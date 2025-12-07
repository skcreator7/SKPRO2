import asyncio
import logging
import re
import time
from datetime import datetime, timedelta

try:
    from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton
    from pyrogram.errors import FloodWait, BadRequest
    PYROGRAM_AVAILABLE = True
except ImportError:
    class InlineKeyboardMarkup:
        def __init__(self, buttons): pass
    class InlineKeyboardButton:
        def __init__(self, text, url=None, callback_data=None): pass
    PYROGRAM_AVAILABLE = False

logger = logging.getLogger(__name__)

def format_size(size_in_bytes):
    """Format file size in human-readable format"""
    if size_in_bytes is None or size_in_bytes == 0:
        return "Unknown"
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.1f} {unit}"
        size_in_bytes /= 1024.0
    return f"{size_in_bytes:.1f} PB"

async def send_file_to_user(client, user_id, file_message, quality="480p", config=None, bot_instance=None):
    """Send file to user with verification check"""
    try:
        # ✅ FIRST CHECK: Verify user is premium/verified/admin
        user_status = "Checking..."
        status_icon = "⏳"
        can_download = False
        
        # Check if user is admin
        is_admin = user_id in getattr(config, 'ADMIN_IDS', [])
        
        if is_admin:
            can_download = True
            user_status = "Admin User 👑"
            status_icon = "👑"
        elif bot_instance and bot_instance.premium_system:
            # Check premium status
            is_premium = await bot_instance.premium_system.is_premium_user(user_id)
            if is_premium:
                can_download = True
                user_status = "Premium User ⭐"
                status_icon = "⭐"
            else:
                # Check verification status
                if bot_instance.verification_system:
                    is_verified, _ = await bot_instance.verification_system.check_user_verified(
                        user_id, bot_instance.premium_system
                    )
                    if is_verified:
                        can_download = True
                        user_status = "Verified User ✅"
                        status_icon = "✅"
                    else:
                        # User needs verification
                        verification_data = await bot_instance.verification_system.create_verification_link(user_id)
                        return False, {
                            'message': f"🔒 **Access Restricted**\n\n❌ You need to verify or purchase premium to download files.",
                            'buttons': [
                                [InlineKeyboardButton("🔗 VERIFY NOW", url=verification_data['short_url'])],
                                [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")]
                            ]
                        }, 0
                else:
                    return False, {
                        'message': "❌ Verification system not available. Please try again later.",
                        'buttons': []
                    }, 0
        else:
            return False, {
                'message': "❌ System temporarily unavailable. Please try again later.",
                'buttons': []
            }, 0
        
        if not can_download:
            return False, {
                'message': "❌ Access denied. Please upgrade to premium or complete verification.",
                'buttons': []
            }, 0
        
        # ✅ FILE SENDING LOGIC
        if file_message.document:
            file_name = file_message.document.file_name or "file"
            file_size = file_message.document.file_size or 0
            file_id = file_message.document.file_id
            is_video = False
        elif file_message.video:
            file_name = file_message.video.file_name or "video.mp4"
            file_size = file_message.video.file_size or 0
            file_id = file_message.video.file_id
            is_video = True
        else:
            return False, {
                'message': "❌ No downloadable file found in this message",
                'buttons': []
            }, 0
        
        # ✅ Validate file ID
        if not file_id:
            logger.error(f"❌ Empty file ID for message {file_message.id}")
            return False, {
                'message': "❌ File ID is empty. Please try download again.",
                'buttons': []
            }, 0
        
        # ✅ Get auto-delete time from config (default 15 minutes)
        auto_delete_minutes = getattr(config, 'AUTO_DELETE_TIME', 15)
        
        # ✅ SIMPLE CAPTION
        file_caption = (
            f"📁 **File:** `{file_name}`\n"
            f"📦 **Size:** {format_size(file_size)}\n"
            f"📹 **Quality:** {quality}\n"
            f"{status_icon} **Status:** {user_status}\n\n"
            f"♻ **Forward to saved messages for safety**\n"
            f"⏰ **Auto-delete in:** {auto_delete_minutes} minutes\n\n"
            f"@SK4FiLM 🎬"
        )
        
        try:
            if file_message.document:
                sent = await client.send_document(
                    user_id,
                    file_id,
                    caption=file_caption,
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")],
                        [InlineKeyboardButton("🌐 OPEN WEBSITE", url=config.WEBSITE_URL)]
                    ])
                )
            else:
                sent = await client.send_video(
                    user_id,
                    file_id,
                    caption=file_caption,
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")],
                        [InlineKeyboardButton("🌐 OPEN WEBSITE", url=config.WEBSITE_URL)]
                    ])
                )
            
            logger.info(f"✅ File sent to {user_status} user {user_id}: {file_name}")
            
            # ✅ Schedule auto-delete
            if bot_instance and auto_delete_minutes > 0:
                task_id = f"{user_id}_{sent.id}"
                
                # Cancel any existing task for this user
                if task_id in bot_instance.auto_delete_tasks:
                    bot_instance.auto_delete_tasks[task_id].cancel()
                
                # Create new auto-delete task
                delete_task = asyncio.create_task(
                    bot_instance.schedule_file_deletion(user_id, sent.id, file_name, auto_delete_minutes)
                )
                bot_instance.auto_delete_tasks[task_id] = delete_task
                bot_instance.file_messages_to_delete[task_id] = {
                    'message_id': sent.id,
                    'file_name': file_name,
                    'scheduled_time': datetime.now() + timedelta(minutes=auto_delete_minutes)
                }
                
                logger.info(f"⏰ Auto-delete scheduled for message {sent.id} in {auto_delete_minutes} minutes")
            
            # ✅ Return success
            return True, {
                'success': True,
                'file_name': file_name,
                'file_size': file_size,
                'quality': quality,
                'user_status': user_status,
                'status_icon': status_icon,
                'auto_delete_minutes': auto_delete_minutes,
                'message_id': sent.id,
                'single_message': True
            }, file_size
            
        except BadRequest as e:
            if "MEDIA_EMPTY" in str(e) or "FILE_REFERENCE_EXPIRED" in str(e):
                logger.error(f"❌ File reference expired or empty: {e}")
                # Try to refresh file reference
                try:
                    # Get fresh message
                    fresh_msg = await client.get_messages(
                        file_message.chat.id,
                        file_message.id
                    )
                    
                    if fresh_msg.document:
                        new_file_id = fresh_msg.document.file_id
                    elif fresh_msg.video:
                        new_file_id = fresh_msg.video.file_id
                    else:
                        return False, {
                            'message': "❌ File reference expired, please try download again",
                            'buttons': []
                        }, 0
                    
                    # Retry with new file ID
                    if file_message.document:
                        sent = await client.send_document(
                            user_id, 
                            new_file_id,
                            caption=file_caption,
                            reply_markup=InlineKeyboardMarkup([
                                [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")],
                                [InlineKeyboardButton("🌐 OPEN WEBSITE", url=config.WEBSITE_URL)]
                            ])
                        )
                    else:
                        sent = await client.send_video(
                            user_id, 
                            new_file_id,
                            caption=file_caption,
                            reply_markup=InlineKeyboardMarkup([
                                [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")],
                                [InlineKeyboardButton("🌐 OPEN WEBSITE", url=config.WEBSITE_URL)]
                            ])
                        )
                    
                    logger.info(f"✅ File sent with refreshed reference to {user_id}")
                    
                    # ✅ Schedule auto-delete for refreshed file
                    if bot_instance and auto_delete_minutes > 0:
                        task_id = f"{user_id}_{sent.id}"
                        
                        # Cancel any existing task for this user
                        if task_id in bot_instance.auto_delete_tasks:
                            bot_instance.auto_delete_tasks[task_id].cancel()
                        
                        # Create new auto-delete task
                        delete_task = asyncio.create_task(
                            bot_instance.schedule_file_deletion(user_id, sent.id, file_name, auto_delete_minutes)
                        )
                        bot_instance.auto_delete_tasks[task_id] = delete_task
                        bot_instance.file_messages_to_delete[task_id] = {
                            'message_id': sent.id,
                            'file_name': file_name,
                            'scheduled_time': datetime.now() + timedelta(minutes=auto_delete_minutes)
                        }
                        
                        logger.info(f"⏰ Auto-delete scheduled for refreshed message {sent.id}")
                    
                    return True, {
                        'success': True,
                        'file_name': file_name,
                        'file_size': file_size,
                        'quality': quality,
                        'user_status': user_status,
                        'status_icon': status_icon,
                        'auto_delete_minutes': auto_delete_minutes,
                        'message_id': sent.id,
                        'refreshed': True,
                        'single_message': True
                    }, file_size
                    
                except Exception as retry_error:
                    logger.error(f"❌ Retry failed: {retry_error}")
                    return False, {
                        'message': "❌ File reference expired, please try download again",
                        'buttons': []
                    }, 0
            else:
                raise e
                
    except FloodWait as e:
        logger.warning(f"⏳ Flood wait: {e.value}s")
        return False, {
            'message': f"⏳ Please wait {e.value} seconds (Telegram limit)",
            'buttons': []
        }, 0
    except Exception as e:
        logger.error(f"File sending error: {e}")
        return False, {
            'message': f"❌ Error: {str(e)}",
            'buttons': []
        }, 0

async def handle_verification_token(client, message, token, bot_instance):
    """Handle verification token from /start verify_<token>"""
    try:
        user_id = message.from_user.id
        user_name = message.from_user.first_name or "User"
        
        # ✅ VERIFICATION RATE LIMIT CHECK
        if not await bot_instance.check_rate_limit(user_id, limit=5, window=60, request_type="verification"):
            await message.reply_text(
                "⚠️ **Verification Rate Limit**\n\n"
                "Too many verification attempts. Please wait 60 seconds."
            )
            return
        
        # ✅ DUPLICATE VERIFICATION CHECK
        if await bot_instance.is_request_duplicate(user_id, token, request_type="verification"):
            logger.warning(f"⚠️ Duplicate verification ignored for user {user_id}")
            await message.reply_text(
                "⏳ **Already Processing Verification**\n\n"
                "Your verification is already being processed. Please wait..."
            )
            return
        
        logger.info(f"🔐 Processing verification token for user {user_id}: {token[:16]}...")
        
        if not bot_instance.verification_system:
            await message.reply_text("❌ Verification system not available. Please try again later.")
            await bot_instance.clear_processing_request(user_id, token, request_type="verification")
            return
        
        # Send processing message
        processing_msg = await message.reply_text(
            f"🔐 **Verifying your access...**\n\n"
            f"**User:** {user_name}\n"
            f"**Token:** `{token[:16]}...`\n"
            f"⏳ **Please wait...**"
        )
        
        # Verify the token
        is_valid, verified_user_id, message_text = await bot_instance.verification_system.verify_user_token(token)
        
        # Clear processing request
        await bot_instance.clear_processing_request(user_id, token, request_type="verification")
        
        if is_valid:
            # Success!
            success_text = (
                f"✅ **Verification Successful!** ✅\n\n"
                f"**Welcome, {user_name}!** 🎉\n\n"
                f"🎬 **You now have access to:**\n"
                f"• File downloads for 6 hours\n"
                f"• All quality options\n"
                f"• Unlimited downloads\n\n"
                f"⏰ **Access valid for:** 6 hours\n"
                f"✅ **Status:** Verified User\n\n"
                f"Visit {bot_instance.config.WEBSITE_URL} to download movies!\n"
                f"🎬 @SK4FiLM"
            )
            
            success_keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🌐 OPEN WEBSITE", url=bot_instance.config.WEBSITE_URL)],
                [InlineKeyboardButton("⭐ GET PREMIUM", callback_data="buy_premium")]
            ])
            
            try:
                await processing_msg.edit_text(
                    text=success_text,
                    reply_markup=success_keyboard,
                    disable_web_page_preview=True
                )
            except:
                await message.reply_text(
                    success_text,
                    reply_markup=success_keyboard,
                    disable_web_page_preview=True
                )
            
            logger.info(f"✅ User {user_id} verified successfully via token")
            
        else:
            # Verification failed
            error_text = (
                f"❌ **Verification Failed**\n\n"
                f"**Reason:** {message_text}\n\n"
                f"🔗 **Get a new verification link:**\n"
                f"Click the button below"
            )
            
            error_keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🔗 GET VERIFICATION LINK", callback_data="get_verified")],
                [InlineKeyboardButton("⭐ BUY PREMIUM (No verification needed)", callback_data="buy_premium")]
            ])
            
            try:
                await processing_msg.edit_text(
                    text=error_text,
                    reply_markup=error_keyboard,
                    disable_web_page_preview=True
                )
            except:
                await message.reply_text(
                    error_text,
                    reply_markup=error_keyboard,
                    disable_web_page_preview=True
                )
            
            logger.warning(f"❌ Verification failed for user {user_id}: {message_text}")
            
    except Exception as e:
        logger.error(f"Verification token handling error: {e}")
        try:
            await message.reply_text(
                "❌ **Verification Error**\n\n"
                "An error occurred during verification. Please try again."
            )
        except:
            pass
        await bot_instance.clear_processing_request(user_id, token, request_type="verification")

async def handle_file_request(client, message, file_text, bot_instance):
    """Handle file download request with user verification"""
    try:
        config = bot_instance.config
        user_id = message.from_user.id
        
        # ✅ FILE RATE LIMIT CHECK
        if not await bot_instance.check_rate_limit(user_id, limit=3, window=60, request_type="file"):
            await message.reply_text(
                "⚠️ **Download Rate Limit Exceeded**\n\n"
                "You're making too many download requests. Please wait 60 seconds and try again."
            )
            return
        
        # ✅ DUPLICATE FILE REQUEST CHECK
        if await bot_instance.is_request_duplicate(user_id, file_text, request_type="file"):
            logger.warning(f"⚠️ Duplicate file request ignored for user {user_id}: {file_text}")
            await message.reply_text(
                "⏳ **Already Processing Download**\n\n"
                "Your previous download request is still being processed. Please wait..."
            )
            return
        
        # Clean the text
        clean_text = file_text.strip()
        logger.info(f"📥 Processing file request from user {user_id}: {clean_text}")
        
        # Parse file request
        if clean_text.startswith('/start'):
            clean_text = clean_text.replace('/start', '').strip()
        
        clean_text = re.sub(r'^/start\s+', '', clean_text)
        
        # Extract file ID parts
        parts = clean_text.split('_')
        logger.info(f"📥 Parts: {parts}")
        
        if len(parts) < 2:
            await message.reply_text(
                "❌ **Invalid file format**\n\n"
                "Correct format: `-1001768249569_16066_480p`\n"
                "Please click download button on website again."
            )
            await bot_instance.clear_processing_request(user_id, file_text, request_type="file")
            return
        
        # Parse channel ID
        channel_str = parts[0].strip()
        try:
            if channel_str.startswith('--'):
                channel_id = int(channel_str[1:])
            else:
                channel_id = int(channel_str)
        except ValueError:
            await message.reply_text(
                "❌ **Invalid channel ID**\n\n"
                f"Channel ID '{channel_str}' is not valid.\n"
                "Please click download button on website again."
            )
            await bot_instance.clear_processing_request(user_id, file_text, request_type="file")
            return
        
        # Parse message ID
        try:
            message_id = int(parts[1].strip())
        except ValueError:
            await message.reply_text(
                "❌ **Invalid message ID**\n\n"
                f"Message ID '{parts[1]}' is not valid."
            )
            await bot_instance.clear_processing_request(user_id, file_text, request_type="file")
            return
        
        # Get quality
        quality = parts[2].strip() if len(parts) > 2 else "480p"
        
        logger.info(f"📥 Parsed: channel={channel_id}, message={message_id}, quality={quality}")
        
        try:
            # Send processing message
            processing_msg = await message.reply_text(
                f"⏳ **Preparing your file...**\n\n"
                f"📹 **Quality:** {quality}\n"
                f"🔄 **Checking access...**"
            )
        except FloodWait as e:
            logger.warning(f"⏳ Can't send processing message - Flood wait: {e.value}s")
            await asyncio.sleep(e.value)
            processing_msg = await message.reply_text(
                f"⏳ **Preparing your file...**\n\n"
                f"📹 **Quality:** {quality}\n"
                f"🔄 **Checking access...**"
            )
        
        # Get file from channel
        file_message = None
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                # Try user client first
                if bot_instance.user_client and bot_instance.user_session_ready:
                    try:
                        file_message = await bot_instance.user_client.get_messages(
                            channel_id, 
                            message_id
                        )
                        logger.info(f"✅ Attempt {attempt+1}: Got file via user client")
                        break
                    except Exception as e:
                        logger.warning(f"Attempt {attempt+1}: User client failed: {e}")
                
                # Try bot client
                try:
                    file_message = await client.get_messages(
                        channel_id, 
                        message_id
                    )
                    logger.info(f"✅ Attempt {attempt+1}: Got file via bot client")
                    break
                except Exception as e:
                    logger.warning(f"Attempt {attempt+1}: Bot client failed: {e}")
                    
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logger.error(f"Attempt {attempt+1} failed: {e}")
        
        if not file_message:
            try:
                await processing_msg.edit_text(
                    "❌ **File not found**\n\n"
                    "The file may have been deleted or I don't have access."
                )
            except:
                pass
            await bot_instance.clear_processing_request(user_id, file_text, request_type="file")
            return
        
        if not file_message.document and not file_message.video:
            try:
                await processing_msg.edit_text(
                    "❌ **Not a downloadable file**\n\n"
                    "This message doesn't contain a video or document file."
                )
            except:
                pass
            await bot_instance.clear_processing_request(user_id, file_text, request_type="file")
            return
        
        # ✅ Send file to user
        success, result_data, file_size = await send_file_to_user(
            client, message.chat.id, file_message, quality, config, bot_instance
        )
        
        if success:
            # File was sent with caption
            try:
                await processing_msg.delete()
            except:
                pass
            
            # ✅ Record download for statistics
            if bot_instance.premium_system:
                await bot_instance.premium_system.record_download(
                    user_id, 
                    file_size, 
                    quality
                )
                logger.info(f"📊 Download recorded for user {user_id}")
            
        else:
            # Handle error with buttons if available
            error_text = result_data['message']
            error_buttons = result_data.get('buttons', [])
            
            try:
                if error_buttons:
                    await processing_msg.edit_text(
                        error_text,
                        reply_markup=InlineKeyboardMarkup(error_buttons),
                        disable_web_page_preview=True
                    )
                else:
                    await processing_msg.edit_text(error_text)
            except:
                pass
        
        # Clear processing request
        await bot_instance.clear_processing_request(user_id, file_text, request_type="file")
        
    except Exception as e:
        logger.error(f"File request handling error: {e}")
        try:
            await message.reply_text(
                "❌ **Download Error**\n\n"
                "An error occurred during download. Please try again."
            )
        except:
            pass
        await bot_instance.clear_processing_request(user_id, file_text, request_type="file")
