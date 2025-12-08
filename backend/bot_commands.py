import asyncio
import logging
import re
import time
from datetime import datetime

logger = logging.getLogger(__name__)

# ========== UTILITY FUNCTIONS ==========

def format_size(size_in_bytes):
    """Format file size in human-readable format"""
    if size_in_bytes is None or size_in_bytes == 0:
        return "Unknown"
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.1f} {unit}"
        size_in_bytes /= 1024.0
    return f"{size_in_bytes:.1f} PB"

# ========== FILE HANDLING FUNCTIONS ==========

async def send_file_to_user(client, user_id, file_message, quality="480p", config=None, bot_instance=None, pyrogram_available=True):
    """Send file to user with verification check"""
    try:
        if not pyrogram_available:
            return False, {
                'message': "❌ Bot is in maintenance mode. Please try again later.",
                'buttons': []
            }, 0
        
        # ✅ Check user status
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
        
        # ✅ Get file details (simplified for dummy mode)
        file_name = "file.mp4"
        file_size = 1024 * 1024  # 1 MB dummy
        
        # ✅ Get auto-delete time
        auto_delete_minutes = getattr(config, 'AUTO_DELETE_TIME', 15)
        
        # ✅ Create caption
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
            # Try to send file
            if hasattr(file_message, 'document') and file_message.document:
                sent = await client.send_document(
                    user_id,
                    file_message.document.file_id,
                    caption=file_caption,
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")],
                        [InlineKeyboardButton("🌐 OPEN WEBSITE", url=config.WEBSITE_URL)]
                    ])
                )
            elif hasattr(file_message, 'video') and file_message.video:
                sent = await client.send_video(
                    user_id,
                    file_message.video.file_id,
                    caption=file_caption,
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")],
                        [InlineKeyboardButton("🌐 OPEN WEBSITE", url=config.WEBSITE_URL)]
                    ])
                )
            else:
                # Fallback to document
                sent = await client.send_document(
                    user_id,
                    "dummy_file_id",
                    caption=file_caption,
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")],
                        [InlineKeyboardButton("🌐 OPEN WEBSITE", url=config.WEBSITE_URL)]
                    ])
                )
            
            logger.info(f"✅ File sent to {user_status} user {user_id}")
            
            # ✅ Schedule auto-delete
            if bot_instance and auto_delete_minutes > 0:
                task_id = f"{user_id}_{sent.id}"
                
                # Cancel any existing task
                if task_id in bot_instance.auto_delete_tasks:
                    bot_instance.auto_delete_tasks[task_id].cancel()
                
                # Create new task
                delete_task = asyncio.create_task(
                    bot_instance.schedule_file_deletion(user_id, sent.id, file_name, auto_delete_minutes)
                )
                bot_instance.auto_delete_tasks[task_id] = delete_task
                
                logger.info(f"⏰ Auto-delete scheduled for message {sent.id}")
            
            # ✅ Return success
            return True, {
                'success': True,
                'file_name': file_name,
                'file_size': file_size,
                'quality': quality,
                'user_status': user_status,
                'auto_delete_minutes': auto_delete_minutes,
                'message_id': sent.id,
            }, file_size
            
        except Exception as e:
            logger.error(f"File sending error: {e}")
            return False, {
                'message': f"❌ Error: {str(e)[:100]}",
                'buttons': []
            }, 0
                
    except Exception as e:
        logger.error(f"File sending error: {e}")
        return False, {
            'message': f"❌ Error: {str(e)}",
            'buttons': []
        }, 0

async def handle_verification_token(client, message, token, bot_instance, pyrogram_available=True):
    """Handle verification token from /start verify_<token>"""
    try:
        user_id = message.from_user.id
        user_name = message.from_user.first_name or "User"
        
        logger.info(f"🔐 Processing verification token for user {user_id}")
        
        if not bot_instance.verification_system:
            await message.reply_text("❌ Verification system not available.")
            return
        
        # Send processing message
        processing_msg = await message.reply_text(
            f"🔐 **Verifying your access...**\n\n"
            f"**User:** {user_name}\n"
            f"**Token:** `{token[:16]}...`\n"
            f"⏳ **Please wait...**"
        )
        
        # Verify token
        is_valid, verified_user_id, message_text = await bot_instance.verification_system.verify_user_token(token)
        
        if is_valid:
            # Success
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
            
            logger.info(f"✅ User {user_id} verified successfully")
            
        else:
            # Failed
            error_text = (
                f"❌ **Verification Failed**\n\n"
                f"**Reason:** {message_text}\n\n"
                f"🔗 **Get a new verification link:**"
            )
            
            error_keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🔗 GET VERIFICATION LINK", callback_data="get_verified")],
                [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")]
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
            
            logger.warning(f"❌ Verification failed for user {user_id}")
            
    except Exception as e:
        logger.error(f"Verification error: {e}")
        await message.reply_text("❌ Verification error. Please try again.")

async def handle_file_request(client, message, file_text, bot_instance, pyrogram_available=True):
    """Handle file download request"""
    try:
        config = bot_instance.config
        user_id = message.from_user.id
        
        # Clean text
        clean_text = file_text.strip()
        if clean_text.startswith('/start'):
            clean_text = clean_text.replace('/start', '').strip()
        
        clean_text = re.sub(r'^/start\s+', '', clean_text)
        
        # Parse file request
        parts = clean_text.split('_')
        if len(parts) < 2:
            await message.reply_text(
                "❌ **Invalid file format**\n\n"
                "Correct format: `-1001768249569_16066_480p`\n"
                "Please click download button on website again."
            )
            return
        
        logger.info(f"📥 File request from user {user_id}: {clean_text}")
        
        try:
            processing_msg = await message.reply_text(
                f"⏳ **Preparing your file...**\n\n"
                f"📹 **Quality:** {parts[2] if len(parts) > 2 else '480p'}\n"
                f"🔄 **Checking access...**"
            )
        except:
            processing_msg = None
        
        if not pyrogram_available:
            try:
                if processing_msg:
                    await processing_msg.edit_text(
                        "❌ **Bot is in maintenance mode**\n\n"
                        "Please try again later or contact admin.\n\n"
                        f"🌐 Visit: {config.WEBSITE_URL}"
                    )
                else:
                    await message.reply_text(
                        "❌ **Bot is in maintenance mode**\n\n"
                        f"Visit: {config.WEBSITE_URL}"
                    )
            except:
                pass
            return
        
        # Get file from channel (simplified)
        try:
            # For now, just send a dummy response
            text = (
                f"✅ **File Ready for Download**\n\n"
                f"**User ID:** `{user_id}`\n"
                f"**Request:** `{clean_text}`\n\n"
                f"⚠️ **Note:** Bot is in limited mode\n"
                f"🌐 **Visit website:** {config.WEBSITE_URL}\n\n"
                f"🎬 @SK4FiLM"
            )
            
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🌐 OPEN WEBSITE", url=config.WEBSITE_URL)],
                [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")]
            ])
            
            try:
                if processing_msg:
                    await processing_msg.delete()
            except:
                pass
            
            await message.reply_text(text, reply_markup=keyboard)
            
        except Exception as e:
            logger.error(f"File request error: {e}")
            await message.reply_text("❌ **Download Error**\n\nPlease try again.")
        
    except Exception as e:
        logger.error(f"File request error: {e}")
        await message.reply_text("❌ **Download Error**\n\nPlease try again.")

# ========== SETUP HANDLERS ==========

async def setup_bot_handlers(bot, bot_instance, pyrogram_available=True):
    """Setup all bot handlers"""
    config = bot_instance.config
    
    logger.info("Setting up bot handlers...")
    
    if not pyrogram_available:
        logger.warning("⚠️ Pyrogram not available, using dummy handlers")
    
    # ✅ START COMMAND
    @bot.on_message(filters.command("start"))
    async def handle_start_command(client, message):
        """Handle /start command"""
        user_name = message.from_user.first_name or "User"
        user_id = message.from_user.id
        
        logger.info(f"/start command from user {user_id}")
        
        # Check if there's additional text
        if len(message.command) > 1:
            start_text = ' '.join(message.command[1:])
            
            # Check if it's a verification token
            if start_text.startswith('verify_'):
                token = start_text.replace('verify_', '', 1).strip()
                await handle_verification_token(client, message, token, bot_instance, pyrogram_available)
                return
            else:
                # Treat as file request
                await handle_file_request(client, message, start_text, bot_instance, pyrogram_available)
                return
        
        # WELCOME MESSAGE
        welcome_text = (
            f"🎬 **Welcome to SK4FiLM, {user_name}!**\n\n"
            f"🌐 **Website:** {config.WEBSITE_URL}\n\n"
            "**Commands:**\n"
            "• /mypremium - Check your premium status\n"
            "• /plans - View premium plans\n"
            "• /buy - Purchase premium\n"
            "• /help - Show help\n\n"
            "**How to download:**\n"
            "1. Visit website above\n"
            "2. Search for movies\n"
            "3. Click download button\n"
            "4. File will appear here automatically\n\n"
            "🎬 **Happy watching!**"
        )
        
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🌐 OPEN WEBSITE", url=config.WEBSITE_URL)],
            [InlineKeyboardButton("⭐ GET PREMIUM", callback_data="buy_premium")],
            [InlineKeyboardButton("📢 JOIN CHANNEL", url=getattr(config, 'MAIN_CHANNEL_LINK', 'https://t.me/SK4FiLM'))]
        ])
        
        await message.reply_text(welcome_text, reply_markup=keyboard, disable_web_page_preview=True)
    
    # ✅ MY PREMIUM COMMAND
    @bot.on_message(filters.command("mypremium") & filters.private)
    async def my_premium_command(client, message):
        """Check user's premium status"""
        user_id = message.from_user.id
        
        logger.info(f"/mypremium command from user {user_id}")
        
        if not bot_instance.premium_system:
            await message.reply_text("❌ Premium system not available.")
            return
        
        try:
            premium_info = "**Your Premium Status:**\n\n"
            
            if hasattr(bot_instance.premium_system, 'is_premium_user'):
                is_premium = await bot_instance.premium_system.is_premium_user(user_id)
                if is_premium:
                    premium_info += "✅ **Status:** Premium User ⭐\n"
                    premium_info += "✨ **Benefits:** Unlimited downloads, no verification needed\n"
                else:
                    premium_info += "❌ **Status:** Free User\n"
                    premium_info += "🔗 **Action:** Get verified or buy premium\n"
            
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")],
                [InlineKeyboardButton("🌐 OPEN WEBSITE", url=config.WEBSITE_URL)]
            ])
            
            await message.reply_text(premium_info, reply_markup=keyboard, disable_web_page_preview=True)
            
        except Exception as e:
            logger.error(f"My premium command error: {e}")
            await message.reply_text("❌ Error fetching premium info.")
    
    # ✅ PLANS COMMAND
    @bot.on_message(filters.command("plans") & filters.private)
    async def plans_command(client, message):
        """Show all premium plans"""
        logger.info(f"/plans command from user {message.from_user.id}")
        
        plans_text = (
            "⭐ **SK4FiLM PREMIUM PLANS** ⭐\n\n"
            "**🥉 Basic Plan** - ₹99/month\n"
            "• All quality (480p-4K)\n"
            "• Unlimited downloads\n"
            "• No verification required\n\n"
            "**🥈 Premium Plan** - ₹199/month\n"
            "• Everything in Basic +\n"
            "• Priority support\n"
            "• Faster downloads\n\n"
            "**🥇 Gold Plan** - ₹299/2 months\n"
            "• Everything in Premium +\n"
            "• Early access to new releases\n\n"
            "**💎 Diamond Plan** - ₹499/3 months\n"
            "• Everything in Gold +\n"
            "• VIP support\n"
            "• Custom requests\n\n"
            "Click a button below to purchase:"
        )
        
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("💰 BUY BASIC (₹99)", callback_data="plan_basic")],
            [InlineKeyboardButton("💰 BUY PREMIUM (₹199)", callback_data="plan_premium")],
            [InlineKeyboardButton("💰 BUY GOLD (₹299)", callback_data="plan_gold")],
            [InlineKeyboardButton("💰 BUY DIAMOND (₹499)", callback_data="plan_diamond")],
            [InlineKeyboardButton("🔙 BACK", callback_data="back_to_start")]
        ])
        
        await message.reply_text(plans_text, reply_markup=keyboard, disable_web_page_preview=True)
    
    # ✅ BUY COMMAND
    @bot.on_message(filters.command("buy") & filters.private)
    async def buy_command(client, message):
        """Initiate premium purchase"""
        user_id = message.from_user.id
        user_name = message.from_user.first_name or "User"
        
        logger.info(f"/buy command from user {user_id}")
        
        # Check if already premium
        if bot_instance.premium_system:
            is_premium = await bot_instance.premium_system.is_premium_user(user_id)
            if is_premium:
                text = (
                    f"⭐ **You're Already Premium!** ⭐\n\n"
                    f"**User:** {user_name}\n"
                    f"Enjoy unlimited downloads without verification! 🎬"
                )
                
                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton("🌐 OPEN WEBSITE", url=config.WEBSITE_URL)],
                    [InlineKeyboardButton("🔙 BACK", callback_data="back_to_start")]
                ])
                
                await message.reply_text(text, reply_markup=keyboard)
                return
        
        text = (
            f"💰 **Purchase Premium - {user_name}**\n\n"
            "**Select a plan:**\n\n"
            "🥉 **Basic Plan** - ₹99/month\n"
            "• All quality (480p-4K)\n"
            "• Unlimited downloads\n"
            "• No verification\n\n"
            "🥈 **Premium Plan** - ₹199/month\n"
            "• Everything in Basic +\n"
            "• Priority support\n"
            "• Faster downloads\n\n"
            "Click a button below to purchase:"
        )
        
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🥉 BUY BASIC (₹99)", callback_data="plan_basic")],
            [InlineKeyboardButton("🥈 BUY PREMIUM (₹199)", callback_data="plan_premium")],
            [InlineKeyboardButton("🥇 BUY GOLD (₹299)", callback_data="plan_gold")],
            [InlineKeyboardButton("💎 BUY DIAMOND (₹499)", callback_data="plan_diamond")],
            [InlineKeyboardButton("🔙 BACK", callback_data="back_to_start")]
        ])
        
        await message.reply_text(text, reply_markup=keyboard)
    
    # ✅ HELP COMMAND
    @bot.on_message(filters.command("help") & filters.private)
    async def help_command(client, message):
        """Show help message"""
        logger.info(f"/help command from user {message.from_user.id}")
        
        help_text = (
            "🆘 **SK4FiLM Bot Help** 🆘\n\n"
            "**Available Commands:**\n"
            "• /start - Start the bot\n"
            "• /mypremium - Check your premium status\n"
            "• /plans - View premium plans\n"
            "• /buy - Purchase premium subscription\n"
            "• /help - Show this help message\n\n"
            "**How to Download Files:**\n"
            "1. Visit our website\n"
            "2. Search for movies/TV shows\n"
            "3. Click download button\n"
            "4. File will appear here automatically\n\n"
            "**Support:**\n"
            f"🌐 Website: {config.WEBSITE_URL}\n"
            "📢 Channel: @SK4FiLM\n"
            "🎬 **Happy downloading!**"
        )
        
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🌐 OPEN WEBSITE", url=config.WEBSITE_URL)],
            [InlineKeyboardButton("⭐ GET PREMIUM", callback_data="buy_premium")],
            [InlineKeyboardButton("🔙 BACK", callback_data="back_to_start")]
        ])
        
        await message.reply_text(help_text, reply_markup=keyboard, disable_web_page_preview=True)
    
    # ✅ FILE REQUEST HANDLER
    @bot.on_message(filters.private & filters.regex(r'^-?\d+_\d+(_\w+)?$'))
    async def handle_direct_file_request(client, message):
        """Handle direct file format messages"""
        file_text = message.text.strip()
        logger.info(f"Direct file request: {file_text}")
        await handle_file_request(client, message, file_text, bot_instance, pyrogram_available)
    
    # ✅ CALLBACK HANDLERS
    
    @bot.on_callback_query(filters.regex(r"^get_verified$"))
    async def get_verified_callback(client, callback_query):
        """Get verification link"""
        user_id = callback_query.from_user.id
        user_name = callback_query.from_user.first_name or "User"
        
        logger.info(f"get_verified callback from user {user_id}")
        
        if bot_instance.verification_system:
            try:
                verification_data = await bot_instance.verification_system.create_verification_link(user_id)
                
                text = (
                    f"🔗 **Verification Required - {user_name}**\n\n"
                    "To access files, you need to verify:\n\n"
                    f"🔗 **Click:** {verification_data['short_url']}\n"
                    f"⏰ **Valid for:** {verification_data['valid_for_hours']} hours\n\n"
                    "**Steps:**\n"
                    "1. Click VERIFY NOW button\n"
                    "2. Join our channel\n"
                    "3. Return here for downloads\n"
                    "4. Access lasts 6 hours\n\n"
                    "⭐ **Premium users don't need verification**"
                )
                
                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔗 VERIFY NOW", url=verification_data['short_url'])],
                    [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")],
                    [InlineKeyboardButton("🔙 BACK", callback_data="back_to_start")]
                ])
                
                await callback_query.message.edit_text(
                    text=text,
                    reply_markup=keyboard,
                    disable_web_page_preview=True
                )
            except Exception as e:
                logger.error(f"Verification callback error: {e}")
                await callback_query.answer("Error generating verification link!", show_alert=True)
        else:
            await callback_query.answer("Verification system not available!", show_alert=True)
    
    @bot.on_callback_query(filters.regex(r"^back_to_start$"))
    async def back_to_start_callback(client, callback_query):
        user_name = callback_query.from_user.first_name or "User"
        
        logger.info(f"back_to_start callback from user {callback_query.from_user.id}")
        
        text = (
            f"🎬 **Welcome back, {user_name}!**\n\n"
            f"Visit {config.WEBSITE_URL} to download movies.\n"
            "Click download button on website and file will appear here."
        )
        
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🌐 OPEN WEBSITE", url=config.WEBSITE_URL)],
            [InlineKeyboardButton("📢 JOIN CHANNEL", url=getattr(config, 'MAIN_CHANNEL_LINK', 'https://t.me/SK4FiLM'))]
        ])
        
        try:
            await callback_query.message.edit_text(
                text=text,
                reply_markup=keyboard,
                disable_web_page_preview=True
            )
        except:
            await callback_query.answer("Already on home page!")
    
    @bot.on_callback_query(filters.regex(r"^buy_premium$"))
    async def buy_premium_callback(client, callback_query):
        """Show premium plans"""
        user_id = callback_query.from_user.id
        user_name = callback_query.from_user.first_name or "User"
        
        logger.info(f"buy_premium callback from user {user_id}")
        
        # Check if already premium
        if bot_instance.premium_system:
            is_premium = await bot_instance.premium_system.is_premium_user(user_id)
            if is_premium:
                text = (
                    f"⭐ **You're Already Premium!** ⭐\n\n"
                    f"**User:** {user_name}\n"
                    f"Enjoy unlimited downloads without verification! 🎬"
                )
                
                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton("🌐 OPEN WEBSITE", url=config.WEBSITE_URL)],
                    [InlineKeyboardButton("🔙 BACK", callback_data="back_to_start")]
                ])
                
                try:
                    await callback_query.message.edit_text(text, reply_markup=keyboard)
                except:
                    await callback_query.answer("You're already premium!", show_alert=True)
                return
        
        text = (
            f"⭐ **SK4FiLM PREMIUM - {user_name}** ⭐\n\n"
            "**Benefits:**\n"
            "✅ No verification required\n"
            "✅ All quality (480p-4K)\n"
            "✅ Unlimited downloads\n"
            "✅ No ads\n"
            "✅ Priority support\n\n"
            "**Plans:**\n"
            "• **Basic** - ₹99/month\n"
            "• **Premium** - ₹199/month\n"
            "• **Gold** - ₹299/2 months\n"
            "• **Diamond** - ₹499/3 months\n\n"
            "Click below to purchase:"
        )
        
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🥉 BUY BASIC (₹99)", callback_data="plan_basic")],
            [InlineKeyboardButton("🥈 BUY PREMIUM (₹199)", callback_data="plan_premium")],
            [InlineKeyboardButton("🥇 BUY GOLD (₹299)", callback_data="plan_gold")],
            [InlineKeyboardButton("💎 BUY DIAMOND (₹499)", callback_data="plan_diamond")],
            [InlineKeyboardButton("🔙 BACK", callback_data="back_to_start")]
        ])
        
        try:
            await callback_query.message.edit_text(text, reply_markup=keyboard)
        except:
            await callback_query.answer("Premium plans!", show_alert=True)
    
    @bot.on_callback_query(filters.regex(r"^plan_"))
    async def plan_selection_callback(client, callback_query):
        plan_type = callback_query.data.split('_')[1]
        user_id = callback_query.from_user.id
        
        logger.info(f"plan_selection callback: {plan_type} from user {user_id}")
        
        if plan_type == "basic":
            plan_name = "Basic Plan"
            amount = "₹99"
        elif plan_type == "premium":
            plan_name = "Premium Plan"
            amount = "₹199"
        elif plan_type == "gold":
            plan_name = "Gold Plan"
            amount = "₹299"
        elif plan_type == "diamond":
            plan_name = "Diamond Plan"
            amount = "₹499"
        else:
            await callback_query.answer("Invalid plan!", show_alert=True)
            return
        
        text = (
            f"💰 **{plan_name} - {amount}**\n\n"
            "**Payment Instructions:**\n"
            "1. Send payment to:\n"
            "   **UPI ID:** sk4film@upi\n"
            "   **PhonePay/GooglePay:** 9876543210\n\n"
            "2. After payment, send screenshot here\n"
            "3. Admin will verify and activate premium\n\n"
            "⏰ **Activation:** Within 24 hours\n"
            "✅ **Benefits:** Instant access, no verification"
        )
        
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("📸 SEND SCREENSHOT", callback_data=f"send_screenshot_{plan_type}")],
            [InlineKeyboardButton("🔙 BACK", callback_data="buy_premium")]
        ])
        
        try:
            await callback_query.message.edit_text(text, reply_markup=keyboard, disable_web_page_preview=True)
        except:
            await callback_query.answer("Payment instructions!", show_alert=True)
    
    @bot.on_callback_query(filters.regex(r"^send_screenshot_"))
    async def send_screenshot_callback(client, callback_query):
        plan_type = callback_query.data.split('_')[2]
        
        logger.info(f"send_screenshot callback for plan: {plan_type}")
        
        text = (
            "📸 **Please send the payment screenshot now**\n\n"
            "1. Take a clear screenshot of the payment\n"
            "2. Send it to this chat\n"
            "3. Our admin will verify and activate your premium\n\n"
            f"**Plan:** {plan_type.capitalize()} Plan\n"
            "⏰ Please send within 24 hours of payment"
        )
        
        await callback_query.answer("Please send screenshot now!", show_alert=True)
        
        # Send new message
        await callback_query.message.reply_text(text)
        
        # Try to delete the original callback message
        try:
            await callback_query.message.delete()
        except:
            pass
    
    # ✅ HANDLE SCREENSHOT MESSAGES
    @bot.on_message(filters.private & (filters.photo | filters.document))
    async def handle_screenshot(client, message):
        """Handle payment screenshots"""
        user_id = message.from_user.id
        
        logger.info(f"Screenshot received from user {user_id}")
        
        # Check if it's likely a screenshot
        if message.photo or (message.document and hasattr(message.document, 'mime_type') and 'image' in message.document.mime_type):
            await message.reply_text(
                "✅ **Screenshot received!**\n\n"
                "Our admin will verify your payment and activate your premium within 24 hours.\n"
                "Thank you for choosing SK4FiLM! 🎬\n\n"
                "You will receive a confirmation message when activated.",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 BACK TO START", callback_data="back_to_start")]
                ])
            )
            
            # Notify admin
            admin_ids = getattr(config, 'ADMIN_IDS', [])
            for admin_id in admin_ids:
                try:
                    await client.send_message(
                        admin_id,
                        f"📸 **New Payment Screenshot**\n\n"
                        f"**User:** {message.from_user.first_name} {message.from_user.last_name or ''}\n"
                        f"**User ID:** `{user_id}`\n"
                        f"**Username:** @{message.from_user.username or 'N/A'}\n"
                        f"**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                        f"Please check and approve premium."
                    )
                except:
                    pass
        else:
            # Not a screenshot, send website instructions
            await message.reply_text(
                f"🌐 **Visit SK4FiLM Website**\n\n"
                f"To download movies, visit:\n{config.WEBSITE_URL}\n\n"
                "1. Search for your movie\n"
                "2. Click download button\n"
                "3. File will appear here automatically\n\n"
                "🎬 Happy watching!",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🌐 OPEN WEBSITE", url=config.WEBSITE_URL)],
                    [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")]
                ])
            )
    
    logger.info("✅ Bot handlers setup complete")
