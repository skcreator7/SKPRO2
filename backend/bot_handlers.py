"""
bot_handlers.py - Telegram Bot Handlers for SK4FiLM
FIXED: 
1. Direct message से site visit message reply
2. Auto-delete 5 मिनट कर दिया
3. Duplicate request warning fix
4. Rate limiting improved
5. Fixed filters.command error with ~ operator
6. Fixed multiple replies issue
"""
import asyncio
import logging
import secrets
import re
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from collections import defaultdict

# ✅ Complete Pyrogram imports
try:
    from pyrogram import Client, filters
    from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton, Message, CallbackQuery
    from pyrogram.errors import FloodWait, BadRequest, MessageDeleteForbidden
    PYROGRAM_AVAILABLE = True
except ImportError:
    # Dummy classes for development
    class Client: pass
    class filters:
        @staticmethod
        def command(cmd): return lambda x: x
        @staticmethod
        def private(): return lambda x: x
        @staticmethod
        def regex(pattern): return lambda x: x
        text = None
        photo = None
        document = None
    class InlineKeyboardMarkup:
        def __init__(self, buttons): pass
    class InlineKeyboardButton:
        def __init__(self, text, url=None, callback_data=None): pass
    class Message: pass
    class CallbackQuery: pass
    PYROGRAM_AVAILABLE = False

logger = logging.getLogger(__name__)

class SK4FiLMBot:
    def __init__(self, config, db_manager=None):
        self.config = config
        self.db_manager = db_manager
        self.bot = None
        self.user_client = None
        self.bot_started = False
        self.user_session_ready = False
        
        # Track auto-delete tasks
        self.auto_delete_tasks = {}
        self.file_messages_to_delete = {}
        
        # Rate limiting and deduplication - IMPROVED
        self.user_request_times = defaultdict(list)
        self.processing_requests = {}
        self.verification_processing = {}
        
        # Track last messages to prevent double replies
        self.last_user_messages = {}
        
        # Track message content to prevent duplicate processing
        self.last_message_content = {}
        
        # Initialize all systems
        try:
            from verification import VerificationSystem
            from premium import PremiumSystem, PremiumTier
            from poster_fetching import PosterFetcher
            from cache import CacheManager
            
            self.verification_system = VerificationSystem(config, db_manager)
            self.premium_system = PremiumSystem(config, db_manager)
            self.PremiumTier = PremiumTier
            self.poster_fetcher = PosterFetcher(config)
            self.cache_manager = CacheManager(config)
            
            # Initialize cache
            asyncio.create_task(self.cache_manager.init_redis())
            
            logger.info("✅ All systems initialized")
        except Exception as e:
            logger.error(f"System initialization error: {e}")
            self.verification_system = None
            self.premium_system = None
            self.PremiumTier = None
            self.poster_fetcher = None
            self.cache_manager = None
    
    async def initialize(self):
        """Initialize bot"""
        try:
            logger.info("🚀 Initializing SK4FiLM Bot...")
            
            # Initialize bot
            self.bot = Client(
                "bot",
                api_id=self.config.API_ID,
                api_hash=self.config.API_HASH,
                bot_token=self.config.BOT_TOKEN,
                workers=20
            )
            
            # Initialize user client if session string is provided
            if hasattr(self.config, 'USER_SESSION_STRING') and self.config.USER_SESSION_STRING:
                self.user_client = Client(
                    "user",
                    api_id=self.config.API_ID,
                    api_hash=self.config.API_HASH,
                    session_string=self.config.USER_SESSION_STRING
                )
                await self.user_client.start()
                self.user_session_ready = True
                logger.info("✅ User session started successfully")
            
            # Start bot
            await self.bot.start()
            self.bot_started = True
            logger.info("✅ Bot started successfully")
            
            # Setup handlers
            await setup_bot_handlers(self.bot, self)
            
            # Start cleanup tasks
            if self.verification_system:
                asyncio.create_task(self.verification_system.start_cleanup_task())
            if self.premium_system:
                asyncio.create_task(self.premium_system.start_cleanup_task())
            if self.cache_manager:
                asyncio.create_task(self.cache_manager.start_cleanup_task())
            
            # Start auto-delete monitor
            asyncio.create_task(self._monitor_auto_delete())
            
            # Start cleanup for tracking dictionaries
            asyncio.create_task(self._cleanup_tracking_data())
            
            return True
            
        except Exception as e:
            logger.error(f"Bot initialization failed: {e}")
            traceback.print_exc()
            return False
    
    async def shutdown(self):
        """Shutdown bot"""
        try:
            # Cancel all auto-delete tasks
            for task_id, task in self.auto_delete_tasks.items():
                task.cancel()
            
            if self.bot and self.bot_started:
                await self.bot.stop()
                logger.info("✅ Bot stopped")
            
            if self.user_client and self.user_session_ready:
                await self.user_client.stop()
                logger.info("✅ User client stopped")
                
            # Stop cleanup tasks
            if self.verification_system:
                await self.verification_system.stop_cleanup_task()
            if self.premium_system:
                await self.premium_system.stop_cleanup_task()
            if self.cache_manager:
                await self.cache_manager.stop()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    # ✅ AUTO-DELETE SYSTEM - 5 MINUTES
    async def schedule_file_deletion(self, user_id: int, message_id: int, file_name: str, delete_after_minutes: int = 5):
        """Schedule file deletion after specified minutes (5 minutes for all users)"""
        try:
            task_id = f"{user_id}_{message_id}"
            
            logger.info(f"⏰ Scheduling auto-delete for message {message_id} in {delete_after_minutes} minutes")
            
            # Wait for the specified time
            await asyncio.sleep(delete_after_minutes * 60)
            
            logger.info(f"⏰ Auto-delete time reached for message {message_id} (user {user_id})")
            
            # Try to delete the file message
            try:
                await self.bot.delete_messages(user_id, message_id)
                logger.info(f"✅ Auto-deleted message {message_id} for user {user_id}")
                
                # Send deletion notification
                await self.send_deletion_notification(user_id, file_name, delete_after_minutes)
                
            except MessageDeleteForbidden:
                logger.warning(f"❌ Cannot delete message {message_id} - forbidden")
                # Still send notification
                await self.send_deletion_notification(user_id, file_name, delete_after_minutes, deleted=False)
            except Exception as e:
                logger.error(f"Error deleting message {message_id}: {e}")
                # Still send notification
                await self.send_deletion_notification(user_id, file_name, delete_after_minutes, deleted=False)
            
            # Remove from tracking
            self.auto_delete_tasks.pop(task_id, None)
            self.file_messages_to_delete.pop(task_id, None)
            
        except asyncio.CancelledError:
            logger.info(f"Auto-delete task cancelled for message {message_id}")
        except Exception as e:
            logger.error(f"Error in auto-delete task: {e}")
    
    async def send_deletion_notification(self, user_id: int, file_name: str, delete_after_minutes: int = 5, deleted: bool = True):
        """Send notification about file deletion"""
        try:
            website_url = getattr(self.config, 'WEBSITE_URL', 'https://sk4film.com')
            
            if deleted:
                text = (
                    f"🗑️ **File Auto-Deleted**\n\n"
                    f"`{file_name}`\n\n"
                    f"⏰ **Deleted after:** {delete_after_minutes} minutes\n"
                    f"✅ **Security measure completed**\n\n"
                    f"🔁 **Need the file again?**\n"
                    f"Visit website and download again\n"
                    f"🎬 @SK4FiLM"
                )
            else:
                text = (
                    f"⏰ **File Auto-Delete Time Reached**\n\n"
                    f"`{file_name}`\n\n"
                    f"⏰ **Delete time:** {delete_after_minutes} minutes\n"
                    f"⚠️ **File not deleted (permissions)**\n\n"
                    f"🔁 **Download again from:** {website_url}\n"
                    f"🎬 @SK4FiLM"
                )
            
            buttons = InlineKeyboardMarkup([
                [InlineKeyboardButton("🌐 VISIT WEBSITE", url=website_url)],
                [InlineKeyboardButton("🔄 GET ANOTHER FILE", callback_data="back_to_start")]
            ])
            
            await self.bot.send_message(user_id, text, reply_markup=buttons)
            logger.info(f"✅ Deletion notification sent to user {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to send deletion notification: {e}")
    
    async def _monitor_auto_delete(self):
        """Monitor and manage auto-delete tasks"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Log active tasks
                if self.auto_delete_tasks:
                    logger.info(f"📊 Auto-delete monitoring: {len(self.auto_delete_tasks)} active tasks")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-delete monitor error: {e}")
    
    async def _cleanup_tracking_data(self):
        """Cleanup old tracking data periodically"""
        while True:
            try:
                await asyncio.sleep(300)  # Clean every 5 minutes
                
                current_time = time.time()
                
                # Clean last_user_messages (older than 5 minutes)
                self.last_user_messages = {
                    k: v for k, v in self.last_user_messages.items() 
                    if current_time - v < 300
                }
                
                # Clean last_message_content (older than 2 minutes)
                self.last_message_content = {
                    k: v for k, v in self.last_message_content.items() 
                    if current_time - v['time'] < 120
                }
                
                # Clean processing_requests (older than 2 minutes)
                self.processing_requests = {
                    k: v for k, v in self.processing_requests.items() 
                    if current_time - v < 120
                }
                
                # Clean verification_processing (older than 2 minutes)
                self.verification_processing = {
                    k: v for k, v in self.verification_processing.items() 
                    if current_time - v < 120
                }
                
                logger.debug(f"🧹 Cleanup complete: {len(self.last_user_messages)} user messages, {len(self.last_message_content)} message contents")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup tracking data error: {e}")
    
    # ✅ ADMIN NOTIFICATION SYSTEM
    async def notify_admin_screenshot(self, user_id: int, message_id: int, payment_id: str):
        """Notify admin about payment screenshot"""
        try:
            admin_ids = getattr(self.config, 'ADMIN_IDS', [])
            if not admin_ids:
                logger.warning("❌ No admin IDs configured")
                return False
            
            # Get user info
            try:
                user = await self.bot.get_users(user_id)
                user_name = f"{user.first_name or ''} {user.last_name or ''}".strip() or f"User {user_id}"
                username = f"@{user.username}" if user.username else "No username"
                user_link = f"[{user_name}](tg://user?id={user_id})"
            except Exception as e:
                logger.error(f"Error getting user info: {e}")
                user_name = f"User {user_id}"
                username = "Unknown"
                user_link = f"`{user_id}`"
            
            # Get payment info
            payment_info = None
            if self.premium_system:
                for pid, payment in self.premium_system.pending_payments.items():
                    if pid == payment_id:
                        payment_info = payment
                        break
            
            # Prepare notification message
            notification_text = (
                f"📸 **NEW PAYMENT SCREENSHOT** 📸\n\n"
                f"👤 **User:** {user_link}\n"
                f"🆔 **User ID:** `{user_id}`\n"
                f"📛 **Username:** {username}\n"
                f"💰 **Payment ID:** `{payment_id}`\n\n"
            )
            
            if payment_info:
                notification_text += (
                    f"📋 **Plan:** {payment_info.get('tier_name', 'Unknown')}\n"
                    f"💵 **Amount:** ₹{payment_info.get('amount', 0)}\n"
                    f"📅 **Duration:** {payment_info.get('duration_days', 0)} days\n\n"
                )
            
            notification_text += (
                f"🔗 **Message Link:** [Click to view](tg://openmessage?user_id={user_id}&message_id={message_id})\n"
                f"⏰ **Time:** {datetime.now().strftime('%d %b %Y, %H:%M:%S')}\n\n"
                f"**Commands:**\n"
                f"✅ `/approve {payment_id}` - Approve payment\n"
                f"❌ `/reject {payment_id} <reason>` - Reject payment\n"
                f"👤 `/checkpremium {user_id}` - Check user status"
            )
            
            # Send to all admins
            success_count = 0
            for admin_id in admin_ids:
                try:
                    # Send notification message
                    await self.bot.send_message(
                        admin_id,
                        notification_text,
                        disable_web_page_preview=True
                    )
                    
                    # Forward the screenshot to admin
                    try:
                        await self.bot.forward_messages(
                            admin_id,
                            user_id,
                            message_id
                        )
                    except Exception as e:
                        logger.error(f"Could not forward screenshot to admin {admin_id}: {e}")
                        # Try to send as copy
                        try:
                            await message.copy(admin_id)
                        except:
                            pass
                    
                    success_count += 1
                    logger.info(f"✅ Admin notification sent to admin {admin_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to send notification to admin {admin_id}: {e}")
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Admin notification error: {e}")
            return False
    
    # ✅ RATE LIMITING & DUPLICATE PREVENTION - IMPROVED
    async def check_rate_limit(self, user_id, limit=3, window=60, request_type="file"):
        """Check if user is within rate limits"""
        now = time.time()
        key = f"{user_id}_{request_type}"
        
        # Clean old requests
        self.user_request_times[key] = [
            t for t in self.user_request_times.get(key, []) 
            if now - t < window
        ]
        
        # Check if limit exceeded
        if len(self.user_request_times[key]) >= limit:
            logger.warning(f"⚠️ Rate limit exceeded for user {user_id} ({request_type})")
            return False
        
        # Add current request
        self.user_request_times[key].append(now)
        return True
    
    async def is_request_duplicate(self, user_id, request_data, request_type="file"):
        """Check if this is a duplicate request - IMPROVED"""
        request_hash = f"{user_id}_{request_type}_{hash(str(request_data))}"
        
        if request_type == "verification":
            processing_dict = self.verification_processing
        else:
            processing_dict = self.processing_requests
        
        # Check if same request is already processing
        if request_hash in processing_dict:
            processing_time = processing_dict[request_hash]
            if time.time() - processing_time < 30:  # 30 second cooldown for same request
                logger.debug(f"Duplicate request detected: {request_hash[:50]}...")
                return True
        
        # Mark as processing
        processing_dict[request_hash] = time.time()
        return False
    
    async def clear_processing_request(self, user_id, request_data, request_type="file"):
        """Clear from processing requests"""
        request_hash = f"{user_id}_{request_type}_{hash(str(request_data))}"
        
        if request_type == "verification":
            self.verification_processing.pop(request_hash, None)
        else:
            self.processing_requests.pop(request_hash, None)
    
    # ✅ PREVENT DOUBLE REPLIES - IMPROVED
    async def should_reply(self, user_id: int, message_id: int) -> bool:
        """Check if we should reply to this message (prevent double replies)"""
        key = f"{user_id}_{message_id}"
        
        # Check if we already replied to this exact message
        if key in self.last_user_messages:
            last_time = self.last_user_messages[key]
            if time.time() - last_time < 2:  # 2 second cooldown
                logger.debug(f"Double reply prevented for message {message_id}")
                return False
        
        # Update last message time
        self.last_user_messages[key] = time.time()
        return True
    
    async def is_message_content_duplicate(self, user_id: int, message_text: str) -> bool:
        """Check if same message content was recently sent by user"""
        content_hash = f"{user_id}_{hash(message_text.strip())}"
        current_time = time.time()
        
        if content_hash in self.last_message_content:
            last_time = self.last_message_content[content_hash]['time']
            if current_time - last_time < 5:  # 5 second cooldown for same content
                logger.debug(f"Duplicate message content from user {user_id}")
                return True
        
        # Store new content
        self.last_message_content[content_hash] = {
            'time': current_time,
            'text': message_text[:100]  # Store first 100 chars for debugging
        }
        return False

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
        
        # ✅ Get auto-delete time - HARDCODED 5 MINUTES
        auto_delete_minutes = 5
        
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
            
            # ✅ Schedule auto-delete - 5 MINUTES
            if bot_instance and auto_delete_minutes > 0:
                task_id = f"{user_id}_{sent.id}"
                
                # Cancel any existing task for this user
                if task_id in bot_instance.auto_delete_tasks:
                    bot_instance.auto_delete_tasks[task_id].cancel()
                
                # Create new auto-delete task for 5 minutes
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
                    
                    # ✅ Schedule auto-delete for refreshed file - 5 MINUTES
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
        
        # ✅ PREVENT DOUBLE PROCESSING - IMPROVED
        message_key = f"{user_id}_{message.id}"
        if not await bot_instance.should_reply(user_id, message.id):
            logger.warning(f"⚠️ Duplicate verification processing ignored for user {user_id}")
            return
        
        # ✅ CHECK DUPLICATE MESSAGE CONTENT
        if await bot_instance.is_message_content_duplicate(user_id, message.text):
            logger.debug(f"Duplicate verification message from user {user_id}")
            return
        
        # ✅ VERIFICATION RATE LIMIT CHECK
        if not await bot_instance.check_rate_limit(user_id, limit=5, window=60, request_type="verification"):
            await message.reply_text(
                "⚠️ **Verification Rate Limit**\n\n"
                "Too many verification attempts. Please wait 60 seconds."
            )
            return
        
        # ✅ DUPLICATE VERIFICATION CHECK
        if await bot_instance.is_request_duplicate(user_id, token, request_type="verification"):
            logger.debug(f"Duplicate verification request from user {user_id}")
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
        
        # ✅ PREVENT DOUBLE PROCESSING - IMPROVED
        message_key = f"{user_id}_{message.id}"
        if not await bot_instance.should_reply(user_id, message.id):
            logger.debug(f"⚠️ Duplicate file request processing ignored for user {user_id}")
            return
        
        # ✅ CHECK DUPLICATE MESSAGE CONTENT
        if await bot_instance.is_message_content_duplicate(user_id, message.text):
            logger.debug(f"Duplicate file request message from user {user_id}")
            return
        
        # ✅ FILE RATE LIMIT CHECK
        if not await bot_instance.check_rate_limit(user_id, limit=3, window=60, request_type="file"):
            await message.reply_text(
                "⚠️ **Download Rate Limit Exceeded**\n\n"
                "You're making too many download requests. Please wait 60 seconds and try again."
            )
            return
        
        # ✅ DUPLICATE FILE REQUEST CHECK
        if await bot_instance.is_request_duplicate(user_id, file_text, request_type="file"):
            logger.debug(f"Duplicate file request from user {user_id}: {file_text[:50]}...")
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
        logger.debug(f"📥 Parts: {parts}")
        
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

async def handle_direct_message(client, message, bot_instance):
    """Handle direct messages from users (not commands)"""
    try:
        user_id = message.from_user.id
        user_name = message.from_user.first_name or "User"
        
        # ✅ PREVENT DOUBLE REPLIES
        if not await bot_instance.should_reply(user_id, message.id):
            return
        
        # ✅ CHECK DUPLICATE MESSAGE CONTENT
        if await bot_instance.is_message_content_duplicate(user_id, message.text):
            logger.debug(f"Duplicate direct message from user {user_id}")
            return
        
        # Check if message contains file request pattern
        if re.match(r'^-?\d+_\d+(_\w+)?$', message.text.strip()):
            # It's a file request, handle it
            await handle_file_request(client, message, message.text.strip(), bot_instance)
            return
        
        # Check if it's a verification token
        if message.text.strip().startswith('verify_'):
            token = message.text.strip().replace('verify_', '', 1)
            await handle_verification_token(client, message, token, bot_instance)
            return
        
        # ✅ RATE LIMIT FOR DIRECT MESSAGES
        if not await bot_instance.check_rate_limit(user_id, limit=5, window=60, request_type="message"):
            await message.reply_text(
                "⚠️ **Message Rate Limit**\n\n"
                "Too many messages. Please wait 60 seconds."
            )
            return
        
        # It's a regular message, respond with website visit message
        website_url = getattr(bot_instance.config, 'WEBSITE_URL', 'https://sk4film.com')
        
        response_text = (
            f"👋 **Hello {user_name}!**\n\n"
            f"🎬 **Welcome to SK4FiLM Bot**\n\n"
            f"To download movies/TV shows:\n"
            f"1. Visit our website: {website_url}\n"
            f"2. Search for your movie\n"
            f"3. Click download button\n"
            f"4. File will appear here automatically\n\n"
            f"**Important Notes:**\n"
            f"⏰ **Files auto-delete after 5 minutes** (security)\n"
            f"✅ **Free users need verification every 6 hours**\n"
            f"⭐ **Premium users get instant access**\n\n"
            f"**Commands:**\n"
            f"• /start - Start bot\n"
            f"• /buy - Buy premium\n"
            f"• /help - Help\n\n"
            f"🎬 **Happy watching!**"
        )
        
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🌐 VISIT WEBSITE", url=website_url)],
            [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")],
            [InlineKeyboardButton("📢 JOIN CHANNEL", url=getattr(bot_instance.config, 'MAIN_CHANNEL_LINK', 'https://t.me/SK4FiLM'))]
        ])
        
        # Send response
        sent_msg = await message.reply_text(response_text, reply_markup=keyboard, disable_web_page_preview=True)
        
        # ✅ Schedule auto-delete for this response message too (5 minutes)
        if bot_instance:
            task_id = f"{user_id}_{sent_msg.id}"
            
            # Cancel any existing task for this user
            if task_id in bot_instance.auto_delete_tasks:
                bot_instance.auto_delete_tasks[task_id].cancel()
            
            # Create new auto-delete task for 5 minutes
            delete_task = asyncio.create_task(
                bot_instance.schedule_file_deletion(user_id, sent_msg.id, "Website Visit Message", 5)
            )
            bot_instance.auto_delete_tasks[task_id] = delete_task
            
            logger.info(f"⏰ Auto-delete scheduled for response message {sent_msg.id}")
        
        logger.info(f"✅ Direct message handled for user {user_id}")
        
    except Exception as e:
        logger.error(f"Direct message handling error: {e}")

async def setup_bot_handlers(bot: Client, bot_instance):
    """Setup bot commands and handlers - FIXED VERSION"""
    config = bot_instance.config
    
    # ✅ USER COMMANDS
    
    @bot.on_message(filters.command("start") & filters.private)
    async def handle_start_command(client, message):
        """Handle /start command with verification token detection"""
        user_id = message.from_user.id
        
        # ✅ PREVENT DOUBLE REPLIES
        if not await bot_instance.should_reply(user_id, message.id):
            return
        
        user_name = message.from_user.first_name or "User"
        
        # Check if there's additional text
        if len(message.command) > 1:
            start_text = ' '.join(message.command[1:])
            
            # Check if it's a verification token
            if start_text.startswith('verify_'):
                token = start_text.replace('verify_', '', 1).strip()
                await handle_verification_token(client, message, token, bot_instance)
                return
            else:
                # Treat as file request
                await handle_file_request(client, message, start_text, bot_instance)
                return
        
        # WELCOME MESSAGE
        welcome_text = (
            f"🎬 **Welcome to SK4FiLM, {user_name}!**\n\n"
            f"🌐 **Website:** {config.WEBSITE_URL}\n\n"
            "**Features:**\n"
            "• Free verification every 6 hours\n"
            "• Premium plans for uninterrupted access\n"
            "• All quality formats (480p-4K)\n"
            "• **Files auto-delete after 5 minutes** ⏰\n\n"
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
    
    @bot.on_message(filters.command("mypremium") & filters.private)
    async def my_premium_command(client, message):
        """Check user's premium status"""
        user_id = message.from_user.id
        
        # ✅ PREVENT DOUBLE REPLIES
        if not await bot_instance.should_reply(user_id, message.id):
            return
        
        user_name = message.from_user.first_name or "User"
        
        if not bot_instance.premium_system:
            await message.reply_text("❌ Premium system not available. Please try again later.")
            return
        
        try:
            # Get premium info
            premium_info = await bot_instance.premium_system.get_my_premium_info(user_id)
            
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")],
                [InlineKeyboardButton("🌐 OPEN WEBSITE", url=config.WEBSITE_URL)]
            ])
            
            await message.reply_text(premium_info, reply_markup=keyboard, disable_web_page_preview=True)
            
        except Exception as e:
            logger.error(f"My premium command error: {e}")
            await message.reply_text("❌ Error fetching premium info. Please try again.")
    
    @bot.on_message(filters.command("plans") & filters.private)
    async def plans_command(client, message):
        """Show all premium plans"""
        user_id = message.from_user.id
        
        # ✅ PREVENT DOUBLE REPLIES
        if not await bot_instance.should_reply(user_id, message.id):
            return
        
        if not bot_instance.premium_system:
            await message.reply_text("❌ Premium system not available. Please try again later.")
            return
        
        try:
            plans_text = await bot_instance.premium_system.get_available_plans_text()
            
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("💰 BUY BASIC (₹99)", callback_data="plan_basic")],
                [InlineKeyboardButton("💰 BUY PREMIUM (₹199)", callback_data="plan_premium")],
                [InlineKeyboardButton("💰 BUY GOLD (₹299)", callback_data="plan_gold")],
                [InlineKeyboardButton("💰 BUY DIAMOND (₹499)", callback_data="plan_diamond")],
                [InlineKeyboardButton("🔙 BACK", callback_data="back_to_start")]
            ])
            
            await message.reply_text(plans_text, reply_markup=keyboard, disable_web_page_preview=True)
            
        except Exception as e:
            logger.error(f"Plans command error: {e}")
            await message.reply_text("❌ Error fetching plans. Please try again.")
    
    @bot.on_message(filters.command("buy") & filters.private)
    async def buy_command(client, message):
        """Initiate premium purchase"""
        user_id = message.from_user.id
        
        # ✅ PREVENT DOUBLE REPLIES
        if not await bot_instance.should_reply(user_id, message.id):
            return
        
        user_name = message.from_user.first_name or "User"
        
        # Check if already premium
        if bot_instance.premium_system:
            is_premium = await bot_instance.premium_system.is_premium_user(user_id)
            if is_premium:
                details = await bot_instance.premium_system.get_subscription_details(user_id)
                
                text = (
                    f"⭐ **You're Already Premium!** ⭐\n\n"
                    f"**User:** {user_name}\n"
                    f"**Plan:** {details.get('tier_name', 'Premium')}\n"
                    f"**Days Left:** {details.get('days_remaining', 0)}\n\n"
                    "Enjoy unlimited downloads without verification! 🎬"
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
            "• No verification\n"
            "⏰ Files auto-delete after 5 minutes\n\n"
            "🥈 **Premium Plan** - ₹199/month\n"
            "• Everything in Basic +\n"
            "• Priority support\n"
            "• Faster downloads\n"
            "⏰ Files auto-delete after 5 minutes\n\n"
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
    
    @bot.on_message(filters.command("help") & filters.private)
    async def help_command(client, message):
        """Show help message"""
        user_id = message.from_user.id
        
        # ✅ PREVENT DOUBLE REPLIES
        if not await bot_instance.should_reply(user_id, message.id):
            return
        
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
            "**Verification System:**\n"
            "• Free users need verification every 6 hours\n"
            "• Premium users don't need verification\n"
            "• Verification link valid for 1 hour\n\n"
            "**Auto-Delete Feature:**\n"
            "• Files auto-delete after 5 minutes\n"
            "• For security and privacy\n"
            "• Download again if needed\n"
            "• Same for all users (free & premium)\n\n"
            "**Support:**\n"
            f"🌐 Website: {config.WEBSITE_URL}\n"
            "📢 Channel: @SK4FiLM\n"
            "🆘 Issues: Contact admin\n\n"
            "🎬 **Happy downloading!**"
        )
        
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🌐 OPEN WEBSITE", url=config.WEBSITE_URL)],
            [InlineKeyboardButton("⭐ GET PREMIUM", callback_data="buy_premium")],
            [InlineKeyboardButton("🔙 BACK", callback_data="back_to_start")]
        ])
        
        await message.reply_text(help_text, reply_markup=keyboard, disable_web_page_preview=True)
    
    # ✅ ADMIN COMMANDS
    
    @bot.on_message(filters.command("addpremium") & filters.user(getattr(config, 'ADMIN_IDS', [])))
    async def add_premium_command(client, message):
        """Add premium user command for admins"""
        try:
            # ✅ PREVENT DOUBLE REPLIES
            if not await bot_instance.should_reply(message.from_user.id, message.id):
                return
            
            if len(message.command) < 4:
                await message.reply_text(
                    "❌ **Usage:** `/addpremium <user_id> <days> <plan_type>`\n\n"
                    "**Examples:**\n"
                    "• `/addpremium 123456789 30 basic`\n"
                    "• `/addpremium 123456789 365 premium`\n\n"
                    "**Plan types:** basic, premium, gold, diamond"
                )
                return
            
            user_id = int(message.command[1])
            days = int(message.command[2])
            plan_type = message.command[3].lower()
            
            # Map plan type to PremiumTier
            plan_map = {
                'basic': bot_instance.PremiumTier.BASIC,
                'premium': bot_instance.PremiumTier.PREMIUM,
                'gold': bot_instance.PremiumTier.GOLD,
                'diamond': bot_instance.PremiumTier.DIAMOND
            }
            
            if plan_type not in plan_map:
                await message.reply_text(
                    "❌ **Invalid plan type**\n\n"
                    "Use: `basic`, `premium`, `gold`, or `diamond`\n"
                    "Example: `/addpremium 123456789 30 basic`"
                )
                return
            
            if days <= 0:
                await message.reply_text("❌ Days must be greater than 0")
                return
            
            tier = plan_map[plan_type]
            
            # Get user info
            try:
                user = await client.get_users(user_id)
                user_name = f"{user.first_name or ''} {user.last_name or ''}".strip() or f"User {user_id}"
                username = f"@{user.username}" if user.username else "No username"
            except:
                user_name = f"User {user_id}"
                username = "Unknown"
            
            # Add premium subscription
            if bot_instance.premium_system:
                subscription_data = await bot_instance.premium_system.add_premium_subscription(
                    admin_id=message.from_user.id,
                    user_id=user_id,
                    tier=tier,
                    days=days,
                    reason="admin_command"
                )
                
                if subscription_data:
                    await message.reply_text(
                        f"✅ **Premium User Added Successfully!**\n\n"
                        f"**User:** {user_name}\n"
                        f"**ID:** `{user_id}`\n"
                        f"**Username:** {username}\n"
                        f"**Plan:** {plan_type.capitalize()}\n"
                        f"**Duration:** {days} days\n\n"
                        f"User can now download files without verification!\n"
                        f"⏰ Files still auto-delete after 5 minutes"
                    )
                    
                    # Notify user
                    try:
                        await client.send_message(
                            user_id,
                            f"🎉 **Congratulations!** 🎉\n\n"
                            f"You've been upgraded to **{plan_type.capitalize()} Premium** by admin!\n\n"
                            f"✅ **Plan:** {plan_type.capitalize()}\n"
                            f"📅 **Valid for:** {days} days\n"
                            f"⭐ **Benefits:**\n"
                            f"• Instant file access\n"
                            f"• No verification required\n"
                            f"• Priority support\n"
                            f"⏰ **Files auto-delete after 5 minutes** (security)\n\n"
                            f"🎬 **Enjoy unlimited downloads!**"
                        )
                    except:
                        pass
                else:
                    await message.reply_text("❌ Failed to add premium subscription.")
            else:
                await message.reply_text("❌ Premium system not available")
                
        except ValueError:
            await message.reply_text(
                "❌ **Invalid parameters**\n\n"
                "Correct format: `/addpremium <user_id> <days> <plan_type>`\n"
                "Example: `/addpremium 123456789 30 basic`"
            )
        except Exception as e:
            logger.error(f"Add premium command error: {e}")
            await message.reply_text(f"❌ Error: {str(e)[:100]}")
    
    @bot.on_message(filters.command("removepremium") & filters.user(getattr(config, 'ADMIN_IDS', [])))
    async def remove_premium_command(client, message):
        """Remove premium user command for admins"""
        try:
            # ✅ PREVENT DOUBLE REPLIES
            if not await bot_instance.should_reply(message.from_user.id, message.id):
                return
            
            if len(message.command) < 2:
                await message.reply_text(
                    "❌ **Usage:** `/removepremium <user_id>`\n\n"
                    "**Example:** `/removepremium 123456789`"
                )
                return
            
            user_id = int(message.command[1])
            
            if bot_instance.premium_system:
                success = await bot_instance.premium_system.remove_premium_subscription(
                    admin_id=message.from_user.id,
                    user_id=user_id,
                    reason="admin_command"
                )
                
                if success:
                    await message.reply_text(
                        f"✅ **Premium Removed Successfully!**\n\n"
                        f"**User ID:** `{user_id}`\n"
                        f"Premium access has been revoked."
                    )
                else:
                    await message.reply_text("❌ User not found or not premium")
            else:
                await message.reply_text("❌ Premium system not available")
                
        except ValueError:
            await message.reply_text("❌ Invalid user ID. Must be a number.")
        except Exception as e:
            logger.error(f"Remove premium command error: {e}")
            await message.reply_text(f"❌ Error: {str(e)[:100]}")
    
    @bot.on_message(filters.command("checkpremium") & filters.user(getattr(config, 'ADMIN_IDS', [])))
    async def check_premium_command(client, message):
        """Check premium status of user"""
        try:
            # ✅ PREVENT DOUBLE REPLIES
            if not await bot_instance.should_reply(message.from_user.id, message.id):
                return
            
            if len(message.command) < 2:
                await message.reply_text(
                    "❌ **Usage:** `/checkpremium <user_id>`\n\n"
                    "**Example:** `/checkpremium 123456789`"
                )
                return
            
            user_id = int(message.command[1])
            
            if bot_instance.premium_system:
                user_info = await bot_instance.premium_system.get_premium_user_info(user_id)
                
                # Get user info
                try:
                    user = await client.get_users(user_id)
                    user_name = f"{user.first_name or ''} {user.last_name or ''}".strip() or f"User {user_id}"
                    username = f"@{user.username}" if user.username else "No username"
                except:
                    user_name = f"User {user_id}"
                    username = "Unknown"
                
                if user_info['tier'] == 'free':
                    await message.reply_text(
                        f"❌ **Not a Premium User**\n\n"
                        f"**User:** {user_name}\n"
                        f"**ID:** `{user_id}`\n"
                        f"**Username:** {username}\n"
                        f"**Status:** Free User\n\n"
                        f"This user does not have premium access.\n"
                        f"⏰ Files auto-delete after 5 minutes"
                    )
                else:
                    await message.reply_text(
                        f"✅ **Premium User Found**\n\n"
                        f"**User:** {user_name}\n"
                        f"**ID:** `{user_id}`\n"
                        f"**Username:** {username}\n"
                        f"**Plan:** {user_info.get('tier_name', 'Unknown')}\n"
                        f"**Status:** {user_info.get('status', 'Unknown').title()}\n"
                        f"**Days Left:** {user_info.get('days_remaining', 0)}\n"
                        f"**Total Downloads:** {user_info.get('total_downloads', 0)}\n"
                        f"**Joined:** {user_info.get('purchased_at', 'Unknown')}\n"
                        f"**Expires:** {user_info.get('expires_at', 'Unknown')}\n"
                        f"⏰ **Files auto-delete after 5 minutes**"
                    )
            else:
                await message.reply_text("❌ Premium system not available")
                
        except ValueError:
            await message.reply_text("❌ Invalid user ID. Must be a number.")
        except Exception as e:
            logger.error(f"Check premium command error: {e}")
            await message.reply_text(f"❌ Error: {str(e)[:100]}")
    
    @bot.on_message(filters.command("stats") & filters.user(getattr(config, 'ADMIN_IDS', [])))
    async def stats_command(client, message):
        """Show bot statistics"""
        try:
            # ✅ PREVENT DOUBLE REPLIES
            if not await bot_instance.should_reply(message.from_user.id, message.id):
                return
            
            if bot_instance.premium_system:
                stats = await bot_instance.premium_system.get_statistics()
                
                stats_text = (
                    f"📊 **SK4FiLM Bot Statistics** 📊\n\n"
                    f"👥 **Total Users:** {stats.get('total_users', 0)}\n"
                    f"⭐ **Premium Users:** {stats.get('premium_users', 0)}\n"
                    f"✅ **Active Premium:** {stats.get('active_premium', 0)}\n"
                    f"🎯 **Free Users:** {stats.get('free_users', 0)}\n\n"
                    f"📥 **Total Downloads:** {stats.get('total_downloads', 0)}\n"
                    f"💾 **Total Data Sent:** {stats.get('total_data_sent', '0 GB')}\n"
                    f"💰 **Total Revenue:** {stats.get('total_revenue', '₹0')}\n"
                    f"🛒 **Premium Sales:** {stats.get('total_premium_sales', 0)}\n"
                    f"⏳ **Pending Payments:** {stats.get('pending_payments', 0)}\n\n"
                    f"🔄 **System Status:**\n"
                    f"• Bot: {'✅ Online' if bot_instance.bot_started else '❌ Offline'}\n"
                    f"• User Client: {'✅ Connected' if bot_instance.user_session_ready else '❌ Disconnected'}\n"
                    f"• Verification: {'✅ Active' if bot_instance.verification_system else '❌ Inactive'}\n"
                    f"• Premium: {'✅ Active' if bot_instance.premium_system else '❌ Inactive'}\n\n"
                    f"⏰ **Auto-delete time:** 5 minutes for all users\n"
                    f"🕐 **Server Time:** {stats.get('server_time', 'Unknown')}"
                )
                
                await message.reply_text(stats_text, disable_web_page_preview=True)
            else:
                await message.reply_text("❌ Premium system not available for stats")
                
        except Exception as e:
            logger.error(f"Stats command error: {e}")
            await message.reply_text(f"❌ Error getting stats: {str(e)[:100]}")
    
    @bot.on_message(filters.command("pending") & filters.user(getattr(config, 'ADMIN_IDS', [])))
    async def pending_payments_command(client, message):
        """Show pending payments"""
        try:
            # ✅ PREVENT DOUBLE REPLIES
            if not await bot_instance.should_reply(message.from_user.id, message.id):
                return
            
            if bot_instance.premium_system:
                pending = await bot_instance.premium_system.get_pending_payments_admin()
                
                if not pending:
                    await message.reply_text("✅ No pending payments!")
                    return
                
                text = f"⏳ **Pending Payments:** {len(pending)}\n\n"
                
                for i, payment in enumerate(pending[:10], 1):  # Show first 10
                    text += (
                        f"{i}. **ID:** `{payment['payment_id']}`\n"
                        f"   **User:** `{payment['user_id']}`\n"
                        f"   **Plan:** {payment['tier_name']}\n"
                        f"   **Amount:** ₹{payment['amount']}\n"
                        f"   **Screenshot:** {'✅ Sent' if payment['screenshot_sent'] else '❌ Not sent'}\n"
                        f"   **Time Left:** {payment['hours_left']} hours\n\n"
                    )
                
                if len(pending) > 10:
                    text += f"... and {len(pending) - 10} more pending payments\n\n"
                
                text += "Use `/approve <payment_id>` to approve payment."
                
                await message.reply_text(text, disable_web_page_preview=True)
            else:
                await message.reply_text("❌ Premium system not available")
                
        except Exception as e:
            logger.error(f"Pending payments command error: {e}")
            await message.reply_text(f"❌ Error: {str(e)[:100]}")
    
    @bot.on_message(filters.command("approve") & filters.user(getattr(config, 'ADMIN_IDS', [])))
    async def approve_payment_command(client, message):
        """Approve pending payment"""
        try:
            # ✅ PREVENT DOUBLE REPLIES
            if not await bot_instance.should_reply(message.from_user.id, message.id):
                return
            
            if len(message.command) < 2:
                await message.reply_text(
                    "❌ **Usage:** `/approve <payment_id>`\n\n"
                    "**Example:** `/approve PAY_ABC123DEF456`"
                )
                return
            
            payment_id = message.command[1].strip()
            
            if bot_instance.premium_system:
                success, result = await bot_instance.premium_system.approve_payment(
                    admin_id=message.from_user.id,
                    payment_id=payment_id
                )
                
                if success:
                    await message.reply_text(f"✅ {result}")
                    
                    # Notify user
                    try:
                        # Find user from payment
                        for pid, payment in bot_instance.premium_system.pending_payments.items():
                            if pid == payment_id:
                                user_id = payment['user_id']
                                plan_name = payment['tier_name']
                                
                                await client.send_message(
                                    user_id,
                                    f"🎉 **Payment Approved!** 🎉\n\n"
                                    f"Your payment for **{plan_name}** has been approved!\n\n"
                                    f"✅ **Status:** Premium Active\n"
                                    f"⭐ **Benefits:**\n"
                                    f"• No verification required\n"
                                    f"• Instant file access\n"
                                    f"• Priority support\n"
                                    f"⏰ **Files auto-delete after 5 minutes** (security)\n\n"
                                    f"🎬 **Enjoy unlimited downloads!**"
                                )
                                break
                    except:
                        pass
                else:
                    await message.reply_text(f"❌ {result}")
            else:
                await message.reply_text("❌ Premium system not available")
                
        except Exception as e:
            logger.error(f"Approve payment command error: {e}")
            await message.reply_text(f"❌ Error: {str(e)[:100]}")
    
    @bot.on_message(filters.command("reject") & filters.user(getattr(config, 'ADMIN_IDS', [])))
    async def reject_payment_command(client, message):
        """Reject pending payment"""
        try:
            # ✅ PREVENT DOUBLE REPLIES
            if not await bot_instance.should_reply(message.from_user.id, message.id):
                return
            
            if len(message.command) < 3:
                await message.reply_text(
                    "❌ **Usage:** `/reject <payment_id> <reason>`\n\n"
                    "**Example:** `/reject PAY_ABC123DEF456 Invalid screenshot`"
                )
                return
            
            payment_id = message.command[1].strip()
            reason = ' '.join(message.command[2:])
            
            if bot_instance.premium_system:
                success = await bot_instance.premium_system.reject_payment(
                    admin_id=message.from_user.id,
                    payment_id=payment_id,
                    reason=reason
                )
                
                if success:
                    await message.reply_text(f"✅ Payment {payment_id} rejected!\n**Reason:** {reason}")
                else:
                    await message.reply_text(f"❌ Failed to reject payment {payment_id}")
            else:
                await message.reply_text("❌ Premium system not available")
                
        except Exception as e:
            logger.error(f"Reject payment command error: {e}")
            await message.reply_text(f"❌ Error: {str(e)[:100]}")
    
    # ✅ FIXED: MAIN TEXT MESSAGE HANDLER
    @bot.on_message(filters.private & filters.text)
    async def handle_private_text(client, message):
        """Handle all private text messages"""
        user_id = message.from_user.id
        
        # ✅ PREVENT DOUBLE REPLIES
        if not await bot_instance.should_reply(user_id, message.id):
            return
        
        # Check if it's a command - commands are handled by separate handlers
        if message.text.startswith('/'):
            # Commands are handled by separate handlers, so just return
            return
        
        # Check if it's a file request pattern
        if re.match(r'^-?\d+_\d+(_\w+)?$', message.text.strip()):
            # It's a file request
            await handle_file_request(client, message, message.text.strip(), bot_instance)
            return
        
        # Check if it's a verification token
        if message.text.strip().startswith('verify_'):
            token = message.text.strip().replace('verify_', '', 1)
            await handle_verification_token(client, message, token, bot_instance)
            return
        
        # It's a regular message
        await handle_direct_message(client, message, bot_instance)
    
    # ✅ CALLBACK HANDLERS
    
    @bot.on_callback_query(filters.regex(r"^get_verified$"))
    async def get_verified_callback(client, callback_query):
        """Get verification link"""
        user_id = callback_query.from_user.id
        user_name = callback_query.from_user.first_name or "User"
        
        if bot_instance.verification_system:
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
                "⭐ **Premium users don't need verification**\n"
                "⏰ **Files auto-delete after 5 minutes**"
            )
            
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🔗 VERIFY NOW", url=verification_data['short_url'])],
                [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")],
                [InlineKeyboardButton("🔙 BACK", callback_data="back_to_start")]
            ])
            
            try:
                await callback_query.message.edit_text(
                    text=text,
                    reply_markup=keyboard,
                    disable_web_page_preview=True
                )
                await callback_query.answer()
            except:
                await callback_query.answer("Click VERIFY NOW button!", show_alert=True)
        else:
            await callback_query.answer("Verification system not available!", show_alert=True)
    
    @bot.on_callback_query(filters.regex(r"^back_to_start$"))
    async def back_to_start_callback(client, callback_query):
        user_name = callback_query.from_user.first_name or "User"
        
        text = (
            f"🎬 **Welcome back, {user_name}!**\n\n"
            f"Visit {config.WEBSITE_URL} to download movies.\n"
            "Click download button on website and file will appear here.\n\n"
            f"⏰ **Files auto-delete after 5 minutes** (security)"
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
            await callback_query.answer()
        except:
            await callback_query.answer("Already on home page!")
    
    @bot.on_callback_query(filters.regex(r"^buy_premium$"))
    async def buy_premium_callback(client, callback_query):
        """Show premium plans"""
        user_id = callback_query.from_user.id
        user_name = callback_query.from_user.first_name or "User"
        
        # Check if already premium
        if bot_instance.premium_system:
            is_premium = await bot_instance.premium_system.is_premium_user(user_id)
            if is_premium:
                details = await bot_instance.premium_system.get_subscription_details(user_id)
                
                text = (
                    f"⭐ **You're Already Premium!** ⭐\n\n"
                    f"**User:** {user_name}\n"
                    f"**Plan:** {details.get('tier_name', 'Premium')}\n"
                    f"**Days Left:** {details.get('days_remaining', 0)}\n"
                    f"**Status:** ✅ Active\n\n"
                    "Enjoy unlimited downloads without verification! 🎬\n"
                    f"⏰ **Files auto-delete after 5 minutes** (security)"
                )
                
                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton("🌐 OPEN WEBSITE", url=config.WEBSITE_URL)],
                    [InlineKeyboardButton("🔙 BACK", callback_data="back_to_start")]
                ])
                
                try:
                    await callback_query.message.edit_text(text, reply_markup=keyboard)
                    await callback_query.answer()
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
            "✅ Priority support\n"
            "⏰ **Files auto-delete after 5 minutes** (security measure)\n\n"
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
            await callback_query.answer()
        except:
            await callback_query.answer("Premium plans!", show_alert=True)
    
    @bot.on_callback_query(filters.regex(r"^plan_"))
    async def plan_selection_callback(client, callback_query):
        plan_type = callback_query.data.split('_')[1]
        user_id = callback_query.from_user.id
        
        if plan_type == "basic":
            tier = bot_instance.PremiumTier.BASIC
            plan_name = "Basic Plan"
        elif plan_type == "premium":
            tier = bot_instance.PremiumTier.PREMIUM
            plan_name = "Premium Plan"
        elif plan_type == "gold":
            tier = bot_instance.PremiumTier.GOLD
            plan_name = "Gold Plan"
        elif plan_type == "diamond":
            tier = bot_instance.PremiumTier.DIAMOND
            plan_name = "Diamond Plan"
        else:
            await callback_query.answer("Invalid plan!", show_alert=True)
            return
        
        if not bot_instance.premium_system:
            await callback_query.answer("Premium system not available!", show_alert=True)
            return
        
        # Initiate purchase
        payment_data = await bot_instance.premium_system.initiate_purchase(user_id, tier)
        
        if not payment_data:
            await callback_query.answer("Failed to initiate purchase!", show_alert=True)
            return
        
        # Get payment instructions
        instructions = await bot_instance.premium_system.get_payment_instructions_text(payment_data['payment_id'])
        
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("📸 SEND SCREENSHOT", callback_data=f"send_screenshot_{payment_data['payment_id']}")],
            [InlineKeyboardButton("🔙 BACK", callback_data="buy_premium")]
        ])
        
        try:
            await callback_query.message.edit_text(instructions, reply_markup=keyboard, disable_web_page_preview=True)
            await callback_query.answer()
        except:
            await callback_query.answer("Payment instructions!", show_alert=True)
    
    @bot.on_callback_query(filters.regex(r"^send_screenshot_"))
    async def send_screenshot_callback(client, callback_query):
        payment_id = callback_query.data.split('_')[2]
        
        text = (
            "📸 **Please send the payment screenshot now**\n\n"
            "1. Take a clear screenshot of the payment\n"
            "2. Send it to this chat\n"
            "3. Our admin will verify and activate your premium\n\n"
            f"**Payment ID:** `{payment_id}`\n"
            "⏰ Please send within 24 hours of payment\n"
            f"⏰ **Files auto-delete after 5 minutes** (security)"
        )
        
        await callback_query.answer("Please send screenshot now!", show_alert=True)
        
        # Send new message
        await callback_query.message.reply_text(text)
        
        # Try to delete the original callback message
        try:
            await callback_query.message.delete()
        except:
            pass
    
    # ✅ HANDLE SCREENSHOT MESSAGES - FIXED ADMIN NOTIFICATION
    @bot.on_message(filters.private & (filters.photo | filters.document))
    async def handle_screenshot(client, message):
        """Handle payment screenshots"""
        user_id = message.from_user.id
        
        # ✅ PREVENT DOUBLE REPLIES
        if not await bot_instance.should_reply(user_id, message.id):
            return
        
        # Check if it's likely a screenshot
        is_screenshot = False
        if message.photo:
            is_screenshot = True
        elif message.document and message.document.mime_type:
            if 'image' in message.document.mime_type:
                is_screenshot = True
            elif message.document.file_name and any(ext in message.document.file_name.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                is_screenshot = True
        
        if not is_screenshot:
            return  # Not a screenshot, ignore
        
        user_name = message.from_user.first_name or "User"
        
        if bot_instance.premium_system:
            # Find pending payment for this user
            payment_id = None
            for pid, payment in bot_instance.premium_system.pending_payments.items():
                if payment['user_id'] == user_id and payment['status'] == 'pending':
                    payment_id = pid
                    break
            
            if not payment_id:
                # No pending payment found
                await message.reply_text(
                    "❌ **No pending payment found!**\n\n"
                    "Please initiate a purchase first using /buy command.\n"
                    "Or send a screenshot only after making payment.\n\n"
                    f"⏰ **Files auto-delete after 5 minutes** (security)",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("💰 BUY PREMIUM", callback_data="buy_premium")],
                        [InlineKeyboardButton("🔙 BACK", callback_data="back_to_start")]
                    ])
                )
                return
            
            # Process screenshot in premium system
            success = await bot_instance.premium_system.process_payment_screenshot(
                user_id, 
                message.id
            )
            
            if success:
                # ✅ NOTIFY ADMINS
                admin_notified = await bot_instance.notify_admin_screenshot(user_id, message.id, payment_id)
                
                reply_text = (
                    "✅ **Screenshot received successfully!**\n\n"
                    f"**Payment ID:** `{payment_id}`\n"
                    f"**User:** {user_name}\n\n"
                )
                
                if admin_notified:
                    reply_text += (
                        "📨 **Admin notified!**\n"
                        "Our admin will verify your payment and activate your premium within 24 hours.\n\n"
                    )
                else:
                    reply_text += (
                        "⚠️ **Admin notification failed!**\n"
                        "Please contact admin manually with your payment ID.\n\n"
                    )
                
                reply_text += (
                    "Thank you for choosing SK4FiLM! 🎬\n"
                    "You will receive a confirmation message when activated.\n"
                    f"⏰ **Files auto-delete after 5 minutes** (security)"
                )
                
                await message.reply_text(
                    reply_text,
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("🔙 BACK TO START", callback_data="back_to_start")]
                    ])
                )
                
                logger.info(f"✅ Screenshot processed for user {user_id}, payment {payment_id}")
                
            else:
                await message.reply_text(
                    "❌ **Failed to process screenshot!**\n\n"
                    "Please try again or contact admin.\n"
                    f"Payment ID: `{payment_id}`\n\n"
                    f"⏰ **Files auto-delete after 5 minutes** (security)",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("🆘 CONTACT ADMIN", url="https://t.me/SKadminrobot")],
                        [InlineKeyboardButton("🔙 BACK", callback_data="back_to_start")]
                    ])
                )
        else:
            await message.reply_text(
                "❌ **Premium system not available**\n\n"
                "Please try again later or contact admin.\n\n"
                f"⏰ **Files auto-delete after 5 minutes** (security)",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🆘 CONTACT ADMIN", url="https://t.me/SKadminrobot")],
                    [InlineKeyboardButton("🔙 BACK", callback_data="back_to_start")]
                ])
            )
    
    logger.info("✅ Bot handlers setup complete with FIXED features")

# Utility function for file size formatting
def format_size(size_in_bytes):
    """Format file size in human-readable format"""
    if size_in_bytes is None or size_in_bytes == 0:
        return "Unknown"
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.1f} {unit}"
        size_in_bytes /= 1024.0
    return f"{size_in_bytes:.1f} PB"
