import asyncio
import logging
import re
import time
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, Any

# Pyrogram imports
try:
    from pyrogram import Client, filters
    from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton, Message, CallbackQuery
    from pyrogram.errors import FloodWait, BadRequest, MessageDeleteForbidden
    PYROGRAM_AVAILABLE = True
except ImportError:
    class Client: pass
    class filters:
        @staticmethod
        def command(cmd): return lambda x: x
        @staticmethod
        def private(): return lambda x: x
        @staticmethod
        def regex(pattern): return lambda x: x
        text = lambda x: x
    class InlineKeyboardMarkup:
        def __init__(self, buttons): pass
    class InlineKeyboardButton:
        def __init__(self, text, url=None, callback_data=None): pass
    class Message: pass
    class CallbackQuery: pass
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
        
        # Rate limiting
        self.user_request_times = defaultdict(list)
        self.processing_requests = {}
        self.verification_processing = {}
        
        # Screenshot processing
        self.screenshot_processing = {}
        self.pending_screenshots = {}
        
        # Initialize systems
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
            await self.setup_bot_handlers()
            
            # Start cleanup tasks
            if self.verification_system:
                asyncio.create_task(self.verification_system.start_cleanup_task())
            if self.premium_system:
                asyncio.create_task(self.premium_system.start_cleanup_task())
            if self.cache_manager:
                asyncio.create_task(self.cache_manager.start_cleanup_task())
            
            # Start auto-delete monitor
            asyncio.create_task(self._monitor_auto_delete())
            
            return True
            
        except Exception as e:
            logger.error(f"Bot initialization failed: {e}")
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
                
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    # ✅ AUTO-DELETE SYSTEM
    async def schedule_file_deletion(self, user_id: int, message_id: int, file_name: str, delete_after_minutes: int):
        """Schedule file deletion after specified minutes"""
        try:
            task_id = f"{user_id}_{message_id}"
            
            await asyncio.sleep(delete_after_minutes * 60)
            
            logger.info(f"⏰ Auto-delete time reached for message {message_id} (user {user_id})")
            
            try:
                await self.bot.delete_messages(user_id, message_id)
                logger.info(f"✅ Auto-deleted message {message_id} for user {user_id}")
                
                await self.send_deletion_notification(user_id, file_name, delete_after_minutes)
                
            except MessageDeleteForbidden:
                logger.warning(f"❌ Cannot delete message {message_id} - forbidden")
                await self.send_deletion_notification(user_id, file_name, delete_after_minutes, deleted=False)
            except Exception as e:
                logger.error(f"Error deleting message {message_id}: {e}")
                await self.send_deletion_notification(user_id, file_name, delete_after_minutes, deleted=False)
            
            self.auto_delete_tasks.pop(task_id, None)
            self.file_messages_to_delete.pop(task_id, None)
            
        except asyncio.CancelledError:
            logger.info(f"Auto-delete task cancelled for message {message_id}")
        except Exception as e:
            logger.error(f"Error in auto-delete task: {e}")
    
    async def send_deletion_notification(self, user_id: int, file_name: str, delete_after_minutes: int, deleted: bool = True):
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
                await asyncio.sleep(60)
                if self.auto_delete_tasks:
                    logger.info(f"📊 Auto-delete monitoring: {len(self.auto_delete_tasks)} active tasks")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-delete monitor error: {e}")
    
    # ✅ RATE LIMITING METHODS
    async def check_rate_limit(self, user_id, limit=3, window=60, request_type="file"):
        """Check if user is within rate limits"""
        now = time.time()
        key = f"{user_id}_{request_type}"
        
        self.user_request_times[key] = [
            t for t in self.user_request_times.get(key, []) 
            if now - t < window
        ]
        
        if len(self.user_request_times[key]) >= limit:
            logger.warning(f"⚠️ Rate limit exceeded for user {user_id} ({request_type})")
            return False
        
        self.user_request_times[key].append(now)
        return True
    
    async def is_request_duplicate(self, user_id, request_data, request_type="file"):
        """Check if this is a duplicate request"""
        request_hash = f"{user_id}_{request_type}_{hash(request_data)}"
        
        if request_type == "verification":
            processing_dict = self.verification_processing
        else:
            processing_dict = self.processing_requests
        
        if request_hash in processing_dict:
            if time.time() - processing_dict[request_hash] < 30:
                return True
        
        processing_dict[request_hash] = time.time()
        return False
    
    async def clear_processing_request(self, user_id, request_data, request_type="file"):
        """Clear from processing requests"""
        request_hash = f"{user_id}_{request_type}_{hash(request_data)}"
        
        if request_type == "verification":
            self.verification_processing.pop(request_hash, None)
        else:
            self.processing_requests.pop(request_hash, None)
    
    # ✅ SCREENSHOT PROCESSING
    async def notify_admins_about_screenshot(self, user_id, message_id, screenshot_message=None):
        """Notify all admins about a new screenshot"""
        try:
            if not getattr(self.config, 'ADMIN_IDS', []):
                logger.warning("No admin IDs configured")
                return False
            
            # Get user info
            try:
                user = await self.bot.get_users(user_id)
                user_name = f"{user.first_name or ''} {user.last_name or ''}".strip() or f"User {user_id}"
                username = f"@{user.username}" if user.username else "No username"
            except:
                user_name = f"User {user_id}"
                username = "Unknown"
            
            # Get screenshot info
            screenshot_info = ""
            if screenshot_message:
                if screenshot_message.photo:
                    file_size = screenshot_message.photo.file_size or "Unknown"
                    screenshot_info = f"📸 Photo - Size: {format_size(file_size)}"
                elif screenshot_message.document:
                    file_name = screenshot_message.document.file_name or "screenshot"
                    file_size = screenshot_message.document.file_size or "Unknown"
                    screenshot_info = f"📄 Document: {file_name} - Size: {format_size(file_size)}"
            
            # Get payment info
            payment_info = ""
            if user_id in self.pending_screenshots:
                payment_data = self.pending_screenshots[user_id]
                payment_info = (
                    f"💰 **Payment ID:** `{payment_data.get('payment_id', 'N/A')}`\n"
                    f"📋 **Plan:** {payment_data.get('plan_name', 'Unknown')}\n"
                    f"💵 **Amount:** ₹{payment_data.get('amount', 0)}\n"
                )
            
            # Create notification
            notification_text = (
                f"📸 **NEW PAYMENT SCREENSHOT RECEIVED** 📸\n\n"
                f"👤 **User:** {user_name}\n"
                f"🆔 **ID:** `{user_id}`\n"
                f"👤 **Username:** {username}\n"
                f"🕐 **Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                f"{payment_info}"
                f"📁 **File Info:** {screenshot_info}\n"
                f"📨 **Message ID:** `{message_id}`\n\n"
                f"**Quick Actions:**\n"
                f"✅ `/approve {user_id}` - Approve payment\n"
                f"❌ `/reject {user_id} <reason>` - Reject payment\n"
                f"👁️ `/checkpremium {user_id}` - Check user status\n"
                f"📋 `/pending` - View all pending"
            )
            
            keyboard = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("✅ APPROVE", callback_data=f"admin_approve_{user_id}"),
                    InlineKeyboardButton("❌ REJECT", callback_data=f"admin_reject_{user_id}")
                ],
                [
                    InlineKeyboardButton("👤 CHECK USER", callback_data=f"admin_check_{user_id}"),
                    InlineKeyboardButton("📋 PENDING LIST", callback_data="admin_pending")
                ]
            ])
            
            # Send to all admins
            for admin_id in self.config.ADMIN_IDS:
                try:
                    if screenshot_message:
                        await screenshot_message.forward(admin_id)
                    
                    await self.bot.send_message(
                        admin_id,
                        notification_text,
                        reply_markup=keyboard
                    )
                    
                    logger.info(f"✅ Screenshot notification sent to admin {admin_id}")
                    
                except Exception as admin_error:
                    logger.error(f"Failed to notify admin {admin_id}: {admin_error}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error notifying admins: {e}")
            return False
    
    async def process_screenshot_message(self, client, message):
        """Process screenshot message from user"""
        user_id = message.from_user.id
        
        # Check if already processing
        if user_id in self.screenshot_processing:
            processing_time = time.time() - self.screenshot_processing[user_id]['timestamp']
            if processing_time < 30:
                return
        
        # Check if user has pending payment
        has_pending = False
        payment_data = {}
        
        if self.premium_system and hasattr(self.premium_system, 'pending_payments'):
            for payment_id, payment in self.premium_system.pending_payments.items():
                if payment['user_id'] == user_id:
                    has_pending = True
                    payment_data = payment
                    break
        
        if not has_pending:
            await message.reply_text(
                "❌ **No Pending Payment Found**\n\n"
                "You don't have any pending payments.\n"
                "Please purchase a premium plan first using /buy command.",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("💰 BUY PREMIUM", callback_data="buy_premium")],
                    [InlineKeyboardButton("🔙 BACK", callback_data="back_to_start")]
                ])
            )
            return
        
        # Store payment data
        self.pending_screenshots[user_id] = payment_data
        
        # Mark as processing
        self.screenshot_processing[user_id] = {
            'message_id': message.id,
            'timestamp': time.time(),
            'processed': False
        }
        
        # Send processing message
        processing_msg = await message.reply_text(
            f"📸 **Processing your screenshot...**\n\n"
            f"**Payment ID:** `{payment_data.get('payment_id', 'N/A')}`\n"
            f"**Plan:** {payment_data.get('tier_name', 'Unknown')}\n"
            f"**Amount:** ₹{payment_data.get('amount', 0)}\n\n"
            f"⏳ Notifying admins..."
        )
        
        # Notify all admins
        notification_sent = await self.notify_admins_about_screenshot(
            user_id, 
            message.id,
            screenshot_message=message
        )
        
        if notification_sent:
            success_text = (
                f"✅ **Screenshot received and forwarded to admins!**\n\n"
                f"**Payment ID:** `{payment_data.get('payment_id', 'N/A')}`\n"
                f"**Plan:** {payment_data.get('tier_name', 'Unknown')}\n"
                f"**Amount:** ₹{payment_data.get('amount', 0)}\n\n"
                f"📋 **Status:** Under review by admin\n"
                f"⏰ **Estimated time:** Within 24 hours\n\n"
                f"Thank you for choosing SK4FiLM! 🎬"
            )
            
            try:
                await processing_msg.edit_text(
                    success_text,
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("🔙 BACK TO START", callback_data="back_to_start")]
                    ])
                )
            except:
                await message.reply_text(
                    success_text,
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("🔙 BACK TO START", callback_data="back_to_start")]
                    ])
                )
            
            self.screenshot_processing[user_id]['processed'] = True
        else:
            error_text = "❌ **Failed to notify admins**\n\nPlease try again or contact support."
            
            try:
                await processing_msg.edit_text(error_text)
            except:
                await message.reply_text(error_text)
            
            self.screenshot_processing.pop(user_id, None)
    
    # ✅ FILE SENDING FUNCTION - ORIGINAL WORKING VERSION
    async def send_file_to_user(self, client, user_id, file_message, quality="480p"):
        """Send file to user with verification check"""
        try:
            # Check user status
            user_status = "Checking..."
            status_icon = "⏳"
            can_download = False
            
            # Check if user is admin
            is_admin = user_id in getattr(self.config, 'ADMIN_IDS', [])
            
            if is_admin:
                can_download = True
                user_status = "Admin User 👑"
                status_icon = "👑"
            elif self.premium_system:
                # Check premium status
                is_premium = await self.premium_system.is_premium_user(user_id)
                if is_premium:
                    can_download = True
                    user_status = "Premium User ⭐"
                    status_icon = "⭐"
                else:
                    # Check verification status
                    if self.verification_system:
                        is_verified, _ = await self.verification_system.check_user_verified(
                            user_id, self.premium_system
                        )
                        if is_verified:
                            can_download = True
                            user_status = "Verified User ✅"
                            status_icon = "✅"
                        else:
                            # User needs verification
                            verification_data = await self.verification_system.create_verification_link(user_id)
                            return False, {
                                'message': f"🔒 **Access Restricted**\n\n❌ You need to verify or purchase premium to download files.",
                                'buttons': [
                                    [InlineKeyboardButton("🔗 VERIFY NOW", url=verification_data['short_url'])],
                                    [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")]
                                ]
                            }, 0
                    else:
                        return False, {
                            'message': "❌ Verification system not available.",
                            'buttons': []
                        }, 0
            else:
                return False, {
                    'message': "❌ System temporarily unavailable.",
                    'buttons': []
                }, 0
            
            if not can_download:
                return False, {
                    'message': "❌ Access denied.",
                    'buttons': []
                }, 0
            
            # Get file details
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
                    'message': "❌ No downloadable file found",
                    'buttons': []
                }, 0
            
            if not file_id:
                return False, {
                    'message': "❌ File ID is empty.",
                    'buttons': []
                }, 0
            
            # Get auto-delete time
            auto_delete_minutes = getattr(self.config, 'AUTO_DELETE_TIME', 15)
            
            # Create caption
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
                            [InlineKeyboardButton("🌐 OPEN WEBSITE", url=self.config.WEBSITE_URL)]
                        ])
                    )
                else:
                    sent = await client.send_video(
                        user_id,
                        file_id,
                        caption=file_caption,
                        reply_markup=InlineKeyboardMarkup([
                            [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")],
                            [InlineKeyboardButton("🌐 OPEN WEBSITE", url=self.config.WEBSITE_URL)]
                        ])
                    )
                
                logger.info(f"✅ File sent to {user_status} user {user_id}: {file_name}")
                
                # Schedule auto-delete
                if auto_delete_minutes > 0:
                    task_id = f"{user_id}_{sent.id}"
                    
                    if task_id in self.auto_delete_tasks:
                        self.auto_delete_tasks[task_id].cancel()
                    
                    delete_task = asyncio.create_task(
                        self.schedule_file_deletion(user_id, sent.id, file_name, auto_delete_minutes)
                    )
                    self.auto_delete_tasks[task_id] = delete_task
                    
                    logger.info(f"⏰ Auto-delete scheduled for message {sent.id}")
                
                return True, {
                    'success': True,
                    'file_name': file_name,
                    'file_size': file_size,
                    'quality': quality,
                    'user_status': user_status,
                    'auto_delete_minutes': auto_delete_minutes,
                    'message_id': sent.id,
                }, file_size
                
            except BadRequest as e:
                if "MEDIA_EMPTY" in str(e) or "FILE_REFERENCE_EXPIRED" in str(e):
                    logger.error(f"❌ File reference expired: {e}")
                    return False, {
                        'message': "❌ File reference expired, please try download again",
                        'buttons': []
                    }, 0
                else:
                    raise e
                    
        except FloodWait as e:
            logger.warning(f"⏳ Flood wait: {e.value}s")
            return False, {
                'message': f"⏳ Please wait {e.value} seconds",
                'buttons': []
            }, 0
        except Exception as e:
            logger.error(f"File sending error: {e}")
            return False, {
                'message': f"❌ Error: {str(e)}",
                'buttons': []
            }, 0
    
    # ✅ HANDLER METHODS
    async def handle_verification_token(self, client, message, token):
        """Handle verification token"""
        try:
            user_id = message.from_user.id
            user_name = message.from_user.first_name or "User"
            
            if not await self.check_rate_limit(user_id, limit=5, window=60, request_type="verification"):
                await message.reply_text("⚠️ **Rate Limit**\n\nToo many verification attempts.")
                return
            
            if await self.is_request_duplicate(user_id, token, request_type="verification"):
                await message.reply_text("⏳ **Already Processing**")
                return
            
            logger.info(f"Processing verification token for user {user_id}")
            
            if not self.verification_system:
                await message.reply_text("❌ Verification system not available.")
                return
            
            processing_msg = await message.reply_text("🔐 **Verifying your access...**")
            
            # Verify the token
            is_valid, verified_user_id, message_text = await self.verification_system.verify_user_token(token)
            
            if is_valid:
                success_text = (
                    f"✅ **Verification Successful!** ✅\n\n"
                    f"**Welcome, {user_name}!** 🎉\n\n"
                    f"🎬 **You now have access to:**\n"
                    f"• File downloads for 6 hours\n"
                    f"• All quality options\n"
                    f"• Unlimited downloads\n\n"
                    f"Visit {self.config.WEBSITE_URL} to download movies!"
                )
                
                success_keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton("🌐 OPEN WEBSITE", url=self.config.WEBSITE_URL)],
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
                
            else:
                error_text = f"❌ **Verification Failed**\n\n**Reason:** {message_text}"
                
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
            
            await self.clear_processing_request(user_id, token, request_type="verification")
                
        except Exception as e:
            logger.error(f"Verification token handling error: {e}")
            await message.reply_text("❌ **Verification Error**")
    
    async def handle_file_request(self, client, message, file_text):
        """Handle file download request"""
        try:
            user_id = message.from_user.id
            
            # Rate limit check
            if not await self.check_rate_limit(user_id, limit=3, window=60, request_type="file"):
                await message.reply_text("⚠️ **Rate Limit Exceeded**")
                return
            
            # Duplicate check
            if await self.is_request_duplicate(user_id, file_text, request_type="file"):
                await message.reply_text("⏳ **Already Processing**")
                return
            
            # Clean the text
            clean_text = file_text.strip()
            logger.info(f"Processing file request: {clean_text}")
            
            # Parse file request
            if clean_text.startswith('/start'):
                clean_text = clean_text.replace('/start', '').strip()
            
            parts = clean_text.split('_')
            
            if len(parts) < 2:
                await message.reply_text("❌ **Invalid file format**")
                await self.clear_processing_request(user_id, file_text, request_type="file")
                return
            
            # Parse channel ID
            channel_str = parts[0].strip()
            try:
                if channel_str.startswith('--'):
                    channel_id = int(channel_str[1:])
                else:
                    channel_id = int(channel_str)
            except ValueError:
                await message.reply_text("❌ **Invalid channel ID**")
                await self.clear_processing_request(user_id, file_text, request_type="file")
                return
            
            # Parse message ID
            try:
                message_id = int(parts[1].strip())
            except ValueError:
                await message.reply_text("❌ **Invalid message ID**")
                await self.clear_processing_request(user_id, file_text, request_type="file")
                return
            
            # Get quality
            quality = parts[2].strip() if len(parts) > 2 else "480p"
            
            logger.info(f"Parsed: channel={channel_id}, message={message_id}, quality={quality}")
            
            # Send processing message
            processing_msg = await message.reply_text(
                f"⏳ **Preparing your file...**\n\n"
                f"📹 **Quality:** {quality}"
            )
            
            # Get file from channel
            file_message = None
            max_retries = 2
            
            for attempt in range(max_retries):
                try:
                    # Try user client first
                    if self.user_client and self.user_session_ready:
                        try:
                            file_message = await self.user_client.get_messages(channel_id, message_id)
                            logger.info(f"Got file via user client")
                            break
                        except:
                            pass
                    
                    # Try bot client
                    try:
                        file_message = await client.get_messages(channel_id, message_id)
                        logger.info(f"Got file via bot client")
                        break
                    except:
                        pass
                        
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)
                        
                except Exception as e:
                    logger.error(f"Attempt {attempt+1} failed: {e}")
            
            if not file_message:
                await processing_msg.edit_text("❌ **File not found**")
                await self.clear_processing_request(user_id, file_text, request_type="file")
                return
            
            if not file_message.document and not file_message.video:
                await processing_msg.edit_text("❌ **Not a downloadable file**")
                await self.clear_processing_request(user_id, file_text, request_type="file")
                return
            
            # Send file to user
            success, result_data, file_size = await self.send_file_to_user(
                client, user_id, file_message, quality
            )
            
            if success:
                try:
                    await processing_msg.delete()
                except:
                    pass
                
                # Record download
                if self.premium_system:
                    await self.premium_system.record_download(user_id, file_size, quality)
                
            else:
                error_text = result_data['message']
                error_buttons = result_data.get('buttons', [])
                
                if error_buttons:
                    await processing_msg.edit_text(
                        error_text,
                        reply_markup=InlineKeyboardMarkup(error_buttons),
                        disable_web_page_preview=True
                    )
                else:
                    await processing_msg.edit_text(error_text)
            
            await self.clear_processing_request(user_id, file_text, request_type="file")
            
        except Exception as e:
            logger.error(f"File request handling error: {e}")
            await message.reply_text("❌ **Download Error**")
            await self.clear_processing_request(user_id, file_text, request_type="file")
    
    # ✅ SETUP BOT HANDLERS - SIMPLIFIED AND WORKING
    async def setup_bot_handlers(self):
        """Setup bot handlers - SIMPLIFIED VERSION"""
        
        # ✅ 1. START COMMAND
        @self.bot.on_message(filters.command("start"))
        async def start_handler(client, message):
            if len(message.command) > 1:
                start_text = ' '.join(message.command[1:])
                
                # Check if verification token
                if start_text.startswith('verify_'):
                    token = start_text.replace('verify_', '', 1).strip()
                    await self.handle_verification_token(client, message, token)
                    return
                else:
                    # Treat as file request
                    await self.handle_file_request(client, message, start_text)
                    return
            
            # Regular start command
            user_name = message.from_user.first_name or "User"
            welcome_text = f"🎬 **Welcome to SK4FiLM, {user_name}!**\n\n🌐 **Website:** {self.config.WEBSITE_URL}"
            
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🌐 OPEN WEBSITE", url=self.config.WEBSITE_URL)],
                [InlineKeyboardButton("⭐ GET PREMIUM", callback_data="buy_premium")]
            ])
            
            await message.reply_text(welcome_text, reply_markup=keyboard, disable_web_page_preview=True)
        
        # ✅ 2. FILE REQUEST HANDLER (PATTERN MATCHING)
        @self.bot.on_message(filters.private & filters.regex(r'^-?\d+_\d+'))
        async def file_handler(client, message):
            file_text = message.text.strip()
            await self.handle_file_request(client, message, file_text)
        
        # ✅ 3. SCREENSHOT HANDLER
        @self.bot.on_message(filters.private & (filters.photo | filters.document))
        async def screenshot_handler(client, message):
            # Simple screenshot detection
            if message.photo:
                await self.process_screenshot_message(client, message)
            elif message.document:
                # Check if it's an image
                mime_type = message.document.mime_type or ""
                file_name = message.document.file_name or ""
                
                image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']
                image_mimes = ['image/', 'application/image']
                
                if any(mime in mime_type.lower() for mime in image_mimes):
                    await self.process_screenshot_message(client, message)
                elif any(ext in file_name.lower() for ext in image_extensions):
                    await self.process_screenshot_message(client, message)
        
        # ✅ 4. NON-COMMAND TEXT HANDLER (MUST BE LAST)
        @self.bot.on_message(filters.private & filters.text & ~filters.command)
        async def text_handler(client, message):
            text = message.text.strip()
            
            # Skip if it's a file pattern
            if re.match(r'^-?\d+_\d+', text):
                return
            
            # Skip if it's a verification token
            if text.startswith('verify_'):
                return
            
            user_name = message.from_user.first_name or "User"
            
            website_text = (
                f"🌐 **Welcome to SK4FiLM, {user_name}!** 🌐\n\n"
                f"**To download files:**\n"
                f"1. Visit our website: {self.config.WEBSITE_URL}\n"
                f"2. Search for movies/TV shows\n"
                f"3. Click download button\n"
                f"4. File will appear here automatically\n\n"
                f"**Need help?** Use /start for main menu"
            )
            
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🌐 OPEN WEBSITE", url=self.config.WEBSITE_URL)],
                [InlineKeyboardButton("💰 BUY PREMIUM", callback_data="buy_premium")]
            ])
            
            await message.reply_text(website_text, reply_markup=keyboard, disable_web_page_preview=True)
        
        # ✅ 5. CALLBACK HANDLERS
        @self.bot.on_callback_query(filters.regex(r"^get_verified$"))
        async def get_verified_callback(client, callback_query):
            user_id = callback_query.from_user.id
            
            if self.verification_system:
                verification_data = await self.verification_system.create_verification_link(user_id)
                
                text = f"🔗 **Verification Required**\n\nClick: {verification_data['short_url']}"
                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔗 VERIFY NOW", url=verification_data['short_url'])],
                    [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")]
                ])
                
                try:
                    await callback_query.message.edit_text(text, reply_markup=keyboard)
                except:
                    await callback_query.answer("Click VERIFY NOW!", show_alert=True)
        
        @self.bot.on_callback_query(filters.regex(r"^back_to_start$"))
        async def back_to_start_callback(client, callback_query):
            user_name = callback_query.from_user.first_name or "User"
            text = f"🎬 **Welcome back, {user_name}!**\n\nVisit {self.config.WEBSITE_URL} to download."
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🌐 OPEN WEBSITE", url=self.config.WEBSITE_URL)]
            ])
            
            try:
                await callback_query.message.edit_text(text, reply_markup=keyboard)
            except:
                pass
        
        @self.bot.on_callback_query(filters.regex(r"^buy_premium$"))
        async def buy_premium_callback(client, callback_query):
            user_name = callback_query.from_user.first_name or "User"
            text = f"⭐ **Premium Plans - {user_name}** ⭐\n\n**Benefits:**\n✅ No verification\n✅ All quality\n✅ Unlimited downloads"
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🥉 BASIC (₹99)", callback_data="plan_basic")],
                [InlineKeyboardButton("🥈 PREMIUM (₹199)", callback_data="plan_premium")],
                [InlineKeyboardButton("🔙 BACK", callback_data="back_to_start")]
            ])
            
            try:
                await callback_query.message.edit_text(text, reply_markup=keyboard)
            except:
                await callback_query.answer("Premium plans!", show_alert=True)
        
        logger.info("✅ Bot handlers setup complete")
