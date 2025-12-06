import asyncio
import logging
import secrets
import re
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
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
        self.file_messages_to_delete = {}  # Track files to delete
        
        # Rate limiting and deduplication
        self.user_request_times = defaultdict(list)
        self.processing_requests = {}
        self.verification_processing = {}
        
        # Screenshot processing
        self.screenshot_processing = {}
        self.pending_screenshots = {}
        
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
                
            # Stop cleanup tasks
            if self.verification_system:
                await self.verification_system.stop_cleanup_task()
            if self.premium_system:
                await self.premium_system.stop_cleanup_task()
            if self.cache_manager:
                await self.cache_manager.stop()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    # ✅ AUTO-DELETE SYSTEM
    async def schedule_file_deletion(self, user_id: int, message_id: int, file_name: str, delete_after_minutes: int):
        """Schedule file deletion after specified minutes"""
        try:
            task_id = f"{user_id}_{message_id}"
            
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
                await asyncio.sleep(60)  # Check every minute
                
                # Log active tasks
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
    
    # ✅ SCREENSHOT PROCESSING METHODS
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
            
            # Get screenshot info if available
            screenshot_info = ""
            if screenshot_message:
                if screenshot_message.photo:
                    file_size = screenshot_message.photo.file_size or "Unknown"
                    screenshot_info = f"📸 Photo - Size: {format_size(file_size)}"
                elif screenshot_message.document:
                    file_name = screenshot_message.document.file_name or "screenshot"
                    file_size = screenshot_message.document.file_size or "Unknown"
                    screenshot_info = f"📄 Document: {file_name} - Size: {format_size(file_size)}"
            
            # Get payment info if available
            payment_info = ""
            if user_id in self.pending_screenshots:
                payment_data = self.pending_screenshots[user_id]
                payment_info = (
                    f"💰 **Payment ID:** `{payment_data.get('payment_id', 'N/A')}`\n"
                    f"📋 **Plan:** {payment_data.get('plan_name', 'Unknown')}\n"
                    f"💵 **Amount:** ₹{payment_data.get('amount', 0)}\n"
                )
            
            # Create notification text
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
            
            # Create inline buttons for quick actions
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
            admin_notifications = []
            for admin_id in self.config.ADMIN_IDS:
                try:
                    # Forward the screenshot message first
                    if screenshot_message:
                        forwarded = await screenshot_message.forward(admin_id)
                        screenshot_msg_id = forwarded.id
                    else:
                        screenshot_msg_id = message_id
                    
                    # Send notification with buttons
                    notification_msg = await self.bot.send_message(
                        admin_id,
                        notification_text,
                        reply_markup=keyboard
                    )
                    
                    admin_notifications.append({
                        'admin_id': admin_id,
                        'notification_id': notification_msg.id,
                        'screenshot_id': screenshot_msg_id
                    })
                    
                    logger.info(f"✅ Screenshot notification sent to admin {admin_id}")
                    
                except Exception as admin_error:
                    logger.error(f"Failed to notify admin {admin_id}: {admin_error}")
            
            # Store notification info
            self.screenshot_processing[user_id] = {
                'message_id': message_id,
                'notifications': admin_notifications,
                'timestamp': time.time(),
                'processed': False
            }
            
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
            if processing_time < 30:  # 30 seconds cooldown
                logger.info(f"Screenshot already processing for user {user_id}")
                return
        
        # Check if user has pending payment
        has_pending = False
        payment_data = {}
        
        if self.premium_system and hasattr(self.premium_system, 'pending_payments'):
            # Find user's pending payment
            for payment_id, payment in self.premium_system.pending_payments.items():
                if payment['user_id'] == user_id:
                    has_pending = True
                    payment_data = payment
                    break
        
        if not has_pending:
            # No pending payment found
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
        
        # Store payment data for reference
        self.pending_screenshots[user_id] = payment_data
        
        # Mark as processing
        self.screenshot_processing[user_id] = {
            'message_id': message.id,
            'timestamp': time.time(),
            'processed': False
        }
        
        # Send processing message to user
        processing_msg = await message.reply_text(
            f"📸 **Processing your screenshot...**\n\n"
            f"**Payment ID:** `{payment_data.get('payment_id', 'N/A')}`\n"
            f"**Plan:** {payment_data.get('tier_name', 'Unknown')}\n"
            f"**Amount:** ₹{payment_data.get('amount', 0)}\n\n"
            f"⏳ Notifying admins... This may take a moment."
        )
        
        # Notify all admins
        notification_sent = await self.notify_admins_about_screenshot(
            user_id, 
            message.id,
            screenshot_message=message
        )
        
        if notification_sent:
            # Update user message
            success_text = (
                f"✅ **Screenshot received and forwarded to admins!**\n\n"
                f"**Payment ID:** `{payment_data.get('payment_id', 'N/A')}`\n"
                f"**Plan:** {payment_data.get('tier_name', 'Unknown')}\n"
                f"**Amount:** ₹{payment_data.get('amount', 0)}\n\n"
                f"📋 **Status:** Under review by admin\n"
                f"⏰ **Estimated time:** Within 24 hours\n\n"
                f"💬 **What happens next?**\n"
                f"1. Admin reviews your screenshot\n"
                f"2. You'll receive confirmation message\n"
                f"3. Premium access will be activated\n\n"
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
            
            # Mark as processed
            self.screenshot_processing[user_id]['processed'] = True
            
            # Update premium system if available
            if self.premium_system and hasattr(self.premium_system, 'update_payment_status'):
                await self.premium_system.update_payment_status(
                    payment_data['payment_id'],
                    {'screenshot_sent': True, 'screenshot_message_id': message.id}
                )
        else:
            # Failed to notify admins
            error_text = (
                f"❌ **Failed to notify admins**\n\n"
                f"Please try again or contact support.\n"
                f"You can also send the screenshot directly to admin."
            )
            
            try:
                await processing_msg.edit_text(error_text)
            except:
                await message.reply_text(error_text)
            
            # Remove from processing
            self.screenshot_processing.pop(user_id, None)
    
    async def handle_admin_approval(self, admin_id, user_id, reason=None):
        """Handle admin approval of payment"""
        try:
            if user_id not in self.pending_screenshots:
                return False, "No pending payment found for this user"
            
            payment_data = self.pending_screenshots[user_id]
            payment_id = payment_data.get('payment_id')
            
            if not payment_id:
                return False, "Payment ID not found"
            
            # Process approval through premium system
            if self.premium_system and hasattr(self.premium_system, 'approve_payment'):
                success, result = await self.premium_system.approve_payment(
                    admin_id=admin_id,
                    payment_id=payment_id,
                    reason=reason or "Approved via admin notification"
                )
                
                if success:
                    # Notify user
                    try:
                        user = await self.bot.get_users(user_id)
                        user_name = user.first_name or "User"
                    except:
                        user_name = "User"
                    
                    await self.bot.send_message(
                        user_id,
                        f"🎉 **Payment Approved!** 🎉\n\n"
                        f"**Hello {user_name}!**\n"
                        f"Your payment for **{payment_data.get('tier_name', 'Premium')}** has been approved!\n\n"
                        f"✅ **Status:** Premium Active\n"
                        f"📅 **Plan:** {payment_data.get('tier_name', 'Premium')}\n"
                        f"💰 **Amount:** ₹{payment_data.get('amount', 0)}\n\n"
                        f"⭐ **Benefits Activated:**\n"
                        f"• No verification required\n"
                        f"• Instant file access\n"
                        f"• Priority support\n\n"
                        f"🎬 **Enjoy unlimited downloads!**",
                        reply_markup=InlineKeyboardMarkup([
                            [InlineKeyboardButton("🌐 OPEN WEBSITE", url=self.config.WEBSITE_URL)],
                            [InlineKeyboardButton("📥 DOWNLOAD NOW", callback_data="back_to_start")]
                        ])
                    )
                    
                    # Cleanup
                    self.pending_screenshots.pop(user_id, None)
                    self.screenshot_processing.pop(user_id, None)
                    
                    return True, f"Payment approved for user {user_id}"
                
                return False, result
            
            return False, "Premium system not available"
            
        except Exception as e:
            logger.error(f"Admin approval error: {e}")
            return False, f"Error: {str(e)}"
    
    async def handle_admin_rejection(self, admin_id, user_id, reason):
        """Handle admin rejection of payment"""
        try:
            if user_id not in self.pending_screenshots:
                return False, "No pending payment found for this user"
            
            payment_data = self.pending_screenshots[user_id]
            payment_id = payment_data.get('payment_id')
            
            if not payment_id:
                return False, "Payment ID not found"
            
            # Process rejection through premium system
            if self.premium_system and hasattr(self.premium_system, 'reject_payment'):
                success = await self.premium_system.reject_payment(
                    admin_id=admin_id,
                    payment_id=payment_id,
                    reason=reason
                )
                
                if success:
                    # Notify user
                    try:
                        user = await self.bot.get_users(user_id)
                        user_name = user.first_name or "User"
                    except:
                        user_name = "User"
                    
                    rejection_text = (
                        f"❌ **Payment Rejected** ❌\n\n"
                        f"**Hello {user_name},**\n"
                        f"Your payment for **{payment_data.get('tier_name', 'Premium')}** was rejected.\n\n"
                        f"📋 **Reason:** {reason}\n\n"
                        f"🔄 **What to do next?**\n"
                        f"1. Check your payment details\n"
                        f"2. Ensure screenshot is clear\n"
                        f"3. Make sure payment is completed\n"
                        f"4. Try again with /buy command\n\n"
                        f"💬 **Need help?** Contact support."
                    )
                    
                    await self.bot.send_message(
                        user_id,
                        rejection_text,
                        reply_markup=InlineKeyboardMarkup([
                            [InlineKeyboardButton("💰 TRY AGAIN", callback_data="buy_premium")],
                            [InlineKeyboardButton("🆘 HELP", callback_data="back_to_start")]
                        ])
                    )
                    
                    # Cleanup
                    self.pending_screenshots.pop(user_id, None)
                    self.screenshot_processing.pop(user_id, None)
                    
                    return True, f"Payment rejected for user {user_id}"
                
                return False, "Failed to reject payment"
            
            return False, "Premium system not available"
            
        except Exception as e:
            logger.error(f"Admin rejection error: {e}")
            return False, f"Error: {str(e)}"

    # ✅ ORIGINAL FILE SENDING FUNCTION (FIXED)
    async def send_file_to_user(self, client, user_id, file_message, quality="480p"):
        """Send file to user with verification check - ORIGINAL WORKING VERSION"""
        try:
            # ✅ FIRST CHECK: Verify user is premium/verified/admin
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
            auto_delete_minutes = getattr(self.config, 'AUTO_DELETE_TIME', 15)
            
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
                
                # ✅ Schedule auto-delete
                if auto_delete_minutes > 0:
                    task_id = f"{user_id}_{sent.id}"
                    
                    # Cancel any existing task for this user
                    if task_id in self.auto_delete_tasks:
                        self.auto_delete_tasks[task_id].cancel()
                    
                    # Create new auto-delete task
                    delete_task = asyncio.create_task(
                        self.schedule_file_deletion(user_id, sent.id, file_name, auto_delete_minutes)
                    )
                    self.auto_delete_tasks[task_id] = delete_task
                    self.file_messages_to_delete[task_id] = {
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
                                    [InlineKeyboardButton("🌐 OPEN WEBSITE", url=self.config.WEBSITE_URL)]
                                ])
                            )
                        else:
                            sent = await client.send_video(
                                user_id, 
                                new_file_id,
                                caption=file_caption,
                                reply_markup=InlineKeyboardMarkup([
                                    [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")],
                                    [InlineKeyboardButton("🌐 OPEN WEBSITE", url=self.config.WEBSITE_URL)]
                                ])
                            )
                        
                        logger.info(f"✅ File sent with refreshed reference to {user_id}")
                        
                        # ✅ Schedule auto-delete for refreshed file
                        if auto_delete_minutes > 0:
                            task_id = f"{user_id}_{sent.id}"
                            
                            # Cancel any existing task for this user
                            if task_id in self.auto_delete_tasks:
                                self.auto_delete_tasks[task_id].cancel()
                            
                            # Create new auto-delete task
                            delete_task = asyncio.create_task(
                                self.schedule_file_deletion(user_id, sent.id, file_name, auto_delete_minutes)
                            )
                            self.auto_delete_tasks[task_id] = delete_task
                            self.file_messages_to_delete[task_id] = {
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

    # ✅ HANDLER METHODS
    async def handle_verification_token(self, client, message, token):
        """Handle verification token from /start verify_<token>"""
        try:
            user_id = message.from_user.id
            user_name = message.from_user.first_name or "User"
            
            # ✅ VERIFICATION RATE LIMIT CHECK
            if not await self.check_rate_limit(user_id, limit=5, window=60, request_type="verification"):
                await message.reply_text(
                    "⚠️ **Verification Rate Limit**\n\n"
                    "Too many verification attempts. Please wait 60 seconds."
                )
                return
            
            # ✅ DUPLICATE VERIFICATION CHECK
            if await self.is_request_duplicate(user_id, token, request_type="verification"):
                logger.warning(f"⚠️ Duplicate verification ignored for user {user_id}")
                await message.reply_text(
                    "⏳ **Already Processing Verification**\n\n"
                    "Your verification is already being processed. Please wait..."
                )
                return
            
            logger.info(f"🔐 Processing verification token for user {user_id}: {token[:16]}...")
            
            if not self.verification_system:
                await message.reply_text("❌ Verification system not available. Please try again later.")
                await self.clear_processing_request(user_id, token, request_type="verification")
                return
            
            # Send processing message
            processing_msg = await message.reply_text(
                f"🔐 **Verifying your access...**\n\n"
                f"**User:** {user_name}\n"
                f"**Token:** `{token[:16]}...`\n"
                f"⏳ **Please wait...**"
            )
            
            # Verify the token
            is_valid, verified_user_id, message_text = await self.verification_system.verify_user_token(token)
            
            # Clear processing request
            await self.clear_processing_request(user_id, token, request_type="verification")
            
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
                    f"Visit {self.config.WEBSITE_URL} to download movies!\n"
                    f"🎬 @SK4FiLM"
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
            await self.clear_processing_request(user_id, token, request_type="verification")

    async def handle_file_request(self, client, message, file_text):
        """Handle file download request with user verification - ORIGINAL WORKING VERSION"""
        try:
            user_id = message.from_user.id
            
            # ✅ FILE RATE LIMIT CHECK
            if not await self.check_rate_limit(user_id, limit=3, window=60, request_type="file"):
                await message.reply_text(
                    "⚠️ **Download Rate Limit Exceeded**\n\n"
                    "You're making too many download requests. Please wait 60 seconds and try again."
                )
                return
            
            # ✅ DUPLICATE FILE REQUEST CHECK
            if await self.is_request_duplicate(user_id, file_text, request_type="file"):
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
                await message.reply_text(
                    "❌ **Invalid channel ID**\n\n"
                    f"Channel ID '{channel_str}' is not valid.\n"
                    "Please click download button on website again."
                )
                await self.clear_processing_request(user_id, file_text, request_type="file")
                return
            
            # Parse message ID
            try:
                message_id = int(parts[1].strip())
            except ValueError:
                await message.reply_text(
                    "❌ **Invalid message ID**\n\n"
                    f"Message ID '{parts[1]}' is not valid."
                )
                await self.clear_processing_request(user_id, file_text, request_type="file")
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
                    if self.user_client and self.user_session_ready:
                        try:
                            file_message = await self.user_client.get_messages(
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
                await self.clear_processing_request(user_id, file_text, request_type="file")
                return
            
            if not file_message.document and not file_message.video:
                try:
                    await processing_msg.edit_text(
                        "❌ **Not a downloadable file**\n\n"
                        "This message doesn't contain a video or document file."
                    )
                except:
                    pass
                await self.clear_processing_request(user_id, file_text, request_type="file")
                return
            
            # ✅ Send file to user using ORIGINAL FUNCTION
            success, result_data, file_size = await self.send_file_to_user(
                client, user_id, file_message, quality
            )
            
            if success:
                # File was sent with caption
                try:
                    await processing_msg.delete()
                except:
                    pass
                
                # ✅ Record download for statistics
                if self.premium_system:
                    await self.premium_system.record_download(
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
            await self.clear_processing_request(user_id, file_text, request_type="file")
            
        except Exception as e:
            logger.error(f"File request handling error: {e}")
            try:
                await message.reply_text(
                    "❌ **Download Error**\n\n"
                    "An error occurred during download. Please try again."
                )
            except:
                pass
            await self.clear_processing_request(user_id, file_text, request_type="file")

    # ✅ SETUP BOT HANDLERS - SIMPLIFIED AND CORRECTED
    async def setup_bot_handlers(self):
        """Setup bot commands and handlers - SIMPLIFIED AND WORKING"""
        
        # ✅ 1. COMMAND HANDLERS
        @self.bot.on_message(filters.command("start"))
        async def handle_start_command(client, message):
            """Handle /start command"""
            user_name = message.from_user.first_name or "User"
            user_id = message.from_user.id
            
            # Check if there's additional text
            if len(message.command) > 1:
                start_text = ' '.join(message.command[1:])
                
                # Check if it's a verification token
                if start_text.startswith('verify_'):
                    token = start_text.replace('verify_', '', 1).strip()
                    await self.handle_verification_token(client, message, token)
                    return
                else:
                    # Treat as file request
                    await self.handle_file_request(client, message, start_text)
                    return
            
            # WELCOME MESSAGE
            welcome_text = (
                f"🎬 **Welcome to SK4FiLM, {user_name}!**\n\n"
                f"🌐 **Website:** {self.config.WEBSITE_URL}\n\n"
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
                [InlineKeyboardButton("🌐 OPEN WEBSITE", url=self.config.WEBSITE_URL)],
                [InlineKeyboardButton("⭐ GET PREMIUM", callback_data="buy_premium")],
                [InlineKeyboardButton("📢 JOIN CHANNEL", url=getattr(self.config, 'MAIN_CHANNEL_LINK', 'https://t.me/SK4FiLM'))]
            ])
            
            await message.reply_text(welcome_text, reply_markup=keyboard, disable_web_page_preview=True)
        
        # ✅ 2. FILE REQUEST HANDLER (MUST COME BEFORE NON-COMMAND TEXT)
        @self.bot.on_message(filters.private & filters.regex(r'^-?\d+_\d+(_\w+)?$'))
        async def handle_direct_file_request(client, message):
            """Handle direct file format messages"""
            file_text = message.text.strip()
            await self.handle_file_request(client, message, file_text)
        
        # ✅ 3. SCREENSHOT HANDLER
        @self.bot.on_message(filters.private & (filters.photo | filters.document))
        async def handle_screenshot(client, message):
            """Handle payment screenshots"""
            # Check if it's likely a screenshot
            is_screenshot = False
            
            if message.photo:
                is_screenshot = True
            elif message.document:
                # Check if it's an image file
                mime_type = message.document.mime_type or ""
                file_name = message.document.file_name or ""
                
                # Common screenshot/image file extensions and mime types
                image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']
                image_mimes = ['image/', 'application/image']
                
                if any(mime in mime_type.lower() for mime in image_mimes):
                    is_screenshot = True
                elif any(ext in file_name.lower() for ext in image_extensions):
                    is_screenshot = True
            
            if is_screenshot:
                # Process screenshot
                await self.process_screenshot_message(client, message)
            else:
                # If not a screenshot, it's probably a file download
                # Let it pass through to other handlers
                pass
        
        # ✅ 4. NON-COMMAND TEXT HANDLER (MUST BE LAST)
        @self.bot.on_message(filters.private & filters.text & ~filters.command)
        async def handle_non_command_text(client, message):
            """Handle non-command text messages"""
            # Check if it's already handled by file request handler
            import re
            if re.match(r'^-?\d+_\d+(_\w+)?$', message.text.strip()):
                return  # Already handled by file request handler
            
            # Check if it's a verification token
            if message.text.strip().startswith('verify_'):
                return  # Will be handled by start command
            
            user_name = message.from_user.first_name or "User"
            
            # Default response for any other text
            website_text = (
                f"🌐 **Welcome to SK4FiLM, {user_name}!** 🌐\n\n"
                f"**To download files:**\n"
                f"1. Visit our website: {self.config.WEBSITE_URL}\n"
                f"2. Search for movies/TV shows\n"
                f"3. Click download button\n"
                f"4. File will appear here automatically\n\n"
                f"**Available Commands:**\n"
                f"• /start - Main menu\n"
                f"• /buy - Purchase premium\n"
                f"• /plans - View plans\n"
                f"• /help - Help guide\n\n"
                f"**Need help with payment screenshot?**\n"
                f"1. Use /buy to purchase premium\n"
                f"2. Complete payment\n"
                f"3. Send screenshot here\n"
                f"4. Admin will approve\n\n"
                f"🎬 **Happy downloading!**"
            )
            
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🌐 OPEN WEBSITE", url=self.config.WEBSITE_URL)],
                [InlineKeyboardButton("💰 BUY PREMIUM", callback_data="buy_premium")],
                [InlineKeyboardButton("🆘 HELP", callback_data="back_to_start")]
            ])
            
            await message.reply_text(
                website_text,
                reply_markup=keyboard,
                disable_web_page_preview=True
            )
        
        # ✅ 5. CALLBACK HANDLERS (KEEP ORIGINAL)
        @self.bot.on_callback_query(filters.regex(r"^get_verified$"))
        async def get_verified_callback(client, callback_query):
            """Get verification link"""
            user_id = callback_query.from_user.id
            user_name = callback_query.from_user.first_name or "User"
            
            if self.verification_system:
                verification_data = await self.verification_system.create_verification_link(user_id)
                
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
                
                try:
                    await callback_query.message.edit_text(
                        text=text,
                        reply_markup=keyboard,
                        disable_web_page_preview=True
                    )
                except:
                    await callback_query.answer("Click VERIFY NOW button!", show_alert=True)
            else:
                await callback_query.answer("Verification system not available!", show_alert=True)
        
        @self.bot.on_callback_query(filters.regex(r"^back_to_start$"))
        async def back_to_start_callback(client, callback_query):
            user_name = callback_query.from_user.first_name or "User"
            
            text = (
                f"🎬 **Welcome back, {user_name}!**\n\n"
                f"Visit {self.config.WEBSITE_URL} to download movies.\n"
                "Click download button on website and file will appear here."
            )
            
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🌐 OPEN WEBSITE", url=self.config.WEBSITE_URL)],
                [InlineKeyboardButton("📢 JOIN CHANNEL", url=getattr(self.config, 'MAIN_CHANNEL_LINK', 'https://t.me/SK4FiLM'))]
            ])
            
            try:
                await callback_query.message.edit_text(
                    text=text,
                    reply_markup=keyboard,
                    disable_web_page_preview=True
                )
            except:
                await callback_query.answer("Already on home page!")
        
        @self.bot.on_callback_query(filters.regex(r"^buy_premium$"))
        async def buy_premium_callback(client, callback_query):
            """Show premium plans"""
            user_id = callback_query.from_user.id
            user_name = callback_query.from_user.first_name or "User"
            
            # Check if already premium
            if self.premium_system:
                is_premium = await self.premium_system.is_premium_user(user_id)
                if is_premium:
                    details = await self.premium_system.get_subscription_details(user_id)
                    
                    text = (
                        f"⭐ **You're Already Premium!** ⭐\n\n"
                        f"**User:** {user_name}\n"
                        f"**Plan:** {details.get('tier_name', 'Premium')}\n"
                        f"**Days Left:** {details.get('days_remaining', 0)}\n"
                        f"**Status:** ✅ Active\n\n"
                        "Enjoy unlimited downloads without verification! 🎬"
                    )
                    
                    keyboard = InlineKeyboardMarkup([
                        [InlineKeyboardButton("🌐 OPEN WEBSITE", url=self.config.WEBSITE_URL)],
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
        
        # ✅ 6. ADMIN CALLBACK HANDLERS
        @self.bot.on_callback_query(filters.regex(r"^admin_"))
        async def admin_callback_handler(client, callback_query):
            """Handle all admin callbacks"""
            admin_id = callback_query.from_user.id
            
            if admin_id not in getattr(self.config, 'ADMIN_IDS', []):
                await callback_query.answer("❌ Admin only!", show_alert=True)
                return
            
            data = callback_query.data
            
            if data.startswith("admin_approve_"):
                user_id = int(data.split('_')[2])
                await callback_query.answer("Processing approval...")
                # Show confirmation
                confirm_text = f"Confirm approval for user {user_id}?\n\nType: `/approve {user_id}`"
                await callback_query.message.reply_text(confirm_text)
            
            elif data.startswith("admin_reject_"):
                user_id = int(data.split('_')[2])
                await callback_query.answer("Enter rejection reason")
                reject_text = f"Enter rejection reason for user {user_id}:\n\nType: `/reject {user_id} <reason>`"
                await callback_query.message.reply_text(reject_text)
        
        logger.info("✅ Bot handlers setup complete - ALL ORIGINAL FUNCTIONS WORKING")
