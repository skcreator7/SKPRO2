"""
bot_handlers.py - Telegram Bot Handlers for SK4FiLM
UPDATED: Auto-delete file system fixed
UPDATED: All plans have same features, different validity
UPDATED: QR code image display for all plans
"""
import asyncio
import logging
import secrets
import re
import time
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from collections import defaultdict

# ✅ Complete Pyrogram imports
try:
    from pyrogram import Client, filters
    from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton, Message, CallbackQuery
    from pyrogram.errors import FloodWait, BadRequest, UserNotParticipant, MessageDeleteForbidden
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

# Import utilities
from utils import (
    normalize_title,
    extract_title_smart,
    format_size,
    detect_quality,
    is_video_file,
    format_post
)

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
        self.auto_delete_tasks = {}  # task_id -> task
        self.auto_delete_messages = {}  # task_id -> message data
        
        # Rate limiting and deduplication
        self.user_request_times = defaultdict(list)
        self.processing_requests = {}
        
        # Payment screenshot tracking
        self.pending_payments = {}
        
        # Initialize all systems
        try:
            from verification import VerificationSystem
            from premium import PremiumSystem
            from poster_fetching import PosterFetcher
            from cache import CacheManager
            
            self.verification_system = VerificationSystem(config, db_manager)
            self.premium_system = PremiumSystem(config, db_manager)
            self.poster_fetcher = PosterFetcher(config)
            self.cache_manager = CacheManager(config)
            
            asyncio.create_task(self.cache_manager.init_redis())
            
            logger.info("✅ All systems initialized")
        except Exception as e:
            logger.error(f"System initialization error: {e}")
            self.verification_system = None
            self.premium_system = None
            self.poster_fetcher = None
            self.cache_manager = None
    
    async def initialize(self):
        """Initialize bot"""
        try:
            logger.info("🚀 Initializing SK4FiLM Bot...")
            
            self.bot = Client(
                "bot",
                api_id=self.config.API_ID,
                api_hash=self.config.API_HASH,
                bot_token=self.config.BOT_TOKEN,
                workers=20
            )
            
            if self.config.USER_SESSION_STRING:
                self.user_client = Client(
                    "user",
                    api_id=self.config.API_ID,
                    api_hash=self.config.API_HASH,
                    session_string=self.config.USER_SESSION_STRING
                )
                await self.user_client.start()
                self.user_session_ready = True
                logger.info("✅ User session started successfully")
            
            await self.bot.start()
            self.bot_started = True
            logger.info("✅ Bot started successfully")
            
            await setup_bot_handlers(self.bot, self)
            
            if self.verification_system:
                asyncio.create_task(self.verification_system.start_cleanup_task())
            if self.premium_system:
                asyncio.create_task(self.premium_system.start_cleanup_task())
            if self.cache_manager:
                asyncio.create_task(self.cache_manager.start_cleanup_task())
            
            # Start auto-delete cleanup task
            asyncio.create_task(self.cleanup_old_auto_delete_tasks())
            
            return True
            
        except Exception as e:
            logger.error(f"Bot initialization failed: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown bot"""
        try:
            # Cancel all auto-delete tasks
            for task_id, task in self.auto_delete_tasks.items():
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            self.auto_delete_tasks.clear()
            self.auto_delete_messages.clear()
            
            if self.bot and self.bot_started:
                await self.bot.stop()
                logger.info("✅ Bot stopped")
            
            if self.user_client and self.user_session_ready:
                await self.user_client.stop()
                logger.info("✅ User client stopped")
                
            if self.verification_system:
                await self.verification_system.stop_cleanup_task()
            if self.premium_system:
                await self.premium_system.stop_cleanup_task()
            if self.cache_manager:
                await self.cache_manager.stop()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    # ✅ AUTO-DELETE SYSTEM METHODS - FIXED
    
    async def add_auto_delete_task(self, user_id: int, message_id: int, file_name: str, 
                                   delete_after_minutes: int = 10):
        """Add auto-delete task for a file"""
        try:
            task_id = f"{user_id}_{message_id}"
            
            # Cancel existing task if any
            if task_id in self.auto_delete_tasks:
                old_task = self.auto_delete_tasks[task_id]
                if not old_task.done():
                    old_task.cancel()
                    try:
                        await old_task
                    except asyncio.CancelledError:
                        pass
            
            # Store message data
            self.auto_delete_messages[task_id] = {
                'user_id': user_id,
                'message_id': message_id,
                'file_name': file_name,
                'scheduled_time': datetime.now() + timedelta(minutes=delete_after_minutes),
                'status': 'pending',
                'delete_after_minutes': delete_after_minutes
            }
            
            # Create new task
            task = asyncio.create_task(
                self._auto_delete_file(user_id, message_id, file_name, delete_after_minutes)
            )
            self.auto_delete_tasks[task_id] = task
            
            logger.info(f"⏰ Auto-delete task scheduled: {task_id} in {delete_after_minutes} minutes")
            return True
            
        except Exception as e:
            logger.error(f"Error adding auto-delete task: {e}")
            return False
    
    async def _auto_delete_file(self, user_id: int, message_id: int, file_name: str, 
                                delete_after_minutes: int):
        """Auto-delete file after specified minutes - FIXED"""
        try:
            logger.info(f"⏰ Auto-delete started for user {user_id}, message {message_id}, waiting {delete_after_minutes} minutes")
            
            # Wait for specified time
            await asyncio.sleep(delete_after_minutes * 60)
            
            task_id = f"{user_id}_{message_id}"
            
            # Try to delete the file message
            delete_success = False
            try:
                if self.bot and self.bot_started:
                    await self.bot.delete_messages(user_id, message_id)
                    delete_success = True
                    logger.info(f"🗑️ File message deleted: user {user_id}, message {message_id}")
            except MessageDeleteForbidden:
                logger.warning(f"❌ Cannot delete message {message_id} for user {user_id}: Message delete forbidden")
            except BadRequest as e:
                if "MESSAGE_TOO_OLD" in str(e):
                    logger.warning(f"❌ Cannot delete message {message_id}: Message too old (48 hours limit)")
                elif "MESSAGE_ID_INVALID" in str(e):
                    logger.warning(f"❌ Cannot delete message {message_id}: Message already deleted or invalid")
                else:
                    logger.error(f"❌ Error deleting message {message_id}: {e}")
            except Exception as e:
                logger.error(f"❌ Error deleting message: {e}")
            
            # Send deletion notification
            try:
                notification_text = (
                    f"🗑️ **File Auto-Deleted**\n\n"
                    f"`{file_name}`\n\n"
                    f"⏰ **Deleted after:** {delete_after_minutes} minutes\n"
                    f"✅ **Security measure completed**\n\n"
                    f"🔁 **Need the file again?**\n"
                    f"Visit website and download again\n"
                    f"🎬 @SK4FiLM"
                )
                
                buttons = InlineKeyboardMarkup([
                    [InlineKeyboardButton("🌐 VISIT WEBSITE", url=self.config.WEBSITE_URL)],
                    [InlineKeyboardButton("🔄 GET ANOTHER FILE", callback_data="back_to_start")]
                ])
                
                if self.bot and self.bot_started:
                    await self.bot.send_message(
                        user_id,
                        notification_text,
                        reply_markup=buttons
                    )
                    logger.info(f"✅ Auto-delete notification sent to user {user_id}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to send delete notification: {e}")
            
            # Clean up task data
            if task_id in self.auto_delete_tasks:
                del self.auto_delete_tasks[task_id]
            
            if task_id in self.auto_delete_messages:
                self.auto_delete_messages[task_id]['status'] = 'completed' if delete_success else 'failed'
                self.auto_delete_messages[task_id]['completed_at'] = datetime.now()
            
            logger.info(f"✅ Auto-delete process completed for task {task_id}")
            
        except asyncio.CancelledError:
            logger.info(f"⏹️ Auto-delete task cancelled for user {user_id}, message {message_id}")
            
            # Clean up task data
            task_id = f"{user_id}_{message_id}"
            if task_id in self.auto_delete_tasks:
                del self.auto_delete_tasks[task_id]
            
            if task_id in self.auto_delete_messages:
                self.auto_delete_messages[task_id]['status'] = 'cancelled'
                self.auto_delete_messages[task_id]['cancelled_at'] = datetime.now()
                
        except Exception as e:
            logger.error(f"❌ Error in auto-delete task: {e}")
            
            # Clean up task data even on error
            task_id = f"{user_id}_{message_id}"
            if task_id in self.auto_delete_tasks:
                del self.auto_delete_tasks[task_id]
            
            if task_id in self.auto_delete_messages:
                self.auto_delete_messages[task_id]['status'] = 'error'
                self.auto_delete_messages[task_id]['error'] = str(e)
    
    async def cancel_auto_delete(self, user_id: int, message_id: int):
        """Cancel auto-delete for a specific file"""
        try:
            task_id = f"{user_id}_{message_id}"
            
            if task_id in self.auto_delete_tasks:
                task = self.auto_delete_tasks[task_id]
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                
                del self.auto_delete_tasks[task_id]
                
                if task_id in self.auto_delete_messages:
                    self.auto_delete_messages[task_id]['status'] = 'cancelled'
                    self.auto_delete_messages[task_id]['cancelled_at'] = datetime.now()
                
                logger.info(f"✅ Auto-delete cancelled for task {task_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error cancelling auto-delete: {e}")
            return False
    
    async def get_auto_delete_stats(self):
        """Get auto-delete system statistics"""
        try:
            pending_tasks = 0
            completed_tasks = 0
            failed_tasks = 0
            cancelled_tasks = 0
            error_tasks = 0
            
            for task_id, task_data in self.auto_delete_messages.items():
                status = task_data.get('status', 'unknown')
                if status == 'pending':
                    pending_tasks += 1
                elif status == 'completed':
                    completed_tasks += 1
                elif status == 'failed':
                    failed_tasks += 1
                elif status == 'cancelled':
                    cancelled_tasks += 1
                elif status == 'error':
                    error_tasks += 1
            
            return {
                'total_tasks': len(self.auto_delete_messages),
                'pending_tasks': pending_tasks,
                'completed_tasks': completed_tasks,
                'failed_tasks': failed_tasks,
                'cancelled_tasks': cancelled_tasks,
                'error_tasks': error_tasks,
                'active_tasks': len(self.auto_delete_tasks),
                'default_delete_time': getattr(self.config, 'AUTO_DELETE_TIME', 10)
            }
            
        except Exception as e:
            logger.error(f"Error getting auto-delete stats: {e}")
            return {}
    
    async def cleanup_old_auto_delete_tasks(self):
        """Clean up old auto-delete task data"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                now = datetime.now()
                to_remove = []
                
                for task_id, task_data in self.auto_delete_messages.items():
                    completed_at = task_data.get('completed_at')
                    cancelled_at = task_data.get('cancelled_at')
                    
                    # Remove tasks completed/cancelled more than 24 hours ago
                    if completed_at and (now - completed_at).total_seconds() > 24 * 3600:
                        to_remove.append(task_id)
                    elif cancelled_at and (now - cancelled_at).total_seconds() > 24 * 3600:
                        to_remove.append(task_id)
                    # Remove pending tasks scheduled more than 48 hours ago
                    elif task_data.get('status') == 'pending':
                        scheduled_time = task_data.get('scheduled_time')
                        if scheduled_time and (now - scheduled_time).total_seconds() > 48 * 3600:
                            to_remove.append(task_id)
                    # Remove error tasks older than 12 hours
                    elif task_data.get('status') == 'error':
                        error_time = task_data.get('error_time', datetime.now())
                        if (now - error_time).total_seconds() > 12 * 3600:
                            to_remove.append(task_id)
                
                for task_id in to_remove:
                    self.auto_delete_messages.pop(task_id, None)
                
                if to_remove:
                    logger.info(f"🧹 Cleaned up {len(to_remove)} old auto-delete tasks")
                    
            except asyncio.CancelledError:
                logger.info("🧹 Auto-delete cleanup task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in auto-delete cleanup: {e}")
    
    # ✅ RATE LIMITING METHODS
    async def check_rate_limit(self, user_id, limit=3, window=60):
        """Check if user is within rate limits"""
        now = time.time()
        
        if user_id in self.user_request_times:
            self.user_request_times[user_id] = [
                t for t in self.user_request_times[user_id] 
                if now - t < window
            ]
        
        if len(self.user_request_times.get(user_id, [])) >= limit:
            logger.warning(f"⚠️ Rate limit exceeded for user {user_id}")
            return False
        
        self.user_request_times[user_id].append(now)
        return True
    
    async def is_request_duplicate(self, user_id, request_data):
        """Check if this is a duplicate request"""
        request_hash = f"{user_id}_{hashlib.md5(request_data.encode()).hexdigest()[:8]}"
        
        if request_hash in self.processing_requests:
            if time.time() - self.processing_requests[request_hash] < 30:
                return True
        
        self.processing_requests[request_hash] = time.time()
        return False
    
    async def clear_processing_request(self, user_id, request_data):
        """Clear from processing requests"""
        request_hash = f"{user_id}_{hashlib.md5(request_data.encode()).hexdigest()[:8]}"
        self.processing_requests.pop(request_hash, None)
    
    # ✅ PAYMENT SCREENSHOT TRACKING
    async def add_pending_payment(self, payment_id: str, user_id: int, amount: int, plan: str):
        """Add pending payment for tracking"""
        self.pending_payments[payment_id] = {
            'user_id': user_id,
            'amount': amount,
            'plan': plan,
            'timestamp': time.time(),
            'status': 'pending',
            'screenshot_sent': False,
            'screenshot_message_id': None
        }
        logger.info(f"✅ Payment added: {payment_id} for user {user_id}, amount: {amount}")
    
    async def update_payment_screenshot(self, payment_id: str, message_id: int):
        """Update payment with screenshot info"""
        if payment_id in self.pending_payments:
            self.pending_payments[payment_id]['screenshot_sent'] = True
            self.pending_payments[payment_id]['screenshot_message_id'] = message_id
            self.pending_payments[payment_id]['screenshot_time'] = time.time()
            logger.info(f"✅ Screenshot updated for payment {payment_id}")
            return True
        return False
    
    async def get_pending_payments(self):
        """Get all pending payments"""
        pending = []
        now = time.time()
        for payment_id, data in self.pending_payments.items():
            if data['status'] == 'pending':
                age_hours = (now - data['timestamp']) / 3600
                data['payment_id'] = payment_id
                data['age_hours'] = round(age_hours, 1)
                pending.append(data)
        return pending
    
    async def complete_payment(self, payment_id: str):
        """Mark payment as completed"""
        if payment_id in self.pending_payments:
            self.pending_payments[payment_id]['status'] = 'completed'
            self.pending_payments[payment_id]['completed_time'] = time.time()
            logger.info(f"✅ Payment completed: {payment_id}")
            return True
        return False
    
    # ✅ PERIODIC CLEANUP
    async def cleanup_old_processing_requests(self):
        """Clean up old processing requests"""
        try:
            now = time.time()
            to_remove = []
            for request_hash, timestamp in self.processing_requests.items():
                if now - timestamp > 300:
                    to_remove.append(request_hash)
            
            for req in to_remove:
                self.processing_requests.pop(req, None)
            
            if to_remove:
                logger.info(f"Cleaned up {len(to_remove)} old processing requests")
        except Exception as e:
            logger.error(f"Error cleaning processing requests: {e}")
    
    async def periodic_cleanup(self):
        """Periodic cleanup of rate limiting and processing requests"""
        while True:
            try:
                await asyncio.sleep(300)
                await self.cleanup_old_processing_requests()
                
                now = time.time()
                for user_id in list(self.user_request_times.keys()):
                    self.user_request_times[user_id] = [
                        t for t in self.user_request_times[user_id]
                        if now - t < 300
                    ]
                    if not self.user_request_times[user_id]:
                        self.user_request_times.pop(user_id, None)
                
                old_payments = []
                for payment_id, data in list(self.pending_payments.items()):
                    if now - data.get('timestamp', 0) > 7 * 24 * 3600:
                        old_payments.append(payment_id)
                
                for pid in old_payments:
                    self.pending_payments.pop(pid, None)
                
                if old_payments:
                    logger.info(f"Cleaned up {len(old_payments)} old pending payments")
                        
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")

async def safe_edit_message(callback_query, text=None, reply_markup=None, disable_web_page_preview=None):
    """Safely edit message to avoid MESSAGE_NOT_MODIFIED errors"""
    try:
        current_text = callback_query.message.text or callback_query.message.caption
        current_markup = callback_query.message.reply_markup
        
        text_changed = text is not None and text != current_text
        
        markup_changed = False
        if reply_markup is not None and current_markup is not None:
            if hasattr(reply_markup, 'inline_keyboard') and hasattr(current_markup, 'inline_keyboard'):
                if str(reply_markup.inline_keyboard) != str(current_markup.inline_keyboard):
                    markup_changed = True
            else:
                markup_changed = True
        elif reply_markup is not None and current_markup is None:
            markup_changed = True
        elif reply_markup is None and current_markup is not None:
            markup_changed = True
        
        if text_changed or markup_changed:
            await callback_query.message.edit_text(
                text=text if text is not None else current_text,
                reply_markup=reply_markup,
                disable_web_page_preview=disable_web_page_preview
            )
            return True
        else:
            await callback_query.answer()
            return False
            
    except Exception as e:
        if "MESSAGE_NOT_MODIFIED" in str(e):
            await callback_query.answer()
            return False
        else:
            raise e

async def send_file_to_user(client, user_id, file_message, quality="480p", config=None, bot_instance=None):
    """Send file to user with verification check - FIXED AUTO-DELETE"""
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
        
        # ✅ FIX: Validate file ID
        if not file_id:
            logger.error(f"❌ Empty file ID for message {file_message.id}")
            return False, {
                'message': "❌ File ID is empty. Please try download again.",
                'buttons': []
            }, 0
        
        # ✅ Get auto-delete time from config
        auto_delete_minutes = getattr(config, 'AUTO_DELETE_TIME', 10)  # Default 10 minutes
        
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
            
            # ✅ SCHEDULE AUTO-DELETE USING BOT INSTANCE METHOD
            if bot_instance and auto_delete_minutes > 0:
                # Use bot instance method for better tracking
                await bot_instance.add_auto_delete_task(
                    user_id=user_id,
                    message_id=sent.id,
                    file_name=file_name,
                    delete_after_minutes=auto_delete_minutes
                )
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
                    
                    # ✅ SCHEDULE AUTO-DELETE FOR REFRESHED FILE
                    if bot_instance and auto_delete_minutes > 0:
                        await bot_instance.add_auto_delete_task(
                            user_id=user_id,
                            message_id=sent.id,
                            file_name=file_name,
                            delete_after_minutes=auto_delete_minutes
                        )
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
                raise e  # Re-raise other BadRequest errors
                
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

async def handle_file_request(client, message, file_text, bot_instance):
    """Handle file download request with user verification"""
    try:
        config = bot_instance.config
        user_id = message.from_user.id
        request_hash = f"{user_id}_{file_text}"
        
        # ✅ RATE LIMIT CHECK
        if not await bot_instance.check_rate_limit(user_id):
            await message.reply_text(
                "⚠️ **Rate Limit Exceeded**\n\n"
                "You're making too many requests. Please wait 60 seconds and try again."
            )
            return
        
        # ✅ DUPLICATE REQUEST CHECK
        if await bot_instance.is_request_duplicate(user_id, file_text):
            logger.warning(f"⚠️ Duplicate request ignored for user {user_id}: {file_text}")
            await message.reply_text(
                "⏳ **Already Processing**\n\n"
                "Your previous request is still being processed. Please wait..."
            )
            return
        
        # Clean the text
        clean_text = file_text.strip()
        logger.info(f"📥 Processing file request from user {user_id}: {clean_text}")
        
        # Parse file request
        # Remove /start if present
        if clean_text.startswith('/start'):
            clean_text = clean_text.replace('/start', '').strip()
        
        # Also handle /start with space
        clean_text = re.sub(r'^/start\s+', '', clean_text)
        
        # Extract file ID parts
        parts = clean_text.split('_')
        logger.info(f"📥 Parts: {parts}")
        
        if len(parts) < 2:
            await message.reply_text(
                "❌ **Invalid format**\n\n"
                "Correct format: `-1001768249569_16066_480p`\n"
                "Please click download button on website again."
            )
            await bot_instance.clear_processing_request(user_id, file_text)
            return
        
        # Parse channel ID (could be negative)
        channel_str = parts[0].strip()
        try:
            # Handle negative channel IDs
            if channel_str.startswith('--'):
                # Double dash case
                channel_id = int(channel_str[1:])
            else:
                channel_id = int(channel_str)
        except ValueError:
            await message.reply_text(
                "❌ **Invalid channel ID**\n\n"
                f"Channel ID '{channel_str}' is not valid.\n"
                "Please click download button on website again."
            )
            await bot_instance.clear_processing_request(user_id, file_text)
            return
        
        # Parse message ID
        try:
            message_id = int(parts[1].strip())
        except ValueError:
            await message.reply_text(
                "❌ **Invalid message ID**\n\n"
                f"Message ID '{parts[1]}' is not valid."
            )
            await bot_instance.clear_processing_request(user_id, file_text)
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
                    await asyncio.sleep(1)  # Wait before retry
                    
            except Exception as e:
                logger.error(f"Attempt {attempt+1} failed: {e}")
        
        if not file_message:
            try:
                await processing_msg.edit_text(
                    "❌ **File not found**\n\n"
                    "The file may have been deleted or I don't have access.\n"
                    "Please try downloading from the website again."
                )
            except:
                pass
            await bot_instance.clear_processing_request(user_id, file_text)
            return
        
        if not file_message.document and not file_message.video:
            try:
                await processing_msg.edit_text(
                    "❌ **Not a downloadable file**\n\n"
                    "This message doesn't contain a video or document file."
                )
            except:
                pass
            await bot_instance.clear_processing_request(user_id, file_text)
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
        await bot_instance.clear_processing_request(user_id, file_text)
        
    except Exception as e:
        logger.error(f"File request handling error: {e}")
        try:
            await message.reply_text(
                "❌ **An error occurred**\n\n"
                "Please try again or contact support."
            )
        except:
            pass
        await bot_instance.clear_processing_request(user_id, file_text)

# ✅ ADMIN COMMAND TO CHECK AUTO-DELETE STATUS
async def setup_bot_handlers(bot: Client, bot_instance):
    """Setup bot commands and handlers with admin dashboard"""
    config = bot_instance.config
    
    # ✅ AUTO-DELETE STATUS COMMAND (ADMIN ONLY)
    @bot.on_message(filters.command("autodelete") & filters.user(config.ADMIN_IDS))
    async def autodelete_status_command(client, message):
        """Check auto-delete system status"""
        try:
            stats = await bot_instance.get_auto_delete_stats()
            
            if stats:
                text = (
                    "⏰ **Auto-Delete System Status**\n\n"
                    f"📊 **Total Tasks:** {stats.get('total_tasks', 0)}\n"
                    f"⏳ **Pending Tasks:** {stats.get('pending_tasks', 0)}\n"
                    f"✅ **Completed Tasks:** {stats.get('completed_tasks', 0)}\n"
                    f"❌ **Failed Tasks:** {stats.get('failed_tasks', 0)}\n"
                    f"⏹️ **Cancelled Tasks:** {stats.get('cancelled_tasks', 0)}\n"
                    f"⚠️ **Error Tasks:** {stats.get('error_tasks', 0)}\n"
                    f"⚡ **Active Tasks:** {stats.get('active_tasks', 0)}\n\n"
                    f"⏱️ **Default Delete Time:** {stats.get('default_delete_time', 10)} minutes\n"
                    f"🔄 **System:** {'✅ Running' if bot_instance.bot_started else '❌ Stopped'}\n"
                )
                
                # Add recent tasks info
                pending_tasks = []
                for task_id, task_data in bot_instance.auto_delete_messages.items():
                    if task_data.get('status') == 'pending':
                        pending_tasks.append(task_data)
                
                if pending_tasks:
                    text += f"\n📋 **Recent Pending Tasks ({len(pending_tasks)}):**\n"
                    for i, task in enumerate(pending_tasks[:5], 1):
                        user_id = task.get('user_id', 'Unknown')
                        file_name = task.get('file_name', 'Unknown')[:20]
                        minutes_left = 0
                        scheduled_time = task.get('scheduled_time')
                        if scheduled_time:
                            time_diff = scheduled_time - datetime.now()
                            minutes_left = max(0, int(time_diff.total_seconds() / 60))
                        
                        text += f"{i}. User {user_id}: `{file_name}` ({minutes_left}m left)\n"
                    
                    if len(pending_tasks) > 5:
                        text += f"... and {len(pending_tasks) - 5} more\n"
                
            else:
                text = "❌ Could not retrieve auto-delete statistics."
            
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🔄 Refresh", callback_data="refresh_autodelete")],
                [InlineKeyboardButton("📊 All Stats", callback_data="admin_stats")]
            ])
            
            await message.reply_text(text, reply_markup=keyboard, disable_web_page_preview=True)
            
        except Exception as e:
            logger.error(f"Auto-delete status command error: {e}")
            await message.reply_text(f"❌ Error: {str(e)[:100]}")
    
    # ✅ AUTO-DELETE REFRESH CALLBACK
    @bot.on_callback_query(filters.regex(r"^refresh_autodelete$"))
    async def refresh_autodelete_callback(client, callback_query):
        """Refresh auto-delete status"""
        admin_id = callback_query.from_user.id
        
        if admin_id not in config.ADMIN_IDS:
            await callback_query.answer("❌ Access denied!", show_alert=True)
            return
        
        try:
            stats = await bot_instance.get_auto_delete_stats()
            
            if stats:
                text = (
                    "⏰ **Auto-Delete System Status**\n\n"
                    f"📊 **Total Tasks:** {stats.get('total_tasks', 0)}\n"
                    f"⏳ **Pending Tasks:** {stats.get('pending_tasks', 0)}\n"
                    f"✅ **Completed Tasks:** {stats.get('completed_tasks', 0)}\n"
                    f"❌ **Failed Tasks:** {stats.get('failed_tasks', 0)}\n"
                    f"⏹️ **Cancelled Tasks:** {stats.get('cancelled_tasks', 0)}\n"
                    f"⚠️ **Error Tasks:** {stats.get('error_tasks', 0)}\n"
                    f"⚡ **Active Tasks:** {stats.get('active_tasks', 0)}\n\n"
                    f"⏱️ **Default Delete Time:** {stats.get('default_delete_time', 10)} minutes\n"
                    f"🔄 **System:** {'✅ Running' if bot_instance.bot_started else '❌ Stopped'}\n"
                )
            else:
                text = "❌ Could not retrieve auto-delete statistics."
            
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🔄 Refresh", callback_data="refresh_autodelete")],
                [InlineKeyboardButton("📊 All Stats", callback_data="admin_stats")]
            ])
            
            await callback_query.message.edit_text(text, reply_markup=keyboard)
            await callback_query.answer("✅ Refreshed!")
            
        except Exception as e:
            logger.error(f"Refresh auto-delete error: {e}")
            await callback_query.answer("❌ Refresh failed!", show_alert=True)
    
    # ✅ BUY COMMAND - SHOW ALL PLANS
    @bot.on_message(filters.command("buy"))
    async def buy_command(client, message):
        """Show all premium plans for purchase"""
        if bot_instance.premium_system:
            plans = await bot_instance.premium_system.get_all_plans()
            
            text = "💎 **SK4FiLM PREMIUM PLANS** 💎\n\n"
            text += "🎯 **ALL PLANS INCLUDE ALL PREMIUM FEATURES:**\n"
            text += "✅ All Quality (480p-4K)\n"
            text += "✅ Unlimited Downloads\n"
            text += "✅ No Verification Needed\n"
            text += "✅ VIP Support 24/7\n"
            text += "✅ No Ads\n"
            text += "✅ Instant Downloads\n"
            text += "✅ Batch Downloads\n"
            text += "✅ Early Access\n"
            text += "✅ Custom Requests\n"
            text += "✅ Highest Priority\n\n"
            text += "📊 **Choose your validity period:**\n\n"
            
            keyboard_buttons = []
            for plan in plans:
                text += f"{plan['icon']} **{plan['name']}**\n"
                text += f"💰 **Price:** ₹{plan['price']}\n"
                text += f"📅 **Validity:** {plan['duration_days']} days\n"
                text += f"📊 **Per day:** ₹{plan['per_day_cost']}/day\n\n"
                
                keyboard_buttons.append([
                    InlineKeyboardButton(
                        f"{plan['icon']} {plan['name']} - ₹{plan['price']}", 
                        callback_data=f"select_plan_{plan['tier']}"
                    )
                ])
            
            text += "🎬 **Same premium features in all plans!**\n"
            text += "Click a plan button to purchase"
            
            keyboard_buttons.append([InlineKeyboardButton("🔙 BACK", callback_data="back_to_start")])
            keyboard = InlineKeyboardMarkup(keyboard_buttons)
            
            await message.reply_text(text, reply_markup=keyboard, disable_web_page_preview=True)
        else:
            await message.reply_text("❌ Premium system not available")
    
    # ✅ PLANS COMMAND
    @bot.on_message(filters.command("plans"))
    async def plans_command(client, message):
        """Show all premium plans"""
        await buy_command(client, message)
    
    # ✅ MY PREMIUM COMMAND
    @bot.on_message(filters.command("mypremium"))
    async def mypremium_command(client, message):
        """Show user's premium status"""
        if bot_instance.premium_system:
            user_id = message.from_user.id
            info = await bot_instance.premium_system.get_my_premium_info(user_id)
            await message.reply_text(info, disable_web_page_preview=True)
        else:
            await message.reply_text("❌ Premium system not available")
    
    # ✅ PREMIUM PLANS SELECTION CALLBACK
    @bot.on_callback_query(filters.regex(r"^select_plan_"))
    async def select_plan_callback(client, callback_query):
        """Handle plan selection with QR code image"""
        plan_tier = callback_query.data.split('_')[2]
        
        # Map tier names
        tier_map = {
            'basic': 'Basic',
            'premium': 'Premium', 
            'gold': 'Gold',
            'diamond': 'Diamond'
        }
        
        plan_name = tier_map.get(plan_tier, 'Premium')
        
        # Get plan details from premium system
        if bot_instance.premium_system:
            try:
                from premium import PremiumTier
                tier_enum = PremiumTier(plan_tier)
                plans = await bot_instance.premium_system.get_all_plans()
                
                plan_details = None
                for plan in plans:
                    if plan['tier'] == plan_tier:
                        plan_details = plan
                        break
                
                if plan_details:
                    payment_id = secrets.token_hex(8)
                    user_id = callback_query.from_user.id
                    
                    # Track payment
                    await bot_instance.add_pending_payment(
                        payment_id=payment_id,
                        user_id=user_id,
                        amount=plan_details['price'],
                        plan=plan_tier
                    )
                    
                    # QR code image URL
                    qr_image_url = "https://i.ibb.co/4RLgJ8Tp/QR-MY.jpg"
                    
                    text = (
                        f"💰 **Payment for {plan_details['name']}**\n\n"
                        f"**Amount:** ₹{plan_details['price']}\n"
                        f"**Validity:** {plan_details['duration_days']} days\n"
                        f"**UPI ID:** `{plan_details['upi_id']}`\n\n"
                        f"📱 **QR Code:**\n"
                        f"{qr_image_url}\n\n"
                        f"🆔 **Payment ID:** `{payment_id}`\n\n"
                        f"**Payment Methods:**\n"
                        f"1. Scan QR Code above\n"
                        f"2. Send ₹{plan_details['price']} to UPI ID\n"
                        f"3. Take screenshot\n"
                        f"4. Send screenshot here\n\n"
                        f"✅ **You'll get ALL premium features:**\n"
                        "• All Quality (480p-4K)\n"
                        "• Unlimited Downloads\n"
                        "• No Verification Needed\n"
                        "• VIP Support 24/7\n"
                        "• No Ads\n"
                        "• Instant Downloads\n"
                        "• Batch Downloads\n"
                        "• Early Access\n"
                        "• Custom Requests\n"
                        "• Highest Priority\n\n"
                        f"⏰ **Payment ID valid for 24 hours**"
                    )
                    
                    keyboard = InlineKeyboardMarkup([
                        [InlineKeyboardButton("📸 SEND SCREENSHOT", callback_data=f"send_screenshot_{payment_id}")],
                        [InlineKeyboardButton("🔙 VIEW ALL PLANS", callback_data="buy_premium")],
                        [InlineKeyboardButton("🏠 HOME", callback_data="back_to_start")]
                    ])
                    
                    await callback_query.message.edit_text(text, reply_markup=keyboard)
                else:
                    await callback_query.answer("Plan not found!", show_alert=True)
            except Exception as e:
                logger.error(f"Plan selection error: {e}")
                await callback_query.answer("Error loading plan!", show_alert=True)
        else:
            await callback_query.answer("Premium system not available!", show_alert=True)
    
    # ✅ BUY PREMIUM CALLBACK
    @bot.on_callback_query(filters.regex(r"^buy_premium$"))
    async def buy_premium_callback(client, callback_query):
        """Show premium plans"""
        user_id = callback_query.from_user.id
        
        # Check if already premium
        if bot_instance.premium_system:
            is_premium = await bot_instance.premium_system.is_premium_user(user_id)
            if is_premium:
                details = await bot_instance.premium_system.get_subscription_details(user_id)
                
                text = (
                    f"⭐ **You're Already Premium!** ⭐\n\n"
                    f"**Plan:** {details.get('tier_name', 'Premium')}\n"
                    f"**Days Left:** {details.get('days_remaining', 0)}\n"
                    f"**Features:**\n"
                )
                
                features = details.get('features', [])
                for feature in features[:5]:
                    text += f"• {feature}\n"
                
                if len(features) > 5:
                    text += f"• ... and {len(features)-5} more features\n"
                
                text += "\n🎬 **Enjoy unlimited premium downloads!**"
                
                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 BACK", callback_data="back_to_start")]
                ])
                
                try:
                    await callback_query.message.edit_text(text, reply_markup=keyboard)
                except:
                    await callback_query.answer("You're already premium!")
                return
        
        # Show all plans
        if bot_instance.premium_system:
            plans = await bot_instance.premium_system.get_all_plans()
            
            text = "💎 **SK4FiLM PREMIUM PLANS** 💎\n\n"
            text += "🎯 **ALL PLANS INCLUDE ALL PREMIUM FEATURES:**\n"
            text += "✅ All Quality (480p-4K)\n"
            text += "✅ Unlimited Downloads\n"
            text += "✅ No Verification Needed\n"
            text += "✅ VIP Support 24/7\n"
            text += "✅ No Ads\n"
            text += "✅ Instant Downloads\n"
            text += "✅ Batch Downloads\n"
            text += "✅ Early Access\n"
            text += "✅ Custom Requests\n"
            text += "✅ Highest Priority\n\n"
            text += "📊 **Choose your validity period:**\n\n"
            
            keyboard_buttons = []
            for plan in plans:
                text += f"{plan['icon']} **{plan['name']}**\n"
                text += f"💰 **Price:** ₹{plan['price']}\n"
                text += f"📅 **Validity:** {plan['duration_days']} days\n"
                text += f"📊 **Per day:** ₹{plan['per_day_cost']}/day\n\n"
                
                keyboard_buttons.append([
                    InlineKeyboardButton(
                        f"{plan['icon']} {plan['name']} - ₹{plan['price']}", 
                        callback_data=f"select_plan_{plan['tier']}"
                    )
                ])
            
            text += "🎬 **Same premium features in all plans!**"
            
            keyboard_buttons.append([InlineKeyboardButton("🔙 BACK", callback_data="back_to_start")])
            keyboard = InlineKeyboardMarkup(keyboard_buttons)
            
            try:
                await callback_query.message.edit_text(text, reply_markup=keyboard)
            except:
                await callback_query.answer("Premium plans!")
        else:
            await callback_query.answer("Premium system not available!", show_alert=True)
    
    # ✅ SCREENSHOT CALLBACK
    @bot.on_callback_query(filters.regex(r"^send_screenshot_"))
    async def send_screenshot_callback(client, callback_query):
        payment_id = callback_query.data.split('_')[2]
        
        text = (
            "📸 **Please send the payment screenshot now**\n\n"
            "1. Take a clear screenshot of the payment\n"
            "2. Send it to this chat\n"
            "3. Our admin will verify and activate your premium\n\n"
            f"**Payment ID:** `{payment_id}`\n"
            "⏰ Please send within 1 hour of payment"
        )
        
        await callback_query.answer("Please send screenshot now!", show_alert=True)
        
        await callback_query.message.reply_text(text)
        
        try:
            await callback_query.message.delete()
        except:
            pass
    
    # ✅ ADMIN DASHBOARD
    @bot.on_message(filters.command("admin") & filters.user(config.ADMIN_IDS))
    async def admin_dashboard_command(client, message):
        """Admin dashboard main menu"""
        admin_text = (
            "🛠️ **SK4FiLM Admin Dashboard**\n\n"
            "**Available Commands:**\n"
            "• `/users` - View all users\n"
            "• `/pending` - View pending payments\n"
            "• `/addpremium <id> <days> <plan>` - Add premium\n"
            "• `/removepremium <id>` - Remove premium\n"
            "• `/checkpremium <id>` - Check premium\n"
            "• `/stats` - Bot statistics\n"
            "• `/autodelete` - Auto-delete system status\n"
            "• `/backup` - Backup database\n"
            "• `/broadcast` - Send message to all users\n\n"
            "**Payment Screenshots:**\n"
            "Screenshots are automatically forwarded to all admins when users send them."
        )
        
        await message.reply_text(admin_text, disable_web_page_preview=True)
    
    # ✅ VIEW ALL USERS
    @bot.on_message(filters.command("users") & filters.user(config.ADMIN_IDS))
    async def users_command(client, message):
        """View all registered users"""
        try:
            if bot_instance.premium_system:
                users_data = await bot_instance.premium_system.get_all_users()
                
                if not users_data:
                    await message.reply_text("📭 No users found in database.")
                    return
                
                page = int(message.command[1]) if len(message.command) > 1 else 1
                per_page = 10
                start_idx = (page - 1) * per_page
                end_idx = start_idx + per_page
                
                users_page = users_data[start_idx:end_idx]
                
                users_text = f"👥 **Registered Users** (Page {page})\n\n"
                
                for i, user in enumerate(users_page, start_idx + 1):
                    user_id = user.get('user_id', 'Unknown')
                    username = user.get('username', 'No username')
                    first_name = user.get('first_name', 'Unknown')
                    last_name = user.get('last_name', '')
                    premium = user.get('premium', '❌')
                    last_active = user.get('last_active', 'Never')
                    
                    users_text += f"{i}. **{first_name} {last_name}**\n"
                    users_text += f"   👤 ID: `{user_id}`\n"
                    users_text += f"   📛 @{username}\n"
                    users_text += f"   ⭐ Premium: {premium}\n"
                    users_text += f"   ⏰ Last Active: {last_active}\n\n"
                
                total_pages = (len(users_data) + per_page - 1) // per_page
                
                keyboard_buttons = []
                if page > 1:
                    keyboard_buttons.append(InlineKeyboardButton("⬅️ Previous", callback_data=f"users_page_{page-1}"))
                if page < total_pages:
                    keyboard_buttons.append(InlineKeyboardButton("Next ➡️", callback_data=f"users_page_{page+1}"))
                
                if keyboard_buttons:
                    keyboard = InlineKeyboardMarkup([keyboard_buttons])
                    await message.reply_text(users_text, reply_markup=keyboard)
                else:
                    await message.reply_text(users_text)
            else:
                await message.reply_text("❌ Premium system not available")
                
        except Exception as e:
            logger.error(f"Users command error: {e}")
            await message.reply_text(f"❌ Error: {str(e)[:100]}")
    
    # ✅ VIEW PENDING PAYMENTS
    @bot.on_message(filters.command("pending") & filters.user(config.ADMIN_IDS))
    async def pending_payments_command(client, message):
        """View pending payment screenshots"""
        try:
            pending_payments = await bot_instance.get_pending_payments()
            
            if not pending_payments:
                await message.reply_text("✅ No pending payments.")
                return
            
            pending_text = "💰 **Pending Payments**\n\n"
            
            for i, payment in enumerate(pending_payments, 1):
                user_id = payment.get('user_id', 'Unknown')
                amount = payment.get('amount', 0)
                plan = payment.get('plan', 'Unknown')
                age_hours = payment.get('age_hours', 0)
                screenshot_sent = "✅ Yes" if payment.get('screenshot_sent') else "❌ No"
                
                try:
                    user = await client.get_users(user_id)
                    user_name = f"{user.first_name or ''} {user.last_name or ''}".strip() or f"User {user_id}"
                    username = f"@{user.username}" if user.username else "No username"
                except:
                    user_name = f"User {user_id}"
                    username = "Unknown"
                
                pending_text += f"{i}. **{user_name}**\n"
                pending_text += f"   👤 {username}\n"
                pending_text += f"   💳 Payment ID: `{payment.get('payment_id', 'Unknown')}`\n"
                pending_text += f"   💰 Amount: ₹{amount}\n"
                pending_text += f"   📋 Plan: {plan}\n"
                pending_text += f"   📸 Screenshot: {screenshot_sent}\n"
                pending_text += f"   ⏰ Age: {age_hours} hours\n\n"
            
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🔄 Refresh", callback_data="refresh_pending")],
                [InlineKeyboardButton("📋 View All Users", callback_data="admin_users")]
            ])
            
            await message.reply_text(pending_text, reply_markup=keyboard)
            
        except Exception as e:
            logger.error(f"Pending payments command error: {e}")
            await message.reply_text(f"❌ Error: {str(e)[:100]}")
    
    # ✅ ADD PREMIUM USER
    @bot.on_message(filters.command("addpremium") & filters.user(config.ADMIN_IDS))
    async def add_premium_command(client, message):
        """Add premium user command for admins"""
        try:
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
            
            if plan_type not in ['basic', 'premium', 'gold', 'diamond']:
                await message.reply_text(
                    "❌ **Invalid plan type**\n\n"
                    "Use: `basic`, `premium`, `gold`, or `diamond`\n"
                    "Example: `/addpremium 123456789 30 basic`"
                )
                return
            
            if days <= 0:
                await message.reply_text("❌ Days must be greater than 0")
                return
            
            try:
                user = await client.get_users(user_id)
                user_name = f"{user.first_name or ''} {user.last_name or ''}".strip() or f"User {user_id}"
                username = f"@{user.username}" if user.username else "No username"
            except:
                user_name = f"User {user_id}"
                username = "Unknown"
            
            if bot_instance.premium_system:
                success = await bot_instance.premium_system.add_premium_subscription(
                    user_id=user_id,
                    tier_name=plan_type.capitalize(),
                    days_valid=days,
                    payment_method="admin",
                    payment_id=f"admin_{int(time.time())}"
                )
                
                if success:
                    admin_msg = (
                        f"✅ **Premium User Added Successfully!**\n\n"
                        f"**User:** {user_name}\n"
                        f"**ID:** `{user_id}`\n"
                        f"**Username:** {username}\n"
                        f"**Plan:** {plan_type.capitalize()}\n"
                        f"**Duration:** {days} days\n\n"
                        f"User gets ALL premium features!"
                    )
                    
                    await message.reply_text(admin_msg)
                    
                    try:
                        await client.send_message(
                            user_id,
                            f"🎉 **Congratulations!** 🎉\n\n"
                            f"You've been upgraded to **{plan_type.capitalize()} Premium** by admin!\n\n"
                            f"✅ **Plan:** {plan_type.capitalize()}\n"
                            f"📅 **Valid for:** {days} days\n"
                            f"⭐ **You get ALL premium features:**\n"
                            f"• All Quality (480p-4K)\n"
                            f"• Unlimited Downloads\n"
                            f"• No verification required\n"
                            f"• VIP Support 24/7\n"
                            f"• No Ads\n"
                            f"• Instant Downloads\n\n"
                            f"🎬 **Enjoy unlimited downloads!**"
                        )
                    except Exception as e:
                        logger.warning(f"Could not notify user {user_id}: {e}")
                else:
                    await message.reply_text("❌ Failed to add premium subscription. Check logs.")
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
    
    # ✅ REMOVE PREMIUM
    @bot.on_message(filters.command("removepremium") & filters.user(config.ADMIN_IDS))
    async def remove_premium_command(client, message):
        """Remove premium user command for admins"""
        try:
            if len(message.command) < 2:
                await message.reply_text(
                    "❌ **Usage:** `/removepremium <user_id>`\n\n"
                    "**Example:** `/removepremium 123456789`"
                )
                return
            
            user_id = int(message.command[1])
            
            if bot_instance.premium_system:
                success = await bot_instance.premium_system.remove_premium_subscription(user_id)
                
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
    
    # ✅ CHECK PREMIUM STATUS
    @bot.on_message(filters.command("checkpremium") & filters.user(config.ADMIN_IDS))
    async def check_premium_command(client, message):
        """Check premium status of user"""
        try:
            if len(message.command) < 2:
                await message.reply_text(
                    "❌ **Usage:** `/checkpremium <user_id>`\n\n"
                    "**Example:** `/checkpremium 123456789`"
                )
                return
            
            user_id = int(message.command[1])
            
            if bot_instance.premium_system:
                is_premium = await bot_instance.premium_system.is_premium_user(user_id)
                
                if is_premium:
                    details = await bot_instance.premium_system.get_subscription_details(user_id)
                    
                    try:
                        user = await client.get_users(user_id)
                        user_name = f"{user.first_name or ''} {user.last_name or ''}".strip() or f"User {user_id}"
                        username = f"@{user.username}" if user.username else "No username"
                    except:
                        user_name = f"User {user_id}"
                        username = "Unknown"
                    
                    await message.reply_text(
                        f"✅ **Premium User Found**\n\n"
                        f"**User:** {user_name}\n"
                        f"**ID:** `{user_id}`\n"
                        f"**Username:** {username}\n"
                        f"**Plan:** {details.get('tier_name', 'Unknown')}\n"
                        f"**Status:** {details.get('status', 'Unknown')}\n"
                        f"**Days Left:** {details.get('days_remaining', 0)}\n"
                        f"**Expires:** {details.get('expires_at', 'Unknown')}\n\n"
                        f"✅ **Has ALL premium features!**"
                    )
                else:
                    await message.reply_text(
                        f"❌ **Not a Premium User**\n\n"
                        f"User ID: `{user_id}`\n"
                        f"This user does not have premium access."
                    )
            else:
                await message.reply_text("❌ Premium system not available")
                
        except ValueError:
            await message.reply_text("❌ Invalid user ID. Must be a number.")
        except Exception as e:
            logger.error(f"Check premium command error: {e}")
            await message.reply_text(f"❌ Error: {str(e)[:100]}")
    
    # ✅ STATISTICS COMMAND
    @bot.on_message(filters.command("stats") & filters.user(config.ADMIN_IDS))
    async def stats_command(client, message):
        """Show bot statistics"""
        try:
            if bot_instance.premium_system:
                stats = await bot_instance.premium_system.get_statistics()
                
                stats_text = (
                    f"📊 **SK4FiLM Bot Statistics**\n\n"
                    f"👥 **Total Users:** {stats.get('total_users', 0)}\n"
                    f"⭐ **Premium Users:** {stats.get('premium_users', 0)}\n"
                    f"✅ **Active Premium:** {stats.get('active_premium', 0)}\n"
                    f"📥 **Total Downloads:** {stats.get('total_downloads', 0)}\n"
                    f"💾 **Total Data Sent:** {stats.get('total_data_sent', '0 GB')}\n\n"
                    f"💰 **Pending Payments:** {len(await bot_instance.get_pending_payments())}\n\n"
                    f"🔄 **System Status:**\n"
                    f"• Bot: {'✅ Online' if bot_instance.bot_started else '❌ Offline'}\n"
                    f"• User Client: {'✅ Connected' if bot_instance.user_session_ready else '❌ Disconnected'}\n"
                    f"• Verification: {'✅ Active' if bot_instance.verification_system else '❌ Inactive'}\n"
                    f"• Premium: {'✅ Active' if bot_instance.premium_system else '❌ Inactive'}\n"
                )
                
                await message.reply_text(stats_text, disable_web_page_preview=True)
            else:
                await message.reply_text("❌ Premium system not available for stats")
                
        except Exception as e:
            logger.error(f"Stats command error: {e}")
            await message.reply_text(f"❌ Error getting stats: {str(e)[:100]}")
    
    # ✅ START COMMAND HANDLER
    @bot.on_message(filters.command("start"))
    async def handle_start_command(client, message):
        """Handle /start command"""
        user_name = message.from_user.first_name or "User"
        user_id = message.from_user.id
        
        if len(message.command) > 1:
            file_text = ' '.join(message.command[1:])
            await handle_file_request(client, message, file_text, bot_instance)
            return
        
        welcome_text = (
            f"🎬 **Welcome to SK4FiLM, {user_name}!**\n\n"
            f"🌐 **Visit:** {config.WEBSITE_URL}\n\n"
            "**How to download:**\n"
            "1. Visit website above\n"
            "2. Search for movies\n"
            "3. Click download button\n"
            "4. File will appear here automatically\n\n"
            "🎬 **Happy watching!**"
        )
        
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🌐 OPEN WEBSITE", url=config.WEBSITE_URL)],
            [InlineKeyboardButton("📢 JOIN CHANNEL", url=config.MAIN_CHANNEL_LINK)],
            [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")]
        ])
        
        await message.reply_text(welcome_text, reply_markup=keyboard, disable_web_page_preview=True)
    
    # ✅ Handle direct file format messages
    @bot.on_message(filters.private & filters.regex(r'^-?\d+_\d+(_\w+)?$'))
    async def handle_direct_file_request(client, message):
        """Handle direct file format messages"""
        file_text = message.text.strip()
        await handle_file_request(client, message, file_text, bot_instance)
    
    # ✅ GET VERIFIED CALLBACK
    @bot.on_callback_query(filters.regex(r"^get_verified$"))
    async def get_verified_callback(client, callback_query):
        """Get verification link"""
        user_id = callback_query.from_user.id
        
        if bot_instance.verification_system:
            verification_data = await bot_instance.verification_system.create_verification_link(user_id)
            
            text = (
                "🔗 **Verification Required**\n\n"
                "Join channel to verify:\n\n"
                f"🔗 **Link:** {verification_data['short_url']}\n"
                f"⏰ **Valid:** {verification_data['valid_for_hours']} hours\n\n"
                "Click link, join channel, then try download again."
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
                await callback_query.answer("Click VERIFY NOW button!")
        else:
            await callback_query.answer("Verification not available!", show_alert=True)
    
    # ✅ BACK TO START
    @bot.on_callback_query(filters.regex(r"^back_to_start$"))
    async def back_to_start_callback(client, callback_query):
        user_name = callback_query.from_user.first_name or "User"
        
        text = (
            f"🎬 **Welcome back, {user_name}!**\n\n"
            f"Visit {config.WEBSITE_URL} to download movies.\n"
            "Click download button on website and file will appear here."
        )
        
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🌐 OPEN WEBSITE", url=config.WEBSITE_URL)],
            [InlineKeyboardButton("📢 JOIN CHANNEL", url=config.MAIN_CHANNEL_LINK)],
            [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")]
        ])
        
        try:
            await callback_query.message.edit_text(
                text=text,
                reply_markup=keyboard,
                disable_web_page_preview=True
            )
        except:
            await callback_query.answer("Already on home page!")
    
    # ✅ FORWARD SCREENSHOTS TO ALL ADMINS
    @bot.on_message(filters.private & (filters.photo | filters.document))
    async def forward_screenshots_to_admins(client, message):
        """Forward payment screenshots to all admins"""
        user_id = message.from_user.id
        
        if user_id in config.ADMIN_IDS:
            return
        
        is_screenshot = False
        
        if message.photo:
            is_screenshot = True
        elif message.document and message.document.mime_type:
            image_types = ['image/jpeg', 'image/png', 'image/jpg', 'image/webp']
            if any(img_type in message.document.mime_type for img_type in image_types):
                is_screenshot = True
        
        if not is_screenshot:
            return
        
        try:
            user = await client.get_users(user_id)
            user_name = f"{user.first_name or ''} {user.last_name or ''}".strip() or f"User {user_id}"
            username = f"@{user.username}" if user.username else "No username"
        except:
            user_name = f"User {user_id}"
            username = "Unknown"
        
        alert_text = (
            f"📸 **New Payment Screenshot Received!**\n\n"
            f"👤 **User:** {user_name}\n"
            f"📛 **Username:** {username}\n"
            f"🆔 **User ID:** `{user_id}`\n"
            f"🕒 **Time:** {datetime.now().strftime('%H:%M:%S')}\n\n"
            f"**Admin Actions:**\n"
            f"• Check pending payments: `/pending`\n"
            f"• Add premium: `/addpremium {user_id} 30 basic`\n"
            f"• View user info: `/checkpremium {user_id}`"
        )
        
        for admin_id in config.ADMIN_IDS:
            try:
                await message.forward(admin_id)
                
                await client.send_message(
                    admin_id,
                    alert_text,
                    reply_markup=InlineKeyboardMarkup([
                        [
                            InlineKeyboardButton("✅ Add Premium", callback_data=f"admin_addpremium_{user_id}"),
                            InlineKeyboardButton("👤 Check User", callback_data=f"admin_checkuser_{user_id}")
                        ],
                        [InlineKeyboardButton("📋 Pending Payments", callback_data="admin_pending")]
                    ])
                )
                
                logger.info(f"✅ Screenshot forwarded to admin {admin_id} from user {user_id}")
                
            except Exception as e:
                logger.error(f"Failed to forward screenshot to admin {admin_id}: {e}")
        
        await message.reply_text(
            "✅ **Screenshot received!**\n\n"
            "Our admin will verify your payment and activate your premium within 24 hours.\n"
            "Thank you for choosing SK4FiLM! 🎬\n\n"
            "You will receive a confirmation message when activated.",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 BACK TO START", callback_data="back_to_start")]
            ])
        )
    
    # ✅ ADMIN CALLBACK QUERIES
    @bot.on_callback_query(filters.regex(r"^admin_"))
    async def admin_callback_handler(client, callback_query):
        """Handle admin callback queries"""
        data = callback_query.data
        admin_id = callback_query.from_user.id
        
        if admin_id not in config.ADMIN_IDS:
            await callback_query.answer("❌ Access denied!", show_alert=True)
            return
        
        if data.startswith("admin_addpremium_"):
            user_id = int(data.split('_')[2])
            
            text = (
                f"⭐ **Add Premium for User**\n\n"
                f"**User ID:** `{user_id}`\n\n"
                f"Select plan and duration:"
            )
            
            keyboard = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("30 Days Basic", callback_data=f"quick_add_{user_id}_30_basic"),
                    InlineKeyboardButton("30 Days Premium", callback_data=f"quick_add_{user_id}_30_premium")
                ],
                [
                    InlineKeyboardButton("60 Days Gold", callback_data=f"quick_add_{user_id}_60_gold"),
                    InlineKeyboardButton("180 Days Diamond", callback_data=f"quick_add_{user_id}_180_diamond")
                ],
                [InlineKeyboardButton("🔙 Back", callback_data="admin_dashboard")]
            ])
            
            await callback_query.message.edit_text(text, reply_markup=keyboard)
            
        elif data.startswith("quick_add_"):
            parts = data.split('_')
            user_id = int(parts[2])
            days = int(parts[3])
            plan_type = parts[4]
            
            if bot_instance.premium_system:
                success = await bot_instance.premium_system.add_premium_subscription(
                    user_id=user_id,
                    tier_name=plan_type.capitalize(),
                    days_valid=days,
                    payment_method="admin_quick",
                    payment_id=f"admin_quick_{int(time.time())}"
                )
                
                if success:
                    try:
                        user = await client.get_users(user_id)
                        user_name = f"{user.first_name or ''} {user.last_name or ''}".strip()
                    except:
                        user_name = f"User {user_id}"
                    
                    await callback_query.answer(f"✅ Premium added for {user_name}!", show_alert=True)
                    
                    await callback_query.message.edit_text(
                        f"✅ **Premium Added Successfully!**\n\n"
                        f"**User:** {user_name}\n"
                        f"**Plan:** {plan_type.capitalize()}\n"
                        f"**Duration:** {days} days\n\n"
                        f"User gets ALL premium features!\n"
                        f"User has been notified.",
                        reply_markup=InlineKeyboardMarkup([
                            [InlineKeyboardButton("📋 Back to Admin", callback_data="admin_dashboard")]
                        ])
                    )
                    
                    try:
                        await client.send_message(
                            user_id,
                            f"🎉 **Congratulations!** 🎉\n\n"
                            f"Your premium has been activated!\n\n"
                            f"✅ **Plan:** {plan_type.capitalize()}\n"
                            f"📅 **Valid for:** {days} days\n"
                            f"⭐ **You get ALL premium features:**\n"
                            f"• All Quality (480p-4K)\n"
                            f"• Unlimited Downloads\n"
                            f"• No verification required\n"
                            f"• VIP Support 24/7\n"
                            f"• No Ads\n"
                            f"• Instant Downloads\n\n"
                            f"🎬 **Enjoy unlimited downloads!**"
                        )
                    except:
                        pass
                else:
                    await callback_query.answer("❌ Failed to add premium", show_alert=True)
            else:
                await callback_query.answer("❌ Premium system not available", show_alert=True)
                
        elif data == "admin_pending":
            pending_payments = await bot_instance.get_pending_payments()
            
            if not pending_payments:
                text = "✅ No pending payments."
                keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back", callback_data="admin_dashboard")]])
            else:
                text = "💰 **Pending Payments**\n\n"
                for i, payment in enumerate(pending_payments[:5], 1):
                    user_id = payment.get('user_id', 'Unknown')
                    amount = payment.get('amount', 0)
                    plan = payment.get('plan', 'Unknown')
                    
                    text += f"{i}. User ID: `{user_id}`\n"
                    text += f"   Amount: ₹{amount} ({plan})\n\n"
                
                keyboard_buttons = []
                for i, payment in enumerate(pending_payments[:5], 1):
                    user_id = payment.get('user_id')
                    keyboard_buttons.append([InlineKeyboardButton(
                        f"✅ Approve #{i}", 
                        callback_data=f"approve_payment_{payment.get('payment_id', '')}"
                    )])
                
                keyboard_buttons.append([InlineKeyboardButton("🔙 Back", callback_data="admin_dashboard")])
                keyboard = InlineKeyboardMarkup(keyboard_buttons)
            
            await callback_query.message.edit_text(text, reply_markup=keyboard)
            
        elif data.startswith("approve_payment_"):
            payment_id = data.split('_')[2]
            
            await bot_instance.complete_payment(payment_id)
            
            await callback_query.answer("✅ Payment approved!", show_alert=True)
            
            await admin_callback_handler(client, callback_query)
            
        elif data == "admin_dashboard":
            text = (
                "🛠️ **SK4FiLM Admin Dashboard**\n\n"
                "**Quick Actions:**"
            )
            
            keyboard = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("👥 Users", callback_data="admin_users"),
                    InlineKeyboardButton("💰 Pending", callback_data="admin_pending")
                ],
                [
                    InlineKeyboardButton("📊 Stats", callback_data="admin_stats"),
                    InlineKeyboardButton("⚙️ Settings", callback_data="admin_settings")
                ],
                [
                    InlineKeyboardButton("⏰ Auto-Delete", callback_data="admin_autodelete")
                ]
            ])
            
            await callback_query.message.edit_text(text, reply_markup=keyboard)
            
        elif data == "admin_users":
            if bot_instance.premium_system:
                users_data = await bot_instance.premium_system.get_all_users()
                
                if users_data:
                    text = f"👥 **Total Users:** {len(users_data)}\n\n"
                    for i, user in enumerate(users_data[:10], 1):
                        user_id = user.get('user_id', 'Unknown')
                        premium = user.get('premium', '❌')
                        text += f"{i}. `{user_id}` - {premium}\n"
                    
                    if len(users_data) > 10:
                        text += f"\n... and {len(users_data) - 10} more users"
                else:
                    text = "📭 No users found."
            else:
                text = "❌ Premium system not available"
            
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Back", callback_data="admin_dashboard")]
            ])
            
            await callback_query.message.edit_text(text, reply_markup=keyboard)
            
        elif data == "admin_stats":
            if bot_instance.premium_system:
                stats = await bot_instance.premium_system.get_statistics()
                text = (
                    f"📊 **Bot Statistics**\n\n"
                    f"👥 Users: {stats.get('total_users', 0)}\n"
                    f"⭐ Premium: {stats.get('premium_users', 0)}\n"
                    f"📥 Downloads: {stats.get('total_downloads', 0)}\n"
                    f"💾 Data: {stats.get('total_data_sent', '0 GB')}"
                )
            else:
                text = "❌ Statistics not available"
            
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Back", callback_data="admin_dashboard")]
            ])
            
            await callback_query.message.edit_text(text, reply_markup=keyboard)
        
        elif data == "admin_settings":
            text = (
                "⚙️ **Bot Settings**\n\n"
                f"• Auto-delete: {getattr(config, 'AUTO_DELETE_TIME', 10)} minutes\n"
                f"• Admins: {len(config.ADMIN_IDS)}\n"
                f"• User Client: {'✅ Connected' if bot_instance.user_session_ready else '❌ Disconnected'}\n"
                f"• Bot Status: {'✅ Online' if bot_instance.bot_started else '❌ Offline'}"
            )
            
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Back", callback_data="admin_dashboard")]
            ])
            
            await callback_query.message.edit_text(text, reply_markup=keyboard)
        
        elif data == "admin_autodelete":
            # Redirect to auto-delete status
            await autodelete_status_command(client, callback_query.message)
            await callback_query.answer()

    logger.info("✅ Bot handlers setup complete with FIXED auto-delete system")
