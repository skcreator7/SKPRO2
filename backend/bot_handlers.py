"""
bot_handlers.py - Telegram Bot Handlers for SK4FiLM
FULL WORKING VERSION - Proper Text Reply & File Send
"""

import asyncio
import logging
import secrets
import re
import time
import traceback
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from collections import defaultdict

# ✅ Proper Pyrogram imports
from pyrogram import Client, filters
from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton, Message, CallbackQuery
from pyrogram.errors import FloodWait, BadRequest, MessageDeleteForbidden, UserNotParticipant

logger = logging.getLogger(__name__)

# ✅ Utility function for file size formatting
def format_size(size_in_bytes: Union[int, float, None]) -> str:
    """Format file size in human-readable format"""
    if size_in_bytes is None or size_in_bytes == 0:
        return "0 B"
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.1f} {unit}"
        size_in_bytes /= 1024.0
    return f"{size_in_bytes:.1f} PB"

class VerificationSystem:
    """Simplified verification system for testing"""
    def __init__(self, config, db_manager=None):
        self.config = config
        self.db_manager = db_manager
        self.verification_tokens = {}  # token -> (user_id, expiry_time)
        self.verified_users = {}  # user_id -> expiry_time
    
    async def create_verification_link(self, user_id: int) -> Dict[str, Any]:
        """Create verification link"""
        token = secrets.token_urlsafe(32)
        expiry = time.time() + (6 * 3600)  # 6 hours
        
        self.verification_tokens[token] = {
            'user_id': user_id,
            'expiry': expiry,
            'created_at': datetime.now().isoformat()
        }
        
        short_url = f"https://t.me/{self.config.BOT_USERNAME}?start=verify_{token}"
        
        return {
            'token': token,
            'short_url': short_url,
            'valid_for_hours': 6
        }
    
    async def verify_user_token(self, token: str) -> tuple:
        """Verify user token"""
        if token not in self.verification_tokens:
            return False, None, "Invalid or expired verification token"
        
        data = self.verification_tokens[token]
        
        if time.time() > data['expiry']:
            del self.verification_tokens[token]
            return False, None, "Verification token expired"
        
        user_id = data['user_id']
        
        # Mark user as verified for 6 hours
        self.verified_users[user_id] = {
            'expiry': time.time() + (6 * 3600),
            'verified_at': datetime.now().isoformat()
        }
        
        # Remove used token
        del self.verification_tokens[token]
        
        return True, user_id, "Verification successful"
    
    async def check_user_verified(self, user_id: int, premium_system=None) -> tuple:
        """Check if user is verified"""
        # Check if user has active premium (overrides verification)
        if premium_system:
            is_premium = await premium_system.is_premium_user(user_id)
            if is_premium:
                return True, "premium"
        
        # Check verification
        if user_id in self.verified_users:
            if time.time() > self.verified_users[user_id]['expiry']:
                del self.verified_users[user_id]
                return False, "verification_expired"
            return True, "verified"
        
        return False, "not_verified"
    
    async def start_cleanup_task(self):
        """Start cleanup task for expired tokens"""
        async def cleanup():
            while True:
                try:
                    await asyncio.sleep(3600)  # Check every hour
                    current_time = time.time()
                    
                    # Clean expired tokens
                    expired_tokens = [
                        token for token, data in self.verification_tokens.items()
                        if current_time > data['expiry']
                    ]
                    for token in expired_tokens:
                        del self.verification_tokens[token]
                    
                    # Clean expired verifications
                    expired_users = [
                        user_id for user_id, data in self.verified_users.items()
                        if current_time > data['expiry']
                    ]
                    for user_id in expired_users:
                        del self.verified_users[user_id]
                    
                    logger.info(f"🧹 Verification cleanup: {len(expired_tokens)} tokens, {len(expired_users)} users")
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Cleanup error: {e}")
        
        self.cleanup_task = asyncio.create_task(cleanup())
    
    async def stop_cleanup_task(self):
        """Stop cleanup task"""
        if hasattr(self, 'cleanup_task'):
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

class PremiumTier:
    """Premium tier enum"""
    BASIC = "basic"
    PREMIUM = "premium"
    GOLD = "gold"
    DIAMOND = "diamond"

class PremiumSystem:
    """Simplified premium system for testing"""
    def __init__(self, config, db_manager=None):
        self.config = config
        self.db_manager = db_manager
        self.premium_users = {}  # user_id -> premium_data
        self.pending_payments = {}  # payment_id -> payment_data
        self.download_stats = defaultdict(lambda: {'count': 0, 'size': 0})
    
    async def is_premium_user(self, user_id: int) -> bool:
        """Check if user is premium"""
        if user_id in self.premium_users:
            data = self.premium_users[user_id]
            if time.time() > data['expires_at']:
                del self.premium_users[user_id]
                return False
            return True
        return False
    
    async def get_subscription_details(self, user_id: int) -> Dict[str, Any]:
        """Get subscription details"""
        if user_id not in self.premium_users:
            return {'status': 'free', 'tier_name': 'Free'}
        
        data = self.premium_users[user_id]
        expires_at = datetime.fromtimestamp(data['expires_at'])
        days_left = max(0, (expires_at - datetime.now()).days)
        
        return {
            'status': 'premium',
            'tier': data['tier'],
            'tier_name': data['tier'].capitalize(),
            'purchased_at': data.get('purchased_at', 'Unknown'),
            'expires_at': expires_at.strftime('%d %b %Y'),
            'days_remaining': days_left
        }
    
    async def get_my_premium_info(self, user_id: int) -> str:
        """Get premium info text for user"""
        if await self.is_premium_user(user_id):
            details = await self.get_subscription_details(user_id)
            text = (
                f"⭐ **PREMIUM USER** ⭐\n\n"
                f"**Plan:** {details['tier_name']}\n"
                f"**Status:** ✅ Active\n"
                f"**Days Left:** {details['days_remaining']}\n"
                f"**Expires:** {details['expires_at']}\n\n"
                f"✅ **Benefits:**\n"
                f"• No verification required\n"
                f"• Unlimited downloads\n"
                f"• All quality formats\n"
                f"• Priority support\n"
                f"⏰ **Files auto-delete after 5 minutes**\n\n"
                f"🎬 **Enjoy unlimited downloads!**"
            )
        else:
            text = (
                "🔓 **FREE USER** 🔓\n\n"
                "**Status:** Free (Verification Required)\n"
                "**Access:** Limited to 6 hours after verification\n\n"
                "⚠️ **Limitations:**\n"
                "• Need verification every 6 hours\n"
                "• Same download speeds as premium\n"
                "⏰ **Files auto-delete after 5 minutes**\n\n"
                "⭐ **Upgrade to Premium for:**\n"
                "• No verification needed\n"
                "• Priority support\n"
                "• All other features same\n"
                "⏰ **Files still auto-delete after 5 minutes**\n\n"
                "Click /buy to upgrade!"
            )
        return text
    
    async def get_available_plans_text(self) -> str:
        """Get available plans text"""
        text = (
            "💰 **PREMIUM PLANS** 💰\n\n"
            "🥉 **Basic Plan** - ₹99/month\n"
            "✅ No verification needed\n"
            "✅ All quality (480p-4K)\n"
            "✅ Unlimited downloads\n"
            "✅ Priority support\n"
            "⏰ Files auto-delete after 5 minutes\n\n"
            "🥈 **Premium Plan** - ₹199/month\n"
            "✅ Everything in Basic +\n"
            "✅ Faster verification (if needed)\n"
            "✅ VIP support\n"
            "⏰ Files auto-delete after 5 minutes\n\n"
            "🥇 **Gold Plan** - ₹299/2 months\n"
            "✅ Everything in Premium +\n"
            "✅ Extended support hours\n"
            "⏰ Files auto-delete after 5 minutes\n\n"
            "💎 **Diamond Plan** - ₹499/3 months\n"
            "✅ Everything in Gold +\n"
            "✅ Highest priority support\n"
            "⏰ Files auto-delete after 5 minutes\n\n"
            "**Note:** All plans have 5-minute auto-delete for security\n"
            "**Payment:** UPI or Bank Transfer\n"
            "**Process:** Send screenshot after payment"
        )
        return text
    
    async def initiate_purchase(self, user_id: int, tier: str) -> Optional[Dict[str, Any]]:
        """Initiate purchase"""
        tier_names = {
            PremiumTier.BASIC: "Basic",
            PremiumTier.PREMIUM: "Premium",
            PremiumTier.GOLD: "Gold",
            PremiumTier.DIAMOND: "Diamond"
        }
        
        amounts = {
            PremiumTier.BASIC: 99,
            PremiumTier.PREMIUM: 199,
            PremiumTier.GOLD: 299,
            PremiumTier.DIAMOND: 499
        }
        
        durations = {
            PremiumTier.BASIC: 30,
            PremiumTier.PREMIUM: 30,
            PremiumTier.GOLD: 60,
            PremiumTier.DIAMOND: 90
        }
        
        payment_id = f"PAY_{secrets.token_hex(8).upper()}"
        
        payment_data = {
            'payment_id': payment_id,
            'user_id': user_id,
            'tier': tier,
            'tier_name': tier_names.get(tier, "Unknown"),
            'amount': amounts.get(tier, 0),
            'duration_days': durations.get(tier, 30),
            'status': 'pending',
            'created_at': datetime.now().isoformat(),
            'expires_at': time.time() + (24 * 3600),  # 24 hours
            'screenshot_sent': False
        }
        
        self.pending_payments[payment_id] = payment_data
        return payment_data
    
    async def get_payment_instructions_text(self, payment_id: str) -> str:
        """Get payment instructions"""
        if payment_id not in self.pending_payments:
            return "❌ Payment not found!"
        
        payment = self.pending_payments[payment_id]
        
        upi_id = getattr(self.config, 'PAYMENT_UPI', 'your-upi@okaxis')
        bank_details = getattr(self.config, 'BANK_DETAILS', 'Bank: Example Bank\nAccount: 1234567890\nIFSC: EXMP0123456')
        
        text = (
            f"💰 **Payment Instructions - {payment['tier_name']} Plan**\n\n"
            f"**Amount:** ₹{payment['amount']}\n"
            f"**Duration:** {payment['duration_days']} days\n"
            f"**Payment ID:** `{payment_id}`\n\n"
            "**Payment Methods:**\n"
            f"1. **UPI:** `{upi_id}`\n"
            f"2. **Bank Transfer:**\n{bank_details}\n\n"
            "**Steps to Complete:**\n"
            "1. Make payment using above details\n"
            "2. Take a screenshot of successful payment\n"
            "3. Click 'SEND SCREENSHOT' button below\n"
            "4. Send the screenshot in this chat\n\n"
            "**Important:**\n"
            "• Include payment ID in screenshot if possible\n"
            "• Admin will verify within 24 hours\n"
            "• You'll get confirmation message\n"
            "⏰ **Files auto-delete after 5 minutes**"
        )
        return text
    
    async def process_payment_screenshot(self, user_id: int, message_id: int) -> bool:
        """Process payment screenshot"""
        # Find pending payment for this user
        for payment_id, payment in self.pending_payments.items():
            if payment['user_id'] == user_id and payment['status'] == 'pending':
                payment['screenshot_sent'] = True
                payment['screenshot_message_id'] = message_id
                payment['screenshot_sent_at'] = datetime.now().isoformat()
                return True
        return False
    
    async def approve_payment(self, admin_id: int, payment_id: str) -> tuple:
        """Approve payment"""
        if payment_id not in self.pending_payments:
            return False, "Payment not found"
        
        payment = self.pending_payments[payment_id]
        
        if payment['status'] != 'pending':
            return False, f"Payment already {payment['status']}"
        
        # Calculate expiry
        expires_at = time.time() + (payment['duration_days'] * 24 * 3600)
        
        # Add to premium users
        self.premium_users[payment['user_id']] = {
            'tier': payment['tier'],
            'tier_name': payment['tier_name'],
            'purchased_at': datetime.now().isoformat(),
            'expires_at': expires_at,
            'payment_id': payment_id,
            'approved_by': admin_id,
            'approved_at': datetime.now().isoformat()
        }
        
        # Update payment status
        payment['status'] = 'approved'
        payment['approved_at'] = datetime.now().isoformat()
        payment['approved_by'] = admin_id
        
        return True, f"Payment approved! User upgraded to {payment['tier_name']} for {payment['duration_days']} days."
    
    async def record_download(self, user_id: int, file_size: int, quality: str):
        """Record download for statistics"""
        self.download_stats[user_id]['count'] += 1
        self.download_stats[user_id]['size'] += file_size
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get statistics"""
        total_users = len(set(list(self.premium_users.keys()) + [u for u in self.download_stats.keys()]))
        
        active_premium = 0
        for user_id, data in self.premium_users.items():
            if time.time() < data['expires_at']:
                active_premium += 1
        
        total_downloads = sum(stats['count'] for stats in self.download_stats.values())
        total_data = sum(stats['size'] for stats in self.download_stats.values())
        
        return {
            'total_users': total_users,
            'premium_users': len(self.premium_users),
            'active_premium': active_premium,
            'free_users': total_users - len(self.premium_users),
            'total_downloads': total_downloads,
            'total_data_sent': format_size(total_data),
            'total_revenue': f"₹{len(self.premium_users) * 100}",  # Simplified
            'total_premium_sales': len(self.premium_users),
            'pending_payments': len([p for p in self.pending_payments.values() if p['status'] == 'pending']),
            'server_time': datetime.now().strftime('%d %b %Y, %H:%M:%S')
        }
    
    async def start_cleanup_task(self):
        """Start cleanup task"""
        async def cleanup():
            while True:
                try:
                    await asyncio.sleep(3600)  # Check every hour
                    current_time = time.time()
                    
                    # Clean expired premium users
                    expired_users = [
                        user_id for user_id, data in self.premium_users.items()
                        if current_time > data['expires_at']
                    ]
                    for user_id in expired_users:
                        del self.premium_users[user_id]
                    
                    # Clean expired pending payments
                    expired_payments = [
                        pid for pid, payment in self.pending_payments.items()
                        if payment['status'] == 'pending' and current_time > payment['expires_at']
                    ]
                    for pid in expired_payments:
                        del self.pending_payments[pid]
                    
                    logger.info(f"🧹 Premium cleanup: {len(expired_users)} users, {len(expired_payments)} payments")
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Premium cleanup error: {e}")
        
        self.cleanup_task = asyncio.create_task(cleanup())
    
    async def stop_cleanup_task(self):
        """Stop cleanup task"""
        if hasattr(self, 'cleanup_task'):
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

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
        self.verification_system = VerificationSystem(config, db_manager)
        self.premium_system = PremiumSystem(config, db_manager)
        self.PremiumTier = PremiumTier
        self.poster_fetcher = None
        self.cache_manager = None
        
        logger.info("✅ All systems initialized")
    
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
                try:
                    self.user_client = Client(
                        "user",
                        api_id=self.config.API_ID,
                        api_hash=self.config.API_HASH,
                        session_string=self.config.USER_SESSION_STRING
                    )
                    await self.user_client.start()
                    self.user_session_ready = True
                    logger.info("✅ User session started successfully")
                except Exception as e:
                    logger.warning(f"User session failed: {e}")
                    self.user_session_ready = False
            
            # Start bot
            await self.bot.start()
            self.bot_started = True
            logger.info("✅ Bot started successfully")
            
            # Setup handlers
            await self.setup_bot_handlers()
            
            # Start cleanup tasks
            asyncio.create_task(self.verification_system.start_cleanup_task())
            asyncio.create_task(self.premium_system.start_cleanup_task())
            
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
            await self.verification_system.stop_cleanup_task()
            await self.premium_system.stop_cleanup_task()
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
                    
                    success_count += 1
                    logger.info(f"✅ Admin notification sent to admin {admin_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to send notification to admin {admin_id}: {e}")
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Admin notification error: {e}")
            return False
    
    # ✅ RATE LIMITING & DUPLICATE PREVENTION
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
        data_str = str(request_data).encode('utf-8')
        request_hash = f"{user_id}_{request_type}_{hashlib.md5(data_str).hexdigest()}"
        
        if request_type == "verification":
            processing_dict = self.verification_processing
        else:
            processing_dict = self.processing_requests
        
        # Check if same request is already processing
        if request_hash in processing_dict:
            processing_time = processing_dict[request_hash]
            if time.time() - processing_time < 30:  # 30 second cooldown
                logger.debug(f"Duplicate request detected: {request_hash[:50]}...")
                return True
        
        # Mark as processing
        processing_dict[request_hash] = time.time()
        return False
    
    async def clear_processing_request(self, user_id, request_data, request_type="file"):
        """Clear from processing requests"""
        data_str = str(request_data).encode('utf-8')
        request_hash = f"{user_id}_{request_type}_{hashlib.md5(data_str).hexdigest()}"
        
        if request_type == "verification":
            self.verification_processing.pop(request_hash, None)
        else:
            self.processing_requests.pop(request_hash, None)
    
    # ✅ PREVENT DOUBLE REPLIES
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
        content_hash = f"{user_id}_{hashlib.md5(message_text.strip().encode()).hexdigest()}"
        current_time = time.time()
        
        if content_hash in self.last_message_content:
            last_time = self.last_message_content[content_hash]['time']
            if current_time - last_time < 5:  # 5 second cooldown for same content
                logger.debug(f"Duplicate message content from user {user_id}")
                return True
        
        # Store new content
        self.last_message_content[content_hash] = {
            'time': current_time,
            'text': message_text[:100]
        }
        return False

    async def setup_bot_handlers(self):
        """Setup all bot handlers"""
        config = self.config
        
        # ✅ USER COMMANDS
        
        @self.bot.on_message(filters.command("start") & filters.private)
        async def handle_start_command(client, message):
            """Handle /start command"""
            user_id = message.from_user.id
            
            # ✅ PREVENT DOUBLE REPLIES
            if not await self.should_reply(user_id, message.id):
                return
            
            user_name = message.from_user.first_name or "User"
            
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
        
        @self.bot.on_message(filters.command("mypremium") & filters.private)
        async def my_premium_command(client, message):
            """Check user's premium status"""
            user_id = message.from_user.id
            
            # ✅ PREVENT DOUBLE REPLIES
            if not await self.should_reply(user_id, message.id):
                return
            
            try:
                # Get premium info
                premium_info = await self.premium_system.get_my_premium_info(user_id)
                
                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")],
                    [InlineKeyboardButton("🌐 OPEN WEBSITE", url=config.WEBSITE_URL)]
                ])
                
                await message.reply_text(premium_info, reply_markup=keyboard, disable_web_page_preview=True)
                
            except Exception as e:
                logger.error(f"My premium command error: {e}")
                await message.reply_text("❌ Error fetching premium info. Please try again.")
        
        @self.bot.on_message(filters.command("plans") & filters.private)
        async def plans_command(client, message):
            """Show all premium plans"""
            user_id = message.from_user.id
            
            # ✅ PREVENT DOUBLE REPLIES
            if not await self.should_reply(user_id, message.id):
                return
            
            try:
                plans_text = await self.premium_system.get_available_plans_text()
                
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
        
        @self.bot.on_message(filters.command("buy") & filters.private)
        async def buy_command(client, message):
            """Initiate premium purchase"""
            user_id = message.from_user.id
            
            # ✅ PREVENT DOUBLE REPLIES
            if not await self.should_reply(user_id, message.id):
                return
            
            user_name = message.from_user.first_name or "User"
            
            # Check if already premium
            is_premium = await self.premium_system.is_premium_user(user_id)
            if is_premium:
                details = await self.premium_system.get_subscription_details(user_id)
                
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
        
        @self.bot.on_message(filters.command("help") & filters.private)
        async def help_command(client, message):
            """Show help message"""
            user_id = message.from_user.id
            
            # ✅ PREVENT DOUBLE REPLIES
            if not await self.should_reply(user_id, message.id):
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
        admin_ids = getattr(config, 'ADMIN_IDS', [])
        
        @self.bot.on_message(filters.command("addpremium") & filters.user(admin_ids))
        async def add_premium_command(client, message):
            """Add premium user command for admins"""
            try:
                # ✅ PREVENT DOUBLE REPLIES
                if not await self.should_reply(message.from_user.id, message.id):
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
                    'basic': self.PremiumTier.BASIC,
                    'premium': self.PremiumTier.PREMIUM,
                    'gold': self.PremiumTier.GOLD,
                    'diamond': self.PremiumTier.DIAMOND
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
                expires_at = time.time() + (days * 24 * 3600)
                
                self.premium_system.premium_users[user_id] = {
                    'tier': tier,
                    'tier_name': plan_type.capitalize(),
                    'purchased_at': datetime.now().isoformat(),
                    'expires_at': expires_at,
                    'approved_by': message.from_user.id,
                    'approved_at': datetime.now().isoformat(),
                    'reason': 'admin_command'
                }
                
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
                    
            except ValueError:
                await message.reply_text(
                    "❌ **Invalid parameters**\n\n"
                    "Correct format: `/addpremium <user_id> <days> <plan_type>`\n"
                    "Example: `/addpremium 123456789 30 basic`"
                )
            except Exception as e:
                logger.error(f"Add premium command error: {e}")
                await message.reply_text(f"❌ Error: {str(e)[:100]}")
        
        @self.bot.on_message(filters.command("removepremium") & filters.user(admin_ids))
        async def remove_premium_command(client, message):
            """Remove premium user command for admins"""
            try:
                # ✅ PREVENT DOUBLE REPLIES
                if not await self.should_reply(message.from_user.id, message.id):
                    return
                
                if len(message.command) < 2:
                    await message.reply_text(
                        "❌ **Usage:** `/removepremium <user_id>`\n\n"
                        "**Example:** `/removepremium 123456789`"
                    )
                    return
                
                user_id = int(message.command[1])
                
                if user_id in self.premium_system.premium_users:
                    del self.premium_system.premium_users[user_id]
                    await message.reply_text(
                        f"✅ **Premium Removed Successfully!**\n\n"
                        f"**User ID:** `{user_id}`\n"
                        f"Premium access has been revoked."
                    )
                else:
                    await message.reply_text("❌ User not found or not premium")
                    
            except ValueError:
                await message.reply_text("❌ Invalid user ID. Must be a number.")
            except Exception as e:
                logger.error(f"Remove premium command error: {e}")
                await message.reply_text(f"❌ Error: {str(e)[:100]}")
        
        @self.bot.on_message(filters.command("checkpremium") & filters.user(admin_ids))
        async def check_premium_command(client, message):
            """Check premium status of user"""
            try:
                # ✅ PREVENT DOUBLE REPLIES
                if not await self.should_reply(message.from_user.id, message.id):
                    return
                
                if len(message.command) < 2:
                    await message.reply_text(
                        "❌ **Usage:** `/checkpremium <user_id>`\n\n"
                        "**Example:** `/checkpremium 123456789`"
                    )
                    return
                
                user_id = int(message.command[1])
                
                user_info = await self.premium_system.get_subscription_details(user_id)
                
                # Get user info
                try:
                    user = await client.get_users(user_id)
                    user_name = f"{user.first_name or ''} {user.last_name or ''}".strip() or f"User {user_id}"
                    username = f"@{user.username}" if user.username else "No username"
                except:
                    user_name = f"User {user_id}"
                    username = "Unknown"
                
                if user_info['status'] == 'free':
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
                        f"**Joined:** {user_info.get('purchased_at', 'Unknown')}\n"
                        f"**Expires:** {user_info.get('expires_at', 'Unknown')}\n"
                        f"⏰ **Files auto-delete after 5 minutes**"
                    )
                    
            except ValueError:
                await message.reply_text("❌ Invalid user ID. Must be a number.")
            except Exception as e:
                logger.error(f"Check premium command error: {e}")
                await message.reply_text(f"❌ Error: {str(e)[:100]}")
        
        @self.bot.on_message(filters.command("stats") & filters.user(admin_ids))
        async def stats_command(client, message):
            """Show bot statistics"""
            try:
                # ✅ PREVENT DOUBLE REPLIES
                if not await self.should_reply(message.from_user.id, message.id):
                    return
                
                stats = await self.premium_system.get_statistics()
                
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
                    f"• Bot: {'✅ Online' if self.bot_started else '❌ Offline'}\n"
                    f"• User Client: {'✅ Connected' if self.user_session_ready else '❌ Disconnected'}\n"
                    f"• Verification: {'✅ Active'}\n"
                    f"• Premium: {'✅ Active'}\n\n"
                    f"⏰ **Auto-delete time:** 5 minutes for all users\n"
                    f"🕐 **Server Time:** {stats.get('server_time', 'Unknown')}"
                )
                
                await message.reply_text(stats_text, disable_web_page_preview=True)
                    
            except Exception as e:
                logger.error(f"Stats command error: {e}")
                await message.reply_text(f"❌ Error getting stats: {str(e)[:100]}")
        
        @self.bot.on_message(filters.command("pending") & filters.user(admin_ids))
        async def pending_payments_command(client, message):
            """Show pending payments"""
            try:
                # ✅ PREVENT DOUBLE REPLIES
                if not await self.should_reply(message.from_user.id, message.id):
                    return
                
                pending = [p for p in self.premium_system.pending_payments.values() if p['status'] == 'pending']
                
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
                        f"   **Screenshot:** {'✅ Sent' if payment.get('screenshot_sent', False) else '❌ Not sent'}\n"
                        f"   **Time Left:** {max(0, int((payment['expires_at'] - time.time()) / 3600))} hours\n\n"
                    )
                
                if len(pending) > 10:
                    text += f"... and {len(pending) - 10} more pending payments\n\n"
                
                text += "Use `/approve <payment_id>` to approve payment."
                
                await message.reply_text(text, disable_web_page_preview=True)
                    
            except Exception as e:
                logger.error(f"Pending payments command error: {e}")
                await message.reply_text(f"❌ Error: {str(e)[:100]}")
        
        @self.bot.on_message(filters.command("approve") & filters.user(admin_ids))
        async def approve_payment_command(client, message):
            """Approve pending payment"""
            try:
                # ✅ PREVENT DOUBLE REPLIES
                if not await self.should_reply(message.from_user.id, message.id):
                    return
                
                if len(message.command) < 2:
                    await message.reply_text(
                        "❌ **Usage:** `/approve <payment_id>`\n\n"
                        "**Example:** `/approve PAY_ABC123DEF456`"
                    )
                    return
                
                payment_id = message.command[1].strip()
                
                success, result = await self.premium_system.approve_payment(
                    admin_id=message.from_user.id,
                    payment_id=payment_id
                )
                
                if success:
                    await message.reply_text(f"✅ {result}")
                    
                    # Notify user
                    try:
                        # Find user from payment
                        if payment_id in self.premium_system.pending_payments:
                            payment = self.premium_system.pending_payments[payment_id]
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
                    except:
                        pass
                else:
                    await message.reply_text(f"❌ {result}")
                    
            except Exception as e:
                logger.error(f"Approve payment command error: {e}")
                await message.reply_text(f"❌ Error: {str(e)[:100]}")
        
        @self.bot.on_message(filters.command("reject") & filters.user(admin_ids))
        async def reject_payment_command(client, message):
            """Reject pending payment"""
            try:
                # ✅ PREVENT DOUBLE REPLIES
                if not await self.should_reply(message.from_user.id, message.id):
                    return
                
                if len(message.command) < 3:
                    await message.reply_text(
                        "❌ **Usage:** `/reject <payment_id> <reason>`\n\n"
                        "**Example:** `/reject PAY_ABC123DEF456 Invalid screenshot`"
                    )
                    return
                
                payment_id = message.command[1].strip()
                reason = ' '.join(message.command[2:])
                
                if payment_id in self.premium_system.pending_payments:
                    self.premium_system.pending_payments[payment_id]['status'] = 'rejected'
                    self.premium_system.pending_payments[payment_id]['reject_reason'] = reason
                    self.premium_system.pending_payments[payment_id]['rejected_at'] = datetime.now().isoformat()
                    
                    await message.reply_text(f"✅ Payment {payment_id} rejected!\n**Reason:** {reason}")
                else:
                    await message.reply_text(f"❌ Failed to reject payment {payment_id}")
                    
            except Exception as e:
                logger.error(f"Reject payment command error: {e}")
                await message.reply_text(f"❌ Error: {str(e)[:100]}")
        
        # ✅ FIXED: MAIN TEXT MESSAGE HANDLER
        @self.bot.on_message(filters.private & filters.text)
        async def handle_private_text(client, message):
            """Handle all private text messages"""
            user_id = message.from_user.id
            
            # ✅ PREVENT DOUBLE REPLIES
            if not await self.should_reply(user_id, message.id):
                return
            
            # Check if it's a command - commands are handled by separate handlers
            if message.text.startswith('/'):
                # Commands are handled by separate handlers, so just return
                return
            
            # Check if it's a file request pattern
            if re.match(r'^-?\d+_\d+(_\w+)?$', message.text.strip()):
                # It's a file request
                await self.handle_file_request(client, message, message.text.strip())
                return
            
            # Check if it's a verification token
            if message.text.strip().startswith('verify_'):
                token = message.text.strip().replace('verify_', '', 1)
                await self.handle_verification_token(client, message, token)
                return
            
            # It's a regular message
            await self.handle_direct_message(client, message)
        
        # ✅ CALLBACK HANDLERS
        
        @self.bot.on_callback_query(filters.regex(r"^get_verified$"))
        async def get_verified_callback(client, callback_query):
            """Get verification link"""
            user_id = callback_query.from_user.id
            user_name = callback_query.from_user.first_name or "User"
            
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
        
        @self.bot.on_callback_query(filters.regex(r"^back_to_start$"))
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
        
        @self.bot.on_callback_query(filters.regex(r"^buy_premium$"))
        async def buy_premium_callback(client, callback_query):
            """Show premium plans"""
            user_id = callback_query.from_user.id
            user_name = callback_query.from_user.first_name or "User"
            
            # Check if already premium
            is_premium = await self.premium_system.is_premium_user(user_id)
            if is_premium:
                details = await self.premium_system.get_subscription_details(user_id)
                
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
        
        @self.bot.on_callback_query(filters.regex(r"^plan_"))
        async def plan_selection_callback(client, callback_query):
            plan_type = callback_query.data.split('_')[1]
            user_id = callback_query.from_user.id
            
            if plan_type == "basic":
                tier = self.PremiumTier.BASIC
                plan_name = "Basic Plan"
            elif plan_type == "premium":
                tier = self.PremiumTier.PREMIUM
                plan_name = "Premium Plan"
            elif plan_type == "gold":
                tier = self.PremiumTier.GOLD
                plan_name = "Gold Plan"
            elif plan_type == "diamond":
                tier = self.PremiumTier.DIAMOND
                plan_name = "Diamond Plan"
            else:
                await callback_query.answer("Invalid plan!", show_alert=True)
                return
            
            # Initiate purchase
            payment_data = await self.premium_system.initiate_purchase(user_id, tier)
            
            if not payment_data:
                await callback_query.answer("Failed to initiate purchase!", show_alert=True)
                return
            
            # Get payment instructions
            instructions = await self.premium_system.get_payment_instructions_text(payment_data['payment_id'])
            
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("📸 SEND SCREENSHOT", callback_data=f"send_screenshot_{payment_data['payment_id']}")],
                [InlineKeyboardButton("🔙 BACK", callback_data="buy_premium")]
            ])
            
            try:
                await callback_query.message.edit_text(instructions, reply_markup=keyboard, disable_web_page_preview=True)
                await callback_query.answer()
            except:
                await callback_query.answer("Payment instructions!", show_alert=True)
        
        @self.bot.on_callback_query(filters.regex(r"^send_screenshot_"))
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
        
        # ✅ HANDLE SCREENSHOT MESSAGES
        @self.bot.on_message(filters.private & (filters.photo | filters.document))
        async def handle_screenshot(client, message):
            """Handle payment screenshots"""
            user_id = message.from_user.id
            
            # ✅ PREVENT DOUBLE REPLIES
            if not await self.should_reply(user_id, message.id):
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
            
            # Find pending payment for this user
            payment_id = None
            for pid, payment in self.premium_system.pending_payments.items():
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
            
            # Process screenshot
            success = await self.premium_system.process_payment_screenshot(
                user_id, 
                message.id
            )
            
            if success:
                # ✅ NOTIFY ADMINS
                admin_notified = await self.notify_admin_screenshot(user_id, message.id, payment_id)
                
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
        
        logger.info("✅ Bot handlers setup complete with FIXED features")
    
    async def send_file_to_user(self, client, user_id, file_message, quality="480p"):
        """Send file to user with verification check"""
        try:
            # ✅ FIRST CHECK: Verify user is premium/verified/admin
            user_status = "Checking..."
            status_icon = "⏳"
            can_download = False
            
            # Check if user is admin
            admin_ids = getattr(self.config, 'ADMIN_IDS', [])
            is_admin = user_id in admin_ids
            
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
                
                # ✅ Schedule auto-delete - 5 MINUTES
                if auto_delete_minutes > 0:
                    task_id = f"{user_id}_{sent.id}"
                    
                    # Cancel any existing task for this user
                    if task_id in self.auto_delete_tasks:
                        self.auto_delete_tasks[task_id].cancel()
                    
                    # Create new auto-delete task for 5 minutes
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
                        
                        # ✅ Schedule auto-delete for refreshed file - 5 MINUTES
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
            traceback.print_exc()
            return False, {
                'message': f"❌ Error sending file: {str(e)[:100]}",
                'buttons': []
            }, 0
    
    async def handle_verification_token(self, client, message, token):
        """Handle verification token from /start verify_<token>"""
        try:
            user_id = message.from_user.id
            user_name = message.from_user.first_name or "User"
            
            # ✅ PREVENT DOUBLE PROCESSING
            if not await self.should_reply(user_id, message.id):
                logger.warning(f"⚠️ Duplicate verification processing ignored for user {user_id}")
                return
            
            # ✅ CHECK DUPLICATE MESSAGE CONTENT
            if await self.is_message_content_duplicate(user_id, message.text):
                logger.debug(f"Duplicate verification message from user {user_id}")
                return
            
            # ✅ VERIFICATION RATE LIMIT CHECK
            if not await self.check_rate_limit(user_id, limit=5, window=60, request_type="verification"):
                await message.reply_text(
                    "⚠️ **Verification Rate Limit**\n\n"
                    "Too many verification attempts. Please wait 60 seconds."
                )
                return
            
            # ✅ DUPLICATE VERIFICATION CHECK
            if await self.is_request_duplicate(user_id, token, request_type="verification"):
                logger.debug(f"Duplicate verification request from user {user_id}")
                await message.reply_text(
                    "⏳ **Already Processing Verification**\n\n"
                    "Your verification is already being processed. Please wait..."
                )
                return
            
            logger.info(f"🔐 Processing verification token for user {user_id}: {token[:16]}...")
            
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
        """Handle file download request with user verification"""
        try:
            user_id = message.from_user.id
            
            # ✅ PREVENT DOUBLE PROCESSING
            if not await self.should_reply(user_id, message.id):
                logger.debug(f"⚠️ Duplicate file request processing ignored for user {user_id}")
                return
            
            # ✅ CHECK DUPLICATE MESSAGE CONTENT
            if await self.is_message_content_duplicate(user_id, message.text):
                logger.debug(f"Duplicate file request message from user {user_id}")
                return
            
            # ✅ FILE RATE LIMIT CHECK
            if not await self.check_rate_limit(user_id, limit=3, window=60, request_type="file"):
                await message.reply_text(
                    "⚠️ **Download Rate Limit Exceeded**\n\n"
                    "You're making too many download requests. Please wait 60 seconds and try again."
                )
                return
            
            # ✅ DUPLICATE FILE REQUEST CHECK
            if await self.is_request_duplicate(user_id, file_text, request_type="file"):
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
            
            # ✅ Send file to user
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
            traceback.print_exc()
            try:
                await message.reply_text(
                    "❌ **Download Error**\n\n"
                    "An error occurred during download. Please try again."
                )
            except:
                pass
            await self.clear_processing_request(user_id, file_text, request_type="file")
    
    async def handle_direct_message(self, client, message):
        """Handle direct messages from users (not commands)"""
        try:
            user_id = message.from_user.id
            user_name = message.from_user.first_name or "User"
            
            # ✅ PREVENT DOUBLE REPLIES
            if not await self.should_reply(user_id, message.id):
                return
            
            # ✅ CHECK DUPLICATE MESSAGE CONTENT
            if await self.is_message_content_duplicate(user_id, message.text):
                logger.debug(f"Duplicate direct message from user {user_id}")
                return
            
            # Check if message contains file request pattern
            if re.match(r'^-?\d+_\d+(_\w+)?$', message.text.strip()):
                # It's a file request, handle it
                await self.handle_file_request(client, message, message.text.strip())
                return
            
            # Check if it's a verification token
            if message.text.strip().startswith('verify_'):
                token = message.text.strip().replace('verify_', '', 1)
                await self.handle_verification_token(client, message, token)
                return
            
            # ✅ RATE LIMIT FOR DIRECT MESSAGES
            if not await self.check_rate_limit(user_id, limit=5, window=60, request_type="message"):
                await message.reply_text(
                    "⚠️ **Message Rate Limit**\n\n"
                    "Too many messages. Please wait 60 seconds."
                )
                return
            
            # It's a regular message, respond with website visit message
            website_url = getattr(self.config, 'WEBSITE_URL', 'https://sk4film.com')
            
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
                [InlineKeyboardButton("📢 JOIN CHANNEL", url=getattr(self.config, 'MAIN_CHANNEL_LINK', 'https://t.me/SK4FiLM'))]
            ])
            
            # Send response
            sent_msg = await message.reply_text(response_text, reply_markup=keyboard, disable_web_page_preview=True)
            
            # ✅ Schedule auto-delete for this response message too (5 minutes)
            task_id = f"{user_id}_{sent_msg.id}"
            
            # Cancel any existing task for this user
            if task_id in self.auto_delete_tasks:
                self.auto_delete_tasks[task_id].cancel()
            
            # Create new auto-delete task for 5 minutes
            delete_task = asyncio.create_task(
                self.schedule_file_deletion(user_id, sent_msg.id, "Website Visit Message", 5)
            )
            self.auto_delete_tasks[task_id] = delete_task
            
            logger.info(f"⏰ Auto-delete scheduled for response message {sent_msg.id}")
            logger.info(f"✅ Direct message handled for user {user_id}")
            
        except Exception as e:
            logger.error(f"Direct message handling error: {e}")
            traceback.print_exc()
