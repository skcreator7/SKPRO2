"""
bot_handlers.py - Telegram Bot Handlers for SK4FiLM
UPDATED: Razorpay integration, referral system, enhanced features
"""
import asyncio
import logging
import secrets
import re
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from collections import defaultdict

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
    class InlineKeyboardMarkup:
        def __init__(self, buttons): pass
    class InlineKeyboardButton:
        def __init__(self, text, url=None, callback_data=None): pass
    class Message: pass
    class CallbackQuery: pass
    PYROGRAM_AVAILABLE = False

from utils import normalize_title, extract_title_smart, format_size, detect_quality, is_video_file, format_post

logger = logging.getLogger(__name__)

class SK4FiLMBot:
    def __init__(self, config, db_manager=None):
        self.config = config
        self.db_manager = db_manager
        self.bot = None
        self.user_client = None
        self.bot_started = False
        self.user_session_ready = False
        self.auto_delete_tasks = {}
        self.auto_delete_messages = {}
        self.user_request_times = defaultdict(list)
        self.processing_requests = {}
        self.pending_payments = {}
        self.user_download_history = defaultdict(list)
        
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
            
            asyncio.create_task(self.cleanup_old_auto_delete_tasks())
            asyncio.create_task(self.periodic_cleanup())
            
            return True
            
        except Exception as e:
            logger.error(f"Bot initialization failed: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown bot"""
        try:
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
    
    async def add_auto_delete_task(self, user_id: int, message_id: int, file_name: str, 
                                   delete_after_minutes: int = 10):
        """Add auto-delete task for a file"""
        try:
            task_id = f"{user_id}_{message_id}"
            
            if task_id in self.auto_delete_tasks:
                old_task = self.auto_delete_tasks[task_id]
                if not old_task.done():
                    old_task.cancel()
                    try:
                        await old_task
                    except asyncio.CancelledError:
                        pass
            
            self.auto_delete_messages[task_id] = {
                'user_id': user_id,
                'message_id': message_id,
                'file_name': file_name,
                'scheduled_time': datetime.now() + timedelta(minutes=delete_after_minutes),
                'status': 'pending',
                'delete_after_minutes': delete_after_minutes
            }
            
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
        """Auto-delete file after specified minutes"""
        try:
            logger.info(f"⏰ Auto-delete started for user {user_id}, message {message_id}")
            await asyncio.sleep(delete_after_minutes * 60)
            
            task_id = f"{user_id}_{message_id}"
            delete_success = False
            
            try:
                if self.bot and self.bot_started:
                    await self.bot.delete_messages(user_id, message_id)
                    delete_success = True
                    logger.info(f"🗑️ File message deleted: user {user_id}, message {message_id}")
            except MessageDeleteForbidden:
                logger.warning(f"❌ Cannot delete message {message_id}: forbidden")
            except BadRequest as e:
                if "MESSAGE_TOO_OLD" in str(e):
                    logger.warning(f"❌ Cannot delete message {message_id}: too old")
                elif "MESSAGE_ID_INVALID" in str(e):
                    logger.warning(f"❌ Cannot delete message {message_id}: invalid")
                else:
                    logger.error(f"❌ Error deleting message {message_id}: {e}")
            except Exception as e:
                logger.error(f"❌ Error deleting message: {e}")
            
            try:
                notification_text = (
                    f"🗑️ **File Auto-Deleted**\n\n"
                    f"✅ **Security measure completed**\n\n"
                    f"> Visit website and download again\n"
                    f"🎬 @SK4FiLM"
                )
                
                buttons = InlineKeyboardMarkup([
                    [InlineKeyboardButton("🌐 VISIT WEBSITE", url=self.config.WEBSITE_URL)],
                    [InlineKeyboardButton("🔄 GET ANOTHER FILE", callback_data="back_to_start")]
                ])
                
                if self.bot and self.bot_started:
                    await self.bot.send_message(user_id, notification_text, reply_markup=buttons)
                    logger.info(f"✅ Auto-delete notification sent to user {user_id}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to send delete notification: {e}")
            
            if task_id in self.auto_delete_tasks:
                del self.auto_delete_tasks[task_id]
            
            if task_id in self.auto_delete_messages:
                self.auto_delete_messages[task_id]['status'] = 'completed' if delete_success else 'failed'
                self.auto_delete_messages[task_id]['completed_at'] = datetime.now()
            
            logger.info(f"✅ Auto-delete process completed for task {task_id}")
            
        except asyncio.CancelledError:
            logger.info(f"⏹️ Auto-delete task cancelled for user {user_id}, message {message_id}")
            task_id = f"{user_id}_{message_id}"
            if task_id in self.auto_delete_tasks:
                del self.auto_delete_tasks[task_id]
            if task_id in self.auto_delete_messages:
                self.auto_delete_messages[task_id]['status'] = 'cancelled'
                
        except Exception as e:
            logger.error(f"❌ Error in auto-delete task: {e}")
            task_id = f"{user_id}_{message_id}"
            if task_id in self.auto_delete_tasks:
                del self.auto_delete_tasks[task_id]
            if task_id in self.auto_delete_messages:
                self.auto_delete_messages[task_id]['status'] = 'error'
                self.auto_delete_messages[task_id]['error'] = str(e)
    
    async def check_rate_limit(self, user_id, limit=3, window=60):
        """Check if user is within rate limits"""
        now = time.time()
        if user_id in self.user_request_times:
            self.user_request_times[user_id] = [
                t for t in self.user_request_times[user_id] 
                if now - t < window
            ]
        
        if len(self.user_request_times.get(user_id, [])) >= limit:
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
    
    async def add_download_history(self, user_id: int, file_name: str, file_size: int, quality: str):
        """Add file download to user history"""
        try:
            download_record = {
                'file_name': file_name,
                'file_size': file_size,
                'quality': quality,
                'timestamp': time.time(),
                'date': datetime.now().isoformat()
            }
            
            self.user_download_history[user_id].append(download_record)
            if len(self.user_download_history[user_id]) > 50:
                self.user_download_history[user_id] = self.user_download_history[user_id][-50:]
            
            return True
        except Exception as e:
            logger.error(f"Error adding download history: {e}")
            return False
    
    async def get_user_download_history(self, user_id: int, limit: int = 10):
        """Get user download history"""
        try:
            history = self.user_download_history.get(user_id, [])
            history.sort(key=lambda x: x['timestamp'], reverse=True)
            return history[:limit]
        except Exception as e:
            logger.error(f"Error getting download history: {e}")
            return []
    
    async def cleanup_old_auto_delete_tasks(self):
        """Clean up old auto-delete task data"""
        while True:
            try:
                await asyncio.sleep(3600)
                now = datetime.now()
                to_remove = []
                
                for task_id, task_data in self.auto_delete_messages.items():
                    completed_at = task_data.get('completed_at')
                    cancelled_at = task_data.get('cancelled_at')
                    
                    if completed_at and (now - completed_at).total_seconds() > 24 * 3600:
                        to_remove.append(task_id)
                    elif cancelled_at and (now - cancelled_at).total_seconds() > 24 * 3600:
                        to_remove.append(task_id)
                
                for task_id in to_remove:
                    self.auto_delete_messages.pop(task_id, None)
                
                if to_remove:
                    logger.info(f"🧹 Cleaned up {len(to_remove)} old auto-delete tasks")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in auto-delete cleanup: {e}")
    
    async def periodic_cleanup(self):
        """Periodic cleanup"""
        while True:
            try:
                await asyncio.sleep(300)
                now = time.time()
                
                for user_id in list(self.user_request_times.keys()):
                    self.user_request_times[user_id] = [
                        t for t in self.user_request_times[user_id]
                        if now - t < 300
                    ]
                    if not self.user_request_times[user_id]:
                        self.user_request_times.pop(user_id, None)
                        
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")


async def setup_bot_handlers(bot: Client, bot_instance):
    """Setup bot commands and handlers"""
    config = bot_instance.config
    
    # ============================================================================
    # ✅ START COMMAND
    # ============================================================================
    @bot.on_message(filters.command("start"))
    async def handle_start_command(client, message):
        user_name = message.from_user.first_name or "User"
        user_id = message.from_user.id
        
        # Check for file request
        if len(message.command) > 1:
            file_text = ' '.join(message.command[1:])
            
            # Check for referral code
            if file_text.startswith('ref_'):
                referral_code = file_text.replace('ref_', '')
                await handle_referral_start(client, message, referral_code, bot_instance)
                return
            
            await handle_file_request(client, message, file_text, bot_instance)
            return
        
        welcome_text = (
            f"🎬 **Welcome to SK4FiLM, {user_name}!**\n\n"
            f"🌐 **Website:** {config.WEBSITE_URL}\n\n"
            "**How to download movies:**\n"
            "1. Visit website above\n"
            "2. Search for any movie\n"
            "3. Click download button\n"
            "4. File will appear here automatically\n\n"
            "**Features:**\n"
            "✅ Multiple quality options\n"
            "✅ Auto-delete for security\n"
            "✅ Fast downloads\n"
            "✅ Premium support\n\n"
            "🎬 **Happy watching!**"
        )
        
        keyboard_buttons = [
            [InlineKeyboardButton("🌐 OPEN WEBSITE", url=config.WEBSITE_URL)],
            [InlineKeyboardButton("📢 JOIN CHANNEL", url=config.MAIN_CHANNEL_LINK)],
            [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")],
            [InlineKeyboardButton("🎁 REFER & GET PREMIUM", callback_data="referral_info")]
        ]
        
        keyboard = InlineKeyboardMarkup(keyboard_buttons)
        await message.reply_text(welcome_text, reply_markup=keyboard, disable_web_page_preview=True)
    
    # ============================================================================
    # ✅ REFERRAL HANDLER
    # ============================================================================
    async def handle_referral_start(client, message, referral_code, bot_instance):
        """Handle referral code from start command"""
        user_id = message.from_user.id
        user_name = message.from_user.first_name or "User"
        
        if bot_instance.premium_system:
            referral_data = await bot_instance.premium_system.validate_referral_code(referral_code, user_id)
            
            if referral_data['valid']:
                text = (
                    f"🎁 **Referral Code Applied!**\n\n"
                    f"👋 Welcome {user_name}!\n"
                    f"✅ You'll get **{referral_data['reward_days']} extra days** FREE when you purchase premium!\n\n"
                    f"💎 **Choose your plan:**\n"
                    f"• 🥉 Basic - ₹9/15 days\n"
                    f"• 🥈 Standard - ₹19/28 days\n"
                    f"• 🥇 Pro - ₹29/49 days\n"
                    f"• 💎 Ultimate - ₹49/90 days\n\n"
                    f"All plans include ALL premium features!"
                )
                
                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data=f"buy_with_ref_{referral_code}")],
                    [InlineKeyboardButton("🌐 VISIT WEBSITE", url=config.WEBSITE_URL)],
                    [InlineKeyboardButton("🔙 HOME", callback_data="back_to_start")]
                ])
                
                await message.reply_text(text, reply_markup=keyboard)
            else:
                await message.reply_text(
                    f"❌ {referral_data.get('error', 'Invalid referral code')}\n\n"
                    f"Please check the code and try again."
                )
        else:
            await message.reply_text("Premium system not available!")
    
    # ============================================================================
    # ✅ BUY PREMIUM CALLBACK
    # ============================================================================
    @bot.on_callback_query(filters.regex(r"^buy_premium$"))
    async def buy_premium_callback(client, callback_query):
        user_id = callback_query.from_user.id
        
        if bot_instance.premium_system:
            is_premium = await bot_instance.premium_system.is_premium_user(user_id)
            if is_premium:
                details = await bot_instance.premium_system.get_subscription_details(user_id)
                text = (
                    f"⭐ **You're Already Premium!** ⭐\n\n"
                    f"**Plan:** {details.get('tier_name', 'Premium')}\n"
                    f"**Days Left:** {details.get('days_remaining', 0)}\n"
                    f"**Expires:** {details.get('expires_at', 'Unknown')}\n\n"
                    f"**Premium Features:**\n"
                    f"✅ All Quality (480p-4K)\n"
                    f"✅ Unlimited Downloads\n"
                    f"✅ No Verification Needed\n"
                    f"✅ VIP Support 24/7\n"
                    f"✅ No Ads\n"
                    f"✅ Custom Requests\n\n"
                    f"🎬 **Enjoy unlimited premium downloads!**"
                )
                
                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton("🎁 REFER & GET PREMIUM", callback_data="referral_info")],
                    [InlineKeyboardButton("🔙 BACK", callback_data="back_to_start")]
                ])
                
                try:
                    await callback_query.message.edit_text(text, reply_markup=keyboard)
                except:
                    await callback_query.answer("You're already premium!")
                return
            
            # Show plans
            text = "💎 **SK4FiLM PREMIUM PLANS** 💎\n\n"
            text += "🎯 **ALL PLANS INCLUDE:**\n"
            text += "✅ All Quality (480p-4K)\n"
            text += "✅ Unlimited Downloads\n"
            text += "✅ No Verification Needed\n"
            text += "✅ VIP Support 24/7\n"
            text += "✅ No Ads\n"
            text += "✅ Custom Requests\n\n"
            text += "📊 **Choose Your Plan:**\n\n"
            
            keyboard_buttons = []
            
            plans = [
                ("basic", "🥉 Basic", "₹9", "15 days"),
                ("standard", "🥈 Standard", "₹19", "28 days"),
                ("pro", "🥇 Pro", "₹29", "49 days"),
                ("ultimate", "💎 Ultimate", "₹49", "90 days")
            ]
            
            for tier_id, icon_name, price, duration in plans:
                keyboard_buttons.append([
                    InlineKeyboardButton(
                        f"{icon_name} - {price}/{duration}",
                        callback_data=f"buy_{tier_id}"
                    )
                ])
            
            keyboard_buttons.append([
                InlineKeyboardButton("🎁 REFER & GET PREMIUM", callback_data="referral_info")
            ])
            keyboard_buttons.append([
                InlineKeyboardButton("🔙 BACK", callback_data="back_to_start")
            ])
            
            keyboard = InlineKeyboardMarkup(keyboard_buttons)
            
            try:
                await callback_query.message.edit_text(text, reply_markup=keyboard)
            except:
                await callback_query.answer("Premium plans!")
        else:
            await callback_query.answer("Premium system not available!", show_alert=True)
    
    # ============================================================================
    # ✅ BUY WITH REFERRAL
    # ============================================================================
    @bot.on_callback_query(filters.regex(r"^buy_with_ref_"))
    async def buy_with_referral_callback(client, callback_query):
        referral_code = callback_query.data.replace("buy_with_ref_", "")
        user_id = callback_query.from_user.id
        
        text = "💎 **Choose Your Plan** 💎\n\n"
        text += f"🎁 **Referral Code Applied:** `{referral_code}`\n"
        text += f"✅ You'll get 3 extra days FREE!\n\n"
        text += "📊 **Select Plan:**\n\n"
        
        keyboard_buttons = []
        plans = [
            ("basic", "🥉 Basic", "₹9", "15 days"),
            ("standard", "🥈 Standard", "₹19", "28 days"),
            ("pro", "🥇 Pro", "₹29", "49 days"),
            ("ultimate", "💎 Ultimate", "₹49", "90 days")
        ]
        
        for tier_id, icon_name, price, duration in plans:
            keyboard_buttons.append([
                InlineKeyboardButton(
                    f"{icon_name} - {price}/{duration}",
                    callback_data=f"pay_{tier_id}_{referral_code}"
                )
            ])
        
        keyboard_buttons.append([
            InlineKeyboardButton("🔙 BACK", callback_data="buy_premium")
        ])
        
        keyboard = InlineKeyboardMarkup(keyboard_buttons)
        await callback_query.message.edit_text(text, reply_markup=keyboard)
    
    # ============================================================================
    # ✅ BUY PLAN - RAZORPAY PAYMENT
    # ============================================================================
    @bot.on_callback_query(filters.regex(r"^buy_(basic|standard|pro|ultimate)$"))
    async def buy_plan_callback(client, callback_query):
        tier_str = callback_query.data.split("_")[1]
        user_id = callback_query.from_user.id
        
        # Map tier
        from premium import PremiumTier
        tier_map = {
            'basic': PremiumTier.BASIC,
            'standard': PremiumTier.STANDARD,
            'pro': PremiumTier.PRO,
            'ultimate': PremiumTier.ULTIMATE
        }
        
        tier = tier_map.get(tier_str)
        if not tier:
            await callback_query.answer("Invalid plan!", show_alert=True)
            return
        
        # Create Razorpay order
        order_data = await bot_instance.premium_system.create_razorpay_order(user_id, tier)
        
        if order_data.get('success'):
            plan = bot_instance.premium_system.plans[tier]
            
            text = (
                f"💳 **Payment Required**\n\n"
                f"{plan['icon']} **Plan:** {plan['name']}\n"
                f"💰 **Amount:** ₹{plan['price']}\n"
                f"📅 **Duration:** {plan['duration_days']} days\n\n"
                f"🔒 **Secure payment via Razorpay**\n\n"
                f"Click below to pay:"
            )
            
            # Create Razorpay payment link
            payment_url = f"https://rzp.io/l/{order_data['order_id']}"
            
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("💳 PAY NOW", url=payment_url)],
                [InlineKeyboardButton("✅ I'VE PAID", callback_data=f"verify_payment_{order_data['order_id']}")],
                [InlineKeyboardButton("❌ CANCEL", callback_data="buy_premium")]
            ])
            
            await callback_query.message.edit_text(text, reply_markup=keyboard)
        else:
            await callback_query.answer("Payment gateway error!", show_alert=True)
    
    # ============================================================================
    # ✅ PAY WITH REFERRAL
    # ============================================================================
    @bot.on_callback_query(filters.regex(r"^pay_(basic|standard|pro|ultimate)_"))
    async def pay_with_referral_callback(client, callback_query):
        parts = callback_query.data.split("_")
        tier_str = parts[1]
        referral_code = parts[2] if len(parts) > 2 else ""
        user_id = callback_query.from_user.id
        
        from premium import PremiumTier
        tier_map = {
            'basic': PremiumTier.BASIC,
            'standard': PremiumTier.STANDARD,
            'pro': PremiumTier.PRO,
            'ultimate': PremiumTier.ULTIMATE
        }
        
        tier = tier_map.get(tier_str)
        if not tier:
            await callback_query.answer("Invalid plan!", show_alert=True)
            return
        
        # Create Razorpay order with referral
        order_data = await bot_instance.premium_system.create_razorpay_order(
            user_id, tier, referral_code
        )
        
        if order_data.get('success'):
            plan = bot_instance.premium_system.plans[tier]
            
            text = (
                f"💳 **Payment Required**\n\n"
                f"{plan['icon']} **Plan:** {plan['name']}\n"
                f"💰 **Amount:** ₹{plan['price']}\n"
                f"📅 **Duration:** {plan['duration_days']} days\n"
                f"🎁 **Referral Bonus:** +3 days FREE\n"
                f"📅 **Total:** {plan['duration_days'] + 3} days\n\n"
                f"🔒 **Secure payment via Razorpay**\n\n"
                f"Click below to pay:"
            )
            
            payment_url = f"https://rzp.io/l/{order_data['order_id']}"
            
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("💳 PAY NOW", url=payment_url)],
                [InlineKeyboardButton("✅ I'VE PAID", callback_data=f"verify_payment_{order_data['order_id']}")],
                [InlineKeyboardButton("❌ CANCEL", callback_data="buy_premium")]
            ])
            
            await callback_query.message.edit_text(text, reply_markup=keyboard)
        else:
            await callback_query.answer("Payment gateway error!", show_alert=True)
    
    # ============================================================================
    # ✅ VERIFY PAYMENT
    # ============================================================================
    @bot.on_callback_query(filters.regex(r"^verify_payment_"))
    async def verify_payment_callback(client, callback_query):
        order_id = callback_query.data.replace("verify_payment_", "")
        user_id = callback_query.from_user.id
        
        text = (
            "🔄 **Verifying Payment...**\n\n"
            f"Order ID: `{order_id}`\n\n"
            "Please wait while we verify your payment..."
        )
        
        await callback_query.message.edit_text(text)
        await callback_query.answer("Checking payment status...")
        
        # Check order status
        if order_id in bot_instance.premium_system.razorpay_orders:
            order_data = bot_instance.premium_system.razorpay_orders[order_id]
            
            if order_data.get('status') == 'paid':
                text = (
                    "✅ **Payment Verified!**\n\n"
                    f"Your premium has been activated!\n"
                    f"Thank you for your purchase! 🎬"
                )
                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 BACK TO HOME", callback_data="back_to_start")]
                ])
                await callback_query.message.edit_text(text, reply_markup=keyboard)
            else:
                text = (
                    "⏳ **Payment Pending**\n\n"
                    f"Order ID: `{order_id}`\n\n"
                    "If you've completed payment, please wait a few moments and try again.\n"
                    "Contact support if issue persists."
                )
                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔄 CHECK AGAIN", callback_data=f"verify_payment_{order_id}")],
                    [InlineKeyboardButton("❌ CANCEL", callback_data="buy_premium")]
                ])
                await callback_query.message.edit_text(text, reply_markup=keyboard)
        else:
            await callback_query.answer("Order not found!", show_alert=True)
    
    # ============================================================================
    # ✅ REFERRAL INFO
    # ============================================================================
    @bot.on_callback_query(filters.regex(r"^referral_info$"))
    async def referral_info_callback(client, callback_query):
        user_id = callback_query.from_user.id
        
        if bot_instance.premium_system:
            referral_info = await bot_instance.premium_system.get_referral_info(user_id)
            
            text = (
                f"🎁 **REFER & GET PREMIUM** 🎁\n\n"
                f"👥 **Your Referral Code:**\n"
                f"`{referral_info['referral_code']}`\n\n"
                f"📊 **Your Stats:**\n"
                f"✅ Total Referrals: {referral_info['total_referrals']}\n"
                f"🎁 Reward Days Earned: {referral_info['total_rewards_days']}\n\n"
                f"💎 **How it Works:**\n"
                f"1️⃣ Share your referral code\n"
                f"2️⃣ Friend buys premium using code\n"
                f"3️⃣ You get {referral_info['reward_per_referral']} days FREE\n"
                f"4️⃣ Friend gets {referral_info['referred_reward_days']} extra days\n\n"
                f"🔗 **Referral Link:**\n"
                f"`{referral_info['referral_link']}`"
            )
            
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("📤 SHARE REFERRAL LINK", url=f"https://t.me/share/url?url={referral_info['referral_link']}&text=Join%20SK4FiLM%20Premium!")],
                [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")],
                [InlineKeyboardButton("🔙 BACK", callback_data="back_to_start")]
            ])
            
            await callback_query.message.edit_text(text, reply_markup=keyboard, disable_web_page_preview=True)
        else:
            await callback_query.answer("Premium system not available!", show_alert=True)
    
    # ============================================================================
    # ✅ BACK TO START
    # ============================================================================
    @bot.on_callback_query(filters.regex(r"^back_to_start$"))
    async def back_to_start_callback(client, callback_query):
        user_name = callback_query.from_user.first_name or "User"
        user_id = callback_query.from_user.id
        
        text = (
            f"🎬 **Welcome back, {user_name}!**\n\n"
            f"Visit {config.WEBSITE_URL} to download movies.\n"
            "Click download button on website and file will appear here."
        )
        
        keyboard_buttons = [
            [InlineKeyboardButton("🌐 OPEN WEBSITE", url=config.WEBSITE_URL)],
            [InlineKeyboardButton("📢 JOIN CHANNEL", url=config.MAIN_CHANNEL_LINK)],
            [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")],
            [InlineKeyboardButton("🎁 REFER & GET PREMIUM", callback_data="referral_info")]
        ]
        
        # Add history button if available
        try:
            history = await bot_instance.get_user_download_history(user_id, limit=1)
            if history:
                keyboard_buttons.append([InlineKeyboardButton("📜 DOWNLOAD HISTORY", callback_data="download_history")])
        except:
            pass
        
        keyboard = InlineKeyboardMarkup(keyboard_buttons)
        
        try:
            await callback_query.message.edit_text(
                text=text,
                reply_markup=keyboard,
                disable_web_page_preview=True
            )
        except:
            await callback_query.answer("Already on home page!")
    
    # ============================================================================
    # ✅ DOWNLOAD HISTORY
    # ============================================================================
    @bot.on_callback_query(filters.regex(r"^download_history$"))
    async def download_history_callback(client, callback_query):
        user_id = callback_query.from_user.id
        
        history = await bot_instance.get_user_download_history(user_id, limit=10)
        
        if not history:
            await callback_query.answer("📭 No download history found!", show_alert=True)
            return
        
        history_text = "📜 **Your Recent Downloads**\n\n"
        
        for i, record in enumerate(history[:10], 1):
            file_name = record.get('file_name', 'Unknown')
            file_size = record.get('file_size', 0)
            quality = record.get('quality', 'Unknown')
            date = record.get('date', '')
            
            if date:
                try:
                    date_obj = datetime.fromisoformat(date.replace('Z', '+00:00'))
                    date_str = date_obj.strftime("%b %d, %H:%M")
                except:
                    date_str = date
            else:
                date_str = "Recent"
            
            history_text += f"{i}. **{file_name[:40]}**\n"
            history_text += f"   📦 {format_size(file_size)} | 📹 {quality} | 🕒 {date_str}\n\n"
        
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔙 BACK TO HOME", callback_data="back_to_start")],
            [InlineKeyboardButton("🔄 CLEAR HISTORY", callback_data="clear_history")]
        ])
        
        await callback_query.message.edit_text(
            history_text,
            reply_markup=keyboard,
            disable_web_page_preview=True
        )
        await callback_query.answer("✅ Download history loaded!")
    
    # ============================================================================
    # ✅ CLEAR HISTORY
    # ============================================================================
    @bot.on_callback_query(filters.regex(r"^clear_history$"))
    async def clear_history_callback(client, callback_query):
        user_id = callback_query.from_user.id
        
        if user_id in bot_instance.user_download_history:
            bot_instance.user_download_history[user_id] = []
            await callback_query.answer("✅ History cleared!", show_alert=True)
            await back_to_start_callback(client, callback_query)
        else:
            await callback_query.answer("📭 No history to clear!", show_alert=True)
    
    logger.info("✅ Bot handlers setup complete with Razorpay integration")
