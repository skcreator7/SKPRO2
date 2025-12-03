"""
bot_handlers.py - Telegram Bot Handlers for SK4FiLM
UPDATED: No circular imports - uses utils.py
"""
import asyncio
import logging
import secrets
from datetime import datetime
from typing import Dict, Any, Optional

# ADD THIS IMPORT at the TOP
from pyrogram import Client, filters
from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton, Message, CallbackQuery
from pyrogram.errors import FloodWait

# Import from utils instead of app
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
            
            # Initialize cache
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
            
            # Initialize bot
            self.bot = Client(
                "bot",
                api_id=self.config.API_ID,
                api_hash=self.config.API_HASH,
                bot_token=self.config.BOT_TOKEN,
                workers=20
            )
            
            # Initialize user client if session string is provided
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
            
            # Start bot
            await self.bot.start()
            self.bot_started = True
            logger.info("✅ Bot started successfully")
            
            # Start cleanup tasks
            if self.verification_system:
                asyncio.create_task(self.verification_system.start_cleanup_task())
            if self.premium_system:
                asyncio.create_task(self.premium_system.start_cleanup_task())
            if self.cache_manager:
                asyncio.create_task(self.cache_manager.start_cleanup_task())
            
            return True
            
        except Exception as e:
            logger.error(f"Bot initialization failed: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown bot"""
        try:
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

async def setup_bot_handlers(bot: Client, bot_instance):
    """Setup bot commands and handlers"""
    config = bot_instance.config
    
    @bot.on_message(filters.command("start") & filters.private)
    async def start_handler(client, message):
        user_id = message.from_user.id
        user_name = message.from_user.first_name or "User"
        
        # Check if this is a verification token
        if len(message.command) > 1:
            command_arg = message.command[1]
            
            if command_arg.startswith('verify_'):
                token = command_arg[7:]
                
                # Verify token if verification system exists
                if bot_instance.verification_system:
                    success, verified_user_id, message_text = await bot_instance.verification_system.verify_user_token(token)
                    if success and verified_user_id == user_id:
                        await message.reply_text(
                            f"✅ **Verification Successful, {user_name}!**\n\n"
                            "You are now verified and can download files.\n\n"
                            f"🌐 **Website:** {config.WEBSITE_URL}\n"
                            f"⏰ **Verification valid for 6 hours**",
                            reply_markup=InlineKeyboardMarkup([
                                [InlineKeyboardButton("🌐 OPEN WEBSITE", url=config.WEBSITE_URL)],
                                [InlineKeyboardButton("📥 DOWNLOAD FILES", url=config.WEBSITE_URL)]
                            ])
                        )
                        return
                else:
                    # Fallback verification
                    await message.reply_text(
                        f"✅ **Verification Successful, {user_name}!**\n\n"
                        "You are now verified and can download files.\n\n"
                        f"🌐 **Website:** {config.WEBSITE_URL}\n"
                        f"⏰ **Verification valid for 6 hours**",
                        reply_markup=InlineKeyboardMarkup([
                            [InlineKeyboardButton("🌐 OPEN WEBSITE", url=config.WEBSITE_URL)]
                        ])
                    )
                    return
        
        # Regular start command with better instructions
        welcome_text = (
            f"🎬 **Welcome to SK4FiLM, {user_name}!**\n\n"
            "**How to download movies:**\n"
            f"1. **Visit:** {config.WEBSITE_URL}\n"
            "2. **Search for any movie**\n"
            "3. **Click download button**\n"
            "4. **File will appear here automatically**\n\n"
        )
        
        # Check premium status
        is_premium = False
        if bot_instance.premium_system is not None:
            try:
                is_premium = await bot_instance.premium_system.is_premium_user(user_id)
            except:
                pass
        
        if is_premium:
            welcome_text += "🌟 **Premium User**\n✅ **Instant access to all files!**\n\n"
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🌐 OPEN WEBSITE", url=config.WEBSITE_URL)],
                [InlineKeyboardButton("📥 START DOWNLOADING", url=config.WEBSITE_URL)],
                [InlineKeyboardButton("⭐ PREMIUM STATUS", callback_data=f"premium_status_{user_id}")]
            ])
        elif config.VERIFICATION_REQUIRED:
            # Create verification link
            if bot_instance.verification_system:
                try:
                    verification_data = await bot_instance.verification_system.create_verification_link(user_id)
                    verification_url = verification_data['short_url']
                    welcome_text += (
                        "🔒 **Verification Required**\n"
                        "Please complete verification to download files:\n\n"
                        f"🔗 **Verification Link:** {verification_url}\n\n"
                        "Click the link above to verify your account.\n"
                        "⏰ **Valid for 1 hour**\n\n"
                        "✨ **Or upgrade to Premium for instant access!**"
                    )
                    keyboard = InlineKeyboardMarkup([
                        [InlineKeyboardButton("🔗 CLICK TO VERIFY", url=verification_url)],
                        [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")],
                        [InlineKeyboardButton("🌐 OPEN WEBSITE", url=config.WEBSITE_URL)]
                    ])
                except Exception as e:
                    logger.error(f"Verification link creation error: {e}")
                    # Fallback
                    verification_url = f"https://t.me/{config.BOT_USERNAME}?start=verify_{secrets.token_urlsafe(16)}"
                    welcome_text += (
                        "🔒 **Verification Required**\n"
                        f"🔗 **Verification Link:** {verification_url}\n\n"
                        "Click the link above to verify your account.\n"
                        "⏰ **Valid for 1 hour**"
                    )
                    keyboard = InlineKeyboardMarkup([
                        [InlineKeyboardButton("🔗 CLICK TO VERIFY", url=verification_url)],
                        [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")]
                    ])
            else:
                # Fallback if verification system not initialized
                verification_url = f"https://t.me/{config.BOT_USERNAME}?start=verify_{secrets.token_urlsafe(16)}"
                welcome_text += (
                    "🔒 **Verification Required**\n"
                    f"🔗 **Verification Link:** {verification_url}\n\n"
                    "Click the link above to verify your account.\n"
                    "⏰ **Valid for 1 hour**"
                )
                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔗 CLICK TO VERIFY", url=verification_url)],
                    [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")]
                ])
        else:
            welcome_text += "✨ **Start downloading movies now!**\n\n"
            welcome_text += "⭐ **Upgrade to Premium for:**\n• Faster downloads\n• No verification\n• Higher priority"
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🌐 OPEN WEBSITE", url=config.WEBSITE_URL)],
                [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")]
            ])
        
        await message.reply_text(welcome_text, reply_markup=keyboard, disable_web_page_preview=True)
    
    @bot.on_callback_query(filters.regex(r"^check_verify_"))
    async def check_verify_callback(client, callback_query):
        user_id = int(callback_query.data.split('_')[2])
        user_name = callback_query.from_user.first_name or "User"
        
        # Create new verification link
        if bot_instance.verification_system:
            try:
                verification_data = await bot_instance.verification_system.create_verification_link(user_id)
                verification_url = verification_data['short_url']
                message_text = (
                    "❌ **Not Verified Yet**\n\n"
                    "Please complete the verification process:\n\n"
                    f"🔗 **Verification Link:** {verification_url}\n\n"
                    "Click the link above to verify your account."
                )
                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔗 VERIFY NOW", url=verification_url)],
                    [InlineKeyboardButton("🔄 CHECK AGAIN", callback_data=f"check_verify_{user_id}")]
                ])
            except Exception as e:
                logger.error(f"Verification link creation error: {e}")
                verification_url = f"https://t.me/{config.BOT_USERNAME}?start=verify_{secrets.token_urlsafe(16)}"
                message_text = (
                    "❌ **Not Verified Yet**\n\n"
                    "Please complete the verification process:\n\n"
                    f"🔗 **Verification Link:** {verification_url}\n\n"
                    "Click the link above and then click 'Start' in the bot."
                )
                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔗 VERIFY NOW", url=verification_url)],
                    [InlineKeyboardButton("🔄 CHECK AGAIN", callback_data=f"check_verify_{user_id}")]
                ])
        else:
            verification_url = f"https://t.me/{config.BOT_USERNAME}?start=verify_{secrets.token_urlsafe(16)}"
            message_text = (
                "❌ **Not Verified Yet**\n\n"
                "Please complete the verification process:\n\n"
                f"🔗 **Verification Link:** {verification_url}\n\n"
                "Click the link above and then click 'Start' in the bot."
            )
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🔗 VERIFY NOW", url=verification_url)],
                [InlineKeyboardButton("🔄 CHECK AGAIN", callback_data=f"check_verify_{user_id}")]
            ])
        
        await callback_query.message.edit_text(
            message_text,
            reply_markup=keyboard,
            disable_web_page_preview=True
        )
    
    @bot.on_callback_query(filters.regex(r"^buy_premium$"))
    async def buy_premium_callback(client, callback_query):
        """Show premium plans"""
        user_id = callback_query.from_user.id
        
        text = "⭐ **SK4FiLM PREMIUM PLANS** ⭐\n\n"
        text += "Upgrade for better quality, more downloads and faster speeds!\n\n"
        
        plans = [
            {"tier": "basic", "name": "Basic Plan", "price": 99, "duration_days": 30, 
             "features": ["1080p Quality", "10 Daily Downloads", "Priority Support"]},
            {"tier": "premium", "name": "Premium Plan", "price": 199, "duration_days": 30,
             "features": ["4K Quality", "Unlimited Downloads", "Priority Support", "No Ads"]}
        ]
        
        keyboard = []
        for plan in plans:
            text += f"**{plan['name']}**\n"
            text += f"💰 **Price:** ₹{plan['price']}\n"
            text += f"⏰ **Duration:** {plan['duration_days']} days\n"
            text += "**Features:**\n"
            for feature in plan['features'][:3]:
                text += f"• {feature}\n"
            text += "\n"
            
            keyboard.append([InlineKeyboardButton(
                f"{plan['name']} - ₹{plan['price']}", 
                callback_data=f"select_plan_{plan['tier']}"
            )])
        
        text += "**How to purchase:**\n1. Select a plan\n2. Pay using UPI\n3. Send screenshot\n4. Get activated!"
        
        keyboard.append([InlineKeyboardButton("🔙 BACK", callback_data="back_to_start")])
        
        await callback_query.message.edit_text(
            text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            disable_web_page_preview=True
        )
    
    @bot.on_callback_query(filters.regex(r"^select_plan_"))
    async def select_plan_callback(client, callback_query):
        """Select premium plan and show payment details"""
        tier_str = callback_query.data.split('_')[2]
        user_id = callback_query.from_user.id
        
        if tier_str == "basic":
            upi_id = config.UPI_ID_BASIC
            amount = 99
            tier_name = "Basic Plan"
        else:
            upi_id = config.UPI_ID_PREMIUM
            amount = 199
            tier_name = "Premium Plan"
        
        payment_id = secrets.token_hex(8)
        
        text = f"💰 **Payment for {tier_name}**\n\n"
        text += f"**Amount:** ₹{amount}\n"
        text += f"**UPI ID:** `{upi_id}`\n\n"
        text += "**Payment Instructions:**\n"
        text += f"1. Send ₹{amount} to UPI ID: `{upi_id}`\n"
        text += "2. Take screenshot of payment\n"
        text += "3. Send screenshot to this bot\n\n"
        text += "⏰ **Payment valid for 1 hour**\n"
        text += "✅ **Admin will activate within 24 hours**"
        
        keyboard = [
            [InlineKeyboardButton("📸 SEND SCREENSHOT", callback_data=f"send_screenshot_{payment_id}")],
            [InlineKeyboardButton("🔙 BACK TO PLANS", callback_data="buy_premium")]
        ]
        
        await callback_query.message.delete()
        await callback_query.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard))
    
    # ... (Add other handlers from your original bot_handlers.py)
    # Make sure to use functions from utils instead of app
    
    logger.info("✅ Bot handlers setup complete")
