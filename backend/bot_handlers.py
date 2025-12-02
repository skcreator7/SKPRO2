"""
bot_handlers.py - Telegram Bot Handlers for SK4FiLM
FIXED: No circular imports!
"""
import asyncio
import logging
import secrets
from datetime import datetime

from pyrogram import Client, filters
from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton, Message, CallbackQuery
from pyrogram.errors import FloodWait

# Import from config and utils ONLY (NOT from app.py!)
from config import Config
from utils import (
    normalize_title, extract_title_from_file, format_size,
    detect_quality, is_video_file, safe_telegram_operation,
    safe_telegram_generator, auto_delete_file, extract_title_smart
)

logger = logging.getLogger(__name__)


class SK4FiLMBot:
    """Main bot class"""
    
    def __init__(self, config, db_manager=None):
        self.config = config
        self.db_manager = db_manager
        self.bot = None
        self.user_client = None
        self.bot_started = False
        self.user_session_ready = False
        
        # System placeholders
        self.verification_system = None
        self.premium_system = None
        self.poster_fetcher = None
        self.cache_manager = None
        self.task_manager = None
    
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
            
            # Initialize user client if session string provided
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
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


async def setup_bot_handlers(bot: Client, bot_instance: SK4FiLMBot):
    """Setup bot commands and handlers"""
    
    @bot.on_message(filters.command("start") & filters.private)
    async def start_handler(client, message):
        user_id = message.from_user.id
        user_name = message.from_user.first_name or "User"
        
        # Check verification token
        if len(message.command) > 1:
            command_arg = message.command[1]
            
            if command_arg.startswith('verify_'):
                await message.reply_text(
                    f"✅ **Verification Successful, {user_name}!**\n\n"
                    "You are now verified and can download files.\n\n"
                    f"🌐 **Website:** {Config.WEBSITE_URL}\n"
                    f"⏰ **Valid for 6 hours**",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)]
                    ])
                )
                return
        
        # Regular start
        welcome_text = (
            f"🎬 **Welcome to SK4FiLM, {user_name}!**\n\n"
            "🌐 **Browse and download movies:**\n"
            f"{Config.WEBSITE_URL}\n\n"
        )
        
        # Check premium (placeholder)
        is_premium = False
        
        if is_premium:
            welcome_text += "🌟 **Premium User** - Full access!\n\n"
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🌐 WEBSITE", url=Config.WEBSITE_URL)],
                [InlineKeyboardButton("📊 STATUS", callback_data=f"premium_status_{user_id}")]
            ])
        elif Config.VERIFICATION_REQUIRED:
            verification_url = f"https://t.me/{Config.BOT_USERNAME}?start=verify_{secrets.token_urlsafe(16)}"
            welcome_text += (
                "🔒 **Verification Required**\n"
                f"🔗 {verification_url}\n\n"
                "⏰ Valid for 1 hour\n"
                "✨ Or upgrade to Premium!"
            )
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🔗 VERIFY", url=verification_url)],
                [InlineKeyboardButton("⭐ PREMIUM", callback_data="buy_premium")]
            ])
        else:
            welcome_text += "✨ Start browsing now!\n⭐ Upgrade for premium features"
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🌐 WEBSITE", url=Config.WEBSITE_URL)],
                [InlineKeyboardButton("⭐ PREMIUM", callback_data="buy_premium")]
            ])
        
        await message.reply_text(welcome_text, reply_markup=keyboard, disable_web_page_preview=True)
    
    # (Continue with remaining handlers...)
    # Copy paste करें बाकी handlers यहाँ
    
    logger.info("✅ All bot handlers registered")
