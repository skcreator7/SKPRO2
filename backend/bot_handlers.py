"""
bot_handlers.py - Telegram Bot Handlers for SK4FiLM
FIXED: Handles /start -1001768249569_16066_480p format
"""
import asyncio
import logging
import secrets
import re
from datetime import datetime
from typing import Dict, Any, Optional

# ✅ Complete Pyrogram imports
try:
    from pyrogram import Client, filters
    from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton, Message, CallbackQuery
    from pyrogram.errors import FloodWait, BadRequest
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
    
    async def send_file_to_user(client, user_id, file_message, quality="480p"):
        """Send file to user with proper error handling"""
        try:
            # Prepare file info
            if file_message.document:
                file_name = file_message.document.file_name or "file"
                file_size = file_message.document.file_size or 0
                file_id = file_message.document.file_id
                is_video = False
            else:
                file_name = file_message.video.file_name or "video.mp4"
                file_size = file_message.video.file_size or 0
                file_id = file_message.video.file_id
                is_video = True
            
            # Send file
            if file_message.document:
                sent = await client.send_document(
                    user_id,
                    file_id,
                    caption=(
                        f"📁 **File:** {file_name}\n"
                        f"📦 **Size:** {format_size(file_size)}\n"
                        f"📹 **Quality:** {quality}\n\n"
                        f"♻ **Forward to saved messages for safety**\n"
                        f"⏰ **Auto-delete in:** {config.AUTO_DELETE_TIME//60} minutes\n\n"
                        f"@SK4FiLM 🎬"
                    )
                )
            else:
                sent = await client.send_video(
                    user_id,
                    file_id,
                    caption=(
                        f"🎬 **Video:** {file_name}\n"
                        f"📦 **Size:** {format_size(file_size)}\n"
                        f"📹 **Quality:** {quality}\n\n"
                        f"♻ **Forward to saved messages for safety**\n"
                        f"⏰ **Auto-delete in:** {config.AUTO_DELETE_TIME//60} minutes\n\n"
                        f"@SK4FiLM 🎬"
                    )
                )
            
            logger.info(f"✅ File sent to user {user_id}: {file_name}")
            return True, file_name, file_size
            
        except FloodWait as e:
            logger.warning(f"⏳ Flood wait: {e.value}s")
            return False, f"Please wait {e.value} seconds (Telegram limit)", 0
        except Exception as e:
            logger.error(f"File sending error: {e}")
            return False, f"Error: {str(e)}", 0
    
    async def handle_file_request(client, message, file_text):
        """Handle file download request"""
        try:
            # Clean the text
            clean_text = file_text.strip()
            logger.info(f"📥 Processing file request: {clean_text}")
            
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
                return
            
            # Parse message ID
            try:
                message_id = int(parts[1].strip())
            except ValueError:
                await message.reply_text(
                    "❌ **Invalid message ID**\n\n"
                    f"Message ID '{parts[1]}' is not valid."
                )
                return
            
            # Get quality
            quality = parts[2].strip() if len(parts) > 2 else "480p"
            
            logger.info(f"📥 Parsed: channel={channel_id}, message={message_id}, quality={quality}")
            
            # Send processing message
            processing_msg = await message.reply_text(
                f"⏳ **Preparing your file...**\n\n"
                f"📹 **Quality:** {quality}\n"
                f"🔄 **Please wait...**"
            )
            
            # Get file from channel
            file_message = None
            
            # Try user client first
            if bot_instance.user_client and bot_instance.user_session_ready:
                try:
                    file_message = await bot_instance.user_client.get_messages(
                        channel_id, 
                        message_id
                    )
                    logger.info("✅ Got file via user client")
                except Exception as e:
                    logger.warning(f"User client failed: {e}")
            
            # Try bot client
            if not file_message:
                try:
                    file_message = await client.get_messages(
                        channel_id, 
                        message_id
                    )
                    logger.info("✅ Got file via bot client")
                except Exception as e:
                    logger.error(f"Bot client failed: {e}")
                    await processing_msg.edit_text(
                        f"❌ **Error getting file**\n\n"
                        f"Error: {str(e)}\n\n"
                        "Please try again or contact support."
                    )
                    return
            
            if not file_message:
                await processing_msg.edit_text(
                    "❌ **File not found**\n\n"
                    "The file may have been deleted or I don't have access."
                )
                return
            
            if not file_message.document and not file_message.video:
                await processing_msg.edit_text(
                    "❌ **Not a downloadable file**\n\n"
                    "This message doesn't contain a video or document file."
                )
                return
            
            # Send file to user
            success, result_msg, file_size = await send_file_to_user(
                client, message.chat.id, file_message, quality
            )
            
            if success:
                await processing_msg.delete()
                
                # Send success message
                success_text = (
                    f"✅ **File sent successfully!**\n\n"
                    f"📁 **File:** {result_msg}\n"
                    f"📦 **Size:** {format_size(file_size)}\n"
                    f"📹 **Quality:** {quality}\n\n"
                    f"♻ **Forward to saved messages**\n"
                    f"⏰ **Auto-deletes in:** {config.AUTO_DELETE_TIME//60} minutes\n\n"
                    f"⭐ **Upgrade to Premium for faster downloads!**"
                )
                
                await message.reply_text(
                    success_text,
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")],
                        [InlineKeyboardButton("🌐 OPEN WEBSITE", url=config.WEBSITE_URL)]
                    ])
                )
            else:
                await processing_msg.edit_text(
                    f"❌ **File sending failed**\n\n"
                    f"{result_msg}\n\n"
                    "Please try again in a few moments."
                )
            
        except Exception as e:
            logger.error(f"File request handling error: {e}")
            await message.reply_text(
                "❌ **An error occurred**\n\n"
                "Please try again or contact support."
            )
    
    # ✅ MAIN MESSAGE HANDLER
    @bot.on_message(filters.text & filters.private)
    async def handle_all_messages(client, message):
        """Handle all private messages"""
        try:
            user_id = message.from_user.id
            user_name = message.from_user.first_name or "User"
            text = message.text.strip()
            
            logger.info(f"📱 Message from {user_id} ({user_name}): {text[:50]}...")
            
            # Handle /start command
            if text.startswith('/start'):
                if len(text.split()) > 1:
                    # It's /start with file ID: /start -1001768249569_16066_480p
                    file_part = text.split(' ', 1)[1]
                    await handle_file_request(client, message, file_part)
                else:
                    # Regular /start command
                    await handle_start_command(client, message)
                return
            
            # Handle file requests (format: -1001768249569_16066_480p)
            if '_' in text and (text.startswith('-100') or text.startswith('--100')):
                await handle_file_request(client, message, text)
                return
            
            # Default: show help
            await message.reply_text(
                "🎬 **SK4FiLM Download Bot**\n\n"
                f"**How to download:**\n"
                f"1. Visit {config.WEBSITE_URL}\n"
                "2. Search for any movie\n"
                "3. Click download button\n"
                "4. File will appear here automatically\n\n"
                "Or send a file ID in this format:\n"
                "`-1001768249569_16066_480p`",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🌐 OPEN WEBSITE", url=config.WEBSITE_URL)],
                    [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")]
                ]),
                disable_web_page_preview=True
            )
            
        except Exception as e:
            logger.error(f"Main handler error: {e}")
            await message.reply_text("❌ **An error occurred**\n\nPlease try again.")
    
    # ✅ START COMMAND HANDLER
    async def handle_start_command(client, message):
        """Handle /start command"""
        user_name = message.from_user.first_name or "User"
        
        welcome_text = (
            f"🎬 **Welcome to SK4FiLM, {user_name}!**\n\n"
            "**How to download movies:**\n"
            f"1. **Visit:** {config.WEBSITE_URL}\n"
            "2. **Search for any movie**\n"
            "3. **Click download button**\n"
            "4. **File will appear here automatically**\n\n"
            "⭐ **Premium users get instant access!**"
        )
        
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🌐 OPEN WEBSITE", url=config.WEBSITE_URL)],
            [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")]
        ])
        
        await message.reply_text(welcome_text, reply_markup=keyboard, disable_web_page_preview=True)
    
    # ✅ PREMIUM CALLBACK
    @bot.on_callback_query(filters.regex(r"^buy_premium$"))
    async def buy_premium_callback(client, callback_query):
        """Show premium plans"""
        text = (
            "⭐ **SK4FiLM PREMIUM PLANS** ⭐\n\n"
            "**Basic Plan - ₹99**\n"
            "• 1080p Quality\n"
            "• 10 Daily Downloads\n"
            "• Priority Support\n\n"
            "**Premium Plan - ₹199**\n"
            "• 4K Quality\n"
            "• Unlimited Downloads\n"
            "• No Ads\n"
            "• Highest Priority\n\n"
            "Click below to purchase:"
        )
        
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("💰 BUY BASIC (₹99)", callback_data="plan_basic")],
            [InlineKeyboardButton("💰 BUY PREMIUM (₹199)", callback_data="plan_premium")],
            [InlineKeyboardButton("🔙 BACK", callback_data="back_to_start")]
        ])
        
        await callback_query.message.edit_text(text, reply_markup=keyboard)
    
    # ✅ BACK TO START
    @bot.on_callback_query(filters.regex(r"^back_to_start$"))
    async def back_to_start_callback(client, callback_query):
        user_name = callback_query.from_user.first_name or "User"
        welcome_text = (
            f"🎬 **Welcome back, {user_name}!**\n\n"
            f"Visit {config.WEBSITE_URL} to download movies.\n"
            "Click any download button and the file will appear here automatically!"
        )
        
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🌐 OPEN WEBSITE", url=config.WEBSITE_URL)],
            [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")]
        ])
        
        await callback_query.message.edit_text(welcome_text, reply_markup=keyboard, disable_web_page_preview=True)
    
    # ✅ PLAN SELECTION CALLBACKS
    @bot.on_callback_query(filters.regex(r"^plan_"))
    async def plan_selection_callback(client, callback_query):
        plan_type = callback_query.data.split('_')[1]
        
        if plan_type == "basic":
            amount = 99
            plan_name = "Basic Plan"
            upi_id = config.UPI_ID_BASIC if hasattr(config, 'UPI_ID_BASIC') else "sk4filmbot@ybl"
        else:
            amount = 199
            plan_name = "Premium Plan"
            upi_id = config.UPI_ID_PREMIUM if hasattr(config, 'UPI_ID_PREMIUM') else "sk4filmbot@ybl"
        
        payment_id = secrets.token_hex(8)
        
        text = (
            f"💰 **Payment for {plan_name}**\n\n"
            f"**Amount:** ₹{amount}\n"
            f"**UPI ID:** `{upi_id}`\n\n"
            "**Payment Instructions:**\n"
            f"1. Send ₹{amount} to UPI ID: `{upi_id}`\n"
            "2. Take screenshot of payment\n"
            "3. Send screenshot to this bot\n\n"
            "⏰ **Payment valid for 1 hour**\n"
            "✅ **Admin will activate within 24 hours**"
        )
        
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("📸 SEND SCREENSHOT", callback_data=f"send_screenshot_{payment_id}")],
            [InlineKeyboardButton("🔙 BACK TO PLANS", callback_data="buy_premium")]
        ])
        
        await callback_query.message.edit_text(text, reply_markup=keyboard)
    
    logger.info("✅ Bot handlers setup complete - Ready to send files!")
    return bot
