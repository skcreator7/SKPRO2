"""
bot_handlers.py - Telegram Bot Handlers for SK4FiLM
FIXED: Complete Pyrogram imports
"""
import asyncio
import logging
import secrets
import re
from datetime import datetime
from typing import Dict, Any, Optional

# ✅ FIX: ADD THESE IMPORTS AT THE VERY TOP
from pyrogram import Client, filters
from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton, Message, CallbackQuery
from pyrogram.errors import FloodWait, BadRequest

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
        
        # Initialize systems
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
            
            # Initialize bot
            self.bot = Client(
                "bot",
                api_id=self.config.API_ID,
                api_hash=self.config.API_HASH,
                bot_token=self.config.BOT_TOKEN,
                workers=20
            )
            
            # Initialize user client
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
    
    # ✅ FILE SENDING HANDLER - MOST IMPORTANT
    @bot.on_message(filters.text & filters.private)
    async def handle_file_request(client, message):
        """Handle file download requests from website"""
        try:
            user_id = message.from_user.id
            user_name = message.from_user.first_name or "User"
            
            text = message.text.strip()
            logger.info(f"📥 File request from {user_id}: {text}")
            
            # Check if it's a file link format: channel_message_quality
            if '_' in text and text.count('_') >= 2:
                try:
                    parts = text.split('_')
                    if len(parts) >= 3:
                        channel_id = int(parts[0])
                        message_id = int(parts[1])
                        quality = parts[2] if len(parts) > 2 else "HD"
                        
                        # Send processing message
                        processing_msg = await message.reply_text(
                            f"⏳ **Preparing your file...**\n\n"
                            f"📹 **Quality:** {quality}\n"
                            f"🔄 **Please wait...**"
                        )
                        
                        # Get file from channel using user client
                        file_message = None
                        if bot_instance.user_client and bot_instance.user_session_ready:
                            try:
                                file_message = await bot_instance.user_client.get_messages(
                                    channel_id, 
                                    message_id
                                )
                            except Exception as e:
                                logger.error(f"User client error: {e}")
                                file_message = None
                        
                        # Fallback to bot client
                        if not file_message:
                            try:
                                file_message = await client.get_messages(
                                    channel_id, 
                                    message_id
                                )
                            except Exception as e:
                                logger.error(f"Bot client error: {e}")
                                await processing_msg.edit_text(
                                    "❌ **File not found or access denied**\n"
                                    "The file may have been deleted or I don't have access."
                                )
                                return
                        
                        if not file_message or (not file_message.document and not file_message.video):
                            await processing_msg.edit_text("❌ **File not found**\n\nThe file may have been deleted.")
                            return
                        
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
                        
                        # Send file to user
                        try:
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
                            
                            await processing_msg.delete()
                            logger.info(f"✅ File sent to user {user_id}: {file_name}")
                            
                            # Send success message
                            success_text = (
                                f"✅ **File sent successfully!**\n\n"
                                f"📁 **File:** {file_name}\n"
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
                            
                            return
                            
                        except FloodWait as e:
                            await processing_msg.edit_text(
                                f"⏳ **Please wait {e.value} seconds**\n"
                                "Telegram rate limit reached. Trying again..."
                            )
                            await asyncio.sleep(e.value)
                            # Retry logic could be added here
                            return
                        except Exception as e:
                            logger.error(f"File sending error: {e}")
                            await processing_msg.edit_text("❌ **Error sending file**\n\nPlease try again later.")
                            return
                            
                except Exception as e:
                    logger.error(f"File download processing error: {e}")
                    try:
                        await processing_msg.edit_text("❌ **Error processing request**\n\nPlease try again.")
                    except:
                        pass
                    return
            
            # If not a file link, show help
            await message.reply_text(
                "🎬 **SK4FiLM File Download**\n\n"
                "**How to download:**\n"
                f"1. **Visit website:** {config.WEBSITE_URL}\n"
                "2. **Find any movie**\n"
                "3. **Click download button**\n"
                "4. **File link will appear here automatically**\n\n"
                "The bot will automatically send you the file! 🍿\n\n"
                "⭐ **Premium users get instant access!**",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🌐 OPEN WEBSITE", url=config.WEBSITE_URL)],
                    [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")]
                ]),
                disable_web_page_preview=True
            )
            
        except Exception as e:
            logger.error(f"Handler error: {e}")
            await message.reply_text("❌ **An error occurred**\n\nPlease try again.")
    
    # ✅ START COMMAND
    @bot.on_message(filters.command("start") & filters.private)
    async def start_handler(client, message):
        user_id = message.from_user.id
        user_name = message.from_user.first_name or "User"
        
        welcome_text = (
            f"🎬 **Welcome to SK4FiLM, {user_name}!**\n\n"
            "**How to download movies:**\n"
            f"1. **Visit:** {config.WEBSITE_URL}\n"
            "2. **Search for any movie**\n"
            "3. **Click download button**\n"
            "4. **File will appear here automatically**\n\n"
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
        text = "⭐ **SK4FiLM PREMIUM PLANS** ⭐\n\n"
        text += "**Basic Plan - ₹99**\n• 1080p Quality\n• 10 Daily Downloads\n• Priority Support\n\n"
        text += "**Premium Plan - ₹199**\n• 4K Quality\n• Unlimited Downloads\n• No Ads\n\n"
        text += "Click below to purchase:"
        
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
        welcome_text = f"🎬 **Welcome back, {user_name}!**\n\nVisit our website to download movies."
        
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🌐 OPEN WEBSITE", url=config.WEBSITE_URL)],
            [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")]
        ])
        
        await callback_query.message.edit_text(welcome_text, reply_markup=keyboard, disable_web_page_preview=True)
    
    logger.info("✅ Bot handlers setup complete - File sending enabled!")
    return bot
