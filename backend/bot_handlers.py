"""
bot_handlers.py - Telegram Bot Handlers with Website Integration
"""
import asyncio
import logging
import re
import secrets
from datetime import datetime
from typing import Dict, Any, Optional

from pyrogram import Client, filters, idle
from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton, Message, CallbackQuery
from pyrogram.errors import FloodWait

logger = logging.getLogger(__name__)

# Import utility functions from app
from app import (
    normalize_title, extract_title_from_file, format_size, 
    detect_quality, is_video_file, safe_telegram_operation,
    safe_telegram_generator, index_single_file, auto_delete_file,
    Config, extract_title_smart
)

class SK4FiLMBot:
    def __init__(self, config, db_manager=None):
        self.config = config
        self.db_manager = db_manager
        self.bot = None
        self.user_client = None
        self.bot_started = False
        self.user_session_ready = False
        
        # Initialize systems
        self.verification_system = None
        self.premium_system = None
        self.poster_fetcher = None
        self.cache_manager = None
        self.task_manager = None
        
        # Website requests queue
        self.website_requests = {}
    
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
    
    async def process_website_download_request(self, user_id: int, channel_id: int, message_id: int, quality: str = "HD") -> Dict[str, Any]:
        """Process download request from website"""
        try:
            logger.info(f"Processing website download request: user={user_id}, channel={channel_id}, msg={message_id}, quality={quality}")
            
            # Get file from Telegram
            file_message = await safe_telegram_operation(
                self.bot.get_messages,
                channel_id, 
                message_id
            )
            
            if not file_message or (not file_message.document and not file_message.video):
                return {
                    'status': 'error',
                    'message': 'File not found'
                }
            
            # Send file to user
            if file_message.document:
                sent = await safe_telegram_operation(
                    self.bot.send_document,
                    user_id, 
                    file_message.document.file_id,
                    caption=(
                        f"✅ **File from SK4FiLM Website**\n\n"
                        f"📁 **File:** {file_message.document.file_name or 'Document'}\n"
                        f"📦 **Quality:** {quality}\n"
                        f"📊 **Size:** {format_size(file_message.document.file_size)}\n"
                        f"⏰ **Auto-delete in:** {self.config.AUTO_DELETE_TIME//60} minutes\n\n"
                        f"♻ **Please forward to saved messages**\n"
                        f"⭐ **Thank you for using SK4FiLM!**\n\n"
                        f"@SK4FiLM 🍿"
                    )
                )
                file_size = file_message.document.file_size
                file_name = file_message.document.file_name
            else:
                sent = await safe_telegram_operation(
                    self.bot.send_video,
                    user_id, 
                    file_message.video.file_id,
                    caption=(
                        f"✅ **File from SK4FiLM Website**\n\n"
                        f"🎬 **Video:** {file_message.video.file_name or 'Video'}\n"
                        f"📦 **Quality:** {quality}\n" 
                        f"📊 **Size:** {format_size(file_message.video.file_size)}\n"
                        f"⏰ **Auto-delete in:** {self.config.AUTO_DELETE_TIME//60} minutes\n\n"
                        f"♻ **Please forward to saved messages**\n"
                        f"⭐ **Thank you for using SK4FiLM!**\n\n"
                        f"@SK4FiLM 🍿"
                    )
                )
                file_size = file_message.video.file_size
                file_name = file_message.video.file_name
            
            if sent:
                # Auto-delete file after specified time
                if self.config.AUTO_DELETE_TIME > 0:
                    asyncio.create_task(auto_delete_file(sent, self.config.AUTO_DELETE_TIME))
                
                # Record download
                if self.premium_system:
                    try:
                        await self.premium_system.record_download(user_id)
                    except:
                        pass
                
                logger.info(f"✅ File sent to user {user_id} via website: {file_name}, {format_size(file_size)}")
                
                return {
                    'status': 'success',
                    'message': 'File sent successfully',
                    'file_name': file_name,
                    'file_size': file_size,
                    'quality': quality,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'status': 'error',
                    'message': 'Failed to send file'
                }
                
        except Exception as e:
            logger.error(f"Error processing website download: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }

async def setup_bot_handlers(bot: Client, bot_instance):
    """Setup bot commands and handlers with website integration"""
    
    @bot.on_message(filters.command("start") & filters.private)
    async def start_handler(client, message):
        user_id = message.from_user.id
        user_name = message.from_user.first_name or "User"
        
        # Check if this is a verification token
        if len(message.command) > 1:
            command_arg = message.command[1]
            
            if command_arg.startswith('verify_'):
                token = command_arg[7:]
                
                # Verify the token
                if bot_instance.verification_system:
                    success, verified_user_id, msg = await bot_instance.verification_system.verify_user_token(token)
                    
                    if success and verified_user_id == user_id:
                        # Successfully verified
                        await message.reply_text(
                            f"✅ **Verification Successful, {user_name}!**\n\n"
                            "🎉 **You are now verified!**\n\n"
                            "✨ **Now you can:**\n"
                            "1. Go to our website\n"
                            "2. Find any movie\n"
                            "3. Click download\n"
                            "4. File will be sent here automatically!\n\n"
                            f"🌐 **Website:** {Config.WEBSITE_URL}\n"
                            f"⏰ **Verification valid for 6 hours**\n\n"
                            "No need to paste anything - just click download on website!",
                            reply_markup=InlineKeyboardMarkup([
                                [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)],
                                [InlineKeyboardButton("🎬 BROWSE MOVIES", url=f"{Config.WEBSITE_URL}/movies")]
                            ])
                        )
                        return
                    else:
                        await message.reply_text(
                            f"❌ **Verification Failed**\n\n"
                            f"{msg}\n\n"
                            f"Please try again or contact support."
                        )
                        return
        
        # Regular start command
        welcome_text = (
            f"🎬 **Welcome to SK4FiLM, {user_name}!**\n\n"
            "🌐 **Use our website to browse and download movies:**\n"
            f"{Config.WEBSITE_URL}\n\n"
        )
        
        # Check premium status
        is_premium = False
        if bot_instance.premium_system is not None:
            try:
                is_premium = await bot_instance.premium_system.is_premium_user(user_id)
            except:
                pass
        
        if is_premium:
            welcome_text += f"🌟 **Premium User**\n✅ **You have full access!**\n\n"
            welcome_text += "✨ **How to download:**\n"
            welcome_text += "1. Visit our website\n"
            welcome_text += "2. Find any movie\n"
            welcome_text += "3. Click download\n"
            welcome_text += "4. File comes here automatically!\n\n"
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)],
                [InlineKeyboardButton("🎬 BROWSE MOVIES", url=f"{Config.WEBSITE_URL}/movies")],
                [InlineKeyboardButton("📊 PREMIUM STATUS", callback_data=f"premium_status_{user_id}")]
            ])
        else:
            # Check if user is already verified
            is_verified = False
            if bot_instance.verification_system:
                is_verified, _ = await bot_instance.verification_system.check_user_verified(user_id)
            
            if is_verified:
                welcome_text += (
                    "✅ **Already Verified**\n"
                    "✨ **You can download from website!**\n\n"
                    "📥 **How to download:**\n"
                    "1. Visit our website\n"
                    "2. Find any movie\n"
                    "3. Click download\n"
                    "4. File comes here automatically!\n\n"
                )
                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)],
                    [InlineKeyboardButton("🎬 BROWSE MOVIES", url=f"{Config.WEBSITE_URL}/movies")]
                ])
            else:
                # User needs verification
                if Config.VERIFICATION_REQUIRED:
                    # Create verification link (will be shortened)
                    if bot_instance.verification_system:
                        verification_link = await bot_instance.verification_system.get_verification_link_for_user(user_id)
                        
                        welcome_text += (
                            "🔒 **One-Time Verification Required**\n\n"
                            "To download from website, complete one-time verification:\n\n"
                            f"🔗 **Verification Link:** {verification_link['short_url']}\n\n"
                            "**Steps:**\n"
                            "1. Click the link above\n"
                            "2. Click 'Start' in the bot\n"
                            "3. You'll be verified for 6 hours\n"
                            "4. Then download from website!\n\n"
                            "⏰ **Link valid for 1 hour**\n"
                            "✅ **Verification valid for 6 hours**\n\n"
                            "⭐ **Or upgrade to Premium for instant access!**"
                        )
                        keyboard = InlineKeyboardMarkup([
                            [InlineKeyboardButton("🔗 VERIFY NOW", url=verification_link['short_url'])],
                            [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")],
                            [InlineKeyboardButton("🔄 CHECK ACCESS", callback_data=f"check_access_{user_id}")]
                        ])
                else:
                    # No verification required
                    welcome_text += "✨ **Start downloading from website!**\n\n"
                    welcome_text += "📥 **How to download:**\n"
                    welcome_text += "1. Visit our website\n"
                    welcome_text += "2. Find any movie\n"
                    welcome_text += "3. Click download\n"
                    welcome_text += "4. File comes here automatically!\n\n"
                    keyboard = InlineKeyboardMarkup([
                        [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)],
                        [InlineKeyboardButton("🎬 BROWSE MOVIES", url=f"{Config.WEBSITE_URL}/movies")],
                        [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")]
                    ])
        
        await message.reply_text(welcome_text, reply_markup=keyboard, disable_web_page_preview=True)
    
    @bot.on_callback_query(filters.regex(r"^check_access_"))
    async def check_access_callback(client, callback_query):
        """Check user access status"""
        user_id = int(callback_query.data.split('_')[2])
        user_name = callback_query.from_user.first_name or "User"
        
        # Check access status
        has_access = False
        access_message = ""
        
        if bot_instance.premium_system:
            try:
                is_premium = await bot_instance.premium_system.is_premium_user(user_id)
                if is_premium:
                    has_access = True
                    access_message = "🌟 **Premium User** - Full website access!"
            except:
                pass
        
        if not has_access and bot_instance.verification_system:
            is_verified, verify_msg = await bot_instance.verification_system.check_user_verified(user_id)
            if is_verified:
                has_access = True
                access_message = f"✅ **Verified User** - {verify_msg}"
        
        if has_access:
            await callback_query.message.edit_text(
                f"🎉 **Access Granted, {user_name}!**\n\n"
                f"{access_message}\n\n"
                "📥 **You can now download from website:**\n"
                "1. Visit our website\n"
                "2. Find any movie\n"
                "3. Click download\n"
                "4. File comes here automatically!\n\n"
                f"🌐 **Website:** {Config.WEBSITE_URL}",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)],
                    [InlineKeyboardButton("🎬 BROWSE MOVIES", url=f"{Config.WEBSITE_URL}/movies")]
                ])
            )
        else:
            # User needs verification
            if bot_instance.verification_system:
                verification_link = await bot_instance.verification_system.get_verification_link_for_user(user_id)
                
                await callback_query.message.edit_text(
                    "🔒 **Verification Required**\n\n"
                    "To download from website, complete verification:\n\n"
                    f"🔗 **Verification Link:** {verification_link['short_url']}\n\n"
                    "**Steps:**\n"
                    "1. Click the link above\n"
                    "2. Click 'Start' in the bot\n"
                    "3. You'll be verified for 6 hours\n"
                    "4. Then download from website!\n\n"
                    "⏰ **Link valid for 1 hour**",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("🔗 VERIFY NOW", url=verification_link['short_url'])],
                        [InlineKeyboardButton("🔄 CHECK AGAIN", callback_data=f"check_access_{user_id}")]
                    ]),
                    disable_web_page_preview=True
                )
    
    @bot.on_message(filters.command("website") & filters.private)
    async def website_command(client, message):
        """Website command - direct users to website"""
        user_id = message.from_user.id
        user_name = message.from_user.first_name or "User"
        
        text = f"🌐 **SK4FiLM Website Guide - {user_name}**\n\n"
        
        # Check access status
        has_access = False
        if bot_instance.premium_system:
            try:
                is_premium = await bot_instance.premium_system.is_premium_user(user_id)
                if is_premium:
                    has_access = True
                    text += "🌟 **Premium User** - Full access!\n\n"
            except:
                pass
        
        if not has_access and bot_instance.verification_system:
            is_verified, verify_msg = await bot_instance.verification_system.check_user_verified(user_id)
            if is_verified:
                has_access = True
                text += f"✅ **Verified User** - {verify_msg}\n\n"
        
        if has_access:
            text += "✨ **How to download from website:**\n"
            text += "1. Visit our website\n"
            text += "2. Browse movies\n"
            text += "3. Select quality\n"
            text += "4. Click download\n"
            text += "5. File comes here automatically!\n\n"
            text += "**No need to paste anything!**\n\n"
            text += f"🌐 **Website:** {Config.WEBSITE_URL}"
            
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)],
                [InlineKeyboardButton("🎬 BROWSE MOVIES", url=f"{Config.WEBSITE_URL}/movies")]
            ])
        else:
            text += "🔒 **Verification Required**\n\n"
            text += "To download from website, you need to complete one-time verification.\n\n"
            text += "**Benefits after verification:**\n"
            text += "✅ Download directly from website\n"
            text += "✅ No need to paste file details\n"
            text += "✅ Just click download on website\n"
            text += "✅ Files come here automatically!\n\n"
            text += f"🌐 **Website:** {Config.WEBSITE_URL}"
            
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🔗 GET VERIFICATION", callback_data=f"check_access_{user_id}")],
                [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")]
            ])
        
        await message.reply_text(text, reply_markup=keyboard, disable_web_page_preview=True)
    
    @bot.on_message(filters.command("help") & filters.private)
    async def help_command(client, message):
        """Help command"""
        user_id = message.from_user.id
        
        text = "🆘 **SK4FiLM Help Guide**\n\n"
        text += "**How to download movies:**\n\n"
        
        text += "📱 **Method 1: Website (Recommended)**\n"
        text += "1. Visit our website\n"
        text += "2. Browse movies\n"
        text += "3. Select quality\n"
        text += "4. Click download\n"
        text += "5. File comes here automatically!\n\n"
        
        text += "**Commands:**\n"
        text += "/start - Start the bot\n"
        text += "/website - Go to website\n"
        text += "/verify - Get verification\n"
        text += "/premium - Premium plans\n"
        text += "/status - Check your status\n"
        text += "/help - This help message\n\n"
        
        text += f"🌐 **Website:** {Config.WEBSITE_URL}"
        
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)],
            [InlineKeyboardButton("🔄 CHECK STATUS", callback_data=f"check_access_{user_id}")]
        ])
        
        await message.reply_text(text, reply_markup=keyboard, disable_web_page_preview=True)
    
    @bot.on_message(filters.command("status") & filters.private)
    async def status_command(client, message):
        """Check user status"""
        user_id = message.from_user.id
        user_name = message.from_user.first_name or "User"
        
        text = f"📊 **Your Status - {user_name}**\n\n"
        
        # Check premium status
        is_premium = False
        premium_info = ""
        if bot_instance.premium_system:
            try:
                is_premium = await bot_instance.premium_system.is_premium_user(user_id)
                if is_premium:
                    tier = await bot_instance.premium_system.get_user_tier(user_id)
                    sub_details = await bot_instance.premium_system.get_subscription_details(user_id)
                    premium_info = f"🌟 **Premium Plan:** {tier.value}\n"
                    premium_info += f"⏰ **Days remaining:** {sub_details.get('days_remaining', 0)}\n\n"
            except:
                pass
        
        if is_premium:
            text += premium_info
            text += "✅ **Website Access:** Full\n"
            text += "✅ **Download Method:** Click & Get\n"
            text += "✅ **Verification:** Not required\n"
            text += "✅ **Status:** Active Premium\n\n"
            text += "✨ **You can download directly from website!**"
        else:
            # Check verification status
            is_verified = False
            verify_info = ""
            if bot_instance.verification_system:
                is_verified, verify_msg = await bot_instance.verification_system.check_user_verified(user_id)
                verify_info = verify_msg
            
            if is_verified:
                text += "✅ **Verification:** Active\n"
                text += f"⏰ **Status:** {verify_info}\n"
                text += "✅ **Website Access:** Full\n"
                text += "✅ **Download Method:** Click & Get\n\n"
                text += "✨ **You can download directly from website!**"
            else:
                text += "❌ **Verification:** Not active\n"
                text += "❌ **Website Access:** Restricted\n"
                text += "🔒 **Download Method:** Verification required\n\n"
                text += "⚠️ **Complete verification to download from website!**"
        
        text += f"\n\n🌐 **Website:** {Config.WEBSITE_URL}"
        
        keyboard = []
        if not is_premium:
            if not is_verified:
                if bot_instance.verification_system:
                    verification_link = await bot_instance.verification_system.get_verification_link_for_user(user_id)
                    keyboard.append([InlineKeyboardButton("🔗 VERIFY NOW", url=verification_link['short_url'])])
            keyboard.append([InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")])
        
        keyboard.append([InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)])
        
        await message.reply_text(
            text, 
            reply_markup=InlineKeyboardMarkup(keyboard),
            disable_web_page_preview=True
        )
    
    @bot.on_message(filters.text & filters.private & ~filters.command(['start', 'stats', 'premium', 'verify', 'status', 'website', 'help', 'index', 'broadcast', 'premiumuser']))
    async def text_handler(client, message):
        """Handle text messages - legacy support for direct paste"""
        user_id = message.from_user.id
        user_name = message.from_user.first_name or "User"
        
        # Check if message is in file format (legacy support)
        if message.text and '_' in message.text:
            try:
                parts = message.text.strip().split('_')
                if len(parts) >= 2 and (parts[0].isdigit() or (parts[0].startswith('-') and parts[0][1:].isdigit())):
                    
                    # Send message about website method
                    await message.reply_text(
                        f"👋 **Hey {user_name}!**\n\n"
                        "✨ **Try our new website method!**\n\n"
                        "**Website Method:**\n"
                        "1. Visit our website\n"
                        "2. Browse movies\n"
                        "3. Click download\n"
                        "4. File comes here automatically!\n\n"
                        "**No need to paste file details!**\n\n"
                        f"🌐 **Website:** {Config.WEBSITE_URL}\n\n"
                        "⭐ **It's easier and faster!**",
                        reply_markup=InlineKeyboardMarkup([
                            [InlineKeyboardButton("🌐 TRY WEBSITE", url=Config.WEBSITE_URL)],
                            [InlineKeyboardButton("📚 HELP", callback_data="help_info")]
                        ]),
                        disable_web_page_preview=True
                    )
                    return
                    
            except:
                pass
        
        # If not a file format, show help
        await message.reply_text(
            "🎬 **SK4FiLM Download Guide**\n\n"
            "**Recommended Method:**\n"
            "📱 **Website Download**\n"
            "1. Visit our website\n"
            "2. Browse movies\n"
            "3. Click download\n"
            "4. File comes here automatically!\n\n"
            "**Legacy Method:**\n"
            "🤖 **Direct Paste**\n"
            "Paste format: `channel_id_message_id_quality`\n\n"
            f"🌐 **Website:** {Config.WEBSITE_URL}\n\n"
            "⭐ **Try the website method - it's easier!**",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)],
                [InlineKeyboardButton("📚 HELP GUIDE", callback_data="help_info")]
            ]),
            disable_web_page_preview=True
        )
    
    @bot.on_callback_query(filters.regex(r"^help_info$"))
    async def help_info_callback(client, callback_query):
        """Show help info"""
        user_id = callback_query.from_user.id
        
        text = "📚 **SK4FiLM Help Guide**\n\n"
        text += "**Two Ways to Download:**\n\n"
        
        text += "📱 **1. Website Method (Easiest)**\n"
        text += "• Visit our website\n"
        text += "• Browse movies\n"
        text += "• Select quality\n"
        text += "• Click download\n"
        text += "• File comes automatically!\n\n"
        
        text += "**First Time?**\n"
        text += "Complete one-time verification to use website method.\n"
        text += "Verification is valid for 6 hours.\n\n"
        
        text += f"🌐 **Website:** {Config.WEBSITE_URL}"
        
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)],
            [InlineKeyboardButton("🔗 GET VERIFIED", callback_data=f"check_access_{user_id}")]
        ])
        
        await callback_query.message.edit_text(text, reply_markup=keyboard, disable_web_page_preview=True)
    
    # Other handlers remain the same (buy_premium, select_plan, premium_status, etc.)
    # ... [Keep all the other handlers from previous version]

async def setup_website_integration(bot_instance):
    """Setup website integration for the bot"""
    logger.info("🔗 Setting up website integration...")
    
    # This function would be called from main app
    pass
