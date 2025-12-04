"""
bot_handlers.py - Telegram Bot Handlers for SK4FiLM
FIXED: Handles /start -1001768249569_16066_480p format with access control
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
            
            # Setup handlers
            await setup_bot_handlers(self.bot, self)
            
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

async def send_file_to_user(client, user_id, file_message, quality="480p", config=None, bot_instance=None):
    """Send file to user with verification check AND SUCCESS MESSAGE WITH BUTTONS"""
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
        
        # ✅ FILE CAPTION WITH STATUS
        file_caption = (
            f"📁 **File:** `{file_name}`\n"
            f"📦 **Size:** {format_size(file_size)}\n"
            f"📹 **Quality:** {quality}\n"
            f"{status_icon} **Status:** {user_status}\n\n"
            f"♻ **Forward to saved messages for safety**\n"
            f"⏰ **Auto-delete in:** {config.AUTO_DELETE_TIME//60} minutes\n\n"
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
            
            # ✅ SUCCESS RESPONSE WITH DATA FOR SUCCESS MESSAGE
            return True, {
                'success': True,
                'file_name': file_name,
                'file_size': file_size,
                'quality': quality,
                'user_status': user_status,
                'status_icon': status_icon,
                'auto_delete_minutes': config.AUTO_DELETE_TIME//60
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
                    
                    return True, {
                        'success': True,
                        'file_name': file_name,
                        'file_size': file_size,
                        'quality': quality,
                        'user_status': user_status,
                        'status_icon': status_icon,
                        'auto_delete_minutes': config.AUTO_DELETE_TIME//60,
                        'refreshed': True
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
        
        # ✅ Send file to user with bot_instance parameter
        success, result_data, file_size = await send_file_to_user(
            client, message.chat.id, file_message, quality, config, bot_instance
        )
        
        if success:
            # File was sent with caption and buttons
            await processing_msg.delete()
            
            # ✅ SEND SUCCESS MESSAGE WITH MORE BUTTONS
            success_buttons = []
            
            if result_data.get('user_status') == "Verified User ✅":
                # Verified users see upgrade button
                success_buttons.append([InlineKeyboardButton("⭐ UPGRADE TO PREMIUM", callback_data="buy_premium")])
            
            success_buttons.extend([
                [InlineKeyboardButton("🔍 SEARCH MORE", url=config.WEBSITE_URL)],
                [InlineKeyboardButton("📱 JOIN CHANNEL", url=config.MAIN_CHANNEL_LINK)],
                [InlineKeyboardButton("🔄 GET ANOTHER FILE", callback_data="back_to_start")]
            ])
            
            success_text = (
                f"🎉 **Download Complete!** 🎉\n\n"
                f"✅ **File successfully sent!**\n"
                f"📁 **Name:** `{result_data['file_name']}`\n"
                f"📦 **Size:** {format_size(result_data['file_size'])}\n"
                f"📹 **Quality:** {result_data['quality']}\n"
                f"{result_data['status_icon']} **Status:** {result_data['user_status']}\n\n"
                f"⚠️ **Important Notes:**\n"
                f"• File auto-deletes in {result_data['auto_delete_minutes']} minutes\n"
                f"• Forward to saved messages for safety\n"
                f"• Check spam folder if not received\n\n"
                f"🎬 **Thank you for using SK4FiLM!**"
            )
            
            await message.reply_text(
                success_text,
                reply_markup=InlineKeyboardMarkup(success_buttons),
                disable_web_page_preview=True
            )
            
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
            
            if error_buttons:
                await processing_msg.edit_text(
                    error_text,
                    reply_markup=InlineKeyboardMarkup(error_buttons),
                    disable_web_page_preview=True
                )
            else:
                await processing_msg.edit_text(error_text)
        
    except Exception as e:
        logger.error(f"File request handling error: {e}")
        await message.reply_text(
            "❌ **An error occurred**\n\n"
            "Please try again or contact support.\n"
            f"Error: {str(e)[:100]}"
        )

async def setup_bot_handlers(bot: Client, bot_instance):
    """Setup bot commands and handlers"""
    config = bot_instance.config
    
    # ✅ START COMMAND HANDLER
    @bot.on_message(filters.command("start"))
    async def handle_start_command(client, message):
        """Handle /start command"""
        user_name = message.from_user.first_name or "User"
        user_id = message.from_user.id
        
        # Check if there's additional text (file request)
        if len(message.command) > 1:
            file_text = ' '.join(message.command[1:])
            await handle_file_request(client, message, file_text, bot_instance)
            return
        
        # Check user status
        user_status = "New User"
        status_icon = "👋"
        
        if user_id in config.ADMIN_IDS:
            user_status = "Admin 👑"
            status_icon = "👑"
        elif bot_instance.premium_system:
            is_premium = await bot_instance.premium_system.is_premium_user(user_id)
            if is_premium:
                user_status = "Premium ⭐"
                status_icon = "⭐"
            elif bot_instance.verification_system:
                is_verified, _ = await bot_instance.verification_system.check_user_verified(
                    user_id, bot_instance.premium_system
                )
                if is_verified:
                    user_status = "Verified ✅"
                    status_icon = "✅"
        
        welcome_text = (
            f"🎬 **Welcome to SK4FiLM, {user_name}!** {status_icon}\n\n"
            f"**Your Status:** {user_status}\n\n"
            "**How to download movies:**\n"
            f"1. **Visit:** {config.WEBSITE_URL}\n"
            "2. **Search for any movie**\n"
            "3. **Click download button**\n"
            "4. **File will appear here automatically**\n\n"
            f"**Current Access:**\n"
            f"{status_icon} **{user_status}** - {'Full access' if status_icon in ['👑', '⭐', '✅'] else 'Limited access'}\n\n"
            f"⭐ **Premium users get instant access!**"
        )
        
        buttons = []
        if status_icon not in ['👑', '⭐', '✅']:
            # Unverified users see verify button
            buttons.append([InlineKeyboardButton("🔗 GET VERIFIED", callback_data="get_verified")])
        
        buttons.append([InlineKeyboardButton("🌐 OPEN WEBSITE", url=config.WEBSITE_URL)])
        
        if status_icon != '👑':  # Non-admins see premium button
            buttons.append([InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")])
        
        buttons.append([InlineKeyboardButton("📢 JOIN CHANNEL", url=config.MAIN_CHANNEL_LINK)])
        
        keyboard = InlineKeyboardMarkup(buttons)
        
        await message.reply_text(welcome_text, reply_markup=keyboard, disable_web_page_preview=True)
    
    # ✅ Also handle direct messages with file format
    @bot.on_message(filters.private & filters.regex(r'^-?\d+_\d+'))
    async def handle_direct_file_request(client, message):
        """Handle direct file format messages like -1001768249569_16066_480p"""
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
                "To download files, you need to verify by joining our channel:\n\n"
                f"🔗 **Verification Link:**\n{verification_data['short_url']}\n\n"
                f"⏰ **Valid for:** {verification_data['valid_for_hours']} hours\n\n"
                "**Steps:**\n"
                "1. Click the link above\n"
                "2. Join the channel\n"
                "3. Click the verify button\n"
                "4. Come back here and try downloading again"
            )
            
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🔗 VERIFY NOW", url=verification_data['short_url'])],
                [InlineKeyboardButton("⭐ BUY PREMIUM (No Verification)", callback_data="buy_premium")],
                [InlineKeyboardButton("🔙 BACK", callback_data="back_to_start")]
            ])
            
            await callback_query.message.edit_text(text, reply_markup=keyboard, disable_web_page_preview=True)
        else:
            await callback_query.message.edit_text(
                "❌ Verification system not available. Please try again later.",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 BACK", callback_data="back_to_start")]
                ])
            )
    
    # ✅ PREMIUM CALLBACK
    @bot.on_callback_query(filters.regex(r"^buy_premium$"))
    async def buy_premium_callback(client, callback_query):
        """Show premium plans"""
        user_id = callback_query.from_user.id
        
        # Check if already premium
        if bot_instance.premium_system:
            is_premium = await bot_instance.premium_system.is_premium_user(user_id)
            if is_premium:
                sub_details = await bot_instance.premium_system.get_subscription_details(user_id)
                
                text = (
                    f"⭐ **You're Already Premium!** ⭐\n\n"
                    f"**Plan:** {sub_details.get('tier_name', 'Premium')}\n"
                    f"**Status:** {sub_details.get('status', 'Active')}\n"
                    f"**Days Left:** {sub_details.get('days_remaining', 0)}\n\n"
                    "Enjoy unlimited downloads! 🎬"
                )
                
                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 BACK", callback_data="back_to_start")]
                ])
                
                await callback_query.message.edit_text(text, reply_markup=keyboard)
                return
        
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
        user_id = callback_query.from_user.id
        
        # Check user status
        user_status = "New User"
        status_icon = "👋"
        
        if user_id in config.ADMIN_IDS:
            user_status = "Admin 👑"
            status_icon = "👑"
        elif bot_instance.premium_system:
            is_premium = await bot_instance.premium_system.is_premium_user(user_id)
            if is_premium:
                user_status = "Premium ⭐"
                status_icon = "⭐"
            elif bot_instance.verification_system:
                is_verified, _ = await bot_instance.verification_system.check_user_verified(
                    user_id, bot_instance.premium_system
                )
                if is_verified:
                    user_status = "Verified ✅"
                    status_icon = "✅"
        
        welcome_text = (
            f"🎬 **Welcome back, {user_name}!** {status_icon}\n\n"
            f"**Your Status:** {user_status}\n\n"
            f"Visit {config.WEBSITE_URL} to download movies.\n"
            "Click any download button and the file will appear here automatically!"
        )
        
        buttons = []
        if status_icon not in ['👑', '⭐', '✅']:
            buttons.append([InlineKeyboardButton("🔗 GET VERIFIED", callback_data="get_verified")])
        
        buttons.append([InlineKeyboardButton("🌐 OPEN WEBSITE", url=config.WEBSITE_URL)])
        
        if status_icon != '👑':
            buttons.append([InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")])
        
        buttons.append([InlineKeyboardButton("📢 JOIN CHANNEL", url=config.MAIN_CHANNEL_LINK)])
        
        keyboard = InlineKeyboardMarkup(buttons)
        
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
        
        await callback_query.message.edit_text(text)
    
    # ✅ HANDLE SCREENSHOT MESSAGES
    @bot.on_message(filters.private & (filters.photo | filters.document))
    async def handle_screenshot(client, message):
        """Handle payment screenshots"""
        # Check if it's likely a screenshot
        if message.photo or (message.document and message.document.mime_type and 'image' in message.document.mime_type):
            await message.reply_text(
                "✅ **Screenshot received!**\n\n"
                "Our admin will verify your payment and activate your premium within 24 hours.\n"
                "Thank you for choosing SK4FiLM! 🎬\n\n"
                "You will receive a confirmation message when activated.",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 BACK TO START", callback_data="back_to_start")]
                ])
            )
    
    logger.info("✅ Bot handlers setup complete - Ready to send files with access control!")
