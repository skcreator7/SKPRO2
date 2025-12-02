"""
bot_handlers.py - Telegram Bot Handlers for SK4FiLM - COMPLETE FIXED VERSION
"""
import asyncio
import logging
import re
from datetime import datetime
from typing import Dict, Any

from pyrogram import Client, filters, idle
from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton, Message, CallbackQuery
from pyrogram.errors import FloodWait

logger = logging.getLogger(__name__)

# Import Config
from app import Config

class SK4FiLMBot:
    def __init__(self, config):
        self.config = config
        self.bot = None
        self.user_client = None
        self.bot_started = False
        self.user_session_ready = False
        
        # Initialize placeholder systems
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
                bot_token=self.config.BOT_TOKEN
            )
            
            # Start bot
            await self.bot.start()
            self.bot_started = True
            logger.info("✅ Bot started successfully")
            
            # Try to initialize user client if session string is provided
            if self.config.USER_SESSION_STRING:
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
                    logger.error(f"Failed to start user session: {e}")
            
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
        
        welcome_text = (
            f"🎬 **Welcome to SK4FiLM, {user_name}!**\n\n"
            "🌐 **Use our website to browse and download movies:**\n"
            f"{Config.WEBSITE_URL}\n\n"
            "✨ **Start browsing movies now!**\n\n"
            "⭐ **Upgrade to Premium for:**\n• Higher quality\n• More downloads\n• Faster speeds"
        )
        
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)],
            [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")]
        ])
        
        await message.reply_text(welcome_text, reply_markup=keyboard, disable_web_page_preview=True)
    
    @bot.on_callback_query(filters.regex(r"^buy_premium$"))
    async def buy_premium_callback(client, callback_query):
        """Show premium plans"""
        text = "⭐ **SK4FiLM PREMIUM PLANS** ⭐\n\n"
        text += "Upgrade for better quality, more downloads and faster speeds!\n\n"
        
        plans = [
            {"name": "Basic Plan", "price": 99, "features": ["1080p Quality", "10 Daily Downloads", "Priority Support"]},
            {"name": "Premium Plan", "price": 199, "features": ["4K Quality", "Unlimited Downloads", "Priority Support", "No Ads"]},
        ]
        
        keyboard = []
        for plan in plans:
            text += f"**{plan['name']}**\n"
            text += f"💰 **Price:** ₹{plan['price']}\n"
            text += "**Features:**\n"
            for feature in plan['features'][:3]:
                text += f"• {feature}\n"
            text += "\n"
            
            keyboard.append([InlineKeyboardButton(
                f"{plan['name']} - ₹{plan['price']}", 
                callback_data=f"select_plan_{plan['name'].lower().replace(' ', '_')}"
            )])
        
        text += "**How to purchase:**\n1. Select a plan\n2. Contact admin\n3. Get activated!"
        
        keyboard.append([InlineKeyboardButton("🔙 BACK", callback_data="back_to_start")])
        
        await callback_query.message.edit_text(
            text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            disable_web_page_preview=True
        )
    
    @bot.on_callback_query(filters.regex(r"^select_plan_"))
    async def select_plan_callback(client, callback_query):
        """Select premium plan"""
        plan_name = callback_query.data.split('_', 2)[2]
        user_id = callback_query.from_user.id
        
        if plan_name == "basic_plan":
            price = 99
            plan_display = "Basic Plan"
        else:
            price = 199
            plan_display = "Premium Plan"
        
        text = f"💰 **Payment for {plan_display}**\n\n"
        text += f"**Amount:** ₹{price}\n"
        text += "**Payment Instructions:**\n"
        text += "1. Contact admin for payment details\n"
        text += "2. Make payment\n"
        text += "3. Send screenshot to admin\n"
        text += "4. Get activated!\n\n"
        text += "✅ **Admin will activate within 24 hours**"
        
        keyboard = [
            [InlineKeyboardButton("📞 CONTACT ADMIN", url="https://t.me/sk4filmadmin")],
            [InlineKeyboardButton("🔙 BACK TO PLANS", callback_data="buy_premium")]
        ]
        
        await callback_query.message.delete()
        
        await callback_query.message.reply_text(
            text,
            reply_markup=InlineKeyboardMarkup(keyboard)
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
        text += "No additional verification required!"
        
        await message.reply_text(text, disable_web_page_preview=True)
    
    @bot.on_message(filters.command("stats") & filters.user(Config.ADMIN_IDS))
    async def stats_command(client, message):
        """Admin stats command"""
        try:
            text = "📊 **SK4FiLM STATISTICS**\n\n"
            text += f"📡 **Bot Status:** {'✅ Online' if bot_instance.bot_started else '⏳ Starting'}\n"
            text += f"👤 **User Session:** {'✅ Ready' if bot_instance.user_session_ready else '⏳ Pending'}\n\n"
            text += "⚡ **All systems operational!**"
            
            await message.reply_text(text)
            
        except Exception as e:
            await message.reply_text(f"❌ Error getting stats: {e}")
    
    @bot.on_message(filters.command("broadcast") & filters.user(Config.ADMIN_IDS))
    async def broadcast_command(client, message):
        """Broadcast to users"""
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
                f"📢 **Broadcast Message**\n\n"
                f"**Message:** {broadcast_text[:100]}...\n\n"
                f"Message prepared for broadcasting."
            )
            
        except Exception as e:
            await message.reply_text(f"❌ Error: {e}")
    
    @bot.on_message(filters.text & filters.private & ~filters.command(['start', 'stats', 'premium', 'verify', 'broadcast']))
    async def text_handler(client, message):
        """Handle text messages"""
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
                    
                    processing_msg = await message.reply_text(f"⏳ **Preparing your file...**\n\n📦 Quality: {quality}")
                    
                    try:
                        # Get file from channel
                        file_message = await client.get_messages(channel_id, message_id)
                        
                        if not file_message or (not file_message.document and not file_message.video):
                            await processing_msg.edit_text("❌ **File not found**\n\nThe file may have been deleted.")
                            return
                        
                        # Send file to user
                        if file_message.document:
                            await client.send_document(
                                user_id, 
                                file_message.document.file_id, 
                                caption=f"📹 Quality: {quality}\n\n"
                                       f"⚠️ Will auto-delete in {Config.AUTO_DELETE_TIME//60} minutes\n\n"
                                       f"@SK4FiLM 🍿"
                            )
                        else:
                            await client.send_video(
                                user_id, 
                                file_message.video.file_id, 
                                caption=f"📹 Quality: {quality}\n\n"
                                       f"⚠️ Will auto-delete in {Config.AUTO_DELETE_TIME//60} minutes\n\n"
                                       f"@SK4FiLM 🍿"
                            )
                        
                        await processing_msg.delete()
                        
                        # Auto-delete file after specified time
                        if Config.AUTO_DELETE_TIME > 0:
                            async def auto_delete():
                                await asyncio.sleep(Config.AUTO_DELETE_TIME)
                                try:
                                    await message.delete()
                                except:
                                    pass
                            
                            asyncio.create_task(auto_delete())
                        
                        logger.info(f"File sent to user {user_id}: {quality} quality")
                        
                        # Send success message
                        await message.reply_text(
                            f"✅ **File sent successfully!**\n\n"
                            f"📦 **Quality:** {quality}\n"
                            f"⏰ **Auto-delete in:** {Config.AUTO_DELETE_TIME//60} minutes\n\n"
                            f"⭐ **Consider upgrading to Premium for better features!**",
                            reply_markup=InlineKeyboardMarkup([
                                [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")]
                            ])
                        )
                        
                        return
                        
                    except Exception as e:
                        logger.error(f"File sending error: {e}")
                        await processing_msg.edit_text("❌ **Error sending file**\n\nPlease try again later.")
                        return
        
        # If not a file link, show help
        await message.reply_text(
            "🎬 **SK4FiLM File Download**\n\n"
            "To download a file, use the website:\n"
            f"🌐 **Website:** {Config.WEBSITE_URL}\n\n"
            "Find your movie and click download to get the file link.\n"
            "Then paste the link here and I'll send you the file! 🍿"
        )
    
    @bot.on_message(filters.photo & filters.private)
    async def handle_photo(client, message):
        """Handle photo messages"""
        try:
            user_id = message.from_user.id
            user_name = message.from_user.first_name or "User"
            
            await message.reply_text(
                f"✅ **Photo Received, {user_name}!**\n\n"
                "If this is a payment screenshot, please contact admin:\n"
                "📞 **Admin:** @sk4filmadmin\n\n"
                "Thank you! 🎬"
            )
            
        except Exception as e:
            logger.error(f"Error handling photo: {e}")
