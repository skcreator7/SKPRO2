"""
bot_handlers.py - Telegram Bot Handlers for SK4FiLM - Updated for Direct File Access
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
    """Setup bot commands and handlers with direct file access"""
    
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
                            "✨ **What you can do now:**\n"
                            "• Download files directly by pasting file details\n"
                            "• No more file links needed\n"
                            "• Direct access for 6 hours\n\n"
                            f"🌐 **Website:** {Config.WEBSITE_URL}\n"
                            f"⏰ **Verification valid for 6 hours**\n\n"
                            "📥 **How to download:**\n"
                            "1. Go to our website\n"
                            "2. Find your movie\n" 
                            "3. Copy file details\n"
                            "4. Paste here and get the file!",
                            reply_markup=InlineKeyboardMarkup([
                                [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)],
                                [InlineKeyboardButton("📥 START DOWNLOADING", url=Config.WEBSITE_URL)]
                            ])
                        )
                        
                        # Also send a sample format
                        await message.reply_text(
                            "📋 **Sample File Format:**\n\n"
                            "`channel_id_message_id_quality`\n\n"
                            "**Example:**\n"
                            "`-1001768249569_1234_1080p`\n\n"
                            "Just paste the format above to get the file!"
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
            welcome_text += f"🌟 **Premium User**\n✅ **You have full access to all features!**\n\n"
            welcome_text += "✨ **You can download files directly by pasting file details!**\n\n"
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)],
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
                    "✨ **You can download files directly!**\n\n"
                    "📥 **How to download:**\n"
                    "1. Go to our website\n"
                    "2. Find your movie\n"
                    "3. Copy file details\n"
                    "4. Paste here and get the file!\n\n"
                )
                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)],
                    [InlineKeyboardButton("📥 START DOWNLOADING", url=Config.WEBSITE_URL)]
                ])
            else:
                # User needs verification
                if Config.VERIFICATION_REQUIRED:
                    # Create verification link (will be shortened)
                    if bot_instance.verification_system:
                        verification_link = await bot_instance.verification_system.get_verification_link_for_user(user_id)
                        
                        welcome_text += (
                            "🔒 **One-Time Verification Required**\n\n"
                            "To download files directly, complete one-time verification:\n\n"
                            f"🔗 **Verification Link:** {verification_link['short_url']}\n\n"
                            "**Steps:**\n"
                            "1. Click the link above\n"
                            "2. Click 'Start' in the bot\n"
                            "3. You'll be verified for 6 hours\n"
                            "4. Then paste file details to download!\n\n"
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
                    welcome_text += "✨ **Start downloading movies now!**\n\n"
                    welcome_text += "📥 **How to download:**\n"
                    welcome_text += "1. Go to our website\n"
                    welcome_text += "2. Find your movie\n"
                    welcome_text += "3. Copy file details\n"
                    welcome_text += "4. Paste here and get the file!\n\n"
                    keyboard = InlineKeyboardMarkup([
                        [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)],
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
                    access_message = "🌟 **Premium User** - Direct file access available!"
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
                "📥 **You can now download files directly:**\n"
                "1. Go to our website\n"
                "2. Find your movie\n"
                "3. Copy file details\n"
                "4. Paste here and get the file!\n\n"
                "📋 **Format:** `channel_id_message_id_quality`\n"
                "**Example:** `-1001768249569_1234_1080p`",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)],
                    [InlineKeyboardButton("📥 START DOWNLOADING", url=Config.WEBSITE_URL)]
                ])
            )
        else:
            # User needs verification
            if bot_instance.verification_system:
                verification_link = await bot_instance.verification_system.get_verification_link_for_user(user_id)
                
                await callback_query.message.edit_text(
                    "🔒 **Verification Required**\n\n"
                    "To download files directly, you need to complete verification:\n\n"
                    f"🔗 **Verification Link:** {verification_link['short_url']}\n\n"
                    "**Steps:**\n"
                    "1. Click the link above\n"
                    "2. Click 'Start' in the bot\n"
                    "3. You'll be verified for 6 hours\n\n"
                    "⏰ **Link valid for 1 hour**",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("🔗 VERIFY NOW", url=verification_link['short_url'])],
                        [InlineKeyboardButton("🔄 CHECK AGAIN", callback_data=f"check_access_{user_id}")]
                    ]),
                    disable_web_page_preview=True
                )
    
    @bot.on_callback_query(filters.regex(r"^check_verify_"))
    async def check_verify_callback(client, callback_query):
        user_id = int(callback_query.data.split('_')[2])
        user_name = callback_query.from_user.first_name or "User"
        
        # Check verification status
        if bot_instance.verification_system:
            verification_link = await bot_instance.verification_system.get_verification_link_for_user(user_id)
            
            await callback_query.message.edit_text(
                "🔒 **Verification Required**\n\n"
                "To download files directly, complete verification:\n\n"
                f"🔗 **Verification Link:** {verification_link['short_url']}\n\n"
                "**Steps:**\n"
                "1. Click the link above\n"
                "2. Click 'Start' in the bot\n"
                "3. You'll be verified for 6 hours\n\n"
                "⏰ **Link valid for 1 hour**",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔗 VERIFY NOW", url=verification_link['short_url'])],
                    [InlineKeyboardButton("🔄 CHECK AGAIN", callback_data=f"check_verify_{user_id}")]
                ]),
                disable_web_page_preview=True
            )
    
    @bot.on_callback_query(filters.regex(r"^buy_premium$"))
    async def buy_premium_callback(client, callback_query):
        """Show premium plans"""
        user_id = callback_query.from_user.id
        
        text = "⭐ **SK4FiLM PREMIUM PLANS** ⭐\n\n"
        text += "Upgrade for instant access, no verification needed!\n\n"
        
        plans = [
            {"tier": "basic", "name": "Basic Plan", "price": 99, "duration_days": 30, 
             "features": ["Direct File Access", "1080p Quality", "10 Daily Downloads", "Priority Support"]},
            {"tier": "premium", "name": "Premium Plan", "price": 199, "duration_days": 30,
             "features": ["Direct File Access", "4K Quality", "Unlimited Downloads", "Priority Support", "No Ads"]}
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
        
        text += "**Benefits of Premium:**\n"
        text += "✅ No verification required\n"
        text += "✅ Instant file access\n"
        text += "✅ Higher quality downloads\n"
        text += "✅ Faster speeds\n\n"
        
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
        text += "✅ **Admin will activate within 24 hours**\n\n"
        text += "**Once activated:**\n"
        text += "✅ No verification required\n"
        text += "✅ Direct file access\n"
        text += "✅ Instant downloads"
        
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
        text += "• Basic access\n"
        text += "• Verification required\n\n"
        text += "**Upgrade to Premium for:**\n"
        text += "✅ No verification needed\n"
        text += "✅ Direct file access\n"
        text += "✅ Instant downloads"
        
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
        text += "Get **direct file access** without verification!\n\n"
        text += "**Benefits:**\n"
        text += "✅ No verification required\n"
        text += "✅ Direct file downloads\n"
        text += "✅ Higher quality (1080p/4K)\n"
        text += "✅ More daily downloads\n"
        text += "✅ Faster download speeds\n"
        text += "✅ Priority support\n\n"
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
        
        # Check if user is already verified
        is_verified = False
        if bot_instance.verification_system:
            is_verified, verify_msg = await bot_instance.verification_system.check_user_verified(user_id)
        
        if is_verified:
            text += "🎉 **You are already verified!**\n\n"
            text += "✨ **You can download files directly:**\n"
            text += "1. Go to our website\n"
            text += "2. Find your movie\n"
            text += "3. Copy file details\n"
            text += "4. Paste here and get the file!\n\n"
            text += f"🌐 **Website:** {Config.WEBSITE_URL}\n\n"
            text += f"⏰ {verify_msg}"
            
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)],
                [InlineKeyboardButton("📥 START DOWNLOADING", url=Config.WEBSITE_URL)]
            ])
        else:
            if Config.VERIFICATION_REQUIRED:
                # Create verification link
                if bot_instance.verification_system:
                    verification_link = await bot_instance.verification_system.get_verification_link_for_user(user_id)
                    
                    text += "🔒 **One-Time Verification Required**\n\n"
                    text += "To download files directly, complete verification:\n\n"
                    text += f"🔗 **Verification Link:** {verification_link['short_url']}\n\n"
                    text += "**Steps:**\n"
                    text += "1. Click the link above\n"
                    text += "2. Click 'Start' in the bot\n"
                    text += "3. You'll be verified for 6 hours\n\n"
                    text += "⏰ **Link valid for 1 hour**\n\n"
                    text += "⭐ **Or upgrade to Premium for instant access!**"
                    
                    keyboard = InlineKeyboardMarkup([
                        [InlineKeyboardButton("🔗 VERIFY NOW", url=verification_link['short_url'])],
                        [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")]
                    ])
            else:
                text += "✨ **No verification required!**\n\n"
                text += "You can download files directly from our website:\n"
                text += f"🌐 **Website:** {Config.WEBSITE_URL}\n\n"
                text += "⭐ **Upgrade to Premium for better features!**"
                
                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)],
                    [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")]
                ])
        
        await message.reply_text(text, reply_markup=keyboard, disable_web_page_preview=True)
    
    @bot.on_message(filters.command("status") & filters.private)
    async def status_command(client, message):
        """Check user status"""
        user_id = message.from_user.id
        user_name = message.from_user.first_name or "User"
        
        text = f"📊 **Status for {user_name}**\n\n"
        
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
            text += "✅ **File Access:** Direct (no verification needed)\n"
            text += "✅ **Download Type:** Instant\n"
            text += "✅ **Status:** Premium User\n\n"
            text += "✨ **You can download files directly!**"
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
                text += "✅ **File Access:** Direct\n"
                text += "✅ **Download Type:** Verified User\n\n"
                text += "✨ **You can download files directly!**"
            else:
                text += "❌ **Verification:** Not active\n"
                text += "❌ **File Access:** Restricted\n"
                text += "🔒 **Download Type:** Verification required\n\n"
                text += "⚠️ **Complete verification to download files directly!**"
        
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
            
            # Notify user
            try:
                await client.send_message(
                    user_id,
                    f"🎉 **Congratulations!**\n\n"
                    f"Your premium subscription has been activated!\n\n"
                    f"**Plan:** {tier_name}\n"
                    f"**Duration:** 30 days\n\n"
                    "✨ **Benefits:**\n"
                    "✅ No verification needed\n"
                    "✅ Direct file access\n"
                    "✅ Instant downloads\n\n"
                    "📥 **Start downloading files directly now!**"
                )
            except Exception as e:
                logger.error(f"Failed to notify user {user_id}: {e}")
            
        except Exception as e:
            await message.reply_text(f"❌ Error: {e}")
    
    @bot.on_message(filters.command("stats") & filters.user(Config.ADMIN_IDS))
    async def stats_command(client, message):
        """Admin stats command"""
        try:
            text = "📊 **SK4FiLM STATISTICS**\n\n"
            text += f"📡 **Bot Status:** {'✅ Online' if bot_instance.bot_started else '⏳ Starting'}\n"
            text += f"👤 **User Session:** {'✅ Ready' if bot_instance.user_session_ready else '⏳ Pending'}\n"
            
            # Verification stats
            if bot_instance.verification_system:
                try:
                    stats = await bot_instance.verification_system.get_user_stats()
                    text += f"✅ **Verified Users:** {stats.get('active_verified_users', 0)}\n"
                    text += f"⏳ **Pending Verifications:** {stats.get('pending_verifications', 0)}\n"
                except:
                    text += f"✅ **Verified Users:** 0\n"
            
            # Premium stats
            if bot_instance.premium_system:
                try:
                    stats = await bot_instance.premium_system.get_admin_stats()
                    text += f"⭐ **Premium Users:** {stats.get('active_premium_users', 0)}\n"
                    text += f"💰 **Total Revenue:** ₹{stats.get('total_revenue', 0)}\n"
                except:
                    text += f"⭐ **Premium Users:** 0\n"
            
            text += f"🔧 **Redis Enabled:** {bot_instance.cache_manager.redis_enabled if bot_instance.cache_manager else False}\n\n"
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
    
    @bot.on_message(filters.text & filters.private & ~filters.command(['start', 'stats', 'premium', 'verify', 'status', 'index', 'broadcast', 'premiumuser']))
    async def direct_file_access_handler(client, message):
        """Handle direct file access for verified/premium users"""
        user_id = message.from_user.id
        user_name = message.from_user.first_name or "User"
        
        # Check if message is in file format: channel_id_message_id_quality
        if message.text and '_' in message.text:
            try:
                parts = message.text.strip().split('_')
                if len(parts) >= 2 and parts[0].isdigit() or (parts[0].startswith('-') and parts[0][1:].isdigit()):
                    channel_id = int(parts[0])
                    message_id = int(parts[1])
                    quality = parts[2] if len(parts) > 2 else "HD"
                    
                    logger.info(f"File request from user {user_id}: {channel_id}_{message_id}_{quality}")
                    
                    # Check user access
                    has_access = False
                    access_message = ""
                    
                    # Check premium first
                    if bot_instance.premium_system:
                        try:
                            is_premium = await bot_instance.premium_system.is_premium_user(user_id)
                            if is_premium:
                                has_access = True
                                access_message = "Premium user - Direct access"
                        except Exception as e:
                            logger.error(f"Premium check error: {e}")
                    
                    # Check verification if not premium
                    if not has_access and bot_instance.verification_system:
                        is_verified, verify_msg = await bot_instance.verification_system.check_user_verified(user_id)
                        if is_verified:
                            has_access = True
                            access_message = f"Verified user - {verify_msg}"
                    
                    if not has_access:
                        # User needs verification
                        if bot_instance.verification_system:
                            verification_link = await bot_instance.verification_system.get_verification_link_for_user(user_id)
                            
                            await message.reply_text(
                                f"🔒 **Verification Required, {user_name}!**\n\n"
                                f"To download files directly, you need to complete one-time verification.\n\n"
                                f"🔗 **Verification Link:** {verification_link['short_url']}\n\n"
                                "**Steps:**\n"
                                "1. Click the link above\n"
                                "2. Click 'Start' in the bot\n"
                                "3. You'll be verified for 6 hours\n"
                                "4. Then paste file details to download!\n\n"
                                "⏰ **Link valid for 1 hour**\n\n"
                                "⭐ **Or upgrade to Premium for instant access!**",
                                reply_markup=InlineKeyboardMarkup([
                                    [InlineKeyboardButton("🔗 VERIFY NOW", url=verification_link['short_url'])],
                                    [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")]
                                ]),
                                disable_web_page_preview=True
                            )
                            return
                        else:
                            await message.reply_text(
                                "❌ **Access Denied**\n\n"
                                "Verification system not available. Please contact support."
                            )
                            return
                    
                    # User has access - proceed with file download
                    processing_msg = await message.reply_text(
                        f"⏳ **Preparing your file...**\n\n"
                        f"👤 **User:** {user_name}\n"
                        f"📦 **Quality:** {quality}\n"
                        f"✅ **Access:** {access_message}\n\n"
                        f"Fetching file from channel..."
                    )
                    
                    # Get file from channel
                    file_message = await safe_telegram_operation(
                        client.get_messages,
                        channel_id, 
                        message_id
                    )
                    
                    if not file_message or (not file_message.document and not file_message.video):
                        await processing_msg.edit_text(
                            "❌ **File not found**\n\n"
                            "The file may have been deleted or moved.\n"
                            "Please check the file details and try again."
                        )
                        return
                    
                    # Determine file type and size
                    file_type = "document" if file_message.document else "video"
                    file_size = file_message.document.file_size if file_message.document else (file_message.video.file_size if file_message.video else 0)
                    file_name = file_message.document.file_name if file_message.document else (file_message.video.file_name if file_message.video else "Unknown")
                    
                    # Send file to user
                    try:
                        if file_message.document:
                            sent = await safe_telegram_operation(
                                client.send_document,
                                user_id, 
                                file_message.document.file_id,
                                caption=(
                                    f"✅ **File Downloaded Successfully!**\n\n"
                                    f"📁 **File:** {file_name}\n"
                                    f"📦 **Quality:** {quality}\n"
                                    f"📊 **Size:** {format_size(file_size)}\n"
                                    f"⏰ **Auto-delete in:** {Config.AUTO_DELETE_TIME//60} minutes\n\n"
                                    f"♻ **Please forward to saved messages**\n"
                                    f"⭐ **Consider upgrading to Premium for better features!**\n\n"
                                    f"@SK4FiLM 🍿"
                                )
                            )
                        else:
                            sent = await safe_telegram_operation(
                                client.send_video,
                                user_id, 
                                file_message.video.file_id,
                                caption=(
                                    f"✅ **File Downloaded Successfully!**\n\n"
                                    f"🎬 **Video:** {file_name}\n"
                                    f"📦 **Quality:** {quality}\n" 
                                    f"📊 **Size:** {format_size(file_size)}\n"
                                    f"⏰ **Auto-delete in:** {Config.AUTO_DELETE_TIME//60} minutes\n\n"
                                    f"♻ **Please forward to saved messages**\n"
                                    f"⭐ **Consider upgrading to Premium for better features!**\n\n"
                                    f"@SK4FiLM 🍿"
                                )
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
                        
                        logger.info(f"✅ File sent to user {user_id}: {file_type}, {quality} quality, {format_size(file_size)}")
                        
                        # Send success message
                        success_text = (
                            f"🎉 **Download Complete, {user_name}!**\n\n"
                            f"✅ **File sent successfully!**\n"
                            f"📁 **Type:** {file_type}\n"
                            f"📦 **Quality:** {quality}\n"
                            f"📊 **Size:** {format_size(file_size)}\n"
                            f"⏰ **Auto-delete in:** {Config.AUTO_DELETE_TIME//60} minutes\n\n"
                        )
                        
                        if has_access and "Premium" in access_message:
                            success_text += "🌟 **Thank you for being a Premium user!**\n\n"
                        else:
                            success_text += (
                                "⏰ **Your verification expires in 6 hours**\n"
                                "📥 **You can download more files during this time!**\n\n"
                            )
                        
                        success_text += "♻ **Please forward the file to your saved messages**\n"
                        success_text += "⭐ **Consider upgrading to Premium for better features!**"
                        
                        keyboard = []
                        if not has_access or "Premium" not in access_message:
                            keyboard.append([InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")])
                        keyboard.append([InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)])
                        
                        await message.reply_text(
                            success_text,
                            reply_markup=InlineKeyboardMarkup(keyboard)
                        )
                        
                        return
                        
                    except Exception as e:
                        logger.error(f"Error sending file to user {user_id}: {e}")
                        await processing_msg.edit_text(
                            "❌ **Error sending file**\n\n"
                            "There was an error sending the file. Please try again later.\n"
                            f"Error: {str(e)[:100]}"
                        )
                        return
                        
            except ValueError:
                # Not a valid file format, treat as regular message
                pass
            except Exception as e:
                logger.error(f"File download error: {e}")
                try:
                    await processing_msg.edit_text(
                        "❌ **Error processing request**\n\n"
                        "There was an error processing your file request. Please try again."
                    )
                except:
                    pass
                return
        
        # If not a file format, show help
        await message.reply_text(
            "🎬 **SK4FiLM Direct File Download**\n\n"
            "📥 **How to download files:**\n"
            "1. Visit our website\n"
            "2. Find your movie\n"
            "3. Copy file details in format:\n"
            "   `channel_id_message_id_quality`\n\n"
            "**Example:**\n"
            "`-1001768249569_1234_1080p`\n\n"
            "4. Paste the format here\n"
            "5. Get your file directly!\n\n"
            f"🌐 **Website:** {Config.WEBSITE_URL}\n\n"
            "🔒 **Note:** First-time users need one-time verification for direct access!"
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
        
        # Check access status
        has_access = False
        if bot_instance.premium_system:
            try:
                is_premium = await bot_instance.premium_system.is_premium_user(user_id)
                if is_premium:
                    has_access = True
                    welcome_text += "🌟 **Premium User** - Direct file access available!\n\n"
            except:
                pass
        
        if not has_access and bot_instance.verification_system:
            is_verified, verify_msg = await bot_instance.verification_system.check_user_verified(user_id)
            if is_verified:
                has_access = True
                welcome_text += f"✅ **Verified User** - {verify_msg}\n\n"
        
        if has_access:
            welcome_text += "📥 **You can download files directly by pasting file details!**\n\n"
            welcome_text += "**Format:** `channel_id_message_id_quality`\n"
            welcome_text += "**Example:** `-1001768249569_1234_1080p`"
            
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)],
                [InlineKeyboardButton("📥 START DOWNLOADING", url=Config.WEBSITE_URL)]
            ])
        else:
            welcome_text += "🔒 **Complete one-time verification for direct file access!**\n\n"
            welcome_text += "⭐ **Or upgrade to Premium for instant access!**"
            
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
            "Please send the payment screenshot to this chat.\n\n"
            "**Make sure:**\n"
            "1. Payment amount is visible\n"
            "2. UPI ID is visible\n"
            "3. Transaction ID is visible\n\n"
            "⚠️ **Send the screenshot now...**\n\n"
            "✅ **Once verified, you'll get instant premium access!**"
        )
    
    @bot.on_message(filters.photo & filters.private)
    async def handle_screenshot(client, message):
        """Handle payment screenshots"""
        try:
            user_id = message.from_user.id
            user_name = message.from_user.first_name or "User"
            
            await message.reply_text(
                f"✅ **Screenshot Received, {user_name}!**\n\n"
                "Your payment screenshot has been received.\n\n"
                "**What happens next:**\n"
                "1. Our admin will verify your payment\n"
                "2. Your premium subscription will be activated\n"
                "3. You'll receive a confirmation message\n\n"
                "⏰ **Processing time:** Within 24 hours\n\n"
                "✨ **Once activated, you'll get:**\n"
                "✅ No verification needed\n"
                "✅ Direct file access\n"
                "✅ Instant downloads\n\n"
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
