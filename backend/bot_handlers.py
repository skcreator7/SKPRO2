"""
bot_handlers.py - Telegram Bot Handlers for SK4FiLM - FIXED VERSION
"""
import asyncio
import logging
import re
from datetime import datetime
from typing import Dict, Any, Optional

from pyrogram import Client, filters, idle
from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton, Message, CallbackQuery
from pyrogram.errors import FloodWait, BadRequest

logger = logging.getLogger(__name__)

# Import utility functions from app
from app import (
    normalize_title, extract_title_from_file, format_size, 
    detect_quality, is_video_file, safe_telegram_operation,
    safe_telegram_generator, index_single_file, auto_delete_file,
    Config
)

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
                
                # Verify token
                if bot_instance.verification_system is not None:
                    is_verified, verified_user_id, verify_message = await bot_instance.verification_system.verify_user_token(token)
                    
                    if is_verified and verified_user_id == user_id:
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
                    else:
                        await message.reply_text(
                            "❌ **Verification Failed**\n\n"
                            f"Error: {verify_message}\n\n"
                            "Please generate a new verification link."
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
        premium_details = {}
        if bot_instance.premium_system is not None:
            is_premium = await bot_instance.premium_system.is_premium_user(user_id)
            premium_details = await bot_instance.premium_system.get_subscription_details(user_id)
        
        if is_premium:
            welcome_text += f"🌟 **Premium Status:** {premium_details.get('tier_name', 'Premium')}\n"
            welcome_text += f"📅 **Days Remaining:** {premium_details.get('days_remaining', 0)}\n\n"
            welcome_text += "✅ **You have full access to all features!**\n\n"
            
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)],
                [InlineKeyboardButton("📊 PREMIUM STATUS", callback_data=f"premium_status_{user_id}")]
            ])
        elif Config.VERIFICATION_REQUIRED and bot_instance.verification_system is not None:
            # Check if user is verified
            is_verified, status = await bot_instance.verification_system.check_user_verified(user_id)
            
            if not is_verified:
                # Create verification link
                verification_data = await bot_instance.verification_system.create_verification_link(user_id)
                
                welcome_text += (
                    "🔒 **Verification Required**\n"
                    "Please complete verification to download files:\n\n"
                    f"🔗 **Verification Link:** {verification_data['short_url']}\n\n"
                    "Click the link above and then click 'Start' in the bot.\n"
                    "⏰ **Valid for 1 hour**\n\n"
                    "✨ **Or upgrade to Premium for instant access!**"
                )
                
                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔗 VERIFY NOW", url=verification_data['short_url'])],
                    [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")],
                    [InlineKeyboardButton("🔄 CHECK VERIFICATION", callback_data=f"check_verify_{user_id}")]
                ])
            else:
                welcome_text += "✅ **You are verified!**\nYou can download files from the website.\n\n"
                welcome_text += "✨ **Upgrade to Premium for more features!**"
                
                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)],
                    [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")]
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
        
        if bot_instance.verification_system is None:
            await callback_query.answer("Verification system not available", show_alert=True)
            return
        
        is_verified, message = await bot_instance.verification_system.check_user_verified(user_id)
        
        if is_verified:
            await callback_query.message.edit_text(
                "✅ **Verification Successful!**\n\n"
                "You can now download files from the website.\n\n"
                f"🌐 **Website:** {Config.WEBSITE_URL}",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)]
                ])
            )
        else:
            verification_data = await bot_instance.verification_system.create_verification_link(user_id)
            await callback_query.message.edit_text(
                "❌ **Not Verified Yet**\n\n"
                "Please complete the verification process:\n\n"
                f"🔗 **Verification Link:** {verification_data['short_url']}\n\n"
                "Click the link above and then click 'Start' in the bot.",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔗 VERIFY NOW", url=verification_data['short_url'])],
                    [InlineKeyboardButton("🔄 CHECK AGAIN", callback_data=f"check_verify_{user_id}")]
                ]),
                disable_web_page_preview=True
            )
    
    @bot.on_callback_query(filters.regex(r"^buy_premium$"))
    async def buy_premium_callback(client, callback_query):
        """Show premium plans"""
        if bot_instance.premium_system is None:
            await callback_query.answer("Premium system not available", show_alert=True)
            return
        
        user_id = callback_query.from_user.id
        plans = await bot_instance.premium_system.get_all_plans()
        
        text = "⭐ **SK4FiLM PREMIUM PLANS** ⭐\n\n"
        text += "Upgrade for better quality, more downloads and faster speeds!\n\n"
        
        keyboard = []
        for plan in plans:
            text += f"**{plan['name']}**\n"
            text += f"💰 **Price:** ₹{plan['price']}\n"
            text += f"⏰ **Duration:** {plan['duration_days']} days\n"
            text += "**Features:**\n"
            for feature in plan['features'][:3]:  # Show only 3 features
                text += f"• {feature}\n"
            text += "\n"
            
            keyboard.append([InlineKeyboardButton(
                f"{plan['name']} - ₹{plan['price']}", 
                callback_data=f"select_plan_{plan['tier']}"
            )])
        
        text += "\n**How to purchase:**\n1. Select a plan\n2. Pay using UPI\n3. Send screenshot\n4. Get activated!"
        
        keyboard.append([InlineKeyboardButton("🔙 BACK", callback_data="back_to_start")])
        
        await callback_query.message.edit_text(
            text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            disable_web_page_preview=True
        )
    
    @bot.on_callback_query(filters.regex(r"^select_plan_"))
    async def select_plan_callback(client, callback_query):
        """Select premium plan and show payment details"""
        if bot_instance.premium_system is None:
            await callback_query.answer("Premium system not available", show_alert=True)
            return
        
        tier_str = callback_query.data.split('_')[2]
        user_id = callback_query.from_user.id
        
        try:
            # Create payment request
            payment_data = await bot_instance.premium_system.create_payment_request(user_id, tier_str)
            
            text = f"💰 **Payment for {payment_data['tier_name']}**\n\n"
            text += f"**Amount:** ₹{payment_data['amount']}\n"
            text += f"**UPI ID:** `{payment_data['upi_id']}`\n\n"
            text += "**Payment Instructions:**\n"
            text += "1. Scan the QR code below OR\n"
            text += f"2. Send ₹{payment_data['amount']} to UPI ID: `{payment_data['upi_id']}`\n"
            text += "3. Take screenshot of payment\n"
            text += "4. Send screenshot to this bot\n\n"
            text += "⏰ **Payment valid for 1 hour**\n"
            text += "✅ **Admin will activate within 24 hours**"
            
            keyboard = [
                [InlineKeyboardButton("📸 SEND SCREENSHOT", callback_data=f"send_screenshot_{payment_data['payment_id']}")],
                [InlineKeyboardButton("🔙 BACK TO PLANS", callback_data="buy_premium")]
            ]
            
            await callback_query.message.delete()
            
            # Send message
            await callback_query.message.reply_text(
                text,
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            
        except Exception as e:
            logger.error(f"Select plan error: {e}")
            await callback_query.answer("Error creating payment request", show_alert=True)
    
    @bot.on_callback_query(filters.regex(r"^premium_status_"))
    async def premium_status_callback(client, callback_query):
        """Show premium status"""
        if bot_instance.premium_system is None:
            await callback_query.answer("Premium system not available", show_alert=True)
            return
        
        user_id = int(callback_query.data.split('_')[2])
        details = await bot_instance.premium_system.get_subscription_details(user_id)
        
        text = f"⭐ **PREMIUM STATUS**\n\n"
        text += f"**Plan:** {details['tier_name']}\n"
        text += f"**Status:** {'✅ Active' if details['is_active'] else '❌ Inactive'}\n"
        
        if details['expires_at']:
            expires = datetime.fromisoformat(details['expires_at']) if isinstance(details['expires_at'], str) else details['expires_at']
            text += f"**Expires:** {expires.strftime('%d %b %Y')}\n"
            text += f"**Days Remaining:** {details['days_remaining']}\n"
        
        text += f"\n**Features:**\n"
        for feature in details['features'][:5]:  # Show only 5 features
            text += f"• {feature}\n"
        
        text += f"\n**Downloads Today:** {details.get('daily_downloads', 0)}/{details['limits']['daily_downloads']}\n"
        text += f"**Total Downloads:** {details.get('total_downloads', 0)}\n"
        
        keyboard = [
            [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)],
            [InlineKeyboardButton("🔙 BACK", callback_data="back_to_start")]
        ]
        
        await callback_query.message.edit_text(
            text,
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    @bot.on_message(filters.command("premium") & filters.private)
    async def premium_command(client, message):
        """Premium command"""
        if bot_instance.premium_system is None:
            await message.reply_text("Premium system is not available.")
            return
        
        user_id = message.from_user.id
        is_premium = await bot_instance.premium_system.is_premium_user(user_id)
        
        if is_premium:
            details = await bot_instance.premium_system.get_subscription_details(user_id)
            text = f"⭐ **You are a Premium User!**\n\n"
            text += f"**Plan:** {details['tier_name']}\n"
            text += f"**Days Remaining:** {details['days_remaining']}\n\n"
            text += "✅ **You have full access to all features!**"
            
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)]
            ])
        else:
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
        if bot_instance.verification_system is None:
            await message.reply_text("Verification system is not available.")
            return
        
        user_id = message.from_user.id
        user_name = message.from_user.first_name or "User"
        
        # Check if premium user (bypass verification)
        if bot_instance.premium_system is not None:
            is_premium = await bot_instance.premium_system.is_premium_user(user_id)
            if is_premium:
                await message.reply_text(
                    f"✅ **Premium User Detected!**\n\n"
                    f"As a premium user, you don't need verification.\n"
                    f"You have full access to all features, {user_name}! 🎬"
                )
                return
        
        is_verified, status = await bot_instance.verification_system.check_user_verified(user_id)
        
        if is_verified:
            await message.reply_text(
                f"✅ **Already Verified, {user_name}!**\n\n"
                f"Your verification is active and valid for 6 hours.\n"
                "You can download files from the website now! 🎬"
            )
        else:
            verification_data = await bot_instance.verification_system.create_verification_link(user_id)
            await message.reply_text(
                f"🔗 **Verification Required, {user_name}**\n\n"
                "To download files, please complete the URL verification:\n\n"
                f"**Verification URL:** {verification_data['short_url']}\n\n"
                "⏰ **Valid for 1 hour**\n\n"
                "Click the link above and then click 'Start' in the bot.",
                disable_web_page_preview=True
            )
    
    @bot.on_message(filters.command("premiumuser") & filters.user(Config.ADMIN_IDS))
    async def premium_user_admin(client, message):
        """Admin command to activate premium for user"""
        if bot_instance.premium_system is None:
            await message.reply_text("Premium system is not available.")
            return
        
        try:
            parts = message.text.split()
            if len(parts) < 2:
                await message.reply_text(
                    "Usage: /premiumuser <user_id> [plan]\n\n"
                    "Plans: basic, premium, ultimate, lifetime\n"
                    "Example: /premiumuser 123456789 premium"
                )
                return
            
            user_id = int(parts[1])
            tier_str = parts[2] if len(parts) > 2 else "premium"
            
            # Activate premium
            subscription = await bot_instance.premium_system.activate_premium(
                admin_id=message.from_user.id,
                user_id=user_id,
                tier=tier_str
            )
            
            await message.reply_text(
                f"✅ **Premium Activated!**\n\n"
                f"**User:** {user_id}\n"
                f"**Plan:** {subscription['tier_name']}\n"
                f"**Expires:** {subscription['expires_at'].strftime('%d %b %Y')}\n"
                f"**Days:** {subscription['duration_days']}\n\n"
                f"User will receive a notification."
            )
            
        except Exception as e:
            await message.reply_text(f"❌ Error: {e}")
    
    @bot.on_message(filters.command("stats") & filters.user(Config.ADMIN_IDS))
    async def stats_command(client, message):
        """Admin stats command"""
        try:
            # Get MongoDB stats
            total_files = await bot_instance.db_manager.files_col.count_documents({}) if bot_instance.db_manager and bot_instance.db_manager.files_col else 0
            video_files = await bot_instance.db_manager.files_col.count_documents({'is_video_file': True}) if bot_instance.db_manager and bot_instance.db_manager.files_col else 0
            
            # Get verification stats
            verification_stats = {}
            if bot_instance.verification_system is not None:
                verification_stats = await bot_instance.verification_system.get_user_stats()
            
            # Get premium stats
            premium_stats = {}
            if bot_instance.premium_system is not None:
                premium_stats = await bot_instance.premium_system.get_admin_stats()
            
            # Get cache stats
            cache_stats = {}
            if bot_instance.cache_manager is not None:
                cache_stats = await bot_instance.cache_manager.get_stats_summary()
            
            text = "📊 **SK4FiLM STATISTICS**\n\n"
            text += f"📁 **Total Files:** {total_files}\n"
            text += f"🎥 **Video Files:** {video_files}\n"
            text += f"🔐 **Pending Verifications:** {verification_stats.get('pending_verifications', 0)}\n"
            text += f"✅ **Verified Users:** {verification_stats.get('verified_users', 0)}\n"
            text += f"⭐ **Premium Users:** {premium_stats.get('active_premium_users', 0)}\n"
            text += f"💰 **Total Revenue:** ₹{premium_stats.get('total_revenue', 0)}\n"
            text += f"🔧 **Redis Enabled:** {cache_stats.get('redis_enabled', False)}\n"
            text += f"📡 **Bot Status:** {'✅ Online' if bot_instance.bot_started else '⏳ Starting'}\n"
            text += f"👤 **User Session:** {'✅ Ready' if bot_instance.user_session_ready else '⏳ Pending'}\n\n"
            text += "⚡ **All systems operational!**"
            
            await message.reply_text(text)
            
        except Exception as e:
            await message.reply_text(f"❌ Error getting stats: {e}")
    
    @bot.on_message(filters.command("broadcast") & filters.user(Config.ADMIN_IDS))
    async def broadcast_command(client, message):
        """Broadcast to premium users"""
        if bot_instance.premium_system is None:
            await message.reply_text("Premium system is not available.")
            return
        
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
            
            result = await bot_instance.premium_system.broadcast_to_premium_users(broadcast_text)
            
            await message.reply_text(
                f"📢 **Broadcast Scheduled**\n\n"
                f"**Message:** {broadcast_text[:50]}...\n"
                f"**Users:** {result.get('user_count', 0)}\n"
                f"**Status:** {result.get('status', 'unknown')}\n\n"
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
                    
                    if bot_instance.premium_system is not None:
                        # Premium users bypass all checks
                        is_premium = await bot_instance.premium_system.is_premium_user(user_id)
                        if not is_premium:
                            # Check download limits for free users
                            can_download, message_text, details = await bot_instance.premium_system.can_user_download(user_id)
                    
                    if not can_download:
                        await message.reply_text(
                            f"❌ **Download Failed**\n\n"
                            f"{message_text}\n\n"
                            f"⭐ **Upgrade to Premium for unlimited downloads!**",
                            reply_markup=InlineKeyboardMarkup([
                                [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")]
                            ])
                        )
                        return
                    
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
                        await bot_instance.premium_system.record_download(user_id)
                    
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
        # This handler was missing a proper try/except block
        try:
            if not message.caption and not message.reply_to_message:
                return
            
            # Check if this is a payment screenshot
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
