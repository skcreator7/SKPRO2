"""
bot_handlers.py - Telegram Bot Handlers for SK4FiLM
COMPLETE UPDATED - Access Denied with 2+1 Button Layout
"""
import asyncio
import logging
import secrets
import re
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from collections import defaultdict

try:
    from pyrogram import Client, filters
    from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton, Message, CallbackQuery
    from pyrogram.errors import FloodWait, BadRequest, MessageDeleteForbidden
    PYROGRAM_AVAILABLE = True
except ImportError:
    PYROGRAM_AVAILABLE = False

from utils import normalize_title, extract_title_smart, format_size, detect_quality, is_video_file, format_post

logger = logging.getLogger(__name__)

class SK4FiLMBot:
    def __init__(self, config, db_manager=None):
        self.config = config
        self.db_manager = db_manager
        self.bot = None
        self.user_client = None
        self.bot_started = False
        self.user_session_ready = False
        self.auto_delete_tasks = {}
        self.auto_delete_messages = {}
        self.user_request_times = defaultdict(list)
        self.processing_requests = {}
        self.user_download_history = defaultdict(list)
        
        self.verification_system = None
        self.premium_system = None
        
        try:
            from premium import PremiumSystem, PremiumTier
            self.premium_system = PremiumSystem(config, db_manager)
            logger.info("✅ Premium system loaded")
        except Exception as e:
            logger.warning(f"⚠️ Premium system not loaded: {e}")
        
        try:
            from verification import VerificationSystem
            self.verification_system = VerificationSystem(config, db_manager)
            logger.info("✅ Verification system loaded")
        except Exception as e:
            logger.warning(f"⚠️ Verification system not loaded: {e}")
    
    async def initialize(self):
        """Initialize bot"""
        try:
            logger.info("🚀 Initializing SK4FiLM Bot...")
            
            self.bot = Client(
                "sk4film_bot_main",
                api_id=self.config.API_ID,
                api_hash=self.config.API_HASH,
                bot_token=self.config.BOT_TOKEN,
                workers=10
            )
            
            await self.bot.start()
            self.bot_started = True
            logger.info("✅ Bot started")
            
            await setup_bot_handlers(self.bot, self)
            
            if self.premium_system:
                asyncio.create_task(self.premium_system.start_cleanup_task())
            if self.verification_system:
                asyncio.create_task(self.verification_system.start_cleanup_task())
            
            return True
        except Exception as e:
            logger.error(f"Bot init error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def shutdown(self):
        """Shutdown"""
        if self.bot and self.bot_started:
            await self.bot.stop()
        if self.premium_system:
            await self.premium_system.stop_cleanup_task()
        if self.verification_system:
            await self.verification_system.stop_cleanup_task()


async def setup_bot_handlers(bot: Client, bot_instance):
    """Setup all bot handlers"""
    config = bot_instance.config
    
    # ============================================================================
    # ACCESS CHECK HELPER
    # ============================================================================
    async def check_access_and_send_denial(user_id: int, chat_id: int) -> bool:
        """Check access and send denial message with 2+1 button layout"""
        
        # Check 1: Admin
        if user_id in config.ADMIN_IDS:
            return True
        
        # Check 2: Premium
        is_premium = False
        if bot_instance.premium_system:
            try:
                is_premium = await bot_instance.premium_system.is_premium_user(user_id)
            except:
                is_premium = False
        
        if is_premium:
            return True
        
        # Check 3: Verified
        is_verified = False
        if bot_instance.verification_system:
            try:
                is_verified, _ = await bot_instance.verification_system.check_user_verified(
                    user_id, bot_instance.premium_system
                )
            except:
                is_verified = False
        
        if is_verified:
            return True
        
        # ❌ ACCESS DENIED
        logger.warning(f"❌ User {user_id} - Access denied")
        
        # Create verification link
        verification_link = None
        if bot_instance.verification_system:
            try:
                verification_data = await bot_instance.verification_system.create_verification_link(user_id)
                verification_link = verification_data.get('short_url')
            except:
                pass
        
        # ✅ BUTTONS - Row 1: 2 buttons, Row 2: 1 button
        buttons = []
        
        # Row 1: Two buttons side by side
        row1 = []
        if verification_link:
            row1.append(InlineKeyboardButton("🔗 VERIFY NOW (FREE)", url=verification_link))
        else:
            row1.append(InlineKeyboardButton("🔗 VERIFY NOW (FREE)", callback_data="verify_free"))
        
        row1.append(InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium"))
        buttons.append(row1)
        
        # Row 2: One button full width
        buttons.append([
            InlineKeyboardButton("🎁 REFER & GET PREMIUM", callback_data="referral_info")
        ])
        
        denial_text = (
            "🔒 **ACCESS DENIED** 🔒\n\n"
            "❌ You need to verify or purchase premium to download files.\n\n"
            "━━━━━━━━━━━━━━━━━━\n\n"
            "🆓 **FREE USER:**\n"
            "• Verify every 6 hours\n\n"
            "⭐ **PREMIUM USER:**\n"
            "• No verification needed\n"
            "• Unlimited downloads\n\n"
            "🎁 **REFER & EARN:**\n"
            "• Refer friends\n"
            "• Get FREE premium\n\n"
            "━━━━━━━━━━━━━━━━━━\n\n"
            "👇 **Choose your option:**"
        )
        
        try:
            await bot.send_message(
                chat_id,
                denial_text,
                reply_markup=InlineKeyboardMarkup(buttons),
                disable_web_page_preview=True
            )
        except Exception as e:
            logger.error(f"Denial message error: {e}")
        
        return False
    
    # ============================================================================
    # START COMMAND
    # ============================================================================
    @bot.on_message(filters.command("start"))
    async def start_command(client, message):
        try:
            user_name = message.from_user.first_name or "User"
            user_id = message.from_user.id
            
            # File request check
            if len(message.command) > 1:
                file_param = message.command[1]
                
                # Referral code check
                if file_param.startswith('ref_'):
                    code = file_param.replace('ref_', '')
                    
                    if bot_instance.premium_system:
                        referral_data = await bot_instance.premium_system.validate_referral_code(code, user_id)
                        
                        if referral_data.get('valid'):
                            text = (
                                f"🎁 **REFERRAL CODE APPLIED!**\n\n"
                                f"👋 Welcome {user_name}!\n"
                                f"✅ Code: `{code}`\n"
                                f"✅ You'll get +{referral_data.get('extra_days', 3)} days FREE on premium purchase!\n\n"
                                f"💎 **Buy Premium Now:**\n"
                                f"Use /buy command or click below!"
                            )
                            keyboard = InlineKeyboardMarkup([
                                [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")],
                                [InlineKeyboardButton("🎁 REFER & EARN", callback_data="referral_info")]
                            ])
                            await message.reply_text(text, reply_markup=keyboard)
                        else:
                            await message.reply_text(
                                f"❌ {referral_data.get('error', 'Invalid referral code')}"
                            )
                    return
                
                # Verification token check
                if file_param.startswith('verify_'):
                    token = file_param.replace('verify_', '')
                    
                    if bot_instance.verification_system:
                        success, verified_user_id, msg = await bot_instance.verification_system.verify_user_token(token)
                        
                        if success:
                            await message.reply_text(
                                "✅ **VERIFICATION SUCCESSFUL!**\n\n"
                                "🎉 You now have access for 6 hours!\n"
                                "🌐 Go back to website and download your file.\n\n"
                                "⭐ Want permanent access?\n"
                                "Use /buy to get premium!"
                            )
                        else:
                            await message.reply_text(f"❌ {msg}")
                    return
                
                # File download request
                parts = file_param.split('_')
                if len(parts) >= 2:
                    try:
                        channel_id = int(parts[0])
                        msg_id = int(parts[1])
                        quality = parts[2] if len(parts) > 2 else "480p"
                        
                        # ✅ ACCESS CHECK
                        has_access = await check_access_and_send_denial(user_id, message.chat.id)
                        
                        if not has_access:
                            return
                        
                        # Send processing
                        processing = await message.reply_text("⏳ **Sending your file...**")
                        
                        # Get file
                        file_msg = await client.get_messages(channel_id, msg_id)
                        
                        if file_msg and (file_msg.document or file_msg.video):
                            caption = (
                                f"📹 **Quality:** {quality}\n"
                                f"⏰ **Auto-delete:** {config.AUTO_DELETE_TIME} minutes\n\n"
                                f"🎬 @SK4FiLM"
                            )
                            
                            buttons = InlineKeyboardMarkup([
                                [InlineKeyboardButton("🌐 WEBSITE", url=config.WEBSITE_URL)],
                                [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")]
                            ])
                            
                            if file_msg.document:
                                await client.send_document(
                                    user_id,
                                    file_msg.document.file_id,
                                    caption=caption,
                                    reply_markup=buttons
                                )
                            else:
                                await client.send_video(
                                    user_id,
                                    file_msg.video.file_id,
                                    caption=caption,
                                    reply_markup=buttons
                                )
                            
                            await processing.delete()
                            logger.info(f"✅ File sent to user {user_id}")
                        else:
                            await processing.edit_text("❌ File not found!")
                        return
                        
                    except Exception as e:
                        logger.error(f"File request error: {e}")
                        await message.reply_text(f"❌ Error: {e}")
                        return
            
            # Welcome message
            text = (
                f"🎬 **SK4FiLM**\n\n"
                f"👋 Welcome {user_name}!\n\n"
                f"**Commands:**\n"
                f"/buy - Premium plans\n"
                f"/plans - View plans\n"
                f"/mypremium - My status\n"
                f"/referral - Refer & earn\n"
                f"/help - Help"
            )
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🌐 WEBSITE", url=config.WEBSITE_URL)],
                [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")],
                [InlineKeyboardButton("🎁 REFER & EARN", callback_data="referral_info")]
            ])
            await message.reply_text(text, reply_markup=keyboard)
            
        except Exception as e:
            logger.error(f"Start error: {e}")
            await message.reply_text("❌ Error!")
    
    # ============================================================================
    # BUY COMMAND
    # ============================================================================
    @bot.on_message(filters.command("buy"))
    async def buy_command(client, message):
        text = (
            "💎 **PREMIUM PLANS** 💎\n\n"
            "🎯 **ALL PLANS INCLUDE:**\n"
            "✅ All Quality (480p-4K)\n"
            "✅ Unlimited Downloads\n"
            "✅ No Verification Needed\n"
            "✅ VIP Support 24/7\n"
            "✅ No Ads\n"
            "✅ Custom Requests\n\n"
            "📊 **Choose Your Plan:**\n\n"
            "🥉 Basic - ₹9/15 days\n"
            "🥈 Standard - ₹19/28 days\n"
            "🥇 Pro - ₹29/49 days\n"
            "💎 Ultimate - ₹49/90 days"
        )
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🥉 Basic - ₹9", callback_data="buy_basic")],
            [InlineKeyboardButton("🥈 Standard - ₹19", callback_data="buy_standard")],
            [InlineKeyboardButton("🥇 Pro - ₹29", callback_data="buy_pro")],
            [InlineKeyboardButton("💎 Ultimate - ₹49", callback_data="buy_ultimate")],
            [InlineKeyboardButton("🎁 Refer & Get Premium", callback_data="referral_info")],
            [InlineKeyboardButton("🔙 BACK", callback_data="back_to_start")]
        ])
        await message.reply_text(text, reply_markup=keyboard)
    
    # ============================================================================
    # PLANS COMMAND
    # ============================================================================
    @bot.on_message(filters.command("plans"))
    async def plans_command(client, message):
        await buy_command(client, message)
    
    # ============================================================================
    # MYPREMIUM COMMAND
    # ============================================================================
    @bot.on_message(filters.command("mypremium"))
    async def mypremium_command(client, message):
        user_id = message.from_user.id
        
        if bot_instance.premium_system:
            is_premium = await bot_instance.premium_system.is_premium_user(user_id)
            if is_premium:
                details = await bot_instance.premium_system.get_subscription_details(user_id)
                await message.reply_text(
                    f"⭐ **Premium Active**\n\n"
                    f"Plan: {details.get('tier_name', 'Premium')}\n"
                    f"Days Left: {details.get('days_remaining', 0)}\n"
                    f"Expires: {details.get('expires_at', 'N/A')}"
                )
            else:
                await message.reply_text(
                    "👤 **Free User**\n\n"
                    "Use /buy to upgrade!"
                )
        else:
            await message.reply_text("❌ Premium system unavailable")
    
    # ============================================================================
    # REFERRAL COMMAND
    # ============================================================================
    @bot.on_message(filters.command("referral"))
    async def referral_command(client, message):
        user_id = message.from_user.id
        
        if bot_instance.premium_system:
            info = await bot_instance.premium_system.get_referral_info(user_id)
            text = (
                f"🎁 **REFER & GET PREMIUM**\n\n"
                f"👥 **Your Code:** `{info['referral_code']}`\n"
                f"📊 **Total Referrals:** {info['total_referrals']}\n\n"
                f"💎 **Milestone Rewards:**\n"
                f"• 3 referrals = Basic (15 days)\n"
                f"• 5 referrals = Standard (28 days)\n"
                f"• 10 referrals = Pro (49 days)\n\n"
                f"🔗 **Your Link:**\n"
                f"`{info['referral_link']}`"
            )
            await message.reply_text(text, disable_web_page_preview=True)
        else:
            await message.reply_text("❌ Premium system unavailable")
    
    # ============================================================================
    # HELP COMMAND
    # ============================================================================
    @bot.on_message(filters.command("help"))
    async def help_command(client, message):
        await message.reply_text(
            "📚 **HELP**\n\n"
            "/start - Start bot\n"
            "/buy - Buy premium\n"
            "/plans - View plans\n"
            "/mypremium - My status\n"
            "/referral - Refer & earn\n"
            "/help - This help"
        )
    
    # ============================================================================
    # CALLBACK QUERIES
    # ============================================================================
    @bot.on_callback_query()
    async def callbacks(client, callback_query):
        try:
            data = callback_query.data
            
            if data == "buy_premium":
                text = (
                    "💎 **PREMIUM PLANS** 💎\n\n"
                    "🥉 Basic - ₹9/15 days\n"
                    "🥈 Standard - ₹19/28 days\n"
                    "🥇 Pro - ₹29/49 days\n"
                    "💎 Ultimate - ₹49/90 days"
                )
                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton("🥉 Basic ₹9", callback_data="buy_basic")],
                    [InlineKeyboardButton("🥈 Standard ₹19", callback_data="buy_standard")],
                    [InlineKeyboardButton("🥇 Pro ₹29", callback_data="buy_pro")],
                    [InlineKeyboardButton("💎 Ultimate ₹49", callback_data="buy_ultimate")],
                    [InlineKeyboardButton("🎁 Refer", callback_data="referral_info")],
                    [InlineKeyboardButton("🔙 Back", callback_data="back_to_start")]
                ])
                await callback_query.message.edit_text(text, reply_markup=keyboard)
            
            elif data == "referral_info":
                user_id = callback_query.from_user.id
                if bot_instance.premium_system:
                    info = await bot_instance.premium_system.get_referral_info(user_id)
                    text = (
                        f"🎁 **REFER & GET PREMIUM**\n\n"
                        f"Code: `{info['referral_code']}`\n"
                        f"Total: {info['total_referrals']}\n\n"
                        f"3 refs = Basic (15 days)\n"
                        f"5 refs = Standard (28 days)\n"
                        f"10 refs = Pro (49 days)"
                    )
                    await callback_query.message.edit_text(text)
            
            elif data == "back_to_start":
                await callback_query.message.edit_text(
                    "🎬 **SK4FiLM**\n\nUse /buy for premium!"
                )
            
            elif data in ["buy_basic", "buy_standard", "buy_pro", "buy_ultimate"]:
                await callback_query.answer(
                    "💳 Use /buy command for plans",
                    show_alert=True
                )
            
            elif data == "verify_free":
                user_id = callback_query.from_user.id
                if bot_instance.verification_system:
                    verification_data = await bot_instance.verification_system.create_verification_link(user_id)
                    await callback_query.answer(
                        f"🔗 Verification link created!\nCheck your messages",
                        show_alert=True
                    )
            
            await callback_query.answer()
            
        except Exception as e:
            logger.error(f"Callback error: {e}")
            await callback_query.answer("❌ Error", show_alert=True)
    
    logger.info("✅ Bot handlers registered successfully")
