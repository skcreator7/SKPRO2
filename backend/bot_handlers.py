import asyncio
import logging
import re
import time
from datetime import datetime
from typing import Dict, Any
from collections import defaultdict

try:
    from pyrogram import filters
    from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton
    from pyrogram.errors import FloodWait
except ImportError:
    # Dummy classes
    class filters:
        @staticmethod
        def command(cmd): return lambda x: x
        @staticmethod
        def private(): return lambda x: x
        @staticmethod
        def regex(pattern): return lambda x: x
        @staticmethod
        def text(): return lambda x: x
        @staticmethod
        def photo(): return lambda x: x
        @staticmethod
        def document(): return lambda x: x
        @staticmethod
        def user(ids): return lambda x: x
    class InlineKeyboardMarkup:
        def __init__(self, buttons): pass
    class InlineKeyboardButton:
        def __init__(self, text, url=None, callback_data=None): pass

logger = logging.getLogger(__name__)

async def setup_bot_handlers(bot, bot_instance):
    """Setup all bot handlers"""
    config = bot_instance.config
    
    # ✅ TEXT MESSAGE HANDLER (Responds to all text messages)
    @bot.on_message(filters.private & filters.text & ~filters.command)
    async def handle_text_messages(client, message):
        """Handle all text messages (non-commands)"""
        try:
            user_id = message.from_user.id
            user_name = message.from_user.first_name or "User"
            message_id = message.id
            
            # Check if this message was already processed
            if message_id in bot_instance.text_message_cache:
                logger.info(f"📝 Text message {message_id} already processed, skipping")
                return
            
            # Mark as processed
            bot_instance.text_message_cache[message_id] = {
                'user_id': user_id,
                'timestamp': time.time(),
                'text': message.text[:100]  # Store first 100 chars
            }
            
            # Check if this looks like a file request
            text = message.text.strip()
            
            # Pattern for file requests: channel_message_quality
            file_pattern = r'^-?\d+_\d+(_\w+)?$'
            
            if re.match(file_pattern, text):
                # It's a file request, process it
                await handle_file_request(client, message, text, bot_instance)
                return
            
            # Check if it looks like a verification token
            if text.startswith('verify_'):
                token = text.replace('verify_', '', 1).strip()
                await handle_verification_token(client, message, token, bot_instance)
                return
            
            # Check if it's a payment ID
            if text.startswith('PAY_') and len(text) > 10:
                # User might be sending payment ID
                await message.reply_text(
                    "⚠️ **Payment ID Detected**\n\n"
                    f"Your Payment ID: `{text[:20]}...`\n\n"
                    "Please send the **payment screenshot** instead of just the ID.\n\n"
                    "📸 **How to send screenshot:**\n"
                    "1. Take screenshot of payment\n"
                    "2. Make sure Payment ID is visible\n"
                    "3. Send the image here\n\n"
                    "Admin will verify within 24 hours.",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("📸 SEND SCREENSHOT", 
                                             callback_data=f"send_screenshot_{text[:20]}")]
                    ])
                )
                return
            
            # Default response for other text messages
            website_url = getattr(config, 'WEBSITE_URL', 'https://sk4film.com')
            
            response_text = (
                f"👋 **Hello {user_name}!**\n\n"
                f"I see you sent: `{text[:50]}{'...' if len(text) > 50 else ''}`\n\n"
                f"🎬 **How to download files:**\n"
                f"1. Visit {website_url}\n"
                f"2. Search for movies\n"
                f"3. Click download button\n"
                f"4. File will appear here automatically\n\n"
                f"📋 **Available commands:**\n"
                f"• /start - Show welcome message\n"
                f"• /buy - Purchase premium\n"
                f"• /plans - View premium plans\n"
                f"• /help - Get help\n\n"
                f"❓ **Need help?**\n"
                f"Use /help command or contact admin"
            )
            
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🌐 OPEN WEBSITE", url=website_url)],
                [InlineKeyboardButton("⭐ GET PREMIUM", callback_data="buy_premium")],
                [InlineKeyboardButton("🆘 HELP", callback_data="show_help")]
            ])
            
            await message.reply_text(
                response_text,
                reply_markup=keyboard,
                disable_web_page_preview=True
            )
            
            logger.info(f"📝 Responded to text message from {user_id}: {text[:50]}")
            
        except Exception as e:
            logger.error(f"Error handling text message: {e}")
            try:
                await message.reply_text(
                    "❌ **Error Processing Message**\n\n"
                    "There was an error processing your message. Please try again."
                )
            except:
                pass
    
    # ✅ SCREENSHOT HANDLER (Processes payment screenshots)
    @bot.on_message(filters.private & (filters.photo | filters.document))
    async def handle_screenshots(client, message):
        """Handle payment screenshots from users"""
        try:
            user_id = message.from_user.id
            user_name = message.from_user.first_name or "User"
            message_id = message.id
            
            # Check if message is likely a screenshot
            is_screenshot = False
            
            if message.photo:
                is_screenshot = True
                file_type = "photo"
                file_id = message.photo.file_id
            elif (message.document and 
                  message.document.mime_type and 
                  'image' in message.document.mime_type.lower()):
                is_screenshot = True
                file_type = "document"
                file_id = message.document.file_id
            else:
                # Not an image, skip
                return
            
            if not is_screenshot:
                return
            
            logger.info(f"📸 Screenshot received from user {user_id}: {file_type}")
            
            # Check if already processed
            if bot_instance.premium_system and message_id in bot_instance.premium_system.processed_screenshots:
                logger.info(f"📸 Screenshot {message_id} already processed, skipping")
                return
            
            # Send processing message
            processing_msg = await message.reply_text(
                "📸 **Processing screenshot...**\n\n"
                "Please wait while we verify your payment screenshot..."
            )
            
            # Process the screenshot
            if bot_instance.premium_system:
                result = await bot_instance.premium_system.process_payment_screenshot(user_id, message_id)
                
                if result['status'] == 'success':
                    # Success - screenshot linked to payment
                    payment_id = result['payment_id']
                    payment_data = result['payment_data']
                    
                    # Send success message to user
                    success_text = (
                        "✅ **Screenshot Received Successfully!**\n\n"
                        f"📋 **Payment ID:** `{payment_id}`\n"
                        f"💰 **Plan:** {payment_data.get('tier_name', 'Premium')}\n"
                        f"💵 **Amount:** ₹{payment_data.get('amount', 0)}\n\n"
                        "⏰ **What happens next:**\n"
                        "1. Admin notified of your payment\n"
                        "2. Admin verifies screenshot\n"
                        "3. You get premium within 24 hours\n"
                        "4. Receive confirmation message\n\n"
                        "📞 **Contact support if delayed:**\n"
                        f"{getattr(config, 'SUPPORT_CHANNEL', '@SK4FiLMSupport')}\n\n"
                        "Thank you for your patience! ❤️"
                    )
                    
                    await processing_msg.edit_text(
                        success_text,
                        disable_web_page_preview=True
                    )
                    
                    # Notify all admins
                    admin_notified = await bot_instance.premium_system.notify_admins_of_screenshot(
                        payment_id, user_id, message_id
                    )
                    
                    if admin_notified:
                        logger.info(f"✅ Screenshot processed for user {user_id}, admins notified")
                    else:
                        logger.warning(f"⚠️ Screenshot processed but admin notification failed")
                    
                elif result['status'] == 'no_pending_payment':
                    # No pending payment found
                    error_text = (
                        "❌ **No Pending Payment Found**\n\n"
                        "We couldn't find a pending payment for your account.\n\n"
                        "📝 **What to do:**\n"
                        "1. Use /buy command first\n"
                        "2. Select a premium plan\n"
                        "3. Get payment instructions\n"
                        "4. Then send screenshot\n\n"
                        "🔄 **Start purchase:**\n"
                        "Click the button below"
                    )
                    
                    keyboard = InlineKeyboardMarkup([
                        [InlineKeyboardButton("💰 BUY PREMIUM", callback_data="buy_premium")],
                        [InlineKeyboardButton("🆘 HELP", callback_data="show_help")]
                    ])
                    
                    await processing_msg.edit_text(
                        error_text,
                        reply_markup=keyboard,
                        disable_web_page_preview=True
                    )
                    
                elif result['status'] == 'already_processed':
                    # Already processed
                    await processing_msg.edit_text(
                        "⚠️ **Already Processed**\n\n"
                        "This screenshot has already been processed.\n"
                        "Admin is reviewing your payment."
                    )
                    
                else:
                    # Error
                    await processing_msg.edit_text(
                        "❌ **Error Processing Screenshot**\n\n"
                        "There was an error processing your screenshot.\n"
                        "Please try again or contact support."
                    )
                    
            else:
                # Premium system not available
                await processing_msg.edit_text(
                    "❌ **Premium System Unavailable**\n\n"
                    "The premium system is temporarily unavailable.\n"
                    "Please try again later or contact admin."
                )
                
        except FloodWait as e:
            logger.warning(f"⏳ Flood wait in screenshot handler: {e.value}s")
            await asyncio.sleep(e.value)
            # Try to send a simple response
            try:
                await message.reply_text(
                    "⚠️ **Please wait...**\n\n"
                    f"Telegram limits: Wait {e.value} seconds and try again."
                )
            except:
                pass
        except Exception as e:
            logger.error(f"Error in screenshot handler: {e}")
            try:
                await message.reply_text(
                    "❌ **Error Processing Screenshot**\n\n"
                    "An error occurred. Please try again."
                )
            except:
                pass
    
    # ✅ ADMIN CALLBACK HANDLERS
    @bot.on_callback_query(filters.regex(r"^admin_"))
    async def handle_admin_callbacks(client, callback_query):
        """Handle admin callback queries"""
        try:
            admin_id = callback_query.from_user.id
            data = callback_query.data
            
            # Check if user is admin
            if admin_id not in getattr(config, 'ADMIN_IDS', []):
                await callback_query.answer("❌ Admin access required!", show_alert=True)
                return
            
            # Parse callback data
            parts = data.split('_')
            action = parts[1] if len(parts) > 1 else ""
            payment_id = parts[2] if len(parts) > 2 else ""
            
            if not payment_id:
                await callback_query.answer("❌ Invalid payment ID!", show_alert=True)
                return
            
            if action == "approve":
                # Handle approval
                await callback_query.answer("Processing approval...")
                
                if bot_instance.premium_system:
                    result = await bot_instance.premium_system.handle_admin_approve_callback(
                        admin_id, payment_id
                    )
                    
                    if result['success']:
                        # Update the callback message
                        new_text = (
                            f"✅ **APPROVED BY ADMIN** ✅\n\n"
                            f"**Payment ID:** `{payment_id}`\n"
                            f"**Approved by:** Admin {admin_id}\n"
                            f"**Approved at:** {datetime.now().strftime('%d %b %Y %I:%M %p')}\n\n"
                            f"✅ **Status:** Approved\n"
                            f"👤 **User ID:** `{result.get('user_id', 'Unknown')}`\n\n"
                            f"User has been upgraded to premium!"
                        )
                        
                        try:
                            await callback_query.message.edit_text(
                                new_text,
                                reply_markup=InlineKeyboardMarkup([
                                    [InlineKeyboardButton("✅ APPROVED", callback_data="none")],
                                    [InlineKeyboardButton("📊 VIEW USER", 
                                                         callback_data=f"admin_viewuser_{payment_id}")]
                                ])
                            )
                        except:
                            await callback_query.answer("✅ Approved successfully!", show_alert=True)
                    else:
                        await callback_query.answer(f"❌ {result['message']}", show_alert=True)
                else:
                    await callback_query.answer("❌ Premium system not available!", show_alert=True)
            
            elif action == "reject":
                # Ask for rejection reason
                await callback_query.answer("Please enter rejection reason using /reject command", show_alert=True)
                
                # Send message to admin about how to reject
                try:
                    await client.send_message(
                        admin_id,
                        f"❌ **To reject payment:**\n\n"
                        f"**Payment ID:** `{payment_id}`\n\n"
                        f"Use command:\n"
                        f"`/reject {payment_id} <reason>`\n\n"
                        f"**Example:**\n"
                        f"`/reject {payment_id} Invalid screenshot`"
                    )
                except:
                    pass
            
            elif action == "viewuser":
                # View user info
                if bot_instance.premium_system:
                    payment = bot_instance.premium_system.pending_payments.get(payment_id)
                    if payment:
                        user_id = payment['user_id']
                        
                        # Get user info
                        try:
                            user = await client.get_users(user_id)
                            user_name = f"{user.first_name or ''} {user.last_name or ''}".strip()
                            username = f"@{user.username}" if user.username else "No username"
                            
                            user_info = (
                                f"👤 **User Information**\n\n"
                                f"**Name:** {user_name}\n"
                                f"**Username:** {username}\n"
                                f"**ID:** `{user_id}`\n"
                                f"**Language:** {user.language_code or 'Unknown'}\n"
                                f"**Premium User:** {'✅ Yes' if user.is_premium else '❌ No'}\n\n"
                                f"💬 **To message user:**\n"
                                f"Click button below"
                            )
                            
                            keyboard = InlineKeyboardMarkup([
                                [InlineKeyboardButton("💬 MESSAGE USER", 
                                                     url=f"tg://user?id={user_id}")],
                                [InlineKeyboardButton("🔙 BACK", 
                                                     callback_data=f"admin_details_{payment_id}")]
                            ])
                            
                            await callback_query.message.edit_text(
                                user_info,
                                reply_markup=keyboard
                            )
                        except Exception as e:
                            await callback_query.answer(f"❌ Error: {str(e)[:50]}", show_alert=True)
                    else:
                        await callback_query.answer("❌ Payment not found!", show_alert=True)
                else:
                    await callback_query.answer("❌ System error!", show_alert=True)
            
            elif action == "details":
                # Show payment details
                if bot_instance.premium_system:
                    payment = bot_instance.premium_system.pending_payments.get(payment_id)
                    if payment:
                        # Calculate hours left
                        expires_at = payment.get('expires_at', datetime.now())
                        time_left = expires_at - datetime.now()
                        hours_left = max(0, int(time_left.total_seconds() / 3600))
                        
                        details_text = (
                            f"📋 **Payment Details**\n\n"
                            f"**Payment ID:** `{payment_id}`\n"
                            f"**User ID:** `{payment['user_id']}`\n"
                            f"**Plan:** {payment['tier_name']}\n"
                            f"**Amount:** ₹{payment['amount']}\n"
                            f"**Duration:** {payment['duration_days']} days\n"
                            f"**Status:** {payment['status'].title()}\n"
                            f"**Screenshot:** {'✅ Sent' if payment.get('screenshot_sent') else '❌ Not sent'}\n"
                            f"**Created:** {payment['created_at'].strftime('%d %b %Y %I:%M %p')}\n"
                            f"**Expires:** {expires_at.strftime('%d %b %Y %I:%M %p')}\n"
                            f"**Hours Left:** {hours_left} hours\n\n"
                            f"⚡ **Quick Actions:**"
                        )
                        
                        keyboard = InlineKeyboardMarkup([
                            [
                                InlineKeyboardButton("✅ APPROVE", callback_data=f"admin_approve_{payment_id}"),
                                InlineKeyboardButton("❌ REJECT", callback_data=f"admin_reject_{payment_id}")
                            ],
                            [
                                InlineKeyboardButton("👁️ VIEW USER", callback_data=f"admin_viewuser_{payment_id}"),
                                InlineKeyboardButton("🔙 BACK", callback_data="admin_back")
                            ]
                        ])
                        
                        await callback_query.message.edit_text(
                            details_text,
                            reply_markup=keyboard
                        )
                    else:
                        await callback_query.answer("❌ Payment not found!", show_alert=True)
                else:
                    await callback_query.answer("❌ System error!", show_alert=True)
            
        except Exception as e:
            logger.error(f"Error in admin callback handler: {e}")
            try:
                await callback_query.answer("❌ Error processing request", show_alert=True)
            except:
                pass
    
    # ✅ USER CALLBACK HANDLERS
    @bot.on_callback_query(filters.regex(r"^show_help$"))
    async def show_help_callback(client, callback_query):
        """Show help information"""
        try:
            user_name = callback_query.from_user.first_name or "User"
            
            help_text = (
                f"🆘 **Help Center - {user_name}** 🆘\n\n"
                "**Common Issues:**\n"
                "❓ **Can't download files?**\n"
                "• You need verification (free users)\n"
                "• Or buy premium (no verification)\n\n"
                "❓ **Verification not working?**\n"
                "• Click verification link\n"
                "• Join required channel\n"
                "• Return to bot\n\n"
                "❓ **Payment issues?**\n"
                "• Send clear screenshot\n"
                "• Include payment ID\n"
                "• Wait 24 hours for verification\n\n"
                "**Commands:**\n"
                "• /start - Show welcome\n"
                "• /buy - Purchase premium\n"
                "• /plans - View plans\n"
                "• /mypremium - Your status\n"
                "• /help - This message\n\n"
                "**Support:**\n"
                f"🌐 Website: {getattr(config, 'WEBSITE_URL', 'https://sk4film.com')}\n"
                "📢 Channel: @SK4FiLM\n"
                "🆘 Support: @SK4FiLMSupport\n\n"
                "🎬 **Happy downloading!**"
            )
            
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("⭐ GET PREMIUM", callback_data="buy_premium")],
                [InlineKeyboardButton("🔗 GET VERIFIED", callback_data="get_verified")],
                [InlineKeyboardButton("🔙 BACK", callback_data="back_to_start")]
            ])
            
            await callback_query.message.edit_text(
                help_text,
                reply_markup=keyboard,
                disable_web_page_preview=True
            )
            
        except Exception as e:
            logger.error(f"Error in show help callback: {e}")
            await callback_query.answer("Error showing help!", show_alert=True)
    
    # ✅ STANDARD COMMAND HANDLERS
    @bot.on_message(filters.command("start"))
    async def handle_start_command(client, message):
        """Handle /start command"""
        user_name = message.from_user.first_name or "User"
        user_id = message.from_user.id
        
        # Check if there's additional text
        if len(message.command) > 1:
            start_text = ' '.join(message.command[1:])
            
            # Check if it's a verification token
            if start_text.startswith('verify_'):
                token = start_text.replace('verify_', '', 1).strip()
                await handle_verification_token(client, message, token, bot_instance)
                return
            else:
                # Treat as file request
                await handle_file_request(client, message, start_text, bot_instance)
                return
        
        # WELCOME MESSAGE
        welcome_text = (
            f"🎬 **Welcome to SK4FiLM, {user_name}!**\n\n"
            f"🌐 **Website:** {config.WEBSITE_URL}\n\n"
            "**Commands:**\n"
            "• /mypremium - Check your premium status\n"
            "• /plans - View premium plans\n"
            "• /buy - Purchase premium\n"
            "• /help - Show help\n\n"
            "**How to download:**\n"
            "1. Visit website above\n"
            "2. Search for movies\n"
            "3. Click download button\n"
            "4. File will appear here automatically\n\n"
            "🎬 **Happy watching!**"
        )
        
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🌐 OPEN WEBSITE", url=config.WEBSITE_URL)],
            [InlineKeyboardButton("⭐ GET PREMIUM", callback_data="buy_premium")],
            [InlineKeyboardButton("📢 JOIN CHANNEL", url=getattr(config, 'MAIN_CHANNEL_LINK', 'https://t.me/SK4FiLM'))]
        ])
        
        await message.reply_text(welcome_text, reply_markup=keyboard, disable_web_page_preview=True)
    
    @bot.on_message(filters.command("mypremium") & filters.private)
    async def my_premium_command(client, message):
        """Check user's premium status"""
        user_id = message.from_user.id
        user_name = message.from_user.first_name or "User"
        
        if not bot_instance.premium_system:
            await message.reply_text("❌ Premium system not available. Please try again later.")
            return
        
        try:
            # Get premium info
            premium_info = await bot_instance.premium_system.get_my_premium_info(user_id)
            
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")],
                [InlineKeyboardButton("🌐 OPEN WEBSITE", url=config.WEBSITE_URL)]
            ])
            
            await message.reply_text(premium_info, reply_markup=keyboard, disable_web_page_preview=True)
            
        except Exception as e:
            logger.error(f"My premium command error: {e}")
            await message.reply_text("❌ Error fetching premium info. Please try again.")
    
    @bot.on_message(filters.command("plans") & filters.private)
    async def plans_command(client, message):
        """Show all premium plans"""
        if not bot_instance.premium_system:
            await message.reply_text("❌ Premium system not available. Please try again later.")
            return
        
        try:
            plans_text = await bot_instance.premium_system.get_available_plans_text()
            
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("💰 BUY BASIC (₹99)", callback_data="plan_basic")],
                [InlineKeyboardButton("💰 BUY PREMIUM (₹199)", callback_data="plan_premium")],
                [InlineKeyboardButton("💰 BUY GOLD (₹299)", callback_data="plan_gold")],
                [InlineKeyboardButton("💰 BUY DIAMOND (₹499)", callback_data="plan_diamond")],
                [InlineKeyboardButton("🔙 BACK", callback_data="back_to_start")]
            ])
            
            await message.reply_text(plans_text, reply_markup=keyboard, disable_web_page_preview=True)
            
        except Exception as e:
            logger.error(f"Plans command error: {e}")
            await message.reply_text("❌ Error fetching plans. Please try again.")
    
    @bot.on_message(filters.command("buy") & filters.private)
    async def buy_command(client, message):
        """Initiate premium purchase"""
        user_id = message.from_user.id
        user_name = message.from_user.first_name or "User"
        
        # Check if already premium
        if bot_instance.premium_system:
            is_premium = await bot_instance.premium_system.is_premium_user(user_id)
            if is_premium:
                details = await bot_instance.premium_system.get_subscription_details(user_id)
                
                text = (
                    f"⭐ **You're Already Premium!** ⭐\n\n"
                    f"**User:** {user_name}\n"
                    f"**Plan:** {details.get('tier_name', 'Premium')}\n"
                    f"**Days Left:** {details.get('days_remaining', 0)}\n\n"
                    "Enjoy unlimited downloads without verification! 🎬"
                )
                
                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton("🌐 OPEN WEBSITE", url=config.WEBSITE_URL)],
                    [InlineKeyboardButton("🔙 BACK", callback_data="back_to_start")]
                ])
                
                await message.reply_text(text, reply_markup=keyboard)
                return
        
        text = (
            f"💰 **Purchase Premium - {user_name}**\n\n"
            "**Select a plan:**\n\n"
            "🥉 **Basic Plan** - ₹99/month\n"
            "• All quality (480p-4K)\n"
            "• Unlimited downloads\n"
            "• No verification\n\n"
            "🥈 **Premium Plan** - ₹199/month\n"
            "• Everything in Basic +\n"
            "• Priority support\n"
            "• Faster downloads\n\n"
            "Click a button below to purchase:"
        )
        
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🥉 BUY BASIC (₹99)", callback_data="plan_basic")],
            [InlineKeyboardButton("🥈 BUY PREMIUM (₹199)", callback_data="plan_premium")],
            [InlineKeyboardButton("🥇 BUY GOLD (₹299)", callback_data="plan_gold")],
            [InlineKeyboardButton("💎 BUY DIAMOND (₹499)", callback_data="plan_diamond")],
            [InlineKeyboardButton("🔙 BACK", callback_data="back_to_start")]
        ])
        
        await message.reply_text(text, reply_markup=keyboard)
    
    @bot.on_message(filters.command("help") & filters.private)
    async def help_command(client, message):
        """Show help message"""
        help_text = (
            "🆘 **SK4FiLM Bot Help** 🆘\n\n"
            "**Available Commands:**\n"
            "• /start - Start the bot\n"
            "• /mypremium - Check your premium status\n"
            "• /plans - View premium plans\n"
            "• /buy - Purchase premium subscription\n"
            "• /help - Show this help message\n\n"
            "**How to Download Files:**\n"
            "1. Visit our website\n"
            "2. Search for movies/TV shows\n"
            "3. Click download button\n"
            "4. File will appear here automatically\n\n"
            "**Verification System:**\n"
            "• Free users need verification every 6 hours\n"
            "• Premium users don't need verification\n"
            "• Verification link valid for 1 hour\n\n"
            "**Auto-Delete Feature:**\n"
            "• Files auto-delete after 15 minutes\n"
            "• For security and privacy\n"
            "• Download again if needed\n\n"
            "**Support:**\n"
            f"🌐 Website: {config.WEBSITE_URL}\n"
            "📢 Channel: @SK4FiLM\n"
            "🆘 Issues: Contact admin\n\n"
            "🎬 **Happy downloading!**"
        )
        
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🌐 OPEN WEBSITE", url=config.WEBSITE_URL)],
            [InlineKeyboardButton("⭐ GET PREMIUM", callback_data="buy_premium")],
            [InlineKeyboardButton("🔙 BACK", callback_data="back_to_start")]
        ])
        
        await message.reply_text(help_text, reply_markup=keyboard, disable_web_page_preview=True)
    
    # ✅ ADMIN COMMANDS
    @bot.on_message(filters.command("addpremium") & filters.user(getattr(config, 'ADMIN_IDS', [])))
    async def add_premium_command(client, message):
        """Add premium user command for admins"""
        try:
            if len(message.command) < 4:
                await message.reply_text(
                    "❌ **Usage:** `/addpremium <user_id> <days> <plan_type>`\n\n"
                    "**Examples:**\n"
                    "• `/addpremium 123456789 30 basic`\n"
                    "• `/addpremium 123456789 365 premium`\n\n"
                    "**Plan types:** basic, premium, gold, diamond"
                )
                return
            
            user_id = int(message.command[1])
            days = int(message.command[2])
            plan_type = message.command[3].lower()
            
            # Map plan type to PremiumTier
            plan_map = {
                'basic': bot_instance.PremiumTier.BASIC,
                'premium': bot_instance.PremiumTier.PREMIUM,
                'gold': bot_instance.PremiumTier.GOLD,
                'diamond': bot_instance.PremiumTier.DIAMOND
            }
            
            if plan_type not in plan_map:
                await message.reply_text(
                    "❌ **Invalid plan type**\n\n"
                    "Use: `basic`, `premium`, `gold`, or `diamond`\n"
                    "Example: `/addpremium 123456789 30 basic`"
                )
                return
            
            if days <= 0:
                await message.reply_text("❌ Days must be greater than 0")
                return
            
            tier = plan_map[plan_type]
            
            # Get user info
            try:
                user = await client.get_users(user_id)
                user_name = f"{user.first_name or ''} {user.last_name or ''}".strip() or f"User {user_id}"
                username = f"@{user.username}" if user.username else "No username"
            except:
                user_name = f"User {user_id}"
                username = "Unknown"
            
            # Add premium subscription
            if bot_instance.premium_system:
                subscription_data = await bot_instance.premium_system.add_premium_subscription(
                    admin_id=message.from_user.id,
                    user_id=user_id,
                    tier=tier,
                    days=days,
                    reason="admin_command"
                )
                
                if subscription_data:
                    await message.reply_text(
                        f"✅ **Premium User Added Successfully!**\n\n"
                        f"**User:** {user_name}\n"
                        f"**ID:** `{user_id}`\n"
                        f"**Username:** {username}\n"
                        f"**Plan:** {plan_type.capitalize()}\n"
                        f"**Duration:** {days} days\n\n"
                        f"User can now download files without verification!"
                    )
                    
                    # Notify user
                    try:
                        await client.send_message(
                            user_id,
                            f"🎉 **Congratulations!** 🎉\n\n"
                            f"You've been upgraded to **{plan_type.capitalize()} Premium** by admin!\n\n"
                            f"✅ **Plan:** {plan_type.capitalize()}\n"
                            f"📅 **Valid for:** {days} days\n"
                            f"⭐ **Benefits:**\n"
                            f"• Instant file access\n"
                            f"• No verification required\n"
                            f"• Priority support\n\n"
                            f"🎬 **Enjoy unlimited downloads!**"
                        )
                    except:
                        pass
                else:
                    await message.reply_text("❌ Failed to add premium subscription.")
            else:
                await message.reply_text("❌ Premium system not available")
                
        except ValueError:
            await message.reply_text(
                "❌ **Invalid parameters**\n\n"
                "Correct format: `/addpremium <user_id> <days> <plan_type>`\n"
                "Example: `/addpremium 123456789 30 basic`"
            )
        except Exception as e:
            logger.error(f"Add premium command error: {e}")
            await message.reply_text(f"❌ Error: {str(e)[:100]}")
    
    @bot.on_message(filters.command("removepremium") & filters.user(getattr(config, 'ADMIN_IDS', [])))
    async def remove_premium_command(client, message):
        """Remove premium user command for admins"""
        try:
            if len(message.command) < 2:
                await message.reply_text(
                    "❌ **Usage:** `/removepremium <user_id>`\n\n"
                    "**Example:** `/removepremium 123456789`"
                )
                return
            
            user_id = int(message.command[1])
            
            if bot_instance.premium_system:
                success = await bot_instance.premium_system.remove_premium_subscription(
                    admin_id=message.from_user.id,
                    user_id=user_id,
                    reason="admin_command"
                )
                
                if success:
                    await message.reply_text(
                        f"✅ **Premium Removed Successfully!**\n\n"
                        f"**User ID:** `{user_id}`\n"
                        f"Premium access has been revoked."
                    )
                else:
                    await message.reply_text("❌ User not found or not premium")
            else:
                await message.reply_text("❌ Premium system not available")
                
        except ValueError:
            await message.reply_text("❌ Invalid user ID. Must be a number.")
        except Exception as e:
            logger.error(f"Remove premium command error: {e}")
            await message.reply_text(f"❌ Error: {str(e)[:100]}")
    
    @bot.on_message(filters.command("checkpremium") & filters.user(getattr(config, 'ADMIN_IDS', [])))
    async def check_premium_command(client, message):
        """Check premium status of user"""
        try:
            if len(message.command) < 2:
                await message.reply_text(
                    "❌ **Usage:** `/checkpremium <user_id>`\n\n"
                    "**Example:** `/checkpremium 123456789`"
                )
                return
            
            user_id = int(message.command[1])
            
            if bot_instance.premium_system:
                user_info = await bot_instance.premium_system.get_premium_user_info(user_id)
                
                # Get user info
                try:
                    user = await client.get_users(user_id)
                    user_name = f"{user.first_name or ''} {user.last_name or ''}".strip() or f"User {user_id}"
                    username = f"@{user.username}" if user.username else "No username"
                except:
                    user_name = f"User {user_id}"
                    username = "Unknown"
                
                if user_info['tier'] == 'free':
                    await message.reply_text(
                        f"❌ **Not a Premium User**\n\n"
                        f"**User:** {user_name}\n"
                        f"**ID:** `{user_id}`\n"
                        f"**Username:** {username}\n"
                        f"**Status:** Free User\n\n"
                        f"This user does not have premium access."
                    )
                else:
                    await message.reply_text(
                        f"✅ **Premium User Found**\n\n"
                        f"**User:** {user_name}\n"
                        f"**ID:** `{user_id}`\n"
                        f"**Username:** {username}\n"
                        f"**Plan:** {user_info.get('tier_name', 'Unknown')}\n"
                        f"**Status:** {user_info.get('status', 'Unknown').title()}\n"
                        f"**Days Left:** {user_info.get('days_remaining', 0)}\n"
                        f"**Total Downloads:** {user_info.get('total_downloads', 0)}\n"
                        f"**Joined:** {user_info.get('purchased_at', 'Unknown')}\n"
                        f"**Expires:** {user_info.get('expires_at', 'Unknown')}"
                    )
            else:
                await message.reply_text("❌ Premium system not available")
                
        except ValueError:
            await message.reply_text("❌ Invalid user ID. Must be a number.")
        except Exception as e:
            logger.error(f"Check premium command error: {e}")
            await message.reply_text(f"❌ Error: {str(e)[:100]}")
    
    @bot.on_message(filters.command("stats") & filters.user(getattr(config, 'ADMIN_IDS', [])))
    async def stats_command(client, message):
        """Show bot statistics"""
        try:
            if bot_instance.premium_system:
                stats = await bot_instance.premium_system.get_statistics()
                
                stats_text = (
                    f"📊 **SK4FiLM Bot Statistics** 📊\n\n"
                    f"👥 **Total Users:** {stats.get('total_users', 0)}\n"
                    f"⭐ **Premium Users:** {stats.get('premium_users', 0)}\n"
                    f"✅ **Active Premium:** {stats.get('active_premium', 0)}\n"
                    f"🎯 **Free Users:** {stats.get('free_users', 0)}\n\n"
                    f"📥 **Total Downloads:** {stats.get('total_downloads', 0)}\n"
                    f"💾 **Total Data Sent:** {stats.get('total_data_sent', '0 GB')}\n"
                    f"💰 **Total Revenue:** {stats.get('total_revenue', '₹0')}\n"
                    f"🛒 **Premium Sales:** {stats.get('total_premium_sales', 0)}\n"
                    f"⏳ **Pending Payments:** {stats.get('pending_payments', 0)}\n\n"
                    f"🔄 **System Status:**\n"
                    f"• Bot: {'✅ Online' if bot_instance.bot_started else '❌ Offline'}\n"
                    f"• User Client: {'✅ Connected' if bot_instance.user_session_ready else '❌ Disconnected'}\n"
                    f"• Verification: {'✅ Active' if bot_instance.verification_system else '❌ Inactive'}\n"
                    f"• Premium: {'✅ Active' if bot_instance.premium_system else '❌ Inactive'}\n\n"
                    f"⏰ **Uptime:** {stats.get('uptime', 'Unknown')}\n"
                    f"🕐 **Server Time:** {stats.get('server_time', 'Unknown')}"
                )
                
                await message.reply_text(stats_text, disable_web_page_preview=True)
            else:
                await message.reply_text("❌ Premium system not available for stats")
                
        except Exception as e:
            logger.error(f"Stats command error: {e}")
            await message.reply_text(f"❌ Error getting stats: {str(e)[:100]}")
    
    @bot.on_message(filters.command("pending") & filters.user(getattr(config, 'ADMIN_IDS', [])))
    async def pending_payments_command(client, message):
        """Show pending payments"""
        try:
            if bot_instance.premium_system:
                pending = await bot_instance.premium_system.get_pending_payments_admin()
                
                if not pending:
                    await message.reply_text("✅ No pending payments!")
                    return
                
                text = f"⏳ **Pending Payments:** {len(pending)}\n\n"
                
                for i, payment in enumerate(pending[:10], 1):  # Show first 10
                    text += (
                        f"{i}. **ID:** `{payment['payment_id']}`\n"
                        f"   **User:** `{payment['user_id']}`\n"
                        f"   **Plan:** {payment['tier_name']}\n"
                        f"   **Amount:** ₹{payment['amount']}\n"
                        f"   **Screenshot:** {'✅ Sent' if payment['screenshot_sent'] else '❌ Not sent'}\n"
                        f"   **Time Left:** {payment['hours_left']} hours\n\n"
                    )
                
                if len(pending) > 10:
                    text += f"... and {len(pending) - 10} more pending payments\n\n"
                
                text += "Use `/approve <payment_id>` to approve payment."
                
                await message.reply_text(text, disable_web_page_preview=True)
            else:
                await message.reply_text("❌ Premium system not available")
                
        except Exception as e:
            logger.error(f"Pending payments command error: {e}")
            await message.reply_text(f"❌ Error: {str(e)[:100]}")
    
    @bot.on_message(filters.command("approve") & filters.user(getattr(config, 'ADMIN_IDS', [])))
    async def approve_payment_command(client, message):
        """Approve pending payment"""
        try:
            if len(message.command) < 2:
                await message.reply_text(
                    "❌ **Usage:** `/approve <payment_id>`\n\n"
                    "**Example:** `/approve PAY_ABC123DEF456`"
                )
                return
            
            payment_id = message.command[1].strip()
            
            if bot_instance.premium_system:
                success, result = await bot_instance.premium_system.approve_payment(
                    admin_id=message.from_user.id,
                    payment_id=payment_id
                )
                
                if success:
                    await message.reply_text(f"✅ {result}")
                    
                    # Notify user
                    try:
                        # Find user from payment
                        for pid, payment in bot_instance.premium_system.pending_payments.items():
                            if pid == payment_id:
                                user_id = payment['user_id']
                                plan_name = payment['tier_name']
                                
                                await client.send_message(
                                    user_id,
                                    f"🎉 **Payment Approved!** 🎉\n\n"
                                    f"Your payment for **{plan_name}** has been approved!\n\n"
                                    f"✅ **Status:** Premium Active\n"
                                    f"⭐ **Benefits:**\n"
                                    f"• No verification required\n"
                                    f"• Instant file access\n"
                                    f"• Priority support\n\n"
                                    f"🎬 **Enjoy unlimited downloads!**"
                                )
                                break
                    except:
                        pass
                else:
                    await message.reply_text(f"❌ {result}")
            else:
                await message.reply_text("❌ Premium system not available")
                
        except Exception as e:
            logger.error(f"Approve payment command error: {e}")
            await message.reply_text(f"❌ Error: {str(e)[:100]}")
    
    @bot.on_message(filters.command("reject") & filters.user(getattr(config, 'ADMIN_IDS', [])))
    async def reject_payment_command(client, message):
        """Reject pending payment"""
        try:
            if len(message.command) < 3:
                await message.reply_text(
                    "❌ **Usage:** `/reject <payment_id> <reason>`\n\n"
                    "**Example:** `/reject PAY_ABC123DEF456 Invalid screenshot`"
                )
                return
            
            payment_id = message.command[1].strip()
            reason = ' '.join(message.command[2:])
            
            if bot_instance.premium_system:
                success = await bot_instance.premium_system.reject_payment(
                    admin_id=message.from_user.id,
                    payment_id=payment_id,
                    reason=reason
                )
                
                if success:
                    await message.reply_text(f"✅ Payment {payment_id} rejected!\n**Reason:** {reason}")
                else:
                    await message.reply_text(f"❌ Failed to reject payment {payment_id}")
            else:
                await message.reply_text("❌ Premium system not available")
                
        except Exception as e:
            logger.error(f"Reject payment command error: {e}")
            await message.reply_text(f"❌ Error: {str(e)[:100]}")
    
    # ✅ FILE REQUEST HANDLER
    @bot.on_message(filters.private & filters.regex(r'^-?\d+_\d+(_\w+)?$'))
    async def handle_direct_file_request(client, message):
        """Handle direct file format messages"""
        file_text = message.text.strip()
        await handle_file_request(client, message, file_text, bot_instance)
    
    # ✅ CALLBACK HANDLERS
    @bot.on_callback_query(filters.regex(r"^get_verified$"))
    async def get_verified_callback(client, callback_query):
        """Get verification link"""
        user_id = callback_query.from_user.id
        user_name = callback_query.from_user.first_name or "User"
        
        if bot_instance.verification_system:
            verification_data = await bot_instance.verification_system.create_verification_link(user_id)
            
            text = (
                f"🔗 **Verification Required - {user_name}**\n\n"
                "To access files, you need to verify:\n\n"
                f"🔗 **Click:** {verification_data['short_url']}\n"
                f"⏰ **Valid for:** {verification_data['valid_for_hours']} hours\n\n"
                "**Steps:**\n"
                "1. Click VERIFY NOW button\n"
                "2. Join our channel\n"
                "3. Return here for downloads\n"
                "4. Access lasts 6 hours\n\n"
                "⭐ **Premium users don't need verification**"
            )
            
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🔗 VERIFY NOW", url=verification_data['short_url'])],
                [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")],
                [InlineKeyboardButton("🔙 BACK", callback_data="back_to_start")]
            ])
            
            try:
                await callback_query.message.edit_text(
                    text=text,
                    reply_markup=keyboard,
                    disable_web_page_preview=True
                )
            except:
                await callback_query.answer("Click VERIFY NOW button!", show_alert=True)
        else:
            await callback_query.answer("Verification system not available!", show_alert=True)
    
    @bot.on_callback_query(filters.regex(r"^back_to_start$"))
    async def back_to_start_callback(client, callback_query):
        user_name = callback_query.from_user.first_name or "User"
        
        text = (
            f"🎬 **Welcome back, {user_name}!**\n\n"
            f"Visit {config.WEBSITE_URL} to download movies.\n"
            "Click download button on website and file will appear here."
        )
        
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🌐 OPEN WEBSITE", url=config.WEBSITE_URL)],
            [InlineKeyboardButton("📢 JOIN CHANNEL", url=getattr(config, 'MAIN_CHANNEL_LINK', 'https://t.me/SK4FiLM'))]
        ])
        
        try:
            await callback_query.message.edit_text(
                text=text,
                reply_markup=keyboard,
                disable_web_page_preview=True
            )
        except:
            await callback_query.answer("Already on home page!")
    
    @bot.on_callback_query(filters.regex(r"^buy_premium$"))
    async def buy_premium_callback(client, callback_query):
        """Show premium plans"""
        user_id = callback_query.from_user.id
        user_name = callback_query.from_user.first_name or "User"
        
        # Check if already premium
        if bot_instance.premium_system:
            is_premium = await bot_instance.premium_system.is_premium_user(user_id)
            if is_premium:
                details = await bot_instance.premium_system.get_subscription_details(user_id)
                
                text = (
                    f"⭐ **You're Already Premium!** ⭐\n\n"
                    f"**User:** {user_name}\n"
                    f"**Plan:** {details.get('tier_name', 'Premium')}\n"
                    f"**Days Left:** {details.get('days_remaining', 0)}\n"
                    f"**Status:** ✅ Active\n\n"
                    "Enjoy unlimited downloads without verification! 🎬"
                )
                
                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton("🌐 OPEN WEBSITE", url=config.WEBSITE_URL)],
                    [InlineKeyboardButton("🔙 BACK", callback_data="back_to_start")]
                ])
                
                try:
                    await callback_query.message.edit_text(text, reply_markup=keyboard)
                except:
                    await callback_query.answer("You're already premium!", show_alert=True)
                return
        
        text = (
            f"⭐ **SK4FiLM PREMIUM - {user_name}** ⭐\n\n"
            "**Benefits:**\n"
            "✅ No verification required\n"
            "✅ All quality (480p-4K)\n"
            "✅ Unlimited downloads\n"
            "✅ No ads\n"
            "✅ Priority support\n\n"
            "**Plans:**\n"
            "• **Basic** - ₹99/month\n"
            "• **Premium** - ₹199/month\n"
            "• **Gold** - ₹299/2 months\n"
            "• **Diamond** - ₹499/3 months\n\n"
            "Click below to purchase:"
        )
        
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🥉 BUY BASIC (₹99)", callback_data="plan_basic")],
            [InlineKeyboardButton("🥈 BUY PREMIUM (₹199)", callback_data="plan_premium")],
            [InlineKeyboardButton("🥇 BUY GOLD (₹299)", callback_data="plan_gold")],
            [InlineKeyboardButton("💎 BUY DIAMOND (₹499)", callback_data="plan_diamond")],
            [InlineKeyboardButton("🔙 BACK", callback_data="back_to_start")]
        ])
        
        try:
            await callback_query.message.edit_text(text, reply_markup=keyboard)
        except:
            await callback_query.answer("Premium plans!", show_alert=True)
    
    @bot.on_callback_query(filters.regex(r"^plan_"))
    async def plan_selection_callback(client, callback_query):
        plan_type = callback_query.data.split('_')[1]
        user_id = callback_query.from_user.id
        
        if plan_type == "basic":
            tier = bot_instance.PremiumTier.BASIC
            plan_name = "Basic Plan"
        elif plan_type == "premium":
            tier = bot_instance.PremiumTier.PREMIUM
            plan_name = "Premium Plan"
        elif plan_type == "gold":
            tier = bot_instance.PremiumTier.GOLD
            plan_name = "Gold Plan"
        elif plan_type == "diamond":
            tier = bot_instance.PremiumTier.DIAMOND
            plan_name = "Diamond Plan"
        else:
            await callback_query.answer("Invalid plan!", show_alert=True)
            return
        
        if not bot_instance.premium_system:
            await callback_query.answer("Premium system not available!", show_alert=True)
            return
        
        # Initiate purchase
        payment_data = await bot_instance.premium_system.initiate_purchase(user_id, tier)
        
        if not payment_data:
            await callback_query.answer("Failed to initiate purchase!", show_alert=True)
            return
        
        # Get payment instructions
        instructions = await bot_instance.premium_system.get_payment_instructions_text(payment_data['payment_id'])
        
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("📸 SEND SCREENSHOT", callback_data=f"send_screenshot_{payment_data['payment_id']}")],
            [InlineKeyboardButton("🔙 BACK", callback_data="buy_premium")]
        ])
        
        try:
            await callback_query.message.edit_text(instructions, reply_markup=keyboard, disable_web_page_preview=True)
        except:
            await callback_query.answer("Payment instructions!", show_alert=True)
    
    @bot.on_callback_query(filters.regex(r"^send_screenshot_"))
    async def send_screenshot_callback(client, callback_query):
        payment_id = callback_query.data.split('_')[2]
        
        text = (
            "📸 **Please send the payment screenshot now**\n\n"
            "1. Take a clear screenshot of the payment\n"
            "2. Send it to this chat\n"
            "3. Our admin will verify and activate your premium\n\n"
            f"**Payment ID:** `{payment_id}`\n"
            "⏰ Please send within 24 hours of payment"
        )
        
        await callback_query.answer("Please send screenshot now!", show_alert=True)
        
        # Send new message
        await callback_query.message.reply_text(text)
        
        # Try to delete the original callback message
        try:
            await callback_query.message.delete()
        except:
            pass
    
    logger.info("✅ Bot handlers setup complete with ALL handlers")

# ✅ IMPORTANT: These functions need to be defined or imported
async def handle_file_request(client, message, file_text, bot_instance):
    """Handle file download request"""
    # This should be imported from your main bot file
    pass

async def handle_verification_token(client, message, token, bot_instance):
    """Handle verification token"""
    # This should be imported from your main bot file
    pass

def format_size(size_in_bytes):
    """Format file size"""
    if size_in_bytes is None or size_in_bytes == 0:
        return "0 B"
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.1f} {unit}"
        size_in_bytes /= 1024.0
    return f"{size_in_bytes:.1f} PB"
