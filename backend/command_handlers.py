import logging
from pyrogram import Client, filters
from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton, Message, CallbackQuery
from file_handlers import handle_verification_token, handle_file_request

logger = logging.getLogger(__name__)

async def setup_bot_handlers(bot: Client, bot_instance):
    """Setup bot commands and handlers - COMPLETE VERSION"""
    config = bot_instance.config
    
    # ✅ USER COMMANDS
    
    @bot.on_message(filters.command("start"))
    async def handle_start_command(client, message):
        """Handle /start command with verification token detection"""
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
    
    # ✅ HANDLE SCREENSHOT MESSAGES
    @bot.on_message(filters.private & (filters.photo | filters.document))
    async def handle_screenshot(client, message):
        """Handle payment screenshots"""
        # Check if it's likely a screenshot
        if message.photo or (message.document and message.document.mime_type and 'image' in message.document.mime_type):
            user_id = message.from_user.id
            
            if bot_instance.premium_system:
                success = await bot_instance.premium_system.process_payment_screenshot(
                    user_id, 
                    message.id
                )
                
                if success:
                    await message.reply_text(
                        "✅ **Screenshot received!**\n\n"
                        "Our admin will verify your payment and activate your premium within 24 hours.\n"
                        "Thank you for choosing SK4FiLM! 🎬\n\n"
                        "You will receive a confirmation message when activated.",
                        reply_markup=InlineKeyboardMarkup([
                            [InlineKeyboardButton("🔙 BACK TO START", callback_data="back_to_start")]
                        ])
                    )
                else:
                    await message.reply_text(
                        "❌ **No pending payment found!**\n\n"
                        "Please initiate a purchase first using /buy command."
                    )
            else:
                await message.reply_text(
                    "❌ **Premium system not available**\n\n"
                    "Please try again later or contact admin."
                )
    
    logger.info("✅ Bot handlers setup complete with ALL commands")

# Utility function for file size formatting
def format_size(size_in_bytes):
    """Format file size in human-readable format"""
    if size_in_bytes is None or size_in_bytes == 0:
        return "Unknown"
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.1f} {unit}"
        size_in_bytes /= 1024.0
    return f"{size_in_bytes:.1f} PB"
