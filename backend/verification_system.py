import os
import logging
from datetime import datetime, timedelta
from motor.motor_asyncio import AsyncIOMotorCollection
import aiohttp
import asyncio
from pyrogram import filters
from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton
import random
import string
import time
import json
from collections import defaultdict
from typing import Optional, Dict, Tuple, Any
import uuid

logger = logging.getLogger(__name__)

class VerificationSystem:
    def __init__(self, verification_col: AsyncIOMotorCollection, config):
        self.verification_col = verification_col
        self.config = config
        self.pending_verifications = {}
        self.verification_tokens = {}
        self.premium_users = {}  # Cache for premium users
        self.rate_limits = defaultdict(list)
        self.max_attempts = 5
        self.rate_limit_window = 300
        self.cleanup_interval = 60
        self.bot = None
        
        # Premium Plans Configuration
        self.premium_plans = {
            'bronze': {
                'name': 'Bronze Plan',
                'price': '10₹',
                'days': 15,
                'description': '15 days access'
            },
            'silver': {
                'name': 'Silver Plan',
                'price': '20₹',
                'days': 30,
                'description': '30 days access'
            },
            'gold': {
                'name': 'Gold Plan',
                'price': '30₹',
                'days': 90,
                'description': '90 days access'
            },
            'platinum': {
                'name': 'Platinum Plan',
                'price': '40₹',
                'days': 180,
                'description': '180 days access'
            },
            'diamond': {
                'name': 'Diamond Plan',
                'price': '50₹',
                'days': 250,
                'description': '250 days access'
            }
        }
        
        self.upi_id = "skfilmbox718186.rzp@icici"
        
        logger.info("✅ VerificationSystem initialized with Premium Support")
    
    def set_bot(self, bot):
        """Set bot instance for sending messages"""
        self.bot = bot
    
    async def initialize(self):
        """Initialize database indexes and cleanup task"""
        try:
            # Create database indexes
            await self.verification_col.create_index("user_id", unique=True)
            await self.verification_col.create_index("verification_token", unique=True, sparse=True)
            await self.verification_col.create_index("verified_at")
            await self.verification_col.create_index("expires_at")
            await self.verification_col.create_index("premium_expiry")
            await self.verification_col.create_index("premium_plan")
            
            # Start cleanup task
            asyncio.create_task(self._cleanup_expired_entries())
            
            logger.info("✅ Verification system indexes created")
        except Exception as e:
            logger.error(f"❌ Initialization error: {e}")
    
    async def _cleanup_expired_entries(self):
        """Periodically cleanup expired entries"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                # Clean expired database entries
                result = await self.verification_col.delete_many({
                    "expires_at": {"$lt": datetime.now()}
                })
                if result.deleted_count:
                    logger.info(f"🧹 Cleaned {result.deleted_count} expired verifications")
                
                # Clean expired premium entries
                premium_result = await self.verification_col.update_many(
                    {"premium_expiry": {"$lt": datetime.now()}},
                    {"$unset": {"premium_plan": "", "premium_expiry": "", "premium_purchased_at": ""}}
                )
                if premium_result.modified_count:
                    logger.info(f"🧹 Cleared {premium_result.modified_count} expired premium plans")
                
                # Clean expired pending verifications (older than 15 minutes)
                now = datetime.now()
                expired_users = []
                for user_id, data in self.pending_verifications.items():
                    if (now - data['created_at']).total_seconds() > 900:  # 15 minutes
                        expired_users.append(user_id)
                
                for user_id in expired_users:
                    if user_id in self.pending_verifications:
                        token = self.pending_verifications[user_id].get('token')
                        if token and token in self.verification_tokens:
                            del self.verification_tokens[token]
                        del self.pending_verifications[user_id]
                
                if expired_users:
                    logger.info(f"🧹 Cleaned {len(expired_users)} expired pending verifications")
                    
            except Exception as e:
                logger.error(f"Cleanup task error: {e}")
    
    def is_admin(self, user_id: int) -> bool:
        """Check if user is admin"""
        return user_id in self.config.ADMIN_IDS
    
    async def is_premium_user(self, user_id: int) -> Tuple[bool, Optional[Dict]]:
        """Check if user has active premium subscription"""
        try:
            # Check cache first
            if user_id in self.premium_users:
                return True, self.premium_users[user_id]
            
            # Check database
            verification = await self.verification_col.find_one({"user_id": user_id})
            
            if verification and verification.get('premium_expiry'):
                expiry_date = verification.get('premium_expiry')
                if isinstance(expiry_date, str):
                    expiry_date = datetime.fromisoformat(expiry_date.replace('Z', '+00:00'))
                
                if expiry_date > datetime.now():
                    premium_info = {
                        'plan': verification.get('premium_plan', 'unknown'),
                        'expires_at': expiry_date,
                        'purchased_at': verification.get('premium_purchased_at'),
                        'days_left': (expiry_date - datetime.now()).days
                    }
                    # Cache the result
                    self.premium_users[user_id] = premium_info
                    return True, premium_info
            
            return False, None
            
        except Exception as e:
            logger.error(f"Premium check error: {e}")
            return False, None
    
    async def get_user_status(self, user_id: int) -> Dict[str, Any]:
        """Get complete user status including verification and premium"""
        try:
            # Get verification status
            is_verified, status, expires_at = await self.check_verification(user_id)
            
            # Get premium status
            is_premium, premium_info = await self.is_premium_user(user_id)
            
            return {
                'user_id': user_id,
                'is_admin': self.is_admin(user_id),
                'is_verified': is_verified,
                'verification_status': status,
                'verification_expires_at': expires_at,
                'is_premium': is_premium,
                'premium_info': premium_info,
                'has_access': self.is_admin(user_id) or is_verified or is_premium
            }
            
        except Exception as e:
            logger.error(f"User status error: {e}")
            return {
                'user_id': user_id,
                'is_admin': False,
                'is_verified': False,
                'verification_status': 'error',
                'is_premium': False,
                'premium_info': None,
                'has_access': False
            }
    
    def check_rate_limit(self, user_id: int) -> Tuple[bool, Optional[int]]:
        """Check rate limit for user"""
        now = time.time()
        user_key = str(user_id)
        
        # Clean old attempts
        valid_attempts = [
            attempt for attempt in self.rate_limits[user_key]
            if now - attempt < self.rate_limit_window
        ]
        self.rate_limits[user_key] = valid_attempts
        
        if len(valid_attempts) >= self.max_attempts:
            oldest_attempt = min(valid_attempts)
            wait_time = int(self.rate_limit_window - (now - oldest_attempt))
            return False, wait_time
        
        return True, None
    
    def add_attempt(self, user_id: int):
        """Add rate limit attempt"""
        self.rate_limits[str(user_id)].append(time.time())
    
    def generate_unique_token(self) -> str:
        """Generate unique verification token using UUID"""
        token = str(uuid.uuid4()).replace('-', '')[:16]
        timestamp = int(time.time())
        return f"{token}{timestamp:08x}"[:24]
    
    async def check_verification(self, user_id: int) -> Tuple[bool, str, Optional[datetime]]:
        """Check if user is verified and return remaining time"""
        try:
            if not self.config.VERIFICATION_REQUIRED:
                return True, "verification_not_required", None
            
            if self.is_admin(user_id):
                return True, "admin_user", None
            
            # Check premium first (premium users don't need verification)
            is_premium, premium_info = await self.is_premium_user(user_id)
            if is_premium:
                return True, "premium_user", premium_info['expires_at']
            
            verification = await self.verification_col.find_one({"user_id": user_id})
            
            if not verification:
                return False, "not_verified", None
            
            # Check if verification is expired
            expires_at = verification.get('expires_at')
            if expires_at:
                if isinstance(expires_at, str):
                    expires_at = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
                
                if expires_at < datetime.now():
                    await self.verification_col.delete_one({"user_id": user_id})
                    return False, "expired", None
                else:
                    # Return remaining time
                    remaining = expires_at - datetime.now()
                    return True, "verified", expires_at
            
            # Check verified_at timestamp
            verified_at = verification.get('verified_at')
            if verified_at:
                if isinstance(verified_at, str):
                    verified_at = datetime.fromisoformat(verified_at.replace('Z', '+00:00'))
                
                # Calculate time elapsed
                time_elapsed = (datetime.now() - verified_at).total_seconds()
                if time_elapsed < self.config.VERIFICATION_DURATION:
                    expires_at = verified_at + timedelta(seconds=self.config.VERIFICATION_DURATION)
                    remaining = expires_at - datetime.now()
                    return True, "verified", expires_at
                else:
                    await self.verification_col.delete_one({"user_id": user_id})
                    return False, "expired", None
            
            return False, "not_verified", None
            
        except Exception as e:
            logger.error(f"Verification check error: {e}")
            return False, "error", None
    
    async def get_shortened_url(self, destination_url: str) -> Tuple[str, str]:
        """Get shortened URL"""
        try:
            if not hasattr(self.config, 'SHORTLINK_API') or not self.config.SHORTLINK_API:
                return destination_url, 'Direct'
            
            api_url = "https://gplinks.in/api"
            params = {
                'api': self.config.SHORTLINK_API,
                'url': destination_url
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(api_url, params=params, timeout=30) as response:
                    if response.status == 200:
                        response_text = await response.text()
                        
                        # Try JSON first
                        try:
                            data = json.loads(response_text)
                            if data.get("status") == "success":
                                return data.get('shortenedUrl', destination_url), 'GPLinks'
                        except json.JSONDecodeError:
                            # Try direct URL
                            if response_text.startswith('http'):
                                return response_text, 'GPLinks'
        
        except Exception as e:
            logger.warning(f"Shortener failed: {e}")
        
        return destination_url, 'Direct'
    
    async def create_verification_link(self, user_id: int) -> Dict[str, Any]:
        """Create verification link with unique token"""
        try:
            # Generate unique verification token
            verification_token = self.generate_unique_token()
            
            # Create Telegram deep link
            bot_username = self.config.BOT_USERNAME
            destination_url = f"https://t.me/{bot_username}?start=verify_{verification_token}"
            
            # Get shortened URL
            short_url, service_name = await self.get_shortened_url(destination_url)
            
            # Store verification data
            verification_data = {
                'user_id': user_id,
                'token': verification_token,
                'created_at': datetime.now(),
                'short_url': short_url,
                'service_name': service_name,
                'destination_url': destination_url,
                'attempts': 0,
                'status': 'pending'
            }
            
            # Store in memory caches
            self.pending_verifications[user_id] = verification_data
            self.verification_tokens[verification_token] = user_id
            
            logger.info(f"✅ Verification link created for user {user_id}")
            
            return verification_data
            
        except Exception as e:
            logger.error(f"❌ Verification link creation error: {e}")
            # Fallback
            verification_token = self.generate_unique_token()
            bot_username = self.config.BOT_USERNAME
            direct_url = f"https://t.me/{bot_username}?start=verify_{verification_token}"
            
            verification_data = {
                'user_id': user_id,
                'token': verification_token,
                'created_at': datetime.now(),
                'short_url': direct_url,
                'service_name': 'Direct',
                'destination_url': direct_url,
                'attempts': 0,
                'status': 'pending'
            }
            
            self.pending_verifications[user_id] = verification_data
            self.verification_tokens[verification_token] = user_id
            
            return verification_data
    
    async def verify_user(self, verification_token: str) -> Tuple[bool, str, Optional[int]]:
        """Verify user using verification token"""
        try:
            # Check if token exists
            if verification_token not in self.verification_tokens:
                return False, "❌ Invalid verification token. Please get a new verification link.", None
            
            user_id = self.verification_tokens[verification_token]
            
            # Check rate limit
            allowed, wait_time = self.check_rate_limit(user_id)
            if not allowed:
                return False, f"❌ Too many verification attempts. Please try again in {wait_time} seconds.", user_id
            
            # Check if user has pending verification
            if user_id not in self.pending_verifications:
                return False, "❌ No pending verification found. Please get a new verification link.", user_id
            
            pending_data = self.pending_verifications[user_id]
            
            # Check if token matches
            if pending_data['token'] != verification_token:
                self.add_attempt(user_id)
                return False, "❌ Invalid verification token. Please try again.", user_id
            
            # Check expiration (15 minutes)
            time_elapsed = (datetime.now() - pending_data['created_at']).total_seconds()
            if time_elapsed > 900:  # 15 minutes
                # Clean up expired token
                del self.verification_tokens[verification_token]
                if user_id in self.pending_verifications:
                    del self.pending_verifications[user_id]
                return False, "❌ Verification link expired. Please get a new one.", user_id
            
            # Add rate limit attempt
            self.add_attempt(user_id)
            
            # Calculate expiration time
            verified_at = datetime.now()
            expires_at = verified_at + timedelta(seconds=self.config.VERIFICATION_DURATION)
            
            # Save to database
            await self.verification_col.update_one(
                {"user_id": user_id},
                {
                    "$set": {
                        "user_id": user_id,
                        "verification_token": verification_token,
                        "verified_at": verified_at,
                        "expires_at": expires_at,
                        "verified_by": "url_shortener",
                        "created_at": datetime.now(),
                        "verification_method": "token"
                    }
                },
                upsert=True
            )
            
            # Clean up from memory caches
            del self.verification_tokens[verification_token]
            if user_id in self.pending_verifications:
                del self.pending_verifications[user_id]
            
            # Calculate remaining time
            remaining_hours = self.config.VERIFICATION_DURATION // 3600
            
            logger.info(f"🎉 User {user_id} verified successfully!")
            return True, f"🎉 **Verification Successful!**\n\nYou can now search and download movies directly! 🎬\n\n⏰ **Valid for {remaining_hours} hours**", user_id
            
        except Exception as e:
            logger.error(f"💥 Verification error: {e}")
            return False, "❌ Verification failed. Please try again.", None
    
    async def send_verification_success_message(self, user_id: int):
        """Send verification success message with search options"""
        try:
            if self.bot:
                # Get user info
                user = await self.bot.get_users(user_id)
                user_name = user.first_name or "User"
                
                await self.bot.send_message(
                    user_id,
                    f"✅ **Verification Completed, {user_name}!**\n\n"
                    "🎬 **You can now:**\n"
                    "• Search for movies directly\n"
                    "• Download files instantly\n"
                    "• Access all features\n\n"
                    "📝 **How to search:**\n"
                    "Simply send movie name to start!\n\n"
                    "⏰ **Verification valid for 6 hours**",
                    reply_markup=self._get_verified_keyboard()
                )
        except Exception as e:
            logger.error(f"Could not send success message: {e}")
    
    def _get_verified_keyboard(self) -> InlineKeyboardMarkup:
        """Get keyboard for verified users"""
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("🔍 SEARCH MOVIES", switch_inline_query_current_chat="")],
            [
                InlineKeyboardButton("📢 MAIN CHANNEL", url=self.config.MAIN_CHANNEL_LINK),
                InlineKeyboardButton("🔎 MOVIES GROUP", url=self.config.UPDATES_CHANNEL_LINK)
            ],
            [InlineKeyboardButton("🌐 VISIT WEBSITE", url=self.config.WEBSITE_URL)]
        ])
    
    def _get_unverified_keyboard(self, user_id: int, short_url: str) -> InlineKeyboardMarkup:
        """Get keyboard for unverified users with VERIFY and BUY PREMIUM options"""
        return InlineKeyboardMarkup([
            [
                InlineKeyboardButton("🔗 FREE VERIFY", url=short_url),
                InlineKeyboardButton("🎖️ BUY PREMIUM", callback_data=f"premium_plans_{user_id}")
            ],
            [InlineKeyboardButton("🔄 CHECK STATUS", callback_data=f"check_verify_{user_id}")],
            [
                InlineKeyboardButton("📢 MAIN CHANNEL", url=self.config.MAIN_CHANNEL_LINK),
                InlineKeyboardButton("🔎 MOVIES GROUP", url=self.config.UPDATES_CHANNEL_LINK)
            ]
        ])
    
    def _get_premium_plans_keyboard(self, user_id: int) -> InlineKeyboardMarkup:
        """Get keyboard for premium plans"""
        buttons = []
        
        # Add premium plan buttons
        buttons.append([InlineKeyboardButton("🎖️ AVAILABLE PLANS", callback_data="show_premium_info")])
        
        for plan_id, plan_data in self.premium_plans.items():
            button_text = f"{plan_data['price']} ➛ {plan_data['name']}"
            buttons.append([InlineKeyboardButton(button_text, callback_data=f"select_plan_{plan_id}_{user_id}")])
        
        # Add back button
        buttons.append([InlineKeyboardButton("🔙 BACK TO VERIFICATION", callback_data=f"check_verify_{user_id}")])
        
        return InlineKeyboardMarkup(buttons)
    
    def _get_payment_keyboard(self, user_id: int, plan_id: str) -> InlineKeyboardMarkup:
        """Get keyboard for payment instructions"""
        plan = self.premium_plans.get(plan_id, {})
        
        return InlineKeyboardMarkup([
            [
                InlineKeyboardButton("📸 SEND SCREENSHOT", url=f"https://t.me/{self.config.BOT_USERNAME}"),
                InlineKeyboardButton("✅ CONFIRM PAYMENT", callback_data=f"confirm_payment_{user_id}_{plan_id}")
            ],
            [InlineKeyboardButton("🔙 BACK TO PLANS", callback_data=f"premium_plans_{user_id}")]
        ])
    
    async def show_premium_plans(self, client, callback_query):
        """Show premium plans to user"""
        try:
            user_id = callback_query.from_user.id
            
            premium_message = (
                "🎖️ **ᴘʀᴇᴍɪᴜᴍ ᴘʟᴀɴꜱ**\n\n"
                "● 10₹ ➛ ʙʀᴏɴᴢᴇ ᴘʟᴀɴ » 15 ᴅᴀʏꜱ\n"
                "● 20₹ ➛ ꜱɪʟᴠᴇʀ ᴘʟᴀɴ » 30 ᴅᴀʏꜱ\n"
                "● 30₹ ➛ ɢᴏʟᴅ ᴘʟᴀɴ » 90 ᴅᴀʏꜱ\n"
                "● 40₹ ➛ ᴘʟᴀᴛɪɴᴜᴍ ᴘʟᴀɴ » 180 ᴅᴀʏꜱ\n"
                "● 50₹ ➛ ᴅɪᴀᴍᴏɴᴅ ᴘʟᴀɴ » 250 ᴅᴀʏꜱ\n\n"
                "💵 **ᴜᴘɪ ɪᴅ** - `skfilmbox718186.rzp@icici`\n\n"
                "📝 **ɪɴꜱᴛʀᴜᴄᴛɪᴏɴꜱ:**\n"
                "1. Select your plan below\n"
                "2. Send payment to the UPI ID\n"
                "3. Send screenshot to @sk4filmbot\n"
                "4. Click 'Confirm Payment' below\n\n"
                "⚡ **ɪɴꜱᴛᴀɴᴛ ᴀᴄᴛɪᴠᴀᴛɪᴏɴ!**"
            )
            
            await callback_query.message.edit_text(
                premium_message,
                reply_markup=self._get_premium_plans_keyboard(user_id)
            )
            await callback_query.answer("📋 Premium plans loaded!", show_alert=False)
            
        except Exception as e:
            logger.error(f"Premium plans error: {e}")
            await callback_query.answer("❌ Error loading plans", show_alert=True)
    
    async def select_premium_plan(self, client, callback_query):
        """Handle premium plan selection"""
        try:
            data = callback_query.data
            parts = data.split('_')
            plan_id = parts[2]
            user_id = int(parts[3])
            
            if callback_query.from_user.id != user_id:
                await callback_query.answer("❌ This is not for you!", show_alert=True)
                return
            
            plan = self.premium_plans.get(plan_id)
            if not plan:
                await callback_query.answer("❌ Invalid plan!", show_alert=True)
                return
            
            payment_message = (
                f"💳 **{plan['name']} - {plan['price']}**\n\n"
                f"📅 **Duration:** {plan['days']} days\n"
                f"📋 **Description:** {plan['description']}\n\n"
                "🔄 **Payment Steps:**\n"
                "1. Send **EXACT** amount to:\n"
                f"   💵 `{self.upi_id}`\n"
                "2. Take **screenshot** of payment\n"
                "3. Send screenshot to @sk4filmbot\n"
                "4. Click 'Confirm Payment' below\n\n"
                "⚠️ **Important:**\n"
                "• Mention your User ID in payment note\n"
                "• Keep screenshot ready\n"
                "• Activation within 5 minutes\n\n"
                f"👤 **Your User ID:** `{user_id}`"
            )
            
            await callback_query.message.edit_text(
                payment_message,
                reply_markup=self._get_payment_keyboard(user_id, plan_id)
            )
            await callback_query.answer(f"Selected {plan['name']}!", show_alert=False)
            
        except Exception as e:
            logger.error(f"Plan selection error: {e}")
            await callback_query.answer("❌ Error selecting plan", show_alert=True)
    
    async def confirm_payment(self, client, callback_query):
        """Handle payment confirmation"""
        try:
            data = callback_query.data
            parts = data.split('_')
            user_id = int(parts[2])
            plan_id = parts[3]
            
            if callback_query.from_user.id != user_id:
                await callback_query.answer("❌ This is not for you!", show_alert=True)
                return
            
            plan = self.premium_plans.get(plan_id)
            if not plan:
                await callback_query.answer("❌ Invalid plan!", show_alert=True)
                return
            
            # Here you would typically verify payment with your payment system
            # For now, we'll simulate successful payment
            
            # Calculate expiry date
            expires_at = datetime.now() + timedelta(days=plan['days'])
            
            # Update user record with premium info
            await self.verification_col.update_one(
                {"user_id": user_id},
                {
                    "$set": {
                        "premium_plan": plan_id,
                        "premium_expiry": expires_at,
                        "premium_purchased_at": datetime.now(),
                        "verified_at": datetime.now(),  # Auto verify premium users
                        "expires_at": expires_at  # Set verification to premium expiry
                    }
                },
                upsert=True
            )
            
            # Update cache
            self.premium_users[user_id] = {
                'plan': plan_id,
                'expires_at': expires_at,
                'purchased_at': datetime.now(),
                'days_left': plan['days']
            }
            
            # Send success message
            success_message = (
                f"🎉 **{plan['name']} ACTIVATED!**\n\n"
                f"✅ **Payment Confirmed**\n"
                f"💰 Amount: {plan['price']}\n"
                f"📅 Valid for: {plan['days']} days\n"
                f"⏰ Expires: {expires_at.strftime('%d %b %Y')}\n\n"
                "✨ **Premium Benefits:**\n"
                "• No verification required\n"
                "• Instant file access\n"
                "• Priority support\n"
                "• All features unlocked\n\n"
                "🎬 **Start downloading now!**"
            )
            
            await callback_query.message.edit_text(
                success_message,
                reply_markup=self._get_verified_keyboard()
            )
            
            # Notify admin about new premium purchase
            try:
                for admin_id in self.config.ADMIN_IDS:
                    await self.bot.send_message(
                        admin_id,
                        f"💰 **NEW PREMIUM PURCHASE**\n\n"
                        f"👤 User ID: `{user_id}`\n"
                        f"📋 Plan: {plan['name']} ({plan['price']})\n"
                        f"📅 Duration: {plan['days']} days\n"
                        f"⏰ Expires: {expires_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
                        f"🕐 Purchased: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )
            except Exception as e:
                logger.error(f"Admin notification failed: {e}")
            
            logger.info(f"💰 Premium activated: User {user_id}, Plan {plan['name']}")
            await callback_query.answer("✅ Premium activated successfully!", show_alert=True)
            
        except Exception as e:
            logger.error(f"Payment confirmation error: {e}")
            await callback_query.message.edit_text(
                "❌ **Payment confirmation failed!**\n\n"
                "Please make sure:\n"
                "1. You have sent the payment\n"
                "2. You have sent the screenshot\n"
                "3. Try again or contact admin\n\n"
                "👨‍💻 Contact: @sk4film",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔄 TRY AGAIN", callback_data=f"premium_plans_{callback_query.from_user.id}")]
                ])
            )
            await callback_query.answer("❌ Confirmation failed!", show_alert=True)
    
    def setup_handlers(self, bot):
        """Setup all bot handlers"""
        self.bot = bot
        
        @bot.on_callback_query(filters.regex(r"^check_verify_"))
        async def check_verify_callback(client, callback_query):
            await self._handle_check_verify(client, callback_query)
        
        @bot.on_callback_query(filters.regex(r"^premium_plans_"))
        async def premium_plans_callback(client, callback_query):
            await self.show_premium_plans(client, callback_query)
        
        @bot.on_callback_query(filters.regex(r"^select_plan_"))
        async def select_plan_callback(client, callback_query):
            await self.select_premium_plan(client, callback_query)
        
        @bot.on_callback_query(filters.regex(r"^confirm_payment_"))
        async def confirm_payment_callback(client, callback_query):
            await self.confirm_payment(client, callback_query)
        
        @bot.on_callback_query(filters.regex(r"^show_premium_info$"))
        async def show_premium_info_callback(client, callback_query):
            await self.show_premium_plans(client, callback_query)
        
        @bot.on_message(filters.command("verify") & filters.private)
        async def verify_command(client, message):
            await self._handle_verify_command(client, message)
        
        @bot.on_message(filters.command("start") & filters.private)
        async def start_verification_handler(client, message):
            await self._handle_start_command(client, message)
        
        @bot.on_message(filters.command("status") & filters.private)
        async def status_command(client, message):
            await self._handle_status_command(client, message)
        
        @bot.on_message(filters.command("premium") & filters.private)
        async def premium_command(client, message):
            await self._handle_premium_command(client, message)
        
        logger.info("✅ Verification handlers setup completed with Premium support")
    
    async def _handle_check_verify(self, client, callback_query):
        """Handle check verify callback"""
        user_id = callback_query.from_user.id
        
        try:
            is_verified, status, expires_at = await self.check_verification(user_id)
            
            if is_verified:
                # User is verified
                if status == "admin_user":
                    message_text = "👑 **You are an ADMIN!**\n\nNo verification required. You can search directly."
                elif status == "premium_user":
                    # Check premium expiry
                    is_premium, premium_info = await self.is_premium_user(user_id)
                    if is_premium:
                        days_left = premium_info['days_left']
                        plan_name = self.premium_plans.get(premium_info['plan'], {}).get('name', 'Premium')
                        message_text = f"🎖️ **PREMIUM USER - {plan_name}**\n\nValid for **{days_left}** more days!\n\nYou have full access!"
                    else:
                        message_text = "✅ **Already Verified!**\n\nYou can search movies directly!"
                else:
                    if expires_at:
                        remaining = expires_at - datetime.now()
                        hours = int(remaining.total_seconds() // 3600)
                        minutes = int((remaining.total_seconds() % 3600) // 60)
                        time_text = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"
                        message_text = f"✅ **Already Verified!**\n\nVerification valid for **{time_text}**\n\nYou can search movies directly!"
                    else:
                        message_text = "✅ **Already Verified!**\n\nYou can search movies directly!"
                
                await callback_query.message.edit_text(
                    message_text,
                    reply_markup=self._get_verified_keyboard()
                )
                await callback_query.answer("✅ Already verified!", show_alert=False)
                
            else:
                # User is NOT verified - Show VERIFY and BUY PREMIUM options
                verification_data = await self.create_verification_link(user_id)
                
                await callback_query.message.edit_text(
                    f"🔗 **Verification Required**\n\n"
                    "📱 **Choose your option:**\n\n"
                    "🆓 **FREE VERIFY:**\n"
                    "• 6 hours access\n"
                    "• Complete shortlink\n"
                    "• Quick process\n\n"
                    "🎖️ **BUY PREMIUM:**\n"
                    "• 15 to 250 days access\n"
                    "• No verification needed\n"
                    "• Instant activation\n\n"
                    "👇 **Select below:**",
                    reply_markup=self._get_unverified_keyboard(user_id, verification_data['short_url']),
                    disable_web_page_preview=True
                )
                await callback_query.answer("🔄 Options loaded!", show_alert=False)
                
        except Exception as e:
            logger.error(f"Check verify callback error: {e}")
            await callback_query.answer("❌ Error checking verification", show_alert=True)
    
    async def _handle_verify_command(self, client, message):
        """Handle /verify command"""
        user_id = message.from_user.id
        user_name = message.from_user.first_name or "User"
        
        if self.is_admin(user_id):
            await message.reply_text(
                f"👑 **Welcome Admin {user_name}!**\n\n"
                "You have **ADMIN privileges** - no verification required.\n\n"
                "You can directly search and download files! 🎬",
                reply_markup=self._get_verified_keyboard()
            )
            return
        
        # Check if user is premium
        is_premium, premium_info = await self.is_premium_user(user_id)
        if is_premium:
            days_left = premium_info['days_left']
            plan_name = self.premium_plans.get(premium_info['plan'], {}).get('name', 'Premium')
            await message.reply_text(
                f"🎖️ **Premium User - {plan_name}**\n\n"
                f"Valid for **{days_left}** more days!\n\n"
                "You have full access to all features! 🎬",
                reply_markup=self._get_verified_keyboard()
            )
            return
        
        if not self.config.VERIFICATION_REQUIRED:
            await message.reply_text(
                "ℹ️ **Verification Not Required**\n\n"
                "You can search movies directly.",
                reply_markup=self._get_verified_keyboard()
            )
            return
        
        is_verified, status, expires_at = await self.check_verification(user_id)
        
        if is_verified:
            # Verified - show direct options
            if expires_at:
                remaining = expires_at - datetime.now()
                hours = int(remaining.total_seconds() // 3600)
                minutes = int((remaining.total_seconds() % 3600) // 60)
                time_text = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"
                time_info = f"\n⏰ **Valid for {time_text}**"
            else:
                time_info = ""
            
            await message.reply_text(
                f"✅ **Already Verified, {user_name}!**\n\n"
                f"Your verification is active.{time_info}\n\n"
                "You can search movies now! 🎬",
                reply_markup=self._get_verified_keyboard()
            )
        else:
            # Not verified - show verify and premium options
            verification_data = await self.create_verification_link(user_id)
            
            await message.reply_text(
                f"🔗 **Access Required, {user_name}**\n\n"
                "📋 **Choose your access method:**\n\n"
                "🆓 **FREE VERIFICATION:**\n"
                "• 6 hours access\n"
                "• Quick shortlink process\n"
                "• Perfect for testing\n\n"
                "🎖️ **PREMIUM SUBSCRIPTION:**\n"
                "• 15-250 days access\n"
                "• No verification needed\n"
                "• Best value for money\n\n"
                "👇 **Select below to continue:**",
                reply_markup=self._get_unverified_keyboard(user_id, verification_data['short_url']),
                disable_web_page_preview=True
            )
    
    async def _handle_start_command(self, client, message):
        """Handle /start command with verification"""
        user_id = message.from_user.id
        user_name = message.from_user.first_name or "User"
        
        # Handle verification token
        if len(message.command) > 1:
            command_text = message.command[1]
            
            if command_text.startswith("verify_"):
                verification_token = command_text[7:]
                logger.info(f"🔑 Verification attempt with token: {verification_token}")
                
                success, result_message, verified_user_id = await self.verify_user(verification_token)
                
                if success:
                    # Send success message
                    await self.send_verification_success_message(verified_user_id)
                    
                    # If someone else clicked the link
                    if verified_user_id != user_id:
                        await message.reply_text(
                            f"⚠️ **Note:** This verification link was for user ID: `{verified_user_id}`\n\n"
                            f"If you need verification, use /verify command.",
                            reply_markup=InlineKeyboardMarkup([
                                [InlineKeyboardButton("🔄 GET MY ACCESS", callback_data=f"check_verify_{user_id}")]
                            ])
                        )
                else:
                    await message.reply_text(
                        result_message,
                        reply_markup=InlineKeyboardMarkup([
                            [InlineKeyboardButton("🔄 GET NEW ACCESS", callback_data=f"check_verify_{user_id}")]
                        ])
                    )
                return
        
        # Normal start command
        user_status = await self.get_user_status(user_id)
        
        if user_status['has_access']:
            # User has access (admin, verified, or premium)
            welcome_text = f"🎬 **Welcome {user_name}!**\n\n"
            
            if user_status['is_admin']:
                welcome_text += "👑 **ADMIN Access Granted**\n\n"
            elif user_status['is_premium']:
                plan_name = self.premium_plans.get(user_status['premium_info']['plan'], {}).get('name', 'Premium')
                days_left = user_status['premium_info']['days_left']
                welcome_text += f"🎖️ **PREMIUM User - {plan_name}**\n"
                welcome_text += f"⏰ **Valid for {days_left} days**\n\n"
            else:
                if user_status['verification_expires_at']:
                    remaining = user_status['verification_expires_at'] - datetime.now()
                    hours = int(remaining.total_seconds() // 3600)
                    minutes = int((remaining.total_seconds() % 3600) // 60)
                    time_text = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"
                    welcome_text += f"✅ **Verified** (Valid: {time_text})\n\n"
            
            welcome_text += "**You can:**\n• Search movies directly\n• Download instantly\n• Access all features\n\n📝 **Send movie name to start!**"
            
            await message.reply_text(
                welcome_text,
                reply_markup=self._get_verified_keyboard()
            )
        else:
            # No access - show verification and premium options
            verification_data = await self.create_verification_link(user_id)
            
            await message.reply_text(
                f"🔗 **Welcome {user_name}!**\n\n"
                "To access movies, choose an option:\n\n"
                "🆓 **Free Verification:**\n"
                "• Quick 6-hour access\n"
                "• Perfect for trying\n\n"
                "🎖️ **Premium Subscription:**\n"
                "• 15-250 days access\n"
                "• Best value\n"
                "• No verification needed\n\n"
                "👇 **Choose below:**",
                reply_markup=self._get_unverified_keyboard(user_id, verification_data['short_url'])
            )
    
    async def _handle_status_command(self, client, message):
        """Handle /status command to check verification status"""
        user_id = message.from_user.id
        user_name = message.from_user.first_name or "User"
        
        user_status = await self.get_user_status(user_id)
        
        status_text = f"📊 **Account Status - {user_name}**\n\n"
        
        if user_status['is_admin']:
            status_text += "👑 **Role:** ADMIN\n"
        elif user_status['is_premium']:
            plan_name = self.premium_plans.get(user_status['premium_info']['plan'], {}).get('name', 'Premium')
            days_left = user_status['premium_info']['days_left']
            status_text += f"🎖️ **Role:** PREMIUM ({plan_name})\n"
            status_text += f"📅 **Days Left:** {days_left} days\n"
        else:
            if user_status['is_verified']:
                if user_status['verification_expires_at']:
                    remaining = user_status['verification_expires_at'] - datetime.now()
                    hours = int(remaining.total_seconds() // 3600)
                    minutes = int((remaining.total_seconds() % 3600) // 60)
                    time_text = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"
                    status_text += f"✅ **Status:** VERIFIED\n"
                    status_text += f"⏰ **Time Left:** {time_text}\n"
                else:
                    status_text += "✅ **Status:** VERIFIED (No expiry)\n"
            else:
                status_text += "🔴 **Status:** NOT VERIFIED\n"
        
        status_text += f"🆔 **User ID:** `{user_id}`\n\n"
        
        if user_status['has_access']:
            status_text += "🎬 **Access:** FULL ACCESS GRANTED\n\n"
            status_text += "You can search and download movies!"
            
            keyboard = self._get_verified_keyboard()
        else:
            status_text += "🔒 **Access:** VERIFICATION REQUIRED\n\n"
            status_text += "Please complete verification to download."
            
            # Create new verification link
            verification_data = await self.create_verification_link(user_id)
            keyboard = self._get_unverified_keyboard(user_id, verification_data['short_url'])
        
        await message.reply_text(
            status_text,
            reply_markup=keyboard,
            disable_web_page_preview=True
        )
    
    async def _handle_premium_command(self, client, message):
        """Handle /premium command"""
        user_id = message.from_user.id
        user_name = message.from_user.first_name or "User"
        
        await self.show_premium_plans(client, message)
    
    async def show_premium_plans(self, client, message):
        """Show premium plans (can be called from command or callback)"""
        try:
            if hasattr(message, 'from_user'):
                user_id = message.from_user.id
            elif hasattr(message, 'message'):
                user_id = message.from_user.id
                message_obj = message.message
            else:
                user_id = message.chat.id
            
            premium_message = (
                "🎖️ **ᴘʀᴇᴍɪᴜᴍ ᴘʟᴀɴꜱ**\n\n"
                "● 10₹ ➛ ʙʀᴏɴᴢᴇ ᴘʟᴀɴ » 15 ᴅᴀʏꜱ\n"
                "● 20₹ ➛ ꜱɪʟᴠᴇʀ ᴘʟᴀɴ » 30 ᴅᴀʏꜱ\n"
                "● 30₹ ➛ ɢᴏʟᴅ ᴘʟᴀɴ » 90 ᴅᴀʏꜱ\n"
                "● 40₹ ➛ ᴘʟᴀᴛɪɴᴜᴍ ᴘʟᴀɴ » 180 ᴅᴀʏꜱ\n"
                "● 50₹ ➛ ᴅɪᴀᴍᴏɴᴅ ᴘʟᴀɴ » 250 ᴅᴀʏꜱ\n\n"
                "💵 **ᴜᴘɪ ɪᴅ** - `skfilmbox718186.rzp@icici`\n\n"
                "📝 **ɪɴꜱᴛʀᴜᴄᴛɪᴏɴꜱ:**\n"
                "1. Select your plan below\n"
                "2. Send payment to the UPI ID\n"
                "3. Send screenshot to @sk4filmbot\n"
                "4. Click 'Confirm Payment' below\n\n"
                "⚡ **ɪɴꜱᴛᴀɴᴛ ᴀᴄᴛɪᴠᴀᴛɪᴏɴ!**"
            )
            
            if hasattr(message, 'reply_text'):
                await message.reply_text(
                    premium_message,
                    reply_markup=self._get_premium_plans_keyboard(user_id)
                )
            else:
                await message.edit_text(
                    premium_message,
                    reply_markup=self._get_premium_plans_keyboard(user_id)
                )
            
            if hasattr(message, 'answer'):
                await message.answer("📋 Premium plans loaded!", show_alert=False)
            
        except Exception as e:
            logger.error(f"Premium plans error: {e}")
            if hasattr(message, 'answer'):
                await message.answer("❌ Error loading plans", show_alert=True)
    
    # API METHODS FOR WEB INTERFACE
    async def api_verify_user(self, request_data):
        """API endpoint for verification"""
        try:
            data = await request_data.get_json()
            user_id = data.get('user_id')
            
            if not user_id:
                return {'status': 'error', 'message': 'User ID required'}, 400
            
            verification_data = await self.create_verification_link(int(user_id))
            
            return {
                'status': 'success',
                'verification_url': verification_data['short_url'],
                'service': verification_data['service_name'],
                'user_id': user_id,
                'expires_in': '15 minutes'
            }, 200
            
        except Exception as e:
            logger.error(f"API verify error: {e}")
            return {'status': 'error', 'message': str(e)}, 500
    
    async def api_check_verification(self, user_id):
        """API endpoint to check verification status"""
        try:
            user_status = await self.get_user_status(int(user_id))
            
            return {
                'status': 'success',
                'user_id': user_id,
                'is_verified': user_status['is_verified'],
                'is_premium': user_status['is_premium'],
                'has_access': user_status['has_access'],
                'verification_status': user_status['verification_status'],
                'premium_info': user_status['premium_info']
            }, 200
            
        except Exception as e:
            logger.error(f"API check error: {e}")
            return {'status': 'error', 'message': str(e)}, 500
    
    async def api_generate_verification_url(self, user_id):
        """API endpoint to generate verification URL"""
        try:
            verification_data = await self.create_verification_link(int(user_id))
            
            return {
                'status': 'success',
                'verification_url': verification_data['short_url'],
                'direct_url': verification_data['destination_url'],
                'service': verification_data['service_name'],
                'user_id': user_id
            }, 200
            
        except Exception as e:
            logger.error(f"API generate URL error: {e}")
            return {'status': 'error', 'message': str(e)}, 500
    
    async def cleanup_expired_verifications(self):
        """Cleanup expired verifications"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Clean expired verifications
                result = await self.verification_col.delete_many({
                    "expires_at": {"$lt": datetime.now()},
                    "premium_expiry": {"$exists": False}  # Don't delete premium users
                })
                
                if result.deleted_count:
                    logger.info(f"🧹 Cleaned {result.deleted_count} expired verifications")
                    
                # Clear expired premium from cache
                now = datetime.now()
                expired_premium = []
                for user_id, premium_info in self.premium_users.items():
                    if premium_info['expires_at'] < now:
                        expired_premium.append(user_id)
                
                for user_id in expired_premium:
                    del self.premium_users[user_id]
                
                if expired_premium:
                    logger.info(f"🧹 Cleared {len(expired_premium)} expired premium users from cache")
                    
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
