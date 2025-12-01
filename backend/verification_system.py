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
        self.rate_limits = defaultdict(list)
        self.max_attempts = 5
        self.rate_limit_window = 300
        self.cleanup_interval = 60
        self.bot = None
        logger.info("✅ VerificationSystem initialized")
    
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
        """Get keyboard for verified users (NO CHECK BUTTON)"""
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("🔍 SEARCH MOVIES", switch_inline_query_current_chat="")],
            [
                InlineKeyboardButton("📢 MAIN CHANNEL", url=self.config.MAIN_CHANNEL_LINK),
                InlineKeyboardButton("🔎 MOVIES GROUP", url=self.config.UPDATES_CHANNEL_LINK)
            ],
            [InlineKeyboardButton("🌐 VISIT WEBSITE", url=self.config.WEBSITE_URL)]
        ])
    
    def _get_unverified_keyboard(self, user_id: int, short_url: str) -> InlineKeyboardMarkup:
        """Get keyboard for unverified users (WITH CHECK BUTTON)"""
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("🔗 VERIFY NOW", url=short_url)],
            [InlineKeyboardButton("🔄 CHECK STATUS", callback_data=f"check_verify_{user_id}")],
            [
                InlineKeyboardButton("📢 MAIN CHANNEL", url=self.config.MAIN_CHANNEL_LINK),
                InlineKeyboardButton("🔎 MOVIES GROUP", url=self.config.UPDATES_CHANNEL_LINK)
            ]
        ])
    
    def setup_handlers(self, bot):
        """Setup all bot handlers"""
        self.bot = bot
        
        @bot.on_callback_query(filters.regex(r"^check_verify_"))
        async def check_verify_callback(client, callback_query):
            await self._handle_check_verify(client, callback_query)
        
        @bot.on_message(filters.command("verify") & filters.private)
        async def verify_command(client, message):
            await self._handle_verify_command(client, message)
        
        @bot.on_message(filters.command("start") & filters.private)
        async def start_verification_handler(client, message):
            await self._handle_start_command(client, message)
        
        @bot.on_message(filters.command("status") & filters.private)
        async def status_command(client, message):
            await self._handle_status_command(client, message)
        
        logger.info("✅ Verification handlers setup completed")
    
    async def _handle_check_verify(self, client, callback_query):
        """Handle check verify callback - SHOW CHECK BUTTON ONLY IF NOT VERIFIED"""
        user_id = callback_query.from_user.id
        
        try:
            is_verified, status, expires_at = await self.check_verification(user_id)
            
            if is_verified:
                # User is verified - SHOW DIRECT OPTIONS, NO CHECK BUTTON
                if status == "admin_user":
                    message_text = "👑 **You are an ADMIN!**\n\nNo verification required. You can search directly."
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
                # User is NOT verified - SHOW VERIFY + CHECK BUTTON
                verification_data = await self.create_verification_link(user_id)
                
                await callback_query.message.edit_text(
                    f"🔗 **{verification_data['service_name']} Verification Required**\n\n"
                    "📱 **Complete Verification:**\n\n"
                    "1. **Click VERIFY NOW below**\n"
                    "2. **You'll be automatically verified**\n"
                    "3. **Return to bot and start downloading**\n\n"
                    "⏰ **Link valid for 15 minutes**\n\n"
                    "🚀 **Click below to verify:**",
                    reply_markup=self._get_unverified_keyboard(user_id, verification_data['short_url']),
                    disable_web_page_preview=True
                )
                await callback_query.answer("🔄 New verification link generated!", show_alert=False)
                
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
            # Not verified - show verify options WITH CHECK BUTTON
            verification_data = await self.create_verification_link(user_id)
            
            await message.reply_text(
                f"🔗 **Verification Required, {user_name}**\n\n"
                "📋 **To download movies, complete verification:**\n\n"
                "1. **Click VERIFY NOW below**\n"
                "2. **You'll be automatically verified**\n"
                "3. **Return and start searching**\n\n"
                "⏰ **Link valid for 15 minutes**\n\n"
                "🚀 **Click below to start:**",
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
                    # Send success message with DIRECT OPTIONS
                    await self.send_verification_success_message(verified_user_id)
                    
                    # If someone else clicked the link
                    if verified_user_id != user_id:
                        await message.reply_text(
                            f"⚠️ **Note:** This verification link was for user ID: `{verified_user_id}`\n\n"
                            f"If you need verification, use /verify command.",
                            reply_markup=InlineKeyboardMarkup([
                                [InlineKeyboardButton("🔄 GET MY VERIFICATION", callback_data=f"check_verify_{user_id}")]
                            ])
                        )
                else:
                    await message.reply_text(
                        result_message,
                        reply_markup=InlineKeyboardMarkup([
                            [InlineKeyboardButton("🔄 GET NEW VERIFICATION", callback_data=f"check_verify_{user_id}")]
                        ])
                    )
                return
        
        # Normal start command
        is_verified, status, expires_at = await self.check_verification(user_id)
        
        if is_verified or self.is_admin(user_id):
            # Verified user - show DIRECT OPTIONS
            welcome_text = f"🎬 **Welcome {user_name}!**\n\n"
            
            if self.is_admin(user_id):
                welcome_text += "👑 **ADMIN Access Granted**\n\n"
            else:
                if expires_at:
                    remaining = expires_at - datetime.now()
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
            # Not verified - show verification option WITH CHECK BUTTON
            verification_data = await self.create_verification_link(user_id)
            
            await message.reply_text(
                f"🔗 **Welcome {user_name}!**\n\n"
                "To download movies, complete a quick verification.\n\n"
                "**It's just 1 click!**\n\n"
                "Click below to start verification:",
                reply_markup=self._get_unverified_keyboard(user_id, verification_data['short_url'])
            )
    
    async def _handle_status_command(self, client, message):
        """Handle /status command to check verification status"""
        user_id = message.from_user.id
        user_name = message.from_user.first_name or "User"
        
        is_verified, status, expires_at = await self.check_verification(user_id)
        
        if is_verified:
            if expires_at:
                remaining = expires_at - datetime.now()
                hours = int(remaining.total_seconds() // 3600)
                minutes = int((remaining.total_seconds() % 3600) // 60)
                time_text = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"
                time_info = f"\n⏰ **Time remaining:** {time_text}"
            else:
                time_info = ""
            
            status_text = (
                f"✅ **Verification Status - {user_name}**\n\n"
                f"🟢 **VERIFIED**\n"
                f"📊 Status: `{status}`{time_info}\n\n"
                f"You can search movies directly!"
            )
            
            keyboard = self._get_verified_keyboard()
            
        else:
            pending_info = "No pending verification"
            if user_id in self.pending_verifications:
                pending_data = self.pending_verifications[user_id]
                time_elapsed = (datetime.now() - pending_data['created_at']).total_seconds()
                pending_info = f"Pending (created {int(time_elapsed)}s ago)"
            
            status_text = (
                f"🔴 **Verification Status - {user_name}**\n\n"
                f"🆔 User ID: `{user_id}`\n"
                f"📊 Status: `{status}`\n"
                f"⏳ Pending: `{pending_info}`\n\n"
                f"Please complete verification to download."
            )
            
            # Create new verification link
            verification_data = await self.create_verification_link(user_id)
            keyboard = self._get_unverified_keyboard(user_id, verification_data['short_url'])
        
        await message.reply_text(
            status_text,
            reply_markup=keyboard,
            disable_web_page_preview=True
        )
