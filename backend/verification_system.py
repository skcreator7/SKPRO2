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
        logger.info("✅ VerificationSystem initialized with token-based verification")
    
    def set_bot(self, bot):
        self.bot = bot
    
    async def initialize(self):
        try:
            await self.verification_col.create_index("user_id", unique=True)
            await self.verification_col.create_index("verification_token", unique=True, sparse=True)
            await self.verification_col.create_index("verified_at")
            await self.verification_col.create_index("expires_at")
            
            asyncio.create_task(self._cleanup_expired_entries())
            logger.info("✅ Verification system indexes created")
        except Exception as e:
            logger.error(f"❌ Initialization error: {e}")
    
    async def _cleanup_expired_entries(self):
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                result = await self.verification_col.delete_many({
                    "expires_at": {"$lt": datetime.now()}
                })
                if result.deleted_count:
                    logger.info(f"🧹 Cleaned {result.deleted_count} expired verifications")
                
                now = datetime.now()
                expired_users = []
                for user_id, data in self.pending_verifications.items():
                    if (now - data['created_at']).total_seconds() > 900:
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
        return user_id in self.config.ADMIN_IDS
    
    def check_rate_limit(self, user_id: int) -> Tuple[bool, Optional[int]]:
        now = time.time()
        user_key = str(user_id)
        
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
        self.rate_limits[str(user_id)].append(time.time())
    
    def generate_unique_token(self) -> str:
        token = str(uuid.uuid4()).replace('-', '')[:16]
        timestamp = int(time.time())
        return f"{token}{timestamp:08x}"[:24]
    
    async def check_verification(self, user_id: int) -> Tuple[bool, str]:
        try:
            if not self.config.VERIFICATION_REQUIRED:
                return True, "verification_not_required"
            
            if self.is_admin(user_id):
                return True, "admin_user"
            
            verification = await self.verification_col.find_one({"user_id": user_id})
            
            if not verification:
                return False, "not_verified"
            
            expires_at = verification.get('expires_at')
            if expires_at:
                if isinstance(expires_at, str):
                    expires_at = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
                
                if expires_at < datetime.now():
                    await self.verification_col.delete_one({"user_id": user_id})
                    return False, "expired"
            
            verified_at = verification.get('verified_at')
            if verified_at:
                if isinstance(verified_at, str):
                    verified_at = datetime.fromisoformat(verified_at.replace('Z', '+00:00'))
                
                time_elapsed = (datetime.now() - verified_at).total_seconds()
                if time_elapsed < self.config.VERIFICATION_DURATION:
                    return True, "verified"
                else:
                    await self.verification_col.delete_one({"user_id": user_id})
                    return False, "expired"
            
            return False, "not_verified"
            
        except Exception as e:
            logger.error(f"Verification check error: {e}")
            return False, "error"
    
    async def get_shortened_url(self, destination_url: str) -> Tuple[str, str]:
        try:
            api_url = "https://gplinks.in/api"
            params = {
                'api': self.config.SHORTLINK_API,
                'url': destination_url
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(api_url, params=params, timeout=30) as response:
                    if response.status == 200:
                        response_text = await response.text()
                        
                        try:
                            data = json.loads(response_text)
                            if data.get("status") == "success":
                                short_url = data.get('shortenedUrl', destination_url)
                                return short_url, 'GPLinks'
                        except json.JSONDecodeError:
                            if response_text.startswith('http'):
                                return response_text, 'GPLinks'
        
        except Exception as e:
            logger.warning(f"Shortener failed: {e}")
        
        return destination_url, 'Direct'
    
    async def create_verification_link(self, user_id: int) -> Dict[str, Any]:
        try:
            verification_token = self.generate_unique_token()
            bot_username = self.config.BOT_USERNAME
            destination_url = f"https://t.me/{bot_username}?start=verify_{verification_token}"
            
            short_url, service_name = await self.get_shortened_url(destination_url)
            
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
            
            self.pending_verifications[user_id] = verification_data
            self.verification_tokens[verification_token] = user_id
            
            logger.info(f"✅ Verification link created for user {user_id}")
            logger.info(f"🔑 Token: {verification_token}")
            
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
        try:
            if verification_token not in self.verification_tokens:
                return False, "❌ Invalid verification token. Please get a new verification link.", None
            
            user_id = self.verification_tokens[verification_token]
            
            allowed, wait_time = self.check_rate_limit(user_id)
            if not allowed:
                return False, f"❌ Too many verification attempts. Please try again in {wait_time} seconds.", user_id
            
            if user_id not in self.pending_verifications:
                return False, "❌ No pending verification found. Please get a new verification link.", user_id
            
            pending_data = self.pending_verifications[user_id]
            
            if pending_data['token'] != verification_token:
                self.add_attempt(user_id)
                return False, "❌ Invalid verification token. Please try again.", user_id
            
            time_elapsed = (datetime.now() - pending_data['created_at']).total_seconds()
            if time_elapsed > 900:
                del self.verification_tokens[verification_token]
                if user_id in self.pending_verifications:
                    del self.pending_verifications[user_id]
                return False, "❌ Verification link expired. Please get a new one.", user_id
            
            self.add_attempt(user_id)
            
            verified_at = datetime.now()
            expires_at = verified_at + timedelta(seconds=self.config.VERIFICATION_DURATION)
            
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
            
            del self.verification_tokens[verification_token]
            if user_id in self.pending_verifications:
                del self.pending_verifications[user_id]
            
            remaining_hours = self.config.VERIFICATION_DURATION // 3600
            
            logger.info(f"🎉 User {user_id} verified successfully!")
            return True, f"🎉 **Verification Successful!**\n\nYou can now download movies! 🎬\n\n⏰ **Valid for {remaining_hours} hours**", user_id
            
        except Exception as e:
            logger.error(f"💥 Verification error: {e}")
            return False, "❌ Verification failed. Please try again.", None
    
    async def send_verification_success_notification(self, user_id: int):
        try:
            if self.bot:
                await self.bot.send_message(
                    user_id,
                    "✅ **Verification Completed!**\n\n"
                    "Your verification was successful! You can now use the bot.\n\n"
                    "Go back to the bot and start downloading! 🎬",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("🎬 START DOWNLOADING", url=f"https://t.me/{self.config.BOT_USERNAME}")]
                    ])
                )
        except Exception as e:
            logger.warning(f"Could not send notification: {e}")
    
    async def force_verify_user(self, user_id: int) -> bool:
        try:
            verified_at = datetime.now()
            expires_at = verified_at + timedelta(seconds=self.config.VERIFICATION_DURATION)
            
            await self.verification_col.update_one(
                {"user_id": user_id},
                {
                    "$set": {
                        "user_id": user_id,
                        "verified_at": verified_at,
                        "expires_at": expires_at,
                        "verified_by": "admin",
                        "created_at": datetime.now(),
                        "verification_method": "manual"
                    }
                },
                upsert=True
            )
            
            if user_id in self.pending_verifications:
                pending_data = self.pending_verifications[user_id]
                if 'token' in pending_data and pending_data['token'] in self.verification_tokens:
                    del self.verification_tokens[pending_data['token']]
                del self.pending_verifications[user_id]
            
            return True
        except Exception as e:
            logger.error(f"Force verify error: {e}")
            return False
    
    def setup_handlers(self, bot):
        self.bot = bot
        
        @bot.on_callback_query(filters.regex(r"^check_verify_"))
        async def check_verify_callback(client, callback_query):
            user_id = callback_query.from_user.id
            try:
                is_verified, status = await self.check_verification(user_id)
                
                if is_verified:
                    message_text = "✅ **Verification Successful!**\n\nYou can now download files! 🎬\n\n⏰ **Valid for 6 hours**"
                    if status == "admin_user":
                        message_text = "✅ **You are an ADMIN!**\n\nNo verification required."
                    
                    await callback_query.message.edit_text(
                        message_text,
                        reply_markup=self._get_main_menu_keyboard()
                    )
                else:
                    verification_data = await self.create_verification_link(user_id)
                    
                    await callback_query.message.edit_text(
                        f"🔗 **{verification_data['service_name']} Verification**\n\n"
                        "📱 **Complete Verification:**\n\n"
                        "1. **Click VERIFY LINK below**\n"
                        "2. **You'll be automatically verified**\n"
                        "3. **Return to bot and start downloading**\n\n"
                        "⏰ **Link valid for 15 minutes**\n\n"
                        "🚀 **Click below to verify:**",
                        reply_markup=InlineKeyboardMarkup([
                            [InlineKeyboardButton("🔗 VERIFY NOW", url=verification_data['short_url'])],
                            [InlineKeyboardButton("🔄 CHECK STATUS", callback_data=f"check_verify_{user_id}")],
                            [
                                InlineKeyboardButton("📢 MAIN CHANNEL", url=self.config.MAIN_CHANNEL_LINK),
                                InlineKeyboardButton("🔎 MOVIES GROUP", url=self.config.UPDATES_CHANNEL_LINK)
                            ]
                        ]),
                        disable_web_page_preview=True
                    )
                    
            except Exception as e:
                logger.error(f"Check verify callback error: {e}")
                await callback_query.answer("Error checking verification", show_alert=True)

        @bot.on_message(filters.command("verify") & filters.private)
        async def verify_command(client, message):
            user_id = message.from_user.id
            user_name = message.from_user.first_name or "User"
            
            if self.is_admin(user_id):
                await message.reply_text(
                    f"👑 **Welcome Admin {user_name}!**\n\n"
                    "You have **ADMIN privileges** - no verification required.\n\n"
                    "You can directly download files! 🎬",
                    reply_markup=self._get_main_menu_keyboard()
                )
                return
            
            if not self.config.VERIFICATION_REQUIRED:
                await message.reply_text(
                    "ℹ️ **Verification Not Required**\n\n"
                    "URL verification is currently disabled.\n"
                    "You can download files directly.",
                    reply_markup=self._get_main_menu_keyboard()
                )
                return
            
            is_verified, status = await self.check_verification(user_id)
            
            if is_verified:
                await message.reply_text(
                    f"✅ **Already Verified, {user_name}!**\n\n"
                    f"Your verification is active and valid for 6 hours.\n\n"
                    "You can download files now! 🎬",
                    reply_markup=self._get_main_menu_keyboard()
                )
            else:
                verification_data = await self.create_verification_link(user_id)
                
                await message.reply_text(
                    f"🔗 **{verification_data['service_name']} Verification Required, {user_name}**\n\n"
                    "📋 **Verification Process:**\n\n"
                    "1. **Click VERIFY NOW below**\n"
                    "2. **You'll be automatically verified**\n"
                    "3. **Return to bot and start downloading**\n\n"
                    "⏰ **Link valid for 15 minutes**\n\n"
                    "🚀 **Click below to start:**",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("🔗 VERIFY NOW", url=verification_data['short_url'])],
                        [InlineKeyboardButton("🔄 CHECK STATUS", callback_data=f"check_verify_{user_id}")],
                        [
                            InlineKeyboardButton("📢 MAIN CHANNEL", url=self.config.MAIN_CHANNEL_LINK),
                            InlineKeyboardButton("🔎 MOVIES GROUP", url=self.config.UPDATES_CHANNEL_LINK)
                        ]
                    ]),
                    disable_web_page_preview=True
                )

        @bot.on_message(filters.command("start") & filters.private)
        async def start_verification_handler(client, message):
            user_id = message.from_user.id
            user_name = message.from_user.first_name or "User"
            
            if len(message.command) > 1:
                command_text = message.command[1]
                
                if command_text.startswith("verify_"):
                    verification_token = command_text[7:]
                    logger.info(f"🔑 Verification token received: {verification_token}")
                    
                    success, result_message, verified_user_id = await self.verify_user(verification_token)
                    
                    if success:
                        await self.send_verification_success_notification(verified_user_id)
                        
                        await message.reply_text(
                            result_message,
                            reply_markup=InlineKeyboardMarkup([
                                [InlineKeyboardButton("🎬 START DOWNLOADING", url=f"https://t.me/{self.config.BOT_USERNAME}")],
                                [
                                    InlineKeyboardButton("📢 MAIN CHANNEL", url=self.config.MAIN_CHANNEL_LINK),
                                    InlineKeyboardButton("🔎 MOVIES GROUP", url=self.config.UPDATES_CHANNEL_LINK)
                                ]
                            ])
                        )
                        
                        if verified_user_id != user_id:
                            await message.reply_text(
                                f"⚠️ **Note:** This verification link was for a different user.\n\n"
                                f"If you need verification for yourself, use /verify command.",
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
            
            is_verified, status = await self.check_verification(user_id)
            
            if is_verified or self.is_admin(user_id):
                await message.reply_text(
                    f"🎬 **Welcome {user_name}!**\n\n"
                    "You can search and download movies.\n\n"
                    "Use our website to browse movies or click below:",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("🌐 VISIT WEBSITE", url=self.config.WEBSITE_URL)],
                        [
                            InlineKeyboardButton("📢 MAIN CHANNEL", url=self.config.MAIN_CHANNEL_LINK),
                            InlineKeyboardButton("🔎 MOVIES GROUP", url=self.config.UPDATES_CHANNEL_LINK)
                        ]
                    ])
                )
            else:
                verification_data = await self.create_verification_link(user_id)
                
                await message.reply_text(
                    f"🔗 **Welcome {user_name}!**\n\n"
                    "To download movies, you need to complete a quick verification.\n\n"
                    "**It's just 1 click!**\n\n"
                    "Click below to start verification:",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("🔗 START VERIFICATION", url=verification_data['short_url'])],
                        [
                            InlineKeyboardButton("📢 MAIN CHANNEL", url=self.config.MAIN_CHANNEL_LINK),
                            InlineKeyboardButton("🔎 MOVIES GROUP", url=self.config.UPDATES_CHANNEL_LINK)
                        ]
                    ])
                )

        @bot.on_message(filters.command("force_verify") & filters.private)
        async def force_verify_command(client, message):
            user_id = message.from_user.id
            
            if not self.is_admin(user_id):
                await message.reply_text("❌ This command is for admins only.")
                return
            
            try:
                if len(message.command) > 1:
                    target_user_id = int(message.command[1])
                    success = await self.force_verify_user(target_user_id)
                    
                    if success:
                        await message.reply_text(f"✅ User {target_user_id} has been force verified.")
                    else:
                        await message.reply_text(f"❌ Failed to verify user {target_user_id}.")
                else:
                    await message.reply_text("Usage: /force_verify <user_id>")
            except Exception as e:
                await message.reply_text(f"❌ Error: {str(e)}")

        @bot.on_message(filters.command("verify_status") & filters.private)
        async def verify_status_command(client, message):
            user_id = message.from_user.id
            user_name = message.from_user.first_name or "User"
            
            is_verified, status = await self.check_verification(user_id)
            
            pending_info = "None"
            if user_id in self.pending_verifications:
                pending_data = self.pending_verifications[user_id]
                time_elapsed = (datetime.now() - pending_data['created_at']).total_seconds()
                pending_info = f"Token: {pending_data['token'][:8]}..., Created: {int(time_elapsed)}s ago"
            
            status_message = (
                f"🔍 **Verification Status for {user_name}**\n\n"
                f"🆔 User ID: `{user_id}`\n"
                f"✅ Verified: `{is_verified}`\n"
                f"📊 Status: `{status}`\n"
                f"👑 Admin: `{self.is_admin(user_id)}`\n\n"
                f"⏳ **Pending Verification:**\n`{pending_info}`\n\n"
                f"📋 **Total Pending Users:** {len(self.pending_verifications)}"
            )
            
            keyboard = [[InlineKeyboardButton("🔄 Check Verify", callback_data=f"check_verify_{user_id}")]]
            
            if self.is_admin(user_id):
                keyboard.append([InlineKeyboardButton("👑 Admin Panel", callback_data="admin_panel")])
            
            await message.reply_text(
                status_message,
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
        
        logger.info("✅ Verification handlers setup completed")
    
    def _get_main_menu_keyboard(self) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("🎬 DOWNLOAD MOVIES", url=f"https://t.me/{self.config.BOT_USERNAME}")],
            [
                InlineKeyboardButton("📢 MAIN CHANNEL", url=self.config.MAIN_CHANNEL_LINK),
                InlineKeyboardButton("🔎 MOVIES GROUP", url=self.config.UPDATES_CHANNEL_LINK)
            ]
        ])
