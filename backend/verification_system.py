import os
import logging
from datetime import datetime, timedelta
from motor.motor_asyncio import AsyncIOMotorCollection
import aiohttp
from quart import jsonify
from pyrogram import filters
from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton
import random
import string
import time
import json
from collections import defaultdict

logger = logging.getLogger(__name__)

class VerificationSystem:
    def __init__(self, verification_col: AsyncIOMotorCollection, config):
        self.verification_col = verification_col
        self.config = config
        self.pending_verifications = {}
        self.rate_limits = defaultdict(list)
        self.max_attempts = 3
        self.rate_limit_window = 300
        logger.info("✅ VerificationSystem initialized")
    
    def is_admin(self, user_id):
        return user_id in self.config.ADMIN_IDS
    
    def check_rate_limit(self, user_id):
        now = time.time()
        user_attempts = [attempt for attempt in self.rate_limits[user_id] 
                        if now - attempt < self.rate_limit_window]
        self.rate_limits[user_id] = user_attempts
        
        if len(user_attempts) >= self.max_attempts:
            return False
        return True
    
    def add_attempt(self, user_id):
        self.rate_limits[user_id].append(time.time())
    
    def generate_verification_code(self):
        return ''.join(random.choices(string.digits, k=6))
    
    async def check_verification(self, user_id):
        try:
            if not self.config.VERIFICATION_REQUIRED:
                return True, "verification_not_required"
            
            if self.is_admin(user_id):
                return True, "admin_user"
            
            verification = await self.verification_col.find_one({"user_id": user_id})
            if verification:
                verified_at = verification.get('verified_at')
                if isinstance(verified_at, datetime):
                    time_elapsed = (datetime.now() - verified_at).total_seconds()
                    if time_elapsed < self.config.VERIFICATION_DURATION:
                        return True, "verified"
                    else:
                        await self.verification_col.delete_one({"user_id": user_id})
                        return False, "expired"
                elif isinstance(verified_at, str):
                    try:
                        verified_at = datetime.fromisoformat(verified_at.replace('Z', '+00:00'))
                        time_elapsed = (datetime.now() - verified_at).total_seconds()
                        if time_elapsed < self.config.VERIFICATION_DURATION:
                            return True, "verified"
                        else:
                            await self.verification_col.delete_one({"user_id": user_id})
                            return False, "expired"
                    except:
                        await self.verification_col.delete_one({"user_id": user_id})
                        return False, "expired"
            return False, "not_verified"
        except Exception as e:
            logger.error(f"Verification check error: {e}")
            return False, "error"

    async def get_shortened_url(self, destination_url):
        try:
            api_url = "https://gplinks.in/api"
            params = {
                'api': self.config.SHORTLINK_API,
                'url': destination_url
            }
            
            logger.info(f"🔄 Shortening URL: {destination_url}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(api_url, params=params, timeout=30) as response:
                    logger.info(f"📡 Shortener response status: {response.status}")
                    
                    if response.status == 200:
                        response_text = await response.text()
                        logger.info(f"📄 Shortener raw response: {response_text}")
                        
                        try:
                            data = json.loads(response_text)
                            if data.get("status") == "success":
                                short_url = data.get('shortenedUrl')
                                if short_url:
                                    logger.info(f"✅ Short URL generated: {short_url}")
                                    return short_url, 'GPLinks'
                        except json.JSONDecodeError:
                            if response_text.startswith('http'):
                                logger.info(f"✅ Direct URL response: {response_text}")
                                return response_text, 'GPLinks'
                    
                    return destination_url, 'Direct'
                    
        except Exception as e:
            logger.error(f"💥 Shortener error: {e}")
            return destination_url, 'Direct'

    async def create_verification_link(self, user_id):
        try:
            verification_code = self.generate_verification_code()
            destination_url = f"https://t.me/{self.config.BOT_USERNAME}?start=verify_{user_id}_{verification_code}"
            
            short_url, service_name = await self.get_shortened_url(destination_url)
            
            # Store with detailed logging
            self.pending_verifications[user_id] = {
                'code': verification_code,
                'created_at': datetime.now(),
                'short_url': short_url,
                'service_name': service_name,
                'destination_url': destination_url
            }
            
            logger.info(f"✅ Verification link created for user {user_id}")
            logger.info(f"🔑 Generated Code: {verification_code}")
            logger.info(f"🔗 Short URL: {short_url}")
            logger.info(f"🎯 Destination: {destination_url}")
            logger.info(f"📋 Total pending verifications: {len(self.pending_verifications)}")
            
            return short_url, verification_code, service_name
            
        except Exception as e:
            logger.error(f"❌ Verification link creation error: {e}")
            verification_code = self.generate_verification_code()
            destination_url = f"https://t.me/{self.config.BOT_USERNAME}?start=verify_{user_id}_{verification_code}"
            return destination_url, verification_code, 'Direct'

    async def verify_user(self, user_id, verification_code):
        try:
            if not self.check_rate_limit(user_id):
                return False, "❌ Too many verification attempts. Please try again in 5 minutes."
            
            logger.info(f"🔍 Verifying user {user_id} with code {verification_code}")
            logger.info(f"📋 All pending users: {list(self.pending_verifications.keys())}")
            
            # Debug: Print all pending verification details
            for pending_user_id, pending_data in self.pending_verifications.items():
                logger.info(f"📝 Pending - User: {pending_user_id}, Code: {pending_data['code']}, Created: {pending_data['created_at']}")
            
            if user_id in self.pending_verifications:
                pending_data = self.pending_verifications[user_id]
                
                time_elapsed = (datetime.now() - pending_data['created_at']).total_seconds()
                
                logger.info(f"⏰ Time elapsed: {time_elapsed}s")
                logger.info(f"🔑 Expected code: '{pending_data['code']}' (type: {type(pending_data['code'])})")
                logger.info(f"🔑 Received code: '{verification_code}' (type: {type(verification_code)})")
                logger.info(f"✅ Codes match: {pending_data['code'] == verification_code}")
                
                if time_elapsed > 600:
                    del self.pending_verifications[user_id]
                    return False, "❌ Verification code expired. Please get a new one."
                
                # CRITICAL FIX: Strip and compare codes properly
                expected_code = str(pending_data['code']).strip()
                received_code = str(verification_code).strip()
                
                logger.info(f"🔍 After stripping - Expected: '{expected_code}', Received: '{received_code}'")
                logger.info(f"✅ Final match: {expected_code == received_code}")
                
                if expected_code == received_code:
                    self.add_attempt(user_id)
                    
                    # Save to MongoDB
                    await self.verification_col.update_one(
                        {"user_id": user_id},
                        {
                            "$set": {
                                "user_id": user_id,
                                "verified_at": datetime.now(),
                                "verified_by": "url_shortener",
                                "verification_code": verification_code,
                                "created_at": datetime.now(),
                                "expires_at": datetime.now() + timedelta(seconds=self.config.VERIFICATION_DURATION)
                            }
                        },
                        upsert=True
                    )
                    
                    # Remove from pending
                    del self.pending_verifications[user_id]
                    
                    logger.info(f"🎉 User {user_id} verified successfully!")
                    return True, "🎉 **Verification Successful!**\n\nYou can now download movies! 🎬\n\n⏰ **Valid for 6 hours**"
                else:
                    self.add_attempt(user_id)
                    remaining_attempts = self.max_attempts - len(self.rate_limits[user_id])
                    return False, f"❌ Invalid verification code. {remaining_attempts} attempts remaining."
            
            logger.error(f"❌ No pending verification found for user {user_id}")
            return False, "❌ No verification found. Please use /verify to get a new link."
            
        except Exception as e:
            logger.error(f"💥 Verification error: {e}")
            return False, "❌ Verification failed. Please try again."

    async def generate_verification_url(self, user_id):
        try:
            if self.is_admin(user_id):
                return None
            
            short_url, verification_code, service_name = await self.create_verification_link(user_id)
            return short_url
                
        except Exception as e:
            logger.error(f"Generate verification URL error: {e}")
            return None

    def setup_handlers(self, bot):
        @bot.on_callback_query(filters.regex(r"^check_verify_"))
        async def check_verify_callback(client, callback_query):
            user_id = callback_query.from_user.id
            try:
                is_verified, status = await self.check_verification(user_id)
                
                if is_verified:
                    if status == "admin_user":
                        message_text = "✅ **You are an ADMIN!**\n\nNo verification required."
                    else:
                        message_text = "✅ **Verification Successful!**\n\nYou can now download files! 🎬\n\n⏰ **Valid for 6 hours**"
                    
                    await callback_query.message.edit_text(
                        message_text,
                        reply_markup=InlineKeyboardMarkup([
                            [InlineKeyboardButton("🎬 DOWNLOAD MOVIES", url=f"https://t.me/{self.config.BOT_USERNAME}")],
                            [
                                InlineKeyboardButton("📢 MAIN CHANNEL", url=self.config.MAIN_CHANNEL_LINK),
                                InlineKeyboardButton("🔎 MOVIES GROUP", url=self.config.UPDATES_CHANNEL_LINK)
                            ]
                        ])
                    )
                else:
                    if self.is_admin(user_id):
                        await callback_query.message.edit_text(
                            "✅ **ADMIN Access Granted!**",
                            reply_markup=InlineKeyboardMarkup([
                                [InlineKeyboardButton("🎬 DOWNLOAD MOVIES", url=f"https://t.me/{self.config.BOT_USERNAME}")],
                                [
                                    InlineKeyboardButton("📢 MAIN CHANNEL", url=self.config.MAIN_CHANNEL_LINK),
                                    InlineKeyboardButton("🔎 MOVIES GROUP", url=self.config.UPDATES_CHANNEL_LINK)
                                ]
                            ])
                        )
                        return
                    
                    short_url, verification_code, service_name = await self.create_verification_link(user_id)
                    
                    await callback_query.message.edit_text(
                        f"🔗 **{service_name} Verification**\n\n"
                        "📱 **Complete Verification:**\n\n"
                        "1. **Click VERIFY LINK below**\n"
                        "2. **You'll be automatically verified**\n" 
                        "3. **Return to bot and start downloading**\n\n"
                        "⏰ **Link valid for 10 minutes**\n\n"
                        "🚀 **Click below to verify:**",
                        reply_markup=InlineKeyboardMarkup([
                            [InlineKeyboardButton("🔗 VERIFY NOW", url=short_url)],
                            [InlineKeyboardButton("🔄 CHECK STATUS", callback_data=f"check_verify_{user_id}")],
                            [
                                InlineKeyboardButton("📢 MAIN CHANNEL", url=self.config.MAIN_CHANNEL_LINK),
                                InlineKeyboardButton("🔎 MOVIES GROUP", url=self.config.UPDATES_CHANNEL_LINK)
                            ]
                        ]),
                        disable_web_page_preview=True
                    )
                    
            except Exception as e:
                logger.error(f"💥 Check verify callback error: {e}")
                await callback_query.answer("Error checking verification", show_alert=True)

        @bot.on_message(filters.command("verify") & filters.private)
        async def verify_command(client, message):
            user_id = message.from_user.id
            user_name = message.from_user.first_name or "User"
            
            logger.info(f"🔍 /verify command from user {user_id}")
            
            if self.is_admin(user_id):
                await message.reply_text(
                    f"👑 **Welcome Admin {user_name}!**\n\n"
                    "You have **ADMIN privileges** - no verification required.\n\n"
                    "You can directly download files! 🎬",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("🎬 DOWNLOAD MOVIES", url=f"https://t.me/{self.config.BOT_USERNAME}")],
                        [
                            InlineKeyboardButton("📢 MAIN CHANNEL", url=self.config.MAIN_CHANNEL_LINK),
                            InlineKeyboardButton("🔎 MOVIES GROUP", url=self.config.UPDATES_CHANNEL_LINK)
                        ]
                    ])
                )
                return
            
            if not self.config.VERIFICATION_REQUIRED:
                await message.reply_text(
                    "ℹ️ **Verification Not Required**\n\n"
                    "URL verification is currently disabled.\n"
                    "You can download files directly.",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("🎬 DOWNLOAD MOVIES", url=f"https://t.me/{self.config.BOT_USERNAME}")],
                        [
                            InlineKeyboardButton("📢 MAIN CHANNEL", url=self.config.MAIN_CHANNEL_LINK),
                            InlineKeyboardButton("🔎 MOVIES GROUP", url=self.config.UPDATES_CHANNEL_LINK)
                        ]
                    ])
                )
                return
            
            is_verified, status = await self.check_verification(user_id)
            
            if is_verified:
                await message.reply_text(
                    f"✅ **Already Verified, {user_name}!**\n\n"
                    f"Your verification is active and valid for 6 hours.\n\n"
                    "You can download files now! 🎬",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("🎬 DOWNLOAD MOVIES", url=f"https://t.me/{self.config.BOT_USERNAME}")],
                        [
                            InlineKeyboardButton("📢 MAIN CHANNEL", url=self.config.MAIN_CHANNEL_LINK),
                            InlineKeyboardButton("🔎 MOVIES GROUP", url=self.config.UPDATES_CHANNEL_LINK)
                        ]
                    ])
                )
            else:
                short_url, verification_code, service_name = await self.create_verification_link(user_id)
                
                await message.reply_text(
                    f"🔗 **{service_name} Verification Required, {user_name}**\n\n"
                    "📋 **Verification Process:**\n\n"
                    "1. **Click VERIFY NOW below**\n"
                    "2. **You'll be automatically verified**\n"
                    "3. **Return to bot and start downloading**\n\n"
                    "⏰ **Link valid for 10 minutes**\n\n"
                    "🚀 **Click below to start:**",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("🔗 VERIFY NOW", url=short_url)],
                        [InlineKeyboardButton("🔄 CHECK STATUS", callback_data=f"check_verify_{user_id}")],
                        [
                            InlineKeyboardButton("📢 MAIN CHANNEL", url=self.config.MAIN_CHANNEL_LINK),
                            InlineKeyboardButton("🔎 MOVIES GROUP", url=self.config.UPDATES_CHANNEL_LINK)
                        ]
                    ]),
                    disable_web_page_preview=True
                )

        # CRITICAL FIX: Enhanced start handler for verification
        @bot.on_message(filters.command("start") & filters.private)
        async def start_verification_handler(client, message):
            user_id = message.from_user.id
            user_name = message.from_user.first_name or "User"
            
            logger.info(f"🔍 /start command from user {user_id}")
            
            # Handle verification start
            if len(message.command) > 1:
                command_text = message.command[1]
                logger.info(f"🎯 Processing command: {command_text}")
                
                if command_text.startswith("verify_"):
                    try:
                        parts = command_text.split('_')
                        logger.info(f"📋 Command parts: {parts}")
                        
                        if len(parts) >= 3:
                            verify_user_id = int(parts[1])
                            verification_code = parts[2]
                            
                            logger.info(f"🎯 Verification attempt:")
                            logger.info(f"   👤 User ID: {verify_user_id}")
                            logger.info(f"   🔑 Received Code: {verification_code}")
                            logger.info(f"   📋 Pending users: {list(self.pending_verifications.keys())}")
                            
                            # Debug: Print current pending verification for this user
                            if verify_user_id in self.pending_verifications:
                                pending_data = self.pending_verifications[verify_user_id]
                                logger.info(f"   📝 Pending data - Code: {pending_data['code']}, Created: {pending_data['created_at']}")
                            
                            # Auto-verify the user
                            is_verified, message_text = await self.verify_user(verify_user_id, verification_code)
                            
                            if is_verified:
                                await message.reply_text(
                                    message_text,
                                    reply_markup=InlineKeyboardMarkup([
                                        [InlineKeyboardButton("🎬 START DOWNLOADING", url=f"https://t.me/{self.config.BOT_USERNAME}")],
                                        [
                                            InlineKeyboardButton("📢 MAIN CHANNEL", url=self.config.MAIN_CHANNEL_LINK),
                                            InlineKeyboardButton("🔎 MOVIES GROUP", url=self.config.UPDATES_CHANNEL_LINK)
                                        ]
                                    ])
                                )
                            else:
                                await message.reply_text(
                                    f"❌ **Verification Failed**\n\n{message_text}",
                                    reply_markup=InlineKeyboardMarkup([
                                        [InlineKeyboardButton("🔄 GET NEW VERIFICATION", callback_data=f"check_verify_{verify_user_id}")]
                                    ])
                                )
                            return
                        else:
                            logger.error(f"❌ Invalid verification format: {parts}")
                            await message.reply_text("❌ **Invalid verification link**")
                    except Exception as e:
                        logger.error(f"💥 Verification start error: {e}")
                        await message.reply_text("❌ **Verification error. Please try again.**")
                    return
            
            # Normal start command
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
                short_url, verification_code, service_name = await self.create_verification_link(user_id)
                
                await message.reply_text(
                    f"🔗 **Welcome {user_name}!**\n\n"
                    "To download movies, you need to complete a quick verification.\n\n"
                    "**It's just 1 click!**\n\n"
                    "Click below to start verification:",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("🔗 START VERIFICATION", url=short_url)],
                        [
                            InlineKeyboardButton("📢 MAIN CHANNEL", url=self.config.MAIN_CHANNEL_LINK),
                            InlineKeyboardButton("🔎 MOVIES GROUP", url=self.config.UPDATES_CHANNEL_LINK)
                        ]
                    ])
                )

        @bot.on_message(filters.command("debug_verify") & filters.private)
        async def debug_verify_command(client, message):
            user_id = message.from_user.id
            user_name = message.from_user.first_name or "User"
            
            is_verified, status = await self.check_verification(user_id)
            
            pending_info = "No pending verification"
            if user_id in self.pending_verifications:
                pending_data = self.pending_verifications[user_id]
                time_elapsed = (datetime.now() - pending_data['created_at']).total_seconds()
                pending_info = f"Code: '{pending_data['code']}', Created: {time_elapsed:.0f}s ago"
            
            await message.reply_text(
                f"🐛 **Debug Info for {user_name}**\n\n"
                f"🆔 User ID: `{user_id}`\n"
                f"✅ Verified: `{is_verified}`\n"
                f"📊 Status: `{status}`\n"
                f"👑 Admin: `{self.is_admin(user_id)}`\n"
                f"⏳ Pending: `{pending_info}`\n"
                f"📋 All pending users: `{list(self.pending_verifications.keys())}`\n\n"
                f"🔑 Pending codes:\n" + 
                "\n".join([f"• {uid}: '{data['code']}'" for uid, data in self.pending_verifications.items()]),
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔄 Check Verify", callback_data=f"check_verify_{user_id}")],
                    [InlineKeyboardButton("🔗 Get New Link", callback_data=f"get_new_link_{user_id}")]
                ])
            )

        logger.info("✅ Verification handlers setup completed")
