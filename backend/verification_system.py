import os
import logging
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorCollection
import aiohttp
from quart import jsonify
from pyrogram import filters
from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton
import random
import string

logger = logging.getLogger(__name__)

class VerificationSystem:
    def __init__(self, verification_col: AsyncIOMotorCollection, config):
        self.verification_col = verification_col
        self.config = config
        self.pending_verifications = {}
        logger.info("✅ VerificationSystem initialized")
    
    def is_admin(self, user_id):
        return user_id in self.config.ADMIN_IDS
    
    def generate_verification_code(self):
        return ''.join(random.choices(string.digits, k=6))
    
    async def check_verification(self, user_id):
        """Check if user is verified in MongoDB"""
        try:
            if not self.config.VERIFICATION_REQUIRED:
                return True, "verification_not_required"
            
            if self.is_admin(user_id):
                return True, "admin_user"
            
            # Check in MongoDB
            verification = await self.verification_col.find_one({"user_id": user_id})
            if verification:
                verified_at = verification.get('verified_at')
                if isinstance(verified_at, datetime):
                    time_elapsed = (datetime.now() - verified_at).total_seconds()
                    if time_elapsed < self.config.VERIFICATION_DURATION:
                        return True, "verified"
                    else:
                        # Remove expired verification
                        await self.verification_col.delete_one({"user_id": user_id})
                        return False, "expired"
            return False, "not_verified"
        except Exception as e:
            logger.error(f"Verification check error: {e}")
            return False, "error"

    async def get_shortened_url(self, destination_url):
        """Get shortened URL using GPLinks"""
        try:
            api_url = f"https://{self.config.SHORTLINK_URL}/api"
            params = {
                'api': self.config.SHORTLINK_API,
                'url': destination_url
            }
            
            logger.info(f"🔄 Shortening URL: {destination_url}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(api_url, params=params, timeout=10) as response:
                    logger.info(f"📡 Shortener response status: {response.status}")
                    
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"📄 Shortener response: {data}")
                        
                        if data.get("status") == "success":
                            short_url = data.get('shortenedUrl')
                            if short_url:
                                logger.info(f"✅ Short URL generated: {short_url}")
                                return short_url, 'GPLinks'
                    
                    # If shortener fails, return original URL
                    logger.warning("❌ Shortener failed, using direct URL")
                    return destination_url, 'Direct'
                    
        except Exception as e:
            logger.error(f"💥 Shortener error: {e}")
            return destination_url, 'Direct'

    async def create_verification_link(self, user_id):
        """Create verification link with shortened URL"""
        try:
            verification_code = self.generate_verification_code()
            destination_url = f"https://t.me/{self.config.BOT_USERNAME}?start=verify_{user_id}_{verification_code}"
            
            # Get shortened URL
            short_url, service_name = await self.get_shortened_url(destination_url)
            
            # Store in pending verifications
            self.pending_verifications[user_id] = {
                'code': verification_code,
                'created_at': datetime.now(),
                'short_url': short_url,
                'service_name': service_name,
                'destination_url': destination_url
            }
            
            logger.info(f"✅ Verification link created for user {user_id}")
            return short_url, verification_code, service_name
            
        except Exception as e:
            logger.error(f"❌ Verification link creation error: {e}")
            # Fallback
            verification_code = self.generate_verification_code()
            destination_url = f"https://t.me/{self.config.BOT_USERNAME}?start=verify_{user_id}_{verification_code}"
            return destination_url, verification_code, 'Direct'

    async def verify_user(self, user_id, verification_code):
        """Verify user and save to MongoDB"""
        try:
            logger.info(f"🔍 Verifying user {user_id} with code {verification_code}")
            
            if user_id in self.pending_verifications:
                pending_data = self.pending_verifications[user_id]
                
                # Check if code is valid and not expired (10 minutes)
                if (pending_data['code'] == verification_code and 
                    (datetime.now() - pending_data['created_at']).total_seconds() < 600):
                    
                    # Save to MongoDB
                    await self.verification_col.update_one(
                        {"user_id": user_id},
                        {
                            "$set": {
                                "user_id": user_id,
                                "verified_at": datetime.now(),
                                "verified_by": "url_shortener",
                                "verification_code": verification_code,
                                "created_at": datetime.now()
                            }
                        },
                        upsert=True
                    )
                    
                    # Remove from pending
                    del self.pending_verifications[user_id]
                    
                    logger.info(f"✅ User {user_id} verified successfully")
                    return True, "🎉 **Verification Successful!**\n\nYou can now download movies! 🎬"
                
                else:
                    logger.error(f"❌ Invalid or expired code for user {user_id}")
                    return False, "❌ Invalid or expired verification code."
            
            logger.error(f"❌ No pending verification for user {user_id}")
            return False, "❌ No verification found. Please use /verify to get a new link."
            
        except Exception as e:
            logger.error(f"💥 Verification error: {e}")
            return False, "❌ Verification failed. Please try again."

    async def generate_verification_url(self, user_id):
        """Generate verification URL for user"""
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
                    
                    # Generate new verification link
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

        @bot.on_message(filters.command("start") & filters.private)
        async def start_command(client, message):
            user_id = message.from_user.id
            user_name = message.from_user.first_name or "User"
            
            logger.info(f"🔍 /start command from user {user_id}")
            
            # Check if it's a verification start
            if len(message.command) > 1:
                command_parts = message.command[1].split('_')
                if len(command_parts) >= 3 and command_parts[0] == "verify":
                    verify_user_id = int(command_parts[1])
                    verification_code = command_parts[2]
                    
                    logger.info(f"🎯 Verification start: user {verify_user_id}, code {verification_code}")
                    
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
            
            # Normal start command
            is_verified, status = await self.check_verification(user_id)
            
            if is_verified or self.is_admin(user_id):
                await message.reply_text(
                    f"🎬 **Welcome {user_name}!**\n\n"
                    "You can search and download movies.\n\n"
                    "Use /search to find movies or click below:",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("🎬 SEARCH MOVIES", url=f"https://t.me/{self.config.BOT_USERNAME}")],
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

        # Test command for debugging
        @bot.on_message(filters.command("test_verify") & filters.private)
        async def test_verify_command(client, message):
            user_id = message.from_user.id
            user_name = message.from_user.first_name or "User"
            
            # Test shortener
            test_url = f"https://t.me/{self.config.BOT_USERNAME}?start=test_123"
            short_url, service_name = await self.get_shortened_url(test_url)
            
            # Check verification status
            is_verified, status = await self.check_verification(user_id)
            
            await message.reply_text(
                f"🧪 **Test Results for {user_name}**\n\n"
                f"🔗 Shortener: `{service_name}`\n"
                f"📝 Original: `{test_url[:50]}...`\n"
                f"🎯 Shortened: `{short_url[:50]}...`\n"
                f"✅ Verified: `{is_verified}`\n"
                f"👑 Admin: `{self.is_admin(user_id)}`\n"
                f"📊 Pending: `{len(self.pending_verifications)}`",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔗 Test Short URL", url=short_url)],
                    [InlineKeyboardButton("🔄 Check Verify", callback_data=f"check_verify_{user_id}")]
                ])
            )

        logger.info("✅ Verification handlers setup completed")
