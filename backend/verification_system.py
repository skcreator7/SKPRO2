import os
import logging
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorCollection
import aiohttp
from pyrogram import filters
from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton
import random
import string

logger = logging.getLogger(__name__)

# Import config
try:
    from config import config
except ImportError:
    # Fallback if config not available
    class FallbackConfig:
        SHORTLINK_API = os.getenv("SHORTLINK_API", "02178e3fdd26bbd8eae0111a7aeb8ad11557c23d")
        SHORTLINK_URL = os.getenv("SHORTLINK_URL", "api.gplinks.com")
        BOT_USERNAME = os.getenv("BOT_USERNAME", "SKadminrobot")
        MAIN_CHANNEL_LINK = os.getenv("MAIN_CHANNEL_LINK", "https://t.me/your_main_channel")
        UPDATES_CHANNEL_LINK = os.getenv("UPDATES_CHANNEL_LINK", "https://t.me/your_movies_group")
        ADMIN_IDS = [int(id.strip()) for id in os.getenv("ADMIN_IDS", "6920962552,7435781940").split(",")]
        VERIFICATION_REQUIRED = os.getenv("VERIFICATION_REQUIRED", "True").lower() == "true"
        VERIFICATION_DURATION = int(os.getenv("VERIFICATION_DURATION", "21600"))
    
    config = FallbackConfig()

# Simple working shortener function
async def get_verify_shorted_link(link):
    """Simple shortener that works with GPLinks"""
    try:
        logger.info(f"🔄 Shortening URL: {link}")
        
        # GPLinks API format
        url = f'https://{config.SHORTLINK_URL}/api'
        params = {
            'api': config.SHORTLINK_API,
            'url': link,
        }
        
        logger.info(f"📡 Calling GPLinks API: {url}")
        logger.info(f"🔑 Using API Key: {config.SHORTLINK_API[:10]}...")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, ssl=False, timeout=10) as response:
                logger.info(f"📡 API Response Status: {response.status}")
                
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"📄 API Response: {data}")
                    
                    if data.get("status") == "success":
                        short_url = data.get('shortenedUrl')
                        if short_url:
                            logger.info(f"✅ Short URL Generated: {short_url}")
                            return short_url
                        else:
                            logger.error("❌ No shortenedUrl in response")
                    else:
                        logger.error(f"❌ API Error: {data.get('message', 'Unknown error')}")
                else:
                    response_text = await response.text()
                    logger.error(f"❌ HTTP Error {response.status}: {response_text}")
        
        # If shortener fails, return original link
        logger.info("🔄 Shortener failed, using direct URL")
        return link
        
    except Exception as e:
        logger.error(f"💥 Shortener Exception: {e}")
        return link

class VerificationSystem:
    def __init__(self, verification_col: AsyncIOMotorCollection):
        self.verification_col = verification_col
        self.pending_verifications = {}
        logger.info("✅ VerificationSystem initialized successfully")
    
    def is_admin(self, user_id):
        return user_id in config.ADMIN_IDS
    
    def generate_verification_code(self):
        return ''.join(random.choices(string.digits, k=6))
    
    async def check_verification(self, user_id):
        """Check if user is verified"""
        try:
            if not config.VERIFICATION_REQUIRED:
                return True, "verification_not_required"
            
            if self.is_admin(user_id):
                return True, "admin_user"
            
            verification = await self.verification_col.find_one({"user_id": user_id})
            if verification:
                verified_at = verification.get('verified_at')
                if isinstance(verified_at, datetime):
                    time_elapsed = (datetime.now() - verified_at).total_seconds()
                    if time_elapsed < config.VERIFICATION_DURATION:
                        return True, "verified"
                    else:
                        await self.verification_col.delete_one({"user_id": user_id})
                        return False, "expired"
            return False, "not_verified"
        except Exception as e:
            logger.error(f"Verification check error: {e}")
            return False, "error"

    async def create_verification_link(self, user_id):
        """Create verification link"""
        try:
            verification_code = self.generate_verification_code()
            destination_url = f"https://t.me/{config.BOT_USERNAME}?start=verify_{user_id}_{verification_code}"
            
            logger.info(f"🔗 Creating verification for user {user_id}")
            
            # Generate short URL
            short_url = await get_verify_shorted_link(destination_url)
            
            # Store verification data
            self.pending_verifications[user_id] = {
                'code': verification_code,
                'created_at': datetime.now(),
                'short_url': short_url,
                'destination_url': destination_url
            }
            
            logger.info(f"✅ Verification created for user {user_id}")
            logger.info(f"🎯 Short URL: {short_url}")
            
            return short_url, verification_code
            
        except Exception as e:
            logger.error(f"❌ Verification link creation error: {e}")
            # Fallback
            verification_code = self.generate_verification_code()
            destination_url = f"https://t.me/{config.BOT_USERNAME}?start=verify_{user_id}_{verification_code}"
            return destination_url, verification_code

    async def verify_user(self, user_id, verification_code):
        """Verify user with code"""
        try:
            logger.info(f"🔍 Verifying user {user_id} with code {verification_code}")
            
            if user_id in self.pending_verifications:
                pending_data = self.pending_verifications[user_id]
                
                # Check if code matches and not expired (10 minutes)
                if (pending_data['code'] == verification_code and 
                    (datetime.now() - pending_data['created_at']).total_seconds() < 600):
                    
                    logger.info(f"✅ Code valid for user {user_id}")
                    
                    # Save to database
                    await self.verification_col.update_one(
                        {"user_id": user_id},
                        {
                            "$set": {
                                "verified_at": datetime.now(),
                                "verified_by": "bot",
                                "verification_code": verification_code
                            }
                        },
                        upsert=True
                    )
                    
                    # Remove from pending
                    del self.pending_verifications[user_id]
                    return True, "✅ Verification successful! You can now download files."
                
                else:
                    logger.error(f"❌ Invalid or expired code for user {user_id}")
                    return False, "❌ Invalid or expired verification code."
            
            logger.error(f"❌ No pending verification found for user {user_id}")
            return False, "❌ No verification found. Please use /verify to get a new link."
            
        except Exception as e:
            logger.error(f"💥 Verification error: {e}")
            return False, "❌ Verification failed. Please try again."

    async def generate_verification_url(self, user_id):
        """Generate verification URL"""
        try:
            if self.is_admin(user_id):
                return None
            
            short_url, verification_code = await self.create_verification_link(user_id)
            return short_url
                
        except Exception as e:
            logger.error(f"Generate verification URL error: {e}")
            return None

    def setup_handlers(self, bot):
        @bot.on_callback_query(filters.regex(r"^check_verify_"))
        async def check_verify_callback(client, callback_query):
            user_id = callback_query.from_user.id
            try:
                logger.info(f"🔄 Check verification for user {user_id}")
                
                is_verified, status = await self.check_verification(user_id)
                
                if is_verified:
                    if status == "admin_user":
                        message_text = "✅ **You are an ADMIN!**\n\nNo verification required."
                    else:
                        message_text = "✅ **Verification Successful!**\n\nYou can now download files! 🎬\n\n⏰ **Valid for 6 hours**"
                    
                    await callback_query.message.edit_text(
                        message_text,
                        reply_markup=InlineKeyboardMarkup([
                            [InlineKeyboardButton("🎬 DOWNLOAD MOVIES", url=f"https://t.me/{config.BOT_USERNAME}")],
                            [
                                InlineKeyboardButton("📢 MAIN CHANNEL", url=config.MAIN_CHANNEL_LINK),
                                InlineKeyboardButton("🔎 MOVIES GROUP", url=config.UPDATES_CHANNEL_LINK)
                            ]
                        ])
                    )
                else:
                    if self.is_admin(user_id):
                        await callback_query.message.edit_text(
                            "✅ **ADMIN Access Granted!**",
                            reply_markup=InlineKeyboardMarkup([
                                [InlineKeyboardButton("🎬 DOWNLOAD MOVIES", url=f"https://t.me/{config.BOT_USERNAME}")],
                                [
                                    InlineKeyboardButton("📢 MAIN CHANNEL", url=config.MAIN_CHANNEL_LINK),
                                    InlineKeyboardButton("🔎 MOVIES GROUP", url=config.UPDATES_CHANNEL_LINK)
                                ]
                            ])
                        )
                        return
                    
                    # Generate verification link
                    short_url, verification_code = await self.create_verification_link(user_id)
                    
                    await callback_query.message.edit_text(
                        f"🔗 **1-Click Auto-Verification**\n\n"
                        "📱 **Complete Verification in 1 Click:**\n\n"
                        "1. **Click VERIFY LINK below**\n"
                        "2. **You'll be automatically verified**\n" 
                        "3. **Return to bot and start downloading**\n\n"
                        "⏰ **Link valid for 10 minutes**\n\n"
                        "🚀 **Click below to auto-verify:**",
                        reply_markup=InlineKeyboardMarkup([
                            [InlineKeyboardButton("🔗 CLICK TO AUTO-VERIFY", url=short_url)],
                            [InlineKeyboardButton("🔄 CHECK STATUS", callback_data=f"check_verify_{user_id}")],
                            [
                                InlineKeyboardButton("📢 MAIN CHANNEL", url=config.MAIN_CHANNEL_LINK),
                                InlineKeyboardButton("🔎 MOVIES GROUP", url=config.UPDATES_CHANNEL_LINK)
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
            
            logger.info(f"🔍 /verify command from user {user_id} ({user_name})")
            
            if self.is_admin(user_id):
                await message.reply_text(
                    f"👑 **Welcome Admin {user_name}!**\n\n"
                    "You have **ADMIN privileges** - no verification required.\n\n"
                    "You can directly download files! 🎬",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("🎬 DOWNLOAD MOVIES", url=f"https://t.me/{config.BOT_USERNAME}")],
                        [
                            InlineKeyboardButton("📢 MAIN CHANNEL", url=config.MAIN_CHANNEL_LINK),
                            InlineKeyboardButton("🔎 MOVIES GROUP", url=config.UPDATES_CHANNEL_LINK)
                        ]
                    ])
                )
                return
            
            if not config.VERIFICATION_REQUIRED:
                await message.reply_text(
                    "ℹ️ **Verification Not Required**\n\n"
                    "URL verification is currently disabled.\n"
                    "You can download files directly.",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("🎬 DOWNLOAD MOVIES", url=f"https://t.me/{config.BOT_USERNAME}")],
                        [
                            InlineKeyboardButton("📢 MAIN CHANNEL", url=config.MAIN_CHANNEL_LINK),
                            InlineKeyboardButton("🔎 MOVIES GROUP", url=config.UPDATES_CHANNEL_LINK)
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
                        [InlineKeyboardButton("🎬 DOWNLOAD MOVIES", url=f"https://t.me/{config.BOT_USERNAME}")],
                        [
                            InlineKeyboardButton("📢 MAIN CHANNEL", url=config.MAIN_CHANNEL_LINK),
                            InlineKeyboardButton("🔎 MOVIES GROUP", url=config.UPDATES_CHANNEL_LINK)
                        ]
                    ])
                )
            else:
                # Generate verification link
                short_url, verification_code = await self.create_verification_link(user_id)
                
                await message.reply_text(
                    f"🔗 **1-Click Auto-Verification, {user_name}**\n\n"
                    "📋 **Just 1 Step:**\n\n"
                    "1. **Click VERIFY NOW below**\n"
                    "2. **You'll be automatically verified**\n\n"
                    "⏰ **Link valid for 10 minutes**\n\n"
                    "🚀 **Click below to start:**",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("🔗 CLICK TO AUTO-VERIFY", url=short_url)],
                        [InlineKeyboardButton("🔄 CHECK STATUS", callback_data=f"check_verify_{user_id}")],
                        [
                            InlineKeyboardButton("📢 MAIN CHANNEL", url=config.MAIN_CHANNEL_LINK),
                            InlineKeyboardButton("🔎 MOVIES GROUP", url=config.UPDATES_CHANNEL_LINK)
                        ]
                    ]),
                    disable_web_page_preview=True
                )

        @bot.on_message(filters.command("start") & filters.private)
        async def start_command(client, message):
            user_id = message.from_user.id
            user_name = message.from_user.first_name or "User"
            
            logger.info(f"🔍 /start command from user {user_id} ({user_name})")
            
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
                            "✅ **Auto-Verification Successful!**\n\n"
                            "You can now download files! 🎬\n\n"
                            "Return to the bot and use /search command.",
                            reply_markup=InlineKeyboardMarkup([
                                [InlineKeyboardButton("🎬 START DOWNLOADING", url=f"https://t.me/{config.BOT_USERNAME}")],
                                [
                                    InlineKeyboardButton("📢 MAIN CHANNEL", url=config.MAIN_CHANNEL_LINK),
                                    InlineKeyboardButton("🔎 MOVIES GROUP", url=config.UPDATES_CHANNEL_LINK)
                                ]
                            ])
                        )
                    else:
                        await message.reply_text(
                            "❌ **Verification Failed**\n\n"
                            f"{message_text}\n\n"
                            "Please use /verify to get a new link.",
                            reply_markup=InlineKeyboardMarkup([
                                [InlineKeyboardButton("🔄 GET NEW VERIFICATION", callback_data=f"check_verify_{verify_user_id}")]
                            ])
                        )
                    return
            
            # Normal start command
            if self.is_admin(user_id):
                await message.reply_text(
                    f"👑 **Welcome Admin {user_name}!**\n\n"
                    "You have **ADMIN privileges** - no verification required.\n\n"
                    "Use /search to find movies! 🎬",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("🎬 SEARCH MOVIES", url=f"https://t.me/{config.BOT_USERNAME}")],
                        [
                            InlineKeyboardButton("📢 MAIN CHANNEL", url=config.MAIN_CHANNEL_LINK),
                            InlineKeyboardButton("🔎 MOVIES GROUP", url=config.UPDATES_CHANNEL_LINK)
                        ]
                    ])
                )
            else:
                is_verified, status = await self.check_verification(user_id)
                
                if is_verified:
                    await message.reply_text(
                        f"🎬 **Welcome {user_name}!**\n\n"
                        "You are already verified and can download files!\n\n"
                        "Use /search to find movies.",
                        reply_markup=InlineKeyboardMarkup([
                            [InlineKeyboardButton("🎬 SEARCH MOVIES", url=f"https://t.me/{config.BOT_USERNAME}")],
                            [
                                InlineKeyboardButton("📢 MAIN CHANNEL", url=config.MAIN_CHANNEL_LINK),
                                InlineKeyboardButton("🔎 MOVIES GROUP", url=config.UPDATES_CHANNEL_LINK)
                            ]
                        ])
                    )
                else:
                    # Generate verification link for start button
                    short_url, verification_code = await self.create_verification_link(user_id)
                    
                    await message.reply_text(
                        f"🔗 **Welcome {user_name}!**\n\n"
                        "To download movies, you need to complete a quick verification.\n\n"
                        "**It's just 1 click!**\n\n"
                        "Click below to start verification:",
                        reply_markup=InlineKeyboardMarkup([
                            [InlineKeyboardButton("🔗 START VERIFICATION", url=short_url)],
                            [
                                InlineKeyboardButton("📢 MAIN CHANNEL", url=config.MAIN_CHANNEL_LINK),
                                InlineKeyboardButton("🔎 MOVIES GROUP", url=config.UPDATES_CHANNEL_LINK)
                            ]
                        ])
                    )

        logger.info("✅ Verification handlers setup completed")
