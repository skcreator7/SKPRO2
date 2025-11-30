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
        self.rate_limit_window = 300  # 5 minutes
        logger.info("✅ VerificationSystem initialized")
    
    def is_admin(self, user_id):
        return user_id in self.config.ADMIN_IDS
    
    def check_rate_limit(self, user_id):
        """Check if user is rate limited"""
        now = time.time()
        user_attempts = [attempt for attempt in self.rate_limits[user_id] 
                        if now - attempt < self.rate_limit_window]
        self.rate_limits[user_id] = user_attempts
        
        if len(user_attempts) >= self.max_attempts:
            return False
        return True
    
    def add_attempt(self, user_id):
        """Add rate limit attempt"""
        self.rate_limits[user_id].append(time.time())
    
    def generate_verification_code(self):
        return ''.join(random.choices(string.digits, k=6))
    
    async def check_verification(self, user_id):
        """Check if user is verified in MongoDB with enhanced error handling"""
        try:
            if not self.config.VERIFICATION_REQUIRED:
                return True, "verification_not_required"
            
            if self.is_admin(user_id):
                return True, "admin_user"
            
            # Check in MongoDB
            verification = await self.verification_col.find_one({"user_id": user_id})
            if verification:
                verified_at = verification.get('verified_at')
                expires_at = verification.get('expires_at')
                
                # Handle different date formats
                if isinstance(verified_at, str):
                    try:
                        verified_at = datetime.fromisoformat(verified_at.replace('Z', '+00:00'))
                    except:
                        verified_at = None
                
                if isinstance(expires_at, str):
                    try:
                        expires_at = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
                    except:
                        expires_at = None
                
                if verified_at:
                    # Use expires_at if available, otherwise calculate from verified_at
                    if expires_at:
                        if datetime.now() < expires_at:
                            return True, "verified"
                        else:
                            await self.verification_col.delete_one({"user_id": user_id})
                            return False, "expired"
                    else:
                        # Fallback to duration calculation
                        time_elapsed = (datetime.now() - verified_at).total_seconds()
                        if time_elapsed < self.config.VERIFICATION_DURATION:
                            return True, "verified"
                        else:
                            await self.verification_col.delete_one({"user_id": user_id})
                            return False, "expired"
            
            return False, "not_verified"
        except Exception as e:
            logger.error(f"Verification check error: {e}")
            # In case of error, assume not verified for security
            return False, "error"

    async def get_shortened_url(self, destination_url):
        """Get shortened URL using GPLinks with proper API integration"""
        try:
            # GPLinks API endpoint
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
                            # Try to parse as JSON
                            data = json.loads(response_text)
                            
                            if data.get("status") == "success" or "shortenedUrl" in data:
                                short_url = data.get('shortenedUrl') or data.get('shortened_url')
                                if short_url:
                                    logger.info(f"✅ Short URL generated: {short_url}")
                                    return short_url, 'GPLinks'
                            
                        except json.JSONDecodeError:
                            # If response is not JSON, check if it's a direct URL
                            if response_text.startswith('http'):
                                logger.info(f"✅ Direct URL response: {response_text}")
                                return response_text, 'GPLinks'
                    
                    # Alternative GPLinks API format
                    alternative_url = f"https://gplinks.in/api?api={self.config.SHORTLINK_API}&url={destination_url}"
                    async with session.get(alternative_url, timeout=30) as alt_response:
                        if alt_response.status == 200:
                            alt_text = await alt_response.text()
                            if alt_text.startswith('http'):
                                logger.info(f"✅ Alternative short URL: {alt_text}")
                                return alt_text, 'GPLinks'
                    
                    # If all shortener methods fail, return original URL
                    logger.warning("❌ All shortener methods failed, using direct URL")
                    return destination_url, 'Direct'
                    
        except aiohttp.ClientError as e:
            logger.error(f"💥 Shortener connection error: {e}")
        except asyncio.TimeoutError:
            logger.error("💥 Shortener timeout error")
        except Exception as e:
            logger.error(f"💥 Shortener unexpected error: {e}")
        
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
        """Verify user with rate limiting and security checks"""
        try:
            # Check rate limit
            if not self.check_rate_limit(user_id):
                return False, "❌ Too many verification attempts. Please try again in 5 minutes."
            
            logger.info(f"🔍 Verifying user {user_id} with code {verification_code}")
            
            if user_id in self.pending_verifications:
                pending_data = self.pending_verifications[user_id]
                
                # Check if code is valid and not expired (10 minutes)
                time_elapsed = (datetime.now() - pending_data['created_at']).total_seconds()
                
                if time_elapsed > 600:  # 10 minutes
                    del self.pending_verifications[user_id]
                    return False, "❌ Verification code expired. Please get a new one."
                
                if pending_data['code'] == verification_code:
                    # Add rate limit attempt
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
                    
                    logger.info(f"✅ User {user_id} verified successfully")
                    return True, "🎉 **Verification Successful!**\n\nYou can now download movies! 🎬\n\n⏰ **Valid for 6 hours**"
                else:
                    # Add failed attempt
                    self.add_attempt(user_id)
                    remaining_attempts = self.max_attempts - len(self.rate_limits[user_id])
                    return False, f"❌ Invalid verification code. {remaining_attempts} attempts remaining."
            
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

    async def cleanup_expired_verifications(self):
        """Clean up expired verifications from MongoDB"""
        try:
            result = await self.verification_col.delete_many({
                "expires_at": {"$lt": datetime.now()}
            })
            if result.deleted_count > 0:
                logger.info(f"🧹 Cleaned up {result.deleted_count} expired verifications")
            
            # Also clean up old pending verifications
            now = datetime.now()
            expired_pending = []
            for user_id, data in self.pending_verifications.items():
                if (now - data['created_at']).total_seconds() > 600:  # 10 minutes
                    expired_pending.append(user_id)
            
            for user_id in expired_pending:
                del self.pending_verifications[user_id]
            
            if expired_pending:
                logger.info(f"🧹 Cleaned up {len(expired_pending)} expired pending verifications")
                
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

    # API Methods for Quart routes
    async def api_verify_user(self, request):
        """API endpoint to verify user"""
        try:
            data = await request.get_json()
            user_id = data.get('user_id')
            verification_code = data.get('verification_code')
            
            if not user_id or not verification_code:
                return jsonify({'status': 'error', 'message': 'Missing user_id or verification_code'}), 400
            
            is_verified, message = await self.verify_user(int(user_id), verification_code)
            
            return jsonify({
                'status': 'success' if is_verified else 'error',
                'message': message,
                'verified': is_verified
            })
            
        except Exception as e:
            logger.error(f"API verification error: {e}")
            return jsonify({'status': 'error', 'message': 'Internal server error'}), 500

    async def api_check_verification(self, user_id):
        """API endpoint to check verification status"""
        try:
            is_verified, status = await self.check_verification(int(user_id))
            
            return jsonify({
                'status': 'success',
                'verified': is_verified,
                'verification_status': status,
                'is_admin': self.is_admin(int(user_id))
            })
            
        except Exception as e:
            logger.error(f"API check verification error: {e}")
            return jsonify({'status': 'error', 'message': 'Internal server error'}), 500

    async def api_generate_verification_url(self, user_id):
        """API endpoint to generate verification URL"""
        try:
            url = await self.generate_verification_url(int(user_id))
            
            return jsonify({
                'status': 'success',
                'verification_url': url,
                'verification_required': self.config.VERIFICATION_REQUIRED
            })
            
        except Exception as e:
            logger.error(f"API generate URL error: {e}")
            return jsonify({'status': 'error', 'message': 'Internal server error'}), 500

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

        @bot.on_message(filters.command("verification_status") & filters.private)
        async def verification_status_command(client, message):
            user_id = message.from_user.id
            user_name = message.from_user.first_name or "User"
            
            is_verified, status = await self.check_verification(user_id)
            
            status_messages = {
                "verified": "✅ **VERIFIED**\nYour verification is active",
                "expired": "❌ **EXPIRED**\nYour verification has expired",
                "not_verified": "❌ **NOT VERIFIED**\nYou need to verify first",
                "admin_user": "👑 **ADMIN**\nNo verification required",
                "verification_not_required": "ℹ️ **NOT REQUIRED**\nVerification is disabled",
                "error": "⚠️ **ERROR**\nCould not check status"
            }
            
            status_text = status_messages.get(status, "❓ **UNKNOWN STATUS**")
            
            # Get verification expiry info if verified
            expiry_info = ""
            if is_verified and status == "verified":
                verification = await self.verification_col.find_one({"user_id": user_id})
                if verification:
                    expires_at = verification.get('expires_at')
                    if expires_at:
                        if isinstance(expires_at, str):
                            expires_at = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
                        time_left = expires_at - datetime.now()
                        hours_left = int(time_left.total_seconds() / 3600)
                        minutes_left = int((time_left.total_seconds() % 3600) / 60)
                        expiry_info = f"\n⏰ **Expires in:** {hours_left}h {minutes_left}m"
            
            await message.reply_text(
                f"🔍 **Verification Status for {user_name}**\n\n"
                f"{status_text}{expiry_info}\n\n"
                f"🆔 User ID: `{user_id}`\n"
                f"👑 Admin: `{self.is_admin(user_id)}`",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔄 Refresh", callback_data=f"check_verify_{user_id}")],
                    [InlineKeyboardButton("📊 System Info", callback_data="system_info")]
                ])
            )
        
        # Add callback for system info
        @bot.on_callback_query(filters.regex(r"^system_info$"))
        async def system_info_callback(client, callback_query):
            user_id = callback_query.from_user.id
            
            total_verified = await self.verification_col.count_documents({})
            total_pending = len(self.pending_verifications)
            
            await callback_query.message.edit_text(
                f"📊 **Verification System Info**\n\n"
                f"✅ Total Verified: `{total_verified}` users\n"
                f"⏳ Pending: `{total_pending}` verifications\n"
                f"🔗 Shortener: `{'ENABLED' if self.config.SHORTLINK_API else 'DISABLED'}`\n"
                f"⏰ Duration: `{self.config.VERIFICATION_DURATION//3600}` hours\n"
                f"🛡️ Required: `{self.config.VERIFICATION_REQUIRED}`",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 Back", callback_data=f"check_verify_{user_id}")]
                ])
            )

        logger.info("✅ Verification handlers setup completed")
