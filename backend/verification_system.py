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
        self.pending_verifications = {}  # Store pending verifications
    
    def is_admin(self, user_id):
        """Check if user is admin"""
        return user_id in self.config.ADMIN_IDS
    
    def generate_verification_code(self):
        """Generate random verification code"""
        return ''.join(random.choices(string.digits, k=6))
    
    async def check_url_shortener_verification(self, user_id):
        """Check if user is verified via URL shortener"""
        if not self.config.VERIFICATION_REQUIRED:
            return True, "verification_not_required"
        
        # Admin users are always verified
        if self.is_admin(user_id):
            return True, "admin_user"
        
        try:
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
            return False, "not_verified"
        except Exception as e:
            logger.error(f"Verification check error: {e}")
            return False, "error"

    async def create_gplink_verification(self, user_id):
        """Create GP Link verification for user"""
        try:
            # Generate verification code
            verification_code = self.generate_verification_code()
            
            # Create GP Link with verification code
            gplink_url = await self.generate_gplink_with_code(verification_code, user_id)
            
            # Store verification data temporarily
            self.pending_verifications[user_id] = {
                'code': verification_code,
                'created_at': datetime.now(),
                'gplink_url': gplink_url
            }
            
            return gplink_url, verification_code
            
        except Exception as e:
            logger.error(f"GP Link creation error: {e}")
            return None, None

    async def generate_gplink_with_code(self, verification_code, user_id):
        """Generate GP Link with verification code"""
        try:
            # GP Link API configuration
            api_key = self.config.URL_SHORTENER_KEY
            base_url = "https://gplinks.in/api"
            
            # Create destination URL with verification data
            destination_url = f"https://t.me/{self.config.BOT_USERNAME}?start=verify_{user_id}_{verification_code}"
            
            # Shorten with GP Link
            async with aiohttp.ClientSession() as session:
                payload = {
                    'url': destination_url,
                    'api_key': api_key
                }
                
                async with session.post(
                    f"{base_url}/shorten",
                    json=payload,
                    timeout=10
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        if result.get('status') == 'success':
                            shortened_url = result.get('shortenedUrl')
                            logger.info(f"✅ GP Link created: {shortened_url}")
                            return shortened_url
                    else:
                        logger.error(f"GP Link API error: {response.status}")
                        
        except Exception as e:
            logger.error(f"GP Link generation failed: {e}")
        
        # Fallback: Create direct bot link
        return f"https://t.me/{self.config.BOT_USERNAME}?start=verify_{user_id}_{verification_code}"

    async def verify_user_with_code(self, user_id, verification_code):
        """Verify user using verification code"""
        try:
            # Check if verification exists and is valid
            if user_id in self.pending_verifications:
                pending_data = self.pending_verifications[user_id]
                
                # Check if code matches and is not expired (10 minutes)
                if (pending_data['code'] == verification_code and 
                    (datetime.now() - pending_data['created_at']).total_seconds() < 600):
                    
                    # Mark user as verified
                    await self.verification_col.update_one(
                        {"user_id": user_id},
                        {
                            "$set": {
                                "verified_at": datetime.now(),
                                "verified_by": "gplink_shortener",
                                "verification_code": verification_code
                            }
                        },
                        upsert=True
                    )
                    
                    # Remove from pending
                    del self.pending_verifications[user_id]
                    
                    return True, "verified_successfully"
            
            return False, "invalid_or_expired_code"
            
        except Exception as e:
            logger.error(f"Verification error: {e}")
            return False, "error"

    async def process_verification_start(self, user_id, verification_code):
        """Process verification when user clicks GP Link"""
        is_verified, message = await self.verify_user_with_code(user_id, verification_code)
        return is_verified, message

    # ADD THIS MISSING METHOD
    async def generate_verification_url(self, user_id):
        """Generate verification URL - wrapper for create_gplink_verification"""
        try:
            # Admin users don't need verification
            if self.is_admin(user_id):
                return None
            
            gplink_url, verification_code = await self.create_gplink_verification(user_id)
            return gplink_url
            
        except Exception as e:
            logger.error(f"Generate verification URL error: {e}")
            return None

    async def api_verify_user(self, request):
        """API endpoint to verify user"""
        try:
            data = await request.get_json()
            user_id = data.get('user_id')
            verification_code = data.get('verification_code')
            
            if not user_id or not verification_code:
                return jsonify({'status': 'error', 'message': 'User ID and verification code required'}), 400
            
            # Check if admin
            if self.is_admin(user_id):
                return jsonify({
                    'status': 'success',
                    'verified': True,
                    'message': 'admin_auto_verified',
                    'user_id': user_id
                })
            
            is_verified, message = await self.verify_user_with_code(user_id, verification_code)
            
            return jsonify({
                'status': 'success' if is_verified else 'error',
                'verified': is_verified,
                'message': message,
                'user_id': user_id
            })
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500

    async def api_check_verification(self, user_id):
        """API endpoint to check verification status"""
        try:
            is_verified, message = await self.check_url_shortener_verification(user_id)
            return jsonify({
                'status': 'success',
                'verified': is_verified,
                'message': message,
                'user_id': user_id,
                'is_admin': self.is_admin(user_id),
                'verification_required': self.config.VERIFICATION_REQUIRED
            })
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500

    async def api_generate_verification_url(self, user_id):
        """API endpoint to generate verification URL"""
        try:
            # Admin users don't need verification
            if self.is_admin(user_id):
                return jsonify({
                    'status': 'success',
                    'verification_url': None,
                    'user_id': user_id,
                    'message': 'admin_no_verification_needed'
                })
            
            gplink_url, verification_code = await self.create_gplink_verification(user_id)
            
            if gplink_url:
                return jsonify({
                    'status': 'success',
                    'verification_url': gplink_url,
                    'verification_code': verification_code,
                    'user_id': user_id,
                    'is_gplink': 'gplinks.in' in gplink_url
                })
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'failed_to_generate_verification'
                })
                
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500

    def setup_bot_handlers(self, bot, User, flood_protection):
        """Setup bot handlers for verification"""
        
        @bot.on_callback_query(filters.regex(r"^check_verify_"))
        async def check_verify_callback(client, callback_query):
            user_id = callback_query.from_user.id
            try:
                is_verified, status = await self.check_url_shortener_verification(user_id)
                
                if is_verified:
                    if status == "admin_user":
                        message_text = "✅ **You are an ADMIN!**\n\nNo verification required. You can download files directly."
                    else:
                        message_text = "✅ **GP Link Verification Successful!**\n\nYou Can Now Download Files! 🎬\n\n⏰ **Verification Valid For 6 Hours**"
                    
                    await callback_query.message.edit_text(
                        message_text,
                        reply_markup=InlineKeyboardMarkup([
                            [InlineKeyboardButton("🎬 DOWNLOAD MOVIES", url=f"https://t.me/{self.config.BOT_USERNAME}")],
                            [
                                InlineKeyboardButton("📢 Mᴀɪɴ Cʜᴀɴɴᴇʟ", url=self.config.MAIN_CHANNEL_LINK),
                                InlineKeyboardButton("🔎 Mᴏᴠɪᴇꜱ Gʀᴏᴜᴘ", url=self.config.UPDATES_CHANNEL_LINK)
                            ]
                        ])
                    )
                else:
                    # Check if user is admin (shouldn't happen but just in case)
                    if self.is_admin(user_id):
                        await callback_query.message.edit_text(
                            "✅ **ADMIN Access Granted!**\n\nYou can download files directly.",
                            reply_markup=InlineKeyboardMarkup([
                                [InlineKeyboardButton("🎬 DOWNLOAD MOVIES", url=f"https://t.me/{self.config.BOT_USERNAME}")],
                                [
                                    InlineKeyboardButton("📢 Mᴀɪɴ Cʜᴀɴɴᴇʟ", url=self.config.MAIN_CHANNEL_LINK),
                                    InlineKeyboardButton("🔎 Mᴏᴠɪᴇꜱ Gʀᴏᴜᴘ", url=self.config.UPDATES_CHANNEL_LINK)
                                ]
                            ])
                        )
                        return
                    
                    gplink_url, verification_code = await self.create_gplink_verification(user_id)
                    
                    if gplink_url:
                        await callback_query.message.edit_text(
                            "🔗 **GP Link Verification Required**\n\n"
                            "📱 **Complete Verification in 3 Steps:**\n\n"
                            "1. **Click GP LINK below**\n"
                            "2. **Complete GP Link action**\n" 
                            "3. **Return to bot automatically**\n\n"
                            "⏰ **Code valid for 10 minutes**\n"
                            f"🔢 **Your Code:** `{verification_code}`\n\n"
                            "🚀 **Click below to verify:**",
                            reply_markup=InlineKeyboardMarkup([
                                [InlineKeyboardButton("🔗 VERIFY WITH GP LINK", url=gplink_url)],
                                [InlineKeyboardButton("🔄 CHECK VERIFICATION", callback_data=f"check_verify_{user_id}")],
                                [
                                    InlineKeyboardButton("📢 Mᴀɪɴ Cʜᴀɴɴᴇʟ", url=self.config.MAIN_CHANNEL_LINK),
                                    InlineKeyboardButton("🔎 Mᴏᴠɪᴇꜱ Gʀᴏᴜᴘ", url=self.config.UPDATES_CHANNEL_LINK)
                                ]
                            ]),
                            disable_web_page_preview=True
                        )
                    else:
                        await callback_query.message.edit_text(
                            "❌ **Failed to generate verification link**\n\nPlease try again later.",
                            reply_markup=InlineKeyboardMarkup([
                                [InlineKeyboardButton("🔄 TRY AGAIN", callback_data=f"check_verify_{user_id}")]
                            ])
                        )
                    
            except Exception as e:
                await callback_query.answer("Error checking verification", show_alert=True)

        @bot.on_message(filters.command("verify") & filters.private)
        async def verify_command(client, message):
            user_id = message.from_user.id
            user_name = message.from_user.first_name or "User"
            
            # Check if user is admin
            if self.is_admin(user_id):
                await message.reply_text(
                    f"👑 **Welcome Admin {user_name}!**\n\n"
                    "You have **ADMIN privileges** and don't require verification.\n\n"
                    "You can directly download files! 🎬",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("🎬 DOWNLOAD MOVIES", url=f"https://t.me/{self.config.BOT_USERNAME}")],
                        [
                            InlineKeyboardButton("📢 Mᴀɪɴ Cʜᴀɴɴᴇʟ", url=self.config.MAIN_CHANNEL_LINK),
                            InlineKeyboardButton("🔎 Mᴏᴠɪᴇꜱ Gʀᴏᴜᴘ", url=self.config.UPDATES_CHANNEL_LINK)
                        ]
                    ])
                )
                return
            
            if self.config.VERIFICATION_REQUIRED:
                is_verified, status = await self.check_url_shortener_verification(user_id)
                
                if is_verified:
                    await message.reply_text(
                        f"✅ **Already Verified, {user_name}!**\n\n"
                        f"Your GP Link verification is active and valid for 6 hours.\n\n"
                        "You can download files now! 🎬",
                        reply_markup=InlineKeyboardMarkup([
                            [InlineKeyboardButton("🎬 DOWNLOAD MOVIES", url=f"https://t.me/{self.config.BOT_USERNAME}")],
                            [
                                InlineKeyboardButton("📢 Mᴀɪɴ Cʜᴀɴɴᴇʟ", url=self.config.MAIN_CHANNEL_LINK),
                                InlineKeyboardButton("🔎 Mᴏᴠɪᴇꜱ Gʀᴏᴜᴘ", url=self.config.UPDATES_CHANNEL_LINK)
                            ]
                        ])
                    )
                else:
                    gplink_url, verification_code = await self.create_gplink_verification(user_id)
                    
                    if gplink_url:
                        await message.reply_text(
                            f"🔗 **GP Link Verification Required, {user_name}**\n\n"
                            "📋 **Verification Process:**\n\n"
                            "1. **Click GP LINK VERIFY below**\n"
                            "2. **Complete GP Link action**\n"
                            "3. **You'll return to bot automatically**\n\n"
                            "⏰ **Code valid for 10 minutes**\n"
                            f"🔢 **Your Code:** `{verification_code}`\n\n"
                            "🚀 **Click below to start verification:**",
                            reply_markup=InlineKeyboardMarkup([
                                [InlineKeyboardButton("🔗 GP LINK VERIFY", url=gplink_url)],
                                [InlineKeyboardButton("🔄 CHECK STATUS", callback_data=f"check_verify_{user_id}")],
                                [
                                    InlineKeyboardButton("📢 Mᴀɪɴ Cʜᴀɴɴᴇʟ", url=self.config.MAIN_CHANNEL_LINK),
                                    InlineKeyboardButton("🔎 Mᴏᴠɪᴇꜱ Gʀᴏᴜᴘ", url=self.config.UPDATES_CHANNEL_LINK)
                                ]
                            ]),
                            disable_web_page_preview=True
                        )
                    else:
                        await message.reply_text(
                            "❌ **Failed to generate verification link**\n\nPlease try the /verify command again.",
                            reply_markup=InlineKeyboardMarkup([
                                [InlineKeyboardButton("🔄 TRY AGAIN", callback_data=f"check_verify_{user_id}")]
                            ])
                        )
            else:
                await message.reply_text(
                    "ℹ️ **Verification Not Required**\n\n"
                    "GP Link verification is currently disabled.\n"
                    "You can download files directly.",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("🎬 DOWNLOAD MOVIES", url=f"https://t.me/{self.config.BOT_USERNAME}")],
                        [
                            InlineKeyboardButton("📢 Mᴀɪɴ Cʜᴀɴɴᴇʟ", url=self.config.MAIN_CHANNEL_LINK),
                            InlineKeyboardButton("🔎 Mᴏᴠɪᴇꜱ Gʀᴏᴜᴘ", url=self.config.UPDATES_CHANNEL_LINK)
                        ]
                    ])
                )
