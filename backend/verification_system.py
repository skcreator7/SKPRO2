import os
import logging
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorCollection
import aiohttp
from quart import jsonify
from pyrogram import filters
from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton

logger = logging.getLogger(__name__)

class VerificationSystem:
    def __init__(self, verification_col: AsyncIOMotorCollection, config):
        self.verification_col = verification_col
        self.config = config
    
    def is_admin(self, user_id):
        """Check if user is admin"""
        return user_id in self.config.ADMIN_IDS
    
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

    async def verify_user_with_url_shortener(self, user_id, verification_url=None):
        """Verify user using URL shortener service"""
        if not self.config.VERIFICATION_REQUIRED:
            return True, "verification_not_required"
        
        # Admin users are always verified automatically
        if self.is_admin(user_id):
            return True, "admin_auto_verified"
        
        try:
            if not verification_url:
                verification_url = await self.generate_verification_url(user_id)
            
            async with aiohttp.ClientSession() as session:
                payload = {
                    'user_id': user_id, 
                    'verification_url': verification_url, 
                    'api_key': self.config.URL_SHORTENER_KEY
                }
                async with session.post(
                    self.config.URL_SHORTENER_API, 
                    json=payload, 
                    timeout=5
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get('verified') == True:
                            await self.verification_col.update_one(
                                {"user_id": user_id},
                                {
                                    "$set": {
                                        "verified_at": datetime.now(), 
                                        "verification_url": verification_url, 
                                        "verified_by": "url_shortener"
                                    }
                                },
                                upsert=True
                            )
                            return True, "verified"
                        else:
                            return False, result.get('message', 'verification_failed')
                    else:
                        return False, "api_error"
        except Exception as e:
            logger.error(f"Verification error: {e}")
            return False, "error"

    async def generate_verification_url(self, user_id):
        """Generate GP Link shortner verification URL for user"""
        # Use GP Link shortner service
        base_url = "https://gplinks.in/api"
        api_key = self.config.URL_SHORTENER_KEY
        
        # Create verification token
        verification_token = f"sk4film_verify_{user_id}_{int(datetime.now().timestamp())}"
        verification_page_url = f"{self.config.WEBSITE_URL}/verify?token={verification_token}"
        
        try:
            # Shorten the verification URL using GP Link
            async with aiohttp.ClientSession() as session:
                payload = {
                    'url': verification_page_url,
                    'api_key': api_key
                }
                async with session.post(
                    f"{base_url}/shorten",
                    json=payload,
                    timeout=5
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get('status') == 'success':
                            return result.get('shortenedUrl', verification_page_url)
        except Exception as e:
            logger.error(f"GP Link shortening failed: {e}")
        
        # Fallback to direct URL if shortening fails
        return verification_page_url

    async def api_verify_user(self, request):
        """API endpoint to verify user"""
        try:
            data = await request.get_json()
            user_id = data.get('user_id')
            verification_url = data.get('verification_url')
            
            if not user_id:
                return jsonify({'status': 'error', 'message': 'User ID required'}), 400
            
            # Check if admin
            if self.is_admin(user_id):
                return jsonify({
                    'status': 'success',
                    'verified': True,
                    'message': 'admin_auto_verified',
                    'user_id': user_id
                })
            
            is_verified, message = await self.verify_user_with_url_shortener(user_id, verification_url)
            
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
            
            verification_url = await self.generate_verification_url(user_id)
            return jsonify({
                'status': 'success',
                'verification_url': verification_url,
                'user_id': user_id,
                'is_gplink': 'gplinks.in' in verification_url
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
                        message_text = "✅ **Verification Successful!**\n\nYou Can Now Download Files From The Website.\n\n⏰ **Verification Valid For 6 Hours**"
                    
                    await callback_query.message.edit_text(
                        message_text,
                        reply_markup=InlineKeyboardMarkup([
                            [InlineKeyboardButton("🌐 OPEN WEBSITE", url=self.config.WEBSITE_URL)],
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
                                [InlineKeyboardButton("🌐 OPEN WEBSITE", url=self.config.WEBSITE_URL)],
                                [
                                    InlineKeyboardButton("📢 Mᴀɪɴ Cʜᴀɴɴᴇʟ", url=self.config.MAIN_CHANNEL_LINK),
                                    InlineKeyboardButton("🔎 Mᴏᴠɪᴇꜱ Gʀᴏᴜᴘ", url=self.config.UPDATES_CHANNEL_LINK)
                                ]
                            ])
                        )
                        return
                    
                    verification_url = await self.generate_verification_url(user_id)
                    await callback_query.message.edit_text(
                        "❌ **Not Verified Yet**\n\n"
                        "Please complete the GP Link verification process first.\n\n"
                        f"🔗 **Verification URL:** `{verification_url}`\n\n"
                        "📱 **Steps:**\n"
                        "1. Click VERIFY NOW below\n"
                        "2. Complete GP Link verification\n"
                        "3. Come back and click CHECK AGAIN",
                        reply_markup=InlineKeyboardMarkup([
                            [InlineKeyboardButton("🔗 VERIFY NOW (GP Link)", url=verification_url)],
                            [InlineKeyboardButton("🔄 CHECK AGAIN", callback_data=f"check_verify_{user_id}")],
                            [
                                InlineKeyboardButton("📢 Mᴀɪɴ Cʜᴀɴɴᴇʟ", url=self.config.MAIN_CHANNEL_LINK),
                                InlineKeyboardButton("🔎 Mᴏᴠɪᴇꜱ Gʀᴏᴜᴘ", url=self.config.UPDATES_CHANNEL_LINK)
                            ]
                        ]),
                        disable_web_page_preview=True
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
                    "You can directly download files from the website! 🎬",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("🌐 OPEN WEBSITE", url=self.config.WEBSITE_URL)],
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
                        "You can download files from the website now! 🎬",
                        reply_markup=InlineKeyboardMarkup([
                            [InlineKeyboardButton("🌐 OPEN WEBSITE", url=self.config.WEBSITE_URL)],
                            [
                                InlineKeyboardButton("📢 Mᴀɪɴ Cʜᴀɴɴᴇʟ", url=self.config.MAIN_CHANNEL_LINK),
                                InlineKeyboardButton("🔎 Mᴏᴠɪᴇꜱ Gʀᴏᴜᴘ", url=self.config.UPDATES_CHANNEL_LINK)
                            ]
                        ])
                    )
                else:
                    verification_url = await self.generate_verification_url(user_id)
                    await message.reply_text(
                        f"🔗 **GP Link Verification Required, {user_name}**\n\n"
                        "To download files, please complete the GP Link URL verification:\n\n"
                        f"**Verification URL:** `{verification_url}`\n\n"
                        "📋 **Process:**\n"
                        "• Click VERIFY NOW below\n"
                        "• Complete GP Link verification\n"
                        "• Return to bot and use /verify again\n\n"
                        "⏰ **Valid for 6 hours after verification**",
                        reply_markup=InlineKeyboardMarkup([
                            [InlineKeyboardButton("🔗 GP LINK VERIFY", url=verification_url)],
                            [
                                InlineKeyboardButton("📢 Mᴀɪɴ Cʜᴀɴɴᴇʟ", url=self.config.MAIN_CHANNEL_LINK),
                                InlineKeyboardButton("🔎 Mᴏᴠɪᴇꜱ Gʀᴏᴜᴘ", url=self.config.UPDATES_CHANNEL_LINK)
                            ]
                        ]),
                        disable_web_page_preview=True
                    )
            else:
                await message.reply_text(
                    "ℹ️ **Verification Not Required**\n\n"
                    "GP Link verification is currently disabled.\n"
                    "You can download files directly from the website.",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("🌐 OPEN WEBSITE", url=self.config.WEBSITE_URL)],
                        [
                            InlineKeyboardButton("📢 Mᴀɪɴ Cʜᴀɴɴᴇʟ", url=self.config.MAIN_CHANNEL_LINK),
                            InlineKeyboardButton("🔎 Mᴏᴠɪᴇꜱ Gʀᴏᴜᴘ", url=self.config.UPDATES_CHANNEL_LINK)
                        ]
                    ])
                )
