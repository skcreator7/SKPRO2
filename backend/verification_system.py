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
    
    async def check_url_shortener_verification(self, user_id):
        """Check if user is verified via URL shortener"""
        if not self.config.VERIFICATION_REQUIRED:
            return True, "verification_not_required"
        
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
        """Generate verification URL for user"""
        base_url = self.config.WEBSITE_URL or self.config.BACKEND_URL
        verification_token = f"verify_{user_id}_{int(datetime.now().timestamp())}"
        return f"{base_url}/verify?token={verification_token}&user_id={user_id}"

    async def api_verify_user(self, request):
        """API endpoint to verify user"""
        try:
            data = await request.get_json()
            user_id = data.get('user_id')
            verification_url = data.get('verification_url')
            
            if not user_id:
                return jsonify({'status': 'error', 'message': 'User ID required'}), 400
            
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
                'verification_required': self.config.VERIFICATION_REQUIRED
            })
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500

    async def api_generate_verification_url(self, user_id):
        """API endpoint to generate verification URL"""
        try:
            verification_url = await self.generate_verification_url(user_id)
            return jsonify({
                'status': 'success',
                'verification_url': verification_url,
                'user_id': user_id
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
                    await callback_query.message.edit_text(
                        "✅ **Verification Successful!**\n\n"
                        "You Can Now Download Files From The Website.\n\n"
                        f"🌐 **Website:** {self.config.WEBSITE_URL}\n\n"
                        "⏰ **Verification Valid For 6 Hours**",
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
                    await callback_query.message.edit_text(
                        "❌ **Not Verified Yet**\n\n"
                        "Please complete the verification process first.\n\n"
                        f"🔗 **Verification URL:** `{verification_url}`",
                        reply_markup=InlineKeyboardMarkup([
                            [InlineKeyboardButton("🔗 VERIFY NOW", url=verification_url)],
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
            
            if self.config.VERIFICATION_REQUIRED:
                is_verified, status = await self.check_url_shortener_verification(user_id)
                
                if is_verified:
                    await message.reply_text(
                        f"✅ **Already Verified, {user_name}!**\n\n"
                        f"Your verification is active and valid for 6 hours.\n\n"
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
                        f"🔗 **Verification Required, {user_name}**\n\n"
                        "To download files, please complete the URL verification:\n\n"
                        f"**Verification URL:** `{verification_url}`\n\n"
                        "⏰ **Valid for 6 hours after verification**",
                        reply_markup=InlineKeyboardMarkup([
                            [InlineKeyboardButton("🔗 VERIFY NOW", url=verification_url)],
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
                    "URL shortener verification is currently disabled.\n"
                    "You can download files directly from the website.",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("🌐 OPEN WEBSITE", url=self.config.WEBSITE_URL)],
                        [
                            InlineKeyboardButton("📢 Mᴀɪɴ Cʜᴀɴɴᴇʟ", url=self.config.MAIN_CHANNEL_LINK),
                            InlineKeyboardButton("🔎 Mᴏᴠɪᴇꜱ Gʀᴏᴜᴘ", url=self.config.UPDATES_CHANNEL_LINK)
                        ]
                    ])
                )
