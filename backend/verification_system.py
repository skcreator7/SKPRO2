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
import urllib.parse

logger = logging.getLogger(__name__)

class VerificationSystem:
    def __init__(self, verification_col: AsyncIOMotorCollection, config):
        self.verification_col = verification_col
        self.config = config
        self.pending_verifications = {}
    
    def is_admin(self, user_id):
        return user_id in self.config.ADMIN_IDS
    
    def generate_verification_code(self):
        return ''.join(random.choices(string.digits, k=6))
    
    async def check_url_shortener_verification(self, user_id):
        if not self.config.VERIFICATION_REQUIRED:
            return True, "verification_not_required"
        
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

    async def generate_short_url(self, destination_url):
        """Generate short URL using configured shortener"""
        api_url = getattr(self.config, 'URL_SHORTENER_API', '').strip()
        api_key = getattr(self.config, 'URL_SHORTENER_KEY', '').strip()
        
        # If no shortener configured, use direct URL
        if not api_url or not api_key:
            logger.info("No shortener configured, using direct URL")
            return destination_url, 'Direct'
        
        logger.info(f"Trying shortener API: {api_url}")
        
        try:
            # For GPLinks - use correct API endpoint
            if 'gplinks' in api_url.lower():
                return await self._try_gplinks_fixed(destination_url, api_key)
            else:
                # For other shorteners, use the provided API URL
                return await self._try_generic_api(destination_url, api_url, api_key)
                
        except Exception as e:
            logger.error(f"Shortener error: {e}")
            return destination_url, 'Direct'

    async def _try_gplinks_fixed(self, destination_url, api_key):
        """Fixed GPLinks shortener with correct API endpoint"""
        try:
            async with aiohttp.ClientSession() as session:
                # GPLinks correct API endpoint
                api_endpoint = "https://gplinks.in/api/shorten"
                
                payload = {
                    'url': destination_url,
                    'api_key': api_key
                }
                
                logger.info(f"Calling GPLinks API: {api_endpoint}")
                
                async with session.post(
                    api_endpoint,
                    json=payload,
                    timeout=10
                ) as response:
                    
                    logger.info(f"GPLinks response status: {response.status}")
                    
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"GPLinks response: {result}")
                        
                        if result.get('status') == 'success':
                            short_url = result.get('shortenedUrl')
                            logger.info(f"✅ GPLinks success: {short_url}")
                            return short_url, 'GPLinks'
                        else:
                            logger.error(f"GPLinks API error: {result}")
                    else:
                        response_text = await response.text()
                        logger.error(f"GPLinks HTTP error {response.status}: {response_text}")
                        
        except Exception as e:
            logger.error(f"GPLinks error: {str(e)}")
        
        return destination_url, 'Direct'

    async def _try_generic_api(self, destination_url, api_url, api_key):
        """Try generic API with common formats"""
        try:
            async with aiohttp.ClientSession() as session:
                logger.info(f"Trying generic API: {api_url}")
                
                # Try format 1: JSON with api_key in payload
                payload1 = {
                    'url': destination_url,
                    'api_key': api_key
                }
                headers1 = {'Content-Type': 'application/json'}
                
                async with session.post(api_url, json=payload1, headers=headers1, timeout=10) as response:
                    logger.info(f"Generic API response status: {response.status}")
                    
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"Generic API response: {result}")
                        
                        if isinstance(result, dict):
                            for key in ['shortenedUrl', 'short_url', 'link', 'url', 'result_url', 'shortened_url']:
                                if result.get(key):
                                    short_url = result[key]
                                    logger.info(f"✅ Generic API success with key '{key}': {short_url}")
                                    return short_url, 'Generic'
                        elif isinstance(result, str) and result.startswith('http'):
                            logger.info(f"✅ Generic API success with string: {result}")
                            return result, 'Generic'
                
                # Try format 2: Authorization header
                headers2 = {
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                }
                payload2 = {'url': destination_url}
                
                async with session.post(api_url, json=payload2, headers=headers2, timeout=10) as response:
                    logger.info(f"Generic API response status (Auth): {response.status}")
                    
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"Generic API response (Auth): {result}")
                        
                        if isinstance(result, dict):
                            for key in ['shortenedUrl', 'short_url', 'link', 'url', 'result_url', 'shortened_url']:
                                if result.get(key):
                                    short_url = result[key]
                                    logger.info(f"✅ Generic API success with Auth: {short_url}")
                                    return short_url, 'Generic'
                
                # Try format 3: public-api-token header
                headers3 = {'public-api-token': api_key}
                payload3 = {'url': destination_url}
                
                async with session.post(api_url, json=payload3, headers=headers3, timeout=10) as response:
                    logger.info(f"Generic API response status (Token): {response.status}")
                    
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"Generic API response (Token): {result}")
                        
                        if isinstance(result, dict) and result.get('shortenedUrl'):
                            short_url = result['shortenedUrl']
                            logger.info(f"✅ Generic API success with Token: {short_url}")
                            return short_url, 'Generic'
                
        except Exception as e:
            logger.error(f"Generic API error: {str(e)}")
        
        return destination_url, 'Direct'

    async def create_verification_link(self, user_id):
        """Create verification link"""
        try:
            verification_code = self.generate_verification_code()
            destination_url = f"https://t.me/{self.config.BOT_USERNAME}?start=verify_{user_id}_{verification_code}"
            
            # Generate short URL
            short_url, service_name = await self.generate_short_url(destination_url)
            
            # Store verification data
            self.pending_verifications[user_id] = {
                'code': verification_code,
                'created_at': datetime.now(),
                'short_url': short_url,
                'service_name': service_name,
                'destination_url': destination_url
            }
            
            logger.info(f"✅ Generated {service_name} URL: {short_url}")
            return short_url, verification_code, service_name
            
        except Exception as e:
            logger.error(f"Verification link creation error: {e}")
            # Fallback to direct URL
            verification_code = self.generate_verification_code()
            destination_url = f"https://t.me/{self.config.BOT_USERNAME}?start=verify_{user_id}_{verification_code}"
            return destination_url, verification_code, 'Direct'

    async def verify_user_with_code(self, user_id, verification_code):
        """Verify user using verification code"""
        try:
            if user_id in self.pending_verifications:
                pending_data = self.pending_verifications[user_id]
                
                if (pending_data['code'] == verification_code and 
                    (datetime.now() - pending_data['created_at']).total_seconds() < 600):
                    
                    await self.verification_col.update_one(
                        {"user_id": user_id},
                        {
                            "$set": {
                                "verified_at": datetime.now(),
                                "verified_by": "url_shortener",
                                "verification_code": verification_code
                            }
                        },
                        upsert=True
                    )
                    
                    del self.pending_verifications[user_id]
                    return True, "verified_successfully"
            
            return False, "invalid_or_expired_code"
            
        except Exception as e:
            logger.error(f"Verification error: {e}")
            return False, "error"

    async def process_verification_start(self, user_id, verification_code):
        return await self.verify_user_with_code(user_id, verification_code)

    async def generate_verification_url(self, user_id):
        """Generate verification URL"""
        try:
            if self.is_admin(user_id):
                return None
            
            short_url, verification_code, service_name = await self.create_verification_link(user_id)
            return short_url
                
        except Exception as e:
            logger.error(f"Generate verification URL error: {e}")
            return None

    async def api_verify_user(self, request):
        try:
            data = await request.get_json()
            user_id = data.get('user_id')
            verification_code = data.get('verification_code')
            
            if not user_id or not verification_code:
                return jsonify({'status': 'error', 'message': 'User ID and verification code required'}), 400
            
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
        try:
            if self.is_admin(user_id):
                return jsonify({
                    'status': 'success',
                    'verification_url': None,
                    'user_id': user_id,
                    'message': 'admin_no_verification_needed'
                })
            
            short_url, verification_code, service_name = await self.create_verification_link(user_id)
            
            return jsonify({
                'status': 'success',
                'verification_url': short_url,
                'verification_code': verification_code,
                'service_name': service_name,
                'user_id': user_id
            })
                
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500

    def setup_bot_handlers(self, bot, User, flood_protection):
        @bot.on_callback_query(filters.regex(r"^check_verify_"))
        async def check_verify_callback(client, callback_query):
            user_id = callback_query.from_user.id
            try:
                is_verified, status = await self.check_url_shortener_verification(user_id)
                
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
                        "2. **Complete the action**\n" 
                        "3. **Return to bot automatically**\n\n"
                        "⏰ **Code valid for 10 minutes**\n"
                        f"🔢 **Your Code:** `{verification_code}`\n\n"
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
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("🎬 DOWNLOAD MOVIES", url=f"https://t.me/{self.config.BOT_USERNAME}")],
                        [
                            InlineKeyboardButton("📢 MAIN CHANNEL", url=self.config.MAIN_CHANNEL_LINK),
                            InlineKeyboardButton("🔎 MOVIES GROUP", url=self.config.UPDATES_CHANNEL_LINK)
                        ]
                    ])
                )
                return
            
            if self.config.VERIFICATION_REQUIRED:
                is_verified, status = await self.check_url_shortener_verification(user_id)
                
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
                        "2. **Complete the action**\n"
                        "3. **You'll return automatically**\n\n"
                        "⏰ **Code valid for 10 minutes**\n"
                        f"🔢 **Your Code:** `{verification_code}`\n\n"
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
            else:
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
