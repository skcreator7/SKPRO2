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

    async def get_configured_shortener(self):
        """Get configured shortener from environment variables"""
        api_url = getattr(self.config, 'URL_SHORTENER_API', '').strip()
        api_key = getattr(self.config, 'URL_SHORTENER_KEY', '').strip()
        
        if not api_url or not api_key:
            logger.warning("No shortener API configured, using public shorteners")
            return None
        
        # Return configured shortener details
        return {
            'api_url': api_url,
            'api_key': api_key,
            'name': self._detect_shortener_name(api_url)
        }

    def _detect_shortener_name(self, api_url):
        """Detect shortener name from API URL"""
        api_url_lower = api_url.lower()
        
        if 'gplinks' in api_url_lower:
            return 'GPLinks'
        elif 'bitly' in api_url_lower:
            return 'Bitly'
        elif 'cuttly' in api_url_lower:
            return 'Cuttly'
        elif 'short.io' in api_url_lower:
            return 'ShortIO'
        elif 'shorte.st' in api_url_lower:
            return 'Shorte'
        elif 'ouo.io' in api_url_lower:
            return 'Ouo'
        elif 'linkshortify' in api_url_lower:
            return 'LinkShortify'
        elif 'tinyurl' in api_url_lower:
            return 'TinyURL'
        else:
            return 'CustomShortener'

    async def generate_short_url_generic(self, destination_url, shortener_config=None):
        """Generic method to generate short URL"""
        try:
            # Try configured shortener first
            if shortener_config:
                custom_url = await self._try_custom_shortener(destination_url, shortener_config)
                if custom_url:
                    return custom_url, shortener_config['name']
            
            # Fallback to public shorteners
            public_shorteners = [
                ('TinyURL', self._try_tinyurl),
                ('Isgd', self._try_isgd),
                ('Dagd', self._try_dagd),
                ('CleanURI', self._try_cleanuri)
            ]
            
            for name, method in public_shorteners:
                try:
                    short_url = await method(destination_url)
                    if short_url:
                        return short_url, name
                except Exception as e:
                    logger.debug(f"{name} failed: {e}")
                    continue
            
            # Ultimate fallback - direct URL
            return destination_url, 'Direct'
            
        except Exception as e:
            logger.error(f"Short URL generation error: {e}")
            return destination_url, 'Direct'

    async def _try_custom_shortener(self, destination_url, shortener_config):
        """Try custom configured shortener"""
        try:
            api_url = shortener_config['api_url']
            api_key = shortener_config['api_key']
            
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {api_key}'
                }
                
                payload = {
                    'url': destination_url,
                    'api_key': api_key
                }
                
                # Try different payload formats for different shorteners
                async with session.post(api_url, json=payload, headers=headers, timeout=10) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Try different response formats
                        if isinstance(result, dict):
                            if result.get('shortenedUrl'):
                                return result['shortenedUrl']
                            elif result.get('short_url'):
                                return result['short_url']
                            elif result.get('link'):
                                return result['link']
                            elif result.get('url'):
                                return result['url']
                            elif result.get('result_url'):
                                return result['result_url']
                        elif isinstance(result, str) and result.startswith('http'):
                            return result
                
                # Alternative format for some shorteners
                headers = {'public-api-token': api_key}
                async with session.post(api_url, json={'url': destination_url}, headers=headers, timeout=10) as response:
                    if response.status == 200:
                        result = await response.json()
                        if isinstance(result, dict) and result.get('shortenedUrl'):
                            return result['shortenedUrl']
                
        except Exception as e:
            logger.error(f"Custom shortener failed: {e}")
        
        return None

    async def _try_tinyurl(self, destination_url):
        """Try TinyURL public API"""
        try:
            async with aiohttp.ClientSession() as session:
                api_url = f"http://tinyurl.com/api-create.php?url={urllib.parse.quote(destination_url)}"
                async with session.get(api_url, timeout=5) as response:
                    if response.status == 200:
                        return await response.text()
        except:
            return None

    async def _try_isgd(self, destination_url):
        """Try Isgd public API"""
        try:
            async with aiohttp.ClientSession() as session:
                api_url = f"https://is.gd/create.php?format=simple&url={urllib.parse.quote(destination_url)}"
                async with session.get(api_url, timeout=5) as response:
                    if response.status == 200:
                        return await response.text()
        except:
            return None

    async def _try_dagd(self, destination_url):
        """Try Dagd public API"""
        try:
            async with aiohttp.ClientSession() as session:
                api_url = f"https://da.gd/s?url={urllib.parse.quote(destination_url)}"
                async with session.get(api_url, timeout=5) as response:
                    if response.status == 200:
                        return (await response.text()).strip()
        except:
            return None

    async def _try_cleanuri(self, destination_url):
        """Try CleanURI public API"""
        try:
            async with aiohttp.ClientSession() as session:
                api_url = "https://cleanuri.com/api/v1/shorten"
                payload = {'url': destination_url}
                async with session.post(api_url, json=payload, timeout=5) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get('result_url')
        except:
            return None

    async def create_verification_link(self, user_id):
        """Create verification link with any available shortener"""
        try:
            verification_code = self.generate_verification_code()
            destination_url = f"https://t.me/{self.config.BOT_USERNAME}?start=verify_{user_id}_{verification_code}"
            
            # Get configured shortener
            shortener_config = await self.get_configured_shortener()
            
            # Generate short URL
            short_url, service_name = await self.generate_short_url_generic(destination_url, shortener_config)
            
            # Store verification data
            self.pending_verifications[user_id] = {
                'code': verification_code,
                'created_at': datetime.now(),
                'short_url': short_url,
                'service': service_name,
                'destination_url': destination_url
            }
            
            logger.info(f"✅ Generated {service_name} URL: {short_url}")
            return short_url, verification_code, service_name
            
        except Exception as e:
            logger.error(f"Verification link creation error: {e}")
            # Ultimate fallback
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
                        f"🔗 **{service_name} Verification Required**\n\n"
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
