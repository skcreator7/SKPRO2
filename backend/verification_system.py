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

# Your existing working shortener function
async def get_verify_shorted_link(link):
    API = SHORTLINK_API
    URL = SHORTLINK_URL
    https = link.split(":")[0]
    if "http" == https:
        https = "https"
        link = link.replace("http", https)

    if URL == "api.shareus.in":
        url = f"https://{URL}/shortLink"
        params = {"token": API,
                  "format": "json",
                  "link": link,
                  }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, raise_for_status=True, ssl=False) as response:
                    data = await response.json(content_type="text/html")
                    if data["status"] == "success":
                        return data["shortlink"]
                    else:
                        logger.error(f"Error: {data['message']}")
                        return f'https://{URL}/shortLink?token={API}&format=json&link={link}'

        except Exception as e:
            logger.error(e)
            return f'https://{URL}/shortLink?token={API}&format=json&link={link}'
    else:
        url = f'https://{URL}/api'
        params = {'api': API,
                  'url': link,
                  }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, raise_for_status=True, ssl=False) as response:
                    data = await response.json()
                    if data["status"] == "success":
                        return data['shortenedUrl']
                    else:
                        logger.error(f"Error: {data['message']}")
                        return f'https://{URL}/api?api={API}&link={link}'

        except Exception as e:
            logger.error(e)
            return f'{URL}/api?api={API}&link={link}'

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
        """Use your existing working shortener function"""
        try:
            # Use your proven working function
            short_url = await get_verify_shorted_link(destination_url)
            
            # Check if shortening was successful
            if short_url and short_url != destination_url and short_url.startswith('http'):
                service_name = "GPLinks" if "gplinks" in SHORTLINK_URL else "Shortener"
                logger.info(f"✅ {service_name} success: {short_url}")
                return short_url, service_name
            else:
                logger.info("Shortener returned original URL, using direct")
                return destination_url, 'Direct'
                
        except Exception as e:
            logger.error(f"Shortener error: {e}")
            return destination_url, 'Direct'

    async def create_verification_link(self, user_id):
        """Create verification link that auto-verifies on click"""
        try:
            verification_code = self.generate_verification_code()
            destination_url = f"https://t.me/{self.config.BOT_USERNAME}?start=verify_{user_id}_{verification_code}"
            
            # Generate short URL using your working function
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

    async def process_verification_start(self, user_id, verification_code):
        """Auto-verify when user clicks the short link"""
        return await self.verify_user_with_code(user_id, verification_code)

    async def verify_user_with_code(self, user_id, verification_code):
        """Verify user automatically when they click the link"""
        try:
            if user_id in self.pending_verifications:
                pending_data = self.pending_verifications[user_id]
                
                # Auto-verify without any manual steps
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
                    return True, "auto_verified_successfully"
            
            return False, "invalid_or_expired_code"
            
        except Exception as e:
            logger.error(f"Verification error: {e}")
            return False, "error"

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
                        message_text = "✅ **Auto-Verification Successful!**\n\nYou can now download files! 🎬\n\n⏰ **Valid for 6 hours**"
                    
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

        @bot.on_message(filters.command("start") & filters.private)
        async def start_command(client, message):
            if len(message.command) > 1:
                command_parts = message.command[1].split('_')
                if len(command_parts) >= 3 and command_parts[0] == "verify":
                    user_id = int(command_parts[1])
                    verification_code = command_parts[2]
                    
                    # Auto-verify the user
                    is_verified, message_text = await self.verify_user_with_code(user_id, verification_code)
                    
                    if is_verified:
                        await message.reply_text(
                            "✅ **Auto-Verification Successful!**\n\n"
                            "You can now download files! 🎬\n\n"
                            "Return to the bot and use /search command.",
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
                            "❌ **Verification Failed**\n\n"
                            "The verification link is invalid or expired.\n"
                            "Please use /verify to get a new link.",
                            reply_markup=InlineKeyboardMarkup([
                                [InlineKeyboardButton("🔄 GET NEW VERIFICATION", callback_data=f"check_verify_{user_id}")]
                            ])
                        )
                    return
            
            # Normal start command
            user_id = message.from_user.id
            user_name = message.from_user.first_name or "User"
            
            if self.is_admin(user_id):
                await message.reply_text(
                    f"👑 **Welcome Admin {user_name}!**\n\n"
                    "You have **ADMIN privileges** - no verification required.\n\n"
                    "Use /search to find movies! 🎬",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("🎬 SEARCH MOVIES", url=f"https://t.me/{self.config.BOT_USERNAME}")],
                        [
                            InlineKeyboardButton("📢 MAIN CHANNEL", url=self.config.MAIN_CHANNEL_LINK),
                            InlineKeyboardButton("🔎 MOVIES GROUP", url=self.config.UPDATES_CHANNEL_LINK)
                        ]
                    ])
                )
            else:
                is_verified, status = await self.check_url_shortener_verification(user_id)
                
                if is_verified:
                    await message.reply_text(
                        f"🎬 **Welcome {user_name}!**\n\n"
                        "You are already verified and can download files!\n\n"
                        "Use /search to find movies.",
                        reply_markup=InlineKeyboardMarkup([
                            [InlineKeyboardButton("🎬 SEARCH MOVIES", url=f"https://t.me/{self.config.BOT_USERNAME}")],
                            [
                                InlineKeyboardButton("📢 MAIN CHANNEL", url=self.config.MAIN_CHANNEL_LINK),
                                InlineKeyboardButton("🔎 MOVIES GROUP", url=self.config.UPDATES_CHANNEL_LINK)
                            ]
                        ])
                    )
                else:
                    await message.reply_text(
                        f"🔗 **Welcome {user_name}!**\n\n"
                        "To download movies, you need to complete a quick verification.\n\n"
                        "**It's just 1 click!**\n\n"
                        "Use /verify to get started.",
                        reply_markup=InlineKeyboardMarkup([
                            [InlineKeyboardButton("🔗 START VERIFICATION", callback_data=f"check_verify_{user_id}")],
                            [
                                InlineKeyboardButton("📢 MAIN CHANNEL", url=self.config.MAIN_CHANNEL_LINK),
                                InlineKeyboardButton("🔎 MOVIES GROUP", url=self.config.UPDATES_CHANNEL_LINK)
                            ]
                        ])
                    )
