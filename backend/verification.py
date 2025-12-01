# verification.py - Verification & Link Shortener System

import logging
import secrets
import string
import json
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Any
from motor.motor_asyncio import AsyncIOMotorClient

logger = logging.getLogger(__name__)

class VerificationManager:
    """Verification System with Link Shortener Support"""
    
    def __init__(self, config, db, premium_manager):
        self.config = config
        self.db = db
        self.verification_col = db.verifications
        self.premium_manager = premium_manager
        
        # In-memory caches
        self.pending_verifications = {}  # user_id -> verification_data
        self.verification_tokens = {}  # token -> user_id
        self.verified_users = {}  # user_id -> verification_data
    
    async def init_indexes(self):
        """Initialize MongoDB indexes"""
        try:
            await self.verification_col.create_index("user_id", unique=True)
            await self.verification_col.create_index("token")
            await self.verification_col.create_index("verified_at")
            logger.info("✅ Verification indexes created")
        except Exception as e:
            logger.warning(f"Verification index creation: {e}")
    
    def generate_unique_token(self, length: int = 16) -> str:
        """Generate unique verification token"""
        characters = string.ascii_letters + string.digits
        return ''.join(secrets.choice(characters) for _ in range(length))
    
    async def get_shortened_url(self, destination_url: str) -> Tuple[str, str]:
        """Get shortened URL using GPLinks API"""
        try:
            if not hasattr(self.config, 'SHORTLINK_API') or not self.config.SHORTLINK_API:
                return destination_url, 'Direct'
            
            api_url = "https://gplinks.in/api"
            params = {
                'api': self.config.SHORTLINK_API,
                'url': destination_url
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(api_url, params=params, timeout=30) as response:
                    if response.status == 200:
                        response_text = await response.text()
                        
                        # Try JSON first
                        try:
                            data = json.loads(response_text)
                            if data.get("status") == "success":
                                return data.get('shortenedUrl', destination_url), 'GPLinks'
                        except json.JSONDecodeError:
                            # Try direct URL
                            if response_text.startswith('http'):
                                return response_text, 'GPLinks'
        
        except Exception as e:
            logger.warning(f"Shortener failed: {e}")
        
        return destination_url, 'Direct'
    
    async def create_verification_link(self, user_id: int) -> Dict[str, Any]:
        """Create verification link with unique token"""
        try:
            # Generate unique verification token
            verification_token = self.generate_unique_token()
            
            # Create Telegram deep link
            bot_username = self.config.BOT_USERNAME
            destination_url = f"https://t.me/{bot_username}?start=verify_{verification_token}"
            
            # Get shortened URL
            short_url, service_name = await self.get_shortened_url(destination_url)
            
            # Store verification data
            verification_data = {
                'user_id': user_id,
                'token': verification_token,
                'created_at': datetime.now(),
                'short_url': short_url,
                'service_name': service_name,
                'destination_url': destination_url,
                'attempts': 0,
                'status': 'pending',
                'verified_at': None
            }
            
            # Store in memory caches
            self.pending_verifications[user_id] = verification_data
            self.verification_tokens[verification_token] = user_id
            
            # Store in database
            await self.verification_col.update_one(
                {'user_id': user_id},
                {'$set': verification_data},
                upsert=True
            )
            
            logger.info(f"✅ Verification link created for user {user_id}")
            
            return verification_data
            
        except Exception as e:
            logger.error(f"❌ Verification link creation error: {e}")
            # Fallback
            verification_token = self.generate_unique_token()
            bot_username = self.config.BOT_USERNAME
            direct_url = f"https://t.me/{bot_username}?start=verify_{verification_token}"
            
            verification_data = {
                'user_id': user_id,
                'token': verification_token,
                'created_at': datetime.now(),
                'short_url': direct_url,
                'service_name': 'Direct',
                'destination_url': direct_url,
                'attempts': 0,
                'status': 'pending',
                'verified_at': None
            }
            
            self.pending_verifications[user_id] = verification_data
            self.verification_tokens[verification_token] = user_id
            
            return verification_data
    
    async def verify_user(self, token: str) -> Tuple[bool, Optional[int]]:
        """Verify user with token"""
        try:
            # Check if token exists
            user_id = self.verification_tokens.get(token)
            
            if not user_id:
                # Try database
                verification_data = await self.verification_col.find_one({'token': token})
                if verification_data:
                    user_id = verification_data['user_id']
                else:
                    return False, None
            
            # Mark as verified
            verification_data = {
                'user_id': user_id,
                'token': token,
                'verified_at': datetime.now(),
                'expires_at': datetime.now() + timedelta(hours=self.config.VERIFICATION_DURATION),
                'status': 'verified'
            }
            
            # Update caches
            self.verified_users[user_id] = verification_data
            if user_id in self.pending_verifications:
                del self.pending_verifications[user_id]
            
            # Update database
            await self.verification_col.update_one(
                {'user_id': user_id},
                {'$set': verification_data},
                upsert=True
            )
            
            logger.info(f"✅ User {user_id} verified successfully")
            return True, user_id
            
        except Exception as e:
            logger.error(f"Verification error: {e}")
            return False, None
    
    async def check_verification_status(self, user_id: int) -> Tuple[bool, Optional[Dict]]:
        """Check if user is verified"""
        try:
            # Check if premium (premium users bypass verification)
            is_premium, premium_data = await self.premium_manager.check_premium_status(user_id)
            if is_premium:
                return True, {'type': 'premium', 'data': premium_data}
            
            # Check cache first
            if user_id in self.verified_users:
                cached_data = self.verified_users[user_id]
                if cached_data['expires_at'] > datetime.now():
                    return True, {'type': 'verified', 'data': cached_data}
                else:
                    del self.verified_users[user_id]
            
            # Check database
            verification_data = await self.verification_col.find_one({
                'user_id': user_id,
                'status': 'verified',
                'expires_at': {'$gt': datetime.now()}
            })
            
            if verification_data:
                self.verified_users[user_id] = verification_data
                return True, {'type': 'verified', 'data': verification_data}
            
            return False, None
            
        except Exception as e:
            logger.error(f"Verification check error: {e}")
            return False, None
    
    async def cleanup_expired_verifications(self) -> int:
        """Remove expired verifications"""
        try:
            result = await self.verification_col.delete_many({
                'expires_at': {'$lt': datetime.now()}
            })
            
            # Clear cache of expired verifications
            expired_users = [uid for uid, data in self.verified_users.items() 
                           if data.get('expires_at') and data['expires_at'] < datetime.now()]
            for uid in expired_users:
                del self.verified_users[uid]
            
            logger.info(f"✅ Cleaned up {result.deleted_count} expired verifications")
            return result.deleted_count
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            return 0


# Verification command handlers
async def setup_verification_handlers(bot, verification_manager, premium_manager, config):
    """Setup verification-related handlers"""
    from pyrogram import filters
    from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton
    
    @bot.on_message(filters.command("start") & filters.private)
    async def start_command(client, message):
        """Handle /start command with verification"""
        user_id = message.from_user.id
        username = message.from_user.first_name or "User"
        
        # Check for verification token
        if len(message.text.split()) > 1:
            token_part = message.text.split()[1]
            
            if token_part.startswith("verify_"):
                token = token_part.replace("verify_", "")
                success, verified_user_id = await verification_manager.verify_user(token)
                
                if success and verified_user_id == user_id:
                    await message.reply_text(
                        f"✅ **Verification Successful!**\n\n"
                        f"Hi {username}! You are now verified for {config.VERIFICATION_DURATION} hours.\n"
                        f"You can now download files from the website.",
                        reply_markup=InlineKeyboardMarkup([
                            [InlineKeyboardButton("🌐 Open Website", url=config.WEBSITEURL)],
                            [InlineKeyboardButton("💎 Get Premium", callback_data="show_premium_plans")]
                        ])
                    )
                    return
                else:
                    await message.reply_text("❌ Invalid or expired verification link.")
                    return
        
        # Normal start message
        await message.reply_text(
            f"👋 **Welcome {username}!**\n\n"
            f"🎬 Search and download movies from our website.\n\n"
            f"**Choose an option:**",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔓 Free Verify (6 hours)", callback_data="verify_free")],
                [InlineKeyboardButton("💎 Buy Premium", callback_data="show_premium_plans")],
                [InlineKeyboardButton("🌐 Open Website", url=config.WEBSITEURL)],
                [InlineKeyboardButton("📢 Join Channel", url=config.MAINCHANNELLINK)]
            ])
        )
    
    @bot.on_callback_query(filters.regex("^verify_free$"))
    async def verify_free_callback(client, callback_query):
        """Handle free verification"""
        user_id = callback_query.from_user.id
        username = callback_query.from_user.first_name or "User"
        
        # Check if already verified
        is_verified, status = await verification_manager.check_verification_status(user_id)
        
        if is_verified:
            if status['type'] == 'premium':
                await callback_query.answer("You already have Premium! 💎", show_alert=True)
            else:
                remaining_time = (status['data']['expires_at'] - datetime.now()).seconds // 3600
                await callback_query.answer(f"Already verified! {remaining_time}h remaining", show_alert=True)
            return
        
        # Create verification link
        verification_data = await verification_manager.create_verification_link(user_id)
        
        await callback_query.message.edit_text(
            f"🔓 **Free Verification**\n\n"
            f"Hi {username}!\n\n"
            f"Complete the verification to get **{config.VERIFICATION_DURATION} hours** of free access.\n\n"
            f"**Step 1:** Click the button below\n"
            f"**Step 2:** Complete the verification\n"
            f"**Step 3:** Come back and enjoy!\n\n"
            f"🔗 Shortener: {verification_data['service_name']}",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("✅ Verify Now", url=verification_data['short_url'])],
                [InlineKeyboardButton("💎 Or Buy Premium", callback_data="show_premium_plans")],
                [InlineKeyboardButton("🔙 Back", callback_data="start_back")]
            ])
        )
        await callback_query.answer()
    
    @bot.on_callback_query(filters.regex("^show_premium_plans$"))
    async def show_premium_plans_callback(client, callback_query):
        """Show premium plans"""
        plans_text = f"""
🌟 **SK4FiLM Premium Plans**

Get premium access and enjoy:
✓ No Ads
✓ Fast Downloads
✓ High Quality
✓ Priority Support

**Choose Your Plan:**
"""
        await callback_query.message.edit_text(
            plans_text,
            reply_markup=premium_manager.get_premium_plans_keyboard()
        )
        await callback_query.answer()
    
    @bot.on_callback_query(filters.regex("^buy_premium_"))
    async def buy_premium_callback(client, callback_query):
        """Handle premium purchase"""
        plan_type = callback_query.data.replace("buy_premium_", "")
        user_id = callback_query.from_user.id
        
        plan_info = premium_manager.format_premium_info(plan_type)
        
        await callback_query.message.edit_text(
            plan_info + "\n\n**Payment Instructions:**\n"
            "1️⃣ Send payment to UPI ID: `yourUPI@paytm`\n"
            "2️⃣ Take screenshot of payment\n"
            "3️⃣ Send screenshot to admin\n\n"
            "After verification, premium will be activated!",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📸 Submit Screenshot", callback_data=f"submit_payment_{plan_type}")],
                [InlineKeyboardButton("🔙 Back", callback_data="show_premium_plans")]
            ])
        )
        await callback_query.answer()
    
    @bot.on_callback_query(filters.regex("^submit_payment_"))
    async def submit_payment_callback(client, callback_query):
        """Handle payment screenshot submission"""
        plan_type = callback_query.data.replace("submit_payment_", "")
        user_id = callback_query.from_user.id
        
        # Notify admins
        for admin_id in config.ADMINIDS:
            try:
                await client.send_message(
                    admin_id,
                    f"💳 **New Premium Purchase Request**\n\n"
                    f"User ID: `{user_id}`\n"
                    f"Username: @{callback_query.from_user.username or 'N/A'}\n"
                    f"Plan: {plan_type.title()}\n\n"
                    f"Waiting for payment screenshot...\n\n"
                    f"Use: `/premiumuser {user_id} add {plan_type}`",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("✅ Approve", callback_data=f"approve_{user_id}_{plan_type}")]
                    ])
                )
            except:
                pass
        
        await callback_query.message.edit_text(
            "📸 **Submit Payment Screenshot**\n\n"
            "Please send your payment screenshot now.\n"
            "Admin will verify and activate your premium within 10-30 minutes.",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Back", callback_data="show_premium_plans")]
            ])
        )
        await callback_query.answer("Please send payment screenshot to admin!", show_alert=True)
    
    logger.info("✅ Verification handlers registered")
