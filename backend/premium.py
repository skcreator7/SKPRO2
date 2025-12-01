# premium.py - Premium Plans & User Management System

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from motor.motor_asyncio import AsyncIOMotorClient
import aiohttp
import json

logger = logging.getLogger(__name__)

class PremiumManager:
    """Premium Plans Management System"""
    
    # Premium Plans Configuration
    PREMIUM_PLANS = {
        'bronze': {
            'name': 'Bronze Plan',
            'duration_days': 7,
            'price': 49,
            'features': ['No Ads', 'Fast Downloads', '720p Quality'],
            'emoji': '🥉'
        },
        'silver': {
            'name': 'Silver Plan', 
            'duration_days': 15,
            'price': 89,
            'features': ['No Ads', 'Fast Downloads', '1080p Quality', 'Priority Support'],
            'emoji': '🥈'
        },
        'gold': {
            'name': 'Gold Plan',
            'duration_days': 30,
            'price': 149,
            'features': ['No Ads', 'Fastest Downloads', '1080p Quality', 'Priority Support', 'Batch Downloads'],
            'emoji': '🥇'
        },
        'platinum': {
            'name': 'Platinum Plan',
            'duration_days': 90,
            'price': 399,
            'features': ['All Gold Features', '4K Quality', 'Exclusive Content', 'VIP Support'],
            'emoji': '💎'
        },
        'diamond': {
            'name': 'Diamond Plan',
            'duration_days': 365,
            'price': 999,
            'features': ['All Platinum Features', 'Lifetime Updates', 'Custom Requests', 'Dedicated Support'],
            'emoji': '💠'
        }
    }
    
    def __init__(self, config, db):
        self.config = config
        self.db = db
        self.premium_col = db.premium_users
        self.transactions_col = db.transactions
        self.cache = {}  # In-memory cache
        
    async def init_indexes(self):
        """Initialize MongoDB indexes"""
        try:
            await self.premium_col.create_index("user_id", unique=True)
            await self.premium_col.create_index("expires_at", expireAfterSeconds=0)
            await self.transactions_col.create_index("user_id")
            await self.transactions_col.create_index("created_at")
            logger.info("✅ Premium indexes created")
        except Exception as e:
            logger.warning(f"Premium index creation: {e}")
    
    async def check_premium_status(self, user_id: int) -> Tuple[bool, Optional[Dict]]:
        """Check if user has active premium"""
        try:
            # Check cache first
            if user_id in self.cache:
                cached_data = self.cache[user_id]
                if cached_data['expires_at'] > datetime.now():
                    return True, cached_data
                else:
                    del self.cache[user_id]
            
            # Check database
            premium_data = await self.premium_col.find_one({
                'user_id': user_id,
                'expires_at': {'$gt': datetime.now()}
            })
            
            if premium_data:
                self.cache[user_id] = premium_data
                return True, premium_data
            
            return False, None
            
        except Exception as e:
            logger.error(f"Premium check error: {e}")
            return False, None
    
    async def add_premium(self, user_id: int, plan_type: str, payment_method: str = 'upi', 
                         transaction_id: Optional[str] = None, admin_id: Optional[int] = None) -> bool:
        """Add premium to user"""
        try:
            if plan_type not in self.PREMIUM_PLANS:
                return False
            
            plan = self.PREMIUM_PLANS[plan_type]
            expires_at = datetime.now() + timedelta(days=plan['duration_days'])
            
            premium_data = {
                'user_id': user_id,
                'plan_type': plan_type,
                'plan_name': plan['name'],
                'activated_at': datetime.now(),
                'expires_at': expires_at,
                'payment_method': payment_method,
                'transaction_id': transaction_id,
                'added_by': admin_id,
                'status': 'active'
            }
            
            # Upsert premium data
            await self.premium_col.update_one(
                {'user_id': user_id},
                {'$set': premium_data},
                upsert=True
            )
            
            # Update cache
            self.cache[user_id] = premium_data
            
            # Record transaction
            await self.transactions_col.insert_one({
                'user_id': user_id,
                'plan_type': plan_type,
                'amount': plan['price'],
                'payment_method': payment_method,
                'transaction_id': transaction_id,
                'created_at': datetime.now(),
                'status': 'completed'
            })
            
            logger.info(f"✅ Premium added: User {user_id}, Plan {plan_type}")
            return True
            
        except Exception as e:
            logger.error(f"Add premium error: {e}")
            return False
    
    async def remove_premium(self, user_id: int) -> bool:
        """Remove premium from user"""
        try:
            await self.premium_col.delete_one({'user_id': user_id})
            if user_id in self.cache:
                del self.cache[user_id]
            logger.info(f"✅ Premium removed: User {user_id}")
            return True
        except Exception as e:
            logger.error(f"Remove premium error: {e}")
            return False
    
    async def extend_premium(self, user_id: int, days: int) -> bool:
        """Extend premium duration"""
        try:
            premium_data = await self.premium_col.find_one({'user_id': user_id})
            
            if premium_data:
                new_expiry = max(
                    premium_data['expires_at'],
                    datetime.now()
                ) + timedelta(days=days)
                
                await self.premium_col.update_one(
                    {'user_id': user_id},
                    {'$set': {'expires_at': new_expiry}}
                )
                
                if user_id in self.cache:
                    self.cache[user_id]['expires_at'] = new_expiry
                
                logger.info(f"✅ Premium extended: User {user_id}, +{days} days")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Extend premium error: {e}")
            return False
    
    async def get_premium_users_count(self) -> int:
        """Get total active premium users"""
        try:
            count = await self.premium_col.count_documents({
                'expires_at': {'$gt': datetime.now()}
            })
            return count
        except:
            return 0
    
    async def cleanup_expired_premium(self) -> int:
        """Remove expired premium users"""
        try:
            result = await self.premium_col.delete_many({
                'expires_at': {'$lt': datetime.now()}
            })
            
            # Clear cache of expired users
            expired_users = [uid for uid, data in self.cache.items() 
                           if data['expires_at'] < datetime.now()]
            for uid in expired_users:
                del self.cache[uid]
            
            logger.info(f"✅ Cleaned up {result.deleted_count} expired premium users")
            return result.deleted_count
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            return 0
    
    def get_premium_plans_keyboard(self):
        """Get inline keyboard with premium plans"""
        from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton
        
        buttons = []
        for plan_id, plan in self.PREMIUM_PLANS.items():
            buttons.append([
                InlineKeyboardButton(
                    f"{plan['emoji']} {plan['name']} - ₹{plan['price']} ({plan['duration_days']} days)",
                    callback_data=f"buy_premium_{plan_id}"
                )
            ])
        
        buttons.append([
            InlineKeyboardButton("🔙 Back", callback_data="premium_back")
        ])
        
        return InlineKeyboardMarkup(buttons)
    
    def format_premium_info(self, plan_type: str) -> str:
        """Format premium plan information"""
        if plan_type not in self.PREMIUM_PLANS:
            return "Invalid plan"
        
        plan = self.PREMIUM_PLANS[plan_type]
        features_text = "\n".join([f"  ✓ {feature}" for feature in plan['features']])
        
        return f"""
{plan['emoji']} **{plan['name']}**

**Price:** ₹{plan['price']}
**Duration:** {plan['duration_days']} days

**Features:**
{features_text}

To purchase, click the button below and send payment screenshot.
"""

    async def get_user_premium_info(self, user_id: int) -> str:
        """Get formatted premium info for user"""
        is_premium, data = await self.check_premium_status(user_id)
        
        if not is_premium:
            return "❌ You don't have active premium subscription."
        
        plan = self.PREMIUM_PLANS.get(data['plan_type'], {})
        remaining = (data['expires_at'] - datetime.now()).days
        
        return f"""
✅ **Premium Active**

{plan.get('emoji', '💎')} **Plan:** {data['plan_name']}
📅 **Activated:** {data['activated_at'].strftime('%d %b %Y')}
⏰ **Expires:** {data['expires_at'].strftime('%d %b %Y')}
⏳ **Remaining:** {remaining} days
"""


# Premium command handlers
async def setup_premium_handlers(bot, premium_manager, config):
    """Setup premium-related command handlers"""
    from pyrogram import filters
    from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton
    
    @bot.on_message(filters.command("premium") & filters.private)
    async def premium_command(client, message):
        """Premium plans command"""
        user_id = message.from_user.id
        username = message.from_user.first_name or "User"
        
        # Check current status
        is_premium, data = await premium_manager.check_premium_status(user_id)
        
        if is_premium:
            info_text = await premium_manager.get_user_premium_info(user_id)
            await message.reply_text(
                info_text,
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔄 Renew Premium", callback_data="show_premium_plans")],
                    [InlineKeyboardButton("🌐 Open Website", url=config.WEBSITEURL)]
                ])
            )
        else:
            plans_text = f"""
🌟 **SK4FiLM Premium Plans**

Get premium access and enjoy ad-free experience with fast downloads!

**Choose Your Plan:**
"""
            await message.reply_text(
                plans_text,
                reply_markup=premium_manager.get_premium_plans_keyboard()
            )
    
    @bot.on_message(filters.command("premiumuser") & filters.user(config.ADMINIDS))
    async def premiumuser_command(client, message):
        """Admin command to manage premium for specific user"""
        try:
            parts = message.text.split()
            
            if len(parts) < 2:
                await message.reply_text(
                    "**Usage:**\n"
                    "`/premiumuser <user_id>` - Check status\n"
                    "`/premiumuser <user_id> add <plan>` - Add premium\n"
                    "`/premiumuser <user_id> remove` - Remove premium\n"
                    "`/premiumuser <user_id> extend <days>` - Extend premium\n\n"
                    "**Plans:** bronze, silver, gold, platinum, diamond"
                )
                return
            
            user_id = int(parts[1])
            
            if len(parts) == 2:
                # Check status
                is_premium, data = await premium_manager.check_premium_status(user_id)
                if is_premium:
                    info = await premium_manager.get_user_premium_info(user_id)
                    await message.reply_text(f"**User {user_id}:**\n{info}")
                else:
                    await message.reply_text(f"User {user_id} has no active premium.")
            
            elif parts[2] == "add" and len(parts) >= 4:
                # Add premium
                plan_type = parts[3].lower()
                success = await premium_manager.add_premium(
                    user_id, plan_type, 
                    payment_method='admin', 
                    admin_id=message.from_user.id
                )
                if success:
                    await message.reply_text(f"✅ Premium added for user {user_id}")
                    # Notify user
                    try:
                        await client.send_message(
                            user_id,
                            f"🎉 Congratulations! You have been granted {plan_type.title()} Premium!"
                        )
                    except:
                        pass
                else:
                    await message.reply_text(f"❌ Failed to add premium")
            
            elif parts[2] == "remove":
                # Remove premium
                success = await premium_manager.remove_premium(user_id)
                if success:
                    await message.reply_text(f"✅ Premium removed for user {user_id}")
                else:
                    await message.reply_text(f"❌ Failed to remove premium")
            
            elif parts[2] == "extend" and len(parts) >= 4:
                # Extend premium
                days = int(parts[3])
                success = await premium_manager.extend_premium(user_id, days)
                if success:
                    await message.reply_text(f"✅ Premium extended by {days} days for user {user_id}")
                else:
                    await message.reply_text(f"❌ Failed to extend premium")
        
        except Exception as e:
            await message.reply_text(f"❌ Error: {e}")
    
    logger.info("✅ Premium handlers registered")
