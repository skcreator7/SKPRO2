"""
premium.py - Premium subscription and tier management
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class PremiumTier(Enum):
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ULTIMATE = "ultimate"

class PremiumFeature:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

class PremiumSystem:
    def __init__(self, config, db_client=None):
        self.config = config
        self.db_client = db_client
        
        # Define tiers and their features
        self.tiers = {
            PremiumTier.FREE: {
                'name': 'Free',
                'price': 0,
                'duration_days': 0,
                'features': [
                    'Basic search',
                    'Limited downloads (5/day)',
                    '480p quality',
                    'Standard support'
                ],
                'limits': {
                    'daily_downloads': 5,
                    'concurrent_downloads': 1,
                    'search_limit': 10,
                    'quality': ['480p']
                }
            },
            PremiumTier.BASIC: {
                'name': 'Basic',
                'price': 49,
                'duration_days': 30,
                'features': [
                    'Unlimited search',
                    'Enhanced downloads (15/day)',
                    '720p quality',
                    'Priority support'
                ],
                'limits': {
                    'daily_downloads': 15,
                    'concurrent_downloads': 2,
                    'search_limit': 50,
                    'quality': ['480p', '720p']
                }
            },
            PremiumTier.PREMIUM: {
                'name': 'Premium',
                'price': 99,
                'duration_days': 30,
                'features': [
                    'Unlimited search',
                    'Unlimited downloads',
                    '1080p quality',
                    'Batch downloads',
                    '24/7 priority support'
                ],
                'limits': {
                    'daily_downloads': 999,
                    'concurrent_downloads': 3,
                    'search_limit': 999,
                    'quality': ['480p', '720p', '1080p']
                }
            },
            PremiumTier.ULTIMATE: {
                'name': 'Ultimate',
                'price': 199,
                'duration_days': 30,
                'features': [
                    'Unlimited search',
                    'Unlimited downloads',
                    '4K/2160p quality',
                    'Batch downloads',
                    'Instant downloads',
                    'VIP support',
                    'Early access'
                ],
                'limits': {
                    'daily_downloads': 9999,
                    'concurrent_downloads': 5,
                    'search_limit': 9999,
                    'quality': ['480p', '720p', '1080p', '2160p']
                }
            }
        }
        
        # User subscriptions cache
        self.user_subscriptions = {}  # user_id -> subscription_data
        self.user_usage = {}          # user_id -> usage_data
        
        # Cleanup task
        self.cleanup_task = None
    
    async def get_user_tier(self, user_id: int) -> PremiumTier:
        """Get user's current premium tier"""
        if user_id in self.user_subscriptions:
            sub_data = self.user_subscriptions[user_id]
            expiry = sub_data.get('expires_at')
            
            if expiry and datetime.now() < expiry:
                return sub_data.get('tier', PremiumTier.FREE)
        
        return PremiumTier.FREE
    
    async def get_subscription_details(self, user_id: int) -> Dict[str, Any]:
        """Get detailed subscription information for user"""
        tier = await self.get_user_tier(user_id)
        tier_info = self.tiers[tier]
        
        # Get subscription data if exists
        sub_data = self.user_subscriptions.get(user_id, {})
        
        return {
            'user_id': user_id,
            'tier': tier.value,
            'tier_name': tier_info['name'],
            'expires_at': sub_data.get('expires_at'),
            'purchased_at': sub_data.get('purchased_at'),
            'features': tier_info['features'],
            'limits': tier_info['limits'],
            'is_active': sub_data.get('is_active', False),
            'days_remaining': self._calculate_days_remaining(sub_data.get('expires_at'))
        }
    
    def _calculate_days_remaining(self, expiry_date: Optional[datetime]) -> int:
        """Calculate days remaining in subscription"""
        if not expiry_date:
            return 0
        if datetime.now() > expiry_date:
            return 0
        return (expiry_date - datetime.now()).days
    
    async def create_subscription(self, user_id: int, tier: PremiumTier, 
                                  payment_id: str, duration_days: int = 30) -> Dict[str, Any]:
        """Create new premium subscription"""
        try:
            tier_info = self.tiers[tier]
            
            subscription_data = {
                'user_id': user_id,
                'tier': tier,
                'tier_name': tier_info['name'],
                'purchased_at': datetime.now(),
                'expires_at': datetime.now() + timedelta(days=duration_days),
                'payment_id': payment_id,
                'is_active': True,
                'duration_days': duration_days,
                'price': tier_info['price']
            }
            
            # Store in cache
            self.user_subscriptions[user_id] = subscription_data
            
            # Initialize usage tracking
            if user_id not in self.user_usage:
                self.user_usage[user_id] = {
                    'daily_downloads': 0,
                    'monthly_downloads': 0,
                    'last_reset': datetime.now().date(),
                    'total_downloads': 0
                }
            
            logger.info(f"✅ Created {tier.value} subscription for user {user_id}")
            
            return subscription_data
            
        except Exception as e:
            logger.error(f"❌ Subscription creation error: {e}")
            raise
    
    async def renew_subscription(self, user_id: int, tier: PremiumTier, 
                                 payment_id: str) -> Dict[str, Any]:
        """Renew existing subscription"""
        try:
            current_sub = self.user_subscriptions.get(user_id, {})
            
            # If expired, start new from now
            # If active, extend from expiry
            current_expiry = current_sub.get('expires_at')
            
            if current_expiry and current_expiry > datetime.now():
                # Extend from current expiry
                new_expiry = current_expiry + timedelta(days=30)
            else:
                # Start from now
                new_expiry = datetime.now() + timedelta(days=30)
            
            tier_info = self.tiers[tier]
            
            subscription_data = {
                'user_id': user_id,
                'tier': tier,
                'tier_name': tier_info['name'],
                'purchased_at': datetime.now(),
                'expires_at': new_expiry,
                'payment_id': payment_id,
                'is_active': True,
                'renewal': True,
                'previous_tier': current_sub.get('tier')
            }
            
            # Update cache
            self.user_subscriptions[user_id] = subscription_data
            
            logger.info(f"✅ Renewed {tier.value} subscription for user {user_id}")
            
            return subscription_data
            
        except Exception as e:
            logger.error(f"❌ Subscription renewal error: {e}")
            raise
    
    async def cancel_subscription(self, user_id: int) -> bool:
        """Cancel user subscription"""
        try:
            if user_id in self.user_subscriptions:
                self.user_subscriptions[user_id]['is_active'] = False
                self.user_subscriptions[user_id]['cancelled_at'] = datetime.now()
                
                logger.info(f"✅ Cancelled subscription for user {user_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"❌ Subscription cancellation error: {e}")
            return False
    
    async def can_user_download(self, user_id: int, file_size: int = 0) -> Tuple[bool, str]:
        """Check if user can download based on tier limits"""
        try:
            # Reset daily usage if needed
            await self._reset_daily_usage_if_needed(user_id)
            
            tier = await self.get_user_tier(user_id)
            tier_info = self.tiers[tier]
            user_usage = self.user_usage.get(user_id, {})
            
            # Check daily download limit
            daily_limit = tier_info['limits']['daily_downloads']
            daily_downloads = user_usage.get('daily_downloads', 0)
            
            if daily_downloads >= daily_limit:
                return False, f"Daily download limit reached ({daily_limit})"
            
            return True, "Can download"
            
        except Exception as e:
            logger.error(f"❌ Download permission check error: {e}")
            return False, "Internal error"
    
    async def record_download(self, user_id: int, file_size: int = 0):
        """Record user download"""
        try:
            if user_id not in self.user_usage:
                self.user_usage[user_id] = {
                    'daily_downloads': 0,
                    'monthly_downloads': 0,
                    'last_reset': datetime.now().date(),
                    'total_downloads': 0
                }
            
            self.user_usage[user_id]['daily_downloads'] += 1
            self.user_usage[user_id]['monthly_downloads'] += 1
            self.user_usage[user_id]['total_downloads'] += 1
            
        except Exception as e:
            logger.error(f"❌ Record download error: {e}")
    
    async def _reset_daily_usage_if_needed(self, user_id: int):
        """Reset daily usage counters if new day"""
        try:
            if user_id in self.user_usage:
                today = datetime.now().date()
                last_reset = self.user_usage[user_id]['last_reset']
                
                if last_reset != today:
                    self.user_usage[user_id]['daily_downloads'] = 0
                    self.user_usage[user_id]['last_reset'] = today
                    
        except Exception as e:
            logger.error(f"❌ Reset usage error: {e}")
    
    async def get_user_usage(self, user_id: int) -> Dict[str, Any]:
        """Get user usage statistics"""
        if user_id not in self.user_usage:
            return {
                'daily_downloads': 0,
                'monthly_downloads': 0,
                'total_downloads': 0,
                'last_reset': None
            }
        
        return self.user_usage[user_id].copy()
    
    async def get_all_tiers(self) -> List[Dict[str, Any]]:
        """Get information about all available tiers"""
        tiers_list = []
        
        for tier_enum, tier_info in self.tiers.items():
            tiers_list.append({
                'tier': tier_enum.value,
                'name': tier_info['name'],
                'price': tier_info['price'],
                'duration_days': tier_info['duration_days'],
                'features': tier_info['features'],
                'limits': tier_info['limits']
            })
        
        return tiers_list
    
    async def get_payment_link(self, user_id: int, tier: PremiumTier) -> str:
        """Generate payment link for tier"""
        # This is a simplified version. In production, integrate with actual payment gateway
        
        tier_info = self.tiers[tier]
        
        # Example using manual payment (replace with actual payment gateway)
        return f"Please contact @Admin to purchase {tier_info['name']} tier for ₹{tier_info['price']}"
    
    async def start_cleanup_task(self):
        """Start background cleanup task"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Deactivate expired subscriptions
                now = datetime.now()
                expired_users = []
                
                for user_id, sub_data in self.user_subscriptions.items():
                    expiry = sub_data.get('expires_at')
                    if expiry and now > expiry:
                        sub_data['is_active'] = False
                        expired_users.append(user_id)
                
                if expired_users:
                    logger.info(f"🧹 Deactivated {len(expired_users)} expired subscriptions")
                    
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
    
    async def stop(self):
        """Stop premium system"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
