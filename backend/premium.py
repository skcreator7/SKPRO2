"""
premium.py - Premium subscription system with Razorpay integration
COMPLETE CODE - Razorpay + Referral System + Milestone Rewards
"""

import asyncio
import secrets
import json
import hashlib
import hmac
import base64
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from enum import Enum
import logging
import os
import aiohttp

logger = logging.getLogger(__name__)

class PremiumTier(Enum):
    FREE = "free"
    BASIC = "basic"
    STANDARD = "standard"
    PRO = "pro"
    ULTIMATE = "ultimate"

class PremiumStatus(Enum):
    ACTIVE = "active"
    PENDING = "pending"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"

class PremiumSystem:
    def __init__(self, config, db_manager=None):
        self.config = config
        self.db_manager = db_manager
        self.logger = logger
        
        self.razorpay_key_id = os.environ.get("RAZORPAY_KEY_ID", "")
        self.razorpay_key_secret = os.environ.get("RAZORPAY_KEY_SECRET", "")
        self.razorpay_webhook_secret = os.environ.get("RAZORPAY_WEBHOOK_SECRET", "")
        
        self.COMMON_FEATURES = [
            "✅ All Quality (480p-4K)",
            "✅ Unlimited Downloads",
            "✅ No Verification Needed",
            "✅ VIP Support 24/7",
            "✅ No Ads",
            "✅ Custom Requests"
        ]
        
        self.COMMON_LIMITS = {
            'daily_downloads': 999999,
            'concurrent_downloads': 10,
            'quality': ['480p', '720p', '1080p', '2160p'],
            'priority': 'highest',
            'verification_bypass': True,
            'is_unlimited': True
        }
        
        self.plans = {
            PremiumTier.BASIC: {
                'tier': PremiumTier.BASIC,
                'name': "Basic Plan",
                'price': 9,
                'duration_days': 15,
                'features': self.COMMON_FEATURES,
                'limits': self.COMMON_LIMITS,
                'description': "Starter plan - 15 days",
                'color_code': "#4CAF50",
                'icon': "🥉",
                'per_day_cost': round(9/15, 2)
            },
            PremiumTier.STANDARD: {
                'tier': PremiumTier.STANDARD,
                'name': "Standard Plan",
                'price': 19,
                'duration_days': 28,
                'features': self.COMMON_FEATURES,
                'limits': self.COMMON_LIMITS,
                'description': "Popular plan - 28 days",
                'color_code': "#2196F3",
                'icon': "🥈",
                'per_day_cost': round(19/28, 2)
            },
            PremiumTier.PRO: {
                'tier': PremiumTier.PRO,
                'name': "Pro Plan",
                'price': 29,
                'duration_days': 49,
                'features': self.COMMON_FEATURES,
                'limits': self.COMMON_LIMITS,
                'description': "Best value - 49 days",
                'color_code': "#FF9800",
                'icon': "🥇",
                'per_day_cost': round(29/49, 2)
            },
            PremiumTier.ULTIMATE: {
                'tier': PremiumTier.ULTIMATE,
                'name': "Ultimate Plan",
                'price': 49,
                'duration_days': 90,
                'features': self.COMMON_FEATURES,
                'limits': self.COMMON_LIMITS,
                'description': "Ultimate - 90 days",
                'color_code': "#E040FB",
                'icon': "💎",
                'per_day_cost': round(49/90, 2)
            }
        }
        
        self.free_limits = {
            'daily_downloads': 999999,
            'concurrent_downloads': 2,
            'quality': ['480p', '720p', '1080p', '2160p'],
            'priority': 'medium',
            'verification_bypass': False,
            'verification_duration': 6 * 60 * 60,
            'is_unlimited': True
        }
        
        self.referral_codes = {}
        self.referral_usage = {}
        self.referral_milestones = {
            3: {'tier': PremiumTier.BASIC, 'tier_name': 'Basic Plan', 'days': 15, 'icon': '🥉'},
            5: {'tier': PremiumTier.STANDARD, 'tier_name': 'Standard Plan', 'days': 28, 'icon': '🥈'},
            10: {'tier': PremiumTier.PRO, 'tier_name': 'Pro Plan', 'days': 49, 'icon': '🥇'}
        }
        
        self.referred_user_reward = {'extra_days': 3}
        
        self.user_subscriptions = {}
        self.user_usage = {}
        self.razorpay_orders = {}
        self.cleanup_task = None
        
        self.statistics = {
            'total_downloads': 0,
            'total_data_sent': 0,
            'total_premium_sales': 0,
            'total_revenue': 0,
            'total_referrals': 0,
            'bot_start_time': datetime.now()
        }
        
        self.logger.info("✅ Premium System initialized")
    
    async def create_razorpay_order(self, user_id: int, tier: PremiumTier, referral_code: Optional[str] = None) -> Dict[str, Any]:
        try:
            if not self.razorpay_key_id or not self.razorpay_key_secret:
                return {'success': False, 'error': 'Razorpay not configured'}
            
            plan = self.plans.get(tier)
            if not plan:
                return {'success': False, 'error': 'Invalid plan'}
            
            amount = plan['price'] * 100
            
            referral_applied = False
            if referral_code:
                referral_data = await self.validate_referral_code(referral_code, user_id)
                if referral_data.get('valid'):
                    referral_applied = True
            
            order_data = {
                'amount': amount,
                'currency': 'INR',
                'receipt': f"SK4_{user_id}_{int(datetime.now().timestamp())}",
                'notes': {
                    'user_id': str(user_id),
                    'tier': tier.value,
                    'referral_code': referral_code if referral_applied else ''
                }
            }
            
            async with aiohttp.ClientSession() as session:
                auth = base64.b64encode(f"{self.razorpay_key_id}:{self.razorpay_key_secret}".encode()).decode()
                headers = {'Authorization': f'Basic {auth}', 'Content-Type': 'application/json'}
                
                async with session.post('https://api.razorpay.com/v1/orders', json=order_data, headers=headers) as response:
                    if response.status == 200:
                        razorpay_order = await response.json()
                        self.razorpay_orders[razorpay_order['id']] = {
                            'order_id': razorpay_order['id'],
                            'user_id': user_id,
                            'tier': tier,
                            'amount': amount / 100,
                            'referral_code': referral_code if referral_applied else None,
                            'created_at': datetime.now(),
                            'status': 'created'
                        }
                        return {
                            'success': True,
                            'order_id': razorpay_order['id'],
                            'amount': amount / 100,
                            'key_id': self.razorpay_key_id,
                            'tier': tier.value,
                            'plan_name': plan['name'],
                            'duration_days': plan['duration_days'],
                            'referral_applied': referral_applied,
                            'payment_url': f"https://checkout.razorpay.com/v1/checkout.js?order_id={razorpay_order['id']}&key_id={self.razorpay_key_id}&amount={amount}&currency=INR&name=SK4FiLM&description={plan['name']}&prefill[contact]=&prefill[email]="
                        }
                    else:
                        return {'success': False, 'error': 'Payment gateway error'}
                        
        except Exception as e:
            self.logger.error(f"Razorpay error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def verify_razorpay_payment(self, razorpay_order_id: str, razorpay_payment_id: str, razorpay_signature: str) -> Tuple[bool, str]:
        try:
            message = f"{razorpay_order_id}|{razorpay_payment_id}"
            expected_signature = hmac.new(self.razorpay_key_secret.encode(), message.encode(), hashlib.sha256).hexdigest()
            
            if hmac.compare_digest(expected_signature, razorpay_signature):
                if razorpay_order_id in self.razorpay_orders:
                    order_data = self.razorpay_orders[razorpay_order_id]
                    order_data['status'] = 'paid'
                    order_data['payment_id'] = razorpay_payment_id
                    await self.activate_premium_after_payment(
                        user_id=order_data['user_id'],
                        tier=order_data['tier'],
                        payment_id=razorpay_payment_id,
                        razorpay_order_id=razorpay_order_id,
                        referral_code=order_data.get('referral_code')
                    )
                    return True, "Payment verified"
                return False, "Order not found"
            return False, "Invalid signature"
        except Exception as e:
            return False, str(e)
    
    async def handle_razorpay_webhook(self, webhook_data: Dict[str, Any]) -> Tuple[bool, str]:
        try:
            event = webhook_data.get('event', '')
            payload = webhook_data.get('payload', {})
            
            if event == 'payment.captured':
                payment_entity = payload.get('payment', {}).get('entity', {})
                order_id = payment_entity.get('order_id')
                payment_id = payment_entity.get('id')
                
                if order_id in self.razorpay_orders:
                    order_data = self.razorpay_orders[order_id]
                    await self.activate_premium_after_payment(
                        user_id=order_data['user_id'],
                        tier=order_data['tier'],
                        payment_id=payment_id,
                        razorpay_order_id=order_id,
                        referral_code=order_data.get('referral_code')
                    )
                    return True, "Payment processed"
            
            return True, "Webhook processed"
        except Exception as e:
            return False, str(e)
    
    async def activate_premium_after_payment(self, user_id: int, tier: PremiumTier, payment_id: str, razorpay_order_id: str, referral_code: Optional[str] = None) -> Dict[str, Any]:
        try:
            plan = self.plans[tier]
            duration_days = plan['duration_days']
            
            if referral_code:
                referral_data = await self.validate_referral_code(referral_code, user_id)
                if referral_data.get('valid'):
                    duration_days += self.referred_user_reward['extra_days']
                    await self.process_referral(referral_code, user_id)
            
            current_tier = await self.get_user_tier(user_id)
            current_sub = self.user_subscriptions.get(user_id, {})
            
            if current_tier != PremiumTier.FREE:
                current_expiry = current_sub.get('expires_at')
                if current_expiry and current_expiry > datetime.now():
                    new_expiry = current_expiry + timedelta(days=duration_days)
                else:
                    new_expiry = datetime.now() + timedelta(days=duration_days)
                subscription_data = {**current_sub, 'expires_at': new_expiry, 'is_renewal': True}
            else:
                subscription_data = {
                    'user_id': user_id,
                    'tier': tier,
                    'tier_name': plan['name'],
                    'tier_icon': plan['icon'],
                    'purchased_at': datetime.now(),
                    'expires_at': datetime.now() + timedelta(days=duration_days),
                    'payment_id': payment_id,
                    'razorpay_order_id': razorpay_order_id,
                    'status': PremiumStatus.ACTIVE.value,
                    'duration_days': duration_days,
                    'price': plan['price'],
                    'color_code': plan['color_code'],
                    'referral_code_used': referral_code
                }
            
            self.user_subscriptions[user_id] = subscription_data
            
            if user_id not in self.user_usage:
                self.user_usage[user_id] = {
                    'daily_downloads': 0,
                    'monthly_downloads': 0,
                    'total_downloads': 0,
                    'last_reset': datetime.now().date(),
                    'premium_started': datetime.now()
                }
            
            self.statistics['total_premium_sales'] += 1
            self.statistics['total_revenue'] += plan['price']
            
            return subscription_data
        except Exception as e:
            self.logger.error(f"Activation error: {e}")
            raise
    
    async def generate_referral_code(self, user_id: int) -> str:
        if user_id in self.referral_codes:
            return self.referral_codes[user_id]
        
        while True:
            code = secrets.token_hex(4).upper()
            if code not in self.referral_usage:
                break
        
        self.referral_codes[user_id] = code
        self.referral_usage[code] = {
            'owner_user_id': user_id,
            'created_at': datetime.now(),
            'total_uses': 0,
            'successful_referrals': 0,
            'referred_users': [],
            'milestones_achieved': []
        }
        return code
    
    async def validate_referral_code(self, code: str, user_id: int) -> Dict[str, Any]:
        if not code or code not in self.referral_usage:
            return {'valid': False, 'error': 'Invalid code'}
        
        referral_data = self.referral_usage[code]
        
        if referral_data['owner_user_id'] == user_id:
            return {'valid': False, 'error': 'Cannot use own code'}
        
        if user_id in referral_data.get('referred_users', []):
            return {'valid': False, 'error': 'Already used'}
        
        return {'valid': True, 'code': code, 'extra_days': self.referred_user_reward['extra_days']}
    
    async def process_referral(self, code: str, referred_user_id: int) -> None:
        try:
            if code not in self.referral_usage:
                return
            
            referral_data = self.referral_usage[code]
            referrer_user_id = referral_data['owner_user_id']
            
            referral_data['successful_referrals'] += 1
            referral_data['referred_users'].append(referred_user_id)
            
            self.statistics['total_referrals'] += 1
            
            total = referral_data['successful_referrals']
            
            for milestone, reward in self.referral_milestones.items():
                if total == milestone and milestone not in referral_data['milestones_achieved']:
                    referral_data['milestones_achieved'].append(milestone)
                    await self.give_milestone_reward(referrer_user_id, reward['tier'], reward['days'], milestone)
        except Exception as e:
            self.logger.error(f"Referral error: {e}")
    
    async def give_milestone_reward(self, user_id: int, tier: PremiumTier, days: int, milestone_count: int) -> bool:
        try:
            if user_id in self.user_subscriptions:
                current_sub = self.user_subscriptions[user_id]
                current_expiry = current_sub.get('expires_at')
                if current_expiry and current_expiry > datetime.now():
                    current_sub['expires_at'] = current_expiry + timedelta(days=days)
                else:
                    current_sub['expires_at'] = datetime.now() + timedelta(days=days)
                self.user_subscriptions[user_id] = current_sub
            else:
                plan = self.plans.get(tier, {})
                self.user_subscriptions[user_id] = {
                    'user_id': user_id,
                    'tier': tier,
                    'tier_name': plan.get('name', tier.value),
                    'tier_icon': plan.get('icon', '🎁'),
                    'purchased_at': datetime.now(),
                    'expires_at': datetime.now() + timedelta(days=days),
                    'payment_id': f"REFERRAL_{milestone_count}",
                    'status': PremiumStatus.ACTIVE.value,
                    'is_referral_reward': True,
                    'duration_days': days,
                    'price': 0
                }
            return True
        except Exception as e:
            return False
    
    async def get_referral_info(self, user_id: int) -> Dict[str, Any]:
        code = await self.generate_referral_code(user_id)
        data = self.referral_usage.get(code, {})
        
        return {
            'referral_code': f"SK-{code}",
            'total_referrals': data.get('successful_referrals', 0),
            'milestones_achieved': data.get('milestones_achieved', []),
            'referred_user_reward': self.referred_user_reward['extra_days'],
            'referral_link': f"https://telegram.me/{getattr(self.config, 'BOT_USERNAME', 'sk4filmbot')}?start=SK-{code}",
            'milestones': self.referral_milestones
        }
    
    async def get_user_tier(self, user_id: int) -> PremiumTier:
        if user_id in self.user_subscriptions:
            sub = self.user_subscriptions[user_id]
            expiry = sub.get('expires_at')
            if expiry and datetime.now() < expiry:
                if sub.get('status') == PremiumStatus.ACTIVE.value:
                    return sub.get('tier', PremiumTier.FREE)
        return PremiumTier.FREE
    
    async def is_premium_user(self, user_id: int) -> bool:
        return await self.get_user_tier(user_id) != PremiumTier.FREE
    
    async def get_subscription_details(self, user_id: int) -> Dict[str, Any]:
        tier = await self.get_user_tier(user_id)
        
        if tier == PremiumTier.FREE:
            return {
                'user_id': user_id,
                'tier': 'free',
                'tier_name': 'Free',
                'days_remaining': 0,
                'is_active': False
            }
        
        plan = self.plans.get(tier, {})
        sub = self.user_subscriptions.get(user_id, {})
        days_left = 0
        if sub.get('expires_at'):
            days_left = max(0, (sub['expires_at'] - datetime.now()).days)
        
        return {
            'user_id': user_id,
            'tier': tier.value,
            'tier_name': plan.get('name', tier.value),
            'days_remaining': days_left,
            'expires_at': sub.get('expires_at'),
            'is_active': True
        }
    
    async def record_download(self, user_id: int, file_size: int = 0, quality: str = "480p"):
        if user_id not in self.user_usage:
            self.user_usage[user_id] = {
                'daily_downloads': 0,
                'total_downloads': 0,
                'last_reset': datetime.now().date()
            }
        self.user_usage[user_id]['daily_downloads'] += 1
        self.user_usage[user_id]['total_downloads'] += 1
        self.statistics['total_downloads'] += 1
        self.statistics['total_data_sent'] += file_size
    
    async def get_all_plans(self) -> List[Dict[str, Any]]:
        plans_list = []
        for tier, plan in self.plans.items():
            plans_list.append({
                'tier': tier.value,
                'name': plan['name'],
                'icon': plan['icon'],
                'price': plan['price'],
                'duration_days': plan['duration_days'],
                'features': plan['features'],
                'per_day_cost': plan['per_day_cost']
            })
        return plans_list
    
    async def get_admin_stats(self) -> Dict[str, Any]:
        premium_users = 0
        total_revenue = 0
        for sub in self.user_subscriptions.values():
            if sub.get('status') == PremiumStatus.ACTIVE.value:
                premium_users += 1
                total_revenue += sub.get('price', 0)
        
        return {
            'total_premium_users': premium_users,
            'total_revenue': total_revenue,
            'total_referrals': self.statistics['total_referrals']
        }
    
    async def get_statistics(self) -> Dict[str, Any]:
        return {
            'total_users': len(self.user_subscriptions) + len(self.user_usage),
            'premium_users': sum(1 for s in self.user_subscriptions.values() if s.get('status') == 'active'),
            'total_downloads': self.statistics['total_downloads'],
            'total_revenue': f"₹{self.statistics['total_revenue']}",
            'total_referrals': self.statistics['total_referrals']
        }
    
    async def start_cleanup_task(self):
        if self.cleanup_task:
            self.cleanup_task.cancel()
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _cleanup_loop(self):
        while True:
            try:
                await asyncio.sleep(3600)
                now = datetime.now()
                for user_id, sub in self.user_subscriptions.items():
                    if sub.get('expires_at') and now > sub['expires_at']:
                        sub['status'] = PremiumStatus.EXPIRED.value
            except asyncio.CancelledError:
                break
            except Exception:
                pass
    
    async def stop_cleanup_task(self):
        if self.cleanup_task:
            self.cleanup_task.cancel()
