"""
premium.py - Premium subscription system with Razorpay integration
UPDATED: Referral system with milestone rewards
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
    BASIC = "basic"      # ₹9 - 15 days
    STANDARD = "standard" # ₹19 - 28 days
    PRO = "pro"          # ₹29 - 49 days
    ULTIMATE = "ultimate" # ₹49 - 90 days

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
        
        # Razorpay Configuration
        self.razorpay_key_id = os.environ.get("RAZORPAY_KEY_ID", "rzp_test_51PfXhWSIYHF8VB")
        self.razorpay_key_secret = os.environ.get("RAZORPAY_KEY_SECRET", "j8XwcLsT0UBb3kUvTXnUvV7F")
        self.razorpay_webhook_secret = os.environ.get("RAZORPAY_WEBHOOK_SECRET", "")
        
        # Common Features for ALL Premium Plans
        self.COMMON_FEATURES = [
            "✅ All Quality (480p-4K)",
            "✅ Unlimited Downloads",
            "✅ No Verification Needed",
            "✅ VIP Support 24/7",
            "✅ No Ads",
            "✅ Custom Requests"
        ]
        
        # Common Limits for ALL Premium Plans
        self.COMMON_LIMITS = {
            'daily_downloads': 999999,
            'concurrent_downloads': 10,
            'quality': ['480p', '720p', '1080p', '2160p'],
            'priority': 'highest',
            'verification_bypass': True,
            'is_unlimited': True
        }
        
        # Define premium plans
        self.plans = {
            PremiumTier.BASIC: {
                'tier': PremiumTier.BASIC,
                'name': "Basic Plan",
                'price': 9,
                'duration_days': 15,
                'features': self.COMMON_FEATURES,
                'limits': self.COMMON_LIMITS,
                'description': "Starter plan - 15 days premium access",
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
                'description': "Popular plan - 28 days premium access",
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
                'description': "Best value - 49 days premium access",
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
                'description': "Ultimate plan - 90 days premium access",
                'color_code': "#E040FB",
                'icon': "💎",
                'per_day_cost': round(49/90, 2)
            }
        }
        
        # Free tier limits
        self.free_limits = {
            'daily_downloads': 999999,
            'concurrent_downloads': 2,
            'quality': ['480p', '720p', '1080p', '2160p'],
            'priority': 'medium',
            'verification_bypass': False,
            'verification_duration': 6 * 60 * 60,
            'is_unlimited': True
        }
        
        # ✅ REFERRAL SYSTEM - UPDATED
        self.referral_codes = {}  # user_id -> referral_code
        self.referral_usage = {}  # referral_code -> usage_data
        
        # ✅ MILESTONE REWARDS - Kitne referrals par kya milega
        self.referral_milestones = {
            3: {
                'tier': PremiumTier.BASIC,
                'tier_name': 'Basic Plan',
                'days': 15,
                'icon': '🥉',
                'message': '3 referrals complete! Basic Plan activated!'
            },
            5: {
                'tier': PremiumTier.STANDARD,
                'tier_name': 'Standard Plan',
                'days': 28,
                'icon': '🥈',
                'message': '5 referrals complete! Standard Plan activated!'
            },
            10: {
                'tier': PremiumTier.PRO,
                'tier_name': 'Pro Plan',
                'days': 49,
                'icon': '🥇',
                'message': '10 referrals complete! Pro Plan activated!'
            }
        }
        
        # Referral rewards for referred user
        self.referred_user_reward = {
            'extra_days': 3,  # Referred user gets 3 extra days
            'discount_percent': 0  # No discount, just extra days
        }
        
        # User subscriptions cache
        self.user_subscriptions = {}
        self.pending_payments = {}
        self.user_usage = {}
        
        # Razorpay order tracking
        self.razorpay_orders = {}
        
        # Cleanup task
        self.cleanup_task = None
        
        # Statistics
        self.statistics = {
            'total_downloads': 0,
            'total_data_sent': 0,
            'total_premium_sales': 0,
            'total_revenue': 0,
            'total_referrals': 0,
            'bot_start_time': datetime.now()
        }
        
        self.logger.info("✅ Premium System initialized with Razorpay + Milestone Referral System")
    
    # ============================================================================
    # ✅ RAZORPAY INTEGRATION (SAME AS BEFORE)
    # ============================================================================
    
    async def create_razorpay_order(self, user_id: int, tier: PremiumTier, 
                                   referral_code: Optional[str] = None) -> Dict[str, Any]:
        """Create Razorpay order for premium purchase"""
        try:
            if not self.razorpay_key_id or not self.razorpay_key_secret:
                self.logger.error("❌ Razorpay credentials not configured")
                return {'success': False, 'error': 'Razorpay not configured'}
            
            plan = self.plans.get(tier)
            if not plan:
                return {'success': False, 'error': 'Invalid plan'}
            
            amount = plan['price'] * 100
            
            referral_applied = False
            if referral_code:
                referral_data = await self.validate_referral_code(referral_code, user_id)
                if referral_data['valid']:
                    referral_applied = True
                    self.logger.info(f"Referral code applied: {referral_code}")
            
            order_data = {
                'amount': amount,
                'currency': 'INR',
                'receipt': f"SK4_{user_id}_{int(datetime.now().timestamp())}",
                'notes': {
                    'user_id': str(user_id),
                    'tier': tier.value,
                    'plan_name': plan['name'],
                    'duration_days': plan['duration_days'],
                    'referral_code': referral_code if referral_applied else ''
                }
            }
            
            async with aiohttp.ClientSession() as session:
                auth = base64.b64encode(
                    f"{self.razorpay_key_id}:{self.razorpay_key_secret}".encode()
                ).decode()
                
                headers = {
                    'Authorization': f'Basic {auth}',
                    'Content-Type': 'application/json'
                }
                
                async with session.post(
                    'https://api.razorpay.com/v1/orders',
                    json=order_data,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        razorpay_order = await response.json()
                        
                        self.razorpay_orders[razorpay_order['id']] = {
                            'order_id': razorpay_order['id'],
                            'user_id': user_id,
                            'tier': tier,
                            'amount': amount / 100,
                            'referral_code': referral_code if referral_applied else None,
                            'created_at': datetime.now(),
                            'status': 'created',
                            'razorpay_data': razorpay_order
                        }
                        
                        return {
                            'success': True,
                            'order_id': razorpay_order['id'],
                            'amount': amount / 100,
                            'currency': 'INR',
                            'key_id': self.razorpay_key_id,
                            'user_id': user_id,
                            'tier': tier.value,
                            'plan_name': plan['name'],
                            'duration_days': plan['duration_days'],
                            'referral_applied': referral_applied
                        }
                    else:
                        error_data = await response.text()
                        self.logger.error(f"Razorpay API error: {error_data}")
                        return {'success': False, 'error': 'Payment gateway error'}
                        
        except Exception as e:
            self.logger.error(f"Razorpay order creation error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def verify_razorpay_payment(self, razorpay_order_id: str, 
                                     razorpay_payment_id: str, 
                                     razorpay_signature: str) -> Tuple[bool, str]:
        """Verify Razorpay payment signature"""
        try:
            message = f"{razorpay_order_id}|{razorpay_payment_id}"
            expected_signature = hmac.new(
                self.razorpay_key_secret.encode(),
                message.encode(),
                hashlib.sha256
            ).hexdigest()
            
            if hmac.compare_digest(expected_signature, razorpay_signature):
                if razorpay_order_id in self.razorpay_orders:
                    order_data = self.razorpay_orders[razorpay_order_id]
                    order_data['payment_id'] = razorpay_payment_id
                    order_data['status'] = 'paid'
                    order_data['paid_at'] = datetime.now()
                    
                    await self.activate_premium_after_payment(
                        user_id=order_data['user_id'],
                        tier=order_data['tier'],
                        payment_id=razorpay_payment_id,
                        razorpay_order_id=razorpay_order_id,
                        referral_code=order_data.get('referral_code')
                    )
                    
                    return True, "Payment verified successfully"
                else:
                    return False, "Order not found"
            else:
                return False, "Invalid signature"
                
        except Exception as e:
            self.logger.error(f"Payment verification error: {e}")
            return False, str(e)
    
    async def handle_razorpay_webhook(self, webhook_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Handle Razorpay webhook events"""
        try:
            event = webhook_data.get('event', '')
            payload = webhook_data.get('payload', {})
            
            if event == 'payment.captured':
                payment_entity = payload.get('payment', {}).get('entity', {})
                order_id = payment_entity.get('order_id')
                payment_id = payment_entity.get('id')
                status = payment_entity.get('status')
                
                if order_id in self.razorpay_orders and status == 'captured':
                    order_data = self.razorpay_orders[order_id]
                    order_data['payment_id'] = payment_id
                    order_data['status'] = 'paid'
                    order_data['paid_at'] = datetime.now()
                    
                    await self.activate_premium_after_payment(
                        user_id=order_data['user_id'],
                        tier=order_data['tier'],
                        payment_id=payment_id,
                        razorpay_order_id=order_id,
                        referral_code=order_data.get('referral_code')
                    )
                    
                    self.logger.info(f"✅ Webhook: Payment captured for order {order_id}")
                    return True, "Payment processed"
                    
            elif event == 'payment.failed':
                payment_entity = payload.get('payment', {}).get('entity', {})
                order_id = payment_entity.get('order_id')
                
                if order_id in self.razorpay_orders:
                    self.razorpay_orders[order_id]['status'] = 'failed'
                    
                return True, "Payment failed recorded"
            
            return True, "Webhook processed"
            
        except Exception as e:
            self.logger.error(f"Webhook handling error: {e}")
            return False, str(e)
    
    async def activate_premium_after_payment(self, user_id: int, tier: PremiumTier, 
                                            payment_id: str, razorpay_order_id: str,
                                            referral_code: Optional[str] = None) -> Dict[str, Any]:
        """Activate premium after successful payment"""
        try:
            plan = self.plans[tier]
            duration_days = plan['duration_days']
            
            # Apply referral bonus for referred user
            if referral_code:
                referral_data = await self.validate_referral_code(referral_code, user_id)
                if referral_data['valid']:
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
                
                subscription_data = {
                    **current_sub,
                    'expires_at': new_expiry,
                    'is_renewal': True,
                    'previous_tier': current_tier.value,
                    'last_payment_id': payment_id,
                    'last_payment_at': datetime.now()
                }
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
                    'is_renewal': False,
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
            
            self.logger.info(f"✅ Premium activated: {tier.value} for user {user_id}")
            return subscription_data
            
        except Exception as e:
            self.logger.error(f"Premium activation error: {e}")
            raise
    
    # ============================================================================
    # ✅ REFERRAL SYSTEM - UPDATED WITH MILESTONES
    # ============================================================================
    
    async def generate_referral_code(self, user_id: int) -> str:
        """Generate unique referral code for user"""
        if user_id in self.referral_codes:
            return self.referral_codes[user_id]
        
        while True:
            code = secrets.token_hex(4).upper()  # 8 characters
            if code not in self.referral_usage:
                break
        
        self.referral_codes[user_id] = code
        self.referral_usage[code] = {
            'owner_user_id': user_id,
            'created_at': datetime.now(),
            'total_uses': 0,
            'successful_referrals': 0,
            'referred_users': [],
            'milestones_achieved': [],
            'total_reward_days': 0
        }
        
        return code
    
    async def validate_referral_code(self, code: str, user_id: int) -> Dict[str, Any]:
        """Validate referral code"""
        if not code or code not in self.referral_usage:
            return {'valid': False, 'error': 'Invalid referral code'}
        
        referral_data = self.referral_usage[code]
        
        if referral_data['owner_user_id'] == user_id:
            return {'valid': False, 'error': 'Cannot use your own referral code'}
        
        if user_id in referral_data.get('referred_users', []):
            return {'valid': False, 'error': 'Referral code already used'}
        
        return {
            'valid': True,
            'code': code,
            'owner_user_id': referral_data['owner_user_id'],
            'extra_days': self.referred_user_reward['extra_days']
        }
    
    async def process_referral(self, code: str, referred_user_id: int) -> None:
        """Process referral and check milestones"""
        try:
            if code not in self.referral_usage:
                return
            
            referral_data = self.referral_usage[code]
            referrer_user_id = referral_data['owner_user_id']
            
            # Update referral stats
            referral_data['total_uses'] += 1
            referral_data['successful_referrals'] += 1
            referral_data['referred_users'].append(referred_user_id)
            referral_data['last_used_at'] = datetime.now()
            
            self.statistics['total_referrals'] += 1
            
            # ✅ CHECK MILESTONES
            total_referrals = referral_data['successful_referrals']
            
            for milestone_count, reward in self.referral_milestones.items():
                if total_referrals == milestone_count and milestone_count not in referral_data['milestones_achieved']:
                    # Milestone achieved!
                    referral_data['milestones_achieved'].append(milestone_count)
                    
                    # Give reward to referrer
                    await self.give_milestone_reward(
                        user_id=referrer_user_id,
                        tier=reward['tier'],
                        days=reward['days'],
                        milestone_count=milestone_count
                    )
                    
                    self.logger.info(f"🎉 Milestone {milestone_count} achieved by user {referrer_user_id}!")
            
            self.logger.info(f"✅ Referral processed: {code} used by {referred_user_id}")
            
        except Exception as e:
            self.logger.error(f"Referral processing error: {e}")
    
    async def give_milestone_reward(self, user_id: int, tier: PremiumTier, 
                                   days: int, milestone_count: int) -> bool:
        """Give milestone reward to referrer"""
        try:
            plan = self.plans.get(tier, {})
            
            if user_id in self.user_subscriptions:
                # Extend existing subscription
                current_sub = self.user_subscriptions[user_id]
                current_expiry = current_sub.get('expires_at')
                
                if current_expiry and current_expiry > datetime.now():
                    current_sub['expires_at'] = current_expiry + timedelta(days=days)
                else:
                    current_sub['expires_at'] = datetime.now() + timedelta(days=days)
                
                current_sub['referral_rewards'] = current_sub.get('referral_rewards', 0) + days
                current_sub['milestones_achieved'] = current_sub.get('milestones_achieved', []) + [milestone_count]
                self.user_subscriptions[user_id] = current_sub
            else:
                # New subscription from referral
                self.user_subscriptions[user_id] = {
                    'user_id': user_id,
                    'tier': tier,
                    'tier_name': plan.get('name', tier.value),
                    'tier_icon': plan.get('icon', '🎁'),
                    'purchased_at': datetime.now(),
                    'expires_at': datetime.now() + timedelta(days=days),
                    'payment_id': f"REFERRAL_{milestone_count}_{secrets.token_hex(4)}",
                    'status': PremiumStatus.ACTIVE.value,
                    'is_referral_reward': True,
                    'duration_days': days,
                    'price': 0,
                    'color_code': plan.get('color_code', '#4CAF50'),
                    'milestone_count': milestone_count
                }
            
            if user_id not in self.user_usage:
                self.user_usage[user_id] = {
                    'daily_downloads': 0,
                    'monthly_downloads': 0,
                    'total_downloads': 0,
                    'last_reset': datetime.now().date(),
                    'premium_started': datetime.now()
                }
            
            self.logger.info(f"🎉 Milestone reward: {days} days {tier.value} for user {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Milestone reward error: {e}")
            return False
    
    async def get_referral_info(self, user_id: int) -> Dict[str, Any]:
        """Get user's referral information with milestones"""
        code = await self.generate_referral_code(user_id)
        referral_data = self.referral_usage.get(code, {})
        
        total_referrals = referral_data.get('successful_referrals', 0)
        
        # Calculate next milestone
        next_milestone = None
        progress_to_next = 0
        
        for milestone_count in sorted(self.referral_milestones.keys()):
            if total_referrals < milestone_count:
                next_milestone = {
                    'count': milestone_count,
                    'reward': self.referral_milestones[milestone_count]
                }
                progress_to_next = total_referrals
                break
        
        return {
            'referral_code': code,
            'total_referrals': total_referrals,
            'milestones_achieved': referral_data.get('milestones_achieved', []),
            'total_reward_days': referral_data.get('total_reward_days', 0),
            'next_milestone': next_milestone,
            'progress_to_next': progress_to_next,
            'referred_user_reward': self.referred_user_reward['extra_days'],
            'referral_link': f"https://t.me/{getattr(self.config, 'BOT_USERNAME', 'sk4filmbot')}?start=ref_{code}",
            'milestones': self.referral_milestones
        }
    
    # ============================================================================
    # ✅ USER METHODS (SAME AS BEFORE)
    # ============================================================================
    
    async def get_user_tier(self, user_id: int) -> PremiumTier:
        """Get user's current premium tier"""
        if user_id in self.user_subscriptions:
            sub_data = self.user_subscriptions[user_id]
            expiry = sub_data.get('expires_at')
            
            if expiry and datetime.now() < expiry:
                if sub_data.get('status') == PremiumStatus.ACTIVE.value:
                    return sub_data.get('tier', PremiumTier.FREE)
        
        return PremiumTier.FREE
    
    async def is_premium_user(self, user_id: int) -> bool:
        """Check if user has active premium"""
        return await self.get_user_tier(user_id) != PremiumTier.FREE
    
    async def get_subscription_details(self, user_id: int) -> Dict[str, Any]:
        """Get detailed subscription information for user"""
        tier = await self.get_user_tier(user_id)
        
        if tier == PremiumTier.FREE:
            return {
                'user_id': user_id,
                'tier': PremiumTier.FREE.value,
                'tier_name': 'Free',
                'status': 'free',
                'expires_at': None,
                'days_remaining': 0,
                'features': [
                    '✅ All Quality (480p-4K)',
                    '✅ Unlimited Downloads',
                    '🔒 URL Verification Required (6 hours)',
                    '✅ Basic Search',
                    '✅ No Ads'
                ],
                'limits': self.free_limits,
                'is_active': True,
                'verification_required': True,
                'verification_hours': 6,
                'is_unlimited': True
            }
        
        plan = self.plans.get(tier, {})
        sub_data = self.user_subscriptions.get(user_id, {})
        
        days_left = 0
        expires_at = sub_data.get('expires_at')
        if expires_at:
            days_left = max(0, (expires_at - datetime.now()).days)
        
        return {
            'user_id': user_id,
            'tier': tier.value,
            'tier_name': plan.get('name', tier.value),
            'tier_icon': plan.get('icon', '⭐'),
            'status': sub_data.get('status', PremiumStatus.ACTIVE.value),
            'expires_at': expires_at,
            'purchased_at': sub_data.get('purchased_at'),
            'payment_id': sub_data.get('payment_id'),
            'features': plan.get('features', self.COMMON_FEATURES),
            'limits': plan.get('limits', self.COMMON_LIMITS),
            'is_active': sub_data.get('status') == PremiumStatus.ACTIVE.value,
            'days_remaining': days_left,
            'total_downloads': self.user_usage.get(user_id, {}).get('total_downloads', 0),
            'verification_required': False,
            'color_code': plan.get('color_code', '#2196F3')
        }
    
    async def can_user_download(self, user_id: int, file_size: int = 0) -> Tuple[bool, str, Dict[str, Any]]:
        """Check if user can download"""
        try:
            tier = await self.get_user_tier(user_id)
            
            if tier == PremiumTier.FREE:
                return True, "Free download allowed - Unlimited (Verification required)", {
                    'tier': 'free',
                    'unlimited': True,
                    'quality': self.free_limits['quality'],
                    'needs_verification': True,
                    'verification_hours': 6
                }
            
            plan = self.plans.get(tier, {})
            return True, f"Premium download allowed - {plan.get('name', 'Premium')}", {
                'tier': tier.value,
                'tier_name': plan.get('name', 'Premium'),
                'unlimited': True,
                'quality': plan.get('limits', {}).get('quality', []),
                'needs_verification': False
            }
            
        except Exception as e:
            logger.error(f"Download permission check error: {e}")
            return False, "Internal error", {'tier': 'error'}
    
    async def record_download(self, user_id: int, file_size: int = 0, quality: str = "480p"):
        """Record user download"""
        try:
            if user_id not in self.user_usage:
                self.user_usage[user_id] = {
                    'daily_downloads': 0,
                    'monthly_downloads': 0,
                    'total_downloads': 0,
                    'last_reset': datetime.now().date(),
                    'premium_started': None
                }
            
            self.user_usage[user_id]['daily_downloads'] += 1
            self.user_usage[user_id]['monthly_downloads'] += 1
            self.user_usage[user_id]['total_downloads'] += 1
            
            self.statistics['total_downloads'] += 1
            self.statistics['total_data_sent'] += file_size
            
            if 'quality_stats' not in self.user_usage[user_id]:
                self.user_usage[user_id]['quality_stats'] = {}
            
            if quality not in self.user_usage[user_id]['quality_stats']:
                self.user_usage[user_id]['quality_stats'][quality] = 0
            
            self.user_usage[user_id]['quality_stats'][quality] += 1
            
        except Exception as e:
            logger.error(f"Record download error: {e}")
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get bot statistics"""
        try:
            total_users = len(self.user_subscriptions) + len(self.user_usage)
            premium_users = 0
            active_premium = 0
            
            for user_id, sub_data in self.user_subscriptions.items():
                if sub_data.get('status') == PremiumStatus.ACTIVE.value:
                    premium_users += 1
                    expiry = sub_data.get('expires_at')
                    if expiry and datetime.now() < expiry:
                        active_premium += 1
            
            uptime = datetime.now() - self.statistics['bot_start_time']
            days = uptime.days
            hours = uptime.seconds // 3600
            minutes = (uptime.seconds % 3600) // 60
            
            data_sent_gb = self.statistics['total_data_sent'] / (1024 ** 3)
            
            return {
                'total_users': total_users,
                'premium_users': premium_users,
                'active_premium': active_premium,
                'free_users': total_users - premium_users,
                'total_downloads': self.statistics['total_downloads'],
                'total_data_sent': f"{data_sent_gb:.2f} GB",
                'total_premium_sales': self.statistics['total_premium_sales'],
                'total_revenue': f"₹{self.statistics['total_revenue']}",
                'total_referrals': self.statistics['total_referrals'],
                'uptime': f"{days}d {hours}h {minutes}m",
                'server_time': datetime.now().strftime('%d %b %Y, %H:%M:%S')
            }
            
        except Exception as e:
            logger.error(f"Statistics error: {e}")
            return {}
    
    async def start_cleanup_task(self):
        """Start background cleanup task"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("🧹 Premium cleanup task started")
    
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while True:
            try:
                await asyncio.sleep(3600)
                
                now = datetime.now()
                expired_users = []
                expired_orders = []
                
                for user_id, sub_data in self.user_subscriptions.items():
                    expiry = sub_data.get('expires_at')
                    if expiry and now > expiry:
                        sub_data['status'] = PremiumStatus.EXPIRED.value
                        expired_users.append(user_id)
                
                for order_id, order_data in self.razorpay_orders.items():
                    if order_data.get('status') == 'created':
                        created_at = order_data.get('created_at')
                        if created_at and (now - created_at).total_seconds() > 3600:
                            expired_orders.append(order_id)
                
                for order_id in expired_orders:
                    del self.razorpay_orders[order_id]
                
                if expired_users or expired_orders:
                    logger.info(f"🧹 Premium cleanup: {len(expired_users)} subscriptions, {len(expired_orders)} orders expired")
                    
            except asyncio.CancelledError:
                logger.info("🧹 Premium cleanup task cancelled")
                break
            except Exception as e:
                logger.error(f"Premium cleanup loop error: {e}")
    
    async def stop_cleanup_task(self):
        """Stop background cleanup task"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
            logger.info("🧹 Premium cleanup task stopped")
    
    async def get_all_plans(self) -> List[Dict[str, Any]]:
        """Get information about all available plans"""
        plans_list = []
        
        for tier_enum, plan in self.plans.items():
            plans_list.append({
                'tier': tier_enum.value,
                'name': plan['name'],
                'icon': plan['icon'],
                'price': plan['price'],
                'duration_days': plan['duration_days'],
                'features': plan['features'],
                'limits': plan['limits'],
                'description': plan['description'],
                'color_code': plan['color_code'],
                'per_day_cost': plan['per_day_cost']
            })
        
        return plans_list
