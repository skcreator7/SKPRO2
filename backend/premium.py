"""
premium.py - Premium subscription system with 4 tiers
"""
import asyncio
import secrets
import json
import qrcode
import base64
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from enum import Enum
from io import BytesIO
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class PremiumTier(Enum):
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    GOLD = "gold"
    DIAMOND = "diamond"

class PremiumStatus(Enum):
    ACTIVE = "active"
    PENDING = "pending"
    EXPIRED = "expired"
    CANCELLED = "cancelled"

@dataclass
class PremiumPlan:
    """Premium plan structure"""
    tier: PremiumTier
    name: str
    price: int
    duration_days: int
    features: List[str]
    limits: Dict[str, Any]
    description: str
    upi_id: str
    color_code: str
    icon: str

class PremiumSystem:
    def __init__(self, config, db_client=None):
        self.config = config
        self.db_client = db_client
        
        # Define premium plans with UPI IDs
        self.plans = {
            PremiumTier.BASIC: PremiumPlan(
                tier=PremiumTier.BASIC,
                name="Basic Plan",
                price=10,
                duration_days=15,
                features=[
                    "✅ All Quality (480p-4K)",
                    "✅ Unlimited Downloads",
                    "✅ No Verification Needed",
                    "✅ Basic Support",
                    "✅ No Ads",
                    "✅ Faster Downloads"
                ],
                limits={
                    'daily_downloads': 999,
                    'concurrent_downloads': 3,
                    'quality': ['480p', '720p', '1080p', '2160p'],
                    'priority': 'medium',
                    'verification_bypass': True
                },
                description="Perfect starter plan - Unlimited access for 15 days",
                upi_id=os.environ.get("UPI_ID_BASIC", "sk4filmbot@ybl"),
                color_code="#4CAF50",  # Green
                icon="🥉"
            ),
            PremiumTier.PREMIUM: PremiumPlan(
                tier=PremiumTier.PREMIUM,
                name="Premium Plan",
                price=25,
                duration_days=30,
                features=[
                    "✅ All Quality (480p-4K)",
                    "✅ Unlimited Downloads",
                    "✅ No Verification Needed",
                    "✅ Priority Support",
                    "✅ No Ads",
                    "✅ Faster Downloads",
                    "✅ Batch Downloads"
                ],
                limits={
                    'daily_downloads': 9999,
                    'concurrent_downloads': 5,
                    'quality': ['480p', '720p', '1080p', '2160p'],
                    'priority': 'high',
                    'verification_bypass': True
                },
                description="Best value - Unlimited access for 30 days",
                upi_id=os.environ.get("UPI_ID_PREMIUM", "sk4filmbot@ybl"),
                color_code="#2196F3",  # Blue
                icon="🥈"
            ),
            PremiumTier.GOLD: PremiumPlan(
                tier=PremiumTier.GOLD,
                name="Gold Plan",
                price=35,
                duration_days=60,
                features=[
                    "✅ All Quality (480p-4K)",
                    "✅ Unlimited Downloads",
                    "✅ No Verification Needed",
                    "✅ VIP Support",
                    "✅ No Ads",
                    "✅ Instant Downloads",
                    "✅ Batch Downloads",
                    "✅ Early Access"
                ],
                limits={
                    'daily_downloads': 99999,
                    'concurrent_downloads': 8,
                    'quality': ['480p', '720p', '1080p', '2160p'],
                    'priority': 'vip',
                    'verification_bypass': True
                },
                description="Premium experience - Unlimited access for 60 days",
                upi_id=os.environ.get("UPI_ID_GOLD", "sk4filmbot@ybl"),
                color_code="#FFC107",  # Gold
                icon="🥇"
            ),
            PremiumTier.DIAMOND: PremiumPlan(
                tier=PremiumTier.DIAMOND,
                name="Diamond Plan",
                price=49,
                duration_days=90,
                features=[
                    "✅ All Quality (480p-4K)",
                    "✅ Unlimited Downloads",
                    "✅ No Verification Needed",
                    "✅ VIP Support 24/7",
                    "✅ No Ads",
                    "✅ Instant Downloads",
                    "✅ Batch Downloads",
                    "✅ Early Access",
                    "✅ Custom Requests",
                    "✅ Highest Priority"
                ],
                limits={
                    'daily_downloads': 999999,
                    'concurrent_downloads': 10,
                    'quality': ['480p', '720p', '1080p', '2160p'],
                    'priority': 'highest',
                    'verification_bypass': True
                },
                description="Ultimate experience - Unlimited access for 90 days",
                upi_id=os.environ.get("UPI_ID_DIAMOND", "sk4filmbot@ybl"),
                color_code="#E040FB",  # Purple
                icon="💎"
            )
        }
        
        # Free tier limits
        self.free_limits = {
            'daily_downloads': 5,
            'concurrent_downloads': 1,
            'quality': ['480p'],
            'priority': 'low',
            'verification_bypass': False,
            'verification_duration': 6 * 60 * 60  # 6 hours
        }
        
        # Payment methods
        self.payment_methods = [
            {
                'name': 'UPI',
                'icon': '💸',
                'description': 'Instant UPI Payment',
                'supported': True
            },
            {
                'name': 'Paytm',
                'icon': '📱',
                'description': 'Paytm Wallet',
                'supported': True
            },
            {
                'name': 'PhonePe',
                'icon': '📱',
                'description': 'PhonePe UPI',
                'supported': True
            }
        ]
        
        # User subscriptions cache
        self.user_subscriptions = {}
        self.pending_payments = {}
        self.user_usage = {}
        
        # Admin commands queue
        self.admin_commands = {}
        
        # Cleanup task
        self.cleanup_task = None
    
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
                    '480p Quality Only',
                    '5 Downloads/Day',
                    'URL Verification Required (6 hours)',
                    'Basic Search'
                ],
                'limits': self.free_limits,
                'is_active': False,
                'verification_required': True
            }
        
        plan = self.plans[tier]
        sub_data = self.user_subscriptions.get(user_id, {})
        
        return {
            'user_id': user_id,
            'tier': tier.value,
            'tier_name': plan.name,
            'tier_icon': plan.icon,
            'status': sub_data.get('status', PremiumStatus.ACTIVE.value),
            'expires_at': sub_data.get('expires_at'),
            'purchased_at': sub_data.get('purchased_at'),
            'payment_id': sub_data.get('payment_id'),
            'features': plan.features,
            'limits': plan.limits,
            'is_active': sub_data.get('status') == PremiumStatus.ACTIVE.value,
            'days_remaining': self._calculate_days_remaining(sub_data.get('expires_at')),
            'total_downloads': self.user_usage.get(user_id, {}).get('total_downloads', 0),
            'verification_required': False,  # Premium users don't need verification
            'color_code': plan.color_code
        }
    
    def _calculate_days_remaining(self, expiry_date: Optional[datetime]) -> int:
        """Calculate days remaining in subscription"""
        if not expiry_date:
            return 0
        if datetime.now() > expiry_date:
            return 0
        return (expiry_date - datetime.now()).days
    
    async def generate_payment_qr(self, upi_id: str, amount: int, note: str = "SK4FiLM Premium") -> str:
        """Generate UPI payment QR code as base64"""
        try:
            # Create UPI payment URL
            upi_url = f"upi://pay?pa={upi_id}&pn=SK4FiLM&am={amount}&tn={note}&cu=INR"
            
            # Generate QR code
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4,
            )
            qr.add_data(upi_url)
            qr.make(fit=True)
            
            # Create image
            img = qr.make_image(fill_color="black", back_color="white")
            
            # Convert to base64
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return f"data:image/png;base64,{img_str}"
            
        except Exception as e:
            logger.error(f"QR generation error: {e}")
            return None
    
    async def create_payment_request(self, user_id: int, tier: PremiumTier) -> Dict[str, Any]:
        """Create payment request for premium tier"""
        try:
            plan = self.plans[tier]
            payment_id = f"PMT_{secrets.token_hex(8).upper()}"
            
            # Generate QR code
            qr_code = await self.generate_payment_qr(plan.upi_id, plan.price, f"SK4FiLM {plan.name}")
            
            payment_data = {
                'payment_id': payment_id,
                'user_id': user_id,
                'tier': tier.value,
                'tier_name': plan.name,
                'tier_icon': plan.icon,
                'amount': plan.price,
                'duration_days': plan.duration_days,
                'upi_id': plan.upi_id,
                'qr_code': qr_code,
                'created_at': datetime.now(),
                'expires_at': datetime.now() + timedelta(hours=1),
                'status': 'pending',
                'screenshot_sent': False,
                'admin_notified': False,
                'color_code': plan.color_code
            }
            
            # Store payment data
            self.pending_payments[payment_id] = payment_data
            
            logger.info(f"💰 Payment request created: {payment_id} for user {user_id}")
            
            return payment_data
            
        except Exception as e:
            logger.error(f"Payment request creation error: {e}")
            raise
    
    async def process_payment_screenshot(self, user_id: int, screenshot_message_id: int) -> bool:
        """Process payment screenshot from user"""
        try:
            # Find pending payment for user
            payment_id = None
            for pid, payment in self.pending_payments.items():
                if payment['user_id'] == user_id and payment['status'] == 'pending':
                    payment_id = pid
                    break
            
            if not payment_id:
                return False
            
            # Update payment data
            payment = self.pending_payments[payment_id]
            payment['screenshot_sent'] = True
            payment['screenshot_message_id'] = screenshot_message_id
            payment['screenshot_sent_at'] = datetime.now()
            
            # Notify admin
            await self.notify_admin_payment(payment_id)
            
            logger.info(f"📸 Payment screenshot received: {payment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Payment screenshot processing error: {e}")
            return False
    
    async def notify_admin_payment(self, payment_id: str):
        """Notify admin about pending payment"""
        try:
            payment = self.pending_payments.get(payment_id)
            if not payment:
                return
            
            payment['admin_notified'] = True
            self.admin_commands[payment_id] = {
                'type': 'payment_approval',
                'payment_data': payment,
                'received_at': datetime.now()
            }
            
            logger.info(f"👑 Admin notification: Payment {payment_id} pending approval")
            
        except Exception as e:
            logger.error(f"Admin notification error: {e}")
    
    async def activate_premium(self, admin_id: int, user_id: int, tier: PremiumTier, 
                               payment_id: Optional[str] = None) -> Dict[str, Any]:
        """Activate premium subscription for user (admin only)"""
        try:
            plan = self.plans[tier]
            
            # Check if user already has active subscription
            current_tier = await self.get_user_tier(user_id)
            current_sub = self.user_subscriptions.get(user_id, {})
            
            if current_tier != PremiumTier.FREE:
                # Extend existing subscription
                current_expiry = current_sub.get('expires_at')
                if current_expiry and current_expiry > datetime.now():
                    # Extend from current expiry
                    new_expiry = current_expiry + timedelta(days=plan.duration_days)
                else:
                    # Start from now
                    new_expiry = datetime.now() + timedelta(days=plan.duration_days)
                
                subscription_data = {
                    'user_id': user_id,
                    'tier': tier,
                    'tier_name': plan.name,
                    'tier_icon': plan.icon,
                    'purchased_at': datetime.now(),
                    'expires_at': new_expiry,
                    'payment_id': payment_id or f"ADMIN_{admin_id}_{secrets.token_hex(4)}",
                    'status': PremiumStatus.ACTIVE.value,
                    'activated_by': admin_id,
                    'activated_at': datetime.now(),
                    'is_renewal': True,
                    'previous_tier': current_tier.value,
                    'duration_days': plan.duration_days,
                    'price': plan.price,
                    'color_code': plan.color_code
                }
            else:
                # New subscription
                subscription_data = {
                    'user_id': user_id,
                    'tier': tier,
                    'tier_name': plan.name,
                    'tier_icon': plan.icon,
                    'purchased_at': datetime.now(),
                    'expires_at': datetime.now() + timedelta(days=plan.duration_days),
                    'payment_id': payment_id or f"ADMIN_{admin_id}_{secrets.token_hex(4)}",
                    'status': PremiumStatus.ACTIVE.value,
                    'activated_by': admin_id,
                    'activated_at': datetime.now(),
                    'is_renewal': False,
                    'duration_days': plan.duration_days,
                    'price': plan.price,
                    'color_code': plan.color_code
                }
            
            # Update cache
            self.user_subscriptions[user_id] = subscription_data
            
            # Initialize usage tracking
            if user_id not in self.user_usage:
                self.user_usage[user_id] = {
                    'daily_downloads': 0,
                    'monthly_downloads': 0,
                    'total_downloads': 0,
                    'last_reset': datetime.now().date(),
                    'premium_started': datetime.now()
                }
            
            # Clear pending payment if exists
            if payment_id and payment_id in self.pending_payments:
                del self.pending_payments[payment_id]
            
            logger.info(f"✅ Premium activated: {tier.value} for user {user_id} by admin {admin_id}")
            
            return subscription_data
            
        except Exception as e:
            logger.error(f"Premium activation error: {e}")
            raise
    
    async def can_user_download(self, user_id: int, file_size: int = 0) -> Tuple[bool, str, Dict[str, Any]]:
        """Check if user can download based on tier limits"""
        try:
            # Reset daily usage if needed
            await self._reset_daily_usage_if_needed(user_id)
            
            tier = await self.get_user_tier(user_id)
            
            if tier == PremiumTier.FREE:
                # Free user limits
                user_usage = self.user_usage.get(user_id, {})
                daily_downloads = user_usage.get('daily_downloads', 0)
                
                if daily_downloads >= self.free_limits['daily_downloads']:
                    return False, "Daily download limit reached (5)", {
                        'tier': 'free', 
                        'limit': self.free_limits['daily_downloads'],
                        'needs_verification': True
                    }
                
                return True, "Free download allowed", {
                    'tier': 'free', 
                    'remaining': self.free_limits['daily_downloads'] - daily_downloads,
                    'needs_verification': True
                }
            
            # Premium user - unlimited downloads
            plan = self.plans[tier]
            return True, f"Premium download allowed - {plan.name}", {
                'tier': tier.value,
                'tier_name': plan.name,
                'unlimited': True,
                'quality': plan.limits['quality'],
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
            
            # Record quality
            if 'quality_stats' not in self.user_usage[user_id]:
                self.user_usage[user_id]['quality_stats'] = {}
            
            if quality not in self.user_usage[user_id]['quality_stats']:
                self.user_usage[user_id]['quality_stats'][quality] = 0
            
            self.user_usage[user_id]['quality_stats'][quality] += 1
            
        except Exception as e:
            logger.error(f"Record download error: {e}")
    
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
            logger.error(f"Reset usage error: {e}")
    
    async def get_user_usage(self, user_id: int) -> Dict[str, Any]:
        """Get user usage statistics"""
        if user_id not in self.user_usage:
            return {
                'daily_downloads': 0,
                'monthly_downloads': 0,
                'total_downloads': 0,
                'last_reset': None,
                'premium_started': None
            }
        
        return self.user_usage[user_id].copy()
    
    async def get_all_plans(self) -> List[Dict[str, Any]]:
        """Get information about all available plans"""
        plans_list = []
        
        for tier_enum, plan in self.plans.items():
            plans_list.append({
                'tier': tier_enum.value,
                'name': plan.name,
                'icon': plan.icon,
                'price': plan.price,
                'duration_days': plan.duration_days,
                'features': plan.features,
                'limits': plan.limits,
                'description': plan.description,
                'color_code': plan.color_code,
                'upi_id': plan.upi_id,
                'per_day_cost': round(plan.price / plan.duration_days, 2)
            })
        
        return plans_list
    
    async def get_admin_stats(self) -> Dict[str, Any]:
        """Get statistics for admin panel"""
        try:
            total_premium_users = 0
            active_premium_users = 0
            total_revenue = 0
            pending_payments = 0
            
            for user_id, sub_data in self.user_subscriptions.items():
                if sub_data.get('status') == PremiumStatus.ACTIVE.value:
                    total_premium_users += 1
                    expiry = sub_data.get('expires_at')
                    if expiry and datetime.now() < expiry:
                        active_premium_users += 1
                
                # Calculate revenue
                if sub_data.get('price'):
                    total_revenue += sub_data.get('price', 0)
            
            pending_payments = len(self.pending_payments)
            
            # Plan distribution
            plan_distribution = {}
            for tier in PremiumTier:
                if tier != PremiumTier.FREE:
                    plan_distribution[tier.value] = 0
            
            for user_id, sub_data in self.user_subscriptions.items():
                if sub_data.get('status') == PremiumStatus.ACTIVE.value:
                    tier = sub_data.get('tier')
                    if tier in plan_distribution:
                        plan_distribution[tier] += 1
            
            return {
                'total_premium_users': total_premium_users,
                'active_premium_users': active_premium_users,
                'total_revenue': total_revenue,
                'pending_payments': pending_payments,
                'plan_distribution': plan_distribution,
                'cache_size': len(self.user_subscriptions),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Admin stats error: {e}")
            return {}
    
    async def get_pending_payments(self) -> List[Dict[str, Any]]:
        """Get all pending payments for admin"""
        pending = []
        
        for payment_id, payment in self.pending_payments.items():
            if payment['status'] == 'pending':
                pending.append({
                    'payment_id': payment_id,
                    'user_id': payment['user_id'],
                    'tier': payment['tier'],
                    'tier_name': payment['tier_name'],
                    'amount': payment['amount'],
                    'duration_days': payment['duration_days'],
                    'created_at': payment['created_at'],
                    'screenshot_sent': payment.get('screenshot_sent', False),
                    'color_code': payment.get('color_code', '#2196F3')
                })
        
        return pending
    
    async def broadcast_to_premium_users(self, message: str) -> Dict[str, Any]:
        """Broadcast message to all premium users"""
        try:
            premium_users = []
            for user_id, sub_data in self.user_subscriptions.items():
                if sub_data.get('status') == PremiumStatus.ACTIVE.value:
                    expiry = sub_data.get('expires_at')
                    if expiry and datetime.now() < expiry:
                        premium_users.append(user_id)
            
            return {
                'status': 'success',
                'message': f'Broadcast scheduled for {len(premium_users)} premium users',
                'user_count': len(premium_users),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Broadcast error: {e}")
            return {'status': 'error', 'message': str(e)}
    
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
                
                now = datetime.now()
                expired_users = []
                expired_payments = []
                
                # Cleanup expired subscriptions
                for user_id, sub_data in self.user_subscriptions.items():
                    expiry = sub_data.get('expires_at')
                    if expiry and now > expiry:
                        sub_data['status'] = PremiumStatus.EXPIRED.value
                        expired_users.append(user_id)
                
                # Cleanup expired pending payments
                for payment_id, payment in self.pending_payments.items():
                    expiry = payment.get('expires_at')
                    if expiry and now > expiry:
                        expired_payments.append(payment_id)
                
                for payment_id in expired_payments:
                    del self.pending_payments[payment_id]
                
                if expired_users or expired_payments:
                    logger.info(f"🧹 Cleanup: {len(expired_users)} expired subs, {len(expired_payments)} expired payments")
                    
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
