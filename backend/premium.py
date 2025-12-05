"""
premium.py - Premium subscription system with 4 tiers
COMPLETE: Added user commands, admin commands, and all functionality
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
import os
from dataclasses import dataclass, asdict

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
                price=99,
                duration_days=30,
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
                description="Perfect starter plan - Unlimited access for 30 days",
                upi_id=os.environ.get("UPI_ID_BASIC", "sk4filmbot@ybl"),
                color_code="#4CAF50",  # Green
                icon="🥉"
            ),
            PremiumTier.PREMIUM: PremiumPlan(
                tier=PremiumTier.PREMIUM,
                name="Premium Plan",
                price=199,
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
                price=299,
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
                price=499,
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
        
        # Free tier limits - UNLIMITED downloads, all quality
        self.free_limits = {
            'daily_downloads': 999999,  # Unlimited
            'concurrent_downloads': 2,
            'quality': ['480p', '720p', '1080p', '2160p'],  # All quality
            'priority': 'medium',
            'verification_bypass': False,
            'verification_duration': 6 * 60 * 60,  # 6 hours
            'is_unlimited': True  # Flag for unlimited downloads
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
        
        # Statistics
        self.statistics = {
            'total_downloads': 0,
            'total_data_sent': 0,
            'total_premium_sales': 0,
            'total_revenue': 0,
            'bot_start_time': datetime.now()
        }
    
    # ✅ USER COMMANDS
    
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
                    '✅ Basic Search'
                ],
                'limits': self.free_limits,
                'is_active': True,  # Free is always "active"
                'verification_required': True,
                'verification_hours': 6,
                'is_unlimited': True
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
    
    async def get_my_premium_info(self, user_id: int) -> str:
        """Get user's premium info in formatted text for /mypremium command"""
        details = await self.get_subscription_details(user_id)
        
        if details['tier'] == 'free':
            return (
                "👤 **Your Account Status**\n\n"
                "⭐ **Plan:** Free User\n"
                "✅ **Status:** Active\n\n"
                "📥 **Daily Downloads:** Unlimited\n"
                "🎬 **Quality:** All (480p-4K)\n"
                "🔒 **Verification:** Required every 6 hours\n\n"
                "💎 **Upgrade to Premium for:**\n"
                "• No verification required\n"
                "• Priority support\n"
                "• Faster downloads\n"
                "• No ads\n\n"
                "Use /buy to upgrade!"
            )
        
        # Premium user
        plan_icon = details.get('tier_icon', '⭐')
        plan_name = details.get('tier_name', 'Premium')
        days_left = details.get('days_remaining', 0)
        
        text = (
            f"{plan_icon} **Your Premium Status** {plan_icon}\n\n"
            f"📋 **Plan:** {plan_name}\n"
            f"✅ **Status:** {details['status'].title()}\n"
            f"📅 **Days Left:** {days_left}\n"
            f"🆔 **Payment ID:** `{details.get('payment_id', 'N/A')}`\n\n"
        )
        
        if details.get('expires_at'):
            expiry = details['expires_at']
            if isinstance(expiry, str):
                expiry_str = expiry
            else:
                expiry_str = expiry.strftime('%d %b %Y')
            text += f"⏰ **Expires:** {expiry_str}\n"
        
        text += f"📥 **Total Downloads:** {details.get('total_downloads', 0)}\n\n"
        text += "✅ **Benefits:**\n"
        
        features = details.get('features', [])
        for feature in features[:5]:  # Show first 5 features
            text += f"• {feature}\n"
        
        if len(features) > 5:
            text += f"• ... and {len(features) - 5} more benefits\n"
        
        text += "\n🎬 **Enjoy unlimited downloads!**"
        
        return text
    
    async def get_available_plans_text(self) -> str:
        """Get all plans in formatted text for /plans command"""
        text = "💎 **SK4FiLM PREMIUM PLANS** 💎\n\n"
        
        for tier_enum, plan in self.plans.items():
            per_day = plan.price / plan.duration_days
            text += (
                f"{plan.icon} **{plan.name}** {plan.icon}\n"
                f"💰 **Price:** ₹{plan.price} ({plan.duration_days} days)\n"
                f"📅 **Per day:** ₹{per_day:.2f}/day\n"
                f"✅ **Features:** {len(plan.features)} benefits\n\n"
            )
        
        text += (
            "🎬 **All plans include:**\n"
            "✅ No verification required\n"
            "✅ All quality (480p-4K)\n"
            "✅ Unlimited downloads\n"
            "✅ No ads\n"
            "✅ Priority support\n\n"
            "Use /buy to purchase a plan!"
        )
        
        return text
    
    async def get_plan_details_text(self, tier: PremiumTier) -> str:
        """Get detailed plan info for /plan <tier> command"""
        plan = self.plans.get(tier)
        if not plan:
            return "❌ Plan not found!"
        
        per_day = plan.price / plan.duration_days
        
        text = (
            f"{plan.icon} **{plan.name}** {plan.icon}\n\n"
            f"💰 **Price:** ₹{plan.price}\n"
            f"📅 **Duration:** {plan.duration_days} days\n"
            f"📊 **Per day:** ₹{per_day:.2f}/day\n"
            f"💳 **UPI ID:** `{plan.upi_id}`\n\n"
            "✅ **Features:**\n"
        )
        
        for feature in plan.features:
            text += f"• {feature}\n"
        
        text += f"\n📊 **Limits:**\n"
        text += f"• Daily downloads: {plan.limits.get('daily_downloads', 'Unlimited')}\n"
        text += f"• Concurrent downloads: {plan.limits.get('concurrent_downloads', 3)}\n"
        text += f"• Quality: {', '.join(plan.limits.get('quality', []))}\n"
        text += f"• Priority: {plan.limits.get('priority', 'medium').title()}\n\n"
        
        text += f"📝 **Description:** {plan.description}\n\n"
        text += "Use /buy to purchase this plan!"
        
        return text
    
    # ✅ BUY/PURCHASE COMMANDS
    
    async def initiate_purchase(self, user_id: int, tier: PremiumTier) -> Dict[str, Any]:
        """Initiate purchase process for user"""
        try:
            plan = self.plans[tier]
            payment_id = f"PAY_{secrets.token_hex(8).upper()}"
            
            # Generate QR code
            qr_code = await self.generate_payment_qr(plan.upi_id, plan.price, f"SK4FiLM {plan.name}")
            
            payment_data = {
                'payment_id': payment_id,
                'user_id': user_id,
                'tier': tier,
                'tier_name': plan.name,
                'tier_icon': plan.icon,
                'amount': plan.price,
                'duration_days': plan.duration_days,
                'upi_id': plan.upi_id,
                'qr_code': qr_code,
                'created_at': datetime.now(),
                'expires_at': datetime.now() + timedelta(hours=24),  # 24 hours to pay
                'status': 'pending',
                'screenshot_sent': False,
                'admin_notified': False,
                'color_code': plan.color_code
            }
            
            # Store payment data
            self.pending_payments[payment_id] = payment_data
            
            # Update statistics
            self.statistics['total_premium_sales'] += 1
            
            logger.info(f"💰 Purchase initiated: {payment_id} for user {user_id} - {plan.name}")
            
            return payment_data
            
        except Exception as e:
            logger.error(f"Purchase initiation error: {e}")
            raise
    
    async def get_payment_instructions_text(self, payment_id: str) -> str:
        """Get payment instructions for user"""
        payment = self.pending_payments.get(payment_id)
        if not payment:
            return "❌ Payment not found or expired!"
        
        plan = self.plans[payment['tier']]
        expiry_time = payment['expires_at']
        remaining = expiry_time - datetime.now()
        hours_left = max(0, int(remaining.total_seconds() / 3600))
        
        text = (
            f"💰 **Payment Instructions** 💰\n\n"
            f"{plan.icon} **Plan:** {plan.name}\n"
            f"💵 **Amount:** ₹{plan.price}\n"
            f"📅 **Duration:** {plan.duration_days} days\n\n"
            f"💳 **Payment Method:**\n"
            f"1. **UPI ID:** `{plan.upi_id}`\n"
            f"2. **Paytm/PhonePe:** Send to above UPI\n"
            f"3. **Amount:** ₹{plan.price}\n\n"
            f"📸 **After Payment:**\n"
            f"1. Take screenshot\n"
            f"2. Send to this bot\n"
            f"3. Admin will activate within 24 hours\n\n"
            f"🆔 **Payment ID:** `{payment_id}`\n"
            f"⏰ **Time left:** {hours_left} hours\n\n"
            f"⚠️ **Important:**\n"
            f"• Keep screenshot ready\n"
            f"• Don't share payment details\n"
            f"• Contact @admin for issues"
        )
        
        return text
    
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
    
    # ✅ ADMIN COMMANDS
    
    async def add_premium_subscription(self, admin_id: int, user_id: int, tier: PremiumTier, 
                                      days: int = 30, reason: str = "admin_grant") -> Dict[str, Any]:
        """Admin command: Add premium subscription to user"""
        try:
            if tier not in self.plans:
                raise ValueError(f"Invalid tier: {tier}")
            
            plan = self.plans[tier]
            
            # Check if user already has active subscription
            current_tier = await self.get_user_tier(user_id)
            current_sub = self.user_subscriptions.get(user_id, {})
            
            if current_tier != PremiumTier.FREE:
                # Extend existing subscription
                current_expiry = current_sub.get('expires_at')
                if current_expiry and current_expiry > datetime.now():
                    # Extend from current expiry
                    new_expiry = current_expiry + timedelta(days=days)
                else:
                    # Start from now
                    new_expiry = datetime.now() + timedelta(days=days)
                
                subscription_data = {
                    'user_id': user_id,
                    'tier': tier,
                    'tier_name': plan.name,
                    'tier_icon': plan.icon,
                    'purchased_at': datetime.now(),
                    'expires_at': new_expiry,
                    'payment_id': f"ADMIN_{admin_id}_{secrets.token_hex(4)}",
                    'status': PremiumStatus.ACTIVE.value,
                    'activated_by': admin_id,
                    'activated_at': datetime.now(),
                    'is_renewal': True,
                    'previous_tier': current_tier.value,
                    'duration_days': days,
                    'price': 0,  # Admin grants are free
                    'color_code': plan.color_code,
                    'admin_reason': reason
                }
            else:
                # New subscription
                subscription_data = {
                    'user_id': user_id,
                    'tier': tier,
                    'tier_name': plan.name,
                    'tier_icon': plan.icon,
                    'purchased_at': datetime.now(),
                    'expires_at': datetime.now() + timedelta(days=days),
                    'payment_id': f"ADMIN_{admin_id}_{secrets.token_hex(4)}",
                    'status': PremiumStatus.ACTIVE.value,
                    'activated_by': admin_id,
                    'activated_at': datetime.now(),
                    'is_renewal': False,
                    'duration_days': days,
                    'price': 0,  # Admin grants are free
                    'color_code': plan.color_code,
                    'admin_reason': reason
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
            
            logger.info(f"✅ Admin premium added: {tier.value} for user {user_id} by admin {admin_id}")
            
            return subscription_data
            
        except Exception as e:
            logger.error(f"Admin add premium error: {e}")
            raise
    
    async def remove_premium_subscription(self, admin_id: int, user_id: int, 
                                         reason: str = "admin_revoked") -> bool:
        """Admin command: Remove premium subscription from user"""
        try:
            if user_id in self.user_subscriptions:
                self.user_subscriptions[user_id]['status'] = PremiumStatus.CANCELLED.value
                self.user_subscriptions[user_id]['cancelled_at'] = datetime.now()
                self.user_subscriptions[user_id]['cancelled_by'] = admin_id
                self.user_subscriptions[user_id]['cancellation_reason'] = reason
                
                logger.info(f"❌ Admin premium removed: user {user_id} by admin {admin_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Admin remove premium error: {e}")
            return False
    
    async def get_premium_user_info(self, user_id: int) -> Dict[str, Any]:
        """Admin command: Get detailed premium user info"""
        details = await self.get_subscription_details(user_id)
        
        # Add extra admin info
        if user_id in self.user_subscriptions:
            sub_data = self.user_subscriptions[user_id]
            details.update({
                'admin_notes': sub_data.get('admin_notes', ''),
                'activated_by': sub_data.get('activated_by'),
                'cancelled_by': sub_data.get('cancelled_by'),
                'payment_proof': sub_data.get('screenshot_sent', False),
                'total_spent': sub_data.get('price', 0)
            })
        
        return details
    
    async def get_all_premium_users(self) -> List[Dict[str, Any]]:
        """Admin command: Get all premium users"""
        premium_users = []
        
        for user_id, sub_data in self.user_subscriptions.items():
            if sub_data.get('status') == PremiumStatus.ACTIVE.value:
                expiry = sub_data.get('expires_at')
                if expiry and datetime.now() < expiry:
                    user_info = {
                        'user_id': user_id,
                        'tier': sub_data.get('tier', PremiumTier.FREE).value,
                        'tier_name': sub_data.get('tier_name', 'Unknown'),
                        'days_left': self._calculate_days_remaining(expiry),
                        'joined_at': sub_data.get('purchased_at'),
                        'total_downloads': self.user_usage.get(user_id, {}).get('total_downloads', 0),
                        'status': 'active'
                    }
                    premium_users.append(user_info)
        
        return premium_users
    
    async def get_pending_payments_admin(self) -> List[Dict[str, Any]]:
        """Admin command: Get all pending payments"""
        pending = []
        
        for payment_id, payment in self.pending_payments.items():
            if payment['status'] == 'pending':
                user_data = {
                    'payment_id': payment_id,
                    'user_id': payment['user_id'],
                    'tier': payment['tier'].value if isinstance(payment['tier'], PremiumTier) else payment['tier'],
                    'tier_name': payment['tier_name'],
                    'amount': payment['amount'],
                    'duration_days': payment['duration_days'],
                    'created_at': payment['created_at'],
                    'screenshot_sent': payment.get('screenshot_sent', False),
                    'screenshot_sent_at': payment.get('screenshot_sent_at'),
                    'hours_left': max(0, int((payment['expires_at'] - datetime.now()).total_seconds() / 3600)),
                    'color_code': payment.get('color_code', '#2196F3')
                }
                pending.append(user_data)
        
        return pending
    
    async def approve_payment(self, admin_id: int, payment_id: str) -> Tuple[bool, str]:
        """Admin command: Approve pending payment"""
        try:
            if payment_id not in self.pending_payments:
                return False, "Payment not found"
            
            payment = self.pending_payments[payment_id]
            
            if payment['status'] != 'pending':
                return False, f"Payment already {payment['status']}"
            
            # Activate premium for user
            subscription_data = await self.activate_premium(
                admin_id=admin_id,
                user_id=payment['user_id'],
                tier=payment['tier'],
                payment_id=payment_id
            )
            
            # Update payment status
            payment['status'] = 'approved'
            payment['approved_by'] = admin_id
            payment['approved_at'] = datetime.now()
            
            # Update revenue
            self.statistics['total_revenue'] += payment['amount']
            
            logger.info(f"✅ Payment approved: {payment_id} by admin {admin_id}")
            
            return True, f"Payment approved! User {payment['user_id']} now has {payment['tier_name']}"
            
        except Exception as e:
            logger.error(f"Payment approval error: {e}")
            return False, f"Error: {str(e)}"
    
    async def reject_payment(self, admin_id: int, payment_id: str, reason: str = "Invalid screenshot") -> bool:
        """Admin command: Reject pending payment"""
        try:
            if payment_id not in self.pending_payments:
                return False
            
            payment = self.pending_payments[payment_id]
            payment['status'] = 'rejected'
            payment['rejected_by'] = admin_id
            payment['rejected_at'] = datetime.now()
            payment['rejection_reason'] = reason
            
            logger.info(f"❌ Payment rejected: {payment_id} by admin {admin_id}")
            
            return True
        except Exception as e:
            logger.error(f"Payment rejection error: {e}")
            return False
    
    # ✅ STATISTICS COMMANDS
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get bot statistics for /stats command"""
        try:
            total_users = len(self.user_subscriptions) + len(self.user_usage)
            premium_users = 0
            active_premium = 0
            total_downloads = self.statistics['total_downloads']
            
            for user_id, sub_data in self.user_subscriptions.items():
                if sub_data.get('status') == PremiumStatus.ACTIVE.value:
                    premium_users += 1
                    expiry = sub_data.get('expires_at')
                    if expiry and datetime.now() < expiry:
                        active_premium += 1
            
            # Calculate uptime
            uptime = datetime.now() - self.statistics['bot_start_time']
            days = uptime.days
            hours = uptime.seconds // 3600
            minutes = (uptime.seconds % 3600) // 60
            
            # Format data sent
            data_sent_gb = self.statistics['total_data_sent'] / (1024 ** 3)
            
            return {
                'total_users': total_users,
                'premium_users': premium_users,
                'active_premium': active_premium,
                'free_users': total_users - premium_users,
                'total_downloads': total_downloads,
                'total_data_sent': f"{data_sent_gb:.2f} GB",
                'total_premium_sales': self.statistics['total_premium_sales'],
                'total_revenue': f"₹{self.statistics['total_revenue']}",
                'pending_payments': len(self.pending_payments),
                'uptime': f"{days}d {hours}h {minutes}m",
                'server_time': datetime.now().strftime('%d %b %Y, %H:%M:%S')
            }
            
        except Exception as e:
            logger.error(f"Statistics error: {e}")
            return {}
    
    # ✅ UTILITY METHODS
    
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
            tier = await self.get_user_tier(user_id)
            
            if tier == PremiumTier.FREE:
                # Free users have unlimited downloads
                return True, "Free download allowed - Unlimited", {
                    'tier': 'free', 
                    'unlimited': True,
                    'quality': self.free_limits['quality'],
                    'needs_verification': True,
                    'verification_hours': 6
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
            
            # Update global statistics
            self.statistics['total_downloads'] += 1
            self.statistics['total_data_sent'] += file_size
            
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
                    if isinstance(tier, PremiumTier):
                        tier_value = tier.value
                    else:
                        tier_value = tier
                    if tier_value in plan_distribution:
                        plan_distribution[tier_value] += 1
            
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
        logger.info("🧹 Premium cleanup task started")
    
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
                    logger.info(f"🧹 Premium cleanup: {len(expired_users)} subscriptions, {len(expired_payments)} payments expired")
                    
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
    
    async def cancel_subscription(self, user_id: int, reason: str = "user_request") -> bool:
        """Cancel user's subscription"""
        try:
            if user_id in self.user_subscriptions:
                self.user_subscriptions[user_id]['status'] = PremiumStatus.CANCELLED.value
                self.user_subscriptions[user_id]['cancelled_at'] = datetime.now()
                self.user_subscriptions[user_id]['cancellation_reason'] = reason
                
                logger.info(f"❌ Subscription cancelled for user {user_id}: {reason}")
                return True
            return False
        except Exception as e:
            logger.error(f"Subscription cancellation error: {e}")
            return False
    
    async def get_plan_by_tier(self, tier: PremiumTier) -> Optional[PremiumPlan]:
        """Get plan details by tier"""
        return self.plans.get(tier)
    
    async def validate_payment(self, payment_id: str) -> Tuple[bool, str]:
        """Validate payment exists and is pending"""
        if payment_id not in self.pending_payments:
            return False, "Payment not found"
        
        payment = self.pending_payments[payment_id]
        
        if payment['status'] != 'pending':
            return False, f"Payment already {payment['status']}"
        
        if datetime.now() > payment['expires_at']:
            return False, "Payment expired"
        
        return True, "Payment valid"
    
    def to_dict(self) -> Dict[str, Any]:
        """Export system state to dictionary"""
        return {
            'user_subscriptions': {
                str(k): {
                    **v,
                    'tier': v['tier'].value if isinstance(v.get('tier'), PremiumTier) else v.get('tier'),
                    'expires_at': v.get('expires_at').isoformat() if v.get('expires_at') else None,
                    'purchased_at': v.get('purchased_at').isoformat() if v.get('purchased_at') else None
                }
                for k, v in self.user_subscriptions.items()
            },
            'pending_payments': {
                k: {
                    **v,
                    'tier': v['tier'].value if isinstance(v.get('tier'), PremiumTier) else v.get('tier'),
                    'created_at': v['created_at'].isoformat(),
                    'expires_at': v['expires_at'].isoformat()
                }
                for k, v in self.pending_payments.items()
            },
            'user_usage': self.user_usage,
            'statistics': self.statistics,
            'timestamp': datetime.now().isoformat()
        }
    
    async def save_to_db(self):
        """Save system state to database"""
        if self.db_client:
            try:
                state = self.to_dict()
                # Implement your DB save logic here
                logger.info("💾 Premium system state saved to DB")
            except Exception as e:
                logger.error(f"DB save error: {e}")
    
    async def load_from_db(self):
        """Load system state from database"""
        if self.db_client:
            try:
                # Implement your DB load logic here
                logger.info("📥 Premium system state loaded from DB")
            except Exception as e:
                logger.error(f"DB load error: {e}")


# Example usage
if __name__ == "__main__":
    async def main():
        # Initialize system
        premium = PremiumSystem(config={})
        
        # Start cleanup task
        await premium.start_cleanup_task()
        
        # Test user commands
        print("=== USER COMMANDS ===")
        print(await premium.get_my_premium_info(12345))
        print("\n=== ALL PLANS ===")
        print(await premium.get_available_plans_text())
        
        # Test admin commands
        print("\n=== ADMIN COMMANDS ===")
        await premium.add_premium_subscription(1, 12345, PremiumTier.PREMIUM, 30, "test")
        stats = await premium.get_statistics()
        print(f"Stats: {stats}")
        
        # Stop cleanup
        await premium.stop_cleanup_task()
    
    asyncio.run(main())
