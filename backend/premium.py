import asyncio
import logging
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from enum import Enum
import json

logger = logging.getLogger(__name__)

class PremiumTier(Enum):
    """Premium subscription tiers"""
    BASIC = "basic"
    PREMIUM = "premium" 
    GOLD = "gold"
    DIAMOND = "diamond"

class PremiumSystem:
    """Premium subscription management system"""
    
    def __init__(self, config, db_manager=None):
        self.config = config
        self.db_manager = db_manager
        self.cleanup_task = None
        self.running = False
        
        # In-memory storage (for development)
        self.premium_users = {}  # user_id -> subscription_data
        self.pending_payments = {}  # payment_id -> payment_data
        self.download_history = defaultdict(list)  # user_id -> list of downloads
        self.statistics = {
            'total_users': 0,
            'premium_users': 0,
            'free_users': 0,
            'total_downloads': 0,
            'total_data_sent': 0,  # in bytes
            'total_revenue': 0,
            'total_premium_sales': 0,
            'pending_payments': 0,
            'bot_start_time': datetime.now()
        }
    
    # ✅ TIER CONFIGURATION
    TIER_CONFIG = {
        PremiumTier.BASIC: {
            'name': 'Basic',
            'price': 99,
            'duration_days': 30,
            'max_daily_downloads': 50,
            'max_daily_data': 10 * 1024 * 1024 * 1024,  # 10GB
            'priority_level': 1,
            'features': [
                'All quality options (480p-4K)',
                'No verification required',
                'Unlimited downloads',
                'Basic support'
            ]
        },
        PremiumTier.PREMIUM: {
            'name': 'Premium',
            'price': 199,
            'duration_days': 30,
            'max_daily_downloads': 100,
            'max_daily_data': 20 * 1024 * 1024 * 1024,  # 20GB
            'priority_level': 2,
            'features': [
                'Everything in Basic',
                'Priority support',
                'Faster downloads',
                'Early access to new content'
            ]
        },
        PremiumTier.GOLD: {
            'name': 'Gold',
            'price': 299,
            'duration_days': 60,  # 2 months
            'max_daily_downloads': 200,
            'max_daily_data': 50 * 1024 * 1024 * 1024,  # 50GB
            'priority_level': 3,
            'features': [
                'Everything in Premium',
                'VIP support',
                'Highest priority',
                'Custom requests (limited)'
            ]
        },
        PremiumTier.DIAMOND: {
            'name': 'Diamond',
            'price': 499,
            'duration_days': 90,  # 3 months
            'max_daily_downloads': 500,
            'max_daily_data': 100 * 1024 * 1024 * 1024,  # 100GB
            'priority_level': 4,
            'features': [
                'Everything in Gold',
                '24/7 priority support',
                'Unlimited data',
                'Custom requests',
                'Beta features access'
            ]
        }
    }
    
    # ✅ PAYMENT METHODS
    PAYMENT_METHODS = {
        'upi': {
            'name': 'UPI',
            'id': getattr(self.config, 'UPI_ID', 'sk4film@upi'),
            'qr_code': getattr(self.config, 'UPI_QR_CODE', 'https://example.com/qr.png'),
            'instructions': 'Send payment to UPI ID or scan QR code'
        },
        'paytm': {
            'name': 'PayTM',
            'number': getattr(self.config, 'PAYTM_NUMBER', '9876543210'),
            'instructions': 'Send payment to PayTM number'
        },
        'google_pay': {
            'name': 'Google Pay',
            'id': getattr(self.config, 'GOOGLE_PAY_ID', '9876543210'),
            'instructions': 'Send payment via Google Pay'
        }
    }
    
    async def is_premium_user(self, user_id: int) -> bool:
        """Check if user has active premium subscription"""
        if user_id in self.premium_users:
            sub_data = self.premium_users[user_id]
            
            # Check if subscription is expired
            expires_at = sub_data.get('expires_at')
            if expires_at and datetime.now() > expires_at:
                # Subscription expired
                await self.deactivate_premium(user_id, reason="expired")
                return False
            
            # Check if manually deactivated
            if not sub_data.get('is_active', True):
                return False
            
            return True
        return False
    
    async def add_premium_subscription(self, admin_id: int, user_id: int, tier: PremiumTier, 
                                     days: int, reason: str = "admin") -> Optional[Dict]:
        """Add premium subscription for user"""
        try:
            tier_config = self.TIER_CONFIG.get(tier)
            if not tier_config:
                logger.error(f"Invalid tier: {tier}")
                return None
            
            start_time = datetime.now()
            expires_at = start_time + timedelta(days=days)
            
            subscription_data = {
                'user_id': user_id,
                'tier': tier.value,
                'tier_name': tier_config['name'],
                'purchased_at': start_time,
                'expires_at': expires_at,
                'days_remaining': days,
                'is_active': True,
                'added_by': admin_id,
                'added_reason': reason,
                'daily_downloads': 0,
                'daily_data': 0,
                'total_downloads': 0,
                'last_reset': start_time.date()
            }
            
            self.premium_users[user_id] = subscription_data
            
            # Update statistics
            self.statistics['premium_users'] = len([u for u in self.premium_users.values() if u.get('is_active', True)])
            
            logger.info(f"✅ Premium added for user {user_id}: {tier_config['name']} for {days} days")
            
            return subscription_data
            
        except Exception as e:
            logger.error(f"Error adding premium subscription: {e}")
            return None
    
    async def remove_premium_subscription(self, admin_id: int, user_id: int, reason: str = "admin") -> bool:
        """Remove premium subscription from user"""
        try:
            if user_id in self.premium_users:
                old_tier = self.premium_users[user_id].get('tier_name', 'Unknown')
                del self.premium_users[user_id]
                
                logger.info(f"✅ Premium removed for user {user_id} by admin {admin_id}. Reason: {reason}")
                
                # Update statistics
                self.statistics['premium_users'] = len([u for u in self.premium_users.values() if u.get('is_active', True)])
                
                return True
            return False
        except Exception as e:
            logger.error(f"Error removing premium subscription: {e}")
            return False
    
    async def deactivate_premium(self, user_id: int, reason: str = "expired"):
        """Deactivate premium subscription"""
        try:
            if user_id in self.premium_users:
                self.premium_users[user_id]['is_active'] = False
                self.premium_users[user_id]['deactivated_at'] = datetime.now()
                self.premium_users[user_id]['deactivation_reason'] = reason
                
                logger.info(f"⚠️ Premium deactivated for user {user_id}: {reason}")
                
                # Update statistics
                self.statistics['premium_users'] = len([u for u in self.premium_users.values() if u.get('is_active', True)])
                
        except Exception as e:
            logger.error(f"Error deactivating premium: {e}")
    
    async def get_subscription_details(self, user_id: int) -> Dict:
        """Get user's subscription details"""
        if await self.is_premium_user(user_id):
            sub_data = self.premium_users.get(user_id, {})
            
            # Calculate remaining days
            expires_at = sub_data.get('expires_at')
            days_remaining = 0
            if expires_at:
                delta = expires_at - datetime.now()
                days_remaining = max(0, delta.days)
            
            return {
                'is_premium': True,
                'tier': sub_data.get('tier', 'basic'),
                'tier_name': sub_data.get('tier_name', 'Basic'),
                'purchased_at': sub_data.get('purchased_at', datetime.now()),
                'expires_at': expires_at,
                'days_remaining': days_remaining,
                'daily_downloads': sub_data.get('daily_downloads', 0),
                'daily_data': sub_data.get('daily_data', 0),
                'total_downloads': sub_data.get('total_downloads', 0),
                'max_daily_downloads': self.TIER_CONFIG.get(PremiumTier(sub_data.get('tier', 'basic')), {}).get('max_daily_downloads', 50),
                'max_daily_data': self.TIER_CONFIG.get(PremiumTier(sub_data.get('tier', 'basic')), {}).get('max_daily_data', 10 * 1024 * 1024 * 1024)
            }
        
        return {
            'is_premium': False,
            'tier': 'free',
            'tier_name': 'Free',
            'daily_downloads': 0,
            'daily_data': 0,
            'max_daily_downloads': 10,
            'max_daily_data': 2 * 1024 * 1024 * 1024  # 2GB for free users
        }
    
    async def get_my_premium_info(self, user_id: int) -> str:
        """Get formatted premium info for user"""
        details = await self.get_subscription_details(user_id)
        
        if details['is_premium']:
            # Premium user
            expires_at = details['expires_at']
            expires_str = expires_at.strftime("%d %b %Y %I:%M %p") if expires_at else "Never"
            
            tier_config = self.TIER_CONFIG.get(PremiumTier(details['tier']), {})
            
            text = (
                f"⭐ **PREMIUM USER STATUS** ⭐\n\n"
                f"✅ **Status:** Active Premium\n"
                f"🏆 **Plan:** {details['tier_name']}\n"
                f"💰 **Price:** ₹{tier_config.get('price', 0)}/{tier_config.get('duration_days', 30)} days\n"
                f"📅 **Started:** {details['purchased_at'].strftime('%d %b %Y')}\n"
                f"⏰ **Expires:** {expires_str}\n"
                f"📊 **Days Left:** {details['days_remaining']}\n\n"
                f"📥 **Daily Usage:**\n"
                f"• Downloads: {details['daily_downloads']}/{details['max_daily_downloads']}\n"
                f"• Data: {format_size(details['daily_data'])}/{format_size(details['max_daily_data'])}\n"
                f"• Total Downloads: {details['total_downloads']}\n\n"
                f"✨ **Benefits:**\n"
            )
            
            for feature in tier_config.get('features', []):
                text += f"• {feature}\n"
            
            text += f"\n🎬 Enjoy unlimited downloads!"
            
        else:
            # Free user
            text = (
                f"🎬 **FREE USER STATUS** 🎬\n\n"
                f"❌ **Status:** Not Premium\n"
                f"💰 **Plan:** Free Tier\n\n"
                f"📥 **Daily Limits:**\n"
                f"• Downloads: {details['daily_downloads']}/{details['max_daily_downloads']}\n"
                f"• Data: {format_size(details['daily_data'])}/{format_size(details['max_daily_data'])}\n\n"
                f"⚠️ **Free User Restrictions:**\n"
                f"• Need verification every 6 hours\n"
                f"• Limited daily downloads\n"
                f"• Limited daily data\n"
                f"• Standard priority\n\n"
                f"⭐ **Upgrade to Premium for:**\n"
                f"• No verification required\n"
                f"• Unlimited downloads\n"
                f"• All quality options\n"
                f"• Priority support\n\n"
                f"Click /buy to upgrade!"
            )
        
        return text
    
    async def get_available_plans_text(self) -> str:
        """Get formatted text of all available plans"""
        text = "💰 **SK4FiLM PREMIUM PLANS** 💰\n\n"
        
        for tier, config in self.TIER_CONFIG.items():
            text += f"🏆 **{config['name']} Plan** - ₹{config['price']}"
            
            if config['duration_days'] > 30:
                text += f" / {config['duration_days']} days\n"
            else:
                text += " / month\n"
            
            text += f"📅 **Duration:** {config['duration_days']} days\n"
            text += f"📥 **Daily:** {config['max_daily_downloads']} downloads\n"
            text += f"💾 **Data:** {format_size(config['max_daily_data'])} per day\n"
            text += f"⚡ **Priority:** Level {config['priority_level']}\n\n"
            
            text += "✨ **Features:**\n"
            for feature in config['features']:
                text += f"• {feature}\n"
            
            text += "\n" + "─" * 30 + "\n\n"
        
        text += (
            "⭐ **Premium Benefits Summary:**\n"
            "✅ No verification required\n"
            "✅ All quality options (480p-4K)\n"
            "✅ Unlimited downloads\n"
            "✅ No ads\n"
            "✅ Priority support\n\n"
            "Click a plan button below to purchase!"
        )
        
        return text
    
    # ✅ PAYMENT SYSTEM
    async def initiate_purchase(self, user_id: int, tier: PremiumTier) -> Optional[Dict]:
        """Initiate premium purchase"""
        try:
            tier_config = self.TIER_CONFIG.get(tier)
            if not tier_config:
                logger.error(f"Invalid tier for purchase: {tier}")
                return None
            
            # Generate unique payment ID
            payment_id = f"PAY_{secrets.token_hex(8).upper()}_{int(time.time())}"
            
            payment_data = {
                'payment_id': payment_id,
                'user_id': user_id,
                'tier': tier.value,
                'tier_name': tier_config['name'],
                'amount': tier_config['price'],
                'duration_days': tier_config['duration_days'],
                'created_at': datetime.now(),
                'expires_at': datetime.now() + timedelta(hours=24),  # 24 hours to pay
                'status': 'pending',
                'screenshot_sent': False,
                'screenshot_message_id': None,
                'approved': False,
                'approved_by': None,
                'approved_at': None
            }
            
            self.pending_payments[payment_id] = payment_data
            self.statistics['pending_payments'] = len(self.pending_payments)
            
            logger.info(f"💰 Purchase initiated for user {user_id}: {tier_config['name']} - {payment_id}")
            
            return payment_data
            
        except Exception as e:
            logger.error(f"Error initiating purchase: {e}")
            return None
    
    async def get_payment_instructions_text(self, payment_id: str) -> str:
        """Get payment instructions for a payment"""
        if payment_id not in self.pending_payments:
            return "❌ Payment ID not found!"
        
        payment = self.pending_payments[payment_id]
        tier_config = self.TIER_CONFIG.get(PremiumTier(payment['tier']), {})
        
        website_url = getattr(self.config, 'WEBSITE_URL', 'https://sk4film.com')
        support_channel = getattr(self.config, 'SUPPORT_CHANNEL', 'https://t.me/SK4FiLMSupport')
        
        text = (
            f"💰 **PAYMENT INSTRUCTIONS** 💰\n\n"
            f"**Payment ID:** `{payment_id}`\n"
            f"**Plan:** {payment['tier_name']}\n"
            f"**Amount:** ₹{payment['amount']}\n"
            f"**Duration:** {payment['duration_days']} days\n\n"
            f"⏰ **Payment valid for:** 24 hours\n\n"
            f"💳 **Payment Methods:**\n"
            f"1. **UPI:** `{self.PAYMENT_METHODS['upi']['id']}`\n"
            f"2. **PayTM:** `{self.PAYMENT_METHODS['paytm']['number']}`\n"
            f"3. **Google Pay:** `{self.PAYMENT_METHODS['google_pay']['id']}`\n\n"
            f"📸 **After Payment:**\n"
            f"1. Take screenshot of payment\n"
            f"2. Click 'SEND SCREENSHOT' button\n"
            f"3. Send screenshot here\n"
            f"4. Admin will verify within 24 hours\n\n"
            f"⚠️ **Important:**\n"
            f"• Include payment ID in screenshot\n"
            f"• Keep payment proof\n"
            f"• Contact support if issues\n\n"
            f"🌐 **Website:** {website_url}\n"
            f"📢 **Support:** {support_channel}\n\n"
            f"🎬 Thank you for choosing SK4FiLM!"
        )
        
        return text
    
    async def process_payment_screenshot(self, user_id: int, screenshot_message_id: int) -> bool:
        """Process payment screenshot sent by user"""
        try:
            # Find pending payment for this user
            payment_id = None
            for pid, payment in self.pending_payments.items():
                if payment['user_id'] == user_id and payment['status'] == 'pending':
                    payment_id = pid
                    break
            
            if not payment_id:
                logger.warning(f"No pending payment found for user {user_id}")
                return False
            
            # Update payment with screenshot info
            self.pending_payments[payment_id]['screenshot_sent'] = True
            self.pending_payments[payment_id]['screenshot_message_id'] = screenshot_message_id
            self.pending_payments[payment_id]['screenshot_sent_at'] = datetime.now()
            
            logger.info(f"📸 Payment screenshot received for {payment_id} from user {user_id}")
            
            # Notify admins
            await self.notify_admins_of_screenshot(payment_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing screenshot: {e}")
            return False
    
    async def notify_admins_of_screenshot(self, payment_id: str):
        """Notify admins about new screenshot"""
        try:
            if payment_id not in self.pending_payments:
                return
            
            payment = self.pending_payments[payment_id]
            
            # This would be implemented with actual bot instance
            # For now, just log
            logger.info(
                f"📢 ADMIN NOTIFICATION - New payment screenshot:\n"
                f"Payment ID: {payment_id}\n"
                f"User ID: {payment['user_id']}\n"
                f"Plan: {payment['tier_name']}\n"
                f"Amount: ₹{payment['amount']}"
            )
            
        except Exception as e:
            logger.error(f"Error notifying admins: {e}")
    
    async def approve_payment(self, admin_id: int, payment_id: str) -> (bool, str):
        """Approve pending payment"""
        try:
            if payment_id not in self.pending_payments:
                return False, "Payment ID not found!"
            
            payment = self.pending_payments[payment_id]
            
            if payment['status'] != 'pending':
                return False, f"Payment already {payment['status']}!"
            
            # Add premium subscription
            tier = PremiumTier(payment['tier'])
            subscription_data = await self.add_premium_subscription(
                admin_id=admin_id,
                user_id=payment['user_id'],
                tier=tier,
                days=payment['duration_days'],
                reason="payment_approved"
            )
            
            if subscription_data:
                # Update payment status
                self.pending_payments[payment_id]['status'] = 'approved'
                self.pending_payments[payment_id]['approved'] = True
                self.pending_payments[payment_id]['approved_by'] = admin_id
                self.pending_payments[payment_id]['approved_at'] = datetime.now()
                
                # Update statistics
                self.statistics['total_revenue'] += payment['amount']
                self.statistics['total_premium_sales'] += 1
                self.statistics['pending_payments'] = len([p for p in self.pending_payments.values() if p['status'] == 'pending'])
                
                logger.info(f"✅ Payment {payment_id} approved by admin {admin_id}")
                
                return True, f"✅ Payment approved! User {payment['user_id']} now has {payment['tier_name']} premium."
            else:
                return False, "❌ Failed to add premium subscription!"
                
        except Exception as e:
            logger.error(f"Error approving payment: {e}")
            return False, f"Error: {str(e)}"
    
    async def reject_payment(self, admin_id: int, payment_id: str, reason: str) -> bool:
        """Reject pending payment"""
        try:
            if payment_id not in self.pending_payments:
                return False
            
            payment = self.pending_payments[payment_id]
            
            if payment['status'] != 'pending':
                return False
            
            # Update payment status
            self.pending_payments[payment_id]['status'] = 'rejected'
            self.pending_payments[payment_id]['rejected_by'] = admin_id
            self.pending_payments[payment_id]['rejected_at'] = datetime.now()
            self.pending_payments[payment_id]['rejection_reason'] = reason
            
            # Update statistics
            self.statistics['pending_payments'] = len([p for p in self.pending_payments.values() if p['status'] == 'pending'])
            
            logger.info(f"❌ Payment {payment_id} rejected by admin {admin_id}: {reason}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error rejecting payment: {e}")
            return False
    
    async def get_pending_payments_admin(self) -> List[Dict]:
        """Get pending payments for admin view"""
        pending = []
        now = datetime.now()
        
        for payment_id, payment in self.pending_payments.items():
            if payment['status'] == 'pending':
                # Calculate hours left
                expires_at = payment.get('expires_at', now)
                time_left = expires_at - now
                hours_left = max(0, int(time_left.total_seconds() / 3600))
                
                pending.append({
                    'payment_id': payment_id,
                    'user_id': payment['user_id'],
                    'tier_name': payment['tier_name'],
                    'amount': payment['amount'],
                    'screenshot_sent': payment.get('screenshot_sent', False),
                    'hours_left': hours_left,
                    'created_at': payment['created_at']
                })
        
        return sorted(pending, key=lambda x: x['hours_left'])
    
    # ✅ DOWNLOAD TRACKING
    async def record_download(self, user_id: int, file_size: int, quality: str):
        """Record download for user statistics"""
        try:
            today = datetime.now().date()
            
            # Get or create user download stats
            if user_id not in self.download_history:
                self.download_history[user_id] = []
            
            # Check if we need to reset daily stats
            if user_id in self.premium_users:
                last_reset = self.premium_users[user_id].get('last_reset')
                if last_reset != today:
                    # Reset daily stats
                    self.premium_users[user_id]['daily_downloads'] = 0
                    self.premium_users[user_id]['daily_data'] = 0
                    self.premium_users[user_id]['last_reset'] = today
            
            # Update user stats
            download_record = {
                'timestamp': datetime.now(),
                'file_size': file_size,
                'quality': quality,
                'date': today
            }
            
            self.download_history[user_id].append(download_record)
            
            # Update premium user stats if applicable
            if user_id in self.premium_users:
                self.premium_users[user_id]['daily_downloads'] = self.premium_users[user_id].get('daily_downloads', 0) + 1
                self.premium_users[user_id]['daily_data'] = self.premium_users[user_id].get('daily_data', 0) + file_size
                self.premium_users[user_id]['total_downloads'] = self.premium_users[user_id].get('total_downloads', 0) + 1
            
            # Update global statistics
            self.statistics['total_downloads'] += 1
            self.statistics['total_data_sent'] += file_size
            
            logger.debug(f"📥 Download recorded for user {user_id}: {format_size(file_size)} {quality}")
            
        except Exception as e:
            logger.error(f"Error recording download: {e}")
    
    # ✅ STATISTICS
    async def get_statistics(self) -> Dict:
        """Get system statistics"""
        try:
            # Calculate uptime
            uptime = datetime.now() - self.statistics['bot_start_time']
            uptime_str = str(uptime).split('.')[0]
            
            # Format total data sent
            total_data_gb = self.statistics['total_data_sent'] / (1024 ** 3)
            
            stats = self.statistics.copy()
            stats.update({
                'uptime': uptime_str,
                'total_data_sent': f"{total_data_gb:.2f} GB",
                'total_revenue': f"₹{stats['total_revenue']}",
                'server_time': datetime.now().strftime("%d %b %Y %I:%M %p"),
                'pending_payments': len([p for p in self.pending_payments.values() if p['status'] == 'pending']),
                'free_users': stats['total_users'] - stats['premium_users']
            })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return self.statistics
    
    async def get_premium_user_info(self, user_id: int) -> Dict:
        """Get premium user info for admin"""
        try:
            if await self.is_premium_user(user_id):
                sub_data = self.premium_users[user_id]
                
                # Calculate days remaining
                expires_at = sub_data.get('expires_at')
                days_remaining = 0
                if expires_at:
                    delta = expires_at - datetime.now()
                    days_remaining = max(0, delta.days)
                
                return {
                    'tier': sub_data.get('tier', 'basic'),
                    'tier_name': sub_data.get('tier_name', 'Basic'),
                    'status': 'active' if sub_data.get('is_active', True) else 'inactive',
                    'purchased_at': sub_data.get('purchased_at', datetime.now()).strftime("%d %b %Y %I:%M %p"),
                    'expires_at': expires_at.strftime("%d %b %Y %I:%M %p") if expires_at else "Never",
                    'days_remaining': days_remaining,
                    'daily_downloads': sub_data.get('daily_downloads', 0),
                    'daily_data': format_size(sub_data.get('daily_data', 0)),
                    'total_downloads': sub_data.get('total_downloads', 0),
                    'added_by': sub_data.get('added_by', 'system'),
                    'added_reason': sub_data.get('added_reason', 'unknown')
                }
            else:
                return {
                    'tier': 'free',
                    'tier_name': 'Free',
                    'status': 'free_user'
                }
                
        except Exception as e:
            logger.error(f"Error getting premium user info: {e}")
            return {'tier': 'error', 'status': 'error'}
    
    # ✅ CLEANUP TASKS
    async def start_cleanup_task(self):
        """Start periodic cleanup task"""
        self.running = True
        self.cleanup_task = asyncio.create_task(self._cleanup_task())
        logger.info("✅ Premium system cleanup task started")
    
    async def stop_cleanup_task(self):
        """Stop cleanup task"""
        self.running = False
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("✅ Premium system cleanup task stopped")
    
    async def _cleanup_task(self):
        """Periodic cleanup of expired subscriptions and payments"""
        while self.running:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                now = datetime.now()
                expired_count = 0
                pending_expired = 0
                
                # Clean expired premium subscriptions
                for user_id in list(self.premium_users.keys()):
                    sub_data = self.premium_users[user_id]
                    expires_at = sub_data.get('expires_at')
                    
                    if expires_at and now > expires_at and sub_data.get('is_active', True):
                        await self.deactivate_premium(user_id, reason="expired")
                        expired_count += 1
                
                # Clean expired pending payments
                for payment_id in list(self.pending_payments.keys()):
                    payment = self.pending_payments[payment_id]
                    expires_at = payment.get('expires_at')
                    
                    if payment['status'] == 'pending' and expires_at and now > expires_at:
                        del self.pending_payments[payment_id]
                        pending_expired += 1
                
                # Update statistics
                self.statistics['pending_payments'] = len([p for p in self.pending_payments.values() if p['status'] == 'pending'])
                
                if expired_count > 0 or pending_expired > 0:
                    logger.info(
                        f"🧹 Premium cleanup: {expired_count} expired subscriptions, "
                        f"{pending_expired} expired payments removed"
                    )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in premium cleanup task: {e}")

# Utility function for file size formatting
def format_size(size_in_bytes):
    """Format file size in human-readable format"""
    if size_in_bytes is None or size_in_bytes == 0:
        return "0 B"
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.1f} {unit}"
        size_in_bytes /= 1024.0
    return f"{size_in_bytes:.1f} PB"
