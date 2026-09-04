"""
verification.py - URL shortener and verification system
UPDATED: Compatible with Razorpay premium system
"""
import asyncio
import json
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, Optional, List
import aiohttp
import logging

logger = logging.getLogger(__name__)

class VerificationSystem:
    def __init__(self, config, db_client=None):
        self.config = config
        self.db_client = db_client
        self.pending_verifications = {}
        self.verification_tokens = {}
        self.verified_users = {}
        
        # Verification duration: 6 hours
        self.verification_duration = 6 * 60 * 60
        
        # Cleanup task
        self.cleanup_task = None
        
    def generate_unique_token(self, length=32) -> str:
        """Generate unique verification token"""
        return secrets.token_urlsafe(length)
    
    async def get_shortened_url(self, destination_url: str) -> Tuple[str, str]:
        """Get shortened URL using sk4link"""
        if not hasattr(self.config, 'SHORTLINK_API') or not self.config.SHORTLINK_API:
            logger.warning("No shortlink API configured, using direct URL")
            return destination_url, 'Direct'

        try:
            api_url = "https://sk4link.vercel.app/api/public/shorten"
            params = {
                'api': self.config.SHORTLINK_API,
                'url': destination_url
            }

            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(api_url, params=params) as response:
                    if response.status == 200:
                        response_text = await response.text()

                        try:
                            data = json.loads(response_text)
                            if data.get("status") == "success":
                                short_url = data.get('shortenedUrl') or data.get('short_link')
                                if short_url:
                                    logger.info(f"✅ URL shortened via sk4link: {short_url}")
                                    return short_url, 'sk4link'
                        except json.JSONDecodeError:
                            if response_text.startswith('http'):
                                logger.info(f"✅ URL shortened via sk4link: {response_text}")
                                return response_text, 'sk4link'

        except Exception as e:
            logger.warning(f"sk4link failed: {e}")

        logger.warning("sk4link failed, using direct URL")
        return destination_url, 'Direct'
    
    async def create_verification_link(self, user_id: int, content_type: str = "download") -> Dict[str, Any]:
        """Create verification link with unique token (valid for 6 hours)"""
        try:
            verification_token = self.generate_unique_token()
            
            bot_username = getattr(self.config, 'BOT_USERNAME', 'sk4filmbot')
            destination_url = f"https://t.me/{bot_username}?start=verify_{verification_token}"
            
            short_url, service_name = await self.get_shortened_url(destination_url)
            
            link_expiry = datetime.now() + timedelta(hours=1)
            verification_expiry = datetime.now() + timedelta(hours=6)
            
            verification_data = {
                'user_id': user_id,
                'token': verification_token,
                'created_at': datetime.now(),
                'short_url': short_url,
                'service_name': service_name,
                'destination_url': destination_url,
                'content_type': content_type,
                'attempts': 0,
                'status': 'pending',
                'valid_for_hours': 6,
                'link_expires_at': link_expiry,
                'verification_expires_at': verification_expiry
            }
            
            self.pending_verifications[user_id] = verification_data
            self.verification_tokens[verification_token] = user_id
            
            logger.info(f"✅ Verification link created for user {user_id} via {service_name}")
            return verification_data
            
        except Exception as e:
            logger.error(f"❌ Verification link creation error: {e}")
            # Fallback to direct link
            verification_token = self.generate_unique_token()
            bot_username = getattr(self.config, 'BOT_USERNAME', 'sk4filmbot')
            direct_url = f"https://t.me/{bot_username}?start=verify_{verification_token}"
            
            verification_data = {
                'user_id': user_id,
                'token': verification_token,
                'created_at': datetime.now(),
                'short_url': direct_url,
                'service_name': 'Direct',
                'destination_url': direct_url,
                'content_type': content_type,
                'attempts': 0,
                'status': 'pending',
                'valid_for_hours': 6,
                'link_expires_at': datetime.now() + timedelta(hours=1),
                'verification_expires_at': datetime.now() + timedelta(hours=6)
            }
            
            self.pending_verifications[user_id] = verification_data
            self.verification_tokens[verification_token] = user_id
            
            return verification_data
    
    async def verify_user_token(self, token: str) -> Tuple[bool, Optional[int], str]:
        """Verify user token and return user_id if valid"""
        try:
            user_id = self.verification_tokens.get(token)
            if not user_id:
                logger.warning(f"❌ Invalid token attempted: {token[:8]}...")
                return False, None, "Invalid or expired token"
            
            verification_data = self.pending_verifications.get(user_id)
            if not verification_data:
                logger.warning(f"❌ No pending verification for user {user_id}")
                return False, user_id, "No pending verification found"
            
            if verification_data['token'] != token:
                logger.warning(f"❌ Token mismatch for user {user_id}")
                return False, user_id, "Token mismatch"
            
            created_at = verification_data['created_at']
            if datetime.now() > verification_data['link_expires_at']:
                self._cleanup_user_verification(user_id)
                logger.info(f"⏰ Verification link expired for user {user_id}")
                return False, user_id, "Verification link expired (max 1 hour)"
            
            verification_data['status'] = 'verified'
            verification_data['verified_at'] = datetime.now()
            
            expiry_time = datetime.now() + timedelta(seconds=self.verification_duration)
            self.verified_users[user_id] = {
                'verified_at': datetime.now(),
                'expires_at': expiry_time,
                'token': token,
                'verification_count': self.verified_users.get(user_id, {}).get('verification_count', 0) + 1
            }
            
            self._cleanup_user_verification(user_id)
            
            logger.info(f"✅ User {user_id} verified successfully (valid for 6 hours)")
            return True, user_id, "Verification successful - Valid for 6 hours"
            
        except Exception as e:
            logger.error(f"❌ Token verification error: {e}")
            return False, None, "Internal verification error"
    
    def _cleanup_user_verification(self, user_id: int):
        """Cleanup verification data for user"""
        if user_id in self.pending_verifications:
            token = self.pending_verifications[user_id].get('token')
            if token and token in self.verification_tokens:
                del self.verification_tokens[token]
            del self.pending_verifications[user_id]
            logger.debug(f"🧹 Cleaned up verification data for user {user_id}")
    
    async def check_user_verified(self, user_id: int, premium_system=None) -> Tuple[bool, str]:
        """Check if user is currently verified (6 hours) or premium"""
        # ✅ Check premium first (Razorpay premium users bypass verification)
        if premium_system:
            try:
                is_premium = await premium_system.is_premium_user(user_id)
                if is_premium:
                    tier = await premium_system.get_user_tier(user_id)
                    tier_name = tier.value if hasattr(tier, 'value') else str(tier)
                    return True, f"Premium user ({tier_name}) - verification not required"
            except Exception as e:
                logger.error(f"Premium check error: {e}")
        
        # Check verification for free users
        if user_id in self.verified_users:
            user_data = self.verified_users[user_id]
            expiry_time = user_data['expires_at']
            remaining = expiry_time - datetime.now()
            
            if datetime.now() < expiry_time:
                hours = int(remaining.total_seconds() / 3600)
                minutes = int((remaining.total_seconds() % 3600) / 60)
                return True, f"Verified ✅ (expires in {hours}h {minutes}m)"
            else:
                del self.verified_users[user_id]
                logger.info(f"⏰ Verification expired for user {user_id}")
                return False, "Verification expired - Please verify again"
        
        return False, "Not verified - 6 hours verification required"
    
    async def check_user_access(self, user_id: int, premium_system=None) -> Tuple[bool, str, Dict[str, Any]]:
        """Check user access with premium bypass"""
        # ✅ Premium users always have access (Razorpay)
        if premium_system:
            try:
                is_premium = await premium_system.is_premium_user(user_id)
                if is_premium:
                    tier = await premium_system.get_user_tier(user_id)
                    sub_details = await premium_system.get_subscription_details(user_id)
                    return True, "Premium access granted", {
                        'access_type': 'premium',
                        'tier': tier.value if hasattr(tier, 'value') else str(tier),
                        'days_remaining': sub_details.get('days_remaining', 0)
                    }
            except Exception as e:
                logger.error(f"Premium check error: {e}")
        
        # Free users need verification
        is_verified, message = await self.check_user_verified(user_id, premium_system)
        
        if is_verified:
            user_data = self.verified_users.get(user_id, {})
            remaining = user_data.get('expires_at', datetime.now()) - datetime.now()
            return True, message, {
                'access_type': 'verified',
                'tier': 'free',
                'hours_remaining': int(remaining.total_seconds() / 3600)
            }
        else:
            return False, message, {
                'access_type': 'none',
                'tier': 'free',
                'needs_verification': True
            }
    
    async def get_user_verification_info(self, user_id: int) -> Dict[str, Any]:
        """Get detailed verification info for user"""
        info = {
            'user_id': user_id,
            'is_verified': user_id in self.verified_users,
            'has_pending': user_id in self.pending_verifications,
            'verification_duration_hours': 6
        }
        
        if user_id in self.verified_users:
            user_data = self.verified_users[user_id]
            remaining = user_data['expires_at'] - datetime.now()
            info.update({
                'verified_at': user_data['verified_at'].isoformat(),
                'expires_at': user_data['expires_at'].isoformat(),
                'hours_remaining': int(remaining.total_seconds() / 3600),
                'minutes_remaining': int((remaining.total_seconds() % 3600) / 60),
                'verification_count': user_data.get('verification_count', 1)
            })
        
        if user_id in self.pending_verifications:
            pending = self.pending_verifications[user_id]
            info.update({
                'pending_created_at': pending['created_at'].isoformat(),
                'pending_short_url': pending['short_url'],
                'pending_service': pending['service_name'],
                'pending_expires_at': pending['link_expires_at'].isoformat()
            })
        
        return info
    
    async def get_user_stats(self) -> Dict[str, Any]:
        """Get verification statistics"""
        active_verified = 0
        expired_verified = 0
        now = datetime.now()
        
        for user_id, user_data in self.verified_users.items():
            if now < user_data['expires_at']:
                active_verified += 1
            else:
                expired_verified += 1
        
        return {
            'pending_verifications': len(self.pending_verifications),
            'active_verified_users': active_verified,
            'expired_verified_users': expired_verified,
            'total_verified_users': len(self.verified_users),
            'active_tokens': len(self.verification_tokens),
            'verification_duration_hours': 6,
            'link_validity_hours': 1,
            'timestamp': datetime.now().isoformat()
        }
    
    async def start_cleanup_task(self):
        """Start background cleanup task"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("🧹 Verification cleanup task started")
    
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while True:
            try:
                await asyncio.sleep(300)
                
                now = datetime.now()
                
                expired_pending = []
                for user_id, data in self.pending_verifications.items():
                    if now > data.get('link_expires_at', now):
                        expired_pending.append(user_id)
                
                for user_id in expired_pending:
                    self._cleanup_user_verification(user_id)
                
                expired_verified = []
                for user_id, user_data in self.verified_users.items():
                    if now > user_data['expires_at']:
                        expired_verified.append(user_id)
                
                for user_id in expired_verified:
                    del self.verified_users[user_id]
                
                if expired_pending or expired_verified:
                    logger.info(
                        f"🧹 Verification cleanup: "
                        f"{len(expired_pending)} pending links, "
                        f"{len(expired_verified)} expired verifications"
                    )
                    
            except asyncio.CancelledError:
                logger.info("🧹 Verification cleanup task cancelled")
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
    
    async def stop_cleanup_task(self):
        """Stop background cleanup task"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
            logger.info("🧹 Verification cleanup task stopped")
    
    async def stop(self):
        """Stop verification system"""
        await self.stop_cleanup_task()
        logger.info("🛑 Verification system stopped")
