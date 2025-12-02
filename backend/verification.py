"""
verification.py - URL shortener and verification system with Website Integration
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
        self.pending_verifications = {}  # user_id -> verification_data
        self.verification_tokens = {}    # token -> user_id
        self.verified_users = {}         # user_id -> expiry_time
        
        # Verification duration: 6 hours
        self.verification_duration = 6 * 60 * 60
        
        # Cleanup task
        self.cleanup_task = None
        
        # Supported URL shortener services
        self.shortener_services = {
            'gplinks': {
                'api_url': 'https://gplinks.in/api',
                'enabled': True,
                'priority': 1
            },
            'short.io': {
                'api_url': 'https://api.short.io/links',
                'enabled': False,
                'priority': 2
            },
            'bitly': {
                'api_url': 'https://api-ssl.bitly.com/v4/shorten',
                'enabled': False,
                'priority': 3
            }
        }
    
    def generate_unique_token(self, length=32) -> str:
        """Generate unique verification token"""
        return secrets.token_urlsafe(length)
    
    def generate_token_hash(self, user_id: int, timestamp: datetime) -> str:
        """Generate hash for token validation"""
        data = f"{user_id}:{timestamp.isoformat()}:{secrets.token_hex(16)}"
        return hashlib.sha256(data.encode()).hexdigest()[:32]
    
    async def get_shortened_url(self, destination_url: str) -> Tuple[str, str]:
        """Get shortened URL using GPLinks or similar service"""
        try:
            # Check if API key is configured
            if not hasattr(self.config, 'SHORTLINK_API') or not self.config.SHORTLINK_API:
                logger.warning("No shortlink API configured, using direct URL")
                return destination_url, 'Direct'
            
            # Try GPLinks first
            api_url = "https://gplinks.in/api"
            params = {
                'api': self.config.SHORTLINK_API,
                'url': destination_url
            }
            
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(api_url, params=params) as response:
                    if response.status == 200:
                        response_text = await response.text()
                        
                        # Try JSON first
                        try:
                            data = json.loads(response_text)
                            if data.get("status") == "success":
                                short_url = data.get('shortenedUrl', destination_url)
                                logger.info(f"✅ URL shortened via GPLinks: {short_url}")
                                return short_url, 'GPLinks'
                        except json.JSONDecodeError:
                            # Try direct URL response
                            if response_text.startswith('http'):
                                logger.info(f"✅ URL shortened via GPLinks (direct): {response_text}")
                                return response_text, 'GPLinks'
                    else:
                        logger.warning(f"GPLinks API returned status {response.status}")
        
        except asyncio.TimeoutError:
            logger.warning("Shortener API timeout")
        except Exception as e:
            logger.warning(f"Shortener failed: {e}")
        
        # Fallback to direct URL
        logger.info("Using direct URL (no shortener)")
        return destination_url, 'Direct'
    
    async def create_verification_link(self, user_id: int, content_type: str = "website_access") -> Dict[str, Any]:
        """Create verification link with unique token (valid for 6 hours) - ALWAYS shortened"""
        try:
            # Generate unique verification token
            verification_token = self.generate_unique_token(32)
            
            # Create Telegram deep link with verification
            bot_username = getattr(self.config, 'BOT_USERNAME', 'sk4filmbot')
            destination_url = f"https://t.me/{bot_username}?start=verify_{verification_token}"
            
            # Get shortened URL - ALWAYS try to shorten
            short_url, service_name = await self.get_shortened_url(destination_url)
            
            # Calculate expiry times
            link_expiry = datetime.now() + timedelta(hours=1)  # Link valid for 1 hour
            verification_expiry = datetime.now() + timedelta(hours=6)  # Verification valid for 6 hours
            
            # Store verification data
            verification_data = {
                'user_id': user_id,
                'token': verification_token,
                'created_at': datetime.now(),
                'short_url': short_url,  # This will be shortened URL
                'service_name': service_name,
                'destination_url': destination_url,
                'content_type': content_type,
                'attempts': 0,
                'status': 'pending',
                'valid_for_hours': 6,
                'link_expires_at': link_expiry,
                'verification_expires_at': verification_expiry,
                'needs_file_access': True,
                'website_access': True,  # Enable website access after verification
                'is_shortened': service_name != 'Direct'  # Track if URL was shortened
            }
            
            # Store in memory caches
            self.pending_verifications[user_id] = verification_data
            self.verification_tokens[verification_token] = user_id
            
            logger.info(f"✅ Verification link created for user {user_id} via {service_name}")
            logger.info(f"   Short URL: {short_url}")
            logger.info(f"   Original URL: {destination_url}")
            logger.info(f"   Token: {verification_token[:16]}...")
            
            return verification_data
            
        except Exception as e:
            logger.error(f"❌ Verification link creation error: {e}")
            # Fallback to direct link (still shortened if possible)
            verification_token = self.generate_unique_token(32)
            bot_username = getattr(self.config, 'BOT_USERNAME', 'sk4filmbot')
            direct_url = f"https://t.me/{bot_username}?start=verify_{verification_token}"
            
            # Try to shorten even in fallback
            short_url, service_name = await self.get_shortened_url(direct_url)
            
            verification_data = {
                'user_id': user_id,
                'token': verification_token,
                'created_at': datetime.now(),
                'short_url': short_url,
                'service_name': service_name,
                'destination_url': direct_url,
                'content_type': content_type,
                'attempts': 0,
                'status': 'pending',
                'valid_for_hours': 6,
                'link_expires_at': datetime.now() + timedelta(hours=1),
                'verification_expires_at': datetime.now() + timedelta(hours=6),
                'needs_file_access': True,
                'website_access': True,
                'is_shortened': service_name != 'Direct'
            }
            
            self.pending_verifications[user_id] = verification_data
            self.verification_tokens[verification_token] = user_id
            
            return verification_data
    
    async def verify_user_token(self, token: str) -> Tuple[bool, Optional[int], str]:
        """Verify user token and return user_id if valid"""
        try:
            # Find user_id from token
            user_id = self.verification_tokens.get(token)
            if not user_id:
                logger.warning(f"❌ Invalid token attempted: {token[:8]}...")
                return False, None, "Invalid or expired token"
            
            # Get verification data
            verification_data = self.pending_verifications.get(user_id)
            if not verification_data:
                logger.warning(f"❌ No pending verification for user {user_id}")
                return False, user_id, "No pending verification found"
            
            # Check if token matches
            if verification_data['token'] != token:
                logger.warning(f"❌ Token mismatch for user {user_id}")
                return False, user_id, "Token mismatch"
            
            # Check link expiry (1 hour)
            created_at = verification_data['created_at']
            if datetime.now() > verification_data['link_expires_at']:
                # Cleanup expired
                self._cleanup_user_verification(user_id)
                logger.info(f"⏰ Verification link expired for user {user_id}")
                return False, user_id, "Verification link expired (max 1 hour)"
            
            # Mark as verified - valid for 6 hours
            verification_data['status'] = 'verified'
            verification_data['verified_at'] = datetime.now()
            verification_data['website_access'] = True  # Enable website access
            
            # Store in verified users (valid for 6 hours)
            expiry_time = datetime.now() + timedelta(seconds=self.verification_duration)
            self.verified_users[user_id] = {
                'verified_at': datetime.now(),
                'expires_at': expiry_time,
                'token': token,
                'verification_count': self.verified_users.get(user_id, {}).get('verification_count', 0) + 1,
                'can_access_files': True,
                'website_access': True,  # User can use website
                'access_granted_at': datetime.now(),
                'last_activity': datetime.now(),
                'can_download_directly': True
            }
            
            # Cleanup from pending
            self._cleanup_user_verification(user_id)
            
            logger.info(f"✅ User {user_id} verified successfully (valid for 6 hours)")
            logger.info(f"   User can now use website to download files directly!")
            logger.info(f"   Expires at: {expiry_time}")
            
            return True, user_id, "Verification successful - You can now use website to download files directly for 6 hours!"
            
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
        # Check if user is premium (premium users don't need verification)
        if premium_system:
            try:
                is_premium = await premium_system.is_premium_user(user_id)
                if is_premium:
                    tier = await premium_system.get_user_tier(user_id)
                    return True, f"Premium user ({tier.value}) - website access available"
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
                return True, f"Verified ✅ (website access for {hours}h {minutes}m)"
            else:
                # Expired, cleanup
                del self.verified_users[user_id]
                logger.info(f"⏰ Verification expired for user {user_id}")
                return False, "Verification expired - Please verify again for website access"
        
        return False, "Not verified - Need verification for website access"
    
    async def check_user_access(self, user_id: int, premium_system=None) -> Tuple[bool, str, Dict[str, Any]]:
        """Check user access with premium bypass - UPDATED for website access"""
        # Premium users always have website access
        if premium_system:
            try:
                is_premium = await premium_system.is_premium_user(user_id)
                if is_premium:
                    tier = await premium_system.get_user_tier(user_id)
                    sub_details = await premium_system.get_subscription_details(user_id)
                    return True, "Premium user - Full website access", {
                        'access_type': 'premium',
                        'tier': tier.value,
                        'days_remaining': sub_details.get('days_remaining', 0),
                        'can_access_files': True,
                        'requires_verification': False,
                        'website_access': True,
                        'can_use_website': True,
                        'can_download_directly': True,
                        'download_method': 'website_click'
                    }
            except Exception as e:
                logger.error(f"Premium check error: {e}")
        
        # Free users need verification for website access
        is_verified, message = await self.check_user_verified(user_id, premium_system)
        
        if is_verified:
            user_data = self.verified_users.get(user_id, {})
            remaining = user_data.get('expires_at', datetime.now()) - datetime.now()
            return True, message, {
                'access_type': 'verified',
                'tier': 'free',
                'hours_remaining': int(remaining.total_seconds() / 3600),
                'can_access_files': True,
                'requires_verification': True,
                'website_access': True,
                'can_use_website': True,
                'can_download_directly': True,
                'download_method': 'website_click'
            }
        else:
            return False, message, {
                'access_type': 'none',
                'tier': 'free',
                'needs_verification': True,
                'can_access_files': False,
                'requires_verification': True,
                'website_access': False,
                'can_use_website': False,
                'can_download_directly': False,
                'download_method': 'verification_required'
            }
    
    async def get_verification_link_for_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get verification link for user, create if not exists"""
        try:
            # Check if user already has pending verification
            if user_id in self.pending_verifications:
                verification_data = self.pending_verifications[user_id]
                
                # Check if link is still valid
                if datetime.now() < verification_data['link_expires_at']:
                    logger.info(f"✅ Using existing verification link for user {user_id}")
                    return verification_data
                else:
                    # Link expired, create new one
                    logger.info(f"⏰ Existing verification link expired for user {user_id}, creating new")
                    self._cleanup_user_verification(user_id)
            
            # Create new verification link
            return await self.create_verification_link(user_id, "website_access")
            
        except Exception as e:
            logger.error(f"Error getting verification link for user {user_id}: {e}")
            return None
    
    async def can_user_use_website(self, user_id: int, premium_system=None) -> Tuple[bool, str, Dict[str, Any]]:
        """Check if user can use website to download files (website calls this)"""
        # Check premium first
        if premium_system:
            try:
                is_premium = await premium_system.is_premium_user(user_id)
                if is_premium:
                    tier = await premium_system.get_user_tier(user_id)
                    sub_details = await premium_system.get_subscription_details(user_id)
                    return True, "Premium user - Full website access", {
                        'user_id': user_id,
                        'has_access': True,
                        'access_type': 'premium',
                        'tier': tier.value,
                        'days_remaining': sub_details.get('days_remaining', 0),
                        'can_use_website': True,
                        'requires_verification': False,
                        'verification_required': False,
                        'download_method': 'website_click',
                        'message': 'Premium user can download directly from website'
                    }
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
                
                return True, f"Verified user - Access for {hours}h {minutes}m", {
                    'user_id': user_id,
                    'has_access': True,
                    'access_type': 'verified',
                    'tier': 'free',
                    'hours_remaining': hours,
                    'minutes_remaining': minutes,
                    'can_use_website': True,
                    'requires_verification': True,
                    'verification_required': False,
                    'download_method': 'website_click',
                    'message': f'Verified user can download from website (expires in {hours}h {minutes}m)'
                }
            else:
                # Expired, cleanup
                del self.verified_users[user_id]
                logger.info(f"⏰ Verification expired for user {user_id}")
        
        # User needs verification
        verification_link = await self.get_verification_link_for_user(user_id)
        
        return False, "Verification required for website access", {
            'user_id': user_id,
            'has_access': False,
            'access_type': 'none',
            'tier': 'free',
            'can_use_website': False,
            'requires_verification': True,
            'verification_required': True,
            'download_method': 'verification_required',
            'verification_link': verification_link['short_url'] if verification_link else None,
            'verification_service': verification_link['service_name'] if verification_link else None,
            'message': 'Complete verification to download from website'
        }
    
    async def process_website_download_request(self, user_id: int, premium_system=None) -> Dict[str, Any]:
        """Process website download request - check if user can download"""
        try:
            # Check if user can use website
            can_use, message, access_details = await self.can_user_use_website(user_id, premium_system)
            
            if can_use:
                # Update last activity
                if user_id in self.verified_users:
                    self.verified_users[user_id]['last_activity'] = datetime.now()
                    self.verified_users[user_id]['website_download_count'] = \
                        self.verified_users[user_id].get('website_download_count', 0) + 1
                
                logger.info(f"✅ Website download access granted for user {user_id}")
                
                return {
                    'status': 'success',
                    'user_id': user_id,
                    'message': 'Access granted for website download',
                    'access_details': access_details,
                    'can_download': True,
                    'requires_verification': False,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                logger.info(f"🔒 Website download access denied for user {user_id}: {message}")
                
                return {
                    'status': 'verification_required',
                    'user_id': user_id,
                    'message': 'Verification required for website download',
                    'access_details': access_details,
                    'can_download': False,
                    'requires_verification': True,
                    'verification_link': access_details.get('verification_link'),
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error processing website download request for user {user_id}: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'can_download': False,
                'requires_verification': False
            }
    
    async def extend_verification(self, user_id: int, hours: int = 6) -> bool:
        """Extend user verification"""
        if user_id in self.verified_users:
            new_expiry = datetime.now() + timedelta(hours=hours)
            self.verified_users[user_id]['expires_at'] = new_expiry
            self.verified_users[user_id]['can_access_files'] = True
            self.verified_users[user_id]['website_access'] = True
            logger.info(f"✅ Extended verification for user {user_id} by {hours} hours")
            return True
        else:
            logger.warning(f"❌ Cannot extend verification for user {user_id} - not verified")
            return False
    
    async def revoke_verification(self, user_id: int) -> bool:
        """Revoke user verification"""
        if user_id in self.verified_users:
            del self.verified_users[user_id]
            logger.info(f"🚫 Revoked verification for user {user_id}")
            return True
        return False
    
    async def get_user_verification_info(self, user_id: int) -> Dict[str, Any]:
        """Get detailed verification info for user"""
        info = {
            'user_id': user_id,
            'is_verified': user_id in self.verified_users,
            'has_pending': user_id in self.pending_verifications,
            'verification_duration_hours': 6,
            'can_access_files': user_id in self.verified_users,
            'can_use_website': user_id in self.verified_users
        }
        
        if user_id in self.verified_users:
            user_data = self.verified_users[user_id]
            remaining = user_data['expires_at'] - datetime.now()
            info.update({
                'verified_at': user_data['verified_at'].isoformat(),
                'expires_at': user_data['expires_at'].isoformat(),
                'hours_remaining': int(remaining.total_seconds() / 3600),
                'minutes_remaining': int((remaining.total_seconds() % 3600) / 60),
                'verification_count': user_data.get('verification_count', 1),
                'can_access_files': True,
                'can_use_website': True,
                'website_access': True,
                'last_activity': user_data.get('last_activity', user_data['verified_at']).isoformat(),
                'website_download_count': user_data.get('website_download_count', 0),
                'file_access_method': 'website_direct',
                'download_method': 'website_click'
            })
        
        if user_id in self.pending_verifications:
            pending = self.pending_verifications[user_id]
            info.update({
                'pending_created_at': pending['created_at'].isoformat(),
                'pending_short_url': pending['short_url'],  # This is the shortened URL
                'pending_service': pending['service_name'],
                'pending_expires_at': pending['link_expires_at'].isoformat(),
                'is_shortened': pending.get('is_shortened', False),
                'can_access_files': False,
                'can_use_website': False,
                'needs_verification': True,
                'verification_link': pending['short_url']
            })
        
        return info
    
    async def get_user_stats(self) -> Dict[str, Any]:
        """Get verification statistics"""
        # Count verified users still within 6 hours
        active_verified = 0
        expired_verified = 0
        website_users = 0
        now = datetime.now()
        
        for user_id, user_data in self.verified_users.items():
            if now < user_data['expires_at']:
                active_verified += 1
                if user_data.get('website_access', False):
                    website_users += 1
            else:
                expired_verified += 1
        
        # Count pending verifications with shortened URLs
        shortened_pending = 0
        direct_pending = 0
        
        for user_id, data in self.pending_verifications.items():
            if data.get('is_shortened', False):
                shortened_pending += 1
            else:
                direct_pending += 1
        
        return {
            'pending_verifications': len(self.pending_verifications),
            'pending_shortened': shortened_pending,
            'pending_direct': direct_pending,
            'active_verified_users': active_verified,
            'expired_verified_users': expired_verified,
            'website_users': website_users,
            'total_verified_users': len(self.verified_users),
            'active_tokens': len(self.verification_tokens),
            'verification_duration_hours': 6,
            'link_validity_hours': 1,
            'timestamp': datetime.now().isoformat()
        }
    
    async def get_all_verified_users(self) -> List[Dict[str, Any]]:
        """Get list of all verified users"""
        verified_list = []
        now = datetime.now()
        
        for user_id, user_data in self.verified_users.items():
            remaining = user_data['expires_at'] - now
            verified_list.append({
                'user_id': user_id,
                'verified_at': user_data['verified_at'].isoformat(),
                'expires_at': user_data['expires_at'].isoformat(),
                'hours_remaining': int(remaining.total_seconds() / 3600),
                'is_expired': now > user_data['expires_at'],
                'verification_count': user_data.get('verification_count', 1),
                'can_access_files': True,
                'website_access': user_data.get('website_access', False),
                'last_activity': user_data.get('last_activity', user_data['verified_at']).isoformat(),
                'website_download_count': user_data.get('website_download_count', 0)
            })
        
        return verified_list
    
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
                await asyncio.sleep(300)  # Run every 5 minutes
                
                now = datetime.now()
                
                # Cleanup expired pending verifications (1 hour)
                expired_pending = []
                for user_id, data in self.pending_verifications.items():
                    if now > data.get('link_expires_at', now):
                        expired_pending.append(user_id)
                
                for user_id in expired_pending:
                    self._cleanup_user_verification(user_id)
                
                # Cleanup expired verified users (6 hours)
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Export verification state to dictionary"""
        return {
            'pending_verifications': {
                str(k): {
                    **v,
                    'created_at': v['created_at'].isoformat(),
                    'link_expires_at': v['link_expires_at'].isoformat(),
                    'verification_expires_at': v['verification_expires_at'].isoformat()
                }
                for k, v in self.pending_verifications.items()
            },
            'verified_users': {
                str(k): {
                    **v,
                    'verified_at': v['verified_at'].isoformat(),
                    'expires_at': v['expires_at'].isoformat(),
                    'last_activity': v.get('last_activity', v['verified_at']).isoformat()
                }
                for k, v in self.verified_users.items()
            },
            'stats': asyncio.run(self.get_user_stats()) if asyncio.get_event_loop().is_running() else {},
            'timestamp': datetime.now().isoformat()
        }
    
    async def save_to_db(self):
        """Save verification state to database"""
        if self.db_client:
            try:
                state = self.to_dict()
                # Implement your DB save logic here
                logger.info("💾 Verification system state saved to DB")
            except Exception as e:
                logger.error(f"DB save error: {e}")
    
    async def load_from_db(self):
        """Load verification state from database"""
        if self.db_client:
            try:
                # Implement your DB load logic here
                logger.info("📥 Verification system state loaded from DB")
            except Exception as e:
                logger.error(f"DB load error: {e}")


# Example usage showing website integration
if __name__ == "__main__":
    async def main():
        # Mock config
        class Config:
            BOT_USERNAME = "sk4filmbot"
            SHORTLINK_API = "your_gplinks_api_key_here"  # Set your API key here
        
        # Initialize system
        verification = VerificationSystem(Config())
        
        # Start cleanup task
        await verification.start_cleanup_task()
        
        # Test 1: Create verification link (will be shortened)
        print("\n" + "="*50)
        print("TEST 1: Creating verification link")
        print("="*50)
        link_data = await verification.create_verification_link(12345, "website_access")
        print(f"✅ Verification link created:")
        print(f"   Short URL: {link_data['short_url']}")
        print(f"   Service: {link_data['service_name']}")
        print(f"   Is Shortened: {link_data.get('is_shortened', False)}")
        print(f"   Token: {link_data['token'][:16]}...")
        print(f"   Valid until: {link_data['link_expires_at']}")
        
        # Test 2: Verify user
        print("\n" + "="*50)
        print("TEST 2: Verifying user token")
        print("="*50)
        success, user_id, message = await verification.verify_user_token(link_data['token'])
        print(f"✅ Verification: {success} - {message}")
        
        # Test 3: Check website access
        print("\n" + "="*50)
        print("TEST 3: Checking website access")
        print("="*50)
        can_use, website_msg, website_details = await verification.can_user_use_website(12345)
        print(f"✅ Website Access: {can_use} - {website_msg}")
        print(f"   Details: {website_details}")
        
        # Test 4: Process website download request
        print("\n" + "="*50)
        print("TEST 4: Processing website download request")
        print("="*50)
        download_result = await verification.process_website_download_request(12345)
        print(f"✅ Download Request: {download_result['status']}")
        print(f"   Can Download: {download_result['can_download']}")
        print(f"   Message: {download_result['message']}")
        
        # Test 5: Get user info
        print("\n" + "="*50)
        print("TEST 5: Getting user verification info")
        print("="*50)
        user_info = await verification.get_user_verification_info(12345)
        print(f"✅ User Info:")
        print(f"   Is Verified: {user_info['is_verified']}")
        print(f"   Can Use Website: {user_info['can_use_website']}")
        print(f"   Hours Remaining: {user_info.get('hours_remaining', 0)}")
        print(f"   Download Method: {user_info.get('download_method', 'N/A')}")
        
        # Test 6: Get stats
        print("\n" + "="*50)
        print("TEST 6: Getting system stats")
        print("="*50)
        stats = await verification.get_user_stats()
        print(f"📊 Stats:")
        print(f"   Active Verified Users: {stats['active_verified_users']}")
        print(f"   Website Users: {stats['website_users']}")
        print(f"   Pending Shortened: {stats['pending_shortened']}")
        print(f"   Pending Direct: {stats['pending_direct']}")
        
        # Test 7: Test non-verified user
        print("\n" + "="*50)
        print("TEST 7: Testing non-verified user")
        print("="*50)
        non_verified_result = await verification.process_website_download_request(99999)
        print(f"✅ Non-verified User Result: {non_verified_result['status']}")
        print(f"   Can Download: {non_verified_result['can_download']}")
        print(f"   Verification Required: {non_verified_result['requires_verification']}")
        if non_verified_result.get('verification_link'):
            print(f"   Verification Link: {non_verified_result['verification_link']}")
        
        # Stop cleanup
        print("\n" + "="*50)
        print("Stopping system...")
        print("="*50)
        await verification.stop()
    
    asyncio.run(main())
