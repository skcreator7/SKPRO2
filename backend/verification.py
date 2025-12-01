"""
verification.py - URL shortener and verification system
"""
import asyncio
import json
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, Optional
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
        
        # Cleanup task
        self.cleanup_task = None
    
    def generate_unique_token(self, length=32) -> str:
        """Generate unique verification token"""
        return secrets.token_urlsafe(length)
    
    async def get_shortened_url(self, destination_url: str) -> Tuple[str, str]:
        """Get shortened URL using GPLinks or similar service"""
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
                'status': 'pending'
            }
            
            # Store in memory caches
            self.pending_verifications[user_id] = verification_data
            self.verification_tokens[verification_token] = user_id
            
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
                'status': 'pending'
            }
            
            self.pending_verifications[user_id] = verification_data
            self.verification_tokens[verification_token] = user_id
            
            return verification_data
    
    async def verify_user_token(self, token: str) -> Tuple[bool, Optional[int], str]:
        """Verify user token and return user_id if valid"""
        try:
            user_id = self.verification_tokens.get(token)
            if not user_id:
                return False, None, "Invalid token"
            
            verification_data = self.pending_verifications.get(user_id)
            if not verification_data:
                return False, user_id, "No pending verification"
            
            # Check if token matches
            if verification_data['token'] != token:
                return False, user_id, "Token mismatch"
            
            # Check expiry (1 hour)
            created_at = verification_data['created_at']
            if datetime.now() - created_at > timedelta(hours=1):
                # Cleanup expired
                self._cleanup_user_verification(user_id)
                return False, user_id, "Token expired"
            
            # Mark as verified
            verification_data['status'] = 'verified'
            verification_data['verified_at'] = datetime.now()
            
            # Store in verified users (valid for 6 hours)
            expiry_time = datetime.now() + timedelta(hours=6)
            self.verified_users[user_id] = expiry_time
            
            # Cleanup from pending
            self._cleanup_user_verification(user_id)
            
            logger.info(f"✅ User {user_id} verified successfully")
            return True, user_id, "Verification successful"
            
        except Exception as e:
            logger.error(f"❌ Token verification error: {e}")
            return False, None, "Internal error"
    
    def _cleanup_user_verification(self, user_id: int):
        """Cleanup verification data for user"""
        if user_id in self.pending_verifications:
            token = self.pending_verifications[user_id].get('token')
            if token and token in self.verification_tokens:
                del self.verification_tokens[token]
            del self.pending_verifications[user_id]
    
    async def check_user_verified(self, user_id: int) -> Tuple[bool, str]:
        """Check if user is currently verified"""
        if user_id in self.verified_users:
            expiry_time = self.verified_users[user_id]
            if datetime.now() < expiry_time:
                return True, f"Verified (expires in {expiry_time - datetime.now()})"
            else:
                # Expired, cleanup
                del self.verified_users[user_id]
                return False, "Verification expired"
        return False, "Not verified"
    
    async def extend_verification(self, user_id: int, hours: int = 6) -> bool:
        """Extend user verification"""
        if user_id in self.verified_users:
            self.verified_users[user_id] = datetime.now() + timedelta(hours=hours)
            logger.info(f"✅ Extended verification for user {user_id} by {hours} hours")
            return True
        return False
    
    async def revoke_verification(self, user_id: int) -> bool:
        """Revoke user verification"""
        if user_id in self.verified_users:
            del self.verified_users[user_id]
            logger.info(f"✅ Revoked verification for user {user_id}")
            return True
        return False
    
    async def get_user_stats(self) -> Dict[str, Any]:
        """Get verification statistics"""
        return {
            'pending_verifications': len(self.pending_verifications),
            'verified_users': len(self.verified_users),
            'active_tokens': len(self.verification_tokens)
        }
    
    async def start_cleanup_task(self):
        """Start background cleanup task"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Cleanup expired pending verifications
                now = datetime.now()
                expired_users = []
                
                for user_id, data in self.pending_verifications.items():
                    created_at = data['created_at']
                    if now - created_at > timedelta(hours=1):
                        expired_users.append(user_id)
                
                for user_id in expired_users:
                    self._cleanup_user_verification(user_id)
                
                # Cleanup expired verified users
                expired_verified = [
                    user_id for user_id, expiry in self.verified_users.items()
                    if now > expiry
                ]
                
                for user_id in expired_verified:
                    del self.verified_users[user_id]
                
                if expired_users or expired_verified:
                    logger.info(f"🧹 Cleanup: {len(expired_users)} pending, {len(expired_verified)} verified")
                    
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
    
    async def stop(self):
        """Stop verification system"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
