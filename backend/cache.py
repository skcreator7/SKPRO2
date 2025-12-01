# cache.py - Redis Cache & Auto Cleanup System

import logging
import asyncio
import redis.asyncio as redis
import json
from datetime import datetime, timedelta
from typing import Optional, Any

logger = logging.getLogger(__name__)

class CacheManager:
    """Redis Cache Manager with Auto Cleanup"""
    
    def __init__(self, config):
        self.config = config
        self.client = None
        self.enabled = False
        self.cleanup_task = None
    
    async def init_redis(self):
        """Initialize Redis connection"""
        try:
            self.client = redis.Redis(
                host='redis-17119.c283.us-east-1-4.ec2.cloud.redislabs.com',
                port=17119,
                username='default',
                password='EjtnvQpIkLv5Z3g9Fr4FQDLfmLKZVqML',
                decode_responses=True,
                encoding='utf-8',
                socket_connect_timeout=10,
                socket_timeout=10,
                max_connections=10,
                health_check_interval=30
            )
            
            await self.client.ping()
            self.enabled = True
            logger.info("✅ Redis cache connected!")
            
            # Test basic operations
            await self.client.set('connection_test', 'success', ex=60)
            test_result = await self.client.get('connection_test')
            
            if test_result == 'success':
                logger.info("✅ Redis operations test PASSED")
            else:
                logger.warning("⚠️ Redis operations test FAILED")
            
            return True
            
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.enabled = False
            return False
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.enabled or not self.client:
            return None
        
        try:
            data = await self.client.get(key)
            if data:
                try:
                    return json.loads(data)
                except:
                    return data
        except Exception as e:
            logger.warning(f"Redis get error: {e}")
        
        return None
    
    async def set(self, key: str, value: Any, expire: int = 3600):
        """Set value in cache"""
        if not self.enabled or not self.client:
            return False
        
        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            
            await self.client.setex(key, expire, value)
            return True
        except Exception as e:
            logger.warning(f"Redis set error: {e}")
            return False
    
    async def delete(self, key: str):
        """Delete key from cache"""
        if not self.enabled or not self.client:
            return False
        
        try:
            await self.client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Redis delete error: {e}")
            return False
    
    async def clear_pattern(self, pattern: str):
        """Clear keys matching pattern"""
        if not self.enabled or not self.client:
            return False
        
        try:
            keys = await self.client.keys(pattern)
            if keys:
                await self.client.delete(*keys)
                logger.info(f"✅ Cleared {len(keys)} keys matching '{pattern}'")
            return True
        except Exception as e:
            logger.warning(f"Redis clear error: {e}")
            return False
    
    # Premium-specific cache methods
    async def cache_premium_user(self, user_id: int, data: dict, expire: int = 3600):
        """Cache premium user data"""
        key = f"premium:{user_id}"
        return await self.set(key, data, expire)
    
    async def get_premium_user(self, user_id: int) -> Optional[dict]:
        """Get cached premium user data"""
        key = f"premium:{user_id}"
        return await self.get(key)
    
    async def remove_premium_user(self, user_id: int):
        """Remove premium user from cache"""
        key = f"premium:{user_id}"
        return await self.delete(key)
    
    # Verification-specific cache methods
    async def cache_verification(self, user_id: int, data: dict, expire: int = 21600):  # 6 hours
        """Cache verification data"""
        key = f"verification:{user_id}"
        return await self.set(key, data, expire)
    
    async def get_verification(self, user_id: int) -> Optional[dict]:
        """Get cached verification data"""
        key = f"verification:{user_id}"
        return await self.get(key)
    
    async def remove_verification(self, user_id: int):
        """Remove verification from cache"""
        key = f"verification:{user_id}"
        return await self.delete(key)
    
    # Auto cleanup tasks
    async def start_auto_cleanup(self, premium_manager, verification_manager):
        """Start automatic cleanup tasks"""
        self.cleanup_task = asyncio.create_task(
            self._cleanup_loop(premium_manager, verification_manager)
        )
        logger.info("✅ Auto cleanup tasks started")
    
    async def _cleanup_loop(self, premium_manager, verification_manager):
        """Background cleanup loop"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every 1 hour
                
                logger.info("🧹 Running auto cleanup...")
                
                # Cleanup expired premium users
                premium_count = await premium_manager.cleanup_expired_premium()
                
                # Cleanup expired verifications
                verification_count = await verification_manager.cleanup_expired_verifications()
                
                # Clear expired cache keys
                await self.clear_pattern("premium:*")
                await self.clear_pattern("verification:*")
                
                logger.info(
                    f"✅ Cleanup complete: "
                    f"{premium_count} premium, "
                    f"{verification_count} verifications removed"
                )
                
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def stop_auto_cleanup(self):
        """Stop cleanup tasks"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            logger.info("✅ Auto cleanup stopped")
    
    async def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        if not self.enabled or not self.client:
            return {'enabled': False}
        
        try:
            info = await self.client.info('stats')
            return {
                'enabled': True,
                'total_keys': await self.client.dbsize(),
                'hits': info.get('keyspace_hits', 0),
                'misses': info.get('keyspace_misses', 0),
                'hit_rate': f"{(info.get('keyspace_hits', 0) / (info.get('keyspace_hits', 0) + info.get('keyspace_misses', 1))) * 100:.1f}%"
            }
        except:
            return {'enabled': True, 'error': 'Could not fetch stats'}


# Initialize cache manager
cache_manager = None

async def init_cache_manager(config):
    """Initialize global cache manager"""
    global cache_manager
    cache_manager = CacheManager(config)
    await cache_manager.init_redis()
    return cache_manager
