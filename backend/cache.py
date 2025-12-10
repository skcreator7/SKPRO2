"""
cache.py - Redis and in-memory caching system
"""
import asyncio
import json
import pickle
import re
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Union, List
import logging
import redis.asyncio as redis

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self, config):
        self.config = config
        self.redis_client = None
        self.redis_enabled = False
        
        # Memory caches with expiry
        self.memory_cache = {}  # key -> (value, expiry_time)
        self.search_cache = {}  # search_key -> (data, timestamp)
        self.stats = {
            'redis_hits': 0,
            'redis_misses': 0,
            'memory_hits': 0,
            'memory_misses': 0,
            'memory_evictions': 0,
            'batch_operations': 0
        }
        
        # Cleanup task
        self.cleanup_task = None
    
    async def init_redis(self) -> bool:
        """Initialize Redis connection"""
        try:
            # Check if Redis URL is configured
            if not hasattr(self.config, 'REDIS_URL') or not self.config.REDIS_URL:
                logger.warning("⚠️ Redis URL not configured, using memory cache only")
                return False
            
            # Parse Redis URL
            redis_url = self.config.REDIS_URL
            
            # Special handling for Redis Labs format
            if 'redislabs.com' in redis_url or 'cloud.redislabs.com' in redis_url:
                # Parse Redis Labs URL format: redis://username:password@host:port
                try:
                    # Extract host and port from URL
                    match = re.search(r'redis-(\d+)\.([^:]+):(\d+)', redis_url)
                    if match:
                        # Full host with port
                        host_with_port = match.group(0)
                        host = host_with_port.split(':')[0]
                        port = int(host_with_port.split(':')[1]) if ':' in host_with_port else 6379
                        
                        # Get password from config or extract from URL
                        password = getattr(self.config, 'REDIS_PASSWORD', None)
                        if not password and '@' in redis_url:
                            # Extract password from URL: redis://default:password@host:port
                            password_part = redis_url.split('@')[0]
                            if ':' in password_part:
                                password = password_part.split(':')[-1]
                        
                        logger.info(f"🔌 Connecting to Redis Labs: {host}:{port}")
                        
                        self.redis_client = redis.Redis(
                            host=host,
                            port=port,
                            password=password,
                            decode_responses=True,
                            encoding='utf-8',
                            socket_connect_timeout=10,
                            socket_timeout=10,
                            max_connections=10,
                            health_check_interval=30
                        )
                    else:
                        # Fallback to from_url
                        logger.info(f"🔌 Connecting to Redis via URL: {redis_url}")
                        self.redis_client = redis.from_url(
                            redis_url,
                            decode_responses=True,
                            encoding='utf-8',
                            socket_connect_timeout=10,
                            socket_timeout=10
                        )
                except Exception as e:
                    logger.warning(f"Redis Labs parsing failed: {e}")
                    # Simple fallback
                    self.redis_client = redis.from_url(
                        redis_url,
                        decode_responses=True,
                        encoding='utf-8'
                    )
            else:
                # Standard Redis URL
                self.redis_client = redis.from_url(
                    redis_url,
                    decode_responses=True,
                    encoding='utf-8'
                )
            
            # Test connection
            await self.redis_client.ping()
            self.redis_enabled = True
            
            # Test basic operations
            await self.redis_client.set('connection_test', 'success', ex=60)
            test_result = await self.redis_client.get('connection_test')
            
            if test_result == 'success':
                logger.info("✅ Redis connected successfully!")
            else:
                logger.warning("⚠️ Redis connected but operations test failed")
                self.redis_enabled = False
            
            return self.redis_enabled
            
        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed: {e}")
            self.redis_enabled = False
            return False
    
    def _serialize_value(self, value: Any) -> str:
        """Serialize value for storage"""
        try:
            return json.dumps(value, default=str)
        except:
            # Fallback to pickle for complex objects
            return pickle.dumps(value).decode('latin-1')
    
    def _deserialize_value(self, value: str, use_json: bool = True) -> Any:
        """Deserialize stored value"""
        if value is None:
            return None
        
        try:
            if use_json:
                return json.loads(value)
            else:
                return pickle.loads(value.encode('latin-1'))
        except:
            return value
    
    async def get(self, key: str, use_redis: bool = True) -> Optional[Any]:
        """Get value from cache"""
        # Try memory cache first
        if key in self.memory_cache:
            value, expiry = self.memory_cache[key]
            if expiry is None or datetime.now() < expiry:
                self.stats['memory_hits'] += 1
                return value
            else:
                # Expired, remove from memory
                del self.memory_cache[key]
                self.stats['memory_evictions'] += 1
        
        self.stats['memory_misses'] += 1
        
        # Try Redis if enabled
        if use_redis and self.redis_enabled and self.redis_client:
            try:
                value = await self.redis_client.get(key)
                if value is not None:
                    self.stats['redis_hits'] += 1
                    
                    # Deserialize based on key prefix
                    if key.startswith('json:'):
                        deserialized = self._deserialize_value(value, use_json=True)
                    else:
                        deserialized = self._deserialize_value(value, use_json=False)
                    
                    # Also store in memory cache for faster access
                    self.memory_cache[key] = (deserialized, datetime.now() + timedelta(minutes=5))
                    
                    return deserialized
                else:
                    self.stats['redis_misses'] += 1
            except Exception as e:
                logger.warning(f"Redis get error: {e}")
        
        return None
    
    async def batch_get(self, keys: List[str], use_redis: bool = True) -> Dict[str, Any]:
        """Get multiple values from cache at once"""
        result = {}
        
        # Check memory cache first
        remaining_keys = []
        for key in keys:
            if key in self.memory_cache:
                value, expiry = self.memory_cache[key]
                if expiry is None or datetime.now() < expiry:
                    result[key] = value
                    self.stats['memory_hits'] += 1
                else:
                    # Expired
                    del self.memory_cache[key]
                    self.stats['memory_evictions'] += 1
                    remaining_keys.append(key)
            else:
                remaining_keys.append(key)
                self.stats['memory_misses'] += 1
        
        if not remaining_keys:
            return result
        
        # Try Redis for remaining keys
        if use_redis and self.redis_enabled and self.redis_client and remaining_keys:
            try:
                # Get values from Redis
                values = await self.redis_client.mget(remaining_keys)
                
                for i, key in enumerate(remaining_keys):
                    value = values[i]
                    if value is not None:
                        self.stats['redis_hits'] += 1
                        
                        # Deserialize based on key prefix
                        if key.startswith('json:'):
                            deserialized = self._deserialize_value(value, use_json=True)
                        else:
                            deserialized = self._deserialize_value(value, use_json=False)
                        
                        result[key] = deserialized
                        
                        # Store in memory cache
                        self.memory_cache[key] = (deserialized, datetime.now() + timedelta(minutes=5))
                    else:
                        self.stats['redis_misses'] += 1
                        result[key] = None
                        
            except Exception as e:
                logger.warning(f"Redis batch get error: {e}")
                # Mark remaining as None
                for key in remaining_keys:
                    result[key] = None
        
        return result
    
    async def set(self, key: str, value: Any, expire_seconds: int = 3600, 
                  use_redis: bool = True) -> bool:
        """Set value in cache"""
        # Store in memory
        expiry = datetime.now() + timedelta(seconds=expire_seconds) if expire_seconds > 0 else None
        self.memory_cache[key] = (value, expiry)
        
        # Store in Redis if enabled
        if use_redis and self.redis_enabled and self.redis_client:
            try:
                # Serialize based on value type
                if isinstance(value, (dict, list, tuple, int, float, bool, str)) and not key.startswith('binary:'):
                    serialized = self._serialize_value(value)
                    prefixed_key = f"json:{key}"
                else:
                    serialized = self._serialize_value(value)
                    prefixed_key = key
                
                # Store with prefix for deserialization
                await self.redis_client.setex(
                    prefixed_key, 
                    expire_seconds, 
                    serialized
                )
                return True
            except Exception as e:
                logger.warning(f"Redis set error: {e}")
                return False
        
        return True
    
    async def batch_set(self, items: Dict[str, Any], ttl: int = None) -> bool:
        """
        Set multiple cache items at once
        items: Dictionary of key-value pairs
        ttl: Time to live in seconds (default: 3600)
        """
        try:
            if ttl is None:
                ttl = 3600
            
            expiry = datetime.now() + timedelta(seconds=ttl) if ttl > 0 else None
            
            # Store in memory cache
            for key, value in items.items():
                self.memory_cache[key] = (value, expiry)
            
            # Store in Redis if enabled
            if self.redis_enabled and self.redis_client:
                pipeline = self.redis_client.pipeline()
                
                for key, value in items.items():
                    # Serialize based on value type
                    if isinstance(value, (dict, list, tuple, int, float, bool, str)) and not key.startswith('binary:'):
                        serialized = self._serialize_value(value)
                        prefixed_key = f"json:{key}"
                    else:
                        serialized = self._serialize_value(value)
                        prefixed_key = key
                    
                    # Add to pipeline
                    pipeline.setex(prefixed_key, ttl, serialized)
                
                # Execute all commands in pipeline
                await pipeline.execute()
            
            self.stats['batch_operations'] += 1
            logger.debug(f"✅ Batch set completed for {len(items)} items")
            return True
            
        except Exception as e:
            logger.error(f"❌ Batch set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        # Delete from memory
        if key in self.memory_cache:
            del self.memory_cache[key]
        
        # Delete from Redis
        if self.redis_enabled and self.redis_client:
            try:
                # Try both prefixed and non-prefixed versions
                await self.redis_client.delete(f"json:{key}", key)
                return True
            except Exception as e:
                logger.warning(f"Redis delete error: {e}")
                return False
        
        return True
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern"""
        cleared_count = 0
        
        # Clear from memory cache
        keys_to_delete = [k for k in self.memory_cache.keys() if pattern in k]
        for key in keys_to_delete:
            del self.memory_cache[key]
            cleared_count += 1
        
        # Clear from Redis
        if self.redis_enabled and self.redis_client:
            try:
                # Find keys matching pattern
                keys = await self.redis_client.keys(f"*{pattern}*")
                if keys:
                    await self.redis_client.delete(*keys)
                    cleared_count += len(keys)
            except Exception as e:
                logger.warning(f"Redis pattern delete error: {e}")
        
        logger.info(f"🧹 Cleared {cleared_count} keys matching pattern: {pattern}")
        return cleared_count
    
    async def clear_search_cache(self) -> int:
        """Clear all search-related cache"""
        return await self.clear_pattern("search:")
    
    async def clear_all(self) -> bool:
        """Clear all cache"""
        try:
            # Clear memory cache
            self.memory_cache.clear()
            
            # Clear Redis
            if self.redis_enabled and self.redis_client:
                await self.redis_client.flushdb()
            
            # Reset stats
            self.stats = {
                'redis_hits': 0,
                'redis_misses': 0,
                'memory_hits': 0,
                'memory_misses': 0,
                'memory_evictions': 0,
                'batch_operations': 0
            }
            
            logger.info("✅ All cache cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Clear all cache error: {e}")
            return False
    
    async def cache_search_results(self, query: str, page: int, limit: int, results: Dict[str, Any]) -> bool:
        """Cache search results"""
        cache_key = f"search:{query}:{page}:{limit}"
        return await self.set(cache_key, results, expire_seconds=1800)
    
    async def get_search_results(self, query: str, page: int, limit: int) -> Optional[Dict[str, Any]]:
        """Get cached search results"""
        cache_key = f"search:{query}:{page}:{limit}"
        return await self.get(cache_key)
    
    async def batch_cache_posters(self, poster_data_dict: Dict[str, Dict[str, Any]]) -> bool:
        """Cache multiple posters at once"""
        try:
            # Prepare items for batch set
            items = {}
            for title, poster_data in poster_data_dict.items():
                cache_key = f"poster:{title.lower()}"
                items[cache_key] = poster_data
            
            # Use batch_set for better performance
            return await self.batch_set(items, ttl=7200)
            
        except Exception as e:
            logger.error(f"Batch cache posters error: {e}")
            return False
    
    async def cache_poster(self, title: str, poster_data: Dict[str, Any]) -> bool:
        """Cache poster data"""
        cache_key = f"poster:{title.lower()}"
        return await self.set(cache_key, poster_data, expire_seconds=7200)
    
    async def get_poster(self, title: str) -> Optional[Dict[str, Any]]:
        """Get cached poster data"""
        cache_key = f"poster:{title.lower()}"
        return await self.get(cache_key)
    
    async def batch_get_posters(self, titles: List[str]) -> Dict[str, Optional[Dict[str, Any]]]:
        """Get multiple posters at once"""
        result = {}
        
        # Prepare cache keys
        cache_keys = [f"poster:{title.lower()}" for title in titles]
        
        # Batch get from cache
        cached_data = await self.batch_get(cache_keys)
        
        # Map back to titles
        for i, title in enumerate(titles):
            cache_key = cache_keys[i]
            result[title] = cached_data.get(cache_key)
        
        return result
    
    async def cache_user_data(self, user_id: int, data: Dict[str, Any]) -> bool:
        """Cache user-specific data"""
        cache_key = f"user:{user_id}:data"
        return await self.set(cache_key, data, expire_seconds=300)
    
    async def get_user_data(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get cached user data"""
        cache_key = f"user:{user_id}:data"
        return await self.get(cache_key)
    
    async def increment_counter(self, key: str, amount: int = 1) -> int:
        """Increment counter in cache"""
        # Memory cache
        current = await self.get(key) or 0
        new_value = current + amount
        
        await self.set(key, new_value, expire_seconds=86400)
        return new_value
    
    async def batch_increment_counters(self, counters: Dict[str, int]) -> Dict[str, int]:
        """Increment multiple counters at once"""
        result = {}
        
        for key, amount in counters.items():
            result[key] = await self.increment_counter(key, amount)
        
        return result
    
    async def get_stats_summary(self) -> Dict[str, Any]:
        """Get cache statistics summary"""
        redis_info = {}
        if self.redis_enabled and self.redis_client:
            try:
                info = await self.redis_client.info()
                redis_info = {
                    'connected_clients': info.get('connected_clients', 0),
                    'used_memory': info.get('used_memory_human', '0'),
                    'total_commands_processed': info.get('total_commands_processed', 0),
                    'keyspace_hits': info.get('keyspace_hits', 0),
                    'keyspace_misses': info.get('keyspace_misses', 0)
                }
            except:
                pass
        
        # Calculate hit rates
        total_redis_ops = self.stats['redis_hits'] + self.stats['redis_misses']
        total_memory_ops = self.stats['memory_hits'] + self.stats['memory_misses']
        
        redis_hit_rate = self.stats['redis_hits'] / total_redis_ops if total_redis_ops > 0 else 0
        memory_hit_rate = self.stats['memory_hits'] / total_memory_ops if total_memory_ops > 0 else 0
        
        return {
            'redis_enabled': self.redis_enabled,
            'redis_info': redis_info,
            'memory_cache_size': len(self.memory_cache),
            'stats': self.stats.copy(),
            'hit_rates': {
                'redis': round(redis_hit_rate, 3),
                'memory': round(memory_hit_rate, 3)
            },
            'operations_summary': {
                'total_redis_ops': total_redis_ops,
                'total_memory_ops': total_memory_ops,
                'batch_operations': self.stats['batch_operations']
            }
        }
    
    async def start_cleanup_task(self):
        """Start background cleanup task for memory cache"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _cleanup_loop(self):
        """Background cleanup loop for expired memory cache entries"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Cleanup expired memory cache entries
                now = datetime.now()
                expired_keys = []
                
                for key, (value, expiry) in self.memory_cache.items():
                    if expiry is not None and now > expiry:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self.memory_cache[key]
                    self.stats['memory_evictions'] += 1
                
                if expired_keys:
                    logger.debug(f"🧹 Cleaned {len(expired_keys)} expired memory cache entries")
                    
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
    
    async def stop(self):
        """Stop cache manager"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self.redis_client:
            await self.redis_client.close()
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on cache system"""
        health = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {}
        }
        
        # Check memory cache
        health['components']['memory_cache'] = {
            'status': 'healthy',
            'size': len(self.memory_cache),
            'details': f"{len(self.memory_cache)} items in memory"
        }
        
        # Check Redis
        if self.redis_enabled and self.redis_client:
            try:
                await self.redis_client.ping()
                info = await self.redis_client.info('memory')
                
                health['components']['redis'] = {
                    'status': 'healthy',
                    'connected': True,
                    'memory_used': info.get('used_memory_human', 'unknown')
                }
            except Exception as e:
                health['components']['redis'] = {
                    'status': 'unhealthy',
                    'connected': False,
                    'error': str(e)
                }
                health['status'] = 'degraded'
        else:
            health['components']['redis'] = {
                'status': 'disabled',
                'connected': False
            }
        
        # Check stats
        health['stats'] = self.get_stats_summary()
        
        return health
