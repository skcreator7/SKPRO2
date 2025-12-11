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
        self.stats = {
            'redis_hits': 0,
            'redis_misses': 0,
            'memory_hits': 0,
            'memory_misses': 0,
            'memory_evictions': 0
        }
        
        # Default fallback poster
        self.default_poster = "https://iili.io/fAeIwv9.th.png"
        
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
                            socket_connect_timeout=5,
                            socket_timeout=5,
                            max_connections=20,
                            health_check_interval=15,
                            retry_on_timeout=True
                        )
                    else:
                        # Fallback to from_url
                        logger.info(f"🔌 Connecting to Redis via URL: {redis_url}")
                        self.redis_client = redis.from_url(
                            redis_url,
                            decode_responses=True,
                            encoding='utf-8',
                            socket_connect_timeout=5,
                            socket_timeout=5,
                            retry_on_timeout=True
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
                    encoding='utf-8',
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
            
            # Test connection with timeout
            try:
                await asyncio.wait_for(self.redis_client.ping(), timeout=3)
                self.redis_enabled = True
                
                # Test basic operations
                await self.redis_client.set('connection_test', 'success', ex=10)
                test_result = await self.redis_client.get('connection_test')
                
                if test_result == 'success':
                    logger.info("✅ Redis connected successfully!")
                else:
                    logger.warning("⚠️ Redis connected but operations test failed")
                    self.redis_enabled = False
            except asyncio.TimeoutError:
                logger.warning("⚠️ Redis connection timeout")
                self.redis_enabled = False
                return False
            
            return self.redis_enabled
            
        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed: {e}")
            self.redis_enabled = False
            return False
    
    def _serialize_value(self, value: Any) -> str:
        """Serialize value for storage, preferring JSON when possible"""
        try:
            # Try JSON first (faster and more standard)
            return json.dumps(value, default=str, ensure_ascii=False)
        except:
            # Fallback to pickle for complex objects
            try:
                return pickle.dumps(value).decode('latin-1')
            except:
                return str(value)
    
    def _deserialize_value(self, value: str, use_json: bool = True) -> Any:
        """Deserialize stored value, preferring JSON when possible"""
        if value is None:
            return None
        
        if use_json:
            try:
                return json.loads(value)
            except:
                # Try pickle as fallback
                try:
                    return pickle.loads(value.encode('latin-1'))
                except:
                    return value
        else:
            # Try pickle first
            try:
                return pickle.loads(value.encode('latin-1'))
            except:
                # Try JSON as fallback
                try:
                    return json.loads(value)
                except:
                    return value
    
    async def get(self, key: str, use_redis: bool = True) -> Optional[Any]:
        """Get value from cache"""
        # Try memory cache first (fastest)
        if key in self.memory_cache:
            value, expiry = self.memory_cache[key]
            if expiry is None or datetime.now() < expiry:
                self.stats['memory_hits'] += 1
                
                # JSON decode if needed
                try:
                    if isinstance(value, str):
                        return json.loads(value)
                    return value
                except:
                    return value
            else:
                # Expired, remove from memory
                del self.memory_cache[key]
                self.stats['memory_evictions'] += 1
        
        self.stats['memory_misses'] += 1
        
        # Try Redis if enabled
        if use_redis and self.redis_enabled and self.redis_client:
            try:
                # First try with json prefix
                value = await self.redis_client.get(f"json:{key}")
                if value is None:
                    # Try without prefix
                    value = await self.redis_client.get(key)
                
                if value is not None:
                    self.stats['redis_hits'] += 1
                    
                    # Try JSON decode
                    try:
                        deserialized = json.loads(value)
                    except:
                        # Fallback to other deserialization methods
                        if 'json:' in str(key):
                            deserialized = self._deserialize_value(value, use_json=True)
                        else:
                            deserialized = self._deserialize_value(value, use_json=False)
                    
                    # Also store in memory cache for faster access (with shorter expiry)
                    memory_expiry = datetime.now() + timedelta(minutes=2)
                    self.memory_cache[key] = (deserialized, memory_expiry)
                    
                    return deserialized
                else:
                    self.stats['redis_misses'] += 1
            except Exception as e:
                logger.debug(f"Redis get error for key {key}: {e}")
                # Continue without Redis
        
        return None
    
    async def set(self, key: str, value: Any, expire_seconds: int = 3600, 
                  use_redis: bool = True, priority: str = 'normal') -> bool:
        """Set value in cache with priority-based expiry"""
        
        # Priority-based expiry adjustments
        if priority == 'high':
            # High priority items have longer memory cache
            memory_expire_seconds = min(expire_seconds, 300)  # Max 5 min in memory
            redis_expire_seconds = expire_seconds
        elif priority == 'low':
            memory_expire_seconds = min(expire_seconds, 60)   # Max 1 min in memory
            redis_expire_seconds = min(expire_seconds, 1800)  # Max 30 min in Redis
        else:  # normal
            memory_expire_seconds = min(expire_seconds, 180)  # Max 3 min in memory
            redis_expire_seconds = expire_seconds
        
        # Store in memory (fast access)
        memory_expiry = datetime.now() + timedelta(seconds=memory_expire_seconds) if memory_expire_seconds > 0 else None
        self.memory_cache[key] = (value, memory_expiry)
        
        # Store in Redis if enabled (persistent storage)
        if use_redis and self.redis_enabled and self.redis_client:
            try:
                # Try JSON serialization first for common data types
                serialized = None
                try:
                    if isinstance(value, (dict, list, tuple, int, float, bool, str)):
                        serialized = json.dumps(value, default=str, ensure_ascii=False)
                        prefixed_key = f"json:{key}"
                    else:
                        raise ValueError("Not JSON serializable")
                except:
                    # Fallback to other serialization methods
                    serialized = self._serialize_value(value)
                    prefixed_key = key
                
                # Store with expiry
                await self.redis_client.setex(
                    prefixed_key, 
                    redis_expire_seconds, 
                    serialized
                )
                return True
            except Exception as e:
                logger.debug(f"Redis set error for key {key}: {e}")
                # Don't fail if Redis is down, memory cache still works
                return True  # Still success because memory cache worked
        
        return True
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        success = True
        
        # Delete from memory
        if key in self.memory_cache:
            del self.memory_cache[key]
        
        # Delete from Redis
        if self.redis_enabled and self.redis_client:
            try:
                # Try both prefixed and non-prefixed versions
                await self.redis_client.delete(f"json:{key}", key)
            except Exception as e:
                logger.debug(f"Redis delete error: {e}")
                success = False
        
        return success
    
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
                logger.debug(f"Redis pattern delete error: {e}")
        
        if cleared_count > 0:
            logger.debug(f"🧹 Cleared {cleared_count} keys matching pattern: {pattern}")
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
                'memory_evictions': 0
            }
            
            logger.info("✅ All cache cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Clear all cache error: {e}")
            return False
    
    async def cache_search_results(self, query: str, page: int, limit: int, results: Dict[str, Any]) -> bool:
        """Cache search results with high priority"""
        cache_key = f"search:{query}:{page}:{limit}"
        return await self.set(cache_key, results, expire_seconds=900, priority='high')
    
    async def get_search_results(self, query: str, page: int, limit: int) -> Optional[Dict[str, Any]]:
        """Get cached search results"""
        cache_key = f"search:{query}:{page}:{limit}"
        return await self.get(cache_key, use_redis=True)
    
    async def cache_poster(self, title: str, poster_data: Dict[str, Any], source: str = 'default') -> bool:
        """Cache poster data with very high priority"""
        cache_key = f"poster:{title.lower()}:{source}"
        # Posters are high priority - cache longer
        return await self.set(cache_key, poster_data, expire_seconds=86400, priority='high')
    
    async def get_poster(self, title: str, source: str = 'default') -> Optional[Dict[str, Any]]:
        """Get cached poster data - check memory first for speed"""
        cache_key = f"poster:{title.lower()}:{source}"
        
        # First try memory cache (fastest)
        result = await self.get(cache_key, use_redis=False)
        if result:
            return result
        
        # Then try Redis
        return await self.get(cache_key, use_redis=True)
    
    async def cache_user_data(self, user_id: int, data: Dict[str, Any]) -> bool:
        """Cache user-specific data"""
        cache_key = f"user:{user_id}:data"
        return await self.set(cache_key, data, expire_seconds=600)
    
    async def get_user_data(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get cached user data"""
        cache_key = f"user:{user_id}:data"
        return await self.get(cache_key)
    
    async def get_default_poster(self) -> str:
        """Get default fallback poster URL"""
        return self.default_poster
    
    async def cache_multi_posters(self, posters_data: Dict[str, Dict[str, Any]]) -> bool:
        """Cache multiple posters at once"""
        success = True
        for title, data in posters_data.items():
            if not await self.cache_poster(title, data):
                success = False
        return success
    
    async def get_multi_posters(self, titles: List[str]) -> Dict[str, Optional[Dict[str, Any]]]:
        """Get multiple posters at once"""
        results = {}
        for title in titles:
            # Try all sources in order
            for source in ['letterboxd', 'omdb', 'imdb', 'impawards', 'justwatch', 'youtube', 'default']:
                poster = await self.get_poster(title, source)
                if poster:
                    results[title] = poster
                    break
            if title not in results:
                results[title] = None
        return results
    
    async def increment_counter(self, key: str, amount: int = 1) -> int:
        """Increment counter in cache"""
        try:
            if self.redis_enabled and self.redis_client:
                # Use Redis atomic increment
                new_value = await self.redis_client.incrby(key, amount)
                # Set expiry if first time
                await self.redis_client.expire(key, 86400)
            else:
                # Fallback to memory
                current = await self.get(key) or 0
                new_value = current + amount
                await self.set(key, new_value, expire_seconds=86400, use_redis=False)
            
            return new_value
        except:
            return 0
    
    async def batch_set(self, data: Dict[str, Any], expire_seconds: int = 3600, 
                        use_redis: bool = True, priority: str = 'normal') -> bool:
        """
        Efficiently store multiple cache items using Redis pipeline
        
        Args:
            data: Dictionary of key-value pairs to cache
            expire_seconds: Cache expiry in seconds
            use_redis: Whether to use Redis (True) or memory only (False)
            priority: Cache priority ('high', 'normal', 'low')
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Priority-based expiry adjustments (same as in set method)
            if priority == 'high':
                memory_expire_seconds = min(expire_seconds, 300)  # Max 5 min in memory
                redis_expire_seconds = expire_seconds
            elif priority == 'low':
                memory_expire_seconds = min(expire_seconds, 60)   # Max 1 min in memory
                redis_expire_seconds = min(expire_seconds, 1800)  # Max 30 min in Redis
            else:  # normal
                memory_expire_seconds = min(expire_seconds, 180)  # Max 3 min in memory
                redis_expire_seconds = expire_seconds
            
            # Store in memory cache for all items
            memory_expiry = datetime.now() + timedelta(seconds=memory_expire_seconds) if memory_expire_seconds > 0 else None
            for key, value in data.items():
                self.memory_cache[key] = (value, memory_expiry)
            
            # Store in Redis if enabled using pipeline
            if use_redis and self.redis_enabled and self.redis_client:
                try:
                    pipe = self.redis_client.pipeline()
                    
                    for key, value in data.items():
                        # Try JSON serialization first
                        try:
                            if isinstance(value, (dict, list, tuple, int, float, bool, str)) and not key.startswith('binary:'):
                                serialized = json.dumps(value, default=str, ensure_ascii=False)
                                prefixed_key = f"json:{key}"
                            else:
                                raise ValueError("Not JSON serializable")
                        except:
                            # Fallback to other serialization
                            serialized = self._serialize_value(value)
                            prefixed_key = key
                        
                        # Store with expiry
                        if redis_expire_seconds > 0:
                            pipe.setex(prefixed_key, redis_expire_seconds, serialized)
                        else:
                            pipe.set(prefixed_key, serialized)
                    
                    await pipe.execute()
                    return True
                except Exception as e:
                    logger.debug(f"Redis batch set error: {e}")
                    # Don't fail if Redis is down, memory cache still worked
                    return True  # Still success because memory cache worked
            
            return True
            
        except Exception as e:
            logger.error(f"Batch set error: {e}")
            return False
    
    async def batch_get(self, keys: List[str], use_redis: bool = True) -> Dict[str, Optional[Any]]:
        """
        Get multiple values from cache at once
        
        Args:
            keys: List of keys to retrieve
            use_redis: Whether to check Redis
        
        Returns:
            Dict of key-value pairs (missing keys will have None value)
        """
        results = {}
        
        # First check memory cache
        for key in keys:
            if key in self.memory_cache:
                value, expiry = self.memory_cache[key]
                if expiry is None or datetime.now() < expiry:
                    self.stats['memory_hits'] += 1
                    
                    # Try JSON decode
                    try:
                        if isinstance(value, str):
                            results[key] = json.loads(value)
                        else:
                            results[key] = value
                    except:
                        results[key] = value
                else:
                    # Expired
                    del self.memory_cache[key]
                    self.stats['memory_evictions'] += 1
                    results[key] = None
            else:
                self.stats['memory_misses'] += 1
                results[key] = None
        
        # For keys not found in memory, check Redis
        if use_redis and self.redis_enabled and self.redis_client:
            try:
                # Get all missing keys from Redis
                missing_keys = [k for k in keys if results.get(k) is None]
                
                if missing_keys:
                    # Try with json prefix first
                    prefixed_keys = [f"json:{k}" for k in missing_keys]
                    redis_values = await self.redis_client.mget(*prefixed_keys)
                    
                    # Process results
                    for i, key in enumerate(missing_keys):
                        value = redis_values[i]
                        if value is not None:
                            self.stats['redis_hits'] += 1
                            
                            # Try JSON decode
                            try:
                                deserialized = json.loads(value)
                            except:
                                # Fallback to other deserialization
                                deserialized = self._deserialize_value(value, use_json='json:' in str(key))
                            
                            results[key] = deserialized
                            
                            # Cache in memory for faster access
                            memory_expiry = datetime.now() + timedelta(minutes=2)
                            self.memory_cache[key] = (deserialized, memory_expiry)
                        else:
                            self.stats['redis_misses'] += 1
            except Exception as e:
                logger.debug(f"Redis batch get error: {e}")
        
        return results
    
    async def exists(self, key: str) -> bool:
        """Check if a key exists in cache (checks both memory and Redis)"""
        # Check memory cache
        if key in self.memory_cache:
            value, expiry = self.memory_cache[key]
            if expiry is None or datetime.now() < expiry:
                return True
            else:
                # Expired, remove it
                del self.memory_cache[key]
                self.stats['memory_evictions'] += 1
        
        # Check Redis
        if self.redis_enabled and self.redis_client:
            try:
                # Check both prefixed and non-prefixed versions
                exists = await self.redis_client.exists(f"json:{key}", key)
                return exists > 0
            except Exception as e:
                logger.debug(f"Redis exists check error: {e}")
        
        return False
    
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
        redis_total = self.stats['redis_hits'] + self.stats['redis_misses']
        memory_total = self.stats['memory_hits'] + self.stats['memory_misses']
        
        return {
            'redis_enabled': self.redis_enabled,
            'redis_info': redis_info,
            'memory_cache_size': len(self.memory_cache),
            'stats': self.stats.copy(),
            'hit_rates': {
                'redis': self.stats['redis_hits'] / max(redis_total, 1),
                'memory': self.stats['memory_hits'] / max(memory_total, 1),
                'overall': (self.stats['redis_hits'] + self.stats['memory_hits']) / max(redis_total + memory_total, 1)
            },
            'default_poster': self.default_poster
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
                await asyncio.sleep(60)  # Run every 1 minute for faster cleanup
                
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
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(30)  # Wait before retry
    
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
