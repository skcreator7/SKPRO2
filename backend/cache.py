"""
cache.py - Redis and in-memory caching system with optimizations
"""
import asyncio
import json
import pickle
import re
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Union, List, Tuple
import logging
import redis.asyncio as redis
import hashlib
import time

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
            'total_requests': 0,
            'batch_hits': 0,
            'pipeline_saves': 0
        }
        
        # Cache keys for fast access
        self.cache_keys = set()
        
        # Performance tracking
        self.performance_stats = {
            'avg_get_time': 0,
            'avg_set_time': 0,
            'total_get_calls': 0,
            'total_set_calls': 0
        }
        
        # Cleanup task
        self.cleanup_task = None
        
        # Connection pool
        self.connection_pool = None
        
        # Cache warming data
        self.warm_cache_data = {}
    
    def _generate_cache_key(self, prefix: str, *args) -> str:
        """Generate consistent cache key"""
        key_str = f"{prefix}:{':'.join(str(arg) for arg in args)}"
        if len(key_str) > 100:  # If key is too long, hash it
            return f"{prefix}:{hashlib.md5(key_str.encode()).hexdigest()[:16]}"
        return key_str
    
    def _generate_fingerprint(self, data: Any) -> str:
        """Generate fingerprint for data validation"""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()[:8]
    
    async def init_redis(self) -> bool:
        """Initialize Redis connection with optimizations"""
        try:
            # Check if Redis URL is configured
            if not hasattr(self.config, 'REDIS_URL') or not self.config.REDIS_URL:
                logger.warning("⚠️ Redis URL not configured, using memory cache only")
                return False
            
            # Parse Redis URL
            redis_url = self.config.REDIS_URL
            
            # Enhanced connection parameters
            connection_params = {
                'decode_responses': True,
                'encoding': 'utf-8',
                'socket_connect_timeout': 3,  # Reduced from 5
                'socket_timeout': 3,  # Reduced from 5
                'max_connections': 50,  # Increased from 20
                'health_check_interval': 15,  # Reduced from 30
                'retry_on_timeout': True,
                'socket_keepalive': True,
                'socket_keepalive_options': {
                    'TCP_KEEPIDLE': 60,
                    'TCP_KEEPINTVL': 30,
                    'TCP_KEEPCNT': 3
                }
            }
            
            # Handle special Redis URLs
            if 'redislabs.com' in redis_url or 'redislabs' in redis_url:
                # Extract credentials for Redis Labs
                parsed = re.search(r'redis://([^:]+):([^@]+)@([^:]+):(\d+)', redis_url)
                if parsed:
                    username, password, host, port = parsed.groups()
                    self.redis_client = redis.Redis(
                        host=host,
                        port=int(port),
                        username=username,
                        password=password,
                        **connection_params
                    )
                else:
                    self.redis_client = redis.from_url(redis_url, **connection_params)
            else:
                self.redis_client = redis.from_url(redis_url, **connection_params)
            
            # Test connection with timeout
            start_time = time.time()
            try:
                pong = await asyncio.wait_for(self.redis_client.ping(), timeout=2)
                if pong:
                    connection_time = time.time() - start_time
                    logger.info(f"✅ Redis connected in {connection_time:.3f}s")
                    
                    # Enable Redis features
                    self.redis_enabled = True
                    
                    # Configure Redis for better performance
                    await self._configure_redis()
                    
                    # Warm up cache
                    await self._warm_up_cache_async()
                    
                    # Test performance
                    await self._test_performance()
                    
                    return True
                else:
                    logger.warning("⚠️ Redis ping returned False")
                    self.redis_enabled = False
                    return False
                    
            except asyncio.TimeoutError:
                logger.warning("⚠️ Redis connection timeout (2s)")
                self.redis_enabled = False
                return False
            except Exception as e:
                logger.warning(f"⚠️ Redis ping failed: {e}")
                self.redis_enabled = False
                return False
            
        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed: {e}")
            self.redis_enabled = False
            return False
    
    async def _configure_redis(self):
        """Configure Redis for optimal performance"""
        try:
            if self.redis_enabled and self.redis_client:
                # Enable keyspace notifications (optional)
                await self.redis_client.config_set('notify-keyspace-events', 'Ex')
                
                # Set max memory policy
                await self.redis_client.config_set('maxmemory-policy', 'allkeys-lru')
                
                # Enable AOF for persistence (if needed)
                # await self.redis_client.config_set('appendonly', 'yes')
                # await self.redis_client.config_set('appendfsync', 'everysec')
                
                logger.info("✅ Redis configured for optimal performance")
        except Exception as e:
            logger.warning(f"Redis configuration failed: {e}")
    
    async def _warm_up_cache_async(self):
        """Warm up cache with common data asynchronously"""
        try:
            if not self.redis_enabled:
                return
            
            warm_data = {
                "system:status": {"status": "ok", "timestamp": datetime.now().isoformat()},
                "config:channels": {"count": 3, "names": ["Main", "Updates", "Files"]},
                "stats:initial": {"users": 0, "files": 0, "cache_hits": 0},
                "cache:warm": True
            }
            
            # Use pipeline for batch set
            pipe = self.redis_client.pipeline()
            for key, value in warm_data.items():
                serialized = json.dumps(value, separators=(',', ':'))
                pipe.setex(key, 7200, serialized)  # 2 hours
            
            await pipe.execute()
            
            # Store in memory for immediate access
            for key, value in warm_data.items():
                self.memory_cache[key] = (value, datetime.now() + timedelta(hours=2))
            
            logger.info(f"🔥 Redis cache warmed up with {len(warm_data)} items")
            
            # Pre-load common search patterns
            await self._preload_common_searches()
            
        except Exception as e:
            logger.error(f"Cache warm-up error: {e}")
    
    async def _preload_common_searches(self):
        """Pre-load common search results"""
        common_searches = [
            "avatar", "avengers", "spider", "batman", "superman",
            "john wick", "fast furious", "mission impossible",
            "tamil", "hindi", "telugu", "malayalam"
        ]
        
        try:
            for search in common_searches:
                key = self._generate_cache_key("search", search, 1, 12)
                placeholder = {
                    "query": search,
                    "results": [],
                    "cached_at": datetime.now().isoformat(),
                    "placeholder": True
                }
                await self.set(key, placeholder, expire_seconds=3600)
            
            logger.info(f"🔍 Pre-loaded {len(common_searches)} common searches")
        except Exception as e:
            logger.error(f"Search pre-load error: {e}")
    
    async def _test_performance(self):
        """Test cache performance"""
        try:
            test_key = "performance_test"
            test_value = {"test": True, "timestamp": datetime.now().isoformat()}
            
            # Test set performance
            start = time.time()
            await self.set(test_key, test_value, expire_seconds=10)
            set_time = time.time() - start
            
            # Test get performance
            start = time.time()
            await self.get(test_key)
            get_time = time.time() - start
            
            # Test pipeline performance
            start = time.time()
            pipe = self.redis_client.pipeline()
            for i in range(5):
                pipe.set(f"test_{i}", f"value_{i}", ex=5)
            await pipe.execute()
            pipe_time = time.time() - start
            
            logger.info(f"⚡ Performance test - Set: {set_time*1000:.1f}ms, Get: {get_time*1000:.1f}ms, Pipeline: {pipe_time*1000:.1f}ms")
            
        except Exception as e:
            logger.error(f"Performance test error: {e}")
    
    def _serialize_value(self, value: Any) -> str:
        """Optimized serialization with compression detection"""
        try:
            # For simple types, use JSON with minimal formatting
            if isinstance(value, (dict, list, tuple, str, int, float, bool, type(None))):
                return json.dumps(value, separators=(',', ':'), default=str)
            else:
                # For complex objects, use pickle
                return f"pickle:{base64.b64encode(pickle.dumps(value)).decode('utf-8')}"
        except Exception:
            # Fallback
            return str(value)
    
    def _deserialize_value(self, value: str) -> Any:
        """Optimized deserialization"""
        if value is None:
            return None
        
        try:
            if value.startswith('pickle:'):
                import base64
                pickled_data = value[7:]  # Remove 'pickle:' prefix
                return pickle.loads(base64.b64decode(pickled_data))
            else:
                return json.loads(value)
        except json.JSONDecodeError:
            try:
                import base64
                return pickle.loads(base64.b64decode(value))
            except:
                return value
        except Exception:
            return value
    
    async def get(self, key: str, use_redis: bool = True) -> Optional[Any]:
        """Get value from cache with enhanced performance"""
        start_time = time.time()
        self.stats['total_requests'] += 1
        
        # Try memory cache first (fastest - O(1))
        if key in self.memory_cache:
            value, expiry = self.memory_cache[key]
            if expiry is None or datetime.now() < expiry:
                self.stats['memory_hits'] += 1
                
                # Update performance stats
                self._update_performance_stats('get', time.time() - start_time)
                
                return value
            else:
                # Expired, remove from memory
                del self.memory_cache[key]
                self.stats['memory_evictions'] += 1
        
        self.stats['memory_misses'] += 1
        
        # Try Redis if enabled
        if use_redis and self.redis_enabled and self.redis_client:
            try:
                # Add fingerprint for validation
                validation_key = f"{key}:fingerprint"
                
                # Get both value and fingerprint in pipeline
                pipe = self.redis_client.pipeline()
                pipe.get(key)
                pipe.get(validation_key)
                results = await pipe.execute()
                
                value_str, fingerprint = results
                
                if value_str is not None:
                    self.stats['redis_hits'] += 1
                    
                    # Deserialize
                    deserialized = self._deserialize_value(value_str)
                    
                    # Verify fingerprint if available
                    if fingerprint and hasattr(deserialized, '__dict__'):
                        current_fp = self._generate_fingerprint(deserialized.__dict__)
                        if fingerprint != current_fp:
                            logger.warning(f"⚠️ Cache fingerprint mismatch for {key}")
                            # Data may be corrupted, don't cache in memory
                            self._update_performance_stats('get', time.time() - start_time)
                            return deserialized
                    
                    # Store in memory cache for faster access (with shorter TTL)
                    memory_expiry = datetime.now() + timedelta(minutes=3)
                    self.memory_cache[key] = (deserialized, memory_expiry)
                    
                    # Update performance stats
                    self._update_performance_stats('get', time.time() - start_time)
                    
                    return deserialized
                else:
                    self.stats['redis_misses'] += 1
                    
                    # Update performance stats
                    self._update_performance_stats('get', time.time() - start_time)
                    
            except Exception as e:
                logger.warning(f"Redis get error: {e}")
                self._update_performance_stats('get', time.time() - start_time)
        
        # Update performance stats for miss
        self._update_performance_stats('get', time.time() - start_time)
        
        return None
    
    def _update_performance_stats(self, operation: str, duration: float):
        """Update performance statistics"""
        if operation == 'get':
            self.performance_stats['total_get_calls'] += 1
            old_avg = self.performance_stats['avg_get_time']
            count = self.performance_stats['total_get_calls']
            self.performance_stats['avg_get_time'] = old_avg + (duration - old_avg) / count
        elif operation == 'set':
            self.performance_stats['total_set_calls'] += 1
            old_avg = self.performance_stats['avg_set_time']
            count = self.performance_stats['total_set_calls']
            self.performance_stats['avg_set_time'] = old_avg + (duration - old_avg) / count
    
    async def set(self, key: str, value: Any, expire_seconds: int = 3600, 
                  use_redis: bool = True) -> bool:
        """Set value in cache with optimizations"""
        start_time = time.time()
        
        # Store in memory
        expiry = datetime.now() + timedelta(seconds=expire_seconds) if expire_seconds > 0 else None
        self.memory_cache[key] = (value, expiry)
        
        # Generate fingerprint for complex objects
        fingerprint = None
        if hasattr(value, '__dict__'):
            fingerprint = self._generate_fingerprint(value.__dict__)
        
        # Store in Redis if enabled
        if use_redis and self.redis_enabled and self.redis_client:
            try:
                serialized = self._serialize_value(value)
                
                # Use pipeline for atomic operations
                pipe = self.redis_client.pipeline()
                pipe.setex(key, expire_seconds, serialized)
                
                # Store fingerprint if available
                if fingerprint:
                    pipe.setex(f"{key}:fingerprint", expire_seconds, fingerprint)
                
                await pipe.execute()
                
                self.stats['pipeline_saves'] += 1
                
                # Track key
                self.cache_keys.add(key)
                
                # Update performance stats
                self._update_performance_stats('set', time.time() - start_time)
                
                return True
            except Exception as e:
                logger.warning(f"Redis set error: {e}")
                self._update_performance_stats('set', time.time() - start_time)
                return False
        
        self._update_performance_stats('set', time.time() - start_time)
        return True
    
    async def get_or_set(self, key: str, coroutine, expire_seconds: int = 3600, 
                        force_refresh: bool = False) -> Any:
        """Get from cache or set using coroutine (cache-aside pattern)"""
        if not force_refresh:
            cached = await self.get(key)
            if cached is not None:
                return cached
        
        # Not in cache or force refresh, execute coroutine
        value = await coroutine()
        await self.set(key, value, expire_seconds)
        return value
    
    async def cache_search_results(self, query: str, page: int, limit: int, results: Dict[str, Any]) -> bool:
        """Cache search results with optimized key and compression"""
        cache_key = self._generate_cache_key("search", query.lower(), page, limit)
        
        # Add metadata
        results['_cache_meta'] = {
            'cached_at': datetime.now().isoformat(),
            'query': query,
            'page': page,
            'limit': limit,
            'results_count': len(results.get('results', []))
        }
        
        return await self.set(cache_key, results, expire_seconds=1800)  # 30 minutes
    
    async def get_search_results(self, query: str, page: int, limit: int) -> Optional[Dict[str, Any]]:
        """Get cached search results with validation"""
        cache_key = self._generate_cache_key("search", query.lower(), page, limit)
        results = await self.get(cache_key)
        
        if results and isinstance(results, dict):
            # Check if cache is stale (older than 25 minutes)
            cache_meta = results.get('_cache_meta', {})
            cached_at_str = cache_meta.get('cached_at')
            
            if cached_at_str:
                try:
                    cached_at = datetime.fromisoformat(cached_at_str.replace('Z', '+00:00'))
                    age_minutes = (datetime.now() - cached_at).total_seconds() / 60
                    
                    # Return stale cache but mark for refresh
                    if age_minutes > 25:
                        results['_cache_stale'] = True
                except:
                    pass
            
            return results
        
        return None
    
    async def batch_get(self, keys: List[str]) -> Dict[str, Any]:
        """Batch get multiple keys with optimization"""
        start_time = time.time()
        results = {}
        
        if not keys:
            return results
        
        # Check memory cache first
        memory_keys = []
        for key in keys:
            if key in self.memory_cache:
                value, expiry = self.memory_cache[key]
                if expiry is None or datetime.now() < expiry:
                    results[key] = value
                else:
                    # Expired
                    del self.memory_cache[key]
                    memory_keys.append(key)
            else:
                memory_keys.append(key)
        
        # Check remaining keys in Redis
        if memory_keys and self.redis_enabled and self.redis_client:
            try:
                # Use MGET for batch retrieval
                redis_results = await self.redis_client.mget(memory_keys)
                
                for key, value_str in zip(memory_keys, redis_results):
                    if value_str is not None:
                        deserialized = self._deserialize_value(value_str)
                        results[key] = deserialized
                        
                        # Store in memory cache
                        self.memory_cache[key] = (deserialized, datetime.now() + timedelta(minutes=2))
                
                self.stats['batch_hits'] += 1
                
            except Exception as e:
                logger.warning(f"Redis batch get error: {e}")
        
        # Log performance
        batch_time = time.time() - start_time
        if batch_time > 0.1:  # Log slow batch operations
            logger.debug(f"Batch get {len(keys)} keys took {batch_time:.3f}s")
        
        return results
    
    async def batch_set(self, items: Dict[str, Any], expire_seconds: int = 3600):
        """Batch set multiple items with pipeline"""
        if not items:
            return
        
        start_time = time.time()
        
        # Store in memory cache
        expiry = datetime.now() + timedelta(seconds=expire_seconds) if expire_seconds > 0 else None
        for key, value in items.items():
            self.memory_cache[key] = (value, expiry)
        
        # Store in Redis if enabled
        if self.redis_enabled and self.redis_client:
            try:
                pipe = self.redis_client.pipeline()
                
                for key, value in items.items():
                    serialized = self._serialize_value(value)
                    pipe.setex(key, expire_seconds, serialized)
                
                await pipe.execute()
                
                # Track keys
                self.cache_keys.update(items.keys())
                
                self.stats['pipeline_saves'] += 1
                
            except Exception as e:
                logger.warning(f"Redis batch set error: {e}")
        
        batch_time = time.time() - start_time
        if batch_time > 0.1:  # Log slow batch operations
            logger.debug(f"Batch set {len(items)} items took {batch_time:.3f}s")
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache keys matching pattern"""
        invalidated = 0
        
        # Invalidate memory cache
        keys_to_delete = [k for k in self.memory_cache.keys() if pattern in k]
        for key in keys_to_delete:
            del self.memory_cache[key]
            invalidated += 1
        
        # Invalidate Redis cache
        if self.redis_enabled and self.redis_client:
            try:
                # Find keys matching pattern
                keys = await self.redis_client.keys(f"*{pattern}*")
                if keys:
                    await self.redis_client.delete(*keys)
                    invalidated += len(keys)
                    
                    # Also delete fingerprints
                    fingerprint_keys = [f"{k}:fingerprint" for k in keys]
                    await self.redis_client.delete(*fingerprint_keys)
            except Exception as e:
                logger.warning(f"Redis pattern invalidation error: {e}")
        
        logger.info(f"🧹 Invalidated {invalidated} keys matching pattern: {pattern}")
        return invalidated
    
    async def clear_search_cache(self) -> int:
        """Clear all search-related cache"""
        return await self.invalidate_pattern("search:")
    
    async def get_stats_summary(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        redis_info = {}
        if self.redis_enabled and self.redis_client:
            try:
                info = await self.redis_client.info()
                redis_info = {
                    'connected_clients': info.get('connected_clients', 0),
                    'used_memory': info.get('used_memory_human', '0'),
                    'used_memory_peak': info.get('used_memory_peak_human', '0'),
                    'total_commands_processed': info.get('total_commands_processed', 0),
                    'keyspace_hits': info.get('keyspace_hits', 0),
                    'keyspace_misses': info.get('keyspace_misses', 0),
                    'hit_rate': info.get('keyspace_hit_rate', 0),
                    'uptime_days': info.get('uptime_in_days', 0)
                }
            except:
                pass
        
        # Calculate hit rates
        total_hits = self.stats['redis_hits'] + self.stats['memory_hits']
        total_misses = self.stats['redis_misses'] + self.stats['memory_misses']
        total_requests = total_hits + total_misses
        
        hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0
        
        # Memory usage estimate (rough)
        import sys
        memory_usage = sum(sys.getsizeof(str(k)) + sys.getsizeof(str(v)) 
                          for k, (v, _) in self.memory_cache.items())
        
        return {
            'redis_enabled': self.redis_enabled,
            'redis_info': redis_info,
            'memory_cache': {
                'size': len(self.memory_cache),
                'estimated_memory': f"{memory_usage / 1024 / 1024:.2f} MB",
                'hit_rate': f"{(self.stats['memory_hits'] / max(self.stats['memory_hits'] + self.stats['memory_misses'], 1)) * 100:.1f}%"
            },
            'cache_keys_count': len(self.cache_keys),
            'stats': self.stats.copy(),
            'performance': {
                'avg_get_time_ms': f"{self.performance_stats['avg_get_time'] * 1000:.2f}",
                'avg_set_time_ms': f"{self.performance_stats['avg_set_time'] * 1000:.2f}",
                'total_get_calls': self.performance_stats['total_get_calls'],
                'total_set_calls': self.performance_stats['total_set_calls']
            },
            'overall_hit_rate': f"{hit_rate:.1f}%",
            'total_requests': self.stats['total_requests'],
            'timestamp': datetime.now().isoformat()
        }
    
    async def clear_all(self) -> bool:
        """Clear all cache with confirmation"""
        try:
            logger.info("🧹 Clearing all cache...")
            
            # Clear memory cache
            memory_size = len(self.memory_cache)
            self.memory_cache.clear()
            
            # Clear Redis
            redis_size = 0
            if self.redis_enabled and self.redis_client:
                await self.redis_client.flushdb()
                redis_size = await self.redis_client.dbsize()
            
            # Reset stats
            self.stats = {
                'redis_hits': 0,
                'redis_misses': 0,
                'memory_hits': 0,
                'memory_misses': 0,
                'memory_evictions': 0,
                'total_requests': 0,
                'batch_hits': 0,
                'pipeline_saves': 0
            }
            
            # Reset performance stats
            self.performance_stats = {
                'avg_get_time': 0,
                'avg_set_time': 0,
                'total_get_calls': 0,
                'total_set_calls': 0
            }
            
            # Clear key tracking
            self.cache_keys.clear()
            
            logger.info(f"✅ All cache cleared - Memory: {memory_size} items, Redis: {redis_size} keys")
            return True
            
        except Exception as e:
            logger.error(f"❌ Clear all cache error: {e}")
            return False
    
    async def start_cleanup_task(self):
        """Start background cleanup task for memory cache"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("🧹 Cache cleanup task started")
    
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
                    
                # Optional: Cleanup old Redis keys (run less frequently)
                if self.redis_enabled and self.redis_client:
                    # This is resource-intensive, run occasionally
                    import random
                    if random.random() < 0.1:  # 10% chance per run
                        await self._cleanup_redis_idle_keys()
                    
            except asyncio.CancelledError:
                logger.info("🧹 Cache cleanup task cancelled")
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
    
    async def _cleanup_redis_idle_keys(self):
        """Cleanup Redis keys that are idle for too long"""
        try:
            # This is a simple implementation
            # In production, you might want more sophisticated logic
            cursor = 0
            idle_keys = []
            
            # Scan for keys (this can be heavy, use with caution)
            while True:
                cursor, keys = await self.redis_client.scan(cursor=cursor, count=100)
                
                for key in keys:
                    # Check idle time (requires Redis 3.2+)
                    try:
                        idle_time = await self.redis_client.object("idletime", key)
                        if idle_time > 86400:  # 24 hours idle
                            idle_keys.append(key)
                    except:
                        pass
                
                if cursor == 0:
                    break
            
            # Delete idle keys (limit to 1000 per run)
            if idle_keys:
                keys_to_delete = idle_keys[:1000]
                await self.redis_client.delete(*keys_to_delete)
                logger.info(f"🧹 Cleaned {len(keys_to_delete)} idle Redis keys")
                
        except Exception as e:
            logger.warning(f"Redis idle cleanup error: {e}")
    
    async def stop(self):
        """Stop cache manager gracefully"""
        logger.info("🛑 Stopping cache manager...")
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
            logger.info("✅ Cleanup task stopped")
        
        if self.redis_client:
            await self.redis_client.close()
            logger.info("✅ Redis connection closed")
        
        # Save final stats
        stats = await self.get_stats_summary()
        logger.info(f"📊 Final cache stats: {stats['overall_hit_rate']} hit rate, {stats['total_requests']} total requests")
        
        logger.info("✅ Cache manager stopped gracefully")
