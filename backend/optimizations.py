"""
optimizations.py - Speed optimizations for SK4FiLM
"""
import asyncio
import time
from functools import wraps, lru_cache
from typing import Dict, Any, Callable, List
import logging

logger = logging.getLogger(__name__)

# ASYNC TIMEOUT DECORATOR
def timeout(seconds=10):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                logger.warning(f"⏰ Function {func.__name__} timed out after {seconds} seconds")
                return None
        return wrapper
    return decorator

# RATE LIMITER
class RateLimiter:
    def __init__(self, calls_per_minute=30):
        self.calls_per_minute = calls_per_minute
        self.calls = []
    
    async def acquire(self):
        now = time.time()
        self.calls = [call for call in self.calls if now - call < 60]
        
        if len(self.calls) >= self.calls_per_minute:
            oldest_call = self.calls[0]
            wait_time = 60 - (now - oldest_call)
            if wait_time > 0:
                logger.debug(f"Rate limit reached, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
                self.calls = []
        
        self.calls.append(time.time())

# CONCURRENT PROCESSING WITH LIMIT
async def process_concurrently(tasks: List, max_concurrent: int = 5):
    """Process tasks concurrently with limit"""
    results = []
    
    for i in range(0, len(tasks), max_concurrent):
        batch = tasks[i:i + max_concurrent]
        batch_results = await asyncio.gather(*batch, return_exceptions=True)
        results.extend(batch_results)
        
        # Small delay between batches to prevent flooding
        if i + max_concurrent < len(tasks):
            await asyncio.sleep(0.1)
    
    return results

# MEMOIZATION FOR SYNC FUNCTIONS
@lru_cache(maxsize=1024)
def memoize_sync(key: str):
    """Memoization for synchronous functions"""
    return None

# ASYNC CACHE DECORATOR
def async_cache(maxsize=128, ttl=300):
    """Async cache decorator with TTL"""
    cache = {}
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key
            key_parts = [func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
            key = ":".join(key_parts)
            
            # Hash long keys
            if len(key) > 100:
                import hashlib
                key = hashlib.md5(key.encode()).hexdigest()
            
            now = time.time()
            
            # Check cache
            if key in cache:
                value, timestamp = cache[key]
                if now - timestamp < ttl:
                    return value
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Store in cache
            cache[key] = (result, now)
            
            # Clean old entries if cache is full
            if len(cache) > maxsize:
                # Remove oldest entry
                oldest_key = min(cache.keys(), key=lambda k: cache[k][1])
                del cache[oldest_key]
            
            return result
        return wrapper
    return decorator

# BATCH PROCESSING
async def process_in_batches(items: List, process_func: Callable, batch_size: int = 10):
    """Process items in batches"""
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_tasks = [process_func(item) for item in batch]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        results.extend(batch_results)
        
        # Small delay between batches
        if i + batch_size < len(items):
            await asyncio.sleep(0.05)
    
    return results

# PERFORMANCE MONITORING
class PerformanceMonitor:
    def __init__(self):
        self.measurements = {}
    
    def measure(self, name: str):
        """Decorator to measure function execution time"""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = await func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                
                if name not in self.measurements:
                    self.measurements[name] = {
                        'count': 0,
                        'total_time': 0,
                        'avg_time': 0
                    }
                
                self.measurements[name]['count'] += 1
                self.measurements[name]['total_time'] += elapsed
                self.measurements[name]['avg_time'] = (
                    self.measurements[name]['total_time'] / self.measurements[name]['count']
                )
                
                if elapsed > 1:  # Log slow operations
                    logger.warning(f"⏱️ {name} took {elapsed:.2f}s")
                
                return result
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                
                if name not in self.measurements:
                    self.measurements[name] = {
                        'count': 0,
                        'total_time': 0,
                        'avg_time': 0
                    }
                
                self.measurements[name]['count'] += 1
                self.measurements[name]['total_time'] += elapsed
                self.measurements[name]['avg_time'] = (
                    self.measurements[name]['total_time'] / self.measurements[name]['count']
                )
                
                if elapsed > 0.5:  # Log slow sync operations
                    logger.warning(f"⏱️ {name} took {elapsed:.2f}s")
                
                return result
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'measurements': self.measurements,
            'total_operations': sum(m['count'] for m in self.measurements.values()),
            'timestamp': time.time()
        }

# Create global performance monitor
performance_monitor = PerformanceMonitor()

# CONNECTION POOL MANAGER
class ConnectionPool:
    def __init__(self, max_connections=10):
        self.max_connections = max_connections
        self.active_connections = 0
        self.queue = asyncio.Queue()
    
    async def acquire(self):
        if self.active_connections < self.max_connections:
            self.active_connections += 1
            return True
        
        # Wait for connection to be available
        await self.queue.put(True)
        return await self.queue.get()
    
    def release(self):
        self.active_connections -= 1
        if not self.queue.empty():
            self.queue.get_nowait()
            self.queue.task_done()

# LAZY LOADING
class LazyLoader:
    def __init__(self, loader_func: Callable):
        self.loader_func = loader_func
        self._loaded = False
        self._value = None
    
    async def get(self):
        if not self._loaded:
            self._value = await self.loader_func()
            self._loaded = True
        return self._value
    
    def clear(self):
        self._loaded = False
        self._value = None
