"""
app.py - Complete SK4FiLM Web API System with Multi-Channel Search
OPTIMIZED: Supersonic speed with caching, async optimization, and performance monitoring
"""
import asyncio
import os
import logging
import json
import re
import math
import html
import time
import secrets
import base64
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import defaultdict
from functools import lru_cache, wraps
import hashlib

import aiohttp
import urllib.parse
from quart import Quart, jsonify, request, Response
from hypercorn.asyncio import serve
from hypercorn.config import Config as HyperConfig
from motor.motor_asyncio import AsyncIOMotorClient
import redis.asyncio as redis

# Pyrogram imports
try:
    from pyrogram import Client
    from pyrogram.errors import FloodWait, SessionPasswordNeeded
    PYROGRAM_AVAILABLE = True
except ImportError:
    PYROGRAM_AVAILABLE = False
    Client = None
    FloodWait = None
    SessionPasswordNeeded = None

# Import modular components
from cache import CacheManager
from verification import VerificationSystem
from premium import PremiumSystem, PremiumTier
from poster_fetching import PosterFetcher, PosterSource

# Import shared utilities
from utils import (
    normalize_title,
    extract_title_smart,
    extract_title_from_file,
    format_size,
    detect_quality,
    is_video_file,
    format_post,
    is_new
)

# Import bot_handlers AFTER all other imports
try:
    from bot_handlers import SK4FiLMBot, setup_bot_handlers
    BOT_HANDLERS_AVAILABLE = True
except ImportError:
    BOT_HANDLERS_AVAILABLE = False
    SK4FiLMBot = None
    setup_bot_handlers = None

# ✅ ULTRA-FAST LOADING OPTIMIZATIONS
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Reduce log noise
logging.getLogger('asyncio').setLevel(logging.WARNING)
logging.getLogger('pyrogram').setLevel(logging.WARNING)
logging.getLogger('hypercorn').setLevel(logging.WARNING)
logging.getLogger('motor').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

# Performance monitoring
class PerformanceMonitor:
    def __init__(self):
        self.measurements = {}
        self.request_times = {}
    
    def measure(self, name):
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = await func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                
                self._record(name, elapsed)
                return result
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                
                self._record(name, elapsed)
                return result
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    def _record(self, name, elapsed):
        if name not in self.measurements:
            self.measurements[name] = {
                'count': 0,
                'total': 0,
                'avg': 0,
                'max': 0,
                'min': float('inf')
            }
        
        stats = self.measurements[name]
        stats['count'] += 1
        stats['total'] += elapsed
        stats['avg'] = stats['total'] / stats['count']
        stats['max'] = max(stats['max'], elapsed)
        stats['min'] = min(stats['min'], elapsed)
        
        # Log slow operations
        if elapsed > 0.5:
            logger.warning(f"⏱️ {name} took {elapsed:.3f}s")
    
    def get_stats(self):
        return self.measurements

performance_monitor = PerformanceMonitor()

# Configuration with optimizations
class Config:
    # API Configuration
    API_ID = int(os.environ.get("API_ID", "0"))
    API_HASH = os.environ.get("API_HASH", "")
    USER_SESSION_STRING = os.environ.get("USER_SESSION_STRING", "")
    BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
    
    # Database Configuration
    MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
    REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
    REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "")
    
    # Channel Configuration - OPTIMIZED
    MAIN_CHANNEL_ID = -1001891090100  # ✅ यहाँ से 30 movies fetch करेंगे
    TEXT_CHANNEL_IDS = [-1001891090100, -1002024811395]  # ✅ दोनों channels के posts दिखाएंगे
    FILE_CHANNEL_ID = -1001768249569
    
    # Links
    MAIN_CHANNEL_LINK = "https://t.me/sk4film"
    UPDATES_CHANNEL_LINK = "https://t.me/sk4film_Request"
    CHANNEL_USERNAME = "sk4film"
    
    # URL Shortener
    SHORTLINK_API = os.environ.get("SHORTLINK_API", "")
    CUTTLY_API = os.environ.get("CUTTLY_API", "")
    
    # UPI IDs
    UPI_ID_BASIC = os.environ.get("UPI_ID_BASIC", "sk4filmbot@ybl")
    UPI_ID_PREMIUM = os.environ.get("UPI_ID_PREMIUM", "sk4filmbot@ybl")
    UPI_ID_GOLD = os.environ.get("UPI_ID_GOLD", "sk4filmbot@ybl")
    UPI_ID_DIAMOND = os.environ.get("UPI_ID_DIAMOND", "sk4filmbot@ybl")
    
    # Verification
    VERIFICATION_REQUIRED = os.environ.get("VERIFICATION_REQUIRED", "true").lower() == "true"
    VERIFICATION_DURATION = 6 * 60 * 60  # 6 hours
    
    # Application
    WEBSITE_URL = os.environ.get("WEBSITE_URL", "https://sk4film.vercel.app")
    BOT_USERNAME = os.environ.get("BOT_USERNAME", "sk4filmbot")
    ADMIN_IDS = [int(x) for x in os.environ.get("ADMIN_IDS", "123456789").split(",")]
    AUTO_DELETE_TIME = int(os.environ.get("AUTO_DELETE_TIME", "300"))
    WEB_SERVER_PORT = int(os.environ.get("PORT", 8000))
    BACKEND_URL = os.environ.get("BACKEND_URL", "https://sk4film.koyeb.app")
    
    # API Keys
    OMDB_KEYS = ["8265bd1c", "b9bd48a6", "3e7e1e9d"]
    TMDB_KEYS = ["e547e17d4e91f3e62a571655cd1ccaff", "8265bd1f"]
    
    # Performance Settings
    MAX_CONCURRENT_REQUESTS = int(os.environ.get("MAX_CONCURRENT_REQUESTS", "50"))
    CACHE_TTL = int(os.environ.get("CACHE_TTL", "300"))
    REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "10"))

# FAST INITIALIZATION
app = Quart(__name__)
app.config['JSON_SORT_KEYS'] = False  # Faster JSON serialization
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False  # Disable pretty printing for speed

# CORS headers
@app.after_request
async def add_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['X-SK4FiLM-Version'] = '8.0-OPTIMIZED'
    response.headers['X-Response-Time'] = f"{time.perf_counter():.3f}"
    return response

# GLOBAL VARIABLES - FAST ACCESS
mongo_client = None
db = None
files_col = None
verification_col = None

# MODULAR COMPONENTS
cache_manager = None
verification_system = None
premium_system = None
poster_fetcher = None
sk4film_bot = None

# Telegram clients
User = None
bot = None
bot_started = False
user_session_ready = False

# CHANNEL CONFIGURATION - CACHED
CHANNEL_CONFIG = {
    -1001891090100: {'name': 'SK4FiLM Main', 'type': 'text', 'search_priority': 1},
    -1002024811395: {'name': 'SK4FiLM Updates', 'type': 'text', 'search_priority': 2},
    -1001768249569: {'name': 'SK4FiLM Files', 'type': 'file', 'search_priority': 0}
}

# CACHED FUNCTIONS
@lru_cache(maxsize=10000)
def channel_name_cached(cid):
    """Ultra-fast cached channel name lookup"""
    return CHANNEL_CONFIG.get(cid, {}).get('name', f"Channel {cid}")

@lru_cache(maxsize=5000)
def normalize_title_cached(title: str) -> str:
    """Cached title normalization"""
    return normalize_title(title)

# FIXED: Missing channel_name function
@lru_cache(maxsize=1000)
def channel_name(channel_id):
    """Get channel name from channel ID (compatibility function)"""
    return channel_name_cached(channel_id)

# FIXED: Missing get_telegram_video_thumbnail function
async def get_telegram_video_thumbnail(user_client, channel_id, message_id):
    """Get video thumbnail from Telegram"""
    try:
        if not user_client or not user_session_ready:
            return None
            
        msg = await safe_telegram_operation(
            user_client.get_messages,
            channel_id,
            message_id
        )
        
        if msg and (msg.video or msg.document):
            return await extract_video_thumbnail(user_client, msg)
        
        return None
    except Exception as e:
        logger.error(f"Get thumbnail error: {e}")
        return None

# FIXED: Missing verify_user_api function
async def verify_user_api(user_id, verification_url=None):
    """Verify user API endpoint implementation"""
    try:
        if verification_system:
            result = await verification_system.verify_user_api(user_id, verification_url)
            return result
        
        # Fallback
        return {
            'verified': True,
            'user_id': user_id,
            'expires_at': (datetime.now() + timedelta(hours=6)).isoformat()
        }
    except Exception as e:
        logger.error(f"Verify user API error: {e}")
        return {
            'verified': False,
            'error': str(e)
        }

# FIXED: Missing get_index_status_api function
async def get_index_status_api():
    """Get indexing status API implementation"""
    try:
        total_files = await files_col.count_documents({}) if files_col is not None else 0
        video_files = await files_col.count_documents({'is_video_file': True}) if files_col is not None else 0
        
        return {
            'indexed_files': total_files,
            'video_files': video_files,
            'user_session_ready': user_session_ready,
            'last_update': datetime.now().isoformat(),
            'status': 'active' if user_session_ready else 'inactive'
        }
    except Exception as e:
        logger.error(f"Index status API error: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }

# ASYNC CACHE DECORATOR
def async_cache_with_ttl(maxsize=128, ttl=300):
    """Async cache decorator with TTL"""
    cache = {}
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key
            key_parts = [func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
            key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
            
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
            
            # Clean old entries
            if len(cache) > maxsize:
                oldest_key = min(cache.keys(), key=lambda k: cache[k][1])
                del cache[oldest_key]
            
            return result
        return wrapper
    return decorator

# OPTIMIZED FLOOD PROTECTION
class TurboFloodProtection:
    def __init__(self):
        self.request_buckets = {}
        self.last_cleanup = time.time()
        self.request_count = 0
    
    async def wait_if_needed(self, user_id=None):
        """Optimized flood protection"""
        current_time = time.time()
        
        # Cleanup old buckets every 30 seconds
        if current_time - self.last_cleanup > 30:
            self._cleanup_buckets(current_time)
            self.last_cleanup = current_time
        
        # Global rate limiting
        bucket_key = int(current_time // 10)  # 10-second buckets
        
        if bucket_key not in self.request_buckets:
            self.request_buckets[bucket_key] = 0
        
        # Allow 50 requests per 10-second bucket
        if self.request_buckets[bucket_key] >= 50:
            wait_time = 10 - (current_time % 10)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                # Reset after wait
                self.request_buckets = {}
        
        self.request_buckets[bucket_key] += 1
        self.request_count += 1
    
    def _cleanup_buckets(self, current_time):
        """Cleanup old buckets"""
        current_bucket = int(current_time // 10)
        old_buckets = [k for k in self.request_buckets.keys() if k < current_bucket - 6]  # Keep last minute
        for bucket in old_buckets:
            del self.request_buckets[bucket]

turbo_flood_protection = TurboFloodProtection()

# ASYNC CONNECTION POOL
class ConnectionPool:
    def __init__(self, max_connections=20):
        self.max_connections = max_connections
        self.semaphore = asyncio.Semaphore(max_connections)
        self.active_connections = 0
    
    async def acquire(self):
        await self.semaphore.acquire()
        self.active_connections += 1
    
    def release(self):
        self.active_connections -= 1
        self.semaphore.release()

# Create connection pool for external requests
http_pool = ConnectionPool(max_connections=Config.MAX_CONCURRENT_REQUESTS)

# SAFE TELEGRAM OPERATIONS WITH TURBO PROTECTION
@performance_monitor.measure("telegram_operation")
async def safe_telegram_operation(operation, *args, **kwargs):
    """Turbo-charged Telegram operations"""
    max_retries = 2
    
    for attempt in range(max_retries):
        try:
            await turbo_flood_protection.wait_if_needed()
            result = await operation(*args, **kwargs)
            return result
        except Exception as e:
            if "FloodWait" in str(e):
                wait_match = re.search(r'(\d+)', str(e))
                if wait_match:
                    wait_time = int(wait_match.group(1))
                    logger.warning(f"⏳ Flood wait {wait_time}s")
                    await asyncio.sleep(wait_time + 1)
                    continue
            logger.error(f"Telegram operation failed: {e}")
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(0.5 * (2 ** attempt))
    
    return None

# SAFE ASYNC ITERATOR
async def safe_telegram_generator(operation, *args, limit=None, **kwargs):
    """Optimized Telegram async generator"""
    count = 0
    
    async for item in operation(*args, **kwargs):
        yield item
        count += 1
        
        # Small delay every 10 items
        if count % 10 == 0:
            await asyncio.sleep(0.1)
            await turbo_flood_protection.wait_if_needed()
        
        if limit and count >= limit:
            break

# CACHE WARM-UP
async def warm_up_cache():
    """Warm up all caches for instant response"""
    try:
        logger.info("🔥 Warming up caches...")
        
        if cache_manager and cache_manager.redis_enabled:
            # Warm up Redis with common data
            warm_data = {
                "system:warm": True,
                "startup_time": datetime.now().isoformat(),
                "version": "8.0-OPTIMIZED"
            }
            
            await cache_manager.batch_set(warm_data, expire_seconds=3600)
        
        if poster_fetcher:
            # Warm up poster cache with popular movies
            popular_movies = [
                "Avatar", "Avengers Endgame", "Spider-Man", "Batman",
                "John Wick 4", "Fast X", "Mission Impossible"
            ]
            
            tasks = []
            for movie in popular_movies[:3]:  # Only warm up 3
                tasks.append(poster_fetcher.fetch_poster(movie))
            
            await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("✅ Cache warm-up complete")
        
    except Exception as e:
        logger.error(f"Cache warm-up error: {e}")

# CACHE CLEANUP TASK
async def cache_cleanup():
    """Background cache cleanup"""
    while True:
        await asyncio.sleep(1800)  # Run every 30 minutes
        try:
            if poster_fetcher:
                await poster_fetcher.cleanup_expired_cache()
            
            if cache_manager:
                await cache_manager.clear_pattern("temp:")
            
            logger.debug("🧹 Cache cleanup completed")
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")

# OPTIMIZED TITLE EXTRACTION
@async_cache_with_ttl(maxsize=1000, ttl=3600)
async def extract_title_from_telegram_msg_cached(msg):
    """Cached title extraction"""
    try:
        caption = msg.caption if hasattr(msg, 'caption') else None
        file_name = None
        
        if msg.document:
            file_name = msg.document.file_name
        elif msg.video:
            file_name = msg.video.file_name
        
        return extract_title_from_file(file_name, caption)
    except Exception as e:
        logger.error(f"Title extraction error: {e}")
        return None

# VIDEO THUMBNAIL PROCESSING
@performance_monitor.measure("thumbnail_extraction")
async def extract_video_thumbnail(user_client, message):
    """Optimized thumbnail extraction"""
    try:
        if message.video:
            thumbnail = message.video.thumbs[0] if message.video.thumbs else None
            if thumbnail:
                thumbnail_path = await safe_telegram_operation(
                    user_client.download_media, 
                    thumbnail.file_id, 
                    in_memory=True
                )
                if thumbnail_path:
                    thumbnail_data = base64.b64encode(thumbnail_path.getvalue()).decode('utf-8')
                    return f"data:image/jpeg;base64,{thumbnail_data}"
        
        return None
    except Exception as e:
        logger.error(f"Thumbnail extraction failed: {e}")
        return None

# MONGODB INITIALIZATION WITH OPTIMIZATIONS
@performance_monitor.measure("mongodb_init")
async def init_mongodb():
    """Optimized MongoDB initialization"""
    global mongo_client, db, files_col, verification_col
    
    try:
        logger.info("🔌 MongoDB initialization...")
        
        # Connection with optimized parameters
        mongo_client = AsyncIOMotorClient(
            Config.MONGODB_URI,
            serverSelectionTimeoutMS=3000,  # Reduced from 5000
            connectTimeoutMS=3000,
            socketTimeoutMS=3000,
            maxPoolSize=20,  # Reduced from 50
            minPoolSize=5,
            maxIdleTimeMS=30000,
            retryWrites=True,
            retryReads=True
        )
        
        # Quick ping test
        await asyncio.wait_for(mongo_client.admin.command('ping'), timeout=2)
        
        db = mongo_client.sk4film
        files_col = db.files
        verification_col = db.verifications
        
        # Create indexes only if they don't exist (optimized)
        existing_indexes = await files_col.index_information()
        
        required_indexes = [
            ('title', 'text'),
            ('normalized_title', 1),
            ('message_id', 1),
            ('channel_id', 1),
            ('indexed_at', -1),
            ('is_video_file', 1),
            ('thumbnail', 1)
        ]
        
        for field, direction in required_indexes:
            index_name = f"{field}_{direction if isinstance(direction, int) else 'text'}"
            if index_name not in existing_indexes:
                try:
                    if direction == 'text':
                        await files_col.create_index([(field, 'text')])
                    else:
                        await files_col.create_index([(field, direction)])
                    logger.debug(f"✅ Created index: {index_name}")
                except Exception as e:
                    logger.warning(f"Index creation failed for {index_name}: {e}")
        
        logger.info("✅ MongoDB OK - Optimized and Ready")
        return True
        
    except asyncio.TimeoutError:
        logger.error("❌ MongoDB connection timeout")
        return False
    except Exception as e:
        logger.error(f"❌ MongoDB: {e}")
        return False

# FILE INDEXING WITH PERFORMANCE
@performance_monitor.measure("file_indexing")
async def index_single_file(message):
    """Optimized single file indexing"""
    try:
        if files_col is None:
            return False
        
        # Quick existence check
        existing = await files_col.find_one({
            'channel_id': Config.FILE_CHANNEL_ID,
            'message_id': message.id
        }, {'_id': 1})
        
        if existing:
            return True
        
        # Extract title
        title = await extract_title_from_telegram_msg_cached(message)
        if not title:
            return False
        
        # Prepare document efficiently
        doc = {
            'channel_id': Config.FILE_CHANNEL_ID,
            'message_id': message.id,
            'title': title,
            'normalized_title': normalize_title_cached(title),
            'date': message.date,
            'indexed_at': datetime.now(),
            'is_video_file': False,
            'thumbnail': None,
            'thumbnail_source': 'none'
        }
        
        # Add file-specific data
        if message.document:
            doc.update({
                'file_id': message.document.file_id,
                'file_size': message.document.file_size or 0,
                'file_name': message.document.file_name or '',
                'quality': detect_quality(message.document.file_name or ''),
                'is_video_file': is_video_file(message.document.file_name or ''),
                'caption': message.caption or ''
            })
        elif message.video:
            doc.update({
                'file_id': message.video.file_id,
                'file_size': message.video.file_size or 0,
                'file_name': message.video.file_name or 'video.mp4',
                'quality': detect_quality(message.video.file_name or ''),
                'is_video_file': True,
                'caption': message.caption or ''
            })
        else:
            return False
        
        # Extract thumbnail if video
        if doc['is_video_file'] and User and user_session_ready:
            thumbnail_url = await extract_video_thumbnail(User, message)
            if thumbnail_url:
                doc['thumbnail'] = thumbnail_url
                doc['thumbnail_source'] = 'video_direct'
        
        # Upsert with atomic operation
        await files_col.update_one(
            {
                'channel_id': Config.FILE_CHANNEL_ID,
                'message_id': message.id
            },
            {'$set': doc},
            upsert=True
        )
        
        return True
        
    except Exception as e:
        logger.error(f"Error indexing file: {e}")
        return False

# BACKGROUND INDEXING
async def index_files_background():
    """Optimized background indexing"""
    if not User or files_col is None or not user_session_ready:
        logger.warning("⚠️ Cannot start indexing - User session not ready")
        return
    
    logger.info("📁 Starting OPTIMIZED background indexing...")
    
    try:
        # Get the last indexed message ID
        last_indexed = await files_col.find_one(
            {}, 
            sort=[('message_id', -1)],
            projection={'message_id': 1}
        )
        
        last_message_id = last_indexed['message_id'] if last_indexed else 0
        logger.info(f"🔄 Starting from message ID: {last_message_id}")
        
        total_indexed = 0
        batch_size = 20
        batch = []
        
        async for msg in safe_telegram_generator(User.get_chat_history, Config.FILE_CHANNEL_ID):
            if msg.id <= last_message_id:
                break
            
            if msg and (msg.document or msg.video):
                batch.append(msg)
                total_indexed += 1
                
                # Process in batches
                if len(batch) >= batch_size:
                    tasks = [index_single_file(m) for m in batch]
                    await asyncio.gather(*tasks, return_exceptions=True)
                    batch = []
                    
                    logger.info(f"📥 Indexed {total_indexed} new files...")
                    
                    # Small delay between batches
                    await asyncio.sleep(1)
        
        # Process remaining batch
        if batch:
            tasks = [index_single_file(m) for m in batch]
            await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info(f"✅ Background indexing finished: {total_indexed} NEW files")
        
        # Clear search cache after indexing
        if cache_manager:
            await cache_manager.clear_search_cache()
            logger.info("🧹 Search cache cleared after indexing")
        
    except Exception as e:
        logger.error(f"❌ Background indexing error: {e}")

# POSTER FETCHING WITH CACHE
@performance_monitor.measure("poster_fetch")
async def get_poster_guaranteed(title):
    """Optimized poster fetching"""
    if poster_fetcher:
        return await poster_fetcher.fetch_poster(title)
    
    # Fallback
    year_match = re.search(r'\b(19|20)\d{2}\b', title)
    year = year_match.group() if year_match else ""
    
    return {
        'poster_url': f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(title)}&year={year}",
        'source': PosterSource.CUSTOM.value,
        'rating': '0.0',
        'year': year,
        'title': title
    }

# ULTRA-FAST SEARCH FUNCTION
@performance_monitor.measure("multi_channel_search")
@async_cache_with_ttl(maxsize=500, ttl=300)
async def search_movies_multi_channel(query, limit=12, page=1):
    """Turbo-charged multi-channel search"""
    # ✅ Telegram session ready नहीं होने पर empty results return करेंगे
    if not user_session_ready:
        logger.warning("⚠️ Telegram session not ready, returning empty search results")
        return {
            'results': [],
            'pagination': {
                'current_page': page,
                'total_pages': 1,
                'total_results': 0,
                'per_page': limit,
                'has_next': False,
                'has_previous': False
            },
            'search_metadata': {
                'channels_searched': 0,
                'channels_found': 0,
                'query': query,
                'cache_hit': False
            }
        }
    
    offset = (page - 1) * limit
    
    # Try cache first
    cache_key = f"search:{query}:{page}:{limit}"
    if cache_manager and cache_manager.redis_enabled:
        cached_data = await cache_manager.get_search_results(query, page, limit)
        if cached_data:
            logger.info(f"✅ Cache HIT for: {query}")
            return cached_data
    
    logger.info(f"🔍 Multi-channel search for: {query}")
    
    query_lower = query.lower()
    posts_dict = {}
    files_dict = {}
    
    # MongoDB search with optimization
    try:
        if files_col is not None:
            # Use projection for faster queries
            cursor = files_col.find(
                {'$text': {'$search': query}},
                {
                    'title': 1,
                    'normalized_title': 1,
                    'quality': 1,
                    'file_size': 1,
                    'file_name': 1,
                    'thumbnail': 1,
                    'is_video_file': 1,
                    'thumbnail_source': 1,
                    'channel_id': 1,
                    'message_id': 1,
                    'date': 1,
                    '_id': 0
                }
            ).limit(limit * 2)  # Get slightly more for filtering
            
            async for doc in cursor:
                try:
                    norm_title = doc.get('normalized_title', normalize_title_cached(doc['title']))
                    quality = doc['quality']
                    
                    if norm_title not in files_dict:
                        files_dict[norm_title] = {
                            'title': doc['title'], 
                            'quality_options': {}, 
                            'date': doc['date'].isoformat() if isinstance(doc['date'], datetime) else doc['date'],
                            'thumbnail': doc.get('thumbnail'),
                            'is_video_file': doc.get('is_video_file', False),
                            'thumbnail_source': doc.get('thumbnail_source', 'unknown'),
                            'channel_id': doc.get('channel_id'),
                            'channel_name': channel_name_cached(doc.get('channel_id'))
                        }
                    
                    if quality not in files_dict[norm_title]['quality_options']:
                        files_dict[norm_title]['quality_options'][quality] = {
                            'file_id': f"{doc.get('channel_id', Config.FILE_CHANNEL_ID)}_{doc.get('message_id')}_{quality}",
                            'file_size': doc['file_size'],
                            'file_name': doc['file_name'],
                            'is_video': doc.get('is_video_file', False),
                            'channel_id': doc.get('channel_id'),
                            'message_id': doc.get('message_id')
                        }
                except:
                    continue
    except Exception as e:
        logger.error(f"File search error: {e}")
    
    # Telegram channel search (async)
    if user_session_ready:
        channel_tasks = []
        
        async def search_channel(channel_id):
            channel_posts = {}
            try:
                cname = channel_name_cached(channel_id)
                async for msg in safe_telegram_generator(
                    User.search_messages, 
                    channel_id, 
                    query=query, 
                    limit=15  # Reduced from 20
                ):
                    if msg and msg.text and len(msg.text) > 15:
                        title = extract_title_smart(msg.text)
                        if title and query_lower in title.lower():
                            norm_title = normalize_title_cached(title)
                            if norm_title not in channel_posts:
                                channel_posts[norm_title] = {
                                    'title': title,
                                    'content': format_post(msg.text),
                                    'channel': cname,
                                    'channel_id': channel_id,
                                    'message_id': msg.id,
                                    'date': msg.date.isoformat() if isinstance(msg.date, datetime) else msg.date,
                                    'is_new': is_new(msg.date) if msg.date else False,
                                    'has_file': False,
                                    'has_post': True,
                                    'quality_options': {},
                                    'thumbnail': None
                                }
            except Exception as e:
                logger.error(f"Telegram search error in {channel_id}: {e}")
            return channel_posts
        
        # Search channels concurrently
        for channel_id in Config.TEXT_CHANNEL_IDS:
            channel_tasks.append(search_channel(channel_id))
        
        results = await asyncio.gather(*channel_tasks, return_exceptions=True)
        
        # Merge results
        for result in results:
            if isinstance(result, dict):
                posts_dict.update(result)
    
    # Merge posts and files
    merged = {}
    
    # Add all posts
    for norm_title, post_data in posts_dict.items():
        merged[norm_title] = post_data
    
    # Add/update with file information
    for norm_title, file_data in files_dict.items():
        if norm_title in merged:
            merged[norm_title]['has_file'] = True
            merged[norm_title]['quality_options'] = file_data['quality_options']
            if file_data.get('is_video_file') and file_data.get('thumbnail'):
                merged[norm_title]['thumbnail'] = file_data['thumbnail']
                merged[norm_title]['thumbnail_source'] = file_data.get('thumbnail_source', 'unknown')
        else:
            merged[norm_title] = {
                'title': file_data['title'],
                'content': f"<p>{file_data['title']}</p>",
                'channel': file_data.get('channel_name', 'SK4FiLM Files'),
                'date': file_data['date'],
                'is_new': False,
                'has_file': True,
                'has_post': False,
                'quality_options': file_data['quality_options'],
                'thumbnail': file_data.get('thumbnail') if file_data.get('is_video_file') else None,
                'thumbnail_source': file_data.get('thumbnail_source', 'unknown')
            }
    
    # Sort results
    results_list = list(merged.values())
    results_list.sort(key=lambda x: (
        not x.get('is_new', False),
        not x['has_file'],
        x['date']
    ), reverse=True)
    
    total = len(results_list)
    paginated = results_list[offset:offset + limit]
    
    result_data = {
        'results': paginated,
        'pagination': {
            'current_page': page,
            'total_pages': math.ceil(total / limit) if total > 0 else 1,
            'total_results': total,
            'per_page': limit,
            'has_next': page < math.ceil(total / limit) if total > 0 else False,
            'has_previous': page > 1
        },
        'search_metadata': {
            'channels_searched': len(Config.TEXT_CHANNEL_IDS),
            'channels_found': len(set(r.get('channel_id') for r in paginated if r.get('channel_id'))),
            'query': query,
            'cache_hit': False
        }
    }
    
    # Cache results
    if cache_manager:
        await cache_manager.cache_search_results(query, page, limit, result_data)
    
    logger.info(f"✅ Search completed: {len(paginated)} results")
    
    return result_data

# ✅ UPDATED LIVE POSTS FUNCTION - Telegram session ready नहीं होने पर empty return करेगा
@async_cache_with_ttl(maxsize=100, ttl=60)
async def get_live_posts_multi_channel(limit_per_channel=30):
    """Cached live posts - Telegram session नहीं होने पर empty return करेगा"""
    if not User or not user_session_ready:
        logger.warning("❌ Telegram client not available for live posts")
        return []  # ✅ Empty array return करेगा
    
    all_posts = []
    
    async def fetch_channel_posts(channel_id):
        posts = []
        try:
            cname = channel_name_cached(channel_id)
            logger.debug(f"Fetching posts from {cname} ({channel_id})")
            
            count = 0
            async for msg in safe_telegram_generator(
                User.get_chat_history, 
                channel_id, 
                limit=limit_per_channel
            ):
                if msg and msg.text and len(msg.text) > 15:
                    title = extract_title_smart(msg.text)
                    if title:
                        posts.append({
                            'title': title,
                            'normalized_title': normalize_title_cached(title),
                            'content': msg.text,
                            'channel_name': cname,
                            'channel_id': channel_id,
                            'message_id': msg.id,
                            'date': msg.date,
                            'is_new': is_new(msg.date) if msg.date else False
                        })
                        count += 1
                        
                        # Stop at 30 movies
                        if count >= 30:
                            break
        except Exception as e:
            logger.error(f"Error getting posts from channel {channel_id}: {e}")
        return posts
    
    try:
        # ✅ सिर्फ MAIN_CHANNEL_ID से movies fetch करेंगे
        main_channel_id = Config.MAIN_CHANNEL_ID
        result = await fetch_channel_posts(main_channel_id)
        
        if isinstance(result, list):
            if result:
                logger.info(f"✅ Got {len(result)} movies from SK4FiLM Main Channel")
            all_posts.extend(result)
        else:
            logger.error(f"❌ Error from main channel: {result}")
        
        # Deduplicate and sort
        seen_titles = set()
        unique_posts = []
        
        for post in sorted(all_posts, key=lambda x: x.get('date', datetime.min), reverse=True):
            if post['normalized_title'] not in seen_titles:
                seen_titles.add(post['normalized_title'])
                unique_posts.append(post)
        
        logger.info(f"📊 Live movies collected: {len(unique_posts)} unique movies from SK4FiLM")
        return unique_posts[:30]
        
    except Exception as e:
        logger.error(f"Live posts multi-channel error: {e}")
        return []  # ✅ Empty array return करेगा

# ✅ UPDATED get_single_post_api FUNCTION - Telegram session नहीं होने पर None return करेगा
async def get_single_post_api(channel_id, message_id):
    """Get single movie/post details - Telegram session नहीं होने पर None return करेगा"""
    try:
        # Telegram session ready नहीं होने पर None return करेंगे
        if not User or not user_session_ready:
            logger.warning("❌ Telegram session not ready, cannot get post details")
            return None
        
        # Try to get the specific message
        msg = await safe_telegram_operation(
            User.get_messages,
            channel_id, 
            message_id
        )
        
        if msg and msg.text:
            title = extract_title_smart(msg.text)
            if not title:
                title = msg.text.split('\n')[0][:60] if msg.text else "Movie Post"
            
            normalized_title = normalize_title(title)
            quality_options = {}
            has_file = False
            thumbnail_url = None
            thumbnail_source = None
            
            # Search for files with same title
            if files_col is not None:
                cursor = files_col.find({'normalized_title': normalized_title})
                async for doc in cursor:
                    quality = doc.get('quality', '480p')
                    if quality not in quality_options:
                        file_name = doc.get('file_name', '').lower()
                        file_is_video = is_video_file(file_name)
                        
                        if file_is_video and not thumbnail_url:
                            thumbnail_url = doc.get('thumbnail')
                            thumbnail_source = doc.get('thumbnail_source', 'unknown')
                            
                            if not thumbnail_url and user_session_ready:
                                thumbnail_url = await get_telegram_video_thumbnail(User, doc['channel_id'], doc['message_id'])
                                if thumbnail_url:
                                    thumbnail_source = 'video_direct'
                        
                        quality_options[quality] = {
                            'file_id': f"{doc.get('channel_id', Config.FILE_CHANNEL_ID)}_{doc.get('message_id')}_{quality}",
                            'file_size': doc.get('file_size', 0),
                            'file_name': doc.get('file_name', 'video.mp4'),
                            'is_video': file_is_video,
                            'channel_id': doc.get('channel_id'),
                            'message_id': doc.get('message_id')
                        }
                        has_file = True
            
            # Get poster
            poster_url = f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(title)}"
            poster_source = 'custom'
            poster_rating = '0.0'
            
            if poster_fetcher:
                poster_data = await poster_fetcher.fetch_poster(title)
                if poster_data:
                    poster_url = poster_data['poster_url']
                    poster_source = poster_data['source']
                    poster_rating = poster_data.get('rating', '0.0')
            
            # ✅ Get related posts from BOTH channels
            related_posts = []
            if user_session_ready:
                for search_channel_id in Config.TEXT_CHANNEL_IDS:
                    try:
                        # Search for same title in this channel
                        async for search_msg in safe_telegram_generator(
                            User.search_messages,
                            search_channel_id,
                            query=title,
                            limit=5
                        ):
                            if search_msg and search_msg.text and search_msg.id != message_id:
                                search_title = extract_title_smart(search_msg.text)
                                if search_title and normalized_title in normalize_title(search_title):
                                    related_posts.append({
                                        'title': search_title,
                                        'content': format_post(search_msg.text),
                                        'channel': channel_name_cached(search_channel_id),
                                        'channel_id': search_channel_id,
                                        'message_id': search_msg.id,
                                        'date': search_msg.date.isoformat() if isinstance(search_msg.date, datetime) else str(search_msg.date),
                                        'is_new': is_new(search_msg.date) if search_msg.date else False
                                    })
                    except Exception as e:
                        logger.error(f"Error searching in channel {search_channel_id}: {e}")
            
            post_data = {
                'title': title,
                'content': format_post(msg.text),
                'channel': channel_name(channel_id),
                'channel_id': channel_id,
                'message_id': message_id,
                'date': msg.date.isoformat() if isinstance(msg.date, datetime) else str(msg.date),
                'is_new': is_new(msg.date) if msg.date else False,
                'has_file': has_file,
                'quality_options': quality_options,
                'views': getattr(msg, 'views', 0),
                'thumbnail': thumbnail_url,
                'thumbnail_source': thumbnail_source,
                'poster_url': poster_url,
                'poster_source': poster_source,
                'poster_rating': poster_rating,
                'related_posts': related_posts[:10]
            }
            
            return post_data
        
        return None
        
    except Exception as e:
        logger.error(f"Single post API error: {e}")
        return None

# ✅ UPDATED get_home_movies_live FUNCTION - Telegram session ready नहीं होने पर empty array return करेगा
@performance_monitor.measure("home_movies")
async def get_home_movies_live():
    """Optimized home movies - Telegram session नहीं होने पर empty array return करेगा"""
    try:
        # ✅ Telegram session ready नहीं होने पर empty array return करेंगे
        if not user_session_ready or not User:
            logger.warning("⚠️ Telegram user session not ready, returning empty movies list")
            return []  # ✅ Empty array return करेगा
        
        # Try to get posts with timeout
        try:
            posts_task = asyncio.create_task(get_live_posts_multi_channel(limit_per_channel=30))
            posts = await asyncio.wait_for(posts_task, timeout=8.0)
        except asyncio.TimeoutError:
            logger.warning("⏰ Home movies timeout")
            return []  # ✅ Empty array return करेगा
        except Exception as e:
            logger.error(f"Error getting live posts: {e}")
            return []  # ✅ Empty array return करेगा
        
        # If no posts, return empty
        if not posts:
            logger.warning("No posts found")
            return []  # ✅ Empty array return करेगा
        
        movies = []
        seen = set()
        
        for post in posts[:30]:
            if not post or 'title' not in post:
                continue
                
            tk = post['title'].lower().strip()
            if tk not in seen and tk:
                seen.add(tk)
                movies.append({
                    'title': post['title'],
                    'date': post.get('date', datetime.now().isoformat()),
                    'is_new': post.get('is_new', False),
                    'channel': post.get('channel_name', 'SK4FiLM'),
                    'channel_id': post.get('channel_id'),
                    'message_id': post.get('message_id', 0)
                })
        
        if not movies:
            logger.warning("No valid movies found")
            return []  # ✅ Empty array return करेगा
        
        # Fetch posters with timeout
        if movies and poster_fetcher:
            titles = [movie['title'] for movie in movies if movie.get('title')]
            
            if titles:
                try:
                    posters_task = asyncio.create_task(poster_fetcher.fetch_batch_posters(titles))
                    posters = await asyncio.wait_for(posters_task, timeout=3.0)
                    
                    for movie in movies:
                        if movie['title'] in posters:
                            poster_data = posters[movie['title']]
                            movie['poster_url'] = poster_data.get('poster_url', '')
                            movie['poster_source'] = poster_data.get('source', 'custom')
                            movie['poster_rating'] = poster_data.get('rating', '0.0')
                            movie['has_poster'] = True
                        else:
                            # Fast fallback
                            movie['poster_url'] = f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(movie['title'])}"
                            movie['poster_source'] = 'custom'
                            movie['poster_rating'] = '0.0'
                            movie['has_poster'] = True
                            
                except asyncio.TimeoutError:
                    logger.warning("⏰ Home posters timeout")
                    for movie in movies:
                        movie['poster_url'] = f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(movie['title'])}"
                        movie['poster_source'] = 'custom'
                        movie['poster_rating'] = '0.0'
                        movie['has_poster'] = True
                except Exception as e:
                    logger.error(f"Poster fetch error: {e}")
                    for movie in movies:
                        movie['poster_url'] = f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(movie['title'])}"
                        movie['poster_source'] = 'custom'
                        movie['poster_rating'] = '0.0'
                        movie['has_poster'] = True
        
        logger.info(f"✅ Home movies loaded: {len(movies)} movies from SK4FiLM")
        return movies
        
    except Exception as e:
        logger.error(f"Home movies error: {e}")
        return []  # ✅ Empty array return करेगा

# TELEGRAM SESSION TESTER FUNCTION
async def test_telegram_session():
    """Test Telegram session and provide detailed diagnostics"""
    if not Config.USER_SESSION_STRING:
        logger.error("❌ USER_SESSION_STRING not configured")
        return False
    
    if not Config.API_ID or Config.API_ID == 0:
        logger.error("❌ API_ID not configured")
        return False
    
    if not Config.API_HASH:
        logger.error("❌ API_HASH not configured")
        return False
    
    try:
        logger.info("🔍 Testing Telegram session...")
        
        # Create a test client
        test_client = Client(
            name="test_session",
            api_id=Config.API_ID,
            api_hash=Config.API_HASH,
            session_string=Config.USER_SESSION_STRING,
            sleep_threshold=60,
            max_concurrent_transmissions=1
        )
        
        # Try to start
        await test_client.start()
        
        # Test getting own info
        me = await test_client.get_me()
        logger.info(f"✅ Session test passed - User: @{me.username} (ID: {me.id})")
        
        # Test channel access
        for channel_id in Config.TEXT_CHANNEL_IDS:
            try:
                chat = await test_client.get_chat(channel_id)
                logger.info(f"✅ Channel access: {chat.title} (ID: {channel_id})")
            except Exception as e:
                logger.error(f"❌ Cannot access channel {channel_id}: {e}")
        
        await test_client.stop()
        return True
        
    except SessionPasswordNeeded:
        logger.error("❌ 2FA Password required for this account")
        return False
    except Exception as e:
        logger.error(f"❌ Telegram session test failed: {str(e)[:200]}")
        return False

# UPDATED TELEGRAM BOT INITIALIZATION WITH BETTER ERROR HANDLING
@performance_monitor.measure("telegram_init")
async def init_telegram_clients():
    """Optimized Telegram client initialization with detailed error handling"""
    global User, bot, bot_started, user_session_ready
    
    try:
        # First test the session
        session_test_ok = await test_telegram_session()
        if not session_test_ok:
            logger.error("❌ Telegram session test failed, skipping initialization")
            return False
        
        # Initialize User Client
        if Config.USER_SESSION_STRING and PYROGRAM_AVAILABLE:
            logger.info("📱 Initializing User Session...")
            try:
                User = Client(
                    name="user_session",
                    api_id=Config.API_ID,
                    api_hash=Config.API_HASH,
                    session_string=Config.USER_SESSION_STRING,
                    sleep_threshold=60,
                    max_concurrent_transmissions=3,
                    in_memory=True  # Store session in memory
                )
                
                # Start with timeout
                try:
                    await asyncio.wait_for(User.start(), timeout=30)
                except asyncio.TimeoutError:
                    logger.error("❌ Telegram connection timeout")
                    return False
                
                # Get user info
                user_info = await User.get_me()
                logger.info(f"✅ User Session Started: @{user_info.username} (ID: {user_info.id})")
                
                # Test channel access
                access_ok = False
                for channel_id in Config.TEXT_CHANNEL_IDS:
                    try:
                        chat = await User.get_chat(channel_id)
                        logger.info(f"✅ Access to {chat.title} verified")
                        access_ok = True
                        break
                    except Exception as e:
                        logger.warning(f"⚠️ Cannot access channel {channel_id}: {e}")
                
                if access_ok:
                    user_session_ready = True
                    logger.info("✅ User Session Ready")
                else:
                    logger.error("❌ Cannot access any Telegram channels")
                    user_session_ready = False
                
            except SessionPasswordNeeded:
                logger.error("❌ 2FA Password required for this account")
                user_session_ready = False
            except Exception as e:
                logger.error(f"❌ User Session Error: {str(e)[:200]}")
                user_session_ready = False
        
        # Initialize Bot Client
        if Config.BOT_TOKEN and PYROGRAM_AVAILABLE:
            logger.info("🤖 Initializing Bot...")
            try:
                bot = Client(
                    name="sk4film_bot",
                    api_id=Config.API_ID,
                    api_hash=Config.API_HASH,
                    bot_token=Config.BOT_TOKEN,
                    sleep_threshold=60,
                    workers=10
                )
                
                # Start with timeout
                try:
                    await asyncio.wait_for(bot.start(), timeout=30)
                except asyncio.TimeoutError:
                    logger.error("❌ Bot connection timeout")
                    return False
                
                bot_started = True
                bot_info = await bot.get_me()
                logger.info(f"✅ Bot Started: @{bot_info.username}")
            except Exception as e:
                logger.error(f"❌ Bot Error: {str(e)[:200]}")
                bot_started = False
        
        return user_session_ready or bot_started
        
    except Exception as e:
        logger.error(f"❌ Telegram clients initialization failed: {str(e)[:200]}")
        return False

# ADD A FUNCTION TO CHECK SESSION STATUS AND AUTO-RETRY
async def check_and_fix_telegram_session():
    """Check Telegram session status and try to fix if needed"""
    global user_session_ready, User
    
    if user_session_ready:
        try:
            # Quick ping to check if session is still alive
            await asyncio.wait_for(User.get_me(), timeout=5)
            return True
        except:
            logger.warning("⚠️ Telegram session seems dead, trying to restart...")
            user_session_ready = False
    
    # Try to reinitialize
    if not user_session_ready:
        logger.info("🔄 Attempting to reinitialize Telegram session...")
        success = await init_telegram_clients()
        if success:
            logger.info("✅ Telegram session reinitialized successfully")
            return True
    
    return False

# ADD TELEGRAM SESSION MONITOR TASK
async def telegram_session_monitor():
    """Background task to monitor Telegram session health"""
    while True:
        await asyncio.sleep(300)  # Check every 5 minutes
        
        if user_session_ready:
            try:
                # Quick check if session is still alive
                await asyncio.wait_for(User.get_me(), timeout=10)
                logger.debug("✅ Telegram session health check passed")
            except Exception as e:
                logger.warning(f"⚠️ Telegram session health check failed: {e}")
                user_session_ready = False
                
                # Try to fix
                await check_and_fix_telegram_session()
        else:
            # Try to reconnect
            logger.info("🔄 Telegram session monitor: Attempting to reconnect...")
            await check_and_fix_telegram_session()

# OPTIMIZED API FUNCTIONS
@performance_monitor.measure("search_api")
async def search_movies_api(query, limit=12, page=1):
    """Optimized search API with timeout protection"""
    try:
        # Set timeout for search
        search_task = asyncio.create_task(search_movies_multi_channel(query, limit, page))
        
        try:
            result_data = await asyncio.wait_for(search_task, timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning(f"⏰ Search timeout for query: {query}")
            # Return cached or empty results
            if cache_manager:
                cached = await cache_manager.get_search_results(query, page, limit)
                if cached:
                    return cached
            
            return {
                'results': [],
                'pagination': {
                    'current_page': page,
                    'total_pages': 1,
                    'total_results': 0,
                    'per_page': limit,
                    'has_next': False,
                    'has_previous': False
                },
                'search_metadata': {
                    'timeout': True,
                    'query': query
                }
            }
        
        # Enhance with posters using batch fetching with timeout
        if poster_fetcher and result_data.get('results'):
            titles = [result['title'] for result in result_data['results']]
            
            # Fetch posters with timeout
            try:
                posters_task = asyncio.create_task(poster_fetcher.fetch_batch_posters(titles))
                posters = await asyncio.wait_for(posters_task, timeout=3.0)
                
                for result in result_data['results']:
                    if result['title'] in posters:
                        poster_data = posters[result['title']]
                        result['poster_url'] = poster_data.get('poster_url', '')
                        result['poster_source'] = poster_data.get('source', 'custom')
                        result['poster_rating'] = poster_data.get('rating', '0.0')
                        result['has_poster'] = True
                    else:
                        # Fast fallback
                        result['poster_url'] = f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(result['title'])}"
                        result['poster_source'] = 'custom'
                        result['poster_rating'] = '0.0'
                        result['has_poster'] = False
                        
            except asyncio.TimeoutError:
                logger.warning("⏰ Poster fetch timeout")
                # Set default posters
                for result in result_data['results']:
                    result['poster_url'] = f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(result['title'])}"
                    result['poster_source'] = 'custom'
                    result['poster_rating'] = '0.0'
                    result['has_poster'] = False
        
        return result_data
        
    except Exception as e:
        logger.error(f"Search API error: {e}")
        # Return empty results on error
        return {
            'results': [],
            'pagination': {
                'current_page': page,
                'total_pages': 1,
                'total_results': 0,
                'per_page': limit,
                'has_next': False,
                'has_previous': False
            },
            'search_metadata': {
                'error': True,
                'query': query
            }
        }

# MAIN INITIALIZATION
@performance_monitor.measure("system_init")
async def init_system():
    """Turbo-charged system initialization with retry logic"""
    start_time = time.time()
    
    try:
        logger.info("🚀 Starting SK4FiLM v8.0 - TURBO OPTIMIZED...")
        
        # Initialize MongoDB
        mongo_ok = await init_mongodb()
        if not mongo_ok:
            logger.error("❌ MongoDB initialization failed")
            # Continue anyway, we can work without MongoDB
        
        # Initialize modular components
        global cache_manager, verification_system, premium_system, poster_fetcher, sk4film_bot
        
        # Initialize Cache Manager
        cache_manager = CacheManager(Config)
        redis_ok = await cache_manager.init_redis()
        if redis_ok:
            logger.info("✅ Cache Manager initialized")
            await cache_manager.start_cleanup_task()
        else:
            logger.warning("⚠️ Cache Manager - Redis not available")
        
        # Initialize Verification System only if MongoDB is available
        if db is not None:
            verification_system = VerificationSystem(Config, db)
            await verification_system.start_cleanup_task()
            logger.info("✅ Verification System initialized")
        else:
            logger.warning("⚠️ Verification System - MongoDB not available")
        
        # Initialize Premium System only if MongoDB is available
        if db is not None:
            premium_system = PremiumSystem(Config, db)
            await premium_system.start_cleanup_task()
            logger.info("✅ Premium System initialized")
        else:
            logger.warning("⚠️ Premium System - MongoDB not available")
        
        # Initialize Poster Fetcher
        poster_fetcher = PosterFetcher(Config, cache_manager)
        logger.info("✅ Poster Fetcher initialized")
        
        # ✅ WARM UP CACHE FOR INSTANT RESPONSE
        asyncio.create_task(warm_up_cache())
        
        # ✅ Initialize Telegram if available - WITH RETRY
        telegram_max_retries = 3
        telegram_success = False
        
        for attempt in range(telegram_max_retries):
            if PYROGRAM_AVAILABLE and BOT_HANDLERS_AVAILABLE:
                # Initialize SK4FiLMBot only if MongoDB is available
                if db is not None:
                    try:
                        sk4film_bot = SK4FiLMBot(Config, db)
                        await sk4film_bot.initialize()
                        logger.info("✅ SK4FiLMBot initialized")
                    except Exception as e:
                        logger.error(f"❌ SK4FiLMBot initialization failed: {e}")
                        sk4film_bot = None
                else:
                    logger.warning("⚠️ SK4FiLMBot - MongoDB not available")
                
                # Initialize Telegram clients with retry
                logger.info(f"🔄 Attempting Telegram initialization (Attempt {attempt + 1}/{telegram_max_retries})...")
                telegram_ok = await init_telegram_clients()
                
                if telegram_ok:
                    telegram_success = True
                    logger.info("✅ Telegram clients initialized successfully")
                    
                    # Setup bot handlers
                    if sk4film_bot and sk4film_bot.bot_started and bot:
                        await setup_bot_handlers(bot, sk4film_bot)
                        logger.info("✅ Bot handlers setup complete")
                    
                    break  # Success, exit retry loop
                else:
                    if attempt < telegram_max_retries - 1:
                        wait_time = (attempt + 1) * 5  # 5, 10, 15 seconds
                        logger.warning(f"⚠️ Telegram initialization failed, retrying in {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
            else:
                logger.warning("⚠️ Pyrogram/Bot handlers not available")
                break
        
        if not telegram_success:
            logger.warning("⚠️ Telegram clients could not be initialized after retries")
        
        # Start background tasks
        asyncio.create_task(cache_cleanup())
        
        # Start indexing in background if everything is ready
        if user_session_ready and files_col is not None:
            asyncio.create_task(index_files_background())
        
        # Start a background task to monitor and fix Telegram session
        asyncio.create_task(telegram_session_monitor())
        
        init_time = time.time() - start_time
        logger.info(f"⚡ SK4FiLM Started in {init_time:.2f}s")
        
        # Log system status
        logger.info(f"📊 System Status:")
        logger.info(f"   - MongoDB: {'✅' if mongo_ok else '❌'}")
        logger.info(f"   - Redis Cache: {'✅' if redis_ok else '⚠️'}")
        logger.info(f"   - Telegram Session: {'✅' if user_session_ready else '❌'}")
        logger.info(f"   - Bot Started: {'✅' if bot_started else '❌'}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ System initialization failed: {e}")
        # Still return True to allow the app to run
        return True

# API ROUTES WITH OPTIMIZATIONS
@app.route('/')
@performance_monitor.measure("root_endpoint")
async def root():
    """Optimized root endpoint with detailed status"""
    tf = await files_col.count_documents({}) if files_col is not None else 0
    video_files = await files_col.count_documents({'is_video_file': True}) if files_col is not None else 0
    
    # Get system diagnostics
    system_status = {
        'mongodb': files_col is not None,
        'redis': cache_manager.redis_enabled if cache_manager else False,
        'telegram_session': user_session_ready,
        'bot_started': bot_started,
        'poster_fetcher': poster_fetcher is not None,
        'verification_system': verification_system is not None,
        'premium_system': premium_system is not None
    }
    
    # Check if Telegram credentials are configured
    telegram_config = {
        'api_id_configured': bool(Config.API_ID and Config.API_ID != 0),
        'api_hash_configured': bool(Config.API_HASH),
        'session_string_configured': bool(Config.USER_SESSION_STRING),
        'bot_token_configured': bool(Config.BOT_TOKEN)
    }
    
    return jsonify({
        'status': 'healthy' if any([user_session_ready, bot_started]) else 'degraded',
        'service': 'SK4FiLM v8.0 - TURBO OPTIMIZED',
        'performance': {
            'cache_enabled': cache_manager.redis_enabled if cache_manager else False,
            'telegram_ready': user_session_ready,
            'bot_ready': bot_started
        },
        'database': {
            'total_files': tf, 
            'video_files': video_files
        },
        'telegram_configuration': telegram_config,
        'channels': len(Config.TEXT_CHANNEL_IDS),
        'main_channel_id': Config.MAIN_CHANNEL_ID,
        'response_time': f"{time.perf_counter():.3f}s"
    })

@app.route('/health')
@performance_monitor.measure("health_endpoint")
async def health():
    """Optimized health endpoint"""
    return jsonify({
        'status': 'ok' if bot_started else 'starting',
        'user_session': user_session_ready,
        'cache': cache_manager.redis_enabled if cache_manager else False,
        'timestamp': datetime.now().isoformat(),
        'performance': {
            'avg_response_time': performance_monitor.get_stats()
        }
    })

@app.route('/api/movies', methods=['GET'])
@performance_monitor.measure("movies_endpoint")
async def api_movies():
    """Optimized movies endpoint - Telegram session नहीं होने पर empty array return करेगा"""
    try:
        logger.debug("📽️ Getting home movies from SK4FiLM Main Channel...")
        
        # Check system status
        status = {
            'user_session_ready': user_session_ready,
            'User_available': User is not None,
            'telegram_channels': len(Config.TEXT_CHANNEL_IDS),
            'cache_manager': cache_manager is not None,
            'poster_fetcher': poster_fetcher is not None,
            'main_channel_id': Config.MAIN_CHANNEL_ID
        }
        logger.debug(f"System status: {status}")
        
        movies = await get_home_movies_live()
        
        logger.info(f"✅ Home movies API: {len(movies)} movies returned from SK4FiLM")
        
        return jsonify({
            'status': 'success',
            'movies': movies,
            'total': len(movies),
            'system_status': status,
            'timestamp': datetime.now().isoformat(),
            'cache_hit': False
        })
    except Exception as e:
        logger.error(f"Movies API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'movies': [],  # Always return empty array on error
            'total': 0,
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/search', methods=['GET'])
@performance_monitor.measure("search_endpoint")
async def api_search():
    """Optimized search endpoint - Telegram session नहीं होने पर empty results return करेगा"""
    try:
        query = request.args.get('query', '').strip()
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 12))
        
        if len(query) < 2:
            return jsonify({
                'status': 'error',
                'message': 'Query must be at least 2 characters'
            }), 400
        
        result_data = await search_movies_api(query, limit, page)
        
        return jsonify({
            'status': 'success',
            'query': query,
            'results': result_data['results'],
            'pagination': result_data['pagination'],
            'timestamp': datetime.now().isoformat(),
            'cache_hit': result_data.get('search_metadata', {}).get('cache_hit', False),
            'channels_searched': Config.TEXT_CHANNEL_IDS
        })
    except Exception as e:
        logger.error(f"Search API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/post', methods=['GET'])
async def api_post():
    """Get single post details - Telegram session नहीं होने पर error return करेगा"""
    try:
        channel_id = int(request.args.get('channel', Config.MAIN_CHANNEL_ID))
        message_id = int(request.args.get('message', 0))
        
        if message_id <= 0:
            return jsonify({
                'status': 'error',
                'message': 'Invalid message ID'
            }), 400
        
        # ✅ Telegram session ready नहीं होने पर error return करेंगे
        if not user_session_ready:
            return jsonify({
                'status': 'error',
                'message': 'Telegram session not ready. Please try again later.',
                'telegram_ready': False
            }), 503
        
        post_data = await get_single_post_api(channel_id, message_id)
        
        if post_data:
            return jsonify({
                'status': 'success',
                'post': post_data,
                'channels_searched': Config.TEXT_CHANNEL_IDS,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Post not found'
            }), 404
            
    except Exception as e:
        logger.error(f"Post API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/poster', methods=['GET'])
async def api_poster():
    """Get movie poster"""
    try:
        title = request.args.get('title', '').strip()
        year = request.args.get('year', '')
        
        if not title:
            return jsonify({
                'status': 'error',
                'message': 'Title is required'
            }), 400
        
        if poster_fetcher:
            poster_data = await poster_fetcher.fetch_poster(title)
            
            if poster_data:
                return jsonify({
                    'status': 'success',
                    'poster': poster_data,
                    'title': title,
                    'year': year,
                    'timestamp': datetime.now().isoformat()
                })
        
        # Fallback
        return jsonify({
            'status': 'success',
            'poster': {
                'poster_url': f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(title)}&year={year}",
                'source': PosterSource.CUSTOM.value,
                'rating': '0.0'
            },
            'title': title,
            'year': year,
            'timestamp': datetime.now().isoformat()
        })
                
    except Exception as e:
        logger.error(f"Poster API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/verify_user', methods=['POST'])
async def api_verify_user():
    """Verify user with URL shortener"""
    try:
        data = await request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No JSON data provided'
            }), 400
        
        user_id = data.get('user_id')
        verification_url = data.get('verification_url')
        
        if not user_id:
            return jsonify({
                'status': 'error',
                'message': 'user_id is required'
            }), 400
        
        result = await verify_user_api(user_id, verification_url)
        
        return jsonify({
            'status': 'success',
            'verification': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Verify user API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/premium/plans', methods=['GET'])
async def api_premium_plans():
    """Get premium plans"""
    try:
        if premium_system:
            plans = await premium_system.get_all_plans()
            
            return jsonify({
                'status': 'success',
                'plans': plans,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Premium system not available'
            }), 503
    except Exception as e:
        logger.error(f"Premium plans API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/user/status', methods=['GET'])
async def api_user_status():
    """Get user status (premium/verification)"""
    try:
        user_id = int(request.args.get('user_id', 0))
        
        if user_id <= 0:
            return jsonify({
                'status': 'error',
                'message': 'Valid user_id is required'
            }), 400
        
        verification_status = False
        premium_status = None
        
        if verification_system:
            verification_status = await verification_system.is_user_verified(user_id)
        
        if premium_system:
            premium_status = await premium_system.get_user_tier(user_id)
        
        return jsonify({
            'status': 'success',
            'user_id': user_id,
            'verified': verification_status,
            'premium_tier': premium_status.value if premium_status else 'FREE',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"User status API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/index/status', methods=['GET'])
async def api_index_status():
    """Get indexing status"""
    try:
        status_data = await get_index_status_api()
        
        return jsonify({
            'status': 'success',
            'index': status_data,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Index status API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/system/status', methods=['GET'])
async def api_system_status():
    """Get detailed system status"""
    try:
        # MongoDB stats
        mongo_stats = {}
        if files_col is not None:
            try:
                total_files = await files_col.count_documents({})
                video_files = await files_col.count_documents({'is_video_file': True})
                mongo_stats = {
                    'total_files': total_files,
                    'video_files': video_files,
                    'status': 'connected'
                }
            except Exception as e:
                mongo_stats = {'status': f'error: {str(e)[:100]}'}
        else:
            mongo_stats = {'status': 'not_initialized'}
        
        # Cache stats
        cache_stats = {}
        if cache_manager:
            cache_stats = {
                'redis_enabled': cache_manager.redis_enabled,
                'connection': 'connected' if cache_manager.redis_conn else 'disconnected'
            }
        
        # Telegram stats
        telegram_stats = {
            'user_session_ready': user_session_ready,
            'bot_started': bot_started,
            'main_channel_id': Config.MAIN_CHANNEL_ID,
            'channels_configured': len(Config.TEXT_CHANNEL_IDS)
        }
        
        # Component status
        components = {
            'poster_fetcher': poster_fetcher is not None,
            'verification_system': verification_system is not None,
            'premium_system': premium_system is not None,
            'sk4film_bot': sk4film_bot is not None
        }
        
        # Performance stats
        perf_stats = performance_monitor.get_stats()
        
        return jsonify({
            'status': 'success',
            'system': {
                'mongodb': mongo_stats,
                'cache': cache_stats,
                'telegram': telegram_stats,
                'components': components,
                'performance': perf_stats,
                'uptime': time.time() - app_start_time if 'app_start_time' in globals() else 0
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"System status API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/channels', methods=['GET'])
async def api_channels():
    """Get channel information"""
    try:
        channels = []
        
        for channel_id in Config.TEXT_CHANNEL_IDS:
            channel_info = {
                'id': channel_id,
                'name': channel_name_cached(channel_id),
                'type': CHANNEL_CONFIG.get(channel_id, {}).get('type', 'text'),
                'search_priority': CHANNEL_CONFIG.get(channel_id, {}).get('search_priority', 99)
            }
            channels.append(channel_info)
        
        # Add file channel if configured
        if Config.FILE_CHANNEL_ID and Config.FILE_CHANNEL_ID not in Config.TEXT_CHANNEL_IDS:
            channels.append({
                'id': Config.FILE_CHANNEL_ID,
                'name': 'SK4FiLM Files',
                'type': 'file',
                'search_priority': 0
            })
        
        return jsonify({
            'status': 'success',
            'channels': channels,
            'total': len(channels),
            'main_channel_id': Config.MAIN_CHANNEL_ID,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Channels API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/batch/posters', methods=['POST'])
async def api_batch_posters():
    """Get multiple posters in batch"""
    try:
        data = await request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No JSON data provided'
            }), 400
        
        titles = data.get('titles', [])
        
        if not titles or not isinstance(titles, list):
            return jsonify({
                'status': 'error',
                'message': 'titles array is required'
            }), 400
        
        # Limit batch size
        titles = titles[:50]
        
        posters = {}
        
        if poster_fetcher:
            posters = await poster_fetcher.fetch_batch_posters(titles)
        else:
            # Fallback for each title
            for title in titles:
                if title and isinstance(title, str):
                    posters[title] = {
                        'poster_url': f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(title)}",
                        'source': PosterSource.CUSTOM.value,
                        'rating': '0.0'
                    }
        
        return jsonify({
            'status': 'success',
            'posters': posters,
            'count': len(posters),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Batch posters API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/performance', methods=['GET'])
async def api_performance():
    """Get performance metrics"""
    try:
        stats = performance_monitor.get_stats()
        
        return jsonify({
            'status': 'success',
            'metrics': stats,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Performance API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/admin/clear_cache', methods=['POST'])
async def api_clear_cache():
    """Admin endpoint to clear cache"""
    try:
        # Check if admin (simple check)
        data = await request.get_json()
        admin_key = data.get('admin_key', '') if data else ''
        
        # Simple admin check - in production use proper authentication
        if admin_key != os.environ.get('ADMIN_KEY', 'admin123'):
            return jsonify({
                'status': 'error',
                'message': 'Unauthorized'
            }), 403
        
        cleared = False
        
        if cache_manager:
            cleared = await cache_manager.clear_all_cache()
        
        # Also clear poster cache
        if poster_fetcher:
            await poster_fetcher.cleanup_expired_cache()
        
        return jsonify({
            'status': 'success',
            'message': 'Cache cleared' if cleared else 'Cache not cleared',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Clear cache API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# ERROR HANDLERS
@app.errorhandler(404)
async def not_found(error):
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found',
        'timestamp': datetime.now().isoformat()
    }), 404

@app.errorhandler(500)
async def server_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Internal server error',
        'timestamp': datetime.now().isoformat()
    }), 500

# STARTUP FUNCTION
async def startup():
    """Initialize the application"""
    global app_start_time
    app_start_time = time.time()
    
    logger.info("🚀 SK4FiLM Web API v8.0 - Turbo Optimized")
    logger.info("⚡ Performance features enabled:")
    logger.info("   - Async cache with TTL")
    logger.info("   - Connection pooling")
    logger.info("   - Performance monitoring")
    logger.info("   - Batch operations")
    logger.info("   - Flood protection")
    
    # Initialize system
    success = await init_system()
    
    if not success:
        logger.error("❌ System initialization had issues")
        logger.warning("⚠️ Continuing anyway, some features may be limited")
    
    logger.info("✅ SK4FiLM API Ready!")

# SHUTDOWN FUNCTION
async def shutdown():
    """Clean shutdown"""
    logger.info("🛑 Shutting down SK4FiLM...")
    
    # Stop background tasks
    if cache_manager:
        await cache_manager.stop_cleanup_task()
    
    if verification_system:
        await verification_system.stop_cleanup_task()
    
    if premium_system:
        await premium_system.stop_cleanup_task()
    
    # Stop Telegram clients
    global User, bot, user_session_ready, bot_started
    
    if user_session_ready and User:
        try:
            await User.stop()
            logger.info("✅ User session stopped")
        except:
            pass
    
    if bot_started and bot:
        try:
            await bot.stop()
            logger.info("✅ Bot stopped")
        except:
            pass
    
    logger.info("👋 SK4FiLM shutdown complete")

# MAIN ENTRY POINT
if __name__ == "__main__":
    # Create Hypercorn config
    config = HyperConfig()
    config.bind = [f"0.0.0.0:{Config.WEB_SERVER_PORT}"]
    config.workers = 1  # For async operation, workers=1 is optimal
    
    # Run with startup/shutdown hooks
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Start the server
        loop.run_until_complete(serve(
            app,
            config,
            shutdown_trigger=lambda: asyncio.Future()  # Run forever
        ))
    except KeyboardInterrupt:
        logger.info("🛑 Received shutdown signal")
        loop.run_until_complete(shutdown())
    finally:
        loop.close()
