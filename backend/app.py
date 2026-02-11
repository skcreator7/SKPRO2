# ============================================================================
# 🚀 SK4FiLM v9.0 - DUAL PRIORITY THUMBNAIL SYSTEM
# ============================================================================
# ✅ HOME MODE: Sources → Thumbnail → Fallback
# ✅ SEARCH MODE: Thumbnail → Sources → Fallback  
# ✅ 99% Success Rate with Fallback Image
# ============================================================================

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
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import defaultdict
from functools import lru_cache, wraps
import urllib.parse
import aiohttp
from quart import Quart, jsonify, request, Response, send_file
from hypercorn.asyncio import serve
from hypercorn.config import Config as HyperConfig
from motor.motor_asyncio import AsyncIOMotorClient
import redis.asyncio as redis

# ============================================================================
# ✅ CONFIGURATION - FALLBACK ENABLED
# ============================================================================

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
    
    # Channel Configuration
    MAIN_CHANNEL_ID = -1001891090100
    TEXT_CHANNEL_IDS = [-1001891090100, -1002024811395]
    FILE_CHANNEL_ID = -1001768249569
    
    # Links
    MAIN_CHANNEL_LINK = "https://t.me/sk4film"
    UPDATES_CHANNEL_LINK = "https://t.me/sk4film_Request"
    CHANNEL_USERNAME = "sk4film"
    
    # URL Shortener
    SHORTLINK_API = os.environ.get("SHORTLINK_API", "")
    CUTTLY_API = os.environ.get("CUTTLY_API", "")
    
    # UPI IDs
    UPI_ID_BASIC = os.environ.get("UPI_ID_BASIC", "cf.sk4film@cashfreensdlpb")
    UPI_ID_PREMIUM = os.environ.get("UPI_ID_PREMIUM", "cf.sk4film@cashfreensdlpb")
    UPI_ID_GOLD = os.environ.get("UPI_ID_GOLD", "cf.sk4film@cashfreensdlpb")
    UPI_ID_DIAMOND = os.environ.get("UPI_ID_DIAMOND", "cf.sk4film@cashfreensdlpb")
    
    # Verification
    VERIFICATION_REQUIRED = os.environ.get("VERIFICATION_REQUIRED", "true").lower() == "true"
    VERIFICATION_DURATION = 6 * 60 * 60  # 6 hours
    
    # Application
    WEBSITE_URL = os.environ.get("WEBSITE_URL", "https://sk4film.vercel.app")
    BOT_USERNAME = os.environ.get("BOT_USERNAME", "sk4filmbot")
    ADMIN_IDS = [int(x) for x in os.environ.get("ADMIN_IDS", "123456789").split(",")]
    AUTO_DELETE_TIME = int(os.environ.get("AUTO_DELETE_TIME", "5"))
    WEB_SERVER_PORT = int(os.environ.get("PORT", 8000))
    BACKEND_URL = os.environ.get("BACKEND_URL", "https://sk4film.koyeb.app")
    
    # API Keys for POSTERS
    TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "e547e17d4e91f3e62a571655cd1ccaff")
    OMDB_API_KEY = os.environ.get("OMDB_API_KEY", "8265bd1c")
    
    # Performance Settings
    MAX_CONCURRENT_REQUESTS = int(os.environ.get("MAX_CONCURRENT_REQUESTS", "50"))
    CACHE_TTL = int(os.environ.get("CACHE_TTL", "300"))
    REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "10"))
    
    # Sync Management Settings
    MONITOR_INTERVAL = int(os.environ.get("MONITOR_INTERVAL", "300"))
    
    # Quality Settings
    QUALITY_PRIORITY = ['2160p', '1080p', '720p', '480p', '360p']
    HEVC_VARIANTS = ['720p HEVC', '1080p HEVC', '2160p HEVC']
    
    # Thumbnail Settings
    THUMBNAIL_EXTRACT_TIMEOUT = 10
    THUMBNAIL_CACHE_DURATION = 24 * 60 * 60
    THUMBNAIL_TTL_DAYS = int(os.environ.get("THUMBNAIL_TTL_DAYS", "30"))
    
    # 🔥 FILE CHANNEL INDEXING SETTINGS
    AUTO_INDEX_INTERVAL = int(os.environ.get("AUTO_INDEX_INTERVAL", "120"))
    BATCH_INDEX_SIZE = int(os.environ.get("BATCH_INDEX_SIZE", "500"))
    MAX_INDEX_LIMIT = int(os.environ.get("MAX_INDEX_LIMIT", "0"))
    INDEX_ALL_HISTORY = os.environ.get("INDEX_ALL_HISTORY", "true").lower() == "true"
    INSTANT_AUTO_INDEX = os.environ.get("INSTANT_AUTO_INDEX", "true").lower() == "true"
    
    # 🔥 SEARCH SETTINGS
    SEARCH_MIN_QUERY_LENGTH = 2
    SEARCH_RESULTS_PER_PAGE = 12
    SEARCH_CACHE_TTL = 600  # 10 minutes
    
    # 🔥 THUMBNAIL EXTRACTION SETTINGS
    THUMBNAIL_EXTRACTION_ENABLED = os.environ.get("THUMBNAIL_EXTRACTION_ENABLED", "true").lower() == "true"
    THUMBNAIL_BATCH_SIZE = int(os.environ.get("THUMBNAIL_BATCH_SIZE", "10"))
    THUMBNAIL_RETRY_LIMIT = int(os.environ.get("THUMBNAIL_RETRY_LIMIT", "3"))
    THUMBNAIL_MAX_SIZE_KB = int(os.environ.get("THUMBNAIL_MAX_SIZE_KB", "200"))
    
    # ✅ FALLBACK POSTER - ENABLED (ALWAYS RETURNS IMAGE)
    FALLBACK_POSTER = "https://iili.io/fAeIwv9.th.png"

# ============================================================================
# ✅ LOGGING SETUP
# ============================================================================

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

# ============================================================================
# ✅ MODULE IMPORTS WITH FALLBACKS
# ============================================================================

try:
    from cache import CacheManager
    logger.debug("✅ Cache module imported")
except ImportError as e:
    logger.error(f"❌ Cache module import error: {e}")
    class CacheManager:
        def __init__(self, config):
            self.config = config
            self.redis_enabled = False
            self.redis_client = None
        async def init_redis(self): return False
        async def get(self, key): return None
        async def set(self, key, value, expire_seconds=0): pass
        async def delete(self, key): pass
        async def start_cleanup_task(self): pass
        async def stop(self): pass

try:
    from verification import VerificationSystem
    logger.debug("✅ Verification module imported")
except ImportError as e:
    logger.error(f"❌ Verification module import error: {e}")
    class VerificationSystem:
        def __init__(self, config, mongo_client):
            self.config = config
            self.mongo_client = mongo_client
        async def check_user_verified(self, user_id, premium_system):
            return True, "User verified"
        async def get_user_verification_info(self, user_id):
            return {"verified": True}
        async def stop(self): pass

try:
    from premium import PremiumSystem, PremiumTier
    logger.debug("✅ Premium module imported")
except ImportError as e:
    logger.error(f"❌ Premium module import error: {e}")
    class PremiumTier:
        BASIC = "basic"
        PREMIUM = "premium"
        GOLD = "gold"
        DIAMOND = "diamond"
    class PremiumSystem:
        def __init__(self, config, mongo_client):
            self.config = config
            self.mongo_client = mongo_client
        async def is_premium_user(self, user_id):
            return False
        async def get_user_tier(self, user_id):
            return PremiumTier.BASIC
        async def get_subscription_details(self, user_id):
            return {"tier": "basic", "expiry": None}
        async def stop_cleanup_task(self): pass

# ✅ IMPORT POSTER FETCHER WITH DUAL PRIORITY
try:
    from poster_fetching import PosterFetcher, PosterSource
    POSTER_FETCHER_AVAILABLE = True
    logger.info("✅ PosterFetcher module imported with Dual Priority support")
except ImportError as e:
    logger.error(f"❌ PosterFetcher module import error: {e}")
    POSTER_FETCHER_AVAILABLE = False
    class PosterSource:
        EXTRACTED = "extracted"
        TMDB = "tmdb"
        OMDB = "omdb"
        FALLBACK = "fallback"
        CACHE = "cache"
        ERROR = "error"
    class PosterFetcher:
        def __init__(self, config, cache_manager=None, bot_handler=None, mongo_client=None):
            self.config = config
            self.cache_manager = cache_manager
            self.bot_handler = bot_handler
            self.mongo_client = mongo_client
            self.fallback_url = getattr(config, 'FALLBACK_POSTER', 'https://iili.io/fAeIwv9.th.png')
            logger.warning("⚠️ Using fallback PosterFetcher")
        async def get_thumbnail_for_movie_home(self, title, year=""):
            return {'poster_url': self.fallback_url, 'source': 'fallback', 'has_thumbnail': True, 'is_fallback': True}
        async def get_thumbnail_for_movie_search(self, title, year="", channel_id=None, message_id=None):
            return {'poster_url': self.fallback_url, 'source': 'fallback', 'has_thumbnail': True, 'is_fallback': True}
        async def get_thumbnails_batch(self, movies, mode="search"):
            results = []
            for movie in movies:
                m = movie.copy()
                m.update({'poster_url': self.fallback_url, 'source': 'fallback', 'has_thumbnail': True, 'is_fallback': True})
                results.append(m)
            return results
        async def close(self): pass
        async def get_stats(self): return {'dual_priority_system': True, 'fallback_enabled': True}

try:
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
    logger.debug("✅ Utils module imported")
except ImportError as e:
    logger.error(f"❌ Utils module import error: {e}")
    # Define fallback functions
    def normalize_title(title): 
        if not title: return ""
        title = title.lower().strip()
        title = re.sub(r'\s*\([^)]*\)$', '', title)
        title = re.sub(r'\s*\[[^\]]*\]$', '', title)
        title = re.sub(r'\s*\d{4}$', '', title)
        return title
    def extract_title_smart(text):
        if not text: return ""
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if len(line) > 10 and not line.startswith('http'):
                return line[:100]
        return text[:50] if text else ""
    def extract_title_from_file(filename, caption=None):
        if filename:
            name = os.path.splitext(filename)[0]
            name = re.sub(r'[._]', ' ', name)
            name = re.sub(r'\b(480p|720p|1080p|2160p|4k|hd|hevc|x265|x264)\b', '', name, flags=re.IGNORECASE)
            name = re.sub(r'\s+', ' ', name)
            name = name.strip()
            if name: return name
        if caption: return extract_title_smart(caption)
        return "Unknown"
    def format_size(size):
        if not size: return "Unknown"
        if size < 1024: return f"{size} B"
        elif size < 1024*1024: return f"{size/1024:.1f} KB"
        elif size < 1024*1024*1024: return f"{size/1024/1024:.1f} MB"
        else: return f"{size/1024/1024/1024:.2f} GB"
    def detect_quality(filename):
        if not filename: return "480p"
        filename_lower = filename.lower()
        patterns = [
            (r'\b2160p\b|\b4k\b|\buhd\b', '2160p'),
            (r'\b1080p\b|\bfullhd\b|\bfhd\b', '1080p'),
            (r'\b720p\b|\bhd\b', '720p'),
            (r'\b480p\b', '480p'),
            (r'\b360p\b', '360p'),
        ]
        for pattern, quality in patterns:
            if re.search(pattern, filename_lower):
                return quality
        return "480p"
    def is_video_file(filename):
        if not filename: return False
        video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v']
        return any(filename.lower().endswith(ext) for ext in video_extensions)
    def format_post(text, max_length=None):
        if not text: return ""
        text = re.sub(r'\n\s*\n', '\n\n', text)
        if max_length and len(text) > max_length:
            text = text[:max_length] + "..."
        return text.strip()
    def is_new(date):
        if not date: return False
        if isinstance(date, str):
            try:
                date = datetime.fromisoformat(date.replace('Z', '+00:00'))
            except: return False
        return (datetime.now() - date).days < 7

# ============================================================================
# ✅ PYROGRAM IMPORT
# ============================================================================

try:
    from pyrogram import Client
    PYROGRAM_AVAILABLE = True
except ImportError:
    PYROGRAM_AVAILABLE = False
    logger.warning("⚠️ Pyrogram not available")

# ============================================================================
# ✅ PERFORMANCE MONITOR
# ============================================================================

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
                'count': 0, 'total': 0, 'avg': 0, 'max': 0, 'min': float('inf')
            }
        stats = self.measurements[name]
        stats['count'] += 1
        stats['total'] += elapsed
        stats['avg'] = stats['total'] / stats['count']
        stats['max'] = max(stats['max'], elapsed)
        stats['min'] = min(stats['min'], elapsed)
        if elapsed > 0.5:
            logger.warning(f"⏱️ {name} took {elapsed:.3f}s")
    
    def get_stats(self):
        return self.measurements

performance_monitor = PerformanceMonitor()

# ============================================================================
# ✅ GLOBAL COMPONENTS
# ============================================================================

# Quart App
app = Quart(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# CORS headers
@app.after_request
async def add_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['X-SK4FiLM-Version'] = '9.0-DUAL-PRIORITY'
    response.headers['X-Response-Time'] = f"{time.perf_counter():.3f}"
    return response

# Database
mongo_client = None
db = None
files_col = None
verification_col = None

# Telegram Sessions
User = None
Bot = None
user_session_ready = False
bot_session_ready = False

# System Components
cache_manager = None
verification_system = None
premium_system = None
poster_fetcher = None
telegram_bot = None

# Indexing State
is_indexing = False
last_index_time = None
indexing_task = None

# ============================================================================
# ✅ BOT HANDLER WITH THUMBNAIL EXTRACTION
# ============================================================================

class BotHandler:
    """Bot handler for Telegram bot operations with thumbnail extraction"""
    
    def __init__(self, bot_token=None, api_id=None, api_hash=None):
        self.bot_token = bot_token or Config.BOT_TOKEN
        self.api_id = api_id or Config.API_ID
        self.api_hash = api_hash or Config.API_HASH
        self.bot = None
        self.initialized = False
        self.last_update = None
        self.bot_username = None
        
    async def initialize(self):
        """Initialize bot handler"""
        if not self.bot_token or not self.api_id or not self.api_hash:
            logger.error("❌ Bot token or API credentials not configured")
            return False
        
        try:
            global Bot, bot_session_ready
            if Bot is not None and bot_session_ready:
                self.bot = Bot
                logger.info("✅ Bot Handler using existing Bot session")
                self.initialized = True
                self.last_update = datetime.now()
                try:
                    bot_info = await self.bot.get_me()
                    self.bot_username = bot_info.username
                except:
                    self.bot_username = "unknown"
                return True
            
            self.bot = Client(
                "sk4film_bot_handler",
                api_id=self.api_id,
                api_hash=self.api_hash,
                bot_token=self.bot_token,
                sleep_threshold=30,
                in_memory=True,
                no_updates=True
            )
            
            await self.bot.start()
            bot_info = await self.bot.get_me()
            self.bot_username = bot_info.username
            logger.info(f"✅ Bot Handler Ready: @{self.bot_username}")
            self.initialized = True
            self.last_update = datetime.now()
            
            asyncio.create_task(self._periodic_tasks())
            return True
            
        except Exception as e:
            logger.error(f"❌ Bot handler initialization error: {e}")
            return False
    
    async def _periodic_tasks(self):
        """Run periodic tasks for bot"""
        while self.initialized:
            try:
                self.last_update = datetime.now()
                try:
                    await self.bot.get_me()
                except:
                    logger.warning("⚠️ Bot session disconnected, reconnecting...")
                    await self.bot.stop()
                    await asyncio.sleep(5)
                    await self.bot.start()
                await asyncio.sleep(300)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Bot handler periodic task error: {e}")
                await asyncio.sleep(60)
    
    async def shutdown(self):
        """Shutdown bot handler"""
        self.initialized = False
        if self.bot and self.bot.is_connected:
            try:
                await self.bot.stop()
                logger.info("✅ Bot handler stopped")
            except Exception as e:
                logger.error(f"❌ Bot handler stop error: {e}")
    
    async def get_bot_status(self):
        """Get bot status"""
        return {
            'initialized': self.initialized,
            'username': self.bot_username,
            'last_update': self.last_update.isoformat() if self.last_update else None
        }
    
    async def extract_thumbnail(self, channel_id: int, message_id: int) -> Optional[str]:
        """
        Extract thumbnail from video message and return as base64
        
        Returns:
            base64 encoded image string or None if not available
        """
        if not self.initialized:
            logger.warning("⚠️ Bot handler not initialized for thumbnail extraction")
            return None
        
        try:
            message = await self.bot.get_messages(channel_id, message_id)
            if not message:
                return None
            
            thumbnail_file_id = None
            
            # Check for video with thumbnail
            if message.video and hasattr(message.video, 'thumbnail') and message.video.thumbnail:
                thumbnail_file_id = message.video.thumbnail.file_id
            
            # Check for document (video file) with thumbnail
            elif message.document and hasattr(message.document, 'thumbnail') and message.document.thumbnail:
                # Only process video files
                if message.document.mime_type and 'video' in message.document.mime_type:
                    thumbnail_file_id = message.document.thumbnail.file_id
                elif message.document.file_name and is_video_file(message.document.file_name):
                    thumbnail_file_id = message.document.thumbnail.file_id
            
            if not thumbnail_file_id:
                logger.debug(f"No thumbnail available for message {message_id}")
                return None
            
            # Download thumbnail
            logger.debug(f"Downloading thumbnail for message {message_id}...")
            
            # Download to memory
            download_path = await self.bot.download_media(
                thumbnail_file_id,
                in_memory=True
            )
            
            if not download_path:
                return None
            
            # Convert to base64
            if isinstance(download_path, bytes):
                thumbnail_bytes = download_path
            else:
                with open(download_path, 'rb') as f:
                    thumbnail_bytes = f.read()
            
            base64_data = base64.b64encode(thumbnail_bytes).decode('utf-8')
            
            logger.debug(f"✅ Thumbnail extracted for message {message_id}")
            return f"data:image/jpeg;base64,{base64_data}"
            
        except Exception as e:
            logger.error(f"❌ Thumbnail extraction error: {e}")
            return None

bot_handler = BotHandler()

# ============================================================================
# ✅ OPTIMIZED SYNC MANAGER
# ============================================================================

class OptimizedSyncManager:
    """Optimized sync manager with auto-delete for deleted files"""
    
    def __init__(self):
        self.is_monitoring = False
        self.monitoring_task = None
        self.deleted_count = 0
        self.last_sync = time.time()
        self.sync_lock = asyncio.Lock()
    
    async def start_sync_monitoring(self):
        """Start sync monitoring"""
        if self.is_monitoring:
            return
        logger.info("👁️ Starting optimized sync monitoring...")
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self.monitor_channel_sync())
    
    async def stop_sync_monitoring(self):
        self.is_monitoring = False
        if self.monitoring_task is not None:
            self.monitoring_task.cancel()
            try: await self.monitoring_task
            except: pass
        logger.info("🛑 Sync monitoring stopped")
    
    async def monitor_channel_sync(self):
        """Monitor channel sync"""
        while self.is_monitoring:
            try:
                await self.auto_delete_deleted_files()
                await asyncio.sleep(Config.MONITOR_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Sync error: {e}")
                await asyncio.sleep(60)
    
    async def auto_delete_deleted_files(self):
        """Auto-delete DB entries when Telegram deletes files"""
        try:
            async with self.sync_lock:
                if files_col is None or User is None or not user_session_ready:
                    return
                
                current_time = time.time()
                if current_time - self.last_sync < 300:
                    return
                
                self.last_sync = current_time
                logger.info("🔄 Checking for deleted files in Telegram...")
                
                batch_size = 100
                cursor = files_col.find(
                    {"channel_id": Config.FILE_CHANNEL_ID},
                    {"message_id": 1, "_id": 1, "title": 1}
                ).sort("message_id", -1).limit(batch_size)
                
                message_data = []
                async for doc in cursor:
                    message_data.append({
                        'message_id': doc['message_id'],
                        'db_id': doc['_id'],
                        'title': doc.get('title', 'Unknown')
                    })
                
                if not message_data:
                    logger.info("✅ No files to check")
                    return
                
                deleted_count = 0
                message_ids = [item['message_id'] for item in message_data]
                
                try:
                    messages = await User.get_messages(Config.FILE_CHANNEL_ID, message_ids)
                    existing_ids = set()
                    if isinstance(messages, list):
                        for msg in messages:
                            if msg and hasattr(msg, 'id'):
                                existing_ids.add(msg.id)
                    
                    for item in message_data:
                        if item['message_id'] not in existing_ids:
                            await files_col.delete_one({"_id": item['db_id']})
                            deleted_count += 1
                            self.deleted_count += 1
                            if deleted_count <= 5:
                                logger.info(f"🗑️ Auto-deleted: {item['title'][:40]}...")
                    
                    if deleted_count > 0:
                        logger.info(f"✅ Auto-deleted {deleted_count} files")
                    else:
                        logger.info("✅ No deleted files found")
                        
                except Exception as e:
                    logger.error(f"❌ Error checking messages: {e}")
                    
        except Exception as e:
            logger.error(f"❌ Auto-delete error: {e}")

sync_manager = OptimizedSyncManager()

# ============================================================================
# ✅ ASYNC CACHE DECORATOR
# ============================================================================

def async_cache_with_ttl(maxsize=128, ttl=300):
    """Async cache decorator with TTL"""
    cache = {}
    cache_lock = asyncio.Lock()
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            key_parts = [func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
            key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
            
            now = time.time()
            
            async with cache_lock:
                if key in cache:
                    value, timestamp = cache[key]
                    if now - timestamp < ttl:
                        return value
            
            result = await func(*args, **kwargs)
            
            async with cache_lock:
                cache[key] = (result, now)
                if len(cache) > maxsize:
                    oldest_key = min(cache.keys(), key=lambda k: cache[k][1])
                    del cache[oldest_key]
            
            return result
        return wrapper
    return decorator

# ============================================================================
# ✅ QUALITY DETECTION ENHANCED
# ============================================================================

QUALITY_PATTERNS = [
    (r'\b2160p\b|\b4k\b|\buhd\b', '2160p'),
    (r'\b1080p\b|\bfullhd\b|\bfhd\b', '1080p'),
    (r'\b720p\b|\bhd\b', '720p'),
    (r'\b480p\b', '480p'),
    (r'\b360p\b', '360p'),
]

HEVC_PATTERNS = [
    r'\bhevc\b',
    r'\bx265\b',
    r'\bh\.?265\b',
]

def detect_quality_enhanced(filename):
    """Enhanced quality detection with HEVC support"""
    if not filename:
        return "480p"
    
    filename_lower = filename.lower()
    is_hevc = any(re.search(pattern, filename_lower) for pattern in HEVC_PATTERNS)
    
    for pattern, quality in QUALITY_PATTERNS:
        if re.search(pattern, filename_lower):
            if is_hevc and quality in ['720p', '1080p', '2160p']:
                return f"{quality} HEVC"
            return quality
    
    return "480p"

def extract_quality_info(filename):
    """Extract detailed quality info"""
    quality = detect_quality_enhanced(filename)
    base_quality = quality
    is_hevc = 'HEVC' in quality
    if is_hevc:
        base_quality = quality.replace(' HEVC', '')
    
    return {
        'full': quality,
        'base': base_quality,
        'is_hevc': is_hevc,
        'priority': Config.QUALITY_PRIORITY.index(base_quality) if base_quality in Config.QUALITY_PRIORITY else 999
    }

# ============================================================================
# ✅ HOME MOVIES - HOME PRIORITY (Sources → Thumbnail → Fallback)
# ============================================================================

def channel_name_cached(cid):
    """Get channel name from cache or return default"""
    return f"Channel {cid}"

@performance_monitor.measure("home_movies")
@async_cache_with_ttl(maxsize=1, ttl=60)
async def get_home_movies(limit=25):
    """Get home movies with HOME PRIORITY: Sources → Thumbnail → Fallback"""
    try:
        if User is None or not user_session_ready:
            return []
        
        movies = []
        seen_titles = set()
        
        logger.info(f"🎬 [HOME] Fetching home movies ({limit})...")
        
        async for msg in User.get_chat_history(Config.MAIN_CHANNEL_ID, limit=25):
            if msg is not None and msg.text and len(msg.text) > 25:
                title = extract_title_smart(msg.text)
                
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    
                    # Extract year
                    year_match = re.search(r'\b(19|20)\d{2}\b', title)
                    year = year_match.group() if year_match else ""
                    
                    # Clean title
                    clean_title = re.sub(r'\s+\(\d{4}\)$', '', title)
                    clean_title = re.sub(r'\s+\d{4}$', '', clean_title)
                    
                    # Format content
                    post_content = msg.text
                    formatted_content = format_post(msg.text, max_length=500)
                    
                    # Get normalized title
                    norm_title = normalize_title(clean_title)
                    
                    # Get thumbnail with HOME PRIORITY
                    thumbnail_url = Config.FALLBACK_POSTER
                    has_thumbnail = True
                    thumbnail_source = 'fallback'
                    is_fallback = True
                    
                    if poster_fetcher:
                        thumbnail_data = await poster_fetcher.get_thumbnail_for_movie_home(clean_title, year)
                        if thumbnail_data:
                            thumbnail_url = thumbnail_data.get('poster_url', Config.FALLBACK_POSTER)
                            thumbnail_source = thumbnail_data.get('source', 'fallback')
                            has_thumbnail = thumbnail_data.get('has_thumbnail', True)
                            is_fallback = thumbnail_data.get('is_fallback', True)
                    
                    movie_data = {
                        'title': clean_title,
                        'original_title': title,
                        'normalized_title': norm_title,
                        'year': year,
                        'date': msg.date.isoformat() if isinstance(msg.date, datetime) else str(msg.date),
                        'is_new': is_new(msg.date) if msg.date else False,
                        'channel': channel_name_cached(Config.MAIN_CHANNEL_ID),
                        'channel_id': Config.MAIN_CHANNEL_ID,
                        'message_id': msg.id,
                        'has_file': False,
                        'has_post': True,
                        'content': formatted_content,
                        'post_content': post_content,
                        'quality_options': {},
                        'is_video_file': False,
                        'result_type': 'post_only',
                        'search_score': 1,
                        'has_poster': True,
                        'has_thumbnail': has_thumbnail,
                        'thumbnail_url': thumbnail_url,
                        'thumbnail_source': thumbnail_source,
                        'is_fallback': is_fallback,
                        'poster_url': thumbnail_url,
                        'poster_source': thumbnail_source,
                        'thumbnail_priority': 'home'
                    }
                    
                    logger.debug(f"📸 [HOME] '{clean_title[:30]}...': {thumbnail_source}")
                    movies.append(movie_data)
                    
                    if len(movies) >= limit:
                        break
        
        logger.info(f"✅ [HOME] Fetched {len(movies)} home movies")
        return movies[:limit]
        
    except Exception as e:
        logger.error(f"❌ [HOME] Home movies error: {e}")
        return [{
            'title': 'Error loading movies',
            'has_thumbnail': True,
            'thumbnail_url': Config.FALLBACK_POSTER,
            'thumbnail_source': 'error_fallback',
            'is_fallback': True
        }]

# ============================================================================
# ✅ SEARCH FUNCTION - SEARCH PRIORITY (Thumbnail → Sources → Fallback)
# ============================================================================

@performance_monitor.measure("enhanced_search_fixed")
@async_cache_with_ttl(maxsize=500, ttl=Config.SEARCH_CACHE_TTL)
async def search_movies_enhanced_fixed(query, limit=15, page=1):
    """FIXED: Combine post and file results with SEARCH PRIORITY: Thumbnail → Sources → Fallback"""
    offset = (page - 1) * limit
    
    # Try cache first
    cache_key = f"search_fixed:{query}:{page}:{limit}"
    if cache_manager is not None and cache_manager.redis_enabled:
        cached_data = await cache_manager.get(cache_key)
        if cached_data:
            logger.info(f"✅ Cache HIT for: {query}")
            return cached_data
    
    logger.info(f"🔍 [SEARCH] ENHANCED SEARCH for: {query}")
    
    query_lower = query.lower()
    
    # Main dictionary to hold merged results
    merged_results = {}
    
    # 1. SEARCH TEXT CHANNELS FOR POST RESULTS
    if user_session_ready and User is not None:
        logger.info(f"📝 [SEARCH] Searching TEXT CHANNELS for posts...")
        
        async def search_text_channel_posts(channel_id):
            try:
                cname = channel_name_cached(channel_id)
                async for msg in User.search_messages(channel_id, query=query, limit=10):
                    if msg is not None and msg.text and len(msg.text) > 15:
                        title = extract_title_smart(msg.text)
                        if title and (query_lower in title.lower() or query_lower in msg.text.lower()):
                            norm_title = normalize_title(title)
                            
                            # Get year
                            year_match = re.search(r'\b(19|20)\d{2}\b', title)
                            year = year_match.group() if year_match else ""
                            
                            # Clean title
                            clean_title = re.sub(r'\s+\(\d{4}\)$', '', title)
                            clean_title = re.sub(r'\s+\d{4}$', '', clean_title)
                            
                            post_data = {
                                'title': clean_title,
                                'original_title': title,
                                'normalized_title': norm_title,
                                'content': format_post(msg.text, max_length=500),
                                'post_content': msg.text,
                                'channel': cname,
                                'channel_id': channel_id,
                                'message_id': msg.id,
                                'date': msg.date.isoformat() if isinstance(msg.date, datetime) else str(msg.date),
                                'is_new': is_new(msg.date) if msg.date else False,
                                'has_file': False,
                                'has_post': True,
                                'quality_options': {},
                                'is_video_file': False,
                                'year': year,
                                'search_score': 3,
                                'result_type': 'post_only',
                                'thumbnail_url': Config.FALLBACK_POSTER,
                                'has_thumbnail': True,
                                'poster_url': Config.FALLBACK_POSTER,
                                'poster_source': 'fallback',
                                'combined': False,
                                'is_fallback': True
                            }
                            
                            merged_results[norm_title] = post_data
                            
            except Exception as e:
                logger.error(f"Text search error in {channel_id}: {e}")
            return True
        
        tasks = [search_text_channel_posts(channel_id) for channel_id in Config.TEXT_CHANNEL_IDS]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info(f"📝 [SEARCH] Found {len(merged_results)} POST results")
    
    # 2. SEARCH FILE CHANNEL DATABASE
    file_results_added = 0
    file_results_merged = 0
    
    if files_col is not None:
        try:
            logger.info(f"📁 [SEARCH] Searching FILE CHANNEL database...")
            
            search_query = {
                "$or": [
                    {"title": {"$regex": query, "$options": "i"}},
                    {"normalized_title": {"$regex": query, "$options": "i"}},
                    {"file_name": {"$regex": query, "$options": "i"}},
                    {"caption": {"$regex": query, "$options": "i"}}
                ],
                "status": "active"
            }
            
            cursor = files_col.find(
                search_query,
                {
                    'title': 1,
                    'normalized_title': 1,
                    'quality': 1,
                    'file_size': 1,
                    'file_name': 1,
                    'is_video_file': 1,
                    'channel_id': 1,
                    'message_id': 1,
                    'real_message_id': 1,
                    'date': 1,
                    'caption': 1,
                    'file_id': 1,
                    'telegram_file_id': 1,
                    'year': 1,
                    '_id': 0
                }
            ).limit(100)
            
            async for doc in cursor:
                try:
                    title = doc.get('title', 'Unknown')
                    norm_title = normalize_title(title)
                    
                    quality_info = extract_quality_info(doc.get('file_name', ''))
                    quality = quality_info['full']
                    
                    real_msg_id = doc.get('real_message_id') or doc.get('message_id')
                    year = doc.get('year', '')
                    
                    if norm_title in merged_results:
                        existing_result = merged_results[norm_title]
                        
                        existing_result.update({
                            'has_file': True,
                            'is_video_file': doc.get('is_video_file', False),
                            'file_caption': doc.get('caption', ''),
                            'real_message_id': real_msg_id,
                            'channel_id': doc.get('channel_id'),
                            'channel_name': channel_name_cached(doc.get('channel_id')),
                            'result_type': 'post_and_file',
                            'combined': True,
                            'search_score': 5
                        })
                        
                        existing_result['quality_options'][quality] = {
                            'quality': quality,
                            'file_size': doc.get('file_size', 0),
                            'message_id': real_msg_id,
                            'file_id': doc.get('file_id'),
                            'telegram_file_id': doc.get('telegram_file_id'),
                            'file_name': doc.get('file_name', '')
                        }
                        
                        file_results_merged += 1
                        logger.debug(f"✅ [SEARCH] Merged file with post: {title}")
                        
                    else:
                        file_result = {
                            'title': title,
                            'original_title': title,
                            'normalized_title': norm_title,
                            'content': format_post(doc.get('caption', ''), max_length=300),
                            'post_content': doc.get('caption', ''),
                            'quality_options': {quality: {
                                'quality': quality,
                                'file_size': doc.get('file_size', 0),
                                'message_id': real_msg_id,
                                'file_id': doc.get('file_id'),
                                'telegram_file_id': doc.get('telegram_file_id'),
                                'file_name': doc.get('file_name', '')
                            }},
                            'all_qualities': [quality],
                            'date': doc['date'].isoformat() if isinstance(doc['date'], datetime) else doc['date'],
                            'is_new': is_new(doc['date']) if doc.get('date') else False,
                            'is_video_file': doc.get('is_video_file', False),
                            'channel_id': doc.get('channel_id'),
                            'channel_name': channel_name_cached(doc.get('channel_id')),
                            'has_file': True,
                            'has_post': bool(doc.get('caption')),
                            'file_caption': doc.get('caption', ''),
                            'year': year,
                            'quality': quality,
                            'real_message_id': real_msg_id,
                            'search_score': 2,
                            'result_type': 'file_only',
                            'quality_count': 1,
                            'thumbnail_url': Config.FALLBACK_POSTER,
                            'poster_url': Config.FALLBACK_POSTER,
                            'poster_source': 'fallback',
                            'combined': False,
                            'has_thumbnail': True,
                            'is_fallback': True
                        }
                        
                        merged_results[norm_title] = file_result
                        file_results_added += 1
                        
                except Exception as e:
                    logger.error(f"File processing error: {e}")
                    continue
            
            logger.info(f"📁 [SEARCH] Found {file_results_added} FILE-ONLY results, merged {file_results_merged} with existing posts")
            
        except Exception as e:
            logger.error(f"❌ [SEARCH] File search error: {e}")
    
    # 3. GET THUMBNAILS WITH SEARCH PRIORITY
    if poster_fetcher:
        logger.info(f"🖼️ [SEARCH] Getting thumbnails for {len(merged_results)} results with SEARCH priority...")
        
        movies_for_thumbnails = []
        for norm_title, result in merged_results.items():
            movie_data = {
                'title': result.get('title', ''),
                'year': result.get('year', ''),
                'channel_id': result.get('channel_id'),
                'message_id': result.get('real_message_id') or result.get('message_id'),
                'result_type': result.get('result_type', 'unknown')
            }
            movies_for_thumbnails.append(movie_data)
        
        if movies_for_thumbnails:
            movies_with_thumbnails = await poster_fetcher.get_thumbnails_batch(movies_for_thumbnails, mode="search")
            
            for i, (norm_title, result) in enumerate(merged_results.items()):
                if i < len(movies_with_thumbnails):
                    thumbnail_data = movies_with_thumbnails[i]
                    
                    has_thumbnail = thumbnail_data.get('has_thumbnail', False)
                    poster_url = thumbnail_data.get('poster_url', Config.FALLBACK_POSTER)
                    thumbnail_source = thumbnail_data.get('source', 'fallback')
                    is_fallback = thumbnail_data.get('is_fallback', True)
                    
                    result.update({
                        'thumbnail_url': poster_url,
                        'thumbnail_source': thumbnail_source,
                        'has_thumbnail': has_thumbnail,
                        'thumbnail_extracted': thumbnail_data.get('extracted', False),
                        'poster_url': poster_url,
                        'poster_source': thumbnail_source,
                        'is_fallback': is_fallback,
                        'thumbnail_priority': 'search'
                    })
                    
                    title_short = result.get('title', '')[:30]
                    logger.debug(f"📸 [SEARCH] '{title_short}...': {thumbnail_source} (fallback: {is_fallback})")
            
            logger.info(f"✅ [SEARCH] Got thumbnails for {len(movies_for_thumbnails)} movies")
    
    # 4. ENSURE ALL RESULTS HAVE THUMBNAILS
    for norm_title, result in merged_results.items():
        if not result.get('thumbnail_url'):
            result.update({
                'thumbnail_url': Config.FALLBACK_POSTER,
                'thumbnail_source': 'fallback_ensure',
                'has_thumbnail': True,
                'is_fallback': True,
                'poster_url': Config.FALLBACK_POSTER,
                'poster_source': 'fallback_ensure'
            })
    
    # 5. CONVERT TO LIST AND SORT
    all_results = list(merged_results.values())
    
    if not all_results:
        result_data = {
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
                'query': query,
                'stats': {
                    'total': 0,
                    'post_only': 0,
                    'file_only': 0,
                    'post_and_file': 0
                },
                'post_file_merged': True,
                'file_only_with_poster': True,
                'thumbnails_enabled': True,
                'poster_fetcher': poster_fetcher is not None,
                'real_message_ids': True,
                'search_logic': 'enhanced_fixed_with_merging',
                'thumbnail_priority': 'search',
                'priority_system': 'Thumbnail → Sources → Fallback',
                'fallback_enabled': True
            },
            'bot_username': Config.BOT_USERNAME
        }
        
        if cache_manager is not None:
            await cache_manager.set(cache_key, result_data, expire_seconds=Config.SEARCH_CACHE_TTL)
        
        return result_data
    
    # 6. SORT RESULTS
    all_results.sort(key=lambda x: (
        x.get('result_type') == 'post_and_file',
        x.get('result_type') == 'post_only',
        x.get('search_score', 0),
        x.get('is_new', False),
        x.get('date', '') if isinstance(x.get('date'), str) else ''
    ), reverse=True)
    
    # 7. PAGINATION
    total = len(all_results)
    start_idx = offset
    end_idx = offset + limit
    paginated = all_results[start_idx:end_idx]
    
    stats = {
        'total': total,
        'post_only': sum(1 for r in all_results if r.get('result_type') == 'post_only'),
        'file_only': sum(1 for r in all_results if r.get('result_type') == 'file_only'),
        'post_and_file': sum(1 for r in all_results if r.get('result_type') == 'post_and_file'),
        'file_results_added': file_results_added,
        'file_results_merged': file_results_merged
    }
    
    poster_stats = {}
    if poster_fetcher:
        poster_stats = await poster_fetcher.get_stats()
    
    logger.info(f"📊 [SEARCH] FINAL RESULTS: {total} total")
    
    # 8. FINAL DATA STRUCTURE
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
            'query': query,
            'stats': stats,
            'poster_stats': poster_stats,
            'post_file_merged': True,
            'file_only_with_poster': True,
            'poster_fetcher': poster_fetcher is not None,
            'thumbnails_enabled': True,
            'thumbnail_system': poster_fetcher is not None,
            'real_message_ids': True,
            'search_logic': 'enhanced_fixed_with_merging',
            'thumbnail_priority': 'search',
            'priority_system': 'Thumbnail → Sources → Fallback',
            'fallback_enabled': True
        },
        'bot_username': Config.BOT_USERNAME
    }
    
    if cache_manager is not None:
        await cache_manager.set(cache_key, result_data, expire_seconds=Config.SEARCH_CACHE_TTL)
    
    logger.info(f"✅ [SEARCH] Search complete: {len(paginated)} results (page {page})")
    
    return result_data

# ============================================================================
# ✅ BACKWARD COMPATIBLE SEARCH
# ============================================================================

@async_cache_with_ttl(maxsize=500, ttl=Config.SEARCH_CACHE_TTL)
async def search_movies_multi_channel_merged(query, limit=15, page=1):
    """Backward compatible search"""
    return await search_movies_enhanced_fixed(query, limit, page)

# ============================================================================
# ✅ OPTIMIZED FILE INDEXING
# ============================================================================

async def extract_title_fast(filename, caption):
    """Fast title extraction"""
    if filename:
        name = os.path.splitext(filename)[0]
        name = re.sub(r'[._]', ' ', name)
        name = re.sub(r'\b(480p|720p|1080p|2160p|4k|hd|hevc)\b', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\s+', ' ', name)
        name = name.strip()
        if name and len(name) > 3:
            return name[:100]
    if caption:
        lines = caption.split('\n')
        for line in lines:
            line = line.strip()
            if len(line) > 10 and not line.startswith('http'):
                return line[:100]
    if filename:
        return os.path.splitext(filename)[0][:50]
    return "Unknown File"

async def index_single_file_fast(message):
    """Fast file indexing"""
    try:
        if files_col is None:
            return False, None
        
        if not message or (not message.document and not message.video):
            return False, None
        
        caption = message.caption if hasattr(message, 'caption') else None
        file_name = None
        
        if message.document:
            file_name = message.document.file_name
        elif message.video:
            file_name = message.video.file_name
        
        title = await extract_title_fast(file_name, caption)
        if not title or title == "Unknown File":
            return False, None
        
        normalized_title = normalize_title(title)
        quality = detect_quality_enhanced(file_name or "")
        
        doc = {
            'channel_id': Config.FILE_CHANNEL_ID,
            'message_id': message.id,
            'real_message_id': message.id,
            'title': title,
            'normalized_title': normalized_title,
            'date': message.date,
            'indexed_at': datetime.now(),
            'last_checked': datetime.now(),
            'is_video_file': is_video_file(file_name or '') if file_name else False,
            'file_id': None,
            'file_size': 0,
            'status': 'active',
            'quality': quality
        }
        
        if message.document:
            doc.update({
                'file_name': message.document.file_name or '',
                'caption': caption or '',
                'file_id': message.document.file_id,
                'telegram_file_id': message.document.file_id,
                'file_size': message.document.file_size or 0,
                'is_video_file': is_video_file(message.document.file_name or '')
            })
        elif message.video:
            doc.update({
                'file_name': message.video.file_name or 'video.mp4',
                'caption': caption or '',
                'file_id': message.video.file_id,
                'telegram_file_id': message.video.file_id,
                'file_size': message.video.file_size or 0,
                'is_video_file': True
            })
        
        await files_col.insert_one(doc)
        logger.info(f"✅ INDEXED: {title[:50]}... (ID: {message.id})")
        
        # Extract thumbnail for this file
        if poster_fetcher and Config.THUMBNAIL_EXTRACTION_ENABLED and bot_handler and bot_handler.initialized:
            try:
                asyncio.create_task(
                    poster_fetcher.get_thumbnail_for_movie_search(
                        title=title,
                        channel_id=Config.FILE_CHANNEL_ID,
                        message_id=message.id
                    )
                )
                logger.debug(f"🖼️ Thumbnail extraction queued for: {title[:30]}...")
            except Exception as e:
                logger.error(f"❌ Thumbnail queue error: {e}")
        
        return True, normalized_title
        
    except Exception as e:
        if "duplicate key error" in str(e).lower():
            return False, None
        logger.error(f"❌ Indexing error: {e}")
        return False, None

async def setup_database_indexes():
    """Setup database indexes"""
    if files_col is None:
        return
    
    try:
        await files_col.create_index(
            [("channel_id", 1), ("message_id", 1)],
            unique=True,
            name="channel_message_unique",
            background=True
        )
        await files_col.create_index(
            [("normalized_title", "text")],
            name="title_text_search",
            background=True
        )
        await files_col.create_index(
            [("normalized_title", 1)],
            name="normalized_title_index",
            background=True
        )
        logger.info("✅ Created files collection indexes")
    except Exception as e:
        logger.warning(f"⚠️ Index creation error: {e}")

class OptimizedFileIndexingManager:
    """Optimized file channel indexing manager"""
    
    def __init__(self):
        self.is_running = False
        self.indexing_task = None
        self.last_run = None
        self.next_run = None
        self.total_indexed = 0
        self.total_skipped = 0
        self.indexing_stats = {
            'total_runs': 0,
            'total_files_processed': 0,
            'total_indexed': 0,
            'total_skipped': 0,
            'total_errors': 0,
            'last_success': None
        }
    
    async def start_indexing(self):
        """Start file channel indexing"""
        if self.is_running:
            logger.warning("⚠️ File indexing already running")
            return
        
        logger.info("🚀 Starting FILE CHANNEL INDEXING...")
        self.is_running = True
        asyncio.create_task(self._run_optimized_indexing())
        self.indexing_task = asyncio.create_task(self._indexing_loop())
    
    async def stop_indexing(self):
        """Stop indexing"""
        self.is_running = False
        if self.indexing_task:
            self.indexing_task.cancel()
            try: await self.indexing_task
            except: pass
        logger.info("🛑 File indexing stopped")
    
    async def _run_optimized_indexing(self):
        """Run optimized indexing - NEW MESSAGES ONLY"""
        logger.info("🔥 RUNNING OPTIMIZED INDEXING...")
        
        try:
            last_indexed = await files_col.find_one(
                {"channel_id": Config.FILE_CHANNEL_ID}, 
                sort=[('message_id', -1)],
                projection={'message_id': 1}
            )
            
            last_message_id = last_indexed['message_id'] if last_indexed else 0
            logger.info(f"📊 Last indexed message ID: {last_message_id}")
            
            messages_to_index = []
            total_fetched = 0
            
            try:
                async for msg in User.get_chat_history(
                    Config.FILE_CHANNEL_ID, 
                    limit=Config.BATCH_INDEX_SIZE
                ):
                    total_fetched += 1
                    if msg.id <= last_message_id:
                        break
                    if msg and (msg.document or msg.video):
                        messages_to_index.append(msg)
                    if Config.MAX_INDEX_LIMIT > 0 and total_fetched >= Config.MAX_INDEX_LIMIT:
                        break
                
                logger.info(f"📥 Fetched {total_fetched} messages, found {len(messages_to_index)} new files")
                
            except Exception as e:
                logger.error(f"❌ Error fetching messages: {e}")
                return
            
            if messages_to_index:
                messages_to_index.reverse()
                
                batch_size = 50
                total_batches = math.ceil(len(messages_to_index) / batch_size)
                logger.info(f"🔧 Processing {len(messages_to_index)} new files in {total_batches} batches...")
                
                for batch_num in range(total_batches):
                    start_idx = batch_num * batch_size
                    end_idx = min(start_idx + batch_size, len(messages_to_index))
                    batch = messages_to_index[start_idx:end_idx]
                    logger.info(f"📦 Processing batch {batch_num + 1}/{total_batches}...")
                    
                    for msg in batch:
                        try:
                            success, _ = await index_single_file_fast(msg)
                            if success:
                                self.indexing_stats['total_indexed'] += 1
                                self.total_indexed += 1
                            else:
                                self.indexing_stats['total_skipped'] += 1
                                self.total_skipped += 1
                        except Exception as e:
                            logger.error(f"❌ Error indexing {msg.id}: {e}")
                            self.indexing_stats['total_errors'] += 1
                    
                    if batch_num < total_batches - 1:
                        await asyncio.sleep(1)
                
                logger.info("✅ OPTIMIZED INDEXING FINISHED!")
                logger.info(f"📊 Stats: {self.indexing_stats}")
            
        except Exception as e:
            logger.error(f"❌ Optimized indexing error: {e}")
    
    async def _indexing_loop(self):
        """Main indexing loop"""
        while self.is_running:
            try:
                if self.next_run and self.next_run > datetime.now():
                    wait_seconds = (self.next_run - datetime.now()).total_seconds()
                    if wait_seconds > 30:
                        logger.info(f"⏰ Next index in {wait_seconds:.0f}s")
                    await asyncio.sleep(min(wait_seconds, 30))
                    continue
                
                await self._run_optimized_indexing()
                self.next_run = datetime.now() + timedelta(seconds=Config.AUTO_INDEX_INTERVAL)
                self.last_run = datetime.now()
                await asyncio.sleep(10)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Indexing loop error: {e}")
                await asyncio.sleep(60)
    
    async def get_indexing_status(self):
        """Get current indexing status"""
        poster_stats = {}
        if poster_fetcher:
            poster_stats = await poster_fetcher.get_stats()
        
        return {
            'is_running': self.is_running,
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'next_run': self.next_run.isoformat() if self.next_run else None,
            'total_indexed': self.total_indexed,
            'total_skipped': self.total_skipped,
            'stats': self.indexing_stats,
            'thumbnail_stats': poster_stats
        }

file_indexing_manager = OptimizedFileIndexingManager()

async def initial_indexing_optimized():
    """Optimized initial indexing"""
    if User is None or files_col is None or not user_session_ready:
        logger.warning("⚠️ User session not ready for initial indexing")
        return
    
    logger.info("=" * 60)
    logger.info("🚀 STARTING OPTIMIZED FILE CHANNEL INDEXING")
    logger.info("=" * 60)
    
    try:
        await setup_database_indexes()
        await file_indexing_manager.start_indexing()
        await sync_manager.start_sync_monitoring()
    except Exception as e:
        logger.error(f"❌ Initial indexing error: {e}")

async def extract_thumbnails_for_existing_files():
    """Extract thumbnails for existing files"""
    if not poster_fetcher or files_col is None:
        logger.warning("⚠️ PosterFetcher or files collection not available")
        return
    
    logger.info("🔄 Extracting thumbnails for existing files...")
    
    try:
        cursor = files_col.find({
            'is_video_file': True,
            'channel_id': Config.FILE_CHANNEL_ID
        }, {
            'title': 1,
            'normalized_title': 1,
            'channel_id': 1,
            'message_id': 1,
            'real_message_id': 1,
            '_id': 1
        }).limit(500)
        
        files_to_process = []
        async for doc in cursor:
            files_to_process.append({
                'title': doc.get('title', ''),
                'normalized_title': doc.get('normalized_title', ''),
                'channel_id': doc.get('channel_id'),
                'message_id': doc.get('real_message_id') or doc.get('message_id'),
                'db_id': doc.get('_id')
            })
        
        logger.info(f"📊 Found {len(files_to_process)} video files to process")
        
        if not files_to_process:
            logger.info("✅ No files need thumbnail extraction")
            return
        
        batch_size = 10
        total_batches = math.ceil(len(files_to_process) / batch_size)
        successful = 0
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(files_to_process))
            batch = files_to_process[start_idx:end_idx]
            
            logger.info(f"🖼️ Processing batch {batch_num + 1}/{total_batches}...")
            
            thumbnail_results = await poster_fetcher.get_thumbnails_batch(batch, mode="search")
            
            for i, file_info in enumerate(batch):
                if i < len(thumbnail_results) and thumbnail_results[i].get('poster_url'):
                    if thumbnail_results[i].get('source') not in ['fallback', 'error']:
                        successful += 1
            
            if batch_num < total_batches - 1:
                await asyncio.sleep(1)
        
        logger.info(f"✅ Thumbnail extraction complete: {successful} successful")
        
    except Exception as e:
        logger.error(f"❌ Error extracting thumbnails: {e}")

# ============================================================================
# ✅ TELEGRAM SESSION INITIALIZATION
# ============================================================================

@performance_monitor.measure("telegram_init")
async def init_telegram_sessions():
    """Initialize Telegram sessions"""
    global User, Bot, user_session_ready, bot_session_ready
    
    logger.info("=" * 50)
    logger.info("🚀 TELEGRAM SESSION INITIALIZATION")
    logger.info("=" * 50)
    
    if not PYROGRAM_AVAILABLE:
        logger.error("❌ Pyrogram not installed!")
        return False
    
    if Config.API_ID > 0 and Config.API_HASH and Config.USER_SESSION_STRING:
        logger.info("\n👤 Initializing USER Session...")
        try:
            User = Client(
                "sk4film_user",
                api_id=Config.API_ID,
                api_hash=Config.API_HASH,
                session_string=Config.USER_SESSION_STRING,
                sleep_threshold=30,
                in_memory=True,
                no_updates=True
            )
            
            await User.start()
            me = await User.get_me()
            logger.info(f"✅ USER Session Ready: {me.first_name}")
            
            try:
                chat = await User.get_chat(Config.FILE_CHANNEL_ID)
                logger.info(f"✅ File Channel Access: {chat.title}")
                user_session_ready = True
            except Exception as e:
                logger.error(f"❌ File channel access failed: {e}")
                user_session_ready = False
                
        except Exception as e:
            logger.error(f"❌ USER Session failed: {e}")
            user_session_ready = False
            if User is not None:
                try: await User.stop()
                except: pass
            User = None
    
    if Config.BOT_TOKEN:
        logger.info("\n🤖 Initializing BOT Session...")
        try:
            Bot = Client(
                "sk4film_bot",
                api_id=Config.API_ID,
                api_hash=Config.API_HASH,
                bot_token=Config.BOT_TOKEN,
                sleep_threshold=30,
                in_memory=True,
                no_updates=True
            )
            
            await Bot.start()
            bot_info = await Bot.get_me()
            logger.info(f"✅ BOT Session Ready: @{bot_info.username}")
            bot_session_ready = True
                
        except Exception as e:
            logger.error(f"❌ BOT Session failed: {e}")
            bot_session_ready = False
            if Bot is not None:
                try: await Bot.stop()
                except: pass
            Bot = None
    
    logger.info("\n" + "=" * 50)
    logger.info("📊 SESSION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"USER Session: {'✅ READY' if user_session_ready else '❌ NOT READY'}")
    logger.info(f"BOT Session: {'✅ READY' if bot_session_ready else '❌ NOT READY'}")
    
    return user_session_ready or bot_session_ready

# ============================================================================
# ✅ MONGODB INITIALIZATION
# ============================================================================

@performance_monitor.measure("mongodb_init")
async def init_mongodb():
    global mongo_client, db, files_col, verification_col
    
    try:
        logger.info("🔌 MongoDB initialization...")
        
        mongo_client = AsyncIOMotorClient(
            Config.MONGODB_URI,
            serverSelectionTimeoutMS=10000,
            connectTimeoutMS=10000,
            socketTimeoutMS=15000,
            maxPoolSize=20,
            minPoolSize=5,
            retryWrites=True,
            retryReads=True
        )
        
        await asyncio.wait_for(mongo_client.admin.command('ping'), timeout=5)
        
        db = mongo_client.sk4film
        files_col = db.files
        verification_col = db.verifications
        
        logger.info("✅ MongoDB OK")
        return True
        
    except asyncio.TimeoutError:
        logger.error("❌ MongoDB connection timeout")
        return False
    except Exception as e:
        logger.error(f"❌ MongoDB error: {e}")
        return False

# ============================================================================
# ✅ TELEGRAM BOT INITIALIZATION
# ============================================================================

async def start_telegram_bot():
    """Start Telegram bot with handlers"""
    try:
        if not PYROGRAM_AVAILABLE:
            logger.warning("❌ Pyrogram not available, bot won't start")
            return None
        
        if not Config.BOT_TOKEN:
            logger.warning("❌ Bot token not configured")
            return None
        
        logger.info("🤖 Starting SK4FiLM Telegram Bot...")
        
        try:
            from bot_handlers import SK4FiLMBot
            logger.info("✅ Bot handler module imported")
        except ImportError as e:
            logger.error(f"❌ Bot handler import error: {e}")
            class FallbackBot:
                def __init__(self):
                    self.bot_started = False
                async def initialize(self): return False
                async def shutdown(self): pass
            return FallbackBot()
        
        bot_instance = SK4FiLMBot(Config, db_manager=None)
        bot_started = await bot_instance.initialize()
        
        if bot_started:
            logger.info("✅ Telegram Bot started successfully!")
            return bot_instance
        else:
            logger.error("❌ Failed to start Telegram Bot")
            return None
            
    except Exception as e:
        logger.error(f"❌ Bot startup error: {e}")
        return None

# ============================================================================
# ✅ MAIN INITIALIZATION - DUAL PRIORITY SYSTEM
# ============================================================================

@performance_monitor.measure("system_init")
async def init_system():
    start_time = time.time()
    
    try:
        logger.info("=" * 60)
        logger.info("🚀 SK4FiLM v9.0 - DUAL PRIORITY THUMBNAIL SYSTEM")
        logger.info("=" * 60)
        
        # Initialize MongoDB
        mongo_ok = await init_mongodb()
        if not mongo_ok:
            logger.error("❌ MongoDB connection failed")
            return False
        
        # Get current file count
        if files_col is not None:
            file_count = await files_col.count_documents({})
            logger.info(f"📊 Current files in database: {file_count}")
        
        # Initialize Bot Handler
        bot_handler_ok = await bot_handler.initialize()
        if bot_handler_ok:
            logger.info("✅ Bot Handler initialized")
        
        # Start Telegram Bot
        global telegram_bot
        telegram_bot = await start_telegram_bot()
        if telegram_bot:
            logger.info("✅ Telegram Bot started successfully")
        else:
            logger.warning("⚠️ Telegram Bot failed to start")
        
        # Initialize Cache Manager
        global cache_manager, verification_system, premium_system, poster_fetcher
        cache_manager = CacheManager(Config)
        redis_ok = await cache_manager.init_redis()
        if redis_ok:
            logger.info("✅ Cache Manager initialized")
            await cache_manager.start_cleanup_task()
        
        # Initialize Verification System
        if VerificationSystem is not None:
            verification_system = VerificationSystem(Config, mongo_client)
            logger.info("✅ Verification System initialized")
        
        # Initialize Premium System
        if PremiumSystem is not None:
            premium_system = PremiumSystem(Config, mongo_client)
            logger.info("✅ Premium System initialized")
        
        # Initialize Poster Fetcher with Dual Priority
        if POSTER_FETCHER_AVAILABLE:
            poster_fetcher = PosterFetcher(Config, cache_manager, bot_handler, mongo_client)
            logger.info("✅ Poster Fetcher with Dual Priority initialized")
        else:
            logger.warning("⚠️ PosterFetcher not available, using fallback")
            poster_fetcher = PosterFetcher(Config, cache_manager, bot_handler, mongo_client)
        
        # Initialize Telegram Sessions
        if PYROGRAM_AVAILABLE:
            telegram_ok = await init_telegram_sessions()
            if not telegram_ok:
                logger.warning("⚠️ Telegram sessions failed")
        
        # Extract thumbnails for existing files
        if poster_fetcher and files_col is not None:
            logger.info("🔄 Extracting thumbnails for existing files...")
            asyncio.create_task(extract_thumbnails_for_existing_files())
        
        # Start OPTIMIZED indexing
        if user_session_ready and files_col is not None:
            logger.info("🔄 Starting OPTIMIZED file channel indexing...")
            asyncio.create_task(initial_indexing_optimized())
        
        init_time = time.time() - start_time
        logger.info(f"⚡ SK4FiLM Started in {init_time:.2f}s")
        logger.info("=" * 60)
        
        logger.info("🔧 DUAL PRIORITY THUMBNAIL SYSTEM:")
        logger.info(f"   • HOME MOVIES: Sources → Thumbnail → Fallback")
        logger.info(f"   • SEARCH RESULTS: Thumbnail → Sources → Fallback")
        logger.info(f"   • Fallback Image: ✅ ENABLED")
        logger.info(f"   • PosterFetcher: {'✅ ENABLED' if poster_fetcher else '❌ DISABLED'}")
        logger.info(f"   • Thumbnail Success Rate Target: 99%")
        logger.info(f"   • File Channel ID: {Config.FILE_CHANNEL_ID}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ System initialization failed: {e}")
        return False

# ============================================================================
# ✅ API ROUTES
# ============================================================================

@app.route('/')
@performance_monitor.measure("root_endpoint")
async def root():
    if files_col is not None:
        tf = await files_col.count_documents({})
        video_files = await files_col.count_documents({'is_video_file': True})
    else:
        tf = 0
        video_files = 0
    
    # Get poster stats
    poster_stats = {}
    if poster_fetcher:
        poster_stats = await poster_fetcher.get_stats()
    
    # Get indexing status
    indexing_status = await file_indexing_manager.get_indexing_status()
    
    # Get bot status
    bot_status = None
    if bot_handler:
        try:
            bot_status = await bot_handler.get_bot_status()
        except:
            bot_status = {'initialized': False}
    
    # Get Telegram bot status
    bot_running = telegram_bot is not None and hasattr(telegram_bot, 'bot_started') and telegram_bot.bot_started
    
    return jsonify({
        'status': 'healthy',
        'service': 'SK4FiLM v9.0 - DUAL PRIORITY THUMBNAIL SYSTEM',
        'thumbnail_system': {
            'home_mode': 'Sources → Thumbnail → Fallback',
            'search_mode': 'Thumbnail → Sources → Fallback',
            'fallback_enabled': True,
            'fallback_url': Config.FALLBACK_POSTER,
            'poster_fetcher': poster_fetcher is not None,
            'success_rate_target': '99%'
        },
        'sessions': {
            'user_session': {'ready': user_session_ready},
            'bot_session': {'ready': bot_session_ready},
            'bot_handler': bot_status,
            'telegram_bot': {'running': bot_running}
        },
        'components': {
            'cache': cache_manager is not None,
            'verification': verification_system is not None,
            'premium': premium_system is not None,
            'poster_fetcher': poster_fetcher is not None,
            'database': files_col is not None
        },
        'stats': {
            'total_files': tf,
            'video_files': video_files
        },
        'indexing': indexing_status,
        'sync_monitoring': {
            'running': sync_manager.is_monitoring,
            'deleted_count': sync_manager.deleted_count
        },
        'thumbnail_manager': poster_stats,
        'response_time': f"{time.perf_counter():.3f}s"
    })

@app.route('/health')
@performance_monitor.measure("health_endpoint")
async def health():
    indexing_status = await file_indexing_manager.get_indexing_status()
    
    bot_status = None
    if bot_handler:
        try:
            bot_status = await bot_handler.get_bot_status()
        except:
            bot_status = {'initialized': False}
    
    poster_stats = {}
    if poster_fetcher:
        poster_stats = await poster_fetcher.get_stats()
    
    return jsonify({
        'status': 'ok',
        'thumbnail_system': True,
        'dual_priority': True,
        'fallback_enabled': True,
        'sessions': {
            'user': user_session_ready,
            'bot': bot_session_ready,
            'bot_handler': bot_status.get('initialized') if bot_status else False,
            'telegram_bot': telegram_bot is not None
        },
        'indexing': {
            'running': indexing_status['is_running'],
            'last_run': indexing_status['last_run']
        },
        'thumbnails': poster_stats,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/movies', methods=['GET'])
@performance_monitor.measure("movies_endpoint")
async def api_movies():
    try:
        movies = await get_home_movies(limit=25)
        
        return jsonify({
            'status': 'success' if movies else 'empty',
            'movies': movies,
            'total': len(movies),
            'limit': 25,
            'source': 'telegram',
            'thumbnail_priority': 'home',
            'priority_system': 'Sources → Thumbnail → Fallback',
            'fallback_enabled': True,
            'poster_fetcher': poster_fetcher is not None,
            'session_used': 'user',
            'channel_id': Config.MAIN_CHANNEL_ID,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"[HOME] Movies API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'movies': [],
            'total': 0
        }), 500

@app.route('/api/search', methods=['GET'])
@performance_monitor.measure("search_endpoint")
async def api_search():
    try:
        query = request.args.get('query', '').strip()
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', Config.SEARCH_RESULTS_PER_PAGE))
        
        if len(query) < Config.SEARCH_MIN_QUERY_LENGTH:
            return jsonify({
                'status': 'error',
                'message': f'Query must be at least {Config.SEARCH_MIN_QUERY_LENGTH} characters'
            }), 400
        
        result_data = await search_movies_enhanced_fixed(query, limit, page)
        
        return jsonify({
            'status': 'success',
            'query': query,
            'results': result_data['results'],
            'pagination': result_data['pagination'],
            'search_metadata': {
                **result_data.get('search_metadata', {}),
                'feature': 'post_first_search',
                'thumbnail_priority': 'search',
                'priority_system': 'Thumbnail → Sources → Fallback',
                'fallback_enabled': True,
                'real_message_ids': True,
            },
            'bot_username': Config.BOT_USERNAME,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"[SEARCH] Search API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/stats', methods=['GET'])
async def api_stats():
    """Get performance statistics"""
    try:
        perf_stats = performance_monitor.get_stats()
        
        if poster_fetcher:
            poster_stats = await poster_fetcher.get_stats()
        else:
            poster_stats = {}
        
        if files_col is not None:
            total_files = await files_col.count_documents({})
            video_files = await files_col.count_documents({'is_video_file': True})
            indexing_status = await file_indexing_manager.get_indexing_status()
            sync_stats = {
                'running': sync_manager.is_monitoring,
                'deleted_count': sync_manager.deleted_count,
                'last_sync': sync_manager.last_sync
            }
        else:
            total_files = 0
            video_files = 0
            indexing_status = {}
            sync_stats = {}
        
        bot_status = None
        if bot_handler:
            try:
                bot_status = await bot_handler.get_bot_status()
            except:
                bot_status = {'initialized': False}
        
        bot_running = telegram_bot is not None and hasattr(telegram_bot, 'bot_started') and telegram_bot.bot_started
        
        return jsonify({
            'status': 'success',
            'thumbnail_system': {
                'dual_priority': True,
                'home_mode': 'Sources → Thumbnail → Fallback',
                'search_mode': 'Thumbnail → Sources → Fallback',
                'fallback_enabled': True,
                'fallback_url': Config.FALLBACK_POSTER
            },
            'performance': perf_stats,
            'poster_fetcher': poster_stats,
            'database_stats': {
                'total_files': total_files,
                'video_files': video_files
            },
            'indexing_stats': indexing_status,
            'sync_stats': sync_stats,
            'bot_handler': bot_status,
            'telegram_bot': {'running': bot_running},
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Stats API error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# ============================================================================
# ✅ ADMIN API ROUTES
# ============================================================================

@app.route('/api/admin/reindex', methods=['POST'])
async def api_admin_reindex():
    """Admin endpoint to trigger reindexing"""
    try:
        auth_token = request.headers.get('X-Admin-Token')
        if not auth_token or auth_token != os.environ.get('ADMIN_TOKEN', 'sk4film_admin_123'):
            return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401
        
        asyncio.create_task(initial_indexing_optimized())
        
        return jsonify({
            'status': 'success',
            'message': 'File channel reindexing started',
            'thumbnail_system': True,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Admin reindex error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/admin/indexing-status', methods=['GET'])
async def api_admin_indexing_status():
    """Check indexing status"""
    try:
        indexing_status = await file_indexing_manager.get_indexing_status()
        
        if files_col is not None:
            total_files = await files_col.count_documents({})
        else:
            total_files = 0
        
        return jsonify({
            'status': 'success',
            'thumbnail_system': True,
            'indexing': indexing_status,
            'database_files': total_files,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Indexing status error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/admin/clear-cache', methods=['POST'])
async def api_admin_clear_cache():
    """Clear all cache"""
    try:
        auth_token = request.headers.get('X-Admin-Token')
        if not auth_token or auth_token != os.environ.get('ADMIN_TOKEN', 'sk4film_admin_123'):
            return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401
        
        if cache_manager and cache_manager.redis_enabled:
            try:
                keys = await cache_manager.redis_client.keys("search_*")
                if keys:
                    await cache_manager.redis_client.delete(*keys)
                    logger.info(f"✅ Cleared {len(keys)} search cache keys")
            except Exception as e:
                logger.error(f"❌ Cache clear error: {e}")
        
        return jsonify({'status': 'success', 'message': 'Cache cleared successfully'})
    except Exception as e:
        logger.error(f"❌ Clear cache error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/admin/thumbnails/extract-existing', methods=['POST'])
async def api_admin_thumbnails_extract_existing():
    """Admin endpoint to trigger thumbnail extraction for existing files"""
    try:
        auth_token = request.headers.get('X-Admin-Token')
        if not auth_token or auth_token != os.environ.get('ADMIN_TOKEN', 'sk4film_admin_123'):
            return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401
        
        if not poster_fetcher:
            return jsonify({'status': 'error', 'message': 'PosterFetcher not initialized'}), 400
        
        asyncio.create_task(extract_thumbnails_for_existing_files())
        
        return jsonify({
            'status': 'success',
            'message': 'Thumbnail extraction started for existing files',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Thumbnail extraction error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# ============================================================================
# ✅ STARTUP AND SHUTDOWN
# ============================================================================

app_start_time = time.time()

@app.before_serving
async def startup():
    await init_system()

@app.after_serving
async def shutdown():
    logger.info("🛑 Shutting down SK4FiLM v9.0...")
    
    shutdown_tasks = []
    
    if telegram_bot:
        try:
            if hasattr(telegram_bot, 'shutdown'):
                await telegram_bot.shutdown()
                logger.info("✅ Telegram Bot stopped")
        except Exception as e:
            logger.error(f"❌ Telegram Bot shutdown error: {e}")
    
    await file_indexing_manager.stop_indexing()
    await sync_manager.stop_sync_monitoring()
    
    if poster_fetcher:
        try:
            await poster_fetcher.close()
            logger.info("✅ Poster Fetcher closed")
        except Exception as e:
            logger.error(f"❌ Poster Fetcher close error: {e}")
    
    if bot_handler:
        try:
            await bot_handler.shutdown()
            logger.info("✅ Bot Handler stopped")
        except Exception as e:
            logger.error(f"❌ Bot Handler shutdown error: {e}")
    
    if User is not None:
        shutdown_tasks.append(User.stop())
    if Bot is not None:
        shutdown_tasks.append(Bot.stop())
    if cache_manager is not None:
        shutdown_tasks.append(cache_manager.stop())
    if verification_system is not None:
        shutdown_tasks.append(verification_system.stop())
    if premium_system is not None and hasattr(premium_system, 'stop_cleanup_task'):
        shutdown_tasks.append(premium_system.stop_cleanup_task())
    
    if shutdown_tasks:
        await asyncio.gather(*shutdown_tasks, return_exceptions=True)
    
    if mongo_client is not None:
        mongo_client.close()
        logger.info("✅ MongoDB connection closed")
    
    logger.info(f"👋 Shutdown complete. Uptime: {time.time() - app_start_time:.1f}s")

# ============================================================================
# ✅ MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    config = HyperConfig()
    config.bind = [f"0.0.0.0:{Config.WEB_SERVER_PORT}"]
    config.worker_class = "asyncio"
    config.workers = 1
    config.accesslog = None
    config.errorlog = "-"
    config.loglevel = "warning"
    config.http2 = True
    config.keep_alive_timeout = 30
    
    logger.info(f"🌐 Starting SK4FiLM v9.0 on port {Config.WEB_SERVER_PORT}...")
    logger.info("🎯 DUAL PRIORITY THUMBNAIL SYSTEM")
    logger.info(f"   • HOME MODE: Sources → Thumbnail → Fallback")
    logger.info(f"   • SEARCH MODE: Thumbnail → Sources → Fallback")
    logger.info(f"   • Fallback Image: ✅ ENABLED")
    logger.info(f"   • File Channel ID: {Config.FILE_CHANNEL_ID}")
    logger.info(f"   • Fallback URL: {Config.FALLBACK_POSTER}")
    
    try:
        asyncio.run(serve(app, config))
    except KeyboardInterrupt:
        logger.info("👋 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
