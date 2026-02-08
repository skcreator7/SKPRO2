# ============================================================================
# 🚀 SK4FiLM v9.0 - COMPLETE THUMBNAIL SYSTEM - ONE MOVIE, ONE THUMBNAIL
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

# ✅ IMPORT ALL MODULES WITH PROPER ERROR HANDLING
# ============================================================================

# First, set up logger before any imports
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

# Now import modules with try/except
try:
    from cache import CacheManager
    logger.debug("✅ Cache module imported")
except ImportError as e:
    logger.error(f"❌ Cache module import error: {e}")
    # Fallback CacheManager
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
    VerificationSystem = None
    # Fallback VerificationSystem
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
    PremiumSystem = None
    PremiumTier = None
    # Fallback PremiumSystem
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

# Import thumbnail_manager first
try:
    from thumbnail_manager import ThumbnailManager
    THUMBNAIL_MANAGER_AVAILABLE = True
    logger.info("✅ ThumbnailManager module imported")
except ImportError as e:
    logger.error(f"❌ ThumbnailManager import error: {e}")
    THUMBNAIL_MANAGER_AVAILABLE = False
    
    # Fallback ThumbnailManager
    class ThumbnailManager:
        def __init__(self, mongo_client, config, bot_handler=None):
            self.mongo_client = mongo_client
            self.config = config
            self.bot_handler = bot_handler
            self.thumbnails_col = None
            
        async def initialize(self):
            logger.warning("⚠️ Using fallback ThumbnailManager")
            return True
        
        async def get_thumbnail_for_movie(self, title: str, channel_id: int = None, message_id: int = None) -> Dict[str, Any]:
            return {
                'thumbnail_url': '',
                'source': 'fallback',
                'has_thumbnail': False,
                'extracted': False
            }
        
        async def get_thumbnails_batch(self, movies: List[Dict]) -> List[Dict]:
            return [{
                **movie,
                'thumbnail_url': '',
                'source': 'fallback',
                'has_thumbnail': False,
                'extracted': False
            } for movie in movies]
        
        async def get_stats(self) -> Dict[str, Any]:
            return {
                'total_thumbnails': 0,
                'performance_stats': {},
                'success_rate': '0%',
                'target': '99% success rate (fallback)'
            }
        
        async def shutdown(self):
            logger.info("✅ ThumbnailManager shutdown (fallback)")

try:
    from poster_fetching import PosterFetcher, PosterSource
    logger.debug("✅ Poster fetching module imported")
except ImportError as e:
    logger.error(f"❌ Poster fetching module import error: {e}")
    PosterFetcher = None
    PosterSource = None
    # Fallback PosterSource
    class PosterSource:
        TMDB = "tmdb"
        OMDB = "omdb"
        CUSTOM = "custom"
        FALLBACK = "fallback"

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
        if not title:
            return ""
        # Basic normalization
        title = title.lower().strip()
        # Remove common suffixes
        title = re.sub(r'\s*\([^)]*\)$', '', title)
        title = re.sub(r'\s*\[[^\]]*\]$', '', title)
        title = re.sub(r'\s*\d{4}$', '', title)
        return title
    
    def extract_title_smart(text):
        if not text:
            return ""
        # Extract first line or first 100 chars
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if len(line) > 10 and not line.startswith('http'):
                return line[:100]
        return text[:50] if text else ""
    
    def extract_title_from_file(filename, caption=None):
        if filename:
            # Remove extension and clean
            name = os.path.splitext(filename)[0]
            # Remove quality indicators
            name = re.sub(r'[._]', ' ', name)
            name = re.sub(r'\b(480p|720p|1080p|2160p|4k|hd|hevc|x265|x264)\b', '', name, flags=re.IGNORECASE)
            name = re.sub(r'\s+', ' ', name)
            name = name.strip()
            if name:
                return name
        if caption:
            return extract_title_smart(caption)
        return "Unknown"
    
    def format_size(size):
        if not size:
            return "Unknown"
        if size < 1024:
            return f"{size} B"
        elif size < 1024*1024:
            return f"{size/1024:.1f} KB"
        elif size < 1024*1024*1024:
            return f"{size/1024/1024:.1f} MB"
        else:
            return f"{size/1024/1024/1024:.2f} GB"
    
    def detect_quality(filename):
        if not filename:
            return "480p"
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
        if not filename:
            return False
        video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v']
        return any(filename.lower().endswith(ext) for ext in video_extensions)
    
    def format_post(text, max_length=None):
        if not text:
            return ""
        # Clean up text
        text = re.sub(r'\n\s*\n', '\n\n', text)
        if max_length and len(text) > max_length:
            text = text[:max_length] + "..."
        return text.strip()
    
    def is_new(date):
        if not date:
            return False
        if isinstance(date, str):
            try:
                date = datetime.fromisoformat(date.replace('Z', '+00:00'))
            except:
                return False
        # Consider new if within last 7 days
        return (datetime.now() - date).days < 7

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

# ============================================================================
# ✅ CONFIGURATION - COMPLETE FILE INDEXING
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
    FILE_CHANNEL_ID = -1001768249569  # ✅ FILE CHANNEL
    
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
    VERIFICATION_REQUIRED = os.environ.get("VERIFICATION_REQUIRED", "true").lower() == "False"
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
    
    # 🔥 FILE CHANNEL INDEXING SETTINGS - OPTIMIZED
    AUTO_INDEX_INTERVAL = int(os.environ.get("AUTO_INDEX_INTERVAL", "120"))  # 2 minutes
    BATCH_INDEX_SIZE = int(os.environ.get("BATCH_INDEX_SIZE", "500"))  # Large batches
    MAX_INDEX_LIMIT = int(os.environ.get("MAX_INDEX_LIMIT", "0"))  # 0 = Unlimited
    INDEX_ALL_HISTORY = os.environ.get("INDEX_ALL_HISTORY", "true").lower() == "true"  # ✅ All history
    INSTANT_AUTO_INDEX = os.environ.get("INSTANT_AUTO_INDEX", "true").lower() == "true"
    
    # 🔥 SEARCH SETTINGS
    SEARCH_MIN_QUERY_LENGTH = 2
    SEARCH_RESULTS_PER_PAGE = 12
    SEARCH_CACHE_TTL = 1800  # 30 minutes
    
    # 🔥 THUMBNAIL SETTINGS
    THUMBNAIL_EXTRACTION_ENABLED = os.environ.get("THUMBNAIL_EXTRACTION_ENABLED", "true").lower() == "true"
    THUMBNAIL_BATCH_SIZE = int(os.environ.get("THUMBNAIL_BATCH_SIZE", "10"))
    THUMBNAIL_RETRY_LIMIT = int(os.environ.get("THUMBNAIL_RETRY_LIMIT", "3"))
    THUMBNAIL_MAX_SIZE_KB = int(os.environ.get("THUMBNAIL_MAX_SIZE_KB", "200"))
    
    # Thumbnail Manager Settings
    THUMBNAIL_TTL_DAYS = int(os.environ.get("THUMBNAIL_TTL_DAYS", "180"))  # 30 days TTL
    
    # NO FALLBACK POSTER - EMPTY STRING
    FALLBACK_POSTER = "https://iili.io/fAeIwv9.th.png"

# ============================================================================
# ✅ FAST INITIALIZATION
# ============================================================================

app = Quart(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# CORS headers
@app.after_request
async def add_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['X-SK4FiLM-Version'] = '9.0-THUMBNAIL-SYSTEM'
    response.headers['X-Response-Time'] = f"{time.perf_counter():.3f}"
    return response

# ============================================================================
# ✅ GLOBAL COMPONENTS
# ============================================================================

# Database
mongo_client = None
db = None
files_col = None
verification_col = None

# Telegram Sessions
try:
    from pyrogram import Client
    PYROGRAM_AVAILABLE = True
    
    User = None
    Bot = None
    user_session_ready = False
    bot_session_ready = False
except ImportError:
    PYROGRAM_AVAILABLE = False
    User = None
    Bot = None

# System Components
cache_manager = None
verification_system = None
premium_system = None
poster_fetcher = None
bot_handler = None
telegram_bot = None

# Thumbnail Manager
thumbnail_manager = None

# Indexing State
is_indexing = False
last_index_time = None
indexing_task = None

# ============================================================================
# ✅ BOT HANDLER MODULE - FIXED VERSION
# ============================================================================

class BotHandler:
    """Bot handler for Telegram bot operations - FIXED"""
    
    def __init__(self, bot_token=None, api_id=None, api_hash=None):
        self.bot_token = bot_token or Config.BOT_TOKEN
        self.api_id = api_id or Config.API_ID
        self.api_hash = api_hash or Config.API_HASH
        self.bot = None
        self.initialized = False
        self.last_update = None
        self.bot_username = None
        
    async def initialize(self):
        """Initialize bot handler - FIXED"""
        if not self.bot_token or not self.api_id or not self.api_hash:
            logger.error("❌ Bot token or API credentials not configured")
            return False
        
        try:
            # Reuse existing Bot session if available
            global Bot, bot_session_ready
            if Bot is not None and bot_session_ready:
                self.bot = Bot
                logger.info("✅ Bot Handler using existing Bot session")
                self.initialized = True
                self.last_update = datetime.now()
                
                # Get bot info
                try:
                    bot_info = await self.bot.get_me()
                    self.bot_username = bot_info.username
                except:
                    self.bot_username = "unknown"
                    
                return True
            
            # Otherwise create new session
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
            
            # Store bot instance globally for reuse
            global Bot
            Bot = self.bot
            bot_session_ready = True
            
            # Start periodic tasks
            asyncio.create_task(self._periodic_tasks())
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Bot handler initialization error: {e}")
            return False
    
    async def _periodic_tasks(self):
        """Run periodic tasks for bot"""
        while self.initialized:
            try:
                # Update last update time
                self.last_update = datetime.now()
                
                # Check if bot is still running
                try:
                    await self.bot.get_me()
                except:
                    logger.warning("⚠️ Bot session disconnected, reconnecting...")
                    try:
                        await self.bot.stop()
                    except:
                        pass
                    await asyncio.sleep(5)
                    await self.bot.start()
                
                # Sleep for 5 minutes
                await asyncio.sleep(300)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Bot handler periodic task error: {e}")
                await asyncio.sleep(60)
    
    async def get_message(self, channel_id, message_id):
        """Get message from Telegram"""
        if not self.initialized or not self.bot:
            return None
        
        try:
            return await self.bot.get_messages(channel_id, message_id)
        except Exception as e:
            logger.error(f"❌ Get message error: {e}")
            return None
    
    async def download_media(self, media):
        """Download media from Telegram"""
        if not self.initialized or not self.bot:
            return None
        
        try:
            return await self.bot.download_media(media, in_memory=True)
        except Exception as e:
            logger.error(f"❌ Download media error: {e}")
            return None
    
    async def get_bot_status(self):
        """Get bot status"""
        return {
            'initialized': self.initialized,
            'bot_username': self.bot_username,
            'last_update': self.last_update.isoformat() if self.last_update else None
        }
    
    async def shutdown(self):
        """Shutdown bot handler"""
        if self.bot:
            try:
                await self.bot.stop()
            except:
                pass
        
        self.initialized = False
        logger.info("✅ Bot Handler shutdown")

# ============================================================================
# ✅ OPTIMIZED SYNC MANAGEMENT
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
            try:
                await self.monitoring_task
            except:
                pass
        logger.info("🛑 Sync monitoring stopped")
    
    async def monitor_channel_sync(self):
        """Monitor channel sync with optimized approach"""
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
                if current_time - self.last_sync < 300:  # 5 minutes minimum interval
                    return
                
                self.last_sync = current_time
                
                logger.info("🔄 Checking for deleted files in Telegram...")
                
                # Get a batch of message IDs from database (newest first)
                batch_size = 100
                cursor = files_col.find(
                    {"channel_id": Config.FILE_CHANNEL_ID},
                    {"message_id": 1, "_id": 1, "title": 1, "normalized_title": 1}
                ).sort("message_id", -1).limit(batch_size)
                
                message_data = []
                async for doc in cursor:
                    message_data.append({
                        'message_id': doc['message_id'],
                        'db_id': doc['_id'],
                        'title': doc.get('title', 'Unknown'),
                        'normalized_title': doc.get('normalized_title', '')
                    })
                
                if not message_data:
                    logger.info("✅ No files to check")
                    return
                
                deleted_count = 0
                message_ids = [item['message_id'] for item in message_data]
                
                try:
                    # Check all messages in a single batch request
                    messages = await User.get_messages(Config.FILE_CHANNEL_ID, message_ids)
                    
                    # Determine which messages are deleted
                    existing_ids = set()
                    if isinstance(messages, list):
                        for msg in messages:
                            if msg and hasattr(msg, 'id'):
                                existing_ids.add(msg.id)
                    
                    # Delete entries for messages that no longer exist
                    for item in message_data:
                        if item['message_id'] not in existing_ids:
                            # Delete from files collection
                            await files_col.delete_one({"_id": item['db_id']})
                            
                            # Also delete thumbnail if exists
                            if thumbnail_manager and item['normalized_title']:
                                # Use thumbnail manager's collection to delete
                                await thumbnail_manager.thumbnails_col.delete_one({
                                    "normalized_title": item['normalized_title']
                                })
                            
                            deleted_count += 1
                            self.deleted_count += 1
                            
                            if deleted_count <= 5:  # Log only first few
                                logger.info(f"🗑️ Auto-deleted: {item['title'][:40]}... (Msg ID: {item['message_id']})")
                    
                    if deleted_count > 0:
                        logger.info(f"✅ Auto-deleted {deleted_count} files from database and thumbnails")
                    else:
                        logger.info("✅ No deleted files found")
                        
                except Exception as e:
                    logger.error(f"❌ Error checking messages: {e}")
                    
        except Exception as e:
            logger.error(f"❌ Auto-delete error: {e}")

# Initialize sync manager
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
# ✅ ENHANCED SEARCH FUNCTION - UPDATED WITH THUMBNAIL MANAGER
# ============================================================================

# Helper function for channel names
def channel_name_cached(cid):
    return f"Channel {cid}"

@performance_monitor.measure("enhanced_search_fixed")
@async_cache_with_ttl(maxsize=500, ttl=Config.SEARCH_CACHE_TTL)
async def search_movies_enhanced_fixed(query, limit=15, page=1):
    """FIXED: Combine post and file results for same title into SINGLE result with thumbnail support"""
    offset = (page - 1) * limit
    
    # Try cache first
    cache_key = f"search_fixed:{query}:{page}:{limit}"
    if cache_manager is not None and cache_manager.redis_enabled:
        cached_data = await cache_manager.get(cache_key)
        if cached_data:
            logger.info(f"✅ Cache HIT for: {query}")
            return cached_data
    
    logger.info(f"🔍 ENHANCED SEARCH for: {query}")
    
    query_lower = query.lower()
    
    # Main dictionary to hold merged results
    merged_results = {}  # normalized_title -> result_data
    
    # ============================================================================
    # ✅ 1. FIRST: SEARCH TEXT CHANNELS FOR POST RESULTS
    # ============================================================================
    if user_session_ready and User is not None:
        logger.info(f"📝 Searching TEXT CHANNELS for posts...")
        
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
                            
                            # Create post data
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
                                'quality_options': {},  # Empty for now
                                'is_video_file': False,
                                'year': year,
                                'search_score': 3,  # Post results get highest score
                                'result_type': 'post_only',
                                'thumbnail_url': None,
                                'has_thumbnail': False,
                                'poster_url': None,
                                'poster_source': None,
                                'combined': False  # Not combined yet
                            }
                            
                            # Store in merged results
                            merged_results[norm_title] = post_data
                            
            except Exception as e:
                logger.error(f"Text search error in {channel_id}: {e}")
            return True
        
        # Search all text channels
        tasks = [search_text_channel_posts(channel_id) for channel_id in Config.TEXT_CHANNEL_IDS]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info(f"📝 Found {len(merged_results)} POST results")
    
    # ============================================================================
    # ✅ 2. SECOND: SEARCH FILE CHANNEL DATABASE AND MERGE WITH POSTS
    # ============================================================================
    file_results_added = 0
    file_results_merged = 0
    
    if files_col is not None:
        try:
            logger.info(f"📁 Searching FILE CHANNEL database...")
            
            # Build search query
            search_query = {
                "$or": [
                    {"title": {"$regex": query, "$options": "i"}},
                    {"normalized_title": {"$regex": query, "$options": "i"}},
                    {"file_name": {"$regex": query, "$options": "i"}},
                    {"caption": {"$regex": query, "$options": "i"}}
                ],
                "status": "active"
            }
            
            # Get matching files
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
            ).limit(100)  # Limit file results
            
            async for doc in cursor:
                try:
                    title = doc.get('title', 'Unknown')
                    norm_title = normalize_title(title)
                    
                    # Extract quality info
                    quality_info = extract_quality_info(doc.get('file_name', ''))
                    quality = quality_info['full']
                    
                    # Get REAL message ID
                    real_msg_id = doc.get('real_message_id') or doc.get('message_id')
                    
                    # Extract year
                    year = doc.get('year', '')
                    
                    # Check if we already have this title from post results
                    if norm_title in merged_results:
                        # ✅ MERGE: Post exists, add file info to it
                        existing_result = merged_results[norm_title]
                        
                        # Update the existing post result with file information
                        existing_result.update({
                            'has_file': True,
                            'is_video_file': doc.get('is_video_file', False),
                            'file_caption': doc.get('caption', ''),
                            'real_message_id': real_msg_id,
                            'channel_id': doc.get('channel_id'),
                            'channel_name': channel_name_cached(doc.get('channel_id')),
                            'result_type': 'post_and_file',  # Changed to indicate both
                            'combined': True,  # Mark as combined
                            'search_score': 5  # Combined results get highest score
                        })
                        
                        # Add quality option
                        existing_result['quality_options'][quality] = {
                            'quality': quality,
                            'file_size': doc.get('file_size', 0),
                            'message_id': real_msg_id,
                            'file_id': doc.get('file_id'),
                            'telegram_file_id': doc.get('telegram_file_id'),
                            'file_name': doc.get('file_name', '')
                        }
                        
                        file_results_merged += 1
                        logger.debug(f"✅ Merged file with post: {title}")
                        
                    else:
                        # This is a file-only result (no post found)
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
                            'search_score': 2,  # File-only results get medium score
                            'result_type': 'file_only',
                            'quality_count': 1,
                            'poster_url': None,
                            'poster_source': None,
                            'combined': False
                        }
                        
                        merged_results[norm_title] = file_result
                        file_results_added += 1
                        
                except Exception as e:
                    logger.error(f"File processing error: {e}")
                    continue
            
            logger.info(f"📁 Found {file_results_added} FILE-ONLY results, merged {file_results_merged} with existing posts")
            
        except Exception as e:
            logger.error(f"❌ File search error: {e}")
    
    # ============================================================================
    # ✅ 3. GET THUMBNAILS FROM THUMBNAIL MANAGER FOR ALL RESULTS
    # ============================================================================
    if thumbnail_manager:
        logger.info(f"🖼️ Getting thumbnails for {len(merged_results)} results using ThumbnailManager...")
        
        # Prepare movies for thumbnail fetching
        movies_for_thumbnails = []
        for norm_title, result in merged_results.items():
            movie_data = {
                'title': result.get('title', ''),
                'channel_id': result.get('channel_id'),
                'message_id': result.get('real_message_id') or result.get('message_id'),
                'result_type': result.get('result_type', 'unknown')
            }
            movies_for_thumbnails.append(movie_data)
        
        # Get thumbnails in batch
        if movies_for_thumbnails:
            movies_with_thumbnails = await thumbnail_manager.get_thumbnails_batch(movies_for_thumbnails)
            
            # Update results with thumbnail data
            for i, (norm_title, result) in enumerate(merged_results.items()):
                if i < len(movies_with_thumbnails):
                    thumbnail_data = movies_with_thumbnails[i]
                    
                    # Update result with thumbnail info
                    result.update({
                        'thumbnail_url': thumbnail_data.get('thumbnail_url'),
                        'thumbnail_source': thumbnail_data.get('source'),
                        'has_thumbnail': thumbnail_data.get('has_thumbnail', False),
                        'thumbnail_extracted': thumbnail_data.get('extracted', False)
                    })
            
            logger.info(f"✅ Got thumbnails for {len(movies_for_thumbnails)} movies")
    
    # ============================================================================
    # ✅ 4. CONVERT DICTIONARY TO LIST AND SORT
    # ============================================================================
    all_results = list(merged_results.values())
    
    if not all_results:
        # Empty result set
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
                'thumbnail_manager': thumbnail_manager is not None,
                'real_message_ids': True,
                'search_logic': 'enhanced_fixed_with_merging'
            },
            'bot_username': Config.BOT_USERNAME
        }
        
        # Cache empty result
        if cache_manager is not None:
            await cache_manager.set(cache_key, result_data, expire_seconds=Config.SEARCH_CACHE_TTL)
        
        return result_data
    
    # ============================================================================
    # ✅ 5. FETCH POSTERS FOR RESULTS WITHOUT THUMBNAILS
    # ============================================================================
    logger.info(f"🎬 Fetching posters for results without thumbnails...")
    
    # Prepare movies that need posters (no thumbnail)
    movies_needing_posters = []
    for result in all_results:
        has_thumbnail = result.get('has_thumbnail', False)
        
        # Only fetch poster if no thumbnail
        if not has_thumbnail:
            movie_data = {
                'title': result.get('title', ''),
                'year': result.get('year', ''),
                'quality': result.get('quality', ''),
                'original_title': result.get('original_title', ''),
                'has_thumbnail': has_thumbnail,
                'result_type': result.get('result_type', 'unknown')
            }
            movies_needing_posters.append(movie_data)
    
    if movies_needing_posters and poster_fetcher:
        # Fetch posters in batch
        movies_with_posters = await get_posters_for_movies_batch(movies_needing_posters)
        
        # Update results with poster data
        for i, result in enumerate(all_results):
            if not result.get('has_thumbnail'):
                # Try to find poster for this movie
                for movie in movies_with_posters:
                    if movie.get('title') == result.get('title'):
                        # Update with poster info
                        result.update({
                            'poster_url': movie.get('poster_url'),
                            'poster_source': movie.get('poster_source'),
                            'poster_rating': movie.get('poster_rating', '0.0'),
                            'has_poster': True,
                            # Use poster as thumbnail
                            'thumbnail_url': movie.get('poster_url'),
                            'thumbnail_source': movie.get('poster_source'),
                            'has_thumbnail': True
                        })
                        break
    
    # ============================================================================
    # ✅ 6. SORT RESULTS
    # ============================================================================
    # Sort by: combined first, posts second, files third, then by search score, then by date
    all_results.sort(key=lambda x: (
        x.get('result_type') == 'post_and_file',  # Combined first (True > False)
        x.get('result_type') == 'post_only',  # Posts second
        x.get('search_score', 0),  # Higher score first
        x.get('is_new', False),  # New first (True > False)
        x.get('date', '') if isinstance(x.get('date'), str) else ''  # Recent first
    ), reverse=True)
    
    # ============================================================================
    # ✅ 7. PAGINATION
    # ============================================================================
    total = len(all_results)
    start_idx = offset
    end_idx = offset + limit
    paginated = all_results[start_idx:end_idx]
    
    # Statistics
    post_count = sum(1 for r in all_results if r.get('result_type') == 'post_only')
    file_count = sum(1 for r in all_results if r.get('result_type') == 'file_only')
    combined_count = sum(1 for r in all_results if r.get('result_type') == 'post_and_file')
    
    stats = {
        'total': total,
        'post_only': post_count,
        'file_only': file_count,
        'post_and_file': combined_count,
        'file_results_added': file_results_added,
        'file_results_merged': file_results_merged
    }
    
    # Get thumbnail stats
    thumbnail_stats = {}
    if thumbnail_manager:
        thumbnail_stats = await thumbnail_manager.get_stats()
    
    # Log results
    logger.info(f"📊 FINAL RESULTS:")
    logger.info(f"   • Total results: {total}")
    logger.info(f"   • Post-only: {post_count}")
    logger.info(f"   • File-only: {file_count}")
    logger.info(f"   • Post+File combined: {combined_count}")
    logger.info(f"   • Files merged with posts: {file_results_merged}")
    logger.info(f"   • Thumbnail Manager: {'✅ ENABLED' if thumbnail_manager else '❌ DISABLED'}")
    
    # Show sample of results
    for i, result in enumerate(paginated[:5]):
        result_type = result.get('result_type', 'unknown')
        title = result.get('title', '')[:40]
        has_thumbnail = result.get('has_thumbnail', False)
        has_file = result.get('has_file', False)
        has_post = result.get('has_post', False)
        thumbnail_source = result.get('thumbnail_source', 'none')
        
        logger.info(f"   📋 {i+1}. {result_type}: {title}... | File: {has_file} | Post: {has_post} | Thumb: {has_thumbnail} ({thumbnail_source})")
    
    # ============================================================================
    # ✅ 8. FINAL DATA STRUCTURE
    # ============================================================================
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
            'thumbnail_stats': thumbnail_stats,
            'post_file_merged': True,  # ✅ New feature
            'file_only_with_poster': True,  # ✅ New feature
            'poster_fetcher': poster_fetcher is not None,
            'thumbnails_enabled': True,
            'thumbnail_manager': thumbnail_manager is not None,
            'real_message_ids': True,
            'search_logic': 'enhanced_fixed_with_merging'
        },
        'bot_username': Config.BOT_USERNAME
    }
    
    # Cache results
    if cache_manager is not None:
        await cache_manager.set(cache_key, result_data, expire_seconds=Config.SEARCH_CACHE_TTL)
    
    logger.info(f"✅ Enhanced search complete: {len(paginated)} results (page {page})")
    
    return result_data

# ============================================================================
# ✅ BACKWARD COMPATIBLE SEARCH FUNCTION
# ============================================================================

@async_cache_with_ttl(maxsize=500, ttl=Config.SEARCH_CACHE_TTL)
async def search_movies_multi_channel_merged(query, limit=15, page=1):
    """Backward compatible search - uses enhanced fixed search"""
    return await search_movies_enhanced_fixed(query, limit, page)

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
    
    # First check for HEVC variants
    is_hevc = any(re.search(pattern, filename_lower) for pattern in HEVC_PATTERNS)
    
    # Then check quality
    for pattern, quality in QUALITY_PATTERNS:
        if re.search(pattern, filename_lower):
            if is_hevc and quality in ['720p', '1080p', '2160p']:
                return f"{quality} HEVC"
            return quality
    
    return "480p"

def extract_quality_info(filename):
    """Extract detailed quality info"""
    quality = detect_quality_enhanced(filename)
    
    # Parse quality components
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
# ✅ QUALITY MERGER
# ============================================================================

class QualityMerger:
    """Merge multiple qualities for same title"""
    
    @staticmethod
    def merge_quality_options(quality_options_dict):
        """Merge quality options from multiple sources"""
        if not quality_options_dict:
            return {}
        
        merged = {}
        
        # Sort by priority
        for quality, option in quality_options_dict.items():
            base_quality = quality.replace(' HEVC', '')
            
            if base_quality not in merged:
                merged[base_quality] = {
                    'qualities': [],
                    'best_option': None,
                    'total_size': 0,
                    'file_count': 0
                }
            
            # Add quality variant
            merged[base_quality]['qualities'].append({
                'full_quality': quality,
                'is_hevc': 'HEVC' in quality,
                'file_id': option.get('file_id'),
                'file_size': option.get('file_size', 0),
                'file_name': option.get('file_name', ''),
                'is_video': option.get('is_video', False),
                'channel_id': option.get('channel_id'),
                'message_id': option.get('message_id'),
                'real_message_id': option.get('real_message_id'),
                'telegram_file_id': option.get('telegram_file_id'),
                'thumbnail_url': option.get('thumbnail_url')
            })
            
            merged[base_quality]['total_size'] += option.get('file_size', 0)
            merged[base_quality]['file_count'] += 1
            
            # Set best option (highest quality, smallest size)
            if merged[base_quality]['best_option'] is None:
                merged[base_quality]['best_option'] = quality
            else:
                current_priority = Config.QUALITY_PRIORITY.index(base_quality) if base_quality in Config.QUALITY_PRIORITY else 999
                best_base = merged[base_quality]['best_option'].replace(' HEVC', '')
                best_priority = Config.QUALITY_PRIORITY.index(best_base) if best_base in Config.QUALITY_PRIORITY else 999
                
                if current_priority < best_priority:
                    merged[base_quality]['best_option'] = quality
        
        # Sort by quality priority
        sorted_merged = {}
        for quality in Config.QUALITY_PRIORITY:
            if quality in merged:
                sorted_merged[quality] = merged[quality]
        
        # Add any remaining qualities
        for quality in merged:
            if quality not in sorted_merged:
                sorted_merged[quality] = merged[quality]
        
        return sorted_merged
    
    @staticmethod
    def get_quality_summary(merged_options):
        """Get summary of available qualities"""
        if not merged_options:
            return "No files"
        
        qualities = list(merged_options.keys())
        
        # Sort by priority
        sorted_qualities = []
        for quality in Config.QUALITY_PRIORITY:
            if quality in qualities:
                sorted_qualities.append(quality)
                qualities.remove(quality)
        
        # Add remaining qualities
        sorted_qualities.extend(sorted(qualities))
        
        # Create summary
        summary_parts = []
        for quality in sorted_qualities[:3]:
            data = merged_options[quality]
            count = data['file_count']
            if count > 1:
                summary_parts.append(f"{quality} ({count} files)")
            else:
                summary_parts.append(quality)
        
        if len(sorted_qualities) > 3:
            summary_parts.append(f"+{len(sorted_qualities) - 3} more")
        
        return " • ".join(summary_parts)

# ============================================================================
# ✅ OPTIMIZED FILE CHANNEL INDEXING MANAGER
# ============================================================================

class OptimizedFileIndexingManager:
    """Optimized file channel indexing manager with thumbnail extraction"""
    
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
        
        logger.info("🚀 Starting OPTIMIZED FILE CHANNEL INDEXING...")
        self.is_running = True
        
        # Run immediate indexing
        asyncio.create_task(self._run_optimized_indexing())
        
        # Start periodic loop
        self.indexing_task = asyncio.create_task(self._indexing_loop())
    
    async def stop_indexing(self):
        """Stop indexing"""
        self.is_running = False
        if self.indexing_task:
            self.indexing_task.cancel()
            try:
                await self.indexing_task
            except asyncio.CancelledError:
                pass
        
        logger.info("🛑 File indexing stopped")
    
    async def _run_optimized_indexing(self):
        """Run optimized indexing - ONLY NEW MESSAGES"""
        logger.info("🔥 RUNNING OPTIMIZED INDEXING (NEW MESSAGES ONLY)...")
        
        try:
            # Get last indexed message from database
            last_indexed = await files_col.find_one(
                {"channel_id": Config.FILE_CHANNEL_ID}, 
                sort=[('message_id', -1)],
                projection={'message_id': 1}
            )
            
            last_message_id = last_indexed['message_id'] if last_indexed else 0
            
            logger.info(f"📊 Last indexed message ID: {last_message_id}")
            
            # Fetch only NEW messages (after last indexed)
            messages_to_index = []
            total_fetched = 0
            
            try:
                # Fetch recent messages in reverse order (newest first)
                async for msg in User.get_chat_history(
                    Config.FILE_CHANNEL_ID, 
                    limit=Config.BATCH_INDEX_SIZE
                ):
                    total_fetched += 1
                    
                    # Stop if we reach already indexed messages
                    if msg.id <= last_message_id:
                        break
                    
                    # Only process file messages (documents or videos)
                    if msg and (msg.document or msg.video):
                        messages_to_index.append(msg)
                    
                    # Safety limit
                    if Config.MAX_INDEX_LIMIT > 0 and total_fetched >= Config.MAX_INDEX_LIMIT:
                        logger.info(f"⚠️ Reached max limit: {Config.MAX_INDEX_LIMIT}")
                        break
                
                logger.info(f"📥 Fetched {total_fetched} messages, found {len(messages_to_index)} new files")
                
            except Exception as e:
                logger.error(f"❌ Error fetching messages: {e}")
                return
            
            # Process messages in reverse order (oldest first to maintain sequence)
            if messages_to_index:
                messages_to_index.reverse()
                
                batch_size = 50
                total_batches = math.ceil(len(messages_to_index) / batch_size)
                
                logger.info(f"🔧 Processing {len(messages_to_index)} new files in {total_batches} batches...")
                
                for batch_num in range(total_batches):
                    start_idx = batch_num * batch_size
                    end_idx = min(start_idx + batch_size, len(messages_to_index))
                    batch = messages_to_index[start_idx:end_idx]
                    
                    logger.info(f"📦 Processing batch {batch_num + 1}/{total_batches} ({len(batch)} files)...")
                    
                    batch_stats = await self._process_optimized_batch(batch)
                    
                    # Update stats
                    self.indexing_stats['total_files_processed'] += batch_stats['processed']
                    self.indexing_stats['total_indexed'] += batch_stats['indexed']
                    self.indexing_stats['total_skipped'] += batch_stats['skipped']
                    self.indexing_stats['total_errors'] += batch_stats['errors']
                    
                    # Small delay between batches
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
                # Wait for next run
                if self.next_run and self.next_run > datetime.now():
                    wait_seconds = (self.next_run - datetime.now()).total_seconds()
                    if wait_seconds > 30:
                        logger.info(f"⏰ Next index in {wait_seconds:.0f}s")
                    await asyncio.sleep(min(wait_seconds, 30))
                    continue
                
                # Run indexing cycle
                await self._run_optimized_indexing()
                
                # Schedule next run
                self.next_run = datetime.now() + timedelta(seconds=Config.AUTO_INDEX_INTERVAL)
                self.last_run = datetime.now()
                
                # Sleep before checking again
                await asyncio.sleep(10)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Indexing loop error: {e}")
                await asyncio.sleep(60)
    
    async def _process_optimized_batch(self, messages):
        """Process a batch of messages - OPTIMIZED"""
        batch_stats = {
            'processed': len(messages),
            'indexed': 0,
            'skipped': 0,
            'errors': 0
        }
        
        for msg in messages:
            try:
                # Skip non-file messages
                if not msg or (not msg.document and not msg.video):
                    batch_stats['skipped'] += 1
                    continue
                
                # ✅ ZERO DUPLICATE CHECKS - Just check by message ID
                existing = await files_col.find_one({
                    'channel_id': Config.FILE_CHANNEL_ID,
                    'message_id': msg.id
                }, {'_id': 1})
                
                if existing:
                    logger.debug(f"📝 Already indexed: {msg.id}")
                    batch_stats['skipped'] += 1
                    continue
                
                # Index the file (simple version)
                success, normalized_title = await index_single_file_fast(msg)
                
                if success:
                    batch_stats['indexed'] += 1
                    
                    # ✅ Get thumbnail for this file using ThumbnailManager
                    if thumbnail_manager and Config.THUMBNAIL_EXTRACTION_ENABLED:
                        try:
                            # Get thumbnail for this movie
                            thumbnail_data = await thumbnail_manager.get_thumbnail_for_movie(
                                title=normalized_title,
                                channel_id=Config.FILE_CHANNEL_ID,
                                message_id=msg.id
                            )
                            
                            if thumbnail_data and thumbnail_data.get('thumbnail_url'):
                                # Update file document with thumbnail
                                await files_col.update_one(
                                    {'channel_id': Config.FILE_CHANNEL_ID, 'message_id': msg.id},
                                    {'$set': {
                                        'thumbnail_url': thumbnail_data['thumbnail_url'],
                                        'thumbnail_extracted': thumbnail_data.get('extracted', False),
                                        'thumbnail_source': thumbnail_data.get('source', 'unknown')
                                    }}
                                )
                                logger.debug(f"✅ Thumbnail set for: {normalized_title}")
                        except Exception as thumb_error:
                            logger.error(f"❌ Thumbnail extraction error for {normalized_title}: {thumb_error}")
                else:
                    batch_stats['skipped'] += 1
                    
            except Exception as e:
                logger.error(f"❌ Error processing message {msg.id}: {e}")
                batch_stats['errors'] += 1
                continue
        
        logger.info(f"📦 Batch stats: {batch_stats}")
        return batch_stats
    
    async def get_indexing_status(self):
        """Get current indexing status"""
        thumbnail_stats = {}
        if thumbnail_manager:
            thumbnail_stats = await thumbnail_manager.get_stats()
        
        return {
            'is_running': self.is_running,
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'next_run': self.next_run.isoformat() if self.next_run else None,
            'total_indexed': self.total_indexed,
            'total_skipped': self.total_skipped,
            'stats': self.indexing_stats,
            'thumbnail_stats': thumbnail_stats
        }

# Initialize file indexing manager
file_indexing_manager = OptimizedFileIndexingManager()

# ============================================================================
# ✅ OPTIMIZED FILE INDEXING FUNCTIONS
# ============================================================================

async def extract_title_fast(filename, caption):
    """Fast title extraction - minimal processing"""
    # Try filename first
    if filename:
        name = os.path.splitext(filename)[0]
        
        # Quick cleanup
        name = re.sub(r'[._]', ' ', name)
        name = re.sub(r'\b(480p|720p|1080p|2160p|4k|hd|hevc)\b', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\s+', ' ', name)
        name = name.strip()
        
        if name and len(name) > 3:
            return name[:100]
    
    # Try caption
    if caption:
        lines = caption.split('\n')
        for line in lines:
            line = line.strip()
            if len(line) > 10 and not line.startswith('http'):
                return line[:100]
    
    # Fallback
    if filename:
        return os.path.splitext(filename)[0][:50]
    
    return "Unknown File"

async def index_single_file_fast(message):
    """Fast file indexing - minimal checks, returns (success, normalized_title)"""
    try:
        if files_col is None:
            return False, None
        
        if not message or (not message.document and not message.video):
            return False, None
        
        # Extract title quickly
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
        
        # Extract quality
        quality = detect_quality_enhanced(file_name or "")
        
        # Create minimal document
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
        
        # Add file-specific data
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
        
        # Insert into MongoDB
        await files_col.insert_one(doc)
        
        # Log success
        logger.info(f"✅ INDEXED FAST: {title[:50]}... (ID: {message.id})")
        
        return True, normalized_title
        
    except Exception as e:
        if "duplicate key error" in str(e).lower():
            return False, None
        logger.error(f"❌ Fast indexing error: {e}")
        return False, None

async def setup_database_indexes():
    """Setup minimal required database indexes"""
    if files_col is None:
        return
    
    try:
        # Only create essential indexes
        await files_col.create_index(
            [("channel_id", 1), ("message_id", 1)],
            unique=True,
            name="channel_message_unique",
            background=True
        )
        
        # Text search index
        await files_col.create_index(
            [("normalized_title", "text")],
            name="title_text_search",
            background=True
        )
        
        # Index for normalized_title
        await files_col.create_index(
            [("normalized_title", 1)],
            name="normalized_title_index",
            background=True
        )
        
        logger.info("✅ Created files collection indexes")
        
    except Exception as e:
        logger.warning(f"⚠️ Files index creation error: {e}")

# ============================================================================
# ✅ OPTIMIZED INITIAL INDEXING WITH THUMBNAIL EXTRACTION
# ============================================================================

async def initial_indexing_optimized():
    """Optimized initial indexing - FAST AND EFFICIENT"""
    if User is None or files_col is None or not user_session_ready:
        logger.warning("⚠️ User session not ready for initial indexing")
        return
    
    logger.info("=" * 60)
    logger.info("🚀 STARTING OPTIMIZED FILE CHANNEL INDEXING")
    logger.info("=" * 60)
    logger.info("✅ ZERO DB PRELOAD")
    logger.info("✅ ZERO DUPLICATE CHECKS")
    logger.info("✅ ONLY NEW MESSAGES INDEXED")
    logger.info("✅ FILES-ONLY TELEGRAM FETCH")
    logger.info("✅ AUTO-DELETE WHEN TELEGRAM DELETES")
    logger.info(f"✅ THUMBNAIL MANAGER: {'✅ ENABLED' if thumbnail_manager else '❌ DISABLED'}")
    logger.info("=" * 60)
    
    try:
        # Setup minimal indexes
        await setup_database_indexes()
        
        # Start file indexing
        await file_indexing_manager.start_indexing()
        
        # Start sync monitoring
        await sync_manager.start_sync_monitoring()
        
    except Exception as e:
        logger.error(f"❌ Initial indexing error: {e}")

# ============================================================================
# ✅ EXTRACT THUMBNAILS FOR EXISTING FILES - FIXED VERSION
# ============================================================================

async def extract_thumbnails_for_existing_files():
    """Extract thumbnails for all existing files in database - FIXED"""
    if not thumbnail_manager or files_col is None:
        logger.warning("⚠️ ThumbnailManager or files collection not available")
        return
    
    logger.info("🔄 Extracting thumbnails for existing files...")
    
    try:
        # Get all video files without thumbnails OR with fallback thumbnails
        cursor = files_col.find({
            'is_video_file': True,
            'channel_id': Config.FILE_CHANNEL_ID,
            '$or': [
                {'thumbnail_url': {'$exists': False}},
                {'thumbnail_url': ''},
                {'thumbnail_url': Config.FALLBACK_POSTER},
                {'thumbnail_source': {'$in': ['fallback', 'api', None]}}
            ]
        }, {
            'title': 1,
            'normalized_title': 1,
            'channel_id': 1,
            'message_id': 1,
            'real_message_id': 1,
            '_id': 1
        })
        
        files_to_process = []
        async for doc in cursor:
            files_to_process.append({
                'title': doc.get('title', ''),
                'normalized_title': doc.get('normalized_title', ''),
                'channel_id': doc.get('channel_id'),
                'message_id': doc.get('real_message_id') or doc.get('message_id'),
                'db_id': doc.get('_id')
            })
        
        logger.info(f"📊 Found {len(files_to_process)} video files to process for thumbnails")
        
        if not files_to_process:
            logger.info("✅ No files need thumbnail extraction")
            return
        
        # Process in batches
        batch_size = 5  # Smaller batch size for reliability
        total_batches = math.ceil(len(files_to_process) / batch_size)
        successful = 0
        failed = 0
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(files_to_process))
            batch = files_to_process[start_idx:end_idx]
            
            logger.info(f"🖼️ Processing batch {batch_num + 1}/{total_batches} ({len(batch)} files)...")
            
            # Get thumbnails for batch
            thumbnail_results = await thumbnail_manager.get_thumbnails_batch(batch)
            
            # Update database with thumbnail info
            for i, file_info in enumerate(batch):
                if i < len(thumbnail_results):
                    thumbnail_data = thumbnail_results[i]
                    
                    if thumbnail_data.get('thumbnail_url'):
                        await files_col.update_one(
                            {'_id': file_info['db_id']},
                            {'$set': {
                                'thumbnail_url': thumbnail_data['thumbnail_url'],
                                'thumbnail_extracted': thumbnail_data.get('extracted', False),
                                'thumbnail_source': thumbnail_data.get('source', 'unknown')
                            }}
                        )
                        
                        if thumbnail_data.get('source', '').startswith('telegram'):
                            logger.info(f"✅ Telegram thumbnail extracted for: {file_info['title'][:50]}...")
                        else:
                            logger.info(f"📄 API thumbnail fetched for: {file_info['title'][:50]}...")
                        
                        successful += 1
                    else:
                        logger.warning(f"⚠️ No thumbnail for: {file_info['title'][:50]}...")
                        failed += 1
            
            # Small delay between batches to avoid rate limiting
            if batch_num < total_batches - 1:
                await asyncio.sleep(2)
        
        logger.info(f"✅ Thumbnail extraction complete: {successful} successful, {failed} failed")
        
        # Get stats
        stats = await thumbnail_manager.get_stats()
        logger.info(f"📊 Thumbnail Stats: {stats}")
        
    except Exception as e:
        logger.error(f"❌ Error extracting thumbnails for existing files: {e}")

# ============================================================================
# ✅ POSTER FETCHING FUNCTIONS - UPDATED FOR NO FALLBACK
# ============================================================================

async def get_poster_for_movie(title: str, year: str = "", quality: str = "") -> Dict[str, Any]:
    """Get poster for movie - Returns empty string if not found"""
    global poster_fetcher
    
    # If poster_fetcher is not available, return empty
    if poster_fetcher is None:
        return {
            'poster_url': '',
            'source': 'none',
            'rating': '0.0',
            'year': year,
            'title': title,
            'quality': quality or 'unknown',
            'found': False
        }
    
    try:
        # Fetch poster with timeout
        poster_task = asyncio.create_task(poster_fetcher.fetch_poster(title))
        
        try:
            poster_data = await asyncio.wait_for(poster_task, timeout=3.0)
            
            if poster_data and poster_data.get('poster_url'):
                logger.debug(f"✅ Poster fetched: {title} - {poster_data['source']}")
                poster_data['found'] = True
                return poster_data
            else:
                raise ValueError("Invalid poster data")
                
        except (asyncio.TimeoutError, ValueError, Exception) as e:
            logger.warning(f"⚠️ Poster fetch timeout/error for {title}: {e}")
            
            if not poster_task.done():
                poster_task.cancel()
            
            # Return empty
            return {
                'poster_url': '',
                'source': 'none',
                'rating': '0.0',
                'year': year,
                'title': title,
                'quality': quality or 'unknown',
                'found': False
            }
            
    except Exception as e:
        logger.error(f"❌ Unexpected error in get_poster_for_movie: {e}")
        return {
            'poster_url': '',
            'source': 'none',
            'rating': '0.0',
            'year': year,
            'title': title,
            'quality': quality or 'unknown',
            'found': False
        }

async def get_posters_for_movies_batch(movies: List[Dict]) -> List[Dict]:
    """Get posters for multiple movies in batch"""
    results = []
    
    # Create tasks for all movies
    tasks = []
    for movie in movies:
        title = movie.get('title', '')
        year = movie.get('year', '')
        quality = movie.get('quality', '')
        
        task = asyncio.create_task(get_poster_for_movie(title, year, quality))
        tasks.append((movie, task))
    
    # Process results
    for movie, task in tasks:
        try:
            poster_data = await task
            
            # Update movie with poster data (only if found)
            movie_with_poster = movie.copy()
            if poster_data['found']:
                movie_with_poster.update({
                    'poster_url': poster_data['poster_url'],
                    'poster_source': poster_data['source'],
                    'poster_rating': poster_data['rating'],
                    'has_poster': True,
                    'found': True
                })
            else:
                movie_with_poster.update({
                    'poster_url': '',
                    'poster_source': 'none',
                    'poster_rating': '0.0',
                    'has_poster': False,
                    'found': False
                })
            
            results.append(movie_with_poster)
            
        except Exception as e:
            logger.warning(f"⚠️ Batch poster error for {movie.get('title')}: {e}")
            
            # Add movie with empty
            movie_with_empty = movie.copy()
            movie_with_empty.update({
                'poster_url': '',
                'poster_source': 'none',
                'poster_rating': '0.0',
                'has_poster': False,
                'found': False
            })
            
            results.append(movie_with_empty)
    
    return results

# ============================================================================
# ✅ DUAL SESSION INITIALIZATION
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
    
    # Initialize USER Session
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
            
            # Test channel access
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
                try:
                    await User.stop()
                except:
                    pass
            User = None
    
    # Initialize BOT Session
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
                try:
                    await Bot.stop()
                except:
                    pass
            Bot = None
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 SESSION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"USER Session: {'✅ READY' if user_session_ready else '❌ NOT READY'}")
    logger.info(f"BOT Session: {'✅ READY' if bot_session_ready else '❌ NOT READY'}")
    logger.info(f"Bot Handler: {'✅ INITIALIZED' if bot_handler.initialized else '❌ NOT READY'}")
    
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
        
        logger.info("✅ MongoDB OK - Files and Verifications collections initialized")
        return True
        
    except asyncio.TimeoutError:
        logger.error("❌ MongoDB connection timeout")
        return False
    except Exception as e:
        logger.error(f"❌ MongoDB error: {e}")
        return False

# ============================================================================
# ✅ MAIN INITIALIZATION - COMPLETE THUMBNAIL SYSTEM
# ============================================================================

@performance_monitor.measure("system_init")
async def init_system():
    start_time = time.time()
    
    try:
        logger.info("=" * 60)
        logger.info("🚀 SK4FiLM v9.0 - COMPLETE THUMBNAIL SYSTEM")
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
        
        # ✅ START TELEGRAM BOT
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
        
        # Initialize Poster Fetcher
        if PosterFetcher is not None:
            poster_fetcher = PosterFetcher(Config, cache_manager)
            logger.info("✅ Poster Fetcher initialized")
        
        # Initialize Thumbnail Manager
        global thumbnail_manager
        if THUMBNAIL_MANAGER_AVAILABLE:
            thumbnail_manager = ThumbnailManager(mongo_client, Config, bot_handler)
            await thumbnail_manager.initialize()
            logger.info("✅ Thumbnail Manager initialized")
        else:
            logger.warning("⚠️ Thumbnail Manager not available, using fallback")
            thumbnail_manager = ThumbnailManager(mongo_client, Config, bot_handler)
            await thumbnail_manager.initialize()
        
        # Initialize Telegram Sessions
        if PYROGRAM_AVAILABLE:
            telegram_ok = await init_telegram_sessions()
            if not telegram_ok:
                logger.warning("⚠️ Telegram sessions failed")
        
        # Extract thumbnails for existing files
        if thumbnail_manager and files_col is not None:
            logger.info("🔄 Extracting thumbnails for existing files...")
            asyncio.create_task(extract_thumbnails_for_existing_files())
        
        # Start OPTIMIZED indexing
        if user_session_ready and files_col is not None:
            logger.info("🔄 Starting OPTIMIZED file channel indexing...")
            asyncio.create_task(initial_indexing_optimized())
        
        init_time = time.time() - start_time
        logger.info(f"⚡ SK4FiLM Started in {init_time:.2f}s")
        logger.info("=" * 60)
        
        logger.info("🔧 COMPLETE THUMBNAIL SYSTEM:")
        logger.info(f"   • One Movie → One Thumbnail: ✅ ENABLED")
        logger.info(f"   • 4 Qualities → Same Thumbnail: ✅ ENABLED")
        logger.info(f"   • Existing Files Auto-Extract: ✅ ENABLED")
        logger.info(f"   • New Files Auto-Extract: ✅ ENABLED")
        logger.info(f"   • Priority: Extracted → Poster → Empty: ✅ ENABLED")
        logger.info(f"   • No Default/Fallback Image: ✅ ENABLED")
        logger.info(f"   • MongoDB Optimized: ✅ ENABLED")
        logger.info(f"   • Thumbnail Success Rate: 99% TARGET")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ System initialization failed: {e}")
        return False

# ============================================================================
# ✅ BOT INITIALIZATION FUNCTION
# ============================================================================

async def start_telegram_bot():
    """Start Telegram bot with handlers"""
    try:
        if not PYROGRAM_AVAILABLE:
            logger.warning("❌ Pyrogram not available, bot won't start")
            return None
        
        # Check if bot token is available
        if not Config.BOT_TOKEN:
            logger.warning("❌ Bot token not configured, bot won't start")
            return None
        
        logger.info("🤖 Starting SK4FiLM Telegram Bot...")
        
        # Import bot handler
        try:
            from bot_handlers import SK4FiLMBot
            logger.info("✅ Bot handler module imported")
        except ImportError as e:
            logger.error(f"❌ Bot handler import error: {e}")
            # Create fallback bot
            class FallbackBot:
                def __init__(self):
                    self.bot_started = False
                async def initialize(self): 
                    logger.warning("⚠️ Using fallback bot")
                    return False
                async def shutdown(self): 
                    logger.info("✅ Fallback bot shutdown")
            return FallbackBot()
        
        # Initialize bot
        bot_instance = SK4FiLMBot(Config, db_manager=None)
        
        # Start bot
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
# ✅ UTILITY FUNCTIONS
# ============================================================================

def channel_name_cached(cid):
    """Get channel name from cache or return default"""
    return f"Channel {cid}"

# ============================================================================
# ✅ HOME MOVIES
# ============================================================================

@performance_monitor.measure("home_movies")
@async_cache_with_ttl(maxsize=1, ttl=60)
async def get_home_movies(limit=25):
    """Get home movies"""
    try:
        if User is None or not user_session_ready:
            return []
        
        movies = []
        seen_titles = set()
        
        logger.info(f"🎬 Fetching home movies ({limit})...")
        
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
                    
                    # Get normalized title for thumbnail lookup
                    norm_title = normalize_title(clean_title)
                    
                    # Try to get thumbnail from ThumbnailManager
                    thumbnail_url = ''
                    has_thumbnail = False
                    thumbnail_source = ''
                    
                    if thumbnail_manager:
                        thumbnail_data = await thumbnail_manager.get_thumbnail_for_movie(clean_title)
                        if thumbnail_data and thumbnail_data.get('thumbnail_url'):
                            thumbnail_url = thumbnail_data['thumbnail_url']
                            thumbnail_source = thumbnail_data.get('source', '')
                            has_thumbnail = thumbnail_data.get('has_thumbnail', False)
                    
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
                        'has_poster': False,
                        'has_thumbnail': has_thumbnail,
                        'thumbnail_url': thumbnail_url,
                        'thumbnail_source': thumbnail_source
                    }
                    
                    movies.append(movie_data)
                    
                    if len(movies) >= limit:
                        break
        
        # Fetch posters for all movies in batch
        if movies and poster_fetcher:
            movies_with_posters = await get_posters_for_movies_batch(movies)
            logger.info(f"✅ Fetched {len(movies_with_posters)} home movies")
            return movies_with_posters[:limit]
        else:
            logger.warning("⚠️ No movies found for home page")
            return []
        
    except Exception as e:
        logger.error(f"❌ Home movies error: {e}")
        return []

# ============================================================================
# ✅ API ROUTES
# ============================================================================

@app.route('/')
@performance_monitor.measure("root_endpoint")
async def root():
    if files_col is not None:
        tf = await files_col.count_documents({})
        video_files = await files_col.count_documents({'is_video_file': True})
        thumbnails_extracted = await files_col.count_documents({'thumbnail_extracted': True})
    else:
        tf = 0
        video_files = 0
        thumbnails_extracted = 0
    
    # Get thumbnail stats
    thumbnail_stats = {}
    if thumbnail_manager:
        thumbnail_stats = await thumbnail_manager.get_stats()
    
    # Get indexing status
    indexing_status = await file_indexing_manager.get_indexing_status()
    
    # Get bot handler status
    bot_status = await bot_handler.get_bot_status() if bot_handler and hasattr(bot_handler, 'get_bot_status') else {'initialized': False, 'reason': 'BotHandler unavailable'}
    
    # Get Telegram bot status
    bot_running = telegram_bot is not None and hasattr(telegram_bot, 'bot_started') and telegram_bot.bot_started
    
    return jsonify({
        'status': 'healthy',
        'service': 'SK4FiLM v9.0 - COMPLETE THUMBNAIL SYSTEM',
        'thumbnail_system': {
            'one_movie_one_thumbnail': True,
            'multi_quality_same_thumbnail': True,
            'existing_files_auto_extract': True,
            'new_files_auto_extract': True,
            'no_fallback_image': True,
            'priority_extracted_first': True,
            'thumbnail_manager': thumbnail_manager is not None,
            'success_rate_target': '99%'
        },
        'sessions': {
            'user_session': {
                'ready': user_session_ready,
                'channels': Config.TEXT_CHANNEL_IDS
            },
            'bot_session': {
                'ready': bot_session_ready,
                'channel': Config.FILE_CHANNEL_ID
            },
            'bot_handler': bot_status,
            'telegram_bot': {
                'running': bot_running,
                'initialized': telegram_bot is not None
            }
        },
        'components': {
            'cache': cache_manager is not None,
            'verification': verification_system is not None,
            'premium': premium_system is not None,
            'poster_fetcher': poster_fetcher is not None,
            'database': files_col is not None,
            'thumbnail_manager': thumbnail_manager is not None,
            'bot_handler': bot_handler is not None and bot_handler.initialized,
            'telegram_bot': telegram_bot is not None
        },
        'stats': {
            'total_files': tf,
            'video_files': video_files,
            'thumbnails_extracted': thumbnails_extracted
        },
        'indexing': indexing_status,
        'sync_monitoring': {
            'running': sync_manager.is_monitoring,
            'deleted_count': sync_manager.deleted_count
        },
        'thumbnail_manager': thumbnail_stats,
        'response_time': f"{time.perf_counter():.3f}s"
    })

@app.route('/health')
@performance_monitor.measure("health_endpoint")
async def health():
    indexing_status = await file_indexing_manager.get_indexing_status()
    
    bot_status = await bot_handler.get_bot_status() if bot_handler and hasattr(bot_handler, 'get_bot_status') else {'initialized': False, 'reason': 'BotHandler unavailable'}
    
    thumbnail_stats = {}
    if thumbnail_manager:
        thumbnail_stats = await thumbnail_manager.get_stats()
    
    return jsonify({
        'status': 'ok',
        'thumbnail_system': True,
        'sessions': {
            'user': user_session_ready,
            'bot': bot_session_ready,
            'bot_handler': bot_status.get('initialized') if bot_status else False,
            'telegram_bot': telegram_bot is not None and hasattr(telegram_bot, 'bot_started') and telegram_bot.bot_started
        },
        'indexing': {
            'running': indexing_status['is_running'],
            'last_run': indexing_status['last_run']
        },
        'sync': {
            'running': sync_manager.is_monitoring,
            'auto_delete_enabled': True
        },
        'thumbnails': thumbnail_stats,
        'features': {
            'one_movie_one_thumbnail': True,
            'multi_quality_same_thumbnail': True,
            'no_fallback_image': True,
            'thumbnail_manager': thumbnail_manager is not None
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/movies', methods=['GET'])
@performance_monitor.measure("movies_endpoint")
async def api_movies():
    try:
        # Get home movies
        movies = await get_home_movies(limit=25)
        
        return jsonify({
            'status': 'success' if movies else 'empty',
            'movies': movies,
            'total': len(movies),
            'limit': 25,
            'source': 'telegram',
            'poster_fetcher': poster_fetcher is not None,
            'thumbnail_manager': thumbnail_manager is not None,
            'session_used': 'user',
            'channel_id': Config.MAIN_CHANNEL_ID,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Movies API error: {e}")
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
                'thumbnail_priority': True,
                'no_fallback_image': True,
                'real_message_ids': True,
            },
            'bot_username': Config.BOT_USERNAME,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Search API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/stats', methods=['GET'])
async def api_stats():
    """Get performance statistics"""
    try:
        perf_stats = performance_monitor.get_stats()
        
        if poster_fetcher and hasattr(poster_fetcher, 'get_stats'):
            poster_stats = poster_fetcher.get_stats()
        else:
            poster_stats = {}
        
        # Get database stats
        if files_col is not None:
            total_files = await files_col.count_documents({})
            video_files = await files_col.count_documents({'is_video_file': True})
            thumbnails_extracted = await files_col.count_documents({'thumbnail_extracted': True})
            
            # Get indexing stats
            indexing_status = await file_indexing_manager.get_indexing_status()
            
            # Get sync stats
            sync_stats = {
                'running': sync_manager.is_monitoring,
                'deleted_count': sync_manager.deleted_count,
                'last_sync': sync_manager.last_sync
            }
        else:
            total_files = 0
            video_files = 0
            thumbnails_extracted = 0
            indexing_status = {}
            sync_stats = {}
        
        # Get thumbnail stats
        thumbnail_stats = {}
        if 'thumbnail_manager' in globals() and thumbnail_manager:
            try:
                thumbnail_stats = await thumbnail_manager.get_stats()
            except Exception as e:
                logger.error(f"Thumbnail stats error: {e}")
                thumbnail_stats = {'error': 'unavailable'}
        
        # Get bot handler status
        bot_status = None
        if bot_handler:
            try:
                bot_status = await bot_handler.get_bot_status()
            except:
                bot_status = {'initialized': False}
        
        # Get Telegram bot status
        bot_running = telegram_bot is not None and hasattr(telegram_bot, 'bot_started') and telegram_bot.bot_started
        
        return jsonify({
            'status': 'success',
            'thumbnail_system': True,
            'performance': perf_stats,
            'poster_fetcher': poster_stats,
            'thumbnail_manager': thumbnail_stats,
            'database_stats': {
                'total_files': total_files,
                'video_files': video_files,
                'thumbnails_extracted': thumbnails_extracted,
                'extraction_rate': f"{(thumbnails_extracted/video_files*100):.1f}%" if video_files > 0 else "0%"
            },
            'indexing_stats': indexing_status,
            'sync_stats': sync_stats,
            'bot_handler': bot_status,
            'telegram_bot': {
                'running': bot_running,
                'initialized': telegram_bot is not None
            },
            'thumbnail_features': {
                'one_movie_one_thumbnail': True,
                'multi_quality_same_thumbnail': True,
                'existing_files_auto_extract': True,
                'new_files_auto_extract': True,
                'no_fallback_image': True,
                'thumbnail_manager': thumbnail_manager is not None,
                'priority_extracted_first': True
            },
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Stats API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

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
        
        # Trigger reindexing
        asyncio.create_task(initial_indexing_optimized())
        
        return jsonify({
            'status': 'success',
            'message': 'File channel reindexing started',
            'thumbnail_system': True,
            'thumbnail_manager': thumbnail_manager is not None,
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
        
        # Get database stats
        if files_col is not None:
            total_files = await files_col.count_documents({})
        else:
            total_files = 0
        
        return jsonify({
            'status': 'success',
            'thumbnail_system': True,
            'indexing': indexing_status,
            'database_files': total_files,
            'thumbnail_features': {
                'one_movie_one_thumbnail': True,
                'multi_quality_same_thumbnail': True,
                'existing_files_auto_extract': True,
                'new_files_auto_extract': True,
                'thumbnail_manager': thumbnail_manager is not None
            },
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
                # Clear all cache keys starting with "search_"
                keys = await cache_manager.redis_client.keys("search_*")
                if keys:
                    await cache_manager.redis_client.delete(*keys)
                    logger.info(f"✅ Cleared {len(keys)} search cache keys")
            except Exception as e:
                logger.error(f"❌ Cache clear error: {e}")
        
        return jsonify({
            'status': 'success',
            'message': 'Cache cleared successfully'
        })
    except Exception as e:
        logger.error(f"❌ Clear cache error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/admin/thumbnails/stats', methods=['GET'])
async def api_admin_thumbnails_stats():
    """Get thumbnail statistics"""
    try:
        auth_token = request.headers.get('X-Admin-Token')
        if not auth_token or auth_token != os.environ.get('ADMIN_TOKEN', 'sk4film_admin_123'):
            return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401
        
        if not thumbnail_manager:
            return jsonify({
                'status': 'error',
                'message': 'ThumbnailManager not initialized'
            }), 400
        
        thumbnail_stats = await thumbnail_manager.get_stats()
        
        return jsonify({
            'status': 'success',
            'thumbnail_stats': thumbnail_stats,
            'features': {
                'one_movie_one_thumbnail': True,
                'multi_quality_same_thumbnail': True,
                'existing_files_auto_extract': True,
                'new_files_auto_extract': True,
                'priority_extracted_first': True,
                'no_fallback_image': True
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Thumbnail stats error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/admin/thumbnails/extract-existing', methods=['POST'])
async def api_admin_thumbnails_extract_existing():
    """Admin endpoint to trigger thumbnail extraction for existing files"""
    try:
        auth_token = request.headers.get('X-Admin-Token')
        if not auth_token or auth_token != os.environ.get('ADMIN_TOKEN', 'sk4film_admin_123'):
            return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401
        
        if not thumbnail_manager:
            return jsonify({
                'status': 'error',
                'message': 'ThumbnailManager not initialized'
            }), 400
        
        # Trigger extraction for existing files
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
    
    # Safely shutdown Telegram bot
    if telegram_bot:
        try:
            if hasattr(telegram_bot, 'shutdown'):
                await telegram_bot.shutdown()
                logger.info("✅ Telegram Bot stopped")
            else:
                logger.warning("⚠️ Telegram Bot has no shutdown method")
        except Exception as e:
            logger.error(f"❌ Telegram Bot shutdown error: {e}")
    
    # Stop indexing
    await file_indexing_manager.stop_indexing()
    await sync_manager.stop_sync_monitoring()
    
    # Safely shutdown Thumbnail Manager
    if thumbnail_manager:
        try:
            await thumbnail_manager.shutdown()
            logger.info("✅ Thumbnail Manager stopped")
        except Exception as e:
            logger.error(f"❌ Thumbnail Manager shutdown error: {e}")
    
    # Safely shutdown bot handler
    if bot_handler:
        try:
            await bot_handler.shutdown()
            logger.info("✅ Bot Handler stopped")
        except Exception as e:
            logger.error(f"❌ Bot Handler shutdown error: {e}")
    
    # Close poster fetcher session
    if poster_fetcher is not None and hasattr(poster_fetcher, 'close'):
        try:
            await poster_fetcher.close()
            logger.info("✅ Poster Fetcher closed")
        except:
            logger.warning("⚠️ Could not close Poster Fetcher")
    
    # Close Telegram sessions
    if User is not None:
        shutdown_tasks.append(User.stop())
    
    if Bot is not None:
        shutdown_tasks.append(Bot.stop())
    
    # Close cache manager
    if cache_manager is not None:
        shutdown_tasks.append(cache_manager.stop())
    
    # Close verification system
    if verification_system is not None:
        shutdown_tasks.append(verification_system.stop())
    
    # Close premium system
    if premium_system is not None and hasattr(premium_system, 'stop_cleanup_task'):
        shutdown_tasks.append(premium_system.stop_cleanup_task())
    
    # Execute all shutdown tasks
    if shutdown_tasks:
        results = await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"❌ Shutdown task {i} failed: {result}")
    
    # Close MongoDB
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
    logger.info("🎯 COMPLETE THUMBNAIL SYSTEM")
    logger.info(f"   • One Movie → One Thumbnail: ✅ ENABLED")
    logger.info(f"   • 4 Qualities → Same Thumbnail: ✅ ENABLED")
    logger.info(f"   • Existing Files Auto-Extract: ✅ ENABLED")
    logger.info(f"   • New Files Auto-Extract: ✅ ENABLED")
    logger.info(f"   • Priority: Extracted → Poster → Empty: ✅ ENABLED")
    logger.info(f"   • No Default/Fallback Image: ✅ ENABLED")
    logger.info(f"   • Thumbnail Success Rate Target: 99%")
    logger.info(f"   • MongoDB Optimized: ✅ ENABLED")
    logger.info(f"   • File Channel ID: {Config.FILE_CHANNEL_ID}")
    logger.info(f"   • Telegram Bot: ✅ ENABLED")
    
    try:
        asyncio.run(serve(app, config))
    except KeyboardInterrupt:
        logger.info("👋 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
