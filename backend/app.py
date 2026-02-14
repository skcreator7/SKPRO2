# ============================================================================
# 🚀 SK4FiLM v9.0 - OPTIMIZED: DIRECT TELEGRAM SEARCH + MONGODB THUMBNAILS
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
# ✅ LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

logging.getLogger('asyncio').setLevel(logging.WARNING)
logging.getLogger('pyrogram').setLevel(logging.WARNING)
logging.getLogger('hypercorn').setLevel(logging.WARNING)
logging.getLogger('motor').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

# ============================================================================
# ✅ MODULE IMPORTS WITH FALLBACKS
# ============================================================================

# Cache Manager
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

# Verification System
try:
    from verification import VerificationSystem
    logger.debug("✅ Verification module imported")
except ImportError as e:
    logger.error(f"❌ Verification module import error: {e}")
    VerificationSystem = None
    class VerificationSystem:
        def __init__(self, config, mongo_client):
            self.config = config
            self.mongo_client = mongo_client
        async def check_user_verified(self, user_id, premium_system):
            return True, "User verified"
        async def get_user_verification_info(self, user_id):
            return {"verified": True}
        async def stop(self): pass

# Premium System
try:
    from premium import PremiumSystem, PremiumTier
    logger.debug("✅ Premium module imported")
except ImportError as e:
    logger.error(f"❌ Premium module import error: {e}")
    PremiumSystem = None
    PremiumTier = None
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

# Thumbnail Manager
try:
    from thumbnail_manager import ThumbnailManager
    THUMBNAIL_MANAGER_AVAILABLE = True
    logger.info("✅ ThumbnailManager module imported")
except ImportError as e:
    logger.error(f"❌ ThumbnailManager import error: {e}")
    THUMBNAIL_MANAGER_AVAILABLE = False

# Poster Fetcher
try:
    from poster_fetching import PosterFetcher, PosterSource
    logger.debug("✅ Poster fetching module imported")
except ImportError as e:
    logger.error(f"❌ Poster fetching module import error: {e}")
    PosterFetcher = None
    PosterSource = None

# Utils
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
        title = title.lower().strip()
        title = re.sub(r'\s*\([^)]*\)$', '', title)
        title = re.sub(r'\s*\[[^\]]*\]$', '', title)
        title = re.sub(r'\s*\d{4}$', '', title)
        return title
    
    def extract_title_smart(text):
        if not text:
            return ""
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
        return (datetime.now() - date).days < 7

# ============================================================================
# ✅ FALLBACK THUMBNAIL URL
# ============================================================================
FALLBACK_THUMBNAIL_URL = "https://iili.io/fAeIwv9.th.png"

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
        
        if elapsed > 0.5:
            logger.warning(f"⏱️ {name} took {elapsed:.3f}s")
    
    def get_stats(self):
        return self.measurements

performance_monitor = PerformanceMonitor()

# ============================================================================
# ✅ CONFIGURATION
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
    VERIFICATION_REQUIRED = os.environ.get("VERIFICATION_REQUIRED", "true").lower() == "False"
    VERIFICATION_DURATION = 6 * 60 * 60
    
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
    
    # 🔥 FILE CHANNEL INDEXING SETTINGS
    AUTO_INDEX_INTERVAL = int(os.environ.get("AUTO_INDEX_INTERVAL", "120"))
    BATCH_INDEX_SIZE = int(os.environ.get("BATCH_INDEX_SIZE", "500"))
    MAX_INDEX_LIMIT = int(os.environ.get("MAX_INDEX_LIMIT", "0"))
    INDEX_ALL_HISTORY = os.environ.get("INDEX_ALL_HISTORY", "true").lower() == "true"
    INSTANT_AUTO_INDEX = os.environ.get("INSTANT_AUTO_INDEX", "true").lower() == "true"
    
    # 🔥 SEARCH SETTINGS
    SEARCH_MIN_QUERY_LENGTH = 2
    SEARCH_RESULTS_PER_PAGE = 12
    SEARCH_CACHE_TTL = 300  # 5 minutes cache
    
    # 🔥 THUMBNAIL SETTINGS
    THUMBNAIL_EXTRACTION_ENABLED = os.environ.get("THUMBNAIL_EXTRACTION_ENABLED", "true").lower() == "true"
    THUMBNAIL_BATCH_SIZE = int(os.environ.get("THUMBNAIL_BATCH_SIZE", "10"))
    THUMBNAIL_RETRY_LIMIT = int(os.environ.get("THUMBNAIL_RETRY_LIMIT", "3"))
    THUMBNAIL_MAX_SIZE_KB = int(os.environ.get("THUMBNAIL_MAX_SIZE_KB", "200"))
    THUMBNAIL_TTL_DAYS = int(os.environ.get("THUMBNAIL_TTL_DAYS", "30"))
    
    # 🔥 OPTIMIZED STORAGE SETTINGS
    STORE_ONLY_THUMBNAILS = True  # Sirf thumbnails wali files store karo
    EXTRACT_IN_BACKGROUND = True   # Background mein thumbnails extract karo

# ============================================================================
# ✅ FAST INITIALIZATION
# ============================================================================

app = Quart(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

@app.after_request
async def add_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['X-SK4FiLM-Version'] = '9.0-OPTIMIZED-SEARCH'
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
# ✅ BOT HANDLER MODULE
# ============================================================================

class BotHandler:
    """Bot handler for Telegram bot operations"""
    
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
    
    async def get_file_info(self, channel_id, message_id):
        """Get file information from message"""
        if not self.initialized:
            return None
        
        try:
            message = await self.bot.get_messages(channel_id, message_id)
            if not message:
                return None
            
            file_info = {
                'channel_id': channel_id,
                'message_id': message_id,
                'has_file': False,
                'file_type': None,
                'file_size': 0,
                'file_name': '',
                'caption': message.caption or ''
            }
            
            if message.document:
                file_info.update({
                    'has_file': True,
                    'file_type': 'document',
                    'file_size': message.document.file_size or 0,
                    'file_name': message.document.file_name or '',
                    'mime_type': message.document.mime_type or '',
                    'file_id': message.document.file_id
                })
            elif message.video:
                file_info.update({
                    'has_file': True,
                    'file_type': 'video',
                    'file_size': message.video.file_size or 0,
                    'file_name': message.video.file_name or 'video.mp4',
                    'duration': message.video.duration if hasattr(message.video, 'duration') else 0,
                    'width': message.video.width if hasattr(message.video, 'width') else 0,
                    'height': message.video.height if hasattr(message.video, 'height') else 0,
                    'file_id': message.video.file_id
                })
            
            return file_info
            
        except Exception as e:
            logger.error(f"❌ Get file info error: {e}")
            return None
    
    async def extract_thumbnail(self, channel_id, message_id):
        """Extract thumbnail from video file"""
        if not self.initialized:
            return None
        
        try:
            message = await self.bot.get_messages(channel_id, message_id)
            if not message:
                return None
            
            thumbnail_data = None
            
            if message.video:
                if hasattr(message.video, 'thumbnail') and message.video.thumbnail:
                    thumbnail_file_id = message.video.thumbnail.file_id
                    thumbnail_data = await self._download_file(thumbnail_file_id)
            
            elif message.document and is_video_file(message.document.file_name or ''):
                if hasattr(message.document, 'thumbnail') and message.document.thumbnail:
                    thumbnail_file_id = message.document.thumbnail.file_id
                    thumbnail_data = await self._download_file(thumbnail_file_id)
            
            if thumbnail_data:
                base64_data = base64.b64encode(thumbnail_data).decode('utf-8')
                return f"data:image/jpeg;base64,{base64_data}"
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Extract thumbnail error: {e}")
            return None
    
    async def _download_file(self, file_id):
        """Download file from Telegram"""
        try:
            download_path = await self.bot.download_media(file_id, in_memory=True)
            if not download_path:
                return None
            if isinstance(download_path, bytes):
                return download_path
            else:
                with open(download_path, 'rb') as f:
                    return f.read()
        except Exception as e:
            logger.error(f"❌ Download file error: {e}")
            return None
    
    async def get_bot_status(self):
        """Get bot status information"""
        if not self.initialized:
            return {
                'initialized': False,
                'error': 'Bot not initialized'
            }
        
        try:
            bot_info = await self.bot.get_me()
            return {
                'initialized': True,
                'bot_username': bot_info.username,
                'bot_id': bot_info.id,
                'first_name': bot_info.first_name,
                'last_update': self.last_update.isoformat() if self.last_update else None,
                'is_connected': True
            }
        except Exception as e:
            logger.error(f"Error getting bot status: {e}")
            return {
                'initialized': False,
                'error': str(e)
            }
    
    async def shutdown(self):
        """Shutdown bot handler"""
        logger.info("Shutting down bot handler...")
        self.initialized = False
        if self.bot:
            try:
                await self.bot.stop()
                logger.info("✅ Bot stopped")
            except Exception as e:
                logger.error(f"❌ Error stopping bot: {e}")

bot_handler = BotHandler()

# ============================================================================
# ✅ OPTIMIZED SYNC MANAGEMENT
# ============================================================================

class OptimizedSyncManager:
    def __init__(self):
        self.is_monitoring = False
        self.monitoring_task = None
        self.deleted_count = 0
        self.last_sync = time.time()
        self.sync_lock = asyncio.Lock()
    
    async def start_sync_monitoring(self):
        if self.is_monitoring:
            return
        logger.info("👁️ Starting optimized sync monitoring...")
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self.monitor_channel_sync())
    
    async def stop_sync_monitoring(self):
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except:
                pass
        logger.info("🛑 Sync monitoring stopped")
    
    async def monitor_channel_sync(self):
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
        try:
            async with self.sync_lock:
                if files_col is None or User is None or not user_session_ready:
                    return
                
                current_time = time.time()
                if current_time - self.last_sync < 300:
                    return
                
                self.last_sync = current_time
                logger.info("🔍 Checking for deleted files in Telegram...")
                
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
                    messages = await User.get_messages(Config.FILE_CHANNEL_ID, message_ids)
                    
                    existing_ids = set()
                    if isinstance(messages, list):
                        for msg in messages:
                            if msg and hasattr(msg, 'id'):
                                existing_ids.add(msg.id)
                    
                    for item in message_data:
                        if item['message_id'] not in existing_ids:
                            await files_col.delete_one({"_id": item['db_id']})
                            
                            if thumbnail_manager and item['normalized_title']:
                                await thumbnail_manager.thumbnails_col.delete_one({
                                    "normalized_title": item['normalized_title']
                                })
                            
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
# ✅ QUALITY DETECTION
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

def channel_name_cached(cid):
    return f"Channel {cid}"

# ============================================================================
# ✅ THUMBNAIL/POSTER PRIORITY LOGIC
# ============================================================================

def get_display_thumbnail(result: Dict[str, Any], scenario: str = "search") -> Dict[str, Any]:
    """Get thumbnail/poster based on scenario priority"""
    thumbnail_data = {
        'thumbnail_url': '',
        'source': 'none',
        'has_thumbnail': False,
        'priority_used': None
    }
    
    extracted_thumbnail = result.get('thumbnail_url') or result.get('extracted_thumbnail')
    poster_url = result.get('poster_url')
    
    has_extracted = bool(extracted_thumbnail and 
                        isinstance(extracted_thumbnail, str) and
                        (extracted_thumbnail.startswith(('http', 'data:image'))))
    
    has_poster = bool(poster_url and 
                     isinstance(poster_url, str) and
                     poster_url.startswith('http'))
    
    if scenario == 'home':
        if has_poster:
            thumbnail_data.update({
                'thumbnail_url': poster_url,
                'source': result.get('poster_source', 'poster'),
                'has_thumbnail': True,
                'priority_used': 'priority_1_poster'
            })
            logger.debug(f"🏠 HOME: Using POSTER (Priority 1) - {result.get('title', '')[:30]}")
        elif has_extracted:
            thumbnail_data.update({
                'thumbnail_url': extracted_thumbnail,
                'source': 'extracted',
                'has_thumbnail': True,
                'priority_used': 'priority_2_extracted'
            })
            logger.debug(f"🏠 HOME: Using EXTRACTED (Priority 2) - {result.get('title', '')[:30]}")
        else:
            thumbnail_data.update({
                'thumbnail_url': FALLBACK_THUMBNAIL_URL,
                'source': 'fallback',
                'has_thumbnail': True,
                'priority_used': 'priority_3_fallback'
            })
            logger.debug(f"🏠 HOME: Using FALLBACK (Priority 3) - {result.get('title', '')[:30]}")
    
    else:  # search, file, post
        if has_extracted:
            thumbnail_data.update({
                'thumbnail_url': extracted_thumbnail,
                'source': 'extracted',
                'has_thumbnail': True,
                'priority_used': 'priority_1_extracted'
            })
            logger.debug(f"🔍 SEARCH: Using EXTRACTED (Priority 1) - {result.get('title', '')[:30]}")
        elif has_poster:
            thumbnail_data.update({
                'thumbnail_url': poster_url,
                'source': result.get('poster_source', 'poster'),
                'has_thumbnail': True,
                'priority_used': 'priority_2_poster'
            })
            logger.debug(f"🔍 SEARCH: Using POSTER (Priority 2) - {result.get('title', '')[:30]}")
        else:
            thumbnail_data.update({
                'thumbnail_url': FALLBACK_THUMBNAIL_URL,
                'source': 'fallback',
                'has_thumbnail': True,
                'priority_used': 'priority_3_fallback'
            })
            logger.debug(f"🔍 SEARCH: Using FALLBACK (Priority 3) - {result.get('title', '')[:30]}")
    
    return thumbnail_data

# ============================================================================
# ✅ HOME MOVIES
# ============================================================================

@performance_monitor.measure("home_movies")
@async_cache_with_ttl(maxsize=1, ttl=60)
async def get_home_movies(limit=25):
    """Get home movies - POSTER FIRST, then EXTRACTED, then FALLBACK"""
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
                    
                    year_match = re.search(r'\b(19|20)\d{2}\b', title)
                    year = year_match.group() if year_match else ""
                    
                    clean_title = re.sub(r'\s+\(\d{4}\)$', '', title)
                    clean_title = re.sub(r'\s+\d{4}$', '', clean_title)
                    
                    post_content = msg.text
                    formatted_content = format_post(msg.text, max_length=500)
                    norm_title = normalize_title(clean_title)
                    
                    # Get POSTER first
                    poster_data = None
                    if poster_fetcher:
                        try:
                            poster_data = await get_poster_for_movie(clean_title, year)
                            logger.debug(f"🏠 HOME: Fetched poster for {clean_title[:30]}: {bool(poster_data.get('poster_url'))}")
                        except Exception as e:
                            logger.error(f"❌ Poster fetch error: {e}")
                    
                    # Get EXTRACTED thumbnail from MongoDB
                    extracted_thumb = None
                    if thumbnail_manager and files_col is not None:
                        try:
                            # Check MongoDB for thumbnail
                            thumb_doc = await files_col.find_one(
                                {'normalized_title': norm_title, 'has_any_thumbnail': True},
                                {'qualities': 1}
                            )
                            if thumb_doc:
                                for quality in Config.QUALITY_PRIORITY:
                                    if quality in thumb_doc.get('qualities', {}):
                                        q_data = thumb_doc['qualities'][quality]
                                        if q_data.get('thumbnail_extracted'):
                                            extracted_thumb = q_data.get('thumbnail_url')
                                            break
                        except Exception as e:
                            logger.error(f"❌ Thumbnail fetch error: {e}")
                    
                    # Apply PRIORITY logic
                    display_thumb = get_display_thumbnail({
                        'title': clean_title,
                        'poster_url': poster_data.get('poster_url') if poster_data else None,
                        'poster_source': poster_data.get('source') if poster_data else None,
                        'thumbnail_url': extracted_thumb,
                        'extracted_thumbnail': extracted_thumb
                    }, scenario='home')
                    
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
                        'thumbnail_url': display_thumb['thumbnail_url'],
                        'thumbnail_source': display_thumb['source'],
                        'has_thumbnail': display_thumb['has_thumbnail'],
                        'thumbnail_priority': display_thumb['priority_used'],
                        'poster_url': poster_data.get('poster_url') if poster_data else None,
                        'poster_source': poster_data.get('source') if poster_data else None,
                        'has_poster': bool(poster_data and poster_data.get('poster_url')),
                        'extracted_thumbnail': extracted_thumb,
                        'has_extracted': bool(extracted_thumb)
                    }
                    
                    movies.append(movie_data)
                    
                    if len(movies) >= limit:
                        break
        
        logger.info(f"✅ Home movies fetched: {len(movies)}")
        
        poster_count = sum(1 for m in movies if m.get('thumbnail_source') in ['tmdb', 'omdb', 'poster'])
        extracted_count = sum(1 for m in movies if m.get('thumbnail_source') == 'extracted')
        fallback_count = sum(1 for m in movies if m.get('thumbnail_source') == 'fallback')
        
        logger.info(f"🏠 HOME Thumbnail Sources: Poster: {poster_count}, Extracted: {extracted_count}, Fallback: {fallback_count}")
        
        return movies[:limit]
        
    except Exception as e:
        logger.error(f"❌ Home movies error: {e}")
        return []

# ============================================================================
# ✅ POSTER FETCHING FUNCTIONS
# ============================================================================

async def get_poster_for_movie(title: str, year: str = "", quality: str = "") -> Dict[str, Any]:
    """Get poster for movie - Returns empty string if not found"""
    global poster_fetcher
    
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
        poster_task = asyncio.create_task(poster_fetcher.fetch_poster(title))
        
        try:
            poster_data = await asyncio.wait_for(poster_task, timeout=3.0)
            
            if poster_data and poster_data.get('poster_url'):
                logger.debug(f"✅ Poster fetched: {title[:30]} - {poster_data['source']}")
                poster_data['found'] = True
                return poster_data
            else:
                raise ValueError("Invalid poster data")
                
        except (asyncio.TimeoutError, ValueError, Exception) as e:
            logger.warning(f"⚠️ Poster fetch timeout/error for {title[:30]}: {e}")
            
            if not poster_task.done():
                poster_task.cancel()
            
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
    
    tasks = []
    for movie in movies:
        title = movie.get('title', '')
        year = movie.get('year', '')
        quality = movie.get('quality', '')
        
        task = asyncio.create_task(get_poster_for_movie(title, year, quality))
        tasks.append((movie, task))
    
    for movie, task in tasks:
        try:
            poster_data = await task
            
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
            logger.warning(f"⚠️ Batch poster error for {movie.get('title', '')[:30]}: {e}")
            
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
# ✅ ENHANCED TITLE EXTRACTION FUNCTIONS
# ============================================================================

def extract_clean_title(filename):
    """Extract clean movie title without quality tags, year, etc."""
    if not filename:
        return "Unknown"
    
    # Remove extension
    name = os.path.splitext(filename)[0]
    
    # Replace separators with spaces
    name = re.sub(r'[._\-]', ' ', name)
    
    # Remove ALL quality-related tags
    name = re.sub(r'\b(480p|720p|1080p|2160p|4k|hd|hevc|x264|x265|web-dl|webrip|bluray|hdtv|hdr|dts|ac3|aac|ddp|5\.1|7\.1|2\.0|esub|sub|multi|dual|audio|hindi|english|tamil|telugu|malayalam|kannada|ben|eng|hin|tam|tel|mal|kan)\b.*$', '', name, flags=re.IGNORECASE)
    
    # Remove year at the end
    name = re.sub(r'\s+(19|20)\d{2}\s*$', '', name)
    
    # Remove parentheses content
    name = re.sub(r'\s*\([^)]*\)', '', name)
    name = re.sub(r'\s*\[[^\]]*\]', '', name)
    
    # Clean up spaces
    name = re.sub(r'\s+', ' ', name)
    name = name.strip()
    
    return name if name else "Unknown"

def extract_year(filename):
    """Extract year from filename"""
    if not filename:
        return ""
    
    year_match = re.search(r'\b(19|20)\d{2}\b', filename)
    return year_match.group() if year_match else ""

def has_telegram_thumbnail(message):
    """Check if message has thumbnail in Telegram"""
    try:
        if message.video and hasattr(message.video, 'thumbnail') and message.video.thumbnail:
            return True
        elif message.document and hasattr(message.document, 'thumbnail') and message.document.thumbnail:
            return True
        return False
    except:
        return False

# ============================================================================
# ✅ OPTIMIZED SEARCH - DIRECT TELEGRAM + MONGODB THUMBNAILS
# ============================================================================

@performance_monitor.measure("optimized_search")
@async_cache_with_ttl(maxsize=500, ttl=300)  # 5 minute cache
async def search_movies_optimized(query, limit=15, page=1):
    """
    🔥 OPTIMIZED SEARCH:
    - Direct Telegram search (fast, real-time)
    - MongoDB sirf thumbnails ke liye
    - No database search overhead
    - Fast results
    """
    offset = (page - 1) * limit
    
    logger.info(f"🔍 OPTIMIZED SEARCH for: '{query}'")
    
    # Dictionary to store unique results (by normalized title)
    results_dict = {}
    
    # ============================================================================
    # ✅ STEP 1: Direct Telegram FILE CHANNEL Search (Primary)
    # ============================================================================
    if user_session_ready and User is not None:
        try:
            logger.info(f"📁 Searching FILE CHANNEL directly: {Config.FILE_CHANNEL_ID}")
            file_count = 0
            
            # Search file channel for documents/videos
            async for msg in User.search_messages(
                Config.FILE_CHANNEL_ID, 
                query=query,
                limit=50  # Fetch more for grouping
            ):
                if not msg or (not msg.document and not msg.video):
                    continue
                
                # Extract file info
                file_name = None
                if msg.document:
                    file_name = msg.document.file_name
                elif msg.video:
                    file_name = msg.video.file_name or "video.mp4"
                
                if not file_name or not is_video_file(file_name):
                    continue
                
                # Extract clean title
                clean_title = extract_clean_title(file_name)
                normalized = normalize_title(clean_title)
                quality = detect_quality_enhanced(file_name)
                year = extract_year(file_name)
                
                # Check if message has thumbnail
                has_thumb = has_telegram_thumbnail(msg)
                
                # Create file entry
                file_data = {
                    'quality': quality,
                    'file_name': file_name,
                    'file_size': msg.document.file_size if msg.document else msg.video.file_size,
                    'file_size_formatted': format_size(msg.document.file_size if msg.document else msg.video.file_size),
                    'message_id': msg.id,
                    'file_id': msg.document.file_id if msg.document else msg.video.file_id,
                    'date': msg.date,
                    'has_thumbnail_in_telegram': has_thumb
                }
                
                if normalized not in results_dict:
                    # New movie
                    results_dict[normalized] = {
                        'title': clean_title,
                        'original_title': clean_title,
                        'normalized_title': normalized,
                        'year': year,
                        'channel_id': Config.FILE_CHANNEL_ID,
                        'qualities': {},
                        'available_qualities': [],
                        'has_file': True,
                        'has_post': False,
                        'result_type': 'file',
                        'date': msg.date,
                        'is_new': is_new(msg.date),
                        'thumbnail_url': None,
                        'thumbnail_source': None,
                        'has_thumbnail': False,
                        'poster_url': None,
                        'has_poster': False,
                        'search_score': 10
                    }
                
                # Add quality
                results_dict[normalized]['qualities'][quality] = file_data
                results_dict[normalized]['available_qualities'].append(quality)
                
                # Update best date
                if msg.date and (not results_dict[normalized].get('date') or msg.date > results_dict[normalized]['date']):
                    results_dict[normalized]['date'] = msg.date
                    results_dict[normalized]['is_new'] = is_new(msg.date)
                
                file_count += 1
            
            logger.info(f"📁 Found {file_count} file results, grouped into {len(results_dict)} movies")
            
        except Exception as e:
            logger.error(f"❌ File channel search error: {e}")
    
    # ============================================================================
    # ✅ STEP 2: Direct Telegram TEXT CHANNELS Search (Secondary)
    # ============================================================================
    if user_session_ready and User is not None:
        try:
            logger.info(f"📝 Searching TEXT CHANNELS directly: {Config.TEXT_CHANNEL_IDS}")
            post_count = 0
            
            for channel_id in Config.TEXT_CHANNEL_IDS:
                try:
                    async for msg in User.search_messages(
                        channel_id, 
                        query=query,
                        limit=20
                    ):
                        if not msg or not msg.text or len(msg.text) < 15:
                            continue
                        
                        title = extract_title_smart(msg.text)
                        if not title:
                            continue
                        
                        normalized = normalize_title(title)
                        clean_title = re.sub(r'\s*\(\d{4}\)', '', title)
                        clean_title = re.sub(r'\s+\d{4}$', '', clean_title).strip()
                        
                        year_match = re.search(r'\b(19|20)\d{2}\b', title)
                        year = year_match.group() if year_match else ""
                        
                        if normalized in results_dict:
                            # Merge with existing file result
                            results_dict[normalized]['has_post'] = True
                            results_dict[normalized]['post_content'] = format_post(msg.text, max_length=500)
                            results_dict[normalized]['post_channel_id'] = channel_id
                            results_dict[normalized]['post_message_id'] = msg.id
                            results_dict[normalized]['result_type'] = 'file_and_post'
                            results_dict[normalized]['search_score'] = 20
                        else:
                            # New post-only result
                            results_dict[normalized] = {
                                'title': clean_title,
                                'original_title': title,
                                'normalized_title': normalized,
                                'year': year,
                                'content': format_post(msg.text, max_length=500),
                                'post_content': msg.text,
                                'channel_id': channel_id,
                                'message_id': msg.id,
                                'date': msg.date,
                                'is_new': is_new(msg.date) if msg.date else False,
                                'has_post': True,
                                'has_file': False,
                                'result_type': 'post',
                                'thumbnail_url': None,
                                'thumbnail_source': None,
                                'has_thumbnail': False,
                                'poster_url': None,
                                'has_poster': False,
                                'search_score': 5
                            }
                        
                        post_count += 1
                        
                except Exception as e:
                    logger.debug(f"Text search error in {channel_id}: {e}")
                    continue
            
            logger.info(f"📝 Found {post_count} post results")
            
        except Exception as e:
            logger.error(f"❌ Text channels search error: {e}")
    
    # ============================================================================
    # ✅ STEP 3: Get Thumbnails from MongoDB (Fast - No extraction)
    # ============================================================================
    if thumbnail_manager and results_dict and files_col is not None:
        logger.info(f"🖼️ Getting thumbnails from MongoDB for {len(results_dict)} results...")
        
        # Get all thumbnails in one query
        normalized_titles = list(results_dict.keys())
        
        # Batch query MongoDB for thumbnails
        cursor = files_col.find(
            {
                'normalized_title': {'$in': normalized_titles},
                'has_any_thumbnail': True
            },
            {
                'normalized_title': 1,
                'qualities': 1,
                'qualities_with_thumbnails': 1
            }
        )
        
        thumbnails_map = {}
        async for doc in cursor:
            thumbnails_map[doc['normalized_title']] = doc
        
        # Apply thumbnails to results
        for normalized, result in results_dict.items():
            if normalized in thumbnails_map:
                thumb_data = thumbnails_map[normalized]
                
                # Get best quality with thumbnail
                for quality in Config.QUALITY_PRIORITY:
                    if quality in thumb_data.get('qualities_with_thumbnails', []):
                        quality_data = thumb_data['qualities'].get(quality, {})
                        if quality_data.get('thumbnail_extracted'):
                            result['thumbnail_url'] = quality_data['thumbnail_url']
                            result['thumbnail_source'] = 'extracted'
                            result['has_thumbnail'] = True
                            break
    
    # ============================================================================
    # ✅ STEP 4: Get Posters for Results Without Thumbnails
    # ============================================================================
    if poster_fetcher:
        logger.info(f"🎬 Getting posters for results without thumbnails...")
        
        for normalized, result in results_dict.items():
            if not result.get('has_thumbnail'):
                try:
                    poster_data = await get_poster_for_movie(
                        result.get('title', ''),
                        result.get('year', '')
                    )
                    
                    if poster_data and poster_data.get('poster_url'):
                        result['poster_url'] = poster_data['poster_url']
                        result['poster_source'] = poster_data.get('source')
                        result['has_poster'] = True
                        result['thumbnail_url'] = poster_data['poster_url']
                        result['thumbnail_source'] = poster_data.get('source')
                        result['has_thumbnail'] = True
                        
                except Exception as e:
                    logger.debug(f"Poster error for {result.get('title', '')[:30]}: {e}")
    
    # ============================================================================
    # ✅ STEP 5: Apply Fallback
    # ============================================================================
    for result in results_dict.values():
        if not result.get('has_thumbnail'):
            result['thumbnail_url'] = FALLBACK_THUMBNAIL_URL
            result['thumbnail_source'] = 'fallback'
            result['has_thumbnail'] = True
    
    # ============================================================================
    # ✅ STEP 6: Convert to List and Sort
    # ============================================================================
    all_results = list(results_dict.values())
    
    # Sort by score and date
    all_results.sort(key=lambda x: (
        x.get('search_score', 0),
        x.get('is_new', False),
        x.get('date') if isinstance(x.get('date'), datetime) else datetime.min
    ), reverse=True)
    
    # ============================================================================
    # ✅ STEP 7: Pagination
    # ============================================================================
    total = len(all_results)
    start_idx = offset
    end_idx = offset + limit
    paginated = all_results[start_idx:end_idx]
    
    # Statistics
    file_count = sum(1 for r in all_results if r.get('has_file'))
    post_count = sum(1 for r in all_results if r.get('has_post'))
    combined_count = sum(1 for r in all_results if r.get('has_file') and r.get('has_post'))
    thumbnail_count = sum(1 for r in all_results if r.get('thumbnail_source') == 'extracted')
    poster_count = sum(1 for r in all_results if r.get('thumbnail_source') in ['tmdb', 'omdb', 'poster'])
    
    logger.info("=" * 60)
    logger.info("📊 SEARCH RESULTS SUMMARY:")
    logger.info(f"   • Query: '{query}'")
    logger.info(f"   • Total results: {total}")
    logger.info(f"   • Files found: {file_count}")
    logger.info(f"   • Posts found: {post_count}")
    logger.info(f"   • Combined (file+post): {combined_count}")
    logger.info(f"   • Extracted thumbnails: {thumbnail_count}")
    logger.info(f"   • Posters: {poster_count}")
    logger.info(f"   • Page: {page}/{max(1, (total + limit - 1) // limit)}")
    logger.info("=" * 60)
    
    return {
        'results': paginated,
        'pagination': {
            'current_page': page,
            'total_pages': max(1, (total + limit - 1) // limit) if total > 0 else 1,
            'total_results': total,
            'per_page': limit,
            'has_next': page < ((total + limit - 1) // limit) if total > 0 else False,
            'has_previous': page > 1
        },
        'search_metadata': {
            'query': query,
            'total_results': total,
            'file_results': file_count,
            'post_results': post_count,
            'combined_results': combined_count,
            'extracted_thumbnails': thumbnail_count,
            'posters': poster_count,
            'mode': 'direct_telegram_search',
            'thumbnail_source': 'mongodb_only'
        },
        'bot_username': Config.BOT_USERNAME
    }

# ============================================================================
# ✅ OPTIMIZED FILE INDEXING MANAGER - STORE ONLY THUMBNAILS
# ============================================================================

class OptimizedFileIndexingManager:
    def __init__(self):
        self.is_running = False
        self.indexing_task = None
        self.last_run = None
        self.next_run = None
        self.total_indexed = 0
        self.total_skipped = 0
        self.total_errors = 0
        self.files_with_thumbnails = 0
        self.files_without_thumbnails = 0
        self.indexing_stats = {
            'total_runs': 0,
            'total_messages_fetched': 0,
            'total_videos_found': 0,
            'videos_with_thumbnails': 0,
            'videos_without_thumbnails': 0,
            'total_movies': 0,
            'total_qualities': 0,
            'last_success': None,
            'last_error': None
        }
    
    async def start_indexing(self, force_reindex=False):
        """Start optimized file channel indexing - Store only thumbnails"""
        if self.is_running:
            logger.warning("⚠️ File indexing already running")
            return
        
        logger.info("=" * 60)
        logger.info("🚀 STARTING OPTIMIZED INDEXING - STORE ONLY THUMBNAILS")
        logger.info("=" * 60)
        
        self.is_running = True
        self.indexing_stats['total_runs'] += 1
        
        try:
            await self._run_optimized_indexing(force_reindex)
        except Exception as e:
            logger.error(f"❌ Indexing error: {e}")
            self.indexing_stats['last_error'] = str(e)
        finally:
            self.is_running = False
        
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
    
    async def _run_optimized_indexing(self, force_reindex=False):
        """Index ALL files but store only those with thumbnails"""
        
        # Clear existing database if force reindex
        if force_reindex:
            logger.info("🧹 Force reindex enabled - Clearing existing database...")
            try:
                result = await files_col.delete_many({"channel_id": Config.FILE_CHANNEL_ID})
                logger.info(f"✅ Deleted {result.deleted_count} existing records")
                self.total_indexed = 0
                self.total_skipped = 0
                self.total_errors = 0
                self.files_with_thumbnails = 0
                self.files_without_thumbnails = 0
            except Exception as e:
                logger.error(f"❌ Failed to clear database: {e}")
        
        # Check if we have any working session
        if not user_session_ready and not bot_session_ready:
            logger.error("❌ No Telegram session available - Cannot index files")
            return
        
        # Use bot session if user session not available
        client = User if user_session_ready else Bot
        
        if client is None:
            logger.error("❌ No working Telegram client")
            return
        
        # Get channel info
        try:
            chat = await client.get_chat(Config.FILE_CHANNEL_ID)
            logger.info(f"📢 Channel: {chat.title} (ID: {chat.id})")
        except Exception as e:
            logger.error(f"❌ Cannot access file channel: {e}")
            return
        
        # Dictionary to group by movie
        movies_dict = {}  # normalized_title -> movie_data
        all_messages = []
        offset_id = 0
        batch_size = 200
        empty_batch_count = 0
        max_empty_batches = 3
        
        logger.info("📥 Fetching ALL messages from file channel...")
        
        while self.is_running:
            try:
                messages = []
                async for msg in client.get_chat_history(
                    Config.FILE_CHANNEL_ID,
                    limit=batch_size,
                    offset_id=offset_id
                ):
                    messages.append(msg)
                    if len(messages) >= batch_size:
                        break
                
                if not messages:
                    empty_batch_count += 1
                    if empty_batch_count >= max_empty_batches:
                        logger.info("✅ No more messages to fetch")
                        break
                    await asyncio.sleep(1)
                    continue
                
                empty_batch_count = 0
                all_messages.extend(messages)
                offset_id = messages[-1].id
                
                logger.info(f"📥 Fetched {len(all_messages)} messages so far...")
                self.indexing_stats['total_messages_fetched'] = len(all_messages)
                
                if len(messages) < batch_size:
                    logger.info("✅ Reached end of channel")
                    break
                
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"❌ Error fetching messages: {e}")
                await asyncio.sleep(2)
                continue
        
        logger.info(f"✅ Total messages fetched: {len(all_messages)}")
        
        # Process video files
        video_count = 0
        thumbnail_count = 0
        no_thumbnail_count = 0
        
        for msg in all_messages:
            if not msg or (not msg.document and not msg.video):
                continue
            
            file_name = None
            if msg.document:
                file_name = msg.document.file_name
            elif msg.video:
                file_name = msg.video.file_name or "video.mp4"
            
            if not file_name or not is_video_file(file_name):
                continue
            
            video_count += 1
            
            # Check if message has thumbnail in Telegram
            has_thumb = has_telegram_thumbnail(msg)
            
            if has_thumb:
                thumbnail_count += 1
            else:
                no_thumbnail_count += 1
            
            # Extract movie info
            clean_title = extract_clean_title(file_name)
            normalized = normalize_title(clean_title)
            quality = detect_quality_enhanced(file_name)
            year = extract_year(file_name)
            
            # Create file entry
            file_entry = {
                'file_name': file_name,
                'file_size': msg.document.file_size if msg.document else msg.video.file_size,
                'file_size_formatted': format_size(msg.document.file_size if msg.document else msg.video.file_size),
                'message_id': msg.id,
                'file_id': msg.document.file_id if msg.document else msg.video.file_id,
                'quality': quality,
                'has_thumbnail_in_telegram': has_thumb,
                'thumbnail_extracted': False,
                'thumbnail_url': None,
                'date': msg.date
            }
            
            # Group by movie
            if normalized not in movies_dict:
                movies_dict[normalized] = {
                    'title': clean_title,
                    'original_title': clean_title,
                    'normalized_title': normalized,
                    'year': year,
                    'qualities': {},
                    'qualities_with_thumbnails': [],
                    'qualities_without_thumbnails': [],
                    'total_files': 0,
                    'files_with_thumbnails': 0,
                    'files_without_thumbnails': 0,
                    'channel_id': Config.FILE_CHANNEL_ID,
                    'has_any_thumbnail': False,
                    'thumbnail_extraction_pending': False
                }
            
            movies_dict[normalized]['qualities'][quality] = file_entry
            movies_dict[normalized]['total_files'] += 1
            
            if has_thumb:
                movies_dict[normalized]['files_with_thumbnails'] += 1
                movies_dict[normalized]['qualities_with_thumbnails'].append(quality)
                movies_dict[normalized]['has_any_thumbnail'] = True
                
                # Mark for extraction if we have a working session
                if user_session_ready or bot_session_ready:
                    movies_dict[normalized]['thumbnail_extraction_pending'] = True
            else:
                movies_dict[normalized]['files_without_thumbnails'] += 1
                movies_dict[normalized]['qualities_without_thumbnails'].append(quality)
        
        self.indexing_stats['total_videos_found'] = video_count
        self.indexing_stats['videos_with_thumbnails'] = thumbnail_count
        self.indexing_stats['videos_without_thumbnails'] = no_thumbnail_count
        self.files_with_thumbnails = thumbnail_count
        self.files_without_thumbnails = no_thumbnail_count
        
        logger.info("=" * 60)
        logger.info("📊 SCANNING COMPLETE")
        logger.info(f"   • Total video files: {video_count}")
        logger.info(f"   • Files WITH thumbnails: {thumbnail_count}")
        logger.info(f"   • Files WITHOUT thumbnails: {no_thumbnail_count}")
        logger.info(f"   • Unique movies: {len(movies_dict)}")
        logger.info(f"   • Avg files per movie: {video_count/len(movies_dict):.1f}" if movies_dict else "0")
        logger.info("=" * 60)
        
        # Save ONLY movies that have thumbnails to database
        logger.info("💾 Saving ONLY movies with thumbnails to database...")
        
        movies_saved = 0
        total_qualities = 0
        movies_with_thumbnails = 0
        
        for normalized, movie in movies_dict.items():
            # Sirf unhi movies ko store karo jinke pas至少 ek thumbnail hai
            if movie['has_any_thumbnail']:
                # Sirf un qualities ko store karo jinke pas thumbnail hai
                qualities_with_thumbnails = {}
                
                for quality, file_data in movie['qualities'].items():
                    if file_data['has_thumbnail_in_telegram']:
                        qualities_with_thumbnails[quality] = file_data
                
                # Create movie document
                movie_doc = {
                    'title': movie['title'],
                    'original_title': movie['original_title'],
                    'normalized_title': movie['normalized_title'],
                    'year': movie['year'],
                    'channel_id': Config.FILE_CHANNEL_ID,
                    'qualities': qualities_with_thumbnails,
                    'available_qualities': list(qualities_with_thumbnails.keys()),
                    'qualities_with_thumbnails': list(qualities_with_thumbnails.keys()),
                    'total_qualities': len(qualities_with_thumbnails),
                    'files_with_thumbnails': movie['files_with_thumbnails'],
                    'has_any_thumbnail': True,
                    'thumbnail_extraction_pending': True,  # Baad mein extract karenge
                    'indexed_at': datetime.now(),
                    'poster_url': None,
                    'poster_source': None,
                    'has_poster': False
                }
                
                # Use update_one with upsert to avoid duplicate key errors
                try:
                    await files_col.update_one(
                        {'normalized_title': normalized, 'channel_id': Config.FILE_CHANNEL_ID},
                        {'$set': movie_doc},
                        upsert=True
                    )
                    movies_saved += 1
                    total_qualities += len(qualities_with_thumbnails)
                    movies_with_thumbnails += 1
                    
                except Exception as e:
                    if "duplicate key" in str(e).lower():
                        logger.debug(f"⚠️ Duplicate movie skipped: {movie['title']}")
                    else:
                        logger.error(f"❌ Error saving movie {movie['title']}: {e}")
        
        self.total_indexed = movies_saved
        self.indexing_stats['total_movies'] = movies_saved
        self.indexing_stats['total_qualities'] = total_qualities
        self.indexing_stats['movies_with_thumbnails'] = movies_with_thumbnails
        self.indexing_stats['movies_without_thumbnails'] = movies_saved - movies_with_thumbnails
        
        logger.info("=" * 60)
        logger.info("✅ DATABASE STORAGE COMPLETE")
        logger.info(f"   • Movies stored: {movies_saved}")
        logger.info(f"   • Total qualities stored: {total_qualities}")
        logger.info(f"   • Storage saved: {video_count - total_qualities} files")
        logger.info("=" * 60)
        
        self.indexing_stats['last_success'] = datetime.now()
        self.last_run = datetime.now()
        
        # Start background thumbnail extraction for movies that have thumbnails
        if Config.EXTRACT_IN_BACKGROUND and (user_session_ready or bot_session_ready) and movies_with_thumbnails > 0:
            asyncio.create_task(self._extract_thumbnails_background())
        
        # Verify indexing
        await self._verify_indexing()
    
    async def _extract_thumbnails_background(self):
        """Background mein dheere dheere saare thumbnails extract karo"""
        logger.info("🔄 Starting background thumbnail extraction...")
        
        # Un movies ko find karo jinka thumbnail extract karna hai
        cursor = files_col.find({
            'channel_id': Config.FILE_CHANNEL_ID,
            'has_any_thumbnail': True,
            'thumbnail_extraction_pending': True
        })
        
        batch_size = 5
        batch = []
        successful = 0
        failed = 0
        total_pending = await files_col.count_documents({
            'channel_id': Config.FILE_CHANNEL_ID,
            'has_any_thumbnail': True,
            'thumbnail_extraction_pending': True
        })
        
        logger.info(f"📊 Found {total_pending} movies pending thumbnail extraction")
        
        async for movie in cursor:
            batch.append(movie)
            
            if len(batch) >= batch_size:
                result = await self._process_thumbnail_batch(batch)
                successful += result['successful']
                failed += result['failed']
                batch = []
                await asyncio.sleep(2)  # Rate limiting
        
        if batch:
            result = await self._process_thumbnail_batch(batch)
            successful += result['successful']
            failed += result['failed']
        
        logger.info(f"✅ Background thumbnail extraction complete: {successful} successful, {failed} failed")
    
    async def _process_thumbnail_batch(self, movies):
        """Ek batch of movies ke thumbnails extract karo"""
        result = {'successful': 0, 'failed': 0}
        
        # Use bot or user session
        client = User if user_session_ready else Bot
        
        for movie in movies:
            for quality, file_data in movie['qualities'].items():
                # Sirf un qualities ke liye extract karo jinke pas thumbnail hai
                if file_data.get('has_thumbnail_in_telegram') and not file_data.get('thumbnail_extracted'):
                    try:
                        thumbnail_url = None
                        
                        # Try bot handler first (it has better thumbnail extraction)
                        if bot_handler and bot_handler.initialized:
                            thumbnail_url = await bot_handler.extract_thumbnail(
                                Config.FILE_CHANNEL_ID,
                                file_data['message_id']
                            )
                        
                        # Try direct client if bot handler fails
                        if not thumbnail_url and client:
                            message = await client.get_messages(
                                Config.FILE_CHANNEL_ID,
                                file_data['message_id']
                            )
                            
                            if message:
                                if message.video and hasattr(message.video, 'thumbnail') and message.video.thumbnail:
                                    thumb_file_id = message.video.thumbnail.file_id
                                    thumb_data = await client.download_media(thumb_file_id, in_memory=True)
                                    if thumb_data:
                                        if isinstance(thumb_data, bytes):
                                            thumbnail_url = f"data:image/jpeg;base64,{base64.b64encode(thumb_data).decode()}"
                                
                                elif message.document and hasattr(message.document, 'thumbnail') and message.document.thumbnail:
                                    thumb_file_id = message.document.thumbnail.file_id
                                    thumb_data = await client.download_media(thumb_file_id, in_memory=True)
                                    if thumb_data:
                                        if isinstance(thumb_data, bytes):
                                            thumbnail_url = f"data:image/jpeg;base64,{base64.b64encode(thumb_data).decode()}"
                        
                        if thumbnail_url:
                            # Store thumbnail in database
                            await files_col.update_one(
                                {'_id': movie['_id']},
                                {
                                    '$set': {
                                        f'qualities.{quality}.thumbnail_url': thumbnail_url,
                                        f'qualities.{quality}.thumbnail_extracted': True,
                                        f'qualities.{quality}.thumbnail_extracted_at': datetime.now()
                                    }
                                }
                            )
                            logger.info(f"✅ Extracted: {movie['title']} - {quality}")
                            result['successful'] += 1
                        else:
                            logger.warning(f"⚠️ Could not extract: {movie['title']} - {quality}")
                            result['failed'] += 1
                        
                    except Exception as e:
                        logger.error(f"❌ Failed: {movie['title']} - {quality}: {e}")
                        result['failed'] += 1
                    
                    await asyncio.sleep(1)  # Rate limiting
            
            # Mark movie as processed
            await files_col.update_one(
                {'_id': movie['_id']},
                {'$set': {'thumbnail_extraction_pending': False}}
            )
        
        return result
    
    async def _verify_indexing(self):
        """Verify indexing is working"""
        try:
            total_movies = await files_col.count_documents({'channel_id': Config.FILE_CHANNEL_ID})
            
            # Count total qualities
            pipeline = [
                {'$match': {'channel_id': Config.FILE_CHANNEL_ID}},
                {'$group': {
                    '_id': None,
                    'total_qualities': {'$sum': '$total_qualities'}
                }}
            ]
            result = await files_col.aggregate(pipeline).to_list(1)
            total_qualities = result[0]['total_qualities'] if result else 0
            
            logger.info("=" * 60)
            logger.info("🔍 VERIFYING INDEXING...")
            logger.info("=" * 60)
            logger.info(f"📊 Movies in database: {total_movies}")
            logger.info(f"📊 Total qualities: {total_qualities}")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"❌ Verification error: {e}")
    
    async def _indexing_loop(self):
        """Background indexing loop for new files"""
        while self.is_running:
            try:
                if self.next_run and self.next_run > datetime.now():
                    wait_seconds = (self.next_run - datetime.now()).total_seconds()
                    if wait_seconds > 30:
                        logger.info(f"⏰ Next index in {wait_seconds:.0f}s")
                    await asyncio.sleep(min(wait_seconds, 30))
                    continue
                
                # Check for new files only
                if not self.is_running:
                    await self._index_new_files()
                    self.next_run = datetime.now() + timedelta(seconds=Config.AUTO_INDEX_INTERVAL)
                    self.last_run = datetime.now()
                
                await asyncio.sleep(10)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Indexing loop error: {e}")
                await asyncio.sleep(60)
    
    async def _index_new_files(self):
        """Index only new files (after last indexed message)"""
        try:
            # Find the latest message_id in database
            latest = await files_col.find_one(
                {'channel_id': Config.FILE_CHANNEL_ID},
                sort=[('message_id', -1)]
            )
            last_message_id = latest.get('message_id', 0) if latest else 0
            
            logger.info(f"🔍 Checking for new files after message ID: {last_message_id}")
            
            # Use appropriate client
            client = User if user_session_ready else Bot
            if not client:
                return
            
            # Fetch new messages
            messages = []
            async for msg in client.get_chat_history(
                Config.FILE_CHANNEL_ID,
                limit=100
            ):
                if msg.id > last_message_id:
                    messages.append(msg)
                else:
                    break
            
            if not messages:
                logger.info("✅ No new files found")
                return
            
            logger.info(f"📥 Found {len(messages)} new messages")
            
            # Process new messages (oldest first)
            for msg in reversed(messages):
                if not msg or (not msg.document and not msg.video):
                    continue
                
                file_name = None
                if msg.document:
                    file_name = msg.document.file_name
                elif msg.video:
                    file_name = msg.video.file_name or "video.mp4"
                
                if not file_name or not is_video_file(file_name):
                    continue
                
                await self._process_single_message(msg)
                await asyncio.sleep(0.5)  # Rate limiting
            
        except Exception as e:
            logger.error(f"❌ Error indexing new files: {e}")
    
    async def _process_single_message(self, msg):
        """Process a single message for indexing"""
        try:
            file_name = None
            if msg.document:
                file_name = msg.document.file_name
            elif msg.video:
                file_name = msg.video.file_name or "video.mp4"
            
            has_thumb = has_telegram_thumbnail(msg)
            
            # Sirf thumbnails wali files ko store karo
            if not has_thumb:
                return
            
            clean_title = extract_clean_title(file_name)
            normalized = normalize_title(clean_title)
            quality = detect_quality_enhanced(file_name)
            year = extract_year(file_name)
            
            file_entry = {
                'file_name': file_name,
                'file_size': msg.document.file_size if msg.document else msg.video.file_size,
                'file_size_formatted': format_size(msg.document.file_size if msg.document else msg.video.file_size),
                'message_id': msg.id,
                'file_id': msg.document.file_id if msg.document else msg.video.file_id,
                'quality': quality,
                'has_thumbnail_in_telegram': has_thumb,
                'thumbnail_extracted': False,
                'thumbnail_url': None,
                'date': msg.date
            }
            
            # Check if movie exists
            existing = await files_col.find_one({
                'normalized_title': normalized,
                'channel_id': Config.FILE_CHANNEL_ID
            })
            
            if existing:
                # Update existing movie
                update = {
                    f'qualities.{quality}': file_entry,
                    '$addToSet': {
                        'available_qualities': quality,
                        'qualities_with_thumbnails': quality
                    },
                    '$inc': {
                        'total_qualities': 1,
                        'files_with_thumbnails': 1
                    },
                    '$set': {
                        'has_any_thumbnail': True,
                        'thumbnail_extraction_pending': True
                    }
                }
                
                await files_col.update_one(
                    {'_id': existing['_id']},
                    update
                )
                
                logger.info(f"✅ Updated movie: {clean_title} - Added {quality}")
                
            else:
                # Create new movie
                movie_doc = {
                    'title': clean_title,
                    'original_title': clean_title,
                    'normalized_title': normalized,
                    'year': year,
                    'channel_id': Config.FILE_CHANNEL_ID,
                    'qualities': {quality: file_entry},
                    'available_qualities': [quality],
                    'qualities_with_thumbnails': [quality],
                    'total_qualities': 1,
                    'files_with_thumbnails': 1,
                    'has_any_thumbnail': True,
                    'thumbnail_extraction_pending': True,
                    'indexed_at': datetime.now()
                }
                
                await files_col.insert_one(movie_doc)
                logger.info(f"✅ New movie: {clean_title} - {quality}")
                
        except Exception as e:
            logger.error(f"❌ Error processing message {msg.id}: {e}")
    
    async def get_indexing_status(self):
        """Get current indexing status"""
        thumbnail_stats = {}
        if thumbnail_manager:
            try:
                thumbnail_stats = await thumbnail_manager.get_stats()
            except:
                pass
        
        total_movies = 0
        total_qualities = 0
        
        if files_col is not None:
            total_movies = await files_col.count_documents({'channel_id': Config.FILE_CHANNEL_ID})
            
            pipeline = [
                {'$match': {'channel_id': Config.FILE_CHANNEL_ID}},
                {'$group': {
                    '_id': None,
                    'total_qualities': {'$sum': '$total_qualities'}
                }}
            ]
            result = await files_col.aggregate(pipeline).to_list(1)
            if result:
                total_qualities = result[0]['total_qualities']
        
        return {
            'is_running': self.is_running,
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'next_run': self.next_run.isoformat() if self.next_run else None,
            'total_movies': total_movies,
            'total_qualities': total_qualities,
            'stats': self.indexing_stats,
            'thumbnail_stats': thumbnail_stats,
            'user_session_ready': user_session_ready,
            'bot_session_ready': bot_session_ready,
            'store_only_thumbnails': Config.STORE_ONLY_THUMBNAILS,
            'extract_in_background': Config.EXTRACT_IN_BACKGROUND
        }

file_indexing_manager = OptimizedFileIndexingManager()

# ============================================================================
# ✅ DATABASE INDEXES
# ============================================================================

async def setup_database_indexes():
    """Create all necessary database indexes"""
    if files_col is None:
        logger.warning("⚠️ Files collection not available")
        return
    
    try:
        # Drop existing text index if it exists (to avoid conflict)
        try:
            await files_col.drop_index("title_text_search")
            logger.info("✅ Dropped existing text index")
        except:
            pass
        
        # Unique index for normalized_title + channel_id
        await files_col.create_index(
            [("normalized_title", 1), ("channel_id", 1)],
            unique=True,
            name="title_channel_unique",
            background=True
        )
        
        # Individual indexes
        await files_col.create_index("year", background=True)
        await files_col.create_index("channel_id", background=True)
        await files_col.create_index("has_any_thumbnail", background=True)
        await files_col.create_index("thumbnail_extraction_pending", background=True)
        
        # Compound indexes
        await files_col.create_index(
            [("channel_id", 1), ("has_any_thumbnail", 1)],
            name="channel_thumbnails",
            background=True
        )
        
        await files_col.create_index(
            [("normalized_title", 1), ("year", 1)],
            name="title_year",
            background=True
        )
        
        # Index for message_id (for sync)
        await files_col.create_index(
            [("channel_id", 1), ("message_id", -1)],
            name="channel_message",
            background=True
        )
        
        logger.info("✅ All database indexes created successfully")
        
    except Exception as e:
        logger.warning(f"⚠️ Index creation error (may already exist): {e}")

# ============================================================================
# ✅ INITIAL INDEXING FUNCTION
# ============================================================================

async def initial_indexing_optimized():
    """Start optimized file channel indexing - store only thumbnails"""
    global file_indexing_manager
    
    # Check if we have any working session
    if not user_session_ready and not bot_session_ready:
        logger.error("❌ No Telegram session available - Cannot index files")
        logger.info("📝 Please check:")
        logger.info("   1. BOT_TOKEN is correct")
        logger.info("   2. USER_SESSION_STRING is correct")
        logger.info("   3. Bot is a member of the file channel")
        return
    
    if files_col is None:
        logger.error("❌ Database not ready - Cannot index files")
        return
    
    logger.info("=" * 60)
    logger.info("🚀 STARTING OPTIMIZED FILE CHANNEL INDEXING")
    logger.info("=" * 60)
    logger.info(f"📁 Channel ID: {Config.FILE_CHANNEL_ID}")
    logger.info(f"🔄 Force reindex: {Config.INDEX_ALL_HISTORY}")
    logger.info(f"✅ Store only thumbnails: {Config.STORE_ONLY_THUMBNAILS}")
    logger.info(f"✅ Background extraction: {Config.EXTRACT_IN_BACKGROUND}")
    logger.info(f"✅ Using: {'User' if user_session_ready else 'Bot'} session")
    logger.info("=" * 60)
    
    try:
        # Create indexes first
        await setup_database_indexes()
        
        # Start indexing with force reindex
        await file_indexing_manager.start_indexing(force_reindex=Config.INDEX_ALL_HISTORY)
        
        # Start sync monitoring if user session available
        if user_session_ready:
            await sync_manager.start_sync_monitoring()
        
        logger.info("✅ Indexing started successfully")
        
    except Exception as e:
        logger.error(f"❌ Initial indexing error: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# ✅ THUMBNAIL EXTRACTION FOR EXISTING FILES
# ============================================================================

async def extract_thumbnails_for_existing_files():
    """Extract thumbnails for existing files in database"""
    if not thumbnail_manager or files_col is None:
        logger.warning("⚠️ ThumbnailManager or files collection not available")
        return
    
    if not user_session_ready and not bot_session_ready:
        logger.warning("⚠️ No Telegram session available for thumbnail extraction")
        return
    
    logger.info("🔍 Extracting thumbnails for existing files...")
    
    try:
        # Find movies that need thumbnail extraction
        cursor = files_col.find({
            'channel_id': Config.FILE_CHANNEL_ID,
            'has_any_thumbnail': True,
            'thumbnail_extraction_pending': True
        })
        
        movies_to_process = []
        async for doc in cursor:
            movies_to_process.append(doc)
        
        logger.info(f"📊 Found {len(movies_to_process)} movies needing thumbnails")
        
        if not movies_to_process:
            logger.info("✅ No movies need thumbnail extraction")
            return
        
        batch_size = 5
        total_batches = math.ceil(len(movies_to_process) / batch_size)
        successful = 0
        failed = 0
        
        client = User if user_session_ready else Bot
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(movies_to_process))
            batch = movies_to_process[start_idx:end_idx]
            
            logger.info(f"🖼️ Processing batch {batch_num + 1}/{total_batches} ({len(batch)} movies)...")
            
            for movie in batch:
                for quality, file_data in movie['qualities'].items():
                    if file_data.get('has_thumbnail_in_telegram') and not file_data.get('thumbnail_extracted'):
                        try:
                            thumbnail_url = None
                            
                            # Try bot handler first
                            if bot_handler and bot_handler.initialized:
                                thumbnail_url = await bot_handler.extract_thumbnail(
                                    Config.FILE_CHANNEL_ID,
                                    file_data['message_id']
                                )
                            
                            # Try direct client
                            if not thumbnail_url and client:
                                message = await client.get_messages(
                                    Config.FILE_CHANNEL_ID,
                                    file_data['message_id']
                                )
                                
                                if message:
                                    if message.video and hasattr(message.video, 'thumbnail') and message.video.thumbnail:
                                        thumb_file_id = message.video.thumbnail.file_id
                                        thumb_data = await client.download_media(thumb_file_id, in_memory=True)
                                        if thumb_data:
                                            if isinstance(thumb_data, bytes):
                                                thumbnail_url = f"data:image/jpeg;base64,{base64.b64encode(thumb_data).decode()}"
                                    
                                    elif message.document and hasattr(message.document, 'thumbnail') and message.document.thumbnail:
                                        thumb_file_id = message.document.thumbnail.file_id
                                        thumb_data = await client.download_media(thumb_file_id, in_memory=True)
                                        if thumb_data:
                                            if isinstance(thumb_data, bytes):
                                                thumbnail_url = f"data:image/jpeg;base64,{base64.b64encode(thumb_data).decode()}"
                            
                            if thumbnail_url:
                                await files_col.update_one(
                                    {'_id': movie['_id']},
                                    {
                                        '$set': {
                                            f'qualities.{quality}.thumbnail_url': thumbnail_url,
                                            f'qualities.{quality}.thumbnail_extracted': True,
                                            f'qualities.{quality}.thumbnail_extracted_at': datetime.now()
                                        }
                                    }
                                )
                                successful += 1
                                logger.info(f"✅ Extracted: {movie['title']} - {quality}")
                            else:
                                failed += 1
                                logger.warning(f"⚠️ No thumbnail: {movie['title']} - {quality}")
                            
                        except Exception as e:
                            logger.error(f"❌ Error: {movie['title']} - {quality}: {e}")
                            failed += 1
                        
                        await asyncio.sleep(1)
                
                # Mark movie as processed
                await files_col.update_one(
                    {'_id': movie['_id']},
                    {'$set': {'thumbnail_extraction_pending': False}}
                )
            
            if batch_num < total_batches - 1:
                await asyncio.sleep(2)
        
        logger.info(f"✅ Thumbnail extraction complete: {successful} successful, {failed} failed")
        
    except Exception as e:
        logger.error(f"❌ Error extracting thumbnails: {e}")

# ============================================================================
# ✅ TELEGRAM SESSION INITIALIZATION
# ============================================================================

@performance_monitor.measure("telegram_init")
async def init_telegram_sessions():
    global User, Bot, user_session_ready, bot_session_ready
    
    logger.info("=" * 50)
    logger.info("🚀 TELEGRAM SESSION INITIALIZATION")
    logger.info("=" * 50)
    
    if not PYROGRAM_AVAILABLE:
        logger.error("❌ Pyrogram not installed!")
        return False
    
    # Initialize BOT first
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
    
    # Initialize USER session
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
    
    logger.info("\n" + "=" * 50)
    logger.info("📊 SESSION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"BOT Session: {'✅ READY' if bot_session_ready else '❌ NOT READY'}")
    logger.info(f"USER Session: {'✅ READY' if user_session_ready else '❌ NOT READY'}")
    logger.info(f"Bot Handler: {'✅ INITIALIZED' if bot_handler.initialized else '❌ NOT READY'}")
    
    return bot_session_ready or user_session_ready

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
# ✅ BOT INITIALIZATION
# ============================================================================

async def start_telegram_bot():
    """Start the Telegram bot for handling user commands"""
    try:
        if not PYROGRAM_AVAILABLE:
            logger.warning("❌ Pyrogram not available, bot won't start")
            return None
        
        if not Config.BOT_TOKEN:
            logger.warning("❌ Bot token not configured, bot won't start")
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
                async def initialize(self): 
                    logger.warning("⚠️ Using fallback bot")
                    return False
                async def shutdown(self): 
                    logger.info("✅ Fallback bot shutdown")
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
# ✅ THUMBNAIL MANAGER INITIALIZATION
# ============================================================================

async def init_thumbnail_manager():
    """Initialize Thumbnail Manager"""
    global thumbnail_manager
    
    try:
        if THUMBNAIL_MANAGER_AVAILABLE:
            from thumbnail_manager import ThumbnailManager
            thumbnail_manager = ThumbnailManager(
                download_path="downloads/thumbnails",
                mongodb=db,
                bot_client=Bot if bot_session_ready else None,
                user_client=User if user_session_ready else None,
                file_channel_id=Config.FILE_CHANNEL_ID
            )
            await thumbnail_manager.initialize()
            logger.info("✅ Thumbnail Manager initialized successfully")
            return True
        else:
            logger.warning("⚠️ Thumbnail Manager not available")
            return False
    except Exception as e:
        logger.error(f"❌ Thumbnail Manager initialization failed: {e}")
        thumbnail_manager = None
        return False

# ============================================================================
# ✅ MAIN INITIALIZATION
# ============================================================================

@performance_monitor.measure("system_init")
async def init_system():
    start_time = time.time()
    
    try:
        logger.info("=" * 60)
        logger.info("🚀 SK4FiLM v9.0 - OPTIMIZED: DIRECT TELEGRAM SEARCH")
        logger.info("=" * 60)
        
        # Initialize MongoDB
        mongo_ok = await init_mongodb()
        if not mongo_ok:
            logger.error("❌ MongoDB connection failed")
            return False
        
        # Initialize Bot Handler
        bot_handler_ok = await bot_handler.initialize()
        if bot_handler_ok:
            logger.info("✅ Bot Handler initialized")
        
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
            poster_fetcher = PosterFetcher(Config, cache_manager.redis_client if cache_manager else None)
            logger.info("✅ Poster Fetcher initialized")
        
        # Initialize Telegram Sessions
        if PYROGRAM_AVAILABLE:
            telegram_ok = await init_telegram_sessions()
            if not telegram_ok:
                logger.warning("⚠️ Telegram sessions partially failed")
        
        # START TELEGRAM BOT
        global telegram_bot
        telegram_bot = await start_telegram_bot()
        if telegram_bot:
            logger.info("✅ Telegram Bot started successfully")
        else:
            logger.warning("⚠️ Telegram Bot failed to start")
        
        # Initialize Thumbnail Manager
        await init_thumbnail_manager()
        
        # Get current file count
        if files_col is not None:
            file_count = await files_col.count_documents({})
            logger.info(f"📊 Current movies in database: {file_count}")
        
        # Start OPTIMIZED indexing (store only thumbnails)
        if (user_session_ready or bot_session_ready) and files_col is not None:
            logger.info("🔍 Starting OPTIMIZED file channel indexing...")
            asyncio.create_task(initial_indexing_optimized())
        
        # Extract thumbnails for existing files
        if thumbnail_manager and files_col is not None and (user_session_ready or bot_session_ready):
            logger.info("🔍 Checking for pending thumbnail extractions...")
            asyncio.create_task(extract_thumbnails_for_existing_files())
        
        init_time = time.time() - start_time
        logger.info(f"⚡ SK4FiLM Started in {init_time:.2f}s")
        logger.info("=" * 60)
        logger.info("🔧 OPTIMIZED FEATURES:")
        logger.info(f"   • Direct Telegram Search: ✅ ENABLED")
        logger.info(f"   • MongoDB only for thumbnails: ✅ ENABLED")
        logger.info(f"   • Background extraction: {Config.EXTRACT_IN_BACKGROUND}")
        logger.info(f"   • One movie → multiple qualities: ✅ ENABLED")
        logger.info(f"   • HOME: Poster first: ✅ ENABLED")
        logger.info(f"   • SEARCH: Extracted first → Poster → Fallback: ✅ ENABLED")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ System initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# ✅ API ROUTES
# ============================================================================

@app.route('/')
@performance_monitor.measure("root_endpoint")
async def root():
    if files_col is not None:
        total_movies = await files_col.count_documents({})
        
        # Count total qualities
        pipeline = [
            {'$group': {
                '_id': None,
                'total_qualities': {'$sum': '$total_qualities'}
            }}
        ]
        result = await files_col.aggregate(pipeline).to_list(1)
        total_qualities = result[0]['total_qualities'] if result else 0
        
        # Count extracted thumbnails
        extracted_pipeline = [
            {'$match': {'thumbnail_extraction_pending': False}},
            {'$group': {
                '_id': None,
                'extracted': {'$sum': '$total_qualities'}
            }}
        ]
        extracted_result = await files_col.aggregate(extracted_pipeline).to_list(1)
        extracted = extracted_result[0]['extracted'] if extracted_result else 0
    else:
        total_movies = 0
        total_qualities = 0
        extracted = 0
    
    thumbnail_stats = {}
    if thumbnail_manager:
        thumbnail_stats = await thumbnail_manager.get_stats()
    
    indexing_status = await file_indexing_manager.get_indexing_status()
    
    bot_status = None
    if bot_handler:
        try:
            bot_status = await bot_handler.get_bot_status()
        except Exception as e:
            logger.error(f"❌ Error getting bot status: {e}")
            bot_status = {'initialized': False, 'error': str(e)}
    
    bot_running = telegram_bot is not None and hasattr(telegram_bot, 'bot_started') and telegram_bot.bot_started
    
    return jsonify({
        'status': 'healthy',
        'service': 'SK4FiLM v9.0 - DIRECT TELEGRAM SEARCH',
        'search_mode': 'direct_telegram',
        'storage_stats': {
            'total_movies': total_movies,
            'total_qualities': total_qualities,
            'thumbnails_extracted': extracted,
            'extraction_progress': f"{(extracted/total_qualities*100):.1f}%" if total_qualities > 0 else "0%"
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
    
    bot_status = None
    if bot_handler:
        try:
            bot_status = await bot_handler.get_bot_status()
        except:
            bot_status = {'initialized': False}
    
    thumbnail_stats = {}
    if thumbnail_manager:
        thumbnail_stats = await thumbnail_manager.get_stats()
    
    return jsonify({
        'status': 'ok',
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
    """Fast search using direct Telegram + MongoDB thumbnails"""
    try:
        query = request.args.get('query', '').strip()
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', Config.SEARCH_RESULTS_PER_PAGE))
        
        if len(query) < Config.SEARCH_MIN_QUERY_LENGTH:
            return jsonify({
                'status': 'error',
                'message': f'Query must be at least {Config.SEARCH_MIN_QUERY_LENGTH} characters'
            }), 400
        
        # Use optimized search
        result_data = await search_movies_optimized(query, limit, page)
        
        return jsonify({
            'status': 'success',
            'query': query,
            'results': result_data['results'],
            'pagination': result_data['pagination'],
            'search_metadata': result_data.get('search_metadata', {}),
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
    try:
        perf_stats = performance_monitor.get_stats()
        
        if poster_fetcher and hasattr(poster_fetcher, 'get_stats'):
            poster_stats = poster_fetcher.get_stats()
        else:
            poster_stats = {}
        
        if files_col is not None:
            total_movies = await files_col.count_documents({})
            
            # Count total qualities
            pipeline = [
                {'$group': {
                    '_id': None,
                    'total_qualities': {'$sum': '$total_qualities'}
                }}
            ]
            result = await files_col.aggregate(pipeline).to_list(1)
            total_qualities = result[0]['total_qualities'] if result else 0
            
            # Count extracted
            extracted_pipeline = [
                {'$match': {'thumbnail_extraction_pending': False}},
                {'$group': {
                    '_id': None,
                    'extracted': {'$sum': '$total_qualities'}
                }}
            ]
            extracted_result = await files_col.aggregate(extracted_pipeline).to_list(1)
            extracted = extracted_result[0]['extracted'] if extracted_result else 0
            
            indexing_status = await file_indexing_manager.get_indexing_status()
            
            sync_stats = {
                'running': sync_manager.is_monitoring,
                'deleted_count': sync_manager.deleted_count,
                'last_sync': sync_manager.last_sync
            }
        else:
            total_movies = 0
            total_qualities = 0
            extracted = 0
            indexing_status = {}
            sync_stats = {}
        
        thumbnail_stats = {}
        if thumbnail_manager:
            thumbnail_stats = await thumbnail_manager.get_stats()
        
        bot_status = None
        if bot_handler:
            try:
                bot_status = await bot_handler.get_bot_status()
            except:
                bot_status = {'initialized': False}
        
        bot_running = telegram_bot is not None and hasattr(telegram_bot, 'bot_started') and telegram_bot.bot_started
        
        return jsonify({
            'status': 'success',
            'performance': perf_stats,
            'poster_fetcher': poster_stats,
            'thumbnail_manager': thumbnail_stats,
            'database_stats': {
                'total_movies': total_movies,
                'total_qualities': total_qualities,
                'thumbnails_extracted': extracted,
                'extraction_rate': f"{(extracted/total_qualities*100):.1f}%" if total_qualities > 0 else "0%"
            },
            'indexing_stats': indexing_status,
            'sync_stats': sync_stats,
            'bot_handler': bot_status,
            'telegram_bot': {
                'running': bot_running,
                'initialized': telegram_bot is not None
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
    """Trigger complete reindexing"""
    try:
        auth_token = request.headers.get('X-Admin-Token')
        if not auth_token or auth_token != os.environ.get('ADMIN_TOKEN', 'sk4film_admin_123'):
            return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401
        
        # Force reindex
        Config.INDEX_ALL_HISTORY = True
        
        # Start indexing
        asyncio.create_task(initial_indexing_optimized())
        
        return jsonify({
            'status': 'success',
            'message': 'Complete file channel reindexing started',
            'channel_id': Config.FILE_CHANNEL_ID,
            'mode': 'store_only_thumbnails',
            'store_only_thumbnails': Config.STORE_ONLY_THUMBNAILS,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Admin reindex error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/admin/indexing-status', methods=['GET'])
async def api_admin_indexing_status():
    try:
        indexing_status = await file_indexing_manager.get_indexing_status()
        
        return jsonify({
            'status': 'success',
            'indexing': indexing_status,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Indexing status error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/admin/clear-cache', methods=['POST'])
async def api_admin_clear_cache():
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
        
        return jsonify({
            'status': 'success',
            'message': 'Cache cleared successfully'
        })
    except Exception as e:
        logger.error(f"❌ Clear cache error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/admin/thumbnails/stats', methods=['GET'])
async def api_admin_thumbnails_stats():
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
            'thumbnail_stats': thumbnail_stats
        })
        
    except Exception as e:
        logger.error(f"❌ Thumbnail stats error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/admin/thumbnails/extract-existing', methods=['POST'])
async def api_admin_thumbnails_extract_existing():
    try:
        auth_token = request.headers.get('X-Admin-Token')
        if not auth_token or auth_token != os.environ.get('ADMIN_TOKEN', 'sk4film_admin_123'):
            return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401
        
        if not thumbnail_manager:
            return jsonify({
                'status': 'error',
                'message': 'ThumbnailManager not initialized'
            }), 400
        
        asyncio.create_task(extract_thumbnails_for_existing_files())
        
        return jsonify({
            'status': 'success',
            'message': 'Thumbnail extraction started for existing files',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Thumbnail extraction error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/debug/indexing', methods=['GET'])
async def debug_indexing():
    """Debug endpoint to check indexing status"""
    try:
        if files_col is None:
            return jsonify({'error': 'Database not initialized'}), 500
        
        # Get total count
        total_movies = await files_col.count_documents({'channel_id': Config.FILE_CHANNEL_ID})
        
        # Get total qualities
        pipeline = [
            {'$match': {'channel_id': Config.FILE_CHANNEL_ID}},
            {'$group': {
                '_id': None,
                'total_qualities': {'$sum': '$total_qualities'}
            }}
        ]
        result = await files_col.aggregate(pipeline).to_list(1)
        total_qualities = result[0]['total_qualities'] if result else 0
        
        # Get pending extractions
        pending = await files_col.count_documents({
            'channel_id': Config.FILE_CHANNEL_ID,
            'thumbnail_extraction_pending': True
        })
        
        # Get recent movies
        cursor = files_col.find(
            {'channel_id': Config.FILE_CHANNEL_ID}
        ).sort('indexed_at', -1).limit(5)
        
        recent_movies = []
        async for doc in cursor:
            doc['_id'] = str(doc['_id'])
            recent_movies.append({
                'title': doc['title'],
                'qualities': list(doc['qualities'].keys()),
                'total_qualities': doc['total_qualities'],
                'has_thumbnails': doc['has_any_thumbnail'],
                'indexed_at': doc['indexed_at'].isoformat() if isinstance(doc['indexed_at'], datetime) else str(doc['indexed_at'])
            })
        
        # Get indexing status
        indexing_status = await file_indexing_manager.get_indexing_status()
        
        return jsonify({
            'status': 'success',
            'total_movies': total_movies,
            'total_qualities': total_qualities,
            'pending_extractions': pending,
            'recent_movies': recent_movies,
            'indexing_status': indexing_status,
            'config': {
                'file_channel_id': Config.FILE_CHANNEL_ID,
                'store_only_thumbnails': Config.STORE_ONLY_THUMBNAILS,
                'extract_in_background': Config.EXTRACT_IN_BACKGROUND,
                'auto_index_interval': Config.AUTO_INDEX_INTERVAL
            },
            'sessions': {
                'user_session_ready': user_session_ready,
                'bot_session_ready': bot_session_ready
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Debug indexing error: {e}")
        return jsonify({'error': str(e)}), 500

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
    
    if thumbnail_manager:
        try:
            await thumbnail_manager.shutdown()
            logger.info("✅ Thumbnail Manager stopped")
        except Exception as e:
            logger.error(f"❌ Thumbnail Manager shutdown error: {e}")
    
    if bot_handler:
        try:
            await bot_handler.shutdown()
            logger.info("✅ Bot Handler stopped")
        except Exception as e:
            logger.error(f"❌ Bot Handler shutdown error: {e}")
    
    if poster_fetcher is not None and hasattr(poster_fetcher, 'close'):
        try:
            await poster_fetcher.close()
            logger.info("✅ Poster Fetcher closed")
        except:
            pass
    
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
    logger.info(f"📁 File Channel ID: {Config.FILE_CHANNEL_ID}")
    logger.info(f"📊 Telegram files: ~25900")
    logger.info(f"🔍 SEARCH MODE: Direct Telegram (Fast, Real-time)")
    logger.info(f"💾 MongoDB: Only thumbnails (Storage optimized)")
    logger.info(f"🔄 Background extraction: {Config.EXTRACT_IN_BACKGROUND}")
    logger.info(f"🔍 SEARCH: Extracted First → Poster → Fallback")
    logger.info(f"🏠 HOME: Poster First → Extracted → Fallback")
    logger.info(f"📊 Database indexes will be created automatically")
    logger.info(f"🔄 Auto-indexing every {Config.AUTO_INDEX_INTERVAL}s")
    
    try:
        asyncio.run(serve(app, config))
    except KeyboardInterrupt:
        logger.info("👋 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
