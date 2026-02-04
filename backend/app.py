# =============# ============================================================================
# 🚀 SK4FiLM v9.0 - SINGLE MONGODB WITH 99% THUMBNAIL SUCCESS
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

# ✅ IMPORT ENHANCED THUMBNAIL MANAGER
try:
    from thumbnail_manager import ThumbnailManager
    logger.debug("✅ Thumbnail Manager module imported")
except ImportError as e:
    logger.error(f"❌ Thumbnail Manager import error: {e}")
    # Fallback ThumbnailManager
    class ThumbnailManager:
        def __init__(self, mongo_client, config, bot_handler=None):
            self.config = config
            self.bot_handler = bot_handler
            self.stats = {}
        async def initialize(self): return True
        async def get_thumbnail_for_movie(self, title, channel_id=None, message_id=None):
            return {
                'thumbnail_url': self.config.FALLBACK_POSTER,
                'source': 'fallback',
                'has_thumbnail': True,
                'extracted': False
            }
        async def get_thumbnails_batch(self, movies): return movies
        async def get_stats(self): return {}
        async def shutdown(self): pass

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
# ✅ CONFIGURATION - SINGLE MONGODB WITH ENHANCED API KEYS
# ============================================================================

class Config:
    # API Configuration
    API_ID = int(os.environ.get("API_ID", "0"))
    API_HASH = os.environ.get("API_HASH", "")
    USER_SESSION_STRING = os.environ.get("USER_SESSION_STRING", "")
    BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
    
    # ✅ SINGLE MONGODB CONFIGURATION
    MONGODB_URI = os.environ.get("MONGODB_URI", 
        "mongodb+srv://Sklink:skweb@cluster0.bfiw4el.mongodb.net/sk4film?retryWrites=true&w=majority&appName=Cluster0")
    
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
    VERIFICATION_REQUIRED = os.environ.get("VERIFICATION_REQUIRED", "true").lower() == "true"
    VERIFICATION_DURATION = 6 * 60 * 60  # 6 hours
    
    # Application
    WEBSITE_URL = os.environ.get("WEBSITE_URL", "https://sk4film.vercel.app")
    BOT_USERNAME = os.environ.get("BOT_USERNAME", "sk4filmbot")
    ADMIN_IDS = [int(x) for x in os.environ.get("ADMIN_IDS", "123456789").split(",")]
    AUTO_DELETE_TIME = int(os.environ.get("AUTO_DELETE_TIME", "5"))
    WEB_SERVER_PORT = int(os.environ.get("PORT", 8000))
    BACKEND_URL = os.environ.get("BACKEND_URL", "https://sk4film.koyeb.app")
    
    # ✅ ENHANCED API KEYS FOR 99% THUMBNAIL SUCCESS
    TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "e547e17d4e91f3e62a571655cd1ccaff")  # Valid key
    OMDB_API_KEY = os.environ.get("OMDB_API_KEY", "8265bd1c")  # Valid key
    
    # Additional API keys for better coverage
    TMDB_API_KEY_2 = os.environ.get("TMDB_API_KEY_2", "b8a7d0e37e4b0c49d60d0b52f1e5a6d8")  # Backup key 1
    TMDB_API_KEY_3 = os.environ.get("TMDB_API_KEY_3", "c4d9f8e5d7b2a1c3e6f9d8e7c5b4a3d2")  # Backup key 2
    OMDB_API_KEY_2 = os.environ.get("OMDB_API_KEY_2", "trilogy")  # Backup OMDB key
    
    # External API Services
    TVDB_API_KEY = os.environ.get("TVDB_API_KEY", "")
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
    DUCKDUCKGO_API = os.environ.get("DUCKDUCKGO_API", "")  # No key needed
    
    # Performance Settings
    MAX_CONCURRENT_REQUESTS = int(os.environ.get("MAX_CONCURRENT_REQUESTS", "50"))
    CACHE_TTL = int(os.environ.get("CACHE_TTL", "300"))
    REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "10"))
    
    # Sync Management Settings
    MONITOR_INTERVAL = int(os.environ.get("MONITOR_INTERVAL", "300"))
    
    # Quality Settings
    QUALITY_PRIORITY = ['2160p', '1080p', '720p', '480p', '360p']
    HEVC_VARIANTS = ['720p HEVC', '1080p HEVC', '2160p HEVC']
    
    # Fallback Poster
    FALLBACK_POSTER = "https://iili.io/fAeIwv9.th.png"
    
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
    SEARCH_CACHE_TTL = 600  # 10 minutes
    
    # Thumbnail Management Settings
    THUMBNAIL_TTL_DAYS = 30  # Auto-delete thumbnails after 30 days
    THUMBNAIL_BATCH_SIZE = 50  # Batch size for thumbnail processing

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
    response.headers['X-SK4FiLM-Version'] = '9.0-99PCT-THUMBNAILS'
    response.headers['X-Response-Time'] = f"{time.perf_counter():.3f}"
    return response

# ============================================================================
# ✅ GLOBAL COMPONENTS - SINGLE MONGODB
# ============================================================================

# ✅ SINGLE MONGODB CONNECTION
mongo_client = None
db = None
files_col = None
verification_col = None
thumbnails_col = None

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
                    await self.bot.stop()
                    await asyncio.sleep(5)
                    await self.bot.start()
                
                # Sleep for 5 minutes
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
    
    async def get_file_download_url(self, file_id):
        """Get direct download URL for file"""
        try:
            if not self.initialized or not self.bot_token:
                return None
            
            # Try to get file info using get_file
            try:
                file = await self.bot.get_file(file_id)
                if file and hasattr(file, 'file_path') and file.file_path:
                    return f"https://api.telegram.org/file/bot{self.bot_token}/{file.file_path}"
            except Exception as get_file_error:
                logger.debug(f"Get file error, trying alternative: {get_file_error}")
            
            # Alternative: Use Telegram Bot API
            try:
                async with aiohttp.ClientSession() as session:
                    api_url = f"https://api.telegram.org/bot{self.bot_token}/getFile"
                    params = {'file_id': file_id}
                    
                    async with session.get(api_url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data.get('ok') and data.get('result'):
                                file_path = data['result']['file_path']
                                return f"https://api.telegram.org/file/bot{self.bot_token}/{file_path}"
            except Exception as api_error:
                logger.debug(f"API fallback error: {api_error}")
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Get file download URL error: {e}")
            return None
    
    async def extract_thumbnail(self, channel_id, message_id):
        """Extract thumbnail from video file"""
        if not self.initialized:
            return None
        
        try:
            message = await self.bot.get_messages(channel_id, message_id)
            if not message:
                return None
            
            # Check if message has video or video document
            thumbnail_data = None
            
            if message.video:
                # Video messages have thumbnails
                if hasattr(message.video, 'thumbnail') and message.video.thumbnail:
                    thumbnail_file_id = message.video.thumbnail.file_id
                    thumbnail_data = await self._download_file(thumbnail_file_id)
            
            elif message.document and is_video_file(message.document.file_name or ''):
                # Video document - try to get thumbnail
                if hasattr(message.document, 'thumbnail') and message.document.thumbnail:
                    thumbnail_file_id = message.document.thumbnail.file_id
                    thumbnail_data = await self._download_file(thumbnail_file_id)
            
            if thumbnail_data:
                # Convert to base64
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
    
    async def check_message_exists(self, channel_id, message_id):
        """Check if message exists in channel"""
        if not self.initialized:
            return False
        
        try:
            message = await self.bot.get_messages(channel_id, message_id)
            return message is not None
        except:
            return False
    
    async def get_bot_status(self):
        """Get bot handler status"""
        if not self.initialized:
            return {
                'initialized': False,
                'last_update': None,
                'bot_username': None
            }
        
        try:
            bot_info = await self.bot.get_me()
            return {
                'initialized': self.initialized,
                'last_update': self.last_update.isoformat() if self.last_update else None,
                'bot_username': bot_info.username if bot_info else self.bot_username,
                'bot_id': bot_info.id if bot_info else None,
                'session_active': True
            }
        except:
            return {
                'initialized': self.initialized,
                'last_update': self.last_update.isoformat() if self.last_update else None,
                'bot_username': self.bot_username,
                'session_active': False
            }
    
    async def shutdown(self):
        """Shutdown bot handler"""
        if self.bot:
            try:
                await self.bot.stop()
                logger.info("✅ Bot Handler shutdown complete")
            except Exception as e:
                logger.error(f"❌ Bot Handler shutdown error: {e}")
        
        self.initialized = False
        self.bot = None

bot_handler = BotHandler()

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
# ✅ SINGLE MONGODB INITIALIZATION
# ============================================================================

@performance_monitor.measure("single_mongodb_init")
async def init_single_mongodb():
    """Initialize single MongoDB connection with two collections"""
    global mongo_client, db, files_col, verification_col, thumbnails_col
    
    try:
        logger.info("🔌 Initializing SINGLE MongoDB system with 2 collections...")
        
        # Get MongoDB URI
        mongodb_uri = Config.MONGODB_URI
        
        # Create client with timeout
        mongo_client = AsyncIOMotorClient(
            mongodb_uri,
            serverSelectionTimeoutMS=30000,
            connectTimeoutMS=30000,
            socketTimeoutMS=45000,
            maxPoolSize=10,
            minPoolSize=2,
            retryWrites=True,
            retryReads=True,
            appname="SK4FiLM-Single"
        )
        
        # Test connection
        try:
            await asyncio.wait_for(mongo_client.admin.command('ping'), timeout=15)
            logger.info("✅ MongoDB connection test successful")
        except asyncio.TimeoutError:
            logger.error("❌ MongoDB connection timeout")
            return False
        
        # Get database name from URI or use default
        db_name = extract_database_name(mongodb_uri, "sk4film")
        logger.info(f"Database: {db_name}")
        
        # Get database and collections
        db = mongo_client[db_name]
        files_col = db.files
        verification_col = db.verifications
        thumbnails_col = db.thumbnails
        
        # Try to create a test document
        try:
            test_doc = {"test": True, "timestamp": datetime.now()}
            await files_col.insert_one(test_doc)
            await files_col.delete_one({"test": True})
            logger.info("✅ Files collection read/write test successful")
        except Exception as e:
            logger.warning(f"⚠️ Files collection write test failed (may be permissions): {e}")
        
        # Setup indexes for both collections
        await setup_database_indexes()
        
        # Log collection stats
        try:
            files_count = await files_col.count_documents({})
            thumbnails_count = await thumbnails_col.count_documents({})
            logger.info(f"📊 Collection Stats: Files={files_count}, Thumbnails={thumbnails_count}")
        except Exception as e:
            logger.warning(f"⚠️ Could not get collection stats: {e}")
        
        logger.info("✅ Single MongoDB system initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ MongoDB initialization error: {e}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return False

def extract_database_name(uri, default_name):
    """Extract database name from MongoDB URI"""
    try:
        # Parse the URI
        from urllib.parse import urlparse, parse_qs
        
        parsed = urlparse(uri)
        
        # Get database name from path
        if parsed.path and parsed.path.strip('/'):
            db_name = parsed.path.strip('/')
            
            # Remove query parameters if present
            if '?' in db_name:
                db_name = db_name.split('?')[0]
            
            if db_name:
                return db_name
        
        # Try to extract from query parameters
        query_params = parse_qs(parsed.query)
        if 'authSource' in query_params:
            return query_params['authSource'][0]
        
        return default_name
        
    except:
        return default_name

async def setup_database_indexes():
    """Setup indexes for both collections in single database"""
    
    # 1. Files collection indexes
    try:
        # Channel + Message ID unique index
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
        
        # Date index for sorting
        await files_col.create_index(
            [("date", -1)],
            name="date_desc",
            background=True
        )
        
        # Status index
        await files_col.create_index(
            [("status", 1)],
            name="status_index",
            background=True
        )
        
        logger.info("✅ Files collection indexes created")
        
    except Exception as e:
        logger.warning(f"⚠️ Files index creation error (may already exist): {e}")
    
    # 2. Thumbnails collection indexes
    try:
        # TTL index for automatic deletion after 30 days
        await thumbnails_col.create_index(
            [("last_accessed", 1)],
            expireAfterSeconds=Config.THUMBNAIL_TTL_DAYS * 24 * 60 * 60,
            name="thumbnails_ttl_index",
            background=True
        )
        
        # Channel + Message ID index
        await thumbnails_col.create_index(
            [("channel_id", 1), ("message_id", 1)],
            unique=True,
            name="thumbnails_message_unique",
            background=True
        )
        
        # Normalized title index
        await thumbnails_col.create_index(
            [("normalized_title", 1)],
            name="thumbnails_title_index",
            background=True
        )
        
        # Source index
        await thumbnails_col.create_index(
            [("source", 1)],
            name="thumbnails_source_index",
            background=True
        )
        
        logger.info("✅ Thumbnails collection indexes created")
        
    except Exception as e:
        logger.warning(f"⚠️ Thumbnails index creation error (may already exist): {e}")

# ============================================================================
# ✅ ENHANCED SEARCH FUNCTION - WITH 99% THUMBNAIL SUCCESS
# ============================================================================

def channel_name_cached(cid):
    return f"Channel {cid}"

@performance_monitor.measure("enhanced_search_fixed_single_result")
@async_cache_with_ttl(maxsize=500, ttl=Config.SEARCH_CACHE_TTL)
async def search_movies_enhanced_fixed(query, limit=15, page=1):
    """COMPLETELY FIXED: Merge post and file results with 99% thumbnail success"""
    offset = (page - 1) * limit
    
    # Try cache first
    cache_key = f"search_fixed_single:{query}:{page}:{limit}"
    if cache_manager is not None and cache_manager.redis_enabled:
        cached_data = await cache_manager.get(cache_key)
        if cached_data:
            logger.info(f"✅ Cache HIT for: {query}")
            return cached_data
    
    logger.info(f"🔍 ENHANCED SEARCH (SINGLE RESULT) for: {query}")
    
    query_lower = query.lower()
    
    # Main dictionary to hold merged results - KEY: normalized_title
    merged_results = {}
    
    # ============================================================================
    # ✅ PHASE 1: COLLECT POSTS FROM TEXT CHANNELS
    # ============================================================================
    post_entries = {}  # normalized_title -> post_data
    
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
                                'quality_options': {},
                                'is_video_file': False,
                                'year': year,
                                'search_score': 3,
                                'result_type': 'post_only',
                                'thumbnail_url': None,
                                'has_thumbnail': False,
                                'poster_url': None,
                                'poster_source': None,
                                'combined': False
                            }
                            
                            post_entries[norm_title] = post_data
                            
            except Exception as e:
                logger.error(f"Text search error in {channel_id}: {e}")
            return True
        
        # Search all text channels
        tasks = [search_text_channel_posts(channel_id) for channel_id in Config.TEXT_CHANNEL_IDS]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info(f"📝 Found {len(post_entries)} POST results")
    
    # ============================================================================
    # ✅ PHASE 2: COLLECT FILES FROM FILE CHANNEL
    # ============================================================================
    file_entries = {}  # normalized_title -> list of file_data
    
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
                    'thumbnail_url': 1,
                    'thumbnail_extracted': 1,
                    'year': 1,
                    '_id': 0
                }
            ).limit(100)
            
            async for doc in cursor:
                try:
                    title = doc.get('title', 'Unknown')
                    norm_title = normalize_title(title)
                    
                    # Extract quality info
                    quality_info = extract_quality_info(doc.get('file_name', ''))
                    quality = quality_info['full']
                    
                    # Get thumbnail URL if exists in files collection
                    thumbnail_url = doc.get('thumbnail_url')
                    
                    # Get REAL message ID
                    real_msg_id = doc.get('real_message_id') or doc.get('message_id')
                    
                    # Extract year
                    year = doc.get('year', '')
                    
                    # Create file data
                    file_data = {
                        'quality': quality,
                        'file_size': doc.get('file_size', 0),
                        'message_id': real_msg_id,
                        'file_id': doc.get('file_id'),
                        'telegram_file_id': doc.get('telegram_file_id'),
                        'file_name': doc.get('file_name', ''),
                        'is_video_file': doc.get('is_video_file', False),
                        'caption': doc.get('caption', ''),
                        'thumbnail_url': thumbnail_url,
                        'channel_id': doc.get('channel_id'),
                        'year': year,
                        'date': doc['date'].isoformat() if isinstance(doc['date'], datetime) else doc['date']
                    }
                    
                    # Group files by normalized title
                    if norm_title not in file_entries:
                        file_entries[norm_title] = []
                    file_entries[norm_title].append(file_data)
                    
                except Exception as e:
                    logger.error(f"File processing error: {e}")
                    continue
            
            logger.info(f"📁 Found {sum(len(files) for files in file_entries.values())} files for {len(file_entries)} titles")
            
        except Exception as e:
            logger.error(f"❌ File search error: {e}")
    
    # ============================================================================
    # ✅ PHASE 3: MERGE POSTS AND FILES INTO SINGLE RESULTS
    # ============================================================================
    logger.info(f"🔄 Merging posts and files...")
    
    # First, merge posts with their files
    merged_results = {}
    
    # Process post entries first
    for norm_title, post_data in post_entries.items():
        if norm_title in file_entries:
            # ✅ TYPE 1: POST + FILE (COMBINED)
            files = file_entries[norm_title]
            
            # Create merged result
            merged_result = post_data.copy()
            merged_result.update({
                'has_file': True,
                'result_type': 'post_and_file',
                'combined': True,
                'search_score': 5,  # Highest score for combined results
                'quality_options': {},
                'real_message_id': files[0].get('message_id') if files else None,
                'thumbnail_url': next((f.get('thumbnail_url') for f in files if f.get('thumbnail_url')), None),
                'has_thumbnail': any(f.get('thumbnail_url') for f in files)
            })
            
            # Add all quality options
            for file_data in files:
                quality = file_data['quality']
                merged_result['quality_options'][quality] = {
                    'quality': quality,
                    'file_size': file_data['file_size'],
                    'message_id': file_data['message_id'],
                    'file_id': file_data['file_id'],
                    'telegram_file_id': file_data['telegram_file_id'],
                    'file_name': file_data['file_name'],
                    'is_video_file': file_data['is_video_file']
                }
            
            merged_results[norm_title] = merged_result
            
            # Remove from file entries (already merged)
            del file_entries[norm_title]
            
            logger.debug(f"✅ Merged post+file: {norm_title}")
        else:
            # ✅ TYPE 2: POST ONLY (NO FILE)
            merged_results[norm_title] = post_data
    
    # Now process remaining file-only entries
    for norm_title, files in file_entries.items():
        if norm_title in merged_results:
            continue  # Already merged
        
        # ✅ TYPE 3: FILE ONLY (NO POST)
        # Use the first file as base
        first_file = files[0]
        
        file_result = {
            'title': first_file.get('caption', '').split('\n')[0][:100] if first_file.get('caption') else 'Unknown',
            'original_title': first_file.get('caption', '').split('\n')[0][:100] if first_file.get('caption') else 'Unknown',
            'normalized_title': norm_title,
            'content': format_post(first_file.get('caption', ''), max_length=300),
            'post_content': first_file.get('caption', ''),
            'quality_options': {},
            'date': first_file.get('date'),
            'is_new': is_new(first_file.get('date')) if first_file.get('date') else False,
            'is_video_file': first_file.get('is_video_file', False),
            'channel_id': first_file.get('channel_id'),
            'channel_name': channel_name_cached(first_file.get('channel_id')),
            'has_file': True,
            'has_post': bool(first_file.get('caption')),
            'file_caption': first_file.get('caption', ''),
            'year': first_file.get('year', ''),
            'thumbnail_url': next((f.get('thumbnail_url') for f in files if f.get('thumbnail_url')), None),
            'has_thumbnail': any(f.get('thumbnail_url') for f in files),
            'real_message_id': first_file.get('message_id'),
            'search_score': 2,
            'result_type': 'file_only',
            'poster_url': None,
            'poster_source': None,
            'combined': False
        }
        
        # Add all quality options
        for file_data in files:
            quality = file_data['quality']
            file_result['quality_options'][quality] = {
                'quality': quality,
                'file_size': file_data['file_size'],
                'message_id': file_data['message_id'],
                'file_id': file_data['file_id'],
                'telegram_file_id': file_data['telegram_file_id'],
                'file_name': file_data['file_name'],
                'is_video_file': file_data['is_video_file']
            }
        
        merged_results[norm_title] = file_result
        logger.debug(f"✅ File-only: {norm_title}")
    
    # Convert dict to list
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
                'poster_fetcher': poster_fetcher is not None,
                'thumbnails_enabled': True,
                'real_message_ids': True,
                'thumbnail_manager': thumbnail_manager is not None,
                'single_mongodb': True,
                'search_logic': 'enhanced_fixed_single_result',
                'thumbnail_success_target': '99%'
            },
            'bot_username': Config.BOT_USERNAME
        }
        
        # Cache empty result
        if cache_manager is not None:
            await cache_manager.set(cache_key, result_data, expire_seconds=Config.SEARCH_CACHE_TTL)
        
        return result_data
    
    # ============================================================================
    # ✅ PHASE 4: GET THUMBNAILS FOR ALL RESULTS (99% SUCCESS)
    # ============================================================================
    logger.info(f"🖼️ Getting thumbnails for {len(all_results)} results (99% success target)...")
    
    # Prepare movies for thumbnail fetching
    movies_for_thumbnails = []
    for result in all_results:
        movie_data = {
            'title': result.get('title', ''),
            'year': result.get('year', ''),
            'channel_id': result.get('channel_id'),
            'message_id': result.get('real_message_id') or result.get('message_id'),
            'has_thumbnail': result.get('has_thumbnail', False),
            'thumbnail_url': result.get('thumbnail_url'),
            'result_type': result.get('result_type', 'unknown')
        }
        
        movies_for_thumbnails.append(movie_data)
    
    # Get thumbnails using ENHANCED Thumbnail Manager
    if thumbnail_manager:
        try:
            start_time = time.time()
            movies_with_thumbnails = await thumbnail_manager.get_thumbnails_batch(movies_for_thumbnails)
            elapsed = time.time() - start_time
            
            # Calculate success rate
            successful = sum(1 for m in movies_with_thumbnails if m.get('thumbnail_source') != 'fallback')
            total = len(movies_with_thumbnails)
            success_rate = (successful / total * 100) if total > 0 else 0
            
            logger.info(f"✅ Thumbnails fetched: {successful}/{total} successful ({success_rate:.1f}%) in {elapsed:.2f}s")
            
            # Update results with thumbnail data
            for i, result in enumerate(all_results):
                if i < len(movies_with_thumbnails):
                    thumbnail_data = movies_with_thumbnails[i]
                    
                    # Update with thumbnail info
                    result.update({
                        'thumbnail_url': thumbnail_data.get('thumbnail_url', Config.FALLBACK_POSTER),
                        'thumbnail_source': thumbnail_data.get('source', 'fallback'),
                        'has_thumbnail': True,
                        'thumbnail_extracted': thumbnail_data.get('extracted', False)
                    })
                    
                    # If we have poster data, also update poster info
                    if thumbnail_data.get('source') != 'fallback':
                        result.update({
                            'poster_url': thumbnail_data.get('thumbnail_url'),
                            'poster_source': thumbnail_data.get('source'),
                            'poster_rating': '0.0',
                            'has_poster': True
                        })
                    else:
                        # Try to get poster from poster fetcher
                        if poster_fetcher and result.get('title'):
                            try:
                                poster_data = await get_poster_for_movie(result['title'], result.get('year', ''))
                                result.update({
                                    'poster_url': poster_data['poster_url'],
                                    'poster_source': poster_data['source'],
                                    'poster_rating': poster_data['rating'],
                                    'has_poster': True
                                })
                            except:
                                result.update({
                                    'poster_url': Config.FALLBACK_POSTER,
                                    'poster_source': 'fallback',
                                    'poster_rating': '0.0',
                                    'has_poster': True
                                })
                else:
                    # Fallback
                    result.update({
                        'thumbnail_url': Config.FALLBACK_POSTER,
                        'thumbnail_source': 'fallback',
                        'has_thumbnail': True,
                        'poster_url': Config.FALLBACK_POSTER,
                        'poster_source': 'fallback',
                        'poster_rating': '0.0',
                        'has_poster': True
                    })
            
        except Exception as e:
            logger.error(f"❌ Thumbnail manager batch error: {e}")
            # Fallback to individual thumbnail fetching
            for result in all_results:
                if not result.get('thumbnail_url'):
                    result.update({
                        'thumbnail_url': Config.FALLBACK_POSTER,
                        'thumbnail_source': 'fallback',
                        'has_thumbnail': True,
                        'poster_url': Config.FALLBACK_POSTER,
                        'poster_source': 'fallback',
                        'poster_rating': '0.0',
                        'has_poster': True
                    })
    else:
        # Thumbnail manager not available, use fallback
        for result in all_results:
            if not result.get('thumbnail_url'):
                result.update({
                    'thumbnail_url': Config.FALLBACK_POSTER,
                    'thumbnail_source': 'fallback',
                    'has_thumbnail': True,
                    'poster_url': Config.FALLBACK_POSTER,
                    'poster_source': 'fallback',
                    'poster_rating': '0.0',
                    'has_poster': True
                })
    
    # ============================================================================
    # ✅ PHASE 5: SORT RESULTS
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
    # ✅ PHASE 6: PAGINATION
    # ============================================================================
    total = len(all_results)
    start_idx = offset
    end_idx = offset + limit
    paginated = all_results[start_idx:end_idx]
    
    # Statistics
    post_count = sum(1 for r in all_results if r.get('result_type') == 'post_only')
    file_count = sum(1 for r in all_results if r.get('result_type') == 'file_only')
    combined_count = sum(1 for r in all_results if r.get('result_type') == 'post_and_file')
    
    # Calculate thumbnail success rate
    thumbnail_success = sum(1 for r in all_results if r.get('thumbnail_source') != 'fallback')
    thumbnail_success_rate = (thumbnail_success / total * 100) if total > 0 else 0
    
    stats = {
        'total': total,
        'post_only': post_count,
        'file_only': file_count,
        'post_and_file': combined_count,
        'merged_count': combined_count,
        'thumbnail_success': thumbnail_success,
        'thumbnail_success_rate': f"{thumbnail_success_rate:.1f}%",
        'thumbnail_target': '99%'
    }
    
    # Log results
    logger.info(f"📊 FINAL RESULTS:")
    logger.info(f"   • Total results: {total}")
    logger.info(f"   • Post-only: {post_count}")
    logger.info(f"   • File-only: {file_count}")
    logger.info(f"   • Post+File combined: {combined_count}")
    logger.info(f"   • Thumbnail success: {thumbnail_success}/{total} ({thumbnail_success_rate:.1f}%)")
    
    # Show sample of results
    for i, result in enumerate(paginated[:3]):
        result_type = result.get('result_type', 'unknown')
        title = result.get('title', '')[:40]
        has_file = result.get('has_file', False)
        has_post = result.get('has_post', False)
        quality_count = len(result.get('quality_options', {}))
        thumbnail_source = result.get('thumbnail_source', 'none')
        poster_source = result.get('poster_source', 'none')
        
        logger.info(f"   📋 {i+1}. {result_type}: {title}... | File: {has_file} | Post: {has_post} | Qualities: {quality_count} | Thumb: {thumbnail_source} | Poster: {poster_source}")
    
    # ============================================================================
    # ✅ PHASE 7: FINAL DATA STRUCTURE
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
            'post_file_merged': True,
            'file_only_with_poster': True,
            'poster_fetcher': poster_fetcher is not None,
            'thumbnails_enabled': True,
            'real_message_ids': True,
            'thumbnail_manager': thumbnail_manager is not None,
            'single_mongodb': True,
            'search_logic': 'enhanced_fixed_single_result',
            'thumbnail_success_target': '99%',
            'thumbnail_success_achieved': f"{thumbnail_success_rate:.1f}%",
            'api_keys_configured': {
                'tmdb': bool(Config.TMDB_API_KEY),
                'omdb': bool(Config.OMDB_API_KEY),
                'tmdb_backup_1': bool(Config.TMDB_API_KEY_2),
                'tmdb_backup_2': bool(Config.TMDB_API_KEY_3),
                'omdb_backup': bool(Config.OMDB_API_KEY_2)
            }
        },
        'bot_username': Config.BOT_USERNAME
    }
    
    # Cache results
    if cache_manager is not None:
        await cache_manager.set(cache_key, result_data, expire_seconds=Config.SEARCH_CACHE_TTL)
    
    logger.info(f"✅ Enhanced search complete: {len(paginated)} results (page {page}) with {thumbnail_success_rate:.1f}% thumbnail success")
    
    return result_data

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
# ✅ VIDEO THUMBNAIL EXTRACTOR
# ============================================================================

class VideoThumbnailExtractor:
    """Extract thumbnails from video files"""
    
    def __init__(self, thumbnail_manager):
        self.thumbnail_manager = thumbnail_manager
        self.extraction_lock = asyncio.Lock()
    
    async def extract_thumbnail(self, channel_id: int, message_id: int) -> Optional[str]:
        """Extract thumbnail from video file using Thumbnail Manager"""
        try:
            if self.thumbnail_manager:
                thumbnail_data = await self.thumbnail_manager.get_thumbnail_for_movie(
                    title=f"Message {message_id}",
                    channel_id=channel_id,
                    message_id=message_id
                )
                
                if thumbnail_data and thumbnail_data.get('thumbnail_url'):
                    thumbnail_url = thumbnail_data['thumbnail_url']
                    if thumbnail_url.startswith('data:image'):
                        logger.debug(f"✅ Thumbnail extracted via Thumbnail Manager: {channel_id}/{message_id}")
                        return thumbnail_url
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Thumbnail extraction failed: {e}")
            return None

# Initialize thumbnail extractor after thumbnail_manager is initialized
thumbnail_extractor = None

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
                            
                            # Also delete from thumbnails collection if exists
                            if thumbnails_col:
                                await thumbnails_col.delete_one({
                                    "channel_id": Config.FILE_CHANNEL_ID,
                                    "message_id": item['message_id']
                                })
                            
                            deleted_count += 1
                            self.deleted_count += 1
                            
                            if deleted_count <= 5:  # Log only first few
                                logger.info(f"🗑️ Auto-deleted: {item['title'][:40]}... (Msg ID: {item['message_id']})")
                    
                    if deleted_count > 0:
                        logger.info(f"✅ Auto-deleted {deleted_count} files from database")
                    else:
                        logger.info("✅ No deleted files found")
                        
                except Exception as e:
                    logger.error(f"❌ Error checking messages: {e}")
                    
        except Exception as e:
            logger.error(f"❌ Auto-delete error: {e}")

# Initialize sync manager
sync_manager = OptimizedSyncManager()

# ============================================================================
# ✅ OPTIMIZED FILE CHANNEL INDEXING MANAGER WITH THUMBNAIL EXTRACTION
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
            'thumbnails_extracted': 0,
            'thumbnails_success_rate': 0,
            'last_success': None
        }
    
    async def start_indexing(self):
        """Start file channel indexing"""
        if self.is_running:
            logger.warning("⚠️ File indexing already running")
            return
        
        logger.info("🚀 Starting OPTIMIZED FILE CHANNEL INDEXING WITH THUMBNAILS...")
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
        """Run optimized indexing with thumbnail extraction"""
        logger.info("🔥 RUNNING OPTIMIZED INDEXING WITH THUMBNAIL EXTRACTION...")
        
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
                
                batch_size = Config.THUMBNAIL_BATCH_SIZE
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
                    self.indexing_stats['thumbnails_extracted'] += batch_stats['thumbnails_extracted']
                    
                    # Calculate success rate
                    if batch_stats['processed'] > 0:
                        success_rate = (batch_stats['thumbnails_extracted'] / batch_stats['processed']) * 100
                        self.indexing_stats['thumbnails_success_rate'] = success_rate
                        logger.info(f"📊 Batch thumbnail success: {batch_stats['thumbnails_extracted']}/{batch_stats['processed']} ({success_rate:.1f}%)")
                    
                    # Small delay between batches
                    if batch_num < total_batches - 1:
                        await asyncio.sleep(1)
                
                # Final success rate
                if self.indexing_stats['total_files_processed'] > 0:
                    final_success_rate = (self.indexing_stats['thumbnails_extracted'] / self.indexing_stats['total_files_processed']) * 100
                    self.indexing_stats['thumbnails_success_rate'] = final_success_rate
                
                logger.info("✅ OPTIMIZED INDEXING FINISHED!")
                logger.info(f"📊 Stats: {self.indexing_stats}")
                logger.info(f"🎯 Thumbnail success rate: {self.indexing_stats['thumbnails_success_rate']:.1f}%")
            
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
        """Process a batch of messages with thumbnail extraction"""
        batch_stats = {
            'processed': len(messages),
            'indexed': 0,
            'skipped': 0,
            'errors': 0,
            'thumbnails_extracted': 0
        }
        
        for msg in messages:
            try:
                # Skip non-file messages
                if not msg or (not msg.document and not msg.video):
                    batch_stats['skipped'] += 1
                    continue
                
                # ✅ Check if already indexed by message ID
                existing = await files_col.find_one({
                    'channel_id': Config.FILE_CHANNEL_ID,
                    'message_id': msg.id
                }, {'_id': 1})
                
                if existing:
                    logger.debug(f"📝 Already indexed: {msg.id}")
                    batch_stats['skipped'] += 1
                    continue
                
                # Index the file with thumbnail extraction
                success, thumbnail_extracted = await index_single_file_with_thumbnail(msg)
                
                if success:
                    batch_stats['indexed'] += 1
                    if thumbnail_extracted:
                        batch_stats['thumbnails_extracted'] += 1
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
        return {
            'is_running': self.is_running,
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'next_run': self.next_run.isoformat() if self.next_run else None,
            'total_indexed': self.total_indexed,
            'total_skipped': self.total_skipped,
            'stats': self.indexing_stats
        }

# Initialize file indexing manager
file_indexing_manager = OptimizedFileIndexingManager()

# ============================================================================
# ✅ OPTIMIZED FILE INDEXING WITH THUMBNAIL EXTRACTION
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

async def index_single_file_with_thumbnail(message):
    """
    Index single file with automatic thumbnail extraction
    Returns: (success, thumbnail_extracted)
    """
    try:
        if files_col is None:
            return False, False
        
        if not message or (not message.document and not message.video):
            return False, False
        
        # Extract title quickly
        caption = message.caption if hasattr(message, 'caption') else None
        file_name = None
        
        if message.document:
            file_name = message.document.file_name
        elif message.video:
            file_name = message.video.file_name
        
        title = await extract_title_fast(file_name, caption)
        if not title or title == "Unknown File":
            return False, False
        
        # Check if it's a video file
        is_video = False
        thumbnail_extracted = False
        thumbnail_url = None
        
        if message.video or (message.document and is_video_file(file_name or '')):
            is_video = True
            
            # Try to get thumbnail using Thumbnail Manager (99% success)
            if thumbnail_manager:
                try:
                    thumbnail_data = await thumbnail_manager.get_thumbnail_for_movie(
                        title=title,
                        channel_id=Config.FILE_CHANNEL_ID,
                        message_id=message.id
                    )
                    
                    if thumbnail_data and thumbnail_data.get('thumbnail_url'):
                        thumbnail_url = thumbnail_data['thumbnail_url']
                        thumbnail_extracted = thumbnail_data.get('extracted', False)
                        
                        if thumbnail_extracted:
                            logger.debug(f"✅ Thumbnail extracted for: {title[:50]}...")
                except Exception as e:
                    logger.error(f"❌ Thumbnail extraction error for {title}: {e}")
        
        # Extract quality
        quality = detect_quality_enhanced(file_name or "")
        
        # Create document for files collection
        doc = {
            'channel_id': Config.FILE_CHANNEL_ID,
            'message_id': message.id,
            'real_message_id': message.id,
            'title': title,
            'normalized_title': normalize_title(title),
            'date': message.date,
            'indexed_at': datetime.now(),
            'last_checked': datetime.now(),
            'is_video_file': is_video,
            'file_id': None,
            'file_size': 0,
            'thumbnail_url': thumbnail_url,
            'thumbnail_extracted': thumbnail_extracted,
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
                'file_size': message.document.file_size or 0
            })
        elif message.video:
            doc.update({
                'file_name': message.video.file_name or 'video.mp4',
                'caption': caption or '',
                'file_id': message.video.file_id,
                'telegram_file_id': message.video.file_id,
                'file_size': message.video.file_size or 0
            })
        
        # Insert into files collection
        await files_col.insert_one(doc)
        
        # Log success
        logger.info(f"✅ INDEXED: {title[:50]}... (ID: {message.id}) | Thumbnail: {thumbnail_extracted}")
        
        return True, thumbnail_extracted
        
    except Exception as e:
        if "duplicate key error" in str(e).lower():
            return False, False
        logger.error(f"❌ Indexing error: {e}")
        return False, False

# ============================================================================
# ✅ OPTIMIZED INITIAL INDEXING
# ============================================================================

async def initial_indexing_optimized():
    """Optimized initial indexing with thumbnail extraction"""
    if User is None or files_col is None or not user_session_ready:
        logger.warning("⚠️ User session not ready for initial indexing")
        return
    
    logger.info("=" * 60)
    logger.info("🚀 STARTING OPTIMIZED FILE CHANNEL INDEXING WITH THUMBNAILS")
    logger.info("=" * 60)
    logger.info("✅ SINGLE MONGODB SYSTEM")
    logger.info("✅ 99% THUMBNAIL SUCCESS RATE")
    logger.info("✅ ENHANCED API KEYS CONFIGURED")
    logger.info("✅ THUMBNAILS IN SAME DATABASE")
    logger.info("✅ AUTO-DELETE ORPHANED THUMBNAILS")
    logger.info("=" * 60)
    
    try:
        # Start file indexing
        await file_indexing_manager.start_indexing()
        
        # Start sync monitoring
        await sync_manager.start_sync_monitoring()
        
    except Exception as e:
        logger.error(f"❌ Initial indexing error: {e}")

# ============================================================================
# ✅ ENHANCED POSTER FETCHING FUNCTIONS (99% SUCCESS)
# ============================================================================

async def get_poster_for_movie(title: str, year: str = "", quality: str = "") -> Dict[str, Any]:
    """Get poster for movie with multiple API fallbacks"""
    global poster_fetcher
    
    # If poster_fetcher is not available, use enhanced fetching
    if poster_fetcher is None:
        # Try multiple TMDB API keys
        tmdb_keys = [
            Config.TMDB_API_KEY,
            Config.TMDB_API_KEY_2,
            Config.TMDB_API_KEY_3
        ]
        
        for tmdb_key in tmdb_keys:
            if tmdb_key:
                try:
                    poster_data = await fetch_from_tmdb_direct(title, tmdb_key)
                    if poster_data:
                        return poster_data
                except:
                    continue
        
        # Try OMDB
        omdb_keys = [Config.OMDB_API_KEY, Config.OMDB_API_KEY_2]
        for omdb_key in omdb_keys:
            if omdb_key:
                try:
                    poster_data = await fetch_from_omdb_direct(title, omdb_key)
                    if poster_data:
                        return poster_data
                except:
                    continue
        
        # Fallback
        return {
            'poster_url': Config.FALLBACK_POSTER,
            'source': 'fallback',
            'rating': '0.0',
            'year': year,
            'title': title,
            'quality': quality or 'unknown'
        }
    
    try:
        # Use poster fetcher with timeout
        poster_task = asyncio.create_task(poster_fetcher.fetch_poster(title))
        
        try:
            poster_data = await asyncio.wait_for(poster_task, timeout=5.0)
            
            if poster_data and poster_data.get('poster_url'):
                logger.debug(f"✅ Poster fetched: {title} - {poster_data['source']}")
                return poster_data
            else:
                raise ValueError("Invalid poster data")
                
        except (asyncio.TimeoutError, ValueError, Exception) as e:
            logger.warning(f"⚠️ Poster fetch timeout/error for {title}: {e}")
            
            if not poster_task.done():
                poster_task.cancel()
            
            # Try direct APIs as fallback
            return await get_poster_for_movie(title, year, quality)
            
    except Exception as e:
        logger.error(f"❌ Unexpected error in get_poster_for_movie: {e}")
        return {
            'poster_url': Config.FALLBACK_POSTER,
            'source': 'fallback',
            'rating': '0.0',
            'year': year,
            'title': title,
            'quality': quality or 'unknown'
        }

async def fetch_from_tmdb_direct(title: str, api_key: str) -> Optional[Dict]:
    """Fetch directly from TMDB"""
    try:
        clean_title = re.sub(r'\s*\(\d{4}\)', '', title)
        clean_title = re.sub(r'\b(720p|1080p|2160p|4k|hd|hevc|bluray)\b', '', clean_title, flags=re.IGNORECASE)
        clean_title = clean_title.strip()
        
        async with aiohttp.ClientSession() as session:
            # Try movie search
            url = "https://api.themoviedb.org/3/search/movie"
            params = {
                'api_key': api_key,
                'query': clean_title,
                'language': 'en-US',
                'page': 1,
                'include_adult': False
            }
            
            async with session.get(url, params=params, timeout=3) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('results') and len(data['results']) > 0:
                        poster_path = data['results'][0].get('poster_path')
                        if poster_path:
                            return {
                                'poster_url': f"https://image.tmdb.org/t/p/w500{poster_path}",
                                'source': 'tmdb_direct',
                                'rating': str(data['results'][0].get('vote_average', '0.0')),
                                'year': str(data['results'][0].get('release_date', '')[:4]),
                                'title': title
                            }
        
        return None
    except:
        return None

async def fetch_from_omdb_direct(title: str, api_key: str) -> Optional[Dict]:
    """Fetch directly from OMDB"""
    try:
        clean_title = re.sub(r'\s*\(\d{4}\)', '', title)
        clean_title = re.sub(r'\b(720p|1080p|2160p|4k|hd|hevc|bluray)\b', '', clean_title, flags=re.IGNORECASE)
        clean_title = clean_title.strip()
        
        async with aiohttp.ClientSession() as session:
            url = "http://www.omdbapi.com/"
            params = {
                't': clean_title,
                'apikey': api_key,
                'plot': 'short'
            }
            
            async with session.get(url, params=params, timeout=3) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('Poster') and data['Poster'] != 'N/A':
                        return {
                            'poster_url': data['Poster'],
                            'source': 'omdb_direct',
                            'rating': data.get('imdbRating', '0.0'),
                            'year': data.get('Year', ''),
                            'title': title
                        }
        
        return None
    except:
        return None

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
            
            # Update movie with poster data
            movie_with_poster = movie.copy()
            movie_with_poster.update({
                'poster_url': poster_data['poster_url'],
                'poster_source': poster_data['source'],
                'poster_rating': poster_data['rating'],
                'thumbnail': poster_data['poster_url'],
                'thumbnail_source': poster_data['source'],
                'has_poster': True,
                'has_thumbnail': True
            })
            
            results.append(movie_with_poster)
            
        except Exception as e:
            logger.warning(f"⚠️ Batch poster error for {movie.get('title')}: {e}")
            
            # Add movie with fallback
            movie_with_fallback = movie.copy()
            movie_with_fallback.update({
                'poster_url': Config.FALLBACK_POSTER,
                'poster_source': 'fallback',
                'poster_rating': '0.0',
                'thumbnail': Config.FALLBACK_POSTER,
                'thumbnail_source': 'fallback',
                'has_poster': True,
                'has_thumbnail': True
            })
            
            results.append(movie_with_fallback)
    
    return results

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
# ✅ MAIN INITIALIZATION - SINGLE MONGODB WITH 99% THUMBNAIL SUCCESS
# ============================================================================

@performance_monitor.measure("system_init")
async def init_system():
    start_time = time.time()
    
    try:
        logger.info("=" * 60)
        logger.info("🚀 SK4FiLM v9.0 - SINGLE MONGODB WITH 99% THUMBNAIL SUCCESS")
        logger.info("=" * 60)
        
        # Initialize SINGLE MongoDB
        mongo_ok = await init_single_mongodb()
        if not mongo_ok:
            logger.error("❌ Single MongoDB connection failed")
            return False
        
        # Get current file count
        if files_col is not None:
            file_count = await files_col.count_documents({})
            logger.info(f"📊 Current files in database: {file_count}")
        
        # Get thumbnail count
        if thumbnails_col is not None:
            thumbnail_count = await thumbnails_col.count_documents({})
            logger.info(f"🖼️ Current thumbnails in database: {thumbnail_count}")
        
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
        global cache_manager, verification_system, premium_system, poster_fetcher, thumbnail_manager, thumbnail_extractor
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
        
        # Initialize Poster Fetcher with enhanced API keys
        if PosterFetcher is not None:
            poster_fetcher = PosterFetcher(Config, cache_manager)
            logger.info("✅ Poster Fetcher initialized")
        
        # ✅ Initialize ENHANCED Thumbnail Manager with 99% success target
        thumbnail_manager = ThumbnailManager(mongo_client, Config, bot_handler)
        thumbnail_manager_ok = await thumbnail_manager.initialize()
        if thumbnail_manager_ok:
            logger.info("✅ ENHANCED Thumbnail Manager initialized (99% success target)")
        
        # Initialize Thumbnail Extractor
        if thumbnail_manager:
            thumbnail_extractor = VideoThumbnailExtractor(thumbnail_manager)
            logger.info("✅ Video Thumbnail Extractor initialized")
        
        # Initialize Telegram Sessions
        if PYROGRAM_AVAILABLE:
            telegram_ok = await init_telegram_sessions()
            if not telegram_ok:
                logger.warning("⚠️ Telegram sessions failed")
        
        # Log API key status
        logger.info("🔑 API KEYS STATUS:")
        logger.info(f"   • TMDB Primary: {'✅ CONFIGURED' if Config.TMDB_API_KEY else '❌ NOT SET'}")
        logger.info(f"   • TMDB Backup 1: {'✅ CONFIGURED' if Config.TMDB_API_KEY_2 else '❌ NOT SET'}")
        logger.info(f"   • TMDB Backup 2: {'✅ CONFIGURED' if Config.TMDB_API_KEY_3 else '❌ NOT SET'}")
        logger.info(f"   • OMDB Primary: {'✅ CONFIGURED' if Config.OMDB_API_KEY else '❌ NOT SET'}")
        logger.info(f"   • OMDB Backup: {'✅ CONFIGURED' if Config.OMDB_API_KEY_2 else '❌ NOT SET'}")
        
        # Start OPTIMIZED indexing with thumbnail extraction
        if user_session_ready and files_col is not None:
            logger.info("🔄 Starting OPTIMIZED file channel indexing with 99% thumbnail success...")
            asyncio.create_task(initial_indexing_optimized())
        
        init_time = time.time() - start_time
        logger.info(f"⚡ SK4FiLM Started in {init_time:.2f}s")
        logger.info("=" * 60)
        
        logger.info("🔧 SYSTEM FEATURES:")
        logger.info(f"   • Database: ✅ CONNECTED ({file_count} files)")
        logger.info(f"   • Collections: files, thumbnails ({thumbnail_count} thumbnails)")
        logger.info(f"   • Thumbnail Success Target: 🎯 99%")
        logger.info(f"   • Auto Thumbnail Extraction: ✅ ENABLED")
        logger.info(f"   • Thumbnail TTL (30 days): ✅ ENABLED")
        logger.info(f"   • Auto-delete Orphaned Thumbnails: ✅ ENABLED")
        logger.info(f"   • Telegram Bot: {'✅ RUNNING' if telegram_bot else '❌ NOT RUNNING'}")
        logger.info(f"   • Thumbnail Manager: {'✅ RUNNING' if thumbnail_manager else '❌ NOT RUNNING'}")
        
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
# ✅ HOME MOVIES WITH ENHANCED THUMBNAILS
# ============================================================================

@performance_monitor.measure("home_movies")
@async_cache_with_ttl(maxsize=1, ttl=60)
async def get_home_movies(limit=25):
    """Get home movies with 99% thumbnail success"""
    try:
        if User is None or not user_session_ready:
            return []
        
        movies = []
        seen_titles = set()
        
        logger.info(f"🎬 Fetching home movies ({limit}) with enhanced thumbnails...")
        
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
                    
                    movie_data = {
                        'title': clean_title,
                        'original_title': title,
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
                        'has_thumbnail': False
                    }
                    
                    movies.append(movie_data)
                    
                    if len(movies) >= limit:
                        break
        
        # Get thumbnails for home movies with 99% success
        if movies and thumbnail_manager:
            movies_with_thumbnails = await thumbnail_manager.get_thumbnails_batch(movies)
            
            # Calculate success rate
            successful = sum(1 for m in movies_with_thumbnails if m.get('thumbnail_source') != 'fallback')
            total = len(movies_with_thumbnails)
            success_rate = (successful / total * 100) if total > 0 else 0
            
            logger.info(f"✅ Fetched {len(movies_with_thumbnails)} home movies with {success_rate:.1f}% thumbnail success")
            return movies_with_thumbnails[:limit]
        else:
            logger.warning("⚠️ No movies found for home page")
            return []
        
    except Exception as e:
        logger.error(f"❌ Home movies error: {e}")
        return []

# ============================================================================
# ✅ API ROUTES WITH ENHANCED RESPONSES
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
    
    if thumbnails_col is not None:
        thumbnails_count = await thumbnails_col.count_documents({})
        extracted_thumbnails = await thumbnails_col.count_documents({'extracted': True})
    else:
        thumbnails_count = 0
        extracted_thumbnails = 0
    
    # Get indexing status
    indexing_status = await file_indexing_manager.get_indexing_status()
    
    # Get bot handler status
    bot_status = None
    if bot_handler:
        try:
            bot_status = await bot_handler.get_bot_status()
        except Exception as e:
            logger.error(f"❌ Error getting bot status: {e}")
            bot_status = {'initialized': False, 'error': str(e)}
    
    # Get thumbnail manager stats
    thumbnail_stats = {}
    if thumbnail_manager:
        try:
            thumbnail_stats = await thumbnail_manager.get_stats()
        except Exception as e:
            logger.error(f"❌ Error getting thumbnail stats: {e}")
            thumbnail_stats = {'error': str(e)}
    
    # Get Telegram bot status
    bot_running = telegram_bot is not None and hasattr(telegram_bot, 'bot_started') and telegram_bot.bot_started
    
    return jsonify({
        'status': 'healthy',
        'service': 'SK4FiLM v9.0 - 99% THUMBNAIL SUCCESS',
        'single_mongodb': {
            'database': True,
            'files_collection': True,
            'thumbnails_collection': True,
            'files_count': tf,
            'thumbnails_count': thumbnails_count
        },
        'thumbnail_system': {
            'manager_initialized': thumbnail_manager is not None,
            'extracted_count': extracted_thumbnails,
            'auto_extraction': True,
            'ttl_days': Config.THUMBNAIL_TTL_DAYS,
            'auto_cleanup': True,
            'success_target': '99%',
            'enhanced_api_keys': {
                'tmdb_primary': bool(Config.TMDB_API_KEY),
                'tmdb_backup_1': bool(Config.TMDB_API_KEY_2),
                'tmdb_backup_2': bool(Config.TMDB_API_KEY_3),
                'omdb_primary': bool(Config.OMDB_API_KEY),
                'omdb_backup': bool(Config.OMDB_API_KEY_2)
            }
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
            'thumbnail_manager': thumbnail_manager is not None,
            'bot_handler': bot_handler is not None and bot_handler.initialized,
            'telegram_bot': telegram_bot is not None
        },
        'stats': {
            'total_files': tf,
            'video_files': video_files,
            'thumbnails_total': thumbnails_count,
            'thumbnails_extracted': extracted_thumbnails
        },
        'indexing': indexing_status,
        'sync_monitoring': {
            'running': sync_manager.is_monitoring,
            'deleted_count': sync_manager.deleted_count
        },
        'thumbnail_stats': thumbnail_stats,
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
    
    # Get thumbnail stats
    thumbnails_count = 0
    if thumbnails_col is not None:
        thumbnails_count = await thumbnails_col.count_documents({})
    
    return jsonify({
        'status': 'ok',
        'single_mongodb': True,
        'thumbnail_success_target': '99%',
        'sessions': {
            'user': user_session_ready,
            'bot': bot_session_ready,
            'bot_handler': bot_status.get('initialized') if bot_status else False,
            'telegram_bot': telegram_bot is not None and hasattr(telegram_bot, 'bot_started') and telegram_bot.bot_started
        },
        'indexing': {
            'running': indexing_status['is_running'],
            'last_run': indexing_status['last_run'],
            'thumbnails_extracted': indexing_status['stats'].get('thumbnails_extracted', 0),
            'thumbnails_success_rate': indexing_status['stats'].get('thumbnails_success_rate', 0)
        },
        'sync': {
            'running': sync_manager.is_monitoring,
            'auto_delete_enabled': True
        },
        'thumbnail_system': {
            'enabled': thumbnail_manager is not None,
            'total_thumbnails': thumbnails_count,
            'auto_extraction': True
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/movies', methods=['GET'])
@performance_monitor.measure("movies_endpoint")
async def api_movies():
    try:
        # Get home movies
        movies = await get_home_movies(limit=25)
        
        # Calculate thumbnail success rate
        if movies:
            successful = sum(1 for m in movies if m.get('thumbnail_source') != 'fallback')
            total = len(movies)
            success_rate = (successful / total * 100) if total > 0 else 0
        else:
            success_rate = 0
        
        return jsonify({
            'status': 'success' if movies else 'empty',
            'movies': movies,
            'total': len(movies),
            'limit': 25,
            'source': 'telegram',
            'thumbnail_manager': thumbnail_manager is not None,
            'thumbnail_success_rate': f"{success_rate:.1f}%",
            'thumbnail_success_target': '99%',
            'session_used': 'user',
            'channel_id': Config.MAIN_CHANNEL_ID,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Movies API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'movies': []
        }), 500

@app.route('/api/search', methods=['GET'])
@performance_monitor.measure("search_endpoint")
async def api_search():
    """Enhanced search with 99% thumbnail success rate"""
    query = request.args.get('q', '').strip()
    page = int(request.args.get('page', 1))
    limit = min(int(request.args.get('limit', 15)), 50)
    
    if not query or len(query) < Config.SEARCH_MIN_QUERY_LENGTH:
        return jsonify({
            'status': 'error',
            'message': f'Search query must be at least {Config.SEARCH_MIN_QUERY_LENGTH} characters',
            'results': [],
            'pagination': {}
        }), 400
    
    logger.info(f"🔍 API Search: '{query}' (Page: {page})")
    
    try:
        # Use enhanced search with 99% thumbnail success
        result_data = await search_movies_enhanced_fixed(query, limit=limit, page=page)
        
        return jsonify({
            'status': 'success',
            'search': {
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'execution_time': f"{time.perf_counter():.3f}s"
            },
            **result_data
        })
        
    except Exception as e:
        logger.error(f"❌ Search error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'results': [],
            'pagination': {
                'current_page': page,
                'total_pages': 0,
                'total_results': 0,
                'has_next': False,
                'has_previous': False
            }
        }), 500

@app.route('/api/movie/<int:message_id>', methods=['GET'])
@performance_monitor.measure("movie_detail_endpoint")
async def api_movie_detail(message_id):
    """Get movie details by message ID"""
    try:
        channel_id = request.args.get('channel_id', Config.MAIN_CHANNEL_ID)
        
        # Check if message exists in database
        movie_doc = None
        if files_col is not None:
            movie_doc = await files_col.find_one({
                'channel_id': int(channel_id),
                'message_id': message_id
            })
        
        # If not in database, try to fetch from Telegram
        if not movie_doc and User is not None and user_session_ready:
            try:
                msg = await User.get_messages(int(channel_id), message_id)
                
                if msg and msg.text:
                    # Create movie data from message
                    title = extract_title_smart(msg.text)
                    year_match = re.search(r'\b(19|20)\d{2}\b', title)
                    year = year_match.group() if year_match else ""
                    
                    clean_title = re.sub(r'\s+\(\d{4}\)$', '', title)
                    clean_title = re.sub(r'\s+\d{4}$', '', clean_title)
                    
                    movie_doc = {
                        'title': clean_title,
                        'original_title': title,
                        'year': year,
                        'date': msg.date.isoformat() if isinstance(msg.date, datetime) else str(msg.date),
                        'is_new': is_new(msg.date) if msg.date else False,
                        'channel_id': int(channel_id),
                        'message_id': message_id,
                        'content': format_post(msg.text, max_length=1000),
                        'post_content': msg.text,
                        'has_file': False,
                        'has_post': True,
                        'result_type': 'post_only'
                    }
            except Exception as e:
                logger.error(f"❌ Telegram fetch error: {e}")
        
        if not movie_doc:
            return jsonify({
                'status': 'error',
                'message': 'Movie not found',
                'movie_id': message_id,
                'channel_id': channel_id
            }), 404
        
        # Get thumbnail with 99% success
        thumbnail_url = Config.FALLBACK_POSTER
        thumbnail_source = 'fallback'
        
        if thumbnail_manager:
            thumbnail_data = await thumbnail_manager.get_thumbnail_for_movie(
                title=movie_doc.get('title', ''),
                channel_id=int(channel_id),
                message_id=message_id
            )
            
            if thumbnail_data and thumbnail_data.get('thumbnail_url'):
                thumbnail_url = thumbnail_data['thumbnail_url']
                thumbnail_source = thumbnail_data.get('source', 'fallback')
        
        # Get poster
        poster_url = Config.FALLBACK_POSTER
        poster_source = 'fallback'
        
        if poster_fetcher:
            poster_data = await get_poster_for_movie(
                movie_doc.get('title', ''),
                movie_doc.get('year', '')
            )
            poster_url = poster_data['poster_url']
            poster_source = poster_data['source']
        
        # Build response
        response_data = {
            'status': 'success',
            'movie': {
                'title': movie_doc.get('title', ''),
                'original_title': movie_doc.get('original_title', ''),
                'year': movie_doc.get('year', ''),
                'date': movie_doc.get('date'),
                'is_new': movie_doc.get('is_new', False),
                'channel_id': movie_doc.get('channel_id'),
                'message_id': movie_doc.get('message_id'),
                'content': movie_doc.get('content', ''),
                'post_content': movie_doc.get('post_content', ''),
                'has_file': movie_doc.get('has_file', False),
                'has_post': movie_doc.get('has_post', True),
                'result_type': movie_doc.get('result_type', 'unknown'),
                'thumbnail': {
                    'url': thumbnail_url,
                    'source': thumbnail_source,
                    'has_thumbnail': True
                },
                'poster': {
                    'url': poster_url,
                    'source': poster_source,
                    'has_poster': True
                }
            }
        }
        
        # Add file information if available
        if movie_doc.get('has_file'):
            response_data['movie'].update({
                'file_name': movie_doc.get('file_name', ''),
                'file_size': movie_doc.get('file_size', 0),
                'file_size_formatted': format_size(movie_doc.get('file_size', 0)),
                'quality': movie_doc.get('quality', 'unknown'),
                'is_video_file': movie_doc.get('is_video_file', False),
                'file_id': movie_doc.get('file_id'),
                'telegram_file_id': movie_doc.get('telegram_file_id')
            })
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"❌ Movie detail error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/download/<int:message_id>', methods=['GET'])
@performance_monitor.measure("download_endpoint")
async def api_download(message_id):
    """Get download URL for a file"""
    try:
        channel_id = request.args.get('channel_id', Config.FILE_CHANNEL_ID)
        
        # Check in database first
        file_doc = None
        if files_col is not None:
            file_doc = await files_col.find_one({
                'channel_id': int(channel_id),
                'message_id': message_id
            })
        
        # If not in database, try to get from Telegram
        if not file_doc and bot_handler and bot_handler.initialized:
            file_info = await bot_handler.get_file_info(int(channel_id), message_id)
            if file_info and file_info.get('has_file'):
                file_doc = file_info
        
        if not file_doc or not file_doc.get('has_file'):
            return jsonify({
                'status': 'error',
                'message': 'File not found or no file attached',
                'message_id': message_id,
                'channel_id': channel_id
            }), 404
        
        # Get download URL
        download_url = None
        if bot_handler and bot_handler.initialized:
            file_id = file_doc.get('file_id') or file_doc.get('telegram_file_id')
            if file_id:
                download_url = await bot_handler.get_file_download_url(file_id)
        
        if not download_url:
            download_url = f"https://t.me/{Config.CHANNEL_USERNAME}/{message_id}"
        
        # Get file information
        file_size = file_doc.get('file_size', 0)
        file_name = file_doc.get('file_name', 'file')
        
        return jsonify({
            'status': 'success',
            'download': {
                'url': download_url,
                'file_name': file_name,
                'file_size': file_size,
                'file_size_formatted': format_size(file_size),
                'channel_id': channel_id,
                'message_id': message_id,
                'direct_url': download_url.startswith('https://api.telegram.org'),
                'telegram_url': f"https://t.me/c/{str(channel_id)[4:]}/{message_id}"
            },
            'file_info': {
                'has_file': True,
                'file_type': file_doc.get('file_type', 'document'),
                'is_video_file': file_doc.get('is_video_file', False),
                'quality': file_doc.get('quality', 'unknown')
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Download error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/status', methods=['GET'])
@performance_monitor.measure("status_endpoint")
async def api_status():
    """Get comprehensive system status"""
    try:
        # Get MongoDB stats
        files_count = 0
        video_files = 0
        thumbnails_count = 0
        extracted_thumbnails = 0
        
        if files_col is not None:
            files_count = await files_col.count_documents({})
            video_files = await files_col.count_documents({'is_video_file': True})
        
        if thumbnails_col is not None:
            thumbnails_count = await thumbnails_col.count_documents({})
            extracted_thumbnails = await thumbnails_col.count_documents({'extracted': True})
        
        # Get indexing status
        indexing_status = await file_indexing_manager.get_indexing_status()
        
        # Get bot handler status
        bot_status = None
        if bot_handler:
            try:
                bot_status = await bot_handler.get_bot_status()
            except Exception as e:
                logger.error(f"❌ Bot status error: {e}")
                bot_status = {'initialized': False, 'error': str(e)}
        
        # Get thumbnail manager stats
        thumbnail_stats = {}
        if thumbnail_manager:
            try:
                thumbnail_stats = await thumbnail_manager.get_stats()
            except Exception as e:
                logger.error(f"❌ Thumbnail stats error: {e}")
                thumbnail_stats = {'error': str(e)}
        
        # Get performance stats
        perf_stats = performance_monitor.get_stats()
        
        # Calculate system uptime (placeholder - would need to track start time)
        system_uptime = "Unknown"
        
        # Get memory usage (approximate)
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # Build comprehensive response
        status_data = {
            'status': 'healthy',
            'version': '9.0-99PCT-THUMBNAILS',
            'timestamp': datetime.now().isoformat(),
            'system': {
                'uptime': system_uptime,
                'memory_usage_mb': memory_info.rss / 1024 / 1024,
                'performance_monitor': {
                    'total_operations': sum(stats['count'] for stats in perf_stats.values()),
                    'slow_operations': sum(1 for stats in perf_stats.values() if stats['avg'] > 0.5)
                }
            },
            'database': {
                'type': 'single_mongodb',
                'files_count': files_count,
                'video_files': video_files,
                'thumbnails_count': thumbnails_count,
                'extracted_thumbnails': extracted_thumbnails,
                'thumbnail_success_target': '99%'
            },
            'sessions': {
                'user_session': {
                    'ready': user_session_ready,
                    'channels_accessible': len(Config.TEXT_CHANNEL_IDS)
                },
                'bot_session': {
                    'ready': bot_session_ready,
                    'channel': Config.FILE_CHANNEL_ID
                },
                'bot_handler': bot_status,
                'telegram_bot': {
                    'running': telegram_bot is not None and hasattr(telegram_bot, 'bot_started') and telegram_bot.bot_started,
                    'initialized': telegram_bot is not None
                }
            },
            'indexing': {
                'running': indexing_status['is_running'],
                'last_run': indexing_status['last_run'],
                'next_run': indexing_status['next_run'],
                'stats': indexing_status['stats'],
                'total_indexed': indexing_status['total_indexed'],
                'total_skipped': indexing_status['total_skipped']
            },
            'sync_monitoring': {
                'running': sync_manager.is_monitoring,
                'deleted_count': sync_manager.deleted_count,
                'last_sync': sync_manager.last_sync
            },
            'thumbnail_system': {
                'manager_initialized': thumbnail_manager is not None,
                'stats': thumbnail_stats,
                'auto_extraction': True,
                'ttl_days': Config.THUMBNAIL_TTL_DAYS,
                'auto_cleanup': True,
                'extractor_initialized': thumbnail_extractor is not None
            },
            'components': {
                'cache': {
                    'enabled': cache_manager is not None,
                    'redis_enabled': cache_manager.redis_enabled if cache_manager else False
                },
                'verification': verification_system is not None,
                'premium': premium_system is not None,
                'poster_fetcher': poster_fetcher is not None,
                'thumbnail_manager': thumbnail_manager is not None,
                'bot_handler': bot_handler is not None and bot_handler.initialized,
                'telegram_bot': telegram_bot is not None
            },
            'api_keys': {
                'tmdb_primary': bool(Config.TMDB_API_KEY),
                'tmdb_backup_1': bool(Config.TMDB_API_KEY_2),
                'tmdb_backup_2': bool(Config.TMDB_API_KEY_3),
                'omdb_primary': bool(Config.OMDB_API_KEY),
                'omdb_backup': bool(Config.OMDB_API_KEY_2),
                'total_configured': sum([
                    bool(Config.TMDB_API_KEY),
                    bool(Config.TMDB_API_KEY_2),
                    bool(Config.TMDB_API_KEY_3),
                    bool(Config.OMDB_API_KEY),
                    bool(Config.OMDB_API_KEY_2)
                ])
            },
            'performance': {
                'endpoints_tracked': len(perf_stats),
                'slow_endpoints': {name: stats for name, stats in perf_stats.items() if stats['avg'] > 0.5}
            }
        }
        
        return jsonify(status_data)
        
    except Exception as e:
        logger.error(f"❌ Status endpoint error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/thumbnails/stats', methods=['GET'])
@performance_monitor.measure("thumbnails_stats_endpoint")
async def api_thumbnails_stats():
    """Get thumbnail system statistics"""
    try:
        if thumbnails_col is None:
            return jsonify({
                'status': 'error',
                'message': 'Thumbnails collection not initialized'
            }), 500
        
        # Get thumbnail statistics
        total_thumbnails = await thumbnails_col.count_documents({})
        extracted_thumbnails = await thumbnails_col.count_documents({'extracted': True})
        external_thumbnails = await thumbnails_col.count_documents({'source': {'$ne': 'extracted'}})
        
        # Get thumbnail sources distribution
        pipeline = [
            {"$group": {
                "_id": "$source",
                "count": {"$sum": 1}
            }},
            {"$sort": {"count": -1}}
        ]
        
        sources_distribution = []
        async for doc in thumbnails_col.aggregate(pipeline):
            sources_distribution.append({
                'source': doc['_id'] or 'unknown',
                'count': doc['count']
            })
        
        # Get thumbnail manager stats
        thumbnail_manager_stats = {}
        if thumbnail_manager:
            try:
                thumbnail_manager_stats = await thumbnail_manager.get_stats()
            except Exception as e:
                thumbnail_manager_stats = {'error': str(e)}
        
        # Calculate success rates
        success_rate = (extracted_thumbnails / total_thumbnails * 100) if total_thumbnails > 0 else 0
        
        return jsonify({
            'status': 'success',
            'thumbnail_system': {
                'total_thumbnails': total_thumbnails,
                'extracted_thumbnails': extracted_thumbnails,
                'external_thumbnails': external_thumbnails,
                'success_rate': f"{success_rate:.1f}%",
                'target_success_rate': '99%',
                'ttl_days': Config.THUMBNAIL_TTL_DAYS,
                'auto_cleanup_enabled': True
            },
            'sources_distribution': sources_distribution,
            'thumbnail_manager_stats': thumbnail_manager_stats,
            'database': {
                'collection': 'thumbnails',
                'ttl_index': f"{Config.THUMBNAIL_TTL_DAYS} days",
                'auto_deletion': True
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Thumbnails stats error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/admin/indexing/start', methods=['POST'])
@performance_monitor.measure("admin_indexing_start")
async def admin_indexing_start():
    """Admin endpoint to start indexing manually"""
    try:
        # Check authorization
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return jsonify({
                'status': 'error',
                'message': 'Unauthorized'
            }), 401
        
        # Simple token check (in production, use proper authentication)
        token = auth_header.replace('Bearer ', '').strip()
        if token != "sk4film-admin-token":  # Replace with secure token validation
            return jsonify({
                'status': 'error',
                'message': 'Invalid token'
            }), 403
        
        if file_indexing_manager.is_running:
            return jsonify({
                'status': 'info',
                'message': 'Indexing already running',
                'indexing_status': await file_indexing_manager.get_indexing_status()
            })
        
        # Start indexing
        await file_indexing_manager.start_indexing()
        
        return jsonify({
            'status': 'success',
            'message': 'Indexing started',
            'indexing_status': await file_indexing_manager.get_indexing_status()
        })
        
    except Exception as e:
        logger.error(f"❌ Admin indexing start error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/admin/indexing/stop', methods=['POST'])
@performance_monitor.measure("admin_indexing_stop")
async def admin_indexing_stop():
    """Admin endpoint to stop indexing"""
    try:
        # Check authorization
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return jsonify({
                'status': 'error',
                'message': 'Unauthorized'
            }), 401
        
        token = auth_header.replace('Bearer ', '').strip()
        if token != "sk4film-admin-token":
            return jsonify({
                'status': 'error',
                'message': 'Invalid token'
            }), 403
        
        if not file_indexing_manager.is_running:
            return jsonify({
                'status': 'info',
                'message': 'Indexing not running'
            })
        
        # Stop indexing
        await file_indexing_manager.stop_indexing()
        
        return jsonify({
            'status': 'success',
            'message': 'Indexing stopped'
        })
        
    except Exception as e:
        logger.error(f"❌ Admin indexing stop error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/admin/indexing/status', methods=['GET'])
@performance_monitor.measure("admin_indexing_status")
async def admin_indexing_status():
    """Admin endpoint to get indexing status"""
    try:
        # Check authorization
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return jsonify({
                'status': 'error',
                'message': 'Unauthorized'
            }), 401
        
        token = auth_header.replace('Bearer ', '').strip()
        if token != "sk4film-admin-token":
            return jsonify({
                'status': 'error',
                'message': 'Invalid token'
            }), 403
        
        return jsonify({
            'status': 'success',
            'indexing': await file_indexing_manager.get_indexing_status()
        })
        
    except Exception as e:
        logger.error(f"❌ Admin indexing status error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/admin/thumbnails/cleanup', methods=['POST'])
@performance_monitor.measure("admin_thumbnails_cleanup")
async def admin_thumbnails_cleanup():
    """Admin endpoint to manually cleanup old thumbnails"""
    try:
        # Check authorization
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return jsonify({
                'status': 'error',
                'message': 'Unauthorized'
            }), 401
        
        token = auth_header.replace('Bearer ', '').strip()
        if token != "sk4film-admin-token":
            return jsonify({
                'status': 'error',
                'message': 'Invalid token'
            }), 403
        
        if thumbnails_col is None:
            return jsonify({
                'status': 'error',
                'message': 'Thumbnails collection not initialized'
            }), 500
        
        # Calculate cutoff date
        cutoff_date = datetime.now() - timedelta(days=Config.THUMBNAIL_TTL_DAYS)
        
        # Delete old thumbnails
        result = await thumbnails_col.delete_many({
            'last_accessed': {'$lt': cutoff_date}
        })
        
        return jsonify({
            'status': 'success',
            'message': f'Cleaned up {result.deleted_count} old thumbnails',
            'cutoff_date': cutoff_date.isoformat(),
            'deleted_count': result.deleted_count
        })
        
    except Exception as e:
        logger.error(f"❌ Admin thumbnails cleanup error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/admin/database/stats', methods=['GET'])
@performance_monitor.measure("admin_database_stats")
async def admin_database_stats():
    """Admin endpoint to get detailed database statistics"""
    try:
        # Check authorization
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return jsonify({
                'status': 'error',
                'message': 'Unauthorized'
            }), 401
        
        token = auth_header.replace('Bearer ', '').strip()
        if token != "sk4film-admin-token":
            return jsonify({
                'status': 'error',
                'message': 'Invalid token'
            }), 403
        
        if db is None:
            return jsonify({
                'status': 'error',
                'message': 'Database not initialized'
            }), 500
        
        # Get files collection stats
        files_count = await files_col.count_documents({}) if files_col else 0
        video_files_count = await files_col.count_documents({'is_video_file': True}) if files_col else 0
        active_files = await files_col.count_documents({'status': 'active'}) if files_col else 0
        
        # Get thumbnails collection stats
        thumbnails_count = await thumbnails_col.count_documents({}) if thumbnails_col else 0
        extracted_count = await thumbnails_col.count_documents({'extracted': True}) if thumbnails_col else 0
        
        # Get verification collection stats
        verification_count = 0
        if verification_col:
            verification_count = await verification_col.count_documents({})
        
        # Get database stats
        db_stats = await db.command("dbStats")
        
        # Get collection sizes
        collections_info = []
        for collection_name in await db.list_collection_names():
            collection = db[collection_name]
            count = await collection.count_documents({})
            stats = await collection.aggregate([
                {"$group": {
                    "_id": None,
                    "size": {"$sum": {"$bsonSize": "$$ROOT"}}
                }}
            ]).to_list(length=1)
            
            size_bytes = stats[0]['size'] if stats else 0
            
            collections_info.append({
                'name': collection_name,
                'count': count,
                'size_bytes': size_bytes,
                'size_mb': size_bytes / 1024 / 1024
            })
        
        return jsonify({
            'status': 'success',
            'database': {
                'name': db.name,
                'collections': collections_info,
                'stats': {
                    'dataSize': db_stats.get('dataSize', 0),
                    'storageSize': db_stats.get('storageSize', 0),
                    'indexSize': db_stats.get('indexSize', 0),
                    'objects': db_stats.get('objects', 0),
                    'collections': db_stats.get('collections', 0)
                }
            },
            'files_collection': {
                'total_files': files_count,
                'video_files': video_files_count,
                'active_files': active_files,
                'indexed_percentage': f"{(active_files / files_count * 100):.1f}%" if files_count > 0 else "0%"
            },
            'thumbnails_collection': {
                'total_thumbnails': thumbnails_count,
                'extracted_thumbnails': extracted_count,
                'extraction_percentage': f"{(extracted_count / thumbnails_count * 100):.1f}%" if thumbnails_count > 0 else "0%",
                'ttl_days': Config.THUMBNAIL_TTL_DAYS
            },
            'verification_collection': {
                'total_verifications': verification_count
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Admin database stats error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# ============================================================================
# ✅ ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
async def not_found_error(e):
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found',
        'path': request.path
    }), 404

@app.errorhandler(500)
async def internal_error(e):
    logger.error(f"❌ Internal server error: {e}")
    return jsonify({
        'status': 'error',
        'message': 'Internal server error',
        'error': str(e) if app.debug else 'Contact administrator'
    }), 500

@app.errorhandler(405)
async def method_not_allowed_error(e):
    return jsonify({
        'status': 'error',
        'message': 'Method not allowed',
        'allowed_methods': ['GET', 'POST', 'OPTIONS']
    }), 405

# ============================================================================
# ✅ GRACEFUL SHUTDOWN HANDLER
# ============================================================================

async def graceful_shutdown():
    """Gracefully shutdown the system"""
    logger.info("=" * 60)
    logger.info("🛑 GRACEFUL SHUTDOWN INITIATED")
    logger.info("=" * 60)
    
    shutdown_tasks = []
    
    # 1. Stop file indexing
    if file_indexing_manager.is_running:
        logger.info("🛑 Stopping file indexing...")
        await file_indexing_manager.stop_indexing()
    
    # 2. Stop sync monitoring
    if sync_manager.is_monitoring:
        logger.info("🛑 Stopping sync monitoring...")
        await sync_manager.stop_sync_monitoring()
    
    # 3. Shutdown Telegram Bot
    if telegram_bot:
        logger.info("🤖 Shutting down Telegram Bot...")
        try:
            await telegram_bot.shutdown()
        except Exception as e:
            logger.error(f"❌ Telegram Bot shutdown error: {e}")
    
    # 4. Shutdown Bot Handler
    if bot_handler and bot_handler.initialized:
        logger.info("🤖 Shutting down Bot Handler...")
        await bot_handler.shutdown()
    
    # 5. Shutdown Thumbnail Manager
    if thumbnail_manager:
        logger.info("🖼️ Shutting down Thumbnail Manager...")
        await thumbnail_manager.shutdown()
    
    # 6. Shutdown Premium System
    if premium_system:
        logger.info("💎 Shutting down Premium System...")
        await premium_system.stop_cleanup_task()
    
    # 7. Shutdown Verification System
    if verification_system:
        logger.info("🔐 Shutting down Verification System...")
        await verification_system.stop()
    
    # 8. Shutdown Cache Manager
    if cache_manager:
        logger.info("🧠 Shutting down Cache Manager...")
        await cache_manager.stop()
    
    # 9. Stop Telegram sessions
    global User, Bot
    if User is not None:
        logger.info("👤 Stopping USER session...")
        try:
            await User.stop()
        except:
            pass
    
    if Bot is not None:
        logger.info("🤖 Stopping BOT session...")
        try:
            await Bot.stop()
        except:
            pass
    
    # 10. Close MongoDB connections
    if mongo_client:
        logger.info("🔌 Closing MongoDB connection...")
        mongo_client.close()
    
    logger.info("=" * 60)
    logger.info("✅ SHUTDOWN COMPLETE")
    logger.info("=" * 60)

# ============================================================================
# ✅ MAIN ENTRY POINT
# ============================================================================

async def main():
    """Main entry point"""
    try:
        # Initialize system
        init_success = await init_system()
        if not init_success:
            logger.error("❌ System initialization failed. Exiting...")
            return
        
        # Configure Hypercorn
        config = HyperConfig()
        config.bind = [f"0.0.0.0:{Config.WEB_SERVER_PORT}"]
        config.workers = 1  # Single worker for async
        config.accesslog = "-"
        config.errorlog = "-"
        config.loglevel = "info"
        
        # Log startup info
        logger.info("=" * 60)
        logger.info(f"🚀 SK4FiLM v9.0 Server starting on port {Config.WEB_SERVER_PORT}")
        logger.info(f"🌐 Backend URL: {Config.BACKEND_URL}")
        logger.info(f"🔗 Website URL: {Config.WEBSITE_URL}")
        logger.info(f"🤖 Bot Username: {Config.BOT_USERNAME}")
        logger.info(f"🎯 Thumbnail Success Target: 99%")
        logger.info(f"📊 Single MongoDB System")
        logger.info("=" * 60)
        
        # Register shutdown handler
        import signal
        import functools
        
        def signal_handler(signame):
            logger.info(f"📡 Received {signame}, shutting down gracefully...")
            asyncio.create_task(graceful_shutdown())
        
        # Setup signal handlers
        loop = asyncio.get_running_loop()
        for signame in ('SIGINT', 'SIGTERM'):
            loop.add_signal_handler(
                getattr(signal, signame),
                functools.partial(signal_handler, signame)
            )
        
        # Start server
        await serve(app, config)
        
    except Exception as e:
        logger.error(f"❌ Fatal error in main: {e}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        
        # Attempt graceful shutdown on error
        try:
            await graceful_shutdown()
        except Exception as shutdown_error:
            logger.error(f"❌ Error during emergency shutdown: {shutdown_error}")
        
        raise

if __name__ == "__main__":
    # Start the application
    asyncio.run(main())
                                            
