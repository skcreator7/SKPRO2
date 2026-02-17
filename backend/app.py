# ============================================================================
# 🚀 SK4FiLM v9.3 - WITH POSTER FETCHING & THUMBNAIL PRIORITY (FULLY FIXED)
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

# Poster Fetcher
try:
    from poster_fetching import PosterFetcher, PosterSource
    logger.debug("✅ Poster fetching module imported")
    POSTER_FETCHER_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ Poster fetching module import error: {e}")
    PosterFetcher = None
    PosterSource = None
    POSTER_FETCHER_AVAILABLE = False

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
        
        if elapsed > 2.0:
            logger.warning(f"⏱️ {name} took {elapsed:.3f}s")
    
    def get_stats(self):
        return self.measurements

performance_monitor = PerformanceMonitor()

# ============================================================================
# ✅ ASYNC CACHE DECORATOR
# ============================================================================

def async_cache_with_ttl(maxsize=128, ttl=300):
    """
    🔥 Cache decorator with TTL (Time To Live)
    maxsize: Maximum number of items in cache
    ttl: Time to live in seconds
    """
    cache = {}
    cache_lock = asyncio.Lock()
    
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
            async with cache_lock:
                if key in cache:
                    value, timestamp = cache[key]
                    if now - timestamp < ttl:
                        return value
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Store in cache
            async with cache_lock:
                cache[key] = (result, now)
                # Limit cache size
                if len(cache) > maxsize:
                    # Remove oldest entry
                    oldest_key = min(cache.keys(), key=lambda k: cache[k][1])
                    del cache[oldest_key]
            
            return result
        return wrapper
    return decorator

# ============================================================================
# ✅ CONFIGURATION - v9.3 WITH POSTER FETCHING
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
    VERIFICATION_REQUIRED = os.environ.get("VERIFICATION_REQUIRED", "False").lower() == "False"
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
    
    # 🔥 OPTIMIZATION SETTINGS
    POSTER_FETCHING_ENABLED = True  # ENABLED
    POSTER_CACHE_TTL = 86400  # 24 hours
    POSTER_FETCH_TIMEOUT = 3  # 3 seconds
    MAX_CONCURRENT_REQUESTS = int(os.environ.get("MAX_CONCURRENT_REQUESTS", "20"))
    CACHE_TTL = int(os.environ.get("CACHE_TTL", "600"))  # 10 minutes
    REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "5"))  # 5 seconds
    
    # Sync Management Settings
    MONITOR_INTERVAL = int(os.environ.get("MONITOR_INTERVAL", "600"))  # 10 minutes
    
    # Quality Settings
    QUALITY_PRIORITY = ['2160p', '1080p', '720p', '480p', '360p']
    HEVC_VARIANTS = ['720p HEVC', '1080p HEVC', '2160p HEVC']
    
    # 🔥 THUMBNAIL EXTRACTION SETTINGS
    THUMBNAIL_EXTRACTION_ENABLED = True
    THUMBNAIL_BATCH_SIZE = 3
    THUMBNAIL_EXTRACT_TIMEOUT = 5
    THUMBNAIL_CACHE_DURATION = 24 * 60 * 60
    THUMBNAIL_RETRY_LIMIT = 1
    THUMBNAIL_MAX_SIZE_KB = 200
    THUMBNAIL_TTL_DAYS = 30
    
    # 🔥 FILE CHANNEL INDEXING SETTINGS
    AUTO_INDEX_INTERVAL = int(os.environ.get("AUTO_INDEX_INTERVAL", "300"))
    BATCH_INDEX_SIZE = int(os.environ.get("BATCH_INDEX_SIZE", "200"))
    MAX_INDEX_LIMIT = int(os.environ.get("MAX_INDEX_LIMIT", "0"))
    INDEX_ALL_HISTORY = os.environ.get("INDEX_ALL_HISTORY", "true").lower() == "true"
    INSTANT_AUTO_INDEX = os.environ.get("INSTANT_AUTO_INDEX", "true").lower() == "true"
    
    # 🔥 SEARCH SETTINGS - OPTIMIZED
    SEARCH_MIN_QUERY_LENGTH = 2
    SEARCH_RESULTS_PER_PAGE = 12
    SEARCH_CACHE_TTL = 600  # 10 minutes cache
    
    # 🔥 OPTIMIZED STORAGE SETTINGS
    STORE_ONLY_THUMBNAILS = True
    EXTRACT_IN_BACKGROUND = True

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
    response.headers['X-SK4FiLM-Version'] = '9.3-POSTER-ENABLED'
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
thumbnails_col = None
posters_col = None

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

def detect_quality_enhanced(filename):
    if not filename:
        return "480p"
    
    filename_lower = filename.lower()
    
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
    
    is_hevc = any(re.search(pattern, filename_lower) for pattern in HEVC_PATTERNS)
    
    for pattern, quality in QUALITY_PATTERNS:
        if re.search(pattern, filename_lower):
            if is_hevc and quality in ['720p', '1080p', '2160p']:
                return f"{quality} HEVC"
            return quality
    
    return "480p"

# ============================================================================
# ✅ POSTER FETCHING FUNCTIONS
# ============================================================================

async def get_poster_for_movie(title: str, year: str = "", quality: str = "") -> Dict[str, Any]:
    """Get poster for movie - Original function"""
    global poster_fetcher, posters_col
    
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
    
    # Check cache first
    if posters_col is not None:
        try:
            cache_key = f"{normalize_title(title)}:{year}"
            cached = await posters_col.find_one({'cache_key': cache_key})
            if cached and cached.get('poster_url'):
                # Check if cache is still valid (24 hours)
                if datetime.now() - cached.get('cached_at', datetime.min) < timedelta(hours=24):
                    logger.debug(f"📦 Cached poster: {title}")
                    cached['found'] = True
                    cached['from_cache'] = True
                    return cached
        except Exception as e:
            logger.debug(f"Cache check error: {e}")
    
    try:
        poster_task = asyncio.create_task(poster_fetcher.fetch_poster(title))
        
        try:
            poster_data = await asyncio.wait_for(poster_task, timeout=Config.POSTER_FETCH_TIMEOUT)
            
            if poster_data and poster_data.get('poster_url'):
                logger.debug(f"✅ Poster fetched: {title[:30]} - {poster_data['source']}")
                poster_data['found'] = True
                poster_data['cached_at'] = datetime.now()
                poster_data['cache_key'] = f"{normalize_title(title)}:{year}"
                
                # Store in cache
                if posters_col is not None:
                    try:
                        await posters_col.update_one(
                            {'cache_key': poster_data['cache_key']},
                            {'$set': poster_data},
                            upsert=True
                        )
                    except Exception as e:
                        logger.debug(f"Cache store error: {e}")
                
                return poster_data
            else:
                raise ValueError("Invalid poster data")
                
        except (asyncio.TimeoutError, ValueError, Exception) as e:
            logger.debug(f"⚠️ Poster fetch timeout/error for {title[:30]}: {e}")
            
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

# ============================================================================
# ✅ THUMBNAIL INTEGRATION FUNCTIONS (CORRECTED)
# ============================================================================

async def get_best_thumbnail(normalized_title: str, clean_title: str = None, 
                            year: str = None, msg=None) -> Tuple[str, str]:
    """
    🎯 Get best thumbnail with priority:
    1. MongoDB (extracted Telegram thumbnails)
    2. Poster (TMDB/OMDB) - TRIGGERS BACKGROUND STORAGE
    3. Fallback
    """
    # PRIORITY 1: Check MongoDB for extracted thumbnail
    if thumbnails_col is not None:
        try:
            # Try exact match first
            doc = await thumbnails_col.find_one(
                {'normalized_title': normalized_title, 'has_thumbnail': True},
                {'thumbnail_url': 1, 'thumbnail_source': 1}
            )
            
            if doc and doc.get('thumbnail_url'):
                logger.debug(f"📦 MongoDB thumbnail found for: {clean_title or normalized_title}")
                return doc['thumbnail_url'], doc.get('thumbnail_source', 'mongodb')
            
            # Try fuzzy match if exact fails
            if clean_title:
                base_title = re.sub(r'\s+\d{4}$', '', clean_title)
                base_norm = normalize_title(base_title)
                
                doc = await thumbnails_col.find_one(
                    {'normalized_title': base_norm, 'has_thumbnail': True},
                    {'thumbnail_url': 1, 'thumbnail_source': 1}
                )
                
                if doc and doc.get('thumbnail_url'):
                    logger.debug(f"📦 MongoDB (fuzzy) thumbnail for: {clean_title}")
                    return doc['thumbnail_url'], doc.get('thumbnail_source', 'mongodb')
                    
        except Exception as e:
            logger.debug(f"⚠️ MongoDB thumbnail fetch error: {e}")
    
    # PRIORITY 2: Try poster fetch
    if Config.POSTER_FETCHING_ENABLED and poster_fetcher and clean_title:
        try:
            poster = await get_poster_for_movie(clean_title, year)
            
            if poster and poster.get('poster_url') and poster.get('found'):
                # 🔥 CRITICAL: If we have a message object, trigger background storage
                if msg and thumbnails_col is not None:
                    logger.debug(f"🎬 Poster found, storing real thumbnail in background for: {clean_title}")
                    asyncio.create_task(
                        try_store_real_thumbnail(normalized_title, clean_title, msg)
                    )
                else:
                    logger.debug(f"🎬 Poster found but no message to store thumbnail for: {clean_title}")
                
                return poster['poster_url'], poster.get('source', 'poster')
                
        except Exception as e:
            logger.debug(f"⚠️ Poster fetch error for {clean_title}: {e}")
    
    # PRIORITY 3: Fallback
    logger.debug(f"⚠️ Using fallback for: {clean_title or normalized_title}")
    return FALLBACK_THUMBNAIL_URL, "fallback"


async def try_store_real_thumbnail(normalized_title: str, clean_title: str, msg) -> None:
    """
    🔄 Background task to store real thumbnail from Telegram message
    """
    try:
        if not msg or thumbnails_col is None:
            return
        
        # Check if already exists
        existing = await thumbnails_col.find_one(
            {'normalized_title': normalized_title, 'has_thumbnail': True}
        )
        if existing:
            logger.debug(f"✅ Thumbnail already exists for {clean_title}")
            return
        
        # Get thumbnail from message
        file_name = None
        has_thumb = False
        thumb_file_id = None
        
        if msg.video:
            file_name = msg.video.file_name or "video.mp4"
            if hasattr(msg.video, 'thumbs') and msg.video.thumbs:
                thumb_file_id = msg.video.thumbs[0].file_id
                has_thumb = True
        elif msg.document:
            file_name = msg.document.file_name
            if hasattr(msg.document, 'thumbs') and msg.document.thumbs:
                thumb_file_id = msg.document.thumbs[0].file_id
                has_thumb = True
        
        if not file_name or not is_video_file(file_name):
            logger.debug(f"⏭️ Not a video file: {file_name}")
            return
        
        if not has_thumb or not thumb_file_id:
            logger.debug(f"⏭️ No thumbnail in Telegram for: {clean_title}")
            # Mark as no thumbnail to avoid re-checking
            await thumbnails_col.update_one(
                {'normalized_title': normalized_title},
                {'$set': {
                    'title': clean_title,
                    'has_thumbnail': False,
                    'checked_at': datetime.now(),
                    'file_name': file_name,
                    'message_id': msg.id,
                    'channel_id': msg.chat.id
                }},
                upsert=True
            )
            return
        
        # Download thumbnail
        client = User if user_session_ready else Bot
        if not client:
            logger.debug("⚠️ No Telegram client available")
            return
        
        download_path = await client.download_media(thumb_file_id, in_memory=True)
        if not download_path:
            logger.debug("⚠️ Failed to download thumbnail")
            return
        
        # Convert to base64
        if isinstance(download_path, bytes):
            thumbnail_data = download_path
        else:
            with open(download_path, 'rb') as f:
                thumbnail_data = f.read()
        
        thumbnail_url = f"data:image/jpeg;base64,{base64.b64encode(thumbnail_data).decode('utf-8')}"
        size_kb = len(thumbnail_url) / 1024
        
        # Extract metadata
        quality = detect_quality_enhanced(file_name)
        year = extract_year(file_name)
        
        # Store in MongoDB
        thumbnail_doc = {
            'normalized_title': normalized_title,
            'title': clean_title,
            'quality': quality,
            'year': year,
            'thumbnail_url': thumbnail_url,
            'thumbnail_source': 'telegram',
            'has_thumbnail': True,
            'extracted_at': datetime.now(),
            'message_id': msg.id,
            'channel_id': msg.chat.id,
            'file_name': file_name,
            'size_kb': size_kb
        }
        
        await thumbnails_col.update_one(
            {'normalized_title': normalized_title},
            {'$set': thumbnail_doc},
            upsert=True
        )
        
        logger.info(f"✅ Stored real thumbnail for: {clean_title} ({size_kb:.1f}KB)")
        
        # Update stats if thumbnail manager exists
        if thumbnail_manager:
            thumbnail_manager.stats['total_extracted'] += 1
            thumbnail_manager.stats['total_size_kb'] += size_kb
            
    except Exception as e:
        logger.debug(f"⚠️ Background thumbnail store error: {e}")


async def get_thumbnails_batch(movies: List[Dict]) -> List[Dict]:
    """
    🎯 Get thumbnails for multiple movies in batch
    """
    if not movies:
        return []
    
    start_time = time.time()
    
    # Extract normalized titles
    normalized_titles = [m.get('normalized_title', '') for m in movies if m.get('normalized_title')]
    
    # Batch fetch from MongoDB
    mongodb_thumbnails = {}
    if thumbnails_col is not None and normalized_titles:
        try:
            cursor = thumbnails_col.find(
                {
                    'normalized_title': {'$in': normalized_titles},
                    'has_thumbnail': True
                },
                {
                    'normalized_title': 1,
                    'thumbnail_url': 1,
                    'thumbnail_source': 1
                }
            )
            
            async for doc in cursor:
                mongodb_thumbnails[doc['normalized_title']] = {
                    'url': doc['thumbnail_url'],
                    'source': doc.get('thumbnail_source', 'mongodb')
                }
                
        except Exception as e:
            logger.debug(f"⚠️ Batch MongoDB fetch error: {e}")
    
    # Apply thumbnails to movies
    for movie in movies:
        normalized = movie.get('normalized_title', '')
        
        if normalized in mongodb_thumbnails:
            movie['thumbnail_url'] = mongodb_thumbnails[normalized]['url']
            movie['thumbnail_source'] = mongodb_thumbnails[normalized]['source']
            movie['has_thumbnail'] = True
        elif Config.POSTER_FETCHING_ENABLED and poster_fetcher and movie.get('title'):
            # Will be handled by individual poster fetch
            pass
        else:
            movie['thumbnail_url'] = FALLBACK_THUMBNAIL_URL
            movie['thumbnail_source'] = 'fallback'
            movie['has_thumbnail'] = True
            movie['is_fallback'] = True
    
    elapsed = time.time() - start_time
    logger.debug(f"🖼️ Batch thumbnails: {len(movies)} movies in {elapsed:.2f}s")
    
    return movies

# ============================================================================
# ✅ OPTIMIZED SEARCH v3.0 - WITH THUMBNAIL PRIORITY (CORRECTED)
# ============================================================================

@performance_monitor.measure("optimized_search")
@async_cache_with_ttl(maxsize=500, ttl=600)
async def search_movies_optimized(query, limit=15, page=1):
    """
    🔥 OPTIMIZED SEARCH v3.0 - WITH THUMBNAIL PRIORITY
    """
    start_time = time.time()
    offset = (page - 1) * limit
    
    logger.info(f"🔍 OPTIMIZED SEARCH for: '{query}'")
    
    # Dictionary to store unique results
    results_dict = {}
    
    # ============================================================================
    # ✅ STEP 1: Direct Telegram FILE CHANNEL Search (Files)
    # ============================================================================
    if user_session_ready and User is not None:
        try:
            file_count = 0
            
            # Search file channel for documents/videos
            async for msg in User.search_messages(
                Config.FILE_CHANNEL_ID, 
                query=query,
                limit=50
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
                
                # Create file entry with FULL MESSAGE OBJECT
                file_data = {
                    'quality': quality,
                    'file_name': file_name,
                    'file_size': msg.document.file_size if msg.document else msg.video.file_size,
                    'file_size_formatted': format_size(msg.document.file_size if msg.document else msg.video.file_size),
                    'message_id': msg.id,
                    'file_id': msg.document.file_id if msg.document else msg.video.file_id,
                    'date': msg.date,
                    'has_thumbnail_in_telegram': has_telegram_thumbnail(msg),
                    'tg_msg': msg  # ⭐ CRITICAL: Store the actual message object
                }
                
                if normalized not in results_dict:
                    # New movie - Store first message for thumbnail extraction
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
                        'poster_source': None,
                        'has_poster': False,
                        'search_score': 5,
                        'first_file_msg': msg  # Store the message for thumbnail
                    }
                
                # Add quality
                results_dict[normalized]['qualities'][quality] = file_data
                if quality not in results_dict[normalized]['available_qualities']:
                    results_dict[normalized]['available_qualities'].append(quality)
                
                # Update best date
                if msg.date and (not results_dict[normalized].get('date') or msg.date > results_dict[normalized]['date']):
                    results_dict[normalized]['date'] = msg.date
                    results_dict[normalized]['is_new'] = is_new(msg.date)
                
                file_count += 1
            
            logger.info(f"📁 Found {file_count} file results")
            
        except Exception as e:
            logger.error(f"❌ File channel search error: {e}")
    
    # ============================================================================
    # ✅ STEP 2: Direct Telegram TEXT CHANNELS Search (Posts)
    # ============================================================================
    if user_session_ready and User is not None:
        try:
            post_count = 0
            
            for channel_id in Config.TEXT_CHANNEL_IDS:
                try:
                    async for msg in User.search_messages(
                        channel_id, 
                        query=query,
                        limit=30
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
                            # UPGRADE to Post+Files
                            results_dict[normalized]['has_post'] = True
                            results_dict[normalized]['post_content'] = format_post(msg.text, max_length=500)
                            results_dict[normalized]['post_channel_id'] = channel_id
                            results_dict[normalized]['post_message_id'] = msg.id
                            results_dict[normalized]['result_type'] = 'file_and_post'
                            results_dict[normalized]['search_score'] = 10
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
                                'poster_source': None,
                                'has_poster': False,
                                'search_score': 7,
                                'first_file_msg': None  # No message for posts
                            }
                        
                        post_count += 1
                        
                except Exception as e:
                    logger.debug(f"Text search error in {channel_id}: {e}")
                    continue
            
            logger.info(f"📝 Found {post_count} post results")
            
        except Exception as e:
            logger.error(f"❌ Text channels search error: {e}")
    
    # ============================================================================
    # ✅ STEP 3: Get Thumbnails for Each Result (with message objects)
    # ============================================================================
    for normalized, result in results_dict.items():
        # Get the stored message for thumbnail extraction (only for file results)
        msg = result.get('first_file_msg')
        
        # Get best thumbnail (will store real one in background if needed)
        thumbnail_url, thumbnail_source = await get_best_thumbnail(
            normalized,
            result.get('title'),
            result.get('year'),
            msg  # Pass the actual message object
        )
        
        result['thumbnail_url'] = thumbnail_url
        result['thumbnail_source'] = thumbnail_source
        result['has_thumbnail'] = True
        
        # Clean up - remove the message object before sending to client
        if 'first_file_msg' in result:
            del result['first_file_msg']
        if 'qualities' in result:
            for quality, q_data in result['qualities'].items():
                if 'tg_msg' in q_data:
                    del q_data['tg_msg']  # Remove from individual quality entries too
    
    # ============================================================================
    # ✅ STEP 4: Convert to List and Sort
    # ============================================================================
    all_results = list(results_dict.values())
    
    # Sort by: search score, is_new, date
    all_results.sort(key=lambda x: (
        x.get('search_score', 0),
        1 if x.get('is_new') else 0,
        x.get('date') if isinstance(x.get('date'), datetime) else datetime.min
    ), reverse=True)
    
    # ============================================================================
    # ✅ STEP 5: Pagination
    # ============================================================================
    total = len(all_results)
    start_idx = offset
    end_idx = offset + limit
    paginated = all_results[start_idx:end_idx]
    
    # Statistics
    file_count = sum(1 for r in all_results if r.get('has_file'))
    post_count = sum(1 for r in all_results if r.get('has_post'))
    combined_count = sum(1 for r in all_results if r.get('has_file') and r.get('has_post'))
    mongodb_count = sum(1 for r in all_results if r.get('thumbnail_source') == 'mongodb')
    poster_count = sum(1 for r in all_results if r.get('thumbnail_source') in ['tmdb', 'omdb', 'poster'])
    fallback_count = sum(1 for r in all_results if r.get('thumbnail_source') == 'fallback')
    
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("📊 SEARCH RESULTS SUMMARY:")
    logger.info(f"   • Query: '{query}'")
    logger.info(f"   • Total results: {total}")
    logger.info(f"   • Post+Files: {combined_count}")
    logger.info(f"   • Post only: {post_count - combined_count}")
    logger.info(f"   • File only: {file_count - combined_count}")
    logger.info(f"   • MongoDB thumbnails: {mongodb_count}")
    logger.info(f"   • Posters: {poster_count}")
    logger.info(f"   • Fallback images: {fallback_count}")
    logger.info(f"   • Time: {elapsed:.2f}s")
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
            'mongodb_thumbnails': mongodb_count,
            'posters': poster_count,
            'fallback': fallback_count,
            'mode': 'optimized_v3',
            'poster_fetching': Config.POSTER_FETCHING_ENABLED
        },
        'bot_username': Config.BOT_USERNAME
    }

# ============================================================================
# ✅ THUMBNAIL MANAGER
# ============================================================================

class ThumbnailManager:
    """🖼️ THUMBNAIL MANAGER v9.2"""
    
    def __init__(self, mongodb=None, bot_client=None, user_client=None, file_channel_id=None):
        self.mongodb = mongodb
        self.bot_client = bot_client
        self.user_client = user_client
        self.file_channel_id = file_channel_id
        
        # Collections
        self.db = None
        self.thumbnails_col = None
        
        # State
        self.initialized = False
        self.is_extracting = False
        
        # Statistics
        self.stats = {
            'total_extracted': 0,
            'total_failed': 0,
            'total_no_thumbnail': 0,
            'total_size_kb': 0,
            'avg_size_kb': 0
        }
        
        logger.info("🖼️ Thumbnail Manager v9.2 initialized")
    
    async def initialize(self):
        """Initialize database collections and indexes"""
        try:
            if not self.mongodb:
                logger.error("❌ MongoDB client not provided")
                return False
            
            # Get database
            self.db = self.mongodb.sk4film
            
            # Create/Get collections
            self.thumbnails_col = self.db.thumbnails
            
            # 🔥 FIX: Drop all existing indexes and create fresh
            await self._reset_indexes()
            
            self.initialized = True
            logger.info("✅ Thumbnail Manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Thumbnail Manager initialization failed: {e}")
            return False
    
    async def _reset_indexes(self):
        """Drop all indexes and create fresh ones"""
        try:
            # Drop all indexes except _id_
            await self.thumbnails_col.drop_indexes()
            logger.info("✅ Dropped all existing indexes")
            
            # Create fresh indexes
            await self.thumbnails_col.create_index(
                [("normalized_title", 1)],
                name="title_idx",
                background=True
            )
            
            await self.thumbnails_col.create_index(
                [("normalized_title", 1), ("quality", 1)],
                unique=True,
                name="title_quality_idx",
                background=True
            )
            
            await self.thumbnails_col.create_index(
                [("has_thumbnail", 1)],
                name="has_thumb_idx",
                background=True
            )
            
            await self.thumbnails_col.create_index(
                [("message_id", 1)],
                name="msg_idx",
                background=True
            )
            
            await self.thumbnails_col.create_index(
                [("channel_id", 1), ("message_id", -1)],
                name="channel_msg_idx",
                background=True
            )
            
            logger.info("✅ Created fresh indexes")
            
        except Exception as e:
            logger.error(f"❌ Index reset error: {e}")
    
    async def extract_thumbnail(self, channel_id, message_id, file_name=None):
        """Extract thumbnail from Telegram message"""
        client = self.bot_client or self.user_client
        if not client:
            logger.error("❌ No Telegram client available")
            return None
        
        try:
            message = await client.get_messages(channel_id, message_id)
            if not message:
                return None
            
            thumbnail_data = None
            
            # Extract from video
            if message.video:
                if hasattr(message.video, 'thumbnail') and message.video.thumbnail:
                    thumbnail_file_id = message.video.thumbnail.file_id
                    download_path = await client.download_media(thumbnail_file_id, in_memory=True)
                    if download_path:
                        if isinstance(download_path, bytes):
                            thumbnail_data = download_path
                        else:
                            with open(download_path, 'rb') as f:
                                thumbnail_data = f.read()
            
            # Extract from document
            elif message.document:
                if file_name and is_video_file(file_name):
                    if hasattr(message.document, 'thumbnail') and message.document.thumbnail:
                        thumbnail_file_id = message.document.thumbnail.file_id
                        download_path = await client.download_media(thumbnail_file_id, in_memory=True)
                        if download_path:
                            if isinstance(download_path, bytes):
                                thumbnail_data = download_path
                            else:
                                with open(download_path, 'rb') as f:
                                    thumbnail_data = f.read()
            
            if thumbnail_data:
                base64_data = base64.b64encode(thumbnail_data).decode('utf-8')
                return f"data:image/jpeg;base64,{base64_data}"
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Thumbnail extraction error: {e}")
            return None
    
    async def extract_and_store(self, channel_id, message_id, file_name, title=None, quality=None, year=None):
        """Extract thumbnail and store in MongoDB"""
        try:
            if not title:
                title = extract_clean_title(file_name)
            
            normalized = normalize_title(title)
            
            if not quality:
                quality = detect_quality_enhanced(file_name)
            
            if not year:
                year = extract_year(file_name)
            
            # Check if already exists
            existing = await self.thumbnails_col.find_one({
                'normalized_title': normalized,
                'quality': quality
            })
            
            if existing and existing.get('has_thumbnail'):
                logger.debug(f"📦 Already exists: {title} - {quality}")
                return {
                    'success': True,
                    'thumbnail_url': existing.get('thumbnail_url'),
                    'cached': True,
                    'has_thumbnail': True
                }
            
            # Extract thumbnail
            thumbnail_url = await self.extract_thumbnail(channel_id, message_id, file_name)
            
            # Prepare document
            thumbnail_doc = {
                'normalized_title': normalized,
                'title': title,
                'quality': quality,
                'year': year,
                'message_id': message_id,
                'channel_id': channel_id,
                'file_name': file_name,
                'extracted_at': datetime.now(),
                'has_thumbnail': False
            }
            
            if thumbnail_url:
                size_kb = len(thumbnail_url) / 1024
                thumbnail_doc.update({
                    'thumbnail_url': thumbnail_url,
                    'thumbnail_source': 'telegram',
                    'size_kb': size_kb,
                    'has_thumbnail': True
                })
                self.stats['total_extracted'] += 1
                self.stats['total_size_kb'] += size_kb
                logger.info(f"✅ Stored thumbnail: {title} - {quality} ({size_kb:.1f}KB)")
            else:
                thumbnail_doc.update({
                    'thumbnail_url': None,
                    'thumbnail_source': None,
                    'size_kb': 0,
                    'has_thumbnail': False
                })
                self.stats['total_no_thumbnail'] += 1
                logger.debug(f"ℹ️ No thumbnail: {title} - {quality}")
            
            # Store in MongoDB
            await self.thumbnails_col.update_one(
                {'normalized_title': normalized, 'quality': quality},
                {'$set': thumbnail_doc},
                upsert=True
            )
            
            return {
                'success': True,
                'thumbnail_url': thumbnail_url,
                'has_thumbnail': bool(thumbnail_url),
                'cached': False
            }
            
        except Exception as e:
            logger.error(f"❌ Extract and store error: {e}")
            self.stats['total_failed'] += 1
            return {'success': False, 'error': str(e), 'has_thumbnail': False}
    
    async def get_stats(self):
        """Get thumbnail statistics"""
        try:
            if self.thumbnails_col:
                total = await self.thumbnails_col.count_documents({})
                with_thumb = await self.thumbnails_col.count_documents({'has_thumbnail': True})
                without_thumb = await self.thumbnails_col.count_documents({'has_thumbnail': False})
                return {**self.stats, 'total_documents': total, 'with_thumbnail': with_thumb, 'without_thumbnail': without_thumb}
        except:
            pass
        return self.stats
    
    async def shutdown(self):
        """Shutdown thumbnail manager"""
        logger.info("🖼️ Shutting down Thumbnail Manager...")
        logger.info(f"✅ Stats - Extracted: {self.stats['total_extracted']}, No thumbnail: {self.stats['total_no_thumbnail']}")

# ============================================================================
# ✅ OPTIMIZED FILE INDEXING MANAGER
# ============================================================================

class OptimizedFileIndexingManager:
    def __init__(self):
        self.is_running = False
        self.indexing_task = None
        self.last_run = None
        self.next_run = None
        self.total_indexed = 0
        self.thumbnails_extracted = 0
        self.indexing_stats = {
            'total_runs': 0,
            'total_messages_fetched': 0,
            'total_videos_found': 0,
            'videos_with_thumbnails': 0,
            'videos_without_thumbnails': 0,
            'thumbnails_extracted': 0,
            'thumbnails_failed': 0,
            'last_success': None,
            'last_error': None
        }
    
    async def start_indexing(self, force_reindex=False):
        if self.is_running:
            logger.warning("⚠️ File indexing already running")
            return
        
        logger.info("=" * 60)
        logger.info("🚀 STARTING THUMBNAIL INDEXING")
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
        self.is_running = False
        if self.indexing_task:
            self.indexing_task.cancel()
            try:
                await self.indexing_task
            except asyncio.CancelledError:
                pass
        logger.info("🛑 File indexing stopped")
    
    async def _run_optimized_indexing(self, force_reindex=False):
        if not user_session_ready and not bot_session_ready:
            logger.error("❌ No Telegram session available")
            return
        
        client = User if user_session_ready else Bot
        if client is None:
            logger.error("❌ No working Telegram client")
            return
        
        try:
            chat = await client.get_chat(Config.FILE_CHANNEL_ID)
            logger.info(f"📢 Channel: {chat.title}")
        except Exception as e:
            logger.error(f"❌ Cannot access file channel: {e}")
            return
        
        # Fetch messages
        all_messages = []
        offset_id = 0
        batch_size = 200
        
        logger.info("📥 Fetching messages from file channel...")
        
        while self.is_running:
            try:
                messages = []
                async for msg in client.get_chat_history(Config.FILE_CHANNEL_ID, limit=batch_size, offset_id=offset_id):
                    messages.append(msg)
                    if len(messages) >= batch_size:
                        break
                
                if not messages:
                    break
                
                all_messages.extend(messages)
                offset_id = messages[-1].id
                logger.info(f"📥 Fetched {len(all_messages)} messages...")
                self.indexing_stats['total_messages_fetched'] = len(all_messages)
                
                if len(messages) < batch_size:
                    break
                
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"❌ Error fetching messages: {e}")
                await asyncio.sleep(2)
                continue
        
        logger.info(f"✅ Total messages fetched: {len(all_messages)}")
        
        # Process video files
        video_count = 0
        thumbnail_candidates = []
        
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
            
            if has_telegram_thumbnail(msg):
                clean_title = extract_clean_title(file_name)
                normalized = normalize_title(clean_title)
                quality = detect_quality_enhanced(file_name)
                year = extract_year(file_name)
                
                thumbnail_candidates.append({
                    'message': msg,
                    'file_name': file_name,
                    'clean_title': clean_title,
                    'normalized': normalized,
                    'quality': quality,
                    'year': year,
                    'message_id': msg.id,
                    'file_id': msg.document.file_id if msg.document else msg.video.file_id
                })
        
        self.indexing_stats['total_videos_found'] = video_count
        self.indexing_stats['videos_with_thumbnails'] = len(thumbnail_candidates)
        self.indexing_stats['videos_without_thumbnails'] = video_count - len(thumbnail_candidates)
        
        logger.info("=" * 60)
        logger.info("📊 SCANNING COMPLETE")
        logger.info(f"   • Total video files: {video_count}")
        logger.info(f"   • Files WITH thumbnails: {len(thumbnail_candidates)}")
        logger.info(f"   • Files WITHOUT thumbnails: {video_count - len(thumbnail_candidates)}")
        logger.info("=" * 60)
        
        if thumbnail_candidates:
            logger.info("🖼️ Extracting thumbnails...")
            await self._extract_thumbnails_batch(thumbnail_candidates)
        
        self.indexing_stats['last_success'] = datetime.now()
        self.last_run = datetime.now()
    
    async def _extract_thumbnails_batch(self, candidates):
        batch_size = 3
        total_batches = math.ceil(len(candidates) / batch_size)
        successful = 0
        failed = 0
        
        client = User if user_session_ready else Bot
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(candidates))
            batch = candidates[start_idx:end_idx]
            logger.info(f"🖼️ Processing batch {batch_num + 1}/{total_batches}...")
            
            for candidate in batch:
                try:
                    thumbnail_url = None
                    
                    if bot_handler and bot_handler.initialized:
                        thumbnail_url = await bot_handler.extract_thumbnail(Config.FILE_CHANNEL_ID, candidate['message_id'])
                    
                    if not thumbnail_url and thumbnail_manager:
                        thumbnail_url = await thumbnail_manager.extract_thumbnail(Config.FILE_CHANNEL_ID, candidate['message_id'], candidate['file_name'])
                    
                    if thumbnail_url and thumbnails_col is not None:
                        thumbnail_doc = {
                            'normalized_title': candidate['normalized'],
                            'title': candidate['clean_title'],
                            'quality': candidate['quality'],
                            'year': candidate['year'],
                            'thumbnail_url': thumbnail_url,
                            'thumbnail_source': 'extracted',
                            'has_thumbnail': True,
                            'extracted_at': datetime.now(),
                            'message_id': candidate['message_id'],
                            'channel_id': Config.FILE_CHANNEL_ID,
                            'file_name': candidate['file_name'],
                            'size_kb': len(thumbnail_url) / 1024
                        }
                        
                        await thumbnails_col.update_one(
                            {'normalized_title': candidate['normalized'], 'quality': candidate['quality']},
                            {'$set': thumbnail_doc},
                            upsert=True
                        )
                        
                        successful += 1
                        logger.info(f"✅ Thumbnail stored: {candidate['clean_title']} - {candidate['quality']}")
                    else:
                        no_thumb_doc = {
                            'normalized_title': candidate['normalized'],
                            'title': candidate['clean_title'],
                            'quality': candidate['quality'],
                            'year': candidate['year'],
                            'has_thumbnail': False,
                            'extracted_at': datetime.now(),
                            'message_id': candidate['message_id'],
                            'channel_id': Config.FILE_CHANNEL_ID,
                            'file_name': candidate['file_name']
                        }
                        
                        await thumbnails_col.update_one(
                            {'normalized_title': candidate['normalized'], 'quality': candidate['quality']},
                            {'$set': no_thumb_doc},
                            upsert=True
                        )
                        
                        failed += 1
                        logger.warning(f"⚠️ No thumbnail: {candidate['clean_title']} - {candidate['quality']}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed: {candidate['clean_title']}: {e}")
                    failed += 1
                
                await asyncio.sleep(1)
            
            if batch_num < total_batches - 1:
                await asyncio.sleep(2)
        
        self.thumbnails_extracted = successful
        self.indexing_stats['thumbnails_extracted'] = successful
        self.indexing_stats['thumbnails_failed'] = failed
        logger.info(f"✅ Extraction complete: {successful} successful, {failed} failed")
    
    async def _indexing_loop(self):
        while self.is_running:
            try:
                await asyncio.sleep(Config.AUTO_INDEX_INTERVAL)
                if self.is_running:
                    await self._index_new_files()
                    self.last_run = datetime.now()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Indexing loop error: {e}")
                await asyncio.sleep(60)
    
    async def _index_new_files(self):
        try:
            if thumbnails_col is None:
                return
            
            latest = await thumbnails_col.find_one({'channel_id': Config.FILE_CHANNEL_ID}, sort=[('message_id', -1)])
            last_message_id = latest.get('message_id', 0) if latest else 0
            logger.info(f"🔍 Checking for new files after {last_message_id}")
            
            client = User if user_session_ready else Bot
            if not client:
                return
            
            messages = []
            async for msg in client.get_chat_history(Config.FILE_CHANNEL_ID, limit=100):
                if msg.id > last_message_id:
                    messages.append(msg)
                else:
                    break
            
            if not messages:
                logger.info("✅ No new files found")
                return
            
            logger.info(f"📥 Found {len(messages)} new messages")
            
            candidates = []
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
                
                if has_telegram_thumbnail(msg):
                    clean_title = extract_clean_title(file_name)
                    normalized = normalize_title(clean_title)
                    quality = detect_quality_enhanced(file_name)
                    year = extract_year(file_name)
                    
                    candidates.append({
                        'message': msg,
                        'file_name': file_name,
                        'clean_title': clean_title,
                        'normalized': normalized,
                        'quality': quality,
                        'year': year,
                        'message_id': msg.id,
                        'file_id': msg.document.file_id if msg.document else msg.video.file_id
                    })
            
            if candidates:
                logger.info(f"🖼️ Found {len(candidates)} new files with thumbnails")
                await self._extract_thumbnails_batch(candidates)
            
        except Exception as e:
            logger.error(f"❌ Error indexing new files: {e}")
    
    async def get_indexing_status(self):
        total_thumbnails = 0
        if thumbnails_col is not None:
            total_thumbnails = await thumbnails_col.count_documents({})
        
        return {
            'is_running': self.is_running,
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'next_run': self.next_run.isoformat() if self.next_run else None,
            'total_thumbnails': total_thumbnails,
            'thumbnails_extracted': self.thumbnails_extracted,
            'stats': self.indexing_stats,
            'user_session_ready': user_session_ready,
            'bot_session_ready': bot_session_ready
        }

file_indexing_manager = OptimizedFileIndexingManager()

# ============================================================================
# ✅ SYNC MANAGER
# ============================================================================

class OptimizedSyncManager:
    def __init__(self):
        self.is_monitoring = False
        self.monitoring_task = None
        self.deleted_count = 0
    
    async def start_sync_monitoring(self):
        if self.is_monitoring:
            return
        logger.info("👁️ Starting sync monitoring...")
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
            if thumbnails_col is None or User is None or not user_session_ready:
                return
            
            logger.info("🔍 Checking for deleted files...")
            
            batch_size = 100
            cursor = thumbnails_col.find(
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
                        await thumbnails_col.delete_one({"_id": item['db_id']})
                        deleted_count += 1
                        self.deleted_count += 1
                        
                        if deleted_count <= 5:
                            logger.info(f"🗑️ Deleted: {item['title'][:40]}...")
                
                if deleted_count > 0:
                    logger.info(f"✅ Deleted {deleted_count} files")
                else:
                    logger.info("✅ No deleted files found")
                    
            except Exception as e:
                logger.error(f"❌ Error checking messages: {e}")
                    
        except Exception as e:
            logger.error(f"❌ Auto-delete error: {e}")

sync_manager = OptimizedSyncManager()

# ============================================================================
# ✅ HOME MOVIES - WITH POSTER PRIORITY
# ============================================================================

@performance_monitor.measure("home_movies")
@async_cache_with_ttl(maxsize=1, ttl=60)
async def get_home_movies(limit=25):
    """
    🎬 HOME MOVIES - Priority: Poster > Extracted Thumbnail > FALLBACK
    """
    try:
        if User is None or not user_session_ready:
            return []
        
        movies = []
        seen_titles = set()
        
        logger.info(f"🎬 Fetching home movies from main channel...")
        
        async for msg in User.get_chat_history(Config.MAIN_CHANNEL_ID, limit=50):
            if msg is not None and msg.text and len(msg.text) > 25:
                title = extract_title_smart(msg.text)
                
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    
                    year_match = re.search(r'\b(19|20)\d{2}\b', title)
                    year = year_match.group() if year_match else ""
                    
                    clean_title = re.sub(r'\s+\(\d{4}\)$', '', title)
                    clean_title = re.sub(r'\s+\d{4}$', '', clean_title)
                    clean_title = clean_title.strip()
                    
                    post_content = msg.text
                    formatted_content = format_post(msg.text, max_length=500)
                    norm_title = normalize_title(clean_title)
                    
                    # ====================================================================
                    # ✅ PRIORITY 1: POSTER
                    # ====================================================================
                    thumbnail_url = None
                    thumbnail_source = None
                    poster_data = None
                    
                    if Config.POSTER_FETCHING_ENABLED and poster_fetcher:
                        try:
                            poster_data = await get_poster_for_movie(clean_title, year)
                            if poster_data and poster_data.get('found') and poster_data.get('poster_url'):
                                thumbnail_url = poster_data['poster_url']
                                thumbnail_source = poster_data.get('source', 'poster')
                                logger.debug(f"✅ POSTER found for: {clean_title[:30]}")
                        except Exception as e:
                            logger.debug(f"Poster error for {clean_title[:30]}: {e}")
                    
                    # ====================================================================
                    # ✅ PRIORITY 2: EXTRACTED THUMBNAIL
                    # ====================================================================
                    if not thumbnail_url and thumbnail_manager and thumbnails_col is not None:
                        try:
                            thumb_doc = await thumbnails_col.find_one(
                                {'normalized_title': norm_title, 'has_thumbnail': True}
                            ).hint([('normalized_title', 1)])
                            
                            if thumb_doc and thumb_doc.get('thumbnail_url'):
                                thumbnail_url = thumb_doc['thumbnail_url']
                                thumbnail_source = 'extracted'
                                logger.debug(f"✅ EXTRACTED THUMBNAIL found for: {clean_title[:30]}")
                        except Exception as e:
                            logger.debug(f"Thumbnail fetch error: {e}")
                    
                    # ====================================================================
                    # ✅ PRIORITY 3: FALLBACK
                    # ====================================================================
                    if not thumbnail_url:
                        thumbnail_url = FALLBACK_THUMBNAIL_URL
                        thumbnail_source = 'fallback'
                        logger.debug(f"⚠️ Using FALLBACK for: {clean_title[:30]}")
                    
                    movie_data = {
                        'title': clean_title,
                        'original_title': title,
                        'normalized_title': norm_title,
                        'year': year,
                        'date': msg.date.isoformat() if isinstance(msg.date, datetime) else str(msg.date),
                        'is_new': is_new(msg.date) if msg.date else False,
                        'channel_id': Config.MAIN_CHANNEL_ID,
                        'message_id': msg.id,
                        'has_file': False,
                        'has_post': True,
                        'content': formatted_content,
                        'post_content': post_content,
                        'thumbnail_url': thumbnail_url,
                        'thumbnail_source': thumbnail_source,
                        'has_thumbnail': True,
                        'poster_url': poster_data.get('poster_url') if poster_data else None,
                        'poster_source': poster_data.get('source') if poster_data else None,
                        'poster_rating': poster_data.get('rating') if poster_data else None,
                        'has_poster': bool(poster_data and poster_data.get('poster_url')),
                        'extracted_thumbnail': thumbnail_url if thumbnail_source == 'extracted' else None,
                        'has_extracted': thumbnail_source == 'extracted',
                        'is_fallback': thumbnail_source == 'fallback',
                        'image_priority': 'poster' if thumbnail_source in ['poster', 'tmdb', 'omdb'] else ('extracted' if thumbnail_source == 'extracted' else 'fallback')
                    }
                    
                    movies.append(movie_data)
                    
                    if len(movies) >= limit:
                        break
        
        # Statistics
        poster_count = sum(1 for m in movies if m.get('has_poster'))
        extracted_count = sum(1 for m in movies if m.get('has_extracted'))
        fallback_count = sum(1 for m in movies if m.get('is_fallback'))
        
        logger.info("=" * 60)
        logger.info("📊 HOME MOVIES SUMMARY:")
        logger.info(f"   • Total movies: {len(movies)}")
        logger.info(f"   • With Posters: {poster_count}")
        logger.info(f"   • With Extracted Thumbnails: {extracted_count}")
        logger.info(f"   • With Fallback Images: {fallback_count}")
        logger.info("=" * 60)
        
        return movies[:limit]
        
    except Exception as e:
        logger.error(f"❌ Home movies error: {e}")
        return []

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
    
    # Initialize BOT
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
    
    # Initialize USER
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
    global mongo_client, db, files_col, verification_col, thumbnails_col, posters_col
    
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
        thumbnails_col = db.thumbnails
        posters_col = db.posters
        
        # Create indexes for posters collection
        if posters_col is not None:
            await posters_col.create_index('cache_key', unique=True)
            await posters_col.create_index('cached_at')
        
        # Create indexes for thumbnails collection
        if thumbnails_col is not None:
            await thumbnails_col.create_index('normalized_title', unique=False)
            await thumbnails_col.create_index([('normalized_title', 1), ('has_thumbnail', 1)])
        
        logger.info("✅ MongoDB OK")
        return True
        
    except asyncio.TimeoutError:
        logger.error("❌ MongoDB connection timeout")
        return False
    except Exception as e:
        logger.error(f"❌ MongoDB error: {e}")
        return False

# ============================================================================
# ✅ POSTER FETCHER INITIALIZATION
# ============================================================================

async def init_poster_fetcher():
    """Initialize poster fetcher"""
    global poster_fetcher
    
    if not POSTER_FETCHER_AVAILABLE:
        logger.warning("⚠️ Poster fetcher module not available")
        return False
    
    if not Config.POSTER_FETCHING_ENABLED:
        logger.info("📌 Poster fetching is disabled in config")
        return False
    
    try:
        poster_fetcher = PosterFetcher(Config, cache_manager.redis_client if cache_manager else None)
        logger.info("✅ Poster Fetcher initialized")
        return True
    except Exception as e:
        logger.error(f"❌ Poster fetcher initialization failed: {e}")
        poster_fetcher = None
        return False

# ============================================================================
# ✅ THUMBNAIL MANAGER INITIALIZATION
# ============================================================================

async def init_thumbnail_manager():
    """Initialize Thumbnail Manager"""
    global thumbnail_manager
    
    thumbnail_manager = ThumbnailManager(
        mongodb=mongo_client,
        bot_client=Bot if bot_session_ready else None,
        user_client=User if user_session_ready else None,
        file_channel_id=Config.FILE_CHANNEL_ID
    )
    
    success = await thumbnail_manager.initialize()
    if success:
        logger.info("✅ Thumbnail Manager initialized")
    else:
        logger.error("❌ Thumbnail Manager initialization failed")
        thumbnail_manager = None
    
    return success

# ============================================================================
# ✅ INITIAL INDEXING FUNCTION
# ============================================================================

async def initial_indexing_optimized():
    """Start optimized file channel indexing"""
    global file_indexing_manager
    
    if not user_session_ready and not bot_session_ready:
        logger.error("❌ No Telegram session available")
        return
    
    if thumbnails_col is None:
        logger.error("❌ Database not ready")
        return
    
    logger.info("=" * 60)
    logger.info("🚀 STARTING THUMBNAIL INDEXING")
    logger.info("=" * 60)
    
    try:
        await file_indexing_manager.start_indexing(force_reindex=Config.INDEX_ALL_HISTORY)
        
        if user_session_ready:
            await sync_manager.start_sync_monitoring()
        
        logger.info("✅ Indexing started")
        
    except Exception as e:
        logger.error(f"❌ Initial indexing error: {e}")

# ============================================================================
# ✅ BOT INITIALIZATION
# ============================================================================

async def start_telegram_bot():
    """Start the Telegram bot"""
    try:
        if not PYROGRAM_AVAILABLE:
            logger.warning("❌ Pyrogram not available")
            return None
        
        if not Config.BOT_TOKEN:
            logger.warning("❌ Bot token not configured")
            return None
        
        logger.info("🤖 Starting Telegram Bot...")
        
        try:
            from bot_handlers import SK4FiLMBot
        except ImportError as e:
            logger.error(f"❌ Bot handler import error: {e}")
            return None
        
        bot_instance = SK4FiLMBot(Config, db_manager=None)
        bot_started = await bot_instance.initialize()
        
        if bot_started:
            logger.info("✅ Telegram Bot started!")
            return bot_instance
        else:
            logger.error("❌ Failed to start Telegram Bot")
            return None
            
    except Exception as e:
        logger.error(f"❌ Bot startup error: {e}")
        return None

# ============================================================================
# ✅ MAIN INITIALIZATION
# ============================================================================

@performance_monitor.measure("system_init")
async def init_system():
    start_time = time.time()
    
    try:
        logger.info("=" * 60)
        logger.info("🚀 SK4FiLM v9.3 - WITH POSTER FETCHING")
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
        global cache_manager, verification_system, premium_system
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
        
        # Initialize Telegram Sessions
        if PYROGRAM_AVAILABLE:
            telegram_ok = await init_telegram_sessions()
            if not telegram_ok:
                logger.warning("⚠️ Telegram sessions partially failed")
        
        # Initialize Thumbnail Manager
        await init_thumbnail_manager()
        
        # Initialize Poster Fetcher
        await init_poster_fetcher()
        
        # START TELEGRAM BOT
        global telegram_bot
        telegram_bot = await start_telegram_bot()
        if telegram_bot:
            logger.info("✅ Telegram Bot started")
        else:
            logger.warning("⚠️ Telegram Bot failed to start")
        
        # Start indexing
        if (user_session_ready or bot_session_ready) and thumbnails_col is not None:
            logger.info("🔍 Starting thumbnail indexing...")
            asyncio.create_task(initial_indexing_optimized())
        
        init_time = time.time() - start_time
        logger.info(f"⚡ SK4FiLM Started in {init_time:.2f}s")
        logger.info("=" * 60)
        logger.info("🔧 FEATURES:")
        logger.info(f"   • Poster Fetching: ✅ ENABLED")
        logger.info(f"   • Thumbnail Extraction: ✅ ENABLED")
        logger.info(f"   • Home Movies Priority: Poster > Thumbnail > Fallback")
        logger.info(f"   • Search Priority: MongoDB > Poster > Fallback")
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
    if thumbnails_col is not None:
        total_thumbnails = await thumbnails_col.count_documents({})
        with_thumb = await thumbnails_col.count_documents({'has_thumbnail': True})
        without_thumb = await thumbnails_col.count_documents({'has_thumbnail': False})
    else:
        total_thumbnails = 0
        with_thumb = 0
        without_thumb = 0
    
    if posters_col is not None:
        total_posters = await posters_col.count_documents({})
    else:
        total_posters = 0
    
    thumbnail_stats = {}
    if thumbnail_manager:
        thumbnail_stats = await thumbnail_manager.get_stats()
    
    indexing_status = await file_indexing_manager.get_indexing_status()
    
    bot_status = None
    if bot_handler:
        try:
            bot_status = await bot_handler.get_bot_status()
        except Exception as e:
            bot_status = {'initialized': False, 'error': str(e)}
    
    bot_running = telegram_bot is not None and hasattr(telegram_bot, 'bot_started') and telegram_bot.bot_started
    
    return jsonify({
        'status': 'healthy',
        'service': 'SK4FiLM v9.3 - WITH POSTER FETCHING',
        'poster_fetching': Config.POSTER_FETCHING_ENABLED,
        'storage_stats': {
            'total_thumbnails': total_thumbnails,
            'with_thumbnail': with_thumb,
            'without_thumbnail': without_thumb,
            'thumbnails_extracted': indexing_status.get('thumbnails_extracted', 0),
            'cached_posters': total_posters
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
            'database': thumbnails_col is not None,
            'thumbnail_manager': thumbnail_manager is not None,
            'poster_fetcher': poster_fetcher is not None,
            'bot_handler': bot_handler is not None and bot_handler.initialized
        },
        'indexing': indexing_status,
        'sync_monitoring': {
            'running': sync_manager.is_monitoring,
            'deleted_count': sync_manager.deleted_count
        },
        'response_time': f"{time.perf_counter():.3f}s"
    })

@app.route('/health')
@performance_monitor.measure("health_endpoint")
async def health():
    return jsonify({
        'status': 'ok',
        'sessions': {
            'user': user_session_ready,
            'bot': bot_session_ready,
            'bot_handler': bot_handler.initialized if bot_handler else False,
            'telegram_bot': telegram_bot is not None
        },
        'poster_fetching': Config.POSTER_FETCHING_ENABLED,
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
            'poster_fetching': Config.POSTER_FETCHING_ENABLED,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Movies API error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/search', methods=['GET'])
@performance_monitor.measure("search_endpoint")
async def api_search():
    """Fast search with poster fetching"""
    try:
        query = request.args.get('query', '').strip()
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', Config.SEARCH_RESULTS_PER_PAGE))
        
        if len(query) < Config.SEARCH_MIN_QUERY_LENGTH:
            return jsonify({
                'status': 'error',
                'message': f'Query must be at least {Config.SEARCH_MIN_QUERY_LENGTH} characters'
            }), 400
        
        result_data = await search_movies_optimized(query, limit, page)
        
        return jsonify({
            'status': 'success',
            'query': query,
            'results': result_data['results'],
            'pagination': result_data['pagination'],
            'search_metadata': result_data.get('search_metadata', {}),
            'bot_username': Config.BOT_USERNAME,
            'poster_fetching': Config.POSTER_FETCHING_ENABLED,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Search API error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
async def api_stats():
    try:
        perf_stats = performance_monitor.get_stats()
        
        if thumbnails_col is not None:
            total_thumbnails = await thumbnails_col.count_documents({})
            indexing_status = await file_indexing_manager.get_indexing_status()
        else:
            total_thumbnails = 0
            indexing_status = {}
        
        if posters_col is not None:
            total_posters = await posters_col.count_documents({})
        else:
            total_posters = 0
        
        thumbnail_stats = {}
        if thumbnail_manager:
            thumbnail_stats = await thumbnail_manager.get_stats()
        
        poster_stats = {}
        if poster_fetcher and hasattr(poster_fetcher, 'get_stats'):
            poster_stats = poster_fetcher.get_stats()
        
        bot_status = None
        if bot_handler:
            try:
                bot_status = await bot_handler.get_bot_status()
            except:
                bot_status = {'initialized': False}
        
        return jsonify({
            'status': 'success',
            'performance': perf_stats,
            'thumbnail_manager': thumbnail_stats,
            'poster_fetcher': poster_stats,
            'database_stats': {
                'total_thumbnails': total_thumbnails,
                'cached_posters': total_posters
            },
            'indexing_stats': indexing_status,
            'sync_stats': {
                'running': sync_manager.is_monitoring,
                'deleted_count': sync_manager.deleted_count
            },
            'bot_handler': bot_status,
            'poster_fetching': Config.POSTER_FETCHING_ENABLED,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Stats API error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/admin/reindex', methods=['POST'])
async def api_admin_reindex():
    try:
        auth_token = request.headers.get('X-Admin-Token')
        if not auth_token or auth_token != os.environ.get('ADMIN_TOKEN', 'sk4film_admin_123'):
            return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401
        
        Config.INDEX_ALL_HISTORY = True
        asyncio.create_task(initial_indexing_optimized())
        
        return jsonify({
            'status': 'success',
            'message': 'Thumbnail reindexing started',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Admin reindex error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/admin/indexing-status', methods=['GET'])
async def api_admin_indexing_status():
    try:
        indexing_status = await file_indexing_manager.get_indexing_status()
        return jsonify({'status': 'success', 'indexing': indexing_status})
    except Exception as e:
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
                    logger.info(f"✅ Cleared {len(keys)} cache keys")
            except Exception as e:
                logger.error(f"❌ Cache clear error: {e}")
        
        return jsonify({'status': 'success', 'message': 'Cache cleared'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/debug/indexing', methods=['GET'])
async def debug_indexing():
    try:
        if thumbnails_col is None:
            return jsonify({'error': 'Database not initialized'}), 500
        
        total_thumbnails = await thumbnails_col.count_documents({})
        with_thumb = await thumbnails_col.count_documents({'has_thumbnail': True})
        
        cursor = thumbnails_col.find({'has_thumbnail': True}).sort('extracted_at', -1).limit(5)
        recent_thumbnails = []
        async for doc in cursor:
            doc['_id'] = str(doc['_id'])
            recent_thumbnails.append({
                'title': doc['title'],
                'quality': doc.get('quality', 'unknown')
            })
        
        indexing_status = await file_indexing_manager.get_indexing_status()
        
        return jsonify({
            'status': 'success',
            'total_thumbnails': total_thumbnails,
            'with_thumbnail': with_thumb,
            'recent_thumbnails': recent_thumbnails,
            'indexing_status': indexing_status,
            'poster_fetching': Config.POSTER_FETCHING_ENABLED,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
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
    logger.info("🛑 Shutting down SK4FiLM v9.3...")
    
    if telegram_bot and hasattr(telegram_bot, 'shutdown'):
        await telegram_bot.shutdown()
    
    await file_indexing_manager.stop_indexing()
    await sync_manager.stop_sync_monitoring()
    
    if thumbnail_manager:
        await thumbnail_manager.shutdown()
    
    if bot_handler:
        await bot_handler.shutdown()
    
    if poster_fetcher and hasattr(poster_fetcher, 'close'):
        await poster_fetcher.close()
    
    if User is not None:
        await User.stop()
    
    if Bot is not None:
        await Bot.stop()
    
    if cache_manager is not None:
        await cache_manager.stop()
    
    if verification_system is not None:
        await verification_system.stop()
    
    if mongo_client is not None:
        mongo_client.close()
    
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
    
    logger.info(f"🌐 Starting SK4FiLM v9.3 on port {Config.WEB_SERVER_PORT}...")
    logger.info(f"📁 File Channel ID: {Config.FILE_CHANNEL_ID}")
    logger.info(f"🎬 Poster Fetching: {'ENABLED' if Config.POSTER_FETCHING_ENABLED else 'DISABLED'}")
    logger.info(f"🖼️ Home Movies: Poster > Thumbnail > Fallback")
    logger.info(f"🔍 Search: MongoDB > Poster > Fallback")
    logger.info(f"💾 MongoDB: Clean indexes + Poster cache")
    logger.info(f"🔄 Auto-indexing every {Config.AUTO_INDEX_INTERVAL}s")
    
    try:
        asyncio.run(serve(app, config))
    except KeyboardInterrupt:
        logger.info("👋 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
