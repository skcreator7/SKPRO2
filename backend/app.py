# ============================================================================
# 🚀 SK4FiLM v9.5 - FILE PREVIEW IMAGE STORE (FULLY FIXED)
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
# ✅ CONFIGURATION - v9.5 WITH FILE PREVIEW STORAGE
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
    POSTER_FETCHING_ENABLED = True
    POSTER_CACHE_TTL = 86400
    POSTER_FETCH_TIMEOUT = 3
    MAX_CONCURRENT_REQUESTS = int(os.environ.get("MAX_CONCURRENT_REQUESTS", "20"))
    CACHE_TTL = int(os.environ.get("CACHE_TTL", "600"))
    REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "5"))
    
    # Sync Management Settings
    MONITOR_INTERVAL = int(os.environ.get("MONITOR_INTERVAL", "600"))
    
    # Quality Settings
    QUALITY_PRIORITY = ['2160p', '1080p', '720p', '480p', '360p']
    HEVC_VARIANTS = ['720p HEVC', '1080p HEVC', '2160p HEVC']
    
    # 🔥 THUMBNAIL EXTRACTION SETTINGS - IMPROVED
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
    
    # 🔥 SEARCH SETTINGS
    SEARCH_MIN_QUERY_LENGTH = 2
    SEARCH_RESULTS_PER_PAGE = 12
    SEARCH_CACHE_TTL = 600
    
    # 🔥 STORAGE SETTINGS
    STORE_ONLY_THUMBNAILS = True
    EXTRACT_IN_BACKGROUND = True
    FORCE_PREVIEW_STORAGE = True  # 🔥 NEW: Force store preview even if extraction fails

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
    response.headers['X-SK4FiLM-Version'] = '9.5-PREVIEW-STORAGE'
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
previews_col = None  # 🔥 NEW: File previews collection

# Telegram Sessions
try:
    from pyrogram import Client
    from pyrogram.types import Message
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

# Preview Manager 🔥 NEW
preview_manager = None

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
            
            # Try to download thumbnail with thumb=0
            if message.video or (message.document and is_video_file(getattr(message.document, 'file_name', ''))):
                download_path = await self.bot.download_media(
                    message, 
                    in_memory=True,
                    thumb=0
                )
                
                if download_path:
                    if isinstance(download_path, bytes):
                        thumbnail_data = download_path
                    else:
                        with open(download_path, 'rb') as f:
                            thumbnail_data = f.read()
            
            if thumbnail_data:
                # Compress if too large
                if len(thumbnail_data) > Config.THUMBNAIL_MAX_SIZE_KB * 1024:
                    thumbnail_data = thumbnail_data[:Config.THUMBNAIL_MAX_SIZE_KB * 1024]
                
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
    """Detect real Telegram streaming preview reliably"""
    try:
        if not message:
            return False
            
        media = message.video or message.document
        if not media:
            return False

        # Check for thumbnails
        if hasattr(media, 'thumbs') and media.thumbs:
            return True
        
        if hasattr(media, 'thumbnail') and media.thumbnail:
            return True
            
        # For videos, always assume possible thumbnail
        if message.video:
            return True

        return False

    except Exception as e:
        logger.debug(f"Thumbnail detect error: {e}")
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
    """Get poster for movie"""
    global poster_fetcher, posters_col
    
    if poster_fetcher is None or not Config.POSTER_FETCHING_ENABLED:
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
                if datetime.now() - cached.get('cached_at', datetime.min) < timedelta(hours=24):
                    logger.debug(f"📦 Cached poster: {title[:30]}")
                    cached['found'] = True
                    cached['from_cache'] = True
                    return cached
        except Exception as e:
            logger.debug(f"Cache check error: {e}")
    
    try:
        poster_task = asyncio.create_task(poster_fetcher.fetch_poster(title, year))
        
        try:
            poster_data = await asyncio.wait_for(poster_task, timeout=Config.POSTER_FETCH_TIMEOUT)
            
            if poster_data and poster_data.get('poster_url'):
                logger.debug(f"✅ Poster fetched: {title[:30]} - {poster_data.get('source', 'unknown')}")
                
                result = {
                    'poster_url': poster_data['poster_url'],
                    'source': poster_data.get('source', 'tmdb'),
                    'rating': poster_data.get('rating', '0.0'),
                    'year': year or poster_data.get('year', ''),
                    'title': title,
                    'quality': quality or 'unknown',
                    'found': True,
                    'cached_at': datetime.now(),
                    'cache_key': f"{normalize_title(title)}:{year}"
                }
                
                # Store in cache
                if posters_col is not None:
                    asyncio.create_task(
                        posters_col.update_one(
                            {'cache_key': result['cache_key']},
                            {'$set': result},
                            upsert=True
                        )
                    )
                
                return result
            else:
                return {
                    'poster_url': '',
                    'source': 'none',
                    'rating': '0.0',
                    'year': year,
                    'title': title,
                    'quality': quality or 'unknown',
                    'found': False
                }
                
        except asyncio.TimeoutError:
            if not poster_task.done():
                poster_task.cancel()
            return {
                'poster_url': '',
                'source': 'timeout',
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
            'source': 'error',
            'rating': '0.0',
            'year': year,
            'title': title,
            'quality': quality or 'unknown',
            'found': False
        }

# ============================================================================
# ✅ 🔥 NEW: FILE PREVIEW MANAGER
# ============================================================================

class FilePreviewManager:
    """🎬 FILE PREVIEW MANAGER - Store and serve file previews"""
    
    def __init__(self, mongodb=None):
        self.mongodb = mongodb
        self.db = None
        self.previews_col = None
        self.initialized = False
        
        self.stats = {
            'total_previews': 0,
            'total_size_kb': 0,
            'avg_size_kb': 0
        }
        
        logger.info("🎬 File Preview Manager initialized")
    
    async def initialize(self):
        """Initialize preview collection"""
        try:
            if not self.mongodb:
                logger.error("❌ MongoDB client not provided")
                return False
            
            self.db = self.mongodb.sk4film
            self.previews_col = self.db.file_previews
            
            # Create indexes
            await self._create_indexes()
            
            self.initialized = True
            logger.info("✅ File Preview Manager initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ File Preview Manager initialization failed: {e}")
            return False
    
    async def _create_indexes(self):
        """Create indexes for previews collection"""
        try:
            existing = await self.previews_col.index_information()
            
            indexes = {
                "title_idx": [("normalized_title", 1)],
                "msg_idx": [("message_id", 1)],
                "channel_msg_idx": [("channel_id", 1), ("message_id", -1)]
            }
            
            for name, keys in indexes.items():
                if name not in existing:
                    await self.previews_col.create_index(keys, name=name, background=True)
            
            logger.info("✅ Preview indexes created")
            
        except Exception as e:
            logger.error(f"❌ Index creation error: {e}")
    
    async def store_preview(self, message_id: int, channel_id: int, file_name: str, 
                           preview_data: bytes, title: str = None, quality: str = None, 
                           year: str = None) -> Dict[str, Any]:
        """Store file preview in MongoDB"""
        try:
            if not self.initialized or not self.previews_col:
                return {'success': False, 'error': 'Preview manager not initialized'}
            
            if not title:
                title = extract_clean_title(file_name)
            
            normalized = normalize_title(title)
            
            if not quality:
                quality = detect_quality_enhanced(file_name)
            
            if not year:
                year = extract_year(file_name)
            
            # Convert to base64
            base64_data = base64.b64encode(preview_data).decode('utf-8')
            preview_url = f"data:image/jpeg;base64,{base64_data}"
            size_kb = len(preview_url) / 1024
            
            # Prepare document
            preview_doc = {
                'normalized_title': normalized,
                'title': title,
                'quality': quality,
                'year': year,
                'message_id': message_id,
                'channel_id': channel_id,
                'file_name': file_name,
                'preview_url': preview_url,
                'preview_source': 'telegram_preview',
                'size_kb': size_kb,
                'stored_at': datetime.now(),
                'has_preview': True
            }
            
            # Store in MongoDB
            result = await self.previews_col.update_one(
                {'message_id': message_id, 'channel_id': channel_id},
                {'$set': preview_doc},
                upsert=True
            )
            
            # Update stats
            self.stats['total_previews'] += 1
            self.stats['total_size_kb'] += size_kb
            self.stats['avg_size_kb'] = self.stats['total_size_kb'] / self.stats['total_previews']
            
            logger.info(f"✅ Preview stored: {title} - {quality} ({size_kb:.1f}KB)")
            
            return {
                'success': True,
                'preview_url': preview_url,
                'size_kb': size_kb
            }
            
        except Exception as e:
            logger.error(f"❌ Store preview error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def get_preview(self, message_id: int, channel_id: int) -> Optional[str]:
        """Get preview URL for a specific message"""
        try:
            if not self.initialized or not self.previews_col:
                return None
            
            doc = await self.previews_col.find_one(
                {'message_id': message_id, 'channel_id': channel_id},
                {'preview_url': 1}
            )
            
            if doc and doc.get('preview_url'):
                return doc['preview_url']
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Get preview error: {e}")
            return None
    
    async def get_preview_by_title(self, normalized_title: str, quality: str = None) -> Optional[str]:
        """Get preview URL by title (for search results)"""
        try:
            if not self.initialized or not self.previews_col:
                return None
            
            query = {'normalized_title': normalized_title}
            if quality:
                query['quality'] = quality
            
            doc = await self.previews_col.find_one(query, {'preview_url': 1})
            
            if doc and doc.get('preview_url'):
                return doc['preview_url']
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Get preview by title error: {e}")
            return None
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get preview statistics"""
        try:
            if self.initialized and self.previews_col:
                total = await self.previews_col.count_documents({})
                return {**self.stats, 'total_documents': total}
        except:
            pass
        return self.stats
    
    async def shutdown(self):
        """Shutdown preview manager"""
        logger.info("🎬 Shutting down File Preview Manager...")
        logger.info(f"✅ Stats - Total previews: {self.stats['total_previews']}")

# Initialize preview manager
preview_manager = FilePreviewManager()

# ============================================================================
# ✅ 🔥 NEW: STORE FILE PREVIEW FUNCTION
# ============================================================================

async def store_file_preview(msg, file_name: str = None, force: bool = False) -> Optional[str]:
    """
    🎯 Store file preview image in MongoDB
    Returns preview URL if successful, None otherwise
    """
    try:
        if not msg or not preview_manager or not preview_manager.initialized:
            return None
        
        if not file_name:
            if msg.document:
                file_name = msg.document.file_name
            elif msg.video:
                file_name = msg.video.file_name or "video.mp4"
            else:
                return None
        
        if not is_video_file(file_name):
            return None
        
        # Check if already exists (optional)
        if not force:
            existing = await preview_manager.get_preview(msg.id, msg.chat.id)
            if existing:
                logger.debug(f"📦 Preview already exists for message {msg.id}")
                return existing
        
        # Determine client
        client = None
        if user_session_ready and User:
            client = User
        elif bot_session_ready and Bot:
            client = Bot
        elif bot_handler and bot_handler.initialized:
            client = bot_handler.bot
        
        if not client:
            logger.warning("⚠️ No client available for preview extraction")
            return None
        
        # Download preview (thumb=0 gets streaming preview)
        logger.debug(f"📥 Downloading preview for: {file_name}")
        data = await client.download_media(
            msg,
            in_memory=True,
            thumb=0  # 🔥 CRITICAL: Gets streaming preview frame
        )
        
        if not data:
            logger.debug(f"❌ No preview data for: {file_name}")
            return None
        
        # Convert to bytes if needed
        if not isinstance(data, bytes):
            with open(data, "rb") as f:
                data = f.read()
        
        # Compress if too large
        if len(data) > Config.THUMBNAIL_MAX_SIZE_KB * 1024:
            logger.debug(f"📦 Compressing preview ({len(data)/1024:.1f}KB)")
            data = data[:Config.THUMBNAIL_MAX_SIZE_KB * 1024]
        
        # Store in preview manager
        title = extract_clean_title(file_name)
        quality = detect_quality_enhanced(file_name)
        year = extract_year(file_name)
        
        result = await preview_manager.store_preview(
            message_id=msg.id,
            channel_id=msg.chat.id,
            file_name=file_name,
            preview_data=data,
            title=title,
            quality=quality,
            year=year
        )
        
        if result.get('success'):
            logger.info(f"✅ Preview stored successfully: {title}")
            return result['preview_url']
        else:
            logger.error(f"❌ Failed to store preview: {result.get('error')}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Store file preview error: {e}")
        return None

# ============================================================================
# ✅ THUMBNAIL INTEGRATION FUNCTIONS - WITH PREVIEW FALLBACK
# ============================================================================

async def get_best_thumbnail(normalized_title: str, clean_title: str = None, 
                            year: str = None, msg=None) -> Tuple[str, str]:
    """
    🎯 Get best thumbnail with priority:
    1. MongoDB (extracted Telegram thumbnails)
    2. File Preview (stored preview images) 🔥 NEW
    3. Poster (TMDB/OMDB)
    4. Fallback
    """
    # PRIORITY 1: Check MongoDB for extracted thumbnail
    if thumbnails_col is not None:
        try:
            doc = await thumbnails_col.find_one(
                {'normalized_title': normalized_title, 'has_thumbnail': True},
                {'thumbnail_url': 1, 'thumbnail_source': 1}
            )
            
            if doc and doc.get('thumbnail_url'):
                logger.debug(f"📦 MongoDB thumbnail found for: {clean_title or normalized_title}")
                return doc['thumbnail_url'], doc.get('thumbnail_source', 'mongodb')
                    
        except Exception as e:
            logger.debug(f"⚠️ MongoDB thumbnail fetch error: {e}")
    
    # PRIORITY 2: Check File Preview (🔥 NEW)
    if preview_manager and preview_manager.initialized:
        try:
            # If we have a message object, try to get preview for this specific message
            if msg:
                preview_url = await preview_manager.get_preview(msg.id, msg.chat.id)
                if preview_url:
                    logger.debug(f"📦 File preview found for message: {clean_title}")
                    return preview_url, 'file_preview'
            
            # Otherwise try by title
            preview_url = await preview_manager.get_preview_by_title(normalized_title)
            if preview_url:
                logger.debug(f"📦 File preview found by title: {clean_title}")
                return preview_url, 'file_preview'
                
        except Exception as e:
            logger.debug(f"⚠️ File preview fetch error: {e}")
    
    # PRIORITY 3: Try poster fetch
    if Config.POSTER_FETCHING_ENABLED and poster_fetcher and clean_title:
        try:
            poster = await get_poster_for_movie(clean_title, year)
            
            if poster and poster.get('poster_url') and poster.get('found'):
                # 🔥 CRITICAL: If we have a message object, trigger background storage
                if msg and thumbnails_col is not None and Config.FORCE_PREVIEW_STORAGE:
                    logger.debug(f"🎬 Poster found, storing file preview in background for: {clean_title}")
                    asyncio.create_task(
                        store_file_preview(msg, file_name=None, force=True)
                    )
                
                return poster['poster_url'], poster.get('source', 'poster')
                
        except Exception as e:
            logger.debug(f"⚠️ Poster fetch error for {clean_title}: {e}")
    
    # PRIORITY 4: Fallback
    logger.debug(f"⚠️ Using fallback for: {clean_title or normalized_title}")
    return FALLBACK_THUMBNAIL_URL, "fallback"

async def try_store_real_thumbnail(normalized_title: str, clean_title: str, msg) -> None:
    """Store real Telegram streaming preview frame"""
    try:
        if not msg or thumbnails_col is None:
            return

        # Check if already exists
        existing = await thumbnails_col.find_one({
            'normalized_title': normalized_title,
            'has_thumbnail': True
        })
        if existing:
            logger.debug(f"✅ Thumbnail already exists for: {clean_title}")
            return

        # 🔥 Use preview manager to store
        await store_file_preview(msg, file_name=None, force=True)
            
    except Exception as e:
        logger.error(f"❌ Thumbnail storage error for {clean_title}: {e}")

# ============================================================================
# ✅ OPTIMIZED SEARCH v3.2 - WITH FILE PREVIEW PRIORITY
# ============================================================================

@performance_monitor.measure("optimized_search")
@async_cache_with_ttl(maxsize=500, ttl=600)
async def search_movies_optimized(query, limit=15, page=1):
    """
    🔥 OPTIMIZED SEARCH v3.2 - WITH FILE PREVIEW PRIORITY
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
                    'tg_msg': msg
                }
                
                if normalized not in results_dict:
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
                        'search_score': 5,
                        'first_file_msg': msg,
                        'file_messages': [msg]  # 🔥 Store all messages for preview
                    }
                else:
                    # Add message to list for preview options
                    if 'file_messages' not in results_dict[normalized]:
                        results_dict[normalized]['file_messages'] = []
                    results_dict[normalized]['file_messages'].append(msg)
                
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
                            results_dict[normalized]['has_post'] = True
                            results_dict[normalized]['post_content'] = format_post(msg.text, max_length=500)
                            results_dict[normalized]['post_channel_id'] = channel_id
                            results_dict[normalized]['post_message_id'] = msg.id
                            results_dict[normalized]['result_type'] = 'file_and_post'
                            results_dict[normalized]['search_score'] = 10
                        else:
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
                                'search_score': 7,
                                'first_file_msg': None,
                                'file_messages': []  # No messages for posts
                            }
                        
                        post_count += 1
                        
                except Exception as e:
                    logger.debug(f"Text search error in {channel_id}: {e}")
                    continue
            
            logger.info(f"📝 Found {post_count} post results")
            
        except Exception as e:
            logger.error(f"❌ Text channels search error: {e}")
    
    # ============================================================================
    # ✅ STEP 3: Get Thumbnails for Each Result (with priority)
    # ============================================================================
    for normalized, result in results_dict.items():
        thumbnail_url = None
        thumbnail_source = None
        
        # Try each message until we find a thumbnail/preview
        messages = result.get('file_messages', [])
        if result.get('first_file_msg'):
            messages.insert(0, result['first_file_msg'])
        
        # Remove duplicates
        seen_ids = set()
        unique_messages = []
        for msg in messages:
            if msg and msg.id not in seen_ids:
                seen_ids.add(msg.id)
                unique_messages.append(msg)
        
        # Try to get thumbnail from any message
        for msg in unique_messages[:3]:  # Try first 3 messages max
            thumb_url, thumb_src = await get_best_thumbnail(
                normalized,
                result.get('title'),
                result.get('year'),
                msg
            )
            
            if thumb_url and thumb_src != 'fallback':
                thumbnail_url = thumb_url
                thumbnail_source = thumb_src
                logger.debug(f"✅ Found thumbnail from {thumb_src} for {result.get('title')}")
                break
        
        # If no thumbnail found from messages, try without message
        if not thumbnail_url:
            thumbnail_url, thumbnail_source = await get_best_thumbnail(
                normalized,
                result.get('title'),
                result.get('year'),
                None
            )
        
        result['thumbnail_url'] = thumbnail_url
        result['thumbnail_source'] = thumbnail_source
        result['has_thumbnail'] = True
        
        # Clean up
        if 'first_file_msg' in result:
            del result['first_file_msg']
        if 'file_messages' in result:
            del result['file_messages']
        if 'qualities' in result:
            for quality, q_data in result['qualities'].items():
                if 'tg_msg' in q_data:
                    del q_data['tg_msg']
    
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
    preview_count = sum(1 for r in all_results if r.get('thumbnail_source') == 'file_preview')
    poster_count = sum(1 for r in all_results if r.get('thumbnail_source') in ['tmdb', 'omdb', 'poster'])
    fallback_count = sum(1 for r in all_results if r.get('thumbnail_source') == 'fallback')
    
    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info("📊 SEARCH RESULTS SUMMARY:")
    logger.info(f"   • Query: '{query}'")
    logger.info(f"   • Total results: {total}")
    logger.info(f"   • Post+Files: {combined_count}")
    logger.info(f"   • Post only: {post_count - combined_count}")
    logger.info(f"   • File only: {file_count - combined_count}")
    logger.info(f"   • MongoDB thumbnails: {mongodb_count}")
    logger.info(f"   • File previews: {preview_count} 🔥")
    logger.info(f"   • Posters: {poster_count}")
    logger.info(f"   • Fallback images: {fallback_count}")
    logger.info(f"   • Time: {elapsed:.2f}s")
    logger.info("=" * 70)
    
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
            'file_previews': preview_count,
            'posters': poster_count,
            'fallback': fallback_count,
            'mode': 'optimized_v3.2',
            'poster_fetching': Config.POSTER_FETCHING_ENABLED,
            'preview_storage': True
        },
        'bot_username': Config.BOT_USERNAME
    }

# ============================================================================
# ✅ THUMBNAIL MANAGER - UPDATED WITH PREVIEW SUPPORT
# ============================================================================

class ThumbnailManager:
    """🖼️ THUMBNAIL MANAGER v12 - WITH PREVIEW SUPPORT"""

    def __init__(self, mongodb=None, bot_client=None, user_client=None, file_channel_id=None):
        self.mongodb = mongodb
        self.bot_client = bot_client
        self.user_client = user_client
        self.file_channel_id = file_channel_id

        self.db = None
        self.thumbnails_col = None

        self.initialized = False
        self.is_extracting = False

        self.stats = {
            'total_extracted': 0,
            'total_failed': 0,
            'total_no_thumbnail': 0,
            'total_size_kb': 0,
            'avg_size_kb': 0
        }

        logger.info("🖼️ Thumbnail Manager v12 initialized")

    async def initialize(self):
        """Initialize database collections and indexes"""
        try:
            if not self.mongodb:
                logger.error("❌ MongoDB client not provided")
                return False

            self.db = self.mongodb.sk4film
            self.thumbnails_col = self.db.thumbnails

            await self._reset_indexes()

            self.initialized = True
            logger.info("✅ Thumbnail Manager initialized successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Thumbnail Manager initialization failed: {e}")
            return False

    async def _reset_indexes(self):
        """Safely manage indexes"""
        try:
            existing = await self.thumbnails_col.index_information()

            indexes = {
                "title_idx": [("normalized_title", 1)],
                "title_quality_idx": [("normalized_title", 1), ("quality", 1)],
                "has_thumbnail_idx": [("has_thumbnail", 1)],
                "msg_idx": [("message_id", 1)],
                "channel_msg_idx": [("channel_id", 1), ("message_id", -1)]
            }

            for name, keys in indexes.items():
                if name in existing:
                    continue

                await self.thumbnails_col.create_index(
                    keys,
                    name=name,
                    background=True
                )

            logger.info("✅ Index management completed")

        except Exception as e:
            logger.error(f"❌ Index management error: {e}")
    
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
            
            if message.video or (message.document and is_video_file(file_name or '')):
                download_path = await client.download_media(
                    message,
                    in_memory=True,
                    thumb=0
                )
                
                if download_path:
                    if isinstance(download_path, bytes):
                        thumbnail_data = download_path
                    else:
                        with open(download_path, 'rb') as f:
                            thumbnail_data = f.read()
            
            if thumbnail_data:
                if len(thumbnail_data) > Config.THUMBNAIL_MAX_SIZE_KB * 1024:
                    thumbnail_data = thumbnail_data[:Config.THUMBNAIL_MAX_SIZE_KB * 1024]
                
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
                'has_thumbnail': True
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
            
            # If extraction fails but preview manager exists, try to get preview
            if not thumbnail_url and preview_manager and preview_manager.initialized:
                logger.debug(f"🔄 Thumbnail extraction failed, checking preview for: {title}")
                preview_url = await preview_manager.get_preview(message_id, channel_id)
                if preview_url:
                    thumbnail_url = preview_url
            
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
                {'normalized_title': normalized},
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
# ✅ OPTIMIZED FILE INDEXING MANAGER - WITH PREVIEW STORAGE
# ============================================================================

class OptimizedFileIndexingManager:
    def __init__(self):
        self.is_running = False
        self.indexing_task = None
        self.last_run = None
        self.next_run = None
        self.total_indexed = 0
        self.thumbnails_extracted = 0
        self.previews_stored = 0  # 🔥 NEW
        self.indexing_stats = {
            'total_runs': 0,
            'total_messages_fetched': 0,
            'total_videos_found': 0,
            'videos_with_thumbnails': 0,
            'videos_without_thumbnails': 0,
            'thumbnails_extracted': 0,
            'previews_stored': 0,
            'thumbnails_failed': 0,
            'last_success': None,
            'last_error': None
        }
    
    async def start_indexing(self, force_reindex=False):
        if self.is_running:
            logger.warning("⚠️ File indexing already running")
            return
        
        logger.info("=" * 60)
        logger.info("🚀 STARTING FILE INDEXING WITH PREVIEW STORAGE")
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
        video_messages = []
        
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
            
            clean_title = extract_clean_title(file_name)
            normalized = normalize_title(clean_title)
            quality = detect_quality_enhanced(file_name)
            year = extract_year(file_name)
            
            video_messages.append({
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
        self.indexing_stats['videos_with_thumbnails'] = len(video_messages)
        
        logger.info("=" * 60)
        logger.info("📊 SCANNING COMPLETE")
        logger.info(f"   • Total video files: {video_count}")
        logger.info(f"   • Processing: {len(video_messages)} files")
        logger.info("=" * 60)
        
        if video_messages:
            logger.info("🖼️ Extracting thumbnails and previews...")
            await self._extract_all_batch(video_messages)
        
        self.indexing_stats['last_success'] = datetime.now()
        self.last_run = datetime.now()
    
    async def _extract_all_batch(self, candidates):
        """Extract thumbnails and store previews for all candidates"""
        batch_size = 3
        total_batches = math.ceil(len(candidates) / batch_size)
        thumb_successful = 0
        preview_successful = 0
        failed = 0
        
        client = User if user_session_ready else Bot
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(candidates))
            batch = candidates[start_idx:end_idx]
            logger.info(f"🖼️ Processing batch {batch_num + 1}/{total_batches}...")
            
            for candidate in batch:
                try:
                    # 1. Try to store preview first (always works if preview available)
                    preview_stored = False
                    if preview_manager and preview_manager.initialized:
                        preview_url = await store_file_preview(
                            candidate['message'],
                            candidate['file_name'],
                            force=True
                        )
                        if preview_url:
                            preview_successful += 1
                            preview_stored = True
                            logger.info(f"✅ Preview stored: {candidate['clean_title']}")
                    
                    # 2. Try thumbnail extraction
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
                            {'normalized_title': candidate['normalized']},
                            {'$set': thumbnail_doc},
                            upsert=True
                        )
                        
                        thumb_successful += 1
                        logger.info(f"✅ Thumbnail stored: {candidate['clean_title']}")
                    elif not preview_stored:
                        # Only count as failed if neither thumbnail nor preview worked
                        failed += 1
                        logger.warning(f"⚠️ No preview/thumbnail: {candidate['clean_title']}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed: {candidate['clean_title']}: {e}")
                    failed += 1
                
                await asyncio.sleep(1)
            
            if batch_num < total_batches - 1:
                await asyncio.sleep(2)
        
        self.thumbnails_extracted = thumb_successful
        self.previews_stored = preview_successful
        self.indexing_stats['thumbnails_extracted'] = thumb_successful
        self.indexing_stats['previews_stored'] = preview_successful
        self.indexing_stats['thumbnails_failed'] = failed
        
        logger.info(f"✅ Extraction complete:")
        logger.info(f"   • Thumbnails: {thumb_successful}")
        logger.info(f"   • Previews: {preview_successful} 🔥")
        logger.info(f"   • Failed: {failed}")
    
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
            if thumbnails_col is None and preview_manager is None:
                return
            
            # Get latest message ID from either collection
            last_message_id = 0
            if thumbnails_col is not None:
                latest = await thumbnails_col.find_one({'channel_id': Config.FILE_CHANNEL_ID}, sort=[('message_id', -1)])
                last_message_id = latest.get('message_id', 0) if latest else 0
            
            if preview_manager and preview_manager.previews_col:
                latest_preview = await preview_manager.previews_col.find_one({'channel_id': Config.FILE_CHANNEL_ID}, sort=[('message_id', -1)])
                if latest_preview:
                    last_message_id = max(last_message_id, latest_preview.get('message_id', 0))
            
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
                logger.info(f"🖼️ Found {len(candidates)} new files")
                await self._extract_all_batch(candidates)
            
        except Exception as e:
            logger.error(f"❌ Error indexing new files: {e}")
    
    async def get_indexing_status(self):
        total_thumbnails = 0
        if thumbnails_col is not None:
            total_thumbnails = await thumbnails_col.count_documents({})
        
        total_previews = 0
        if preview_manager and preview_manager.previews_col:
            total_previews = await preview_manager.previews_col.count_documents({})
        
        return {
            'is_running': self.is_running,
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'next_run': self.next_run.isoformat() if self.next_run else None,
            'total_thumbnails': total_thumbnails,
            'total_previews': total_previews,
            'thumbnails_extracted': self.thumbnails_extracted,
            'previews_stored': self.previews_stored,
            'stats': self.indexing_stats,
            'user_session_ready': user_session_ready,
            'bot_session_ready': bot_session_ready
        }

file_indexing_manager = OptimizedFileIndexingManager()

# ============================================================================
# ✅ SYNC MANAGER - UPDATED WITH PREVIEW SUPPORT
# ============================================================================

class OptimizedSyncManager:
    def __init__(self):
        self.is_monitoring = False
        self.monitoring_task = None
        self.deleted_count = 0
        self.preview_deleted_count = 0  # 🔥 NEW
    
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
            if User is None or not user_session_ready:
                return
            
            logger.info("🔍 Checking for deleted files...")
            
            # Check thumbnails collection
            if thumbnails_col is not None:
                await self._check_and_delete_from_collection(thumbnails_col)
            
            # Check previews collection
            if preview_manager and preview_manager.previews_col:
                await self._check_and_delete_from_collection(preview_manager.previews_col, is_preview=True)
            
        except Exception as e:
            logger.error(f"❌ Auto-delete error: {e}")
    
    async def _check_and_delete_from_collection(self, collection, is_preview=False):
        """Check and delete deleted files from a collection"""
        try:
            batch_size = 100
            cursor = collection.find(
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
                        await collection.delete_one({"_id": item['db_id']})
                        deleted_count += 1
                        
                        if is_preview:
                            self.preview_deleted_count += 1
                        else:
                            self.deleted_count += 1
                        
                        if deleted_count <= 5:
                            logger.info(f"🗑️ Deleted from {'preview' if is_preview else 'thumbnail'}: {item['title'][:40]}...")
                
                if deleted_count > 0:
                    logger.info(f"✅ Deleted {deleted_count} {'previews' if is_preview else 'thumbnails'}")
                    
            except Exception as e:
                logger.error(f"❌ Error checking messages: {e}")
                
        except Exception as e:
            logger.error(f"❌ Collection check error: {e}")

sync_manager = OptimizedSyncManager()

# ============================================================================
# ✅ HOME MOVIES - WITH FILE PREVIEW PRIORITY
# ============================================================================

@performance_monitor.measure("home_movies")
@async_cache_with_ttl(maxsize=1, ttl=60)
async def get_home_movies(limit=25):
    """
    🎬 HOME MOVIES - Priority: File Preview > Poster > Thumbnail > Fallback
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
                    # ✅ PRIORITY 1: FILE PREVIEW (if available)
                    # ====================================================================
                    thumbnail_url = None
                    thumbnail_source = None
                    
                    if preview_manager and preview_manager.initialized:
                        try:
                            # Try to find a preview for this movie
                            preview_url = await preview_manager.get_preview_by_title(norm_title)
                            if preview_url:
                                thumbnail_url = preview_url
                                thumbnail_source = 'file_preview'
                                logger.debug(f"✅ FILE PREVIEW found for: {clean_title[:30]}")
                        except Exception as e:
                            logger.debug(f"Preview fetch error: {e}")
                    
                    # ====================================================================
                    # ✅ PRIORITY 2: POSTER
                    # ====================================================================
                    if not thumbnail_url and Config.POSTER_FETCHING_ENABLED and poster_fetcher:
                        try:
                            poster_data = await get_poster_for_movie(clean_title, year)
                            if poster_data and poster_data.get('found') and poster_data.get('poster_url'):
                                thumbnail_url = poster_data['poster_url']
                                thumbnail_source = poster_data.get('source', 'poster')
                                logger.debug(f"✅ POSTER found for: {clean_title[:30]}")
                        except Exception as e:
                            logger.debug(f"Poster error for {clean_title[:30]}: {e}")
                    
                    # ====================================================================
                    # ✅ PRIORITY 3: EXTRACTED THUMBNAIL
                    # ====================================================================
                    if not thumbnail_url and thumbnails_col is not None:
                        try:
                            thumb_doc = await thumbnails_col.find_one(
                                {'normalized_title': norm_title, 'has_thumbnail': True}
                            )
                            
                            if thumb_doc and thumb_doc.get('thumbnail_url'):
                                thumbnail_url = thumb_doc['thumbnail_url']
                                thumbnail_source = thumb_doc.get('thumbnail_source', 'extracted')
                                logger.debug(f"✅ EXTRACTED THUMBNAIL found for: {clean_title[:30]}")
                        except Exception as e:
                            logger.debug(f"Thumbnail fetch error: {e}")
                    
                    # ====================================================================
                    # ✅ PRIORITY 4: FALLBACK
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
                        'has_preview': thumbnail_source == 'file_preview',
                        'has_poster': thumbnail_source in ['tmdb', 'omdb', 'poster'],
                        'has_extracted': thumbnail_source == 'extracted',
                        'is_fallback': thumbnail_source == 'fallback'
                    }
                    
                    movies.append(movie_data)
                    
                    if len(movies) >= limit:
                        break
        
        # Statistics
        preview_count = sum(1 for m in movies if m.get('has_preview'))
        poster_count = sum(1 for m in movies if m.get('has_poster'))
        extracted_count = sum(1 for m in movies if m.get('has_extracted'))
        fallback_count = sum(1 for m in movies if m.get('is_fallback'))
        
        logger.info("=" * 70)
        logger.info("📊 HOME MOVIES SUMMARY:")
        logger.info(f"   • Total movies: {len(movies)}")
        logger.info(f"   • With File Previews: {preview_count} 🔥")
        logger.info(f"   • With Posters: {poster_count}")
        logger.info(f"   • With Extracted Thumbnails: {extracted_count}")
        logger.info(f"   • With Fallback Images: {fallback_count}")
        logger.info("=" * 70)
        
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
# ✅ MONGODB INITIALIZATION - WITH PREVIEW COLLECTION
# ============================================================================

@performance_monitor.measure("mongodb_init")
async def init_mongodb():
    global mongo_client, db, files_col, verification_col, thumbnails_col, posters_col, previews_col
    
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
        previews_col = db.file_previews  # 🔥 NEW


        # ---------- SAFE INDEX CREATOR ----------
        async def safe_index(col, keys, **kwargs):
            try:
                await col.create_index(keys, **kwargs)
            except Exception as e:
                msg = str(e)
                if "already exists" in msg or "IndexOptionsConflict" in msg:
                    logger.warning(f"⚠️ Index already exists ignored: {keys}")
                else:
                    raise


        # ---------- POSTER INDEXES ----------
        if posters_col is not None:
            await safe_index(posters_col, 'cache_key', unique=True)
            await safe_index(posters_col, 'cached_at')


        # ---------- THUMBNAIL INDEXES ----------
        if thumbnails_col is not None:
            await safe_index(thumbnails_col, 'normalized_title')
            await safe_index(thumbnails_col, [('normalized_title', 1), ('has_thumbnail', 1)])
            await safe_index(thumbnails_col, 'has_thumbnail')

        # ---------- PREVIEW INDEXES ----------
        if previews_col is not None:
            await safe_index(previews_col, 'normalized_title')
            await safe_index(previews_col, [('normalized_title', 1)])
            await safe_index(previews_col, [('message_id', 1), ('channel_id', 1)], unique=True)

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
# ✅ PREVIEW MANAGER INITIALIZATION
# ============================================================================

async def init_preview_manager():
    """Initialize File Preview Manager"""
    global preview_manager
    
    preview_manager = FilePreviewManager(mongodb=mongo_client)
    
    success = await preview_manager.initialize()
    if success:
        logger.info("✅ File Preview Manager initialized")
    else:
        logger.error("❌ File Preview Manager initialization failed")
        preview_manager = None
    
    return success

# ============================================================================
# ✅ INITIAL INDEXING FUNCTION
# ============================================================================

async def initial_indexing_optimized():
    """Start optimized file channel indexing with preview storage"""
    global file_indexing_manager
    
    if not user_session_ready and not bot_session_ready:
        logger.error("❌ No Telegram session available")
        return
    
    if thumbnails_col is None and preview_manager is None:
        logger.error("❌ No storage collections available")
        return
    
    logger.info("=" * 60)
    logger.info("🚀 STARTING FILE INDEXING WITH PREVIEW STORAGE")
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
# ✅ MAIN INITIALIZATION - WITH PREVIEW MANAGER
# ============================================================================

@performance_monitor.measure("system_init")
async def init_system():
    start_time = time.time()
    
    try:
        logger.info("=" * 70)
        logger.info("🚀 SK4FiLM v9.5 - FILE PREVIEW STORAGE")
        logger.info("=" * 70)
        
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
        
        # Initialize Preview Manager 🔥 NEW
        await init_preview_manager()
        
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
        if (user_session_ready or bot_session_ready) and (thumbnails_col is not None or preview_manager is not None):
            logger.info("🔍 Starting file indexing with preview storage...")
            asyncio.create_task(initial_indexing_optimized())
        
        init_time = time.time() - start_time
        logger.info(f"⚡ SK4FiLM Started in {init_time:.2f}s")
        logger.info("=" * 70)
        logger.info("🔧 FEATURES:")
        logger.info(f"   • File Preview Storage: ✅ ENABLED 🔥")
        logger.info(f"   • Poster Fetching: ✅ ENABLED")
        logger.info(f"   • Thumbnail Extraction: ✅ ENABLED")
        logger.info(f"   • Home Movies Priority: Preview > Poster > Thumbnail > Fallback")
        logger.info(f"   • Search Priority: Preview > MongoDB > Poster > Fallback")
        logger.info(f"   • Background Storage: ✅ ACTIVE")
        logger.info("=" * 70)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ System initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# ✅ API ROUTES - UPDATED WITH PREVIEW STATS
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
    
    if preview_manager and preview_manager.previews_col:
        total_previews = await preview_manager.previews_col.count_documents({})
    else:
        total_previews = 0
    
    thumbnail_stats = {}
    if thumbnail_manager:
        thumbnail_stats = await thumbnail_manager.get_stats()
    
    preview_stats = {}
    if preview_manager:
        preview_stats = await preview_manager.get_stats()
    
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
        'service': 'SK4FiLM v9.5 - FILE PREVIEW STORAGE',
        'poster_fetching': Config.POSTER_FETCHING_ENABLED,
        'preview_storage': True,
        'storage_stats': {
            'total_thumbnails': total_thumbnails,
            'with_thumbnail': with_thumb,
            'without_thumbnail': without_thumb,
            'total_previews': total_previews,  # 🔥 NEW
            'thumbnails_extracted': indexing_status.get('thumbnails_extracted', 0),
            'previews_stored': indexing_status.get('previews_stored', 0),
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
            'preview_manager': preview_manager is not None,
            'poster_fetcher': poster_fetcher is not None,
            'bot_handler': bot_handler is not None and bot_handler.initialized
        },
        'indexing': indexing_status,
        'sync_monitoring': {
            'running': sync_manager.is_monitoring,
            'deleted_count': sync_manager.deleted_count,
            'preview_deleted_count': sync_manager.preview_deleted_count
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
        'preview_storage': True,
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
            'preview_storage': True,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Movies API error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/search', methods=['GET'])
@performance_monitor.measure("search_endpoint")
async def api_search():
    """Fast search with file preview priority"""
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
            'preview_storage': True,
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
        
        if preview_manager and preview_manager.previews_col:
            total_previews = await preview_manager.previews_col.count_documents({})
            preview_stats = await preview_manager.get_stats()
        else:
            total_previews = 0
            preview_stats = {}
        
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
            'preview_manager': preview_stats,
            'poster_fetcher': poster_stats,
            'database_stats': {
                'total_thumbnails': total_thumbnails,
                'total_previews': total_previews,
                'cached_posters': total_posters
            },
            'indexing_stats': indexing_status,
            'sync_stats': {
                'running': sync_manager.is_monitoring,
                'deleted_count': sync_manager.deleted_count,
                'preview_deleted_count': sync_manager.preview_deleted_count
            },
            'bot_handler': bot_status,
            'poster_fetching': Config.POSTER_FETCHING_ENABLED,
            'preview_storage': True,
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
            'message': 'File indexing with preview storage started',
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
        
        total_previews = 0
        if preview_manager and preview_manager.previews_col:
            total_previews = await preview_manager.previews_col.count_documents({})
        
        cursor = thumbnails_col.find({'has_thumbnail': True}).sort('extracted_at', -1).limit(5)
        recent_thumbnails = []
        async for doc in cursor:
            doc['_id'] = str(doc['_id'])
            recent_thumbnails.append({
                'title': doc['title'],
                'quality': doc.get('quality', 'unknown')
            })
        
        preview_cursor = None
        recent_previews = []
        if preview_manager and preview_manager.previews_col:
            preview_cursor = preview_manager.previews_col.find().sort('stored_at', -1).limit(5)
            async for doc in preview_cursor:
                doc['_id'] = str(doc['_id'])
                recent_previews.append({
                    'title': doc['title'],
                    'quality': doc.get('quality', 'unknown'),
                    'size_kb': doc.get('size_kb', 0)
                })
        
        indexing_status = await file_indexing_manager.get_indexing_status()
        
        return jsonify({
            'status': 'success',
            'total_thumbnails': total_thumbnails,
            'with_thumbnail': with_thumb,
            'total_previews': total_previews,
            'recent_thumbnails': recent_thumbnails,
            'recent_previews': recent_previews,
            'indexing_status': indexing_status,
            'poster_fetching': Config.POSTER_FETCHING_ENABLED,
            'preview_storage': True,
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
    logger.info("🛑 Shutting down SK4FiLM v9.5...")
    
    if telegram_bot and hasattr(telegram_bot, 'shutdown'):
        await telegram_bot.shutdown()
    
    await file_indexing_manager.stop_indexing()
    await sync_manager.stop_sync_monitoring()
    
    if thumbnail_manager:
        await thumbnail_manager.shutdown()
    
    if preview_manager:
        await preview_manager.shutdown()
    
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
    
    logger.info(f"🌐 Starting SK4FiLM v9.5 on port {Config.WEB_SERVER_PORT}...")
    logger.info(f"📁 File Channel ID: {Config.FILE_CHANNEL_ID}")
    logger.info(f"🎬 Poster Fetching: {'ENABLED' if Config.POSTER_FETCHING_ENABLED else 'DISABLED'}")
    logger.info(f"🖼️ File Preview Storage: ENABLED 🔥")
    logger.info(f"🎯 Home Movies: Preview > Poster > Thumbnail > Fallback")
    logger.info(f"🔍 Search: Preview > MongoDB > Poster > Fallback")
    logger.info(f"💾 MongoDB: Thumbnails + Previews + Posters")
    logger.info(f"🔄 Auto-indexing every {Config.AUTO_INDEX_INTERVAL}s")
    
    try:
        asyncio.run(serve(app, config))
    except KeyboardInterrupt:
        logger.info("👋 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
