# ============================================================================
# 🚀 SK4FiLM v9.0 - STREAMING & DOWNLOAD SUPPORT WITH REAL MESSAGE IDS - OPTIMIZED
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
    response.headers['X-SK4FiLM-Version'] = '9.0-STREAMING-ENABLED'
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

# Indexing State
is_indexing = False
last_index_time = None
indexing_task = None

# ============================================================================
# ✅ BOT HANDLER MODULE - FIXED WITH MISSING METHODS
# ============================================================================

class BotHandler:
    """Bot handler for Telegram bot operations - FIXED VERSION"""
    
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
# ✅ QUALITY MERGER - IMPORTANT FOR MULTIPLE FILES WITH SAME TITLE
# ============================================================================

class QualityMerger:
    """Merge multiple qualities for same title"""
    
    @staticmethod
    def merge_quality_options(quality_options_list):
        """Merge quality options from multiple files"""
        if not quality_options_list:
            return {}
        
        merged = {}
        
        for quality_option in quality_options_list:
            for quality, data in quality_option.items():
                if quality not in merged:
                    merged[quality] = data
                else:
                    # Keep the one with larger file size (better quality)
                    if data.get('file_size', 0) > merged[quality].get('file_size', 0):
                        merged[quality] = data
        
        return merged
    
    @staticmethod
    def create_combined_quality_options(file_list):
        """Create combined quality options from multiple files"""
        if not file_list:
            return {}
        
        all_quality_options = []
        
        for file_data in file_list:
            quality = file_data.get('quality', '480p')
            quality_options = {
                quality: {
                    'quality': quality,
                    'file_size': file_data.get('file_size', 0),
                    'message_id': file_data.get('message_id'),
                    'file_id': file_data.get('file_id'),
                    'telegram_file_id': file_data.get('telegram_file_id'),
                    'file_name': file_data.get('file_name', ''),
                    'is_video_file': file_data.get('is_video_file', False),
                    'real_message_id': file_data.get('real_message_id')
                }
            }
            all_quality_options.append(quality_options)
        
        return QualityMerger.merge_quality_options(all_quality_options)

# ============================================================================
# ✅ VIDEO THUMBNAIL EXTRACTOR
# ============================================================================

class VideoThumbnailExtractor:
    """Extract thumbnails from video files"""
    
    def __init__(self):
        self.extraction_lock = asyncio.Lock()
    
    async def extract_thumbnail(self, channel_id: int, message_id: int) -> Optional[str]:
        """
        Extract thumbnail from video file
        Returns base64 data URL or None
        """
        try:
            # Use bot handler to extract thumbnail
            if bot_handler and bot_handler.initialized:
                thumbnail_url = await bot_handler.extract_thumbnail(channel_id, message_id)
                if thumbnail_url:
                    logger.debug(f"✅ Thumbnail extracted via bot handler: {channel_id}/{message_id}")
                    return thumbnail_url
            
            # Fallback to Bot session if available
            if Bot is not None and bot_session_ready:
                try:
                    message = await Bot.get_messages(channel_id, message_id)
                    if not message:
                        return None
                    
                    thumbnail_data = None
                    
                    if message.video:
                        if hasattr(message.video, 'thumbnail') and message.video.thumbnail:
                            thumbnail_file_id = message.video.thumbnail.file_id
                            download_path = await Bot.download_media(thumbnail_file_id, in_memory=True)
                            
                            if download_path:
                                if isinstance(download_path, bytes):
                                    thumbnail_data = download_path
                                else:
                                    with open(download_path, 'rb') as f:
                                        thumbnail_data = f.read()
                    
                    elif message.document and is_video_file(message.document.file_name or ''):
                        if hasattr(message.document, 'thumbnail') and message.document.thumbnail:
                            thumbnail_file_id = message.document.thumbnail.file_id
                            download_path = await Bot.download_media(thumbnail_file_id, in_memory=True)
                            
                            if download_path:
                                if isinstance(download_path, bytes):
                                    thumbnail_data = download_path
                                else:
                                    with open(download_path, 'rb') as f:
                                        thumbnail_data = f.read()
                    
                    if thumbnail_data:
                        base64_data = base64.b64encode(thumbnail_data).decode('utf-8')
                        return f"data:image/jpeg;base64,{base64_data}"
                    
                except Exception as e:
                    logger.error(f"❌ Bot session thumbnail extraction error: {e}")
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Thumbnail extraction failed: {e}")
            return None

thumbnail_extractor = VideoThumbnailExtractor()

# ============================================================================
# ✅ ENHANCED SEARCH FUNCTION - COMPLETELY FIXED WITH PROPER MERGING
# ============================================================================

def channel_name_cached(cid):
    return f"Channel {cid}"

@performance_monitor.measure("enhanced_search_fixed_complete")
@async_cache_with_ttl(maxsize=500, ttl=Config.SEARCH_CACHE_TTL)
async def search_movies_enhanced_fixed(query, limit=15, page=1):
    """COMPLETELY FIXED: Properly merge posts and files with quality merging"""
    offset = (page - 1) * limit
    
    # Try cache first
    cache_key = f"search_fixed_complete:{query}:{page}:{limit}"
    if cache_manager is not None and cache_manager.redis_enabled:
        cached_data = await cache_manager.get(cache_key)
        if cached_data:
            logger.info(f"✅ Cache HIT for: {query}")
            return cached_data
    
    logger.info(f"🔍 ENHANCED SEARCH (COMPLETE MERGE) for: {query}")
    
    query_lower = query.lower()
    
    # Main dictionary to hold ALL entries by normalized_title
    all_entries = {}  # normalized_title -> {'posts': [], 'files': []}
    
    # ============================================================================
    # ✅ PHASE 1: COLLECT ALL POSTS FROM TEXT CHANNELS
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
                                'quality_options': {},
                                'is_video_file': False,
                                'year': year,
                                'search_score': 3,
                                'result_type': 'post_only',
                                'thumbnail_url': None,
                                'has_thumbnail': False,
                                'poster_url': None,
                                'poster_source': None,
                                'bot_username': Config.BOT_USERNAME
                            }
                            
                            if norm_title not in all_entries:
                                all_entries[norm_title] = {'posts': [], 'files': []}
                            
                            all_entries[norm_title]['posts'].append(post_data)
                            
            except Exception as e:
                logger.error(f"Text search error in {channel_id}: {e}")
            return True
        
        # Search all text channels
        tasks = [search_text_channel_posts(channel_id) for channel_id in Config.TEXT_CHANNEL_IDS]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info(f"📝 Found posts for {len(all_entries)} titles")
    
    # ============================================================================
    # ✅ PHASE 2: COLLECT ALL FILES FROM FILE CHANNEL
    # ============================================================================
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
                    quality = doc.get('quality', '480p')
                    if not quality or quality == 'unknown':
                        quality = detect_quality_enhanced(doc.get('file_name', ''))
                    
                    # Get thumbnail URL
                    thumbnail_url = doc.get('thumbnail_url')
                    
                    # Get REAL message ID
                    real_msg_id = doc.get('real_message_id') or doc.get('message_id')
                    
                    # Extract year
                    year = doc.get('year', '')
                    
                    # Create file data
                    file_data = {
                        'title': title,
                        'original_title': title,
                        'normalized_title': norm_title,
                        'quality': quality,
                        'file_size': doc.get('file_size', 0),
                        'message_id': real_msg_id,
                        'real_message_id': real_msg_id,
                        'file_id': doc.get('file_id'),
                        'telegram_file_id': doc.get('telegram_file_id'),
                        'file_name': doc.get('file_name', ''),
                        'is_video_file': doc.get('is_video_file', False),
                        'caption': doc.get('caption', ''),
                        'thumbnail_url': thumbnail_url,
                        'channel_id': doc.get('channel_id'),
                        'channel_name': channel_name_cached(doc.get('channel_id')),
                        'year': year,
                        'date': doc['date'].isoformat() if isinstance(doc['date'], datetime) else doc['date'],
                        'has_file': True,
                        'has_post': bool(doc.get('caption')),
                        'post_content': doc.get('caption', ''),
                        'content': format_post(doc.get('caption', ''), max_length=300),
                        'has_thumbnail': thumbnail_url is not None,
                        'search_score': 2,
                        'result_type': 'file_only',
                        'poster_url': None,
                        'poster_source': None,
                        'bot_username': Config.BOT_USERNAME
                    }
                    
                    if norm_title not in all_entries:
                        all_entries[norm_title] = {'posts': [], 'files': []}
                    
                    all_entries[norm_title]['files'].append(file_data)
                    
                except Exception as e:
                    logger.error(f"File processing error: {e}")
                    continue
            
            logger.info(f"📁 Found files for {len([k for k, v in all_entries.items() if v['files']])} titles")
            
        except Exception as e:
            logger.error(f"❌ File search error: {e}")
    
    # ============================================================================
    # ✅ PHASE 3: MERGE POSTS AND FILES INTO SINGLE RESULTS
    # ============================================================================
    logger.info(f"🔄 Merging posts and files into single results...")
    
    final_results = []
    
    for norm_title, data in all_entries.items():
        posts = data.get('posts', [])
        files = data.get('files', [])
        
        if posts and files:
            # ✅ TYPE 1: POST + FILE (COMBINED) - MOST IMPORTANT
            # Use the first post as base
            base_post = posts[0]
            
            # Merge all files quality options
            combined_quality_options = QualityMerger.create_combined_quality_options(files)
            
            # Get best thumbnail from files
            best_thumbnail = None
            for file_data in files:
                if file_data.get('thumbnail_url'):
                    best_thumbnail = file_data.get('thumbnail_url')
                    break
            
            # Create merged result
            merged_result = {
                'title': base_post.get('title', ''),
                'original_title': base_post.get('original_title', ''),
                'normalized_title': norm_title,
                'content': base_post.get('content', ''),
                'post_content': base_post.get('post_content', ''),
                'channel': base_post.get('channel', ''),
                'channel_id': base_post.get('channel_id'),
                'message_id': base_post.get('message_id'),
                'date': base_post.get('date'),
                'is_new': base_post.get('is_new', False),
                'has_file': True,
                'has_post': True,
                'quality_options': combined_quality_options,
                'is_video_file': any(f.get('is_video_file', False) for f in files),
                'year': base_post.get('year', '') or (files[0].get('year', '') if files else ''),
                'search_score': 5,  # Highest score for combined
                'result_type': 'post_and_file',
                'thumbnail_url': best_thumbnail or base_post.get('thumbnail_url'),
                'has_thumbnail': bool(best_thumbnail or base_post.get('thumbnail_url')),
                'poster_url': None,
                'poster_source': None,
                'real_message_id': files[0].get('real_message_id') if files else None,
                'bot_username': Config.BOT_USERNAME,
                'quality_count': len(combined_quality_options),
                'all_qualities': list(combined_quality_options.keys()),
                'combined': True
            }
            
            final_results.append(merged_result)
            logger.debug(f"✅ Merged post+file: {norm_title} ({len(files)} files)")
            
        elif posts and not files:
            # ✅ TYPE 2: POST ONLY (NO FILE)
            for post in posts:
                post_only_result = post.copy()
                post_only_result.update({
                    'has_file': False,
                    'result_type': 'post_only',
                    'search_score': 3,
                    'combined': False
                })
                final_results.append(post_only_result)
            
        elif not posts and files:
            # ✅ TYPE 3: FILE ONLY (NO POST)
            # Combine all files for this title
            combined_quality_options = QualityMerger.create_combined_quality_options(files)
            
            # Use first file as base
            base_file = files[0]
            
            # Get best thumbnail
            best_thumbnail = None
            for file_data in files:
                if file_data.get('thumbnail_url'):
                    best_thumbnail = file_data.get('thumbnail_url')
                    break
            
            file_only_result = {
                'title': base_file.get('title', ''),
                'original_title': base_file.get('original_title', ''),
                'normalized_title': norm_title,
                'content': base_file.get('content', ''),
                'post_content': base_file.get('post_content', ''),
                'channel': base_file.get('channel_name', ''),
                'channel_id': base_file.get('channel_id'),
                'message_id': base_file.get('message_id'),
                'date': base_file.get('date'),
                'is_new': is_new(base_file.get('date')) if base_file.get('date') else False,
                'has_file': True,
                'has_post': bool(base_file.get('post_content')),
                'quality_options': combined_quality_options,
                'is_video_file': any(f.get('is_video_file', False) for f in files),
                'year': base_file.get('year', ''),
                'search_score': 2,
                'result_type': 'file_only',
                'thumbnail_url': best_thumbnail,
                'has_thumbnail': bool(best_thumbnail),
                'poster_url': None,
                'poster_source': None,
                'real_message_id': base_file.get('real_message_id'),
                'bot_username': Config.BOT_USERNAME,
                'quality_count': len(combined_quality_options),
                'all_qualities': list(combined_quality_options.keys()),
                'combined': False
            }
            
            final_results.append(file_only_result)
            logger.debug(f"✅ File-only: {norm_title} ({len(files)} files)")
    
    # ============================================================================
    # ✅ PHASE 4: CHECK IF WE HAVE RESULTS
    # ============================================================================
    if not final_results:
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
                'search_logic': 'enhanced_fixed_complete'
            },
            'bot_username': Config.BOT_USERNAME
        }
        
        # Cache empty result
        if cache_manager is not None:
            await cache_manager.set(cache_key, result_data, expire_seconds=Config.SEARCH_CACHE_TTL)
        
        return result_data
    
    # ============================================================================
    # ✅ PHASE 5: FETCH POSTERS FOR ALL RESULTS
    # ============================================================================
    logger.info(f"🎬 Fetching posters for {len(final_results)} results...")
    
    # Prepare movies for poster fetching
    movies_for_posters = []
    for result in final_results:
        movie_data = {
            'title': result.get('title', ''),
            'year': result.get('year', ''),
            'original_title': result.get('original_title', ''),
            'has_thumbnail': result.get('has_thumbnail', False),
            'thumbnail_url': result.get('thumbnail_url'),
            'result_type': result.get('result_type', 'unknown')
        }
        
        movies_for_posters.append(movie_data)
    
    if movies_for_posters:
        # Fetch posters in batch
        movies_with_posters = await get_posters_for_movies_batch(movies_for_posters)
        
        # Update results with poster data
        for i, result in enumerate(final_results):
            title = result.get('title', '')
            
            # Find matching poster data
            poster_data = None
            for movie in movies_with_posters:
                if movie.get('title') == title:
                    poster_data = movie
                    break
            
            if poster_data:
                # Update with poster info
                result.update({
                    'poster_url': poster_data.get('poster_url', Config.FALLBACK_POSTER),
                    'poster_source': poster_data.get('poster_source', 'fallback'),
                    'poster_rating': poster_data.get('poster_rating', '0.0'),
                    'has_poster': True,
                    'thumbnail': result.get('thumbnail_url') or poster_data.get('poster_url'),
                    'thumbnail_source': result.get('thumbnail_source') or poster_data.get('poster_source', 'fallback'),
                    'has_thumbnail': True
                })
            else:
                # Fallback for movies without poster
                result.update({
                    'poster_url': Config.FALLBACK_POSTER,
                    'poster_source': 'fallback',
                    'poster_rating': '0.0',
                    'has_poster': True,
                    'thumbnail': result.get('thumbnail_url') or Config.FALLBACK_POSTER,
                    'thumbnail_source': result.get('thumbnail_source') or 'fallback',
                    'has_thumbnail': True
                })
    
    # ============================================================================
    # ✅ PHASE 6: SORT RESULTS
    # ============================================================================
    # Sort by: combined first, then posts, then files, then score, then date
    final_results.sort(key=lambda x: (
        x.get('result_type') == 'post_and_file',  # Combined first
        x.get('result_type') == 'post_only',      # Posts second
        x.get('search_score', 0),                 # Higher score
        x.get('is_new', False),                   # New first
        x.get('date', '') if isinstance(x.get('date'), str) else ''
    ), reverse=True)
    
    # ============================================================================
    # ✅ PHASE 7: PAGINATION
    # ============================================================================
    total = len(final_results)
    start_idx = offset
    end_idx = offset + limit
    paginated = final_results[start_idx:end_idx]
    
    # Statistics
    post_count = sum(1 for r in final_results if r.get('result_type') == 'post_only')
    file_count = sum(1 for r in final_results if r.get('result_type') == 'file_only')
    combined_count = sum(1 for r in final_results if r.get('result_type') == 'post_and_file')
    
    stats = {
        'total': total,
        'post_only': post_count,
        'file_only': file_count,
        'post_and_file': combined_count
    }
    
    # Log results
    logger.info(f"📊 FINAL RESULTS:")
    logger.info(f"   • Total: {total}")
    logger.info(f"   • Post+File: {combined_count}")
    logger.info(f"   • Post-only: {post_count}")
    logger.info(f"   • File-only: {file_count}")
    
    # Show sample
    for i, result in enumerate(paginated[:3]):
        result_type = result.get('result_type', 'unknown')
        title = result.get('title', '')[:40]
        has_file = result.get('has_file', False)
        has_post = result.get('has_post', False)
        quality_count = len(result.get('quality_options', {}))
        
        logger.info(f"   📋 {i+1}. {result_type}: {title}... | File: {has_file} | Post: {has_post} | Qualities: {quality_count}")
    
    # ============================================================================
    # ✅ PHASE 8: FINAL DATA STRUCTURE
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
            'search_logic': 'enhanced_fixed_complete'
        },
        'bot_username': Config.BOT_USERNAME
    }
    
    # Cache results
    if cache_manager is not None:
        await cache_manager.set(cache_key, result_data, expire_seconds=Config.SEARCH_CACHE_TTL)
    
    logger.info(f"✅ Enhanced search complete: {len(paginated)} results (page {page})")
    
    return result_data

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
                            await files_col.delete_one({"_id": item['db_id']})
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
# ✅ OPTIMIZED FILE CHANNEL INDEXING MANAGER
# ============================================================================

class OptimizedFileIndexingManager:
    """Optimized file channel indexing manager - ULTRA EFFICIENT"""
    
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
                success = await index_single_file_fast(msg)
                
                if success:
                    batch_stats['indexed'] += 1
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
# ✅ OPTIMIZED FILE INDEXING FUNCTIONS - ULTRA FAST
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
    """Fast file indexing - minimal checks"""
    try:
        if files_col is None:
            return False
        
        if not message or (not message.document and not message.video):
            return False
        
        # Extract title quickly
        caption = message.caption if hasattr(message, 'caption') else None
        file_name = None
        
        if message.document:
            file_name = message.document.file_name
        elif message.video:
            file_name = message.video.file_name
        
        title = await extract_title_fast(file_name, caption)
        if not title or title == "Unknown File":
            return False
        
        # Extract thumbnail only for video files
        thumbnail_url = None
        is_video = False
        
        if message.video or (message.document and is_video_file(file_name or '')):
            is_video = True
            # Try thumbnail extraction (non-blocking)
            try:
                thumbnail_url = await thumbnail_extractor.extract_thumbnail(
                    Config.FILE_CHANNEL_ID,
                    message.id
                )
            except:
                pass
        
        # Extract quality
        quality = detect_quality_enhanced(file_name or "")
        
        # Create minimal document
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
            'thumbnail_extracted': thumbnail_url is not None,
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
        
        # Insert into MongoDB
        await files_col.insert_one(doc)
        
        # Log success
        logger.info(f"✅ INDEXED FAST: {title[:50]}... (ID: {message.id})")
        
        return True
        
    except Exception as e:
        if "duplicate key error" in str(e).lower():
            return False
        logger.error(f"❌ Fast indexing error: {e}")
        return False

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
        
        logger.info("✅ Created minimal database indexes")
        
    except Exception as e:
        logger.warning(f"⚠️ Index creation error: {e}")

# ============================================================================
# ✅ OPTIMIZED INITIAL INDEXING
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
# ✅ POSTER FETCHING FUNCTIONS
# ============================================================================

async def get_poster_for_movie(title: str, year: str = "", quality: str = "") -> Dict[str, Any]:
    """Get poster for movie"""
    global poster_fetcher
    
    # If poster_fetcher is not available, use fallback
    if poster_fetcher is None:
        return {
            'poster_url': Config.FALLBACK_POSTER,
            'source': 'custom',
            'rating': '0.0',
            'year': year,
            'title': title,
            'quality': quality or 'unknown'
        }
    
    try:
        # Fetch poster with timeout
        poster_task = asyncio.create_task(poster_fetcher.fetch_poster(title))
        
        try:
            poster_data = await asyncio.wait_for(poster_task, timeout=3.0)
            
            if poster_data and poster_data.get('poster_url'):
                logger.debug(f"✅ Poster fetched: {title} - {poster_data['source']}")
                return poster_data
            else:
                raise ValueError("Invalid poster data")
                
        except (asyncio.TimeoutError, ValueError, Exception) as e:
            logger.warning(f"⚠️ Poster fetch timeout/error for {title}: {e}")
            
            if not poster_task.done():
                poster_task.cancel()
            
            # Return fallback
            return {
                'poster_url': Config.FALLBACK_POSTER,
                'source': 'custom',
                'rating': '0.0',
                'year': year,
                'title': title,
                'quality': quality or 'unknown'
            }
            
    except Exception as e:
        logger.error(f"❌ Unexpected error in get_poster_for_movie: {e}")
        return {
            'poster_url': Config.FALLBACK_POSTER,
            'source': 'custom',
            'rating': '0.0',
            'year': year,
            'title': title,
            'quality': quality or 'unknown'
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
        
        logger.info("✅ MongoDB OK")
        return True
        
    except asyncio.TimeoutError:
        logger.error("❌ MongoDB connection timeout")
        return False
    except Exception as e:
        logger.error(f"❌ MongoDB error: {e}")
        return False

# ============================================================================
# ✅ MAIN INITIALIZATION - OPTIMIZED
# ============================================================================

@performance_monitor.measure("system_init")
async def init_system():
    start_time = time.time()
    
    try:
        logger.info("=" * 60)
        logger.info("🚀 SK4FiLM v9.0 - OPTIMIZED INDEXING")
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
        
        # Initialize Telegram Sessions
        if PYROGRAM_AVAILABLE:
            telegram_ok = await init_telegram_sessions()
            if not telegram_ok:
                logger.warning("⚠️ Telegram sessions failed")
        
        # Start OPTIMIZED indexing
        if user_session_ready and files_col is not None:
            logger.info("🔄 Starting OPTIMIZED file channel indexing...")
            asyncio.create_task(initial_indexing_optimized())
        
        init_time = time.time() - start_time
        logger.info(f"⚡ SK4FiLM Started in {init_time:.2f}s")
        logger.info("=" * 60)
        
        logger.info("🔧 OPTIMIZED FEATURES:")
        logger.info(f"   • Zero DB Preload: ✅ ENABLED")
        logger.info(f"   • Zero Duplicate Checks: ✅ ENABLED")
        logger.info(f"   • Zero Wasted Telegram Calls: ✅ ENABLED")
        logger.info(f"   • Only New Messages: ✅ ENABLED")
        logger.info(f"   • Files-Only Fetch: ✅ ENABLED")
        logger.info(f"   • Auto-Delete DB Entries: ✅ ENABLED")
        logger.info(f"   • Real Message IDs: ✅ ENABLED")
        logger.info(f"   • Telegram Bot: {'✅ RUNNING' if telegram_bot else '❌ NOT RUNNING'}")
        
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
        
        # Fetch posters for all movies in batch
        if movies:
            movies_with_posters = await get_posters_for_movies_batch(movies)
            logger.info(f"✅ Fetched {len(movies_with_posters)} home movies with posters")
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
    
    # Get Telegram bot status
    bot_running = telegram_bot is not None and hasattr(telegram_bot, 'bot_started') and telegram_bot.bot_started
    
    return jsonify({
        'status': 'healthy',
        'service': 'SK4FiLM v9.0 - OPTIMIZED INDEXING',
        'optimized_features': {
            'zero_db_preload': True,
            'zero_duplicate_checks': True,
            'zero_wasted_telegram_calls': True,
            'only_new_messages': True,
            'files_only_fetch': True,
            'auto_delete_deleted': True,
            'real_message_ids': True,
            'post_file_merge': True,
            'file_only_with_poster': True
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
    
    return jsonify({
        'status': 'ok',
        'optimized': True,
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
        'features': {
            'post_file_merge': True,
            'file_only_poster': True
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
                'poster_priority': True,
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
            'optimized': True,
            'performance': perf_stats,
            'poster_fetcher': poster_stats,
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
            'optimization_features': {
                'zero_db_preload': True,
                'zero_duplicate_checks': True,
                'only_new_messages': True,
                'auto_delete_deleted': True,
                'post_file_merge': True,
                'file_only_poster': True
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
            'optimized': True,
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
            'optimized': True,
            'indexing': indexing_status,
            'database_files': total_files,
            'optimization_features': {
                'zero_db_preload': True,
                'zero_duplicate_checks': True,
                'only_new_messages': True,
                'auto_delete_deleted': True,
                'post_file_merge': True
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

@app.route('/api/admin/db-stats', methods=['GET'])
async def api_admin_db_stats():
    """Get detailed database stats"""
    try:
        auth_token = request.headers.get('X-Admin-Token')
        if not auth_token or auth_token != os.environ.get('ADMIN_TOKEN', 'sk4film_admin_123'):
            return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401
        
        if files_col is None:
            return jsonify({'status': 'error', 'message': 'Database not connected'})
        
        # Get total count
        total = await files_col.count_documents({})
        
        # Get sample documents with REAL MESSAGE IDS
        sample = await files_col.find({}, {
            'title': 1, 
            'message_id': 1, 
            'real_message_id': 1,
            'quality': 1, 
            '_id': 0
        }).limit(5).to_list(length=5)
        
        # Get quality distribution
        pipeline = [
            {"$group": {"_id": "$quality", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        quality_dist = await files_col.aggregate(pipeline).to_list(length=10)
        
        # Get recent files
        recent = await files_col.find({}, {
            'title': 1, 
            'date': 1, 
            'real_message_id': 1,
            '_id': 0
        }).sort('date', -1).limit(5).to_list(length=5)
        
        return jsonify({
            'status': 'success',
            'optimized': True,
            'total_files': total,
            'sample_files': sample,
            'quality_distribution': quality_dist,
            'recent_files': recent
        })
        
    except Exception as e:
        logger.error(f"❌ DB stats error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/admin/bot-status', methods=['GET'])
async def api_admin_bot_status():
    """Get Telegram bot status"""
    try:
        auth_token = request.headers.get('X-Admin-Token')
        if not auth_token or auth_token != os.environ.get('ADMIN_TOKEN', 'sk4film_admin_123'):
            return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401
        
        bot_running = telegram_bot is not None and hasattr(telegram_bot, 'bot_started') and telegram_bot.bot_started
        
        return jsonify({
            'status': 'success',
            'optimized': True,
            'telegram_bot': {
                'running': bot_running,
                'initialized': telegram_bot is not None,
                'started': telegram_bot.bot_started if telegram_bot and hasattr(telegram_bot, 'bot_started') else False
            },
            'config': {
                'bot_token_configured': bool(Config.BOT_TOKEN),
                'api_id_configured': bool(Config.API_ID),
                'api_hash_configured': bool(Config.API_HASH),
                'admin_ids': Config.ADMIN_IDS
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Bot status error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# ============================================================================
# ✅ STARTUP AND SHUTDOWN - FIXED
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
    logger.info("🎯 FEATURES: OPTIMIZED INDEXING")
    logger.info(f"   • Zero DB Preload: ✅ ENABLED")
    logger.info(f"   • Zero Duplicate Checks: ✅ ENABLED")  
    logger.info(f"   • Zero Wasted Telegram Calls: ✅ ENABLED")
    logger.info(f"   • Only New Messages: ✅ ENABLED")
    logger.info(f"   • Files-Only Fetch: ✅ ENABLED")
    logger.info(f"   • Auto-Delete DB Entries: ✅ ENABLED")
    logger.info(f"   • Post+File Merge: ✅ ENABLED")
    logger.info(f"   • File-Only with Poster: ✅ ENABLED")
    logger.info(f"   • File Channel ID: {Config.FILE_CHANNEL_ID}")
    logger.info(f"   • Max Messages: {'Unlimited' if Config.MAX_INDEX_LIMIT == 0 else Config.MAX_INDEX_LIMIT}")
    logger.info(f"   • Telegram Bot: ✅ ENABLED")
    
    try:
        asyncio.run(serve(app, config))
    except KeyboardInterrupt:
        logger.info("👋 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
