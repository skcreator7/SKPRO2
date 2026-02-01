# ============================================================================
# 🚀 SK4FiLM v9.0 - POSTS FIRST, FILE-ONLY WITHOUT POST CONTENT
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
    
    # Quality Settings
    QUALITY_PRIORITY = ['2160p', '1080p', '720p', '480p', '360p']
    HEVC_VARIANTS = ['720p HEVC', '1080p HEVC', '2160p HEVC']
    
    # Fallback Poster
    FALLBACK_POSTER = "https://iili.io/fAeIwv9.th.png"
    
    # Thumbnail Settings
    THUMBNAIL_EXTRACT_TIMEOUT = 10
    THUMBNAIL_CACHE_DURATION = 24 * 60 * 60
    
    # 🔥 FILE CHANNEL INDEXING SETTINGS
    AUTO_INDEX_INTERVAL = int(os.environ.get("AUTO_INDEX_INTERVAL", "120"))  # 2 minutes
    BATCH_INDEX_SIZE = int(os.environ.get("BATCH_INDEX_SIZE", "500"))  # Large batches
    MAX_INDEX_LIMIT = int(os.environ.get("MAX_INDEX_LIMIT", "0"))  # 0 = Unlimited
    INDEX_ALL_HISTORY = os.environ.get("INDEX_ALL_HISTORY", "true").lower() == "true"  # ✅ All history
    INSTANT_AUTO_INDEX = os.environ.get("INSTANT_AUTO_INDEX", "true").lower() == "true"
    
    # 🔥 SEARCH SETTINGS
    SEARCH_MIN_QUERY_LENGTH = 2
    SEARCH_RESULTS_PER_PAGE = 12
    SEARCH_CACHE_TTL = 600  # 10 minutes
    
    # 🔥 THUMBNAIL PRIORITY SETTINGS
    THUMBNAIL_PRIORITY = ['extracted', 'tmdb', 'omdb', 'fallback']

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
    response.headers['X-SK4FiLM-Version'] = '9.0-POSTS-FIRST'
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
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Bot handler initialization error: {e}")
            return False
    
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
                    download_path = await self.bot.download_media(thumbnail_file_id, in_memory=True)
                    
                    if download_path:
                        if isinstance(download_path, bytes):
                            thumbnail_data = download_path
                        else:
                            with open(download_path, 'rb') as f:
                                thumbnail_data = f.read()
            
            elif message.document and is_video_file(message.document.file_name or ''):
                # Video document - try to get thumbnail
                if hasattr(message.document, 'thumbnail') and message.document.thumbnail:
                    thumbnail_file_id = message.document.thumbnail.file_id
                    download_path = await self.bot.download_media(thumbnail_file_id, in_memory=True)
                    
                    if download_path:
                        if isinstance(download_path, bytes):
                            thumbnail_data = download_path
                        else:
                            with open(download_path, 'rb') as f:
                                thumbnail_data = f.read()
            
            if thumbnail_data:
                # Convert to base64
                base64_data = base64.b64encode(thumbnail_data).decode('utf-8')
                return f"data:image/jpeg;base64,{base64_data}"
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Extract thumbnail error: {e}")
            return None
    
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
# ✅ THUMBNAIL PRIORITY SYSTEM
# ============================================================================

class ThumbnailPrioritySystem:
    """Thumbnail priority system with fallback"""
    
    def __init__(self):
        self.thumbnail_cache = {}
        
    async def get_thumbnail_for_movie(self, movie_data, title, year=""):
        """
        Get thumbnail with priority:
        1. Extracted thumbnail (from Telegram)
        2. TMDB poster
        3. OMDB poster
        4. Fallback image
        """
        thumbnail_data = {
            'thumbnail': None,
            'thumbnail_source': None,
            'has_thumbnail': False
        }
        
        # Check 1: Extracted thumbnail from movie_data
        if movie_data.get('thumbnail_url'):
            thumbnail_data.update({
                'thumbnail': movie_data['thumbnail_url'],
                'thumbnail_source': 'extracted',
                'has_thumbnail': True
            })
            return thumbnail_data
        
        # Check 2: Try TMDB
        try:
            tmdb_result = await self._try_tmdb_poster(title, year)
            if tmdb_result:
                thumbnail_data.update({
                    'thumbnail': tmdb_result['poster_url'],
                    'thumbnail_source': 'tmdb',
                    'has_thumbnail': True,
                    'poster_url': tmdb_result['poster_url'],
                    'poster_source': 'tmdb',
                    'poster_rating': tmdb_result.get('rating', '0.0')
                })
                return thumbnail_data
        except Exception as e:
            logger.debug(f"TMDB fetch failed for {title}: {e}")
        
        # Check 3: Try OMDB
        try:
            omdb_result = await self._try_omdb_poster(title, year)
            if omdb_result:
                thumbnail_data.update({
                    'thumbnail': omdb_result['poster_url'],
                    'thumbnail_source': 'omdb',
                    'has_thumbnail': True,
                    'poster_url': omdb_result['poster_url'],
                    'poster_source': 'omdb',
                    'poster_rating': omdb_result.get('rating', '0.0')
                })
                return thumbnail_data
        except Exception as e:
            logger.debug(f"OMDB fetch failed for {title}: {e}")
        
        # Final fallback
        thumbnail_data.update({
            'thumbnail': Config.FALLBACK_POSTER,
            'thumbnail_source': 'fallback',
            'has_thumbnail': True,
            'poster_url': Config.FALLBACK_POSTER,
            'poster_source': 'fallback',
            'poster_rating': '0.0'
        })
        
        return thumbnail_data
    
    async def _try_tmdb_poster(self, title, year=""):
        """Try to get poster from TMDB"""
        try:
            # Clean title for TMDB search
            clean_title = self._clean_title_for_search(title)
            
            # TMDB API call
            tmdb_url = f"https://api.themoviedb.org/3/search/movie"
            params = {
                'api_key': Config.TMDB_API_KEY,
                'query': clean_title,
                'language': 'en-US',
                'page': 1
            }
            
            if year and year.isdigit():
                params['year'] = year
            
            async with aiohttp.ClientSession() as session:
                async with session.get(tmdb_url, params=params, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('results') and len(data['results']) > 0:
                            movie = data['results'][0]
                            if movie.get('poster_path'):
                                poster_url = f"https://image.tmdb.org/t/p/w500{movie['poster_path']}"
                                return {
                                    'poster_url': poster_url,
                                    'rating': str(movie.get('vote_average', '0.0')),
                                    'title': movie.get('title', title),
                                    'year': str(movie.get('release_date', '')[:4]) if movie.get('release_date') else year
                                }
        except Exception as e:
            logger.debug(f"TMDB API error: {e}")
        
        return None
    
    async def _try_omdb_poster(self, title, year=""):
        """Try to get poster from OMDB"""
        try:
            # Clean title for OMDB search
            clean_title = self._clean_title_for_search(title)
            
            # OMDB API call
            omdb_url = f"http://www.omdbapi.com/"
            params = {
                'apikey': Config.OMDB_API_KEY,
                't': clean_title,
                'type': 'movie',
                'plot': 'short'
            }
            
            if year:
                params['y'] = year
            
            async with aiohttp.ClientSession() as session:
                async with session.get(omdb_url, params=params, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('Poster') and data['Poster'] != 'N/A':
                            return {
                                'poster_url': data['Poster'],
                                'rating': data.get('imdbRating', '0.0'),
                                'title': data.get('Title', title),
                                'year': data.get('Year', year)
                            }
        except Exception as e:
            logger.debug(f"OMDB API error: {e}")
        
        return None
    
    def _clean_title_for_search(self, title):
        """Clean title for API search"""
        if not title:
            return ""
        
        # Remove year in parentheses
        title = re.sub(r'\s*\(\d{4}\)', '', title)
        
        # Remove quality indicators
        patterns_to_remove = [
            r'\b(480p|720p|1080p|2160p|4k|uhd|hd|hevc|x265|x264)\b',
            r'\b(web|webrip|webdl|bluray|brrip|dvdrip|hdtv)\b',
            r'\b(hindi|english|tamil|telugu|malayalam)\b',
            r'[._]'
        ]
        
        for pattern in patterns_to_remove:
            title = re.sub(pattern, ' ', title, flags=re.IGNORECASE)
        
        # Clean up
        title = re.sub(r'\s+', ' ', title)
        return title.strip()

thumbnail_system = ThumbnailPrioritySystem()

# ============================================================================
# ✅ FILE INDEXING FUNCTIONS
# ============================================================================

async def generate_file_hash(message):
    """Generate unique hash for file to detect duplicates"""
    try:
        hash_parts = []
        
        if message.document:
            file_attrs = message.document
            # Use file_id for hash
            hash_parts.append(f"doc_{file_attrs.file_id}")
            if file_attrs.file_name:
                name_hash = hashlib.md5(file_attrs.file_name.encode()).hexdigest()[:16]
                hash_parts.append(f"name_{name_hash}")
            if file_attrs.file_size:
                hash_parts.append(f"size_{file_attrs.file_size}")
        elif message.video:
            file_attrs = message.video
            hash_parts.append(f"vid_{file_attrs.file_id}")
            if file_attrs.file_name:
                name_hash = hashlib.md5(file_attrs.file_name.encode()).hexdigest()[:16]
                hash_parts.append(f"name_{name_hash}")
            if file_attrs.file_size:
                hash_parts.append(f"size_{file_attrs.file_size}")
            if hasattr(file_attrs, 'duration'):
                hash_parts.append(f"dur_{file_attrs.duration}")
        else:
            return None
        
        # Add caption hash only if exists
        if message.caption:
            caption_hash = hashlib.md5(message.caption.encode()).hexdigest()[:12]
            hash_parts.append(f"cap_{caption_hash}")
        
        # Final hash
        final_hash = hashlib.sha256("_".join(hash_parts).encode()).hexdigest()
        return final_hash
        
    except Exception as e:
        logger.debug(f"Hash generation error: {e}")
        return None

async def extract_title_improved(filename, caption):
    """Improved title extraction"""
    # Try filename first
    if filename:
        # Clean filename
        name = os.path.splitext(filename)[0]
        
        # Remove common patterns
        patterns_to_remove = [
            r'\b(480p|720p|1080p|2160p|4k|uhd|hd|hevc|x265|x264|h264|h265)\b',
            r'\b(web|webrip|webdl|bluray|brrip|dvdrip|hdtv)\b',
            r'\b(hindi|english|tamil|telugu|malayalam|bengali)\b',
            r'\b(dual|multi)\b',
            r'\b(ac3|aac|dd5\.1|dts)\b',
            r'\b(\d{3,4}p)\b',
            r'[._]'
        ]
        
        for pattern in patterns_to_remove:
            name = re.sub(pattern, ' ', name, flags=re.IGNORECASE)
        
        # Clean up
        name = re.sub(r'\s+', ' ', name)
        name = name.strip()
        
        # Extract year if present
        year_match = re.search(r'\b(19|20)\d{2}\b', name)
        if year_match:
            year = year_match.group()
            # Remove year from name
            name = re.sub(r'\s*\b(19|20)\d{2}\b', '', name)
            name = f"{name.strip()} ({year})"
        
        if name and len(name) > 3:
            return name
    
    # Try caption
    if caption:
        # Extract first meaningful line
        lines = caption.split('\n')
        for line in lines:
            line = line.strip()
            if len(line) > 10 and not line.startswith('http'):
                # Clean the line
                line = re.sub(r'📥.*', '', line)  # Remove download indicators
                line = re.sub(r'🎬.*', '', line)  # Remove movie indicators
                line = re.sub(r'⚡.*', '', line)  # Remove speed indicators
                line = re.sub(r'✅.*', '', line)  # Remove check indicators
                line = re.sub(r'[⭐🌟]+', '', line)  # Remove stars
                line = line.strip()
                
                if line and len(line) > 5:
                    return line[:200]  # Limit length
    
    # Fallback to filename
    if filename:
        return os.path.splitext(filename)[0][:100]
    
    return "Unknown File"

async def index_single_file_smart(message):
    """Index single file with improved logic and REAL MESSAGE IDS"""
    try:
        if files_col is None:
            logger.error("❌ Database not ready for indexing")
            return False
        
        if not message or (not message.document and not message.video):
            logger.debug(f"❌ Not a file message: {message.id}")
            return False
        
        # Extract title
        caption = message.caption if hasattr(message, 'caption') else None
        file_name = None
        
        if message.document:
            file_name = message.document.file_name
        elif message.video:
            file_name = message.video.file_name
        
        title = await extract_title_improved(file_name, caption)
        if not title or title == "Unknown File":
            logger.debug(f"📝 Skipping - No valid title: {message.id}")
            return False
        
        normalized_title = normalize_title(title)
        
        # Check if already exists by message ID
        existing_by_id = await files_col.find_one({
            'channel_id': Config.FILE_CHANNEL_ID,
            'message_id': message.id
        }, {'_id': 1})
        
        if existing_by_id:
            logger.debug(f"📝 Already indexed: {title[:50]}... (ID: {message.id})")
            return False
        
        # Extract thumbnail if video file
        thumbnail_url = None
        is_video = False
        
        if message.video or (message.document and is_video_file(file_name or '')):
            is_video = True
            # Try to extract thumbnail using bot handler
            try:
                if bot_handler and bot_handler.initialized:
                    thumbnail_url = await bot_handler.extract_thumbnail(
                        Config.FILE_CHANNEL_ID,
                        message.id
                    )
            except Exception as e:
                logger.debug(f"⚠️ Thumbnail extraction failed: {e}")
        
        # Extract year from title
        year_match = re.search(r'\b(19|20)\d{2}\b', title)
        year = year_match.group() if year_match else ""
        
        # Extract quality
        quality = detect_quality_enhanced(file_name or "")
        
        # Create document with REAL MESSAGE ID
        doc = {
            'channel_id': Config.FILE_CHANNEL_ID,
            'message_id': message.id,  # 🔥 REAL MESSAGE ID
            'real_message_id': message.id,  # 🔥 Store separately for consistency
            'title': title,
            'normalized_title': normalized_title,
            'date': message.date,
            'indexed_at': datetime.now(),
            'last_checked': datetime.now(),
            'is_video_file': is_video,
            'file_id': None,
            'file_size': 0,
            'thumbnail_url': thumbnail_url,
            'thumbnail_extracted': thumbnail_url is not None,
            'status': 'active',
            'is_duplicate': False,
            'quality': quality,
            'year': year
        }
        
        # Add file-specific data
        if message.document:
            doc.update({
                'file_name': message.document.file_name or '',
                'is_video_file': is_video_file(message.document.file_name or ''),
                'caption': caption or '',
                'mime_type': message.document.mime_type or '',
                'file_id': message.document.file_id,
                'telegram_file_id': message.document.file_id,  # 🔥 Store Telegram file_id
                'file_size': message.document.file_size or 0
            })
        elif message.video:
            doc.update({
                'file_name': message.video.file_name or 'video.mp4',
                'is_video_file': True,
                'caption': caption or '',
                'duration': message.video.duration if hasattr(message.video, 'duration') else 0,
                'width': message.video.width if hasattr(message.video, 'width') else 0,
                'height': message.video.height if hasattr(message.video, 'height') else 0,
                'file_id': message.video.file_id,
                'telegram_file_id': message.video.file_id,  # 🔥 Store Telegram file_id
                'file_size': message.video.file_size or 0
            })
        else:
            return False
        
        # Insert into MongoDB
        try:
            await files_col.insert_one(doc)
            
            # Log success
            file_type = "📹 Video" if doc['is_video_file'] else "📄 File"
            size_str = format_size(doc['file_size']) if doc['file_size'] > 0 else "Unknown"
            
            logger.info(f"✅ INDEXED: {title[:60]}...")
            logger.info(f"   📊 Real Message ID: {message.id} | Size: {size_str} | Quality: {quality}")
            
            return True
            
        except Exception as e:
            if "duplicate key error" in str(e).lower():
                logger.debug(f"📝 Duplicate key error: {message.id}")
                return False
            else:
                logger.error(f"❌ Insert error: {e}")
                return False
        
    except Exception as e:
        logger.error(f"❌ Indexing error for message {message.id}: {e}")
        return False

async def initial_indexing():
    """Initial indexing on startup"""
    if User is None or files_col is None or not user_session_ready:
        logger.warning("⚠️ User session not ready for initial indexing")
        return
    
    logger.info("=" * 60)
    logger.info("🚀 STARTING FILE CHANNEL INDEXING")
    logger.info("=" * 60)
    
    try:
        # Setup indexes
        await setup_database_indexes()
        
    except Exception as e:
        logger.error(f"❌ Initial indexing error: {e}")

async def setup_database_indexes():
    """Setup all required database indexes"""
    if files_col is None:
        return
    
    try:
        # Unique index for channel + message
        await files_col.create_index(
            [("channel_id", 1), ("message_id", 1)],
            unique=True,
            name="channel_message_unique",
            background=True
        )
        
        # Text search index
        await files_col.create_index(
            [("normalized_title", "text"), ("title", "text")],
            name="title_text_search",
            background=True
        )
        
        logger.info("✅ Created database indexes")
        
    except Exception as e:
        logger.warning(f"⚠️ Index creation error: {e}")

# ============================================================================
# ✅ SEARCH FUNCTION - POSTS FIRST, FILE-ONLY WITHOUT POST CONTENT
# ============================================================================

@performance_monitor.measure("multi_channel_search_merged")
@async_cache_with_ttl(maxsize=500, ttl=Config.SEARCH_CACHE_TTL)
async def search_movies_multi_channel_merged(query, limit=15, page=1):
    """POSTS FIRST, File-only results show only file name (no post content)"""
    offset = (page - 1) * limit
    
    # Try cache first
    cache_key = f"search_merged:{query}:{page}:{limit}"
    if cache_manager is not None and cache_manager.redis_enabled:
        cached_data = await cache_manager.get(cache_key)
        if cached_data:
            logger.info(f"✅ Cache HIT for: {query}")
            return cached_data
    
    logger.info(f"🔍 SEARCHING for: {query}")
    
    query_lower = query.lower()
    posts_dict = {}
    files_dict = {}
    
    # ============================================================================
    # ✅ 1. SEARCH TEXT CHANNELS (Posts/Messages) - FIRST PRIORITY
    # ============================================================================
    if user_session_ready and User is not None:
        async def search_text_channel(channel_id):
            channel_posts = {}
            try:
                async for msg in User.search_messages(channel_id, query=query, limit=20):
                    if msg is not None and msg.text and len(msg.text) > 15:
                        title = extract_title_smart(msg.text)
                        if title and (query_lower in title.lower() or query_lower in msg.text.lower()):
                            norm_title = normalize_title(title)
                            if norm_title not in channel_posts:
                                # Get year
                                year_match = re.search(r'\b(19|20)\d{2}\b', title)
                                year = year_match.group() if year_match else ""
                                
                                # Clean title
                                clean_title = re.sub(r'\s+\(\d{4}\)$', '', title)
                                clean_title = re.sub(r'\s+\d{4}$', '', clean_title)
                                
                                # Create post data (WITH POST CONTENT)
                                post_data = {
                                    'title': clean_title,
                                    'original_title': title,
                                    'normalized_title': norm_title,
                                    'content': format_post(msg.text, max_length=1000),
                                    'post_content': msg.text,
                                    'channel_id': channel_id,
                                    'message_id': msg.id,
                                    'date': msg.date.isoformat() if isinstance(msg.date, datetime) else str(msg.date),
                                    'is_new': is_new(msg.date) if msg.date else False,
                                    'has_file': False,
                                    'has_post': True,
                                    'quality_options': {},
                                    'is_video_file': False,
                                    'year': year,
                                    'search_score': 5 if query_lower in title.lower() else 4,  # Higher score for posts
                                    'result_type': 'post',
                                    'has_post_content': True  # ✅ Flag for post content
                                }
                                
                                channel_posts[norm_title] = post_data
            except Exception as e:
                logger.error(f"Text search error in {channel_id}: {e}")
            return channel_posts
        
        # Search text channels
        tasks = [search_text_channel(channel_id) for channel_id in Config.TEXT_CHANNEL_IDS]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, dict):
                posts_dict.update(result)
        
        logger.info(f"📝 Found {len(posts_dict)} posts in text channels")
    
    # ============================================================================
    # ✅ 2. SEARCH FILE CHANNEL DATABASE
    # ============================================================================
    if files_col is not None:
        try:
            logger.info(f"🔍 Searching FILE CHANNEL database for: {query}")
            
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
            ).limit(200)
            
            file_count = 0
            quality_counts = defaultdict(int)
            
            async for doc in cursor:
                file_count += 1
                try:
                    title = doc.get('title', 'Unknown')
                    norm_title = normalize_title(title)
                    
                    # Extract quality info
                    quality_info = extract_quality_info(doc.get('file_name', ''))
                    quality = quality_info['full']
                    base_quality = quality_info['base']
                    
                    # Count quality occurrences
                    quality_counts[quality] += 1
                    
                    # Get thumbnail URL (extracted thumbnail)
                    thumbnail_url = doc.get('thumbnail_url')
                    
                    # Get REAL message ID
                    real_msg_id = doc.get('real_message_id') or doc.get('message_id')
                    
                    # Check if this title already has a post
                    has_post_content = norm_title in posts_dict
                    
                    # For file-only results (no post), show only file name as content
                    file_caption = doc.get('caption', '')
                    if has_post_content:
                        # If post exists, use post content
                        content_to_show = posts_dict[norm_title].get('content', '')
                    else:
                        # File-only: show only file name (no post content)
                        content_to_show = f"📁 File: {doc.get('file_name', '')}"
                    
                    # Create quality option
                    quality_option = {
                        'file_id': f"{doc.get('channel_id', Config.FILE_CHANNEL_ID)}_{real_msg_id}_{quality}",
                        'file_size': doc.get('file_size', 0),
                        'file_name': doc.get('file_name', ''),
                        'is_video': doc.get('is_video_file', False),
                        'channel_id': doc.get('channel_id'),
                        'message_id': real_msg_id,
                        'real_message_id': real_msg_id,
                        'quality': quality,
                        'base_quality': base_quality,
                        'is_hevc': quality_info['is_hevc'],
                        'priority': quality_info['priority'],
                        'thumbnail_url': thumbnail_url,
                        'has_thumbnail': thumbnail_url is not None,
                        'date': doc.get('date'),
                        'telegram_file_id': doc.get('telegram_file_id')
                    }
                    
                    # If this normalized title doesn't exist in files_dict
                    if norm_title not in files_dict:
                        year = doc.get('year', '')
                        
                        # Create file entry
                        files_dict[norm_title] = {
                            'title': title,
                            'original_title': title,
                            'normalized_title': norm_title,
                            'content': content_to_show,  # ✅ File name only for file-only
                            'post_content': file_caption if has_post_content else '',  # Empty for file-only
                            'quality_options': {quality: quality_option},
                            'quality_list': [quality],
                            'date': doc['date'].isoformat() if isinstance(doc['date'], datetime) else doc['date'],
                            'is_new': is_new(doc['date']) if doc.get('date') else False,
                            'is_video_file': doc.get('is_video_file', False),
                            'channel_id': doc.get('channel_id'),
                            'has_file': True,
                            'has_post': has_post_content,  # True only if post exists
                            'file_caption': file_caption,
                            'year': year,
                            'quality': quality,
                            'has_thumbnail': thumbnail_url is not None,
                            'thumbnail_url': thumbnail_url,
                            'real_message_id': real_msg_id,
                            'search_score': 3 if query_lower in title.lower() else 2,  # Lower score than posts
                            'result_type': 'file' if not has_post_content else 'both',
                            'total_files': 1,
                            'file_sizes': [doc.get('file_size', 0)],
                            'thumbnail_source': 'extracted' if thumbnail_url else None,
                            'has_post_content': has_post_content  # ✅ Flag for post content
                        }
                    else:
                        # MERGE: Same title, different file - add quality option
                        existing = files_dict[norm_title]
                        
                        # Only add if this quality doesn't already exist
                        if quality not in existing['quality_options']:
                            existing['quality_options'][quality] = quality_option
                            existing['quality_list'].append(quality)
                            existing['total_files'] += 1
                            existing['file_sizes'].append(doc.get('file_size', 0))
                        
                        # Update thumbnail if we have extracted one
                        if thumbnail_url and not existing.get('has_thumbnail'):
                            existing['thumbnail'] = thumbnail_url
                            existing['thumbnail_url'] = thumbnail_url
                            existing['has_thumbnail'] = True
                            existing['thumbnail_source'] = 'extracted'
                        
                        # Update date to latest
                        new_date = doc.get('date')
                        if new_date and isinstance(new_date, datetime):
                            existing_date = existing.get('date')
                            if isinstance(existing_date, str):
                                try:
                                    existing_date = datetime.fromisoformat(existing_date.replace('Z', '+00:00'))
                                except:
                                    existing_date = None
                            
                            if existing_date is None or new_date > existing_date:
                                existing['date'] = new_date.isoformat() if isinstance(new_date, datetime) else new_date
                                existing['is_new'] = is_new(new_date)
                
                except Exception as e:
                    logger.error(f"File processing error: {e}")
                    continue
            
            logger.info(f"✅ Found {file_count} files in database for query: {query}")
            logger.info(f"📊 Quality distribution: {dict(quality_counts)}")
            logger.info(f"📦 After processing: {len(files_dict)} unique file titles")
            
        except Exception as e:
            logger.error(f"❌ File search error: {e}")
    
    # ============================================================================
    # ✅ 3. MERGE POSTS AND FILES - POSTS FIRST, FILE-ONLY WITHOUT POST CONTENT
    # ============================================================================
    merged = {}
    
    # First, add all POSTS (with post content)
    for norm_title, post_data in posts_dict.items():
        merged[norm_title] = post_data
    
    # Then, add FILES (file-only show only file name)
    for norm_title, file_data in files_dict.items():
        # If this title already has a post (already in merged), skip file-only version
        # Because post already includes file information
        if norm_title not in merged:
            merged[norm_title] = file_data
    
    # ============================================================================
    # ✅ 4. PROCESS QUALITY OPTIONS FOR FILE RESULTS
    # ============================================================================
    for norm_title, movie_data in merged.items():
        if movie_data.get('quality_options'):
            # Get all qualities sorted by priority
            qualities = list(movie_data['quality_options'].keys())
            
            # Sort qualities by priority
            def get_quality_priority(q):
                base_q = q.replace(' HEVC', '')
                if base_q in Config.QUALITY_PRIORITY:
                    return Config.QUALITY_PRIORITY.index(base_q)
                return 999
            
            qualities.sort(key=get_quality_priority)
            
            # Calculate total size
            total_size = sum(movie_data['file_sizes'])
            
            # Determine best quality (highest priority)
            best_quality = qualities[0] if qualities else ''
            
            # Create quality summary
            quality_summary_parts = []
            for q in qualities[:5]:  # Show up to 5 qualities
                quality_summary_parts.append(q)
            
            if len(qualities) > 5:
                quality_summary_parts.append(f"+{len(qualities) - 5} more")
            
            quality_summary = " • ".join(quality_summary_parts)
            
            # Update movie data with merged info
            movie_data.update({
                'quality': best_quality,
                'quality_summary': quality_summary,
                'all_qualities': qualities,
                'available_qualities': qualities,
                'quality_count': len(qualities),
                'total_size': total_size,
                'size_formatted': format_size(total_size),
                'best_quality': best_quality
            })
    
    # ============================================================================
    # ✅ 5. FETCH THUMBNAILS WITH PRIORITY SYSTEM
    # ============================================================================
    if merged:
        logger.info(f"🖼️  Fetching thumbnails for {len(merged)} results...")
        
        # Process thumbnails for each result
        thumbnail_tasks = []
        for norm_title, movie_data in merged.items():
            # Skip if already has extracted thumbnail
            if movie_data.get('thumbnail_source') == 'extracted':
                continue
            
            # Get thumbnail with priority system
            task = thumbnail_system.get_thumbnail_for_movie(
                movie_data, 
                movie_data['title'], 
                movie_data.get('year', '')
            )
            thumbnail_tasks.append((norm_title, task))
        
        # Execute thumbnail fetching in parallel
        if thumbnail_tasks:
            for norm_title, task in thumbnail_tasks:
                try:
                    thumbnail_data = await task
                    merged[norm_title].update(thumbnail_data)
                except Exception as e:
                    logger.error(f"❌ Thumbnail error for {norm_title}: {e}")
                    # Set fallback
                    merged[norm_title].update({
                        'thumbnail': Config.FALLBACK_POSTER,
                        'thumbnail_source': 'fallback',
                        'has_thumbnail': True,
                        'poster_url': Config.FALLBACK_POSTER,
                        'poster_source': 'fallback',
                        'poster_rating': '0.0'
                    })
    
    # ============================================================================
    # ✅ 6. SORT AND PAGINATE - POSTS FIRST
    # ============================================================================
    results_list = list(merged.values())
    
    # Sorting: Posts first, then files
    # Higher score for posts (5/4) vs files (3/2)
    results_list.sort(key=lambda x: (
        x.get('has_post_content', False),  # Posts with content first
        x.get('search_score', 0),  # Higher search score
        x.get('has_file', False),  # Files
        x.get('is_new', False),  # New first
        x.get('quality_count', 0),  # More qualities
    ), reverse=True)
    
    total = len(results_list)
    paginated = results_list[offset:offset + limit]
    
    # Statistics
    posts_with_content = sum(1 for r in results_list if r.get('has_post_content', False))
    file_only = sum(1 for r in results_list if r.get('has_file', False) and not r.get('has_post_content', False))
    both = sum(1 for r in results_list if r.get('has_file', False) and r.get('has_post_content', False))
    
    stats = {
        'total': total,
        'posts_with_content': posts_with_content,
        'file_only': file_only,
        'both': both,
        'video_files': sum(1 for r in results_list if r.get('is_video_file', False)),
        'with_thumbnails': sum(1 for r in results_list if r.get('has_thumbnail', False)),
        'extracted_thumbnails': sum(1 for r in results_list if r.get('thumbnail_source') == 'extracted'),
        'multi_quality': sum(1 for r in results_list if r.get('quality_count', 0) > 1),
        'thumbnail_sources': defaultdict(int)
    }
    
    # Count thumbnail sources
    for r in results_list:
        source = r.get('thumbnail_source')
        if source:
            stats['thumbnail_sources'][source] += 1
    
    # Log detailed info
    logger.info(f"📊 FINAL RESULTS:")
    logger.info(f"   • Total: {total}")
    logger.info(f"   • Posts with content: {posts_with_content}") 
    logger.info(f"   • File-only: {file_only} (showing file name only)")
    logger.info(f"   • Both: {both}")
    
    # Show examples
    for i, result in enumerate(paginated[:3]):
        has_post = result.get('has_post_content', False)
        has_file = result.get('has_file', False)
        result_type = "📝 POST" if has_post and not has_file else "📁 FILE-ONLY" if has_file and not has_post else "📝+📁 BOTH"
        logger.info(f"   {result_type}: {result.get('title', '')[:40]}...")
    
    # Final data structure
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
            'posts_first': True,
            'file_only_shows_filename': True,
            'thumbnail_priority': Config.THUMBNAIL_PRIORITY,
            'real_message_ids': True
        },
        'bot_username': Config.BOT_USERNAME
    }
    
    # Cache results
    if cache_manager is not None:
        await cache_manager.set(cache_key, result_data, expire_seconds=Config.SEARCH_CACHE_TTL)
    
    logger.info(f"✅ Search complete: {len(paginated)} results (showing page {page})")
    
    return result_data

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
                        'channel_id': Config.MAIN_CHANNEL_ID,
                        'message_id': msg.id,
                        'has_file': False,
                        'has_post': True,
                        'content': formatted_content,
                        'post_content': post_content,
                        'quality_options': {},
                        'is_video_file': False,
                        'has_post_content': True  # ✅ Flag for post content
                    }
                    
                    movies.append(movie_data)
                    
                    if len(movies) >= limit:
                        break
        
        # Fetch thumbnails for all movies
        if movies:
            for movie in movies:
                thumbnail_data = await thumbnail_system.get_thumbnail_for_movie(
                    {},  # No extracted thumbnail for posts
                    movie['title'],
                    movie.get('year', '')
                )
                movie.update(thumbnail_data)
            
            logger.info(f"✅ Fetched {len(movies)} home movies with thumbnails")
            return movies[:limit]
        else:
            logger.warning("⚠️ No movies found for home page")
            return []
        
    except Exception as e:
        logger.error(f"❌ Home movies error: {e}")
        return []

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
# ✅ MAIN INITIALIZATION
# ============================================================================

@performance_monitor.measure("system_init")
async def init_system():
    start_time = time.time()
    
    try:
        logger.info("=" * 60)
        logger.info("🚀 SK4FiLM v9.0 - POSTS FIRST, FILE-ONLY SHOWS FILENAME")
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
        
        # Initialize Telegram Sessions
        if PYROGRAM_AVAILABLE:
            telegram_ok = await init_telegram_sessions()
            if not telegram_ok:
                logger.warning("⚠️ Telegram sessions failed")
        
        # Start initial indexing
        if user_session_ready and files_col is not None:
            logger.info("🔄 Starting file channel indexing...")
            asyncio.create_task(initial_indexing())
        
        init_time = time.time() - start_time
        logger.info(f"⚡ SK4FiLM Started in {init_time:.2f}s")
        logger.info("=" * 60)
        
        logger.info("🔧 INTEGRATED FEATURES:")
        logger.info(f"   • Posts First: ✅ ENABLED")
        logger.info(f"   • File-Only Shows Filename: ✅ ENABLED")
        logger.info(f"   • Post Content: ✅ SHOWN FOR POSTS")
        logger.info(f"   • Real Message IDs: ✅ ENABLED")
        logger.info(f"   • Thumbnail Priority: ✅ ENABLED")
        logger.info(f"   • Quality Merging: ✅ ENABLED")
        
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
        thumbnails_extracted = await files_col.count_documents({'thumbnail_extracted': True})
    else:
        tf = 0
        video_files = 0
        thumbnails_extracted = 0
    
    # Get bot handler status
    bot_status = None
    if bot_handler:
        try:
            bot_status = await bot_handler.get_bot_status()
        except Exception as e:
            logger.error(f"❌ Error getting bot status: {e}")
            bot_status = {'initialized': False, 'error': str(e)}
    
    return jsonify({
        'status': 'healthy',
        'service': 'SK4FiLM v9.0 - POSTS FIRST',
        'sessions': {
            'user_session': {
                'ready': user_session_ready,
                'channels': Config.TEXT_CHANNEL_IDS
            },
            'bot_session': {
                'ready': bot_session_ready,
                'channel': Config.FILE_CHANNEL_ID
            },
            'bot_handler': bot_status
        },
        'features': {
            'posts_first': True,
            'file_only_shows_filename': True,
            'post_content_for_posts': True,
            'real_message_ids': True,
            'thumbnail_priority': True,
            'quality_merging': True
        },
        'thumbnail_priority': Config.THUMBNAIL_PRIORITY,
        'stats': {
            'total_files': tf,
            'video_files': video_files,
            'thumbnails_extracted': thumbnails_extracted
        },
        'response_time': f"{time.perf_counter():.3f}s"
    })

@app.route('/health')
@performance_monitor.measure("health_endpoint")
async def health():
    # Safe bot status retrieval
    bot_status = None
    if bot_handler:
        try:
            bot_status = await bot_handler.get_bot_status()
        except:
            bot_status = {'initialized': False}
    
    return jsonify({
        'status': 'ok',
        'sessions': {
            'user': user_session_ready,
            'bot': bot_session_ready,
            'bot_handler': bot_status.get('initialized') if bot_status else False
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
            'thumbnail_priority': Config.THUMBNAIL_PRIORITY,
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
        
        result_data = await search_movies_multi_channel_merged(query, limit, page)
        
        return jsonify({
            'status': 'success',
            'query': query,
            'results': result_data['results'],
            'pagination': result_data['pagination'],
            'search_metadata': {
                **result_data.get('search_metadata', {}),
                'feature': 'posts_first',
                'quality_priority': Config.QUALITY_PRIORITY,
                'real_message_ids': True,
                'thumbnail_priority': Config.THUMBNAIL_PRIORITY
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
        
        # Get database stats
        if files_col is not None:
            total_files = await files_col.count_documents({})
            video_files = await files_col.count_documents({'is_video_file': True})
            thumbnails_extracted = await files_col.count_documents({'thumbnail_extracted': True})
        else:
            total_files = 0
            video_files = 0
            thumbnails_extracted = 0
        
        # Get bot handler status
        bot_status = None
        if bot_handler:
            try:
                bot_status = await bot_handler.get_bot_status()
            except:
                bot_status = {'initialized': False}
        
        return jsonify({
            'status': 'success',
            'performance': perf_stats,
            'database_stats': {
                'total_files': total_files,
                'video_files': video_files,
                'thumbnails_extracted': thumbnails_extracted,
                'extraction_rate': f"{(thumbnails_extracted/video_files*100):.1f}%" if video_files > 0 else "0%"
            },
            'bot_handler': bot_status,
            'features': {
                'posts_first': True,
                'file_only_shows_filename': True,
                'post_content_for_posts': True
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
    
    # Safely shutdown bot handler
    if bot_handler:
        try:
            await bot_handler.shutdown()
            logger.info("✅ Bot Handler stopped")
        except Exception as e:
            logger.error(f"❌ Bot Handler shutdown error: {e}")
    
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
    logger.info("🎯 FEATURES: POSTS FIRST, FILE-ONLY SHOWS FILENAME")
    logger.info(f"   • Posts First: ✅ ENABLED")
    logger.info(f"   • File-Only Shows Filename: ✅ ENABLED")
    logger.info(f"   • Post Content for Posts: ✅ ENABLED")
    logger.info(f"   • Thumbnail Priority: {Config.THUMBNAIL_PRIORITY}")
    
    try:
        asyncio.run(serve(app, config))
    except KeyboardInterrupt:
        logger.info("👋 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
