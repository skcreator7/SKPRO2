# ============================================================================
# 🚀 SK4FiLM v9.0 - COMPLETE STREAMING & DOWNLOAD SYSTEM - FIXED
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
from quart import Quart, jsonify, request, Response, render_template_string
from hypercorn.asyncio import serve
from hypercorn.config import Config as HyperConfig
from motor.motor_asyncio import AsyncIOMotorClient
import redis.asyncio as redis

# ============================================================================
# LOGGER SETUP
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
# MODULE IMPORTS WITH FALLBACKS
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

try:
    from poster_fetching import PosterFetcher, PosterSource
    logger.debug("✅ Poster fetching module imported")
except ImportError as e:
    logger.error(f"❌ Poster fetching module import error: {e}")
    class PosterSource:
        TMDB = "tmdb"
        OMDB = "omdb"
        CUSTOM = "custom"
        FALLBACK = "fallback"

# Import utils with fallbacks
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
# PERFORMANCE MONITOR
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
# CONFIGURATION
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
    TMDB_KEYS = [TMDB_API_KEY]
    OMDB_KEYS = [OMDB_API_KEY]
    
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
    
    # Poster Fetch Settings
    POSTER_FETCH_TIMEOUT = int(os.environ.get("POSTER_FETCH_TIMEOUT", "3"))
    POSTER_FETCH_BATCH_SIZE = int(os.environ.get("POSTER_FETCH_BATCH_SIZE", "10"))
    
    # Search Settings
    SEARCH_MIN_QUERY_LENGTH = 2
    SEARCH_RESULTS_PER_PAGE = 20
    SEARCH_CACHE_TTL = 300
    MAX_SEARCH_RESULTS = 100
    
    # Streaming Settings
    STREAMING_ENABLED = os.environ.get("STREAMING_ENABLED", "true").lower() == "true"
    STREAMING_PROXY_URL = os.environ.get("STREAMING_PROXY_URL", "https://stream.sk4film.workers.dev")
    STREAMING_TIMEOUT = int(os.environ.get("STREAMING_TIMEOUT", "30"))
    
    # Download Settings
    DIRECT_DOWNLOAD_ENABLED = os.environ.get("DIRECT_DOWNLOAD_ENABLED", "true").lower() == "true"
    TELEGRAM_CDN_BASE = "https://cdn5.telegram-cdn.org/file"
    TELEGRAM_DOWNLOAD_URL = "https://t.me/{bot_username}?start={file_id}"
    
    # Telegram Bot File Format
    TELEGRAM_FILE_FORMAT = "{channel_id}_{message_id}_{quality}"

# ============================================================================
# QUART APP SETUP
# ============================================================================
app = Quart(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

@app.after_request
async def add_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['X-SK4FiLM-Version'] = '9.0-FIXED'
    response.headers['X-Response-Time'] = f"{time.perf_counter():.3f}"
    return response

# ============================================================================
# GLOBAL COMPONENTS
# ============================================================================
mongo_client = None
db = None
files_col = None
verification_col = None

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

cache_manager = None
verification_system = None
premium_system = None
poster_fetcher = None
bot_handler = None

is_indexing = False
last_index_time = None
indexing_task = None

# ============================================================================
# STREAMING PROXY MANAGER
# ============================================================================
class StreamingProxyManager:
    def __init__(self):
        self.proxy_url = Config.STREAMING_PROXY_URL
        self.session = None
    
    async def get_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=Config.STREAMING_TIMEOUT)
            )
        return self.session
    
    async def get_stream_url(self, file_id: str, quality: str = "auto") -> Optional[str]:
        if not Config.STREAMING_ENABLED:
            return None
        
        try:
            session = await self.get_session()
            proxy_params = {
                "file_id": file_id,
                "quality": quality,
                "format": "stream"
            }
            stream_url = f"{self.proxy_url}/stream"
            
            try:
                async with session.get(stream_url, params=proxy_params, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("stream_url")
                    else:
                        logger.warning(f"Stream proxy error: {response.status}")
                        return None
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(f"Stream proxy connection failed: {e}")
                return None
        except Exception as e:
            logger.error(f"Stream URL error: {e}")
            return None
    
    async def get_direct_download_url(self, file_id: str) -> Optional[Dict]:
        try:
            parts = file_id.split('_')
            if len(parts) < 2:
                return None
            
            channel_id = parts[0]
            message_id = parts[1]
            quality = parts[2] if len(parts) > 2 else "480p"
            
            telegram_bot_url = f"https://t.me/{Config.BOT_USERNAME}?start={file_id}"
            
            file_info = await self.get_file_info(file_id)
            if file_info:
                return {
                    "telegram_bot_url": telegram_bot_url,
                    "quality": quality,
                    "file_name": file_info.get('file_name', 'video.mp4'),
                    "file_size": file_info.get('file_size', 0),
                    "size_formatted": format_size(file_info.get('file_size', 0)),
                    "direct_download": False,
                    "download_instructions": f"Click the link to download via Telegram bot"
                }
            else:
                return {
                    "telegram_bot_url": telegram_bot_url,
                    "quality": quality,
                    "file_name": "video.mp4",
                    "file_size": 0,
                    "size_formatted": "Unknown",
                    "direct_download": False,
                    "download_instructions": f"Click the link to download via Telegram bot"
                }
        except Exception as e:
            logger.error(f"Direct download URL error: {e}")
            return None
    
    async def get_file_info(self, file_id: str) -> Optional[Dict]:
        try:
            parts = file_id.split('_')
            if len(parts) < 2:
                return None
            
            channel_id = int(parts[0])
            message_id = int(parts[1])
            quality = parts[2] if len(parts) > 2 else "480p"
            
            if files_col is None:
                return None
            
            file_doc = await files_col.find_one({
                "channel_id": channel_id,
                "message_id": message_id
            })
            
            if not file_doc:
                return None
            
            duration = file_doc.get('duration', 0)
            duration_formatted = self.format_duration(duration)
            
            return {
                "title": file_doc.get('title', ''),
                "file_name": file_doc.get('file_name', ''),
                "file_size": file_doc.get('file_size', 0),
                "quality": file_doc.get('quality', quality),
                "duration": duration,
                "duration_formatted": duration_formatted,
                "thumbnail_url": file_doc.get('thumbnail_url'),
                "telegram_file_id": file_doc.get('telegram_file_id'),
                "channel_id": channel_id,
                "message_id": message_id,
                "is_video_file": file_doc.get('is_video_file', False),
                "caption": file_doc.get('caption', ''),
                "date": file_doc.get('date'),
                "year": file_doc.get('year', '')
            }
        except Exception as e:
            logger.error(f"Get file info error: {e}")
            return None
    
    def format_duration(self, seconds: int) -> str:
        if not seconds or seconds <= 0:
            return "Unknown"
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    async def close(self):
        if self.session:
            await self.session.close()

streaming_proxy = StreamingProxyManager()

# ============================================================================
# 🔥 POSTER FETCHING - QUICK VERSION (FIXED)
# ============================================================================
async def get_poster_for_movie_quick(title: str, year: str = "") -> Dict[str, Any]:
    """Quick poster fetch with proper fallback"""
    try:
        # Use global poster_fetcher if available
        if poster_fetcher:
            poster_data = await poster_fetcher.fetch_poster(title)
            return poster_data
        
        # Fallback to TMDB direct
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    "https://api.themoviedb.org/3/search/movie",
                    params={"api_key": Config.TMDB_API_KEY, "query": title},
                    timeout=aiohttp.ClientTimeout(total=Config.POSTER_FETCH_TIMEOUT)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("results"):
                            movie = data["results"][0]
                            if movie.get("poster_path"):
                                return {
                                    'poster_url': f"https://image.tmdb.org/t/p/w780{movie['poster_path']}",
                                    'source': 'tmdb',
                                    'rating': str(movie.get('vote_average', '0.0')),
                                    'year': (movie.get('release_date') or '')[:4],
                                    'title': movie.get('title', title)
                                }
            except Exception as e:
                logger.error(f"TMDB fetch error: {e}")
    except Exception as e:
        logger.error(f"Poster fetch error: {e}")
    
    # Fallback
    return {
        'poster_url': Config.FALLBACK_POSTER,
        'source': 'fallback',
        'rating': '0.0',
        'year': year,
        'title': title
    }

async def get_posters_for_movies_batch_quick(movies: List[Dict]) -> List[Dict]:
    """Batch poster fetch with proper handling"""
    results = []
    limited_movies = movies[:Config.POSTER_FETCH_BATCH_SIZE]
    
    tasks = []
    for movie in limited_movies:
        title = movie.get('title', '')
        year = movie.get('year', '')
        task = asyncio.create_task(get_poster_for_movie_quick(title, year))
        tasks.append((movie, task))
    
    for movie, task in tasks:
        try:
            poster_data = await asyncio.wait_for(task, timeout=Config.POSTER_FETCH_TIMEOUT)
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
        except (asyncio.TimeoutError, Exception) as e:
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
    
    for movie in movies[Config.POSTER_FETCH_BATCH_SIZE:]:
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
# 🔥 THUMBNAIL EXTRACTION FROM TEXT (FIXED)
# ============================================================================
def extract_thumbnail_from_text(text: str) -> Optional[str]:
    """Extract image URL from text/caption"""
    if not text:
        return None
    
    # Try to find image URLs in text
    url_patterns = [
        r'(https?://[^\s]+\.(?:jpg|jpeg|png|gif|webp))',
        r'(https?://i\.imgur\.com/[^\s]+)',
        r'(https?://iili\.io/[^\s]+)',
        r'(https?://.*?/photo/[^\s]+)',
    ]
    
    for pattern in url_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return None

# ============================================================================
# DATABASE INITIALIZATION
# ============================================================================
async def init_database():
    global mongo_client, db, files_col, verification_col
    
    logger.info("=" * 50)
    logger.info("📊 DATABASE INITIALIZATION")
    logger.info("=" * 50)
    
    try:
        mongo_client = AsyncIOMotorClient(
            Config.MONGODB_URI,
            maxPoolSize=50,
            minPoolSize=10,
            serverSelectionTimeoutMS=5000
        )
        
        await mongo_client.admin.command('ping')
        logger.info("✅ MongoDB connected")
        
        db = mongo_client.sk4film
        files_col = db.files
        verification_col = db.verification
        
        # Create indexes
        await files_col.create_index([("normalized_title", 1)])
        await files_col.create_index([("quality", 1)])
        await files_col.create_index([("status", 1)])
        await files_col.create_index([("is_duplicate", 1)])
        await files_col.create_index([("date", -1)])
        
        logger.info("✅ Database indexes created")
        return True
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        return False

# ============================================================================
# TELEGRAM SESSION INITIALIZATION
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
    
    logger.info("\n" + "=" * 50)
    logger.info("📊 SESSION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"USER Session: {'✅ READY' if user_session_ready else '❌ NOT READY'}")
    logger.info(f"BOT Session: {'✅ READY' if bot_session_ready else '❌ NOT READY'}")
    
    return user_session_ready or bot_session_ready

# ============================================================================
# SYSTEM INITIALIZATION
# ============================================================================
async def init_systems():
    global cache_manager, verification_system, premium_system, poster_fetcher
    
    logger.info("=" * 50)
    logger.info("⚙️ SYSTEM COMPONENTS INITIALIZATION")
    logger.info("=" * 50)
    
    # Cache Manager
    cache_manager = CacheManager(Config)
    await cache_manager.init_redis()
    logger.info("✅ Cache Manager initialized")
    
    # Verification System
    verification_system = VerificationSystem(Config, mongo_client)
    logger.info("✅ Verification System initialized")
    
    # Premium System
    premium_system = PremiumSystem(Config, mongo_client)
    logger.info("✅ Premium System initialized")
    
    # Poster Fetcher
    try:
        if PosterFetcher:
            poster_fetcher = PosterFetcher(Config, cache_manager.redis_client if cache_manager.redis_enabled else None)
            logger.info("✅ Poster Fetcher initialized")
        else:
            logger.warning("⚠️ Poster Fetcher not available")
    except Exception as e:
        logger.error(f"❌ Poster Fetcher initialization failed: {e}")
    
    logger.info("=" * 50)

# ============================================================================
# 🔥 API ENDPOINTS - FIXED
# ============================================================================

# Home Page - Latest Movies with Posters
@app.route('/api/movies', methods=['GET'])
@performance_monitor.measure("api_movies")
async def get_movies():
    """Get latest movies for home page with posters"""
    try:
        if files_col is None:
            return jsonify({"error": "Database not available"}), 503
        
        # Aggregate to get unique movies (by normalized_title)
        pipeline = [
            {
                "$match": {
                    "status": "active",
                    "is_duplicate": False,
                    "is_video_file": True
                }
            },
            {
                "$sort": {"date": -1}
            },
            {
                "$group": {
                    "_id": "$normalized_title",
                    "title": {"$first": "$title"},
                    "normalized_title": {"$first": "$normalized_title"},
                    "year": {"$first": "$year"},
                    "date": {"$first": "$date"},
                    "channel_id": {"$first": "$channel_id"},
                    "message_id": {"$first": "$message_id"},
                    "quality": {"$first": "$quality"},
                    "file_size": {"$first": "$file_size"},
                    "thumbnail_url": {"$first": "$thumbnail_url"},
                    "caption": {"$first": "$caption"}
                }
            },
            {
                "$sort": {"date": -1}
            },
            {
                "$limit": 20
            }
        ]
        
        movies = []
        async for doc in files_col.aggregate(pipeline):
            # Create file_id for stream button
            file_id = f"{doc.get('channel_id', Config.FILE_CHANNEL_ID)}_{doc.get('message_id')}_{doc.get('quality', '480p')}"
            
            # Extract thumbnail from caption if not available
            thumbnail = doc.get('thumbnail_url')
            if not thumbnail and doc.get('caption'):
                thumbnail = extract_thumbnail_from_text(doc.get('caption', ''))
            
            movie = {
                'title': doc.get('title', 'Unknown'),
                'normalized_title': doc.get('normalized_title', ''),
                'year': doc.get('year', ''),
                'quality': doc.get('quality', '480p'),
                'size': format_size(doc.get('file_size', 0)),
                'file_size': doc.get('file_size', 0),
                'date': doc.get('date').isoformat() if doc.get('date') else None,
                'thumbnail': thumbnail or Config.FALLBACK_POSTER,
                'has_thumbnail': thumbnail is not None,
                'file_id': file_id,
                'channel_id': doc.get('channel_id'),
                'message_id': doc.get('message_id'),
                'is_new': is_new(doc.get('date'))
            }
            movies.append(movie)
        
        # Fetch posters for all movies
        movies_with_posters = await get_posters_for_movies_batch_quick(movies)
        
        logger.info(f"✅ Returned {len(movies_with_posters)} movies with posters")
        
        return jsonify({
            "success": True,
            "count": len(movies_with_posters),
            "movies": movies_with_posters
        })
    
    except Exception as e:
        logger.error(f"❌ Get movies error: {e}")
        return jsonify({"error": str(e)}), 500

# Search API - Fixed with proper thumbnail handling
@app.route('/api/search', methods=['GET'])
@performance_monitor.measure("api_search")
async def search_api():
    """Search API with proper thumbnail extraction"""
    try:
        query = request.args.get('q', '').strip()
        
        if len(query) < Config.SEARCH_MIN_QUERY_LENGTH:
            return jsonify({
                "error": f"Query must be at least {Config.SEARCH_MIN_QUERY_LENGTH} characters",
                "results": []
            }), 400
        
        if files_col is None:
            return jsonify({"error": "Database not available"}), 503
        
        # Normalize query
        normalized_query = normalize_title(query)
        
        # Search in files collection
        search_filter = {
            "$and": [
                {"status": "active"},
                {"is_duplicate": False},
                {
                    "$or": [
                        {"normalized_title": {"$regex": normalized_query, "$options": "i"}},
                        {"title": {"$regex": query, "$options": "i"}},
                        {"caption": {"$regex": query, "$options": "i"}}
                    ]
                }
            ]
        }
        
        cursor = files_col.find(search_filter).sort("date", -1).limit(Config.MAX_SEARCH_RESULTS)
        
        results = []
        grouped_results = {}
        
        async for doc in cursor:
            normalized_title = doc.get('normalized_title', '')
            if not normalized_title:
                continue
            
            # Group by normalized title
            if normalized_title not in grouped_results:
                # Extract thumbnail from caption if not available
                thumbnail = doc.get('thumbnail_url')
                if not thumbnail and doc.get('caption'):
                    thumbnail = extract_thumbnail_from_text(doc.get('caption', ''))
                
                grouped_results[normalized_title] = {
                    'title': doc.get('title', 'Unknown'),
                    'normalized_title': normalized_title,
                    'year': doc.get('year', ''),
                    'thumbnail': thumbnail or Config.FALLBACK_POSTER,
                    'has_thumbnail': thumbnail is not None,
                    'caption': doc.get('caption', ''),
                    'has_posts': bool(doc.get('caption')),
                    'post_text': doc.get('caption', '')[:200] if doc.get('caption') else '',
                    'files': [],
                    'qualities': [],
                    'date': doc.get('date')
                }
            
            # Add file info
            is_video = doc.get('is_video_file', False)
            if is_video:
                file_quality = doc.get('quality', '480p')
                file_id = f"{doc.get('channel_id', Config.FILE_CHANNEL_ID)}_{doc.get('message_id')}_{file_quality}"
                
                file_info = {
                    'quality': file_quality,
                    'size': format_size(doc.get('file_size', 0)),
                    'file_size': doc.get('file_size', 0),
                    'file_name': doc.get('file_name', ''),
                    'file_id': file_id,
                    'channel_id': doc.get('channel_id'),
                    'message_id': doc.get('message_id'),
                    'is_video': True,
                    'has_file': True
                }
                
                grouped_results[normalized_title]['files'].append(file_info)
                grouped_results[normalized_title]['qualities'].append(file_quality)
        
        # Convert to list and add posters
        results = list(grouped_results.values())
        
        # Sort files by quality priority
        for result in results:
            result['files'].sort(key=lambda x: Config.QUALITY_PRIORITY.index(x['quality']) if x['quality'] in Config.QUALITY_PRIORITY else 999)
            result['has_files'] = len(result['files']) > 0
        
        # Sort results by date
        results.sort(key=lambda x: x.get('date') or datetime.min, reverse=True)
        
        # Fetch posters
        results_with_posters = await get_posters_for_movies_batch_quick(results)
        
        logger.info(f"✅ Search '{query}' returned {len(results_with_posters)} results")
        
        return jsonify({
            "success": True,
            "query": query,
            "count": len(results_with_posters),
            "results": results_with_posters
        })
    
    except Exception as e:
        logger.error(f"❌ Search error: {e}")
        return jsonify({"error": str(e)}), 500

# File Info API - For View Page
@app.route('/api/file/<path:file_id>', methods=['GET'])
@performance_monitor.measure("api_file_info")
async def get_file_info_api(file_id: str):
    """Get enhanced file info for view page"""
    try:
        parts = file_id.split('_')
        if len(parts) < 2:
            return jsonify({"error": "Invalid file ID format"}), 400
        
        channel_id = int(parts[0])
        message_id = int(parts[1])
        quality = parts[2] if len(parts) > 2 else ""
        
        if files_col is None:
            return jsonify({"error": "Database not available"}), 503
        
        # Get the specific file
        file_doc = await files_col.find_one({
            "channel_id": channel_id,
            "message_id": message_id
        })
        
        if not file_doc:
            return jsonify({"error": "File not found"}), 404
        
        # Get normalized title
        normalized_title = file_doc.get('normalized_title')
        if not normalized_title:
            normalized_title = normalize_title(file_doc.get('title', ''))
        
        # Find ALL files with same normalized title (all qualities)
        all_files_cursor = files_col.find({
            "normalized_title": normalized_title,
            "status": "active",
            "is_duplicate": False
        }).limit(20)
        
        all_files = await all_files_cursor.to_list(length=20)
        
        # Organize by quality
        quality_options = {}
        selected_quality_info = None
        
        for file_data in all_files:
            file_quality = file_data.get('quality', 'Unknown')
            file_msg_id = file_data.get('message_id')
            file_unique_id = f"{file_data.get('channel_id', Config.FILE_CHANNEL_ID)}_{file_msg_id}_{file_quality}"
            
            # Extract thumbnail
            thumbnail = file_data.get('thumbnail_url')
            if not thumbnail and file_data.get('caption'):
                thumbnail = extract_thumbnail_from_text(file_data.get('caption', ''))
            
            quality_option = {
                'file_id': file_unique_id,
                'file_size': file_data.get('file_size', 0),
                'size_formatted': format_size(file_data.get('file_size', 0)),
                'file_name': file_data.get('file_name', ''),
                'is_video': file_data.get('is_video_file', False),
                'channel_id': file_data.get('channel_id'),
                'message_id': file_msg_id,
                'quality': file_quality,
                'thumbnail_url': thumbnail,
                'has_thumbnail': thumbnail is not None,
                'date': file_data.get('date'),
                'telegram_file_id': file_data.get('telegram_file_id'),
                'duration': file_data.get('duration', 0),
                'duration_formatted': streaming_proxy.format_duration(file_data.get('duration', 0))
            }
            
            if file_unique_id == file_id:
                selected_quality_info = quality_option
            
            quality_options[file_quality] = quality_option
        
        # If no specific quality selected, use first one
        if not selected_quality_info and quality_options:
            first_quality = list(quality_options.keys())[0]
            selected_quality_info = quality_options[first_quality]
        
        # Get streaming URL
        stream_url = None
        if Config.STREAMING_ENABLED and selected_quality_info:
            stream_url = await streaming_proxy.get_stream_url(
                selected_quality_info['file_id'],
                selected_quality_info['quality']
            )
        
        # Get download info
        download_info = await streaming_proxy.get_direct_download_url(
            selected_quality_info['file_id'] if selected_quality_info else file_id
        )
        
        # Get poster
        poster_data = await get_poster_for_movie_quick(
            file_doc.get('title', ''),
            file_doc.get('year', '')
        )
        
        # Prepare quality list sorted by priority
        qualities_list = list(quality_options.keys())
        def get_quality_priority(q):
            base_q = q.replace(' HEVC', '')
            if base_q in Config.QUALITY_PRIORITY:
                return Config.QUALITY_PRIORITY.index(base_q)
            return 999
        qualities_list.sort(key=get_quality_priority)
        
        return jsonify({
            "success": True,
            "file_id": file_id,
            "title": file_doc.get('title', 'Unknown'),
            "year": file_doc.get('year', ''),
            "poster_url": poster_data['poster_url'],
            "poster_source": poster_data['source'],
            "rating": poster_data['rating'],
            "thumbnail": selected_quality_info['thumbnail_url'] if selected_quality_info else Config.FALLBACK_POSTER,
            "selected_quality": selected_quality_info['quality'] if selected_quality_info else quality,
            "qualities": qualities_list,
            "quality_options": quality_options,
            "stream_url": stream_url,
            "streaming_enabled": Config.STREAMING_ENABLED,
            "download_info": download_info,
            "caption": file_doc.get('caption', ''),
            "date": file_doc.get('date').isoformat() if file_doc.get('date') else None
        })
    
    except Exception as e:
        logger.error(f"❌ File info error: {e}")
        return jsonify({"error": str(e)}), 500

# Health Check
@app.route('/health', methods=['GET'])
async def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "version": "9.0-FIXED",
        "database": "connected" if mongo_client else "disconnected",
        "telegram_user": "ready" if user_session_ready else "not ready",
        "telegram_bot": "ready" if bot_session_ready else "not ready"
    })

# ============================================================================
# STARTUP & SHUTDOWN
# ============================================================================
@app.before_serving
async def startup():
    """Initialize all systems"""
    logger.info("=" * 60)
    logger.info("🚀 SK4FiLM v9.0 STARTUP - FIXED VERSION")
    logger.info("=" * 60)
    
    # Initialize database
    db_success = await init_database()
    if not db_success:
        logger.error("❌ Database initialization failed")
        return
    
    # Initialize Telegram sessions
    telegram_success = await init_telegram_sessions()
    if not telegram_success:
        logger.warning("⚠️ Telegram sessions not fully initialized")
    
    # Initialize systems
    await init_systems()
    
    logger.info("=" * 60)
    logger.info("✅ SK4FiLM v9.0 - READY - FIXED VERSION")
    logger.info("=" * 60)

@app.after_serving
async def shutdown():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down...")
    
    if User:
        try:
            await User.stop()
            logger.info("✅ User session stopped")
        except:
            pass
    
    if Bot:
        try:
            await Bot.stop()
            logger.info("✅ Bot session stopped")
        except:
            pass
    
    if mongo_client:
        mongo_client.close()
        logger.info("✅ MongoDB connection closed")
    
    if cache_manager:
        await cache_manager.stop()
        logger.info("✅ Cache manager stopped")
    
    await streaming_proxy.close()
    logger.info("✅ Streaming proxy closed")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    hyper_config = HyperConfig()
    hyper_config.bind = [f"0.0.0.0:{Config.WEB_SERVER_PORT}"]
    hyper_config.workers = 1
    hyper_config.use_reloader = False
    hyper_config.accesslog = None
    hyper_config.errorlog = "-"
    
    logger.info(f"🚀 Starting server on port {Config.WEB_SERVER_PORT}")
    asyncio.run(serve(app, hyper_config))
