# ============================================================================
# 🚀 SK4FiLM v9.0 - COMPLETE WORKING VERSION WITH ALL FIXES
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

# ✅ SETUP LOGGING FIRST
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
# ✅ CONFIGURATION - UPDATED
# ============================================================================

class Config:
    # API Configuration
    API_ID = int(os.environ.get("API_ID", "0"))
    API_HASH = os.environ.get("API_HASH", "")
    USER_SESSION_STRING = os.environ.get("USER_SESSION_STRING", "")
    BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
    
    # Database Configuration - SIMPLIFIED
    MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017/sk4film")
    REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
    
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
    
    # Application
    WEBSITE_URL = os.environ.get("WEBSITE_URL", "https://sk4film.vercel.app")
    BOT_USERNAME = os.environ.get("BOT_USERNAME", "sk4filmbot")
    ADMIN_IDS = [int(x) for x in os.environ.get("ADMIN_IDS", "123456789").split(",")]
    WEB_SERVER_PORT = int(os.environ.get("PORT", 8000))
    BACKEND_URL = os.environ.get("BACKEND_URL", "https://sk4film.koyeb.app")
    
    # API Keys
    TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "e547e17d4e91f3e62a571655cd1ccaff")
    OMDB_API_KEY = os.environ.get("OMDB_API_KEY", "8265bd1c")
    
    # Performance Settings
    MAX_CONCURRENT_REQUESTS = 50
    CACHE_TTL = 300
    REQUEST_TIMEOUT = 10
    
    # Sync Settings
    MONITOR_INTERVAL = 300
    
    # Quality Settings
    QUALITY_PRIORITY = ['2160p', '1080p', '720p', '480p', '360p']
    
    # Fallback Poster
    FALLBACK_POSTER = "https://iili.io/fAeIwv9.th.png"
    
    # Thumbnail Settings
    THUMBNAIL_EXTRACT_TIMEOUT = 10
    THUMBNAIL_CACHE_DURATION = 24 * 60 * 60
    THUMBNAIL_EXTRACTION_ENABLED = True
    
    # Indexing Settings
    AUTO_INDEX_INTERVAL = 120
    BATCH_INDEX_SIZE = 500
    MAX_INDEX_LIMIT = 0
    INDEX_ALL_HISTORY = True
    INSTANT_AUTO_INDEX = True
    
    # Search Settings
    SEARCH_MIN_QUERY_LENGTH = 2
    SEARCH_RESULTS_PER_PAGE = 12
    SEARCH_CACHE_TTL = 600

# ============================================================================
# ✅ PERFORMANCE MONITOR
# ============================================================================

class PerformanceMonitor:
    def __init__(self):
        self.measurements = {}
    
    def measure(self, name):
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = await func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                self._record(name, elapsed)
                return result
            return async_wrapper
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

performance_monitor = PerformanceMonitor()

# ============================================================================
# ✅ APP INITIALIZATION
# ============================================================================

app = Quart(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

@app.after_request
async def add_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['X-SK4FiLM-Version'] = '9.0-WORKING'
    return response

# ============================================================================
# ✅ GLOBAL COMPONENTS
# ============================================================================

# Database
mongo_client = None
db = None
files_col = None
verification_col = None
movie_thumbnails_col = None

# Telegram
try:
    from pyrogram import Client
    PYROGRAM_AVAILABLE = True
except ImportError:
    PYROGRAM_AVAILABLE = False

User = None
Bot = None
user_session_ready = False
bot_session_ready = False

# Systems
cache_manager = None
poster_fetcher = None
telegram_bot = None

# ============================================================================
# ✅ UTILITY FUNCTIONS
# ============================================================================

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

def format_post(text, max_length=None):
    if not text:
        return ""
    text = re.sub(r'\n\s*\n', '\n\n', text)
    if max_length and len(text) > max_length:
        text = text[:max_length] + "..."
    return text.strip()

def is_video_file(filename):
    if not filename:
        return False
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v']
    return any(filename.lower().endswith(ext) for ext in video_extensions)

def detect_quality_enhanced(filename):
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
    
    is_hevc = any(re.search(pattern, filename_lower) 
                  for pattern in [r'\bhevc\b', r'\bx265\b', r'\bh\.?265\b'])
    
    for pattern, quality in patterns:
        if re.search(pattern, filename_lower):
            if is_hevc and quality in ['720p', '1080p', '2160p']:
                return f"{quality} HEVC"
            return quality
    
    return "480p"

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
# ✅ CACHE DECORATOR
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
# ✅ CACHE MANAGER
# ============================================================================

class CacheManager:
    def __init__(self, config):
        self.config = config
        self.redis_enabled = False
        self.redis_client = None
    
    async def init_redis(self):
        try:
            if self.config.REDIS_URL:
                self.redis_client = redis.from_url(
                    self.config.REDIS_URL,
                    decode_responses=True
                )
                await self.redis_client.ping()
                self.redis_enabled = True
                logger.info("✅ Redis connected")
                return True
        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed: {e}")
        return False
    
    async def get(self, key):
        if not self.redis_enabled:
            return None
        try:
            return await self.redis_client.get(key)
        except:
            return None
    
    async def set(self, key, value, expire_seconds=0):
        if not self.redis_enabled:
            return
        try:
            if expire_seconds > 0:
                await self.redis_client.setex(key, expire_seconds, value)
            else:
                await self.redis_client.set(key, value)
        except:
            pass
    
    async def delete(self, key):
        if not self.redis_enabled:
            return
        try:
            await self.redis_client.delete(key)
        except:
            pass
    
    async def start_cleanup_task(self):
        pass
    
    async def stop(self):
        if self.redis_client:
            await self.redis_client.close()

# ============================================================================
# ✅ POSTER FETCHER
# ============================================================================

class PosterFetcher:
    def __init__(self, config, cache_manager=None):
        self.config = config
        self.cache_manager = cache_manager
        self.session = None
    
    async def fetch_poster(self, title, year=""):
        """Fetch movie poster with fallback"""
        try:
            # Clean title
            clean_title = self._clean_title(title)
            
            # Try TMDB
            poster_url = await self._fetch_from_tmdb(clean_title, year)
            if poster_url:
                return {
                    'poster_url': poster_url,
                    'source': 'tmdb',
                    'rating': '0.0',
                    'title': clean_title,
                    'year': year
                }
            
            # Fallback
            return {
                'poster_url': self.config.FALLBACK_POSTER,
                'source': 'fallback',
                'rating': '0.0',
                'title': clean_title,
                'year': year
            }
            
        except Exception as e:
            logger.error(f"Poster fetch error: {e}")
            return {
                'poster_url': self.config.FALLBACK_POSTER,
                'source': 'error',
                'rating': '0.0',
                'title': title,
                'year': year
            }
    
    def _clean_title(self, title):
        """Clean movie title"""
        if not title:
            return ""
        
        patterns_to_remove = [
            r'\b\d{3,4}p\b',
            r'\bHD\b',
            r'\bHEVC\b',
            r'\bx265\b',
            r'\bWEB-DL\b',
            r'\bWEBRip\b',
            r'\bHDRip\b',
            r'\bBluRay\b',
            r'\bDVDRip\b',
            r'\bTC\b',
            r'\bTS\b',
            r'\bCAM\b',
            r'\[.*?\]',
            r'\(.*?\)',
        ]
        
        for pattern in patterns_to_remove:
            title = re.sub(pattern, '', title, flags=re.IGNORECASE)
        
        title = re.sub(r'\s+', ' ', title)
        title = title.strip()
        title = re.sub(r'\s*\d{4}$', '', title)
        
        return title
    
    async def _fetch_from_tmdb(self, title, year=""):
        """Fetch from TMDB API"""
        try:
            if not self.config.TMDB_API_KEY:
                return None
            
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            search_url = "https://api.themoviedb.org/3/search/movie"
            params = {
                'api_key': self.config.TMDB_API_KEY,
                'query': title,
                'year': year,
                'language': 'en-US',
                'page': 1
            }
            
            async with self.session.get(search_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('results') and len(data['results']) > 0:
                        movie = data['results'][0]
                        poster_path = movie.get('poster_path')
                        if poster_path:
                            return f"https://image.tmdb.org/t/p/w500{poster_path}"
        except:
            pass
        return None
    
    async def close(self):
        if self.session:
            await self.session.close()

# ============================================================================
# ✅ TELEGRAM BOT HANDLER
# ============================================================================

class BotHandler:
    def __init__(self, bot_token=None, api_id=None, api_hash=None):
        self.bot_token = bot_token or Config.BOT_TOKEN
        self.api_id = api_id or Config.API_ID
        self.api_hash = api_hash or Config.API_HASH
        self.bot = None
        self.initialized = False
        
    async def initialize(self):
        if not self.bot_token or not self.api_id or not self.api_hash:
            logger.error("❌ Bot credentials missing")
            return False
        
        try:
            global Bot
            if Bot is not None:
                self.bot = Bot
                self.initialized = True
                logger.info("✅ Bot Handler using existing Bot session")
                return True
            
            if PYROGRAM_AVAILABLE:
                self.bot = Client(
                    "sk4film_bot_handler",
                    api_id=self.api_id,
                    api_hash=self.api_hash,
                    bot_token=self.bot_token,
                    in_memory=True,
                    no_updates=True
                )
                await self.bot.start()
                bot_info = await self.bot.get_me()
                logger.info(f"✅ Bot Handler Ready: @{bot_info.username}")
                self.initialized = True
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Bot handler error: {e}")
            return False
    
    async def shutdown(self):
        if self.bot and self.bot != Bot:
            try:
                await self.bot.stop()
            except:
                pass
        self.initialized = False

bot_handler = BotHandler()

# ============================================================================
# ✅ TELEGRAM SESSIONS
# ============================================================================

@performance_monitor.measure("telegram_init")
async def init_telegram_sessions():
    global User, Bot, user_session_ready, bot_session_ready
    
    logger.info("🚀 Initializing Telegram sessions...")
    
    if not PYROGRAM_AVAILABLE:
        logger.error("❌ Pyrogram not installed!")
        return False
    
    # User Session
    if Config.API_ID > 0 and Config.API_HASH and Config.USER_SESSION_STRING:
        try:
            User = Client(
                "sk4film_user",
                api_id=Config.API_ID,
                api_hash=Config.API_HASH,
                session_string=Config.USER_SESSION_STRING,
                in_memory=True,
                no_updates=True
            )
            await User.start()
            me = await User.get_me()
            logger.info(f"✅ User Session: {me.first_name}")
            user_session_ready = True
        except Exception as e:
            logger.error(f"❌ User session failed: {e}")
            user_session_ready = False
    
    # Bot Session
    if Config.BOT_TOKEN:
        try:
            Bot = Client(
                "sk4film_bot",
                api_id=Config.API_ID,
                api_hash=Config.API_HASH,
                bot_token=Config.BOT_TOKEN,
                in_memory=True,
                no_updates=True
            )
            await Bot.start()
            bot_info = await Bot.get_me()
            logger.info(f"✅ Bot Session: @{bot_info.username}")
            bot_session_ready = True
        except Exception as e:
            logger.error(f"❌ Bot session failed: {e}")
            bot_session_ready = False
    
    return user_session_ready or bot_session_ready

# ============================================================================
# ✅ MONGODB INITIALIZATION - SIMPLIFIED
# ============================================================================

@performance_monitor.measure("mongodb_init")
async def init_mongodb():
    global mongo_client, db, files_col, verification_col, movie_thumbnails_col
    
    try:
        logger.info("🔌 Initializing MongoDB...")
        
        # Ensure URI has database name
        mongodb_uri = Config.MONGODB_URI
        if 'mongodb.net/' in mongodb_uri:
            if '?' in mongodb_uri and '/?' in mongodb_uri:
                # Fix: mongodb+srv://...mongodb.net/?...
                base = mongodb_uri.split('?')[0]
                params = mongodb_uri.split('?')[1]
                if not base.endswith('/'):
                    base += '/'
                mongodb_uri = f"{base}sk4film?{params}"
            elif '?' in mongodb_uri and not '/?' in mongodb_uri:
                # Already has database
                pass
            else:
                # No query params
                if not mongodb_uri.endswith('/'):
                    mongodb_uri += '/'
                mongodb_uri += "sk4film"
        
        logger.info(f"📡 Connecting to MongoDB...")
        
        mongo_client = AsyncIOMotorClient(
            mongodb_uri,
            serverSelectionTimeoutMS=10000,
            connectTimeoutMS=10000
        )
        
        # Test connection
        await mongo_client.admin.command('ping')
        
        # Get database (extract from URI or use default)
        db_name = "sk4film"
        if 'mongodb.net/' in mongodb_uri:
            # Extract database name from URI
            path_part = mongodb_uri.split('mongodb.net/')[1]
            if '/' in path_part:
                db_name = path_part.split('/')[0]
            elif '?' in path_part:
                db_name = path_part.split('?')[0]
        
        db = mongo_client[db_name]
        files_col = db.files
        verification_col = db.verifications
        movie_thumbnails_col = db.movie_thumbnails
        
        # Create indexes
        await files_col.create_index([("channel_id", 1), ("message_id", 1)], unique=True)
        await files_col.create_index([("normalized_title", "text")])
        await movie_thumbnails_col.create_index([("normalized_title", 1)], unique=True)
        
        logger.info(f"✅ MongoDB connected to database: {db_name}")
        return True
        
    except Exception as e:
        logger.error(f"❌ MongoDB error: {e}")
        return False

# ============================================================================
# ✅ FILE INDEXING MANAGER
# ============================================================================

class OptimizedFileIndexingManager:
    def __init__(self):
        self.is_running = False
        self.total_indexed = 0
    
    async def start_indexing(self):
        if self.is_running:
            return
        
        logger.info("🚀 Starting file indexing...")
        self.is_running = True
        
        # Start indexing in background
        asyncio.create_task(self._index_files())
    
    async def stop_indexing(self):
        self.is_running = False
        logger.info("🛑 File indexing stopped")
    
    async def _index_files(self):
        """Simple file indexing"""
        try:
            if not user_session_ready or User is None:
                logger.warning("⚠️ User session not ready for indexing")
                return
            
            # Get last indexed message
            last_msg = await files_col.find_one(
                {"channel_id": Config.FILE_CHANNEL_ID},
                sort=[("message_id", -1)]
            )
            last_msg_id = last_msg["message_id"] if last_msg else 0
            
            logger.info(f"📊 Last indexed message ID: {last_msg_id}")
            
            # Fetch new messages
            messages_to_index = []
            async for msg in User.get_chat_history(
                Config.FILE_CHANNEL_ID,
                limit=Config.BATCH_INDEX_SIZE
            ):
                if msg.id <= last_msg_id:
                    break
                
                if msg and (msg.document or msg.video):
                    messages_to_index.append(msg)
            
            logger.info(f"📥 Found {len(messages_to_index)} new files")
            
            # Index each file
            for msg in messages_to_index:
                await self._index_single_file(msg)
            
        except Exception as e:
            logger.error(f"❌ Indexing error: {e}")
    
    async def _index_single_file(self, message):
        """Index a single file"""
        try:
            # Check if already exists
            existing = await files_col.find_one({
                'channel_id': Config.FILE_CHANNEL_ID,
                'message_id': message.id
            })
            
            if existing:
                return
            
            # Extract title
            caption = message.caption or ""
            file_name = ""
            if message.document:
                file_name = message.document.file_name or ""
            elif message.video:
                file_name = message.video.file_name or ""
            
            # Simple title extraction
            title = self._extract_title(file_name, caption)
            
            # Create document
            doc = {
                'channel_id': Config.FILE_CHANNEL_ID,
                'message_id': message.id,
                'real_message_id': message.id,
                'title': title,
                'normalized_title': normalize_title(title),
                'date': message.date,
                'indexed_at': datetime.now(),
                'is_video_file': is_video_file(file_name),
                'file_size': 0,
                'status': 'active',
                'quality': detect_quality_enhanced(file_name)
            }
            
            # Add file info
            if message.document:
                doc.update({
                    'file_name': file_name,
                    'caption': caption,
                    'file_id': message.document.file_id,
                    'file_size': message.document.file_size or 0
                })
            elif message.video:
                doc.update({
                    'file_name': file_name,
                    'caption': caption,
                    'file_id': message.video.file_id,
                    'file_size': message.video.file_size or 0
                })
            
            await files_col.insert_one(doc)
            self.total_indexed += 1
            
            logger.info(f"✅ Indexed: {title[:50]}...")
            
        except Exception as e:
            logger.error(f"❌ File indexing error: {e}")
    
    def _extract_title(self, filename, caption):
        """Simple title extraction"""
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
        
        return filename or "Unknown File"

file_indexing_manager = OptimizedFileIndexingManager()

# ============================================================================
# ✅ THUMBNAIL MANAGER
# ============================================================================

class VideoThumbnailManager:
    def __init__(self):
        self.thumbnail_cache = {}
    
    async def get_thumbnail_for_movie(self, normalized_title):
        """Get thumbnail for movie"""
        if not normalized_title:
            return None
        
        # Check cache
        if normalized_title in self.thumbnail_cache:
            return self.thumbnail_cache[normalized_title]
        
        # Check database
        if movie_thumbnails_col:
            thumb = await movie_thumbnails_col.find_one(
                {'normalized_title': normalized_title},
                {'thumbnail_url': 1}
            )
            if thumb and thumb.get('thumbnail_url'):
                self.thumbnail_cache[normalized_title] = thumb['thumbnail_url']
                return thumb['thumbnail_url']
        
        return None
    
    async def get_stats(self):
        """Get thumbnail stats"""
        if movie_thumbnails_col:
            count = await movie_thumbnails_col.count_documents({})
            return {'total_thumbnails': count}
        return {'total_thumbnails': 0}

thumbnail_manager = VideoThumbnailManager()

# ============================================================================
# ✅ POSTER FETCHING FUNCTION
# ============================================================================

async def get_poster_for_movie(title, year="", quality=""):
    """Get poster for movie"""
    global poster_fetcher
    
    if poster_fetcher is None:
        return {
            'poster_url': Config.FALLBACK_POSTER,
            'source': 'fallback',
            'rating': '0.0'
        }
    
    try:
        poster_data = await poster_fetcher.fetch_poster(title, year)
        return poster_data
    except:
        return {
            'poster_url': Config.FALLBACK_POSTER,
            'source': 'fallback',
            'rating': '0.0'
        }

# ============================================================================
# ✅ ENHANCED SEARCH FUNCTION - WORKING VERSION
# ============================================================================

@performance_monitor.measure("enhanced_search")
@async_cache_with_ttl(maxsize=500, ttl=Config.SEARCH_CACHE_TTL)
async def search_movies_enhanced_fixed(query, limit=15, page=1):
    """Working search function"""
    offset = (page - 1) * limit
    
    logger.info(f"🔍 Searching for: {query}")
    
    # Search in database
    results = []
    if files_col is not None:
        try:
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
                    '_id': 0
                }
            ).limit(100)
            
            async for doc in cursor:
                title = doc.get('title', 'Unknown')
                norm_title = doc.get('normalized_title', '')
                quality = doc.get('quality', '480p')
                
                # Get thumbnail if available
                thumbnail_url = None
                if norm_title:
                    thumbnail_url = await thumbnail_manager.get_thumbnail_for_movie(norm_title)
                
                # Create result
                result = {
                    'title': title,
                    'normalized_title': norm_title,
                    'content': format_post(doc.get('caption', ''), max_length=300),
                    'post_content': doc.get('caption', ''),
                    'quality_options': {quality: {
                        'quality': quality,
                        'file_size': doc.get('file_size', 0),
                        'message_id': doc.get('real_message_id') or doc.get('message_id'),
                        'file_name': doc.get('file_name', '')
                    }},
                    'date': doc['date'].isoformat() if isinstance(doc['date'], datetime) else doc['date'],
                    'is_new': is_new(doc['date']) if doc.get('date') else False,
                    'is_video_file': doc.get('is_video_file', False),
                    'channel_id': doc.get('channel_id'),
                    'has_file': True,
                    'has_post': bool(doc.get('caption')),
                    'quality': quality,
                    'real_message_id': doc.get('real_message_id') or doc.get('message_id'),
                    'result_type': 'file_only',
                    'thumbnail_url': thumbnail_url,
                    'has_thumbnail': thumbnail_url is not None
                }
                
                results.append(result)
            
            logger.info(f"📁 Found {len(results)} files")
            
        except Exception as e:
            logger.error(f"❌ Search error: {e}")
    
    # Apply pagination
    total = len(results)
    start_idx = offset
    end_idx = offset + limit
    paginated = results[start_idx:end_idx]
    
    # Add posters to results
    for result in paginated:
        title = result.get('title', '')
        poster_data = await get_poster_for_movie(title)
        
        result.update({
            'poster_url': poster_data['poster_url'],
            'poster_source': poster_data['source'],
            'poster_rating': poster_data['rating'],
            'has_poster': True
        })
        
        # Use poster as thumbnail if no thumbnail
        if not result.get('thumbnail_url'):
            result['thumbnail_url'] = poster_data['poster_url']
            result['has_thumbnail'] = True
    
    # Return results
    return {
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
            'stats': {'total': total},
            'poster_fetcher': poster_fetcher is not None
        },
        'bot_username': Config.BOT_USERNAME
    }

# ============================================================================
# ✅ HOME MOVIES
# ============================================================================

@performance_monitor.measure("home_movies")
@async_cache_with_ttl(maxsize=1, ttl=60)
async def get_home_movies(limit=25):
    """Get home movies"""
    movies = []
    
    if user_session_ready and User is not None:
        try:
            async for msg in User.get_chat_history(Config.MAIN_CHANNEL_ID, limit=limit):
                if msg and msg.text and len(msg.text) > 25:
                    title = extract_title_smart(msg.text)
                    if title:
                        # Create movie data
                        movie_data = {
                            'title': title,
                            'date': msg.date.isoformat() if isinstance(msg.date, datetime) else str(msg.date),
                            'channel_id': Config.MAIN_CHANNEL_ID,
                            'message_id': msg.id,
                            'has_file': False,
                            'has_post': True,
                            'content': format_post(msg.text, max_length=500)
                        }
                        
                        # Get poster
                        poster_data = await get_poster_for_movie(title)
                        movie_data.update({
                            'poster_url': poster_data['poster_url'],
                            'poster_source': poster_data['source'],
                            'poster_rating': poster_data['rating'],
                            'has_poster': True
                        })
                        
                        movies.append(movie_data)
                        
                        if len(movies) >= limit:
                            break
            
            logger.info(f"✅ Fetched {len(movies)} home movies")
            
        except Exception as e:
            logger.error(f"❌ Home movies error: {e}")
    
    return movies

# ============================================================================
# ✅ SYSTEM INITIALIZATION
# ============================================================================

@performance_monitor.measure("system_init")
async def init_system():
    logger.info("=" * 60)
    logger.info("🚀 SK4FiLM v9.0 - WORKING VERSION")
    logger.info("=" * 60)
    
    try:
        # Initialize MongoDB
        mongo_ok = await init_mongodb()
        if not mongo_ok:
            logger.error("❌ MongoDB failed")
            return False
        
        # Get current stats
        if files_col:
            file_count = await files_col.count_documents({})
            logger.info(f"📊 Current files in database: {file_count}")
        
        # Initialize Bot Handler
        await bot_handler.initialize()
        
        # Initialize Cache
        global cache_manager
        cache_manager = CacheManager(Config)
        await cache_manager.init_redis()
        
        # Initialize Poster Fetcher
        global poster_fetcher
        poster_fetcher = PosterFetcher(Config, cache_manager)
        
        # Initialize Telegram Sessions
        telegram_ok = await init_telegram_sessions()
        
        # Start indexing if user session ready
        if user_session_ready and files_col:
            await file_indexing_manager.start_indexing()
        
        logger.info("✅ SK4FiLM initialized successfully")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ System init error: {e}")
        return False

# ============================================================================
# ✅ API ROUTES
# ============================================================================

@app.route('/')
@performance_monitor.measure("root_endpoint")
async def root():
    # Get stats
    file_count = 0
    if files_col:
        file_count = await files_col.count_documents({})
    
    return jsonify({
        'status': 'healthy',
        'service': 'SK4FiLM v9.0',
        'sessions': {
            'user': user_session_ready,
            'bot': bot_session_ready
        },
        'stats': {
            'total_files': file_count
        },
        'features': {
            'search': True,
            'posters': poster_fetcher is not None,
            'indexing': file_indexing_manager.is_running
        }
    })

@app.route('/health')
async def health():
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/movies', methods=['GET'])
async def api_movies():
    try:
        movies = await get_home_movies(limit=25)
        return jsonify({
            'status': 'success',
            'movies': movies,
            'total': len(movies)
        })
    except Exception as e:
        logger.error(f"Movies API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/search', methods=['GET'])
async def api_search():
    try:
        query = request.args.get('query', '').strip()
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', Config.SEARCH_RESULTS_PER_PAGE))
        
        if len(query) < Config.SEARCH_MIN_QUERY_LENGTH:
            return jsonify({
                'status': 'error',
                'message': f'Query too short'
            }), 400
        
        result_data = await search_movies_enhanced_fixed(query, limit, page)
        
        return jsonify({
            'status': 'success',
            'query': query,
            'results': result_data['results'],
            'pagination': result_data['pagination'],
            'bot_username': Config.BOT_USERNAME
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
        file_count = 0
        if files_col:
            file_count = await files_col.count_documents({})
        
        return jsonify({
            'status': 'success',
            'database_files': file_count,
            'indexing': {
                'running': file_indexing_manager.is_running,
                'total_indexed': file_indexing_manager.total_indexed
            },
            'sessions': {
                'user': user_session_ready,
                'bot': bot_session_ready
            }
        })
    except Exception as e:
        logger.error(f"Stats API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# ============================================================================
# ✅ ADMIN ROUTES
# ============================================================================

@app.route('/api/admin/reindex', methods=['POST'])
async def api_admin_reindex():
    try:
        # Simple auth check
        auth_token = request.headers.get('X-Admin-Token')
        if not auth_token or auth_token != os.environ.get('ADMIN_TOKEN', 'sk4film_admin'):
            return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401
        
        # Trigger reindex
        await file_indexing_manager.start_indexing()
        
        return jsonify({
            'status': 'success',
            'message': 'Reindexing started'
        })
    except Exception as e:
        logger.error(f"Admin reindex error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# ============================================================================
# ✅ STARTUP AND SHUTDOWN
# ============================================================================

@app.before_serving
async def startup():
    await init_system()

@app.after_serving
async def shutdown():
    logger.info("🛑 Shutting down SK4FiLM...")
    
    # Stop indexing
    await file_indexing_manager.stop_indexing()
    
    # Close bot handler
    await bot_handler.shutdown()
    
    # Close poster fetcher
    if poster_fetcher:
        await poster_fetcher.close()
    
    # Close Telegram sessions
    if User:
        await User.stop()
    if Bot:
        await Bot.stop()
    
    # Close cache
    if cache_manager:
        await cache_manager.stop()
    
    # Close MongoDB
    if mongo_client:
        mongo_client.close()
    
    logger.info("👋 Shutdown complete")

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
    
    logger.info(f"🌐 Starting SK4FiLM on port {Config.WEB_SERVER_PORT}...")
    logger.info("✅ Features: Search, Indexing, Posters, Thumbnails")
    
    try:
        asyncio.run(serve(app, config))
    except KeyboardInterrupt:
        logger.info("👋 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
