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
from quart import Quart, jsonify, request, Response
from hypercorn.asyncio import serve
from hypercorn.config import Config as HyperConfig
from motor.motor_asyncio import AsyncIOMotorClient
import redis.asyncio as redis

# Pyrogram imports
try:
    from pyrogram import Client
    from pyrogram.errors import FloodWait, SessionPasswordNeeded, PhoneCodeInvalid
    PYROGRAM_AVAILABLE = True
except ImportError:
    PYROGRAM_AVAILABLE = False
    Client = None
    FloodWait = None

# ✅ ULTRA-FAST LOADING OPTIMIZATIONS
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

# Performance monitoring
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

# Configuration with optimizations
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
    
    # Channel Configuration - DUAL SESSION
    MAIN_CHANNEL_ID = -1001891090100          # ✅ User Session
    TEXT_CHANNEL_IDS = [-1001891090100, -1002024811395]  # ✅ User Session
    FILE_CHANNEL_ID = -1001768249569          # ✅ Bot Session
    
    # Links
    MAIN_CHANNEL_LINK = "https://t.me/sk4film"
    UPDATES_CHANNEL_LINK = "https://t.me/sk4film_Request"
    CHANNEL_USERNAME = "sk4film"
    
    # URL Shortener
    SHORTLINK_API = os.environ.get("SHORTLINK_API", "")
    CUTTLY_API = os.environ.get("CUTTLY_API", "")
    
    # UPI IDs
    UPI_ID_BASIC = os.environ.get("UPI_ID_BASIC", "sk4filmbot@ybl")
    UPI_ID_PREMIUM = os.environ.get("UPI_ID_PREMIUM", "sk4filmbot@ybl")
    UPI_ID_GOLD = os.environ.get("UPI_ID_GOLD", "sk4filmbot@ybl")
    UPI_ID_DIAMOND = os.environ.get("UPI_ID_DIAMOND", "sk4filmbot@ybl")
    
    # Verification
    VERIFICATION_REQUIRED = os.environ.get("VERIFICATION_REQUIRED", "true").lower() == "true"
    VERIFICATION_DURATION = 6 * 60 * 60  # 6 hours
    
    # Application
    WEBSITE_URL = os.environ.get("WEBSITE_URL", "https://sk4film.vercel.app")
    BOT_USERNAME = os.environ.get("BOT_USERNAME", "sk4filmbot")
    ADMIN_IDS = [int(x) for x in os.environ.get("ADMIN_IDS", "123456789").split(",")]
    AUTO_DELETE_TIME = int(os.environ.get("AUTO_DELETE_TIME", "300"))
    WEB_SERVER_PORT = int(os.environ.get("PORT", 8000))
    BACKEND_URL = os.environ.get("BACKEND_URL", "https://sk4film.koyeb.app")
    
    # API Keys
    OMDB_KEYS = ["8265bd1c", "b9bd48a6", "3e7e1e9d"]
    TMDB_KEYS = ["e547e17d4e91f3e62a571655cd1ccaff", "8265bd1f"]
    
    # Performance Settings
    MAX_CONCURRENT_REQUESTS = int(os.environ.get("MAX_CONCURRENT_REQUESTS", "50"))
    CACHE_TTL = int(os.environ.get("CACHE_TTL", "300"))
    REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "10"))
    
    # Sync Management Settings
    MONITOR_INTERVAL = int(os.environ.get("MONITOR_INTERVAL", "300"))
    
    @staticmethod
    def get_poster(title, year=""):
        """Generate poster URL for fallback"""
        if not title:
            return f"https://via.placeholder.com/300x450/1a1a2e/ffffff?text=No+Poster"
        
        encoded_title = urllib.parse.quote(title[:50])
        if year:
            return f"{Config.BACKEND_URL}/api/poster?title={encoded_title}&year={year}"
        else:
            return f"{Config.BACKEND_URL}/api/poster?title={encoded_title}"

# FAST INITIALIZATION
app = Quart(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# CORS headers
@app.after_request
async def add_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['X-SK4FiLM-Version'] = '8.0-DUAL-SESSION'
    response.headers['X-Response-Time'] = f"{time.perf_counter():.3f}"
    return response

# ============================================================================
# ✅ DUAL SESSION ARCHITECTURE
# ============================================================================

# GLOBAL SESSIONS
User = None        # ✅ For TEXT channel searches (-1001891090100, -1002024811395)
Bot = None         # ✅ For FILE channel operations (-1001768249569)
user_session_ready = False
bot_session_ready = False

# Database
mongo_client = None
db = None
files_col = None
verification_col = None
poster_col = None

# Components
cache_manager = None
poster_fetcher = None

# CHANNEL CONFIGURATION
CHANNEL_CONFIG = {
    -1001891090100: {'name': 'SK4FiLM Main', 'type': 'text', 'session': 'user'},
    -1002024811395: {'name': 'SK4FiLM Updates', 'type': 'text', 'session': 'user'},
    -1001768249569: {'name': 'SK4FiLM Files', 'type': 'file', 'session': 'bot', 'sync_manage': True}
}

# ============================================================================
# ✅ UTILITY FUNCTIONS (INLINE)
# ============================================================================

def normalize_title(title):
    """Normalize title for consistent search"""
    if not title:
        return ""
    
    title = title.lower()
    title = re.sub(r'\s*(?:\(|\[)?(?:19|20)\d{2}(?:\)|\])?', '', title)
    title = re.sub(r'\.(mp4|mkv|avi|mov|wmv|flv|webm|m4v|3gp)$', '', title)
    title = re.sub(r'\s*(?:480p|720p|1080p|2160p|4k|hd|fullhd|uhd|bluray|dvdrip|webrip|webdl|hdtv|brrip)', '', title)
    title = re.sub(r'[^\w\s]', ' ', title)
    title = re.sub(r'\s+', ' ', title).strip()
    return title

def extract_title_smart(text):
    """Extract title from Telegram message text"""
    if not text:
        return ""
    
    text = re.sub(r'https?://\S+', '', text)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    for line in lines:
        if len(line) > 10:
            clean = re.sub(r'^[▶►▷•\-*›»›]\s*', '', line)
            clean = re.sub(r'\s*[⬇️👇📥🎬✨🔥💥⭐🌟🎥📽️]\s*$', '', clean)
            clean = clean.strip()
            if len(clean) > 10:
                return clean[:200]
    
    return lines[0][:200] if lines else "Untitled"

def extract_title_from_file(filename, caption=None):
    """Extract title from filename or caption"""
    if caption and len(caption) > 10:
        title = extract_title_smart(caption)
        if title and len(title) > 10:
            return title
    
    if filename:
        name = re.sub(r'\.[a-zA-Z0-9]+$', '', filename)
        name = re.sub(r'\s*[\(\[]?\d{3,4}p[\)\]]?', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\s*(?:hd|fullhd|4k|uhd|bluray|webrip|hdtv)', '', name, flags=re.IGNORECASE)
        return name.strip()
    
    return "Unknown Title"

def format_size(size_bytes):
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ("B", "KB", "MB", "GB", "TB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"

def detect_quality(filename):
    """Detect video quality from filename"""
    if not filename:
        return "480p"
    
    filename_lower = filename.lower()
    if '2160p' in filename_lower or '4k' in filename_lower:
        return "2160p"
    elif '1080p' in filename_lower or 'fullhd' in filename_lower:
        return "1080p"
    elif '720p' in filename_lower or 'hd' in filename_lower:
        return "720p"
    elif '480p' in filename_lower:
        return "480p"
    else:
        return "480p"

def is_video_file(filename):
    """Check if file is a video"""
    if not filename:
        return False
    
    video_extensions = {'.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.mpg', '.mpeg'}
    return any(filename.lower().endswith(ext) for ext in video_extensions)

def format_post(text, max_length=500):
    """Format Telegram post for display"""
    if not text:
        return "<p>No content</p>"
    
    text = html.escape(text)
    paragraphs = []
    for p in text.split('\n'):
        p = p.strip()
        if p:
            paragraphs.append(p)
    
    if not paragraphs:
        return "<p>No content</p>"
    
    formatted = ""
    for i, p in enumerate(paragraphs):
        if i == 0 and len(p) > 30:
            formatted += f'<h3>{p[:100]}{"..." if len(p) > 100 else ""}</h3>'
        else:
            if len(p) > max_length:
                p = p[:max_length] + "..."
            formatted += f'<p>{p}</p>'
    
    return formatted

def is_new(date, hours=48):
    """Check if content is new (within hours)"""
    if not date:
        return False
    
    try:
        if isinstance(date, str):
            if 'Z' in date:
                date = date.replace('Z', '+00:00')
            date_obj = datetime.fromisoformat(date)
        elif isinstance(date, datetime):
            date_obj = date
        else:
            return False
        
        time_diff = datetime.now() - date_obj
        return time_diff.total_seconds() < hours * 3600
    except:
        return False

# ============================================================================
# ✅ CACHE MANAGER (INLINE)
# ============================================================================

class CacheManager:
    def __init__(self, config):
        self.config = config
        self.redis_client = None
        self.redis_enabled = False
        self.memory_cache = {}
    
    async def init_redis(self):
        try:
            self.redis_client = redis.from_url(
                self.config.REDIS_URL,
                password=self.config.REDIS_PASSWORD or None,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_keepalive=True
            )
            await self.redis_client.ping()
            self.redis_enabled = True
            logger.info("✅ Redis connected successfully!")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Redis not available: {e}")
            self.redis_enabled = False
            return False
    
    async def get(self, key: str, default=None):
        try:
            if self.redis_enabled and self.redis_client:
                value = await self.redis_client.get(key)
                if value:
                    return json.loads(value)
        except:
            pass
        
        if key in self.memory_cache:
            data, expiry = self.memory_cache[key]
            if expiry > datetime.now():
                return data
        
        return default
    
    async def set(self, key: str, value: Any, expire_seconds: int = 300):
        try:
            if self.redis_enabled and self.redis_client:
                await self.redis_client.setex(key, expire_seconds, json.dumps(value))
        except:
            pass
        
        expiry = datetime.now() + timedelta(seconds=expire_seconds)
        self.memory_cache[key] = (value, expiry)
    
    async def get_search_results(self, query: str, page: int, limit: int):
        cache_key = f"search:{query}:{page}:{limit}"
        return await self.get(cache_key)
    
    async def cache_search_results(self, query: str, page: int, limit: int, data: Dict):
        cache_key = f"search:{query}:{page}:{limit}"
        await self.set(cache_key, data, expire_seconds=600)
    
    async def clear_all(self):
        try:
            if self.redis_enabled and self.redis_client:
                await self.redis_client.flushdb()
        except:
            pass
        self.memory_cache.clear()
    
    async def start_cleanup_task(self):
        async def cleanup():
            while True:
                await asyncio.sleep(300)
                self._cleanup_memory_cache()
        
        asyncio.create_task(cleanup())
    
    def _cleanup_memory_cache(self):
        now = datetime.now()
        expired_keys = [k for k, (_, expiry) in self.memory_cache.items() if expiry < now]
        for key in expired_keys:
            del self.memory_cache[key]
    
    async def stop(self):
        pass

# ============================================================================
# ✅ POSTER FETCHER (INLINE)
# ============================================================================

class PosterFetcher:
    def __init__(self, cache_manager):
        self.cache_manager = cache_manager
        self.memory_cache = {}
    
    async def fetch_poster(self, title: str, year: str = ""):
        cache_key = f"poster:{title}:{year}"
        
        if cache_key in self.memory_cache:
            data, expiry = self.memory_cache[cache_key]
            if expiry > datetime.now():
                return data
        
        if self.cache_manager and self.cache_manager.redis_enabled:
            cached_data = await self.cache_manager.get(cache_key)
            if cached_data:
                self.memory_cache[cache_key] = (cached_data, datetime.now() + timedelta(hours=24))
                return cached_data
        
        # Create fallback poster
        poster_data = self._create_fallback_poster(title, year)
        
        if poster_data:
            self.memory_cache[cache_key] = (poster_data, datetime.now() + timedelta(hours=24))
            if self.cache_manager:
                await self.cache_manager.set(cache_key, poster_data, expire_seconds=7*24*3600)
        
        return poster_data
    
    def _create_fallback_poster(self, title: str, year: str):
        clean_title = ''.join(c for c in title if c.isalnum() or c in ' _-')
        encoded_title = urllib.parse.quote(clean_title[:50])
        
        title_hash = hashlib.md5(title.encode()).hexdigest()
        color = f"#{title_hash[:6]}"
        
        poster_url = f"https://via.placeholder.com/300x450/{color[1:]}/ffffff?text={encoded_title}"
        
        if year:
            poster_url += f"%28{year}%29"
        
        return {
            'url': poster_url,
            'source': 'placeholder',
            'rating': '0.0',
            'year': year,
            'title': title
        }
    
    def clear_cache(self):
        self.memory_cache.clear()

# ============================================================================
# ✅ DUAL SESSION INITIALIZATION
# ============================================================================

@performance_monitor.measure("telegram_init")
async def init_telegram_sessions():
    """Initialize DUAL sessions: User for text, Bot for files"""
    global User, Bot, user_session_ready, bot_session_ready
    
    logger.info("=" * 60)
    logger.info("🚀 DUAL SESSION INITIALIZATION")
    logger.info("=" * 60)
    
    if not PYROGRAM_AVAILABLE:
        logger.error("❌ Pyrogram not installed!")
        return False
    
    # ============================================================================
    # ✅ 1. INITIALIZE USER SESSION (for TEXT channels)
    # ============================================================================
    if Config.API_ID > 0 and Config.API_HASH and Config.USER_SESSION_STRING:
        logger.info("\n👤 Initializing USER Session for TEXT channels...")
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
            
            # Test TEXT channel access
            try:
                chat = await User.get_chat(Config.MAIN_CHANNEL_ID)
                logger.info(f"✅ TEXT Channel Access: {chat.title}")
                user_session_ready = True
            except Exception as e:
                logger.error(f"❌ TEXT Channel access failed: {e}")
                user_session_ready = False
                
        except Exception as e:
            logger.error(f"❌ USER Session failed: {e}")
            user_session_ready = False
    
    # ============================================================================
    # ✅ 2. INITIALIZE BOT SESSION (for FILE channel)
    # ============================================================================
    if Config.BOT_TOKEN:
        logger.info("\n🤖 Initializing BOT Session for FILE channel...")
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
            
            # Test FILE channel access
            try:
                chat = await Bot.get_chat(Config.FILE_CHANNEL_ID)
                logger.info(f"✅ FILE Channel Access: {chat.title}")
                bot_session_ready = True
            except Exception as e:
                logger.error(f"❌ FILE Channel access failed: {e}")
                bot_session_ready = False
                
        except Exception as e:
            logger.error(f"❌ BOT Session failed: {e}")
            bot_session_ready = False
    
    # ============================================================================
    # ✅ 3. SUMMARY
    # ============================================================================
    logger.info("\n" + "=" * 60)
    logger.info("📊 SESSION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"USER Session (TEXT): {'✅ READY' if user_session_ready else '❌ NOT READY'}")
    logger.info(f"BOT Session (FILE): {'✅ READY' if bot_session_ready else '❌ NOT READY'}")
    logger.info(f"TEXT Channels: {Config.TEXT_CHANNEL_IDS}")
    logger.info(f"FILE Channel: {Config.FILE_CHANNEL_ID}")
    
    return user_session_ready or bot_session_ready

# ============================================================================
# ✅ MONGODB INITIALIZATION
# ============================================================================

@performance_monitor.measure("mongodb_init")
async def init_mongodb():
    global mongo_client, db, files_col, verification_col, poster_col
    
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
        poster_col = db.posters
        
        logger.info("✅ MongoDB OK")
        return True
        
    except asyncio.TimeoutError:
        logger.error("❌ MongoDB connection timeout")
        return False
    except Exception as e:
        logger.error(f"❌ MongoDB error: {e}")
        return False

# ============================================================================
# ✅ CACHE FUNCTIONS
# ============================================================================

@lru_cache(maxsize=10000)
def channel_name_cached(cid):
    return CHANNEL_CONFIG.get(cid, {}).get('name', f"Channel {cid}")

@lru_cache(maxsize=5000)
def normalize_title_cached(title: str) -> str:
    return normalize_title(title)

@lru_cache(maxsize=1000)
def channel_name(channel_id):
    return channel_name_cached(channel_id)

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
# ✅ FILE INDEXING WITH BOT SESSION
# ============================================================================

async def generate_file_hash(message):
    """Generate unique hash for file to detect duplicates"""
    try:
        hash_parts = []
        
        if message.document:
            file_attrs = message.document
            hash_parts.append(f"doc_{file_attrs.file_id}")
            if file_attrs.file_name:
                hash_parts.append(f"name_{hashlib.md5(file_attrs.file_name.encode()).hexdigest()[:8]}")
            if file_attrs.file_size:
                hash_parts.append(f"size_{file_attrs.file_size}")
        elif message.video:
            file_attrs = message.video
            hash_parts.append(f"vid_{file_attrs.file_id}")
            if file_attrs.file_name:
                hash_parts.append(f"name_{hashlib.md5(file_attrs.file_name.encode()).hexdigest()[:8]}")
            if file_attrs.file_size:
                hash_parts.append(f"size_{file_attrs.file_size}")
            if hasattr(file_attrs, 'duration'):
                hash_parts.append(f"dur_{file_attrs.duration}")
        else:
            return None
        
        if message.caption:
            caption_hash = hashlib.md5(message.caption.encode()).hexdigest()[:12]
            hash_parts.append(f"cap_{caption_hash}")
        
        return "_".join(hash_parts)
    except Exception as e:
        logger.debug(f"Hash generation error: {e}")
        return None

@performance_monitor.measure("smart_file_indexing")
async def index_single_file_smart(message):
    """Index single file using BOT session"""
    try:
        if not files_col or not Bot or not bot_session_ready:
            logger.error("❌ Bot session not ready for indexing")
            return False
        
        if not message or (not message.document and not message.video):
            return False
        
        # Check if already exists
        existing_by_id = await files_col.find_one({
            'channel_id': Config.FILE_CHANNEL_ID,
            'message_id': message.id
        }, {'_id': 1})
        
        if existing_by_id:
            logger.debug(f"📝 Already indexed: {message.id}")
            return True
        
        # Extract title
        caption = message.caption if hasattr(message, 'caption') else None
        file_name = None
        
        if message.document:
            file_name = message.document.file_name
        elif message.video:
            file_name = message.video.file_name
        
        title = extract_title_from_file(file_name, caption)
        if not title:
            logger.debug(f"📝 Skipping - No title: {message.id}")
            return False
        
        normalized_title = normalize_title_cached(title)
        
        # Create document
        doc = {
            'channel_id': Config.FILE_CHANNEL_ID,
            'message_id': message.id,
            'title': title,
            'normalized_title': normalized_title,
            'date': message.date,
            'indexed_at': datetime.now(),
            'last_checked': datetime.now(),
            'is_video_file': False,
            'thumbnail': None,
            'thumbnail_source': 'none',
            'file_id': None,
            'file_size': 0,
            'file_hash': None,
            'status': 'active'
        }
        
        # Add file-specific data
        if message.document:
            doc.update({
                'file_name': message.document.file_name or '',
                'quality': detect_quality(message.document.file_name or ''),
                'is_video_file': is_video_file(message.document.file_name or ''),
                'caption': caption or '',
                'mime_type': message.document.mime_type or '',
                'file_id': message.document.file_id,
                'file_size': message.document.file_size or 0
            })
            doc['file_hash'] = await generate_file_hash(message)
        elif message.video:
            doc.update({
                'file_name': message.video.file_name or 'video.mp4',
                'quality': detect_quality(message.video.file_name or ''),
                'is_video_file': True,
                'caption': caption or '',
                'duration': message.video.duration if hasattr(message.video, 'duration') else 0,
                'width': message.video.width if hasattr(message.video, 'width') else 0,
                'height': message.video.height if hasattr(message.video, 'height') else 0,
                'file_id': message.video.file_id,
                'file_size': message.video.file_size or 0
            })
            doc['file_hash'] = await generate_file_hash(message)
        else:
            return False
        
        # Insert into MongoDB
        try:
            await files_col.insert_one(doc)
            
            file_type = "📹 Video" if doc['is_video_file'] else "📄 File"
            size_str = format_size(doc['file_size']) if doc['file_size'] > 0 else "Unknown"
            
            logger.info(f"✅ {file_type} indexed via BOT: {title}")
            logger.info(f"   📊 Size: {size_str} | Quality: {doc.get('quality', 'Unknown')}")
            
            return True
        except Exception as e:
            if "duplicate key error" in str(e).lower():
                return True
            else:
                logger.error(f"❌ Insert error: {e}")
                return False
        
    except Exception as e:
        logger.error(f"❌ Indexing error: {e}")
        return False

async def index_files_background_smart():
    """Background indexing using BOT session"""
    if not Bot or not files_col or not bot_session_ready:
        logger.warning("⚠️ Bot session not ready for indexing")
        return
    
    logger.info("📁 Starting background indexing via BOT session...")
    
    try:
        # Setup indexes
        if files_col is not None:
            try:
                await files_col.create_index(
                    [("channel_id", 1), ("message_id", 1)],
                    unique=True,
                    name="channel_message_unique"
                )
                await files_col.create_index(
                    [("normalized_title", "text")],
                    name="text_search_index"
                )
            except:
                pass
        
        # Get last indexed message
        last_indexed = await files_col.find_one(
            {"channel_id": Config.FILE_CHANNEL_ID}, 
            sort=[('message_id', -1)],
            projection={'message_id': 1}
        )
        
        last_message_id = last_indexed['message_id'] if last_indexed else 0
        
        logger.info(f"🔄 Starting from message ID: {last_message_id}")
        
        # Fetch and process new messages
        total_indexed = 0
        messages = []
        
        async for msg in Bot.get_chat_history(Config.FILE_CHANNEL_ID, limit=100):
            if msg.id <= last_message_id:
                break
            
            if msg and (msg.document or msg.video):
                messages.append(msg)
        
        messages.reverse()
        logger.info(f"📥 Found {len(messages)} new files to index")
        
        for msg in messages:
            try:
                success = await index_single_file_smart(msg)
                if success:
                    total_indexed += 1
                await asyncio.sleep(0.2)
            except Exception as e:
                logger.error(f"❌ Error processing {msg.id}: {e}")
                continue
        
        if total_indexed > 0:
            logger.info(f"✅ Indexing complete: {total_indexed} new files")
        else:
            logger.info("✅ No new files to index")
        
    except Exception as e:
        logger.error(f"❌ Background indexing error: {e}")

# ============================================================================
# ✅ SYNC MANAGEMENT WITH BOT SESSION
# ============================================================================

class ChannelSyncManager:
    def __init__(self):
        self.is_monitoring = False
        self.monitoring_task = None
        self.deleted_count = 0
        self.last_sync = time.time()
    
    async def start_sync_monitoring(self):
        """Start sync monitoring using BOT session"""
        if not Bot or not bot_session_ready:
            logger.warning("⚠️ Bot session not ready for sync")
            return
        
        if self.is_monitoring:
            return
        
        logger.info("👁️ Starting sync monitoring via BOT...")
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
                await self.sync_deletions_from_telegram()
                await asyncio.sleep(Config.MONITOR_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Sync error: {e}")
                await asyncio.sleep(60)
    
    async def sync_deletions_from_telegram(self):
        """Sync deletions using BOT session"""
        try:
            if not files_col or not Bot:
                return
            
            current_time = time.time()
            if current_time - self.last_sync < 300:
                return
            
            self.last_sync = current_time
            
            # Get message IDs from MongoDB
            cursor = files_col.find(
                {"channel_id": Config.FILE_CHANNEL_ID},
                {"message_id": 1, "_id": 0}
            )
            
            message_ids_in_db = []
            async for doc in cursor:
                msg_id = doc.get('message_id')
                if msg_id:
                    message_ids_in_db.append(msg_id)
            
            if not message_ids_in_db:
                return
            
            deleted_count = 0
            batch_size = 50
            
            for i in range(0, len(message_ids_in_db), batch_size):
                batch = message_ids_in_db[i:i + batch_size]
                
                try:
                    # Check if messages exist using BOT
                    messages = await Bot.get_messages(Config.FILE_CHANNEL_ID, batch)
                    
                    existing_ids = set()
                    if isinstance(messages, list):
                        for msg in messages:
                            if msg and hasattr(msg, 'id'):
                                existing_ids.add(msg.id)
                    elif messages and hasattr(messages, 'id'):
                        existing_ids.add(messages.id)
                    
                    # Find deleted IDs
                    deleted_ids = [msg_id for msg_id in batch if msg_id not in existing_ids]
                    
                    # Delete from MongoDB
                    if deleted_ids:
                        result = await files_col.delete_many({
                            "channel_id": Config.FILE_CHANNEL_ID,
                            "message_id": {"$in": deleted_ids}
                        })
                        
                        if result.deleted_count > 0:
                            deleted_count += result.deleted_count
                            self.deleted_count += result.deleted_count
                
                except Exception as e:
                    logger.error(f"❌ Batch check error: {e}")
                    continue
            
            if deleted_count > 0:
                logger.info(f"✅ Sync: {deleted_count} files deleted")
            
        except Exception as e:
            logger.error(f"❌ Sync deletions error: {e}")
    
    async def manual_sync(self):
        await self.sync_deletions_from_telegram()

channel_sync_manager = ChannelSyncManager()

# ============================================================================
# ✅ TEXT CHANNEL SEARCH WITH USER SESSION
# ============================================================================

@async_cache_with_ttl(maxsize=1000, ttl=3600)
async def extract_title_from_telegram_msg_cached(msg):
    try:
        caption = msg.caption if hasattr(msg, 'caption') else None
        file_name = None
        
        if msg.document:
            file_name = msg.document.file_name
        elif msg.video:
            file_name = msg.video.file_name
        
        return extract_title_from_file(file_name, caption)
    except Exception as e:
        logger.error(f"Title extraction error: {e}")
        return None

@performance_monitor.measure("home_movies_telegram")
@async_cache_with_ttl(maxsize=1, ttl=60)
async def get_home_movies_telegram(limit=30):
    """Get movies from TEXT channels using USER session"""
    try:
        if not User or not user_session_ready:
            return []
        
        movies = []
        seen_titles = set()
        
        logger.info(f"🎬 Fetching movies via USER session...")
        
        async for msg in User.get_chat_history(Config.MAIN_CHANNEL_ID, limit=limit * 2):
            if msg and msg.text and len(msg.text) > 20:
                title = extract_title_smart(msg.text)
                
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    
                    year_match = re.search(r'\b(19|20)\d{2}\b', title)
                    year = year_match.group() if year_match else ""
                    
                    clean_title = re.sub(r'\s+\(\d{4}\)$', '', title)
                    clean_title = re.sub(r'\s+\d{4}$', '', clean_title)
                    
                    movies.append({
                        'title': clean_title,
                        'original_title': title,
                        'year': year,
                        'date': msg.date.isoformat() if isinstance(msg.date, datetime) else str(msg.date),
                        'is_new': is_new(msg.date) if msg.date else False,
                        'channel': channel_name_cached(Config.MAIN_CHANNEL_ID),
                        'channel_id': Config.MAIN_CHANNEL_ID,
                        'message_id': msg.id,
                        'has_poster': True,
                        'poster_url': Config.get_poster(clean_title, year),
                        'poster_source': 'telegram',
                        'poster_rating': '0.0'
                    })
                    
                    if len(movies) >= limit:
                        break
        
        logger.info(f"✅ Fetched {len(movies)} movies via USER")
        return movies[:limit]
        
    except Exception as e:
        logger.error(f"❌ Home movies error: {e}")
        return []

# ============================================================================
# ✅ COMBINED SEARCH (USER + BOT)
# ============================================================================

@performance_monitor.measure("multi_channel_search")
@async_cache_with_ttl(maxsize=500, ttl=300)
async def search_movies_multi_channel(query, limit=12, page=1):
    """COMBINED search: USER for text, BOT for files"""
    offset = (page - 1) * limit
    
    # Try cache first
    cache_key = f"search:{query}:{page}:{limit}"
    if cache_manager and cache_manager.redis_enabled:
        cached_data = await cache_manager.get(cache_key)
        if cached_data:
            logger.info(f"✅ Cache HIT for: {query}")
            return cached_data
    
    logger.info(f"🔍 DUAL search for: {query}")
    
    query_lower = query.lower()
    posts_dict = {}    # From USER session (text channels)
    files_dict = {}    # From BOT session (file channel)
    
    # ============================================================================
    # ✅ 1. SEARCH TEXT CHANNELS (USER SESSION)
    # ============================================================================
    if user_session_ready and User:
        async def search_text_channel(channel_id):
            channel_posts = {}
            try:
                cname = channel_name_cached(channel_id)
                async for msg in User.search_messages(channel_id, query=query, limit=10):
                    if msg and msg.text and len(msg.text) > 15:
                        title = extract_title_smart(msg.text)
                        if title and query_lower in title.lower():
                            norm_title = normalize_title_cached(title)
                            if norm_title not in channel_posts:
                                channel_posts[norm_title] = {
                                    'title': title,
                                    'content': format_post(msg.text),
                                    'channel': cname,
                                    'channel_id': channel_id,
                                    'message_id': msg.id,
                                    'date': msg.date.isoformat() if isinstance(msg.date, datetime) else str(msg.date),
                                    'is_new': is_new(msg.date) if msg.date else False,
                                    'has_file': False,
                                    'has_post': True,
                                    'quality_options': {},
                                    'thumbnail': None
                                }
            except Exception as e:
                logger.error(f"Text search error in {channel_id}: {e}")
            return channel_posts
        
        # Search text channels concurrently
        tasks = [search_text_channel(channel_id) for channel_id in Config.TEXT_CHANNEL_IDS]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, dict):
                posts_dict.update(result)
    
    # ============================================================================
    # ✅ 2. SEARCH FILE CHANNEL (BOT SESSION) via MongoDB
    # ============================================================================
    if files_col is not None:
        try:
            cursor = files_col.find(
                {
                    "channel_id": Config.FILE_CHANNEL_ID,
                    "$or": [
                        {"title": {"$regex": query, "$options": "i"}},
                        {"normalized_title": {"$regex": query, "$options": "i"}},
                        {"file_name": {"$regex": query, "$options": "i"}}
                    ]
                },
                {
                    'title': 1,
                    'normalized_title': 1,
                    'quality': 1,
                    'file_size': 1,
                    'file_name': 1,
                    'is_video_file': 1,
                    'channel_id': 1,
                    'message_id': 1,
                    'date': 1,
                    '_id': 0
                }
            ).limit(limit * 2)
            
            async for doc in cursor:
                try:
                    norm_title = doc.get('normalized_title', normalize_title_cached(doc['title']))
                    quality = doc['quality']
                    
                    if norm_title not in files_dict:
                        files_dict[norm_title] = {
                            'title': doc['title'], 
                            'quality_options': {}, 
                            'date': doc['date'].isoformat() if isinstance(doc['date'], datetime) else doc['date'],
                            'is_video_file': doc.get('is_video_file', False),
                            'channel_id': doc.get('channel_id'),
                            'channel_name': channel_name_cached(doc.get('channel_id'))
                        }
                    
                    if quality not in files_dict[norm_title]['quality_options']:
                        files_dict[norm_title]['quality_options'][quality] = {
                            'file_id': f"{doc.get('channel_id', Config.FILE_CHANNEL_ID)}_{doc.get('message_id')}_{quality}",
                            'file_size': doc['file_size'],
                            'file_name': doc['file_name'],
                            'is_video': doc.get('is_video_file', False),
                            'channel_id': doc.get('channel_id'),
                            'message_id': doc.get('message_id')
                        }
                except:
                    continue
        except Exception as e:
            logger.error(f"File search error: {e}")
    
    # ============================================================================
    # ✅ 3. MERGE RESULTS
    # ============================================================================
    merged = {}
    
    # Add text posts from USER session
    for norm_title, post_data in posts_dict.items():
        merged[norm_title] = post_data
    
    # Add/update with file information from BOT session
    for norm_title, file_data in files_dict.items():
        if norm_title in merged:
            merged[norm_title]['has_file'] = True
            merged[norm_title]['quality_options'] = file_data['quality_options']
        else:
            merged[norm_title] = {
                'title': file_data['title'],
                'content': f"<p>File available in {file_data['channel_name']}</p>",
                'channel': file_data.get('channel_name', 'SK4FiLM Files'),
                'date': file_data['date'],
                'is_new': False,
                'has_file': True,
                'has_post': False,
                'quality_options': file_data['quality_options'],
                'thumbnail': None,
                'thumbnail_source': 'none'
            }
    
    # Sort and paginate
    results_list = list(merged.values())
    results_list.sort(key=lambda x: (
        not x.get('is_new', False),
        not x['has_file'],
        x['date']
    ), reverse=True)
    
    total = len(results_list)
    paginated = results_list[offset:offset + limit]
    
    # Add posters
    for result in paginated:
        title = result.get('title', '')
        year_match = re.search(r'\b(19|20)\d{2}\b', title)
        year = year_match.group() if year_match else ""
        result['poster_url'] = Config.get_poster(title, year)
        result['poster_source'] = 'custom'
        result['poster_rating'] = '0.0'
        result['has_poster'] = True
    
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
            'user_session_used': user_session_ready,
            'bot_session_used': bot_session_ready,
            'text_channels': len(Config.TEXT_CHANNEL_IDS),
            'file_channel': Config.FILE_CHANNEL_ID,
            'query': query,
            'cache_hit': False
        }
    }
    
    # Cache results
    if cache_manager:
        await cache_manager.cache_search_results(query, page, limit, result_data)
    
    logger.info(f"✅ DUAL search complete: {len(paginated)} results")
    return result_data

# ============================================================================
# ✅ SINGLE POST API
# ============================================================================

async def get_single_post_api(channel_id, message_id):
    """Get single post - uses appropriate session based on channel"""
    try:
        # Determine which session to use
        if channel_id == Config.FILE_CHANNEL_ID:
            # Use BOT session for file channel
            if not Bot or not bot_session_ready:
                return None
            msg = await Bot.get_messages(channel_id, message_id)
        else:
            # Use USER session for text channels
            if not User or not user_session_ready:
                return None
            msg = await User.get_messages(channel_id, message_id)
        
        if msg and msg.text:
            title = extract_title_smart(msg.text)
            if not title:
                title = msg.text.split('\n')[0][:60] if msg.text else "Movie Post"
            
            normalized_title = normalize_title(title)
            quality_options = {}
            has_file = False
            
            # Search for files with same title in FILE_CHANNEL_ID
            if files_col is not None:
                cursor = files_col.find({
                    'channel_id': Config.FILE_CHANNEL_ID,
                    'normalized_title': normalized_title
                })
                async for doc in cursor:
                    quality = doc.get('quality', '480p')
                    if quality not in quality_options:
                        quality_options[quality] = {
                            'file_id': f"{doc.get('channel_id', Config.FILE_CHANNEL_ID)}_{doc.get('message_id')}_{quality}",
                            'file_size': doc.get('file_size', 0),
                            'file_name': doc.get('file_name', 'video.mp4'),
                            'is_video': doc.get('is_video_file', False),
                            'channel_id': doc.get('channel_id'),
                            'message_id': doc.get('message_id')
                        }
                        has_file = True
            
            # Get poster
            year_match = re.search(r'\b(19|20)\d{2}\b', title)
            year = year_match.group() if year_match else ""
            
            post_data = {
                'title': title,
                'content': format_post(msg.text),
                'channel': channel_name(channel_id),
                'channel_id': channel_id,
                'message_id': message_id,
                'date': msg.date.isoformat() if isinstance(msg.date, datetime) else str(msg.date),
                'is_new': is_new(msg.date) if msg.date else False,
                'has_file': has_file,
                'quality_options': quality_options,
                'views': getattr(msg, 'views', 0),
                'thumbnail': None,
                'thumbnail_source': 'none',
                'poster_url': Config.get_poster(title, year),
                'poster_source': 'custom',
                'poster_rating': '0.0'
            }
            
            return post_data
        
        return None
        
    except Exception as e:
        logger.error(f"Single post API error: {e}")
        return None

# ============================================================================
# ✅ API FUNCTIONS
# ============================================================================

@performance_monitor.measure("search_api")
async def search_movies_api(query, limit=12, page=1):
    try:
        result_data = await search_movies_multi_channel(query, limit, page)
        return result_data
    except Exception as e:
        logger.error(f"Search API error: {e}")
        return {
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
                'error': True,
                'query': query
            }
        }

@performance_monitor.measure("home_movies")
@async_cache_with_ttl(maxsize=1, ttl=300)
async def get_home_movies_live():
    try:
        movies = await get_home_movies_telegram(limit=30)
        return movies
    except Exception as e:
        logger.error(f"Home movies error: {e}")
        return []

async def get_index_status_api():
    try:
        total_files = await files_col.count_documents({}) if files_col is not None else 0
        video_files = await files_col.count_documents({'is_video_file': True}) if files_col is not None else 0
        
        if files_col is not None:
            file_channel_files = await files_col.count_documents({
                "channel_id": Config.FILE_CHANNEL_ID
            })
        else:
            file_channel_files = 0
        
        return {
            'indexed_files': total_files,
            'video_files': video_files,
            'file_channel_files': file_channel_files,
            'sync_monitoring': channel_sync_manager.is_monitoring,
            'deleted_by_sync': channel_sync_manager.deleted_count,
            'user_session_ready': user_session_ready,
            'bot_session_ready': bot_session_ready,
            'last_update': datetime.now().isoformat(),
            'status': 'active' if (user_session_ready or bot_session_ready) else 'inactive'
        }
    except Exception as e:
        logger.error(f"Index status API error: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }

# ============================================================================
# ✅ MAIN INITIALIZATION
# ============================================================================

@performance_monitor.measure("system_init")
async def init_system():
    start_time = time.time()
    
    try:
        logger.info("🚀 Starting SK4FiLM v8.0 - DUAL SESSION ARCHITECTURE...")
        logger.info("👤 User Session: TEXT channels")
        logger.info("🤖 Bot Session: FILE channel")
        
        # Initialize MongoDB
        mongo_ok = await init_mongodb()
        if not mongo_ok:
            logger.warning("⚠️ MongoDB connection failed")
        
        # Initialize Cache Manager
        global cache_manager, poster_fetcher
        cache_manager = CacheManager(Config)
        redis_ok = await cache_manager.init_redis()
        if redis_ok:
            logger.info("✅ Cache Manager initialized")
            await cache_manager.start_cleanup_task()
        
        # Initialize Poster Fetcher
        poster_fetcher = PosterFetcher(cache_manager)
        logger.info("✅ Poster Fetcher initialized")
        
        # Initialize Telegram DUAL Sessions
        if PYROGRAM_AVAILABLE:
            telegram_ok = await init_telegram_sessions()
            if not telegram_ok:
                logger.warning("⚠️ Telegram sessions failed")
        else:
            logger.warning("⚠️ Pyrogram not available")
        
        # Start background tasks
        if bot_session_ready and files_col:
            asyncio.create_task(index_files_background_smart())
            logger.info("✅ Started BOT indexing")
            await channel_sync_manager.start_sync_monitoring()
            logger.info("✅ Started BOT sync monitoring")
        
        init_time = time.time() - start_time
        logger.info(f"⚡ SK4FiLM Started in {init_time:.2f}s")
        
        logger.info("🔧 DUAL SESSION ARCHITECTURE:")
        logger.info(f"   • USER Session: {'✅ READY' if user_session_ready else '❌ NOT READY'}")
        logger.info(f"   • BOT Session: {'✅ READY' if bot_session_ready else '❌ NOT READY'}")
        logger.info(f"   • TEXT Channels: {len(Config.TEXT_CHANNEL_IDS)}")
        logger.info(f"   • FILE Channel: {Config.FILE_CHANNEL_ID}")
        logger.info(f"   • Cache: {'✅ ENABLED' if redis_ok else '❌ DISABLED'}")
        
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
    tf = await files_col.count_documents({}) if files_col is not None else 0
    video_files = await files_col.count_documents({'is_video_file': True}) if files_col is not None else 0
    
    return jsonify({
        'status': 'healthy',
        'service': 'SK4FiLM v8.0 - DUAL SESSION',
        'sessions': {
            'user_session': {
                'ready': user_session_ready,
                'channels': Config.TEXT_CHANNEL_IDS
            },
            'bot_session': {
                'ready': bot_session_ready,
                'channel': Config.FILE_CHANNEL_ID
            }
        },
        'database': {
            'total_files': tf, 
            'video_files': video_files,
            'connected': files_col is not None
        },
        'cache': {
            'redis_enabled': cache_manager.redis_enabled if cache_manager else False
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
            'bot': bot_session_ready
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/movies', methods=['GET'])
@performance_monitor.measure("movies_endpoint")
async def api_movies():
    try:
        movies = await get_home_movies_live()
        
        return jsonify({
            'status': 'success' if movies else 'empty',
            'movies': movies,
            'total': len(movies),
            'source': 'telegram',
            'session_used': 'user',
            'channel_id': Config.MAIN_CHANNEL_ID,
            'timestamp': datetime.now().isoformat(),
            'cache_hit': True if movies else False
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
        limit = int(request.args.get('limit', 12))
        
        if len(query) < 2:
            return jsonify({
                'status': 'error',
                'message': 'Query must be at least 2 characters'
            }), 400
        
        result_data = await search_movies_api(query, limit, page)
        
        return jsonify({
            'status': 'success',
            'query': query,
            'results': result_data['results'],
            'pagination': result_data['pagination'],
            'search_metadata': result_data.get('search_metadata', {}),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Search API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/post', methods=['GET'])
async def api_post():
    try:
        channel_id = int(request.args.get('channel', Config.MAIN_CHANNEL_ID))
        message_id = int(request.args.get('message', 0))
        
        if message_id <= 0:
            return jsonify({
                'status': 'error',
                'message': 'Invalid message ID'
            }), 400
        
        post_data = await get_single_post_api(channel_id, message_id)
        
        if post_data:
            session_used = 'bot' if channel_id == Config.FILE_CHANNEL_ID else 'user'
            return jsonify({
                'status': 'success',
                'post': post_data,
                'session_used': session_used,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Post not found'
            }), 404
            
    except Exception as e:
        logger.error(f"Post API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/poster', methods=['GET'])
async def api_poster():
    try:
        title = request.args.get('title', '').strip()
        year = request.args.get('year', '')
        
        if not title:
            return jsonify({
                'status': 'error',
                'message': 'Title is required'
            }), 400
        
        if poster_fetcher:
            poster_data = await poster_fetcher.fetch_poster(title, year)
            if poster_data:
                return jsonify({
                    'status': 'success',
                    'poster': poster_data,
                    'timestamp': datetime.now().isoformat()
                })
        
        return jsonify({
            'status': 'success',
            'poster': {
                'url': Config.get_poster(title, year),
                'source': 'custom',
                'rating': '0.0',
                'year': year,
                'title': title
            },
            'timestamp': datetime.now().isoformat()
        })
                
    except Exception as e:
        logger.error(f"Poster API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/index_status', methods=['GET'])
async def api_index_status():
    try:
        status_data = await get_index_status_api()
        return jsonify({
            'status': 'success',
            'indexing': status_data,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Index status API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/sync/status', methods=['GET'])
async def api_sync_status():
    try:
        return jsonify({
            'status': 'success',
            'sync': {
                'monitoring': channel_sync_manager.is_monitoring,
                'deleted_count': channel_sync_manager.deleted_count,
                'file_channel': Config.FILE_CHANNEL_ID,
                'session_used': 'bot',
                'bot_session_ready': bot_session_ready,
                'timestamp': datetime.now().isoformat()
            }
        })
    except Exception as e:
        logger.error(f"Sync status API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/telegram/status', methods=['GET'])
async def api_telegram_status():
    try:
        return jsonify({
            'status': 'success',
            'sessions': {
                'user': {
                    'ready': user_session_ready,
                    'channels': Config.TEXT_CHANNEL_IDS
                },
                'bot': {
                    'ready': bot_session_ready,
                    'channel': Config.FILE_CHANNEL_ID
                }
            },
            'architecture': 'dual-session',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Telegram status API error: {e}")
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
    logger.info("🛑 Shutting down SK4FiLM...")
    
    shutdown_tasks = []
    
    await channel_sync_manager.stop_sync_monitoring()
    
    if User:
        shutdown_tasks.append(User.stop())
    
    if Bot:
        shutdown_tasks.append(Bot.stop())
    
    if cache_manager:
        shutdown_tasks.append(cache_manager.stop())
    
    if shutdown_tasks:
        await asyncio.gather(*shutdown_tasks, return_exceptions=True)
    
    if mongo_client:
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
    
    logger.info(f"🌐 Starting Quart server on port {Config.WEB_SERVER_PORT}...")
    
    asyncio.run(serve(app, config))
