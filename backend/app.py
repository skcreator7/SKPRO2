# ============================================================================
# 🚀 SK4FiLM v10.0 - ULTRA OPTIMIZED BACKEND
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

try:
    from premium import PremiumSystem, PremiumTier
    logger.debug("✅ Premium module imported")
except ImportError as e:
    logger.error(f"❌ Premium module import error: {e}")
    PremiumSystem = None
    PremiumTier = None

# ============================================================================
# ✅ CONFIGURATION - SIMPLIFIED
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
    MAX_CONCURRENT_REQUESTS = int(os.environ.get("MAX_CONCURRENT_REQUESTS", "10"))
    CACHE_TTL = int(os.environ.get("CACHE_TTL", "300"))
    REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "10"))
    
    # Fallback Poster
    FALLBACK_POSTER = "https://iili.io/fAeIwv9.th.png"
    
    # SEARCH SETTINGS
    SEARCH_MIN_QUERY_LENGTH = 2
    SEARCH_RESULTS_PER_PAGE = 12
    SEARCH_CACHE_TTL = 600

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
    response.headers['X-SK4FiLM-Version'] = '10.0-ULTRA'
    response.headers['X-Response-Time'] = f"{time.perf_counter():.3f}"
    return response

# ============================================================================
# ✅ GLOBAL COMPONENTS
# ============================================================================

# Database
mongo_client = None
db = None
files_col = None

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
    
    def get_stats(self):
        return self.measurements

performance_monitor = PerformanceMonitor()

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
# ✅ UTILITY FUNCTIONS
# ============================================================================

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

# ============================================================================
# ✅ THUMBNAIL FETCHER
# ============================================================================

class SimplePosterFetcher:
    """Simple poster fetcher - works directly"""
    
    def __init__(self, config, cache_manager=None):
        self.config = config
        self.cache_manager = cache_manager
        self.stats = {
            'total_requests': 0,
            'successful': 0,
            'failed': 0,
            'cache_hits': 0
        }
        
    async def fetch_poster(self, title, year=None, quality=None):
        """Fetch poster for movie"""
        self.stats['total_requests'] += 1
        
        # Check cache first
        if self.cache_manager and self.cache_manager.redis_enabled:
            cache_key = f"poster:{title}"
            cached = await self.cache_manager.get(cache_key)
            if cached:
                self.stats['cache_hits'] += 1
                return cached
        
        # Try TMDB first (public API key)
        poster_data = await self._fetch_from_tmdb(title, year)
        
        # If TMDB fails, use fallback
        if not poster_data:
            poster_data = {
                'poster_url': self.config.FALLBACK_POSTER,
                'source': 'fallback',
                'rating': '0.0',
                'year': year or '',
                'title': title,
                'quality': quality or 'unknown'
            }
            self.stats['failed'] += 1
        else:
            self.stats['successful'] += 1
        
        # Cache the result
        if poster_data and self.cache_manager and self.cache_manager.redis_enabled:
            await self.cache_manager.set(
                f"poster:{title}", 
                poster_data, 
                expire_seconds=86400
            )
        
        return poster_data
    
    async def _fetch_from_tmdb(self, title, year=None):
        """Fetch from TMDB API"""
        try:
            # Clean title
            clean_title = re.sub(r'\s*\(\d{4}\)$', '', title)
            clean_title = re.sub(r'\s+\d{4}$', '', clean_title)
            
            url = "https://api.themoviedb.org/3/search/movie"
            params = {
                'api_key': self.config.TMDB_API_KEY,
                'query': clean_title,
                'language': 'en-US',
                'page': 1,
                'include_adult': False
            }
            
            if year:
                params['year'] = year
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get('results') and len(data['results']) > 0:
                            movie = data['results'][0]
                            
                            poster_path = movie.get('poster_path')
                            if poster_path:
                                poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
                                
                                return {
                                    'poster_url': poster_url,
                                    'source': 'tmdb',
                                    'rating': str(movie.get('vote_average', '0.0')),
                                    'year': str(movie.get('release_date', '')[:4]),
                                    'title': movie.get('title', title),
                                    'quality': 'unknown'
                                }
        
        except Exception as e:
            logger.debug(f"TMDB fetch error: {e}")
        
        return None
    
    def get_stats(self):
        return self.stats

# Initialize poster fetcher
poster_fetcher = None

# ============================================================================
# ✅ OPTIMIZED FILE INDEXING
# ============================================================================

class FileIndexingManager:
    """Simple file indexing manager"""
    
    def __init__(self):
        self.is_running = False
        self.indexing_task = None
        self.total_indexed = 0
        self.last_run = None
    
    async def start_indexing(self):
        """Start file indexing"""
        if self.is_running:
            return
        
        logger.info("🚀 Starting file indexing...")
        self.is_running = True
        self.indexing_task = asyncio.create_task(self._indexing_loop())
    
    async def stop_indexing(self):
        """Stop indexing"""
        self.is_running = False
        if self.indexing_task:
            self.indexing_task.cancel()
            try:
                await self.indexing_task
            except:
                pass
    
    async def _indexing_loop(self):
        """Main indexing loop"""
        while self.is_running:
            try:
                await self._run_indexing_cycle()
                await asyncio.sleep(300)  # Wait 5 minutes between cycles
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Indexing error: {e}")
                await asyncio.sleep(300)
    
    async def _run_indexing_cycle(self):
        """Run one indexing cycle"""
        if not user_session_ready or User is None or files_col is None:
            return
        
        try:
            # Get last indexed message
            last_indexed = await files_col.find_one(
                {"channel_id": Config.FILE_CHANNEL_ID}, 
                sort=[('message_id', -1)],
                projection={'message_id': 1}
            )
            
            last_message_id = last_indexed['message_id'] if last_indexed else 0
            
            logger.info(f"📥 Fetching new messages from {last_message_id}...")
            
            # Fetch new messages
            indexed_count = 0
            try:
                async for msg in User.get_chat_history(
                    Config.FILE_CHANNEL_ID, 
                    limit=50
                ):
                    if msg.id <= last_message_id:
                        break
                    
                    if msg and (msg.document or msg.video):
                        success = await index_single_file(msg)
                        if success:
                            indexed_count += 1
            except Exception as e:
                logger.error(f"Error fetching messages: {e}")
            
            if indexed_count > 0:
                self.total_indexed += indexed_count
                logger.info(f"✅ Indexed {indexed_count} new files")
            
            self.last_run = datetime.now()
            
        except Exception as e:
            logger.error(f"Indexing cycle error: {e}")

# Initialize indexing manager
file_indexing_manager = FileIndexingManager()

async def index_single_file(message):
    """Index a single file"""
    try:
        if not message or (not message.document and not message.video):
            return False
        
        # Check if already indexed
        existing = await files_col.find_one({
            'channel_id': Config.FILE_CHANNEL_ID,
            'message_id': message.id
        })
        
        if existing:
            return False
        
        # Extract file info
        caption = message.caption if hasattr(message, 'caption') else None
        file_name = None
        file_size = 0
        file_id = None
        
        if message.document:
            file_name = message.document.file_name
            file_size = message.document.file_size or 0
            file_id = message.document.file_id
        elif message.video:
            file_name = message.video.file_name or 'video.mp4'
            file_size = message.video.file_size or 0
            file_id = message.video.file_id
        
        # Extract title
        if file_name:
            # Remove extension
            name = os.path.splitext(file_name)[0]
            # Clean up
            name = re.sub(r'[._]', ' ', name)
            name = re.sub(r'\b(480p|720p|1080p|2160p|4k|hd|hevc)\b', '', name, flags=re.IGNORECASE)
            name = re.sub(r'\s+', ' ', name)
            title = name.strip()
        elif caption:
            title = extract_title_smart(caption)
        else:
            title = "Unknown File"
        
        # Extract quality
        quality = detect_quality(file_name or "")
        
        # Extract year
        year_match = re.search(r'\b(19|20)\d{2}\b', title)
        year = year_match.group() if year_match else ""
        
        # Get poster
        poster_url = Config.FALLBACK_POSTER
        poster_source = 'fallback'
        if poster_fetcher is not None:
            try:
                poster_data = await poster_fetcher.fetch_poster(title, year, quality)
                if poster_data:
                    poster_url = poster_data['poster_url']
                    poster_source = poster_data['source']
            except:
                pass
        
        # Create document
        doc = {
            'channel_id': Config.FILE_CHANNEL_ID,
            'message_id': message.id,
            'real_message_id': message.id,
            'title': title,
            'normalized_title': normalize_title(title),
            'date': message.date,
            'indexed_at': datetime.now(),
            'is_video_file': True,
            'file_id': file_id,
            'file_size': file_size,
            'file_name': file_name,
            'caption': caption or '',
            'quality': quality,
            'year': year,
            'thumbnail_url': poster_url,
            'poster_url': poster_url,
            'poster_source': poster_source,
            'status': 'active',
            'searchable': True
        }
        
        # Insert into database
        await files_col.insert_one(doc)
        
        logger.debug(f"Indexed: {title[:50]}... (ID: {message.id})")
        return True
        
    except Exception as e:
        if "duplicate key" in str(e).lower():
            return False
        logger.error(f"Indexing error: {e}")
        return False

# ============================================================================
# ✅ INITIAL BULK INDEXING
# ============================================================================

async def bulk_index_files():
    """Bulk index files from file channel"""
    if not user_session_ready or User is None or files_col is None:
        logger.warning("⚠️ Cannot bulk index: Telegram session not ready")
        return
    
    logger.info("🚀 Starting bulk file indexing...")
    
    try:
        total_indexed = 0
        batch_size = 100
        last_message_id = 0
        
        # Get existing max message ID
        last_indexed = await files_col.find_one(
            {"channel_id": Config.FILE_CHANNEL_ID}, 
            sort=[('message_id', -1)],
            projection={'message_id': 1}
        )
        
        if last_indexed:
            last_message_id = last_indexed['message_id']
            logger.info(f"Resuming from message ID: {last_message_id}")
        
        # Fetch messages in batches
        messages_to_index = []
        async for msg in User.get_chat_history(Config.FILE_CHANNEL_ID, limit=500):
            if msg.id <= last_message_id:
                break
            
            if msg and (msg.document or msg.video):
                messages_to_index.append(msg)
                
                if len(messages_to_index) >= batch_size:
                    # Index batch
                    indexed = 0
                    for msg in messages_to_index:
                        success = await index_single_file(msg)
                        if success:
                            indexed += 1
                    
                    total_indexed += indexed
                    logger.info(f"Indexed {indexed} files in batch, total: {total_indexed}")
                    messages_to_index = []
                    
                    # Small delay to avoid flood
                    await asyncio.sleep(2)
        
        # Index remaining messages
        if messages_to_index:
            indexed = 0
            for msg in messages_to_index:
                success = await index_single_file(msg)
                if success:
                    indexed += 1
            
            total_indexed += indexed
            logger.info(f"Indexed {indexed} remaining files, total: {total_indexed}")
        
        logger.info(f"✅ Bulk indexing complete: {total_indexed} files indexed")
        
    except Exception as e:
        logger.error(f"❌ Bulk indexing error: {e}")

# ============================================================================
# ✅ TELEGRAM SESSION INITIALIZATION
# ============================================================================

async def init_telegram_sessions():
    """Initialize Telegram sessions"""
    global User, Bot, user_session_ready, bot_session_ready
    
    if not PYROGRAM_AVAILABLE:
        logger.error("❌ Pyrogram not installed!")
        return False
    
    logger.info("=" * 50)
    logger.info("🚀 TELEGRAM SESSION INITIALIZATION")
    logger.info("=" * 50)
    
    # Initialize USER Session
    if Config.API_ID > 0 and Config.API_HASH and Config.USER_SESSION_STRING:
        logger.info("\n👤 Initializing USER Session...")
        try:
            User = Client(
                "sk4film_user",
                api_id=Config.API_ID,
                api_hash=Config.API_HASH,
                session_string=Config.USER_SESSION_STRING,
                sleep_threshold=120,
                in_memory=True,
                no_updates=True
            )
            
            await User.start()
            me = await User.get_me()
            logger.info(f"✅ USER Session Ready: {me.first_name}")
            user_session_ready = True
                
        except Exception as e:
            logger.error(f"❌ USER Session failed: {e}")
            user_session_ready = False
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
                sleep_threshold=120,
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
            Bot = None
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 SESSION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"USER Session: {'✅ READY' if user_session_ready else '❌ NOT READY'}")
    logger.info(f"BOT Session: {'✅ READY' if bot_session_ready else '❌ NOT READY'}")
    
    return user_session_ready or bot_session_ready

# ============================================================================
# ✅ MONGODB INITIALIZATION
# ============================================================================

async def init_mongodb():
    global mongo_client, db, files_col
    
    try:
        logger.info("🔌 MongoDB initialization...")
        
        mongo_client = AsyncIOMotorClient(
            Config.MONGODB_URI,
            serverSelectionTimeoutMS=10000,
            connectTimeoutMS=10000,
            socketTimeoutMS=15000,
            maxPoolSize=20,
            minPoolSize=5
        )
        
        await asyncio.wait_for(mongo_client.admin.command('ping'), timeout=5)
        
        db = mongo_client.sk4film
        files_col = db.files
        
        # Create indexes
        await files_col.create_index(
            [("channel_id", 1), ("message_id", 1)],
            unique=True,
            name="channel_message_unique"
        )
        
        await files_col.create_index(
            [("normalized_title", "text")],
            name="title_text_search"
        )
        
        logger.info("✅ MongoDB OK")
        return True
        
    except asyncio.TimeoutError:
        logger.error("❌ MongoDB connection timeout")
        return False
    except Exception as e:
        logger.error(f"❌ MongoDB error: {e}")
        return False

# ============================================================================
# ✅ SEARCH FUNCTION - FIXED FOR FILE RESULTS
# ============================================================================

@performance_monitor.measure("enhanced_search")
@async_cache_with_ttl(maxsize=500, ttl=Config.SEARCH_CACHE_TTL)
async def search_movies_enhanced_fixed(query, limit=15, page=1):
    """Search endpoint that actually returns file results"""
    offset = (page - 1) * limit
    
    # Cache check
    cache_key = f"search:{query}:{page}:{limit}"
    if cache_manager is not None and cache_manager.redis_enabled:
        cached = await cache_manager.get(cache_key)
        if cached:
            return cached
    
    logger.info(f"🔍 Searching: {query}")
    
    results = []
    
    # 1. FIRST: Search in FILES collection (file channel)
    if files_col is not None:
        try:
            logger.info(f"🔍 Searching in files collection for: {query}")
            
            # Create search query
            search_query = {
                "$or": [
                    {"title": {"$regex": query, "$options": "i"}},
                    {"normalized_title": {"$regex": query, "$options": "i"}},
                    {"caption": {"$regex": query, "$options": "i"}},
                ],
                "status": "active",
                "searchable": True
            }
            
            # Execute query
            cursor = files_col.find(
                search_query,
                {
                    '_id': 0,
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
                    'thumbnail_url': 1,
                    'poster_url': 1,
                    'poster_source': 1,
                    'year': 1,
                    'searchable': 1
                }
            ).sort("date", -1).limit(limit * 2).skip(offset)
            
            file_count = 0
            async for doc in cursor:
                try:
                    file_count += 1
                    
                    title = doc.get('title', 'Unknown')
                    quality = doc.get('quality', '480p')
                    year = doc.get('year', '')
                    message_id = doc.get('real_message_id') or doc.get('message_id')
                    
                    # Get thumbnail URL
                    thumbnail_url = doc.get('thumbnail_url') or doc.get('poster_url') or Config.FALLBACK_POSTER
                    
                    # Create quality options
                    quality_options = {
                        quality: {
                            'quality': quality,
                            'file_size': doc.get('file_size', 0),
                            'message_id': message_id,
                            'real_message_id': message_id,
                            'is_video_file': True,
                            'file_id': doc.get('file_id')
                        }
                    }
                    
                    # Create result
                    result = {
                        'title': title,
                        'normalized_title': doc.get('normalized_title', normalize_title(title)),
                        'content': format_post(doc.get('caption', ''), max_length=300),
                        'post_content': doc.get('caption', ''),
                        'quality_options': quality_options,
                        'date': doc['date'].isoformat() if isinstance(doc['date'], datetime) else doc['date'],
                        'is_new': is_new(doc['date']) if doc.get('date') else False,
                        'is_video_file': True,
                        'channel_id': doc.get('channel_id'),
                        'has_file': True,
                        'has_post': bool(doc.get('caption')),
                        'year': year,
                        'quality': quality,
                        'has_thumbnail': True,
                        'thumbnail': thumbnail_url,
                        'thumbnail_url': thumbnail_url,
                        'poster_url': thumbnail_url,
                        'poster_source': doc.get('poster_source', 'fallback'),
                        'real_message_id': message_id,
                        'message_id': message_id,
                        'result_type': 'file_only',
                        'quality_count': 1,
                        'has_poster': True,
                        'bot_username': Config.BOT_USERNAME,
                        'file_channel_id': Config.FILE_CHANNEL_ID
                    }
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error processing file result: {e}")
                    continue
            
            logger.info(f"✅ Found {file_count} files for query: {query}")
            
        except Exception as e:
            logger.error(f"Database search error: {e}")
    
    # 2. SECOND: If no file results, try text channels
    if not results and user_session_ready and User is not None:
        try:
            logger.info(f"🔍 No files found, searching in text channels for: {query}")
            
            for channel_id in Config.TEXT_CHANNEL_IDS:
                try:
                    async for msg in User.search_messages(channel_id, query=query, limit=10):
                        if msg and msg.text and len(msg.text) > 50:
                            title = extract_title_smart(msg.text)
                            if title:
                                # Get poster
                                thumbnail_url = Config.FALLBACK_POSTER
                                if poster_fetcher is not None:
                                    try:
                                        poster_data = await poster_fetcher.fetch_poster(title)
                                        if poster_data:
                                            thumbnail_url = poster_data['poster_url']
                                    except:
                                        pass
                                
                                result = {
                                    'title': title,
                                    'content': format_post(msg.text, max_length=300),
                                    'post_content': msg.text,
                                    'date': msg.date.isoformat() if isinstance(msg.date, datetime) else str(msg.date),
                                    'is_new': is_new(msg.date),
                                    'channel_id': channel_id,
                                    'message_id': msg.id,
                                    'has_file': False,
                                    'has_post': True,
                                    'has_thumbnail': True,
                                    'thumbnail': thumbnail_url,
                                    'thumbnail_url': thumbnail_url,
                                    'poster_url': thumbnail_url,
                                    'result_type': 'post_only',
                                    'bot_username': Config.BOT_USERNAME
                                }
                                results.append(result)
                except Exception as e:
                    logger.error(f"Text channel search error: {e}")
                    continue
        except Exception as e:
            logger.error(f"Text search error: {e}")
    
    # Sort results by date (newest first)
    results.sort(key=lambda x: x.get('date', ''), reverse=True)
    
    total = len(results)
    paginated = results[offset:offset + limit]
    
    # Final data
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
            'total': total,
            'source': 'files' if file_count > 0 else 'text'
        },
        'bot_username': Config.BOT_USERNAME
    }
    
    # Cache results
    if cache_manager is not None:
        await cache_manager.set(cache_key, result_data, expire_seconds=300)
    
    logger.info(f"✅ Search complete: {len(paginated)} results")
    
    return result_data

# ============================================================================
# ✅ HOME MOVIES FUNCTION
# ============================================================================

@performance_monitor.measure("home_movies")
@async_cache_with_ttl(maxsize=1, ttl=60)
async def get_home_movies(limit=25):
    """Get home movies"""
    try:
        if files_col is None:
            return []
        
        # Get latest files from database
        cursor = files_col.find(
            {"status": "active", "searchable": True},
            {
                '_id': 0,
                'title': 1,
                'normalized_title': 1,
                'quality': 1,
                'file_size': 1,
                'channel_id': 1,
                'message_id': 1,
                'real_message_id': 1,
                'date': 1,
                'caption': 1,
                'thumbnail_url': 1,
                'poster_url': 1,
                'poster_source': 1,
                'year': 1
            }
        ).sort("date", -1).limit(limit)
        
        movies = []
        async for doc in cursor:
            title = doc.get('title', 'Unknown')
            quality = doc.get('quality', '480p')
            year = doc.get('year', '')
            message_id = doc.get('real_message_id') or doc.get('message_id')
            
            # Get thumbnail
            thumbnail_url = doc.get('thumbnail_url') or doc.get('poster_url') or Config.FALLBACK_POSTER
            
            # Create quality options
            quality_options = {
                quality: {
                    'quality': quality,
                    'file_size': doc.get('file_size', 0),
                    'message_id': message_id,
                    'real_message_id': message_id,
                    'is_video_file': True
                }
            }
            
            movie = {
                'title': title,
                'normalized_title': doc.get('normalized_title', normalize_title(title)),
                'content': format_post(doc.get('caption', ''), max_length=300),
                'post_content': doc.get('caption', ''),
                'quality_options': quality_options,
                'date': doc['date'].isoformat() if isinstance(doc['date'], datetime) else doc['date'],
                'is_new': is_new(doc['date']) if doc.get('date') else False,
                'is_video_file': True,
                'channel_id': doc.get('channel_id'),
                'has_file': True,
                'has_post': bool(doc.get('caption')),
                'year': year,
                'quality': quality,
                'has_thumbnail': True,
                'thumbnail': thumbnail_url,
                'thumbnail_url': thumbnail_url,
                'poster_url': thumbnail_url,
                'poster_source': doc.get('poster_source', 'fallback'),
                'real_message_id': message_id,
                'message_id': message_id,
                'result_type': 'file_only',
                'quality_count': 1,
                'has_poster': True,
                'bot_username': Config.BOT_USERNAME,
                'file_channel_id': Config.FILE_CHANNEL_ID
            }
            
            movies.append(movie)
        
        if not movies:
            # Fallback to text channel if no files
            if user_session_ready and User is not None:
                movies = await _get_text_channel_movies(limit)
        
        logger.info(f"✅ Fetched {len(movies)} home movies")
        return movies[:limit]
        
    except Exception as e:
        logger.error(f"❌ Home movies error: {e}")
        return []

async def _get_text_channel_movies(limit):
    """Get movies from text channel as fallback"""
    movies = []
    try:
        async for msg in User.get_chat_history(Config.MAIN_CHANNEL_ID, limit=limit):
            if msg and msg.text and len(msg.text) > 50:
                title = extract_title_smart(msg.text)
                if title:
                    # Get poster
                    thumbnail_url = Config.FALLBACK_POSTER
                    if poster_fetcher is not None:
                        try:
                            poster_data = await poster_fetcher.fetch_poster(title)
                            if poster_data:
                                thumbnail_url = poster_data['poster_url']
                        except:
                            pass
                    
                    movie = {
                        'title': title,
                        'content': format_post(msg.text, max_length=300),
                        'post_content': msg.text,
                        'date': msg.date.isoformat() if isinstance(msg.date, datetime) else str(msg.date),
                        'is_new': is_new(msg.date),
                        'channel_id': Config.MAIN_CHANNEL_ID,
                        'message_id': msg.id,
                        'has_file': False,
                        'has_post': True,
                        'has_thumbnail': True,
                        'thumbnail': thumbnail_url,
                        'thumbnail_url': thumbnail_url,
                        'poster_url': thumbnail_url,
                        'result_type': 'post_only',
                        'bot_username': Config.BOT_USERNAME
                    }
                    movies.append(movie)
    except Exception as e:
        logger.error(f"Text channel fallback error: {e}")
    
    return movies

# ============================================================================
# ✅ SYSTEM INITIALIZATION
# ============================================================================

async def init_system():
    start_time = time.time()
    
    try:
        logger.info("=" * 60)
        logger.info("🚀 SK4FiLM v10.0 - ULTRA FAST")
        logger.info("=" * 60)
        
        # Initialize MongoDB
        mongo_ok = await init_mongodb()
        if not mongo_ok:
            logger.error("❌ MongoDB connection failed")
            return False
        
        # Initialize Cache
        global cache_manager, poster_fetcher
        try:
            cache_manager = CacheManager(Config)
            redis_ok = await cache_manager.init_redis()
            if redis_ok:
                logger.info("✅ Cache Manager initialized")
        except:
            logger.warning("⚠️ Cache init failed")
            cache_manager = None
        
        # Initialize Poster Fetcher
        poster_fetcher = SimplePosterFetcher(Config, cache_manager)
        logger.info("✅ Poster Fetcher initialized")
        
        # Initialize Telegram
        if PYROGRAM_AVAILABLE:
            await init_telegram_sessions()
        
        # Get file count
        if files_col is not None:
            try:
                file_count = await files_col.count_documents({})
                logger.info(f"📊 Files in database: {file_count}")
                
                # If database is empty, do initial bulk indexing
                if file_count == 0 and user_session_ready:
                    logger.info("📥 Database empty, starting initial bulk indexing...")
                    await bulk_index_files()
                else:
                    # Start background indexing
                    await file_indexing_manager.start_indexing()
            except Exception as e:
                logger.error(f"File count error: {e}")
        
        init_time = time.time() - start_time
        logger.info(f"⚡ System started in {init_time:.2f}s")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ System initialization failed: {e}")
        return False

# ============================================================================
# ✅ API ROUTES
# ============================================================================

@app.route('/')
async def root():
    """Root endpoint"""
    try:
        total_files = 0
        if files_col is not None:
            try:
                total_files = await files_col.count_documents({})
            except:
                pass
        
        return jsonify({
            'status': 'healthy',
            'service': 'SK4FiLM v10.0',
            'stats': {
                'total_files': total_files
            },
            'sessions': {
                'user': user_session_ready,
                'bot': bot_session_ready
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Root endpoint error: {e}")
        return jsonify({'status': 'error'}), 500

@app.route('/health')
async def health():
    """Health check"""
    try:
        return jsonify({
            'status': 'ok',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'status': 'error'}), 500

@app.route('/api/movies', methods=['GET'])
async def api_movies():
    """Get home movies"""
    try:
        movies = await get_home_movies(limit=25)
        
        return jsonify({
            'status': 'success' if movies else 'empty',
            'movies': movies,
            'total': len(movies),
            'limit': 25,
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
async def api_search():
    """Search API endpoint"""
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
    """Get statistics"""
    try:
        # Get database stats
        if files_col is not None:
            total_files = await files_col.count_documents({})
            video_files = await files_col.count_documents({'is_video_file': True})
        else:
            total_files = 0
            video_files = 0
        
        return jsonify({
            'status': 'success',
            'database_stats': {
                'total_files': total_files,
                'video_files': video_files,
            },
            'indexing_stats': {
                'running': file_indexing_manager.is_running,
                'total_indexed': file_indexing_manager.total_indexed,
                'last_run': file_indexing_manager.last_run.isoformat() if file_indexing_manager.last_run else None
            },
            'poster_stats': poster_fetcher.get_stats() if poster_fetcher else {},
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Stats API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/admin/reindex', methods=['POST'])
async def api_admin_reindex():
    """Trigger reindexing"""
    try:
        auth_token = request.headers.get('X-Admin-Token')
        if not auth_token or auth_token != os.environ.get('ADMIN_TOKEN', 'sk4film_admin_123'):
            return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401
        
        # Trigger bulk indexing
        asyncio.create_task(bulk_index_files())
        
        return jsonify({
            'status': 'success',
            'message': 'Bulk indexing started',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Admin reindex error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/admin/check-files', methods=['GET'])
async def api_admin_check_files():
    """Check if files are being indexed"""
    try:
        if files_col is None:
            return jsonify({
                'status': 'error',
                'message': 'Database not connected'
            }), 500
        
        # Get sample files
        sample_files = []
        cursor = files_col.find({}, {
            'title': 1,
            'message_id': 1,
            'quality': 1,
            'date': 1,
            '_id': 0
        }).limit(10)
        
        async for doc in cursor:
            sample_files.append(doc)
        
        total_files = await files_col.count_documents({})
        
        return jsonify({
            'status': 'success',
            'total_files': total_files,
            'sample_files': sample_files,
            'user_session_ready': user_session_ready,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Check files error: {e}")
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
    logger.info("🛑 Shutting down SK4FiLM...")
    
    # Stop indexing
    await file_indexing_manager.stop_indexing()
    
    # Close Telegram sessions
    if User is not None:
        try:
            await User.stop()
        except:
            pass
    
    if Bot is not None:
        try:
            await Bot.stop()
        except:
            pass
    
    # Close cache manager
    if cache_manager is not None:
        try:
            await cache_manager.stop()
        except:
            pass
    
    # Close MongoDB
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
    
    logger.info(f"🌐 Starting SK4FiLM on port {Config.WEB_SERVER_PORT}...")
    logger.info("🔧 Features: File Indexing • Quick Search • Real Message IDs")
    
    try:
        asyncio.run(serve(app, config))
    except KeyboardInterrupt:
        logger.info("👋 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
