# ============================================================================
# 🚀 SK4FiLM v8.6.1 - FIXED ERRORS
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

# ✅ IMPORT ALL MODULES
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

# Import modules with fallbacks
try:
    from cache import CacheManager
    logger.debug("✅ Cache module imported")
except ImportError:
    class CacheManager:
        def __init__(self, config): pass
        async def init_redis(self): return False
        async def get(self, key): return None
        async def set(self, key, value, expire_seconds=0): pass
        async def delete(self, key): pass
        async def start_cleanup_task(self): pass
        async def stop(self): pass

try:
    from verification import VerificationSystem
    logger.debug("✅ Verification module imported")
except ImportError:
    VerificationSystem = None
    class VerificationSystem:
        def __init__(self, config, mongo_client): pass
        async def check_user_verified(self, user_id, premium_system): return True, "User verified"
        async def get_user_verification_info(self, user_id): return {"verified": True}
        async def stop(self): pass

try:
    from premium import PremiumSystem, PremiumTier
    logger.debug("✅ Premium module imported")
except ImportError:
    PremiumSystem = None
    PremiumTier = None
    class PremiumTier:
        BASIC = "basic"
        PREMIUM = "premium"
        GOLD = "gold"
        DIAMOND = "diamond"
    
    class PremiumSystem:
        def __init__(self, config, mongo_client): pass
        async def is_premium_user(self, user_id): return False
        async def get_user_tier(self, user_id): return PremiumTier.BASIC
        async def get_subscription_details(self, user_id): return {"tier": "basic", "expiry": None}
        async def stop_cleanup_task(self): pass

try:
    from poster_fetching import PosterFetcher, PosterSource
    logger.debug("✅ Poster fetching module imported")
except ImportError:
    PosterFetcher = None
    PosterSource = None
    class PosterSource:
        TMDB = "tmdb"
        OMDB = "omdb"
        CUSTOM = "custom"
        FALLBACK = "fallback"

try:
    from utils import normalize_title, extract_title_smart, extract_title_from_file, format_size, detect_quality, is_video_file, format_post, is_new
    logger.debug("✅ Utils module imported")
except ImportError:
    def normalize_title(title):
        if not title: return ""
        title = title.lower().strip()
        title = re.sub(r'\s*\([^)]*\)$', '', title)
        title = re.sub(r'\s*\[[^\]]*\]$', '', title)
        title = re.sub(r'\s*\d{4}$', '', title)
        return title
    
    def extract_title_smart(text):
        if not text: return ""
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
            if name: return name
        if caption: return extract_title_smart(caption)
        return "Unknown"
    
    def format_size(size):
        if not size: return "Unknown"
        if size < 1024: return f"{size} B"
        elif size < 1024*1024: return f"{size/1024:.1f} KB"
        elif size < 1024*1024*1024: return f"{size/1024/1024:.1f} MB"
        else: return f"{size/1024/1024/1024:.2f} GB"
    
    def detect_quality(filename):
        if not filename: return "480p"
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
        if not filename: return False
        video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v']
        return any(filename.lower().endswith(ext) for ext in video_extensions)
    
    def format_post(text, max_length=None):
        if not text: return ""
        text = re.sub(r'\n\s*\n', '\n\n', text)
        if max_length and len(text) > max_length:
            text = text[:max_length] + "..."
        return text.strip()
    
    def is_new(date):
        if not date: return False
        if isinstance(date, str):
            try:
                date = datetime.fromisoformat(date.replace('Z', '+00:00'))
            except:
                return False
        return (datetime.now() - date).days < 7

# ============================================================================
# ✅ CONFIGURATION
# ============================================================================

class Config:
    API_ID = int(os.environ.get("API_ID", "0"))
    API_HASH = os.environ.get("API_HASH", "")
    USER_SESSION_STRING = os.environ.get("USER_SESSION_STRING", "")
    BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
    
    MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
    REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
    REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "")
    
    MAIN_CHANNEL_ID = -1001891090100
    TEXT_CHANNEL_IDS = [-1001891090100, -1002024811395]
    FILE_CHANNEL_ID = -1001768249569
    
    MAIN_CHANNEL_LINK = "https://t.me/sk4film"
    UPDATES_CHANNEL_LINK = "https://t.me/sk4film_Request"
    CHANNEL_USERNAME = "sk4film"
    
    SHORTLINK_API = os.environ.get("SHORTLINK_API", "")
    CUTTLY_API = os.environ.get("CUTTLY_API", "")
    
    UPI_ID_BASIC = os.environ.get("UPI_ID_BASIC", "cf.sk4film@cashfreensdlpb")
    UPI_ID_PREMIUM = os.environ.get("UPI_ID_PREMIUM", "cf.sk4film@cashfreensdlpb")
    UPI_ID_GOLD = os.environ.get("UPI_ID_GOLD", "cf.sk4film@cashfreensdlpb")
    UPI_ID_DIAMOND = os.environ.get("UPI_ID_DIAMOND", "cf.sk4film@cashfreensdlpb")
    
    VERIFICATION_REQUIRED = os.environ.get("VERIFICATION_REQUIRED", "true").lower() == "true"
    VERIFICATION_DURATION = 6 * 60 * 60
    
    WEBSITE_URL = os.environ.get("WEBSITE_URL", "https://sk4film.vercel.app")
    BOT_USERNAME = os.environ.get("BOT_USERNAME", "sk4filmbot")
    ADMIN_IDS = [int(x) for x in os.environ.get("ADMIN_IDS", "123456789").split(",")]
    AUTO_DELETE_TIME = int(os.environ.get("AUTO_DELETE_TIME", "5"))
    WEB_SERVER_PORT = int(os.environ.get("PORT", 8000))
    BACKEND_URL = os.environ.get("BACKEND_URL", "https://sk4film.koyeb.app")
    
    TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "e547e17d4e91f3e62a571655cd1ccaff")
    OMDB_API_KEY = os.environ.get("OMDB_API_KEY", "8265bd1c")
    
    MAX_CONCURRENT_REQUESTS = int(os.environ.get("MAX_CONCURRENT_REQUESTS", "50"))
    CACHE_TTL = int(os.environ.get("CACHE_TTL", "300"))
    REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "10"))
    
    MONITOR_INTERVAL = int(os.environ.get("MONITOR_INTERVAL", "300"))
    QUALITY_PRIORITY = ['2160p', '1080p', '720p', '480p', '360p']
    HEVC_VARIANTS = ['720p HEVC', '1080p HEVC', '2160p HEVC']
    
    FALLBACK_POSTER = "https://iili.io/fAeIwv9.th.png"
    THUMBNAIL_EXTRACT_TIMEOUT = 10
    THUMBNAIL_CACHE_DURATION = 24 * 60 * 60
    
    AUTO_INDEX_INTERVAL = int(os.environ.get("AUTO_INDEX_INTERVAL", "120"))
    BATCH_INDEX_SIZE = int(os.environ.get("BATCH_INDEX_SIZE", "500"))
    MAX_INDEX_LIMIT = int(os.environ.get("MAX_INDEX_LIMIT", "0"))
    INDEX_ALL_HISTORY = os.environ.get("INDEX_ALL_HISTORY", "true").lower() == "true"
    INSTANT_AUTO_INDEX = os.environ.get("INSTANT_AUTO_INDEX", "true").lower() == "true"
    
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
            self.measurements[name] = {'count': 0, 'total': 0, 'avg': 0}
        stats = self.measurements[name]
        stats['count'] += 1
        stats['total'] += elapsed
        stats['avg'] = stats['total'] / stats['count']
    
    def get_stats(self):
        return self.measurements

performance_monitor = PerformanceMonitor()

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
    response.headers['X-SK4FiLM-Version'] = '8.6.1-FIXED-ERRORS'
    return response

# ============================================================================
# ✅ GLOBAL COMPONENTS
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
telegram_bot = None

is_indexing = False
last_index_time = None
indexing_task = None

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
    if not filename: return "480p"
    filename_lower = filename.lower()
    is_hevc = any(re.search(pattern, filename_lower) for pattern in HEVC_PATTERNS)
    for pattern, quality in QUALITY_PATTERNS:
        if re.search(pattern, filename_lower):
            if is_hevc and quality in ['720p', '1080p', '2160p']:
                return f"{quality} HEVC"
            return quality
    return "480p"

# ============================================================================
# ✅ BOT HANDLER - FIXED WITH get_bot_status METHOD
# ============================================================================

class BotHandler:
    def __init__(self, bot_token=None, api_id=None, api_hash=None):
        self.bot_token = bot_token or Config.BOT_TOKEN
        self.api_id = api_id or Config.API_ID
        self.api_hash = api_hash or Config.API_HASH
        self.bot = None
        self.initialized = False
        self.last_update = None
        self.bot_username = None
    
    async def initialize(self):
        if not self.bot_token or not self.api_id or not self.api_hash:
            return False
        try:
            global Bot, bot_session_ready
            if Bot is not None and bot_session_ready:
                self.bot = Bot
                self.initialized = True
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
            return True
        except Exception as e:
            logger.error(f"❌ Bot handler initialization error: {e}")
            return False
    
    async def get_file_info(self, channel_id, message_id):
        if not self.initialized: return None
        try:
            message = await self.bot.get_messages(channel_id, message_id)
            if not message: return None
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
                    'file_id': message.video.file_id
                })
            return file_info
        except Exception as e:
            logger.error(f"❌ Get file info error: {e}")
            return None
    
    # ✅ ADDED MISSING METHOD
    async def get_bot_status(self):
        """Get bot handler status"""
        return {
            'initialized': self.initialized,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'bot_username': self.bot_username,
            'connected': self.bot is not None
        }
    
    async def shutdown(self):
        if self.bot and not bot_session_ready:
            await self.bot.stop()
        self.initialized = False

bot_handler = BotHandler()

# ============================================================================
# ✅ FILE CHANNEL INDEXING MANAGER
# ============================================================================

class FileChannelIndexingManager:
    def __init__(self):
        self.is_running = False
        self.indexing_task = None
        self.last_run = None
        self.next_run = None
        self.total_indexed = 0
        self.total_duplicates = 0
        self.indexing_stats = {
            'total_runs': 0,
            'total_files_processed': 0,
            'total_indexed': 0,
            'total_duplicates': 0,
            'total_errors': 0,
            'last_success': None
        }
    
    async def start_indexing(self):
        if self.is_running: return
        logger.info("🚀 Starting FILE CHANNEL INDEXING...")
        self.is_running = True
        self.indexing_task = asyncio.create_task(self._indexing_loop())
    
    async def stop_indexing(self):
        self.is_running = False
        if self.indexing_task:
            self.indexing_task.cancel()
            try:
                await self.indexing_task
            except asyncio.CancelledError:
                pass
    
    async def _indexing_loop(self):
        while self.is_running:
            try:
                await self._run_indexing_cycle()
                self.next_run = datetime.now() + timedelta(seconds=Config.AUTO_INDEX_INTERVAL)
                self.last_run = datetime.now()
                await asyncio.sleep(Config.AUTO_INDEX_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Indexing loop error: {e}")
                await asyncio.sleep(60)
    
    async def _run_indexing_cycle(self):
        logger.info("🔄 FILE INDEXING CYCLE")
        start_time = time.time()
        cycle_stats = {'processed': 0, 'indexed': 0, 'duplicates': 0, 'errors': 0}
        
        try:
            if not user_session_ready or User is None:
                logger.warning("⚠️ User session not ready for indexing")
                return
            
            last_indexed = await files_col.find_one(
                {"channel_id": Config.FILE_CHANNEL_ID}, 
                sort=[('message_id', -1)],
                projection={'message_id': 1}
            )
            last_message_id = last_indexed['message_id'] if last_indexed else 0
            
            messages_to_index = []
            async for msg in User.get_chat_history(Config.FILE_CHANNEL_ID, limit=Config.BATCH_INDEX_SIZE):
                if msg.id <= last_message_id:
                    break
                if msg and (msg.document or msg.video):
                    messages_to_index.append(msg)
            
            logger.info(f"📥 Found {len(messages_to_index)} new files")
            
            for msg in messages_to_index:
                try:
                    existing = await files_col.find_one({
                        'channel_id': Config.FILE_CHANNEL_ID,
                        'message_id': msg.id
                    })
                    if existing:
                        cycle_stats['duplicates'] += 1
                        continue
                    
                    success = await self._index_single_file(msg)
                    if success:
                        cycle_stats['indexed'] += 1
                    else:
                        cycle_stats['duplicates'] += 1
                        
                    cycle_stats['processed'] += 1
                    
                except Exception as e:
                    logger.error(f"❌ Error processing message {msg.id}: {e}")
                    cycle_stats['errors'] += 1
                    continue
            
            self.indexing_stats['total_runs'] += 1
            self.indexing_stats['total_files_processed'] += cycle_stats['processed']
            self.indexing_stats['total_indexed'] += cycle_stats['indexed']
            self.indexing_stats['total_duplicates'] += cycle_stats['duplicates']
            self.indexing_stats['total_errors'] += cycle_stats['errors']
            self.indexing_stats['last_success'] = datetime.now()
            self.total_indexed += cycle_stats['indexed']
            self.total_duplicates += cycle_stats['duplicates']
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Indexing complete: {cycle_stats['indexed']} new files in {elapsed:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Indexing cycle failed: {e}")
    
    async def _index_single_file(self, message):
        try:
            caption = message.caption if hasattr(message, 'caption') else None
            file_name = None
            
            if message.document:
                file_name = message.document.file_name
            elif message.video:
                file_name = message.video.file_name
            
            title = self._extract_title_improved(file_name, caption)
            if not title or title == "Unknown File":
                return False
            
            normalized_title = normalize_title(title)
            
            year_match = re.search(r'\b(19|20)\d{2}\b', title)
            year = year_match.group() if year_match else ""
            
            quality = detect_quality_enhanced(file_name or "")
            
            doc = {
                'channel_id': Config.FILE_CHANNEL_ID,
                'message_id': message.id,
                'real_message_id': message.id,
                'title': title,
                'normalized_title': normalized_title,
                'date': message.date,
                'indexed_at': datetime.now(),
                'is_video_file': is_video_file(file_name or ''),
                'file_size': 0,
                'caption': caption or '',
                'thumbnail_url': None,
                'thumbnail_extracted': False,
                'status': 'active',
                'quality': quality,
                'year': year,
                'result_type': 'file_only'
            }
            
            if message.document:
                doc.update({
                    'file_name': message.document.file_name or '',
                    'mime_type': message.document.mime_type or '',
                    'telegram_file_id': message.document.file_id,
                    'file_size': message.document.file_size or 0
                })
            elif message.video:
                doc.update({
                    'file_name': message.video.file_name or 'video.mp4',
                    'duration': message.video.duration if hasattr(message.video, 'duration') else 0,
                    'telegram_file_id': message.video.file_id,
                    'file_size': message.video.file_size or 0
                })
            else:
                return False
            
            await files_col.insert_one(doc)
            logger.info(f"✅ INDEXED: {title[:60]}... (Msg ID: {message.id})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Indexing error for message {message.id}: {e}")
            return False
    
    def _extract_title_improved(self, filename, caption):
        if filename:
            name = os.path.splitext(filename)[0]
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
            name = re.sub(r'\s+', ' ', name)
            name = name.strip()
            year_match = re.search(r'\b(19|20)\d{2}\b', name)
            if year_match:
                year = year_match.group()
                name = re.sub(r'\s*\b(19|20)\d{2}\b', '', name)
                name = f"{name.strip()} ({year})"
            if name and len(name) > 3:
                return name
        
        if caption:
            lines = caption.split('\n')
            for line in lines:
                line = line.strip()
                if len(line) > 10 and not line.startswith('http'):
                    line = re.sub(r'📥.*', '', line)
                    line = re.sub(r'🎬.*', '', line)
                    line = re.sub(r'⚡.*', '', line)
                    line = re.sub(r'✅.*', '', line)
                    line = re.sub(r'[⭐🌟]+', '', line)
                    line = line.strip()
                    if line and len(line) > 5:
                        return line[:200]
        
        if filename:
            return os.path.splitext(filename)[0][:100]
        
        return "Unknown File"
    
    async def get_indexing_status(self):
        return {
            'is_running': self.is_running,
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'next_run': self.next_run.isoformat() if self.next_run else None,
            'total_indexed': self.total_indexed,
            'total_duplicates': self.total_duplicates,
            'stats': self.indexing_stats
        }

file_indexing_manager = FileChannelIndexingManager()

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
# ✅ TELEGRAM SESSION INITIALIZATION
# ============================================================================

@performance_monitor.measure("telegram_init")
async def init_telegram_sessions():
    global User, Bot, user_session_ready, bot_session_ready
    
    logger.info("🚀 TELEGRAM SESSION INITIALIZATION")
    
    if not PYROGRAM_AVAILABLE:
        logger.error("❌ Pyrogram not installed!")
        return False
    
    # Initialize USER Session
    if Config.API_ID > 0 and Config.API_HASH and Config.USER_SESSION_STRING:
        logger.info("👤 Initializing USER Session...")
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
        logger.info("🤖 Initializing BOT Session...")
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
    
    logger.info(f"USER Session: {'✅ READY' if user_session_ready else '❌ NOT READY'}")
    logger.info(f"BOT Session: {'✅ READY' if bot_session_ready else '❌ NOT READY'}")
    
    return user_session_ready or bot_session_ready

# ============================================================================
# ✅ MONGODB INITIALIZATION - FIXED INDEX CREATION
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
            minPoolSize=5
        )
        
        await asyncio.wait_for(mongo_client.admin.command('ping'), timeout=5)
        
        db = mongo_client.sk4film
        files_col = db.files
        verification_col = db.verifications
        
        # ✅ FIXED: Safe index creation with try-except
        try:
            await files_col.create_index([("channel_id", 1), ("message_id", 1)], unique=True)
            logger.info("✅ Created channel_id + message_id unique index")
        except Exception as e:
            if "already exists" in str(e) or "IndexOptionsConflict" in str(e):
                logger.info("ℹ️ Index already exists, skipping creation")
            else:
                logger.warning(f"⚠️ Index creation error (safe to ignore): {e}")
        
        try:
            await files_col.create_index([("normalized_title", "text"), ("title", "text")])
            logger.info("✅ Created text search index")
        except Exception as e:
            if "already exists" in str(e):
                logger.info("ℹ️ Text index already exists")
            else:
                logger.warning(f"⚠️ Text index creation error: {e}")
        
        try:
            await files_col.create_index([("quality", 1)])
            logger.info("✅ Created quality index")
        except Exception as e:
            if "already exists" in str(e):
                logger.info("ℹ️ Quality index already exists")
        
        try:
            await files_col.create_index([("date", -1)])
            logger.info("✅ Created date index")
        except Exception as e:
            if "already exists" in str(e):
                logger.info("ℹ️ Date index already exists")
        
        try:
            await files_col.create_index([("real_message_id", 1)])
            logger.info("✅ Created real_message_id index")
        except Exception as e:
            if "already exists" in str(e):
                logger.info("ℹ️ real_message_id index already exists")
        
        logger.info("✅ MongoDB OK")
        return True
        
    except asyncio.TimeoutError:
        logger.error("❌ MongoDB connection timeout")
        return False
    except Exception as e:
        logger.error(f"❌ MongoDB error: {e}")
        return False

# ============================================================================
# ✅ SYSTEM INITIALIZATION
# ============================================================================

@performance_monitor.measure("system_init")
async def init_system():
    start_time = time.time()
    
    try:
        logger.info("=" * 60)
        logger.info("🚀 SK4FiLM v8.6.1 - FIXED ERRORS")
        logger.info("=" * 60)
        
        # Initialize MongoDB
        mongo_ok = await init_mongodb()
        if not mongo_ok:
            logger.error("❌ MongoDB connection failed")
            # Don't return False, continue without MongoDB
            logger.warning("⚠️ Continuing without MongoDB connection")
        
        # Get current file count if MongoDB is connected
        if files_col is not None:
            try:
                file_count = await files_col.count_documents({})
                logger.info(f"📊 Current files in database: {file_count}")
            except Exception as e:
                logger.warning(f"⚠️ Could not get file count: {e}")
        
        # Initialize Bot Handler
        try:
            await bot_handler.initialize()
            logger.info("✅ Bot Handler initialized")
        except Exception as e:
            logger.error(f"❌ Bot Handler initialization failed: {e}")
        
        # Initialize Cache Manager
        global cache_manager, verification_system, premium_system, poster_fetcher
        try:
            cache_manager = CacheManager(Config)
            redis_ok = await cache_manager.init_redis()
            if redis_ok:
                logger.info("✅ Cache Manager initialized")
                await cache_manager.start_cleanup_task()
        except Exception as e:
            logger.error(f"❌ Cache Manager initialization failed: {e}")
        
        # Initialize Verification System
        if VerificationSystem is not None:
            try:
                verification_system = VerificationSystem(Config, mongo_client)
                logger.info("✅ Verification System initialized")
            except Exception as e:
                logger.error(f"❌ Verification System initialization failed: {e}")
        
        # Initialize Premium System
        if PremiumSystem is not None:
            try:
                premium_system = PremiumSystem(Config, mongo_client)
                logger.info("✅ Premium System initialized")
            except Exception as e:
                logger.error(f"❌ Premium System initialization failed: {e}")
        
        # Initialize Poster Fetcher
        if PosterFetcher is not None:
            try:
                poster_fetcher = PosterFetcher(Config, cache_manager)
                logger.info("✅ Poster Fetcher initialized")
            except Exception as e:
                logger.error(f"❌ Poster Fetcher initialization failed: {e}")
        
        # Initialize Telegram Sessions
        if PYROGRAM_AVAILABLE:
            try:
                telegram_ok = await init_telegram_sessions()
                if not telegram_ok:
                    logger.warning("⚠️ Telegram sessions failed")
            except Exception as e:
                logger.error(f"❌ Telegram sessions initialization failed: {e}")
        
        # Start indexing if conditions are met
        if user_session_ready and files_col is not None:
            try:
                logger.info("🔄 Starting file channel indexing...")
                await file_indexing_manager.start_indexing()
            except Exception as e:
                logger.error(f"❌ File indexing failed to start: {e}")
        
        init_time = time.time() - start_time
        logger.info(f"⚡ SK4FiLM Started in {init_time:.2f}s")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ System initialization failed: {e}")
        # Don't crash, continue with partial initialization
        return True

# ============================================================================
# ✅ POSTER FETCHING
# ============================================================================

async def get_poster_for_movie(title: str, year: str = "", quality: str = "") -> Dict[str, Any]:
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
        poster_task = asyncio.create_task(poster_fetcher.fetch_poster(title))
        poster_data = await asyncio.wait_for(poster_task, timeout=3.0)
        
        if poster_data and poster_data.get('poster_url'):
            return poster_data
        else:
            raise ValueError("Invalid poster data")
            
    except (asyncio.TimeoutError, ValueError, Exception):
        return {
            'poster_url': Config.FALLBACK_POSTER,
            'source': 'custom',
            'rating': '0.0',
            'year': year,
            'title': title,
            'quality': quality or 'unknown'
        }

async def get_posters_for_movies_batch(movies: List[Dict]) -> List[Dict]:
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
            movie_with_poster.update({
                'poster_url': poster_data['poster_url'],
                'poster_source': poster_data['source'],
                'poster_rating': poster_data['rating'],
                'thumbnail': poster_data['poster_url'],
                'thumbnail_source': poster_data['source'],
                'has_poster': True
            })
            results.append(movie_with_poster)
        except Exception as e:
            movie_with_fallback = movie.copy()
            movie_with_fallback.update({
                'poster_url': Config.FALLBACK_POSTER,
                'poster_source': 'fallback',
                'poster_rating': '0.0',
                'thumbnail': Config.FALLBACK_POSTER,
                'thumbnail_source': 'fallback',
                'has_poster': True
            })
            results.append(movie_with_fallback)
    
    return results

# ============================================================================
# ✅ FIXED SEARCH FUNCTION
# ============================================================================

@performance_monitor.measure("multi_channel_search_merged")
@async_cache_with_ttl(maxsize=500, ttl=Config.SEARCH_CACHE_TTL)
async def search_movies_multi_channel_merged(query, limit=15, page=1):
    """Search movies across all channels with proper result types"""
    offset = (page - 1) * limit
    
    # Try cache first
    cache_key = f"search_merged:{query}:{page}:{limit}"
    if cache_manager is not None and cache_manager.redis_enabled:
        try:
            cached_data = await cache_manager.get(cache_key)
            if cached_data:
                logger.info(f"✅ Cache HIT for: {query}")
                return cached_data
        except:
            pass
    
    logger.info(f"🔍 SEARCHING for: {query}")
    
    query_lower = query.lower()
    posts_dict = {}
    files_dict = {}
    
    # ============================================================================
    # ✅ 1. SEARCH TEXT CHANNELS (Posts Only)
    # ============================================================================
    if user_session_ready and User is not None:
        async def search_text_channel(channel_id):
            channel_posts = {}
            try:
                channel_name = f"Channel {channel_id}"
                async for msg in User.search_messages(channel_id, query=query, limit=20):
                    if msg is not None and msg.text and len(msg.text) > 15:
                        title = extract_title_smart(msg.text)
                        if title and (query_lower in title.lower() or query_lower in msg.text.lower()):
                            norm_title = normalize_title(title)
                            
                            # Extract year
                            year_match = re.search(r'\b(19|20)\d{2}\b', title)
                            year = year_match.group() if year_match else ""
                            
                            # Check if this post mentions any files
                            has_file_mention = False
                            if msg.text:
                                file_indicators = ['📥', '📁', '📎', '💾', '🎬']
                                for indicator in file_indicators:
                                    if indicator in msg.text:
                                        has_file_mention = True
                                        break
                            
                            if norm_title not in channel_posts:
                                post_data = {
                                    'title': title,
                                    'original_title': title,
                                    'normalized_title': norm_title,
                                    'content': format_post(msg.text, max_length=1000),
                                    'post_content': msg.text,
                                    'channel': channel_name,
                                    'channel_id': channel_id,
                                    'message_id': msg.id,
                                    'date': msg.date.isoformat() if isinstance(msg.date, datetime) else str(msg.date),
                                    'is_new': is_new(msg.date) if msg.date else False,
                                    'has_file': False,
                                    'has_post': True,
                                    'quality_options': {},
                                    'is_video_file': False,
                                    'year': year,
                                    'search_score': 2 if query_lower in title.lower() else 1,
                                    'result_type': 'post_only' if not has_file_mention else 'post_with_file_mention',
                                    'file_mentioned': has_file_mention,
                                    'file_caption': msg.text if has_file_mention else ''
                                }
                                
                                channel_posts[norm_title] = post_data
            except Exception as e:
                logger.error(f"Text search error in {channel_id}: {e}")
            return channel_posts
        
        # Search all text channels
        try:
            tasks = [search_text_channel(channel_id) for channel_id in Config.TEXT_CHANNEL_IDS]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, dict):
                    posts_dict.update(result)
            
            logger.info(f"📝 Found {len(posts_dict)} posts in text channels")
        except Exception as e:
            logger.error(f"Error searching text channels: {e}")
    
    # ============================================================================
    # ✅ 2. SEARCH FILE CHANNEL DATABASE (Files Only)
    # ============================================================================
    if files_col is not None:
        try:
            logger.info(f"🔍 Searching FILE CHANNEL database for: {query}")
            
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
                    'telegram_file_id': 1,
                    'year': 1,
                    '_id': 0
                }
            ).limit(1000)
            
            file_count = 0
            
            async for doc in cursor:
                file_count += 1
                try:
                    title = doc.get('title', 'Unknown')
                    norm_title = normalize_title(title)
                    
                    quality = doc.get('quality', 'Unknown')
                    
                    quality_option = {
                        'file_id': f"{doc.get('channel_id', Config.FILE_CHANNEL_ID)}_{doc.get('real_message_id') or doc.get('message_id')}_{quality}",
                        'file_size': doc.get('file_size', 0),
                        'file_name': doc.get('file_name', ''),
                        'is_video': doc.get('is_video_file', False),
                        'channel_id': doc.get('channel_id'),
                        'message_id': doc.get('message_id'),
                        'real_message_id': doc.get('real_message_id') or doc.get('message_id'),
                        'quality': quality,
                        'date': doc.get('date'),
                        'telegram_file_id': doc.get('telegram_file_id')
                    }
                    
                    if norm_title not in files_dict:
                        year = doc.get('year', '')
                        
                        files_dict[norm_title] = {
                            'title': title,
                            'original_title': title,
                            'normalized_title': norm_title,
                            'content': format_post(doc.get('caption', ''), max_length=500),
                            'post_content': doc.get('caption', ''),
                            'quality_options': {quality: quality_option},
                            'quality_list': [quality],
                            'date': doc['date'].isoformat() if isinstance(doc['date'], datetime) else doc['date'],
                            'is_new': is_new(doc['date']) if doc.get('date') else False,
                            'is_video_file': doc.get('is_video_file', False),
                            'channel_id': doc.get('channel_id'),
                            'channel_name': f"File Channel {doc.get('channel_id')}",
                            'has_file': True,
                            'has_post': bool(doc.get('caption')),
                            'file_caption': doc.get('caption', ''),
                            'year': year,
                            'quality': quality,
                            'real_message_id': doc.get('real_message_id') or doc.get('message_id'),
                            'search_score': 3 if query_lower in title.lower() else 2,
                            'result_type': 'file_only',
                            'total_files': 1,
                            'file_sizes': [doc.get('file_size', 0)],
                            'is_file_channel': True
                        }
                    else:
                        existing = files_dict[norm_title]
                        if quality not in existing['quality_options']:
                            existing['quality_options'][quality] = quality_option
                            existing['quality_list'].append(quality)
                            existing['total_files'] += 1
                            existing['file_sizes'].append(doc.get('file_size', 0))
                        
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
                        
                        existing['has_file'] = True
                        
                except Exception as e:
                    logger.error(f"File processing error: {e}")
                    continue
            
            logger.info(f"✅ Found {file_count} files in database for query: {query}")
            
        except Exception as e:
            logger.error(f"❌ File search error: {e}")
    
    # ============================================================================
    # ✅ 3. MERGE POSTS AND FILES - PROPER LOGIC
    # ============================================================================
    merged = {}
    
    # First, handle TEXT CHANNEL posts
    for norm_title, post_data in posts_dict.items():
        merged[norm_title] = post_data.copy()
        
        if norm_title in files_dict:
            file_data = files_dict[norm_title]
            
            merged[norm_title].update({
                'has_file': True,
                'file_available': True,
                'quality_options': file_data['quality_options'],
                'quality_list': file_data['quality_list'],
                'quality': file_data.get('quality', 'Unknown'),
                'total_files': file_data.get('total_files', 1),
                'file_sizes': file_data.get('file_sizes', []),
                'result_type': 'post_with_file',
                'file_channel_id': file_data.get('channel_id'),
                'file_message_id': file_data.get('real_message_id'),
                'search_score': max(merged[norm_title].get('search_score', 0), file_data.get('search_score', 0))
            })
            
            del files_dict[norm_title]
        else:
            if post_data.get('file_mentioned', False):
                merged[norm_title]['result_type'] = 'post_with_file_mention'
            else:
                merged[norm_title]['result_type'] = 'post_only'
    
    # Second, add FILE-ONLY results
    for norm_title, file_data in files_dict.items():
        merged[norm_title] = file_data.copy()
        merged[norm_title]['result_type'] = 'file_only'
    
    # ============================================================================
    # ✅ 4. FETCH POSTERS
    # ============================================================================
    if merged:
        logger.info(f"🎬 Fetching posters for {len(merged)} movies...")
        
        movies_for_posters = []
        for norm_title, movie_data in merged.items():
            movies_for_posters.append(movie_data)
        
        movies_without_thumbnails = [m for m in movies_for_posters if not m.get('has_poster')]
        
        if movies_without_thumbnails:
            try:
                movies_with_posters = await get_posters_for_movies_batch(movies_without_thumbnails)
            except Exception as e:
                logger.error(f"Error fetching posters: {e}")
                movies_with_posters = []
        else:
            movies_with_posters = []
        
        # Update merged with poster data
        poster_map = {}
        for movie in movies_with_posters:
            norm_title = movie.get('normalized_title', normalize_title(movie['title']))
            poster_map[norm_title] = movie
        
        for norm_title, movie_data in merged.items():
            if norm_title in poster_map:
                poster_data = poster_map[norm_title]
                merged[norm_title].update({
                    'poster_url': poster_data['poster_url'],
                    'poster_source': poster_data['poster_source'],
                    'poster_rating': poster_data['poster_rating'],
                    'thumbnail': poster_data['thumbnail'],
                    'thumbnail_source': poster_data['thumbnail_source'],
                    'has_poster': True,
                    'has_thumbnail': True
                })
            elif not movie_data.get('has_thumbnail'):
                merged[norm_title].update({
                    'poster_url': Config.FALLBACK_POSTER,
                    'poster_source': 'fallback',
                    'poster_rating': '0.0',
                    'thumbnail': Config.FALLBACK_POSTER,
                    'thumbnail_source': 'fallback',
                    'has_poster': True,
                    'has_thumbnail': True
                })
    
    # ============================================================================
    # ✅ 5. SORT AND PAGINATE
    # ============================================================================
    results_list = list(merged.values())
    
    # Sort by relevance
    results_list.sort(key=lambda x: (
        x.get('has_file', False),
        x.get('search_score', 0),
        x.get('is_new', False),
        x.get('has_thumbnail', False),
    ), reverse=True)
    
    total = len(results_list)
    paginated = results_list[offset:offset + limit]
    
    # Statistics
    result_types = {}
    for result in results_list:
        result_type = result.get('result_type', 'unknown')
        result_types[result_type] = result_types.get(result_type, 0) + 1
    
    stats = {
        'total': total,
        'result_types': result_types,
        'with_files': sum(1 for r in results_list if r.get('has_file', False)),
        'with_posts': sum(1 for r in results_list if r.get('has_post', False))
    }
    
    logger.info(f"📊 Result types: {result_types}")
    
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
            'result_types': result_types,
            'quality_merging': True,
            'poster_fetcher': poster_fetcher is not None,
            'user_session_used': user_session_ready,
            'cache_hit': False,
            'real_message_ids': True
        },
        'bot_username': Config.BOT_USERNAME
    }
    
    # Cache results
    if cache_manager is not None:
        try:
            await cache_manager.set(cache_key, result_data, expire_seconds=Config.SEARCH_CACHE_TTL)
        except:
            pass
    
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
                    
                    year_match = re.search(r'\b(19|20)\d{2}\b', title)
                    year = year_match.group() if year_match else ""
                    
                    clean_title = re.sub(r'\s+\(\d{4}\)$', '', title)
                    clean_title = re.sub(r'\s+\d{4}$', '', clean_title)
                    
                    post_content = msg.text
                    formatted_content = format_post(msg.text, max_length=500)
                    
                    movie_data = {
                        'title': clean_title,
                        'original_title': title,
                        'year': year,
                        'date': msg.date.isoformat() if isinstance(msg.date, datetime) else str(msg.date),
                        'is_new': is_new(msg.date) if msg.date else False,
                        'channel': f"Channel {Config.MAIN_CHANNEL_ID}",
                        'channel_id': Config.MAIN_CHANNEL_ID,
                        'message_id': msg.id,
                        'has_file': False,
                        'has_post': True,
                        'content': formatted_content,
                        'post_content': post_content,
                        'quality_options': {},
                        'is_video_file': False,
                        'result_type': 'post_only'
                    }
                    
                    movies.append(movie_data)
                    
                    if len(movies) >= limit:
                        break
        
        if movies:
            try:
                movies_with_posters = await get_posters_for_movies_batch(movies)
                logger.info(f"✅ Fetched {len(movies_with_posters)} home movies")
                return movies_with_posters[:limit]
            except Exception as e:
                logger.error(f"Error fetching posters for home movies: {e}")
                return movies[:limit]
        else:
            logger.warning("⚠️ No movies found for home page")
            return []
        
    except Exception as e:
        logger.error(f"❌ Home movies error: {e}")
        return []

# ============================================================================
# ✅ API ROUTES - FIXED ERROR HANDLING
# ============================================================================

@app.route('/')
@performance_monitor.measure("root_endpoint")
async def root():
    try:
        tf = video_files = thumbnails_extracted = 0
        if files_col is not None:
            try:
                tf = await files_col.count_documents({})
                video_files = await files_col.count_documents({'is_video_file': True})
                thumbnails_extracted = await files_col.count_documents({'thumbnail_extracted': True})
            except Exception as e:
                logger.warning(f"Could not get database stats: {e}")
        
        indexing_status = {}
        try:
            indexing_status = await file_indexing_manager.get_indexing_status()
        except Exception as e:
            logger.warning(f"Could not get indexing status: {e}")
        
        bot_status = None
        if bot_handler:
            try:
                bot_status = await bot_handler.get_bot_status()
            except Exception as e:
                logger.warning(f"Could not get bot status: {e}")
        
        return jsonify({
            'status': 'healthy',
            'service': 'SK4FiLM v8.6.1 - FIXED ERRORS',
            'sessions': {
                'user_session': {'ready': user_session_ready},
                'bot_session': {'ready': bot_session_ready},
                'bot_handler': bot_status
            },
            'features': {
                'result_types': True,
                'file_channel_indexing': True,
                'real_message_ids': True,
                'quality_merging': True
            },
            'stats': {
                'total_files': tf,
                'video_files': video_files,
                'thumbnails_extracted': thumbnails_extracted
            },
            'indexing': indexing_status,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error in root endpoint: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Internal server error',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/health')
@performance_monitor.measure("health_endpoint")
async def health():
    try:
        indexing_status = {}
        try:
            indexing_status = await file_indexing_manager.get_indexing_status()
        except:
            pass
        
        bot_status = None
        if bot_handler:
            try:
                bot_status = await bot_handler.get_bot_status()
            except:
                pass
        
        return jsonify({
            'status': 'ok',
            'sessions': {
                'user': user_session_ready,
                'bot': bot_session_ready,
                'bot_handler': bot_status.get('initialized') if bot_status else False
            },
            'indexing': {
                'running': indexing_status.get('is_running', False),
                'last_run': indexing_status.get('last_run')
            },
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error in health endpoint: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Internal server error'
        }), 500

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
                'feature': 'file_channel_search',
                'quality_priority': Config.QUALITY_PRIORITY,
                'real_message_ids': True
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
        
        poster_stats = {}
        if poster_fetcher and hasattr(poster_fetcher, 'get_stats'):
            try:
                poster_stats = poster_fetcher.get_stats()
            except:
                pass
        
        total_files = video_files = thumbnails_extracted = 0
        indexing_status = {}
        if files_col is not None:
            try:
                total_files = await files_col.count_documents({})
                video_files = await files_col.count_documents({'is_video_file': True})
                thumbnails_extracted = await files_col.count_documents({'thumbnail_extracted': True})
                indexing_status = await file_indexing_manager.get_indexing_status()
            except Exception as e:
                logger.warning(f"Could not get database stats: {e}")
        
        bot_status = None
        if bot_handler:
            try:
                bot_status = await bot_handler.get_bot_status()
            except:
                pass
        
        return jsonify({
            'status': 'success',
            'performance': perf_stats,
            'poster_fetcher': poster_stats,
            'database_stats': {
                'total_files': total_files,
                'video_files': video_files,
                'thumbnails_extracted': thumbnails_extracted
            },
            'indexing_stats': indexing_status,
            'bot_handler': bot_status,
            'real_message_ids': True,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Stats API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# ============================================================================
# ✅ VIEW PAGE API - FOR VIDEO STREAMING
# ============================================================================

@app.route('/api/view', methods=['GET'])
@performance_monitor.measure("view_endpoint")
async def api_view():
    """Get file information for viewing/streaming"""
    try:
        channel_id = request.args.get('channel_id', type=int)
        message_id = request.args.get('message_id', type=int)
        quality = request.args.get('quality', '')
        
        if not channel_id or not message_id:
            return jsonify({
                'status': 'error',
                'message': 'Missing channel_id or message_id'
            }), 400
        
        file_info = None
        if files_col is not None:
            try:
                query = {
                    'channel_id': channel_id,
                    'message_id': message_id
                }
                if quality:
                    query['quality'] = quality
                
                file_info = await files_col.find_one(query)
            except Exception as e:
                logger.error(f"Database error in view API: {e}")
        
        if not file_info and user_session_ready:
            try:
                message = await User.get_messages(channel_id, message_id)
                if message and (message.document or message.video):
                    file_info = {
                        'channel_id': channel_id,
                        'message_id': message_id,
                        'title': message.caption or '',
                        'file_name': '',
                        'file_size': 0,
                        'is_video_file': False,
                        'telegram_file_id': '',
                        'quality': quality or 'Unknown'
                    }
                    
                    if message.document:
                        file_info.update({
                            'file_name': message.document.file_name or '',
                            'file_size': message.document.file_size or 0,
                            'is_video_file': is_video_file(message.document.file_name or ''),
                            'telegram_file_id': message.document.file_id,
                            'mime_type': message.document.mime_type or ''
                        })
                    elif message.video:
                        file_info.update({
                            'file_name': message.video.file_name or 'video.mp4',
                            'file_size': message.video.file_size or 0,
                            'is_video_file': True,
                            'telegram_file_id': message.video.file_id,
                            'duration': message.video.duration if hasattr(message.video, 'duration') else 0,
                            'width': message.video.width if hasattr(message.video, 'width') else 0,
                            'height': message.video.height if hasattr(message.video, 'height') else 0
                        })
            except Exception as e:
                logger.error(f"Error getting message from Telegram: {e}")
        
        if not file_info:
            return jsonify({
                'status': 'error',
                'message': 'File not found'
            }), 404
        
        # Get streaming URL
        streaming_url = ""
        if file_info.get('telegram_file_id') and user_session_ready:
            streaming_url = f"https://t.me/{Config.CHANNEL_USERNAME}/{message_id}"
        
        # Get available qualities for this title
        available_qualities = []
        if files_col is not None and file_info.get('title'):
            try:
                norm_title = normalize_title(file_info['title'])
                pipeline = [
                    {"$match": {"normalized_title": norm_title}},
                    {"$group": {"_id": "$quality", "count": {"$sum": 1}}},
                    {"$sort": {"_id": -1}}
                ]
                quality_cursor = await files_col.aggregate(pipeline).to_list(length=10)
                available_qualities = [q['_id'] for q in quality_cursor if q['_id']]
            except Exception as e:
                logger.error(f"Error getting available qualities: {e}")
        
        return jsonify({
            'status': 'success',
            'file': {
                'title': file_info.get('title', ''),
                'file_name': file_info.get('file_name', ''),
                'file_size': file_info.get('file_size', 0),
                'file_size_formatted': format_size(file_info.get('file_size', 0)),
                'is_video_file': file_info.get('is_video_file', False),
                'quality': file_info.get('quality', 'Unknown'),
                'channel_id': channel_id,
                'message_id': message_id,
                'telegram_file_id': file_info.get('telegram_file_id', ''),
                'streaming_url': streaming_url,
                'duration': file_info.get('duration', 0),
                'width': file_info.get('width', 0),
                'height': file_info.get('height', 0),
                'mime_type': file_info.get('mime_type', '')
            },
            'available_qualities': available_qualities,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"View API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/stream', methods=['GET'])
async def api_stream():
    """Streaming endpoint (placeholder)"""
    try:
        channel_id = request.args.get('channel_id', type=int)
        message_id = request.args.get('message_id', type=int)
        
        if not channel_id or not message_id:
            return Response("Missing parameters", status=400)
        
        return Response(
            f"Streaming not implemented yet. File: {channel_id}/{message_id}",
            mimetype="text/plain"
        )
        
    except Exception as e:
        logger.error(f"Stream API error: {e}")
        return Response(f"Error: {str(e)}", status=500)

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
        
        asyncio.create_task(file_indexing_manager.start_indexing())
        
        return jsonify({
            'status': 'success',
            'message': 'File channel reindexing started',
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
        
        total_files = 0
        if files_col is not None:
            try:
                total_files = await files_col.count_documents({})
            except:
                pass
        
        return jsonify({
            'status': 'success',
            'indexing': indexing_status,
            'database_files': total_files,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Indexing status error: {e}")
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
    logger.info("🛑 Shutting down SK4FiLM v8.6.1...")
    
    # Stop indexing
    try:
        await file_indexing_manager.stop_indexing()
    except:
        pass
    
    # Shutdown bot handler
    if bot_handler:
        try:
            await bot_handler.shutdown()
        except:
            pass
    
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
    
    logger.info(f"🌐 Starting SK4FiLM v8.6.1 on port {Config.WEB_SERVER_PORT}...")
    logger.info("🎯 FEATURES: FIXED ERRORS VERSION")
    logger.info("   • Fixed MongoDB index conflicts")
    logger.info("   • Added missing get_bot_status method")
    logger.info("   • Enhanced error handling")
    logger.info("   • Post-only Results: ✅")
    logger.info("   • Post + File Results: ✅")
    logger.info("   • File-only Results: ✅")
    logger.info("   • View Page API: ✅")
    
    asyncio.run(serve(app, config))
