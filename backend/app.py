# ============================================================================
# 🚀 SK4FiLM v9.2 - FULLY FIXED WITH PERFORMANCE OPTIMIZATIONS
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
from quart import Quart, jsonify, request, Response, send_file, redirect
from hypercorn.asyncio import serve
from hypercorn.config import Config as HyperConfig
from motor.motor_asyncio import AsyncIOMotorClient
import redis.asyncio as redis

# ✅ IMPORT ALL MODULES WITH PROPER ERROR HANDLING
# ============================================================================

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
# ✅ PERFORMANCE MONITOR WITH OPTIMIZATIONS
# ============================================================================

class PerformanceMonitor:
    def __init__(self):
        self.measurements = {}
        self.request_times = {}
        self.slow_threshold = 1.0  # 1 second threshold
    
    def measure(self, name):
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = await func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                
                self._record(name, elapsed)
                
                # Log warning if too slow
                if elapsed > self.slow_threshold:
                    logger.warning(f"⏱️ {name} took {elapsed:.3f}s - OPTIMIZATION NEEDED")
                
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
                'min': float('inf'),
                'slow_count': 0
            }
        
        stats = self.measurements[name]
        stats['count'] += 1
        stats['total'] += elapsed
        stats['avg'] = stats['total'] / stats['count']
        stats['max'] = max(stats['max'], elapsed)
        stats['min'] = min(stats['min'], elapsed)
        
        if elapsed > self.slow_threshold:
            stats['slow_count'] += 1
    
    def get_stats(self):
        return self.measurements

performance_monitor = PerformanceMonitor()

# ============================================================================
# ✅ CONFIGURATION - OPTIMIZED WITH PERFORMANCE SETTINGS
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
    
    # PERFORMANCE OPTIMIZATION SETTINGS
    MAX_CONCURRENT_REQUESTS = int(os.environ.get("MAX_CONCURRENT_REQUESTS", "30"))  # Reduced
    CACHE_TTL = int(os.environ.get("CACHE_TTL", "600"))  # Increased
    REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "15"))  # Increased
    
    # Sync Management Settings
    MONITOR_INTERVAL = int(os.environ.get("MONITOR_INTERVAL", "300"))
    
    # Quality Settings
    QUALITY_PRIORITY = ['2160p', '1080p', '720p', '480p', '360p']
    HEVC_VARIANTS = ['720p HEVC', '1080p HEVC', '2160p HEVC']
    
    # Fallback Poster
    FALLBACK_POSTER = "https://iili.io/fAeIwv9.th.png"
    
    # Thumbnail Settings
    THUMBNAIL_EXTRACT_TIMEOUT = 15  # Increased
    THUMBNAIL_CACHE_DURATION = 24 * 60 * 60
    
    # 🔥 FILE CHANNEL INDEXING SETTINGS - OPTIMIZED
    AUTO_INDEX_INTERVAL = int(os.environ.get("AUTO_INDEX_INTERVAL", "1800"))  # 30 minutes
    BATCH_INDEX_SIZE = int(os.environ.get("BATCH_INDEX_SIZE", "50"))  # Smaller batches
    MAX_INDEX_LIMIT = int(os.environ.get("MAX_INDEX_LIMIT", "0"))
    INDEX_ALL_HISTORY = os.environ.get("INDEX_ALL_HISTORY", "false").lower() == "true"
    INSTANT_AUTO_INDEX = os.environ.get("INSTANT_AUTO_INDEX", "false").lower() == "true"
    
    # 🔥 SEARCH SETTINGS - OPTIMIZED
    SEARCH_MIN_QUERY_LENGTH = 2
    SEARCH_RESULTS_PER_PAGE = 12
    SEARCH_CACHE_TTL = 1200  # Increased to 20 minutes
    
    # 🔥 STREAMING SETTINGS
    STREAMING_ENABLED = os.environ.get("STREAMING_ENABLED", "true").lower() == "true"
    STREAMING_CACHE_TTL = 3600
    MAX_STREAM_SIZE = 2 * 1024 * 1024 * 1024
    STREAM_CHUNK_SIZE = 1024 * 1024
    
    # 🔥 RATE LIMITING SETTINGS - OPTIMIZED
    TELEGRAM_API_RATE_LIMIT = float(os.environ.get("TELEGRAM_API_RATE_LIMIT", "0.3"))  # Reduced
    MIN_BATCH_DELAY = int(os.environ.get("MIN_BATCH_DELAY", "10"))  # Increased
    
    # 🔥 NEW: PERFORMANCE SETTINGS
    ENABLE_AGGRESSIVE_CACHING = os.environ.get("ENABLE_AGGRESSIVE_CACHING", "true").lower() == "true"
    CACHE_HOME_MOVIES_FOR = int(os.environ.get("CACHE_HOME_MOVIES_FOR", "300"))  # 5 minutes
    CACHE_SEARCH_RESULTS_FOR = int(os.environ.get("CACHE_SEARCH_RESULTS_FOR", "600"))  # 10 minutes
    MAX_PARALLEL_POSTER_REQUESTS = int(os.environ.get("MAX_PARALLEL_POSTER_REQUESTS", "5"))  # Reduced

# ============================================================================
# ✅ RATE LIMITER - OPTIMIZED
# ============================================================================

class RateLimiter:
    def __init__(self, calls_per_second=0.3):
        self.calls_per_second = calls_per_second
        self.last_call = 0
        self.lock = asyncio.Lock()
        self.queue_size = 0
    
    async def wait_if_needed(self):
        async with self.lock:
            self.queue_size += 1
            if self.queue_size > 5:
                logger.warning(f"⚠️ High rate limit queue: {self.queue_size}")
            
            now = time.time()
            time_since_last = now - self.last_call
            if time_since_last < 1.0 / self.calls_per_second:
                wait_time = (1.0 / self.calls_per_second) - time_since_last
                if wait_time > 0.5:
                    logger.debug(f"⏳ Rate limiting: waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
            self.last_call = time.time()
            self.queue_size -= 1
    
    def get_queue_size(self):
        return self.queue_size

telegram_rate_limiter = RateLimiter(calls_per_second=Config.TELEGRAM_API_RATE_LIMIT)

# ============================================================================
# ✅ FAST INITIALIZATION
# ============================================================================

app = Quart(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

@app.after_request
async def add_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS, HEAD'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['X-SK4FiLM-Version'] = '9.2-PERFORMANCE-OPTIMIZED'
    response.headers['X-Response-Time'] = f"{time.perf_counter():.3f}"
    
    # Add cache headers for performance
    if request.path in ['/api/movies', '/api/search', '/']:
        response.headers['Cache-Control'] = f'public, max-age={Config.CACHE_TTL}'
    
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
# ✅ BOT HANDLER MODULE - OPTIMIZED WITH ERROR RECOVERY
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
        self.error_count = 0
        self.max_errors = 5
        
    async def initialize(self):
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
                sleep_threshold=60,  # Increased
                in_memory=True,
                no_updates=True
            )
            
            await self.bot.start()
            bot_info = await self.bot.get_me()
            self.bot_username = bot_info.username
            logger.info(f"✅ Bot Handler Ready: @{self.bot_username}")
            self.initialized = True
            self.last_update = datetime.now()
            self.error_count = 0
            
            asyncio.create_task(self._periodic_tasks())
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Bot handler initialization error: {e}")
            self.error_count += 1
            return False
    
    async def _periodic_tasks(self):
        while self.initialized:
            try:
                self.last_update = datetime.now()
                
                # Check connection
                try:
                    await telegram_rate_limiter.wait_if_needed()
                    await self.bot.get_me()
                    self.error_count = 0  # Reset on success
                except Exception as e:
                    self.error_count += 1
                    logger.warning(f"⚠️ Bot session check error #{self.error_count}: {e}")
                    
                    if self.error_count >= self.max_errors:
                        logger.warning("⚠️ Too many errors, reconnecting bot...")
                        try:
                            await self.bot.stop()
                            await asyncio.sleep(10)
                            await self.bot.start()
                            self.error_count = 0
                            logger.info("✅ Bot reconnected successfully")
                        except Exception as reconnect_error:
                            logger.error(f"❌ Bot reconnection failed: {reconnect_error}")
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Bot handler periodic task error: {e}")
                await asyncio.sleep(60)
    
    async def get_file_info(self, channel_id, message_id):
        if not self.initialized:
            return None
        
        try:
            await telegram_rate_limiter.wait_if_needed()
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
            self.error_count += 1
            return None
    
    async def get_file_download_url(self, file_id):
        try:
            if not self.initialized or not self.bot_token:
                return None
            
            try:
                await telegram_rate_limiter.wait_if_needed()
                file = await self.bot.get_file(file_id)
                if file and hasattr(file, 'file_path') and file.file_path:
                    return f"https://api.telegram.org/file/bot{self.bot_token}/{file.file_path}"
            except Exception as get_file_error:
                logger.debug(f"Get file error, trying alternative: {get_file_error}")
            
            # Alternative method with timeout
            try:
                async with aiohttp.ClientSession() as session:
                    api_url = f"https://api.telegram.org/bot{self.bot_token}/getFile"
                    params = {'file_id': file_id}
                    
                    async with session.get(api_url, params=params, timeout=10) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data.get('ok') and data.get('result'):
                                file_path = data['result']['file_path']
                                return f"https://api.telegram.org/file/bot{self.bot_token}/{file_path}"
            except asyncio.TimeoutError:
                logger.warning(f"⚠️ Timeout getting file URL for {file_id[:20]}...")
            except Exception as api_error:
                logger.debug(f"API fallback error: {api_error}")
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Get file download URL error: {e}")
            return None
    
    async def extract_thumbnail(self, channel_id, message_id):
        if not self.initialized:
            return None
        
        try:
            await telegram_rate_limiter.wait_if_needed()
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
        try:
            await telegram_rate_limiter.wait_if_needed()
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
        if not self.initialized:
            return False
        
        try:
            await telegram_rate_limiter.wait_if_needed()
            message = await self.bot.get_messages(channel_id, message_id)
            return message is not None
        except:
            return False
    
    async def get_bot_status(self):
        if not self.initialized:
            return {
                'initialized': False,
                'last_update': None,
                'bot_username': None,
                'error_count': self.error_count
            }
        
        try:
            await telegram_rate_limiter.wait_if_needed()
            bot_info = await self.bot.get_me()
            return {
                'initialized': self.initialized,
                'last_update': self.last_update.isoformat() if self.last_update else None,
                'bot_username': bot_info.username if bot_info else self.bot_username,
                'bot_id': bot_info.id if bot_info else None,
                'session_active': True,
                'error_count': self.error_count,
                'rate_limit_queue': telegram_rate_limiter.get_queue_size()
            }
        except:
            return {
                'initialized': self.initialized,
                'last_update': self.last_update.isoformat() if self.last_update else None,
                'bot_username': self.bot_username,
                'session_active': False,
                'error_count': self.error_count
            }
    
    async def shutdown(self):
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
# ✅ BOT INITIALIZATION FUNCTION
# ============================================================================

async def start_telegram_bot():
    try:
        if not PYROGRAM_AVAILABLE:
            logger.warning("❌ Pyrogram not available, bot won't start")
            return None
        
        if not Config.BOT_TOKEN:
            logger.warning("❌ Bot token not configured, bot won't start")
            return None
        
        logger.info("🤖 Starting SK4FiLM Telegram Bot...")
        
        try:
            from bot_handlers import SK4FiLMBot
            logger.info("✅ Bot handler module imported")
        except ImportError as e:
            logger.error(f"❌ Bot handler import error: {e}")
            class FallbackBot:
                def __init__(self):
                    self.bot_started = False
                async def initialize(self): 
                    logger.warning("⚠️ Using fallback bot")
                    return False
                async def shutdown(self): 
                    logger.info("✅ Fallback bot shutdown")
            return FallbackBot()
        
        bot_instance = SK4FiLMBot(Config, db_manager=None)
        
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
# ✅ ASYNC CACHE DECORATOR - ENHANCED
# ============================================================================

def async_cache_with_ttl(maxsize=128, ttl=300):
    cache = {}
    cache_timestamps = {}
    cache_lock = asyncio.Lock()
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not Config.ENABLE_AGGRESSIVE_CACHING:
                return await func(*args, **kwargs)
            
            key_parts = [func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
            key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
            
            now = time.time()
            
            async with cache_lock:
                if key in cache:
                    value, timestamp = cache[key]
                    if now - timestamp < ttl:
                        logger.debug(f"✅ Cache HIT for {func.__name__}")
                        return value
                    else:
                        # Expired, remove
                        del cache[key]
                        del cache_timestamps[key]
            
            result = await func(*args, **kwargs)
            
            async with cache_lock:
                cache[key] = (result, now)
                cache_timestamps[key] = now
                
                # Clean up if cache too big
                if len(cache) > maxsize:
                    # Remove oldest entries
                    sorted_keys = sorted(cache_timestamps.keys(), key=lambda k: cache_timestamps[k])
                    keys_to_remove = sorted_keys[:len(cache) - maxsize]
                    for k in keys_to_remove:
                        del cache[k]
                        del cache_timestamps[k]
            
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
    if not filename:
        return "480p"
    
    filename_lower = filename.lower()
    
    is_hevc = any(re.search(pattern, filename_lower) for pattern in HEVC_PATTERNS)
    
    for pattern, quality in QUALITY_PATTERNS:
        if re.search(pattern, filename_lower):
            if is_hevc and quality in ['720p', '1080p', '2160p']:
                return f"{quality} HEVC"
            return quality
    
    return "480p"

def extract_quality_info(filename):
    quality = detect_quality_enhanced(filename)
    
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
# ✅ QUALITY MERGER
# ============================================================================

class QualityMerger:
    @staticmethod
    def merge_quality_options(quality_options_dict):
        if not quality_options_dict:
            return {}
        
        merged = {}
        
        for quality, option in quality_options_dict.items():
            base_quality = quality.replace(' HEVC', '')
            
            if base_quality not in merged:
                merged[base_quality] = {
                    'qualities': [],
                    'best_option': None,
                    'total_size': 0,
                    'file_count': 0
                }
            
            merged[base_quality]['qualities'].append({
                'full_quality': quality,
                'is_hevc': 'HEVC' in quality,
                'file_id': option.get('file_id'),
                'file_size': option.get('file_size', 0),
                'file_name': option.get('file_name', ''),
                'is_video': option.get('is_video', False),
                'channel_id': option.get('channel_id'),
                'message_id': option.get('message_id'),
                'real_message_id': option.get('real_message_id'),
                'telegram_file_id': option.get('telegram_file_id'),
                'thumbnail_url': option.get('thumbnail_url')
            })
            
            merged[base_quality]['total_size'] += option.get('file_size', 0)
            merged[base_quality]['file_count'] += 1
            
            if merged[base_quality]['best_option'] is None:
                merged[base_quality]['best_option'] = quality
            else:
                current_priority = Config.QUALITY_PRIORITY.index(base_quality) if base_quality in Config.QUALITY_PRIORITY else 999
                best_base = merged[base_quality]['best_option'].replace(' HEVC', '')
                best_priority = Config.QUALITY_PRIORITY.index(best_base) if best_base in Config.QUALITY_PRIORITY else 999
                
                if current_priority < best_priority:
                    merged[base_quality]['best_option'] = quality
        
        sorted_merged = {}
        for quality in Config.QUALITY_PRIORITY:
            if quality in merged:
                sorted_merged[quality] = merged[quality]
        
        for quality in merged:
            if quality not in sorted_merged:
                sorted_merged[quality] = merged[quality]
        
        return sorted_merged
    
    @staticmethod
    def get_quality_summary(merged_options):
        if not merged_options:
            return "No files"
        
        qualities = list(merged_options.keys())
        
        sorted_qualities = []
        for quality in Config.QUALITY_PRIORITY:
            if quality in qualities:
                sorted_qualities.append(quality)
                qualities.remove(quality)
        
        sorted_qualities.extend(sorted(qualities))
        
        summary_parts = []
        for quality in sorted_qualities[:3]:
            data = merged_options[quality]
            count = data['file_count']
            if count > 1:
                summary_parts.append(f"{quality} ({count} files)")
            else:
                summary_parts.append(quality)
        
        if len(sorted_qualities) > 3:
            summary_parts.append(f"+{len(sorted_qualities) - 3} more")
        
        return " • ".join(summary_parts)

# ============================================================================
# ✅ VIDEO THUMBNAIL EXTRACTOR - OPTIMIZED
# ============================================================================

class VideoThumbnailExtractor:
    def __init__(self):
        self.extraction_lock = asyncio.Lock()
        self.thumbnail_cache = {}
    
    async def extract_thumbnail(self, channel_id: int, message_id: int) -> Optional[str]:
        cache_key = f"thumb_{channel_id}_{message_id}"
        
        # Check cache first
        if cache_key in self.thumbnail_cache:
            cached_data = self.thumbnail_cache[cache_key]
            if time.time() - cached_data['timestamp'] < Config.THUMBNAIL_CACHE_DURATION:
                logger.debug(f"✅ Thumbnail from cache: {channel_id}/{message_id}")
                return cached_data['thumbnail']
        
        try:
            if bot_handler and bot_handler.initialized:
                thumbnail_url = await bot_handler.extract_thumbnail(channel_id, message_id)
                if thumbnail_url:
                    logger.debug(f"✅ Thumbnail extracted via bot handler: {channel_id}/{message_id}")
                    # Cache it
                    self.thumbnail_cache[cache_key] = {
                        'thumbnail': thumbnail_url,
                        'timestamp': time.time()
                    }
                    return thumbnail_url
            
            if Bot is not None and bot_session_ready:
                try:
                    await telegram_rate_limiter.wait_if_needed()
                    message = await Bot.get_messages(channel_id, message_id)
                    if not message:
                        return None
                    
                    thumbnail_data = None
                    
                    if message.video:
                        if hasattr(message.video, 'thumbnail') and message.video.thumbnail:
                            thumbnail_file_id = message.video.thumbnail.file_id
                            await telegram_rate_limiter.wait_if_needed()
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
                            await telegram_rate_limiter.wait_if_needed()
                            download_path = await Bot.download_media(thumbnail_file_id, in_memory=True)
                            
                            if download_path:
                                if isinstance(download_path, bytes):
                                    thumbnail_data = download_path
                                else:
                                    with open(download_path, 'rb') as f:
                                        thumbnail_data = f.read()
                    
                    if thumbnail_data:
                        base64_data = base64.b64encode(thumbnail_data).decode('utf-8')
                        thumbnail_url = f"data:image/jpeg;base64,{base64_data}"
                        
                        # Cache it
                        self.thumbnail_cache[cache_key] = {
                            'thumbnail': thumbnail_url,
                            'timestamp': time.time()
                        }
                        
                        return thumbnail_url
                    
                except Exception as e:
                    logger.error(f"❌ Bot session thumbnail extraction error: {e}")
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Thumbnail extraction failed: {e}")
            return None

thumbnail_extractor = VideoThumbnailExtractor()

# ============================================================================
# ✅ DUPLICATE PREVENTION SYSTEM - FIXED VERSION
# ============================================================================

class DuplicatePreventionSystem:
    def __init__(self):
        self.file_hashes = set()
        self.title_cache = defaultdict(set)
        self.lock = asyncio.Lock()
    
    async def initialize_from_database(self):
        if files_col is None:
            return
        
        try:
            async with self.lock:
                self.file_hashes.clear()
                self.title_cache.clear()
                
                # Only get recent files to reduce memory usage
                cursor = files_col.find(
                    {"file_hash": {"$ne": None}},
                    {"file_hash": 1, "normalized_title": 1, "_id": 0}
                ).limit(10000)  # Limit to 10k files
                
                async for doc in cursor:
                    file_hash = doc.get('file_hash')
                    normalized_title = doc.get('normalized_title')
                    
                    if file_hash:
                        self.file_hashes.add(file_hash)
                    
                    if normalized_title and file_hash:
                        self.title_cache[normalized_title].add(file_hash)
                
                logger.info(f"✅ Loaded {len(self.file_hashes)} file hashes from database")
                logger.info(f"✅ Loaded {len(self.title_cache)} unique titles from database")
                
        except Exception as e:
            logger.error(f"❌ Error initializing duplicate prevention: {e}")
    
    async def is_duplicate_file(self, file_hash, normalized_title=None):
        if not file_hash:
            return False, "no_hash"
        
        async with self.lock:
            if file_hash in self.file_hashes:
                logger.debug(f"🔍 Duplicate found by hash: {file_hash[:16]}...")
                return True, "same_hash"
            
            # Only check title cache if we have the exact same file hash
            if normalized_title and normalized_title in self.title_cache:
                if file_hash in self.title_cache[normalized_title]:
                    logger.debug(f"🔍 Duplicate found in title cache: {normalized_title[:30]}")
                    return True, "title_hash_match"
            
            return False, "unique"
    
    async def add_file_hash(self, file_hash, normalized_title=None):
        if not file_hash:
            return
        
        async with self.lock:
            self.file_hashes.add(file_hash)
            
            if normalized_title:
                self.title_cache[normalized_title].add(file_hash)
    
    async def remove_file_hash(self, file_hash, normalized_title=None):
        if not file_hash:
            return
        
        async with self.lock:
            if file_hash in self.file_hashes:
                self.file_hashes.remove(file_hash)
            
            if normalized_title and normalized_title in self.title_cache:
                if file_hash in self.title_cache[normalized_title]:
                    self.title_cache[normalized_title].remove(file_hash)
                
                if not self.title_cache[normalized_title]:
                    del self.title_cache[normalized_title]
    
    async def get_duplicate_stats(self):
        async with self.lock:
            return {
                'total_unique_hashes': len(self.file_hashes),
                'total_unique_titles': len(self.title_cache),
                'files_per_title': {
                    title: len(hashes) 
                    for title, hashes in list(self.title_cache.items())[:10]
                }
            }

duplicate_prevention = DuplicatePreventionSystem()

# ============================================================================
# ✅ FILE CHANNEL INDEXING MANAGER - PERFORMANCE OPTIMIZED
# ============================================================================

class FileChannelIndexingManager:
    def __init__(self):
        self.is_running = False
        self.indexing_task = None
        self.last_run = None
        self.next_run = None
        self.indexed_files = 0
        self.total_duplicates = 0
        
        self.indexing_stats = {
            'total_runs': 0,
            'total_files_processed': 0,
            'total_indexed': 0,
            'total_duplicates': 0,
            'total_errors': 0,
            'last_success': None
        }
        
        self.is_first_run = True
        self.emergency_stop = False
    
    async def start_indexing(self):
        if self.is_running:
            logger.warning("⚠️ File indexing already running")
            return
        
        logger.info("🚀 Starting SAFE FILE CHANNEL INDEXING...")
        self.is_running = True
        self.emergency_stop = False
        
        await duplicate_prevention.initialize_from_database()
        
        # Only index NEW files, not complete history
        if Config.INDEX_ALL_HISTORY and self.is_first_run:
            logger.info("🔍 Running LIMITED complete indexing (new files only)...")
            asyncio.create_task(self._run_limited_indexing())
        else:
            logger.info("🔍 Starting incremental indexing loop...")
        
        self.indexing_task = asyncio.create_task(self._indexing_loop())
    
    async def stop_indexing(self):
        self.is_running = False
        self.emergency_stop = True
        if self.indexing_task:
            self.indexing_task.cancel()
            try:
                await self.indexing_task
            except asyncio.CancelledError:
                pass
        
        logger.info("🛑 File indexing stopped")
    
    async def emergency_stop_indexing(self):
        """Emergency stop - immediately halt all indexing"""
        self.emergency_stop = True
        self.is_running = False
        if self.indexing_task:
            self.indexing_task.cancel()
        
        logger.critical("🛑 EMERGENCY STOP: Indexing halted immediately!")
    
    async def _run_limited_indexing(self):
        """Only index NEW files, not everything"""
        logger.info("🔥 RUNNING LIMITED INDEXING (NEW FILES ONLY)...")
        
        try:
            # Get the highest message ID already indexed
            last_indexed = await files_col.find_one(
                {"channel_id": Config.FILE_CHANNEL_ID}, 
                sort=[('message_id', -1)],
                projection={'message_id': 1}
            )
            
            last_message_id = last_indexed['message_id'] if last_indexed else 0
            logger.info(f"📊 Starting from message ID: {last_message_id}")
            
            messages_to_index = []
            total_fetched = 0
            
            # Only fetch messages newer than what we already have
            logger.info(f"📡 Fetching NEW messages from file channel (ID > {last_message_id})...")
            
            try:
                # Fetch in reverse order (newest first)
                async for msg in User.get_chat_history(
                    Config.FILE_CHANNEL_ID, 
                    limit=1000  # Limit to 1000 messages max
                ):
                    if self.emergency_stop:
                        logger.warning("🛑 Emergency stop triggered during fetch")
                        return
                    
                    total_fetched += 1
                    
                    if msg.id <= last_message_id:
                        # We've reached already indexed messages
                        logger.info(f"📊 Reached already indexed messages at ID: {msg.id}")
                        break
                    
                    if msg is not None and (msg.document or msg.video):
                        messages_to_index.append(msg)
                    
                    if total_fetched % 100 == 0:
                        logger.info(f"📥 Fetched {total_fetched} messages...")
                
                logger.info(f"✅ Found {len(messages_to_index)} new files (from {total_fetched} messages)")
                
            except Exception as e:
                logger.error(f"❌ Error fetching messages: {e}")
                return
            
            if messages_to_index:
                # Process newest first (reverse the order)
                messages_to_index.reverse()
                
                batch_size = 50  # Smaller batches
                total_batches = math.ceil(len(messages_to_index) / batch_size)
                
                logger.info(f"🔧 Processing {len(messages_to_index)} new files in {total_batches} batches...")
                
                for batch_num in range(total_batches):
                    if self.emergency_stop:
                        logger.warning("🛑 Emergency stop triggered during batch processing")
                        return
                    
                    start_idx = batch_num * batch_size
                    end_idx = min(start_idx + batch_size, len(messages_to_index))
                    batch = messages_to_index[start_idx:end_idx]
                    
                    logger.info(f"📦 Processing batch {batch_num + 1}/{total_batches} ({len(batch)} files)...")
                    
                    batch_stats = await self._process_indexing_batch(batch)
                    
                    self.indexing_stats['total_files_processed'] += batch_stats['processed']
                    self.indexing_stats['total_indexed'] += batch_stats['indexed']
                    self.indexed_files += batch_stats['indexed']
                    self.total_duplicates += batch_stats['duplicates']
                    self.indexing_stats['total_duplicates'] += batch_stats['duplicates']
                    self.indexing_stats['total_errors'] += batch_stats['errors']
                    
                    # Be gentle with Telegram API
                    if batch_num < total_batches - 1:
                        await asyncio.sleep(Config.MIN_BATCH_DELAY)
                
                logger.info("✅ LIMITED INDEXING FINISHED!")
                logger.info(f"📊 Stats: Indexed: {self.indexed_files}, Duplicates: {self.total_duplicates}")
            
            self.is_first_run = False
            
        except Exception as e:
            logger.error(f"❌ Complete indexing error: {e}")
    
    async def _indexing_loop(self):
        """Main incremental indexing loop"""
        while self.is_running and not self.emergency_stop:
            try:
                if self.next_run and self.next_run > datetime.now():
                    wait_seconds = (self.next_run - datetime.now()).total_seconds()
                    if wait_seconds > 60:
                        logger.info(f"⏰ Next index in {wait_seconds:.0f}s")
                    await asyncio.sleep(min(wait_seconds, 60))
                    continue
                
                await self._run_indexing_cycle()
                
                self.next_run = datetime.now() + timedelta(seconds=Config.AUTO_INDEX_INTERVAL)
                self.last_run = datetime.now()
                
                await asyncio.sleep(10)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Indexing loop error: {e}")
                await asyncio.sleep(60)
    
    async def _run_indexing_cycle(self):
        """Run a single indexing cycle"""
        logger.info("=" * 50)
        logger.info("🔄 FILE INDEXING CYCLE")
        logger.info("=" * 50)
        
        start_time = time.time()
        cycle_stats = {
            'processed': 0,
            'indexed': 0,
            'duplicates': 0,
            'errors': 0
        }
        
        try:
            if self.emergency_stop:
                return
            
            # Get the highest message ID already indexed
            last_indexed = await files_col.find_one(
                {"channel_id": Config.FILE_CHANNEL_ID}, 
                sort=[('message_id', -1)],
                projection={'message_id': 1}
            )
            
            last_message_id = last_indexed['message_id'] if last_indexed else 0
            
            logger.info(f"📊 Last indexed message ID: {last_message_id}")
            
            messages_to_index = []
            fetched_count = 0
            
            try:
                # Fetch only NEW messages
                async for msg in User.get_chat_history(
                    Config.FILE_CHANNEL_ID, 
                    limit=Config.BATCH_INDEX_SIZE
                ):
                    if self.emergency_stop:
                        break
                    
                    fetched_count += 1
                    
                    if msg.id <= last_message_id:
                        break
                    
                    if msg and (msg.document or msg.video):
                        messages_to_index.append(msg)
                
                logger.info(f"📥 Fetched {fetched_count} messages, found {len(messages_to_index)} new files")
                
            except Exception as e:
                logger.error(f"❌ Error fetching messages: {e}")
                return
            
            if messages_to_index:
                # Process newest first
                messages_to_index.reverse()
                batch_stats = await self._process_indexing_batch(messages_to_index)
                cycle_stats.update(batch_stats)
            
            self.indexing_stats['total_runs'] += 1
            self.indexing_stats['total_files_processed'] += cycle_stats['processed']
            self.indexing_stats['total_indexed'] += cycle_stats['indexed']
            self.indexed_files += cycle_stats['indexed']
            self.total_duplicates += cycle_stats['duplicates']
            self.indexing_stats['total_duplicates'] += cycle_stats['duplicates']
            self.indexing_stats['total_errors'] += cycle_stats['errors']
            self.indexing_stats['last_success'] = datetime.now()
            
            elapsed = time.time() - start_time
            
            logger.info("=" * 50)
            logger.info("📊 INDEXING CYCLE COMPLETE")
            logger.info("=" * 50)
            logger.info(f"⏱️  Time: {elapsed:.2f}s")
            logger.info(f"📥 Fetched: {fetched_count} messages")
            logger.info(f"📄 Processed: {cycle_stats['processed']} files")
            logger.info(f"✅ Indexed: {cycle_stats['indexed']} new files")
            logger.info(f"🔄 Duplicates: {cycle_stats['duplicates']} skipped")
            logger.info(f"❌ Errors: {cycle_stats['errors']}")
            logger.info(f"📈 Total Indexed: {self.indexed_files}")
            logger.info("=" * 50)
            
        except Exception as e:
            logger.error(f"❌ Indexing cycle failed: {e}")
            self.indexing_stats['total_errors'] += 1
    
    async def _process_indexing_batch(self, messages):
        """Process a batch of messages for indexing"""
        batch_stats = {
            'processed': 0,
            'indexed': 0,
            'duplicates': 0,
            'errors': 0
        }
        
        for msg in messages:
            if self.emergency_stop:
                break
            
            try:
                batch_stats['processed'] += 1
                
                # Check if already indexed
                existing = await files_col.find_one({
                    'channel_id': Config.FILE_CHANNEL_ID,
                    'message_id': msg.id
                }, {'_id': 1})
                
                if existing:
                    logger.debug(f"📝 Already indexed: {msg.id}")
                    batch_stats['duplicates'] += 1
                    continue
                
                success = await index_single_file_smart(msg)
                
                if success:
                    batch_stats['indexed'] += 1
                else:
                    batch_stats['duplicates'] += 1
                    
            except Exception as e:
                logger.error(f"❌ Error processing message {msg.id}: {e}")
                batch_stats['errors'] += 1
                continue
        
        logger.info(f"📦 Batch stats: {batch_stats}")
        return batch_stats
    
    async def get_indexing_status(self):
        return {
            'is_running': self.is_running,
            'is_first_run': self.is_first_run,
            'emergency_stop': self.emergency_stop,
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'next_run': self.next_run.isoformat() if self.next_run else None,
            'indexed_files': self.indexed_files,
            'total_duplicates': self.total_duplicates,
            'stats': self.indexing_stats
        }

file_indexing_manager = FileChannelIndexingManager()

# ============================================================================
# ✅ SYNC MANAGEMENT
# ============================================================================

class ChannelSyncManager:
    def __init__(self):
        self.is_monitoring = False
        self.monitoring_task = None
        self.deleted_count = 0
        self.last_sync = time.time()
    
    async def start_sync_monitoring(self):
        if self.is_monitoring:
            return
        
        logger.info("👁️ Starting sync monitoring...")
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
        try:
            if files_col is None:
                return
            
            current_time = time.time()
            if current_time - self.last_sync < 300:
                return
            
            self.last_sync = current_time
            
            cursor = files_col.find(
                {"channel_id": Config.FILE_CHANNEL_ID},
                {"message_id": 1, "_id": 0, "file_hash": 1, "normalized_title": 1}
            )
            
            message_data = []
            async for doc in cursor:
                msg_id = doc.get('message_id')
                if msg_id:
                    message_data.append({
                        'message_id': msg_id,
                        'file_hash': doc.get('file_hash'),
                        'normalized_title': doc.get('normalized_title')
                    })
            
            if not message_data:
                return
            
            deleted_count = 0
            batch_size = 50
            
            for i in range(0, len(message_data), batch_size):
                batch = message_data[i:i + batch_size]
                message_ids = [item['message_id'] for item in batch]
                
                try:
                    await telegram_rate_limiter.wait_if_needed()
                    messages = await User.get_messages(Config.FILE_CHANNEL_ID, message_ids)
                    
                    existing_ids = set()
                    if isinstance(messages, list):
                        for msg in messages:
                            if msg and hasattr(msg, 'id'):
                                existing_ids.add(msg.id)
                    elif messages is not None and hasattr(messages, 'id'):
                        existing_ids.add(messages.id)
                    
                    for item in batch:
                        if item['message_id'] not in existing_ids:
                            await files_col.delete_one({
                                "channel_id": Config.FILE_CHANNEL_ID,
                                "message_id": item['message_id']
                            })
                            
                            if item.get('file_hash'):
                                await duplicate_prevention.remove_file_hash(
                                    item['file_hash'],
                                    item.get('normalized_title')
                                )
                            
                            deleted_count += 1
                            self.deleted_count += 1
                
                except Exception as e:
                    logger.error(f"❌ Batch check error: {e}")
                    continue
            
            if deleted_count > 0:
                logger.info(f"✅ Sync: {deleted_count} files deleted")
            
        except Exception as e:
            logger.error(f"❌ Sync deletions error: {e}")

channel_sync_manager = ChannelSyncManager()

# ============================================================================
# ✅ FILE INDEXING FUNCTIONS - OPTIMIZED
# ============================================================================

async def generate_file_hash(message):
    try:
        hash_parts = []
        
        if message.document:
            file_attrs = message.document
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
        
        if message.caption:
            caption_hash = hashlib.md5(message.caption.encode()).hexdigest()[:12]
            hash_parts.append(f"cap_{caption_hash}")
        
        final_hash = hashlib.sha256("_".join(hash_parts).encode()).hexdigest()
        return final_hash
        
    except Exception as e:
        logger.debug(f"Hash generation error: {e}")
        return None

async def extract_title_improved(filename, caption):
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

async def index_single_file_smart(message):
    try:
        if files_col is None:
            logger.error("❌ Database not ready for indexing")
            return False
        
        if not message or (not message.document and not message.video):
            logger.debug(f"❌ Not a file message: {message.id}")
            return False
        
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
        
        # Check if already indexed by message ID
        existing_by_id = await files_col.find_one({
            'channel_id': Config.FILE_CHANNEL_ID,
            'message_id': message.id
        }, {'_id': 1})
        
        if existing_by_id:
            logger.debug(f"📝 Already indexed by ID: {title[:50]}... (ID: {message.id})")
            return False
        
        # Generate file hash
        file_hash = await generate_file_hash(message)
        
        if file_hash:
            is_duplicate, reason = await duplicate_prevention.is_duplicate_file(
                file_hash, normalized_title
            )
            
            if is_duplicate:
                logger.info(f"🔄 DUPLICATE SKIP: {title[:50]}... - Reason: {reason}")
                return False
        
        # Extract thumbnail only for video files
        thumbnail_url = None
        is_video = False
        
        if message.video or (message.document and is_video_file(file_name or '')):
            is_video = True
            try:
                thumbnail_url = await thumbnail_extractor.extract_thumbnail(
                    Config.FILE_CHANNEL_ID,
                    message.id
                )
                
                if thumbnail_url:
                    logger.debug(f"✅ Thumbnail extracted for: {title[:50]}...")
            except Exception as e:
                logger.debug(f"⚠️ Thumbnail extraction failed: {e}")
        
        # Extract year from title
        year_match = re.search(r'\b(19|20)\d{2}\b', title)
        year = year_match.group() if year_match else ""
        
        # Detect quality
        quality = detect_quality_enhanced(file_name or "")
        
        # Prepare document
        doc = {
            'channel_id': Config.FILE_CHANNEL_ID,
            'message_id': message.id,
            'real_message_id': message.id,
            'title': title,
            'normalized_title': normalized_title,
            'date': message.date if hasattr(message, 'date') else datetime.now(),
            'indexed_at': datetime.now(),
            'last_checked': datetime.now(),
            'is_video_file': is_video,
            'file_id': None,
            'file_size': 0,
            'file_hash': file_hash,
            'thumbnail_url': thumbnail_url,
            'thumbnail_extracted': thumbnail_url is not None,
            'status': 'active',
            'is_duplicate': False,
            'quality': quality,
            'year': year
        }
        
        # Add file-specific information
        if message.document:
            doc.update({
                'file_name': message.document.file_name or '',
                'is_video_file': is_video_file(message.document.file_name or ''),
                'caption': caption or '',
                'mime_type': message.document.mime_type or '',
                'file_id': message.document.file_id,
                'telegram_file_id': message.document.file_id,
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
                'telegram_file_id': message.video.file_id,
                'file_size': message.video.file_size or 0
            })
        else:
            return False
        
        try:
            # Insert into database
            await files_col.insert_one(doc)
            
            # Update duplicate prevention cache
            if file_hash:
                await duplicate_prevention.add_file_hash(file_hash, normalized_title)
            
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
    """Safe initialization of indexing"""
    if User is None or files_col is None or not user_session_ready:
        logger.warning("⚠️ User session not ready for initial indexing")
        return
    
    logger.info("=" * 60)
    logger.info("🚀 SAFE FILE CHANNEL INDEXING INITIALIZATION")
    logger.info("=" * 60)
    
    try:
        await setup_database_indexes()
        
        # Check if we should do complete indexing
        total_files = await files_col.count_documents({})
        
        if total_files == 0:
            logger.info("📊 Database is empty, starting fresh indexing...")
            await file_indexing_manager.start_indexing()
        elif Config.INDEX_ALL_HISTORY:
            logger.info("📊 Database has files, indexing new files only...")
            # Only index new files
            asyncio.create_task(file_indexing_manager._run_limited_indexing())
        else:
            logger.info("📊 Database populated, skipping complete indexing")
        
        # Start sync monitoring gently
        await channel_sync_manager.start_sync_monitoring()
        
    except Exception as e:
        logger.error(f"❌ Initial indexing error: {e}")

async def setup_database_indexes():
    if files_col is None:
        return
    
    try:
        # Drop old indexes first
        existing_indexes = await files_col.index_information()
        for index_name in list(existing_indexes.keys()):
            if index_name != '_id_':
                try:
                    await files_col.drop_index(index_name)
                except:
                    pass
        
        # Create new indexes
        await files_col.create_index(
            [("channel_id", 1), ("message_id", 1)],
            unique=True,
            name="channel_message_unique",
            background=True
        )
        
        await files_col.create_index(
            [("normalized_title", "text")],
            name="title_text_search",
            background=True
        )
        
        await files_col.create_index(
            [("quality", 1)],
            name="quality_index",
            background=True
        )
        
        await files_col.create_index(
            [("date", -1)],
            name="date_index",
            background=True
        )
        
        await files_col.create_index(
            [("real_message_id", 1)],
            name="real_message_id_index",
            background=True
        )
        
        await files_col.create_index(
            [("file_hash", 1)],
            name="file_hash_index",
            background=True,
            sparse=True
        )
        
        logger.info("✅ Created fresh database indexes")
        
    except Exception as e:
        logger.warning(f"⚠️ Index creation error: {e}")

# ============================================================================
# ✅ POSTER FETCHING FUNCTIONS - OPTIMIZED
# ============================================================================

async def get_poster_for_movie(title: str, year: str = "", quality: str = "") -> Dict[str, Any]:
    global poster_fetcher
    
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
        # Create task with timeout
        poster_task = asyncio.create_task(poster_fetcher.fetch_poster(title))
        
        try:
            # Use wait_for with timeout
            poster_data = await asyncio.wait_for(poster_task, timeout=5.0)  # Increased timeout
            
            if poster_data and poster_data.get('poster_url'):
                logger.debug(f"✅ Poster fetched: {title} - {poster_data['source']}")
                return poster_data
            else:
                raise ValueError("Invalid poster data")
                
        except (asyncio.TimeoutError, ValueError, Exception) as e:
            logger.warning(f"⚠️ Poster fetch timeout/error for {title}: {e}")
            
            # Cancel task if still running
            if not poster_task.done():
                poster_task.cancel()
                try:
                    await poster_task
                except asyncio.CancelledError:
                    pass
            
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
    """Optimized batch poster fetching with concurrency limit"""
    results = []
    
    semaphore = asyncio.Semaphore(Config.MAX_PARALLEL_POSTER_REQUESTS)
    
    async def fetch_poster_with_semaphore(movie):
        async with semaphore:
            title = movie.get('title', '')
            year = movie.get('year', '')
            quality = movie.get('quality', '')
            
            poster_data = await get_poster_for_movie(title, year, quality)
            
            movie_with_poster = movie.copy()
            movie_with_poster.update({
                'poster_url': poster_data['poster_url'],
                'poster_source': poster_data['source'],
                'poster_rating': poster_data['rating'],
                'thumbnail': poster_data['poster_url'],
                'thumbnail_source': poster_data['source'],
                'has_poster': True
            })
            
            return movie_with_poster
    
    # Create tasks
    tasks = []
    for movie in movies:
        task = asyncio.create_task(fetch_poster_with_semaphore(movie))
        tasks.append(task)
    
    # Wait for all tasks
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        final_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"⚠️ Poster batch error: {result}")
            else:
                final_results.append(result)
        
        return final_results
    except Exception as e:
        logger.error(f"❌ Batch poster gathering error: {e}")
        
        # Fallback - return movies with fallback posters
        for movie in movies:
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
# ✅ FILE STREAMING AND DOWNLOAD FUNCTIONS - OPTIMIZED
# ============================================================================

class StreamingManager:
    def __init__(self):
        self.stream_cache = {}
        self.cache_lock = asyncio.Lock()
        self.file_info_cache = {}
        
    async def get_file_stream_info(self, channel_id, message_id):
        cache_key = f"file_info_{channel_id}_{message_id}"
        
        # Check cache first
        now = time.time()
        if cache_key in self.file_info_cache:
            cached_info, timestamp = self.file_info_cache[cache_key]
            if now - timestamp < 300:  # 5 minutes cache
                logger.debug(f"✅ File info from cache: {channel_id}/{message_id}")
                return cached_info
        
        try:
            if bot_handler and bot_handler.initialized:
                await telegram_rate_limiter.wait_if_needed()
                message = await bot_handler.bot.get_messages(channel_id, message_id)
                if not message:
                    return None
                
                file_info = {
                    'channel_id': channel_id,
                    'message_id': message_id,
                    'has_file': False,
                    'file_type': None,
                    'file_size': 0,
                    'file_name': '',
                    'mime_type': '',
                    'duration': 0,
                    'width': 0,
                    'height': 0
                }
                
                if message.document:
                    file_info.update({
                        'has_file': True,
                        'file_type': 'document',
                        'file_size': message.document.file_size or 0,
                        'file_name': message.document.file_name or '',
                        'mime_type': message.document.mime_type or '',
                        'file_id': message.document.file_id,
                        'is_video': is_video_file(message.document.file_name or '')
                    })
                    
                    if message.document.mime_type and 'video' in message.document.mime_type:
                        if hasattr(message.document, 'duration'):
                            file_info['duration'] = message.document.duration
                        if hasattr(message.document, 'width'):
                            file_info['width'] = message.document.width
                        if hasattr(message.document, 'height'):
                            file_info['height'] = message.document.height
                
                elif message.video:
                    file_info.update({
                        'has_file': True,
                        'file_type': 'video',
                        'file_size': message.video.file_size or 0,
                        'file_name': message.video.file_name or 'video.mp4',
                        'mime_type': 'video/mp4',
                        'duration': message.video.duration if hasattr(message.video, 'duration') else 0,
                        'width': message.video.width if hasattr(message.video, 'width') else 0,
                        'height': message.video.height if hasattr(message.video, 'height') else 0,
                        'file_id': message.video.file_id,
                        'is_video': True
                    })
                
                # Cache the result
                async with self.cache_lock:
                    self.file_info_cache[cache_key] = (file_info, now)
                    # Clean old cache entries
                    if len(self.file_info_cache) > 100:
                        oldest_key = min(self.file_info_cache.keys(), 
                                       key=lambda k: self.file_info_cache[k][1])
                        del self.file_info_cache[oldest_key]
                
                return file_info
                
        except Exception as e:
            logger.error(f"❌ Get file stream info error: {e}")
            return None
    
    async def get_direct_download_url(self, file_id):
        try:
            if bot_handler and bot_handler.initialized:
                return await bot_handler.get_file_download_url(file_id)
        except Exception as e:
            logger.error(f"❌ Get direct download URL error: {e}")
        
        return None
    
    async def get_streaming_url(self, channel_id, message_id, quality=None):
        try:
            file_info = await self.get_file_stream_info(channel_id, message_id)
            if not file_info or not file_info['has_file']:
                return None
            
            if not file_info.get('is_video', False):
                return None
            
            direct_url = await self.get_direct_download_url(file_info['file_id'])
            if direct_url:
                return {
                    'stream_url': direct_url,
                    'direct_url': direct_url,
                    'file_name': file_info['file_name'],
                    'file_size': file_info['file_size'],
                    'duration': file_info.get('duration', 0),
                    'quality': quality or 'Unknown',
                    'mime_type': file_info['mime_type'],
                    'is_streamable': True
                }
            
        except Exception as e:
            logger.error(f"❌ Get streaming URL error: {e}")
        
        return None
    
    async def get_file_metadata(self, channel_id, message_id):
        cache_key = f"metadata_{channel_id}_{message_id}"
        
        # Check cache first
        if cache_manager is not None and cache_manager.redis_enabled:
            cached_metadata = await cache_manager.get(cache_key)
            if cached_metadata:
                logger.debug(f"✅ Metadata from cache: {channel_id}/{message_id}")
                return cached_metadata
        
        try:
            if files_col is not None:
                doc = await files_col.find_one({
                    'channel_id': channel_id,
                    'message_id': int(message_id)
                }, {
                    'title': 1,
                    'file_name': 1,
                    'file_size': 1,
                    'quality': 1,
                    'thumbnail_url': 1,
                    'caption': 1,
                    'year': 1,
                    '_id': 0
                })
                
                if doc:
                    # Cache the result
                    if cache_manager is not None:
                        await cache_manager.set(cache_key, doc, expire_seconds=600)
                    return doc
            
            if bot_handler and bot_handler.initialized:
                await telegram_rate_limiter.wait_if_needed()
                message = await bot_handler.bot.get_messages(channel_id, int(message_id))
                if message and (message.document or message.video):
                    file_name = ''
                    file_size = 0
                    
                    if message.document:
                        file_name = message.document.file_name
                        file_size = message.document.file_size or 0
                    elif message.video:
                        file_name = message.video.file_name
                        file_size = message.video.file_size or 0
                    
                    metadata = {
                        'title': file_name,
                        'file_name': file_name,
                        'file_size': file_size,
                        'quality': detect_quality_enhanced(file_name),
                        'caption': message.caption or '',
                        'year': ''
                    }
                    
                    # Cache the result
                    if cache_manager is not None:
                        await cache_manager.set(cache_key, metadata, expire_seconds=300)
                    
                    return metadata
        
        except Exception as e:
            logger.error(f"❌ Get file metadata error: {e}")
        
        return None

streaming_manager = StreamingManager()

# ============================================================================
# ✅ DUAL SESSION INITIALIZATION - OPTIMIZED
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
    
    # Initialize User session
    user_session_success = False
    if Config.API_ID > 0 and Config.API_HASH and Config.USER_SESSION_STRING:
        logger.info("\n👤 Initializing USER Session...")
        try:
            User = Client(
                "sk4film_user",
                api_id=Config.API_ID,
                api_hash=Config.API_HASH,
                session_string=Config.USER_SESSION_STRING,
                sleep_threshold=120,  # Increased for stability
                in_memory=True,
                no_updates=True
            )
            
            await User.start()
            me = await User.get_me()
            logger.info(f"✅ USER Session Ready: {me.first_name}")
            
            # Test channel access with timeout
            try:
                await asyncio.wait_for(
                    User.get_chat(Config.FILE_CHANNEL_ID),
                    timeout=10
                )
                user_session_success = True
                logger.info(f"✅ File Channel Access: {Config.FILE_CHANNEL_ID}")
            except asyncio.TimeoutError:
                logger.warning("⚠️ File channel access timeout, may have limited functionality")
                user_session_success = True  # Still mark as success for basic operations
            except Exception as e:
                logger.error(f"❌ File channel access failed: {e}")
                user_session_success = False
                
        except Exception as e:
            logger.error(f"❌ USER Session failed: {e}")
            user_session_success = False
            if User is not None:
                try:
                    await User.stop()
                except:
                    pass
            User = None
    else:
        logger.warning("⚠️ User session credentials not configured")
    
    # Initialize Bot session
    bot_session_success = False
    if Config.BOT_TOKEN:
        logger.info("\n🤖 Initializing BOT Session...")
        try:
            Bot = Client(
                "sk4film_bot",
                api_id=Config.API_ID,
                api_hash=Config.API_HASH,
                bot_token=Config.BOT_TOKEN,
                sleep_threshold=120,  # Increased
                in_memory=True,
                no_updates=True
            )
            
            await Bot.start()
            bot_info = await Bot.get_me()
            logger.info(f"✅ BOT Session Ready: @{bot_info.username}")
            bot_session_success = True
                
        except Exception as e:
            if "FLOOD_WAIT" in str(e):
                logger.warning(f"⚠️ Bot session delayed due to flood wait, will retry later")
                # Don't mark as failed, will retry later
            else:
                logger.error(f"❌ BOT Session failed: {e}")
                bot_session_success = False
                if Bot is not None:
                    try:
                        await Bot.stop()
                    except:
                        pass
                Bot = None
    
    user_session_ready = user_session_success
    bot_session_ready = bot_session_success
    
    logger.info("\n" + "=" * 50)
    logger.info("📊 SESSION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"USER Session: {'✅ READY' if user_session_ready else '❌ NOT READY'}")
    logger.info(f"BOT Session: {'✅ READY' if bot_session_ready else '⚠️ LIMITED'}")
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
            serverSelectionTimeoutMS=20000,  # Increased
            connectTimeoutMS=20000,
            socketTimeoutMS=30000,
            maxPoolSize=20,
            minPoolSize=5,
            retryWrites=True,
            retryReads=True,
            maxIdleTimeMS=30000
        )
        
        await asyncio.wait_for(mongo_client.admin.command('ping'), timeout=15)
        
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
# ✅ SHUTDOWN HANDLER
# ============================================================================

async def shutdown_system():
    """Gracefully shutdown all components"""
    logger.info("🛑 Shutting down SK4FiLM system...")
    
    # Stop indexing
    if file_indexing_manager:
        await file_indexing_manager.stop_indexing()
    
    # Stop sync monitoring
    if channel_sync_manager:
        await channel_sync_manager.stop_sync_monitoring()
    
    # Stop Telegram sessions
    if User:
        try:
            await User.stop()
            logger.info("✅ User session stopped")
        except Exception as e:
            logger.error(f"❌ Error stopping user session: {e}")
    
    if Bot:
        try:
            await Bot.stop()
            logger.info("✅ Bot session stopped")
        except Exception as e:
            logger.error(f"❌ Error stopping bot session: {e}")
    
    # Stop bot handler
    if bot_handler:
        await bot_handler.shutdown()
    
    # Stop Telegram bot
    if telegram_bot and hasattr(telegram_bot, 'shutdown'):
        await telegram_bot.shutdown()
    
    # Stop cache manager
    if cache_manager:
        await cache_manager.stop()
    
    # Close MongoDB connection
    if mongo_client:
        mongo_client.close()
        logger.info("✅ MongoDB connection closed")
    
    logger.info("✅ System shutdown complete")

# ============================================================================
# ✅ MAIN INITIALIZATION - OPTIMIZED
# ============================================================================

@performance_monitor.measure("system_init")
async def init_system():
    start_time = time.time()
    
    try:
        logger.info("=" * 60)
        logger.info("🚀 SK4FiLM v9.2 - PERFORMANCE OPTIMIZED")
        logger.info("=" * 60)
        
        # Initialize MongoDB first
        mongo_ok = await init_mongodb()
        if not mongo_ok:
            logger.error("❌ MongoDB connection failed")
            return False
        
        if files_col is not None:
            file_count = await files_col.count_documents({})
            logger.info(f"📊 Current files in database: {file_count}")
        
        # Initialize Bot Handler (but don't fail if it doesn't work)
        bot_handler_ok = await bot_handler.initialize()
        if bot_handler_ok:
            logger.info("✅ Bot Handler initialized")
        else:
            logger.warning("⚠️ Bot Handler failed to initialize, some features disabled")
        
        # Start Telegram Bot (optional)
        global telegram_bot
        telegram_bot = await start_telegram_bot()
        if telegram_bot:
            logger.info("✅ Telegram Bot started successfully")
        else:
            logger.warning("⚠️ Telegram Bot failed to start, bot features disabled")
        
        # Initialize other components
        global cache_manager, verification_system, premium_system, poster_fetcher
        cache_manager = CacheManager(Config)
        redis_ok = await cache_manager.init_redis()
        if redis_ok:
            logger.info("✅ Cache Manager initialized")
            await cache_manager.start_cleanup_task()
        
        if VerificationSystem is not None:
            verification_system = VerificationSystem(Config, mongo_client)
            logger.info("✅ Verification System initialized")
        
        if PremiumSystem is not None:
            premium_system = PremiumSystem(Config, mongo_client)
            logger.info("✅ Premium System initialized")
        
        if PosterFetcher is not None:
            poster_fetcher = PosterFetcher(Config, cache_manager)
            logger.info("✅ Poster Fetcher initialized")
        
        # Initialize Telegram sessions with retry
        if PYROGRAM_AVAILABLE:
            max_retries = 2  # Reduced retries
            for retry in range(max_retries):
                telegram_ok = await init_telegram_sessions()
                if telegram_ok:
                    break
                elif retry < max_retries - 1:
                    wait_time = (retry + 1) * 5  # Exponential backoff
                    logger.warning(f"⚠️ Telegram session failed, retrying in {wait_time}s ({retry + 1}/{max_retries})...")
                    await asyncio.sleep(wait_time)
            if not telegram_ok:
                logger.warning("⚠️ Telegram sessions failed after retries")
        
        # Start indexing only if we have user session
        if user_session_ready and files_col is not None:
            logger.info("🔄 Starting safe file channel indexing...")
            asyncio.create_task(initial_indexing())
        else:
            logger.warning("⚠️ Skipping indexing - user session not ready")
        
        init_time = time.time() - start_time
        logger.info(f"⚡ SK4FiLM Started in {init_time:.2f}s")
        logger.info("=" * 60)
        
        logger.info("🔧 OPTIMIZED FEATURES:")
        logger.info(f"   • Safe Indexing: ✅ ENABLED (New files only)")
        logger.info(f"   • Rate Limiting: ✅ ENABLED ({Config.TELEGRAM_API_RATE_LIMIT} calls/sec)")
        logger.info(f"   • Aggressive Caching: ✅ {'ENABLED' if Config.ENABLE_AGGRESSIVE_CACHING else 'DISABLED'}")
        logger.info(f"   • Real Message IDs: ✅ ENABLED")
        logger.info(f"   • Quality Merging: ✅ ENABLED")
        logger.info(f"   • User Session: {'✅ READY' if user_session_ready else '❌ NOT READY'}")
        logger.info(f"   • Bot Session: {'✅ READY' if bot_session_ready else '⚠️ LIMITED'}")
        logger.info(f"   • Telegram Bot: {'✅ RUNNING' if telegram_bot else '❌ NOT RUNNING'}")
        logger.info(f"   • Video Streaming: {'✅ ENABLED' if Config.STREAMING_ENABLED else '❌ DISABLED'}")
        logger.info(f"   • Emergency Stop: ✅ AVAILABLE")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ System initialization failed: {e}")
        return False

# ============================================================================
# ✅ OPTIMIZED SEARCH FUNCTION
# ============================================================================

def channel_name_cached(cid):
    return f"Channel {cid}"

@performance_monitor.measure("multi_channel_search_merged")
@async_cache_with_ttl(maxsize=1000, ttl=Config.SEARCH_CACHE_TTL)
async def search_movies_multi_channel_merged(query, limit=15, page=1):
    """OPTIMIZED: Returns results with real_message_id for view.html"""
    offset = (page - 1) * limit
    
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
    
    # Limit concurrent operations
    if user_session_ready and User is not None:
        async def search_text_channel(channel_id):
            channel_posts = {}
            try:
                cname = channel_name_cached(channel_id)
                await telegram_rate_limiter.wait_if_needed()
                async for msg in User.search_messages(channel_id, query=query, limit=15):
                    if msg is not None and msg.text and len(msg.text) > 15:
                        title = extract_title_smart(msg.text)
                        if title and (query_lower in title.lower() or query_lower in msg.text.lower()):
                            norm_title = normalize_title(title)
                            if norm_title not in channel_posts:
                                year_match = re.search(r'\b(19|20)\d{2}\b', title)
                                year = year_match.group() if year_match else ""
                                
                                movie_data = {
                                    'title': title,
                                    'original_title': title,
                                    'normalized_title': norm_title,
                                    'content': format_post(msg.text, max_length=1000),
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
                                    'search_score': 2 if query_lower in title.lower() else 1,
                                    'result_type': 'post'
                                }
                                
                                channel_posts[norm_title] = movie_data
            except Exception as e:
                logger.error(f"Text search error in {channel_id}: {e}")
            return channel_posts
        
        # Search text channels in parallel but limited
        tasks = []
        for channel_id in Config.TEXT_CHANNEL_IDS[:2]:  # Limit to 2 channels
            tasks.append(search_text_channel(channel_id))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, dict):
                posts_dict.update(result)
        
        logger.info(f"📝 Found {len(posts_dict)} posts in text channels")
    
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
                "status": "active",
                "is_duplicate": False
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
                    'file_id': 1,
                    'telegram_file_id': 1,
                    'thumbnail_url': 1,
                    'thumbnail_extracted': 1,
                    'year': 1,
                    '_id': 0
                }
            ).limit(500)  # Reduced limit for performance
            
            file_count = 0
            quality_counts = defaultdict(int)
            
            async for doc in cursor:
                file_count += 1
                try:
                    title = doc.get('title', 'Unknown')
                    norm_title = normalize_title(title)
                    
                    quality_info = extract_quality_info(doc.get('file_name', ''))
                    quality = quality_info['full']
                    base_quality = quality_info['base']
                    
                    quality_counts[quality] += 1
                    
                    thumbnail_url = doc.get('thumbnail_url')
                    
                    real_msg_id = doc.get('real_message_id') or doc.get('message_id')
                    
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
                            'channel_name': channel_name_cached(doc.get('channel_id')),
                            'has_file': True,
                            'has_post': bool(doc.get('caption')),
                            'file_caption': doc.get('caption', ''),
                            'year': year,
                            'quality': quality,
                            'has_thumbnail': thumbnail_url is not None,
                            'thumbnail_url': thumbnail_url,
                            'real_message_id': real_msg_id,  # ✅ CRITICAL FOR VIEW.HTML
                            'search_score': 3 if query_lower in title.lower() else 2,
                            'result_type': 'file',
                            'total_files': 1,
                            'file_sizes': [doc.get('file_size', 0)]
                        }
                        
                        if thumbnail_url:
                            files_dict[norm_title]['thumbnail'] = thumbnail_url
                    else:
                        existing = files_dict[norm_title]
                        
                        if quality not in existing['quality_options']:
                            existing['quality_options'][quality] = quality_option
                            existing['quality_list'].append(quality)
                            existing['total_files'] += 1
                            existing['file_sizes'].append(doc.get('file_size', 0))
                        
                        if thumbnail_url and not existing.get('has_thumbnail'):
                            existing['thumbnail'] = thumbnail_url
                            existing['thumbnail_url'] = thumbnail_url
                            existing['has_thumbnail'] = True
                        
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
            
            if quality_counts:
                logger.info(f"📊 Quality distribution: {dict(quality_counts)}")
            
            grouped_count = len(files_dict)
            logger.info(f"📦 After merging: {grouped_count} unique titles (from {file_count} files)")
            
        except Exception as e:
            logger.error(f"❌ File search error: {e}")
    
    for norm_title, movie_data in files_dict.items():
        if movie_data['quality_options']:
            qualities = list(movie_data['quality_options'].keys())
            
            def get_quality_priority(q):
                base_q = q.replace(' HEVC', '')
                if base_q in Config.QUALITY_PRIORITY:
                    return Config.QUALITY_PRIORITY.index(base_q)
                return 999
            
            qualities.sort(key=get_quality_priority)
            
            total_size = sum(movie_data['file_sizes'])
            
            best_quality = qualities[0] if qualities else ''
            
            quality_summary_parts = []
            for q in qualities[:5]:
                quality_summary_parts.append(q)
            
            if len(qualities) > 5:
                quality_summary_parts.append(f"+{len(qualities) - 5} more")
            
            quality_summary = " • ".join(quality_summary_parts)
            
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
    
    merged = {}
    
    for norm_title, file_data in files_dict.items():
        merged[norm_title] = file_data.copy()
        
        if norm_title in posts_dict:
            post_data = posts_dict[norm_title]
            
            merged[norm_title].update({
                'has_post': True,
                'post_content': post_data.get('post_content', ''),
                'content': post_data.get('content', '') or merged[norm_title].get('content', ''),
                'search_score': max(merged[norm_title].get('search_score', 0), post_data.get('search_score', 0)),
                'result_type': 'both'
            })
            
            del posts_dict[norm_title]
    
    for norm_title, post_data in posts_dict.items():
        merged[norm_title] = post_data.copy()
    
    # Fetch posters only if we have results
    if merged:
        movies_for_posters = []
        for norm_title, movie_data in merged.items():
            movies_for_posters.append(movie_data)
        
        # Only fetch posters for movies without thumbnails
        movies_without_thumbnails = [m for m in movies_for_posters if not m.get('has_thumbnail')]
        
        if movies_without_thumbnails:
            logger.info(f"🎬 Fetching posters for {len(movies_without_thumbnails)} movies...")
            movies_with_posters = await get_posters_for_movies_batch(movies_without_thumbnails)
            
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
    
    results_list = list(merged.values())
    
    # Optimized sorting
    results_list.sort(key=lambda x: (
        x.get('has_file', False),
        x.get('quality_count', 0),
        x.get('has_thumbnail', False),
        x.get('is_new', False),
        x.get('search_score', 0)
    ), reverse=True)
    
    total = len(results_list)
    paginated = results_list[offset:offset + limit]
    
    stats = {
        'total': total,
        'with_files': sum(1 for r in results_list if r.get('has_file', False)),
        'with_posts': sum(1 for r in results_list if r.get('has_post', False)),
        'both': sum(1 for r in results_list if r.get('has_file', False) and r.get('has_post', False)),
        'video_files': sum(1 for r in results_list if r.get('is_video_file', False)),
        'with_thumbnails': sum(1 for r in results_list if r.get('has_thumbnail', False)),
        'multi_quality': sum(1 for r in results_list if r.get('quality_count', 0) > 1),
        'avg_qualities_per_title': sum(r.get('quality_count', 0) for r in results_list) / total if total > 0 else 0,
        'real_message_ids': sum(1 for r in results_list if r.get('real_message_id'))
    }
    
    logger.info(f"📊 FINAL MERGING STATS:")
    logger.info(f"   • Total unique titles: {total}")
    logger.info(f"   • Titles with files: {stats['with_files']}")
    
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
            'quality_merging': True,
            'duplicate_prevention': True,
            'poster_fetcher': poster_fetcher is not None,
            'user_session_used': user_session_ready,
            'cache_hit': False,
            'real_message_ids': True,
            'merge_stats': {
                'files_merged': file_count - grouped_count if 'file_count' in locals() else 0,
                'unique_titles': grouped_count if 'grouped_count' in locals() else 0
            }
        },
        'bot_username': Config.BOT_USERNAME
    }
    
    if cache_manager is not None:
        await cache_manager.set(cache_key, result_data, expire_seconds=Config.SEARCH_CACHE_TTL)
    
    logger.info(f"✅ Search complete: {len(paginated)} results (showing page {page})")
    
    return result_data

# ============================================================================
# ✅ OPTIMIZED HOME MOVIES
# ============================================================================

@performance_monitor.measure("home_movies")
@async_cache_with_ttl(maxsize=1, ttl=Config.CACHE_HOME_MOVIES_FOR)
async def get_home_movies(limit=25):
    """Optimized home movies with database fallback"""
    try:
        movies = []
        
        # Try to get from database first (faster)
        if files_col is not None:
            cursor = files_col.find(
                {'status': 'active', 'is_duplicate': False},
                {
                    'title': 1,
                    'file_name': 1,
                    'quality': 1,
                    'file_size': 1,
                    'thumbnail_url': 1,
                    'caption': 1,
                    'year': 1,
                    'date': 1,
                    'is_video_file': 1,
                    'channel_id': 1,
                    'message_id': 1,
                    'real_message_id': 1
                }
            ).sort('date', -1).limit(limit)
            
            async for doc in cursor:
                title = doc.get('title') or doc.get('file_name', 'Unknown')
                year = doc.get('year', '')
                quality = doc.get('quality', 'Unknown')
                thumbnail_url = doc.get('thumbnail_url')
                
                movie_data = {
                    'title': title,
                    'original_title': title,
                    'year': year,
                    'date': doc['date'].isoformat() if isinstance(doc['date'], datetime) else str(doc['date']),
                    'is_new': is_new(doc['date']) if doc.get('date') else False,
                    'channel_id': doc.get('channel_id', Config.FILE_CHANNEL_ID),
                    'channel_name': channel_name_cached(doc.get('channel_id', Config.FILE_CHANNEL_ID)),
                    'message_id': doc.get('message_id'),
                    'real_message_id': doc.get('real_message_id') or doc.get('message_id'),
                    'has_file': True,
                    'has_post': bool(doc.get('caption')),
                    'content': format_post(doc.get('caption', ''), max_length=500),
                    'post_content': doc.get('caption', ''),
                    'quality_options': {quality: {
                        'quality': quality,
                        'file_size': doc.get('file_size', 0)
                    }},
                    'is_video_file': doc.get('is_video_file', False),
                    'quality': quality,
                    'thumbnail_url': thumbnail_url,
                    'has_thumbnail': thumbnail_url is not None,
                    'result_type': 'file'
                }
                
                if thumbnail_url:
                    movie_data['thumbnail'] = thumbnail_url
                
                movies.append(movie_data)
            
            if movies:
                logger.info(f"✅ Got {len(movies)} movies from database")
                # Add posters
                movies_with_posters = await get_posters_for_movies_batch(movies)
                return movies_with_posters[:limit]
        
        # Fallback to Telegram if database doesn't have enough
        if User is not None and user_session_ready and len(movies) < limit:
            logger.info("📡 Falling back to Telegram for home movies...")
            seen_titles = set(m['title'] for m in movies)
            
            try:
                await telegram_rate_limiter.wait_if_needed()
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
                                'channel': channel_name_cached(Config.MAIN_CHANNEL_ID),
                                'channel_id': Config.MAIN_CHANNEL_ID,
                                'message_id': msg.id,
                                'has_file': False,
                                'has_post': True,
                                'content': formatted_content,
                                'post_content': post_content,
                                'quality_options': {},
                                'is_video_file': False
                            }
                            
                            movies.append(movie_data)
                            
                            if len(movies) >= limit:
                                break
            except Exception as e:
                logger.error(f"❌ Error fetching home movies from Telegram: {e}")
        
        if movies:
            movies_with_posters = await get_posters_for_movies_batch(movies)
            logger.info(f"✅ Fetched {len(movies_with_posters)} home movies")
            return movies_with_posters[:limit]
        else:
            logger.warning("⚠️ No movies found for home page")
            return []
        
    except Exception as e:
        logger.error(f"❌ Home movies error: {e}")
        return []

# ============================================================================
# ✅ API ROUTES - FIXED AND OPTIMIZED
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
    
    indexing_status = await file_indexing_manager.get_indexing_status()
    
    bot_status = None
    if bot_handler:
        try:
            bot_status = await bot_handler.get_bot_status()
        except Exception as e:
            logger.error(f"❌ Error getting bot status: {e}")
            bot_status = {'initialized': False, 'error': str(e)}
    
    bot_running = telegram_bot is not None and hasattr(telegram_bot, 'bot_started') and telegram_bot.bot_started
    
    return jsonify({
        'status': 'healthy',
        'service': 'SK4FiLM v9.2 - PERFORMANCE OPTIMIZED',
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
        'features': {
            'safe_indexing': True,
            'rate_limiting': True,
            'real_message_ids': True,
            'file_channel_indexing': True,
            'complete_history': Config.INDEX_ALL_HISTORY,
            'duplicate_prevention': True,
            'quality_merging': True,
            'thumbnail_extraction': True,
            'telegram_bot': True,
            'video_streaming': Config.STREAMING_ENABLED,
            'direct_download': True,
            'emergency_stop': True,
            'aggressive_caching': Config.ENABLE_AGGRESSIVE_CACHING
        },
        'stats': {
            'total_files': tf,
            'video_files': video_files,
            'thumbnails_extracted': thumbnails_extracted
        },
        'indexing': indexing_status,
        'performance': performance_monitor.get_stats(),
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
        'sessions': {
            'user': user_session_ready,
            'bot': bot_session_ready,
            'bot_handler': bot_status.get('initialized') if bot_status else False,
            'telegram_bot': telegram_bot is not None and hasattr(telegram_bot, 'bot_started') and telegram_bot.bot_started
        },
        'indexing': {
            'running': indexing_status['is_running'],
            'is_first_run': indexing_status.get('is_first_run', False),
            'emergency_stop': indexing_status.get('emergency_stop', False),
            'last_run': indexing_status['last_run']
        },
        'timestamp': datetime.now().isoformat()
    })

# ============================================================================
# ✅ FIXED: MISSING ENDPOINTS
# ============================================================================

@app.route('/api/file/metadata', methods=['GET'])
@performance_monitor.measure("file_metadata_endpoint")
async def api_file_metadata():
    """Alternative name for file details endpoint"""
    return await api_file_details()

@app.route('/download/<int:channel_id>/<int:message_id>/<quality>', methods=['GET'])
@performance_monitor.measure("direct_download_endpoint")
async def direct_download(channel_id, message_id, quality):
    """Direct download endpoint (redirects to Telegram)"""
    try:
        # Get file info
        if bot_handler and bot_handler.initialized:
            await telegram_rate_limiter.wait_if_needed()
            message = await bot_handler.bot.get_messages(channel_id, message_id)
            
            if message and (message.document or message.video):
                # Get direct download URL
                if message.document:
                    file_id = message.document.file_id
                elif message.video:
                    file_id = message.video.file_id
                else:
                    return jsonify({'error': 'File not found'}), 404
                
                download_url = await streaming_manager.get_direct_download_url(file_id)
                
                if download_url:
                    # Redirect to direct download URL
                    return redirect(download_url, code=302)
        
        # Fallback: Return file info
        file_info = await streaming_manager.get_file_stream_info(channel_id, message_id)
        
        if file_info and file_info['has_file']:
            return jsonify({
                'status': 'success',
                'file_info': file_info,
                'quality': quality,
                'message': 'Use the download_url from /api/download/url endpoint'
            })
        
        return jsonify({'error': 'File not found'}), 404
        
    except Exception as e:
        logger.error(f"Direct download error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/stream/<int:channel_id>/<int:message_id>/<quality>', methods=['GET'])
@performance_monitor.measure("direct_stream_endpoint")
async def direct_stream(channel_id, message_id, quality):
    """Direct streaming endpoint"""
    try:
        stream_info = await streaming_manager.get_streaming_url(channel_id, message_id, quality)
        
        if stream_info:
            # Redirect to streaming URL
            return redirect(stream_info['stream_url'], code=302)
        
        # Fallback to download URL
        if bot_handler and bot_handler.initialized:
            await telegram_rate_limiter.wait_if_needed()
            message = await bot_handler.bot.get_messages(channel_id, message_id)
            
            if message and (message.document or message.video):
                if message.document:
                    file_id = message.document.file_id
                elif message.video:
                    file_id = message.video.file_id
                else:
                    return jsonify({'error': 'File not found'}), 404
                
                download_url = await streaming_manager.get_direct_download_url(file_id)
                
                if download_url:
                    # For streaming, we might want to add headers for video playback
                    return redirect(download_url, code=302)
        
        return jsonify({
            'error': 'Stream not available',
            'suggestion': 'Use /api/download/url for direct download'
        }), 404
        
    except Exception as e:
        logger.error(f"Direct stream error: {e}")
        return jsonify({'error': str(e)}), 500

# ============================================================================
# ✅ EMERGENCY ENDPOINTS
# ============================================================================

@app.route('/emergency/stop_indexing', methods=['POST'])
async def emergency_stop_indexing():
    """Emergency stop indexing - use this if indexing is flooding"""
    try:
        await file_indexing_manager.emergency_stop_indexing()
        logger.critical("🛑 EMERGENCY STOP: Indexing halted immediately!")
        return jsonify({
            'status': 'success',
            'message': 'Indexing stopped immediately',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"❌ Emergency stop error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/emergency/cleanup_database', methods=['POST'])
async def emergency_cleanup_database():
    """Emergency cleanup of database duplicates"""
    try:
        if files_col is None:
            return jsonify({
                'status': 'error',
                'message': 'Database not available'
            }), 500
        
        # Find and remove exact duplicates by message_id
        pipeline = [
            {
                "$group": {
                    "_id": {"channel_id": "$channel_id", "message_id": "$message_id"},
                    "count": {"$sum": 1},
                    "ids": {"$push": "$_id"}
                }
            },
            {
                "$match": {
                    "count": {"$gt": 1}
                }
            }
        ]
        
        duplicates_found = 0
        async for group in files_col.aggregate(pipeline):
            ids = group["ids"]
            # Keep the first one, delete the rest
            keep_id = ids[0]
            delete_ids = ids[1:]
            
            await files_col.delete_many({"_id": {"$in": delete_ids}})
            duplicates_found += len(delete_ids)
        
        return jsonify({
            'status': 'success',
            'message': f'Removed {duplicates_found} duplicate entries',
            'duplicates_removed': duplicates_found,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Emergency cleanup error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# ============================================================================
# ✅ STANDARD API ENDPOINTS - OPTIMIZED
# ============================================================================

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
            'source': 'database',
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
                'real_message_ids': True,
                'streaming_enabled': Config.STREAMING_ENABLED
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

# ============================================================================
# ✅ STREAMING AND DOWNLOAD ENDPOINTS - OPTIMIZED
# ============================================================================

@app.route('/api/stream/info', methods=['GET'])
@performance_monitor.measure("stream_info_endpoint")
async def api_stream_info():
    """Get streaming information for a file"""
    try:
        channel_id = request.args.get('channel_id', type=int)
        message_id = request.args.get('message_id', type=int)
        quality = request.args.get('quality', '1080p')
        
        if not channel_id or not message_id:
            return jsonify({
                'status': 'error',
                'message': 'channel_id and message_id are required'
            }), 400
        
        # Get streaming info
        stream_info = await streaming_manager.get_streaming_url(channel_id, message_id, quality)
        
        if not stream_info:
            # Fallback
            stream_info = {
                'stream_url': f"{Config.BACKEND_URL}/stream/{channel_id}/{message_id}/{quality}",
                'direct_url': f"{Config.BACKEND_URL}/stream/{channel_id}/{message_id}/{quality}",
                'file_name': 'movie.mp4',
                'file_size': 0,
                'duration': 0,
                'quality': quality,
                'mime_type': 'video/mp4',
                'is_streamable': True
            }
        
        # Get file metadata
        metadata = await streaming_manager.get_file_metadata(channel_id, message_id)
        
        return jsonify({
            'status': 'success',
            'stream_info': stream_info,
            'metadata': metadata,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Stream info error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/download/url', methods=['GET'])
@performance_monitor.measure("download_url_endpoint")
async def api_download_url():
    """Get direct download URL"""
    try:
        channel_id = request.args.get('channel_id', type=int)
        message_id = request.args.get('message_id', type=int)
        quality = request.args.get('quality', '1080p')
        
        if not channel_id or not message_id:
            return jsonify({
                'status': 'error',
                'message': 'channel_id and message_id are required'
            }), 400
        
        # Get file info
        file_info = await streaming_manager.get_file_stream_info(channel_id, message_id)
        
        if not file_info or not file_info['has_file']:
            return jsonify({
                'status': 'error',
                'message': 'File not found'
            }), 404
        
        # Get direct download URL
        download_url = await streaming_manager.get_direct_download_url(file_info['file_id'])
        
        if not download_url:
            # Fallback
            download_url = f"{Config.BACKEND_URL}/download/{channel_id}/{message_id}/{quality}"
        
        return jsonify({
            'status': 'success',
            'download_url': download_url,
            'file_info': {
                'file_name': file_info['file_name'],
                'file_size': file_info['file_size'],
                'mime_type': file_info['mime_type'],
                'quality': quality or detect_quality_enhanced(file_info['file_name'])
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Download URL error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/file/details', methods=['GET'])
@performance_monitor.measure("file_details_endpoint")
async def api_file_details():
    """Get complete file details for view.html"""
    try:
        channel_id = request.args.get('channel_id', type=int)
        message_id = request.args.get('message_id', type=int)
        
        if not channel_id or not message_id:
            return jsonify({
                'status': 'error',
                'message': 'channel_id and message_id are required'
            }), 400
        
        # Get file metadata from database
        file_data = None
        if files_col is not None:
            file_data = await files_col.find_one({
                'channel_id': channel_id,
                'message_id': int(message_id)
            }, {
                '_id': 0,
                'title': 1,
                'file_name': 1,
                'file_size': 1,
                'quality': 1,
                'thumbnail_url': 1,
                'caption': 1,
                'year': 1,
                'date': 1,
                'duration': 1,
                'is_video_file': 1,
                'mime_type': 1,
                'real_message_id': 1
            })
        
        # If not in database, try to get from Telegram
        if not file_data and bot_handler and bot_handler.initialized:
            await telegram_rate_limiter.wait_if_needed()
            message = await bot_handler.bot.get_messages(channel_id, message_id)
            if message and (message.document or message.video):
                file_data = {
                    'title': '',
                    'file_name': '',
                    'file_size': 0,
                    'quality': 'Unknown',
                    'caption': message.caption or '',
                    'year': '',
                    'is_video_file': False,
                    'mime_type': '',
                    'real_message_id': message_id
                }
                
                if message.document:
                    file_data.update({
                        'file_name': message.document.file_name or '',
                        'file_size': message.document.file_size or 0,
                        'mime_type': message.document.mime_type or '',
                        'is_video_file': is_video_file(message.document.file_name or '')
                    })
                elif message.video:
                    file_data.update({
                        'file_name': message.video.file_name or 'video.mp4',
                        'file_size': message.video.file_size or 0,
                        'mime_type': 'video/mp4',
                        'is_video_file': True,
                        'duration': message.video.duration if hasattr(message.video, 'duration') else 0
                    })
                
                file_data['quality'] = detect_quality_enhanced(file_data['file_name'])
        
        if not file_data:
            return jsonify({
                'status': 'error',
                'message': 'File not found'
            }), 404
        
        # Get poster
        title = file_data.get('title') or file_data.get('file_name', '')
        poster_data = await get_poster_for_movie(title, file_data.get('year', ''), file_data.get('quality', ''))
        
        # Get streaming info
        stream_info = await streaming_manager.get_streaming_url(channel_id, message_id, file_data.get('quality', ''))
        
        # Get download URL
        download_info = None
        if bot_handler and bot_handler.initialized:
            await telegram_rate_limiter.wait_if_needed()
            message = await bot_handler.bot.get_messages(channel_id, message_id)
            if message:
                if message.document:
                    file_id = message.document.file_id
                elif message.video:
                    file_id = message.video.file_id
                else:
                    file_id = None
                
                if file_id:
                    download_url = await streaming_manager.get_direct_download_url(file_id)
                    if download_url:
                        download_info = {
                            'direct_url': download_url,
                            'telegram_url': f"https://t.me/{Config.CHANNEL_USERNAME}/{message_id}",
                            'bot_url': f"https://t.me/{Config.BOT_USERNAME}?start=file_{channel_id}_{message_id}"
                        }
        
        response_data = {
            'status': 'success',
            'file': file_data,
            'poster': poster_data,
            'streaming': stream_info,
            'download': download_info,
            'telegram': {
                'channel_username': Config.CHANNEL_USERNAME,
                'bot_username': Config.BOT_USERNAME,
                'message_id': message_id
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"File details error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# ============================================================================
# ✅ ADMIN ENDPOINTS
# ============================================================================

@app.route('/admin/indexing/status', methods=['GET'])
async def admin_indexing_status():
    """Get indexing status"""
    try:
        status = await file_indexing_manager.get_indexing_status()
        return jsonify({
            'status': 'success',
            'indexing': status
        })
    except Exception as e:
        logger.error(f"Indexing status error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/admin/indexing/start', methods=['POST'])
async def admin_start_indexing():
    """Start indexing"""
    try:
        await file_indexing_manager.start_indexing()
        return jsonify({
            'status': 'success',
            'message': 'Indexing started'
        })
    except Exception as e:
        logger.error(f"Start indexing error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/admin/indexing/stop', methods=['POST'])
async def admin_stop_indexing():
    """Stop indexing"""
    try:
        await file_indexing_manager.stop_indexing()
        return jsonify({
            'status': 'success',
            'message': 'Indexing stopped'
        })
    except Exception as e:
        logger.error(f"Stop indexing error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# ============================================================================
# ✅ APPLICATION STARTUP AND SHUTDOWN HANDLERS
# ============================================================================

@app.before_serving
async def startup():
    """Initialize system on startup"""
    await init_system()

@app.after_serving  
async def cleanup():
    """Cleanup on shutdown"""
    await shutdown_system()

# ============================================================================
# ✅ MAIN APPLICATION RUNNER
# ============================================================================

def main():
    """Main application runner"""
    global start_time
    start_time = time.time()
    
    try:
        config = HyperConfig()
        config.bind = [f"0.0.0.0:{Config.WEB_SERVER_PORT}"]
        config.worker_class = "asyncio"
        config.workers = 1
        config.accesslog = "-"
        config.errorlog = "-"
        config.loglevel = "warning"  # Reduced log level for performance
        config.keep_alive_timeout = 65
        config.graceful_timeout = 30
        
        logger.info(f"🚀 Starting SK4FiLM Server on port {Config.WEB_SERVER_PORT}")
        logger.info(f"🌐 Website URL: {Config.WEBSITE_URL}")
        logger.info(f"🤖 Bot Username: @{Config.BOT_USERNAME}")
        logger.info(f"📁 File Channel: {Config.FILE_CHANNEL_ID}")
        logger.info(f"⚡ Streaming: {'ENABLED' if Config.STREAMING_ENABLED else 'DISABLED'}")
        logger.info(f"🔄 Safe Indexing: {'ENABLED' if not Config.INDEX_ALL_HISTORY else 'DISABLED'}")
        logger.info(f"⏱️  Rate Limiting: {Config.TELEGRAM_API_RATE_LIMIT} calls/sec")
        logger.info(f"💾 Aggressive Caching: {'ENABLED' if Config.ENABLE_AGGRESSIVE_CACHING else 'DISABLED'}")
        
        asyncio.run(serve(app, config))
        
    except KeyboardInterrupt:
        logger.info("🛑 Received interrupt signal")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
    finally:
        logger.info("👋 Server shutdown complete")

if __name__ == "__main__":
    # Store start time for uptime calculation
    start_time = time.time()
    
    # Run the application
    main()
