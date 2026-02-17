# ============================================================================
# 🚀 SK4FiLM v9.1 - COMPLETE THUMBNAIL EXTRACTION + HOME MOVIES + SEARCH PRIORITY
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

# Thumbnail Manager
try:
    from thumbnail_manager import ThumbnailManager, FallbackThumbnailManager
    THUMBNAIL_MANAGER_AVAILABLE = True
    logger.info("✅ ThumbnailManager module imported")
except ImportError as e:
    logger.error(f"❌ ThumbnailManager import error: {e}")
    THUMBNAIL_MANAGER_AVAILABLE = False

# Poster Fetcher
try:
    from poster_fetching import PosterFetcher, PosterSource
    logger.debug("✅ Poster fetching module imported")
except ImportError as e:
    logger.error(f"❌ Poster fetching module import error: {e}")
    PosterFetcher = None
    PosterSource = None

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
        is_new,
        extract_clean_title,
        extract_year,
        detect_quality_enhanced,
        has_telegram_thumbnail
    )
    logger.debug("✅ Utils module imported")
except ImportError as e:
    logger.error(f"❌ Utils module import error: {e}")
    
    # Define fallback functions
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
    
    def extract_clean_title(filename):
        if not filename: return "Unknown"
        name = os.path.splitext(filename)[0]
        name = re.sub(r'[._\-]', ' ', name)
        name = re.sub(r'\b(480p|720p|1080p|2160p|4k|hd|hevc|x264|x265|web-dl|webrip|bluray|hdtv|hdr|dts|ac3|aac|ddp|5\.1|7\.1|2\.0|esub|sub|multi|dual|audio|hindi|english|tamil|telugu|malayalam|kannada|ben|eng|hin|tam|tel|mal|kan)\b.*$', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\s+(19|20)\d{2}\s*$', '', name)
        name = re.sub(r'\s*\([^)]*\)', '', name)
        name = re.sub(r'\s*\[[^\]]*\]', '', name)
        name = re.sub(r'\s+', ' ', name)
        return name.strip() or "Unknown"
    
    def extract_year(filename):
        if not filename: return ""
        match = re.search(r'\b(19|20)\d{2}\b', filename)
        return match.group() if match else ""
    
    def detect_quality_enhanced(filename):
        if not filename: return "480p"
        filename_lower = filename.lower()
        hevc_patterns = [r'\bhevc\b', r'\bx265\b', r'\bh\.?265\b']
        is_hevc = any(re.search(p, filename_lower) for p in hevc_patterns)
        patterns = [
            (r'\b2160p\b|\b4k\b|\buhd\b', '2160p'),
            (r'\b1080p\b|\bfullhd\b|\bfhd\b', '1080p'),
            (r'\b720p\b|\bhd\b', '720p'),
            (r'\b480p\b', '480p'),
            (r'\b360p\b', '360p'),
        ]
        for pattern, quality in patterns:
            if re.search(pattern, filename_lower):
                if is_hevc and quality in ['720p', '1080p', '2160p']:
                    return f"{quality} HEVC"
                return quality
        return "480p"
    
    def has_telegram_thumbnail(message):
        try:
            if message.video and hasattr(message.video, 'thumbnail') and message.video.thumbnail:
                return True
            elif message.document and hasattr(message.document, 'thumbnail') and message.document.thumbnail:
                return True
            return False
        except:
            return False

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
    FILE_CHANNEL_ID = -1001768249569
    
    # Links
    MAIN_CHANNEL_LINK = "https://t.me/sk4film"
    UPDATES_CHANNEL_LINK = "https://t.me/sk4film_Request"
    CHANNEL_USERNAME = "sk4film"
    
    # Application
    WEBSITE_URL = os.environ.get("WEBSITE_URL", "https://sk4film.vercel.app")
    BOT_USERNAME = os.environ.get("BOT_USERNAME", "sk4filmbot")
    ADMIN_IDS = [int(x) for x in os.environ.get("ADMIN_IDS", "123456789").split(",")]
    WEB_SERVER_PORT = int(os.environ.get("PORT", 8000))
    
    # API Keys for POSTERS
    TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "e547e17d4e91f3e62a571655cd1ccaff")
    OMDB_API_KEY = os.environ.get("OMDB_API_KEY", "8265bd1c")
    
    # Performance Settings
    MAX_CONCURRENT_REQUESTS = int(os.environ.get("MAX_CONCURRENT_REQUESTS", "50"))
    CACHE_TTL = int(os.environ.get("CACHE_TTL", "300"))
    
    # Sync Management Settings
    MONITOR_INTERVAL = int(os.environ.get("MONITOR_INTERVAL", "300"))
    
    # Quality Settings
    QUALITY_PRIORITY = ['2160p HEVC', '2160p', '1080p HEVC', '1080p', '720p HEVC', '720p', '480p', '360p']
    
    # 🔥 SEARCH SETTINGS
    SEARCH_MIN_QUERY_LENGTH = 2
    SEARCH_RESULTS_PER_PAGE = 10
    
    # 🔥 HOME SETTINGS
    HOME_MOVIES_LIMIT = 25
    
    # 🔥 THUMBNAIL SETTINGS
    THUMBNAIL_EXTRACTION_ENABLED = os.environ.get("THUMBNAIL_EXTRACTION_ENABLED", "true").lower() == "true"
    THUMBNAIL_EXTRACT_ALL = os.environ.get("THUMBNAIL_EXTRACT_ALL", "true").lower() == "true"
    THUMBNAIL_BATCH_SIZE = int(os.environ.get("THUMBNAIL_BATCH_SIZE", "10"))

# ============================================================================
# ✅ GLOBAL COMPONENTS
# ============================================================================

# Database
mongo_client = None
db = None
files_col = None
thumbnails_col = None

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

# ============================================================================
# ✅ ASYNC CACHE DECORATOR
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
# ✅ BOT HANDLER
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
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Bot handler initialization error: {e}")
            return False
    
    async def get_bot_status(self):
        """Get bot status information"""
        if not self.initialized:
            return {'initialized': False, 'error': 'Bot not initialized'}
        
        try:
            bot_info = await self.bot.get_me()
            return {
                'initialized': True,
                'bot_username': bot_info.username,
                'bot_id': bot_info.id,
                'last_update': self.last_update.isoformat() if self.last_update else None,
                'is_connected': True
            }
        except Exception as e:
            return {'initialized': False, 'error': str(e)}
    
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
# ✅ OPTIMIZED SYNC MANAGER
# ============================================================================

class OptimizedSyncManager:
    def __init__(self):
        self.is_monitoring = False
        self.monitoring_task = None
        self.deleted_count = 0
        self.last_sync = time.time()
        self.sync_lock = asyncio.Lock()
    
    async def start_sync_monitoring(self):
        if self.is_monitoring:
            return
        logger.info("👁️ Starting optimized sync monitoring...")
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
            async with self.sync_lock:
                if files_col is None or User is None or not user_session_ready:
                    return
                
                current_time = time.time()
                if current_time - self.last_sync < 300:
                    return
                
                self.last_sync = current_time
                logger.info("🔍 Checking for deleted files in Telegram...")
                
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
                    messages = await User.get_messages(Config.FILE_CHANNEL_ID, message_ids)
                    
                    existing_ids = set()
                    if isinstance(messages, list):
                        for msg in messages:
                            if msg and hasattr(msg, 'id'):
                                existing_ids.add(msg.id)
                    
                    for item in message_data:
                        if item['message_id'] not in existing_ids:
                            await files_col.delete_one({"_id": item['db_id']})
                            deleted_count += 1
                            self.deleted_count += 1
                            
                            if deleted_count <= 5:
                                logger.info(f"🗑️ Auto-deleted: {item['title'][:40]}...")
                    
                    if deleted_count > 0:
                        logger.info(f"✅ Auto-deleted {deleted_count} files")
                    else:
                        logger.info("✅ No deleted files found")
                        
                except Exception as e:
                    logger.error(f"❌ Error checking messages: {e}")
                    
        except Exception as e:
            logger.error(f"❌ Auto-delete error: {e}")

sync_manager = OptimizedSyncManager()

# ============================================================================
# ✅ CHANNEL NAME CACHE
# ============================================================================

def channel_name_cached(cid):
    return f"Channel {cid}"

# ============================================================================
# ✅ HOME MOVIES - COMPLETE FUNCTION
# ============================================================================

@performance_monitor.measure("home_movies")
@async_cache_with_ttl(maxsize=1, ttl=60)
async def get_home_movies(limit=25):
    """
    🏠 HOME MOVIES - Latest posts from main channel
    - Fetches latest posts from MAIN_CHANNEL_ID
    - Gets thumbnails from database or fallback
    - Gets posters from TMDB/OMDB
    """
    try:
        if User is None or not user_session_ready:
            logger.warning("⚠️ User session not ready for home movies")
            return []
        
        movies = []
        seen_titles = set()
        
        logger.info(f"🎬 Fetching home movies from channel {Config.MAIN_CHANNEL_ID}...")
        
        # Get latest messages from main channel
        async for msg in User.get_chat_history(Config.MAIN_CHANNEL_ID, limit=50):
            if msg is not None and msg.text and len(msg.text) > 15:
                # Extract title
                title = extract_title_smart(msg.text)
                
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    
                    # Extract year
                    year_match = re.search(r'\b(19|20)\d{2}\b', title)
                    year = year_match.group() if year_match else ""
                    
                    # Clean title
                    clean_title = re.sub(r'\s+\(\d{4}\)$', '', title)
                    clean_title = re.sub(r'\s+\d{4}$', '', clean_title).strip()
                    
                    # Get thumbnail from database
                    thumbnail_info = await thumbnail_manager.get_thumbnail_for_movie(clean_title)
                    
                    # Get poster if no thumbnail
                    if thumbnail_info['source'] == 'fallback' and poster_fetcher:
                        try:
                            poster_data = await get_poster_for_movie(clean_title, year)
                            if poster_data and poster_data.get('poster_url'):
                                thumbnail_info['thumbnail_url'] = poster_data['poster_url']
                                thumbnail_info['source'] = poster_data.get('source', 'poster')
                        except Exception as e:
                            logger.debug(f"Poster fetch error for {clean_title}: {e}")
                    
                    # Check if this movie has files
                    has_files = False
                    if files_col is not None:
                        file_doc = await files_col.find_one({
                            'normalized_title': normalize_title(clean_title),
                            'has_files': True
                        })
                        has_files = file_doc is not None
                    
                    movie_data = {
                        'title': clean_title,
                        'original_title': title,
                        'normalized_title': normalize_title(clean_title),
                        'year': year,
                        'date': msg.date.isoformat() if isinstance(msg.date, datetime) else str(msg.date),
                        'is_new': is_new(msg.date) if msg.date else False,
                        'channel': channel_name_cached(Config.MAIN_CHANNEL_ID),
                        'channel_id': Config.MAIN_CHANNEL_ID,
                        'message_id': msg.id,
                        'has_files': has_files,
                        'has_post': True,
                        'content': format_post(msg.text, max_length=500),
                        'post_content': msg.text,
                        'result_type': 'post',
                        'thumbnail_url': thumbnail_info['thumbnail_url'],
                        'thumbnail_source': thumbnail_info['source'],
                        'has_thumbnail': thumbnail_info['source'] != 'fallback'
                    }
                    
                    movies.append(movie_data)
                    
                    if len(movies) >= limit:
                        break
        
        logger.info(f"✅ Home movies fetched: {len(movies)}")
        
        # Statistics
        thumb_count = sum(1 for m in movies if m.get('thumbnail_source') == 'extracted')
        poster_count = sum(1 for m in movies if m.get('thumbnail_source') in ['tmdb', 'omdb', 'poster'])
        fallback_count = sum(1 for m in movies if m.get('thumbnail_source') == 'fallback')
        
        logger.info(f"🏠 HOME Thumbnail Sources: Extracted: {thumb_count}, Poster: {poster_count}, Fallback: {fallback_count}")
        
        return movies[:limit]
        
    except Exception as e:
        logger.error(f"❌ Home movies error: {e}")
        return []

# ============================================================================
# ✅ POSTER FETCHING FUNCTIONS
# ============================================================================

async def get_poster_for_movie(title: str, year: str = "") -> Dict[str, Any]:
    """Get poster for movie - Returns empty string if not found"""
    global poster_fetcher
    
    if poster_fetcher is None:
        return {
            'poster_url': '',
            'source': 'none',
            'rating': '0.0',
            'year': year,
            'title': title,
            'found': False
        }
    
    try:
        poster_task = asyncio.create_task(poster_fetcher.fetch_poster(title))
        
        try:
            poster_data = await asyncio.wait_for(poster_task, timeout=3.0)
            
            if poster_data and poster_data.get('poster_url'):
                logger.debug(f"✅ Poster fetched: {title[:30]} - {poster_data['source']}")
                poster_data['found'] = True
                return poster_data
            else:
                raise ValueError("Invalid poster data")
                
        except (asyncio.TimeoutError, ValueError, Exception) as e:
            logger.warning(f"⚠️ Poster fetch timeout/error for {title[:30]}: {e}")
            
            if not poster_task.done():
                poster_task.cancel()
            
            return {
                'poster_url': '',
                'source': 'none',
                'rating': '0.0',
                'year': year,
                'title': title,
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
            'found': False
        }

async def get_posters_for_movies_batch(movies: List[Dict]) -> List[Dict]:
    """Get posters for multiple movies in batch"""
    results = []
    
    tasks = []
    for movie in movies:
        title = movie.get('title', '')
        year = movie.get('year', '')
        
        task = asyncio.create_task(get_poster_for_movie(title, year))
        tasks.append((movie, task))
    
    for movie, task in tasks:
        try:
            poster_data = await task
            
            movie_with_poster = movie.copy()
            if poster_data['found']:
                movie_with_poster.update({
                    'poster_url': poster_data['poster_url'],
                    'poster_source': poster_data['source'],
                    'poster_rating': poster_data['rating'],
                    'has_poster': True,
                    'found': True
                })
            else:
                movie_with_poster.update({
                    'poster_url': '',
                    'poster_source': 'none',
                    'poster_rating': '0.0',
                    'has_poster': False,
                    'found': False
                })
            
            results.append(movie_with_poster)
            
        except Exception as e:
            logger.warning(f"⚠️ Batch poster error for {movie.get('title', '')[:30]}: {e}")
            
            movie_with_empty = movie.copy()
            movie_with_empty.update({
                'poster_url': '',
                'poster_source': 'none',
                'poster_rating': '0.0',
                'has_poster': False,
                'found': False
            })
            
            results.append(movie_with_empty)
    
    return results

# ============================================================================
# ✅ COMPLETE SEARCH - MERGED RESULTS WITH PRIORITY
# ============================================================================

async def search_movies_complete(query, limit=10, page=1):
    """
    🔥 COMPLETE SEARCH with PRIORITY:
    - Priority 1: Post + Files (merged quality options)
    - Priority 2: Post only
    - Priority 3: File only
    """
    offset = (page - 1) * limit
    
    logger.info(f"🔍 COMPLETE SEARCH for: '{query}'")
    
    # Results by type
    post_file_results = []  # Priority 1: Has both post and file
    post_only_results = []   # Priority 2: Has post but no file
    file_only_results = []   # Priority 3: Has file but no post
    
    seen_titles = set()
    
    # ============================================================================
    # ✅ STEP 1: Search TEXT CHANNELS for posts
    # ============================================================================
    if user_session_ready and User is not None:
        logger.info(f"📝 Searching TEXT CHANNELS for posts...")
        
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
                    
                    # Skip if already seen
                    if normalized in seen_titles:
                        continue
                    
                    seen_titles.add(normalized)
                    
                    clean_title = re.sub(r'\s*\(\d{4}\)', '', title)
                    clean_title = re.sub(r'\s+\d{4}$', '', clean_title).strip()
                    
                    year_match = re.search(r'\b(19|20)\d{2}\b', title)
                    year = year_match.group() if year_match else ""
                    
                    # Check if this movie has files
                    has_files = False
                    qualities = {}
                    
                    if files_col is not None:
                        file_doc = await files_col.find_one({
                            'normalized_title': normalized,
                            'has_files': True
                        })
                        if file_doc:
                            has_files = True
                            qualities = file_doc.get('qualities', {})
                    
                    # Get thumbnail
                    thumbnail_info = await thumbnail_manager.get_thumbnail_for_movie(clean_title)
                    
                    # Get poster if no thumbnail
                    if thumbnail_info['source'] == 'fallback' and poster_fetcher:
                        poster_data = await get_poster_for_movie(clean_title, year)
                        if poster_data.get('poster_url'):
                            thumbnail_info['thumbnail_url'] = poster_data['poster_url']
                            thumbnail_info['source'] = poster_data.get('source', 'poster')
                    
                    movie_data = {
                        'title': clean_title,
                        'original_title': title,
                        'normalized_title': normalized,
                        'year': year,
                        'content': format_post(msg.text, max_length=500),
                        'post_content': msg.text,
                        'channel_id': channel_id,
                        'message_id': msg.id,
                        'date': msg.date.isoformat() if msg.date else None,
                        'is_new': is_new(msg.date) if msg.date else False,
                        'has_post': True,
                        'has_files': has_files,
                        'qualities': qualities if has_files else {},
                        'available_qualities': list(qualities.keys()) if has_files else [],
                        'result_type': 'post_and_files' if has_files else 'post_only',
                        'thumbnail_url': thumbnail_info['thumbnail_url'],
                        'thumbnail_source': thumbnail_info['source']
                    }
                    
                    if has_files:
                        post_file_results.append(movie_data)
                    else:
                        post_only_results.append(movie_data)
                    
                    if len(post_file_results) + len(post_only_results) >= 50:
                        break
                        
            except Exception as e:
                logger.debug(f"Text search error in {channel_id}: {e}")
                continue
    
    # ============================================================================
    # ✅ STEP 2: Search FILES collection for files only results
    # ============================================================================
    if files_col is not None:
        logger.info(f"📁 Searching FILES collection...")
        
        # Simple regex search
        regex_pattern = re.escape(query)
        
        cursor = files_col.find({
            'title': {'$regex': regex_pattern, '$options': 'i'},
            'has_files': True
        }).sort('date', -1).limit(50)
        
        async for doc in cursor:
            normalized = doc['normalized_title']
            
            # Skip if already seen (from posts)
            if normalized in seen_titles:
                continue
            
            seen_titles.add(normalized)
            
            # Get thumbnail
            thumbnail_info = await thumbnail_manager.get_thumbnail_for_movie(doc['title'])
            
            # Get poster if no thumbnail
            if thumbnail_info['source'] == 'fallback' and poster_fetcher:
                poster_data = await get_poster_for_movie(doc['title'], doc.get('year', ''))
                if poster_data.get('poster_url'):
                    thumbnail_info['thumbnail_url'] = poster_data['poster_url']
                    thumbnail_info['source'] = poster_data.get('source', 'poster')
            
            movie_data = {
                'title': doc['title'],
                'original_title': doc['title'],
                'normalized_title': normalized,
                'year': doc.get('year', ''),
                'has_post': False,
                'has_files': True,
                'qualities': doc.get('qualities', {}),
                'available_qualities': doc.get('available_qualities', []),
                'result_type': 'files_only',
                'thumbnail_url': thumbnail_info['thumbnail_url'],
                'thumbnail_source': thumbnail_info['source']
            }
            
            file_only_results.append(movie_data)
    
    # ============================================================================
    # ✅ STEP 3: Apply PRIORITY ORDER
    # ============================================================================
    
    # Sort each group by date (newest first)
    post_file_results.sort(key=lambda x: x.get('date', ''), reverse=True)
    post_only_results.sort(key=lambda x: x.get('date', ''), reverse=True)
    file_only_results.sort(key=lambda x: x.get('date', ''), reverse=True)
    
    # Final ordered results
    final_results = []
    
    # Priority 1: Post + Files
    for result in post_file_results[:limit]:
        final_results.append(result)
    
    # Priority 2: Post Only
    if len(final_results) < limit:
        remaining = limit - len(final_results)
        for result in post_only_results[:remaining]:
            final_results.append(result)
    
    # Priority 3: Files Only
    if len(final_results) < limit:
        remaining = limit - len(final_results)
        for result in file_only_results[:remaining]:
            final_results.append(result)
    
    # ============================================================================
    # ✅ STEP 4: PAGINATION
    # ============================================================================
    total = len(post_file_results) + len(post_only_results) + len(file_only_results)
    start_idx = offset
    end_idx = offset + limit
    paginated = final_results[start_idx:end_idx]
    
    logger.info("=" * 60)
    logger.info("📊 SEARCH RESULTS SUMMARY:")
    logger.info(f"   • Query: '{query}'")
    logger.info(f"   • Total results: {total}")
    logger.info(f"   • Priority 1 (Post+Files): {len(post_file_results)}")
    logger.info(f"   • Priority 2 (Post Only): {len(post_only_results)}")
    logger.info(f"   • Priority 3 (Files Only): {len(file_only_results)}")
    logger.info(f"   • Page: {page}/{max(1, (total + limit - 1) // limit)}")
    logger.info("=" * 60)
    
    return {
        'results': paginated,
        'pagination': {
            'current_page': page,
            'total_pages': max(1, (total + limit - 1) // limit) if total > 0 else 1,
            'total_results': total,
            'per_page': limit,
            'has_next': page < ((total + limit - 1) // limit) if total > 0 else False,
            'has_previous': page > 1,
            'counts': {
                'post_files': len(post_file_results),
                'post_only': len(post_only_results),
                'files_only': len(file_only_results)
            }
        },
        'bot_username': Config.BOT_USERNAME
    }

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
    
    # Initialize USER session
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
    
    return bot_session_ready or user_session_ready

# ============================================================================
# ✅ MONGODB INITIALIZATION
# ============================================================================

@performance_monitor.measure("mongodb_init")
async def init_mongodb():
    global mongo_client, db, files_col, thumbnails_col
    
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
        thumbnails_col = db.thumbnails
        
        logger.info("✅ MongoDB OK - Files and Thumbnails collections initialized")
        return True
        
    except asyncio.TimeoutError:
        logger.error("❌ MongoDB connection timeout")
        return False
    except Exception as e:
        logger.error(f"❌ MongoDB error: {e}")
        return False

# ============================================================================
# ✅ DATABASE INDEXES
# ============================================================================

async def setup_database_indexes():
    """Create all necessary database indexes"""
    if files_col is None:
        logger.warning("⚠️ Files collection not available")
        return
    
    try:
        # Files collection indexes
        await files_col.create_index(
            [("normalized_title", 1), ("channel_id", 1)],
            unique=True,
            name="title_channel_unique",
            background=True
        )
        
        await files_col.create_index("year", background=True)
        await files_col.create_index("channel_id", background=True)
        await files_col.create_index("has_files", background=True)
        
        # Thumbnails collection indexes
        if thumbnails_col is not None:
            await thumbnails_col.create_index(
                [("normalized_title", 1), ("quality", 1)],
                unique=True,
                name="title_quality_unique",
                background=True
            )
            
            await thumbnails_col.create_index(
                [("normalized_title", 1), ("has_thumbnail", 1)],
                name="title_thumbnail",
                background=True
            )
            
            await thumbnails_col.create_index(
                "expires_at",
                expireAfterSeconds=0,
                name="ttl_index",
                background=True
            )
        
        logger.info("✅ All database indexes created successfully")
        
    except Exception as e:
        logger.warning(f"⚠️ Index creation error: {e}")

# ============================================================================
# ✅ THUMBNAIL MANAGER INITIALIZATION
# ============================================================================

async def init_thumbnail_manager():
    """Initialize Thumbnail Manager"""
    global thumbnail_manager
    
    try:
        if THUMBNAIL_MANAGER_AVAILABLE:
            thumbnail_manager = ThumbnailManager(
                download_path="downloads/thumbnails",
                mongodb=db,
                bot_client=Bot if bot_session_ready else None,
                user_client=User if user_session_ready else None,
                file_channel_id=Config.FILE_CHANNEL_ID,
                redis_client=cache_manager.redis_client if cache_manager else None
            )
            await thumbnail_manager.initialize()
            logger.info("✅ Thumbnail Manager initialized successfully")
            
            # Start complete thumbnail extraction
            if Config.THUMBNAIL_EXTRACT_ALL and (user_session_ready or bot_session_ready):
                logger.info("🔍 Starting complete thumbnail extraction for all files...")
                asyncio.create_task(thumbnail_manager.extract_all_thumbnails_from_channel())
            
            return True
        else:
            logger.warning("⚠️ Thumbnail Manager not available")
            thumbnail_manager = FallbackThumbnailManager()
            await thumbnail_manager.initialize()
            return False
    except Exception as e:
        logger.error(f"❌ Thumbnail Manager initialization failed: {e}")
        thumbnail_manager = FallbackThumbnailManager()
        await thumbnail_manager.initialize()
        return False

# ============================================================================
# ✅ MAIN INITIALIZATION
# ============================================================================

@performance_monitor.measure("system_init")
async def init_system():
    start_time = time.time()
    
    try:
        logger.info("=" * 60)
        logger.info("🚀 SK4FiLM v9.1 - COMPLETE THUMBNAIL EXTRACTION")
        logger.info("=" * 60)
        
        # Initialize MongoDB
        mongo_ok = await init_mongodb()
        if not mongo_ok:
            logger.error("❌ MongoDB connection failed")
            return False
        
        # Initialize Cache Manager
        global cache_manager, verification_system, premium_system, poster_fetcher
        cache_manager = CacheManager(Config)
        redis_ok = await cache_manager.init_redis()
        if redis_ok:
            logger.info("✅ Cache Manager initialized")
            await cache_manager.start_cleanup_task()
        
        # Initialize Bot Handler
        bot_handler_ok = await bot_handler.initialize()
        if bot_handler_ok:
            logger.info("✅ Bot Handler initialized")
        
        # Initialize Telegram Sessions
        if PYROGRAM_AVAILABLE:
            telegram_ok = await init_telegram_sessions()
            if not telegram_ok:
                logger.warning("⚠️ Telegram sessions partially failed")
        
        # Setup database indexes
        await setup_database_indexes()
        
        # Initialize Poster Fetcher
        if PosterFetcher is not None:
            poster_fetcher = PosterFetcher(Config, cache_manager.redis_client if cache_manager else None)
            logger.info("✅ Poster Fetcher initialized")
        
        # Initialize Thumbnail Manager
        await init_thumbnail_manager()
        
        # Start sync monitoring
        if user_session_ready:
            await sync_manager.start_sync_monitoring()
            logger.info("✅ Sync monitoring started")
        
        init_time = time.time() - start_time
        logger.info(f"⚡ SK4FiLM Started in {init_time:.2f}s")
        logger.info("=" * 60)
        logger.info("🔧 FEATURES:")
        logger.info(f"   • Complete Thumbnail Extraction: ✅ ENABLED")
        logger.info(f"   • Search Priority: Post+Files → Post Only → Files Only")
        logger.info(f"   • Merged Quality Options: ✅ ENABLED")
        logger.info(f"   • Home Movies: ✅ ENABLED")
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

app = Quart(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

@app.after_request
async def add_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['X-SK4FiLM-Version'] = '9.1-COMPLETE-EXTRACTION'
    response.headers['X-Response-Time'] = f"{time.perf_counter():.3f}"
    return response

@app.route('/')
@performance_monitor.measure("root_endpoint")
async def root():
    if files_col is not None:
        total_movies = await files_col.count_documents({'has_files': True})
    else:
        total_movies = 0
    
    thumbnail_stats = {}
    if thumbnail_manager:
        thumbnail_stats = await thumbnail_manager.get_stats()
    
    bot_status = None
    if bot_handler:
        try:
            bot_status = await bot_handler.get_bot_status()
        except Exception as e:
            bot_status = {'initialized': False, 'error': str(e)}
    
    return jsonify({
        'status': 'healthy',
        'service': 'SK4FiLM v9.1 - Complete Thumbnail Extraction',
        'features': {
            'thumbnail_extraction': 'ALL files',
            'search_priority': 'Post+Files → Post Only → Files Only',
            'merged_qualities': True,
            'home_movies': True
        },
        'stats': {
            'total_movies': total_movies,
            'thumbnails_extracted': thumbnail_stats.get('extraction_success', 0)
        },
        'sessions': {
            'user_session': user_session_ready,
            'bot_session': bot_session_ready,
            'bot_handler': bot_status
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/movies', methods=['GET'])
@performance_monitor.measure("movies_endpoint")
async def api_movies():
    """Get home movies"""
    try:
        limit = int(request.args.get('limit', Config.HOME_MOVIES_LIMIT))
        movies = await get_home_movies(limit=limit)
        
        return jsonify({
            'status': 'success' if movies else 'empty',
            'movies': movies,
            'total': len(movies),
            'limit': limit,
            'source': 'telegram',
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
    """Complete search with priority results"""
    try:
        query = request.args.get('query', '').strip()
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', Config.SEARCH_RESULTS_PER_PAGE))
        
        if len(query) < Config.SEARCH_MIN_QUERY_LENGTH:
            return jsonify({
                'status': 'error',
                'message': f'Query must be at least {Config.SEARCH_MIN_QUERY_LENGTH} characters'
            }), 400
        
        result_data = await search_movies_complete(query, limit, page)
        
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
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/thumbnails/status', methods=['GET'])
async def thumbnail_status():
    """Get thumbnail extraction status"""
    if thumbnail_manager:
        stats = await thumbnail_manager.get_stats()
        return jsonify({
            'status': 'success',
            'thumbnail_stats': stats,
            'extraction_complete': stats.get('extraction_success', 0) > 0
        })
    else:
        return jsonify({
            'status': 'error',
            'message': 'Thumbnail manager not initialized'
        }), 400

@app.route('/api/thumbnails/extract', methods=['POST'])
async def trigger_extraction():
    """Manually trigger thumbnail extraction"""
    auth_token = request.headers.get('X-Admin-Token')
    if not auth_token or auth_token != os.environ.get('ADMIN_TOKEN', 'sk4film_admin_123'):
        return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401
    
    if thumbnail_manager and (user_session_ready or bot_session_ready):
        asyncio.create_task(thumbnail_manager.extract_all_thumbnails_from_channel(force_re_extract=True))
        return jsonify({
            'status': 'success',
            'message': 'Thumbnail extraction started',
            'timestamp': datetime.now().isoformat()
        })
    else:
        return jsonify({
            'status': 'error',
            'message': 'Thumbnail manager not ready'
        }), 400

@app.route('/api/health')
async def health():
    return jsonify({
        'status': 'ok',
        'sessions': {
            'user': user_session_ready,
            'bot': bot_session_ready
        },
        'thumbnail_manager': thumbnail_manager is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/stats')
async def stats():
    perf_stats = performance_monitor.get_stats()
    
    thumbnail_stats = {}
    if thumbnail_manager:
        thumbnail_stats = await thumbnail_manager.get_stats()
    
    return jsonify({
        'status': 'success',
        'performance': perf_stats,
        'thumbnail_manager': thumbnail_stats,
        'sessions': {
            'user': user_session_ready,
            'bot': bot_session_ready
        },
        'timestamp': datetime.now().isoformat()
    })

# ============================================================================
# ✅ STARTUP AND SHUTDOWN
# ============================================================================

app_start_time = time.time()

@app.before_serving
async def startup():
    await init_system()

@app.after_serving
async def shutdown():
    logger.info("🛑 Shutting down SK4FiLM v9.1...")
    
    shutdown_tasks = []
    
    if thumbnail_manager:
        try:
            await thumbnail_manager.shutdown()
            logger.info("✅ Thumbnail Manager stopped")
        except Exception as e:
            logger.error(f"❌ Thumbnail Manager shutdown error: {e}")
    
    await sync_manager.stop_sync_monitoring()
    
    if bot_handler:
        try:
            await bot_handler.shutdown()
            logger.info("✅ Bot Handler stopped")
        except Exception as e:
            logger.error(f"❌ Bot Handler shutdown error: {e}")
    
    if poster_fetcher is not None and hasattr(poster_fetcher, 'close'):
        try:
            await poster_fetcher.close()
            logger.info("✅ Poster Fetcher closed")
        except:
            pass
    
    if User is not None:
        shutdown_tasks.append(User.stop())
    
    if Bot is not None:
        shutdown_tasks.append(Bot.stop())
    
    if cache_manager is not None:
        shutdown_tasks.append(cache_manager.stop())
    
    if shutdown_tasks:
        await asyncio.gather(*shutdown_tasks, return_exceptions=True)
    
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
    
    logger.info(f"🌐 Starting SK4FiLM v9.1 on port {Config.WEB_SERVER_PORT}...")
    logger.info(f"📁 File Channel ID: {Config.FILE_CHANNEL_ID}")
    logger.info(f"🔍 SEARCH: Post+Files → Post Only → Files Only")
    logger.info(f"🖼️ Thumbnails: Extracting ALL files in background")
    logger.info(f"🏠 HOME: Latest movies from main channel")
    
    try:
        asyncio.run(serve(app, config))
    except KeyboardInterrupt:
        logger.info("👋 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
