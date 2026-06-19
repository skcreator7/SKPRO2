# ============================================================================
# 🚀 SK4FiLM v9.5 - COMPLETE FIXED BOT WITH /api/movies
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
    from poster_fetching import PosterFetcher, PosterSource, create_poster_fetcher
    logger.debug("✅ Poster fetching module imported")
    POSTER_FETCHER_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ Poster fetching module import error: {e}")
    PosterFetcher = None
    PosterSource = None
    POSTER_FETCHER_AVAILABLE = False
    async def create_poster_fetcher(*args, **kwargs):
        return None

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
    response.headers['X-SK4FiLM-Version'] = '9.5-FIXED-BOT'
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

# Telegram Sessions
try:
    from pyrogram import Client, filters
    from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery, Message
    from pyrogram.errors import FloodWait, BadRequest, MessageDeleteForbidden
    PYROGRAM_AVAILABLE = True
    User = None
    Bot = None
    user_session_ready = False
    bot_session_ready = False
except ImportError:
    PYROGRAM_AVAILABLE = False
    User = None
    Bot = None
    class filters:
        @staticmethod
        def command(cmd): return lambda x: x
        @staticmethod
        def private(): return lambda x: x
    class InlineKeyboardMarkup:
        def __init__(self, buttons): pass
    class InlineKeyboardButton:
        def __init__(self, text, url=None, callback_data=None): pass
    class CallbackQuery: pass
    class Message: pass

# System Components
cache_manager = None
verification_system = None
premium_system = None
poster_fetcher = None
bot_handler = None
telegram_bot = None

# Thumbnail Manager
thumbnail_manager = None

# Indexing State
is_indexing = False
last_index_time = None
indexing_task = None

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
    MAIN_CHANNEL_ID = -1001767371495
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
    VERIFICATION_REQUIRED = os.environ.get("VERIFICATION_REQUIRED", "False").lower() == "True"
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
    
    # Optimization Settings
    POSTER_FETCHING_ENABLED = True
    POSTER_CACHE_TTL = 86400
    POSTER_FETCH_TIMEOUT = 5
    MAX_CONCURRENT_REQUESTS = int(os.environ.get("MAX_CONCURRENT_REQUESTS", "20"))
    CACHE_TTL = int(os.environ.get("CACHE_TTL", "1200"))
    REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "5"))
    
    # Sync Management Settings
    MONITOR_INTERVAL = int(os.environ.get("MONITOR_INTERVAL", "600"))
    
    # Quality Settings
    QUALITY_PRIORITY = ['2160p', '1080p', '720p', '480p', '360p']
    HEVC_VARIANTS = ['720p HEVC', '1080p HEVC', '2160p HEVC']
    
    # Thumbnail Extraction Settings
    THUMBNAIL_EXTRACTION_ENABLED = True
    THUMBNAIL_BATCH_SIZE = 3
    THUMBNAIL_EXTRACT_TIMEOUT = 5
    THUMBNAIL_CACHE_DURATION = 24 * 60 * 60
    THUMBNAIL_RETRY_LIMIT = 1
    THUMBNAIL_MAX_SIZE_KB = 200
    THUMBNAIL_TTL_DAYS = 365
    
    # File Channel Indexing Settings
    AUTO_INDEX_INTERVAL = 3600
    BATCH_INDEX_SIZE = 50
    MAX_INDEX_LIMIT = 500
    INDEX_ALL_HISTORY = False
    INSTANT_AUTO_INDEX = True
    
    # Search Settings
    SEARCH_MIN_QUERY_LENGTH = 2
    SEARCH_RESULTS_PER_PAGE = 12
    SEARCH_CACHE_TTL = 1800
    
    # Home Movies Settings
    HOME_MOVIES_LIMIT = 100
    HOME_MOVIES_CACHE_TTL = 3600

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
    def __init__(self, bot_token=None, api_id=None, api_hash=None):
        self.bot_token = bot_token or Config.BOT_TOKEN
        self.api_id = api_id or Config.API_ID
        self.api_hash = api_hash or Config.API_HASH
        self.bot = None
        self.initialized = False
        self.last_update = None
        self.bot_username = None
        self.auto_delete_tasks = {}
        self.auto_delete_messages = {}
        
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
            asyncio.create_task(self._cleanup_old_auto_delete_tasks())
            return True
            
        except Exception as e:
            logger.error(f"❌ Bot handler initialization error: {e}")
            return False
    
    async def _periodic_tasks(self):
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
    
    async def send_file_to_user(self, user_id: int, channel_id: int, message_id: int, quality: str = "480p") -> Tuple[bool, Dict, int]:
        try:
            if not self.initialized:
                return False, {'message': 'Bot not initialized'}, 0
            
            logger.info(f"📤 Sending file to user {user_id}: channel={channel_id}, msg={message_id}, quality={quality}")
            
            message = await self.bot.get_messages(channel_id, message_id)
            if not message:
                return False, {'message': 'File not found in channel'}, 0
            
            if not message.document and not message.video:
                return False, {'message': 'Not a downloadable file'}, 0
            
            if message.document:
                file_name = message.document.file_name or "file.bin"
                file_size = message.document.file_size or 0
                file_id = message.document.file_id
                is_video = False
            else:
                file_name = message.video.file_name or "video.mp4"
                file_size = message.video.file_size or 0
                file_id = message.video.file_id
                is_video = True
            
            caption = (
                f"📁 **File:** `{file_name}`\n"
                f"📦 **Size:** {format_size(file_size)}\n"
                f"📹 **Quality:** {quality}\n"
                f"⏰ **Auto-delete in:** {Config.AUTO_DELETE_TIME} minutes\n\n"
                f"🔗 **More movies:** {Config.WEBSITE_URL}\n"
                f"🎬 **@SK4FiLM**"
            )
            
            buttons = InlineKeyboardMarkup([
                [InlineKeyboardButton("🌐 VISIT WEBSITE", url=Config.WEBSITE_URL)],
                [InlineKeyboardButton("📢 JOIN CHANNEL", url=Config.MAIN_CHANNEL_LINK)],
                [InlineKeyboardButton("🔄 GET ANOTHER", callback_data="back_to_start")]
            ])
            
            try:
                if is_video:
                    sent = await self.bot.send_video(
                        user_id,
                        file_id,
                        caption=caption,
                        reply_markup=buttons,
                        supports_streaming=True
                    )
                else:
                    sent = await self.bot.send_document(
                        user_id,
                        file_id,
                        caption=caption,
                        reply_markup=buttons
                    )
                
                logger.info(f"✅ File sent to user {user_id}: {file_name} ({format_size(file_size)})")
                
                task_id = f"{user_id}_{sent.id}"
                task = asyncio.create_task(
                    self._auto_delete_file(user_id, sent.id, file_name, Config.AUTO_DELETE_TIME)
                )
                self.auto_delete_tasks[task_id] = task
                self.auto_delete_messages[task_id] = {
                    'user_id': user_id,
                    'message_id': sent.id,
                    'file_name': file_name,
                    'scheduled_time': datetime.now() + timedelta(minutes=Config.AUTO_DELETE_TIME),
                    'status': 'pending'
                }
                
                return True, {
                    'success': True,
                    'file_name': file_name,
                    'file_size': file_size,
                    'quality': quality,
                    'message_id': sent.id,
                    'auto_delete_minutes': Config.AUTO_DELETE_TIME
                }, file_size
                
            except FloodWait as e:
                logger.warning(f"⏳ Flood wait: {e.value}s")
                await asyncio.sleep(e.value)
                return await self.send_file_to_user(user_id, channel_id, message_id, quality)
                
        except Exception as e:
            logger.error(f"❌ Send file error: {e}")
            return False, {'message': f'Error: {str(e)}'}, 0
    
    async def _auto_delete_file(self, user_id: int, message_id: int, file_name: str, minutes: int):
        try:
            task_id = f"{user_id}_{message_id}"
            logger.info(f"⏰ Auto-delete scheduled for user {user_id}, message {message_id} in {minutes} minutes")
            
            await asyncio.sleep(minutes * 60)
            
            try:
                await self.bot.delete_messages(user_id, message_id)
                logger.info(f"🗑️ Auto-deleted file for user {user_id}: {file_name}")
                
                if task_id in self.auto_delete_messages:
                    self.auto_delete_messages[task_id]['status'] = 'completed'
                    self.auto_delete_messages[task_id]['completed_at'] = datetime.now()
                
                await self.bot.send_message(
                    user_id,
                    f"🗑️ **File Auto-Deleted**\n\n"
                    f"✅ Security measure completed\n\n"
                    f"🌐 Visit website to download again: {Config.WEBSITE_URL}",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)]
                    ])
                )
                
            except Exception as e:
                logger.debug(f"Delete error: {e}")
                    
        except asyncio.CancelledError:
            task_id = f"{user_id}_{message_id}"
            if task_id in self.auto_delete_messages:
                self.auto_delete_messages[task_id]['status'] = 'cancelled'
        except Exception as e:
            logger.error(f"❌ Auto-delete error: {e}")
        finally:
            task_id = f"{user_id}_{message_id}"
            if task_id in self.auto_delete_tasks:
                del self.auto_delete_tasks[task_id]
    
    async def _cleanup_old_auto_delete_tasks(self):
        while self.initialized:
            try:
                await asyncio.sleep(3600)
                now = datetime.now()
                to_remove = []
                for task_id, data in self.auto_delete_messages.items():
                    completed_at = data.get('completed_at')
                    if completed_at and (now - completed_at).total_seconds() > 24 * 3600:
                        to_remove.append(task_id)
                for task_id in to_remove:
                    self.auto_delete_messages.pop(task_id, None)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-delete cleanup error: {e}")
    
    async def get_auto_delete_stats(self):
        stats = {
            'total_tasks': len(self.auto_delete_messages),
            'active_tasks': len(self.auto_delete_tasks),
            'pending': 0,
            'completed': 0,
            'cancelled': 0
        }
        for data in self.auto_delete_messages.values():
            status = data.get('status', 'unknown')
            if status == 'pending':
                stats['pending'] += 1
            elif status == 'completed':
                stats['completed'] += 1
            elif 'cancelled' in status:
                stats['cancelled'] += 1
        return stats
    
    async def get_bot_status(self):
        if not self.initialized:
            return {'initialized': False, 'error': 'Bot not initialized'}
        try:
            bot_info = await self.bot.get_me()
            auto_delete_stats = await self.get_auto_delete_stats()
            return {
                'initialized': True,
                'bot_username': bot_info.username,
                'bot_id': bot_info.id,
                'first_name': bot_info.first_name,
                'last_update': self.last_update.isoformat() if self.last_update else None,
                'is_connected': True,
                'auto_delete': auto_delete_stats
            }
        except Exception as e:
            return {'initialized': False, 'error': str(e)}
    
    async def shutdown(self):
        logger.info("Shutting down bot handler...")
        for task_id, task in list(self.auto_delete_tasks.items()):
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self.auto_delete_tasks.clear()
        self.auto_delete_messages.clear()
        self.initialized = False
        if self.bot:
            try:
                await self.bot.stop()
            except Exception as e:
                logger.error(f"❌ Error stopping bot: {e}")

bot_handler = BotHandler()

# ============================================================================
# ✅ HELPER FUNCTIONS
# ============================================================================

def extract_clean_title(filename):
    if not filename:
        return "Unknown"
    name = os.path.splitext(filename)[0]
    name = re.sub(r'[._\-]', ' ', name)
    name = re.sub(r'\b(480p|720p|1080p|2160p|4k|hd|hevc|x265|x264|web-dl|webrip|bluray|hdtv|hdr|dts|ac3|aac|ddp|5\.1|7\.1|2\.0|esub|sub|multi|dual|audio|hindi|english|tamil|telugu|malayalam|kannada|ben|eng|hin|tam|tel|mal|kan)\b.*$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s+(19|20)\d{2}\s*$', '', name)
    name = re.sub(r'\s*\([^)]*\)', '', name)
    name = re.sub(r'\s*\[[^\]]*\]', '', name)
    name = re.sub(r'\s+', ' ', name)
    return name.strip() if name else "Unknown"

def extract_year(filename):
    if not filename:
        return ""
    year_match = re.search(r'\b(19|20)\d{2}\b', filename)
    return year_match.group() if year_match else ""

def has_telegram_thumbnail(message):
    try:
        media = message.video or message.document
        if not media:
            return False
        if getattr(media, "thumbs", None):
            return len(media.thumbs) > 0
        if getattr(media, "sizes", None):
            return len(media.sizes) > 0
        if getattr(media, "thumbnail", None):
            return True
        return False
    except Exception:
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
    HEVC_PATTERNS = [r'\bhevc\b', r'\bx265\b', r'\bh\.?265\b']
    is_hevc = any(re.search(pattern, filename_lower) for pattern in HEVC_PATTERNS)
    for pattern, quality in QUALITY_PATTERNS:
        if re.search(pattern, filename_lower):
            if is_hevc and quality in ['720p', '1080p', '2160p']:
                return f"{quality} HEVC"
            return quality
    return "480p"

async def try_store_real_thumbnail(normalized_title: str, clean_title: str, msg) -> None:
    try:
        if not msg or thumbnails_col is None:
            return
        existing = await thumbnails_col.find_one({
            'normalized_title': normalized_title,
            'has_thumbnail': True
        })
        if existing:
            return
        media = msg.video or msg.document
        if not media:
            return
        file_name = getattr(media, "file_name", "video.mp4")
        if not is_video_file(file_name):
            return
        thumbnail_file_id = None
        if msg.video:
            if hasattr(msg.video, 'thumbnail') and msg.video.thumbnail:
                thumbnail_file_id = msg.video.thumbnail.file_id
            elif hasattr(msg.video, 'thumbs') and msg.video.thumbs:
                thumbnail_file_id = msg.video.thumbs[0].file_id
        elif msg.document:
            if hasattr(msg.document, 'thumbnail') and msg.document.thumbnail:
                thumbnail_file_id = msg.document.thumbnail.file_id
            elif hasattr(msg.document, 'thumbs') and msg.document.thumbs:
                thumbnail_file_id = msg.document.thumbs[0].file_id
        if not thumbnail_file_id:
            return
        client = User if user_session_ready else Bot
        if not client:
            return
        downloaded = await client.download_media(thumbnail_file_id, in_memory=True)
        if not downloaded:
            return
        if isinstance(downloaded, bytes):
            thumbnail_data = downloaded
        else:
            downloaded.seek(0)
            thumbnail_data = downloaded.read()
        thumbnail_url = f"data:image/jpeg;base64,{base64.b64encode(thumbnail_data).decode()}"
        size_kb = len(thumbnail_url) / 1024
        await thumbnails_col.update_one(
            {'normalized_title': normalized_title},
            {'$set': {
                'normalized_title': normalized_title,
                'title': clean_title,
                'year': extract_year(file_name),
                'thumbnail_url': thumbnail_url,
                'thumbnail_source': 'telegram',
                'has_thumbnail': True,
                'extracted_at': datetime.now(),
                'message_id': msg.id,
                'channel_id': Config.FILE_CHANNEL_ID,
                'file_name': file_name,
                'size_kb': size_kb,
                'file_id': thumbnail_file_id
            }},
            upsert=True
        )
        logger.info(f"✅✅✅ Thumbnail stored: {clean_title} ({size_kb:.1f}KB)")
    except Exception as e:
        logger.error(f"❌ Thumbnail error for {clean_title}: {e}")

async def get_best_thumbnail(normalized_title: str, clean_title: str = None, 
                            year: str = None, msg=None, is_post: bool = False) -> Tuple[str, str]:
    if thumbnails_col is not None:
        try:
            doc = await thumbnails_col.find_one({
                'normalized_title': normalized_title,
                'has_thumbnail': True,
                'thumbnail_url': {'$exists': True, '$ne': None}
            })
            if doc and doc.get('thumbnail_url'):
                return doc['thumbnail_url'], 'mongodb'
        except Exception:
            pass
    if Config.POSTER_FETCHING_ENABLED and poster_fetcher and clean_title:
        try:
            poster = await poster_fetcher.fetch_poster(clean_title, year or "")
            if poster and poster.get('poster_url') and poster.get('found'):
                source = poster.get('source', 'unknown')
                if not is_post and msg and has_telegram_thumbnail(msg):
                    asyncio.create_task(try_store_real_thumbnail(normalized_title, clean_title, msg))
                return poster['poster_url'], source
        except Exception:
            pass
    if not is_post and msg and clean_title:
        asyncio.create_task(try_store_real_thumbnail(normalized_title, clean_title, msg))
    return FALLBACK_THUMBNAIL_URL, 'fallback'

# ============================================================================
# ✅ GET HOME MOVIES - FIXED VERSION
# ============================================================================

@performance_monitor.measure("get_home_movies")
@async_cache_with_ttl(maxsize=10, ttl=3600)
async def get_home_movies(limit: int = 100) -> Dict[str, Any]:
    """Get 100 movies for home page - FIXED"""
    start_time = time.time()
    logger.info(f"🏠 Fetching {limit} movies for home page...")
    
    results_dict = {}
    
    # ========== STEP 1: Get files from FILE CHANNEL ==========
    if user_session_ready and User is not None:
        try:
            file_count = 0
            
            # ✅ FIXED: Use search_messages with filter
            async for msg in User.search_messages(
                Config.FILE_CHANNEL_ID,
                limit=min(limit * 3, 300),  # Get more to ensure enough unique movies
                filter='video'
            ):
                if not msg or not msg.video:
                    continue
                
                file_name = msg.video.file_name or "video.mp4"
                if not is_video_file(file_name):
                    continue
                
                clean_title = extract_clean_title(file_name)
                normalized = normalize_title(clean_title)
                quality = detect_quality_enhanced(file_name)
                year = extract_year(file_name)
                
                if normalized in results_dict:
                    if quality not in results_dict[normalized]['qualities']:
                        results_dict[normalized]['qualities'][quality] = {
                            'quality': quality,
                            'file_name': file_name,
                            'file_size': msg.video.file_size,
                            'file_size_formatted': format_size(msg.video.file_size),
                            'message_id': msg.id,
                            'channel_id': Config.FILE_CHANNEL_ID,
                            'file_id': msg.video.file_id,
                            'date': msg.date
                        }
                        results_dict[normalized]['available_qualities'].append(quality)
                    continue
                
                results_dict[normalized] = {
                    'title': clean_title,
                    'normalized_title': normalized,
                    'year': year,
                    'qualities': {
                        quality: {
                            'quality': quality,
                            'file_name': file_name,
                            'file_size': msg.video.file_size,
                            'file_size_formatted': format_size(msg.video.file_size),
                            'message_id': msg.id,
                            'channel_id': Config.FILE_CHANNEL_ID,
                            'file_id': msg.video.file_id,
                            'date': msg.date
                        }
                    },
                    'available_qualities': [quality],
                    'has_file': True,
                    'has_post': False,
                    'result_type': 'file',
                    'best_date': msg.date,
                    'is_new': is_new(msg.date),
                    'thumbnail_url': None,
                    'thumbnail_source': None,
                    'has_thumbnail': False,
                    'first_file_msg': msg
                }
                file_count += 1
                
                if len(results_dict) >= limit:
                    break
            
            logger.info(f"📁 Found {file_count} files from file channel")
            
        except Exception as e:
            logger.error(f"❌ File channel fetch error: {e}")
            import traceback
            traceback.print_exc()
    
    # ========== STEP 2: Get posts from TEXT CHANNELS ==========
    if user_session_ready and User is not None and len(results_dict) < limit:
        try:
            post_count = 0
            
            for channel_id in Config.TEXT_CHANNEL_IDS:
                try:
                    async for msg in User.search_messages(
                        channel_id,
                        limit=50
                    ):
                        if not msg or not msg.text or len(msg.text) < 15:
                            continue
                        
                        title = extract_title_smart(msg.text)
                        if not title:
                            continue
                        
                        normalized = normalize_title(title)
                        
                        if normalized in results_dict:
                            results_dict[normalized]['has_post'] = True
                            results_dict[normalized]['post_content'] = format_post(msg.text, max_length=500)
                            results_dict[normalized]['post_channel_id'] = channel_id
                            results_dict[normalized]['post_message_id'] = msg.id
                            results_dict[normalized]['result_type'] = 'file_and_post'
                            continue
                        
                        clean_title = re.sub(r'\s*\(\d{4}\)', '', title)
                        clean_title = re.sub(r'\s+\d{4}$', '', clean_title).strip()
                        
                        year_match = re.search(r'\b(19|20)\d{2}\b', title)
                        year = year_match.group() if year_match else ""
                        
                        results_dict[normalized] = {
                            'title': clean_title,
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
                            'qualities': {},
                            'available_qualities': [],
                            'thumbnail_url': None,
                            'thumbnail_source': None,
                            'has_thumbnail': False,
                            'first_file_msg': None
                        }
                        post_count += 1
                        
                        if len(results_dict) >= limit:
                            break
                            
                except Exception as e:
                    logger.debug(f"Text channel fetch error in {channel_id}: {e}")
                    continue
                
                if len(results_dict) >= limit:
                    break
            
            logger.info(f"📝 Found {post_count} posts from text channels")
            
        except Exception as e:
            logger.error(f"❌ Text channels fetch error: {e}")
    
    # ========== STEP 3: Get Thumbnails ==========
    logger.info(f"🖼️ Fetching thumbnails for {len(results_dict)} movies...")
    
    items = list(results_dict.items())
    batch_size = 10
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        tasks = []
        
        for normalized, result in batch:
            if result.get('has_file'):
                msg = result.get('first_file_msg')
                task = get_best_thumbnail(
                    normalized,
                    result.get('title'),
                    result.get('year'),
                    msg,
                    is_post=False
                )
            else:
                task = get_best_thumbnail(
                    normalized,
                    result.get('title'),
                    result.get('year'),
                    None,
                    is_post=True
                )
            tasks.append((normalized, task))
        
        for normalized, task in tasks:
            try:
                thumbnail_url, thumbnail_source = await task
                results_dict[normalized]['thumbnail_url'] = thumbnail_url
                results_dict[normalized]['thumbnail_source'] = thumbnail_source
                results_dict[normalized]['has_thumbnail'] = True
            except Exception as e:
                logger.error(f"❌ Thumbnail error for {normalized}: {e}")
                results_dict[normalized]['thumbnail_url'] = FALLBACK_THUMBNAIL_URL
                results_dict[normalized]['thumbnail_source'] = 'fallback'
                results_dict[normalized]['has_thumbnail'] = False
        
        for normalized, result in batch:
            if 'first_file_msg' in result:
                del result['first_file_msg']
    
    # ========== STEP 4: Sort and return ==========
    all_results = list(results_dict.values())
    
    all_results.sort(key=lambda x: (
        1 if x.get('has_file') else 0,
        1 if x.get('is_new') else 0,
        x.get('best_date') if isinstance(x.get('best_date'), datetime) else 
        (x.get('date') if isinstance(x.get('date'), datetime) else datetime.min)
    ), reverse=True)
    
    all_results = all_results[:limit]
    
    source_counts = {}
    file_count = 0
    post_count = 0
    
    for r in all_results:
        if r.get('has_file'):
            file_count += 1
        if r.get('has_post'):
            post_count += 1
        source = r.get('thumbnail_source', 'unknown')
        source_counts[source] = source_counts.get(source, 0) + 1
    
    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info("🏠 HOME MOVIES SUMMARY:")
    logger.info(f"   • Total movies: {len(all_results)}")
    logger.info(f"   • Files: {file_count}")
    logger.info(f"   • Posts: {post_count}")
    logger.info(f"   • Thumbnail sources: {source_counts}")
    logger.info(f"   • Time: {elapsed:.2f}s")
    logger.info("=" * 70)
    
    return {
        'success': True,
        'movies': all_results,
        'total': len(all_results),
        'metadata': {
            'file_count': file_count,
            'post_count': post_count,
            'thumbnail_sources': source_counts,
            'fetched_at': datetime.now().isoformat(),
            'fetch_time_seconds': elapsed
        }
    }

# ============================================================================
# ✅ SEARCH FUNCTION
# ============================================================================

@performance_monitor.measure("optimized_search")
@async_cache_with_ttl(maxsize=500, ttl=600)
async def search_movies_optimized(query, limit=15, page=1):
    start_time = time.time()
    offset = (page - 1) * limit
    logger.info(f"🔍 OPTIMIZED SEARCH for: '{query}'")
    results_dict = {}
    
    if user_session_ready and User is not None:
        try:
            file_count = 0
            async for msg in User.search_messages(
                Config.FILE_CHANNEL_ID,
                query=query,
                limit=50
            ):
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
                quality_entry = {
                    'quality': quality,
                    'file_name': file_name,
                    'file_size': msg.document.file_size if msg.document else msg.video.file_size,
                    'file_size_formatted': format_size(msg.document.file_size if msg.document else msg.video.file_size),
                    'message_id': msg.id,
                    'channel_id': Config.FILE_CHANNEL_ID,
                    'file_id': msg.document.file_id if msg.document else msg.video.file_id,
                    'date': msg.date,
                    'has_thumbnail_in_telegram': has_telegram_thumbnail(msg)
                }
                if normalized not in results_dict:
                    results_dict[normalized] = {
                        'title': clean_title,
                        'normalized_title': normalized,
                        'year': year,
                        'qualities': {},
                        'available_qualities': [],
                        'has_file': True,
                        'has_post': False,
                        'result_type': 'file',
                        'best_date': msg.date,
                        'is_new': is_new(msg.date),
                        'thumbnail_url': None,
                        'thumbnail_source': None,
                        'has_thumbnail': False,
                        'first_file_msg': msg,
                    }
                    results_dict[normalized]['qualities'][quality] = quality_entry
                    results_dict[normalized]['available_qualities'].append(quality)
                else:
                    if quality not in results_dict[normalized]['qualities']:
                        results_dict[normalized]['qualities'][quality] = quality_entry
                        results_dict[normalized]['available_qualities'].append(quality)
                    if msg.date and msg.date > results_dict[normalized]['best_date']:
                        results_dict[normalized]['best_date'] = msg.date
                        results_dict[normalized]['is_new'] = is_new(msg.date)
                    if not results_dict[normalized].get('first_file_msg'):
                        results_dict[normalized]['first_file_msg'] = msg
                file_count += 1
            logger.info(f"📁 Found {file_count} file results from channel {Config.FILE_CHANNEL_ID}")
        except Exception as e:
            logger.error(f"❌ File channel search error: {e}")
    
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
                        else:
                            results_dict[normalized] = {
                                'title': clean_title,
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
                                'qualities': {},
                                'available_qualities': [],
                                'thumbnail_url': None,
                                'thumbnail_source': None,
                                'has_thumbnail': False,
                                'first_file_msg': None
                            }
                        post_count += 1
                except Exception as e:
                    logger.debug(f"Text search error in {channel_id}: {e}")
                    continue
            logger.info(f"📝 Found {post_count} post results from text channels")
        except Exception as e:
            logger.error(f"❌ Text channels search error: {e}")
    
    for normalized, result in results_dict.items():
        if result.get('has_file'):
            msg = result.get('first_file_msg')
            thumbnail_url, thumbnail_source = await get_best_thumbnail(
                normalized,
                result.get('title'),
                result.get('year'),
                msg,
                is_post=False
            )
        else:
            thumbnail_url, thumbnail_source = await get_best_thumbnail(
                normalized,
                result.get('title'),
                result.get('year'),
                None,
                is_post=True
            )
        result['thumbnail_url'] = thumbnail_url
        result['thumbnail_source'] = thumbnail_source
        result['has_thumbnail'] = True
        if 'first_file_msg' in result:
            del result['first_file_msg']
    
    all_results = list(results_dict.values())
    all_results.sort(key=lambda x: (
        1 if x.get('has_file') else 0,
        1 if x.get('is_new') else 0,
        x.get('best_date') if isinstance(x.get('best_date'), datetime) else 
        (x.get('date') if isinstance(x.get('date'), datetime) else datetime.min)
    ), reverse=True)
    
    total = len(all_results)
    start_idx = offset
    end_idx = offset + limit
    paginated = all_results[start_idx:end_idx]
    
    file_count = sum(1 for r in all_results if r.get('has_file'))
    post_count = sum(1 for r in all_results if r.get('has_post'))
    combined_count = sum(1 for r in all_results if r.get('has_file') and r.get('has_post'))
    
    source_counts = {}
    for r in paginated:
        source = r.get('thumbnail_source', 'unknown')
        source_counts[source] = source_counts.get(source, 0) + 1
    
    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info("📊 SEARCH RESULTS SUMMARY:")
    logger.info(f"   • Query: '{query}'")
    logger.info(f"   • Total movies: {total}")
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
            'total_movies': total,
            'file_results': file_count,
            'post_results': post_count,
            'combined_results': combined_count,
            'thumbnail_sources': source_counts,
            'mode': 'optimized_v4.3_with_all_sources',
            'poster_fetching': Config.POSTER_FETCHING_ENABLED
        },
        'bot_username': Config.BOT_USERNAME
    }

# ============================================================================
# ✅ THUMBNAIL MANAGER
# ============================================================================

class ThumbnailManager:
    def __init__(self, mongodb=None, bot_client=None, user_client=None, file_channel_id=None):
        self.mongodb = mongodb
        self.bot_client = bot_client
        self.user_client = user_client
        self.file_channel_id = file_channel_id or Config.FILE_CHANNEL_ID
        self.db = None
        self.thumbnails_col = None
        self.initialized = False
        self.is_extracting = False
        self.stats = {
            'total_movies_with_thumbnails': 0,
            'total_movies_without_thumbnails': 0,
            'total_failed': 0,
            'total_size_kb': 0,
            'avg_size_kb': 0
        }
        logger.info("🖼️ Thumbnail Manager v12 initialized")

    async def initialize(self):
        try:
            if not self.mongodb:
                logger.error("❌ MongoDB client not provided")
                return False
            self.db = self.mongodb.sk4film
            self.thumbnails_col = self.db.thumbnails
            await self._create_indexes_safely()
            self.initialized = True
            logger.info("✅ Thumbnail Manager initialized successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Thumbnail Manager initialization failed: {e}")
            return False

    async def _create_indexes_safely(self):
        try:
            existing = await self.thumbnails_col.index_information()
            indexes = {
                "title_idx": [("normalized_title", 1)],
                "has_thumbnail_idx": [("has_thumbnail", 1)],
                "extracted_at_idx": [("extracted_at", -1)],
                "channel_msg_idx": [("channel_id", 1), ("message_id", 1)]
            }
            for name, keys in indexes.items():
                if name not in existing:
                    try:
                        await self.thumbnails_col.create_index(keys, name=name, background=True)
                    except Exception as e:
                        logger.warning(f"⚠️ Could not create index {name}: {e}")
            logger.info("✅ Index management completed")
        except Exception as e:
            logger.error(f"❌ Index management error: {e}")

    async def get_thumbnail(self, normalized_title: str) -> Optional[str]:
        try:
            if not self.thumbnails_col:
                return None
            doc = await self.thumbnails_col.find_one({
                'normalized_title': normalized_title,
                'has_thumbnail': True
            })
            if doc and doc.get('thumbnail_url'):
                return doc['thumbnail_url']
            return None
        except Exception as e:
            logger.error(f"❌ Get thumbnail error: {e}")
            return None

    async def shutdown(self):
        logger.info("🖼️ Thumbnail Manager shutdown complete")

thumbnail_manager = ThumbnailManager()

# ============================================================================
# ✅ TELEGRAM SESSION INITIALIZATION
# ============================================================================

async def initialize_telegram_sessions():
    global User, Bot, user_session_ready, bot_session_ready, bot_handler, poster_fetcher
    
    logger.info("🔐 Initializing Telegram sessions...")
    
    if Config.USER_SESSION_STRING and Config.API_ID and Config.API_HASH:
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
            user_info = await User.get_me()
            user_session_ready = True
            logger.info(f"✅ User Session Ready: @{user_info.username or user_info.first_name}")
            if poster_fetcher:
                poster_fetcher.set_telegram_clients(user_client=User, bot_client=Bot)
        except Exception as e:
            logger.error(f"❌ User session error: {e}")
            User = None
            user_session_ready = False
    else:
        logger.warning("⚠️ No user session string provided")
    
    if Config.BOT_TOKEN and Config.API_ID and Config.API_HASH:
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
            bot_session_ready = True
            logger.info(f"✅ Bot Session Ready: @{bot_info.username}")
            bot_handler.bot = Bot
            bot_handler.initialized = True
            bot_handler.bot_username = bot_info.username
            if poster_fetcher:
                poster_fetcher.set_telegram_clients(user_client=User, bot_client=Bot)
        except Exception as e:
            logger.error(f"❌ Bot session error: {e}")
            Bot = None
            bot_session_ready = False
    else:
        logger.warning("⚠️ No bot token provided")
    
    global thumbnail_manager
    if mongo_client:
        thumbnail_manager.mongodb = mongo_client
        thumbnail_manager.bot_client = Bot
        thumbnail_manager.user_client = User
        await thumbnail_manager.initialize()
    
    logger.info("✅ Telegram sessions initialized")

# ============================================================================
# ✅ AUTO INDEXER - FIXED
# ============================================================================

class AutoIndexer:
    def __init__(self):
        self.is_indexing = False
        self.last_index_time = None
        self.indexing_task = None
        self.stats = {
            'total_files_indexed': 0,
            'total_files_failed': 0,
            'last_index_duration': 0,
            'last_index_count': 0
        }
    
    async def index_file_channel(self, limit: int = None):
        global is_indexing, last_index_time
        
        if self.is_indexing:
            logger.warning("⚠️ Indexing already in progress")
            return None
        
        self.is_indexing = True
        is_indexing = True
        start_time = time.time()
        
        try:
            if User is None or not user_session_ready:
                logger.error("❌ User session not ready for indexing")
                return None
            
            limit = limit or Config.MAX_INDEX_LIMIT
            logger.info(f"🔄 Starting indexing of file channel {Config.FILE_CHANNEL_ID}...")
            
            indexed_count = 0
            failed_count = 0
            
            # ✅ FIXED: Use search_messages with filter
            async for msg in User.search_messages(
                Config.FILE_CHANNEL_ID,
                limit=limit,
                filter='video'
            ):
                try:
                    if not msg or not msg.video:
                        continue
                    
                    file_name = msg.video.file_name or "video.mp4"
                    if not is_video_file(file_name):
                        continue
                    
                    clean_title = extract_clean_title(file_name)
                    normalized = normalize_title(clean_title)
                    year = extract_year(file_name)
                    quality = detect_quality_enhanced(file_name)
                    
                    existing = await files_col.find_one({
                        'normalized_title': normalized,
                        'channel_id': Config.FILE_CHANNEL_ID,
                        'message_id': msg.id
                    })
                    
                    if existing:
                        continue
                    
                    await files_col.insert_one({
                        'title': clean_title,
                        'normalized_title': normalized,
                        'year': year,
                        'quality': quality,
                        'file_name': file_name,
                        'file_size': msg.video.file_size,
                        'file_size_formatted': format_size(msg.video.file_size),
                        'message_id': msg.id,
                        'channel_id': Config.FILE_CHANNEL_ID,
                        'file_id': msg.video.file_id,
                        'date': msg.date,
                        'indexed_at': datetime.now(),
                        'has_thumbnail': has_telegram_thumbnail(msg)
                    })
                    
                    if Config.THUMBNAIL_EXTRACTION_ENABLED:
                        await try_store_real_thumbnail(normalized, clean_title, msg)
                    
                    indexed_count += 1
                    
                    if indexed_count % 10 == 0:
                        logger.info(f"📊 Indexed {indexed_count} files...")
                    
                except Exception as e:
                    logger.error(f"❌ Error indexing message {msg.id}: {e}")
                    failed_count += 1
            
            duration = time.time() - start_time
            self.last_index_time = datetime.now()
            last_index_time = self.last_index_time
            
            self.stats.update({
                'total_files_indexed': self.stats['total_files_indexed'] + indexed_count,
                'last_index_duration': duration,
                'last_index_count': indexed_count,
                'total_files_failed': self.stats['total_files_failed'] + failed_count
            })
            
            logger.info(f"✅ Indexing complete: {indexed_count} files, {failed_count} failed in {duration:.2f}s")
            return {
                'indexed': indexed_count,
                'failed': failed_count,
                'duration': duration
            }
            
        except Exception as e:
            logger.error(f"❌ Indexing error: {e}")
            import traceback
            traceback.print_exc()
            return None
            
        finally:
            self.is_indexing = False
            is_indexing = False
    
    async def start_auto_indexing(self):
        async def indexing_loop():
            while True:
                try:
                    await asyncio.sleep(Config.AUTO_INDEX_INTERVAL)
                    if not self.is_indexing:
                        await self.index_file_channel(limit=Config.BATCH_INDEX_SIZE)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"❌ Auto-indexing error: {e}")
                    await asyncio.sleep(60)
        
        self.indexing_task = asyncio.create_task(indexing_loop())
        logger.info("🔄 Auto-indexing started")
        return self.indexing_task
    
    async def stop(self):
        if self.indexing_task:
            self.indexing_task.cancel()
            try:
                await self.indexing_task
            except asyncio.CancelledError:
                pass
            self.indexing_task = None
        logger.info("🔄 Auto-indexing stopped")

auto_indexer = AutoIndexer()

# ============================================================================
# ✅ QUART ROUTES
# ============================================================================

@app.route('/')
async def home():
    return jsonify({
        'status': 'online',
        'version': '9.5-FIXED-BOT',
        'name': 'SK4FiLM API',
        'features': [
            '🏠 /api/movies - 100 Movies for Home Page',
            '🔍 /search - Search Movies',
            '📤 /files/send - Send Files to Users',
            '🖼️ All Poster Sources',
            '📁 File Sending with Auto-Delete',
            '🔄 Auto Indexing'
        ],
        'endpoints': [
            '/api/movies - Get 100 movies for home page',
            '/search?q=movie&page=1 - Search movies',
            '/files/send/<channel_id>/<message_id> - Send file to user',
            '/status - System status',
            '/index - Trigger indexing',
            '/health - Health check'
        ]
    })

@app.route('/api/movies', methods=['GET'])
async def api_movies():
    try:
        limit = request.args.get('limit', 100, type=int)
        limit = min(limit, 200)
        logger.info(f"🏠 /api/movies called with limit={limit}")
        
        result = await get_home_movies(limit=limit)
        
        if result.get('success'):
            return jsonify({
                'success': True,
                'movies': result['movies'],
                'total': result['total'],
                'metadata': result['metadata'],
                'version': '9.5-FIXED-BOT'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to fetch movies',
                'message': 'Please try again later'
            }), 500
            
    except Exception as e:
        logger.error(f"❌ /api/movies error: {e}")
        return jsonify({
            'success': False,
            'error': 'Server error',
            'message': str(e)
        }), 500

@app.route('/health', methods=['GET'])
async def health_check():
    bot_status = await bot_handler.get_bot_status() if bot_handler else {'initialized': False}
    poster_stats = poster_fetcher.get_stats() if poster_fetcher else {}
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'database': mongo_client is not None,
        'cache': cache_manager is not None,
        'user_session': user_session_ready,
        'bot_session': bot_session_ready,
        'bot_handler': bot_handler.initialized if bot_handler else False,
        'poster_fetcher': poster_fetcher is not None,
        'poster_stats': poster_stats,
        'auto_indexer': auto_indexer.is_indexing,
        'last_index_time': auto_indexer.last_index_time.isoformat() if auto_indexer.last_index_time else None,
        'thumbnail_manager': thumbnail_manager.initialized if thumbnail_manager else False
    })

@app.route('/status', methods=['GET'])
async def status():
    db_stats = {}
    if files_col:
        db_stats['total_files'] = await files_col.count_documents({})
        db_stats['posts'] = await files_col.count_documents({'is_post': True})
        db_stats['videos'] = await files_col.count_documents({'is_post': {'$ne': True}})
    if thumbnails_col:
        db_stats['thumbnails'] = await thumbnails_col.count_documents({'has_thumbnail': True})
    
    bot_status = await bot_handler.get_bot_status() if bot_handler else {'initialized': False}
    auto_delete_stats = await bot_handler.get_auto_delete_stats() if bot_handler else {}
    indexer_stats = auto_indexer.stats if auto_indexer else {}
    poster_stats = poster_fetcher.get_stats() if poster_fetcher else {}
    
    return jsonify({
        'status': 'online',
        'version': '9.5-FIXED-BOT',
        'config': {
            'main_channel': Config.MAIN_CHANNEL_ID,
            'file_channel': Config.FILE_CHANNEL_ID,
            'text_channels': Config.TEXT_CHANNEL_IDS,
            'auto_delete_minutes': Config.AUTO_DELETE_TIME,
            'poster_fetching': Config.POSTER_FETCHING_ENABLED
        },
        'database': db_stats,
        'sessions': {'user': user_session_ready, 'bot': bot_session_ready},
        'bot': bot_status,
        'auto_delete': auto_delete_stats,
        'indexer': indexer_stats,
        'poster_fetcher': poster_stats,
        'thumbnail_manager': {
            'initialized': thumbnail_manager.initialized if thumbnail_manager else False,
            'stats': thumbnail_manager.stats if thumbnail_manager and thumbnail_manager.initialized else {}
        }
    })

@app.route('/search', methods=['GET'])
async def search():
    query = request.args.get('q', '')
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 12, type=int)
    
    if not query or len(query) < Config.SEARCH_MIN_QUERY_LENGTH:
        return jsonify({
            'error': 'Query too short',
            'message': f'Please provide at least {Config.SEARCH_MIN_QUERY_LENGTH} characters'
        }), 400
    
    try:
        result = await search_movies_optimized(query, limit=min(limit, 50), page=page)
        return jsonify(result)
    except Exception as e:
        logger.error(f"❌ Search error: {e}")
        return jsonify({'error': 'Search failed', 'message': str(e)}), 500

@app.route('/files/send/<int:channel_id>/<int:message_id>', methods=['GET', 'POST'])
async def send_file_to_user(channel_id: int, message_id: int):
    try:
        if request.method == 'POST':
            data = await request.get_json()
            user_id = data.get('user_id')
            quality = data.get('quality', '480p')
        else:
            user_id = request.args.get('user_id', type=int)
            quality = request.args.get('quality', '480p')
        
        if not user_id:
            return jsonify({'error': 'User ID required'}), 400
        
        valid_channels = [Config.FILE_CHANNEL_ID] + Config.TEXT_CHANNEL_IDS + [Config.MAIN_CHANNEL_ID]
        if channel_id not in valid_channels:
            return jsonify({'error': 'Invalid channel'}), 400
        
        success, result, file_size = await bot_handler.send_file_to_user(
            user_id, channel_id, message_id, quality
        )
        
        if success:
            return jsonify({
                'success': True,
                'user_id': user_id,
                'channel_id': channel_id,
                'message_id': message_id,
                'quality': quality,
                'file_size': file_size,
                'file_size_formatted': format_size(file_size),
                'auto_delete_minutes': Config.AUTO_DELETE_TIME
            })
        else:
            return jsonify({'error': 'Send failed', 'message': result.get('message', 'Unknown error')}), 500
    except Exception as e:
        logger.error(f"❌ Send file error: {e}")
        return jsonify({'error': 'Server error', 'message': str(e)}), 500

@app.route('/index', methods=['POST'])
async def index_files():
    try:
        if auto_indexer.is_indexing:
            return jsonify({'status': 'busy', 'message': 'Indexing already in progress'}), 409
        limit = request.args.get('limit', Config.BATCH_INDEX_SIZE, type=int)
        result = await auto_indexer.index_file_channel(limit=limit)
        if result:
            return jsonify({
                'status': 'success',
                'indexed': result.get('indexed', 0),
                'failed': result.get('failed', 0),
                'duration_seconds': result.get('duration', 0)
            })
        else:
            return jsonify({'error': 'Indexing failed'}), 500
    except Exception as e:
        logger.error(f"❌ Index error: {e}")
        return jsonify({'error': 'Server error', 'message': str(e)}), 500

@app.route('/bot/status', methods=['GET'])
async def bot_status():
    status = await bot_handler.get_bot_status()
    return jsonify(status)

@app.route('/bot/auto-delete/stats', methods=['GET'])
async def auto_delete_stats():
    stats = await bot_handler.get_auto_delete_stats()
    return jsonify(stats)

@app.errorhandler(404)
async def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
async def server_error(error):
    return jsonify({'error': 'Server error'}), 500

# ============================================================================
# ✅ APPLICATION STARTUP & SHUTDOWN
# ============================================================================

@app.before_serving
async def startup():
    global mongo_client, db, files_col, verification_col, thumbnails_col, posters_col
    global cache_manager, verification_system, premium_system, poster_fetcher
    
    logger.info("=" * 60)
    logger.info("🚀 SK4FiLM v9.5 STARTING...")
    logger.info("📌 Poster Sources: MongoDB → TMDB → OMDB → Letterboxd → IMDB → JustWatch → IMPAwards → TELEGRAM → Fallback")
    logger.info("📌 /api/movies - 100 Movies for Home Page")
    logger.info("=" * 60)
    
    try:
        mongo_client = AsyncIOMotorClient(Config.MONGODB_URI)
        db = mongo_client.sk4film
        files_col = db.files
        verification_col = db.verification
        thumbnails_col = db.thumbnails
        posters_col = db.posters
        
        await files_col.create_index('normalized_title')
        await files_col.create_index('channel_id')
        await files_col.create_index('message_id')
        await files_col.create_index([('channel_id', 1), ('message_id', 1)])
        
        logger.info("✅ MongoDB connected")
    except Exception as e:
        logger.error(f"❌ MongoDB error: {e}")
    
    try:
        cache_manager = CacheManager(Config)
        await cache_manager.init_redis()
        await cache_manager.start_cleanup_task()
        logger.info("✅ Cache initialized")
    except Exception as e:
        logger.error(f"❌ Cache error: {e}")
        cache_manager = None
    
    if Config.POSTER_FETCHING_ENABLED and POSTER_FETCHER_AVAILABLE:
        try:
            redis_client = None
            if Config.REDIS_URL:
                try:
                    redis_client = redis.from_url(
                        Config.REDIS_URL,
                        password=Config.REDIS_PASSWORD if Config.REDIS_PASSWORD else None,
                        decode_responses=True
                    )
                    await redis_client.ping()
                    logger.info("✅ Redis connected for poster fetcher")
                except Exception as e:
                    logger.warning(f"⚠️ Redis not available: {e}")
            
            poster_fetcher = await create_poster_fetcher(
                config=Config,
                mongo_client=mongo_client,
                redis_client=redis_client
            )
            logger.info("✅ Poster fetcher initialized with all sources")
        except Exception as e:
            logger.error(f"❌ Poster fetcher error: {e}")
            poster_fetcher = None
    else:
        poster_fetcher = None
    
    if VerificationSystem:
        try:
            verification_system = VerificationSystem(Config, mongo_client)
            logger.info("✅ Verification system initialized")
        except Exception as e:
            logger.error(f"❌ Verification system error: {e}")
            verification_system = None
    
    if PremiumSystem:
        try:
            premium_system = PremiumSystem(Config, mongo_client)
            logger.info("✅ Premium system initialized")
        except Exception as e:
            logger.error(f"❌ Premium system error: {e}")
            premium_system = None
    
    await initialize_telegram_sessions()
    
    if mongo_client and user_session_ready:
        await auto_indexer.start_auto_indexing()
        logger.info("✅ Auto-indexing started")
        if Config.INSTANT_AUTO_INDEX and not auto_indexer.last_index_time:
            asyncio.create_task(auto_indexer.index_file_channel(limit=100))
    
    logger.info("=" * 60)
    logger.info("✅ SK4FiLM v9.5 STARTUP COMPLETE")
    logger.info("   🔍 All poster sources ready!")
    logger.info("   🏠 /api/movies ready with 100 movies!")
    logger.info("=" * 60)

@app.after_serving
async def shutdown():
    logger.info("🛑 Shutting down SK4FiLM v9.5...")
    if auto_indexer:
        await auto_indexer.stop()
    if bot_handler:
        await bot_handler.shutdown()
    if cache_manager:
        await cache_manager.stop()
    if verification_system:
        await verification_system.stop()
    if premium_system and hasattr(premium_system, 'stop_cleanup_task'):
        await premium_system.stop_cleanup_task()
    if thumbnail_manager:
        await thumbnail_manager.shutdown()
    if poster_fetcher:
        await poster_fetcher.close()
    if User and user_session_ready:
        try:
            await User.stop()
        except:
            pass
    if Bot and bot_session_ready:
        try:
            await Bot.stop()
        except:
            pass
    if mongo_client:
        try:
            mongo_client.close()
        except:
            pass
    logger.info("✅ SK4FiLM v9.5 shutdown complete")

# ============================================================================
# ✅ MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("🚀 SK4FiLM v9.5 - COMPLETE FIXED BOT")
    print("📅 Version: 9.5-FIXED-BOT")
    print("🏠 NEW: /api/movies - 100 Movies for Home Page")
    print("🎬 Poster Sources:")
    print("   1️⃣ MongoDB Cache")
    print("   2️⃣ TMDB")
    print("   3️⃣ OMDB")
    print("   4️⃣ Letterboxd")
    print("   5️⃣ IMDB")
    print("   6️⃣ JustWatch")
    print("   7️⃣ IMPAwards")
    print("   8️⃣ TELEGRAM")
    print("   9️⃣ Fallback")
    print("=" * 60)
    
    config = HyperConfig()
    config.bind = [f"0.0.0.0:{Config.WEB_SERVER_PORT}"]
    config.use_reloader = False
    config.worker_class = "asyncio"
    config.accesslog = "-"
    config.errorlog = "-"
    config.loglevel = "info"
    
    try:
        asyncio.run(serve(app, config))
    except KeyboardInterrupt:
        logger.info("🛑 Shutdown by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        sys.exit(1)
