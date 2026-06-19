# ============================================================================
# 🚀 SK4FiLM v9.5 - COMPLETE FIXED BOT WITH /api/movies ENDPOINT
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

# Poster Fetcher - UPDATED WITH TELEGRAM SUPPORT
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
    # Dummy classes
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
# ✅ CONFIGURATION - v9.5 WITH FIXED IDs AND BOT
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
    
    # ✅ FIXED: Channel Configuration - Proper negative IDs
    MAIN_CHANNEL_ID = -1001767371495  # Main channel (posts)
    TEXT_CHANNEL_IDS = [-1001891090100, -1002024811395]  # Text channels for posts
    FILE_CHANNEL_ID = -1001768249569   # File channel for videos
    
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
    AUTO_DELETE_TIME = int(os.environ.get("AUTO_DELETE_TIME", "5"))  # 5 minutes default
    WEB_SERVER_PORT = int(os.environ.get("PORT", 8000))
    BACKEND_URL = os.environ.get("BACKEND_URL", "https://sk4film.koyeb.app")
    
    # API Keys for POSTERS
    TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "e547e17d4e91f3e62a571655cd1ccaff")
    OMDB_API_KEY = os.environ.get("OMDB_API_KEY", "8265bd1c")
    
    # 🔥 OPTIMIZATION SETTINGS
    POSTER_FETCHING_ENABLED = True
    POSTER_CACHE_TTL = 86400  # 24 hours
    POSTER_FETCH_TIMEOUT = 5  # 5 seconds
    MAX_CONCURRENT_REQUESTS = int(os.environ.get("MAX_CONCURRENT_REQUESTS", "20"))
    CACHE_TTL = int(os.environ.get("CACHE_TTL", "1200"))  # 10 minutes
    REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "5"))  # 5 seconds
    
    # Sync Management Settings
    MONITOR_INTERVAL = int(os.environ.get("MONITOR_INTERVAL", "600"))  # 10 minutes
    
    # Quality Settings
    QUALITY_PRIORITY = ['2160p', '1080p', '720p', '480p', '360p']
    HEVC_VARIANTS = ['720p HEVC', '1080p HEVC', '2160p HEVC']
    
    # 🔥 THUMBNAIL EXTRACTION SETTINGS
    THUMBNAIL_EXTRACTION_ENABLED = True
    THUMBNAIL_BATCH_SIZE = 3
    THUMBNAIL_EXTRACT_TIMEOUT = 5
    THUMBNAIL_CACHE_DURATION = 24 * 60 * 60
    THUMBNAIL_RETRY_LIMIT = 1
    THUMBNAIL_MAX_SIZE_KB = 200
    THUMBNAIL_TTL_DAYS = 365
    
    # 🔥 FILE CHANNEL INDEXING SETTINGS
    AUTO_INDEX_INTERVAL = 3600  # 1 hour
    BATCH_INDEX_SIZE = 50
    MAX_INDEX_LIMIT = 500
    INDEX_ALL_HISTORY = False
    INSTANT_AUTO_INDEX = True
    
    # 🔥 SEARCH SETTINGS - OPTIMIZED
    SEARCH_MIN_QUERY_LENGTH = 2
    SEARCH_RESULTS_PER_PAGE = 12
    SEARCH_CACHE_TTL = 1800  # 10 minutes cache
    
    # 🔥 HOME MOVIES SETTINGS
    HOME_MOVIES_LIMIT = 100  # 100 movies for home page
    HOME_MOVIES_CACHE_TTL = 3600  # 1 hour

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
    """
    🔥 Cache decorator with TTL (Time To Live)
    maxsize: Maximum number of items in cache
    ttl: Time to live in seconds
    """
    cache = {}
    cache_lock = asyncio.Lock()
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key
            key_parts = [func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
            key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
            
            now = time.time()
            
            # Check cache
            async with cache_lock:
                if key in cache:
                    value, timestamp = cache[key]
                    if now - timestamp < ttl:
                        return value
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Store in cache
            async with cache_lock:
                cache[key] = (result, now)
                # Limit cache size
                if len(cache) > maxsize:
                    # Remove oldest entry
                    oldest_key = min(cache.keys(), key=lambda k: cache[k][1])
                    del cache[oldest_key]
            
            return result
        return wrapper
    return decorator

# ============================================================================
# ✅ BOT HANDLER MODULE - WITH FILE SENDING
# ============================================================================

class BotHandler:
    """Bot handler for Telegram bot operations - ENHANCED with file sending"""
    
    def __init__(self, bot_token=None, api_id=None, api_hash=None):
        self.bot_token = bot_token or Config.BOT_TOKEN
        self.api_id = api_id or Config.API_ID
        self.api_hash = api_hash or Config.API_HASH
        self.bot = None
        self.initialized = False
        self.last_update = None
        self.bot_username = None
        
        # Auto-delete tracking
        self.auto_delete_tasks = {}
        self.auto_delete_messages = {}
        
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
            
            asyncio.create_task(self._periodic_tasks())
            asyncio.create_task(self._cleanup_old_auto_delete_tasks())
            return True
            
        except Exception as e:
            logger.error(f"❌ Bot handler initialization error: {e}")
            return False
    
    async def _periodic_tasks(self):
        """Run periodic tasks for bot"""
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
    
    # ============================================================================
    # ✅ FILE SENDING METHODS
    # ============================================================================
    
    async def send_file_to_user(self, user_id: int, channel_id: int, message_id: int, quality: str = "480p") -> Tuple[bool, Dict, int]:
        """
        🚀 Send file directly to user
        Returns: (success, result_data, file_size)
        """
        try:
            if not self.initialized:
                return False, {'message': 'Bot not initialized'}, 0
            
            logger.info(f"📤 Sending file to user {user_id}: channel={channel_id}, msg={message_id}, quality={quality}")
            
            # Get message from channel
            message = await self.bot.get_messages(channel_id, message_id)
            if not message:
                return False, {'message': 'File not found in channel'}, 0
            
            if not message.document and not message.video:
                return False, {'message': 'Not a downloadable file'}, 0
            
            # Prepare file info
            if message.document:
                file_name = message.document.file_name or "file.bin"
                file_size = message.document.file_size or 0
                file_id = message.document.file_id
                mime_type = message.document.mime_type or "application/octet-stream"
                is_video = False
            else:  # video
                file_name = message.video.file_name or "video.mp4"
                file_size = message.video.file_size or 0
                file_id = message.video.file_id
                mime_type = "video/mp4"
                is_video = True
            
            # Create caption
            caption = (
                f"📁 **File:** `{file_name}`\n"
                f"📦 **Size:** {format_size(file_size)}\n"
                f"📹 **Quality:** {quality}\n"
                f"⏰ **Auto-delete in:** {Config.AUTO_DELETE_TIME} minutes\n\n"
                f"🔗 **More movies:** {Config.WEBSITE_URL}\n"
                f"🎬 **@SK4FiLM**"
            )
            
            # Create buttons
            buttons = InlineKeyboardMarkup([
                [InlineKeyboardButton("🌐 VISIT WEBSITE", url=Config.WEBSITE_URL)],
                [InlineKeyboardButton("📢 JOIN CHANNEL", url=Config.MAIN_CHANNEL_LINK)],
                [InlineKeyboardButton("🔄 GET ANOTHER", callback_data="back_to_start")]
            ])
            
            # Send file
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
                
                # Schedule auto-delete
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
                # Retry after flood wait
                return await self.send_file_to_user(user_id, channel_id, message_id, quality)
                
            except BadRequest as e:
                if "FILE_REFERENCE_EXPIRED" in str(e):
                    # Try to get fresh message
                    fresh_msg = await self.bot.get_messages(channel_id, message_id)
                    if fresh_msg and (fresh_msg.document or fresh_msg.video):
                        # Update file_id
                        if fresh_msg.document:
                            file_id = fresh_msg.document.file_id
                        else:
                            file_id = fresh_msg.video.file_id
                        
                        # Retry with fresh file_id
                        if is_video:
                            sent = await self.bot.send_video(
                                user_id, file_id, caption=caption, reply_markup=buttons
                            )
                        else:
                            sent = await self.bot.send_document(
                                user_id, file_id, caption=caption, reply_markup=buttons
                            )
                        
                        logger.info(f"✅ File sent with refreshed reference to user {user_id}")
                        return True, {
                            'success': True,
                            'file_name': file_name,
                            'file_size': file_size,
                            'quality': quality,
                            'message_id': sent.id,
                            'refreshed': True
                        }, file_size
                raise e
                
        except Exception as e:
            logger.error(f"❌ Send file error: {e}")
            return False, {'message': f'Error: {str(e)}'}, 0
    
    async def _auto_delete_file(self, user_id: int, message_id: int, file_name: str, minutes: int):
        """Auto-delete file after specified minutes"""
        try:
            task_id = f"{user_id}_{message_id}"
            logger.info(f"⏰ Auto-delete scheduled for user {user_id}, message {message_id} in {minutes} minutes")
            
            await asyncio.sleep(minutes * 60)
            
            # Delete the message
            try:
                await self.bot.delete_messages(user_id, message_id)
                logger.info(f"🗑️ Auto-deleted file for user {user_id}: {file_name}")
                
                # Update status
                if task_id in self.auto_delete_messages:
                    self.auto_delete_messages[task_id]['status'] = 'completed'
                    self.auto_delete_messages[task_id]['completed_at'] = datetime.now()
                
                # Send notification
                await self.bot.send_message(
                    user_id,
                    f"🗑️ **File Auto-Deleted**\n\n"
                    f"✅ Security measure completed\n\n"
                    f"🌐 Visit website to download again: {Config.WEBSITE_URL}",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)]
                    ])
                )
                
            except MessageDeleteForbidden:
                logger.warning(f"❌ Cannot delete message {message_id}: Message delete forbidden")
            except BadRequest as e:
                if "MESSAGE_ID_INVALID" in str(e):
                    logger.info(f"Message {message_id} already deleted")
                else:
                    logger.error(f"Delete error: {e}")
                    
        except asyncio.CancelledError:
            logger.info(f"⏹️ Auto-delete cancelled for user {user_id}, message {message_id}")
            task_id = f"{user_id}_{message_id}"
            if task_id in self.auto_delete_messages:
                self.auto_delete_messages[task_id]['status'] = 'cancelled'
        except Exception as e:
            logger.error(f"❌ Auto-delete error: {e}")
        finally:
            # Clean up task
            task_id = f"{user_id}_{message_id}"
            if task_id in self.auto_delete_tasks:
                del self.auto_delete_tasks[task_id]
    
    async def cancel_auto_delete(self, user_id: int, message_id: int):
        """Cancel auto-delete for a specific file"""
        task_id = f"{user_id}_{message_id}"
        if task_id in self.auto_delete_tasks:
            task = self.auto_delete_tasks[task_id]
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            del self.auto_delete_tasks[task_id]
            if task_id in self.auto_delete_messages:
                self.auto_delete_messages[task_id]['status'] = 'cancelled_by_user'
            logger.info(f"✅ Auto-delete cancelled for user {user_id}, message {message_id}")
            return True
        return False
    
    async def _cleanup_old_auto_delete_tasks(self):
        """Clean up old auto-delete task data"""
        while self.initialized:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                now = datetime.now()
                to_remove = []
                
                for task_id, data in self.auto_delete_messages.items():
                    completed_at = data.get('completed_at')
                    if completed_at and (now - completed_at).total_seconds() > 24 * 3600:
                        to_remove.append(task_id)
                
                for task_id in to_remove:
                    self.auto_delete_messages.pop(task_id, None)
                
                if to_remove:
                    logger.info(f"🧹 Cleaned up {len(to_remove)} old auto-delete records")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-delete cleanup error: {e}")
    
    async def get_auto_delete_stats(self):
        """Get auto-delete statistics"""
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
    
    # ============================================================================
    # ✅ OTHER BOT METHODS
    # ============================================================================
    
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
    
    async def extract_thumbnail(self, channel_id, message_id):
        """Extract thumbnail from video file"""
        if not self.initialized:
            return None
        
        try:
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
    
    async def get_bot_status(self):
        """Get bot status information"""
        if not self.initialized:
            return {
                'initialized': False,
                'error': 'Bot not initialized'
            }
        
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
            logger.error(f"Error getting bot status: {e}")
            return {
                'initialized': False,
                'error': str(e)
            }
    
    async def shutdown(self):
        """Shutdown bot handler"""
        logger.info("Shutting down bot handler...")
        
        # Cancel all auto-delete tasks
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
                logger.info("✅ Bot stopped")
            except Exception as e:
                logger.error(f"❌ Error stopping bot: {e}")

bot_handler = BotHandler()

# ============================================================================
# ✅ ENHANCED TITLE EXTRACTION FUNCTIONS
# ============================================================================

def extract_clean_title(filename):
    """Extract clean movie title without quality tags, year, etc."""
    if not filename:
        return "Unknown"
    
    # Remove extension
    name = os.path.splitext(filename)[0]
    
    # Replace separators with spaces
    name = re.sub(r'[._\-]', ' ', name)
    
    # Remove ALL quality-related tags
    name = re.sub(r'\b(480p|720p|1080p|2160p|4k|hd|hevc|x264|x265|web-dl|webrip|bluray|hdtv|hdr|dts|ac3|aac|ddp|5\.1|7\.1|2\.0|esub|sub|multi|dual|audio|hindi|english|tamil|telugu|malayalam|kannada|ben|eng|hin|tam|tel|mal|kan)\b.*$', '', name, flags=re.IGNORECASE)
    
    # Remove year at the end
    name = re.sub(r'\s+(19|20)\d{2}\s*$', '', name)
    
    # Remove parentheses content
    name = re.sub(r'\s*\([^)]*\)', '', name)
    name = re.sub(r'\s*\[[^\]]*\]', '', name)
    
    # Clean up spaces
    name = re.sub(r'\s+', ' ', name)
    name = name.strip()
    
    return name if name else "Unknown"

def extract_year(filename):
    """Extract year from filename"""
    if not filename:
        return ""
    
    year_match = re.search(r'\b(19|20)\d{2}\b', filename)
    return year_match.group() if year_match else ""

def has_telegram_thumbnail(message):
    """Reliable Telegram thumbnail detection"""
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

    except Exception as e:
        logger.debug(f"Thumbnail check error: {e}")
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
    
    HEVC_PATTERNS = [
        r'\bhevc\b',
        r'\bx265\b',
        r'\bh\.?265\b',
    ]
    
    is_hevc = any(re.search(pattern, filename_lower) for pattern in HEVC_PATTERNS)
    
    for pattern, quality in QUALITY_PATTERNS:
        if re.search(pattern, filename_lower):
            if is_hevc and quality in ['720p', '1080p', '2160p']:
                return f"{quality} HEVC"
            return quality
    
    return "480p"

# ============================================================================
# ✅ FIXED try_store_real_thumbnail - CORRECT CHANNEL ID
# ============================================================================

async def try_store_real_thumbnail(normalized_title: str, clean_title: str, msg) -> None:
    """
    🎯 FIXED: Store ONE thumbnail per movie with correct channel ID
    """
    try:
        if not msg or thumbnails_col is None:
            return

        # Check if movie already has thumbnail
        existing = await thumbnails_col.find_one({
            'normalized_title': normalized_title,
            'has_thumbnail': True
        })
        
        if existing:
            logger.debug(f"📦 Movie already has thumbnail: {clean_title}")
            return

        # Get media
        media = msg.video or msg.document
        if not media:
            return

        file_name = getattr(media, "file_name", "video.mp4")
        if not is_video_file(file_name):
            return

        # Find thumbnail file_id
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
            await thumbnails_col.update_one(
                {'normalized_title': normalized_title},
                {'$set': {
                    'normalized_title': normalized_title,
                    'title': clean_title,
                    'has_thumbnail': False,
                    'checked_at': datetime.now()
                }},
                upsert=True
            )
            return

        # Download thumbnail
        client = User if user_session_ready else Bot
        if not client:
            return

        downloaded = await client.download_media(thumbnail_file_id, in_memory=True)
        if not downloaded:
            return

        # Handle both bytes and BytesIO
        if isinstance(downloaded, bytes):
            thumbnail_data = downloaded
        else:
            # BytesIO object - read directly
            downloaded.seek(0)
            thumbnail_data = downloaded.read()

        # Convert to base64
        thumbnail_url = f"data:image/jpeg;base64,{base64.b64encode(thumbnail_data).decode()}"
        size_kb = len(thumbnail_url) / 1024

        # ✅ FIXED: Store with correct channel ID
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
                'channel_id': Config.FILE_CHANNEL_ID,  # ✅ FIXED: Use Config.FILE_CHANNEL_ID
                'file_name': file_name,
                'size_kb': size_kb,
                'file_id': thumbnail_file_id
            }},
            upsert=True
        )

        logger.info(f"✅✅✅ Thumbnail stored: {clean_title} ({size_kb:.1f}KB) [Channel: {Config.FILE_CHANNEL_ID}, Msg: {msg.id}]")

    except Exception as e:
        logger.error(f"❌ Thumbnail error for {clean_title}: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# ✅ UPDATED get_best_thumbnail - WITH ALL POSTER SOURCES + TELEGRAM
# ============================================================================

async def get_best_thumbnail(normalized_title: str, clean_title: str = None, 
                            year: str = None, msg=None, is_post: bool = False) -> Tuple[str, str]:
    """
    🎯 UPDATED: Get thumbnail with ALL sources + TELEGRAM
    Priority: MongoDB > TMDB > OMDB > Letterboxd > IMDB > JustWatch > IMPAwards > TELEGRAM > Fallback
    """
    # ========== STEP 1: MongoDB Check (Fastest) ==========
    if thumbnails_col is not None:
        try:
            doc = await thumbnails_col.find_one({
                'normalized_title': normalized_title,
                'has_thumbnail': True,
                'thumbnail_url': {'$exists': True, '$ne': None}
            })
            
            if doc and doc.get('thumbnail_url'):
                logger.info(f"📦 MongoDB thumbnail FOUND for {clean_title} (Post: {is_post})")
                return doc['thumbnail_url'], 'mongodb'
                    
        except Exception as e:
            logger.debug(f"⚠️ MongoDB fetch error: {e}")
    
    # ========== STEP 2: Poster Fetcher (All Sources + Telegram) ==========
    if Config.POSTER_FETCHING_ENABLED and poster_fetcher and clean_title:
        try:
            # Use the enhanced poster fetcher with all sources
            poster = await poster_fetcher.fetch_poster(clean_title, year or "")
            
            if poster and poster.get('poster_url') and poster.get('found'):
                source = poster.get('source', 'unknown')
                logger.info(f"🎬 POSTER FOUND for {clean_title} from {source} (Post: {is_post})")
                
                # For file results, try to store real thumbnail if from Telegram
                if not is_post and msg and has_telegram_thumbnail(msg):
                    asyncio.create_task(
                        try_store_real_thumbnail(normalized_title, clean_title, msg)
                    )
                
                return poster['poster_url'], source
                
        except Exception as e:
            logger.debug(f"⚠️ Poster fetch error: {e}")
    
    # ========== STEP 3: For files only - try extraction ==========
    if not is_post and msg and clean_title:
        asyncio.create_task(
            try_store_real_thumbnail(normalized_title, clean_title, msg)
        )
    
    # ========== STEP 4: Fallback ==========
    logger.info(f"⚠️ FALLBACK for {clean_title} (Post: {is_post})")
    return FALLBACK_THUMBNAIL_URL, 'fallback'

# ============================================================================
# ✅ UPDATED POSTER FETCHING WRAPPER
# ============================================================================

async def get_poster_for_movie(title: str, year: str = "", quality: str = "") -> Dict[str, Any]:
    """Get poster for movie using enhanced fetcher"""
    global poster_fetcher
    
    if poster_fetcher is None:
        return {
            'poster_url': FALLBACK_THUMBNAIL_URL,
            'source': 'fallback',
            'rating': '0.0',
            'year': year,
            'title': title,
            'found': False
        }
    
    try:
        result = await poster_fetcher.fetch_poster(title, year, quality)
        return result
    except Exception as e:
        logger.error(f"❌ Poster fetch error: {e}")
        return {
            'poster_url': FALLBACK_THUMBNAIL_URL,
            'source': 'error',
            'rating': '0.0',
            'year': year,
            'title': title,
            'found': False
        }

# ============================================================================
# ✅ FIXED GET HOME MOVIES - CORRECT PYROGRAM METHODS
# ============================================================================

@performance_monitor.measure("get_home_movies")
@async_cache_with_ttl(maxsize=10, ttl=3600)  # Cache for 1 hour
async def get_home_movies(limit: int = 100) -> Dict[str, Any]:
    """
    🏠 GET HOME MOVIES - 100 MOVIES FOR HOME PAGE
    Fetches latest movies from file channel and text channels
    """
    start_time = time.time()
    logger.info(f"🏠 Fetching {limit} movies for home page...")
    
    results_dict = {}
    
    # ========== STEP 1: Get files from FILE CHANNEL ==========
    if user_session_ready and User is not None:
        try:
            file_count = 0
            
            # ✅ FIXED: Use get_messages with message_ids or use search_messages
            # Since we want latest messages, use search_messages with filter='video'
            async for msg in User.search_messages(
                Config.FILE_CHANNEL_ID,
                limit=min(limit * 2, 200),  # Get more to ensure we have enough
                filter='video'  # Only videos
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
                
                # Skip if already have this movie
                if normalized in results_dict:
                    # Add quality if not already present
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
                
                # New movie
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
                
                # Stop if we have enough movies
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
                    # Use search_messages to get latest posts
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
                        
                        # Skip if already have this movie
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
                        
                        # Stop if we have enough movies
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
    
    # ========== STEP 3: Get Thumbnails for Each Movie ==========
    logger.info(f"🖼️ Fetching thumbnails for {len(results_dict)} movies...")
    
    # Process in batches for efficiency
    items = list(results_dict.items())
    batch_size = 10
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        tasks = []
        
        for normalized, result in batch:
            is_post_only = result.get('has_file') == False and result.get('has_post') == True
            
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
        
        # Wait for batch
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
        
        # Clean up first_file_msg to avoid memory issues
        for normalized, result in batch:
            if 'first_file_msg' in result:
                del result['first_file_msg']
    
    # ========== STEP 4: Convert to List and Sort ==========
    all_results = list(results_dict.values())
    
    # Sort by: has_file (priority), is_new, best_date
    all_results.sort(key=lambda x: (
        1 if x.get('has_file') else 0,  # Files first
        1 if x.get('is_new') else 0,
        x.get('best_date') if isinstance(x.get('best_date'), datetime) else 
        (x.get('date') if isinstance(x.get('date'), datetime) else datetime.min)
    ), reverse=True)
    
    # Limit to requested number
    all_results = all_results[:limit]
    
    # Count thumbnail sources
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
# ✅ OPTIMIZED SEARCH v4.3 - WITH ALL POSTER SOURCES
# ============================================================================

@performance_monitor.measure("optimized_search")
@async_cache_with_ttl(maxsize=500, ttl=600)
async def search_movies_optimized(query, limit=15, page=1):
    """
    🔥 OPTIMIZED SEARCH v4.3 - All poster sources + Telegram
    """
    start_time = time.time()
    offset = (page - 1) * limit
    
    logger.info(f"🔍 OPTIMIZED SEARCH for: '{query}'")
    
    # Dictionary to store unique results
    results_dict = {}
    
    # ============================================================================
    # ✅ STEP 1: Direct Telegram FILE CHANNEL Search (Files)
    # ============================================================================
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
                        'all_messages': [msg]
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
                    
                    results_dict[normalized]['all_messages'].append(msg)
                
                file_count += 1
            
            logger.info(f"📁 Found {file_count} file results from channel {Config.FILE_CHANNEL_ID}")
            
        except Exception as e:
            logger.error(f"❌ File channel search error: {e}")
    
    # ============================================================================
    # ✅ STEP 2: Direct Telegram TEXT CHANNELS Search (Posts)
    # ============================================================================
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
    
    # ============================================================================
    # ✅ STEP 3: Get Thumbnails with ALL Sources
    # ============================================================================
    for normalized, result in results_dict.items():
        is_post_only = result.get('has_file') == False and result.get('has_post') == True
        
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
        
        # Clean up
        if 'first_file_msg' in result:
            del result['first_file_msg']
        if 'all_messages' in result:
            del result['all_messages']
    
    # ============================================================================
    # ✅ STEP 4: Convert to List and Sort
    # ============================================================================
    all_results = list(results_dict.values())
    
    all_results.sort(key=lambda x: (
        1 if x.get('has_file') else 0,
        1 if x.get('is_new') else 0,
        x.get('best_date') if isinstance(x.get('best_date'), datetime) else 
        (x.get('date') if isinstance(x.get('date'), datetime) else datetime.min)
    ), reverse=True)
    
    # ============================================================================
    # ✅ STEP 5: Pagination
    # ============================================================================
    total = len(all_results)
    start_idx = offset
    end_idx = offset + limit
    paginated = all_results[start_idx:end_idx]
    
    # Statistics
    file_count = sum(1 for r in all_results if r.get('has_file'))
    post_count = sum(1 for r in all_results if r.get('has_post'))
    combined_count = sum(1 for r in all_results if r.get('has_file') and r.get('has_post'))
    
    # Count thumbnail sources
    source_counts = {}
    for r in paginated:
        source = r.get('thumbnail_source', 'unknown')
        source_counts[source] = source_counts.get(source, 0) + 1
    
    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info("📊 SEARCH RESULTS SUMMARY:")
    logger.info(f"   • Query: '{query}'")
    logger.info(f"   • Total movies: {total}")
    logger.info(f"   • Post+Files: {combined_count}")
    logger.info(f"   • Post only: {post_count - combined_count}")
    logger.info(f"   • File only: {file_count - combined_count}")
    logger.info(f"   • Thumbnail sources: {source_counts}")
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
            'poster_fetching': Config.POSTER_FETCHING_ENABLED,
            'sources_used': list(source_counts.keys())
        },
        'bot_username': Config.BOT_USERNAME
    }

# ============================================================================
# ✅ THUMBNAIL MANAGER
# ============================================================================

class ThumbnailManager:
    """🖼️ THUMBNAIL MANAGER v12 - With all poster sources"""

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
        """Initialize database collections and indexes"""
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
        """Create indexes safely without conflicts"""
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
    
    async def extract_and_store(self, channel_id, message_id, file_name):
        """Extract thumbnail from message and store in database"""
        try:
            if not self.initialized:
                logger.warning("⚠️ Thumbnail Manager not initialized")
                return False

            client = self.user_client or self.bot_client
            if not client:
                logger.warning("⚠️ No client available")
                return False

            message = await client.get_messages(channel_id, message_id)
            if not message:
                logger.warning(f"⚠️ Message {message_id} not found")
                return False

            media = message.video or message.document
            if not media:
                logger.warning(f"⚠️ No media in message {message_id}")
                return False

            if not file_name:
                file_name = getattr(media, "file_name", "video.mp4")

            if not is_video_file(file_name):
                return False

            clean_title = extract_clean_title(file_name)
            normalized = normalize_title(clean_title)
            year = extract_year(file_name)

            existing = await self.thumbnails_col.find_one({
                'normalized_title': normalized,
                'has_thumbnail': True
            })
            if existing:
                logger.debug(f"📦 Thumbnail already exists for: {clean_title}")
                return True

            thumbnail_file_id = None
            if message.video:
                if hasattr(message.video, 'thumbnail') and message.video.thumbnail:
                    thumbnail_file_id = message.video.thumbnail.file_id
                elif hasattr(message.video, 'thumbs') and message.video.thumbs:
                    thumbnail_file_id = message.video.thumbs[0].file_id
            elif message.document:
                if hasattr(message.document, 'thumbnail') and message.document.thumbnail:
                    thumbnail_file_id = message.document.thumbnail.file_id
                elif hasattr(message.document, 'thumbs') and message.document.thumbs:
                    thumbnail_file_id = message.document.thumbs[0].file_id

            if not thumbnail_file_id:
                logger.debug(f"⚠️ No thumbnail in message {message_id}")
                await self.thumbnails_col.update_one(
                    {'normalized_title': normalized},
                    {'$set': {
                        'normalized_title': normalized,
                        'title': clean_title,
                        'year': year,
                        'has_thumbnail': False,
                        'checked_at': datetime.now()
                    }},
                    upsert=True
                )
                return False

            try:
                downloaded = await client.download_media(thumbnail_file_id, in_memory=True)
                if not downloaded:
                    logger.warning(f"⚠️ Failed to download thumbnail for {clean_title}")
                    return False

                if isinstance(downloaded, bytes):
                    thumbnail_data = downloaded
                else:
                    downloaded.seek(0)
                    thumbnail_data = downloaded.read()

                thumbnail_url = f"data:image/jpeg;base64,{base64.b64encode(thumbnail_data).decode()}"
                size_kb = len(thumbnail_url) / 1024

                await self.thumbnails_col.update_one(
                    {'normalized_title': normalized},
                    {'$set': {
                        'normalized_title': normalized,
                        'title': clean_title,
                        'year': year,
                        'thumbnail_url': thumbnail_url,
                        'thumbnail_source': 'telegram',
                        'has_thumbnail': True,
                        'extracted_at': datetime.now(),
                        'message_id': message_id,
                        'channel_id': channel_id,
                        'file_name': file_name,
                        'size_kb': size_kb,
                        'file_id': thumbnail_file_id
                    }},
                    upsert=True
                )

                logger.info(f"✅✅✅ Thumbnail stored: {clean_title} ({size_kb:.1f}KB)")
                return True

            except Exception as e:
                logger.error(f"❌ Download error for {clean_title}: {e}")
                return False

        except Exception as e:
            logger.error(f"❌ extract_and_store error: {e}")
            return False

    async def get_thumbnail_stats(self):
        """Get thumbnail statistics"""
        try:
            if not self.thumbnails_col:
                return self.stats

            total_with = await self.thumbnails_col.count_documents({'has_thumbnail': True})
            total_without = await self.thumbnails_col.count_documents({'has_thumbnail': False})
            total_failed = await self.thumbnails_col.count_documents({'has_thumbnail': False, 'failed': True})

            pipeline = [
                {'$match': {'has_thumbnail': True, 'size_kb': {'$exists': True}}},
                {'$group': {
                    '_id': None,
                    'total_size': {'$sum': '$size_kb'},
                    'avg_size': {'$avg': '$size_kb'}
                }}
            ]
            result = await self.thumbnails_col.aggregate(pipeline).to_list(1)

            self.stats.update({
                'total_movies_with_thumbnails': total_with,
                'total_movies_without_thumbnails': total_without,
                'total_failed': total_failed,
                'total_size_kb': result[0]['total_size'] if result else 0,
                'avg_size_kb': result[0]['avg_size'] if result else 0
            })

            return self.stats

        except Exception as e:
            logger.error(f"❌ Stats error: {e}")
            return self.stats

    async def get_thumbnail(self, normalized_title: str) -> Optional[str]:
        """Get thumbnail URL for a movie"""
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
        """Shutdown thumbnail manager"""
        logger.info("🖼️ Thumbnail Manager shutdown complete")

thumbnail_manager = ThumbnailManager()

# ============================================================================
# ✅ TELEGRAM SESSION INITIALIZATION
# ============================================================================

async def initialize_telegram_sessions():
    """Initialize User and Bot sessions with proper error handling"""
    global User, Bot, user_session_ready, bot_session_ready, bot_handler, poster_fetcher
    
    logger.info("🔐 Initializing Telegram sessions...")
    
    # ========== USER SESSION ==========
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
            
            # Set user client for poster fetcher
            if poster_fetcher:
                poster_fetcher.set_telegram_clients(user_client=User, bot_client=Bot)
            
        except Exception as e:
            logger.error(f"❌ User session error: {e}")
            User = None
            user_session_ready = False
    else:
        logger.warning("⚠️ No user session string provided")
    
    # ========== BOT SESSION ==========
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
            
            # Initialize bot handler
            bot_handler.bot = Bot
            bot_handler.initialized = True
            bot_handler.bot_username = bot_info.username
            
            # Set bot client for poster fetcher
            if poster_fetcher:
                poster_fetcher.set_telegram_clients(user_client=User, bot_client=Bot)
            
        except Exception as e:
            logger.error(f"❌ Bot session error: {e}")
            Bot = None
            bot_session_ready = False
    else:
        logger.warning("⚠️ No bot token provided")
    
    # Initialize thumbnail manager
    global thumbnail_manager
    if mongo_client:
        thumbnail_manager.mongodb = mongo_client
        thumbnail_manager.bot_client = Bot
        thumbnail_manager.user_client = User
        await thumbnail_manager.initialize()
    
    logger.info("✅ Telegram sessions initialized")

# ============================================================================
# ✅ AUTO INDEXING v9.5
# ============================================================================

class AutoIndexer:
    """🔄 AUTO INDEXER v9.5 - FIXED CHANNEL IDs"""
    
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
    
    async def index_file_channel(self, limit: int = None, since: datetime = None):
        """Index files from file channel - CORRECT CHANNEL ID"""
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
            
            await self._index_text_channels()
            
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
            return None
            
        finally:
            self.is_indexing = False
            is_indexing = False
    
    async def _index_text_channels(self):
        """Index text channels for posts"""
        try:
            for channel_id in Config.TEXT_CHANNEL_IDS:
                try:
                    count = 0
                    async for msg in User.search_messages(channel_id, limit=50):
                        if not msg or not msg.text:
                            continue
                        
                        title = extract_title_smart(msg.text)
                        if not title or len(title) < 3:
                            continue
                        
                        normalized = normalize_title(title)
                        
                        existing = await files_col.find_one({
                            'normalized_title': normalized,
                            'channel_id': channel_id,
                            'is_post': True
                        })
                        
                        if existing:
                            continue
                        
                        await files_col.insert_one({
                            'title': title,
                            'normalized_title': normalized,
                            'content': msg.text,
                            'message_id': msg.id,
                            'channel_id': channel_id,
                            'date': msg.date,
                            'indexed_at': datetime.now(),
                            'is_post': True,
                            'has_file': False
                        })
                        count += 1
                    
                    if count > 0:
                        logger.info(f"📝 Indexed {count} posts from channel {channel_id}")
                        
                except Exception as e:
                    logger.error(f"❌ Text channel {channel_id} error: {e}")
                    
        except Exception as e:
            logger.error(f"❌ Text indexing error: {e}")
    
    async def start_auto_indexing(self):
        """Start automatic indexing loop"""
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
        """Stop indexing"""
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
# ✅ QUART ROUTES - WITH /api/movies ENDPOINT
# ============================================================================

@app.route('/')
async def home():
    """Home page"""
    return jsonify({
        'status': 'online',
        'version': '9.5-FIXED-BOT',
        'name': 'SK4FiLM API',
        'features': [
            '🏠 /api/movies - 100 Movies for Home Page',
            '🔍 /search - Search Movies',
            '📤 /files/send - Send Files to Users',
            '🖼️ All Poster Sources (MongoDB → TMDB → OMDB → Letterboxd → IMDB → JustWatch → IMPAwards → TELEGRAM → Fallback)',
            '📁 File Sending with Auto-Delete',
            '🔄 Auto Indexing',
            '📊 Thumbnail Extraction'
        ],
        'endpoints': [
            '/api/movies - Get 100 movies for home page',
            '/search?q=movie&page=1 - Search movies',
            '/files/send/<channel_id>/<message_id> - Send file to user',
            '/status - System status',
            '/index - Trigger indexing',
            '/thumbnails/<title> - Get thumbnail',
            '/bot/status - Bot status',
            '/poster/stats - Poster fetcher stats',
            '/health - Health check'
        ]
    })

@app.route('/api/movies', methods=['GET'])
async def api_movies():
    """
    🏠 HOME PAGE MOVIES ENDPOINT
    Returns 100 movies with thumbnails for home page
    """
    try:
        limit = request.args.get('limit', 100, type=int)
        limit = min(limit, 200)  # Max 200 movies
        
        logger.info(f"🏠 /api/movies called with limit={limit}")
        
        # Get movies
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
    """Health check endpoint"""
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
    """Full status page with poster stats"""
    # Get database stats
    db_stats = {}
    if files_col:
        db_stats['total_files'] = await files_col.count_documents({})
        db_stats['posts'] = await files_col.count_documents({'is_post': True})
        db_stats['videos'] = await files_col.count_documents({'is_post': {'$ne': True}})
    
    if thumbnails_col:
        db_stats['thumbnails'] = await thumbnails_col.count_documents({'has_thumbnail': True})
    
    # Bot status
    bot_status = await bot_handler.get_bot_status() if bot_handler else {'initialized': False}
    
    # Auto-delete stats
    auto_delete_stats = await bot_handler.get_auto_delete_stats() if bot_handler else {}
    
    # Indexer stats
    indexer_stats = auto_indexer.stats if auto_indexer else {}
    
    # Poster fetcher stats
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
        'sessions': {
            'user': user_session_ready,
            'bot': bot_session_ready
        },
        'bot': bot_status,
        'auto_delete': auto_delete_stats,
        'indexer': indexer_stats,
        'poster_fetcher': poster_stats,
        'thumbnail_manager': {
            'initialized': thumbnail_manager.initialized if thumbnail_manager else False,
            'stats': thumbnail_manager.stats if thumbnail_manager and thumbnail_manager.initialized else {}
        },
        'performance': performance_monitor.get_stats() if performance_monitor else {}
    })

@app.route('/search', methods=['GET'])
async def search():
    """Search endpoint with all poster sources"""
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
        return jsonify({
            'error': 'Search failed',
            'message': str(e)
        }), 500

@app.route('/files/send/<int:channel_id>/<int:message_id>', methods=['GET', 'POST'])
async def send_file_to_user(channel_id: int, message_id: int):
    """Send file directly to user"""
    try:
        if request.method == 'POST':
            data = await request.get_json()
            user_id = data.get('user_id')
            quality = data.get('quality', '480p')
        else:
            user_id = request.args.get('user_id', type=int)
            quality = request.args.get('quality', '480p')
        
        if not user_id:
            return jsonify({
                'error': 'User ID required',
                'message': 'Please provide user_id in request'
            }), 400
        
        valid_channels = [Config.FILE_CHANNEL_ID] + Config.TEXT_CHANNEL_IDS + [Config.MAIN_CHANNEL_ID]
        if channel_id not in valid_channels:
            return jsonify({
                'error': 'Invalid channel',
                'message': 'Channel ID not in allowed channels'
            }), 400
        
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
                'auto_delete_minutes': Config.AUTO_DELETE_TIME,
                'result': result
            })
        else:
            return jsonify({
                'error': 'Send failed',
                'message': result.get('message', 'Unknown error')
            }), 500
            
    except Exception as e:
        logger.error(f"❌ Send file error: {e}")
        return jsonify({
            'error': 'Server error',
            'message': str(e)
        }), 500

@app.route('/files/info/<int:channel_id>/<int:message_id>', methods=['GET'])
async def get_file_info(channel_id: int, message_id: int):
    """Get file information without sending"""
    try:
        file_info = await bot_handler.get_file_info(channel_id, message_id)
        
        if file_info:
            return jsonify(file_info)
        else:
            return jsonify({
                'error': 'File not found',
                'message': 'Could not retrieve file information'
            }), 404
            
    except Exception as e:
        logger.error(f"❌ File info error: {e}")
        return jsonify({
            'error': 'Server error',
            'message': str(e)
        }), 500

@app.route('/thumbnails/<path:title>', methods=['GET'])
async def get_thumbnail_by_title(title: str):
    """Get thumbnail with ALL poster sources"""
    try:
        normalized = normalize_title(title)
        thumbnail_url = await thumbnail_manager.get_thumbnail(normalized)
        
        if thumbnail_url:
            return jsonify({
                'title': title,
                'normalized': normalized,
                'thumbnail_url': thumbnail_url,
                'source': 'mongodb',
                'found': True
            })
        else:
            # Use poster fetcher with all sources
            if Config.POSTER_FETCHING_ENABLED and poster_fetcher:
                poster = await poster_fetcher.fetch_poster(title)
                if poster and poster.get('poster_url') and poster.get('found'):
                    return jsonify({
                        'title': title,
                        'normalized': normalized,
                        'thumbnail_url': poster['poster_url'],
                        'source': poster.get('source', 'poster'),
                        'rating': poster.get('rating', '0.0'),
                        'year': poster.get('year', ''),
                        'found': True
                    })
            
            return jsonify({
                'title': title,
                'normalized': normalized,
                'thumbnail_url': FALLBACK_THUMBNAIL_URL,
                'source': 'fallback',
                'found': False
            })
            
    except Exception as e:
        logger.error(f"❌ Thumbnail error: {e}")
        return jsonify({
            'error': 'Server error',
            'message': str(e)
        }), 500

@app.route('/thumbnails/extract/<int:channel_id>/<int:message_id>', methods=['POST'])
async def extract_thumbnail_endpoint(channel_id: int, message_id: int):
    """Extract thumbnail from message"""
    try:
        if not thumbnail_manager or not thumbnail_manager.initialized:
            return jsonify({
                'error': 'Thumbnail manager not initialized'
            }), 503
        
        result = await thumbnail_manager.extract_and_store(
            channel_id, message_id, None
        )
        
        return jsonify({
            'channel_id': channel_id,
            'message_id': message_id,
            'extracted': result
        })
        
    except Exception as e:
        logger.error(f"❌ Extract thumbnail error: {e}")
        return jsonify({
            'error': 'Server error',
            'message': str(e)
        }), 500

@app.route('/index', methods=['POST'])
async def index_files():
    """Manually trigger indexing"""
    try:
        if auto_indexer.is_indexing:
            return jsonify({
                'status': 'busy',
                'message': 'Indexing already in progress'
            }), 409
        
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
            return jsonify({
                'error': 'Indexing failed',
                'message': 'Check logs for details'
            }), 500
            
    except Exception as e:
        logger.error(f"❌ Index error: {e}")
        return jsonify({
            'error': 'Server error',
            'message': str(e)
        }), 500

@app.route('/poster/stats', methods=['GET'])
async def poster_stats():
    """Get poster fetcher statistics"""
    if not poster_fetcher:
        return jsonify({
            'error': 'Poster fetcher not initialized'
        }), 503
    
    stats = poster_fetcher.get_stats()
    return jsonify(stats)

@app.route('/poster/clear-cache', methods=['POST'])
async def clear_poster_cache():
    """Clear poster cache"""
    if not poster_fetcher:
        return jsonify({
            'error': 'Poster fetcher not initialized'
        }), 503
    
    await poster_fetcher.clear_cache()
    return jsonify({
        'status': 'success',
        'message': 'Poster cache cleared'
    })

@app.route('/bot/status', methods=['GET'])
async def bot_status():
    """Get bot status"""
    status = await bot_handler.get_bot_status()
    return jsonify(status)

@app.route('/bot/auto-delete/cancel/<int:user_id>/<int:message_id>', methods=['POST'])
async def cancel_auto_delete(user_id: int, message_id: int):
    """Cancel auto-delete for a specific message"""
    try:
        result = await bot_handler.cancel_auto_delete(user_id, message_id)
        
        return jsonify({
            'user_id': user_id,
            'message_id': message_id,
            'cancelled': result
        })
        
    except Exception as e:
        logger.error(f"❌ Cancel auto-delete error: {e}")
        return jsonify({
            'error': 'Server error',
            'message': str(e)
        }), 500

@app.route('/bot/auto-delete/stats', methods=['GET'])
async def auto_delete_stats():
    """Get auto-delete statistics"""
    stats = await bot_handler.get_auto_delete_stats()
    return jsonify(stats)

@app.errorhandler(404)
async def not_found(error):
    return jsonify({'error': 'Not found', 'message': 'Endpoint does not exist'}), 404

@app.errorhandler(500)
async def server_error(error):
    return jsonify({'error': 'Server error', 'message': 'Internal server error'}), 500

# ============================================================================
# ✅ APPLICATION STARTUP & SHUTDOWN
# ============================================================================

@app.before_serving
async def startup():
    """Startup tasks with all poster sources"""
    global mongo_client, db, files_col, verification_col, thumbnails_col, posters_col
    global cache_manager, verification_system, premium_system, poster_fetcher
    
    logger.info("=" * 60)
    logger.info("🚀 SK4FiLM v9.5 STARTING...")
    logger.info("📌 Poster Sources: MongoDB → TMDB → OMDB → Letterboxd → IMDB → JustWatch → IMPAwards → TELEGRAM → Fallback")
    logger.info("📌 /api/movies - 100 Movies for Home Page")
    logger.info("=" * 60)
    
    # ========== DATABASE ==========
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
    
    # ========== CACHE ==========
    try:
        cache_manager = CacheManager(Config)
        await cache_manager.init_redis()
        await cache_manager.start_cleanup_task()
        logger.info("✅ Cache initialized")
    except Exception as e:
        logger.error(f"❌ Cache error: {e}")
        cache_manager = None
    
    # ========== POSTER FETCHER - WITH ALL SOURCES ==========
    if Config.POSTER_FETCHING_ENABLED and POSTER_FETCHER_AVAILABLE:
        try:
            # Initialize Redis client for poster fetcher
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
            
            # Create poster fetcher with all sources
            poster_fetcher = await create_poster_fetcher(
                config=Config,
                mongo_client=mongo_client,
                redis_client=redis_client
            )
            
            logger.info("✅ Poster fetcher initialized with all sources")
            logger.info("   Sources: TMDB, OMDB, Letterboxd, IMDB, JustWatch, IMPAwards, TELEGRAM")
            
        except Exception as e:
            logger.error(f"❌ Poster fetcher error: {e}")
            poster_fetcher = None
    else:
        poster_fetcher = None
        logger.warning("⚠️ Poster fetching disabled")
    
    # ========== VERIFICATION SYSTEM ==========
    if VerificationSystem:
        try:
            verification_system = VerificationSystem(Config, mongo_client)
            logger.info("✅ Verification system initialized")
        except Exception as e:
            logger.error(f"❌ Verification system error: {e}")
            verification_system = None
    
    # ========== PREMIUM SYSTEM ==========
    if PremiumSystem:
        try:
            premium_system = PremiumSystem(Config, mongo_client)
            logger.info("✅ Premium system initialized")
        except Exception as e:
            logger.error(f"❌ Premium system error: {e}")
            premium_system = None
    
    # ========== TELEGRAM SESSIONS ==========
    await initialize_telegram_sessions()
    
    # ========== AUTO INDEXING ==========
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
    """Shutdown tasks"""
    logger.info("🛑 Shutting down SK4FiLM v9.5...")
    
    # Stop auto-indexer
    if auto_indexer:
        await auto_indexer.stop()
    
    # Stop bot handler
    if bot_handler:
        await bot_handler.shutdown()
    
    # Stop cache
    if cache_manager:
        await cache_manager.stop()
    
    # Stop verification
    if verification_system:
        await verification_system.stop()
    
    # Stop premium
    if premium_system and hasattr(premium_system, 'stop_cleanup_task'):
        await premium_system.stop_cleanup_task()
    
    # Stop thumbnail manager
    if thumbnail_manager:
        await thumbnail_manager.shutdown()
    
    # Stop poster fetcher
    if poster_fetcher:
        await poster_fetcher.close()
    
    # Stop Telegram sessions
    if User and user_session_ready:
        try:
            await User.stop()
            logger.info("✅ User session stopped")
        except:
            pass
    
    if Bot and bot_session_ready:
        try:
            await Bot.stop()
            logger.info("✅ Bot session stopped")
        except:
            pass
    
    # Stop MongoDB
    if mongo_client:
        try:
            mongo_client.close()
            logger.info("✅ MongoDB disconnected")
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
