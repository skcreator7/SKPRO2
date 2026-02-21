# ============================================================================
# 🚀 SK4FiLM v9.5 - COMPLETE FIXED BOT WITH FILE SENDING
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
    from poster_fetching import PosterFetcher, PosterSource
    logger.debug("✅ Poster fetching module imported")
    POSTER_FETCHER_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ Poster fetching module import error: {e}")
    PosterFetcher = None
    PosterSource = None
    POSTER_FETCHER_AVAILABLE = False

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
    MAIN_CHANNEL_ID = -1001891090100  # Main channel (posts)
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
    POSTER_FETCH_TIMEOUT = 3  # 3 seconds
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
# ✅ FIXED get_best_thumbnail - POSTS KE LIYE BHI THUMBNAIL
# ============================================================================

async def get_best_thumbnail(normalized_title: str, clean_title: str = None, 
                            year: str = None, msg=None, is_post: bool = False) -> Tuple[str, str]:
    """
    🎯 FIXED: Get thumbnail for both files AND posts
    Priority: MongoDB > Poster > Fallback
    """
    # ========== STEP 1: MongoDB Check (WORKS FOR BOTH) ==========
    if thumbnails_col is not None:
        try:
            # Check if movie has ANY thumbnail in MongoDB
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
    
    # ========== STEP 2: Poster Fetch (WORKS FOR BOTH) ==========
    if Config.POSTER_FETCHING_ENABLED and poster_fetcher and clean_title:
        try:
            poster = await get_poster_for_movie(clean_title, year)
            
            if poster and poster.get('poster_url') and poster.get('found'):
                logger.info(f"🎬 POSTER FOUND for {clean_title} (Post: {is_post})")
                
                # For file results, try to store real thumbnail
                if not is_post and msg and has_telegram_thumbnail(msg):
                    asyncio.create_task(
                        try_store_real_thumbnail(normalized_title, clean_title, msg)
                    )
                
                return poster['poster_url'], 'poster'
                
        except Exception as e:
            logger.debug(f"⚠️ Poster error: {e}")
    
    # ========== STEP 3: For files only - try extraction ==========
    if not is_post and msg and clean_title:
        asyncio.create_task(
            try_store_real_thumbnail(normalized_title, clean_title, msg)
        )
    
    # ========== STEP 4: Fallback ==========
    logger.info(f"⚠️ FALLBACK for {clean_title} (Post: {is_post})")
    return FALLBACK_THUMBNAIL_URL, 'fallback'
                                
# ============================================================================
# ✅ POSTER FETCHING FUNCTIONS
# ============================================================================

async def get_poster_for_movie(title: str, year: str = "", quality: str = "") -> Dict[str, Any]:
    """Get poster for movie"""
    global poster_fetcher, posters_col
    
    if poster_fetcher is None:
        return {
            'poster_url': '',
            'source': 'none',
            'rating': '0.0',
            'year': year,
            'title': title,
            'found': False
        }
    
    # Check cache first
    if posters_col is not None:
        try:
            cache_key = f"{normalize_title(title)}:{year}"
            cached = await posters_col.find_one({'cache_key': cache_key})
            if cached and cached.get('poster_url'):
                if datetime.now() - cached.get('cached_at', datetime.min) < timedelta(hours=24):
                    logger.debug(f"📦 Cached poster: {title}")
                    cached['found'] = True
                    cached['from_cache'] = True
                    return cached
        except Exception as e:
            logger.debug(f"Cache check error: {e}")
    
    try:
        poster_task = asyncio.create_task(poster_fetcher.fetch_poster(title))
        
        try:
            poster_data = await asyncio.wait_for(poster_task, timeout=Config.POSTER_FETCH_TIMEOUT)
            
            if poster_data and poster_data.get('poster_url'):
                logger.debug(f"✅ Poster fetched: {title[:30]} - {poster_data['source']}")
                poster_data['found'] = True
                poster_data['cached_at'] = datetime.now()
                poster_data['cache_key'] = f"{normalize_title(title)}:{year}"
                
                # Store in cache
                if posters_col is not None:
                    try:
                        await posters_col.update_one(
                            {'cache_key': poster_data['cache_key']},
                            {'$set': poster_data},
                            upsert=True
                        )
                    except Exception as e:
                        logger.debug(f"Cache store error: {e}")
                
                return poster_data
            else:
                raise ValueError("Invalid poster data")
                
        except (asyncio.TimeoutError, ValueError, Exception) as e:
            logger.debug(f"⚠️ Poster fetch timeout/error for {title[:30]}: {e}")
            
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

# ============================================================================
# ✅ OPTIMIZED SEARCH v4.2 - FIXED CHANNEL IDs
# ============================================================================

@performance_monitor.measure("optimized_search")
@async_cache_with_ttl(maxsize=500, ttl=600)
async def search_movies_optimized(query, limit=15, page=1):
    """
    🔥 OPTIMIZED SEARCH v4.2 - One thumbnail per movie, all qualities shown
    FIXED: Posts ke liye bhi thumbnail, correct channel IDs
    """
    start_time = time.time()
    offset = (page - 1) * limit
    
    logger.info(f"🔍 OPTIMIZED SEARCH for: '{query}'")
    
    # Dictionary to store unique results
    results_dict = {}
    
    # ============================================================================
    # ✅ STEP 1: Direct Telegram FILE CHANNEL Search (Files) - CORRECT CHANNEL ID
    # ============================================================================
    if user_session_ready and User is not None:
        try:
            file_count = 0
            
            async for msg in User.search_messages(
                Config.FILE_CHANNEL_ID,  # ✅ FIXED: Using correct FILE_CHANNEL_ID
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
                
                # Extract clean title (without quality)
                clean_title = extract_clean_title(file_name)
                normalized = normalize_title(clean_title)
                quality = detect_quality_enhanced(file_name)
                year = extract_year(file_name)
                
                # ✅ FIXED: Store correct channel ID and message ID
                quality_entry = {
                    'quality': quality,
                    'file_name': file_name,
                    'file_size': msg.document.file_size if msg.document else msg.video.file_size,
                    'file_size_formatted': format_size(msg.document.file_size if msg.document else msg.video.file_size),
                    'message_id': msg.id,
                    'channel_id': Config.FILE_CHANNEL_ID,  # ✅ FIXED: Correct channel ID
                    'file_id': msg.document.file_id if msg.document else msg.video.file_id,
                    'date': msg.date,
                    'has_thumbnail_in_telegram': has_telegram_thumbnail(msg)
                }
                
                if normalized not in results_dict:
                    # NEW MOVIE - Create main entry
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
                    
                    # Add this quality
                    results_dict[normalized]['qualities'][quality] = quality_entry
                    results_dict[normalized]['available_qualities'].append(quality)
                    
                else:
                    # EXISTING MOVIE - Just add this quality
                    if quality not in results_dict[normalized]['qualities']:
                        results_dict[normalized]['qualities'][quality] = quality_entry
                        results_dict[normalized]['available_qualities'].append(quality)
                        
                    # Update best date if newer
                    if msg.date and msg.date > results_dict[normalized]['best_date']:
                        results_dict[normalized]['best_date'] = msg.date
                        results_dict[normalized]['is_new'] = is_new(msg.date)
                    
                    # Store message for potential thumbnail extraction (if we don't have one)
                    if not results_dict[normalized].get('first_file_msg'):
                        results_dict[normalized]['first_file_msg'] = msg
                    
                    results_dict[normalized]['all_messages'].append(msg)
                
                file_count += 1
            
            logger.info(f"📁 Found {file_count} file results from channel {Config.FILE_CHANNEL_ID}")
            
        except Exception as e:
            logger.error(f"❌ File channel search error: {e}")
    
    # ============================================================================
    # ✅ STEP 2: Direct Telegram TEXT CHANNELS Search (Posts) - CORRECT CHANNEL IDs
    # ============================================================================
    if user_session_ready and User is not None:
        try:
            post_count = 0
            
            for channel_id in Config.TEXT_CHANNEL_IDS:  # ✅ FIXED: Using TEXT_CHANNEL_IDS
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
                            # EXISTING MOVIE - Add post info
                            results_dict[normalized]['has_post'] = True
                            results_dict[normalized]['post_content'] = format_post(msg.text, max_length=500)
                            results_dict[normalized]['post_channel_id'] = channel_id
                            results_dict[normalized]['post_message_id'] = msg.id
                            results_dict[normalized]['result_type'] = 'file_and_post'
                        else:
                            # NEW POST-ONLY MOVIE
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
    # ✅ STEP 3: Get Thumbnails for Each Result (FIXED FOR POSTS)
    # ============================================================================
    for normalized, result in results_dict.items():
        # Check if it's a post-only result
        is_post_only = result.get('has_file') == False and result.get('has_post') == True
        
        if result.get('has_file'):
            # File result - has message object
            msg = result.get('first_file_msg')
            thumbnail_url, thumbnail_source = await get_best_thumbnail(
                normalized,
                result.get('title'),
                result.get('year'),
                msg,
                is_post=False
            )
        else:
            # Post-only result - no message object
            thumbnail_url, thumbnail_source = await get_best_thumbnail(
                normalized,
                result.get('title'),
                result.get('year'),
                None,  # No message
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
    
    # Sort by: has_file (priority), is_new, best_date
    all_results.sort(key=lambda x: (
        1 if x.get('has_file') else 0,  # Files first
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
    mongodb_count = 0
    poster_count = 0
    fallback_count = 0
    
    for r in paginated:
        if r.get('thumbnail_source') == 'mongodb':
            mongodb_count += 1
        elif r.get('thumbnail_source') in ['tmdb', 'omdb', 'poster']:
            poster_count += 1
        elif r.get('thumbnail_source') == 'fallback':
            fallback_count += 1
    
    # Count total qualities
    total_qualities = sum(len(r.get('available_qualities', [])) for r in all_results)
    
    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info("📊 SEARCH RESULTS SUMMARY:")
    logger.info(f"   • Query: '{query}'")
    logger.info(f"   • Total movies: {total}")
    logger.info(f"   • Total qualities: {total_qualities}")
    logger.info(f"   • Post+Files: {combined_count}")
    logger.info(f"   • Post only: {post_count - combined_count}")
    logger.info(f"   • File only: {file_count - combined_count}")
    logger.info(f"   • MongoDB thumbnails: {mongodb_count}")
    logger.info(f"   • Posters: {poster_count}")
    logger.info(f"   • Fallback images: {fallback_count}")
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
            'total_qualities': total_qualities,
            'file_results': file_count,
            'post_results': post_count,
            'combined_results': combined_count,
            'mongodb_thumbnails': mongodb_count,
            'posters': poster_count,
            'fallback': fallback_count,
            'mode': 'optimized_v4.2',
            'poster_fetching': Config.POSTER_FETCHING_ENABLED
        },
        'bot_username': Config.BOT_USERNAME
    }

# ============================================================================
# ✅ THUMBNAIL MANAGER - FIXED CHANNEL IDS
# ============================================================================

class ThumbnailManager:
    """🖼️ THUMBNAIL MANAGER v12 - FIXED CHANNEL IDs"""

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
        """Extract thumbnail and store in MongoDB (ONE per movie)"""
        try:
            title = extract_clean_title(file_name)
            normalized = normalize_title(title)
            
            # Check if movie already has thumbnail
            existing = await self.thumbnails_col.find_one({
                'normalized_title': normalized,
                'has_thumbnail': True
            })
            
            if existing:
                logger.debug(f"📦 Movie already has thumbnail: {title}")
                return {
                    'success': True,
                    'thumbnail_url': existing.get('thumbnail_url'),
                    'cached': True,
                    'has_thumbnail': True
                }
            
            # Extract thumbnail
            client = self.user_client or self.bot_client
            if not client:
                return {'success': False, 'error': 'No client'}
            
            message = await client.get_messages(channel_id, message_id)
            if not message:
                return {'success': False, 'error': 'Message not found'}
            
            # Find thumbnail file_id
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
                # No thumbnail - mark as checked
                await self.thumbnails_col.update_one(
                    {'normalized_title': normalized},
                    {'$set': {
                        'normalized_title': normalized,
                        'title': title,
                        'has_thumbnail': False,
                        'checked_at': datetime.now()
                    }},
                    upsert=True
                )
                self.stats['total_movies_without_thumbnails'] += 1
                return {'success': True, 'has_thumbnail': False}
            
            # Download thumbnail
            downloaded = await client.download_media(thumbnail_file_id, in_memory=True)
            if not downloaded:
                return {'success': False, 'error': 'Download failed'}
            
            if isinstance(downloaded, bytes):
                thumbnail_data = downloaded
            else:
                with open(downloaded, 'rb') as f:
                    thumbnail_data = f.read()
            
            # Convert to base64
            thumbnail_url = f"data:image/jpeg;base64,{base64.b64encode(thumbnail_data).decode()}"
            size_kb = len(thumbnail_url) / 1024
            
            # ✅ FIXED: Store with correct channel ID
            await self.thumbnails_col.update_one(
                {'normalized_title': normalized},
                {'$set': {
                    'normalized_title': normalized,
                    'title': title,
                    'year': extract_year(file_name),
                    'thumbnail_url': thumbnail_url,
                    'thumbnail_source': 'telegram',
                    'has_thumbnail': True,
                    'extracted_at': datetime.now(),
                    'message_id': message_id,
                    'channel_id': Config.FILE_CHANNEL_ID,  # ✅ FIXED: Use Config.FILE_CHANNEL_ID
                    'file_name': file_name,
                    'size_kb': size_kb,
                    'file_id': thumbnail_file_id
                }},
                upsert=True
            )
            
            self.stats['total_movies_with_thumbnails'] += 1
            self.stats['total_size_kb'] += size_kb
            self.stats['avg_size_kb'] = self.stats['total_size_kb'] / self.stats['total_movies_with_thumbnails']
            
            logger.info(f"✅ Movie thumbnail stored: {title} ({size_kb:.1f}KB) [Channel: {Config.FILE_CHANNEL_ID}, Msg: {message_id}]")
            
            return {
                'success': True,
                'thumbnail_url': thumbnail_url,
                'has_thumbnail': True,
                'size_kb': size_kb
            }
            
        except Exception as e:
            logger.error(f"❌ Extract and store error: {e}")
            self.stats['total_failed'] += 1
            return {'success': False, 'error': str(e), 'has_thumbnail': False}
    
    async def get_stats(self):
        """Get thumbnail statistics"""
        try:
            if self.thumbnails_col:
                total = await self.thumbnails_col.count_documents({})
                with_thumb = await self.thumbnails_col.count_documents({'has_thumbnail': True})
                without_thumb = await self.thumbnails_col.count_documents({'has_thumbnail': False})
                
                # Get channel-specific stats
                file_channel_thumbnails = await self.thumbnails_col.count_documents({
                    'channel_id': Config.FILE_CHANNEL_ID,
                    'has_thumbnail': True
                })
                
                return {
                    **self.stats,
                    'total_documents': total,
                    'with_thumbnail': with_thumb,
                    'without_thumbnail': without_thumb,
                    'file_channel_thumbnails': file_channel_thumbnails
                }
        except:
            pass
        return self.stats
    
    async def shutdown(self):
        """Shutdown thumbnail manager"""
        logger.info("🖼️ Shutting down Thumbnail Manager...")
        logger.info(f"✅ Stats - Movies with thumbnails: {self.stats['total_movies_with_thumbnails']}")

# ============================================================================
# ✅ OPTIMIZED FILE INDEXING MANAGER - LIMITED
# ============================================================================

class OptimizedFileIndexingManager:
    def __init__(self):
        self.is_running = False
        self.indexing_task = None
        self.last_run = None
        self.total_indexed = 0
        self.movies_with_thumbnails = 0
        self.indexing_stats = {
            'total_runs': 0,
            'total_messages_fetched': 0,
            'total_videos_found': 0,
            'unique_movies_found': 0,
            'movies_with_thumbnails': 0,
            'movies_without_thumbnails': 0,
            'thumbnails_extracted': 0,
            'thumbnails_failed': 0,
            'last_success': None,
            'last_error': None
        }
    
    async def start_indexing(self, force_reindex=False):
        if self.is_running:
            logger.warning("⚠️ File indexing already running")
            return
        
        logger.info("=" * 60)
        logger.info(f"🚀 STARTING MOVIE THUMBNAIL INDEXING - Channel: {Config.FILE_CHANNEL_ID}")
        logger.info("=" * 60)
        
        self.is_running = True
        self.indexing_stats['total_runs'] += 1
        
        try:
            await self._run_optimized_indexing(force_reindex)
        except Exception as e:
            logger.error(f"❌ Indexing error: {e}")
            self.indexing_stats['last_error'] = str(e)
        finally:
            self.is_running = False
        
        self.indexing_task = asyncio.create_task(self._indexing_loop())
    
    async def stop_indexing(self):
        self.is_running = False
        if self.indexing_task:
            self.indexing_task.cancel()
            try:
                await self.indexing_task
            except asyncio.CancelledError:
                pass
        logger.info("🛑 File indexing stopped")
    
    async def _run_optimized_indexing(self, force_reindex=False):
        if not user_session_ready and not bot_session_ready:
            logger.error("❌ No Telegram session available")
            return
        
        client = User if user_session_ready else Bot
        if client is None:
            logger.error("❌ No working Telegram client")
            return
        
        try:
            chat = await client.get_chat(Config.FILE_CHANNEL_ID)
            logger.info(f"📢 Channel: {chat.title} (ID: {Config.FILE_CHANNEL_ID})")
        except Exception as e:
            logger.error(f"❌ Cannot access file channel: {e}")
            return
        
        # Fetch only recent messages (limit 500)
        all_messages = []
        offset_id = 0
        batch_size = 50
        max_messages = 500
        
        logger.info(f"📥 Fetching up to {max_messages} messages from file channel...")
        
        while self.is_running and len(all_messages) < max_messages:
            try:
                messages = []
                async for msg in client.get_chat_history(Config.FILE_CHANNEL_ID, limit=batch_size, offset_id=offset_id):
                    messages.append(msg)
                    if len(messages) >= batch_size:
                        break
                
                if not messages:
                    break
                
                all_messages.extend(messages)
                offset_id = messages[-1].id
                logger.info(f"📥 Fetched {len(all_messages)}/{max_messages} messages...")
                self.indexing_stats['total_messages_fetched'] = len(all_messages)
                
                if len(messages) < batch_size:
                    break
                
                # Rate limiting
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"❌ Error fetching messages: {e}")
                await asyncio.sleep(5)
                continue
        
        logger.info(f"✅ Total messages fetched: {len(all_messages)}")
        
        # Process video files and group by movie
        movie_files = {}
        video_count = 0
        
        for msg in all_messages:
            if not msg or (not msg.document and not msg.video):
                continue
            
            file_name = None
            if msg.document:
                file_name = msg.document.file_name
            elif msg.video:
                file_name = msg.video.file_name or "video.mp4"
            
            if not file_name or not is_video_file(file_name):
                continue
            
            video_count += 1
            clean_title = extract_clean_title(file_name)
            normalized = normalize_title(clean_title)
            
            if normalized not in movie_files:
                movie_files[normalized] = {
                    'title': clean_title,
                    'messages': [],
                    'has_thumbnail': False
                }
            
            movie_files[normalized]['messages'].append({
                'message': msg,
                'file_name': file_name,
                'has_thumbnail': has_telegram_thumbnail(msg),
                'message_id': msg.id
            })
            
            if has_telegram_thumbnail(msg):
                movie_files[normalized]['has_thumbnail'] = True
        
        unique_movies = len(movie_files)
        movies_with_thumbnails = sum(1 for m in movie_files.values() if m['has_thumbnail'])
        movies_without_thumbnails = unique_movies - movies_with_thumbnails
        
        self.indexing_stats['total_videos_found'] = video_count
        self.indexing_stats['unique_movies_found'] = unique_movies
        self.indexing_stats['movies_with_thumbnails'] = movies_with_thumbnails
        self.indexing_stats['movies_without_thumbnails'] = movies_without_thumbnails
        
        logger.info("=" * 60)
        logger.info("📊 SCANNING COMPLETE")
        logger.info(f"   • Total video files: {video_count}")
        logger.info(f"   • Unique movies: {unique_movies}")
        logger.info(f"   • Movies WITH thumbnails: {movies_with_thumbnails}")
        logger.info(f"   • Movies WITHOUT thumbnails: {movies_without_thumbnails}")
        logger.info("=" * 60)
        
        # Extract thumbnails for movies that don't have them yet
        movies_to_extract = []
        for normalized, movie in movie_files.items():
            # Check if already in DB
            existing = await thumbnails_col.find_one({
                'normalized_title': normalized,
                'has_thumbnail': True
            })
            
            if not existing and movie['has_thumbnail']:
                # Find first message with thumbnail
                for msg_data in movie['messages']:
                    if msg_data['has_thumbnail']:
                        movies_to_extract.append({
                            'normalized': normalized,
                            'title': movie['title'],
                            'message': msg_data['message'],
                            'file_name': msg_data['file_name'],
                            'message_id': msg_data['message_id']
                        })
                        break
        
        if movies_to_extract:
            logger.info(f"🖼️ Extracting thumbnails for {len(movies_to_extract)} movies...")
            await self._extract_thumbnails_batch(movies_to_extract)
        
        self.indexing_stats['last_success'] = datetime.now()
        self.last_run = datetime.now()
    
    async def _extract_thumbnails_batch(self, movies):
        batch_size = 3
        total_batches = math.ceil(len(movies) / batch_size)
        successful = 0
        failed = 0
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(movies))
            batch = movies[start_idx:end_idx]
            logger.info(f"🖼️ Processing batch {batch_num + 1}/{total_batches}...")
            
            for movie in batch:
                try:
                    await try_store_real_thumbnail(
                        movie['normalized'],
                        movie['title'],
                        movie['message']
                    )
                    successful += 1
                    logger.info(f"✅ Thumbnail stored for: {movie['title']} (Msg: {movie['message_id']})")
                    
                except Exception as e:
                    logger.error(f"❌ Failed for {movie['title']}: {e}")
                    failed += 1
                
                await asyncio.sleep(1)
            
            if batch_num < total_batches - 1:
                await asyncio.sleep(2)
        
        self.movies_with_thumbnails = successful
        self.indexing_stats['thumbnails_extracted'] = successful
        self.indexing_stats['thumbnails_failed'] = failed
        logger.info(f"✅ Extraction complete: {successful} successful, {failed} failed")
    
    async def _indexing_loop(self):
        while self.is_running:
            try:
                await asyncio.sleep(Config.AUTO_INDEX_INTERVAL)
                if self.is_running:
                    await self._index_new_files()
                    self.last_run = datetime.now()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Indexing loop error: {e}")
                await asyncio.sleep(60)
    
    async def _index_new_files(self):
        """Index only new files since last check"""
        try:
            if thumbnails_col is None:
                return
            
            # Get latest message_id we've processed
            latest = await thumbnails_col.find_one(
                {'channel_id': Config.FILE_CHANNEL_ID}, 
                sort=[('message_id', -1)]
            )
            last_message_id = latest.get('message_id', 0) if latest else 0
            logger.info(f"🔍 Checking for new files after message ID {last_message_id} in channel {Config.FILE_CHANNEL_ID}")
            
            client = User if user_session_ready else Bot
            if not client:
                return
            
            messages = []
            async for msg in client.get_chat_history(Config.FILE_CHANNEL_ID, limit=100):
                if msg.id > last_message_id:
                    messages.append(msg)
                else:
                    break
            
            if not messages:
                logger.info("✅ No new files found")
                return
            
            logger.info(f"📥 Found {len(messages)} new messages")
            
            # Group by movie
            new_movies = {}
            for msg in messages:
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
                
                if normalized not in new_movies and has_telegram_thumbnail(msg):
                    new_movies[normalized] = {
                        'normalized': normalized,
                        'title': clean_title,
                        'message': msg,
                        'file_name': file_name,
                        'message_id': msg.id
                    }
            
            if new_movies:
                logger.info(f"🖼️ Found {len(new_movies)} new movies with thumbnails")
                await self._extract_thumbnails_batch(list(new_movies.values()))
            
        except Exception as e:
            logger.error(f"❌ Error indexing new files: {e}")
    
    async def get_indexing_status(self):
        total_thumbnails = 0
        if thumbnails_col is not None:
            total_thumbnails = await thumbnails_col.count_documents({'has_thumbnail': True})
        
        return {
            'is_running': self.is_running,
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'total_movies_with_thumbnails': total_thumbnails,
            'movies_extracted': self.movies_with_thumbnails,
            'stats': self.indexing_stats,
            'user_session_ready': user_session_ready,
            'bot_session_ready': bot_session_ready,
            'file_channel_id': Config.FILE_CHANNEL_ID
        }

file_indexing_manager = OptimizedFileIndexingManager()

# ============================================================================
# ✅ SYNC MANAGER - FIXED CHANNEL ID
# ============================================================================

class OptimizedSyncManager:
    def __init__(self):
        self.is_monitoring = False
        self.monitoring_task = None
        self.deleted_count = 0
    
    async def start_sync_monitoring(self):
        if self.is_monitoring:
            return
        logger.info("👁️ Starting sync monitoring...")
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
            if thumbnails_col is None or User is None or not user_session_ready:
                return
            
            logger.info(f"🔍 Checking for deleted files in channel {Config.FILE_CHANNEL_ID}...")
            
            batch_size = 100
            cursor = thumbnails_col.find(
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
                        await thumbnails_col.delete_one({"_id": item['db_id']})
                        deleted_count += 1
                        self.deleted_count += 1
                        
                        if deleted_count <= 5:
                            logger.info(f"🗑️ Deleted: {item['title'][:40]}... (Msg: {item['message_id']})")
                
                if deleted_count > 0:
                    logger.info(f"✅ Deleted {deleted_count} files from channel {Config.FILE_CHANNEL_ID}")
                else:
                    logger.info("✅ No deleted files found")
                    
            except Exception as e:
                logger.error(f"❌ Error checking messages: {e}")
                    
        except Exception as e:
            logger.error(f"❌ Auto-delete error: {e}")

sync_manager = OptimizedSyncManager()

# ============================================================================
# ✅ UPDATED HOME MOVIES - FIXED CHANNEL ID
# ============================================================================

@performance_monitor.measure("home_movies")
@async_cache_with_ttl(maxsize=1, ttl=60)
async def get_home_movies(limit=25):
    """
    🎬 HOME MOVIES - FIXED CHANNEL ID
    """
    try:
        if User is None or not user_session_ready:
            return []
        
        movies = []
        seen_titles = set()
        
        logger.info(f"🎬 Fetching home movies from main channel {Config.MAIN_CHANNEL_ID}...")
        
        async for msg in User.get_chat_history(Config.MAIN_CHANNEL_ID, limit=50):
            if msg is not None and msg.text and len(msg.text) > 25:
                title = extract_title_smart(msg.text)
                
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    
                    year_match = re.search(r'\b(19|20)\d{2}\b', title)
                    year = year_match.group() if year_match else ""
                    
                    clean_title = re.sub(r'\s+\(\d{4}\)$', '', title)
                    clean_title = re.sub(r'\s+\d{4}$', '', clean_title)
                    clean_title = clean_title.strip()
                    
                    post_content = msg.text
                    formatted_content = format_post(msg.text, max_length=500)
                    norm_title = normalize_title(clean_title)
                    
                    # ========== STEP 1: TRY POSTER FIRST ==========
                    thumbnail_url = None
                    thumbnail_source = None
                    poster_data = None
                    
                    if Config.POSTER_FETCHING_ENABLED and poster_fetcher:
                        try:
                            poster_data = await get_poster_for_movie(clean_title, year)
                            if poster_data and poster_data.get('found') and poster_data.get('poster_url'):
                                thumbnail_url = poster_data['poster_url']
                                thumbnail_source = poster_data.get('source', 'poster')
                                logger.info(f"🎬 POSTER FOUND for home movie: {clean_title}")
                        except Exception as e:
                            logger.debug(f"Poster error for {clean_title}: {e}")
                    
                    # ========== STEP 2: IF NO POSTER, TRY MONGODB ==========
                    if not thumbnail_url and thumbnails_col is not None:
                        try:
                            thumb_doc = await thumbnails_col.find_one({
                                'normalized_title': norm_title,
                                'has_thumbnail': True,
                                'thumbnail_url': {'$exists': True, '$ne': None}
                            })
                            
                            if thumb_doc and thumb_doc.get('thumbnail_url'):
                                thumbnail_url = thumb_doc['thumbnail_url']
                                thumbnail_source = 'mongodb'
                                logger.info(f"📦 MONGODB thumbnail FOUND for home movie: {clean_title}")
                        except Exception as e:
                            logger.debug(f"MongoDB fetch error for {clean_title}: {e}")
                    
                    # ========== STEP 3: FALLBACK ==========
                    if not thumbnail_url:
                        thumbnail_url = FALLBACK_THUMBNAIL_URL
                        thumbnail_source = 'fallback'
                        logger.warning(f"⚠️ FALLBACK for home movie: {clean_title}")
                    
                    movie_data = {
                        'title': clean_title,
                        'original_title': title,
                        'normalized_title': norm_title,
                        'year': year,
                        'date': msg.date.isoformat() if isinstance(msg.date, datetime) else str(msg.date),
                        'is_new': is_new(msg.date) if msg.date else False,
                        'channel_id': Config.MAIN_CHANNEL_ID,
                        'message_id': msg.id,
                        'has_post': True,
                        'content': formatted_content,
                        'post_content': post_content,
                        'thumbnail_url': thumbnail_url,
                        'thumbnail_source': thumbnail_source,
                        'has_thumbnail': True,
                        'poster_url': poster_data.get('poster_url') if poster_data else None,
                        'poster_source': poster_data.get('source') if poster_data else None,
                        'poster_rating': poster_data.get('rating') if poster_data else None,
                        'has_poster': bool(poster_data and poster_data.get('poster_url'))
                    }
                    
                    movies.append(movie_data)
                    
                    if len(movies) >= limit:
                        break
        
        # Statistics
        poster_count = sum(1 for m in movies if m.get('thumbnail_source') in ['poster', 'tmdb', 'omdb'])
        mongodb_count = sum(1 for m in movies if m.get('thumbnail_source') == 'mongodb')
        fallback_count = sum(1 for m in movies if m.get('thumbnail_source') == 'fallback')
        
        logger.info("=" * 60)
        logger.info("📊 HOME MOVIES SUMMARY:")
        logger.info(f"   • Total movies: {len(movies)}")
        logger.info(f"   • POSTERS (Priority 1): {poster_count}")
        logger.info(f"   • MONGODB Thumbnails (Priority 2): {mongodb_count}")
        logger.info(f"   • FALLBACK Images (Priority 3): {fallback_count}")
        logger.info("=" * 60)
        
        return movies[:limit]
        
    except Exception as e:
        logger.error(f"❌ Home movies error: {e}")
        import traceback
        traceback.print_exc()
        return []

# ============================================================================
# ✅ TELEGRAM BOT SETUP - WITH ALL HANDLERS
# ============================================================================

async def setup_telegram_bot_handlers(bot):
    """Setup all bot command handlers"""
    
    @bot.on_message(filters.command("start"))
    async def start_command(client, message):
        """Handle /start command"""
        try:
            user_name = message.from_user.first_name or "User"
            
            # Check if it's a file request
            if len(message.command) > 1:
                file_param = message.command[1]
                parts = file_param.split('_')
                
                if len(parts) >= 2:
                    try:
                        channel_id = int(parts[0])
                        message_id = int(parts[1])
                        quality = parts[2] if len(parts) > 2 else "480p"
                        
                        # Send processing message
                        processing = await message.reply_text("⏳ **Sending your file...**")
                        
                        # Send file using bot handler
                        success, result, size = await bot_handler.send_file_to_user(
                            message.chat.id,
                            channel_id,
                            message_id,
                            quality
                        )
                        
                        if success:
                            await processing.delete()
                            await message.reply_text(
                                f"✅ **File sent successfully!**\n\n"
                                f"📁 **File:** `{result['file_name']}`\n"
                                f"📦 **Size:** {format_size(size)}\n"
                                f"📹 **Quality:** {quality}\n"
                                f"⏰ **Auto-delete in:** {Config.AUTO_DELETE_TIME} minutes",
                                reply_markup=InlineKeyboardMarkup([
                                    [InlineKeyboardButton("🌐 VISIT WEBSITE", url=Config.WEBSITE_URL)],
                                    [InlineKeyboardButton("📢 JOIN CHANNEL", url=Config.MAIN_CHANNEL_LINK)]
                                ])
                            )
                        else:
                            await processing.edit_text(f"❌ {result.get('message', 'Error sending file')}")
                        return
                        
                    except Exception as e:
                        logger.error(f"File request error: {e}")
                        await message.reply_text("❌ Invalid file request")
                        return
            
            # Normal start message
            welcome_text = (
                f"🎬 **Welcome to SK4FiLM, {user_name}!**\n\n"
                f"🌐 **Website:** {Config.WEBSITE_URL}\n\n"
                f"**How to download:**\n"
                f"1. Visit website\n"
                f"2. Search movie\n"
                f"3. Click download\n"
                f"4. File appears here\n\n"
                f"⏰ Files auto-delete in {Config.AUTO_DELETE_TIME} minutes\n\n"
                f"🎬 **Happy watching!**"
            )
            
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)],
                [InlineKeyboardButton("📢 JOIN CHANNEL", url=Config.MAIN_CHANNEL_LINK)],
                [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")]
            ])
            
            await message.reply_text(welcome_text, reply_markup=keyboard)
            
        except Exception as e:
            logger.error(f"Start command error: {e}")
            await message.reply_text("❌ Error. Please try again.")
    
    @bot.on_callback_query()
    async def handle_callbacks(client, callback_query):
        """Handle all callback queries"""
        try:
            data = callback_query.data
            
            if data == "back_to_start":
                user_name = callback_query.from_user.first_name or "User"
                text = (
                    f"🎬 **Welcome back, {user_name}!**\n\n"
                    f"Visit {Config.WEBSITE_URL} to download movies."
                )
                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)],
                    [InlineKeyboardButton("📢 JOIN CHANNEL", url=Config.MAIN_CHANNEL_LINK)],
                    [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")]
                ])
                await callback_query.message.edit_text(text, reply_markup=keyboard)
                await callback_query.answer()
                
            elif data == "buy_premium":
                text = (
                    "💎 **Premium Plans** 💎\n\n"
                    f"**Basic** - ₹50\n"
                    f"📅 30 days\n\n"
                    f"**Premium** - ₹100\n"
                    f"📅 90 days\n\n"
                    f"**Gold** - ₹200\n"
                    f"📅 180 days\n\n"
                    f"**Diamond** - ₹500\n"
                    f"📅 365 days\n\n"
                    f"**UPI ID:** `{Config.UPI_ID_BASIC}`\n\n"
                    f"📸 Send payment screenshot here to activate"
                )
                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 BACK", callback_data="back_to_start")]
                ])
                await callback_query.message.edit_text(text, reply_markup=keyboard)
                await callback_query.answer()
                
        except Exception as e:
            logger.error(f"Callback error: {e}")
            await callback_query.answer("❌ Error", show_alert=True)
    
    @bot.on_message(filters.private & filters.photo)
    async def handle_screenshot(client, message):
        """Forward payment screenshots to admins"""
        try:
            user_id = message.from_user.id
            
            # Skip if admin
            if user_id in Config.ADMIN_IDS:
                return
            
            # Get user info
            user = await client.get_users(user_id)
            user_name = f"{user.first_name or ''} {user.last_name or ''}".strip() or f"User {user_id}"
            username = f"@{user.username}" if user.username else "No username"
            
            # Alert text for admins
            alert = (
                f"📸 **Payment Screenshot**\n\n"
                f"👤 **User:** {user_name}\n"
                f"🆔 **ID:** `{user_id}`\n"
                f"📛 **Username:** {username}\n"
                f"🕒 **Time:** {datetime.now().strftime('%H:%M:%S')}"
            )
            
            # Forward to all admins
            for admin_id in Config.ADMIN_IDS:
                try:
                    await message.copy(admin_id, caption=alert)
                    logger.info(f"✅ Screenshot forwarded to admin {admin_id}")
                except Exception as e:
                    logger.error(f"Failed to forward to admin {admin_id}: {e}")
            
            # Acknowledge user
            await message.reply_text(
                "✅ **Screenshot received!**\n\n"
                "Admin will verify and activate your premium within 24 hours.\n"
                "Thank you for choosing SK4FiLM! 🎬",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔙 BACK TO START", callback_data="back_to_start")]
                ])
            )
            
        except Exception as e:
            logger.error(f"Screenshot handler error: {e}")
    
    # Admin commands
    @bot.on_message(filters.command("autodelete") & filters.user(Config.ADMIN_IDS))
    async def autodelete_status(client, message):
        """Check auto-delete status (admin only)"""
        try:
            stats = await bot_handler.get_auto_delete_stats()
            text = (
                f"⏰ **Auto-Delete System Status**\n\n"
                f"📊 **Total Tasks:** {stats.get('total_tasks', 0)}\n"
                f"⏳ **Pending:** {stats.get('pending', 0)}\n"
                f"✅ **Completed:** {stats.get('completed', 0)}\n"
                f"⏹️ **Cancelled:** {stats.get('cancelled', 0)}\n"
                f"⚡ **Active Tasks:** {stats.get('active_tasks', 0)}\n\n"
                f"⏱️ **Delete Time:** {Config.AUTO_DELETE_TIME} minutes"
            )
            await message.reply_text(text)
        except Exception as e:
            await message.reply_text(f"❌ Error: {e}")
    
    logger.info("✅ Bot handlers setup complete")

# ============================================================================
# ✅ TELEGRAM BOT START FUNCTION
# ============================================================================

async def start_telegram_bot():
    """Start the Telegram bot with all handlers"""
    try:
        if not PYROGRAM_AVAILABLE:
            logger.warning("❌ Pyrogram not available")
            return None
        
        if not Config.BOT_TOKEN:
            logger.warning("❌ Bot token not configured")
            return None
        
        logger.info("🤖 Starting Telegram Bot...")
        
        # Ensure bot handler is initialized
        if not bot_handler.initialized:
            await bot_handler.initialize()
        
        # Create bot client
        bot = Client(
            "sk4film_bot",
            api_id=Config.API_ID,
            api_hash=Config.API_HASH,
            bot_token=Config.BOT_TOKEN
        )
        
        # Start bot
        await bot.start()
        logger.info("✅ Bot client started")
        
        # Setup handlers
        await setup_telegram_bot_handlers(bot)
        
        logger.info("✅ Telegram Bot fully initialized with handlers")
        return bot
        
    except Exception as e:
        logger.error(f"❌ Bot startup error: {e}")
        return None

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
    
    # Initialize USER
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
            
            # Test access to FILE_CHANNEL
            try:
                chat = await User.get_chat(Config.FILE_CHANNEL_ID)
                logger.info(f"✅ File Channel Access: {chat.title} (ID: {Config.FILE_CHANNEL_ID})")
                user_session_ready = True
            except Exception as e:
                logger.error(f"❌ File channel access failed for {Config.FILE_CHANNEL_ID}: {e}")
                user_session_ready = False
                
        except Exception as e:
            logger.error(f"❌ USER Session failed: {e}")
            user_session_ready = False
    
    logger.info("\n" + "=" * 50)
    logger.info("📊 SESSION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"BOT Session: {'✅ READY' if bot_session_ready else '❌ NOT READY'}")
    logger.info(f"USER Session: {'✅ READY' if user_session_ready else '❌ NOT READY'}")
    logger.info(f"Bot Handler: {'✅ INITIALIZED' if bot_handler.initialized else '❌ NOT READY'}")
    logger.info(f"File Channel ID: {Config.FILE_CHANNEL_ID}")
    logger.info(f"Main Channel ID: {Config.MAIN_CHANNEL_ID}")
    logger.info(f"Text Channels: {Config.TEXT_CHANNEL_IDS}")
    
    return bot_session_ready or user_session_ready

# ============================================================================
# ✅ MONGODB INITIALIZATION
# ============================================================================

@performance_monitor.measure("mongodb_init")
async def init_mongodb():
    global mongo_client, db, files_col, verification_col, thumbnails_col, posters_col
    
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
        thumbnails_col = db.thumbnails
        posters_col = db.posters

        # Safe index creation
        async def safe_index_creation():
            try:
                existing = await thumbnails_col.index_information()
                
                # Drop problematic unique index if exists
                if "normalized_title_1" in existing:
                    try:
                        await thumbnails_col.drop_index("normalized_title_1")
                        logger.info("✅ Dropped problematic unique index")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not drop index: {e}")
                
                # Handle duplicates
                pipeline = [
                    {"$group": {
                        "_id": "$normalized_title",
                        "count": {"$sum": 1},
                        "docs": {"$push": "$_id"}
                    }},
                    {"$match": {"count": {"$gt": 1}}}
                ]
                
                duplicates = await thumbnails_col.aggregate(pipeline).to_list(length=100)
                
                if duplicates:
                    logger.warning(f"⚠️ Found {len(duplicates)} duplicate titles")
                    for dup in duplicates:
                        keep_id = dup['docs'][0]
                        delete_ids = dup['docs'][1:]
                        if delete_ids:
                            await thumbnails_col.delete_many({"_id": {"$in": delete_ids}})
                            logger.info(f"🧹 Cleaned duplicates for: {dup['_id']}")
                
                # Create new indexes
                await thumbnails_col.create_index([("normalized_title", 1)], name="title_idx", background=True)
                await thumbnails_col.create_index([("has_thumbnail", 1)], name="has_thumbnail_idx", background=True)
                await thumbnails_col.create_index([("channel_id", 1), ("message_id", 1)], name="channel_msg_idx", background=True)
                
                logger.info("✅ Indexes created")
                
            except Exception as e:
                logger.error(f"❌ Index creation error: {e}")
        
        await safe_index_creation()
        
        # Poster indexes
        if posters_col is not None:
            try:
                await posters_col.create_index('cache_key', unique=True, sparse=True)
            except Exception as e:
                logger.warning(f"⚠️ Poster index error: {e}")

        logger.info("✅ MongoDB OK")
        return True
        
    except asyncio.TimeoutError:
        logger.error("❌ MongoDB connection timeout")
        return False
    except Exception as e:
        logger.error(f"❌ MongoDB error: {e}")
        return False

# ============================================================================
# ✅ POSTER FETCHER INITIALIZATION
# ============================================================================

async def init_poster_fetcher():
    """Initialize poster fetcher"""
    global poster_fetcher
    
    if not POSTER_FETCHER_AVAILABLE:
        logger.warning("⚠️ Poster fetcher module not available")
        return False
    
    if not Config.POSTER_FETCHING_ENABLED:
        logger.info("📌 Poster fetching is disabled in config")
        return False
    
    try:
        poster_fetcher = PosterFetcher(Config, cache_manager.redis_client if cache_manager else None)
        logger.info("✅ Poster Fetcher initialized")
        return True
    except Exception as e:
        logger.error(f"❌ Poster fetcher initialization failed: {e}")
        poster_fetcher = None
        return False

# ============================================================================
# ✅ THUMBNAIL MANAGER INITIALIZATION
# ============================================================================

async def init_thumbnail_manager():
    """Initialize Thumbnail Manager"""
    global thumbnail_manager
    
    thumbnail_manager = ThumbnailManager(
        mongodb=mongo_client,
        bot_client=Bot if bot_session_ready else None,
        user_client=User if user_session_ready else None,
        file_channel_id=Config.FILE_CHANNEL_ID
    )
    
    success = await thumbnail_manager.initialize()
    if success:
        logger.info("✅ Thumbnail Manager initialized")
    else:
        logger.error("❌ Thumbnail Manager initialization failed")
        thumbnail_manager = None
    
    return success

# ============================================================================
# ✅ INITIAL INDEXING FUNCTION
# ============================================================================

async def initial_indexing_optimized():
    """Start optimized file channel indexing"""
    global file_indexing_manager
    
    if not user_session_ready and not bot_session_ready:
        logger.error("❌ No Telegram session available")
        return
    
    if thumbnails_col is None:
        logger.error("❌ Database not ready")
        return
    
    logger.info("=" * 60)
    logger.info(f"🚀 STARTING MOVIE THUMBNAIL INDEXING - Channel: {Config.FILE_CHANNEL_ID}")
    logger.info("=" * 60)
    
    try:
        await file_indexing_manager.start_indexing(force_reindex=Config.INDEX_ALL_HISTORY)
        
        if user_session_ready:
            await sync_manager.start_sync_monitoring()
        
        logger.info("✅ Indexing started")
        
    except Exception as e:
        logger.error(f"❌ Initial indexing error: {e}")

# ============================================================================
# ✅ MAIN INITIALIZATION
# ============================================================================

@performance_monitor.measure("system_init")
async def init_system():
    start_time = time.time()
    
    try:
        logger.info("=" * 60)
        logger.info("🚀 SK4FiLM v9.5 - COMPLETE FIXED BOT")
        logger.info("=" * 60)
        
        # Initialize MongoDB
        mongo_ok = await init_mongodb()
        if not mongo_ok:
            logger.error("❌ MongoDB connection failed")
            return False
        
        # Initialize Bot Handler
        bot_handler_ok = await bot_handler.initialize()
        if bot_handler_ok:
            logger.info("✅ Bot Handler initialized")
        
        # Initialize Cache Manager
        global cache_manager, verification_system, premium_system
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
                logger.warning("⚠️ Telegram sessions partially failed")
        
        # Initialize Thumbnail Manager
        await init_thumbnail_manager()
        
        # Initialize Poster Fetcher
        await init_poster_fetcher()
        
        # START TELEGRAM BOT
        global telegram_bot
        telegram_bot = await start_telegram_bot()
        if telegram_bot:
            logger.info("✅ Telegram Bot started with all handlers")
        else:
            logger.warning("⚠️ Telegram Bot failed to start")
        
        # Start indexing (limited)
        if (user_session_ready or bot_session_ready) and thumbnails_col is not None:
            logger.info("🔍 Starting thumbnail indexing (limited to 500 messages)...")
            asyncio.create_task(initial_indexing_optimized())
        
        init_time = time.time() - start_time
        logger.info(f"⚡ SK4FiLM Started in {init_time:.2f}s")
        logger.info("=" * 60)
        logger.info("🔧 FEATURES:")
        logger.info(f"   • File Channel ID: {Config.FILE_CHANNEL_ID}")
        logger.info(f"   • Main Channel ID: {Config.MAIN_CHANNEL_ID}")
        logger.info(f"   • Poster Fetching: ✅ ENABLED")
        logger.info(f"   • File Sending: ✅ ENABLED")
        logger.info(f"   • Auto-Delete: ✅ {Config.AUTO_DELETE_TIME} minutes")
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

@app.route('/')
@performance_monitor.measure("root_endpoint")
async def root():
    if thumbnails_col is not None:
        movies_with_thumb = await thumbnails_col.count_documents({'has_thumbnail': True})
        movies_without_thumb = await thumbnails_col.count_documents({'has_thumbnail': False})
    else:
        movies_with_thumb = 0
        movies_without_thumb = 0
    
    if posters_col is not None:
        total_posters = await posters_col.count_documents({})
    else:
        total_posters = 0
    
    thumbnail_stats = {}
    if thumbnail_manager:
        thumbnail_stats = await thumbnail_manager.get_stats()
    
    indexing_status = await file_indexing_manager.get_indexing_status()
    
    bot_status = None
    if bot_handler:
        try:
            bot_status = await bot_handler.get_bot_status()
        except Exception as e:
            bot_status = {'initialized': False, 'error': str(e)}
    
    bot_running = telegram_bot is not None
    
    return jsonify({
        'status': 'healthy',
        'service': 'SK4FiLM v9.5 - COMPLETE FIXED BOT',
        'poster_fetching': Config.POSTER_FETCHING_ENABLED,
        'auto_delete_minutes': Config.AUTO_DELETE_TIME,
        'storage_stats': {
            'movies_with_thumbnails': movies_with_thumb,
            'movies_without_thumbnails': movies_without_thumb,
            'cached_posters': total_posters
        },
        'sessions': {
            'user_session': {'ready': user_session_ready},
            'bot_session': {'ready': bot_session_ready},
            'bot_handler': bot_status,
            'telegram_bot': {'running': bot_running}
        },
        'components': {
            'cache': cache_manager is not None,
            'verification': verification_system is not None,
            'premium': premium_system is not None,
            'database': thumbnails_col is not None,
            'thumbnail_manager': thumbnail_manager is not None,
            'poster_fetcher': poster_fetcher is not None,
            'bot_handler': bot_handler is not None and bot_handler.initialized
        },
        'indexing': indexing_status,
        'sync_monitoring': {
            'running': sync_manager.is_monitoring,
            'deleted_count': sync_manager.deleted_count
        },
        'channel_config': {
            'file_channel_id': Config.FILE_CHANNEL_ID,
            'main_channel_id': Config.MAIN_CHANNEL_ID,
            'text_channels': Config.TEXT_CHANNEL_IDS
        },
        'response_time': f"{time.perf_counter():.3f}s"
    })

@app.route('/health')
async def health():
    return jsonify({
        'status': 'ok',
        'sessions': {
            'user': user_session_ready,
            'bot': bot_session_ready,
            'bot_handler': bot_handler.initialized if bot_handler else False,
            'telegram_bot': telegram_bot is not None
        },
        'poster_fetching': Config.POSTER_FETCHING_ENABLED,
        'auto_delete_minutes': Config.AUTO_DELETE_TIME,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/movies', methods=['GET'])
async def api_movies():
    try:
        movies = await get_home_movies(limit=25)
        
        return jsonify({
            'status': 'success' if movies else 'empty',
            'movies': movies,
            'total': len(movies),
            'limit': 25,
            'source': 'telegram',
            'poster_fetching': Config.POSTER_FETCHING_ENABLED,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Movies API error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/search', methods=['GET'])
async def api_search():
    """Fast search with one thumbnail per movie"""
    try:
        query = request.args.get('query', '').strip()
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', Config.SEARCH_RESULTS_PER_PAGE))
        
        if len(query) < Config.SEARCH_MIN_QUERY_LENGTH:
            return jsonify({
                'status': 'error',
                'message': f'Query must be at least {Config.SEARCH_MIN_QUERY_LENGTH} characters'
            }), 400
        
        result_data = await search_movies_optimized(query, limit, page)
        
        return jsonify({
            'status': 'success',
            'query': query,
            'results': result_data['results'],
            'pagination': result_data['pagination'],
            'search_metadata': result_data.get('search_metadata', {}),
            'bot_username': Config.BOT_USERNAME,
            'poster_fetching': Config.POSTER_FETCHING_ENABLED,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Search API error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
async def api_stats():
    try:
        perf_stats = performance_monitor.get_stats()
        
        if thumbnails_col is not None:
            movies_with_thumb = await thumbnails_col.count_documents({'has_thumbnail': True})
            indexing_status = await file_indexing_manager.get_indexing_status()
        else:
            movies_with_thumb = 0
            indexing_status = {}
        
        if posters_col is not None:
            total_posters = await posters_col.count_documents({})
        else:
            total_posters = 0
        
        thumbnail_stats = {}
        if thumbnail_manager:
            thumbnail_stats = await thumbnail_manager.get_stats()
        
        poster_stats = {}
        if poster_fetcher and hasattr(poster_fetcher, 'get_stats'):
            poster_stats = poster_fetcher.get_stats()
        
        bot_status = None
        if bot_handler:
            try:
                bot_status = await bot_handler.get_bot_status()
            except:
                bot_status = {'initialized': False}
        
        return jsonify({
            'status': 'success',
            'performance': perf_stats,
            'thumbnail_manager': thumbnail_stats,
            'poster_fetcher': poster_stats,
            'database_stats': {
                'movies_with_thumbnails': movies_with_thumb,
                'cached_posters': total_posters
            },
            'indexing_stats': indexing_status,
            'sync_stats': {
                'running': sync_manager.is_monitoring,
                'deleted_count': sync_manager.deleted_count
            },
            'bot_handler': bot_status,
            'poster_fetching': Config.POSTER_FETCHING_ENABLED,
            'auto_delete_minutes': Config.AUTO_DELETE_TIME,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Stats API error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/debug/channel-info')
async def debug_channel_info():
    """Debug endpoint to check channel access"""
    try:
        result = {
            'file_channel_id': Config.FILE_CHANNEL_ID,
            'main_channel_id': Config.MAIN_CHANNEL_ID,
            'text_channels': Config.TEXT_CHANNEL_IDS,
            'user_session_ready': user_session_ready,
            'bot_session_ready': bot_session_ready,
            'bot_handler_initialized': bot_handler.initialized if bot_handler else False
        }
        
        if user_session_ready and User:
            try:
                # Try to get file channel info
                file_chat = await User.get_chat(Config.FILE_CHANNEL_ID)
                result['file_channel'] = {
                    'title': file_chat.title,
                    'id': file_chat.id,
                    'type': str(file_chat.type),
                    'username': file_chat.username
                }
                
                # Try to get a sample message
                messages = []
                async for msg in User.get_chat_history(Config.FILE_CHANNEL_ID, limit=5):
                    if msg.document or msg.video:
                        messages.append({
                            'id': msg.id,
                            'type': 'document' if msg.document else 'video',
                            'file_name': msg.document.file_name if msg.document else (msg.video.file_name or 'video.mp4'),
                            'has_thumbnail': has_telegram_thumbnail(msg)
                        })
                
                result['sample_messages'] = messages
                
            except Exception as e:
                result['file_channel_error'] = str(e)
        
        # Bot info
        if bot_handler and bot_handler.initialized:
            try:
                bot_info = await bot_handler.get_bot_status()
                result['bot_info'] = bot_info
            except:
                pass
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# ✅ STARTUP AND SHUTDOWN
# ============================================================================

app_start_time = time.time()

@app.before_serving
async def startup():
    await init_system()

@app.after_serving
async def shutdown():
    logger.info("🛑 Shutting down SK4FiLM v9.5...")
    
    if telegram_bot:
        try:
            await telegram_bot.stop()
            logger.info("✅ Telegram bot stopped")
        except:
            pass
    
    await file_indexing_manager.stop_indexing()
    await sync_manager.stop_sync_monitoring()
    
    if thumbnail_manager:
        await thumbnail_manager.shutdown()
    
    if bot_handler:
        await bot_handler.shutdown()
    
    if poster_fetcher and hasattr(poster_fetcher, 'close'):
        await poster_fetcher.close()
    
    if User is not None:
        await User.stop()
    
    if Bot is not None:
        await Bot.stop()
    
    if cache_manager is not None:
        await cache_manager.stop()
    
    if verification_system is not None:
        await verification_system.stop()
    
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
    
    logger.info(f"🌐 Starting SK4FiLM v9.5 on port {Config.WEB_SERVER_PORT}...")
    logger.info(f"📁 File Channel ID: {Config.FILE_CHANNEL_ID}")
    logger.info(f"🤖 Bot Token: {'✅ Configured' if Config.BOT_TOKEN else '❌ Missing'}")
    logger.info(f"⏰ Auto-Delete Time: {Config.AUTO_DELETE_TIME} minutes")
    logger.info(f"🎬 Poster Fetching: {'ENABLED' if Config.POSTER_FETCHING_ENABLED else 'DISABLED'}")
    logger.info(f"🖼️ ONE Thumbnail Per Movie: ENABLED")
    logger.info(f"🎞️ All Qualities Shown: ENABLED")
    logger.info(f"📤 File Sending: ENABLED")
    logger.info(f"🔄 Auto-indexing: LIMITED (500 messages)")
    
    try:
        asyncio.run(serve(app, config))
    except KeyboardInterrupt:
        logger.info("👋 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
