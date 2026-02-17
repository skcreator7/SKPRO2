# ============================================================================
# 🚀 SK4FiLM v9.0 - ZERO CPU, TELEGRAM METADATA ONLY, BASE64 STORAGE
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
    
    # URL Shortener
    SHORTLINK_API = os.environ.get("SHORTLINK_API", "")
    CUTTLY_API = os.environ.get("CUTTLY_API", "")
    
    # UPI IDs
    UPI_ID_BASIC = os.environ.get("UPI_ID_BASIC", "cf.sk4film@cashfreensdlpb")
    UPI_ID_PREMIUM = os.environ.get("UPI_ID_PREMIUM", "cf.sk4film@cashfreensdlpb")
    UPI_ID_GOLD = os.environ.get("UPI_ID_GOLD", "cf.sk4film@cashfreensdlpb")
    UPI_ID_DIAMOND = os.environ.get("UPI_ID_DIAMOND", "cf.sk4film@cashfreensdlpb")
    
    # Verification
    VERIFICATION_REQUIRED = os.environ.get("VERIFICATION_REQUIRED", "true").lower() == "False"
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
    
    # Performance Settings
    MAX_CONCURRENT_REQUESTS = int(os.environ.get("MAX_CONCURRENT_REQUESTS", "50"))
    CACHE_TTL = int(os.environ.get("CACHE_TTL", "300"))
    REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "10"))
    
    # Sync Management Settings
    MONITOR_INTERVAL = int(os.environ.get("MONITOR_INTERVAL", "300"))
    
    # Quality Settings
    QUALITY_PRIORITY = ['2160p', '1080p', '720p', '480p', '360p']
    HEVC_VARIANTS = ['720p HEVC', '1080p HEVC', '2160p HEVC']
    
    # 🔥 OPTIMIZED SETTINGS - ZERO CPU, METADATA ONLY
    THUMBNAIL_TTL_DAYS = int(os.environ.get("THUMBNAIL_TTL_DAYS", "30"))  # Auto cleanup after 30 days
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "5"))  # Rate limiting safe
    EXTRACT_DELAY = int(os.environ.get("EXTRACT_DELAY", "1"))  # 1 second between extractions
    
    # 🔥 SEARCH SETTINGS
    SEARCH_MIN_QUERY_LENGTH = 2
    SEARCH_RESULTS_PER_PAGE = 12
    SEARCH_CACHE_TTL = 300  # 5 minutes cache

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
    response.headers['X-SK4FiLM-Version'] = '9.0-ZERO-CPU'
    response.headers['X-Response-Time'] = f"{time.perf_counter():.3f}"
    return response

# ============================================================================
# ✅ GLOBAL COMPONENTS
# ============================================================================

# Database
mongo_client = None
db = None
thumbnails_col = None  # Only thumbnails stored
verification_col = None

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

# ============================================================================
# ✅ BOT HANDLER MODULE
# ============================================================================

class BotHandler:
    """Bot handler for Telegram bot operations - NO FFMPEG, METADATA ONLY"""
    
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
            
            asyncio.create_task(self._periodic_tasks())
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
    
    async def extract_thumbnail_metadata(self, channel_id, message_id):
        """
        🔥 EXTRACT THUMBNAIL FROM TELEGRAM METADATA ONLY
        - NO FFMPEG
        - NO CPU USAGE
        - Uses Telegram's built-in thumbnail
        - Returns Base64 (~50KB per file)
        """
        if not self.initialized:
            return None
        
        try:
            message = await self.bot.get_messages(channel_id, message_id)
            if not message:
                return None
            
            thumbnail_data = None
            
            # Get thumbnail from video metadata (Telegram already has it)
            if message.video:
                if hasattr(message.video, 'thumbnail') and message.video.thumbnail:
                    thumbnail_file_id = message.video.thumbnail.file_id
                    logger.debug(f"🎬 Got thumbnail metadata for video: {message.video.file_id[:20]}...")
                    
                    # Download thumbnail (Telegram's built-in thumbnail, small size)
                    download_path = await self.bot.download_media(thumbnail_file_id, in_memory=True)
                    if download_path:
                        if isinstance(download_path, bytes):
                            thumbnail_data = download_path
                        else:
                            with open(download_path, 'rb') as f:
                                thumbnail_data = f.read()
            
            # Get thumbnail from document metadata
            elif message.document and is_video_file(message.document.file_name or ''):
                if hasattr(message.document, 'thumbnail') and message.document.thumbnail:
                    thumbnail_file_id = message.document.thumbnail.file_id
                    logger.debug(f"📄 Got thumbnail metadata for document: {message.document.file_name}")
                    
                    download_path = await self.bot.download_media(thumbnail_file_id, in_memory=True)
                    if download_path:
                        if isinstance(download_path, bytes):
                            thumbnail_data = download_path
                        else:
                            with open(download_path, 'rb') as f:
                                thumbnail_data = f.read()
            
            if thumbnail_data:
                # Convert to base64 for instant display (no processing needed)
                size_kb = len(thumbnail_data) / 1024
                logger.debug(f"✅ Thumbnail size: {size_kb:.1f}KB")
                
                base64_data = base64.b64encode(thumbnail_data).decode('utf-8')
                return f"data:image/jpeg;base64,{base64_data}"
            
            logger.debug(f"⚠️ No thumbnail metadata for message {message_id}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Extract thumbnail error: {e}")
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
            return {
                'initialized': True,
                'bot_username': bot_info.username,
                'bot_id': bot_info.id,
                'first_name': bot_info.first_name,
                'last_update': self.last_update.isoformat() if self.last_update else None,
                'is_connected': True
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
        self.initialized = False
        if self.bot:
            try:
                await self.bot.stop()
                logger.info("✅ Bot stopped")
            except Exception as e:
                logger.error(f"❌ Error stopping bot: {e}")

bot_handler = BotHandler()

# ============================================================================
# ✅ THUMBNAIL EXTRACTOR - ZERO CPU, METADATA ONLY
# ============================================================================

class ThumbnailExtractor:
    """
    🔥 EXTRACTS THUMBNAILS FROM TELEGRAM METADATA
    - NO FFMPEG
    - ZERO CPU USAGE
    - 50KB per file average
    - Base64 storage for instant display
    - Rate limited for Telegram safety
    - Auto cleanup after 30 days
    """
    
    def __init__(self):
        self.is_running = False
        self.extraction_task = None
        self.extracted_count = 0
        self.failed_count = 0
        self.total_size_kb = 0
        self.extraction_stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'total_size_kb': 0,
            'avg_size_kb': 0,
            'last_extraction': None,
            'pending': 0
        }
        self.rate_limit_lock = asyncio.Lock()
    
    async def start_extraction(self, scan_all=True):
        """Start thumbnail extraction - ZERO CPU, METADATA ONLY"""
        if self.is_running:
            logger.warning("⚠️ Extraction already running")
            return
        
        logger.info("=" * 60)
        logger.info("🚀 STARTING THUMBNAIL EXTRACTION - ZERO CPU, METADATA ONLY")
        logger.info("=" * 60)
        logger.info("📊 Features:")
        logger.info("   • NO FFmpeg - Zero CPU usage")
        logger.info("   • Telegram metadata only - ~50KB/file")
        logger.info("   • Base64 storage - Instant display")
        logger.info("   • Rate limiting - Telegram safe")
        logger.info("   • Auto cleanup - 30 days TTL")
        logger.info("=" * 60)
        
        self.is_running = True
        
        try:
            if scan_all:
                await self._extract_all_thumbnails()
            else:
                await self._extract_pending_thumbnails()
            
            # Start background monitoring for new files
            self.extraction_task = asyncio.create_task(self._monitor_new_files())
            
            # Start auto cleanup task
            asyncio.create_task(self._auto_cleanup_loop())
            
        except Exception as e:
            logger.error(f"❌ Extraction error: {e}")
            self.extraction_stats['last_error'] = str(e)
    
    async def stop_extraction(self):
        """Stop thumbnail extraction"""
        self.is_running = False
        if self.extraction_task:
            self.extraction_task.cancel()
            try:
                await self.extraction_task
            except asyncio.CancelledError:
                pass
        logger.info("🛑 Thumbnail extraction stopped")
    
    async def _extract_all_thumbnails(self):
        """Extract thumbnails from ALL files - METADATA ONLY"""
        if not user_session_ready and not bot_session_ready:
            logger.error("❌ No Telegram session available")
            return
        
        client = User if user_session_ready else Bot
        if not client:
            return
        
        logger.info("📥 Fetching ALL messages from file channel (metadata only)...")
        
        all_messages = []
        offset_id = 0
        batch_size = 200
        
        while self.is_running:
            try:
                messages = []
                async for msg in client.get_chat_history(
                    Config.FILE_CHANNEL_ID,
                    limit=batch_size,
                    offset_id=offset_id
                ):
                    messages.append(msg)
                    if len(messages) >= batch_size:
                        break
                
                if not messages:
                    break
                
                all_messages.extend(messages)
                offset_id = messages[-1].id
                logger.info(f"📥 Fetched {len(all_messages)} messages (metadata only)...")
                
                if len(messages) < batch_size:
                    break
                
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"❌ Error fetching messages: {e}")
                await asyncio.sleep(2)
                continue
        
        logger.info(f"✅ Total messages fetched: {len(all_messages)}")
        
        # Filter video files
        video_files = []
        for msg in all_messages:
            if not msg or (not msg.document and not msg.video):
                continue
            
            file_name = None
            if msg.document:
                file_name = msg.document.file_name
            elif msg.video:
                file_name = msg.video.file_name or "video.mp4"
            
            if file_name and is_video_file(file_name):
                video_files.append(msg)
        
        logger.info(f"🎬 Found {len(video_files)} video files")
        
        # Extract thumbnails in batches with rate limiting
        await self._extract_from_messages(video_files)
    
    async def _extract_pending_thumbnails(self):
        """Extract thumbnails for pending files only"""
        if thumbnails_col is None:
            return
        
        # Find already processed message IDs
        processed = set()
        cursor = thumbnails_col.find({}, {"message_id": 1})
        async for doc in cursor:
            processed.add(doc['message_id'])
        
        if not user_session_ready and not bot_session_ready:
            return
        
        client = User if user_session_ready else Bot
        if not client:
            return
        
        logger.info("🔍 Finding new files for thumbnail extraction...")
        
        all_messages = []
        offset_id = 0
        batch_size = 200
        
        while self.is_running:
            try:
                messages = []
                async for msg in client.get_chat_history(
                    Config.FILE_CHANNEL_ID,
                    limit=batch_size,
                    offset_id=offset_id
                ):
                    messages.append(msg)
                    if len(messages) >= batch_size:
                        break
                
                if not messages:
                    break
                
                all_messages.extend(messages)
                offset_id = messages[-1].id
                
                if len(messages) < batch_size:
                    break
                
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"❌ Error fetching messages: {e}")
                await asyncio.sleep(2)
                continue
        
        # Filter unprocessed video files
        pending_files = []
        for msg in all_messages:
            if msg.id in processed:
                continue
            
            if not msg or (not msg.document and not msg.video):
                continue
            
            file_name = None
            if msg.document:
                file_name = msg.document.file_name
            elif msg.video:
                file_name = msg.video.file_name or "video.mp4"
            
            if file_name and is_video_file(file_name):
                pending_files.append(msg)
        
        self.extraction_stats['pending'] = len(pending_files)
        logger.info(f"📊 Found {len(pending_files)} pending files for extraction")
        
        if pending_files:
            await self._extract_from_messages(pending_files)
    
    async def _extract_from_messages(self, messages):
        """
        🔥 EXTRACT THUMBNAILS - ZERO CPU, METADATA ONLY
        - Uses Telegram's built-in thumbnails
        - No FFmpeg processing
        - Rate limited (1 second between extractions)
        - Base64 storage for instant display
        """
        batch_size = Config.BATCH_SIZE
        total_batches = math.ceil(len(messages) / batch_size)
        
        client = User if user_session_ready else Bot
        
        for batch_num in range(total_batches):
            if not self.is_running:
                break
            
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(messages))
            batch = messages[start_idx:end_idx]
            
            logger.info(f"🖼️ Processing batch {batch_num + 1}/{total_batches}...")
            
            for msg in batch:
                async with self.rate_limit_lock:
                    try:
                        # Extract file info
                        file_name = None
                        if msg.document:
                            file_name = msg.document.file_name
                        elif msg.video:
                            file_name = msg.video.file_name or "video.mp4"
                        
                        clean_title = extract_clean_title(file_name)
                        normalized = normalize_title(clean_title)
                        quality = detect_quality_enhanced(file_name)
                        year = extract_year(file_name)
                        
                        # Extract thumbnail from Telegram metadata (NO FFMPEG)
                        thumbnail_url = None
                        
                        if bot_handler and bot_handler.initialized:
                            thumbnail_url = await bot_handler.extract_thumbnail_metadata(
                                Config.FILE_CHANNEL_ID,
                                msg.id
                            )
                        
                        if thumbnail_url and thumbnails_col is not None:
                            # Calculate size
                            size_kb = len(thumbnail_url) / 1024
                            self.total_size_kb += size_kb
                            
                            # Store in MongoDB (Base64 format)
                            thumbnail_doc = {
                                'normalized_title': normalized,
                                'title': clean_title,
                                'quality': quality,
                                'year': year,
                                'thumbnail_url': thumbnail_url,  # Base64 data
                                'thumbnail_source': 'telegram_metadata',
                                'extracted_at': datetime.now(),
                                'message_id': msg.id,
                                'channel_id': Config.FILE_CHANNEL_ID,
                                'file_name': file_name,
                                'file_size': msg.document.file_size if msg.document else msg.video.file_size,
                                'size_kb': size_kb,
                                'expires_at': datetime.now() + timedelta(days=Config.THUMBNAIL_TTL_DAYS)
                            }
                            
                            await thumbnails_col.update_one(
                                {
                                    'normalized_title': normalized,
                                    'quality': quality
                                },
                                {'$set': thumbnail_doc},
                                upsert=True
                            )
                            
                            self.extracted_count += 1
                            self.extraction_stats['successful'] += 1
                            self.extraction_stats['total_size_kb'] += size_kb
                            logger.info(f"✅ Extracted: {clean_title} - {quality} ({size_kb:.1f}KB)")
                            
                        else:
                            self.failed_count += 1
                            self.extraction_stats['failed'] += 1
                            logger.warning(f"⚠️ No thumbnail metadata: {clean_title} - {quality}")
                        
                        self.extraction_stats['total_processed'] += 1
                        
                    except Exception as e:
                        logger.error(f"❌ Extraction error: {e}")
                        self.failed_count += 1
                        self.extraction_stats['failed'] += 1
                        self.extraction_stats['total_processed'] += 1
                    
                    # Rate limiting - Telegram safe
                    await asyncio.sleep(Config.EXTRACT_DELAY)
            
            if batch_num < total_batches - 1:
                await asyncio.sleep(2)
        
        # Calculate average size
        if self.extraction_stats['successful'] > 0:
            self.extraction_stats['avg_size_kb'] = self.extraction_stats['total_size_kb'] / self.extraction_stats['successful']
        
        self.extraction_stats['last_extraction'] = datetime.now()
        
        logger.info("=" * 60)
        logger.info("📊 EXTRACTION COMPLETE - ZERO CPU, METADATA ONLY")
        logger.info(f"   • Processed: {self.extraction_stats['total_processed']}")
        logger.info(f"   • Successful: {self.extraction_stats['successful']}")
        logger.info(f"   • Failed: {self.extraction_stats['failed']}")
        logger.info(f"   • Total size: {self.extraction_stats['total_size_kb']:.1f}KB")
        logger.info(f"   • Average size: {self.extraction_stats['avg_size_kb']:.1f}KB per file")
        logger.info("=" * 60)
    
    async def _monitor_new_files(self):
        """Monitor for new files and extract automatically"""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                if not self.is_running or thumbnails_col is None:
                    continue
                
                # Find latest message ID in database
                latest = await thumbnails_col.find_one(
                    {'channel_id': Config.FILE_CHANNEL_ID},
                    sort=[('message_id', -1)]
                )
                last_message_id = latest.get('message_id', 0) if latest else 0
                
                # Fetch new messages
                client = User if user_session_ready else Bot
                if not client:
                    continue
                
                new_messages = []
                async for msg in client.get_chat_history(
                    Config.FILE_CHANNEL_ID,
                    limit=50
                ):
                    if msg.id > last_message_id:
                        new_messages.append(msg)
                    else:
                        break
                
                if new_messages:
                    # Filter video files
                    video_files = []
                    for msg in new_messages:
                        if not msg or (not msg.document and not msg.video):
                            continue
                        
                        file_name = None
                        if msg.document:
                            file_name = msg.document.file_name
                        elif msg.video:
                            file_name = msg.video.file_name or "video.mp4"
                        
                        if file_name and is_video_file(file_name):
                            video_files.append(msg)
                    
                    if video_files:
                        logger.info(f"🆕 Found {len(video_files)} new video files")
                        await self._extract_from_messages(video_files)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Monitor error: {e}")
                await asyncio.sleep(120)
    
    async def _auto_cleanup_loop(self):
        """
        🔥 AUTO CLEANUP - 30 DAYS TTL
        - Automatically removes expired thumbnails
        - Runs daily
        """
        while self.is_running:
            try:
                await asyncio.sleep(24 * 60 * 60)  # Run daily
                
                if thumbnails_col is None:
                    continue
                
                logger.info("🧹 Running auto cleanup for expired thumbnails...")
                
                # Delete expired thumbnails
                result = await thumbnails_col.delete_many({
                    'expires_at': {'$lt': datetime.now()}
                })
                
                if result.deleted_count > 0:
                    logger.info(f"✅ Auto cleanup removed {result.deleted_count} expired thumbnails")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Auto cleanup error: {e}")
    
    async def get_stats(self):
        """Get extraction statistics"""
        total_thumbnails = 0
        total_size_mb = 0
        if thumbnails_col is not None:
            total_thumbnails = await thumbnails_col.count_documents({})
            
            # Calculate total size
            pipeline = [
                {'$group': {
                    '_id': None,
                    'total_size_kb': {'$sum': '$size_kb'}
                }}
            ]
            result = await thumbnails_col.aggregate(pipeline).to_list(1)
            if result:
                total_size_mb = result[0]['total_size_kb'] / 1024
        
        return {
            'is_running': self.is_running,
            'extracted_count': self.extracted_count,
            'failed_count': self.failed_count,
            'total_thumbnails': total_thumbnails,
            'total_storage_mb': total_size_mb,
            'avg_size_kb': self.extraction_stats['avg_size_kb'],
            'stats': self.extraction_stats,
            'features': {
                'ffmpeg_used': False,
                'cpu_usage': 'zero',
                'storage_type': 'base64',
                'avg_file_size_kb': '~50',
                'ttl_days': Config.THUMBNAIL_TTL_DAYS,
                'rate_limited': True
            }
        }

thumbnail_extractor = ThumbnailExtractor()

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

def extract_clean_title(filename):
    """Extract clean movie title without quality tags, year, etc."""
    if not filename:
        return "Unknown"
    
    name = os.path.splitext(filename)[0]
    name = re.sub(r'[._\-]', ' ', name)
    name = re.sub(r'\b(480p|720p|1080p|2160p|4k|hd|hevc|x264|x265|web-dl|webrip|bluray|hdtv|hdr|dts|ac3|aac|ddp|5\.1|7\.1|2\.0|esub|sub|multi|dual|audio|hindi|english|tamil|telugu|malayalam|kannada|ben|eng|hin|tam|tel|mal|kan)\b.*$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s+(19|20)\d{2}\s*$', '', name)
    name = re.sub(r'\s*\([^)]*\)', '', name)
    name = re.sub(r'\s*\[[^\]]*\]', '', name)
    name = re.sub(r'\s+', ' ', name)
    name = name.strip()
    
    return name if name else "Unknown"

def extract_year(filename):
    if not filename:
        return ""
    year_match = re.search(r'\b(19|20)\d{2}\b', filename)
    return year_match.group() if year_match else ""

def is_video_file(filename):
    if not filename:
        return False
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v']
    return any(filename.lower().endswith(ext) for ext in video_extensions)

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

def channel_name_cached(cid):
    return f"Channel {cid}"

# ============================================================================
# ✅ OPTIMIZED SEARCH - DIRECT TELEGRAM + MONGODB THUMBNAILS
# ============================================================================

@performance_monitor.measure("optimized_search")
@async_cache_with_ttl(maxsize=500, ttl=300)
async def search_movies_optimized(query, limit=15, page=1):
    """
    🔥 OPTIMIZED SEARCH:
    - Direct Telegram search
    - MongoDB thumbnails only
    - Results priority: Post+Files > Post only > File only
    """
    offset = (page - 1) * limit
    
    logger.info(f"🔍 OPTIMIZED SEARCH for: '{query}'")
    
    results_dict = {}
    
    # STEP 1: Search FILE CHANNEL (Files)
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
                
                file_data = {
                    'quality': quality,
                    'file_name': file_name,
                    'file_size': msg.document.file_size if msg.document else msg.video.file_size,
                    'file_size_formatted': format_size(msg.document.file_size if msg.document else msg.video.file_size),
                    'message_id': msg.id,
                    'file_id': msg.document.file_id if msg.document else msg.video.file_id,
                    'date': msg.date
                }
                
                if normalized not in results_dict:
                    results_dict[normalized] = {
                        'title': clean_title,
                        'original_title': clean_title,
                        'normalized_title': normalized,
                        'year': year,
                        'channel_id': Config.FILE_CHANNEL_ID,
                        'qualities': {},
                        'available_qualities': [],
                        'has_file': True,
                        'has_post': False,
                        'result_type': 'file',
                        'date': msg.date,
                        'is_new': is_new(msg.date),
                        'thumbnail_url': None,
                        'thumbnail_source': None,
                        'has_thumbnail': False,
                        'poster_url': None,
                        'has_poster': False,
                        'search_score': 5
                    }
                
                results_dict[normalized]['qualities'][quality] = file_data
                results_dict[normalized]['available_qualities'].append(quality)
                
                if msg.date and (not results_dict[normalized].get('date') or msg.date > results_dict[normalized]['date']):
                    results_dict[normalized]['date'] = msg.date
                    results_dict[normalized]['is_new'] = is_new(msg.date)
                
                file_count += 1
            
            logger.info(f"📁 Found {file_count} file results")
            
        except Exception as e:
            logger.error(f"❌ File channel search error: {e}")
    
    # STEP 2: Search TEXT CHANNELS (Posts)
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
                            # Upgrade to Post+Files
                            results_dict[normalized]['has_post'] = True
                            results_dict[normalized]['post_content'] = format_post(msg.text, max_length=500)
                            results_dict[normalized]['post_channel_id'] = channel_id
                            results_dict[normalized]['post_message_id'] = msg.id
                            results_dict[normalized]['result_type'] = 'file_and_post'
                            results_dict[normalized]['search_score'] = 10
                        else:
                            # Post only
                            results_dict[normalized] = {
                                'title': clean_title,
                                'original_title': title,
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
                                'thumbnail_url': None,
                                'thumbnail_source': None,
                                'has_thumbnail': False,
                                'poster_url': None,
                                'has_poster': False,
                                'search_score': 7
                            }
                        
                        post_count += 1
                        
                except Exception as e:
                    logger.debug(f"Text search error in {channel_id}: {e}")
                    continue
            
            logger.info(f"📝 Found {post_count} post results")
            
        except Exception as e:
            logger.error(f"❌ Text channels search error: {e}")
    
    # STEP 3: Get Thumbnails from MongoDB (Fast indexes)
    if thumbnails_col is not None and results_dict:
        logger.info(f"🖼️ Getting thumbnails from MongoDB (fast indexes)...")
        
        normalized_titles = list(results_dict.keys())
        
        # Use MongoDB indexes for fast lookup
        cursor = thumbnails_col.find(
            {
                'normalized_title': {'$in': normalized_titles}
            },
            {
                'normalized_title': 1,
                'thumbnail_url': 1,
                'thumbnail_source': 1,
                'quality': 1
            }
        ).hint([('normalized_title', 1)])  # Use index
        
        thumbnails_map = {}
        async for doc in cursor:
            if doc['normalized_title'] not in thumbnails_map:
                thumbnails_map[doc['normalized_title']] = doc
        
        for normalized, result in results_dict.items():
            if normalized in thumbnails_map:
                thumb_data = thumbnails_map[normalized]
                result['thumbnail_url'] = thumb_data['thumbnail_url']
                result['thumbnail_source'] = 'telegram_metadata'
                result['has_thumbnail'] = True
    
    # STEP 4: Get Posters for Results Without Thumbnails
    if poster_fetcher:
        for normalized, result in results_dict.items():
            if not result.get('has_thumbnail'):
                try:
                    poster_data = await get_poster_for_movie(
                        result.get('title', ''),
                        result.get('year', '')
                    )
                    
                    if poster_data and poster_data.get('poster_url'):
                        result['poster_url'] = poster_data['poster_url']
                        result['poster_source'] = poster_data.get('source')
                        result['has_poster'] = True
                        result['thumbnail_url'] = poster_data['poster_url']
                        result['thumbnail_source'] = poster_data.get('source')
                        result['has_thumbnail'] = True
                        
                except Exception as e:
                    logger.debug(f"Poster error: {e}")
    
    # STEP 5: Apply Fallback
    FALLBACK_THUMBNAIL_URL_GLOBAL = "https://iili.io/fAeIwv9.th.png"
    for result in results_dict.values():
        if not result.get('has_thumbnail'):
            result['thumbnail_url'] = FALLBACK_THUMBNAIL_URL_GLOBAL
            result['thumbnail_source'] = 'fallback'
            result['has_thumbnail'] = True
    
    # STEP 6: Sort by Priority
    all_results = list(results_dict.values())
    
    all_results.sort(key=lambda x: (
        x.get('search_score', 0),
        x.get('is_new', False),
        x.get('date') if isinstance(x.get('date'), datetime) else datetime.min
    ), reverse=True)
    
    # STEP 7: Pagination
    total = len(all_results)
    start_idx = offset
    end_idx = offset + limit
    paginated = all_results[start_idx:end_idx]
    
    # Statistics
    file_count = sum(1 for r in all_results if r.get('has_file'))
    post_count = sum(1 for r in all_results if r.get('has_post'))
    combined_count = sum(1 for r in all_results if r.get('has_file') and r.get('has_post'))
    thumbnail_count = sum(1 for r in all_results if r.get('thumbnail_source') == 'telegram_metadata')
    poster_count = sum(1 for r in all_results if r.get('thumbnail_source') in ['tmdb', 'omdb', 'poster'])
    
    logger.info("=" * 60)
    logger.info("📊 SEARCH RESULTS SUMMARY:")
    logger.info(f"   • Query: '{query}'")
    logger.info(f"   • Total results: {total}")
    logger.info(f"   • Post+Files (Priority 1): {combined_count}")
    logger.info(f"   • Post only (Priority 2): {post_count - combined_count}")
    logger.info(f"   • File only (Priority 3): {file_count - combined_count}")
    logger.info(f"   • Telegram metadata thumbnails: {thumbnail_count}")
    logger.info(f"   • Posters: {poster_count}")
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
            'has_previous': page > 1
        },
        'search_metadata': {
            'query': query,
            'total_results': total,
            'file_results': file_count,
            'post_results': post_count,
            'combined_results': combined_count,
            'extracted_thumbnails': thumbnail_count,
            'posters': poster_count,
            'mode': 'direct_telegram_search'
        },
        'bot_username': Config.BOT_USERNAME
    }

# ============================================================================
# ✅ POSTER FETCHING FUNCTIONS
# ============================================================================

async def get_poster_for_movie(title: str, year: str = "", quality: str = "") -> Dict[str, Any]:
    """Get poster for movie - Returns empty string if not found"""
    global poster_fetcher
    
    if poster_fetcher is None:
        return {
            'poster_url': '',
            'source': 'none',
            'rating': '0.0',
            'year': year,
            'title': title,
            'quality': quality or 'unknown',
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
                'quality': quality or 'unknown',
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
            'quality': quality or 'unknown',
            'found': False
        }

# ============================================================================
# ✅ HOME MOVIES
# ============================================================================

@performance_monitor.measure("home_movies")
@async_cache_with_ttl(maxsize=1, ttl=60)
async def get_home_movies(limit=25):
    """Get home movies - Poster first, then extracted, then fallback"""
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
                    norm_title = normalize_title(clean_title)
                    
                    # Get POSTER first
                    poster_data = None
                    if poster_fetcher:
                        try:
                            poster_data = await get_poster_for_movie(clean_title, year)
                        except Exception as e:
                            logger.error(f"❌ Poster fetch error: {e}")
                    
                    # Get EXTRACTED thumbnail from MongoDB
                    extracted_thumb = None
                    extracted_source = None
                    if thumbnails_col is not None:
                        try:
                            thumb_doc = await thumbnails_col.find_one(
                                {'normalized_title': norm_title}
                            )
                            if thumb_doc and thumb_doc.get('thumbnail_url'):
                                extracted_thumb = thumb_doc['thumbnail_url']
                                extracted_source = 'telegram_metadata'
                        except Exception as e:
                            logger.error(f"❌ Thumbnail fetch error: {e}")
                    
                    # Determine thumbnail: Poster > Extracted > Fallback
                    FALLBACK_THUMBNAIL_URL_GLOBAL = "https://iili.io/fAeIwv9.th.png"
                    
                    if poster_data and poster_data.get('poster_url'):
                        thumbnail_url = poster_data['poster_url']
                        thumbnail_source = poster_data.get('source', 'poster')
                    elif extracted_thumb:
                        thumbnail_url = extracted_thumb
                        thumbnail_source = extracted_source
                    else:
                        thumbnail_url = FALLBACK_THUMBNAIL_URL_GLOBAL
                        thumbnail_source = 'fallback'
                    
                    movie_data = {
                        'title': clean_title,
                        'original_title': title,
                        'normalized_title': norm_title,
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
                        'thumbnail_url': thumbnail_url,
                        'thumbnail_source': thumbnail_source,
                        'has_thumbnail': True,
                        'poster_url': poster_data.get('poster_url') if poster_data else None,
                        'poster_source': poster_data.get('source') if poster_data else None,
                        'has_poster': bool(poster_data and poster_data.get('poster_url')),
                        'extracted_thumbnail': extracted_thumb,
                        'has_extracted': bool(extracted_thumb)
                    }
                    
                    movies.append(movie_data)
                    
                    if len(movies) >= limit:
                        break
        
        logger.info(f"✅ Home movies fetched: {len(movies)}")
        return movies[:limit]
        
    except Exception as e:
        logger.error(f"❌ Home movies error: {e}")
        return []

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
# ✅ DATABASE INDEXES
# ============================================================================

async def setup_database_indexes():
    """Create optimized database indexes for fast search"""
    global thumbnails_col
    
    try:
        # Create thumbnails collection if it doesn't exist
        if 'thumbnails' not in await db.list_collection_names():
            thumbnails_col = db.thumbnails
            logger.info("✅ Created thumbnails collection")
        else:
            thumbnails_col = db.thumbnails
        
        # 🔥 OPTIMIZED INDEXES FOR FAST SEARCH
        await thumbnails_col.create_index(
            [("normalized_title", 1)],
            name="title_index",
            background=True
        )
        
        await thumbnails_col.create_index(
            [("normalized_title", 1), ("quality", 1)],
            unique=True,
            name="title_quality_unique",
            background=True
        )
        
        await thumbnails_col.create_index(
            [("message_id", 1)],
            name="message_index",
            background=True
        )
        
        await thumbnails_col.create_index(
            [("expires_at", 1)],
            name="ttl_cleanup",
            expireAfterSeconds=0,
            background=True
        )
        
        await thumbnails_col.create_index(
            [("extracted_at", -1)],
            name="recent_index",
            background=True
        )
        
        logger.info("✅ All optimized database indexes created successfully")
        
    except Exception as e:
        logger.warning(f"⚠️ Index creation error (may already exist): {e}")

# ============================================================================
# ✅ SYNC MANAGER
# ============================================================================

class SyncManager:
    def __init__(self):
        self.is_monitoring = False
        self.monitoring_task = None
        self.deleted_count = 0
        self.last_sync = time.time()
        self.sync_lock = asyncio.Lock()
    
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
            async with self.sync_lock:
                if thumbnails_col is None or User is None or not user_session_ready:
                    return
                
                current_time = time.time()
                if current_time - self.last_sync < 300:
                    return
                
                self.last_sync = current_time
                logger.info("🔍 Checking for deleted files in Telegram...")
                
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
                    logger.info("✅ No thumbnails to check")
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
                                logger.info(f"🗑️ Auto-deleted thumbnail: {item['title'][:40]}...")
                    
                    if deleted_count > 0:
                        logger.info(f"✅ Auto-deleted {deleted_count} thumbnails")
                    else:
                        logger.info("✅ No deleted files found")
                        
                except Exception as e:
                    logger.error(f"❌ Error checking messages: {e}")
                    
        except Exception as e:
            logger.error(f"❌ Auto-delete error: {e}")

sync_manager = SyncManager()

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
    
    # Initialize BOT first
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
    logger.info(f"Bot Handler: {'✅ INITIALIZED' if bot_handler.initialized else '❌ NOT READY'}")
    
    return bot_session_ready or user_session_ready

# ============================================================================
# ✅ MONGODB INITIALIZATION
# ============================================================================

@performance_monitor.measure("mongodb_init")
async def init_mongodb():
    global mongo_client, db, thumbnails_col, verification_col
    
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
        thumbnails_col = db.thumbnails
        verification_col = db.verifications
        
        logger.info("✅ MongoDB OK - Thumbnails collection ready")
        return True
        
    except asyncio.TimeoutError:
        logger.error("❌ MongoDB connection timeout")
        return False
    except Exception as e:
        logger.error(f"❌ MongoDB error: {e}")
        return False

# ============================================================================
# ✅ BOT INITIALIZATION
# ============================================================================

async def start_telegram_bot():
    """Start the Telegram bot for handling user commands"""
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
# ✅ POSTER FETCHER INITIALIZATION
# ============================================================================

async def init_poster_fetcher():
    """Initialize Poster Fetcher"""
    global poster_fetcher
    
    if PosterFetcher is not None:
        poster_fetcher = PosterFetcher(Config, cache_manager.redis_client if cache_manager else None)
        logger.info("✅ Poster Fetcher initialized")
        return True
    return False

# ============================================================================
# ✅ MAIN INITIALIZATION
# ============================================================================

@performance_monitor.measure("system_init")
async def init_system():
    start_time = time.time()
    
    try:
        logger.info("=" * 60)
        logger.info("🚀 SK4FiLM v9.0 - ZERO CPU, METADATA ONLY THUMBNAILS")
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
        
        # Initialize Poster Fetcher
        await init_poster_fetcher()
        
        # Initialize Telegram Sessions
        if PYROGRAM_AVAILABLE:
            telegram_ok = await init_telegram_sessions()
            if not telegram_ok:
                logger.warning("⚠️ Telegram sessions partially failed")
        
        # START TELEGRAM BOT
        global telegram_bot
        telegram_bot = await start_telegram_bot()
        if telegram_bot:
            logger.info("✅ Telegram Bot started successfully")
        else:
            logger.warning("⚠️ Telegram Bot failed to start")
        
        # Create database indexes
        await setup_database_indexes()
        
        # Start THUMBNAIL EXTRACTION (ZERO CPU, METADATA ONLY)
        if (user_session_ready or bot_session_ready) and thumbnails_col is not None:
            logger.info("🔍 Starting thumbnail extraction - ZERO CPU, METADATA ONLY...")
            asyncio.create_task(thumbnail_extractor.start_extraction(scan_all=True))
        
        # Start sync monitoring
        if user_session_ready:
            await sync_manager.start_sync_monitoring()
        
        init_time = time.time() - start_time
        logger.info(f"⚡ SK4FiLM Started in {init_time:.2f}s")
        logger.info("=" * 60)
        logger.info("🔧 OPTIMIZED FEATURES:")
        logger.info(f"   • NO FFmpeg - Zero CPU usage: ✅")
        logger.info(f"   • Telegram metadata only - ~50KB/file: ✅")
        logger.info(f"   • Base64 storage - Instant display: ✅")
        logger.info(f"   • MongoDB indexes - Fast search: ✅")
        logger.info(f"   • Rate limiting - Telegram safe: ✅")
        logger.info(f"   • Auto cleanup - {Config.THUMBNAIL_TTL_DAYS} days TTL: ✅")
        logger.info(f"   • Search Priority: Post+Files > Post only > File only: ✅")
        logger.info(f"   • Thumbnail Priority: Extracted > Poster > Fallback: ✅")
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
    extraction_stats = await thumbnail_extractor.get_stats()
    
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
        'service': 'SK4FiLM v9.0 - ZERO CPU, METADATA ONLY',
        'search_mode': 'direct_telegram',
        'thumbnail_stats': extraction_stats,
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
            'database': thumbnails_col is not None,
            'bot_handler': bot_handler is not None and bot_handler.initialized,
            'telegram_bot': telegram_bot is not None,
            'extractor': thumbnail_extractor.is_running
        },
        'sync_monitoring': {
            'running': sync_manager.is_monitoring,
            'deleted_count': sync_manager.deleted_count
        },
        'response_time': f"{time.perf_counter():.3f}s"
    })

@app.route('/health')
@performance_monitor.measure("health_endpoint")
async def health():
    extraction_stats = await thumbnail_extractor.get_stats()
    
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
        'extraction': {
            'running': thumbnail_extractor.is_running,
            'stats': extraction_stats
        },
        'sync': {
            'running': sync_manager.is_monitoring,
            'auto_delete_enabled': True
        },
        'timestamp': datetime.now().isoformat()
    })

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
    """Fast search using direct Telegram + MongoDB thumbnails"""
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
    try:
        perf_stats = performance_monitor.get_stats()
        extraction_stats = await thumbnail_extractor.get_stats()
        
        bot_status = None
        if bot_handler:
            try:
                bot_status = await bot_handler.get_bot_status()
            except:
                bot_status = {'initialized': False}
        
        bot_running = telegram_bot is not None and hasattr(telegram_bot, 'bot_started') and telegram_bot.bot_started
        
        return jsonify({
            'status': 'success',
            'performance': perf_stats,
            'thumbnail_extraction': extraction_stats,
            'sync_stats': {
                'running': sync_manager.is_monitoring,
                'deleted_count': sync_manager.deleted_count,
                'last_sync': sync_manager.last_sync
            },
            'bot_handler': bot_status,
            'telegram_bot': {
                'running': bot_running,
                'initialized': telegram_bot is not None
            },
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Stats API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# ============================================================================
# ✅ ADMIN API ROUTES
# ============================================================================

@app.route('/api/admin/extract-all', methods=['POST'])
async def api_admin_extract_all():
    """Trigger extraction of all thumbnails"""
    try:
        auth_token = request.headers.get('X-Admin-Token')
        if not auth_token or auth_token != os.environ.get('ADMIN_TOKEN', 'sk4film_admin_123'):
            return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401
        
        asyncio.create_task(thumbnail_extractor.start_extraction(scan_all=True))
        
        return jsonify({
            'status': 'success',
            'message': 'Full thumbnail extraction started (ZERO CPU, METADATA ONLY)',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Admin extract error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/admin/extract-pending', methods=['POST'])
async def api_admin_extract_pending():
    """Extract thumbnails for pending files"""
    try:
        auth_token = request.headers.get('X-Admin-Token')
        if not auth_token or auth_token != os.environ.get('ADMIN_TOKEN', 'sk4film_admin_123'):
            return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401
        
        asyncio.create_task(thumbnail_extractor.start_extraction(scan_all=False))
        
        return jsonify({
            'status': 'success',
            'message': 'Pending thumbnail extraction started',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Admin extract error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/admin/extraction-status', methods=['GET'])
async def api_admin_extraction_status():
    try:
        extraction_stats = await thumbnail_extractor.get_stats()
        
        return jsonify({
            'status': 'success',
            'extraction': extraction_stats,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Extraction status error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/admin/clear-cache', methods=['POST'])
async def api_admin_clear_cache():
    try:
        auth_token = request.headers.get('X-Admin-Token')
        if not auth_token or auth_token != os.environ.get('ADMIN_TOKEN', 'sk4film_admin_123'):
            return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401
        
        if cache_manager and cache_manager.redis_enabled:
            try:
                keys = await cache_manager.redis_client.keys("search_*")
                if keys:
                    await cache_manager.redis_client.delete(*keys)
                    logger.info(f"✅ Cleared {len(keys)} search cache keys")
            except Exception as e:
                logger.error(f"❌ Cache clear error: {e}")
        
        return jsonify({
            'status': 'success',
            'message': 'Cache cleared successfully'
        })
    except Exception as e:
        logger.error(f"❌ Clear cache error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/debug/thumbnails', methods=['GET'])
async def debug_thumbnails():
    """Debug endpoint to check thumbnail status"""
    try:
        if thumbnails_col is None:
            return jsonify({'error': 'Database not initialized'}), 500
        
        total_thumbnails = await thumbnails_col.count_documents({})
        
        # Get recent thumbnails
        cursor = thumbnails_col.find({}).sort('extracted_at', -1).limit(10)
        
        recent_thumbnails = []
        async for doc in cursor:
            doc['_id'] = str(doc['_id'])
            recent_thumbnails.append({
                'title': doc['title'],
                'quality': doc.get('quality', 'unknown'),
                'size_kb': doc.get('size_kb', 0),
                'extracted_at': doc['extracted_at'].isoformat() if isinstance(doc['extracted_at'], datetime) else str(doc['extracted_at']),
                'expires_at': doc['expires_at'].isoformat() if isinstance(doc['expires_at'], datetime) else str(doc['expires_at'])
            })
        
        extraction_stats = await thumbnail_extractor.get_stats()
        
        return jsonify({
            'status': 'success',
            'total_thumbnails': total_thumbnails,
            'total_storage_mb': extraction_stats.get('total_storage_mb', 0),
            'avg_size_kb': extraction_stats.get('avg_size_kb', 0),
            'recent_thumbnails': recent_thumbnails,
            'extraction_stats': extraction_stats,
            'config': {
                'file_channel_id': Config.FILE_CHANNEL_ID,
                'ttl_days': Config.THUMBNAIL_TTL_DAYS,
                'batch_size': Config.BATCH_SIZE,
                'extract_delay': Config.EXTRACT_DELAY
            },
            'sessions': {
                'user_session_ready': user_session_ready,
                'bot_session_ready': bot_session_ready
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Debug error: {e}")
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
    logger.info("🛑 Shutting down SK4FiLM v9.0...")
    
    shutdown_tasks = []
    
    if telegram_bot:
        try:
            if hasattr(telegram_bot, 'shutdown'):
                await telegram_bot.shutdown()
                logger.info("✅ Telegram Bot stopped")
        except Exception as e:
            logger.error(f"❌ Telegram Bot shutdown error: {e}")
    
    await thumbnail_extractor.stop_extraction()
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
    
    if verification_system is not None:
        shutdown_tasks.append(verification_system.stop())
    
    if premium_system is not None and hasattr(premium_system, 'stop_cleanup_task'):
        shutdown_tasks.append(premium_system.stop_cleanup_task())
    
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
    
    logger.info(f"🌐 Starting SK4FiLM v9.0 on port {Config.WEB_SERVER_PORT}...")
    logger.info(f"📁 File Channel ID: {Config.FILE_CHANNEL_ID}")
    logger.info("=" * 60)
    logger.info("✅ OPTIMIZED FEATURES ENABLED:")
    logger.info("   • NO FFmpeg - Zero CPU usage")
    logger.info("   • Telegram metadata only - ~50KB per file")
    logger.info("   • Base64 storage - Instant display")
    logger.info("   • MongoDB indexes - Fast search")
    logger.info("   • Rate limiting - Telegram safe")
    logger.info(f"   • Auto cleanup - {Config.THUMBNAIL_TTL_DAYS} days TTL")
    logger.info("   • Search Priority: Post+Files > Post only > File only")
    logger.info("   • Thumbnail Priority: Extracted > Poster > Fallback")
    logger.info("=" * 60)
    
    try:
        asyncio.run(serve(app, config))
    except KeyboardInterrupt:
        logger.info("👋 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
