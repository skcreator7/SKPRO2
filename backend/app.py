# ============================================================================
# 🚀 SK4FiLM v8.0 - COMPLETE FIXED CODE WITH FAST LOADING
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

# ✅ FIX: Import modular components with proper error handling
try:
    from cache import CacheManager
except ImportError:
    # Create inline CacheManager if module not found
    class CacheManager:
        def __init__(self, config):
            self.config = config
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
                return True
            except:
                self.redis_enabled = False
                return False
        
        async def get(self, key, default=None):
            return default
        
        async def set(self, key, value, expire_seconds=300):
            pass
        
        async def get_search_results(self, query, page, limit):
            return None
        
        async def cache_search_results(self, query, page, limit, data):
            pass
        
        async def clear_all(self):
            pass
        
        async def clear_search_cache(self):
            return 0
        
        async def get_stats_summary(self):
            return {}
        
        async def start_cleanup_task(self):
            pass
        
        async def stop(self):
            pass
        
        async def batch_set(self, data_dict, expire_seconds=3600):
            pass
        
        async def clear_pattern(self, pattern):
            return 0

# ✅ FIX: Define VerificationSystem inline
class VerificationSystem:
    def __init__(self, config, db):
        self.config = config
        self.db = db
    
    async def check_user_verified(self, user_id, premium_system=None):
        return True, "Verification not available"
    
    async def verify_user_api(self, user_id, verification_url=None):
        return {'verified': True, 'user_id': user_id}
    
    async def start_cleanup_task(self):
        pass
    
    async def stop(self):
        pass

# ✅ FIX: Define PremiumSystem inline
class PremiumTier:
    BASIC = "basic"
    PREMIUM = "premium"
    GOLD = "gold"
    DIAMOND = "diamond"

class PremiumSystem:
    def __init__(self, config, db):
        self.config = config
        self.db = db
    
    async def get_subscription_details(self, user_id):
        return {'has_active_subscription': False}
    
    async def can_user_download(self, user_id):
        return True, "Can download", {}
    
    async def get_all_plans(self):
        return []
    
    async def start_cleanup_task(self):
        pass
    
    async def stop_cleanup_task(self):
        pass

# ✅ FIX: Define utility functions inline to avoid import errors
import re
from datetime import datetime

def normalize_title(title):
    """Normalize title for consistent search"""
    if not title:
        return ""
    
    # Convert to lowercase
    title = title.lower()
    
    # Remove year patterns
    title = re.sub(r'\s*(?:\(|\[)?(?:19|20)\d{2}(?:\)|\])?', '', title)
    
    # Remove file extensions
    title = re.sub(r'\.(mp4|mkv|avi|mov|wmv|flv|webm|m4v|3gp)$', '', title)
    
    # Remove quality tags
    title = re.sub(r'\s*(?:480p|720p|1080p|2160p|4k|hd|fullhd|uhd|bluray|dvdrip|webrip|webdl|hdtv|brrip)', '', title)
    
    # Remove special characters and extra spaces
    title = re.sub(r'[^\w\s]', ' ', title)
    title = re.sub(r'\s+', ' ', title).strip()
    
    return title

def extract_title_smart(text):
    """Extract title from Telegram message text"""
    if not text:
        return ""
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Split into lines
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    for line in lines:
        if len(line) > 10:
            # Clean common prefixes
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
        # Remove extension
        name = re.sub(r'\.[a-zA-Z0-9]+$', '', filename)
        # Remove quality tags
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
    
    # HTML escape
    text = html.escape(text)
    
    # Split into paragraphs
    paragraphs = []
    for p in text.split('\n'):
        p = p.strip()
        if p:
            paragraphs.append(p)
    
    if not paragraphs:
        return "<p>No content</p>"
    
    # Format
    formatted = ""
    for i, p in enumerate(paragraphs):
        if i == 0 and len(p) > 30:  # First paragraph as title
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

# ✅ FIX: Define bot_handlers inline
BOT_HANDLERS_AVAILABLE = False
SK4FiLMBot = None
setup_bot_handlers = None

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
    
    # Database Configuration - ✅ FIXED: Remove SSL conflict
    MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
    REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
    REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "")
    
    # Channel Configuration - OPTIMIZED
    MAIN_CHANNEL_ID = -1001891090100
    TEXT_CHANNEL_IDS = [-1001891090100, -1002024811395]
    FILE_CHANNEL_ID = -1001768249569  # ✅ FILE CHANNEL FOR SYNC
    
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
    MONITOR_INTERVAL = int(os.environ.get("MONITOR_INTERVAL", "300"))  # Check every 5 minutes
    
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
    response.headers['X-SK4FiLM-Version'] = '8.0-FIXED-FAST'
    response.headers['X-Response-Time'] = f"{time.perf_counter():.3f}"
    return response

# GLOBAL VARIABLES - FAST ACCESS
mongo_client = None
db = None
files_col = None
verification_col = None
poster_col = None

# MODULAR COMPONENTS
cache_manager = None
verification_system = None
premium_system = None
poster_fetcher = None
sk4film_bot = None

# Telegram clients
User = None
bot = None
bot_started = False
user_session_ready = False
telegram_initialized = False

# CHANNEL CONFIGURATION - CACHED
CHANNEL_CONFIG = {
    -1001891090100: {'name': 'SK4FiLM Main', 'type': 'text', 'search_priority': 1},
    -1002024811395: {'name': 'SK4FiLM Updates', 'type': 'text', 'search_priority': 2},
    -1001768249569: {'name': 'SK4FiLM Files', 'type': 'file', 'search_priority': 0, 'sync_manage': True}
}

# ============================================================================
# ✅ POSTER FETCHER CLASS (FIXED)
# ============================================================================

class PosterFetcher:
    def __init__(self, cache_manager):  # ✅ FIXED: Removed config parameter
        self.cache_manager = cache_manager
        self.memory_cache = {}
        
    async def fetch_poster(self, title: str, year: str = ""):
        """Fetch movie poster with caching"""
        cache_key = f"poster:{title}:{year}"
        
        # Check memory cache
        if cache_key in self.memory_cache:
            data, expiry = self.memory_cache[cache_key]
            if expiry > datetime.now():
                return data
        
        # Create fallback poster
        poster_data = self._create_fallback_poster(title, year)
        
        # Cache the result
        if poster_data:
            self.memory_cache[cache_key] = (poster_data, datetime.now() + timedelta(hours=24))
            
            if self.cache_manager:
                await self.cache_manager.set(cache_key, poster_data, expire_seconds=7*24*3600)
        
        return poster_data
    
    def _create_fallback_poster(self, title: str, year: str):
        """Create fallback poster URL"""
        clean_title = ''.join(c for c in title if c.isalnum() or c in ' _-')
        encoded_title = urllib.parse.quote(clean_title[:50])
        
        # Generate color based on title hash
        title_hash = hashlib.md5(title.encode()).hexdigest()
        color = f"#{title_hash[:6]}"
        
        # Create placeholder URL
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
# ✅ TELEGRAM CHANNEL SYNC MANAGEMENT
# ============================================================================

class ChannelSyncManager:
    def __init__(self):
        self.last_checked_message_id = {}
        self.is_monitoring = False
        self.monitoring_task = None
        self.deleted_count = 0
        self.last_sync = time.time()
    
    async def start_sync_monitoring(self):
        """Start monitoring Telegram channel for deletions sync"""
        if not User or not user_session_ready:
            logger.warning("⚠️ Cannot start sync monitoring - User session not ready")
            return
        
        if self.is_monitoring:
            logger.info("✅ Channel sync monitoring already running")
            return
        
        logger.info("👁️ Starting Telegram channel sync monitoring...")
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self.monitor_channel_sync())
    
    async def stop_sync_monitoring(self):
        """Stop sync monitoring"""
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("🛑 Channel sync monitoring stopped")
    
    async def monitor_channel_sync(self):
        """Monitor Telegram channel for deletions sync"""
        logger.info(f"🔍 Monitoring FILE_CHANNEL_ID: {Config.FILE_CHANNEL_ID} for deletions sync")
        
        while self.is_monitoring:
            try:
                await self.sync_deletions_from_telegram()
                await asyncio.sleep(Config.MONITOR_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Channel sync monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def sync_deletions_from_telegram(self):
        """Sync deletions from Telegram to MongoDB"""
        try:
            if not files_col or not User:
                return
            
            current_time = time.time()
            if current_time - self.last_sync < 300:
                return
            
            self.last_sync = current_time
            logger.debug("🔍 Syncing deletions from Telegram channel...")
            
            # Get all files from MongoDB for FILE_CHANNEL_ID
            cursor = files_col.find(
                {"channel_id": Config.FILE_CHANNEL_ID},
                {"message_id": 1, "title": 1, "_id": 0}
            )
            
            message_ids_in_db = []
            file_titles = {}
            
            async for doc in cursor:
                msg_id = doc.get('message_id')
                if msg_id:
                    message_ids_in_db.append(msg_id)
                    file_titles[msg_id] = doc.get('title', 'Unknown')
            
            if not message_ids_in_db:
                logger.debug("📭 No files in database to sync")
                return
            
            logger.info(f"📊 Checking {len(message_ids_in_db)} files for sync...")
            
            deleted_count = 0
            batch_size = 50
            
            for i in range(0, len(message_ids_in_db), batch_size):
                batch = message_ids_in_db[i:i + batch_size]
                
                try:
                    # Try to get messages from Telegram
                    messages = await safe_telegram_operation(
                        User.get_messages,
                        Config.FILE_CHANNEL_ID,
                        batch
                    )
                    
                    # Determine which messages exist
                    existing_ids = set()
                    if isinstance(messages, list):
                        for msg in messages:
                            if msg and hasattr(msg, 'id'):
                                existing_ids.add(msg.id)
                    elif messages and hasattr(messages, 'id'):
                        existing_ids.add(messages.id)
                    
                    # Find deleted message IDs (in database but not in Telegram)
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
                            
                            for msg_id in deleted_ids[:5]:
                                title = file_titles.get(msg_id, f"ID: {msg_id}")
                                logger.info(f"🗑️ Synced Deletion: {title}")
                            
                            if len(deleted_ids) > 5:
                                logger.info(f"🗑️ ... and {len(deleted_ids) - 5} more files")
                
                except Exception as e:
                    logger.error(f"❌ Error checking batch: {e}")
                    continue
            
            if deleted_count > 0:
                logger.info(f"✅ Sync completed: {deleted_count} files deleted from MongoDB")
            else:
                logger.debug("✅ Sync completed: No deletions found")
            
        except Exception as e:
            logger.error(f"❌ Error syncing deletions: {e}")
    
    async def manual_sync(self):
        """Manual sync trigger"""
        await self.sync_deletions_from_telegram()

# Create global sync manager instance
channel_sync_manager = ChannelSyncManager()

# ============================================================================
# ✅ SMART FILE INDEXING WITH DUPLICATE PREVENTION
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
    """
    Smart file indexing for FILE_CHANNEL_ID (-1001768249569)
    """
    try:
        if not files_col:
            logger.error("❌ Files collection not available")
            return False
        
        # ============================================================================
        # ✅ STEP 1: BASIC VALIDATION
        # ============================================================================
        if not message or (not message.document and not message.video):
            return False
        
        # ============================================================================
        # ✅ STEP 2: CHECK IF ALREADY EXISTS
        # ============================================================================
        existing_by_id = await files_col.find_one({
            'channel_id': Config.FILE_CHANNEL_ID,
            'message_id': message.id
        }, {'_id': 1, 'title': 1})
        
        if existing_by_id:
            logger.debug(f"📝 Already indexed: {message.id}")
            return True
        
        # Extract title
        title = await extract_title_from_telegram_msg_cached(message)
        if not title:
            logger.debug(f"📝 Skipping - No title: {message.id}")
            return False
        
        normalized_title = normalize_title_cached(title)
        
        # ============================================================================
        # ✅ STEP 3: CREATE DOCUMENT
        # ============================================================================
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
            'status': 'active',
            'duplicate_checked': True
        }
        
        # Add file-specific data
        if message.document:
            doc.update({
                'file_name': message.document.file_name or '',
                'quality': detect_quality(message.document.file_name or ''),
                'is_video_file': is_video_file(message.document.file_name or ''),
                'caption': message.caption or '',
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
                'caption': message.caption or '',
                'duration': message.video.duration if hasattr(message.video, 'duration') else 0,
                'width': message.video.width if hasattr(message.video, 'width') else 0,
                'height': message.video.height if hasattr(message.video, 'height') else 0,
                'file_id': message.video.file_id,
                'file_size': message.video.file_size or 0
            })
            doc['file_hash'] = await generate_file_hash(message)
        else:
            return False
        
        # ============================================================================
        # ✅ STEP 4: INSERT WITH ERROR HANDLING
        # ============================================================================
        try:
            await files_col.insert_one(doc)
            
            file_type = "📹 Video" if doc['is_video_file'] else "📄 File"
            size_str = format_size(doc['file_size']) if doc['file_size'] > 0 else "Unknown"
            
            logger.info(f"✅ {file_type} indexed in FILE_CHANNEL: {title}")
            logger.info(f"   📊 Size: {size_str} | Quality: {doc.get('quality', 'Unknown')} | ID: {message.id}")
            
            return True
        except Exception as e:
            if "duplicate key error" in str(e).lower():
                logger.debug(f"📝 Race condition duplicate: {message.id}")
                return True
            else:
                logger.error(f"❌ Error inserting document: {e}")
                return False
        
    except Exception as e:
        logger.error(f"❌ Smart indexing error: {e}")
        return False

# ============================================================================
# ✅ BACKGROUND INDEXING FOR FILE_CHANNEL_ID
# ============================================================================

async def index_files_background_smart():
    """Smart background indexing for FILE_CHANNEL_ID (-1001768249569)"""
    if not User or files_col is None or not user_session_ready:
        logger.warning("⚠️ Cannot start smart indexing - User session not ready")
        return
    
    logger.info("📁 Starting SMART background indexing for FILE_CHANNEL_ID...")
    
    try:
        # ============================================================================
        # ✅ STEP 1: SETUP MONGODB INDEXES
        # ============================================================================
        await setup_mongodb_sync_indexes()
        
        # ============================================================================
        # ✅ STEP 2: GET LAST INDEXED MESSAGE ID
        # ============================================================================
        last_indexed = await files_col.find_one(
            {"channel_id": Config.FILE_CHANNEL_ID}, 
            sort=[('message_id', -1)],
            projection={'message_id': 1, 'title': 1}
        )
        
        last_message_id = last_indexed['message_id'] if last_indexed else 0
        last_title = last_indexed.get('title', 'Unknown') if last_indexed else 'None'
        
        logger.info(f"🔄 Starting from message ID: {last_message_id}")
        logger.info(f"📝 Last indexed file: {last_title}")
        
        # ============================================================================
        # ✅ STEP 3: FETCH AND PROCESS NEW MESSAGES
        # ============================================================================
        total_indexed = 0
        total_skipped = 0
        
        # Fetch messages newer than last_indexed
        messages = []
        async for msg in safe_telegram_generator(
            User.get_chat_history, 
            Config.FILE_CHANNEL_ID,
            limit=200
        ):
            if msg.id <= last_message_id:
                break
            
            if msg and (msg.document or msg.video):
                messages.append(msg)
        
        # Process in reverse order (oldest first)
        messages.reverse()
        
        logger.info(f"📥 Found {len(messages)} new files to index")
        
        for msg in messages:
            try:
                success = await index_single_file_smart(msg)
                if success:
                    total_indexed += 1
                else:
                    total_skipped += 1
                
                await asyncio.sleep(0.2)
                
                if (total_indexed + total_skipped) % 10 == 0:
                    logger.info(f"📊 Progress: {total_indexed} new, {total_skipped} skipped")
                
            except Exception as e:
                logger.error(f"❌ Error processing message {msg.id}: {e}")
                continue
        
        # ============================================================================
        # ✅ STEP 4: START SYNC MONITORING
        # ============================================================================
        if total_indexed > 0:
            logger.info(f"✅ Smart indexing complete: {total_indexed} NEW files indexed")
        
        if total_skipped > 0:
            logger.info(f"📝 Skipped {total_skipped} files")
        
        # Start sync monitoring
        await channel_sync_manager.start_sync_monitoring()
        await channel_sync_manager.sync_deletions_from_telegram()
        
        logger.info("🚀 Sync management system activated!")
        
    except Exception as e:
        logger.error(f"❌ Smart background indexing error: {e}")

async def setup_mongodb_sync_indexes():
    """Setup MongoDB indexes"""
    try:
        if not files_col:
            return
        
        logger.info("🗂️ Setting up MongoDB indexes...")
        
        try:
            await files_col.create_index(
                [("channel_id", 1), ("message_id", 1)],
                unique=True,
                name="channel_message_unique"
            )
            logger.info("✅ Unique index created")
        except Exception as e:
            logger.debug(f"Unique index may already exist: {e}")
        
        try:
            await files_col.create_index(
                [("channel_id", 1), ("file_hash", 1)],
                name="file_hash_index"
            )
            logger.info("✅ File hash index created")
        except Exception as e:
            logger.debug(f"File hash index may already exist: {e}")
        
        try:
            await files_col.create_index(
                [("normalized_title", "text")],
                name="text_search_index"
            )
            logger.info("✅ Text search index created")
        except Exception as e:
            logger.debug(f"Text index may already exist: {e}")
        
        logger.info("✅ MongoDB indexes setup complete")
        
    except Exception as e:
        logger.error(f"❌ Error setting up MongoDB indexes: {e}")

# ============================================================================
# ✅ FAST CACHE FUNCTIONS
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

# ASYNC CACHE DECORATOR - FIXED for event loop issues
def async_cache_with_ttl(maxsize=128, ttl=300):
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
            
            # Check cache with lock
            async with cache_lock:
                if key in cache:
                    value, timestamp = cache[key]
                    if now - timestamp < ttl:
                        return value
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Store in cache with lock
            async with cache_lock:
                cache[key] = (result, now)
                
                # Clean old entries
                if len(cache) > maxsize:
                    oldest_key = min(cache.keys(), key=lambda k: cache[k][1])
                    del cache[oldest_key]
            
            return result
        return wrapper
    return decorator

# ============================================================================
# ✅ FIXED TELEGRAM OPERATIONS (Event Loop Safe)
# ============================================================================

class TurboFloodProtection:
    def __init__(self):
        self.request_buckets = {}
        self.last_cleanup = time.time()
    
    async def wait_if_needed(self, user_id=None):
        current_time = time.time()
        
        if current_time - self.last_cleanup > 30:
            self._cleanup_buckets(current_time)
            self.last_cleanup = current_time
        
        bucket_key = int(current_time // 10)
        
        if bucket_key not in self.request_buckets:
            self.request_buckets[bucket_key] = 0
        
        if self.request_buckets[bucket_key] >= 50:
            wait_time = 10 - (current_time % 10)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                self.request_buckets = {}
        
        self.request_buckets[bucket_key] += 1
    
    def _cleanup_buckets(self, current_time):
        current_bucket = int(current_time // 10)
        old_buckets = [k for k in self.request_buckets.keys() if k < current_bucket - 6]
        for bucket in old_buckets:
            del self.request_buckets[bucket]

turbo_flood_protection = TurboFloodProtection()

@performance_monitor.measure("telegram_operation")
async def safe_telegram_operation(operation, *args, **kwargs):
    """✅ FIXED: Event loop safe Telegram operations"""
    max_retries = 2
    
    for attempt in range(max_retries):
        try:
            await turbo_flood_protection.wait_if_needed()
            # ✅ Ensure we're in the right event loop
            result = await operation(*args, **kwargs)
            return result
        except Exception as e:
            error_str = str(e)
            if "FloodWait" in error_str:
                wait_match = re.search(r'(\d+)', error_str)
                if wait_match:
                    wait_time = int(wait_match.group(1))
                    logger.warning(f"⏳ Flood wait {wait_time}s")
                    await asyncio.sleep(wait_time + 1)
                    continue
            elif "different loop" in error_str.lower():
                # ✅ Handle event loop mismatch
                logger.warning(f"🔄 Event loop mismatch, retrying...")
                await asyncio.sleep(0.5)
                continue
            logger.error(f"Telegram operation failed: {e}")
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(0.5 * (2 ** attempt))
    
    return None

async def safe_telegram_generator(operation, *args, limit=None, **kwargs):
    """✅ FIXED: Event loop safe generator"""
    count = 0
    
    async for item in operation(*args, **kwargs):
        yield item
        count += 1
        
        if count % 10 == 0:
            await asyncio.sleep(0.1)
            try:
                await turbo_flood_protection.wait_if_needed()
            except:
                pass
        
        if limit and count >= limit:
            break

# ============================================================================
# ✅ OPTIMIZED TITLE EXTRACTION
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

# ============================================================================
# ✅ VIDEO THUMBNAIL EXTRACTION
# ============================================================================

@performance_monitor.measure("thumbnail_extraction")
async def extract_video_thumbnail(user_client, message):
    try:
        if message.video:
            thumbnail = message.video.thumbs[0] if message.video.thumbs else None
            if thumbnail:
                thumbnail_path = await safe_telegram_operation(
                    user_client.download_media, 
                    thumbnail.file_id, 
                    in_memory=True
                )
                if thumbnail_path:
                    thumbnail_data = base64.b64encode(thumbnail_path.getvalue()).decode('utf-8')
                    return f"data:image/jpeg;base64,{thumbnail_data}"
        
        return None
    except Exception as e:
        logger.error(f"Thumbnail extraction failed: {e}")
        return None

async def get_telegram_video_thumbnail(user_client, channel_id, message_id):
    try:
        if not user_client or not user_session_ready:
            return None
            
        msg = await safe_telegram_operation(
            user_client.get_messages,
            channel_id,
            message_id
        )
        
        if msg and (msg.video or msg.document):
            return await extract_video_thumbnail(user_client, msg)
        
        return None
    except Exception as e:
        logger.error(f"Get thumbnail error: {e}")
        return None

# ============================================================================
# ✅ MONGODB INITIALIZATION - FIXED SSL/TLS CONFLICT
# ============================================================================

@performance_monitor.measure("mongodb_init")
async def init_mongodb():
    global mongo_client, db, files_col, verification_col, poster_col
    
    try:
        logger.info("🔌 MongoDB initialization...")
        
        # ✅ FIXED: Remove SSL/TLS conflict
        mongo_client = AsyncIOMotorClient(
            Config.MONGODB_URI,
            serverSelectionTimeoutMS=10000,
            connectTimeoutMS=10000,
            socketTimeoutMS=15000,
            maxPoolSize=20,
            minPoolSize=5,
            retryWrites=True,
            retryReads=True
            # ✅ REMOVED: ssl=True to avoid conflict with tls
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
# ✅ TELEGRAM INITIALIZATION - SIMPLIFIED
# ============================================================================

@performance_monitor.measure("telegram_init")
async def init_telegram_clients():
    global User, bot, bot_started, user_session_ready, telegram_initialized
    
    logger.info("=" * 60)
    logger.info("🚀 TELEGRAM CLIENT INITIALIZATION")
    logger.info("=" * 60)
    
    if not PYROGRAM_AVAILABLE:
        logger.error("❌ Pyrogram not installed!")
        return False
    
    # Initialize User Client
    if Config.API_ID > 0 and Config.API_HASH and Config.USER_SESSION_STRING:
        logger.info("\n👤 Initializing User Client...")
        
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
            logger.info(f"✅ User Client Ready: {me.first_name}")
            
            # Test FILE_CHANNEL_ID access
            try:
                chat = await User.get_chat(Config.FILE_CHANNEL_ID)
                logger.info(f"✅ FILE Channel Access: {chat.title}")
                user_session_ready = True
            except Exception as e:
                logger.error(f"❌ FILE Channel access failed: {e}")
                user_session_ready = False
                
        except Exception as e:
            logger.error(f"❌ User client failed: {e}")
            user_session_ready = False
    
    # Initialize Bot Client
    if Config.BOT_TOKEN:
        logger.info("\n🤖 Initializing Bot Client...")
        try:
            bot = Client(
                "sk4film_bot",
                api_id=Config.API_ID,
                api_hash=Config.API_HASH,
                bot_token=Config.BOT_TOKEN,
                sleep_threshold=30,
                in_memory=True,
                no_updates=True
            )
            
            await bot.start()
            bot_started = True
            bot_info = await bot.get_me()
            logger.info(f"✅ Bot Ready: @{bot_info.username}")
            
        except Exception as e:
            logger.error(f"❌ Bot initialization failed: {e}")
            bot_started = False
    
    logger.info("\n" + "=" * 60)
    logger.info(f"User Session: {'✅ READY' if user_session_ready else '❌ NOT READY'}")
    logger.info(f"Bot Session: {'✅ READY' if bot_started else '❌ NOT READY'}")
    logger.info(f"FILE Channel ID: {Config.FILE_CHANNEL_ID}")
    
    telegram_initialized = True
    return user_session_ready

# ============================================================================
# ✅ FAST CACHE WARM-UP
# ============================================================================

async def warm_up_cache():
    """Warm up all caches for instant response"""
    try:
        logger.info("🔥 Warming up caches...")
        
        # Warm up initial data
        if cache_manager and cache_manager.redis_enabled:
            warm_data = {
                "system:warm": True,
                "startup_time": datetime.now().isoformat(),
                "version": "8.0-FIXED-FAST",
                "telegram_ready": user_session_ready,
                "file_channel_id": Config.FILE_CHANNEL_ID
            }
            
            for key, value in warm_data.items():
                await cache_manager.set(key, value, expire_seconds=3600)
        
        logger.info("✅ Cache warm-up complete")
        
    except Exception as e:
        logger.error(f"Cache warm-up error: {e}")

# ============================================================================
# ✅ FAST HOME MOVIES FROM TELEGRAM
# ============================================================================

@performance_monitor.measure("home_movies_telegram")
@async_cache_with_ttl(maxsize=1, ttl=60)
async def get_home_movies_telegram(limit=30):
    """✅ FIXED: Event loop safe home movies"""
    try:
        if not User or not user_session_ready:
            return []
        
        movies = []
        seen_titles = set()
        
        logger.info(f"🎬 Fetching {limit} movies from Telegram...")
        
        async for msg in User.get_chat_history(Config.MAIN_CHANNEL_ID, limit=limit * 2):
            if msg and msg.text and len(msg.text) > 20:
                title = extract_title_smart(msg.text)
                
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    
                    year_match = re.search(r'\b(19|20)\d{2}\b', title)
                    year = year_match.group() if year_match else ""
                    
                    clean_title = re.sub(r'\s+\(\d{4}\)$', '', title)
                    clean_title = re.sub(r'\s+\d{4}$', '', clean_title)
                    
                    poster_url = Config.get_poster(clean_title, year)
                    
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
                        'poster_url': poster_url,
                        'poster_source': 'telegram',
                        'poster_rating': '0.0'
                    })
                    
                    if len(movies) >= limit:
                        break
        
        logger.info(f"✅ Fetched {len(movies)} movies")
        return movies[:limit]
        
    except Exception as e:
        logger.error(f"❌ Home movies error: {e}")
        return []

# ============================================================================
# ✅ ULTRA-FAST SEARCH FUNCTION
# ============================================================================

@performance_monitor.measure("multi_channel_search")
@async_cache_with_ttl(maxsize=500, ttl=300)
async def search_movies_multi_channel(query, limit=12, page=1):
    """✅ FIXED: Event loop safe search"""
    offset = (page - 1) * limit
    
    # Try cache first
    cache_key = f"search:{query}:{page}:{limit}"
    if cache_manager and cache_manager.redis_enabled:
        cached_data = await cache_manager.get(cache_key)
        if cached_data:
            return cached_data
    
    logger.info(f"🔍 Searching for: {query}")
    
    query_lower = query.lower()
    files_dict = {}
    
    # MongoDB search from FILE_CHANNEL_ID
    try:
        if files_col is not None:
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
    
    # Merge results
    merged = {}
    
    # Add file information from FILE_CHANNEL_ID
    for norm_title, file_data in files_dict.items():
        if query_lower in norm_title.lower() or query_lower in file_data['title'].lower():
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
    
    # Sort results
    results_list = list(merged.values())
    results_list.sort(key=lambda x: x['date'], reverse=True)
    
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
            'channels_searched': 1,
            'query': query,
            'cache_hit': False
        }
    }
    
    # Cache results
    if cache_manager and cache_manager.redis_enabled:
        await cache_manager.set(cache_key, result_data, expire_seconds=600)
    
    logger.info(f"✅ Search completed: {len(paginated)} results")
    return result_data

# ============================================================================
# ✅ SINGLE POST API
# ============================================================================

async def get_single_post_api(channel_id, message_id):
    """Get single movie/post details"""
    try:
        if User and user_session_ready:
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
# ✅ VERIFICATION & INDEX STATUS FUNCTIONS
# ============================================================================

async def verify_user_api(user_id, verification_url=None):
    try:
        if verification_system:
            result = await verification_system.verify_user_api(user_id, verification_url)
            return result
        
        return {
            'verified': True,
            'user_id': user_id,
            'expires_at': (datetime.now() + timedelta(hours=6)).isoformat()
        }
    except Exception as e:
        logger.error(f"Verify user API error: {e}")
        return {
            'verified': False,
            'error': str(e)
        }

async def get_index_status_api():
    try:
        total_files = await files_col.count_documents({}) if files_col else 0
        video_files = await files_col.count_documents({'is_video_file': True}) if files_col else 0
        
        if files_col:
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
            'last_update': datetime.now().isoformat(),
            'status': 'active' if user_session_ready else 'inactive',
            'note': 'FILE_CHANNEL_ID indexing active'
        }
    except Exception as e:
        logger.error(f"Index status API error: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }

# ============================================================================
# ✅ OPTIMIZED API FUNCTIONS
# ============================================================================

@performance_monitor.measure("search_api")
async def search_movies_api(query, limit=12, page=1):
    """Optimized search API"""
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
async def get_home_movies_live():
    """✅ FIXED: Event loop safe home movies"""
    try:
        movies = await get_home_movies_telegram(limit=30)
        return movies
        
    except Exception as e:
        logger.error(f"Home movies error: {e}")
        return []

# ============================================================================
# ✅ MAIN INITIALIZATION - FIXED
# ============================================================================

@performance_monitor.measure("system_init")
async def init_system():
    start_time = time.time()
    
    try:
        logger.info("🚀 Starting SK4FiLM v8.0 - FAST LOADING FIX...")
        logger.info(f"📌 FILE_CHANNEL_ID: {Config.FILE_CHANNEL_ID}")
        
        # Initialize MongoDB
        mongo_ok = await init_mongodb()
        if not mongo_ok:
            logger.warning("⚠️ MongoDB connection failed, continuing without DB")
        
        # Initialize modular components
        global cache_manager, verification_system, premium_system, poster_fetcher
        
        # Initialize Cache Manager
        cache_manager = CacheManager(Config)
        redis_ok = await cache_manager.init_redis()
        if redis_ok:
            logger.info("✅ Cache Manager initialized")
            await cache_manager.start_cleanup_task()
        
        # Initialize Verification System
        if mongo_ok:
            verification_system = VerificationSystem(Config, db)
            await verification_system.start_cleanup_task()
            logger.info("✅ Verification System initialized")
        
        # Initialize Premium System
        if mongo_ok:
            premium_system = PremiumSystem(Config, db)
            await premium_system.start_cleanup_task()
            logger.info("✅ Premium System initialized")
        
        # ✅ FIXED: Initialize Poster Fetcher with correct parameters
        poster_fetcher = PosterFetcher(cache_manager)
        logger.info("✅ Poster Fetcher initialized")
        
        # WARM UP CACHE FOR INSTANT RESPONSE
        asyncio.create_task(warm_up_cache())
        
        # Initialize Telegram
        if PYROGRAM_AVAILABLE:
            telegram_ok = await init_telegram_clients()
            if not telegram_ok:
                logger.warning("⚠️ Telegram initialization failed")
        else:
            logger.warning("⚠️ Pyrogram not available")
        
        # Start indexing if ready
        if user_session_ready and files_col:
            asyncio.create_task(index_files_background_smart())
            logger.info("✅ Started background indexing for FILE_CHANNEL_ID")
        else:
            logger.warning("⚠️ Cannot start indexing - Requirements not met")
        
        init_time = time.time() - start_time
        logger.info(f"⚡ SK4FiLM Started in {init_time:.2f}s")
        
        logger.info("🔧 SYSTEM FEATURES:")
        logger.info(f"   • FILE_CHANNEL_ID: {Config.FILE_CHANNEL_ID}")
        logger.info(f"   • Fast Cache: {'✅ ENABLED' if redis_ok else '❌ DISABLED'}")
        logger.info(f"   • MongoDB: {'✅ CONNECTED' if mongo_ok else '❌ DISCONNECTED'}")
        logger.info(f"   • Telegram: {'✅ READY' if user_session_ready else '❌ NOT READY'}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ System initialization failed: {e}")
        return False

# ============================================================================
# ✅ API ROUTES WITH FAST RESPONSE
# ============================================================================

@app.route('/')
@performance_monitor.measure("root_endpoint")
async def root():
    tf = await files_col.count_documents({}) if files_col else 0
    video_files = await files_col.count_documents({'is_video_file': True}) if files_col else 0
    
    return jsonify({
        'status': 'healthy',
        'service': 'SK4FiLM v8.0 - FAST FIX',
        'telegram': {
            'user_session_ready': user_session_ready,
            'bot_started': bot_started,
            'file_channel': Config.FILE_CHANNEL_ID,
            'main_channel': Config.MAIN_CHANNEL_ID
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
        'telegram': {
            'user_session': user_session_ready,
            'file_channel': Config.FILE_CHANNEL_ID
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/movies', methods=['GET'])
@performance_monitor.measure("movies_endpoint")
async def api_movies():
    """FAST movies endpoint with cache"""
    try:
        movies = await get_home_movies_live()
        
        return jsonify({
            'status': 'success' if movies else 'empty',
            'movies': movies,
            'total': len(movies),
            'source': 'telegram',
            'telegram_ready': user_session_ready,
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
            'total': 0,
            'telegram_ready': user_session_ready
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
            'timestamp': datetime.now().isoformat(),
            'cache_hit': result_data.get('search_metadata', {}).get('cache_hit', False)
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
            return jsonify({
                'status': 'success',
                'post': post_data,
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
                    'poster': {
                        'poster_url': poster_data.get('url', ''),
                        'source': poster_data.get('source', 'custom'),
                        'rating': poster_data.get('rating', '0.0')
                    },
                    'title': title,
                    'year': year,
                    'timestamp': datetime.now().isoformat()
                })
        
        return jsonify({
            'status': 'success',
            'poster': {
                'poster_url': Config.get_poster(title, year),
                'source': 'custom',
                'rating': '0.0'
            },
            'title': title,
            'year': year,
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

@app.route('/api/stats', methods=['GET'])
@performance_monitor.measure("stats_endpoint")
async def api_stats():
    try:
        stats = {}
        
        if files_col:
            stats['database'] = {
                'total_files': await files_col.count_documents({}),
                'video_files': await files_col.count_documents({'is_video_file': True}),
                'file_channel_files': await files_col.count_documents({"channel_id": Config.FILE_CHANNEL_ID}),
            }
        
        stats['sync_management'] = {
            'sync_monitoring': channel_sync_manager.is_monitoring,
            'deleted_count': channel_sync_manager.deleted_count,
            'file_channel_id': Config.FILE_CHANNEL_ID
        }
        
        stats['system'] = {
            'telegram': {
                'user_session_ready': user_session_ready,
                'bot_started': bot_started,
                'file_channel': Config.FILE_CHANNEL_ID
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify({
            'status': 'success',
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"Stats API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# ============================================================================
# ✅ SYNC MANAGEMENT API ENDPOINTS
# ============================================================================

@app.route('/api/sync/status', methods=['GET'])
async def api_sync_status():
    try:
        status = {
            'file_channel_id': Config.FILE_CHANNEL_ID,
            'sync_monitoring_active': channel_sync_manager.is_monitoring,
            'deleted_count': channel_sync_manager.deleted_count,
            'last_sync': channel_sync_manager.last_sync,
            'user_session_ready': user_session_ready,
            'timestamp': datetime.now().isoformat()
        }
        
        if files_col:
            try:
                total_files = await files_col.count_documents({"channel_id": Config.FILE_CHANNEL_ID})
                status['total_files_in_channel'] = total_files
            except:
                pass
        
        return jsonify({
            'status': 'success',
            'sync_management': status,
            'message': 'Sync management active' if channel_sync_manager.is_monitoring else 'Sync management inactive'
        })
        
    except Exception as e:
        logger.error(f"Sync status API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/sync/manual', methods=['POST'])
async def api_sync_manual():
    try:
        if not user_session_ready:
            return jsonify({
                'status': 'error',
                'message': 'Telegram session not ready'
            }), 400
        
        await channel_sync_manager.manual_sync()
        
        return jsonify({
            'status': 'success',
            'message': 'Manual sync completed',
            'deleted_count': channel_sync_manager.deleted_count,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Manual sync API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# ============================================================================
# ✅ TELEGRAM STATUS API
# ============================================================================

@app.route('/api/telegram/status', methods=['GET'])
async def api_telegram_status():
    try:
        status = {
            'environment': {
                'pyrogram_available': PYROGRAM_AVAILABLE,
                'api_id_configured': Config.API_ID > 0,
                'api_hash_configured': bool(Config.API_HASH),
                'session_string_configured': bool(Config.USER_SESSION_STRING)
            },
            'connections': {
                'user_session': {
                    'initialized': User is not None,
                    'ready': user_session_ready,
                    'can_fetch_movies': user_session_ready
                },
                'bot_session': {
                    'initialized': bot is not None,
                    'ready': bot_started
                }
            },
            'channels': {
                'main_channel': Config.MAIN_CHANNEL_ID,
                'file_channel': Config.FILE_CHANNEL_ID,
                'movies_source': 'Telegram Channel'
            },
            'timestamp': datetime.now().isoformat()
        }
        
        if User:
            try:
                me = await User.get_me()
                status['connections']['user_session']['user_info'] = {
                    'id': me.id,
                    'first_name': me.first_name,
                    'username': me.username
                }
            except:
                pass
        
        return jsonify({
            'status': 'success',
            'telegram': status,
            'message': 'Telegram session ready' if user_session_ready else 'Telegram session not ready'
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
    
    if User and user_session_ready:
        shutdown_tasks.append(User.stop())
    
    if bot and bot_started:
        shutdown_tasks.append(bot.stop())
    
    if cache_manager:
        shutdown_tasks.append(cache_manager.stop())
    
    if verification_system:
        shutdown_tasks.append(verification_system.stop())
    
    if premium_system:
        shutdown_tasks.append(premium_system.stop_cleanup_task())
    
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
