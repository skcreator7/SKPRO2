#all features keep don't remove
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

# ✅ FIRST: Setup logging before any other imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

logger.info("=" * 80)
logger.info("🚀 STARTING SK4FiLM v8.0 - SYNC MANAGEMENT EDITION")
logger.info("=" * 80)

import aiohttp
import urllib.parse
from quart import Quart, jsonify, request, Response
from hypercorn.asyncio import serve
from hypercorn.config import Config as HyperConfig
from motor.motor_asyncio import AsyncIOMotorClient
import redis.asyncio as redis

# Pyrogram imports - LOGGING IMPROVED
try:
    from pyrogram import Client
    from pyrogram.errors import FloodWait, SessionPasswordNeeded, PhoneCodeInvalid
    from pyrogram.types import Message
    PYROGRAM_AVAILABLE = True
    logger.info("✅ Pyrogram imported successfully")
except ImportError as e:
    logger.error(f"❌ Pyrogram import error: {e}")
    PYROGRAM_AVAILABLE = False
    Client = None
    FloodWait = None
    Message = None

# Import modular components with LOGGING
try:
    from cache import CacheManager
    logger.info("✅ CacheManager imported")
except ImportError as e:
    logger.error(f"❌ CacheManager import error: {e}")
    class CacheManager: pass

try:
    from verification import VerificationSystem
    logger.info("✅ VerificationSystem imported")
except ImportError as e:
    logger.error(f"❌ VerificationSystem import error: {e}")
    class VerificationSystem: pass

try:
    from premium import PremiumSystem, PremiumTier
    logger.info("✅ PremiumSystem imported")
except ImportError as e:
    logger.error(f"❌ PremiumSystem import error: {e}")
    class PremiumSystem: pass
    class PremiumTier: pass

try:
    from poster_fetching import PosterFetcher
    logger.info("✅ PosterFetcher imported")
except ImportError as e:
    logger.error(f"❌ PosterFetcher import error: {e}")
    class PosterFetcher: pass

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
    UTILS_AVAILABLE = True
    logger.info("✅ Utils imported")
except ImportError as e:
    logger.error(f"❌ Utils import error: {e}")
    UTILS_AVAILABLE = False
    
    def normalize_title(title): 
        logger.debug(f"Fallback normalize_title: {title}")
        return title.lower().strip() if title else ""
    
    def extract_title_smart(text): 
        logger.debug(f"Fallback extract_title_smart: {text[:50]}")
        if not text: return ""
        lines = text.split('\n')
        if lines:
            return lines[0].strip()[:50]
        return text[:50]
    
    def extract_title_from_file(file_name, caption): 
        logger.debug(f"Fallback extract_title_from_file: {file_name}, {caption}")
        if caption:
            return extract_title_smart(caption)
        if file_name:
            return file_name.rsplit('.', 1)[0]
        return "Unknown"
    
    def format_size(size): 
        logger.debug(f"Fallback format_size: {size}")
        if not size: return "Unknown"
        if size < 1024*1024:
            return f"{size/1024:.1f} KB"
        elif size < 1024*1024*1024:
            return f"{size/(1024*1024):.1f} MB"
        else:
            return f"{size/(1024*1024*1024):.2f} GB"
    
    def detect_quality(filename): 
        logger.debug(f"Fallback detect_quality: {filename}")
        if not filename: return "480p"
        fl = filename.lower()
        if '2160p' in fl or '4k' in fl:
            return "2160p"
        elif '1080p' in fl:
            return "1080p"
        elif '720p' in fl:
            return "720p"
        elif '480p' in fl:
            return "480p"
        return "480p"
    
    def is_video_file(file_name): 
        logger.debug(f"Fallback is_video_file: {file_name}")
        if not file_name: return False
        video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.mpg', '.mpeg']
        file_name_lower = file_name.lower()
        return any(file_name_lower.endswith(ext) for ext in video_extensions)
    
    def format_post(text): 
        logger.debug(f"Fallback format_post: {text[:50]}")
        if not text: return ""
        text = html.escape(text)
        text = re.sub(r'(https?://[^\s]+)', r'<a href="\1" target="_blank" style="color:#00ccff">\1</a>', text)
        return text.replace('\n', '<br>')
    
    def is_new(date): 
        logger.debug(f"Fallback is_new: {date}")
        try:
            if isinstance(date, str):
                date = datetime.fromisoformat(date.replace('Z', '+00:00'))
            hours = (datetime.now() - date.replace(tzinfo=None)).total_seconds() / 3600
            return hours <= 48
        except:
            return False

# Import bot_handlers AFTER all other imports
try:
    from bot_handlers import SK4FiLMBot, setup_bot_handlers
    BOT_HANDLERS_AVAILABLE = True
    logger.info("✅ Bot handlers imported")
except ImportError as e:
    logger.error(f"❌ Bot handlers import error: {e}")
    BOT_HANDLERS_AVAILABLE = False
    SK4FiLMBot = None
    setup_bot_handlers = None

# ✅ ULTRA-FAST LOADING OPTIMIZATIONS WITH DETAILED LOGGING

# Reduce log noise but keep important logs
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
        logger.info("✅ PerformanceMonitor initialized")
    
    def measure(self, name):
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start = time.perf_counter()
                logger.debug(f"⏱️ Starting measurement for: {name}")
                result = await func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                
                self._record(name, elapsed)
                logger.debug(f"⏱️ Completed {name} in {elapsed:.3f}s")
                return result
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start = time.perf_counter()
                logger.debug(f"⏱️ Starting measurement for: {name}")
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                
                self._record(name, elapsed)
                logger.debug(f"⏱️ Completed {name} in {elapsed:.3f}s")
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
            logger.warning(f"⚠️ {name} took {elapsed:.3f}s (slow operation)")
    
    def get_stats(self):
        logger.debug("📊 Performance stats requested")
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
    
    # Channel Configuration - OPTIMIZED
    MAIN_CHANNEL_ID = -1001891090100
    TEXT_CHANNEL_IDS = [-1001891090100, -1002024811395]  # Only active channels
    FILE_CHANNEL_ID = -1001768249569  # ✅ FILE CHANNEL FOR SYNC MANAGEMENT
    
    logger.info(f"📊 Channel Configuration:")
    logger.info(f"   • MAIN_CHANNEL_ID: {MAIN_CHANNEL_ID}")
    logger.info(f"   • TEXT_CHANNEL_IDS: {TEXT_CHANNEL_IDS}")
    logger.info(f"   • FILE_CHANNEL_ID: {FILE_CHANNEL_ID}")
    
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
        logger.debug(f"Generating poster URL for: {title} ({year})")
        if not title:
            return f"https://via.placeholder.com/300x450/1a1a2e/ffffff?text=No+Poster"
        
        encoded_title = urllib.parse.quote(title)
        return f"{Config.BACKEND_URL}/api/poster?title={encoded_title}&year={year}"

# FAST INITIALIZATION
app = Quart(__name__)
app.config['JSON_SORT_KEYS'] = False  # Faster JSON serialization
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False  # Disable pretty printing for speed

logger.info("✅ Quart app initialized")

# CORS headers
@app.after_request
async def add_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['X-SK4FiLM-Version'] = '8.0-SYNC-ONLY'
    response.headers['X-Response-Time'] = f"{time.perf_counter():.3f}"
    return response

# GLOBAL VARIABLES - FAST ACCESS
mongo_client = None
db = None
files_col = None
verification_col = None
poster_col = None

logger.info("✅ Global variables initialized")

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

# Telegram session checker flag
telegram_initialized = False

# CHANNEL CONFIGURATION - CACHED
CHANNEL_CONFIG = {
    -1001891090100: {'name': 'SK4FiLM Main', 'type': 'text', 'search_priority': 1},
    -1002024811395: {'name': 'SK4FiLM Updates', 'type': 'text', 'search_priority': 2},
    -1001768249569: {'name': 'SK4FiLM Files', 'type': 'file', 'search_priority': 0, 'sync_manage': True}
}

logger.info("✅ Channel configuration loaded")

# ============================================================================
# ✅ TELEGRAM CHANNEL SYNC MANAGEMENT (DELETE SYNC ONLY)
# ============================================================================

class ChannelSyncManager:
    def __init__(self):
        self.last_checked_message_id = {}
        self.is_monitoring = False
        self.monitoring_task = None
        self.deleted_count = 0
        self.last_sync = time.time()
        logger.info("✅ ChannelSyncManager initialized")
    
    async def start_sync_monitoring(self):
        """Start monitoring Telegram channel for deletions sync"""
        logger.info("🔄 Starting sync monitoring...")
        
        if not User or not user_session_ready:
            logger.warning("⚠️ Cannot start sync monitoring - User session not ready")
            return
        
        if self.is_monitoring:
            logger.info("✅ Channel sync monitoring already running")
            return
        
        logger.info("👁️ Starting Telegram channel sync monitoring (Delete Sync Only)...")
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self.monitor_channel_sync())
        logger.info("✅ Sync monitoring task created")
    
    async def stop_sync_monitoring(self):
        """Stop sync monitoring"""
        logger.info("🛑 Stopping sync monitoring...")
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("✅ Channel sync monitoring stopped")
    
    async def monitor_channel_sync(self):
        """Monitor Telegram channel for deletions sync"""
        logger.info(f"🔍 Monitoring FILE_CHANNEL_ID: {Config.FILE_CHANNEL_ID} for deletions sync")
        
        while self.is_monitoring:
            try:
                logger.debug("🔄 Running sync check...")
                await self.sync_deletions_from_telegram()
                logger.debug(f"⏳ Waiting {Config.MONITOR_INTERVAL} seconds for next sync...")
                await asyncio.sleep(Config.MONITOR_INTERVAL)
            except asyncio.CancelledError:
                logger.info("🔌 Sync monitoring cancelled")
                break
            except Exception as e:
                logger.error(f"❌ Channel sync monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def sync_deletions_from_telegram(self):
        """Sync deletions from Telegram to MongoDB"""
        try:
            logger.info("🔄 Running deletion sync from Telegram...")
            
            if not files_col or not User:
                logger.warning("⚠️ Files collection or User client not available")
                return
            
            current_time = time.time()
            if current_time - self.last_sync < 300:  # Run every 5 minutes
                logger.debug("⏳ Sync too recent, skipping")
                return
            
            self.last_sync = current_time
            logger.debug("🔍 Checking for deleted files...")
            
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
                logger.info("📭 No files in database to sync")
                return
            
            logger.info(f"📊 Checking {len(message_ids_in_db)} files for sync...")
            
            deleted_count = 0
            batch_size = 50
            
            for i in range(0, len(message_ids_in_db), batch_size):
                batch = message_ids_in_db[i:i + batch_size]
                logger.debug(f"🔍 Checking batch {i//batch_size + 1} ({len(batch)} files)...")
                
                try:
                    messages = await safe_telegram_operation(
                        User.get_messages,
                        Config.FILE_CHANNEL_ID,
                        batch
                    )
                    
                    existing_ids = set()
                    if isinstance(messages, list):
                        for msg in messages:
                            if msg and hasattr(msg, 'id'):
                                existing_ids.add(msg.id)
                    elif messages and hasattr(messages, 'id'):
                        existing_ids.add(messages.id)
                    
                    deleted_ids = [msg_id for msg_id in batch if msg_id not in existing_ids]
                    
                    if deleted_ids:
                        logger.info(f"🗑️ Found {len(deleted_ids)} deleted files in batch")
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
                logger.info("✅ Sync completed: No deletions found")
            
        except Exception as e:
            logger.error(f"❌ Error syncing deletions: {e}")
    
    async def manual_sync(self):
        """Manual sync trigger"""
        logger.info("🔄 Manual sync triggered")
        await self.sync_deletions_from_telegram()

# Create global sync manager instance
channel_sync_manager = ChannelSyncManager()

# ============================================================================
# ✅ SMART FILE INDEXING WITH DUPLICATE PREVENTION
# ============================================================================

async def generate_file_hash(message):
    """Generate unique hash for file to detect duplicates"""
    try:
        logger.debug(f"Generating hash for message {message.id}")
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
            logger.debug("No document or video found for hash generation")
            return None
        
        if message.caption:
            caption_hash = hashlib.md5(message.caption.encode()).hexdigest()[:12]
            hash_parts.append(f"cap_{caption_hash}")
        
        final_hash = "_".join(hash_parts)
        logger.debug(f"Generated hash: {final_hash[:50]}...")
        return final_hash
    except Exception as e:
        logger.error(f"Hash generation error: {e}")
        return None

@performance_monitor.measure("smart_file_indexing")
async def index_single_file_smart(message):
    """
    Smart file indexing with:
    1. Duplicate prevention (multiple checks)
    2. Delete sync tracking
    3. Hash-based duplicate detection
    """
    try:
        logger.info(f"📁 Indexing file from message {message.id}")
        
        if not files_col:
            logger.error("❌ Files collection not available")
            return False
        
        # ✅ STEP 1: BASIC VALIDATION
        if not message or (not message.document and not message.video):
            logger.debug(f"❌ Message {message.id} is not a document or video")
            return False
        
        # ✅ STEP 2: CHECK IF ALREADY EXISTS
        logger.debug(f"🔍 Checking if message {message.id} already exists...")
        
        existing_by_id = await files_col.find_one({
            'channel_id': Config.FILE_CHANNEL_ID,
            'message_id': message.id
        }, {'_id': 1, 'title': 1})
        
        if existing_by_id:
            logger.info(f"📝 Skipping - Already indexed: {message.id}")
            return True
        
        title = await extract_title_from_telegram_msg_cached(message)
        if not title:
            logger.debug(f"📝 Skipping - No title: {message.id}")
            return False
        
        logger.info(f"📝 Title extracted: {title}")
        normalized_title = normalize_title_cached(title)
        
        # Layer 2: Generate file hash for duplicate detection
        file_hash = await generate_file_hash(message)
        
        if file_hash:
            existing_by_hash = await files_col.find_one({
                'channel_id': Config.FILE_CHANNEL_ID,
                'file_hash': file_hash
            }, {'_id': 1, 'title': 1})
            
            if existing_by_hash:
                logger.info(f"🔁 Skipping duplicate (same file hash): {title}")
                return False
        
        # Layer 3: Check by file_id
        file_id = None
        file_size = 0
        
        if message.document:
            file_id = message.document.file_id
            file_size = message.document.file_size or 0
        elif message.video:
            file_id = message.video.file_id
            file_size = message.video.file_size or 0
        
        if file_id:
            existing_by_file_id = await files_col.find_one({
                'channel_id': Config.FILE_CHANNEL_ID,
                'file_id': file_id
            }, {'_id': 1, 'title': 1})
            
            if existing_by_file_id:
                logger.info(f"🔁 Skipping duplicate (same file_id): {title}")
                return False
        
        # ✅ STEP 3: CREATE DOCUMENT
        logger.debug(f"📄 Creating document for message {message.id}")
        
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
            'file_id': file_id,
            'file_size': file_size,
            'file_hash': file_hash,
            'status': 'active',
            'duplicate_checked': True
        }
        
        if message.document:
            doc.update({
                'file_name': message.document.file_name or '',
                'quality': detect_quality(message.document.file_name or ''),
                'is_video_file': is_video_file(message.document.file_name or ''),
                'caption': message.caption or '',
                'mime_type': message.document.mime_type or ''
            })
        elif message.video:
            doc.update({
                'file_name': message.video.file_name or 'video.mp4',
                'quality': detect_quality(message.video.file_name or ''),
                'is_video_file': True,
                'caption': message.caption or '',
                'duration': message.video.duration if hasattr(message.video, 'duration') else 0,
                'width': message.video.width if hasattr(message.video, 'width') else 0,
                'height': message.video.height if hasattr(message.video, 'height') else 0
            })
        else:
            return False
        
        # Extract thumbnail if video
        if doc['is_video_file'] and User:
            try:
                logger.debug(f"🖼️ Extracting thumbnail for video {message.id}")
                thumbnail_url = await extract_video_thumbnail(User, message)
                if thumbnail_url:
                    doc['thumbnail'] = thumbnail_url
                    doc['thumbnail_source'] = 'video_direct'
                    logger.debug(f"✅ Thumbnail extracted for {message.id}")
            except Exception as e:
                logger.warning(f"⚠️ Thumbnail extraction failed: {e}")
        
        # ✅ STEP 4: INSERT WITH ERROR HANDLING
        try:
            await files_col.insert_one(doc)
            
            file_type = "📹 Video" if doc['is_video_file'] else "📄 File"
            size_str = format_size(file_size) if file_size > 0 else "Unknown size"
            
            logger.info(f"✅ {file_type} indexed: {title}")
            logger.info(f"   📊 Size: {size_str} | Quality: {doc.get('quality', 'Unknown')} | ID: {message.id}")
            
            return True
        except Exception as e:
            if "duplicate key error" in str(e).lower():
                logger.info(f"📝 Race condition duplicate: {message.id}")
                return True
            else:
                logger.error(f"❌ Error inserting document: {e}")
                return False
        
    except Exception as e:
        logger.error(f"❌ Smart indexing error: {e}")
        return False

# ============================================================================
# ✅ BACKGROUND INDEXING WITH SYNC MANAGEMENT
# ============================================================================

async def index_files_background_smart():
    """Smart background indexing with sync management features"""
    logger.info("📁 Starting SMART background indexing with sync management...")
    
    if not User or files_col is None or not user_session_ready:
        logger.warning("⚠️ Cannot start smart indexing - User session not ready")
        logger.info("📊 Status check:")
        logger.info(f"   User: {'✅' if User else '❌'}")
        logger.info(f"   files_col: {'✅' if files_col else '❌'}")
        logger.info(f"   user_session_ready: {'✅' if user_session_ready else '❌'}")
        return
    
    try:
        # First check Telegram channel access
        logger.info("🔍 Checking Telegram channel access...")
        try:
            chat = await User.get_chat(Config.FILE_CHANNEL_ID)
            logger.info(f"✅ FILE Channel accessible: {chat.title}")
        except Exception as e:
            logger.error(f"❌ Cannot access FILE channel: {e}")
            return
        
        # ✅ STEP 1: SETUP MONGODB INDEXES FOR SYNC MANAGEMENT
        await setup_mongodb_sync_indexes()
        
        # ✅ STEP 2: GET LAST INDEXED MESSAGE ID
        last_indexed = await files_col.find_one(
            {"channel_id": Config.FILE_CHANNEL_ID}, 
            sort=[('message_id', -1)],
            projection={'message_id': 1, 'title': 1}
        )
        
        last_message_id = last_indexed['message_id'] if last_indexed else 0
        last_title = last_indexed.get('title', 'Unknown') if last_indexed else 'None'
        
        logger.info(f"🔄 Starting from message ID: {last_message_id}")
        logger.info(f"📝 Last indexed file: {last_title}")
        
        # ✅ STEP 3: EXPLORE CHANNEL FIRST
        logger.info("🔍 Exploring FILE channel to understand structure...")
        message_stats = {
            'total': 0,
            'files': 0,
            'texts': 0,
            'others': 0
        }
        
        sample_messages = []
        async for msg in safe_telegram_generator(
            User.get_chat_history, 
            Config.FILE_CHANNEL_ID,
            limit=30
        ):
            message_stats['total'] += 1
            
            if msg.document:
                message_stats['files'] += 1
                file_name = msg.document.file_name or 'Unknown'
                sample_messages.append(f"📄 Document: {file_name}")
            elif msg.video:
                message_stats['files'] += 1
                file_name = msg.video.file_name or 'Unknown'
                sample_messages.append(f"🎬 Video: {file_name}")
            elif msg.text:
                message_stats['texts'] += 1
                sample_messages.append(f"📝 Text: {msg.text[:30]}...")
            else:
                message_stats['others'] += 1
        
        logger.info(f"📊 Channel exploration:")
        logger.info(f"   Total messages: {message_stats['total']}")
        logger.info(f"   Files/Documents: {message_stats['files']}")
        logger.info(f"   Text messages: {message_stats['texts']}")
        logger.info(f"   Others: {message_stats['others']}")
        
        if message_stats['files'] == 0:
            logger.warning("⚠️ No files found in FILE channel!")
            logger.warning("   Check if channel actually contains files")
            
            # Show sample messages
            if sample_messages:
                logger.info("📋 Sample messages from channel:")
                for sample in sample_messages[:5]:
                    logger.info(f"   {sample}")
            return
        
        # ✅ STEP 4: FETCH AND PROCESS NEW MESSAGES
        total_indexed = 0
        total_skipped = 0
        
        logger.info("🔍 Fetching new messages from Telegram...")
        messages = []
        
        async for msg in safe_telegram_generator(
            User.get_chat_history, 
            Config.FILE_CHANNEL_ID,
            limit=200  # Increased limit
        ):
            if msg.id <= last_message_id:
                logger.debug(f"Reached last indexed message {msg.id}, stopping")
                break
            
            if msg and (msg.document or msg.video):
                messages.append(msg)
        
        messages.reverse()
        logger.info(f"📥 Found {len(messages)} potential new files to index")
        
        if not messages:
            logger.info("📭 No new files to index")
        else:
            for idx, msg in enumerate(messages, 1):
                try:
                    logger.debug(f"📄 Processing message {idx}/{len(messages)} (ID: {msg.id})")
                    success = await index_single_file_smart(msg)
                    if success:
                        total_indexed += 1
                    else:
                        total_skipped += 1
                    
                    await asyncio.sleep(0.3)
                    
                    if (total_indexed + total_skipped) % 5 == 0:
                        logger.info(f"📊 Progress: {total_indexed} new, {total_skipped} skipped")
                    
                except Exception as e:
                    logger.error(f"❌ Error processing message {msg.id}: {e}")
                    continue
        
        # ✅ STEP 5: START SYNC MONITORING
        logger.info(f"✅ Smart indexing complete: {total_indexed} NEW files indexed")
        
        if total_skipped > 0:
            logger.info(f"📝 Skipped {total_skipped} files (duplicates or errors)")
        
        # Start sync monitoring for deletions
        await channel_sync_manager.start_sync_monitoring()
        
        # Initial sync check
        await channel_sync_manager.sync_deletions_from_telegram()
        
        # Log final stats
        total_in_db = await files_col.count_documents({}) if files_col else 0
        logger.info(f"📊 FINAL STATS: {total_in_db} total files in database")
        
        logger.info("🚀 Sync management system activated!")
        logger.info(f"   • Delete sync monitoring: Every {Config.MONITOR_INTERVAL//60} minutes")
        logger.info(f"   • Auto expiry: DISABLED (No TTL)")
        logger.info(f"   • Duplicate prevention: Active")
        
    except Exception as e:
        logger.error(f"❌ Smart background indexing error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

async def setup_mongodb_sync_indexes():
    """Setup MongoDB indexes for sync management (NO TTL)"""
    try:
        logger.info("🗂️ Setting up MongoDB indexes for sync management...")
        
        if not files_col:
            logger.error("❌ Files collection not available")
            return
        
        # 1. Compound index for fast channel + message lookups
        try:
            await files_col.create_index(
                [("channel_id", 1), ("message_id", 1)],
                unique=True,
                name="channel_message_unique"
            )
            logger.info("✅ Unique index created (channel_id + message_id)")
        except Exception as e:
            logger.debug(f"Unique index may already exist: {e}")
        
        # 2. Index for file hash (duplicate detection)
        try:
            await files_col.create_index(
                [("channel_id", 1), ("file_hash", 1)],
                name="file_hash_index"
            )
            logger.info("✅ File hash index created (duplicate detection)")
        except Exception as e:
            logger.debug(f"File hash index may already exist: {e}")
        
        # 3. Index for file_id (Telegram duplicate detection)
        try:
            await files_col.create_index(
                [("channel_id", 1), ("file_id", 1)],
                name="file_id_index"
            )
            logger.info("✅ File ID index created")
        except Exception as e:
            logger.debug(f"File ID index may already exist: {e}")
        
        # 4. Index for last_checked (monitoring performance)
        try:
            await files_col.create_index(
                [("channel_id", 1), ("last_checked", 1)],
                name="last_checked_index"
            )
            logger.info("✅ Last checked index created")
        except Exception as e:
            logger.debug(f"Last checked index may already exist: {e}")
        
        # 5. Text index for search
        try:
            await files_col.create_index(
                [("normalized_title", "text"), ("title", "text")],
                name="text_search_index"
            )
            logger.info("✅ Text search index created")
        except Exception as e:
            logger.debug(f"Text index may already exist: {e}")
        
        logger.info("✅ MongoDB indexes setup complete (NO TTL INDEXES)")
        
    except Exception as e:
        logger.error(f"❌ Error setting up MongoDB indexes: {e}")

# ============================================================================
# ✅ CHECK TELEGRAM SESSION DETAILS
# ============================================================================

async def check_telegram_session_details():
    """Check detailed Telegram session status"""
    logger.info("🔍 Checking Telegram session details...")
    
    if not User or not user_session_ready:
        logger.error("❌ Telegram User session not ready")
        return False
    
    try:
        # Get user info
        me = await User.get_me()
        logger.info(f"👤 User: {me.first_name} (@{me.username})")
        
        # Test MAIN_CHANNEL access
        try:
            main_chat = await User.get_chat(Config.MAIN_CHANNEL_ID)
            logger.info(f"📺 MAIN Channel: {main_chat.title}")
            
            # Try to get a message
            message_count = 0
            async for msg in User.get_chat_history(Config.MAIN_CHANNEL_ID, limit=5):
                message_count += 1
                if message_count == 1:
                    logger.info(f"📝 Sample message: {msg.text[:50] if msg.text else 'No text'}...")
            
            logger.info(f"📊 MAIN Channel messages accessible: {message_count} messages")
        except Exception as e:
            logger.error(f"❌ MAIN Channel access error: {e}")
        
        # Test FILE_CHANNEL access
        try:
            file_chat = await User.get_chat(Config.FILE_CHANNEL_ID)
            logger.info(f"📁 FILE Channel: {file_chat.title}")
            
            # Count messages in FILE_CHANNEL
            message_count = 0
            file_count = 0
            async for msg in User.get_chat_history(Config.FILE_CHANNEL_ID, limit=20):
                message_count += 1
                if msg and (msg.document or msg.video):
                    file_count += 1
                    file_name = msg.document.file_name if msg.document else msg.video.file_name
                    logger.info(f"📄 Found file {file_count}: {file_name}")
            
            logger.info(f"📊 FILE Channel Stats: {message_count} messages, {file_count} files")
            
            if file_count == 0:
                logger.warning("⚠️ No files found in FILE channel!")
                logger.warning("   Please check if the channel actually contains files")
            else:
                logger.info("✅ Files found in FILE channel - Ready for indexing")
            
        except Exception as e:
            logger.error(f"❌ FILE Channel access error: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Telegram session check failed: {e}")
        return False

# ============================================================================
# TELEGRAM SESSION GENERATOR FUNCTION (BUILT-IN)
# ============================================================================

async def generate_telegram_session():
    """Generate Telegram session string if not available"""
    logger.info("🎯 Generating new Telegram session...")
    
    if not PYROGRAM_AVAILABLE:
        logger.error("❌ Pyrogram not installed. Run: pip install pyrogram")
        return None
    
    try:
        api_id = input("Enter your API_ID from https://my.telegram.org: ").strip()
        api_hash = input("Enter your API_HASH: ").strip()
        
        if not api_id.isdigit() or not api_hash:
            logger.error("❌ Invalid API credentials")
            return None
        
        temp_client = Client(
            "sk4film_temp_session",
            api_id=int(api_id),
            api_hash=api_hash,
            in_memory=True
        )
        
        await temp_client.start()
        
        me = await temp_client.get_me()
        logger.info(f"✅ Logged in as: {me.first_name} (@{me.username})")
        
        session_string = await temp_client.export_session_string()
        
        logger.info("🎉 SESSION GENERATED SUCCESSFULLY!")
        logger.info(f"Session String: {session_string}")
        
        logger.info("🔍 Testing channel access...")
        try:
            chat = await temp_client.get_chat(Config.MAIN_CHANNEL_ID)
            logger.info(f"✅ Channel accessible: {chat.title}")
        except Exception as e:
            logger.warning(f"⚠️ Cannot access channel: {e}")
            logger.warning("Make sure you're a member of the channel!")
        
        await temp_client.stop()
        
        os.environ["API_ID"] = api_id
        os.environ["API_HASH"] = api_hash
        os.environ["USER_SESSION_STRING"] = session_string
        
        logger.info("✅ Environment variables updated")
        logger.info("📋 Set these in your deployment:")
        logger.info(f'export API_ID="{api_id}"')
        logger.info(f'export API_HASH="{api_hash}"')
        logger.info(f'export USER_SESSION_STRING="{session_string}"')
        
        return session_string
        
    except Exception as e:
        logger.error(f"❌ Session generation failed: {e}")
        return None

# ============================================================================
# TELEGRAM INITIALIZATION WITH AUTO-FIX
# ============================================================================

@performance_monitor.measure("telegram_init")
async def init_telegram_clients():
    """Smart Telegram client initialization with auto-fix capabilities"""
    global User, bot, bot_started, user_session_ready, telegram_initialized
    
    logger.info("=" * 60)
    logger.info("🚀 TELEGRAM CLIENT INITIALIZATION")
    logger.info("=" * 60)
    
    if not PYROGRAM_AVAILABLE:
        logger.error("❌ CRITICAL: Pyrogram not installed!")
        logger.error("   Run: pip install pyrogram")
        return False
    
    logger.info("🔍 Checking environment variables...")
    
    env_status = {
        "API_ID": Config.API_ID > 0,
        "API_HASH": bool(Config.API_HASH and len(Config.API_HASH) > 10),
        "USER_SESSION_STRING": bool(Config.USER_SESSION_STRING and len(Config.USER_SESSION_STRING) > 100),
        "BOT_TOKEN": bool(Config.BOT_TOKEN)
    }
    
    for key, status in env_status.items():
        logger.info(f"   {key}: {'✅' if status else '❌'}")
    
    if not env_status["USER_SESSION_STRING"]:
        logger.warning("⚠️ Session string missing or invalid")
    
    if env_status["API_ID"] and env_status["API_HASH"] and env_status["USER_SESSION_STRING"]:
        logger.info("\n👤 Initializing User Client...")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"   Attempt {attempt + 1}/{max_retries}")
                
                User = Client(
                    "sk4film_user",
                    api_id=Config.API_ID,
                    api_hash=Config.API_HASH,
                    session_string=Config.USER_SESSION_STRING,
                    sleep_threshold=30,
                    max_concurrent_transmissions=1,
                    in_memory=True,
                    no_updates=True
                )
                
                await asyncio.wait_for(User.start(), timeout=15)
                
                me = await User.get_me()
                logger.info(f"✅ User Client Ready: {me.first_name}")
                logger.info(f"   User ID: {me.id}, Username: @{me.username}")
                
                try:
                    chat = await User.get_chat(Config.FILE_CHANNEL_ID)
                    logger.info(f"✅ FILE Channel Access: {chat.title} (ID: {Config.FILE_CHANNEL_ID})")
                    
                    try:
                        async for msg in User.get_chat_history(Config.FILE_CHANNEL_ID, limit=1):
                            if msg:
                                logger.info(f"✅ Can fetch FILE channel messages: YES")
                                break
                    except:
                        logger.warning("⚠️ Can read channel but may need admin rights")
                    
                    user_session_ready = True
                    
                except Exception as e:
                    logger.warning(f"⚠️ FILE Channel access issue: {e}")
                    logger.warning(f"   Make sure you're a member of channel {Config.FILE_CHANNEL_ID}!")
                    user_session_ready = False
                
                break
                
            except asyncio.TimeoutError:
                logger.error(f"⏰ Timeout on attempt {attempt + 1}")
                if User:
                    await User.stop()
                    User = None
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                    
            except Exception as e:
                error_msg = str(e)
                logger.error(f"❌ Attempt {attempt + 1} failed: {error_msg}")
                
                if "SessionRevoked" in error_msg or "session expired" in error_msg.lower():
                    logger.critical("🚨 SESSION EXPIRED! Need new session string")
                    break
                    
                elif "AUTH_KEY_UNREGISTERED" in error_msg:
                    logger.critical("🚨 AUTH KEY INVALID! Session string wrong")
                    break
                    
                elif "API_ID_INVALID" in error_msg:
                    logger.critical("🚨 API_ID INVALID! Check from my.telegram.org")
                    break
                
                if User:
                    await User.stop()
                    User = None
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
    
    if env_status["BOT_TOKEN"]:
        logger.info("\n🤖 Initializing Bot Client...")
        try:
            bot = Client(
                "sk4film_bot",
                api_id=Config.API_ID,
                api_hash=Config.API_HASH,
                bot_token=Config.BOT_TOKEN,
                sleep_threshold=30,
                workers=3,
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
    else:
        logger.warning("⚠️ Bot token not configured")
        bot_started = False
    
    logger.info("\n" + "=" * 60)
    logger.info("📊 INITIALIZATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"User Session: {'✅ READY' if user_session_ready else '❌ NOT READY'}")
    logger.info(f"Bot Session: {'✅ READY' if bot_started else '❌ NOT READY'}")
    logger.info(f"Movies Fetch: {'✅ ENABLED' if user_session_ready else '❌ DISABLED'}")
    logger.info(f"FILE Channel ID: {Config.FILE_CHANNEL_ID}")
    logger.info(f"Delete Sync: {'✅ ENABLED' if user_session_ready else '❌ DISABLED'}")
    
    if not user_session_ready:
        logger.warning("\n⚠️ WARNING: Movies will be EMPTY without Telegram session")
        logger.warning("   To fix:")
        logger.warning(f"   1. Check API_ID, API_HASH, USER_SESSION_STRING")
        logger.warning(f"   2. Ensure you're member of channel {Config.FILE_CHANNEL_ID}")
        logger.warning("   3. Regenerate session if expired")
    
    telegram_initialized = True
    return user_session_ready or bot_started

# CACHED FUNCTIONS
@lru_cache(maxsize=10000)
def channel_name_cached(cid):
    """Ultra-fast cached channel name lookup"""
    name = CHANNEL_CONFIG.get(cid, {}).get('name', f"Channel {cid}")
    logger.debug(f"Channel name cached for {cid}: {name}")
    return name

@lru_cache(maxsize=5000)
def normalize_title_cached(title: str) -> str:
    """Cached title normalization"""
    normalized = normalize_title(title)
    logger.debug(f"Title normalized: '{title}' -> '{normalized}'")
    return normalized

@lru_cache(maxsize=1000)
def channel_name(channel_id):
    """Get channel name from channel ID (compatibility function)"""
    name = channel_name_cached(channel_id)
    logger.debug(f"Channel name for {channel_id}: {name}")
    return name

async def get_telegram_video_thumbnail(user_client, channel_id, message_id):
    """Get video thumbnail from Telegram"""
    logger.debug(f"Getting thumbnail for {channel_id}/{message_id}")
    try:
        if not user_client or not user_session_ready:
            logger.warning("User client not ready for thumbnail")
            return None
            
        msg = await safe_telegram_operation(
            user_client.get_messages,
            channel_id,
            message_id
        )
        
        if msg and (msg.video or msg.document):
            thumbnail = await extract_video_thumbnail(user_client, msg)
            logger.debug(f"Thumbnail {'found' if thumbnail else 'not found'} for {message_id}")
            return thumbnail
        
        return None
    except Exception as e:
        logger.error(f"Get thumbnail error: {e}")
        return None

async def verify_user_api(user_id, verification_url=None):
    """Verify user API endpoint implementation"""
    logger.info(f"Verifying user {user_id}")
    try:
        if verification_system:
            if hasattr(verification_system, 'verify_user_api'):
                result = await verification_system.verify_user_api(user_id, verification_url)
                return result
            elif hasattr(verification_system, 'create_verification_link'):
                verification_data = await verification_system.create_verification_link(user_id)
                return {
                    'verified': False,
                    'verification_url': verification_data.get('short_url'),
                    'user_id': user_id,
                    'expires_at': verification_data.get('verification_expires_at')
                }
        
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
    """Get indexing status API implementation"""
    logger.info("Getting index status")
    try:
        total_files = await files_col.count_documents({}) if files_col else 0
        video_files = await files_col.count_documents({'is_video_file': True}) if files_col else 0
        
        if files_col:
            channel_files = await files_col.count_documents({
                "channel_id": Config.FILE_CHANNEL_ID
            })
        else:
            channel_files = 0
        
        logger.info(f"Index status: {total_files} total, {video_files} videos, {channel_files} in file channel")
        
        return {
            'indexed_files': total_files,
            'video_files': video_files,
            'file_channel_files': channel_files,
            'sync_monitoring': channel_sync_manager.is_monitoring,
            'deleted_by_sync': channel_sync_manager.deleted_count,
            'user_session_ready': user_session_ready,
            'last_update': datetime.now().isoformat(),
            'status': 'active' if user_session_ready else 'inactive',
            'note': 'NO TTL - Sync deletions only'
        }
    except Exception as e:
        logger.error(f"Index status API error: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }

# ASYNC CACHE DECORATOR
def async_cache_with_ttl(maxsize=128, ttl=300):
    """Async cache decorator with TTL"""
    cache = {}
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            key_parts = [func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
            key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
            
            now = time.time()
            
            if key in cache:
                value, timestamp = cache[key]
                if now - timestamp < ttl:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return value
            
            result = await func(*args, **kwargs)
            
            cache[key] = (result, now)
            
            if len(cache) > maxsize:
                oldest_key = min(cache.keys(), key=lambda k: cache[k][1])
                del cache[oldest_key]
            
            return result
        return wrapper
    return decorator

# OPTIMIZED FLOOD PROTECTION
class TurboFloodProtection:
    def __init__(self):
        self.request_buckets = {}
        self.last_cleanup = time.time()
        self.request_count = 0
        logger.info("✅ TurboFloodProtection initialized")
    
    async def wait_if_needed(self, user_id=None):
        """Optimized flood protection"""
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
                logger.warning(f"⏳ Flood protection: waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
                self.request_buckets = {}
        
        self.request_buckets[bucket_key] += 1
        self.request_count += 1
    
    def _cleanup_buckets(self, current_time):
        """Cleanup old buckets"""
        current_bucket = int(current_time // 10)
        old_buckets = [k for k in self.request_buckets.keys() if k < current_bucket - 6]
        for bucket in old_buckets:
            del self.request_buckets[bucket]

turbo_flood_protection = TurboFloodProtection()

# ASYNC CONNECTION POOL
class ConnectionPool:
    def __init__(self, max_connections=20):
        self.max_connections = max_connections
        self.semaphore = asyncio.Semaphore(max_connections)
        self.active_connections = 0
        logger.info(f"✅ ConnectionPool initialized with {max_connections} max connections")
    
    async def acquire(self):
        await self.semaphore.acquire()
        self.active_connections += 1
        logger.debug(f"Connection acquired. Active: {self.active_connections}")
    
    def release(self):
        self.active_connections -= 1
        self.semaphore.release()
        logger.debug(f"Connection released. Active: {self.active_connections}")

http_pool = ConnectionPool(max_connections=Config.MAX_CONCURRENT_REQUESTS)

# SAFE TELEGRAM OPERATIONS WITH TURBO PROTECTION
@performance_monitor.measure("telegram_operation")
async def safe_telegram_operation(operation, *args, **kwargs):
    """Turbo-charged Telegram operations"""
    logger.debug(f"Telegram operation: {operation.__name__}")
    max_retries = 2
    
    for attempt in range(max_retries):
        try:
            await turbo_flood_protection.wait_if_needed()
            result = await operation(*args, **kwargs)
            return result
        except Exception as e:
            if "FloodWait" in str(e):
                wait_match = re.search(r'(\d+)', str(e))
                if wait_match:
                    wait_time = int(wait_match.group(1))
                    logger.warning(f"⏳ Flood wait {wait_time}s")
                    await asyncio.sleep(wait_time + 1)
                    continue
            logger.error(f"Telegram operation failed: {e}")
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(0.5 * (2 ** attempt))
    
    return None

# SAFE ASYNC ITERATOR
async def safe_telegram_generator(operation, *args, limit=None, **kwargs):
    """Optimized Telegram async generator"""
    count = 0
    
    async for item in operation(*args, **kwargs):
        yield item
        count += 1
        
        if count % 10 == 0:
            await asyncio.sleep(0.1)
            await turbo_flood_protection.wait_if_needed()
        
        if limit and count >= limit:
            logger.debug(f"Generator limit reached: {limit}")
            break

# CACHE WARM-UP
async def warm_up_cache():
    """Warm up all caches for instant response"""
    logger.info("🔥 Warming up caches...")
    try:
        if cache_manager and cache_manager.redis_enabled:
            warm_data = {
                "system:warm": True,
                "startup_time": datetime.now().isoformat(),
                "version": "8.0-SYNC-ONLY"
            }
            
            if hasattr(cache_manager, 'batch_set'):
                await cache_manager.batch_set(warm_data, expire_seconds=3600)
        
        logger.info("✅ Cache warm-up complete")
        
    except Exception as e:
        logger.error(f"Cache warm-up error: {e}")

# CACHE CLEANUP TASK
async def cache_cleanup():
    """Background cache cleanup"""
    logger.info("🧹 Starting cache cleanup task")
    while True:
        await asyncio.sleep(1800)
        try:
            if cache_manager and hasattr(cache_manager, 'clear_pattern'):
                await cache_manager.clear_pattern("temp:")
            
            logger.debug("🧹 Cache cleanup completed")
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")

# OPTIMIZED TITLE EXTRACTION
@async_cache_with_ttl(maxsize=1000, ttl=3600)
async def extract_title_from_telegram_msg_cached(msg):
    """Cached title extraction"""
    logger.debug(f"Extracting title from message {msg.id}")
    try:
        caption = msg.caption if hasattr(msg, 'caption') else None
        file_name = None
        
        if msg.document:
            file_name = msg.document.file_name
        elif msg.video:
            file_name = msg.video.file_name
        
        title = extract_title_from_file(file_name, caption)
        logger.debug(f"Extracted title: {title}")
        return title
    except Exception as e:
        logger.error(f"Title extraction error: {e}")
        return None

# VIDEO THUMBNAIL PROCESSING
@performance_monitor.measure("thumbnail_extraction")
async def extract_video_thumbnail(user_client, message):
    """Optimized thumbnail extraction"""
    logger.debug(f"Extracting thumbnail for message {message.id}")
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
                    logger.debug(f"Thumbnail extracted successfully for {message.id}")
                    return f"data:image/jpeg;base64,{thumbnail_data}"
        
        logger.debug(f"No thumbnail found for {message.id}")
        return None
    except Exception as e:
        logger.error(f"Thumbnail extraction failed: {e}")
        return None

# MONGODB INITIALIZATION WITH OPTIMIZATIONS
@performance_monitor.measure("mongodb_init")
async def init_mongodb():
    """Optimized MongoDB initialization"""
    global mongo_client, db, files_col, verification_col, poster_col
    
    logger.info("🔌 MongoDB initialization...")
    
    try:
        mongo_client = AsyncIOMotorClient(
            Config.MONGODB_URI,
            serverSelectionTimeoutMS=10000,
            connectTimeoutMS=10000,
            socketTimeoutMS=15000,
            maxPoolSize=20,
            minPoolSize=5,
            maxIdleTimeMS=30000,
            retryWrites=True,
            retryReads=True,
            ssl=True
        )
        
        await asyncio.wait_for(mongo_client.admin.command('ping'), timeout=5)
        
        db = mongo_client.sk4film
        files_col = db.files
        verification_col = db.verifications
        poster_col = db.posters
        
        logger.info("✅ MongoDB OK - Optimized and Ready (NO TTL)")
        return True
        
    except asyncio.TimeoutError:
        logger.error("❌ MongoDB connection timeout")
        return False
    except Exception as e:
        logger.error(f"❌ MongoDB error: {e}")
        return False

# OLD FILE INDEXING FUNCTION (FOR BACKWARD COMPATIBILITY)
@performance_monitor.measure("file_indexing")
async def index_single_file(message):
    """Optimized single file indexing (legacy function)"""
    logger.info(f"Legacy indexing for message {message.id}")
    return await index_single_file_smart(message)

# OLD BACKGROUND INDEXING FUNCTION (FOR BACKWARD COMPATIBILITY)
async def index_files_background():
    """Optimized background indexing (legacy function)"""
    logger.info("Starting legacy background indexing")
    await index_files_background_smart()

# POSTER FETCHING WITH CACHE
@performance_monitor.measure("poster_fetch")
async def get_poster_guaranteed(title, year=""):
    """Optimized poster fetching"""
    logger.debug(f"Fetching poster for: {title} ({year})")
    if poster_fetcher:
        poster_data = await poster_fetcher.fetch_poster(title, year)
        if poster_data:
            logger.debug(f"Poster found via PosterFetcher: {title}")
            return {
                'poster_url': poster_data.get('url', ''),
                'source': poster_data.get('source', 'custom'),
                'rating': poster_data.get('rating', '0.0'),
                'year': year
            }
    
    logger.debug(f"Using fallback poster for: {title}")
    return {
        'poster_url': Config.get_poster(title, year),
        'source': 'custom',
        'rating': '0.0',
        'year': year,
        'title': title
    }

# ✅ FIXED: ULTRA-FAST SEARCH FUNCTION WITH BETTER TEXT SEARCH
@performance_monitor.measure("multi_channel_search")
@async_cache_with_ttl(maxsize=500, ttl=300)
async def search_movies_multi_channel(query, limit=12, page=1):
    """Turbo-charged multi-channel search with improved text search"""
    logger.info(f"🔍 Multi-channel search for: '{query}' (page {page}, limit {limit})")
    offset = (page - 1) * limit
    
    # Try cache first
    cache_key = f"search:{query}:{page}:{limit}"
    if cache_manager and cache_manager.redis_enabled:
        if hasattr(cache_manager, 'get_search_results'):
            cached_data = await cache_manager.get_search_results(query, page, limit)
            if cached_data:
                logger.info(f"✅ Cache HIT for: {query}")
                return cached_data
    
    query_lower = query.lower()
    posts_dict = {}
    files_dict = {}
    
    # ✅ FIXED: MongoDB search with regex for better matching
    logger.debug("Searching MongoDB files...")
    try:
        if files_col is not None:
            regex_pattern = f".*{re.escape(query)}.*"
            
            cursor = files_col.find(
                {
                    '$or': [
                        {'title': {'$regex': regex_pattern, '$options': 'i'}},
                        {'normalized_title': {'$regex': regex_pattern, '$options': 'i'}},
                        {'file_name': {'$regex': regex_pattern, '$options': 'i'}}
                    ]
                },
                {
                    'title': 1,
                    'normalized_title': 1,
                    'quality': 1,
                    'file_size': 1,
                    'file_name': 1,
                    'thumbnail': 1,
                    'is_video_file': 1,
                    'thumbnail_source': 1,
                    'channel_id': 1,
                    'message_id': 1,
                    'date': 1,
                    '_id': 0
                }
            ).limit(limit * 2)
            
            files_found = 0
            async for doc in cursor:
                try:
                    files_found += 1
                    norm_title = doc.get('normalized_title', normalize_title_cached(doc['title']))
                    quality = doc['quality']
                    
                    if norm_title not in files_dict:
                        files_dict[norm_title] = {
                            'title': doc['title'], 
                            'quality_options': {}, 
                            'date': doc['date'].isoformat() if isinstance(doc['date'], datetime) else doc['date'],
                            'thumbnail': doc.get('thumbnail'),
                            'is_video_file': doc.get('is_video_file', False),
                            'thumbnail_source': doc.get('thumbnail_source', 'unknown'),
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
                except Exception as e:
                    logger.error(f"Error processing document: {e}")
                    continue
            
            logger.debug(f"Found {files_found} files in MongoDB")
    except Exception as e:
        logger.error(f"File search error: {e}")
    
    # Telegram channel search (async)
    logger.debug("Searching Telegram channels...")
    if user_session_ready:
        channel_tasks = []
        
        async def search_channel(channel_id):
            channel_posts = {}
            try:
                cname = channel_name_cached(channel_id)
                logger.debug(f"Searching channel {cname} ({channel_id}) for: {query}")
                
                async for msg in safe_telegram_generator(
                    User.search_messages, 
                    channel_id, 
                    query=query, 
                    limit=15
                ):
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
                                    'date': msg.date.isoformat() if isinstance(msg.date, datetime) else msg.date,
                                    'is_new': is_new(msg.date) if msg.date else False,
                                    'has_file': False,
                                    'has_post': True,
                                    'quality_options': {},
                                    'thumbnail': None
                                }
            except Exception as e:
                logger.error(f"Telegram search error in {channel_id}: {e}")
            return channel_posts
        
        for channel_id in Config.TEXT_CHANNEL_IDS:
            channel_tasks.append(search_channel(channel_id))
        
        results = await asyncio.gather(*channel_tasks, return_exceptions=True)
        
        posts_found = 0
        for result in results:
            if isinstance(result, dict):
                posts_found += len(result)
                posts_dict.update(result)
        
        logger.debug(f"Found {posts_found} posts in Telegram channels")
    
    # Merge posts and files
    logger.debug("Merging search results...")
    merged = {}
    
    for norm_title, post_data in posts_dict.items():
        merged[norm_title] = post_data
    
    for norm_title, file_data in files_dict.items():
        if norm_title in merged:
            merged[norm_title]['has_file'] = True
            merged[norm_title]['quality_options'] = file_data['quality_options']
            if file_data.get('is_video_file') and file_data.get('thumbnail'):
                merged[norm_title]['thumbnail'] = file_data['thumbnail']
                merged[norm_title]['thumbnail_source'] = file_data.get('thumbnail_source', 'unknown')
        else:
            merged[norm_title] = {
                'title': file_data['title'],
                'content': f"<p>{file_data['title']}</p>",
                'channel': file_data.get('channel_name', 'SK4FiLM Files'),
                'date': file_data['date'],
                'is_new': False,
                'has_file': True,
                'has_post': False,
                'quality_options': file_data['quality_options'],
                'thumbnail': file_data.get('thumbnail') if file_data.get('is_video_file') else None,
                'thumbnail_source': file_data.get('thumbnail_source', 'unknown')
            }
    
    # Sort results
    results_list = list(merged.values())
    results_list.sort(key=lambda x: (
        not x.get('is_new', False),
        not x['has_file'],
        x['date']
    ), reverse=True)
    
    total = len(results_list)
    paginated = results_list[offset:offset + limit]
    
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
            'channels_searched': len(Config.TEXT_CHANNEL_IDS),
            'channels_found': len(set(r.get('channel_id') for r in paginated if r.get('channel_id'))),
            'query': query,
            'cache_hit': False
        }
    }
    
    if cache_manager and hasattr(cache_manager, 'cache_search_results'):
        await cache_manager.cache_search_results(query, page, limit, result_data)
    
    logger.info(f"✅ Search completed: {len(paginated)} results, {len(files_dict)} files found")
    
    return result_data

# ✅ FIXED: BETTER INDEXING FUNCTION THAT ACTUALLY WORKS
async def index_all_files_from_telegram():
    """Index all files from Telegram channel"""
    logger.info("📁 Starting to index ALL files from Telegram channel...")
    
    if not User or not user_session_ready:
        logger.error("❌ Cannot index - User session not ready")
        return
    
    try:
        total_indexed = 0
        total_skipped = 0
        
        batch_size = 50
        offset_id = 0
        
        while True:
            try:
                messages = []
                async for msg in User.get_chat_history(
                    Config.FILE_CHANNEL_ID,
                    limit=batch_size,
                    offset_id=offset_id
                ):
                    messages.append(msg)
                
                if not messages:
                    logger.info("✅ No more messages to index")
                    break
                
                logger.info(f"📥 Processing batch of {len(messages)} messages...")
                
                for msg in messages:
                    if msg and (msg.document or msg.video):
                        success = await index_single_file_smart(msg)
                        if success:
                            total_indexed += 1
                        else:
                            total_skipped += 1
                    
                    offset_id = msg.id
                
                logger.info(f"📊 Progress: {total_indexed} indexed, {total_skipped} skipped")
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Error processing batch: {e}")
                break
        
        logger.info(f"✅ Indexing complete: {total_indexed} files indexed, {total_skipped} skipped")
        
    except Exception as e:
        logger.error(f"❌ Indexing error: {e}")

# ✅ FIXED: ADDED DIRECT INDEXING ENDPOINT
@app.route('/api/index/all', methods=['POST'])
async def api_index_all():
    """Manually trigger indexing of all files"""
    logger.info("📥 API: Index all files")
    try:
        data = await request.get_json()
        admin_key = data.get('admin_key') if data else request.headers.get('X-Admin-Key')
        
        if not admin_key or admin_key != os.environ.get('ADMIN_KEY', 'sk4film_admin_123'):
            logger.warning(f"Unauthorized index attempt with key: {admin_key}")
            return jsonify({
                'status': 'error',
                'message': 'Unauthorized'
            }), 401
        
        if not user_session_ready:
            logger.warning("User session not ready for indexing")
            return jsonify({
                'status': 'error',
                'message': 'Telegram session not ready'
            }), 400
        
        asyncio.create_task(index_all_files_from_telegram())
        
        logger.info("✅ Indexing started in background")
        return jsonify({
            'status': 'success',
            'message': 'Indexing started in background',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Index all API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# ✅ ADDED: IMMEDIATE INDEXING API
@app.route('/api/index/now', methods=['POST'])
async def api_index_now():
    """Immediate file indexing"""
    logger.info("📥 API: Immediate indexing")
    try:
        data = await request.get_json()
        admin_key = data.get('admin_key') if data else request.headers.get('X-Admin-Key')
        
        if not admin_key or admin_key != os.environ.get('ADMIN_KEY', 'sk4film_admin_123'):
            logger.warning(f"Unauthorized indexing attempt")
            return jsonify({
                'status': 'error',
                'message': 'Unauthorized'
            }), 401
        
        if not user_session_ready:
            return jsonify({
                'status': 'error',
                'message': 'Telegram session not ready'
            }), 400
        
        # Run immediate indexing
        await index_all_files_from_telegram()
        
        # Count files
        total_files = await files_col.count_documents({}) if files_col else 0
        
        return jsonify({
            'status': 'success',
            'message': 'Immediate indexing completed',
            'files_indexed': total_files,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Immediate indexing API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# ✅ FIXED: ADDED FORCE SYNC ENDPOINT
@app.route('/api/sync/force', methods=['POST'])
async def api_sync_force():
    """Force sync of deletions"""
    logger.info("📥 API: Force sync")
    try:
        data = await request.get_json()
        admin_key = data.get('admin_key') if data else request.headers.get('X-Admin-Key')
        
        if not admin_key or admin_key != os.environ.get('ADMIN_KEY', 'sk4film_admin_123'):
            logger.warning(f"Unauthorized sync attempt with key: {admin_key}")
            return jsonify({
                'status': 'error',
                'message': 'Unauthorized'
            }), 401
        
        await channel_sync_manager.manual_sync()
        
        logger.info("✅ Force sync completed")
        return jsonify({
            'status': 'success',
            'message': 'Force sync completed',
            'deleted_count': channel_sync_manager.deleted_count,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Force sync API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# LIVE POSTS WITH CACHE
@async_cache_with_ttl(maxsize=100, ttl=60)
async def get_live_posts_multi_channel(limit_per_channel=10):
    """Cached live posts"""
    logger.debug(f"Getting live posts from {len(Config.TEXT_CHANNEL_IDS)} channels")
    if not User or not user_session_ready:
        logger.warning("User session not ready for live posts")
        return []
    
    all_posts = []
    
    async def fetch_channel_posts(channel_id):
        posts = []
        try:
            cname = channel_name_cached(channel_id)
            logger.debug(f"Fetching posts from {cname}")
            
            async for msg in safe_telegram_generator(
                User.get_chat_history, 
                channel_id, 
                limit=limit_per_channel
            ):
                if msg and msg.text and len(msg.text) > 15:
                    title = extract_title_smart(msg.text)
                    if title:
                        posts.append({
                            'title': title,
                            'normalized_title': normalize_title_cached(title),
                            'content': msg.text,
                            'channel_name': cname,
                            'channel_id': channel_id,
                            'message_id': msg.id,
                            'date': msg.date,
                            'is_new': is_new(msg.date) if msg.date else False
                        })
        except Exception as e:
            logger.error(f"Error getting posts from channel {channel_id}: {e}")
        return posts
    
    tasks = [fetch_channel_posts(channel_id) for channel_id in Config.TEXT_CHANNEL_IDS]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for result in results:
        if isinstance(result, list):
            all_posts.extend(result)
    
    seen_titles = set()
    unique_posts = []
    
    for post in sorted(all_posts, key=lambda x: x.get('date', datetime.min), reverse=True):
        if post['normalized_title'] not in seen_titles:
            seen_titles.add(post['normalized_title'])
            unique_posts.append(post)
    
    logger.debug(f"Returning {len(unique_posts[:20])} unique posts")
    return unique_posts[:20]

# ================================
# ✅ Fetch 30 Real Movies from Telegram - NO FALLBACK
# ================================
@performance_monitor.measure("home_movies_telegram")
async def get_home_movies_telegram(limit=30):
    """Fetch 30 real movies directly from Telegram MAIN_CHANNEL_ID - NO FALLBACK"""
    logger.info(f"🎬 Fetching {limit} real movies from Telegram channel {Config.MAIN_CHANNEL_ID}...")
    try:
        if not User or not user_session_ready:
            logger.warning("❌ User session not ready for Telegram fetch")
            return []
        
        movies = []
        seen_titles = set()
        
        async for msg in safe_telegram_generator(
            User.get_chat_history, 
            Config.MAIN_CHANNEL_ID, 
            limit=limit * 2
        ):
            if msg and msg.text and len(msg.text) > 20:
                title = extract_title_smart(msg.text)
                
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    
                    year_match = re.search(r'\b(19|20)\d{2}\b', title)
                    year = year_match.group() if year_match else ""
                    
                    clean_title = re.sub(r'\s+\(\d{4}\)$', '', title)
                    clean_title = re.sub(r'\s+\d{4}$', '', clean_title)
                    
                    poster_url = Config.get_poster(title, year)
                    
                    if poster_fetcher:
                        try:
                            poster_data = await poster_fetcher.fetch_poster(title, year)
                            if poster_data and poster_data.get('url'):
                                poster_url = poster_data['url']
                        except:
                            pass
                    
                    movies.append({
                        'title': clean_title,
                        'original_title': title,
                        'year': year,
                        'date': msg.date.isoformat() if isinstance(msg.date, datetime) else msg.date,
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
        
        logger.info(f"✅ Fetched {len(movies)} real movies from Telegram")
        return movies[:limit]
        
    except Exception as e:
        logger.error(f"❌ Telegram movies fetch error: {e}")
        return []

async def get_single_post_api(channel_id, message_id):
    """Get single movie/post details"""
    logger.info(f"Getting single post: {channel_id}/{message_id}")
    try:
        if User and user_session_ready:
            msg = await safe_telegram_operation(
                User.get_messages,
                channel_id, 
                message_id
            )
            
            if msg and msg.text:
                title = extract_title_smart(msg.text)
                if not title:
                    title = msg.text.split('\n')[0][:60] if msg.text else "Movie Post"
                
                normalized_title = normalize_title(title)
                quality_options = {}
                has_file = False
                thumbnail_url = None
                thumbnail_source = None
                
                logger.debug(f"Searching for files with title: {normalized_title}")
                
                if files_col is not None:
                    cursor = files_col.find({'normalized_title': normalized_title})
                    async for doc in cursor:
                        quality = doc.get('quality', '480p')
                        if quality not in quality_options:
                            file_name = doc.get('file_name', '').lower()
                            file_is_video = is_video_file(file_name)
                            
                            if file_is_video and not thumbnail_url:
                                thumbnail_url = doc.get('thumbnail')
                                thumbnail_source = doc.get('thumbnail_source', 'unknown')
                                
                                if not thumbnail_url and user_session_ready:
                                    thumbnail_url = await get_telegram_video_thumbnail(User, doc['channel_id'], doc['message_id'])
                                    if thumbnail_url:
                                        thumbnail_source = 'video_direct'
                            
                            quality_options[quality] = {
                                'file_id': f"{doc.get('channel_id', Config.FILE_CHANNEL_ID)}_{doc.get('message_id')}_{quality}",
                                'file_size': doc.get('file_size', 0),
                                'file_name': doc.get('file_name', 'video.mp4'),
                                'is_video': file_is_video,
                                'channel_id': doc.get('channel_id'),
                                'message_id': doc.get('message_id')
                            }
                            has_file = True
                
                year_match = re.search(r'\b(19|20)\d{2}\b', title)
                year = year_match.group() if year_match else ""
                
                poster_url = Config.get_poster(title, year)
                poster_source = 'custom'
                poster_rating = '0.0'
                
                if poster_fetcher:
                    try:
                        poster_data = await poster_fetcher.fetch_poster(title, year)
                        if poster_data:
                            poster_url = poster_data.get('url', poster_url)
                            poster_source = poster_data.get('source', poster_source)
                            poster_rating = poster_data.get('rating', poster_rating)
                    except:
                        pass
                
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
                    'thumbnail': thumbnail_url,
                    'thumbnail_source': thumbnail_source,
                    'poster_url': poster_url,
                    'poster_source': poster_source,
                    'poster_rating': poster_rating
                }
                
                logger.info(f"✅ Post data retrieved for {title}")
                return post_data
        
        logger.warning(f"Post not found: {channel_id}/{message_id}")
        return None
        
    except Exception as e:
        logger.error(f"Single post API error: {e}")
        return None

# OPTIMIZED API FUNCTIONS
@performance_monitor.measure("search_api")
async def search_movies_api(query, limit=12, page=1):
    """Optimized search API with timeout protection"""
    logger.info(f"Search API called: '{query}' (page {page})")
    try:
        search_task = asyncio.create_task(search_movies_multi_channel(query, limit, page))
        
        try:
            result_data = await asyncio.wait_for(search_task, timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning(f"⏰ Search timeout for query: {query}")
            
            if cache_manager and hasattr(cache_manager, 'get_search_results'):
                cached = await cache_manager.get_search_results(query, page, limit)
                if cached:
                    logger.info(f"✅ Using cached results for {query}")
                    return cached
            
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
                    'timeout': True,
                    'query': query
                }
            }
        
        if result_data.get('results'):
            for result in result_data['results']:
                title = result.get('title', '')
                year_match = re.search(r'\b(19|20)\d{2}\b', title)
                year = year_match.group() if year_match else ""
                
                result['poster_url'] = Config.get_poster(title, year)
                result['poster_source'] = 'custom'
                result['poster_rating'] = '0.0'
                result['has_poster'] = True
        
        logger.info(f"✅ Search API completed for '{query}'")
        return result_data
        
    except Exception as e:
        logger.error(f"Search API error: {e}")
        raise

@performance_monitor.measure("home_movies")
async def get_home_movies_live():
    """Optimized home movies with timeout - Now uses Telegram, NO FALLBACK"""
    logger.info("Getting home movies...")
    try:
        posts_task = asyncio.create_task(get_home_movies_telegram(limit=30))
        posts = await asyncio.wait_for(posts_task, timeout=5.0)
        
        logger.info(f"✅ Home movies retrieved: {len(posts)} movies")
        return posts
        
    except asyncio.TimeoutError:
        logger.warning("⏰ Home movies timeout")
        return []
    
    except Exception as e:
        logger.error(f"Home movies error: {e}")
        return []

# ============================================================================
# ✅ SYNC MANAGEMENT API ENDPOINTS
# ============================================================================

@app.route('/api/sync/status', methods=['GET'])
async def api_sync_status():
    """Get sync management status"""
    logger.info("📥 API: Sync status")
    try:
        status = {
            'file_channel_id': Config.FILE_CHANNEL_ID,
            'sync_monitoring_active': channel_sync_manager.is_monitoring,
            'monitoring_interval': Config.MONITOR_INTERVAL,
            'deleted_count': channel_sync_manager.deleted_count,
            'last_sync': channel_sync_manager.last_sync,
            'user_session_ready': user_session_ready,
            'timestamp': datetime.now().isoformat(),
            'note': 'NO TTL - Only sync deletions from Telegram'
        }
        
        if files_col:
            try:
                total_files = await files_col.count_documents({"channel_id": Config.FILE_CHANNEL_ID})
                
                status['mongo_stats'] = {
                    'total_files_in_channel': total_files,
                    'no_expiry': True,
                    'sync_only': True
                }
            except:
                pass
        
        logger.info(f"✅ Sync status retrieved")
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

@app.route('/api/sync/start', methods=['POST'])
async def api_sync_start():
    """Start sync management"""
    logger.info("📥 API: Start sync")
    try:
        if not user_session_ready:
            logger.warning("User session not ready for sync start")
            return jsonify({
                'status': 'error',
                'message': 'Telegram session not ready'
            }), 400
        
        await channel_sync_manager.start_sync_monitoring()
        
        logger.info("✅ Sync management started")
        return jsonify({
            'status': 'success',
            'message': 'Sync management started',
            'sync_monitoring': True,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Start sync management API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/sync/stop', methods=['POST'])
async def api_sync_stop():
    """Stop sync management"""
    logger.info("📥 API: Stop sync")
    try:
        await channel_sync_manager.stop_sync_monitoring()
        
        logger.info("✅ Sync management stopped")
        return jsonify({
            'status': 'success',
            'message': 'Sync management stopped',
            'sync_monitoring': False,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Stop sync management API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/sync/manual', methods=['POST'])
async def api_sync_manual():
    """Manual sync trigger"""
    logger.info("📥 API: Manual sync")
    try:
        if not user_session_ready:
            logger.warning("User session not ready for manual sync")
            return jsonify({
                'status': 'error',
                'message': 'Telegram session not ready'
            }), 400
        
        await channel_sync_manager.manual_sync()
        
        logger.info("✅ Manual sync completed")
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

@app.route('/api/sync/stats', methods=['GET'])
async def api_sync_stats():
    """Get detailed sync management stats"""
    logger.info("📥 API: Sync stats")
    try:
        if not files_col:
            logger.warning("Files collection not available for sync stats")
            return jsonify({
                'status': 'error',
                'message': 'MongoDB not available'
            }), 500
        
        total_files = await files_col.count_documents({})
        file_channel_files = await files_col.count_documents({"channel_id": Config.FILE_CHANNEL_ID})
        video_files = await files_col.count_documents({"is_video_file": True})
        
        size_stats = await files_col.aggregate([
            {"$match": {"channel_id": Config.FILE_CHANNEL_ID}},
            {"$group": {
                "_id": None,
                "total_size": {"$sum": "$file_size"},
                "avg_size": {"$avg": "$file_size"},
                "max_size": {"$max": "$file_size"},
                "min_size": {"$min": "$file_size"},
                "count": {"$sum": 1}
            }}
        ]).to_list(length=1)
        
        stats = {
            'total_files': total_files,
            'file_channel_files': file_channel_files,
            'video_files': video_files,
            'sync_enabled': True,
            'ttl_expiry': False,
            'monitoring_active': channel_sync_manager.is_monitoring,
            'monitoring_interval': Config.MONITOR_INTERVAL,
            'total_synced_deletions': channel_sync_manager.deleted_count,
            'file_size_stats': size_stats[0] if size_stats else {},
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Sync stats retrieved: {file_channel_files} files in channel")
        return jsonify({
            'status': 'success',
            'stats': stats,
            'note': 'No automatic expiry - Only sync deletions from Telegram'
        })
        
    except Exception as e:
        logger.error(f"Sync stats API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# ============================================================================
# ✅ DEBUG ENDPOINTS FOR TROUBLESHOOTING
# ============================================================================

@app.route('/api/debug/telegram', methods=['GET'])
async def api_debug_telegram():
    """Debug Telegram connection"""
    logger.info("📥 API: Debug Telegram")
    try:
        result = await check_telegram_session_details()
        
        return jsonify({
            'status': 'success' if result else 'error',
            'telegram_session_ready': user_session_ready,
            'bot_started': bot_started,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Debug Telegram API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/debug/database', methods=['GET'])
async def api_debug_database():
    """Debug database status"""
    logger.info("📥 API: Debug database")
    try:
        stats = {
            'files_collection_exists': files_col is not None,
            'total_files': 0,
            'file_channel_files': 0,
            'sample_files': []
        }
        
        if files_col:
            stats['total_files'] = await files_col.count_documents({})
            stats['file_channel_files'] = await files_col.count_documents({"channel_id": Config.FILE_CHANNEL_ID})
            
            # Get sample files
            cursor = files_col.find().limit(5)
            async for doc in cursor:
                stats['sample_files'].append({
                    'title': doc.get('title', 'Unknown'),
                    'message_id': doc.get('message_id'),
                    'channel_id': doc.get('channel_id'),
                    'is_video_file': doc.get('is_video_file', False)
                })
        
        return jsonify({
            'status': 'success',
            'database': stats,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Debug database API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# ============================================================================
# TELEGRAM STATUS API ENDPOINTS
# ============================================================================

@app.route('/api/telegram/status', methods=['GET'])
async def api_telegram_status():
    """Get detailed Telegram connection status"""
    logger.info("📥 API: Telegram status")
    try:
        status = {
            'environment': {
                'pyrogram_available': PYROGRAM_AVAILABLE,
                'api_id_configured': Config.API_ID > 0,
                'api_hash_configured': bool(Config.API_HASH),
                'session_string_configured': bool(Config.USER_SESSION_STRING),
                'bot_token_configured': bool(Config.BOT_TOKEN)
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
                'movies_source': 'Telegram Channel',
                'total_channels': len(Config.TEXT_CHANNEL_IDS)
            },
            'movies': {
                'fetch_enabled': user_session_ready,
                'source': 'telegram',
                'limit': 30,
                'no_fallback': True,
                'current_status': 'active' if user_session_ready else 'inactive'
            },
            'sync_management': {
                'enabled': channel_sync_manager.is_monitoring,
                'file_channel_id': Config.FILE_CHANNEL_ID,
                'ttl_expiry': False,
                'sync_only': True,
                'monitoring_interval': Config.MONITOR_INTERVAL
            },
            'timestamp': datetime.now().isoformat(),
            'server_time': time.time()
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
        
        if bot and bot_started:
            try:
                bot_info = await bot.get_me()
                status['connections']['bot_session']['bot_info'] = {
                    'username': bot_info.username,
                    'id': bot_info.id
                }
            except:
                pass
        
        logger.info(f"✅ Telegram status retrieved: User={'READY' if user_session_ready else 'NOT READY'}")
        return jsonify({
            'status': 'success',
            'telegram': status,
            'message': 'Telegram session ready for movies' if user_session_ready else 'Telegram session not ready'
        })
        
    except Exception as e:
        logger.error(f"Telegram status API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/telegram/test', methods=['GET'])
async def api_telegram_test():
    """Test Telegram connection and channel access"""
    logger.info("📥 API: Telegram test")
    try:
        test_results = {
            'pyrogram_installed': PYROGRAM_AVAILABLE,
            'environment_check': {
                'api_id': Config.API_ID > 0,
                'api_hash': bool(Config.API_HASH),
                'session_string': bool(Config.USER_SESSION_STRING)
            },
            'connection_test': {
                'user_client_initialized': User is not None,
                'user_session_ready': user_session_ready,
                'bot_initialized': bot is not None
            },
            'channel_tests': [],
            'messages_test': None,
            'timestamp': datetime.now().isoformat()
        }
        
        if User and user_session_ready:
            try:
                chat = await User.get_chat(Config.MAIN_CHANNEL_ID)
                test_results['channel_tests'].append({
                    'channel_id': Config.MAIN_CHANNEL_ID,
                    'name': chat.title,
                    'accessible': True,
                    'type': 'main'
                })
                
                try:
                    async for msg in User.get_chat_history(Config.MAIN_CHANNEL_ID, limit=1):
                        test_results['messages_test'] = {
                            'can_fetch': True,
                            'sample_text': msg.text[:50] + '...' if msg.text else 'No text',
                            'channel': 'main'
                        }
                        break
                except Exception as e:
                    test_results['messages_test'] = {
                        'can_fetch': False,
                        'error': str(e),
                        'channel': 'main'
                    }
                    
            except Exception as e:
                test_results['channel_tests'].append({
                    'channel_id': Config.MAIN_CHANNEL_ID,
                    'accessible': False,
                    'error': str(e),
                    'type': 'main'
                })
            
            try:
                file_chat = await User.get_chat(Config.FILE_CHANNEL_ID)
                test_results['channel_tests'].append({
                    'channel_id': Config.FILE_CHANNEL_ID,
                    'name': file_chat.title,
                    'accessible': True,
                    'type': 'file',
                    'sync_management': True
                })
                
                try:
                    async for msg in User.get_chat_history(Config.FILE_CHANNEL_ID, limit=1):
                        has_file = bool(msg.document or msg.video)
                        test_results['file_channel_test'] = {
                            'can_fetch': True,
                            'has_file': has_file,
                            'channel': 'file'
                        }
                        break
                except Exception as e:
                    test_results['file_channel_test'] = {
                        'can_fetch': False,
                        'error': str(e),
                        'channel': 'file'
                    }
                    
            except Exception as e:
                test_results['channel_tests'].append({
                    'channel_id': Config.FILE_CHANNEL_ID,
                    'accessible': False,
                    'error': str(e),
                    'type': 'file',
                    'sync_management': False
                })
        
        test_results['overall'] = 'READY' if user_session_ready else 'NOT READY'
        test_results['movies_available'] = user_session_ready
        test_results['sync_management_ready'] = user_session_ready
        
        logger.info(f"✅ Telegram test completed: {test_results['overall']}")
        return jsonify({
            'status': 'success',
            'test_results': test_results,
            'recommendation': 'All good!' if user_session_ready else 'Check Telegram credentials'
        })
        
    except Exception as e:
        logger.error(f"Telegram test API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# ============================================================================
# MAIN INITIALIZATION
# ============================================================================

@performance_monitor.measure("system_init")
async def init_system():
    """Turbo-charged system initialization with sync management"""
    start_time = time.time()
    
    try:
        logger.info("🚀 Starting SK4FiLM v8.0 - SYNC MANAGEMENT EDITION...")
        logger.info("📌 NO TTL - Only sync deletions from Telegram")
        
        # Initialize MongoDB
        logger.info("🔌 Initializing MongoDB...")
        mongo_ok = await init_mongodb()
        if not mongo_ok:
            logger.error("❌ MongoDB initialization failed")
        
        # Initialize modular components
        global cache_manager, verification_system, premium_system, poster_fetcher, sk4film_bot
        
        # Initialize Cache Manager
        try:
            logger.info("💾 Initializing Cache Manager...")
            cache_manager = CacheManager(Config)
            redis_ok = False
            if hasattr(cache_manager, 'init_redis'):
                redis_ok = await cache_manager.init_redis()
            if redis_ok:
                logger.info("✅ Cache Manager initialized")
                if hasattr(cache_manager, 'start_cleanup_task'):
                    await cache_manager.start_cleanup_task()
            else:
                logger.warning("⚠️ Cache Manager - Redis not available")
        except Exception as e:
            logger.warning(f"⚠️ Cache Manager initialization failed: {e}")
            cache_manager = None
        
        # Initialize Verification System
        if mongo_ok:
            try:
                logger.info("🔐 Initializing Verification System...")
                verification_system = VerificationSystem(Config, db)
                if hasattr(verification_system, 'start_cleanup_task'):
                    await verification_system.start_cleanup_task()
                logger.info("✅ Verification System initialized")
            except Exception as e:
                logger.warning(f"⚠️ Verification System - Initialization failed: {e}")
                verification_system = None
        else:
            logger.warning("⚠️ Verification System - MongoDB not available")
        
        # Initialize Premium System
        if mongo_ok:
            try:
                logger.info("💎 Initializing Premium System...")
                premium_system = PremiumSystem(Config, db)
                if hasattr(premium_system, 'start_cleanup_task'):
                    await premium_system.start_cleanup_task()
                logger.info("✅ Premium System initialized")
            except Exception as e:
                logger.warning(f"⚠️ Premium System - Initialization failed: {e}")
                premium_system = None
        else:
            logger.warning("⚠️ Premium System - MongoDB not available")
        
        # ✅ Initialize Poster Fetcher
        try:
            logger.info("🖼️ Initializing Poster Fetcher...")
            poster_fetcher = PosterFetcher(cache_manager, Config)
            logger.info("✅ Poster Fetcher initialized")
        except Exception as e:
            logger.warning(f"⚠️ Poster Fetcher - Initialization failed: {e}")
            poster_fetcher = None
        
        # ✅ WARM UP CACHE FOR INSTANT RESPONSE
        asyncio.create_task(warm_up_cache())
        
        # ✅ Initialize Telegram if available
        if PYROGRAM_AVAILABLE:
            logger.info("📱 Initializing Telegram clients...")
            telegram_ok = await init_telegram_clients()
            if not telegram_ok:
                logger.warning("⚠️ Telegram clients not initialized - Movies will be empty")
            else:
                # Wait a bit for Telegram to stabilize
                await asyncio.sleep(2)
                
                # Check Telegram session details
                await check_telegram_session_details()
        else:
            logger.warning("⚠️ Pyrogram not available - Movies will be empty")
            sk4film_bot = None
        
        # Start background tasks
        asyncio.create_task(cache_cleanup())
        
        # Start SMART indexing in background only if Telegram is ready
        if user_session_ready:
            logger.info("⏳ Waiting 5 seconds before starting indexing...")
            await asyncio.sleep(5)
            
            logger.info("🚀 Starting SMART background indexing...")
            asyncio.create_task(index_files_background_smart())
        else:
            logger.warning("⚠️ Cannot start indexing - User session not ready")
            logger.info("📋 Please check:")
            logger.info("   1. API_ID, API_HASH, USER_SESSION_STRING environment variables")
            logger.info("   2. Telegram session is valid")
            logger.info("   3. You're a member of the required channels")
        
        init_time = time.time() - start_time
        logger.info(f"⚡ SK4FiLM Started in {init_time:.2f}s - SYNC MANAGEMENT READY")
        
        logger.info("🔧 SYNC MANAGEMENT FEATURES:")
        logger.info(f"   • FILE_CHANNEL_ID: {Config.FILE_CHANNEL_ID}")
        logger.info(f"   • Auto expiry: DISABLED (No TTL)")
        logger.info(f"   • Delete sync: Every {Config.MONITOR_INTERVAL//60} minutes")
        logger.info(f"   • Duplicate Prevention: Active")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ System initialization failed: {e}")
        logger.error("Traceback:", exc_info=True)
        return False

# ============================================================================
# API ROUTES WITH OPTIMIZATIONS
# ============================================================================

@app.route('/')
@performance_monitor.measure("root_endpoint")
async def root():
    """Optimized root endpoint"""
    logger.info("📥 Root endpoint accessed")
    tf = await files_col.count_documents({}) if files_col is not None else 0
    video_files = await files_col.count_documents({'is_video_file': True}) if files_col is not None else 0
    posters_cached = await poster_col.count_documents({}) if poster_col is not None else 0
    
    sync_stats = {
        'sync_monitoring': channel_sync_manager.is_monitoring,
        'file_channel_id': Config.FILE_CHANNEL_ID,
        'ttl_expiry': False,
        'deleted_count': channel_sync_manager.deleted_count
    }
    
    logger.info(f"✅ Root endpoint response: {tf} files, {video_files} videos")
    return jsonify({
        'status': 'healthy',
        'service': 'SK4FiLM v8.0 - SYNC MANAGEMENT EDITION',
        'telegram': {
            'user_session_ready': user_session_ready,
            'bot_started': bot_started,
            'movies_fetch': user_session_ready,
            'main_channel': Config.MAIN_CHANNEL_ID,
            'file_channel': Config.FILE_CHANNEL_ID
        },
        'sync_management': sync_stats,
        'performance': {
            'cache_enabled': cache_manager.redis_enabled if cache_manager else False,
            'modules_loaded': all([
                verification_system is not None,
                premium_system is not None,
                poster_fetcher is not None
            ])
        },
        'database': {
            'total_files': tf, 
            'video_files': video_files,
            'posters_cached': posters_cached,
            'no_ttl': True
        },
        'channels': len(Config.TEXT_CHANNEL_IDS),
        'response_time': f"{time.perf_counter():.3f}s"
    })

@app.route('/health')
@performance_monitor.measure("health_endpoint")
async def health():
    """Optimized health endpoint"""
    logger.info("📥 Health endpoint accessed")
    return jsonify({
        'status': 'ok' if bot_started else 'starting',
        'telegram': {
            'user_session': {
                'ready': user_session_ready,
                'initialized': User is not None
            },
            'bot': {
                'started': bot_started,
                'initialized': bot is not None
            }
        },
        'sync_management': {
            'monitoring': channel_sync_manager.is_monitoring,
            'file_channel': Config.FILE_CHANNEL_ID,
            'ttl_expiry': False
        },
        'cache': cache_manager.redis_enabled if cache_manager else False,
        'timestamp': datetime.now().isoformat(),
        'performance': {
            'avg_response_time': performance_monitor.get_stats()
        }
    })

@app.route('/api/movies', methods=['GET'])
@performance_monitor.measure("movies_endpoint")
async def api_movies():
    """Optimized movies endpoint - ONLY Telegram, NO FALLBACK"""
    logger.info("📥 API: Movies endpoint")
    try:
        movies = await get_home_movies_live()
        
        logger.info(f"✅ Movies endpoint: {len(movies)} movies returned")
        return jsonify({
            'status': 'success' if movies else 'empty',
            'movies': movies,
            'total': len(movies),
            'source': 'telegram',
            'telegram_ready': user_session_ready,
            'channel_id': Config.MAIN_CHANNEL_ID,
            'timestamp': datetime.now().isoformat(),
            'cache_hit': False,
            'message': 'No movies found' if not movies else None
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
    """Optimized search endpoint"""
    logger.info("📥 API: Search endpoint")
    try:
        query = request.args.get('query', '').strip()
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 12))
        
        if len(query) < 2:
            logger.warning(f"Search query too short: '{query}'")
            return jsonify({
                'status': 'error',
                'message': 'Query must be at least 2 characters'
            }), 400
        
        logger.info(f"Search query: '{query}' (page {page})")
        result_data = await search_movies_api(query, limit, page)
        
        logger.info(f"✅ Search completed: {len(result_data['results'])} results")
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
    """Get single post details"""
    logger.info("📥 API: Post endpoint")
    try:
        channel_id = int(request.args.get('channel', Config.MAIN_CHANNEL_ID))
        message_id = int(request.args.get('message', 0))
        
        if message_id <= 0:
            logger.warning(f"Invalid message ID: {message_id}")
            return jsonify({
                'status': 'error',
                'message': 'Invalid message ID'
            }), 400
        
        logger.info(f"Getting post: {channel_id}/{message_id}")
        post_data = await get_single_post_api(channel_id, message_id)
        
        if post_data:
            logger.info(f"✅ Post found: {post_data.get('title', 'Unknown')}")
            return jsonify({
                'status': 'success',
                'post': post_data,
                'timestamp': datetime.now().isoformat()
            })
        else:
            logger.warning(f"Post not found: {channel_id}/{message_id}")
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
    """Get movie poster"""
    logger.info("📥 API: Poster endpoint")
    try:
        title = request.args.get('title', '').strip()
        year = request.args.get('year', '')
        
        if not title:
            logger.warning("Poster request without title")
            return jsonify({
                'status': 'error',
                'message': 'Title is required'
            }), 400
        
        logger.info(f"Getting poster for: {title} ({year})")
        if poster_fetcher:
            poster_data = await poster_fetcher.fetch_poster(title, year)
            
            if poster_data:
                logger.info(f"✅ Poster found via fetcher: {title}")
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
        
        logger.info(f"Using fallback poster for: {title}")
        return jsonify({
            'status': 'success',
            'poster': {
                'poster_url': Config.get_poster(title, year),
                'source': 'custom',
                'rating': '0.0',
                'year': year,
                'title': title
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

@app.route('/api/verify_user', methods=['POST'])
async def api_verify_user():
    """Verify user with URL shortener"""
    logger.info("📥 API: Verify user")
    try:
        data = await request.get_json()
        if not data:
            logger.warning("No JSON data in verify user request")
            return jsonify({
                'status': 'error',
                'message': 'No JSON data provided'
            }), 400
        
        user_id = data.get('user_id')
        verification_url = data.get('verification_url')
        
        if not user_id:
            logger.warning("No user_id in verify user request")
            return jsonify({
                'status': 'error',
                'message': 'user_id is required'
            }), 400
        
        logger.info(f"Verifying user: {user_id}")
        result = await verify_user_api(user_id, verification_url)
        
        logger.info(f"✅ User verification result: {result.get('verified', False)}")
        return jsonify({
            'status': 'success',
            'verification': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Verify user API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/premium/plans', methods=['GET'])
async def api_premium_plans():
    """Get premium plans"""
    logger.info("📥 API: Premium plans")
    try:
        if premium_system:
            if hasattr(premium_system, 'get_all_plans'):
                plans = await premium_system.get_all_plans()
            elif hasattr(premium_system, 'plans'):
                plans = []
                for tier_enum, plan in premium_system.plans.items():
                    plans.append({
                        'tier': tier_enum.value,
                        'name': plan.name,
                        'price': plan.price,
                        'duration_days': plan.duration_days,
                        'features': plan.features
                    })
            else:
                plans = []
            
            logger.info(f"✅ Premium plans retrieved: {len(plans)} plans")
            return jsonify({
                'status': 'success',
                'plans': plans,
                'timestamp': datetime.now().isoformat()
            })
        else:
            logger.warning("Premium system not available")
            return jsonify({
                'status': 'error',
                'message': 'Premium system not available'
            }), 503
    except Exception as e:
        logger.error(f"Premium plans API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/user/status', methods=['GET'])
async def api_user_status():
    """Get user status (premium/verification)"""
    logger.info("📥 API: User status")
    try:
        user_id = int(request.args.get('user_id', 0))
        
        if user_id <= 0:
            logger.warning(f"Invalid user_id: {user_id}")
            return jsonify({
                'status': 'error',
                'message': 'Valid user_id is required'
            }), 400
        
        logger.info(f"Getting status for user: {user_id}")
        result = {}
        
        if premium_system:
            if hasattr(premium_system, 'get_subscription_details'):
                premium_status = await premium_system.get_subscription_details(user_id)
                result['premium'] = premium_status
        
        if verification_system:
            if hasattr(verification_system, 'check_user_verified'):
                is_verified, message = await verification_system.check_user_verified(user_id, premium_system)
                result['verification'] = {
                    'is_verified': is_verified,
                    'message': message,
                    'needs_verification': not is_verified and Config.VERIFICATION_REQUIRED
                }
        
        if premium_system and hasattr(premium_system, 'can_user_download'):
            can_download, download_message, download_details = await premium_system.can_user_download(user_id)
            result['download'] = {
                'can_download': can_download,
                'message': download_message,
                'details': download_details
            }
        
        logger.info(f"✅ User status retrieved for {user_id}")
        return jsonify({
            'status': 'success',
            'user_id': user_id,
            'user_status': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"User status API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/clear_cache', methods=['POST'])
async def api_clear_cache():
    """Clear all caches (admin only)"""
    logger.info("📥 API: Clear cache")
    try:
        data = await request.get_json()
        admin_key = data.get('admin_key') if data else request.headers.get('X-Admin-Key')
        
        if not admin_key or admin_key != os.environ.get('ADMIN_KEY', 'sk4film_admin_123'):
            logger.warning(f"Unauthorized cache clear attempt with key: {admin_key}")
            return jsonify({
                'status': 'error',
                'message': 'Unauthorized'
            }), 401
        
        cleared = {
            'cache_manager': False,
            'poster_fetcher': False,
            'search_cache': 0,
            'poster_collection': False
        }
        
        if cache_manager:
            if hasattr(cache_manager, 'clear_all'):
                await cache_manager.clear_all()
                cleared['cache_manager'] = True
        
        if poster_fetcher and hasattr(poster_fetcher, 'clear_cache'):
            poster_fetcher.clear_cache()
            cleared['poster_fetcher'] = True
        
        if cache_manager and hasattr(cache_manager, 'clear_search_cache'):
            cleared['search_cache'] = await cache_manager.clear_search_cache()
        
        if poster_col:
            await poster_col.delete_many({})
            cleared['poster_collection'] = True
        
        logger.info("✅ Cache cleared successfully")
        return jsonify({
            'status': 'success',
            'message': 'Cache cleared successfully',
            'cleared': cleared,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Clear cache API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/index_status', methods=['GET'])
async def api_index_status():
    """Get indexing status"""
    logger.info("📥 API: Index status")
    try:
        status_data = await get_index_status_api()
        
        logger.info("✅ Index status retrieved")
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
    """Optimized stats endpoint"""
    logger.info("📥 API: Stats endpoint")
    try:
        stats = {}
        
        if files_col:
            stats['database'] = {
                'total_files': await files_col.count_documents({}),
                'video_files': await files_col.count_documents({'is_video_file': True}),
                'unique_titles': len(await files_col.distinct('normalized_title')),
                'file_channel_files': await files_col.count_documents({"channel_id": Config.FILE_CHANNEL_ID}),
                'ttl_expiry': False,
                'sync_only': True
            }
        
        stats['sync_management'] = {
            'sync_monitoring': channel_sync_manager.is_monitoring,
            'deleted_count': channel_sync_manager.deleted_count,
            'monitoring_interval': Config.MONITOR_INTERVAL,
            'file_channel_id': Config.FILE_CHANNEL_ID,
            'last_sync': channel_sync_manager.last_sync
        }
        
        if poster_col:
            stats['posters'] = {
                'total_posters': await poster_col.count_documents({}),
                'expired_posters': await poster_col.count_documents({"expires_at": {"$lt": datetime.utcnow()}})
            }
        
        if cache_manager and hasattr(cache_manager, 'get_stats_summary'):
            cache_stats = await cache_manager.get_stats_summary()
            stats['cache'] = cache_stats
        
        stats['performance'] = performance_monitor.get_stats()
        
        stats['system'] = {
            'telegram': {
                'user_session_ready': user_session_ready,
                'bot_started': bot_started,
                'main_channel': Config.MAIN_CHANNEL_ID,
                'file_channel': Config.FILE_CHANNEL_ID,
                'movies_fetch': user_session_ready
            },
            'uptime': time.time() - app_start_time if 'app_start_time' in globals() else 0,
            'timestamp': datetime.now().isoformat(),
            'note': 'NO TTL - Only sync deletions from Telegram'
        }
        
        logger.info(f"✅ Stats retrieved: {stats['database']['total_files'] if 'database' in stats else 0} files")
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
# TELEGRAM SESSION GENERATION ENDPOINT (SAFE MODE)
# ============================================================================

@app.route('/api/telegram/generate_session', methods=['POST'])
async def api_generate_session():
    """Generate Telegram session string (development only)"""
    logger.info("📥 API: Generate session")
    try:
        if os.environ.get('ENVIRONMENT') != 'development':
            logger.warning("Session generation attempted in non-development environment")
            return jsonify({
                'status': 'error',
                'message': 'Session generation only allowed in development'
            }), 403
        
        data = await request.get_json()
        if not data:
            logger.warning("No data in generate session request")
            return jsonify({
                'status': 'error',
                'message': 'No data provided'
            }), 400
        
        api_id = data.get('api_id')
        api_hash = data.get('api_hash')
        
        if not api_id or not api_hash:
            logger.warning("Missing API ID or hash in generate session request")
            return jsonify({
                'status': 'error',
                'message': 'API ID and API Hash required'
            }), 400
        
        logger.info("🔧 Generating Telegram session...")
        
        return jsonify({
            'status': 'info',
            'message': 'Session generation requires interactive input',
            'instructions': [
                '1. Make sure pyrogram is installed: pip install pyrogram',
                '2. Run the following Python code:',
                '   from pyrogram import Client',
                '   async with Client("session", api_id, api_hash) as app:',
                '       session_string = await app.export_session_string()',
                '       print(session_string)',
                '3. Set the session string as USER_SESSION_STRING environment variable'
            ]
        })
            
    except Exception as e:
        logger.error(f"Generate session API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# ============================================================================
# STARTUP AND SHUTDOWN
# ============================================================================

app_start_time = time.time()

@app.before_serving
async def startup():
    """Startup initialization"""
    logger.info("🚀 Starting up SK4FiLM...")
    await init_system()

@app.after_serving
async def shutdown():
    """Optimized shutdown"""
    logger.info("🛑 Shutting down SK4FiLM...")
    
    shutdown_tasks = []
    
    await channel_sync_manager.stop_sync_monitoring()
    
    if User and user_session_ready:
        shutdown_tasks.append(User.stop())
    
    if bot and bot_started:
        shutdown_tasks.append(bot.stop())
    
    if sk4film_bot and hasattr(sk4film_bot, 'shutdown'):
        shutdown_tasks.append(sk4film_bot.shutdown())
    
    if cache_manager and hasattr(cache_manager, 'stop'):
        shutdown_tasks.append(cache_manager.stop())
    
    if verification_system and hasattr(verification_system, 'stop'):
        shutdown_tasks.append(verification_system.stop())
    
    if premium_system and hasattr(premium_system, 'stop_cleanup_task'):
        shutdown_tasks.append(premium_system.stop_cleanup_task())
    
    if shutdown_tasks:
        await asyncio.gather(*shutdown_tasks, return_exceptions=True)
    
    if mongo_client:
        mongo_client.close()
    
    logger.info(f"👋 Shutdown complete. Uptime: {time.time() - app_start_time:.1f}s")

# ============================================================================
# MAIN ENTRY POINT
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
    
    try:
        asyncio.run(serve(app, config))
    except KeyboardInterrupt:
        logger.info("👋 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        exit(1)
