"""
app.py - Complete SK4FiLM Web API System with Multi-Channel Search
OPTIMIZED: Supersonic speed with caching, async optimization, and performance monitoring
ENHANCED: Auto delete from MongoDB when deleted from Telegram, duplicate prevention
NO TTL: No automatic time-based deletion
"""
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

import aiohttp
import urllib.parse
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

# Import modular components
from cache import CacheManager
from verification import VerificationSystem
from premium import PremiumSystem, PremiumTier

# Import shared utilities
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

# Import bot_handlers AFTER all other imports
try:
    from bot_handlers import SK4FiLMBot, setup_bot_handlers
    BOT_HANDLERS_AVAILABLE = True
except ImportError:
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
    
    # Database Configuration
    MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
    REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
    REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "")
    
    # Channel Configuration - OPTIMIZED
    MAIN_CHANNEL_ID = -1001891090100
    TEXT_CHANNEL_IDS = [-1001891090100, -1002024811395]  # Only active channels
    FILE_CHANNEL_ID = -1001768249569  # ✅ FILE CHANNEL FOR SYNC MANAGEMENT
    
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
        
        encoded_title = urllib.parse.quote(title)
        return f"{Config.BACKEND_URL}/api/poster?title={encoded_title}&year={year}"

# FAST INITIALIZATION
app = Quart(__name__)
app.config['JSON_SORT_KEYS'] = False  # Faster JSON serialization
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False  # Disable pretty printing for speed

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
    
    async def start_sync_monitoring(self):
        """Start monitoring Telegram channel for deletions sync"""
        if not User or not user_session_ready:
            logger.warning("⚠️ Cannot start sync monitoring - User session not ready")
            return
        
        if self.is_monitoring:
            logger.info("✅ Channel sync monitoring already running")
            return
        
        logger.info("👁️ Starting Telegram channel sync monitoring (Delete Sync Only)...")
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
                await asyncio.sleep(Config.MONITOR_INTERVAL)  # Check every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Channel sync monitoring error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def sync_deletions_from_telegram(self):
        """Sync deletions from Telegram to MongoDB"""
        try:
            if not files_col or not User:
                return
            
            current_time = time.time()
            if current_time - self.last_sync < 300:  # Run every 5 minutes
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
            batch_size = 50  # Telegram allows up to 200 messages per request
            
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
                    
                    # Delete from MongoDB (SYNC DELETION)
                    if deleted_ids:
                        result = await files_col.delete_many({
                            "channel_id": Config.FILE_CHANNEL_ID,
                            "message_id": {"$in": deleted_ids}
                        })
                        
                        if result.deleted_count > 0:
                            deleted_count += result.deleted_count
                            self.deleted_count += result.deleted_count
                            
                            # Log deleted files
                            for msg_id in deleted_ids[:5]:  # Log first 5
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
        
        # Add file-specific information
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
        
        # Add caption hash if available
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
    Smart file indexing with:
    1. Duplicate prevention (multiple checks)
    2. Delete sync tracking
    3. Hash-based duplicate detection
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
        # ✅ STEP 2: CHECK IF ALREADY EXISTS (MULTI-LAYER DUPLICATE DETECTION)
        # ============================================================================
        
        # Layer 1: Check by channel_id + message_id (exact match)
        existing_by_id = await files_col.find_one({
            'channel_id': Config.FILE_CHANNEL_ID,
            'message_id': message.id
        }, {'_id': 1, 'title': 1})
        
        if existing_by_id:
            logger.debug(f"📝 Skipping - Already indexed: {message.id}")
            return True  # Already exists
        
        # Extract title
        title = await extract_title_from_telegram_msg_cached(message)
        if not title:
            logger.debug(f"📝 Skipping - No title: {message.id}")
            return False
        
        normalized_title = normalize_title_cached(title)
        
        # Layer 2: Generate file hash for duplicate detection
        file_hash = await generate_file_hash(message)
        
        if file_hash:
            # Check if same file hash exists (same file reposted)
            existing_by_hash = await files_col.find_one({
                'channel_id': Config.FILE_CHANNEL_ID,
                'file_hash': file_hash
            }, {'_id': 1, 'title': 1})
            
            if existing_by_hash:
                logger.info(f"🔁 Skipping duplicate (same file hash): {title}")
                return False
        
        # Layer 3: Check by file_id (for Telegram file duplicates)
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
        
        # Layer 4: Check by title + similar size (within 20%)
        if file_size > 0:
            # Look for files with same normalized title
            similar_title_files = await files_col.find({
                'channel_id': Config.FILE_CHANNEL_ID,
                'normalized_title': normalized_title
            }).limit(3).to_list(length=3)
            
            if similar_title_files:
                # Check if any have similar file size
                for existing_file in similar_title_files:
                    existing_size = existing_file.get('file_size', 0)
                    if existing_size > 0:
                        # Check if size is within 20%
                        size_ratio = file_size / existing_size if existing_size > 0 else 0
                        if 0.8 <= size_ratio <= 1.2:  # Within 20%
                            logger.info(f"🔁 Skipping duplicate (similar title & size): {title}")
                            return False
        
        # ============================================================================
        # ✅ STEP 3: CREATE DOCUMENT WITHOUT EXPIRY (NO TTL)
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
            'file_id': file_id,
            'file_size': file_size,
            'file_hash': file_hash,
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
                thumbnail_url = await extract_video_thumbnail(User, message)
                if thumbnail_url:
                    doc['thumbnail'] = thumbnail_url
                    doc['thumbnail_source'] = 'video_direct'
            except:
                pass
        
        # ============================================================================
        # ✅ STEP 4: INSERT WITH ERROR HANDLING
        # ============================================================================
        try:
            await files_col.insert_one(doc)
            
            # Log successful indexing
            file_type = "📹 Video" if doc['is_video_file'] else "📄 File"
            size_str = format_size(file_size) if file_size > 0 else "Unknown size"
            
            logger.info(f"✅ {file_type} indexed: {title}")
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
# ✅ BACKGROUND INDEXING WITH SYNC MANAGEMENT
# ============================================================================

async def index_files_background_smart():
    """Smart background indexing with sync management features"""
    if not User or files_col is None or not user_session_ready:
        logger.warning("⚠️ Cannot start smart indexing - User session not ready")
        return
    
    logger.info("📁 Starting SMART background indexing with sync management...")
    
    try:
        # ============================================================================
        # ✅ STEP 1: SETUP MONGODB INDEXES FOR SYNC MANAGEMENT
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
        batch_size = 20
        
        # Fetch messages newer than last_indexed
        messages = []
        async for msg in safe_telegram_generator(
            User.get_chat_history, 
            Config.FILE_CHANNEL_ID,
            limit=100  # Start with 100 messages
        ):
            if msg.id <= last_message_id:
                break
            
            if msg and (msg.document or msg.video):
                messages.append(msg)
        
        # Process in reverse order (oldest first for consistency)
        messages.reverse()
        
        logger.info(f"📥 Found {len(messages)} potential new files to index")
        
        for msg in messages:
            try:
                success = await index_single_file_smart(msg)
                if success:
                    total_indexed += 1
                else:
                    total_skipped += 1
                
                # Small delay to avoid rate limiting
                await asyncio.sleep(0.3)
                
                # Log progress every 5 files
                if (total_indexed + total_skipped) % 5 == 0:
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
            logger.info(f"📝 Skipped {total_skipped} files (duplicates or errors)")
        
        # Start sync monitoring for deletions
        await channel_sync_manager.start_sync_monitoring()
        
        # Initial sync check
        await channel_sync_manager.sync_deletions_from_telegram()
        
        logger.info("🚀 Sync management system activated!")
        logger.info(f"   • Delete sync monitoring: Every {Config.MONITOR_INTERVAL//60} minutes")
        logger.info(f"   • Auto expiry: DISABLED (No TTL)")
        logger.info(f"   • Duplicate prevention: Active")
        
    except Exception as e:
        logger.error(f"❌ Smart background indexing error: {e}")

async def setup_mongodb_sync_indexes():
    """Setup MongoDB indexes for sync management (NO TTL)"""
    try:
        if not files_col:
            return
        
        logger.info("🗂️ Setting up MongoDB indexes for sync management...")
        
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
# TELEGRAM SESSION GENERATOR FUNCTION (BUILT-IN)
# ============================================================================

async def generate_telegram_session():
    """Generate Telegram session string if not available"""
    if not PYROGRAM_AVAILABLE:
        logger.error("❌ Pyrogram not installed. Run: pip install pyrogram")
        return None
    
    logger.info("🎯 Generating new Telegram session...")
    
    try:
        api_id = input("Enter your API_ID from https://my.telegram.org: ").strip()
        api_hash = input("Enter your API_HASH: ").strip()
        
        if not api_id.isdigit() or not api_hash:
            logger.error("❌ Invalid API credentials")
            return None
        
        # Create temporary client
        temp_client = Client(
            "sk4film_temp_session",
            api_id=int(api_id),
            api_hash=api_hash,
            in_memory=True
        )
        
        await temp_client.start()
        
        # Get user info
        me = await temp_client.get_me()
        logger.info(f"✅ Logged in as: {me.first_name} (@{me.username})")
        
        # Export session string
        session_string = await temp_client.export_session_string()
        
        logger.info("🎉 SESSION GENERATED SUCCESSFULLY!")
        logger.info(f"Session String: {session_string}")
        
        # Test channel access
        logger.info("🔍 Testing channel access...")
        try:
            chat = await temp_client.get_chat(Config.MAIN_CHANNEL_ID)
            logger.info(f"✅ Channel accessible: {chat.title}")
        except Exception as e:
            logger.warning(f"⚠️ Cannot access channel: {e}")
            logger.warning("Make sure you're a member of the channel!")
        
        await temp_client.stop()
        
        # Update environment
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
    
    # Check Pyrogram availability
    if not PYROGRAM_AVAILABLE:
        logger.error("❌ CRITICAL: Pyrogram not installed!")
        logger.error("   Run: pip install pyrogram")
        return False
    
    # Environment validation
    logger.info("🔍 Checking environment variables...")
    
    env_status = {
        "API_ID": Config.API_ID > 0,
        "API_HASH": bool(Config.API_HASH and len(Config.API_HASH) > 10),
        "USER_SESSION_STRING": bool(Config.USER_SESSION_STRING and len(Config.USER_SESSION_STRING) > 100),
        "BOT_TOKEN": bool(Config.BOT_TOKEN)
    }
    
    for key, status in env_status.items():
        logger.info(f"   {key}: {'✅' if status else '❌'}")
    
    # If session string is missing or too short
    if not env_status["USER_SESSION_STRING"]:
        logger.warning("⚠️ Session string missing or invalid")
        logger.info("   Would you like to generate a new session? (y/n): ")
        # Note: In production, you'd need to handle this differently
        # For now, we'll just log and continue
    
    # Initialize User Client with retry logic
    if env_status["API_ID"] and env_status["API_HASH"] and env_status["USER_SESSION_STRING"]:
        logger.info("\n👤 Initializing User Client...")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"   Attempt {attempt + 1}/{max_retries}")
                
                # Create user client
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
                
                # Start with timeout
                await asyncio.wait_for(User.start(), timeout=15)
                
                # Verify connection
                me = await User.get_me()
                logger.info(f"✅ User Client Ready: {me.first_name}")
                logger.info(f"   User ID: {me.id}, Username: @{me.username}")
                
                # Test channel access
                try:
                    chat = await User.get_chat(Config.FILE_CHANNEL_ID)  # ✅ Test FILE_CHANNEL_ID
                    logger.info(f"✅ FILE Channel Access: {chat.title} (ID: {Config.FILE_CHANNEL_ID})")
                    
                    # Quick message test
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
                    user_session_ready = False  # Can't fetch movies
                
                break  # Success
                
            except asyncio.TimeoutError:
                logger.error(f"⏰ Timeout on attempt {attempt + 1}")
                if User:
                    await User.stop()
                    User = None
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                    
            except Exception as e:
                error_msg = str(e)
                logger.error(f"❌ Attempt {attempt + 1} failed: {error_msg}")
                
                # Handle specific errors
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
    
    # Initialize Bot Client
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
    
    # Final status
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
    return CHANNEL_CONFIG.get(cid, {}).get('name', f"Channel {cid}")

@lru_cache(maxsize=5000)
def normalize_title_cached(title: str) -> str:
    """Cached title normalization"""
    return normalize_title(title)

# FIXED: Missing channel_name function
@lru_cache(maxsize=1000)
def channel_name(channel_id):
    """Get channel name from channel ID (compatibility function)"""
    return channel_name_cached(channel_id)

# FIXED: Missing get_telegram_video_thumbnail function
async def get_telegram_video_thumbnail(user_client, channel_id, message_id):
    """Get video thumbnail from Telegram"""
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

# FIXED: Missing verify_user_api function
async def verify_user_api(user_id, verification_url=None):
    """Verify user API endpoint implementation"""
    try:
        if verification_system:
            result = await verification_system.verify_user_api(user_id, verification_url)
            return result
        
        # Fallback
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

# FIXED: Missing get_index_status_api function
async def get_index_status_api():
    """Get indexing status API implementation"""
    try:
        total_files = await files_col.count_documents({}) if files_col else 0
        video_files = await files_col.count_documents({'is_video_file': True}) if files_col else 0
        
        # ✅ ADDED: Sync management stats
        if files_col:
            channel_files = await files_col.count_documents({
                "channel_id": Config.FILE_CHANNEL_ID
            })
        else:
            channel_files = 0
        
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
            # Create cache key
            key_parts = [func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
            key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
            
            now = time.time()
            
            # Check cache
            if key in cache:
                value, timestamp = cache[key]
                if now - timestamp < ttl:
                    return value
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Store in cache
            cache[key] = (result, now)
            
            # Clean old entries
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
    
    async def wait_if_needed(self, user_id=None):
        """Optimized flood protection"""
        current_time = time.time()
        
        # Cleanup old buckets every 30 seconds
        if current_time - self.last_cleanup > 30:
            self._cleanup_buckets(current_time)
            self.last_cleanup = current_time
        
        # Global rate limiting
        bucket_key = int(current_time // 10)  # 10-second buckets
        
        if bucket_key not in self.request_buckets:
            self.request_buckets[bucket_key] = 0
        
        # Allow 50 requests per 10-second bucket
        if self.request_buckets[bucket_key] >= 50:
            wait_time = 10 - (current_time % 10)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                # Reset after wait
                self.request_buckets = {}
        
        self.request_buckets[bucket_key] += 1
        self.request_count += 1
    
    def _cleanup_buckets(self, current_time):
        """Cleanup old buckets"""
        current_bucket = int(current_time // 10)
        old_buckets = [k for k in self.request_buckets.keys() if k < current_bucket - 6]  # Keep last minute
        for bucket in old_buckets:
            del self.request_buckets[bucket]

turbo_flood_protection = TurboFloodProtection()

# ASYNC CONNECTION POOL
class ConnectionPool:
    def __init__(self, max_connections=20):
        self.max_connections = max_connections
        self.semaphore = asyncio.Semaphore(max_connections)
        self.active_connections = 0
    
    async def acquire(self):
        await self.semaphore.acquire()
        self.active_connections += 1
    
    def release(self):
        self.active_connections -= 1
        self.semaphore.release()

# Create connection pool for external requests
http_pool = ConnectionPool(max_connections=Config.MAX_CONCURRENT_REQUESTS)

# SAFE TELEGRAM OPERATIONS WITH TURBO PROTECTION
@performance_monitor.measure("telegram_operation")
async def safe_telegram_operation(operation, *args, **kwargs):
    """Turbo-charged Telegram operations"""
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
        
        # Small delay every 10 items
        if count % 10 == 0:
            await asyncio.sleep(0.1)
            await turbo_flood_protection.wait_if_needed()
        
        if limit and count >= limit:
            break

# CACHE WARM-UP
async def warm_up_cache():
    """Warm up all caches for instant response"""
    try:
        logger.info("🔥 Warming up caches...")
        
        if cache_manager and cache_manager.redis_enabled:
            # Warm up Redis with common data
            warm_data = {
                "system:warm": True,
                "startup_time": datetime.now().isoformat(),
                "version": "8.0-SYNC-ONLY"
            }
            
            await cache_manager.batch_set(warm_data, expire_seconds=3600)
        
        logger.info("✅ Cache warm-up complete")
        
    except Exception as e:
        logger.error(f"Cache warm-up error: {e}")

# CACHE CLEANUP TASK
async def cache_cleanup():
    """Background cache cleanup"""
    while True:
        await asyncio.sleep(1800)  # Run every 30 minutes
        try:
            if cache_manager:
                await cache_manager.clear_pattern("temp:")
            
            logger.debug("🧹 Cache cleanup completed")
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")

# OPTIMIZED TITLE EXTRACTION
@async_cache_with_ttl(maxsize=1000, ttl=3600)
async def extract_title_from_telegram_msg_cached(msg):
    """Cached title extraction"""
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

# VIDEO THUMBNAIL PROCESSING
@performance_monitor.measure("thumbnail_extraction")
async def extract_video_thumbnail(user_client, message):
    """Optimized thumbnail extraction"""
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

# MONGODB INITIALIZATION WITH OPTIMIZATIONS
@performance_monitor.measure("mongodb_init")
async def init_mongodb():
    """Optimized MongoDB initialization"""
    global mongo_client, db, files_col, verification_col, poster_col
    
    try:
        logger.info("🔌 MongoDB initialization...")
        
        # ✅ FIXED: Increase timeout for cloud databases
        mongo_client = AsyncIOMotorClient(
            Config.MONGODB_URI,
            serverSelectionTimeoutMS=10000,  # ✅ Increased to 10 seconds
            connectTimeoutMS=10000,
            socketTimeoutMS=15000,
            maxPoolSize=20,
            minPoolSize=5,
            maxIdleTimeMS=30000,
            retryWrites=True,
            retryReads=True,
            ssl=True  # ✅ Add SSL for cloud databases
        )
        
        # Quick ping test with longer timeout
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
    return await index_single_file_smart(message)

# OLD BACKGROUND INDEXING FUNCTION (FOR BACKWARD COMPATIBILITY)
async def index_files_background():
    """Optimized background indexing (legacy function)"""
    await index_files_background_smart()

# POSTER FETCHING WITH CACHE
@performance_monitor.measure("poster_fetch")
async def get_poster_guaranteed(title, year=""):
    """Optimized poster fetching"""
    if poster_fetcher:
        poster_data = await poster_fetcher.fetch_poster(title, year)
        if poster_data:
            return {
                'poster_url': poster_data.get('url', ''),
                'source': poster_data.get('source', 'custom'),
                'rating': poster_data.get('rating', '0.0'),
                'year': year
            }
    
    # Fallback using Config.get_poster()
    return {
        'poster_url': Config.get_poster(title, year),
        'source': 'custom',
        'rating': '0.0',
        'year': year,
        'title': title
    }

# ULTRA-FAST SEARCH FUNCTION
@performance_monitor.measure("multi_channel_search")
@async_cache_with_ttl(maxsize=500, ttl=300)
async def search_movies_multi_channel(query, limit=12, page=1):
    """Turbo-charged multi-channel search"""
    offset = (page - 1) * limit
    
    # Try cache first
    cache_key = f"search:{query}:{page}:{limit}"
    if cache_manager and cache_manager.redis_enabled:
        cached_data = await cache_manager.get_search_results(query, page, limit)
        if cached_data:
            logger.info(f"✅ Cache HIT for: {query}")
            return cached_data
    
    logger.info(f"🔍 Multi-channel search for: {query}")
    
    query_lower = query.lower()
    posts_dict = {}
    files_dict = {}
    
    # MongoDB search with optimization
    try:
        if files_col is not None:
            # Use projection for faster queries
            cursor = files_col.find(
                {'$text': {'$search': query}},
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
            ).limit(limit * 2)  # Get slightly more for filtering
            
            async for doc in cursor:
                try:
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
                except:
                    continue
    except Exception as e:
        logger.error(f"File search error: {e}")
    
    # Telegram channel search (async)
    if user_session_ready:
        channel_tasks = []
        
        async def search_channel(channel_id):
            channel_posts = {}
            try:
                cname = channel_name_cached(channel_id)
                async for msg in safe_telegram_generator(
                    User.search_messages, 
                    channel_id, 
                    query=query, 
                    limit=15  # Reduced from 20
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
        
        # Search channels concurrently
        for channel_id in Config.TEXT_CHANNEL_IDS:
            channel_tasks.append(search_channel(channel_id))
        
        results = await asyncio.gather(*channel_tasks, return_exceptions=True)
        
        # Merge results
        for result in results:
            if isinstance(result, dict):
                posts_dict.update(result)
    
    # Merge posts and files
    merged = {}
    
    # Add all posts
    for norm_title, post_data in posts_dict.items():
        merged[norm_title] = post_data
    
    # Add/update with file information
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
    
    # Cache results
    if cache_manager:
        await cache_manager.cache_search_results(query, page, limit, result_data)
    
    logger.info(f"✅ Search completed: {len(paginated)} results")
    
    return result_data

# LIVE POSTS WITH CACHE
@async_cache_with_ttl(maxsize=100, ttl=60)
async def get_live_posts_multi_channel(limit_per_channel=10):
    """Cached live posts"""
    if not User or not user_session_ready:
        return []
    
    all_posts = []
    
    async def fetch_channel_posts(channel_id):
        posts = []
        try:
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
                            'channel_name': channel_name_cached(channel_id),
                            'channel_id': channel_id,
                            'message_id': msg.id,
                            'date': msg.date,
                            'is_new': is_new(msg.date) if msg.date else False
                        })
        except Exception as e:
            logger.error(f"Error getting posts from channel {channel_id}: {e}")
        return posts
    
    # Fetch concurrently
    tasks = [fetch_channel_posts(channel_id) for channel_id in Config.TEXT_CHANNEL_IDS]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for result in results:
        if isinstance(result, list):
            all_posts.extend(result)
    
    # Deduplicate and sort
    seen_titles = set()
    unique_posts = []
    
    for post in sorted(all_posts, key=lambda x: x.get('date', datetime.min), reverse=True):
        if post['normalized_title'] not in seen_titles:
            seen_titles.add(post['normalized_title'])
            unique_posts.append(post)
    
    return unique_posts[:20]

# ================================
# ✅ Fetch 30 Real Movies from Telegram - NO FALLBACK
# ================================
@performance_monitor.measure("home_movies_telegram")
async def get_home_movies_telegram(limit=30):
    """Fetch 30 real movies directly from Telegram MAIN_CHANNEL_ID - NO FALLBACK"""
    try:
        if not User or not user_session_ready:
            logger.warning("❌ User session not ready for Telegram fetch")
            return []  # ✅ EMPTY ARRAY - NO FALLBACK
        
        movies = []
        seen_titles = set()
        
        logger.info(f"🎬 Fetching {limit} real movies from Telegram channel {Config.MAIN_CHANNEL_ID}...")
        
        async for msg in safe_telegram_generator(
            User.get_chat_history, 
            Config.MAIN_CHANNEL_ID, 
            limit=limit * 2  # Fetch extra to account for non-movie posts
        ):
            if msg and msg.text and len(msg.text) > 20:  # Minimum text length
                # Extract title from message
                title = extract_title_smart(msg.text)
                
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    
                    # Parse year from title if available
                    year_match = re.search(r'\b(19|20)\d{2}\b', title)
                    year = year_match.group() if year_match else ""
                    
                    # Clean title for display
                    clean_title = re.sub(r'\s+\(\d{4}\)$', '', title)
                    clean_title = re.sub(r'\s+\d{4}$', '', clean_title)
                    
                    # Get poster URL
                    poster_url = Config.get_poster(title, year)
                    
                    # If poster_fetcher is available, try to get better poster
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
        return movies[:limit]  # Ensure exact limit
        
    except Exception as e:
        logger.error(f"❌ Telegram movies fetch error: {e}")
        return []  # ✅ EMPTY ARRAY ON ERROR - NO FALLBACK

async def get_single_post_api(channel_id, message_id):
    """Get single movie/post details"""
    try:
        # Try to get the message
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
                
                # Search for files with same title
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
                
                # Get poster - using Config.get_poster()
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
                
                # FIXED: Using channel_name function
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
                
                return post_data
        
        return None
        
    except Exception as e:
        logger.error(f"Single post API error: {e}")
        return None

# OPTIMIZED API FUNCTIONS
@performance_monitor.measure("search_api")
async def search_movies_api(query, limit=12, page=1):
    """Optimized search API with timeout protection"""
    try:
        # Set timeout for search
        search_task = asyncio.create_task(search_movies_multi_channel(query, limit, page))
        
        try:
            result_data = await asyncio.wait_for(search_task, timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning(f"⏰ Search timeout for query: {query}")
            # Return cached or empty results
            if cache_manager:
                cached = await cache_manager.get_search_results(query, page, limit)
                if cached:
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
        
        # Enhance with posters
        if result_data.get('results'):
            for result in result_data['results']:
                title = result.get('title', '')
                year_match = re.search(r'\b(19|20)\d{2}\b', title)
                year = year_match.group() if year_match else ""
                
                result['poster_url'] = Config.get_poster(title, year)
                result['poster_source'] = 'custom'
                result['poster_rating'] = '0.0'
                result['has_poster'] = True
        
        return result_data
        
    except Exception as e:
        logger.error(f"Search API error: {e}")
        raise

# Update get_home_movies_live function:
@performance_monitor.measure("home_movies")
async def get_home_movies_live():
    """Optimized home movies with timeout - Now uses Telegram, NO FALLBACK"""
    try:
        posts_task = asyncio.create_task(get_home_movies_telegram(limit=30))
        posts = await asyncio.wait_for(posts_task, timeout=5.0)
        
        return posts  # Could be empty array
        
    except asyncio.TimeoutError:
        logger.warning("⏰ Home movies timeout")
        return []  # ✅ EMPTY ARRAY ON TIMEOUT - NO FALLBACK
    
    except Exception as e:
        logger.error(f"Home movies error: {e}")
        return []  # ✅ EMPTY ARRAY ON ERROR - NO FALLBACK

# ============================================================================
# ✅ SYNC MANAGEMENT API ENDPOINTS
# ============================================================================

@app.route('/api/sync/status', methods=['GET'])
async def api_sync_status():
    """Get sync management status"""
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
        
        # Add MongoDB stats if available
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
    try:
        if not user_session_ready:
            return jsonify({
                'status': 'error',
                'message': 'Telegram session not ready'
            }), 400
        
        await channel_sync_manager.start_sync_monitoring()
        
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
    try:
        await channel_sync_manager.stop_sync_monitoring()
        
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
    try:
        if not user_session_ready:
            return jsonify({
                'status': 'error',
                'message': 'Telegram session not ready'
            }), 400
        
        # Run manual sync
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

@app.route('/api/sync/stats', methods=['GET'])
async def api_sync_stats():
    """Get detailed sync management stats"""
    try:
        if not files_col:
            return jsonify({
                'status': 'error',
                'message': 'MongoDB not available'
            }), 500
        
        # Get detailed stats
        total_files = await files_col.count_documents({})
        file_channel_files = await files_col.count_documents({"channel_id": Config.FILE_CHANNEL_ID})
        video_files = await files_col.count_documents({"is_video_file": True})
        
        # Get file size distribution
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
# TELEGRAM STATUS API ENDPOINTS
# ============================================================================

@app.route('/api/telegram/status', methods=['GET'])
async def api_telegram_status():
    """Get detailed Telegram connection status"""
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
        
        # Test user connection if available
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
        
        # Test bot connection if available
        if bot and bot_started:
            try:
                bot_info = await bot.get_me()
                status['connections']['bot_session']['bot_info'] = {
                    'username': bot_info.username,
                    'id': bot_info.id
                }
            except:
                pass
        
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
        
        # Test channel access
        if User and user_session_ready:
            # Test main channel
            try:
                chat = await User.get_chat(Config.MAIN_CHANNEL_ID)
                test_results['channel_tests'].append({
                    'channel_id': Config.MAIN_CHANNEL_ID,
                    'name': chat.title,
                    'accessible': True,
                    'type': 'main'
                })
                
                # Try to fetch a message
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
            
            # Test FILE channel
            try:
                file_chat = await User.get_chat(Config.FILE_CHANNEL_ID)
                test_results['channel_tests'].append({
                    'channel_id': Config.FILE_CHANNEL_ID,
                    'name': file_chat.title,
                    'accessible': True,
                    'type': 'file',
                    'sync_management': True
                })
                
                # Try to fetch a message from FILE channel
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
        
        # Import PosterFetcher here to avoid circular imports
        from poster_fetching import PosterFetcher
        
        # Initialize MongoDB
        mongo_ok = await init_mongodb()
        if not mongo_ok:
            logger.error("❌ MongoDB initialization failed")
            # Continue anyway - some features will work without DB
        
        # Initialize modular components
        global cache_manager, verification_system, premium_system, poster_fetcher, sk4film_bot
        
        # Initialize Cache Manager
        cache_manager = CacheManager(Config)
        redis_ok = await cache_manager.init_redis()
        if redis_ok:
            logger.info("✅ Cache Manager initialized")
            await cache_manager.start_cleanup_task()
        else:
            logger.warning("⚠️ Cache Manager - Redis not available")
        
        # Initialize Verification System
        if mongo_ok:
            verification_system = VerificationSystem(Config, db)
            await verification_system.start_cleanup_task()
            logger.info("✅ Verification System initialized")
        else:
            logger.warning("⚠️ Verification System - MongoDB not available")
        
        # Initialize Premium System
        if mongo_ok:
            premium_system = PremiumSystem(Config, db)
            await premium_system.start_cleanup_task()
            logger.info("✅ Premium System initialized")
        else:
            logger.warning("⚠️ Premium System - MongoDB not available")
        
        # ✅ Initialize Poster Fetcher
        poster_fetcher = PosterFetcher(cache_manager, Config)
        logger.info("✅ Poster Fetcher initialized")
        
        # ✅ WARM UP CACHE FOR INSTANT RESPONSE
        asyncio.create_task(warm_up_cache())
        
        # ✅ Initialize Telegram if available
        if PYROGRAM_AVAILABLE:
            # Initialize Telegram clients
            telegram_ok = await init_telegram_clients()
            if not telegram_ok:
                logger.warning("⚠️ Telegram clients not initialized - Movies will be empty")
            
            # Initialize SK4FiLMBot if available
            if BOT_HANDLERS_AVAILABLE:
                sk4film_bot = SK4FiLMBot(Config, db)
                await sk4film_bot.initialize()
                logger.info("✅ SK4FiLMBot initialized")
                
                # Setup bot handlers
                if sk4film_bot and sk4film_bot.bot_started and bot:
                    await setup_bot_handlers(bot, sk4film_bot)
                    logger.info("✅ Bot handlers setup complete")
        else:
            logger.warning("⚠️ Pyrogram not available - Movies will be empty")
            sk4film_bot = None
        
        # Start background tasks
        asyncio.create_task(cache_cleanup())
        
        # Start SMART indexing in background only if Telegram is ready
        if user_session_ready:
            asyncio.create_task(index_files_background_smart())
            logger.info("✅ Started SMART background indexing with sync management")
        else:
            logger.warning("⚠️ Cannot start indexing - User session not ready")
        
        init_time = time.time() - start_time
        logger.info(f"⚡ SK4FiLM Started in {init_time:.2f}s - SYNC MANAGEMENT READY")
        
        # Log initial performance
        logger.info(f"📊 Initial performance: {performance_monitor.get_stats()}")
        
        # Log sync management status
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
    tf = await files_col.count_documents({}) if files_col is not None else 0
    video_files = await files_col.count_documents({'is_video_file': True}) if files_col is not None else 0
    posters_cached = await poster_col.count_documents({}) if poster_col is not None else 0
    
    # Sync management stats
    sync_stats = {
        'sync_monitoring': channel_sync_manager.is_monitoring,
        'file_channel_id': Config.FILE_CHANNEL_ID,
        'ttl_expiry': False,
        'deleted_count': channel_sync_manager.deleted_count
    }
    
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
    try:
        movies = await get_home_movies_live()
        
        return jsonify({
            'status': 'success' if movies else 'empty',
            'movies': movies,  # Could be empty array
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
            'movies': [],  # ✅ EMPTY ARRAY ON ERROR
            'total': 0,
            'telegram_ready': user_session_ready
        }), 500

@app.route('/api/search', methods=['GET'])
@performance_monitor.measure("search_endpoint")
async def api_search():
    """Optimized search endpoint"""
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
    """Get single post details"""
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
    """Get movie poster"""
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
        
        # ✅ FIXED: Using Config.get_poster() for fallback
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
    try:
        data = await request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No JSON data provided'
            }), 400
        
        user_id = data.get('user_id')
        verification_url = data.get('verification_url')
        
        if not user_id:
            return jsonify({
                'status': 'error',
                'message': 'user_id is required'
            }), 400
        
        result = await verify_user_api(user_id, verification_url)
        
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
    try:
        if premium_system:
            plans = await premium_system.get_all_plans()
            
            return jsonify({
                'status': 'success',
                'plans': plans,
                'timestamp': datetime.now().isoformat()
            })
        else:
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
    try:
        user_id = int(request.args.get('user_id', 0))
        
        if user_id <= 0:
            return jsonify({
                'status': 'error',
                'message': 'Valid user_id is required'
            }), 400
        
        result = {}
        
        # Get premium status
        if premium_system:
            premium_status = await premium_system.get_subscription_details(user_id)
            result['premium'] = premium_status
        
        # Get verification status
        if verification_system:
            is_verified, message = await verification_system.check_user_verified(user_id, premium_system)
            result['verification'] = {
                'is_verified': is_verified,
                'message': message,
                'needs_verification': not is_verified and Config.VERIFICATION_REQUIRED
            }
        
        # Get download permissions
        if premium_system:
            can_download, download_message, download_details = await premium_system.can_user_download(user_id)
            result['download'] = {
                'can_download': can_download,
                'message': download_message,
                'details': download_details
            }
        
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
    try:
        data = await request.get_json()
        admin_key = data.get('admin_key') if data else request.headers.get('X-Admin-Key')
        
        if not admin_key or admin_key != os.environ.get('ADMIN_KEY', 'sk4film_admin_123'):
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
            await cache_manager.clear_all()
            cleared['cache_manager'] = True
        
        if poster_fetcher:
            poster_fetcher.clear_cache()
            cleared['poster_fetcher'] = True
        
        if cache_manager:
            cleared['search_cache'] = await cache_manager.clear_search_cache()
        
        # Clear MongoDB poster collection
        if poster_col:
            await poster_col.delete_many({})
            cleared['poster_collection'] = True
        
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
    """Optimized stats endpoint"""
    try:
        stats = {}
        
        # Database stats
        if files_col:
            stats['database'] = {
                'total_files': await files_col.count_documents({}),
                'video_files': await files_col.count_documents({'is_video_file': True}),
                'unique_titles': len(await files_col.distinct('normalized_title')),
                'file_channel_files': await files_col.count_documents({"channel_id": Config.FILE_CHANNEL_ID}),
                'ttl_expiry': False,
                'sync_only': True
            }
        
        # Sync management stats
        stats['sync_management'] = {
            'sync_monitoring': channel_sync_manager.is_monitoring,
            'deleted_count': channel_sync_manager.deleted_count,
            'monitoring_interval': Config.MONITOR_INTERVAL,
            'file_channel_id': Config.FILE_CHANNEL_ID,
            'last_sync': channel_sync_manager.last_sync
        }
        
        # Poster stats
        if poster_col:
            stats['posters'] = {
                'total_posters': await poster_col.count_documents({}),
                'expired_posters': await poster_col.count_documents({"expires_at": {"$lt": datetime.utcnow()}})
            }
        
        # Cache stats
        if cache_manager:
            cache_stats = await cache_manager.get_stats_summary()
            stats['cache'] = cache_stats
        
        # Performance stats
        stats['performance'] = performance_monitor.get_stats()
        
        # System info
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
    try:
        # Security check - only allow in development
        if os.environ.get('ENVIRONMENT') != 'development':
            return jsonify({
                'status': 'error',
                'message': 'Session generation only allowed in development'
            }), 403
        
        data = await request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No data provided'
            }), 400
        
        api_id = data.get('api_id')
        api_hash = data.get('api_hash')
        
        if not api_id or not api_hash:
            return jsonify({
                'status': 'error',
                'message': 'API ID and API Hash required'
            }), 400
        
        logger.info("🔧 Generating Telegram session...")
        
        try:
            # This would need to be implemented with proper async handling
            # For now, just return instructions
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
            logger.error(f"Session generation error: {e}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
            
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
    await init_system()

@app.after_serving
async def shutdown():
    """Optimized shutdown"""
    logger.info("🛑 Shutting down SK4FiLM...")
    
    shutdown_tasks = []
    
    # Stop sync monitoring
    await channel_sync_manager.stop_sync_monitoring()
    
    if User and user_session_ready:
        shutdown_tasks.append(User.stop())
    
    if bot and bot_started:
        shutdown_tasks.append(bot.stop())
    
    if sk4film_bot:
        shutdown_tasks.append(sk4film_bot.shutdown())
    
    if cache_manager:
        shutdown_tasks.append(cache_manager.stop())
    
    if verification_system:
        shutdown_tasks.append(verification_system.stop())
    
    if premium_system:
        shutdown_tasks.append(premium_system.stop_cleanup_task())
    
    # Execute shutdown tasks concurrently
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
    config.accesslog = None  # Disable access log for performance
    config.errorlog = "-"
    config.loglevel = "warning"  # Reduced logging
    config.http2 = True  # Enable HTTP/2
    config.keep_alive_timeout = 30
    
    logger.info(f"🌐 Starting Quart server on port {Config.WEB_SERVER_PORT}...")
    
    # Run with performance monitoring
    asyncio.run(serve(app, config))
