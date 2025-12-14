#all features keep don't remove
import asyncio
import os
import logging
import json
import re
import math
import time
import base64
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from functools import lru_cache, wraps
import urllib.parse

from quart import Quart, jsonify, request
from motor.motor_asyncio import AsyncIOMotorClient
import redis.asyncio as redis

# Pyrogram imports
try:
    from pyrogram import Client
    from pyrogram.errors import FloodWait
    PYROGRAM_AVAILABLE = True
except ImportError:
    PYROGRAM_AVAILABLE = False
    Client = None
    FloodWait = None

# ============================================================================
# ✅ MODULAR COMPONENT IMPORTS
# ============================================================================
try:
    from cache import CacheManager
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    CacheManager = None

try:
    from verification import VerificationSystem
    VERIFICATION_AVAILABLE = True
except ImportError:
    VERIFICATION_AVAILABLE = False
    VerificationSystem = None

try:
    from premium import PremiumSystem, PremiumTier
    PREMIUM_AVAILABLE = True
except ImportError:
    PREMIUM_AVAILABLE = False
    PremiumSystem = None
    PremiumTier = None

try:
    from poster_fetching import PosterFetcher
    POSTER_AVAILABLE = True
except ImportError:
    POSTER_AVAILABLE = False
    PosterFetcher = None

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
except ImportError:
    UTILS_AVAILABLE = False
    # Fallback functions
    def normalize_title(title):
        return title.lower().strip() if title else ""
    
    def extract_title_smart(text):
        if not text:
            return ""
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if len(line) > 10:
                return line[:60]
        return text[:60] if text else ""
    
    def extract_title_from_file(file_name, caption):
        if caption and len(caption) > 10:
            return caption.split('\n')[0][:60]
        if file_name:
            clean_name = re.sub(r'[._-]', ' ', file_name)
            clean_name = re.sub(r'\b(1080p|720p|480p|HD|WEBRip|BluRay|x264)\b', '', clean_name, flags=re.IGNORECASE)
            return clean_name.strip()[:60]
        return "Unknown"
    
    def format_size(size_bytes):
        if size_bytes == 0:
            return "0 B"
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_names[i]}"
    
    def detect_quality(filename):
        filename_lower = filename.lower()
        if any(q in filename_lower for q in ['4k', '2160p', 'uhd']):
            return '4K'
        elif any(q in filename_lower for q in ['1080p', 'fhd']):
            return '1080p'
        elif any(q in filename_lower for q in ['720p', 'hd']):
            return '720p'
        elif any(q in filename_lower for q in ['480p', 'sd']):
            return '480p'
        else:
            return 'Unknown'
    
    def is_video_file(filename):
        video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm']
        return any(filename.lower().endswith(ext) for ext in video_extensions)
    
    def format_post(text):
        return text if text else ""
    
    def is_new(date):
        if isinstance(date, datetime):
            return (datetime.now() - date).days <= 7
        return False

try:
    from bot_handlers import SK4FiLMBot, setup_bot_handlers
    BOT_HANDLERS_AVAILABLE = True
except ImportError:
    BOT_HANDLERS_AVAILABLE = False
    SK4FiLMBot = None
    setup_bot_handlers = None

# ============================================================================
# LOGGING SETUP
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

# ============================================================================
# PERFORMANCE MONITORING
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
            self.measurements[name] = {'count': 0, 'total': 0, 'avg': 0}
        stats = self.measurements[name]
        stats['count'] += 1
        stats['total'] += elapsed
        stats['avg'] = stats['total'] / stats['count']
        if elapsed > 0.5:
            logger.warning(f"⏱️ {name} took {elapsed:.3f}s")
    
    def get_stats(self):
        return self.measurements

performance_monitor = PerformanceMonitor()

# ============================================================================
# CONFIGURATION - FIXED WITH ALL MISSING ATTRIBUTES
# ============================================================================
class Config:
    API_ID = int(os.environ.get("API_ID", "0"))
    API_HASH = os.environ.get("API_HASH", "")
    USER_SESSION_STRING = os.environ.get("USER_SESSION_STRING", "")
    BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
    
    MONGODB_URI = os.environ.get("MONGODB_URI", "")
    REDIS_URL = os.environ.get("REDIS_URL", "")
    REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "")
    
    MAIN_CHANNEL_ID = -1001891090100
    TEXT_CHANNEL_IDS = [-1001891090100, -1002024811395]
    FILE_CHANNEL_ID = -1001768249569
    
    BACKEND_URL = os.environ.get("BACKEND_URL", "https://sk4film.koyeb.app")
    WEBSITE_URL = os.environ.get("WEBSITE_URL", "https://sk4film.vercel.app")
    WEB_SERVER_PORT = int(os.environ.get("PORT", "8000"))  # ADDED
    
    OMDB_KEYS = ["8265bd1c", "b9bd48a6"]
    MONITOR_INTERVAL = 300
    
    # ADDED ADMIN IDs
    ADMIN_IDS = [int(x) for x in os.environ.get("ADMIN_IDS", "123456789").split(",") if x]
    
    @staticmethod
    def get_poster(title, year=""):
        if not title:
            return "https://via.placeholder.com/300x450/1a1a2e/ffffff?text=No+Poster"
        encoded_title = urllib.parse.quote(title)
        return f"{Config.BACKEND_URL}/api/poster?title={encoded_title}&year={year}"

# ============================================================================
# QUART APP
# ============================================================================
app = Quart(__name__)
app.config['JSON_SORT_KEYS'] = False

@app.after_request
async def add_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================
mongo_client = None
db = None
files_col = None
verification_col = None
poster_col = None

cache_manager = None
verification_system = None
premium_system = None
poster_fetcher = None
sk4film_bot = None

User = None
bot = None
bot_started = False
user_session_ready = False

CHANNEL_CONFIG = {
    -1001891090100: {'name': 'SK4FiLM Main', 'type': 'text'},
    -1002024811395: {'name': 'SK4FiLM Updates', 'type': 'text'},
    -1001768249569: {'name': 'SK4FiLM Files', 'type': 'file', 'sync': True}
}

# ============================================================================
# CHANNEL SYNC MANAGER
# ============================================================================
class ChannelSyncManager:
    def __init__(self):
        self.is_monitoring = False
        self.monitoring_task = None
        self.deleted_count = 0
        self.last_sync = time.time()
    
    async def start_sync_monitoring(self):
        if not User or not user_session_ready:
            logger.warning("⚠️ Cannot start sync monitoring")
            return
        
        if self.is_monitoring:
            return
        
        logger.info("👁️ Starting sync monitoring...")
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self.monitor_channel_sync())
    
    async def stop_sync_monitoring(self):
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
    
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
            if not files_col or not User:
                return
            
            current_time = time.time()
            if current_time - self.last_sync < 300:
                return
            
            self.last_sync = current_time
            
            cursor = files_col.find(
                {"channel_id": Config.FILE_CHANNEL_ID},
                {"message_id": 1, "title": 1}
            )
            
            message_ids_in_db = []
            file_titles = {}
            
            async for doc in cursor:
                msg_id = doc.get('message_id')
                if msg_id:
                    message_ids_in_db.append(msg_id)
                    file_titles[msg_id] = doc.get('title', 'Unknown')
            
            if not message_ids_in_db:
                return
            
            deleted_count = 0
            batch_size = 50
            
            for i in range(0, len(message_ids_in_db), batch_size):
                batch = message_ids_in_db[i:i + batch_size]
                
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
                        result = await files_col.delete_many({
                            "channel_id": Config.FILE_CHANNEL_ID,
                            "message_id": {"$in": deleted_ids}
                        })
                        
                        if result.deleted_count > 0:
                            deleted_count += result.deleted_count
                            self.deleted_count += result.deleted_count
                            logger.info(f"🗑️ Deleted {result.deleted_count} files")
                
                except Exception as e:
                    logger.error(f"❌ Batch error: {e}")
            
            if deleted_count > 0:
                logger.info(f"✅ Sync complete: {deleted_count} deleted")
        
        except Exception as e:
            logger.error(f"❌ Sync error: {e}")

channel_sync_manager = ChannelSyncManager()

# ============================================================================
# FILE INDEXING WITH DUPLICATE PREVENTION - FIXED
# ============================================================================
async def generate_file_hash(message):
    """Generate unique hash for duplicate detection"""
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
    """Smart file indexing with 4-layer duplicate prevention"""
    try:
        if not files_col:
            logger.warning("⚠️ files_col not available")
            return False
        
        if not message or (not message.document and not message.video):
            return False
        
        # Layer 1: Check by channel_id + message_id
        existing_by_id = await files_col.find_one({
            'channel_id': Config.FILE_CHANNEL_ID,
            'message_id': message.id
        })
        
        if existing_by_id:
            logger.debug(f"📝 Already indexed: {message.id}")
            return True
        
        # Extract title
        title = await extract_title_from_telegram_msg_cached(message)
        if not title:
            logger.warning(f"⚠️ No title for message {message.id}")
            return False
        
        normalized_title = normalize_title(title)
        
        # Layer 2: Check by file hash
        file_hash = await generate_file_hash(message)
        if file_hash:
            existing_by_hash = await files_col.find_one({
                'channel_id': Config.FILE_CHANNEL_ID,
                'file_hash': file_hash
            })
            if existing_by_hash:
                logger.info(f"🔁 Duplicate (hash): {title}")
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
            })
            if existing_by_file_id:
                logger.info(f"🔁 Duplicate (file_id): {title}")
                return False
        
        # Layer 4: Check by title + similar size
        if file_size > 0:
            similar_files = await files_col.find({
                'channel_id': Config.FILE_CHANNEL_ID,
                'normalized_title': normalized_title
            }).limit(3).to_list(length=3)
            
            for existing_file in similar_files:
                existing_size = existing_file.get('file_size', 0)
                if existing_size > 0:
                    size_ratio = file_size / existing_size
                    if 0.8 <= size_ratio <= 1.2:
                        logger.info(f"🔁 Duplicate (title+size): {title}")
                        return False
        
        # Create document
        doc = {
            'channel_id': Config.FILE_CHANNEL_ID,
            'message_id': message.id,
            'title': title,
            'normalized_title': normalized_title,
            'date': message.date,
            'indexed_at': datetime.now(),
            'is_video_file': False,
            'thumbnail': None,
            'file_id': file_id,
            'file_size': file_size,
            'file_hash': file_hash,
            'status': 'active'
        }
        
        # Add file-specific data
        if message.document:
            doc.update({
                'file_name': message.document.file_name or '',
                'quality': detect_quality(message.document.file_name or ''),
                'is_video_file': is_video_file(message.document.file_name or ''),
                'caption': message.caption or ''
            })
        elif message.video:
            doc.update({
                'file_name': message.video.file_name or 'video.mp4',
                'quality': detect_quality(message.video.file_name or ''),
                'is_video_file': True,
                'caption': message.caption or '',
                'duration': getattr(message.video, 'duration', 0)
            })
        
        # Extract thumbnail if video
        if doc['is_video_file'] and User:
            try:
                thumbnail_url = await extract_video_thumbnail(User, message)
                if thumbnail_url:
                    doc['thumbnail'] = thumbnail_url
                    doc['thumbnail_source'] = 'video_direct'
            except Exception as e:
                logger.debug(f"Thumbnail error: {e}")
        
        # Insert document
        try:
            await files_col.insert_one(doc)
            file_type = "📹" if doc['is_video_file'] else "📄"
            logger.info(f"✅ {file_type} Indexed: {title}")
            return True
        except Exception as e:
            if "duplicate key" in str(e).lower():
                return True
            logger.error(f"❌ Insert error: {e}")
            return False
    
    except Exception as e:
        logger.error(f"❌ Indexing error: {e}")
        return False

async def index_files_background_smart():
    """Background file indexing with sync management"""
    if not User or not files_col or not user_session_ready:
        logger.warning(f"⚠️ Cannot start indexing - User:{User is not None}, files_col:{files_col is not None}, user_session_ready:{user_session_ready}")
        return
    
    logger.info("📁 Starting smart indexing...")
    
    try:
        await setup_mongodb_indexes()
        
        last_indexed = await files_col.find_one(
            {"channel_id": Config.FILE_CHANNEL_ID},
            sort=[('message_id', -1)]
        )
        
        last_message_id = last_indexed['message_id'] if last_indexed else 0
        logger.info(f"🔄 Starting from message: {last_message_id}")
        
        total_indexed = 0
        total_skipped = 0
        messages = []
        
        async for msg in safe_telegram_generator(
            User.get_chat_history,
            Config.FILE_CHANNEL_ID,
            limit=100
        ):
            if msg.id <= last_message_id:
                break
            if msg and (msg.document or msg.video):
                messages.append(msg)
        
        messages.reverse()
        
        for msg in messages:
            try:
                success = await index_single_file_smart(msg)
                if success:
                    total_indexed += 1
                else:
                    total_skipped += 1
                await asyncio.sleep(0.3)
            except Exception as e:
                logger.error(f"❌ Error processing {msg.id}: {e}")
        
        if total_indexed > 0:
            logger.info(f"✅ Indexed: {total_indexed} new files")
        if total_skipped > 0:
            logger.info(f"📝 Skipped: {total_skipped} duplicates")
        
        await channel_sync_manager.start_sync_monitoring()
        await channel_sync_manager.sync_deletions_from_telegram()
    
    except Exception as e:
        logger.error(f"❌ Background indexing error: {e}")

async def setup_mongodb_indexes():
    """Setup MongoDB indexes"""
    try:
        if not files_col:
            return
        
        try:
            await files_col.create_index(
                [("channel_id", 1), ("message_id", 1)],
                unique=True
            )
        except Exception as e:
            logger.debug(f"Index 1 error: {e}")
        
        try:
            await files_col.create_index([("file_hash", 1)])
        except Exception as e:
            logger.debug(f"Index 2 error: {e}")
        
        try:
            await files_col.create_index([("file_id", 1)])
        except Exception as e:
            logger.debug(f"Index 3 error: {e}")
        
        try:
            await files_col.create_index([("normalized_title", "text")])
        except Exception as e:
            logger.debug(f"Index 4 error: {e}")
        
        logger.info("✅ MongoDB indexes ready")
    except Exception as e:
        logger.error(f"❌ Index setup error: {e}")

# ============================================================================
# HELPER FUNCTIONS - FIXED
# ============================================================================
@lru_cache(maxsize=10000)
def channel_name_cached(cid):
    return CHANNEL_CONFIG.get(cid, {}).get('name', f"Channel {cid}")

def channel_name(channel_id):
    return channel_name_cached(channel_id)

async def extract_video_thumbnail(user_client, message):
    """Extract video thumbnail"""
    try:
        if message.video:
            thumbnail = message.video.thumbs[0] if message.video.thumbs else None
            if thumbnail:
                thumb_path = await safe_telegram_operation(
                    user_client.download_media,
                    thumbnail.file_id,
                    in_memory=True
                )
                if thumb_path:
                    thumb_data = base64.b64encode(thumb_path.getvalue()).decode('utf-8')
                    return f"data:image/jpeg;base64,{thumb_data}"
        return None
    except Exception as e:
        logger.debug(f"Thumbnail error: {e}")
        return None

async def extract_title_from_telegram_msg_cached(msg):
    """Cached title extraction"""
    try:
        caption = msg.caption if hasattr(msg, 'caption') else None
        file_name = None
        
        if msg.document:
            file_name = msg.document.file_name
        elif msg.video:
            file_name = msg.video.file_name
        
        if UTILS_AVAILABLE:
            return extract_title_from_file(file_name, caption)
        else:
            # Use our fallback function
            if caption and len(caption) > 10:
                return caption.split('\n')[0][:60]
            if file_name:
                clean_name = re.sub(r'[._-]', ' ', file_name)
                clean_name = re.sub(r'\b(1080p|720p|480p|HD|WEBRip|BluRay|x264)\b', '', clean_name, flags=re.IGNORECASE)
                return clean_name.strip()[:60]
            return "Unknown"
    except Exception as e:
        logger.debug(f"Title extraction error: {e}")
        return None

class TurboFloodProtection:
    def __init__(self):
        self.request_buckets = {}
        self.last_cleanup = time.time()
    
    async def wait_if_needed(self):
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
        
        self.request_buckets[bucket_key] += 1
    
    def _cleanup_buckets(self, current_time):
        current_bucket = int(current_time // 10)
        old_buckets = [k for k in self.request_buckets.keys() if k < current_bucket - 6]
        for bucket in old_buckets:
            del self.request_buckets[bucket]

turbo_flood_protection = TurboFloodProtection()

@performance_monitor.measure("telegram_operation")
async def safe_telegram_operation(operation, *args, **kwargs):
    """Safe Telegram operations with retry"""
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
                    await asyncio.sleep(wait_time + 1)
                    continue
            if attempt == max_retries - 1:
                logger.error(f"❌ Telegram operation failed: {e}")
                raise
            await asyncio.sleep(0.5 * (2 ** attempt))
    return None

async def safe_telegram_generator(operation, *args, limit=None, **kwargs):
    """Safe async generator"""
    count = 0
    async for item in operation(*args, **kwargs):
        yield item
        count += 1
        if count % 10 == 0:
            await asyncio.sleep(0.1)
        if limit and count >= limit:
            break

# ============================================================================
# DATABASE INITIALIZATION - FIXED
# ============================================================================
@performance_monitor.measure("mongodb_init")
async def init_mongodb():
    """MongoDB initialization"""
    global mongo_client, db, files_col, verification_col, poster_col
    
    try:
        logger.info("🔌 MongoDB initialization...")
        
        mongo_uri = Config.MONGODB_URI
        
        if not mongo_uri:
            logger.warning("⚠️ MONGODB_URI not set")
            return False
        
        try:
            mongo_client = AsyncIOMotorClient(
                mongo_uri,
                serverSelectionTimeoutMS=10000,
                connectTimeoutMS=10000,
                socketTimeoutMS=15000,
                maxPoolSize=20,
                retryWrites=True,
                ssl=True if "mongodb+srv://" in mongo_uri else False
            )
            
            await asyncio.wait_for(mongo_client.admin.command('ping'), timeout=10)
            
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning(f"⚠️ Primary connection failed: {e}")
            
            # Try without SSL
            mongo_client = AsyncIOMotorClient(
                mongo_uri,
                serverSelectionTimeoutMS=15000,
                connectTimeoutMS=15000,
                socketTimeoutMS=20000,
                maxPoolSize=5,
                retryWrites=True,
                ssl=False
            )
            await asyncio.wait_for(mongo_client.admin.command('ping'), timeout=15)
        
        db = mongo_client.sk4film
        files_col = db.files
        verification_col = db.verifications
        poster_col = db.posters
        
        logger.info("✅ MongoDB connected")
        return True
    
    except asyncio.TimeoutError:
        logger.error("❌ MongoDB timeout - Check your connection string and network")
        logger.warning("⚠️ Proceeding without MongoDB - Some features will be limited")
        return False
    except Exception as e:
        logger.error(f"❌ MongoDB error: {e}")
        logger.warning("⚠️ Proceeding without MongoDB - Some features will be limited")
        return False

# ============================================================================
# TELEGRAM INITIALIZATION
# ============================================================================
@performance_monitor.measure("telegram_init")
async def init_telegram_clients():
    """Initialize Telegram clients"""
    global User, bot, bot_started, user_session_ready
    
    logger.info("=" * 60)
    logger.info("🚀 TELEGRAM INITIALIZATION")
    logger.info("=" * 60)
    
    if not PYROGRAM_AVAILABLE:
        logger.error("❌ Pyrogram not installed")
        return False
    
    # Check environment
    if not Config.API_ID or not Config.API_HASH or not Config.USER_SESSION_STRING:
        logger.error("❌ Missing Telegram credentials")
        return False
    
    # Initialize User Client
    logger.info("👤 Initializing User Client...")
    
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
        
        await asyncio.wait_for(User.start(), timeout=20)
        
        me = await User.get_me()
        logger.info(f"✅ User Client: {me.first_name}")
        
        # Test channel access
        try:
            chat = await User.get_chat(Config.FILE_CHANNEL_ID)
            logger.info(f"✅ Channel Access: {chat.title}")
            user_session_ready = True
        except Exception as e:
            logger.warning(f"⚠️ Channel access issue: {e}")
            user_session_ready = False
    
    except Exception as e:
        logger.error(f"❌ User client failed: {e}")
        user_session_ready = False
    
    # Initialize Bot Client
    if Config.BOT_TOKEN:
        logger.info("🤖 Initializing Bot Client...")
        
        try:
            bot = Client(
                "sk4film_bot",
                api_id=Config.API_ID,
                api_hash=Config.API_HASH,
                bot_token=Config.BOT_TOKEN,
                in_memory=True,
                no_updates=True
            )
            
            await bot.start()
            bot_info = await bot.get_me()
            logger.info(f"✅ Bot Ready: @{bot_info.username}")
            bot_started = True
        
        except Exception as e:
            logger.error(f"❌ Bot failed: {e}")
            bot_started = False
    
    logger.info("=" * 60)
    logger.info(f"User Session: {'✅ READY' if user_session_ready else '❌ NOT READY'}")
    logger.info(f"Bot Session: {'✅ READY' if bot_started else '❌ NOT READY'}")
    logger.info("=" * 60)
    
    return user_session_ready or bot_started

# ============================================================================
# SYSTEM INITIALIZATION - FIXED
# ============================================================================
@performance_monitor.measure("system_init")
async def init_system():
    """Initialize all systems"""
    global cache_manager, verification_system, premium_system, poster_fetcher, sk4film_bot
    
    start_time = time.time()
    
    try:
        logger.info("🚀 Starting SK4FiLM v8.0 - MODULAR EDITION...")
        
        # Initialize MongoDB
        mongo_ok = await init_mongodb()
        
        # Initialize Cache Manager
        if CACHE_AVAILABLE:
            try:
                cache_manager = CacheManager(Config)
                redis_ok = await cache_manager.init_redis()
                if redis_ok:
                    logger.info("✅ Cache Manager initialized")
                else:
                    logger.warning("⚠️ Redis not available")
            except Exception as e:
                logger.warning(f"⚠️ Cache Manager not available: {e}")
                cache_manager = None
        else:
            logger.warning("⚠️ Cache module not available")
        
        # Initialize Poster Fetcher - FIXED
        if POSTER_AVAILABLE and cache_manager:
            try:
                # Pass OMDB keys and cache_manager
                poster_fetcher = PosterFetcher(Config.OMDB_KEYS, cache_manager)
                logger.info("✅ Poster Fetcher initialized")
            except Exception as e:
                logger.warning(f"⚠️ Poster Fetcher not available: {e}")
                poster_fetcher = None
        else:
            logger.warning("⚠️ Poster Fetcher not available")
        
        # Initialize Verification System
        if VERIFICATION_AVAILABLE and mongo_ok:
            try:
                verification_system = VerificationSystem(Config, db)
                logger.info("✅ Verification System initialized")
            except Exception as e:
                logger.warning(f"⚠️ Verification System not available: {e}")
                verification_system = None
        
        # Initialize Premium System
        if PREMIUM_AVAILABLE and mongo_ok:
            try:
                premium_system = PremiumSystem(Config, db)
                logger.info("✅ Premium System initialized")
            except Exception as e:
                logger.warning(f"⚠️ Premium System not available: {e}")
                premium_system = None
        
        # Initialize Telegram
        telegram_ok = await init_telegram_clients()
        
        if not telegram_ok:
            logger.warning("⚠️ Telegram not initialized - Movies will be empty")
        
        # Setup Bot Handlers
        if BOT_HANDLERS_AVAILABLE and bot_started and bot:
            try:
                if SK4FiLMBot:
                    sk4film_bot = SK4FiLMBot(Config, db)
                    await sk4film_bot.initialize()
                    logger.info("✅ SK4FiLMBot initialized")
                
                if setup_bot_handlers:
                    await setup_bot_handlers(bot, sk4film_bot)
                    logger.info("✅ Bot handlers setup complete")
            except Exception as e:
                logger.warning(f"⚠️ Bot handlers not available: {e}")
        
        # Start background indexing
        if user_session_ready:
            asyncio.create_task(index_files_background_smart())
            logger.info("✅ Background indexing started")
        else:
            logger.warning("⚠️ Cannot start indexing - User session not ready")
        
        init_time = time.time() - start_time
        logger.info(f"⚡ SK4FiLM Started in {init_time:.2f}s")
        logger.info(f"📊 Performance: {performance_monitor.get_stats()}")
    
    except Exception as e:
        logger.error(f"❌ System initialization error: {e}")

# ============================================================================
# HOME MOVIES FUNCTION
# ============================================================================
@performance_monitor.measure("home_movies")
async def get_home_movies_telegram(limit=30):
    """Fetch home movies from Telegram"""
    try:
        if not User or not user_session_ready:
            logger.warning("⚠️ User session not ready for home movies")
            return []
        
        movies = []
        seen_titles = set()
        
        async for msg in safe_telegram_generator(
            User.get_chat_history,
            Config.MAIN_CHANNEL_ID,
            limit=limit + 10
        ):
            if not msg or not msg.text:
                continue
            
            title = extract_title_smart(msg.text) if UTILS_AVAILABLE else msg.text.split('\n')[0][:60]
            
            if not title or title in seen_titles:
                continue
            
            seen_titles.add(title)
            
            # Extract year
            year_match = re.search(r'\b(19|20)\d{2}\b', title)
            year = year_match.group() if year_match else ""
            
            # Get poster
            poster_url = Config.get_poster(title, year)
            
            if poster_fetcher:
                try:
                    poster_data = await poster_fetcher.fetch_poster(title, year)
                    if poster_data and poster_data.get('url'):
                        poster_url = poster_data['url']
                except Exception as e:
                    logger.debug(f"Poster fetch error: {e}")
            
            movies.append({
                'title': title,
                'year': year,
                'date': msg.date.isoformat() if isinstance(msg.date, datetime) else str(msg.date),
                'is_new': is_new(msg.date) if UTILS_AVAILABLE and msg.date else False,
                'channel': channel_name(Config.MAIN_CHANNEL_ID),
                'channel_id': Config.MAIN_CHANNEL_ID,
                'message_id': msg.id,
                'poster_url': poster_url,
                'poster_source': 'telegram'
            })
            
            if len(movies) >= limit:
                break
        
        logger.info(f"✅ Fetched {len(movies)} movies from Telegram")
        return movies[:limit]
    
    except Exception as e:
        logger.error(f"❌ Home movies error: {e}")
        return []

# ============================================================================
# SEARCH FUNCTION
# ============================================================================
@performance_monitor.measure("multi_channel_search")
async def search_movies_multi_channel(query, limit=12, page=1):
    """Multi-channel search"""
    try:
        offset = (page - 1) * limit
        query_lower = query.lower()
        
        posts_dict = {}
        files_dict = {}
        
        # MongoDB search
        if files_col:
            try:
                # Try text search first, then fallback to regex
                search_filter = {"$text": {"$search": query}} if "text" in await files_col.index_information() else {"normalized_title": {"$regex": query_lower, "$options": "i"}}
                
                cursor = files_col.find(
                    search_filter,
                    {"title": 1, "normalized_title": 1, "quality": 1, 
                     "file_size": 1, "file_name": 1, "thumbnail": 1,
                     "channel_id": 1, "message_id": 1, "date": 1}
                ).limit(limit + 2)
                
                async for doc in cursor:
                    norm_title = doc.get('normalized_title', normalize_title(doc.get('title', '')))
                    quality = doc.get('quality', '480p')
                    
                    if norm_title not in files_dict:
                        files_dict[norm_title] = {
                            'title': doc.get('title', 'Unknown'),
                            'quality_options': {},
                            'date': doc.get('date', datetime.now()).isoformat() if isinstance(doc.get('date'), datetime) else str(doc.get('date', '')),
                            'thumbnail': doc.get('thumbnail'),
                            'channel_id': doc.get('channel_id'),
                            'channel_name': channel_name(doc.get('channel_id'))
                        }
                    
                    if quality not in files_dict[norm_title]['quality_options']:
                        files_dict[norm_title]['quality_options'][quality] = {
                            'file_id': f"{doc.get('channel_id')}_{doc.get('message_id')}_{quality}",
                            'file_size': doc.get('file_size', 0),
                            'file_name': doc.get('file_name', 'video.mp4'),
                            'channel_id': doc.get('channel_id'),
                            'message_id': doc.get('message_id')
                        }
            except Exception as e:
                logger.error(f"❌ File search error: {e}")
        
        # Telegram search
        if user_session_ready:
            async def search_channel(channel_id):
                channel_posts = {}
                try:
                    async for msg in safe_telegram_generator(
                        User.search_messages,
                        channel_id,
                        query=query,
                        limit=15
                    ):
                        if msg and msg.text and len(msg.text) > 15:
                            title = extract_title_smart(msg.text) if UTILS_AVAILABLE else msg.text.split('\n')[0][:60]
                            
                            if title and query_lower in title.lower():
                                norm_title = normalize_title(title)
                                
                                if norm_title not in channel_posts:
                                    channel_posts[norm_title] = {
                                        'title': title,
                                        'content': format_post(msg.text) if UTILS_AVAILABLE else msg.text,
                                        'channel': channel_name(channel_id),
                                        'channel_id': channel_id,
                                        'message_id': msg.id,
                                        'date': msg.date.isoformat() if isinstance(msg.date, datetime) else str(msg.date),
                                        'is_new': is_new(msg.date) if UTILS_AVAILABLE and msg.date else False,
                                        'has_file': False,
                                        'has_post': True,
                                        'quality_options': {}
                                    }
                except Exception as e:
                    logger.error(f"❌ Search error in {channel_id}: {e}")
                return channel_posts
            
            tasks = [search_channel(cid) for cid in Config.TEXT_CHANNEL_IDS]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, dict):
                    posts_dict.update(result)
        
        # Merge results
        merged = {}
        
        for norm_title, post_data in posts_dict.items():
            merged[norm_title] = post_data
        
        for norm_title, file_data in files_dict.items():
            if norm_title in merged:
                merged[norm_title]['has_file'] = True
                merged[norm_title]['quality_options'] = file_data['quality_options']
                if file_data.get('thumbnail'):
                    merged[norm_title]['thumbnail'] = file_data['thumbnail']
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
                    'thumbnail': file_data.get('thumbnail')
                }
        
        # Sort and paginate
        results_list = list(merged.values())
        results_list.sort(key=lambda x: (not x.get('is_new', False), not x.get('has_file'), x.get('date', '')), reverse=True)
        
        total = len(results_list)
        paginated = results_list[offset:offset + limit]
        
        # Add posters
        for result in paginated:
            title = result.get('title', '')
            year_match = re.search(r'\b(19|20)\d{2}\b', title)
            year = year_match.group() if year_match else ""
            
            result['poster_url'] = Config.get_poster(title, year)
            result['poster_source'] = 'custom'
            result['has_poster'] = True
        
        return {
            'results': paginated,
            'pagination': {
                'current_page': page,
                'total_pages': math.ceil(total / limit) if total > 0 else 1,
                'total_results': total,
                'per_page': limit,
                'has_next': page < math.ceil(total / limit),
                'has_previous': page > 1
            },
            'search_metadata': {
                'channels_searched': len(Config.TEXT_CHANNEL_IDS),
                'query': query
            }
        }
    
    except Exception as e:
        logger.error(f"❌ Search error: {e}")
        raise

# ============================================================================
# API ENDPOINTS
# ============================================================================
@app.route('/')
@performance_monitor.measure("root_endpoint")
async def root():
    """Root endpoint"""
    total_files = await files_col.count_documents({}) if files_col else 0
    video_files = await files_col.count_documents({'is_video_file': True}) if files_col else 0
    
    return jsonify({
        'status': 'ok',
        'version': '8.0-MODULAR',
        'features': {
            'cache': CACHE_AVAILABLE,
            'verification': VERIFICATION_AVAILABLE,
            'premium': PREMIUM_AVAILABLE,
            'poster_fetching': POSTER_AVAILABLE,
            'bot_handlers': BOT_HANDLERS_AVAILABLE,
            'utils': UTILS_AVAILABLE
        },
        'stats': {
            'total_files': total_files,
            'video_files': video_files,
            'user_session_ready': user_session_ready,
            'bot_started': bot_started
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/health')
async def health():
    """Health check"""
    return jsonify({
        'status': 'ok' if bot_started or user_session_ready else 'starting',
        'telegram': {
            'user_session': {'ready': user_session_ready},
            'bot': {'started': bot_started}
        },
        'sync_management': {
            'monitoring': channel_sync_manager.is_monitoring,
            'file_channel': Config.FILE_CHANNEL_ID
        },
        'cache': cache_manager.redis_enabled if cache_manager else False,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/movies', methods=['GET'])
@performance_monitor.measure("movies_endpoint")
async def api_movies():
    """Get home movies"""
    try:
        movies = await get_home_movies_telegram(limit=30)
        
        return jsonify({
            'status': 'success' if movies else 'empty',
            'movies': movies,
            'total': len(movies),
            'source': 'telegram',
            'telegram_ready': user_session_ready,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"❌ Movies API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'movies': [],
            'total': 0
        }), 500

@app.route('/api/search', methods=['GET'])
@performance_monitor.measure("search_endpoint")
async def api_search():
    """Search movies"""
    try:
        query = request.args.get('query', '').strip()
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 12))
        
        if len(query) < 2:
            return jsonify({
                'status': 'error',
                'message': 'Query must be at least 2 characters'
            }), 400
        
        result_data = await search_movies_multi_channel(query, limit, page)
        
        return jsonify({
            'status': 'success',
            'query': query,
            'results': result_data['results'],
            'pagination': result_data['pagination'],
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"❌ Search API error: {e}")
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
            try:
                poster_data = await poster_fetcher.fetch_poster(title, year)
                if poster_data:
                    return jsonify({
                        'status': 'success',
                        'poster': {
                            'url': poster_data.get('url', ''),
                            'source': poster_data.get('source', 'custom'),
                            'rating': poster_data.get('rating', '0.0')
                        },
                        'title': title,
                        'year': year,
                        'timestamp': datetime.now().isoformat()
                    })
            except Exception as e:
                logger.debug(f"Poster fetcher error: {e}")
        
        return jsonify({
            'status': 'success',
            'poster': {
                'url': Config.get_poster(title, year),
                'source': 'custom',
                'rating': '0.0'
            },
            'title': title,
            'year': year,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"❌ Poster API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/stats', methods=['GET'])
async def api_stats():
    """Get system stats"""
    try:
        total_files = await files_col.count_documents({}) if files_col else 0
        video_files = await files_col.count_documents({'is_video_file': True}) if files_col else 0
        
        return jsonify({
            'status': 'success',
            'stats': {
                'total_files': total_files,
                'video_files': video_files,
                'sync_monitoring': channel_sync_manager.is_monitoring,
                'deleted_by_sync': channel_sync_manager.deleted_count,
                'performance': performance_monitor.get_stats()
            },
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"❌ Stats API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# ============================================================================
# MAIN ENTRY POINT - FIXED
# ============================================================================
if __name__ == '__main__':
    import uvicorn
    
    # Create event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    async def startup():
        await init_system()
    
    # Run startup
    try:
        loop.run_until_complete(startup())
    except KeyboardInterrupt:
        logger.info("👋 Shutting down...")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
    
    # Get port
    port = getattr(Config, 'WEB_SERVER_PORT', 8000)
    
    # Run uvicorn
    uvicorn.run(
        app,
        host='0.0.0.0',
        port=port,
        log_level='info',
        loop='asyncio'
    )
