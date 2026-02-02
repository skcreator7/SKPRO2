# ============================================================================
# 🚀 SK4FiLM v9.0 - WITH BOT HANDLER FIX
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

# ✅ IMPORT ALL MODULES WITH PROPER ERROR HANDLING
# ============================================================================

# First, set up logger before any imports
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

# Now import modules with try/except
try:
    from cache import CacheManager
    logger.debug("✅ Cache module imported")
except ImportError as e:
    logger.error(f"❌ Cache module import error: {e}")
    # Fallback CacheManager
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
    VerificationSystem = None

try:
    from premium import PremiumSystem, PremiumTier
    logger.debug("✅ Premium module imported")
except ImportError as e:
    logger.error(f"❌ Premium module import error: {e}")
    PremiumSystem = None
    PremiumTier = None

try:
    from poster_fetching import PosterFetcher, PosterSource
    logger.debug("✅ Poster fetching module imported")
except ImportError as e:
    logger.error(f"❌ Poster fetching module import error: {e}")
    PosterFetcher = None
    PosterSource = None

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
# ✅ BOT HANDLER MODULE - ADDED
# ============================================================================

class BotHandler:
    """Simple Bot Handler for file downloads"""
    
    def __init__(self, config, mongo_client, bot_client=None):
        self.config = config
        self.mongo_client = mongo_client
        self.bot = bot_client
        self.stats = {
            'download_requests': 0,
            'successful_downloads': 0,
            'failed_downloads': 0
        }
    
    async def handle_start_command(self, user_id, params=None):
        """Handle /start command with parameters"""
        try:
            self.stats['download_requests'] += 1
            
            if not params:
                return {
                    'status': 'info',
                    'message': 'Welcome to SK4FiLM Bot! Use the website to search movies.',
                    'help': 'Search at: https://sk4film.vercel.app'
                }
            
            # Parse parameters (format: channel_id_message_id_quality)
            parts = params.split('_')
            if len(parts) < 3:
                return {
                    'status': 'error',
                    'message': 'Invalid download link'
                }
            
            channel_id = int(parts[0])
            message_id = int(parts[1])
            quality = '_'.join(parts[2:]) if len(parts) > 2 else 'unknown'
            
            # Get file from database
            db = self.mongo_client.sk4film
            files_col = db.files
            
            file_doc = await files_col.find_one({
                'channel_id': channel_id,
                'message_id': message_id
            })
            
            if not file_doc:
                return {
                    'status': 'error',
                    'message': 'File not found in database'
                }
            
            # Return file info
            file_info = {
                'status': 'success',
                'title': file_doc.get('title', 'Unknown'),
                'quality': file_doc.get('quality', quality),
                'file_size': file_doc.get('file_size', 0),
                'file_id': file_doc.get('file_id'),
                'channel_id': channel_id,
                'message_id': message_id,
                'bot_username': self.config.BOT_USERNAME
            }
            
            self.stats['successful_downloads'] += 1
            return file_info
            
        except Exception as e:
            self.stats['failed_downloads'] += 1
            logger.error(f"Bot handler error: {e}")
            return {
                'status': 'error',
                'message': f'Download error: {str(e)}'
            }
    
    async def send_file_to_user(self, user_id, file_id, caption=None):
        """Send file to user via bot"""
        if not self.bot:
            return False
        
        try:
            await self.bot.send_document(
                chat_id=user_id,
                document=file_id,
                caption=caption or "File from SK4FiLM"
            )
            return True
        except Exception as e:
            logger.error(f"Send file error: {e}")
            return False
    
    def get_stats(self):
        return self.stats

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
    FILE_CHANNEL_ID = -1001768249569  # ✅ FILE CHANNEL
    
    # Links
    MAIN_CHANNEL_LINK = "https://t.me/sk4film"
    UPDATES_CHANNEL_LINK = "https://t.me/sk4film_Request"
    CHANNEL_USERNAME = "sk4film"
    
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
    MAX_CONCURRENT_REQUESTS = int(os.environ.get("MAX_CONCURRENT_REQUESTS", "10"))
    CACHE_TTL = int(os.environ.get("CACHE_TTL", "300"))
    REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "10"))
    
    # Sync Management Settings
    MONITOR_INTERVAL = int(os.environ.get("MONITOR_INTERVAL", "600"))
    
    # Quality Settings
    QUALITY_PRIORITY = ['2160p', '1080p', '720p', '480p', '360p']
    HEVC_VARIANTS = ['720p HEVC', '1080p HEVC', '2160p HEVC']
    
    # Fallback Poster
    FALLBACK_POSTER = "https://iili.io/fAeIwv9.th.png"
    
    # FILE CHANNEL INDEXING SETTINGS - OPTIMIZED
    AUTO_INDEX_INTERVAL = int(os.environ.get("AUTO_INDEX_INTERVAL", "120"))
    BATCH_INDEX_SIZE = int(os.environ.get("BATCH_INDEX_SIZE", "20"))
    
    # SEARCH SETTINGS
    SEARCH_MIN_QUERY_LENGTH = 2
    SEARCH_RESULTS_PER_PAGE = 12
    SEARCH_CACHE_TTL = 600

# ============================================================================
# ✅ FAST INITIALIZATION
# ============================================================================

app = Quart(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# CORS headers
@app.after_request
async def add_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['X-SK4FiLM-Version'] = '9.0-OPTIMIZED'
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
bot_handler = None  # ✅ Bot Handler initialized
telegram_bot = None

# ============================================================================
# ✅ PERFORMANCE MONITOR
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
    
    def get_stats(self):
        return self.measurements

performance_monitor = PerformanceMonitor()

# ============================================================================
# ✅ ASYNC CACHE DECORATOR
# ============================================================================

def async_cache_with_ttl(maxsize=128, ttl=300):
    """Async cache decorator with TTL"""
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
    """Enhanced quality detection with HEVC support"""
    if not filename:
        return "480p"
    
    filename_lower = filename.lower()
    
    # First check for HEVC variants
    is_hevc = any(re.search(pattern, filename_lower) for pattern in HEVC_PATTERNS)
    
    # Then check quality
    for pattern, quality in QUALITY_PATTERNS:
        if re.search(pattern, filename_lower):
            if is_hevc and quality in ['720p', '1080p', '2160p']:
                return f"{quality} HEVC"
            return quality
    
    return "480p"

# ============================================================================
# ✅ OPTIMIZED FILE INDEXING MANAGER
# ============================================================================

class OptimizedFileIndexingManager:
    """Optimized file channel indexing manager"""
    
    def __init__(self):
        self.is_running = False
        self.indexing_task = None
        self.last_run = None
        self.next_run = None
        self.total_indexed = 0
        self.indexing_lock = asyncio.Lock()
        
        self.indexing_stats = {
            'total_runs': 0,
            'total_indexed': 0,
            'total_errors': 0,
            'last_success': None
        }
    
    async def start_indexing(self):
        """Start file channel indexing"""
        if self.is_running:
            logger.warning("⚠️ File indexing already running")
            return
        
        logger.info("🚀 Starting OPTIMIZED FILE CHANNEL INDEXING...")
        self.is_running = True
        
        # Start periodic loop with delay
        await asyncio.sleep(30)
        self.indexing_task = asyncio.create_task(self._indexing_loop())
    
    async def stop_indexing(self):
        """Stop indexing"""
        self.is_running = False
        if self.indexing_task:
            self.indexing_task.cancel()
            try:
                await self.indexing_task
            except asyncio.CancelledError:
                pass
        
        logger.info("🛑 File indexing stopped")
    
    async def _indexing_loop(self):
        """Main indexing loop"""
        while self.is_running:
            try:
                # Wait for next run
                if self.next_run and self.next_run > datetime.now():
                    wait_seconds = (self.next_run - datetime.now()).total_seconds()
                    await asyncio.sleep(min(wait_seconds, 60))
                    continue
                
                # Run indexing cycle
                await self._run_indexing_cycle()
                
                # Schedule next run (2 minutes)
                self.next_run = datetime.now() + timedelta(seconds=120)
                self.last_run = datetime.now()
                
                # Sleep before checking again
                await asyncio.sleep(60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Indexing loop error: {e}")
                await asyncio.sleep(300)
    
    async def _run_indexing_cycle(self):
        """Run one indexing cycle"""
        try:
            if not user_session_ready or User is None:
                logger.warning("⚠️ User session not ready")
                return
            
            # Get last indexed message
            last_indexed = await files_col.find_one(
                {"channel_id": Config.FILE_CHANNEL_ID}, 
                sort=[('message_id', -1)],
                projection={'message_id': 1}
            )
            
            last_message_id = last_indexed['message_id'] if last_indexed else 0
            
            # Fetch new messages
            messages_to_index = []
            try:
                async for msg in User.get_chat_history(
                    Config.FILE_CHANNEL_ID, 
                    limit=20
                ):
                    if msg.id <= last_message_id:
                        break
                    
                    if msg and (msg.document or msg.video):
                        messages_to_index.append(msg)
                    
                    # Stop after 10 new files
                    if len(messages_to_index) >= 10:
                        break
            except Exception as e:
                logger.error(f"❌ Error fetching messages: {e}")
                return
            
            if not messages_to_index:
                logger.info("✅ No new files to index")
                return
            
            logger.info(f"📥 Found {len(messages_to_index)} new files")
            
            # Index files
            indexed_count = 0
            for msg in messages_to_index:
                try:
                    # Quick check if exists
                    existing = await files_col.find_one({
                        'channel_id': Config.FILE_CHANNEL_ID,
                        'message_id': msg.id
                    })
                    
                    if not existing:
                        success = await index_single_file_fast(msg)
                        if success:
                            indexed_count += 1
                            self.total_indexed += 1
                except Exception as e:
                    logger.debug(f"Index error: {e}")
            
            # Update stats
            self.indexing_stats['total_runs'] += 1
            self.indexing_stats['total_indexed'] += indexed_count
            self.indexing_stats['last_success'] = datetime.now()
            
            if indexed_count > 0:
                logger.info(f"✅ Indexed {indexed_count} new files")
            
        except Exception as e:
            logger.error(f"❌ Indexing cycle failed: {e}")
            self.indexing_stats['total_errors'] += 1
    
    async def get_indexing_status(self):
        """Get current indexing status"""
        return {
            'is_running': self.is_running,
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'next_run': self.next_run.isoformat() if self.next_run else None,
            'total_indexed': self.total_indexed,
            'stats': self.indexing_stats
        }

# Initialize file indexing manager
file_indexing_manager = OptimizedFileIndexingManager()

# ============================================================================
# ✅ OPTIMIZED FILE INDEXING FUNCTIONS
# ============================================================================

async def extract_title_fast(filename, caption):
    """Fast title extraction"""
    # Try filename first
    if filename:
        name = os.path.splitext(filename)[0]
        name = re.sub(r'[._]', ' ', name)
        name = re.sub(r'\b(480p|720p|1080p|2160p|4k|hd|hevc)\b', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\s+', ' ', name)
        name = name.strip()
        
        if name and len(name) > 3:
            return name[:100]
    
    # Try caption
    if caption:
        lines = caption.split('\n')
        for line in lines:
            line = line.strip()
            if len(line) > 10 and not line.startswith('http'):
                return line[:100]
    
    # Fallback
    if filename:
        return os.path.splitext(filename)[0][:50]
    
    return "Unknown File"

async def index_single_file_fast(message):
    """Fast file indexing"""
    try:
        if files_col is None:
            logger.error("❌ Files collection not initialized")
            return False
        
        if not message or (not message.document and not message.video):
            logger.debug("❌ Not a document or video message")
            return False
        
        # Check if already indexed
        existing = await files_col.find_one({
            'channel_id': Config.FILE_CHANNEL_ID,
            'message_id': message.id
        }, {'_id': 1})
        
        if existing:
            logger.debug(f"✅ Already indexed: {message.id}")
            return False
        
        # Extract basic info
        caption = message.caption if hasattr(message, 'caption') else None
        file_name = None
        
        if message.document:
            file_name = message.document.file_name
            file_size = message.document.file_size or 0
            file_id = message.document.file_id
        elif message.video:
            file_name = message.video.file_name or 'video.mp4'
            file_size = message.video.file_size or 0
            file_id = message.video.file_id
        else:
            logger.debug("❌ Not a file message")
            return False
        
        # Extract title
        title = await extract_title_fast(file_name, caption)
        if not title or title == "Unknown File":
            title = file_name or "Unknown"
        
        # Extract quality
        quality = detect_quality_enhanced(file_name or "")
        
        # Extract year from title
        year_match = re.search(r'\b(19|20)\d{2}\b', title)
        year = year_match.group() if year_match else ""
        
        # Get poster
        poster_url = Config.FALLBACK_POSTER
        poster_source = 'fallback'
        poster_rating = '0.0'
        
        # Create document
        doc = {
            'channel_id': Config.FILE_CHANNEL_ID,
            'message_id': message.id,
            'real_message_id': message.id,
            'title': title,
            'normalized_title': normalize_title(title),
            'date': message.date if hasattr(message, 'date') else datetime.now(),
            'indexed_at': datetime.now(),
            'last_checked': datetime.now(),
            'is_video_file': True,
            'file_id': file_id,
            'file_size': file_size,
            'file_name': file_name,
            'caption': caption or '',
            'quality': quality,
            'status': 'active',
            'thumbnail_extracted': False,
            'year': year,
            'thumbnail_url': poster_url,
            'poster_url': poster_url,
            'poster_source': poster_source,
            'poster_rating': poster_rating,
            'searchable': True
        }
        
        # Insert into database
        result = await files_col.insert_one(doc)
        
        logger.info(f"✅ Indexed: {title[:50]}... (ID: {message.id})")
        
        return True
        
    except Exception as e:
        if "duplicate key error" in str(e).lower():
            return False
        logger.error(f"❌ Indexing error: {e}")
        return False

async def setup_database_indexes():
    """Setup database indexes"""
    if files_col is None:
        logger.error("❌ Files collection not initialized")
        return
    
    try:
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
            [("searchable", 1)],
            name="searchable_index",
            background=True
        )
        
        logger.info("✅ Created database indexes")
        
    except Exception as e:
        logger.warning(f"⚠️ Index creation error: {e}")

# ============================================================================
# ✅ POSTER FETCHING FUNCTIONS
# ============================================================================

async def get_poster_for_movie(title: str, year: str = "", quality: str = "") -> Dict[str, Any]:
    """Get poster for movie"""
    global poster_fetcher
    
    if poster_fetcher is None:
        return {
            'poster_url': Config.FALLBACK_POSTER,
            'source': 'fallback',
            'rating': '0.0',
            'year': year,
            'title': title,
            'quality': quality or 'unknown'
        }
    
    try:
        poster_task = asyncio.create_task(poster_fetcher.fetch_poster(title, year, quality))
        
        try:
            poster_data = await asyncio.wait_for(poster_task, timeout=3.0)
            
            if poster_data and poster_data.get('poster_url'):
                return poster_data
            else:
                raise ValueError("Invalid poster data")
                
        except (asyncio.TimeoutError, ValueError, Exception):
            if not poster_task.done():
                poster_task.cancel()
            
            return {
                'poster_url': Config.FALLBACK_POSTER,
                'source': 'fallback',
                'rating': '0.0',
                'year': year,
                'title': title,
                'quality': quality or 'unknown'
            }
            
    except Exception as e:
        logger.error(f"❌ Error in get_poster_for_movie: {e}")
        return {
            'poster_url': Config.FALLBACK_POSTER,
            'source': 'fallback',
            'rating': '0.0',
            'year': year,
            'title': title,
            'quality': quality or 'unknown'
        }

# ============================================================================
# ✅ TELEGRAM SESSION INITIALIZATION
# ============================================================================

async def init_telegram_sessions():
    """Initialize Telegram sessions"""
    global User, Bot, user_session_ready, bot_session_ready, bot_handler
    
    logger.info("=" * 50)
    logger.info("🚀 TELEGRAM SESSION INITIALIZATION")
    logger.info("=" * 50)
    
    if not PYROGRAM_AVAILABLE:
        logger.error("❌ Pyrogram not installed!")
        return False
    
    # Initialize USER Session
    if Config.API_ID > 0 and Config.API_HASH and Config.USER_SESSION_STRING:
        logger.info("\n👤 Initializing USER Session...")
        try:
            User = Client(
                "sk4film_user",
                api_id=Config.API_ID,
                api_hash=Config.API_HASH,
                session_string=Config.USER_SESSION_STRING,
                sleep_threshold=120,
                in_memory=True,
                no_updates=True
            )
            
            await User.start()
            me = await User.get_me()
            logger.info(f"✅ USER Session Ready: {me.first_name}")
            user_session_ready = True
                
        except Exception as e:
            logger.error(f"❌ USER Session failed: {e}")
            user_session_ready = False
            User = None
    
    # Initialize BOT Session
    if Config.BOT_TOKEN:
        logger.info("\n🤖 Initializing BOT Session...")
        try:
            Bot = Client(
                "sk4film_bot",
                api_id=Config.API_ID,
                api_hash=Config.API_HASH,
                bot_token=Config.BOT_TOKEN,
                sleep_threshold=120,
                in_memory=True,
                no_updates=True
            )
            
            await Bot.start()
            bot_info = await Bot.get_me()
            logger.info(f"✅ BOT Session Ready: @{bot_info.username}")
            bot_session_ready = True
            
            # Initialize Bot Handler
            bot_handler = BotHandler(Config, mongo_client, Bot)
            logger.info("✅ Bot Handler initialized")
                
        except Exception as e:
            logger.error(f"❌ BOT Session failed: {e}")
            bot_session_ready = False
            Bot = None
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 SESSION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"USER Session: {'✅ READY' if user_session_ready else '❌ NOT READY'}")
    logger.info(f"BOT Session: {'✅ READY' if bot_session_ready else '❌ NOT READY'}")
    
    return user_session_ready or bot_session_ready

# ============================================================================
# ✅ MONGODB INITIALIZATION
# ============================================================================

async def init_mongodb():
    global mongo_client, db, files_col, verification_col
    
    try:
        logger.info("🔌 MongoDB initialization...")
        
        mongo_client = AsyncIOMotorClient(
            Config.MONGODB_URI,
            serverSelectionTimeoutMS=10000,
            connectTimeoutMS=10000,
            socketTimeoutMS=15000,
            maxPoolSize=20,
            minPoolSize=5
        )
        
        await asyncio.wait_for(mongo_client.admin.command('ping'), timeout=5)
        
        db = mongo_client.sk4film
        files_col = db.files
        verification_col = db.verifications
        
        # Setup indexes
        await setup_database_indexes()
        
        logger.info("✅ MongoDB OK")
        return True
        
    except asyncio.TimeoutError:
        logger.error("❌ MongoDB connection timeout")
        return False
    except Exception as e:
        logger.error(f"❌ MongoDB error: {e}")
        return False

# ============================================================================
# ✅ BULK INDEXING
# ============================================================================

async def bulk_fetch_recent_files():
    """Bulk fetch recent files from channel"""
    try:
        logger.info("📥 Bulk fetching recent files...")
        
        if not user_session_ready or User is None:
            logger.error("❌ User session not ready for bulk fetch")
            return
        
        indexed_count = 0
        batch_size = 30
        
        async for msg in User.get_chat_history(Config.FILE_CHANNEL_ID, limit=100):
            if msg and (msg.document or msg.video):
                success = await index_single_file_fast(msg)
                if success:
                    indexed_count += 1
                
                # Small delay to avoid flood
                if indexed_count % batch_size == 0:
                    await asyncio.sleep(1)
        
        logger.info(f"✅ Bulk indexed {indexed_count} files")
        
        # Start background indexing
        await file_indexing_manager.start_indexing()
        
    except Exception as e:
        logger.error(f"❌ Bulk fetch error: {e}")

# ============================================================================
# ✅ SEARCH FUNCTION
# ============================================================================

@performance_monitor.measure("enhanced_search")
@async_cache_with_ttl(maxsize=500, ttl=Config.SEARCH_CACHE_TTL)
async def search_movies_enhanced_fixed(query, limit=15, page=1):
    """Search endpoint"""
    offset = (page - 1) * limit
    
    # Cache check
    cache_key = f"search:{query}:{page}:{limit}"
    if cache_manager is not None and cache_manager.redis_enabled:
        cached = await cache_manager.get(cache_key)
        if cached:
            logger.debug(f"✅ Cache hit for: {query}")
            return cached
    
    logger.info(f"🔍 Searching: {query}")
    
    results = []
    
    # Search database
    if files_col is not None:
        try:
            search_query = {
                "$or": [
                    {"title": {"$regex": query, "$options": "i"}},
                    {"normalized_title": {"$regex": query, "$options": "i"}},
                    {"caption": {"$regex": query, "$options": "i"}},
                ],
                "status": "active",
                "searchable": True
            }
            
            cursor = files_col.find(
                search_query,
                {
                    '_id': 0,
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
                    'thumbnail_url': 1,
                    'poster_url': 1,
                    'poster_source': 1,
                    'poster_rating': 1,
                    'year': 1,
                    'searchable': 1
                }
            ).sort("date", -1).limit(100).skip(offset)
            
            file_count = 0
            async for doc in cursor:
                try:
                    file_count += 1
                    
                    title = doc.get('title', 'Unknown')
                    quality = doc.get('quality', '480p')
                    year = doc.get('year', '')
                    message_id = doc.get('real_message_id') or doc.get('message_id')
                    
                    # Get thumbnail
                    thumbnail_url = doc.get('thumbnail_url') or doc.get('poster_url') or Config.FALLBACK_POSTER
                    
                    # Create quality options
                    quality_options = {
                        quality: {
                            'quality': quality,
                            'file_size': doc.get('file_size', 0),
                            'message_id': message_id,
                            'real_message_id': message_id,
                            'is_video_file': True,
                            'file_id': doc.get('file_id')
                        }
                    }
                    
                    # Create result
                    result = {
                        'title': title,
                        'normalized_title': doc.get('normalized_title', normalize_title(title)),
                        'content': format_post(doc.get('caption', ''), max_length=300),
                        'post_content': doc.get('caption', ''),
                        'quality_options': quality_options,
                        'date': doc['date'].isoformat() if isinstance(doc['date'], datetime) else doc['date'],
                        'is_new': is_new(doc['date']) if doc.get('date') else False,
                        'is_video_file': True,
                        'channel_id': doc.get('channel_id'),
                        'has_file': True,
                        'has_post': bool(doc.get('caption')),
                        'year': year,
                        'quality': quality,
                        'has_thumbnail': True,
                        'thumbnail': thumbnail_url,
                        'thumbnail_url': thumbnail_url,
                        'poster_url': thumbnail_url,
                        'poster_source': doc.get('poster_source', 'fallback'),
                        'poster_rating': doc.get('poster_rating', '0.0'),
                        'real_message_id': message_id,
                        'message_id': message_id,
                        'result_type': 'file_only',
                        'quality_count': 1,
                        'has_poster': True,
                        'bot_username': Config.BOT_USERNAME,
                        'file_channel_id': Config.FILE_CHANNEL_ID
                    }
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Result processing error: {e}")
                    continue
            
            logger.info(f"✅ Found {file_count} files for query: {query}")
            
        except Exception as e:
            logger.error(f"Database search error: {e}")
    
    # Sort results
    results.sort(key=lambda x: x.get('date', ''), reverse=True)
    
    total = len(results)
    paginated = results[offset:offset + limit]
    
    # Final data
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
            'total': total
        },
        'bot_username': Config.BOT_USERNAME
    }
    
    # Cache results
    if cache_manager is not None:
        await cache_manager.set(cache_key, result_data, expire_seconds=300)
    
    logger.info(f"✅ Search complete: {len(paginated)} results")
    
    return result_data

# ============================================================================
# ✅ HOME MOVIES FUNCTION
# ============================================================================

@performance_monitor.measure("home_movies")
@async_cache_with_ttl(maxsize=1, ttl=60)
async def get_home_movies(limit=25):
    """Get home movies"""
    try:
        if files_col is None:
            logger.error("❌ Files collection not initialized")
            return []
        
        # Get latest files from database
        cursor = files_col.find(
            {"status": "active", "searchable": True},
            {
                '_id': 0,
                'title': 1,
                'normalized_title': 1,
                'quality': 1,
                'file_size': 1,
                'channel_id': 1,
                'message_id': 1,
                'real_message_id': 1,
                'date': 1,
                'caption': 1,
                'thumbnail_url': 1,
                'poster_url': 1,
                'poster_source': 1,
                'year': 1
            }
        ).sort("date", -1).limit(limit)
        
        movies = []
        async for doc in cursor:
            title = doc.get('title', 'Unknown')
            quality = doc.get('quality', '480p')
            year = doc.get('year', '')
            message_id = doc.get('real_message_id') or doc.get('message_id')
            
            # Get thumbnail
            thumbnail_url = doc.get('thumbnail_url') or doc.get('poster_url') or Config.FALLBACK_POSTER
            
            # Create quality options
            quality_options = {
                quality: {
                    'quality': quality,
                    'file_size': doc.get('file_size', 0),
                    'message_id': message_id,
                    'real_message_id': message_id,
                    'is_video_file': True
                }
            }
            
            movie = {
                'title': title,
                'normalized_title': doc.get('normalized_title', normalize_title(title)),
                'content': format_post(doc.get('caption', ''), max_length=300),
                'post_content': doc.get('caption', ''),
                'quality_options': quality_options,
                'date': doc['date'].isoformat() if isinstance(doc['date'], datetime) else doc['date'],
                'is_new': is_new(doc['date']) if doc.get('date') else False,
                'is_video_file': True,
                'channel_id': doc.get('channel_id'),
                'has_file': True,
                'has_post': bool(doc.get('caption')),
                'year': year,
                'quality': quality,
                'has_thumbnail': True,
                'thumbnail': thumbnail_url,
                'thumbnail_url': thumbnail_url,
                'poster_url': thumbnail_url,
                'poster_source': doc.get('poster_source', 'fallback'),
                'real_message_id': message_id,
                'message_id': message_id,
                'result_type': 'file_only',
                'quality_count': 1,
                'has_poster': True,
                'bot_username': Config.BOT_USERNAME,
                'file_channel_id': Config.FILE_CHANNEL_ID
            }
            
            movies.append(movie)
        
        logger.info(f"✅ Fetched {len(movies)} home movies")
        return movies[:limit]
        
    except Exception as e:
        logger.error(f"❌ Home movies error: {e}")
        return []

# ============================================================================
# ✅ SYSTEM INITIALIZATION
# ============================================================================

async def init_system():
    start_time = time.time()
    
    try:
        logger.info("=" * 60)
        logger.info("🚀 SK4FiLM v9.0 - WITH BOT HANDLER")
        logger.info("=" * 60)
        
        # Initialize MongoDB
        mongo_ok = await init_mongodb()
        if not mongo_ok:
            logger.error("❌ MongoDB connection failed")
            return False
        
        # Get file count
        if files_col is not None:
            try:
                file_count = await files_col.count_documents({})
                logger.info(f"📊 Files in database: {file_count}")
            except:
                pass
        
        # Initialize components
        global cache_manager, poster_fetcher, bot_handler
        
        # Initialize Cache
        try:
            cache_manager = CacheManager(Config)
            redis_ok = await cache_manager.init_redis()
            if redis_ok:
                logger.info("✅ Cache Manager initialized")
        except:
            logger.warning("⚠️ Cache init failed")
            cache_manager = None
        
        # Initialize PosterFetcher
        if PosterFetcher is not None:
            poster_fetcher = PosterFetcher(Config, cache_manager)
            logger.info("✅ Poster Fetcher initialized")
        else:
            logger.info("⚠️ Using fallback poster fetcher")
            poster_fetcher = None
        
        # Initialize Telegram sessions
        if PYROGRAM_AVAILABLE:
            await asyncio.sleep(2)
            telegram_ok = await init_telegram_sessions()
            if not telegram_ok:
                logger.warning("⚠️ Telegram sessions failed")
        
        # Initialize Bot Handler (even without bot session)
        if mongo_client is not None:
            bot_handler = BotHandler(Config, mongo_client, Bot)
            logger.info("✅ Bot Handler initialized")
        
        # Start bulk indexing if few files
        if user_session_ready and files_col is not None:
            file_count = await files_col.count_documents({})
            if file_count < 50:
                logger.info(f"🔄 Starting bulk indexing (only {file_count} files)...")
                asyncio.create_task(bulk_fetch_recent_files())
            else:
                await file_indexing_manager.start_indexing()
        else:
            logger.warning("⚠️ Cannot start indexing: Telegram or DB not ready")
        
        init_time = time.time() - start_time
        logger.info(f"⚡ System started in {init_time:.2f}s")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ System initialization failed: {e}")
        return False

# ============================================================================
# ✅ API ROUTES
# ============================================================================

@app.route('/')
async def root():
    """Root endpoint"""
    try:
        total_files = 0
        if files_col is not None:
            try:
                total_files = await files_col.count_documents({})
            except:
                pass
        
        return jsonify({
            'status': 'healthy',
            'service': 'SK4FiLM v9.0',
            'stats': {
                'total_files': total_files
            },
            'sessions': {
                'user': user_session_ready,
                'bot': bot_session_ready
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Root endpoint error: {e}")
        return jsonify({'status': 'error'}), 500

@app.route('/health')
async def health():
    """Health check"""
    try:
        return jsonify({
            'status': 'ok',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'status': 'error'}), 500

@app.route('/api/movies', methods=['GET'])
async def api_movies():
    """Get home movies"""
    try:
        movies = await get_home_movies(limit=25)
        
        return jsonify({
            'status': 'success' if movies else 'empty',
            'movies': movies,
            'total': len(movies),
            'limit': 25,
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
async def api_search():
    """Search API endpoint"""
    try:
        query = request.args.get('query', '').strip()
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', Config.SEARCH_RESULTS_PER_PAGE))
        
        if len(query) < Config.SEARCH_MIN_QUERY_LENGTH:
            return jsonify({
                'status': 'error',
                'message': f'Query must be at least {Config.SEARCH_MIN_QUERY_LENGTH} characters'
            }), 400
        
        result_data = await search_movies_enhanced_fixed(query, limit, page)
        
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
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/stats', methods=['GET'])
async def api_stats():
    """Get statistics"""
    try:
        # Get database stats
        if files_col is not None:
            total_files = await files_col.count_documents({})
            searchable_files = await files_col.count_documents({'searchable': True})
        else:
            total_files = 0
            searchable_files = 0
        
        # Get indexing stats
        indexing_status = await file_indexing_manager.get_indexing_status()
        
        # Get bot handler stats
        bot_stats = bot_handler.get_stats() if bot_handler else {}
        
        return jsonify({
            'status': 'success',
            'database_stats': {
                'total_files': total_files,
                'searchable_files': searchable_files,
            },
            'indexing_stats': indexing_status,
            'bot_stats': bot_stats,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Stats API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/admin/reindex', methods=['POST'])
async def api_admin_reindex():
    """Trigger reindexing"""
    try:
        auth_token = request.headers.get('X-Admin-Token')
        if not auth_token or auth_token != os.environ.get('ADMIN_TOKEN', 'sk4film_admin_123'):
            return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401
        
        # Trigger bulk indexing
        asyncio.create_task(bulk_fetch_recent_files())
        
        return jsonify({
            'status': 'success',
            'message': 'File channel reindexing started',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Admin reindex error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/admin/check-db', methods=['GET'])
async def api_admin_check_db():
    """Check database for files"""
    try:
        if files_col is None:
            return jsonify({
                'status': 'error',
                'message': 'Database not connected'
            }), 500
        
        # Get sample of files
        sample_files = []
        cursor = files_col.find({}, {
            'title': 1,
            'message_id': 1,
            'quality': 1,
            'date': 1,
            'searchable': 1,
            '_id': 0
        }).limit(10)
        
        async for doc in cursor:
            sample_files.append(doc)
        
        total_files = await files_col.count_documents({})
        searchable_files = await files_col.count_documents({'searchable': True})
        
        return jsonify({
            'status': 'success',
            'total_files': total_files,
            'searchable_files': searchable_files,
            'sample_files': sample_files,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Check DB error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# ============================================================================
# ✅ BOT API ROUTES
# ============================================================================

@app.route('/api/bot/start', methods=['POST'])
async def api_bot_start():
    """Bot start command handler"""
    try:
        data = await request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': 'No data provided'}), 400
        
        user_id = data.get('user_id')
        params = data.get('params')
        
        if not user_id:
            return jsonify({'status': 'error', 'message': 'User ID required'}), 400
        
        if bot_handler is None:
            return jsonify({'status': 'error', 'message': 'Bot handler not initialized'}), 500
        
        result = await bot_handler.handle_start_command(user_id, params)
        
        return jsonify({
            'status': 'success',
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Bot start API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/bot/send-file', methods=['POST'])
async def api_bot_send_file():
    """Send file via bot"""
    try:
        data = await request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': 'No data provided'}), 400
        
        user_id = data.get('user_id')
        file_id = data.get('file_id')
        caption = data.get('caption')
        
        if not user_id or not file_id:
            return jsonify({'status': 'error', 'message': 'User ID and File ID required'}), 400
        
        if bot_handler is None:
            return jsonify({'status': 'error', 'message': 'Bot handler not initialized'}), 500
        
        success = await bot_handler.send_file_to_user(user_id, file_id, caption)
        
        return jsonify({
            'status': 'success' if success else 'error',
            'sent': success,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Bot send file error: {e}")
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
    
    # Stop indexing
    await file_indexing_manager.stop_indexing()
    
    # Close Telegram sessions
    if User is not None:
        try:
            await User.stop()
        except:
            pass
    
    if Bot is not None:
        try:
            await Bot.stop()
        except:
            pass
    
    # Close cache manager
    if cache_manager is not None:
        try:
            await cache_manager.stop()
        except:
            pass
    
    # Close MongoDB
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
    
    logger.info(f"🌐 Starting SK4FiLM with Bot Handler on port {Config.WEB_SERVER_PORT}...")
    
    try:
        asyncio.run(serve(app, config))
    except KeyboardInterrupt:
        logger.info("👋 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
