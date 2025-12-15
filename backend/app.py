# ============================================================================
# 🚀 SK4FiLM v8.2 - COMPLETE INTEGRATED SYSTEM
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

# ✅ IMPORT ALL MODULES
try:
    from cache import CacheManager
    from verification import VerificationSystem
    from premium import PremiumSystem, PremiumTier
    from poster_fetching import PosterFetcher, PosterSource
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
    MODULES_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ Module import error: {e}")
    MODULES_AVAILABLE = False

# ✅ LOGGING SETUP
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
        
        # Log slow operations
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
    
    # Channel Configuration - DUAL SESSION
    MAIN_CHANNEL_ID = -1001891090100          # ✅ User Session
    TEXT_CHANNEL_IDS = [-1001891090100, -1002024811395]  # ✅ User Session
    FILE_CHANNEL_ID = -1001768249569          # ✅ Bot Session
    
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
    
    # Fallback Poster
    FALLBACK_POSTER = "https://iili.io/fAeIwv9.th.png"

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
    response.headers['X-SK4FiLM-Version'] = '8.2-COMPLETE-INTEGRATED'
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
    
    User = None        # ✅ For TEXT channel searches
    Bot = None         # ✅ For FILE channel operations
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
# ✅ QUALITY DETECTION ENHANCED
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

def extract_quality_info(filename):
    """Extract detailed quality info"""
    quality = detect_quality_enhanced(filename)
    
    # Parse quality components
    base_quality = quality
    is_hevc = 'HEVC' in quality
    
    if is_hevc:
        base_quality = quality.replace(' HEVC', '')
    
    return {
        'full': quality,
        'base': base_quality,
        'is_hevc': is_hevc,
        'priority': Config.QUALITY_PRIORITY.index(base_quality) if base_quality in Config.QUALITY_PRIORITY else 999
    }

# ============================================================================
# ✅ QUALITY MERGER
# ============================================================================

class QualityMerger:
    """Merge multiple qualities for same title"""
    
    @staticmethod
    def merge_quality_options(quality_options_dict):
        """Merge quality options from multiple sources"""
        if not quality_options_dict:
            return {}
        
        merged = {}
        
        # Sort by priority
        for quality, option in quality_options_dict.items():
            base_quality = quality.replace(' HEVC', '')
            
            if base_quality not in merged:
                merged[base_quality] = {
                    'qualities': [],
                    'best_option': None,
                    'total_size': 0,
                    'file_count': 0
                }
            
            # Add quality variant
            merged[base_quality]['qualities'].append({
                'full_quality': quality,
                'is_hevc': 'HEVC' in quality,
                'file_id': option.get('file_id'),
                'file_size': option.get('file_size', 0),
                'file_name': option.get('file_name', ''),
                'is_video': option.get('is_video', False),
                'channel_id': option.get('channel_id'),
                'message_id': option.get('message_id')
            })
            
            merged[base_quality]['total_size'] += option.get('file_size', 0)
            merged[base_quality]['file_count'] += 1
            
            # Set best option (highest quality, smallest size)
            if merged[base_quality]['best_option'] is None:
                merged[base_quality]['best_option'] = quality
            else:
                current_priority = Config.QUALITY_PRIORITY.index(base_quality) if base_quality in Config.QUALITY_PRIORITY else 999
                best_base = merged[base_quality]['best_option'].replace(' HEVC', '')
                best_priority = Config.QUALITY_PRIORITY.index(best_base) if best_base in Config.QUALITY_PRIORITY else 999
                
                if current_priority < best_priority:
                    merged[base_quality]['best_option'] = quality
        
        # Sort by quality priority
        sorted_merged = {}
        for quality in Config.QUALITY_PRIORITY:
            if quality in merged:
                sorted_merged[quality] = merged[quality]
        
        # Add any remaining qualities
        for quality in merged:
            if quality not in sorted_merged:
                sorted_merged[quality] = merged[quality]
        
        return sorted_merged
    
    @staticmethod
    def get_quality_summary(merged_options):
        """Get summary of available qualities"""
        if not merged_options:
            return "No files"
        
        qualities = list(merged_options.keys())
        
        # Sort by priority
        sorted_qualities = []
        for quality in Config.QUALITY_PRIORITY:
            if quality in qualities:
                sorted_qualities.append(quality)
                qualities.remove(quality)
        
        # Add remaining qualities
        sorted_qualities.extend(sorted(qualities))
        
        # Create summary
        summary_parts = []
        for quality in sorted_qualities[:3]:  # Show top 3 qualities
            data = merged_options[quality]
            count = data['file_count']
            if count > 1:
                summary_parts.append(f"{quality} ({count} files)")
            else:
                summary_parts.append(quality)
        
        if len(sorted_qualities) > 3:
            summary_parts.append(f"+{len(sorted_qualities) - 3} more")
        
        return " • ".join(summary_parts)

# ============================================================================
# ✅ THUMBNAIL EXTRACTOR
# ============================================================================

class FileThumbnailExtractor:
    """Extract thumbnails from Telegram video files"""
    
    @staticmethod
    def generate_video_thumbnail(title, quality):
        """Generate video-specific thumbnail"""
        # Create a video-themed thumbnail
        clean_title = title[:30]
        encoded_title = urllib.parse.quote(clean_title)
        
        # Different colors for different qualities
        quality_colors = {
            '2160p': '4a148c',  # Purple
            '1080p': '1565c0',  # Blue
            '720p': '0277bd',   # Light Blue
            '480p': '00838f',   # Teal
            '360p': '00695c',   # Dark Teal
        }
        
        color = quality_colors.get(quality.split()[0], '1a237e')  # Default dark blue
        
        # Add quality badge
        quality_badge = f"&badge={urllib.parse.quote(quality)}&badgeColor=ff4081"
        
        # Add video icon
        video_icon = "&logo=https://img.icons8.com/color/96/000000/video.png"
        
        thumbnail_url = f"https://via.placeholder.com/300x450/{color}/ffffff?text={encoded_title}{quality_badge}{video_icon}"
        
        return thumbnail_url
    
    @staticmethod
    def get_file_type_icon(file_name):
        """Get icon based on file type"""
        if not file_name:
            return "📁"
        
        file_ext = os.path.splitext(file_name)[1].lower()
        
        icons = {
            '.mp4': '🎬',
            '.mkv': '🎥',
            '.avi': '📽️',
            '.mov': '📹',
            '.wmv': '📺',
            '.flv': '📼',
            '.webm': '🌐',
            '.m4v': '📱',
            '.3gp': '📲',
            '.mp3': '🎵',
            '.wav': '🎶',
            '.zip': '📦',
            '.rar': '🗜️',
            '.srt': '📝',
            '.ass': '✏️',
        }
        
        return icons.get(file_ext, '📁')

# ============================================================================
# ✅ SYNC MANAGEMENT
# ============================================================================

class ChannelSyncManager:
    def __init__(self):
        self.is_monitoring = False
        self.monitoring_task = None
        self.deleted_count = 0
        self.last_sync = time.time()
    
    async def start_sync_monitoring(self):
        """Start sync monitoring using BOT session"""
        if Bot is None or not bot_session_ready:
            logger.warning("⚠️ Bot session not ready for sync")
            return
        
        if self.is_monitoring:
            return
        
        logger.info("👁️ Starting sync monitoring via BOT...")
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self.monitor_channel_sync())
    
    async def stop_sync_monitoring(self):
        self.is_monitoring = False
        if self.monitoring_task is not None:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except:
                pass
        logger.info("🛑 Sync monitoring stopped")
    
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
        """Sync deletions using BOT session"""
        try:
            if files_col is None or Bot is None:
                return
            
            current_time = time.time()
            if current_time - self.last_sync < 300:
                return
            
            self.last_sync = current_time
            
            # Get message IDs from MongoDB
            cursor = files_col.find(
                {"channel_id": Config.FILE_CHANNEL_ID},
                {"message_id": 1, "_id": 0}
            )
            
            message_ids_in_db = []
            async for doc in cursor:
                msg_id = doc.get('message_id')
                if msg_id:
                    message_ids_in_db.append(msg_id)
            
            if not message_ids_in_db:
                return
            
            deleted_count = 0
            batch_size = 50
            
            for i in range(0, len(message_ids_in_db), batch_size):
                batch = message_ids_in_db[i:i + batch_size]
                
                try:
                    # Check if messages exist using BOT
                    messages = await Bot.get_messages(Config.FILE_CHANNEL_ID, batch)
                    
                    existing_ids = set()
                    if isinstance(messages, list):
                        for msg in messages:
                            if msg and hasattr(msg, 'id'):
                                existing_ids.add(msg.id)
                    elif messages is not None and hasattr(messages, 'id'):
                        existing_ids.add(messages.id)
                    
                    # Find deleted IDs
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
                
                except Exception as e:
                    logger.error(f"❌ Batch check error: {e}")
                    continue
            
            if deleted_count > 0:
                logger.info(f"✅ Sync: {deleted_count} files deleted")
            
        except Exception as e:
            logger.error(f"❌ Sync deletions error: {e}")
    
    async def manual_sync(self):
        await self.sync_deletions_from_telegram()

channel_sync_manager = ChannelSyncManager()

# ============================================================================
# ✅ POSTER FETCHING FUNCTIONS
# ============================================================================

async def get_poster_for_movie(title: str, year: str = "", quality: str = "") -> Dict[str, Any]:
    """
    Get poster for movie using PosterFetcher
    """
    global poster_fetcher
    
    # If poster_fetcher is not available, create fallback immediately
    if poster_fetcher is None:
        return {
            'poster_url': Config.FALLBACK_POSTER,
            'source': PosterSource.CUSTOM.value,
            'rating': '0.0',
            'year': year,
            'title': title,
            'quality': quality or 'unknown'
        }
    
    try:
        # Fetch poster with timeout protection
        poster_task = asyncio.create_task(poster_fetcher.fetch_poster(title))
        
        try:
            # Wait for poster with timeout
            poster_data = await asyncio.wait_for(poster_task, timeout=3.0)
            
            # Ensure we have valid data
            if poster_data and poster_data.get('poster_url'):
                logger.debug(f"✅ Poster fetched: {title} - {poster_data['source']}")
                return poster_data
            else:
                raise ValueError("Invalid poster data")
                
        except (asyncio.TimeoutError, ValueError, Exception) as e:
            logger.warning(f"⚠️ Poster fetch timeout/error for {title}: {e}")
            
            # Cancel the task if it's still running
            if not poster_task.done():
                poster_task.cancel()
            
            # Return fallback
            return {
                'poster_url': Config.FALLBACK_POSTER,
                'source': PosterSource.CUSTOM.value,
                'rating': '0.0',
                'year': year,
                'title': title,
                'quality': quality or 'unknown'
            }
            
    except Exception as e:
        logger.error(f"❌ Unexpected error in get_poster_for_movie: {e}")
        # Always return fallback
        return {
            'poster_url': Config.FALLBACK_POSTER,
            'source': PosterSource.CUSTOM.value,
            'rating': '0.0',
            'year': year,
            'title': title,
            'quality': quality or 'unknown'
        }

async def get_posters_for_movies_batch(movies: List[Dict]) -> List[Dict]:
    """
    Get posters for multiple movies in batch
    """
    results = []
    
    # Create tasks for all movies
    tasks = []
    for movie in movies:
        title = movie.get('title', '')
        year = movie.get('year', '')
        quality = movie.get('quality', '')
        
        task = asyncio.create_task(get_poster_for_movie(title, year, quality))
        tasks.append((movie, task))
    
    # Process results as they complete
    for movie, task in tasks:
        try:
            poster_data = await task
            
            # Update movie with poster data
            movie_with_poster = movie.copy()
            movie_with_poster.update({
                'poster_url': poster_data['poster_url'],
                'poster_source': poster_data['source'],
                'poster_rating': poster_data['rating'],
                'thumbnail': poster_data['poster_url'],
                'thumbnail_source': poster_data['source'],
                'has_poster': True
            })
            
            results.append(movie_with_poster)
            
        except Exception as e:
            logger.warning(f"⚠️ Batch poster error for {movie.get('title')}: {e}")
            
            # Add movie with fallback
            movie_with_fallback = movie.copy()
            movie_with_fallback.update({
                'poster_url': Config.FALLBACK_POSTER,
                'poster_source': PosterSource.CUSTOM.value,
                'poster_rating': '0.0',
                'thumbnail': Config.FALLBACK_POSTER,
                'thumbnail_source': 'fallback',
                'has_poster': True
            })
            
            results.append(movie_with_fallback)
    
    return results

# ============================================================================
# ✅ DUAL SESSION INITIALIZATION
# ============================================================================

@performance_monitor.measure("telegram_init")
async def init_telegram_sessions():
    """Initialize DUAL sessions: User for text, Bot for files"""
    global User, Bot, user_session_ready, bot_session_ready
    
    logger.info("=" * 60)
    logger.info("🚀 DUAL SESSION INITIALIZATION")
    logger.info("=" * 60)
    
    if not PYROGRAM_AVAILABLE:
        logger.error("❌ Pyrogram not installed!")
        return False
    
    # ============================================================================
    # ✅ 1. INITIALIZE USER SESSION (for TEXT channels)
    # ============================================================================
    if Config.API_ID > 0 and Config.API_HASH and Config.USER_SESSION_STRING:
        logger.info("\n👤 Initializing USER Session for TEXT channels...")
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
            
            # Test TEXT channel access
            try:
                chat = await User.get_chat(Config.MAIN_CHANNEL_ID)
                logger.info(f"✅ TEXT Channel Access: {chat.title}")
                user_session_ready = True
            except Exception as e:
                logger.error(f"❌ TEXT Channel access failed: {e}")
                user_session_ready = False
                
        except Exception as e:
            logger.error(f"❌ USER Session failed: {e}")
            user_session_ready = False
            if User is not None:
                try:
                    await User.stop()
                except:
                    pass
            User = None
    
    # ============================================================================
    # ✅ 2. INITIALIZE BOT SESSION (for FILE channel)
    # ============================================================================
    if Config.BOT_TOKEN:
        logger.info("\n🤖 Initializing BOT Session for FILE channel...")
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
            
            # Test FILE channel access
            try:
                chat = await Bot.get_chat(Config.FILE_CHANNEL_ID)
                logger.info(f"✅ FILE Channel Access: {chat.title}")
                bot_session_ready = True
            except Exception as e:
                logger.error(f"❌ FILE Channel access failed: {e}")
                bot_session_ready = False
                
        except Exception as e:
            logger.error(f"❌ BOT Session failed: {e}")
            bot_session_ready = False
            if Bot is not None:
                try:
                    await Bot.stop()
                except:
                    pass
            Bot = None
    
    # ============================================================================
    # ✅ 3. SUMMARY
    # ============================================================================
    logger.info("\n" + "=" * 60)
    logger.info("📊 SESSION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"USER Session (TEXT): {'✅ READY' if user_session_ready else '❌ NOT READY'}")
    logger.info(f"BOT Session (FILE): {'✅ READY' if bot_session_ready else '❌ NOT READY'}")
    logger.info(f"TEXT Channels: {Config.TEXT_CHANNEL_IDS}")
    logger.info(f"FILE Channel: {Config.FILE_CHANNEL_ID}")
    
    return user_session_ready or bot_session_ready

# ============================================================================
# ✅ MONGODB INITIALIZATION
# ============================================================================

@performance_monitor.measure("mongodb_init")
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
            minPoolSize=5,
            retryWrites=True,
            retryReads=True
        )
        
        await asyncio.wait_for(mongo_client.admin.command('ping'), timeout=5)
        
        db = mongo_client.sk4film
        files_col = db.files
        verification_col = db.verifications
        
        logger.info("✅ MongoDB OK")
        return True
        
    except asyncio.TimeoutError:
        logger.error("❌ MongoDB connection timeout")
        return False
    except Exception as e:
        logger.error(f"❌ MongoDB error: {e}")
        return False

# ============================================================================
# ✅ SYSTEM COMPONENTS INITIALIZATION
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

async def index_single_file_smart(message):
    """Index single file using BOT session"""
    try:
        if files_col is None or Bot is None or not bot_session_ready:
            logger.error("❌ Bot session not ready for indexing")
            return False
        
        if not message or (not message.document and not message.video):
            return False
        
        # Check if already exists
        existing_by_id = await files_col.find_one({
            'channel_id': Config.FILE_CHANNEL_ID,
            'message_id': message.id
        }, {'_id': 1})
        
        if existing_by_id:
            logger.debug(f"📝 Already indexed: {message.id}")
            return True
        
        # Extract title
        caption = message.caption if hasattr(message, 'caption') else None
        file_name = None
        
        if message.document:
            file_name = message.document.file_name
        elif message.video:
            file_name = message.video.file_name
        
        title = extract_title_from_file(file_name, caption)
        if not title:
            logger.debug(f"📝 Skipping - No title: {message.id}")
            return False
        
        normalized_title = normalize_title(title)
        
        # Create document
        doc = {
            'channel_id': Config.FILE_CHANNEL_ID,
            'message_id': message.id,
            'title': title,
            'normalized_title': normalized_title,
            'date': message.date,
            'indexed_at': datetime.now(),
            'last_checked': datetime.now(),
            'is_video_file': False,
            'file_id': None,
            'file_size': 0,
            'file_hash': None,
            'status': 'active'
        }
        
        # Add file-specific data
        if message.document:
            doc.update({
                'file_name': message.document.file_name or '',
                'quality': detect_quality_enhanced(message.document.file_name or ''),
                'is_video_file': is_video_file(message.document.file_name or ''),
                'caption': caption or '',
                'mime_type': message.document.mime_type or '',
                'file_id': message.document.file_id,
                'file_size': message.document.file_size or 0
            })
            doc['file_hash'] = await generate_file_hash(message)
        elif message.video:
            doc.update({
                'file_name': message.video.file_name or 'video.mp4',
                'quality': detect_quality_enhanced(message.video.file_name or ''),
                'is_video_file': True,
                'caption': caption or '',
                'duration': message.video.duration if hasattr(message.video, 'duration') else 0,
                'width': message.video.width if hasattr(message.video, 'width') else 0,
                'height': message.video.height if hasattr(message.video, 'height') else 0,
                'file_id': message.video.file_id,
                'file_size': message.video.file_size or 0
            })
            doc['file_hash'] = await generate_file_hash(message)
        else:
            return False
        
        # Insert into MongoDB
        try:
            await files_col.insert_one(doc)
            
            file_type = "📹 Video" if doc['is_video_file'] else "📄 File"
            size_str = format_size(doc['file_size']) if doc['file_size'] > 0 else "Unknown"
            
            logger.info(f"✅ {file_type} indexed via BOT: {title}")
            logger.info(f"   📊 Size: {size_str} | Quality: {doc.get('quality', 'Unknown')}")
            
            return True
        except Exception as e:
            if "duplicate key error" in str(e).lower():
                return True
            else:
                logger.error(f"❌ Insert error: {e}")
                return False
        
    except Exception as e:
        logger.error(f"❌ Indexing error: {e}")
        return False

async def index_files_background_smart():
    """Background indexing - Use USER session to fetch, BOT for file ops"""
    if User is None or files_col is None or not user_session_ready:
        logger.warning("⚠️ User session not ready for indexing")
        return
    
    logger.info("📁 Starting hybrid indexing (USER fetches, BOT processes)...")
    
    try:
        # Setup indexes
        if files_col is not None:
            try:
                await files_col.create_index(
                    [("channel_id", 1), ("message_id", 1)],
                    unique=True,
                    name="channel_message_unique"
                )
                await files_col.create_index(
                    [("normalized_title", "text")],
                    name="text_search_index"
                )
            except:
                pass
        
        # Get last indexed message
        last_indexed = await files_col.find_one(
            {"channel_id": Config.FILE_CHANNEL_ID}, 
            sort=[('message_id', -1)],
            projection={'message_id': 1}
        )
        
        last_message_id = last_indexed['message_id'] if last_indexed else 0
        
        logger.info(f"🔄 Starting from message ID: {last_message_id}")
        
        # ✅ USE USER SESSION to fetch FILE channel history
        total_indexed = 0
        messages = []
        
        async for msg in User.get_chat_history(Config.FILE_CHANNEL_ID, limit=50):
            if msg.id <= last_message_id:
                break
            
            if msg is not None and (msg.document or msg.video):
                messages.append(msg)
        
        messages.reverse()
        logger.info(f"📥 Found {len(messages)} new files to index")
        
        for msg in messages:
            try:
                success = await index_single_file_smart(msg)
                if success:
                    total_indexed += 1
                await asyncio.sleep(0.3)  # Rate limiting
            except Exception as e:
                logger.error(f"❌ Error processing {msg.id}: {e}")
                continue
        
        if total_indexed > 0:
            logger.info(f"✅ Hybrid indexing complete: {total_indexed} new files")
        else:
            logger.info("✅ No new files to index")
        
        # Start sync monitoring with BOT session
        if bot_session_ready:
            await channel_sync_manager.start_sync_monitoring()
            logger.info("✅ Started BOT sync monitoring")
        
    except Exception as e:
        logger.error(f"❌ Background indexing error: {e}")

# ============================================================================
# ✅ COMBINED SEARCH WITH QUALITY MERGING
# ============================================================================

@performance_monitor.measure("multi_channel_search_merged")
@async_cache_with_ttl(maxsize=500, ttl=300)
async def search_movies_multi_channel_merged(query, limit=12, page=1):
    """COMBINED search with QUALITY MERGING"""
    offset = (page - 1) * limit
    
    # Try cache first
    cache_key = f"search_merged:{query}:{page}:{limit}"
    if cache_manager is not None and cache_manager.redis_enabled:
        cached_data = await cache_manager.get(cache_key)
        if cached_data:
            logger.info(f"✅ Cache HIT for: {query}")
            return cached_data
    
    logger.info(f"🔍 QUALITY-MERGED search for: {query}")
    
    query_lower = query.lower()
    posts_dict = {}    # From USER session (text channels) 
    files_dict = {}    # From BOT session (file channel)
    
    # ============================================================================
    # ✅ 1. SEARCH TEXT CHANNELS (USER SESSION)
    # ============================================================================
    if user_session_ready and User is not None:
        async def search_text_channel(channel_id):
            channel_posts = {}
            try:
                cname = channel_name_cached(channel_id)
                async for msg in User.search_messages(channel_id, query=query, limit=15):
                    if msg is not None and msg.text and len(msg.text) > 15:
                        title = extract_title_smart(msg.text)
                        if title and (query_lower in title.lower() or query_lower in msg.text.lower()):
                            norm_title = normalize_title(title)
                            if norm_title not in channel_posts:
                                # Get poster
                                year_match = re.search(r'\b(19|20)\d{2}\b', title)
                                year = year_match.group() if year_match else ""
                                
                                # Create movie data
                                movie_data = {
                                    'title': title,
                                    'normalized_title': norm_title,
                                    'content': format_post(msg.text, max_length=1000),
                                    'post_content': msg.text,
                                    'channel': cname,
                                    'channel_id': channel_id,
                                    'message_id': msg.id,
                                    'date': msg.date.isoformat() if isinstance(msg.date, datetime) else str(msg.date),
                                    'is_new': is_new(msg.date) if msg.date else False,
                                    'has_file': False,
                                    'has_post': True,
                                    'quality_options': {},
                                    'is_video_file': False,
                                    'year': year
                                }
                                
                                channel_posts[norm_title] = movie_data
            except Exception as e:
                logger.error(f"Text search error in {channel_id}: {e}")
            return channel_posts
        
        # Search text channels concurrently
        tasks = [search_text_channel(channel_id) for channel_id in Config.TEXT_CHANNEL_IDS]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, dict):
                posts_dict.update(result)
    
    # ============================================================================
    # ✅ 2. SEARCH FILE CHANNEL (BOT SESSION) - WITH QUALITY DETECTION
    # ============================================================================
    if files_col is not None:
        try:
            cursor = files_col.find(
                {
                    "channel_id": Config.FILE_CHANNEL_ID,
                    "$or": [
                        {"title": {"$regex": query, "$options": "i"}},
                        {"normalized_title": {"$regex": query, "$options": "i"}},
                        {"file_name": {"$regex": query, "$options": "i"}},
                        {"caption": {"$regex": query, "$options": "i"}}
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
                    'caption': 1,
                    'file_id': 1,
                    '_id': 0
                }
            ).limit(limit * 3)  # Get more for quality merging
        
            async for doc in cursor:
                try:
                    norm_title = doc.get('normalized_title', normalize_title(doc['title']))
                    quality_info = extract_quality_info(doc.get('file_name', ''))
                    quality = quality_info['full']
                    
                    # Quality option
                    quality_option = {
                        'file_id': f"{doc.get('channel_id', Config.FILE_CHANNEL_ID)}_{doc.get('message_id')}_{quality}",
                        'file_size': doc.get('file_size', 0),
                        'file_name': doc.get('file_name', ''),
                        'is_video': doc.get('is_video_file', False),
                        'channel_id': doc.get('channel_id'),
                        'message_id': doc.get('message_id'),
                        'quality_info': quality_info
                    }
                    
                    if norm_title not in files_dict:
                        # Create movie data
                        title = doc['title']
                        year_match = re.search(r'\b(19|20)\d{2}\b', title)
                        year = year_match.group() if year_match else ""
                        
                        files_dict[norm_title] = {
                            'title': title,
                            'normalized_title': norm_title,
                            'content': format_post(doc.get('caption', ''), max_length=500),
                            'post_content': doc.get('caption', ''),
                            'quality_options': {quality: quality_option},  # Start with dict
                            'date': doc['date'].isoformat() if isinstance(doc['date'], datetime) else doc['date'],
                            'is_new': is_new(doc['date']) if doc.get('date') else False,
                            'is_video_file': doc.get('is_video_file', False),
                            'channel_id': doc.get('channel_id'),
                            'channel_name': channel_name_cached(doc.get('channel_id')),
                            'has_file': True,
                            'has_post': bool(doc.get('caption')),
                            'file_caption': doc.get('caption', ''),
                            'year': year,
                            'quality': quality
                        }
                    else:
                        # Add quality option to existing entry
                        files_dict[norm_title]['quality_options'][quality] = quality_option
                        
                except Exception as e:
                    logger.error(f"File processing error: {e}")
                    continue
        except Exception as e:
            logger.error(f"File search error: {e}")
    
    # ============================================================================
    # ✅ 3. MERGE QUALITY OPTIONS
    # ============================================================================
    for norm_title in files_dict:
        if files_dict[norm_title]['quality_options']:
            files_dict[norm_title]['quality_options'] = QualityMerger.merge_quality_options(
                files_dict[norm_title]['quality_options']
            )
            files_dict[norm_title]['quality_summary'] = QualityMerger.get_quality_summary(
                files_dict[norm_title]['quality_options']
            )
    
    # ============================================================================
    # ✅ 4. MERGE POSTS AND FILES - SINGLE RESULT PER TITLE
    # ============================================================================
    merged = {}
    
    all_titles = set(list(posts_dict.keys()) + list(files_dict.keys()))
    
    # Create list for batch poster fetching
    movies_for_posters = []
    
    for norm_title in all_titles:
        post_data = posts_dict.get(norm_title)
        file_data = files_dict.get(norm_title)
        
        # If both post and file exist
        if post_data and file_data:
            # Merge: Use post data as base, add file data
            result = post_data.copy()
            result['has_file'] = True
            result['quality_options'] = file_data['quality_options']
            result['quality_summary'] = file_data.get('quality_summary', '')
            result['quality'] = file_data.get('quality', '')
            
            # If post has no content but file has caption, use it
            if not result.get('post_content') and file_data.get('file_caption'):
                result['post_content'] = file_data['file_caption']
                result['content'] = format_post(file_data['file_caption'], max_length=500)
        
        # If only post exists
        elif post_data:
            result = post_data.copy()
        
        # If only file exists
        elif file_data:
            result = file_data.copy()
        
        else:
            continue
        
        # Add to batch for poster fetching
        movies_for_posters.append(result)
        merged[norm_title] = result
    
    # ============================================================================
    # ✅ 5. FETCH POSTERS IN BATCH
    # ============================================================================
    logger.info(f"🎬 Fetching posters for {len(movies_for_posters)} movies...")
    
    # Get posters for all movies in batch
    movies_with_posters = await get_posters_for_movies_batch(movies_for_posters)
    
    # Update merged dict with poster data
    for movie in movies_with_posters:
        norm_title = movie.get('normalized_title', normalize_title(movie['title']))
        if norm_title in merged:
            merged[norm_title].update({
                'poster_url': movie['poster_url'],
                'poster_source': movie['poster_source'],
                'poster_rating': movie['poster_rating'],
                'thumbnail': movie['thumbnail'],
                'thumbnail_source': movie['thumbnail_source'],
                'has_poster': movie['has_poster']
            })
            
            # Generate video thumbnail if it's a video file
            if merged[norm_title].get('is_video_file') and merged[norm_title].get('quality'):
                quality = merged[norm_title]['quality']
                title = merged[norm_title]['title']
                merged[norm_title]['thumbnail'] = FileThumbnailExtractor.generate_video_thumbnail(title, quality)
                merged[norm_title]['thumbnail_source'] = 'video_generated'
    
    # ============================================================================
    # ✅ 6. SORT AND PAGINATE
    # ============================================================================
    results_list = list(merged.values())
    
    # Sort: Has files first, then new, then by date
    results_list.sort(key=lambda x: (
        x.get('has_file', False),
        x.get('is_new', False),
        x.get('date', '')
    ), reverse=True)
    
    total = len(results_list)
    paginated = results_list[offset:offset + limit]
    
    # Statistics
    stats = {
        'total': total,
        'with_files': sum(1 for r in results_list if r.get('has_file', False)),
        'with_posts': sum(1 for r in results_list if r.get('has_post', False)),
        'both': sum(1 for r in results_list if r.get('has_file', False) and r.get('has_post', False)),
        'video_files': sum(1 for r in results_list if r.get('is_video_file', False))
    }
    
    # Final data structure
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
            'stats': stats,
            'quality_merging': True,
            'poster_fetcher': poster_fetcher is not None,
            'user_session_used': user_session_ready,
            'bot_session_used': bot_session_ready,
            'cache_hit': False
        }
    }
    
    # Cache results
    if cache_manager is not None:
        await cache_manager.set(cache_key, result_data, expire_seconds=600)
    
    logger.info(f"✅ QUALITY-MERGED search complete: {len(paginated)} results")
    logger.info(f"   📊 Stats: {stats}")
    
    return result_data

# ============================================================================
# ✅ HOME MOVIES (6/6)
# ============================================================================

def channel_name_cached(cid):
    return f"Channel {cid}"

@performance_monitor.measure("home_movies")
@async_cache_with_ttl(maxsize=1, ttl=60)
async def get_home_movies(limit=6):
    """Get home movies"""
    try:
        if User is None or not user_session_ready:
            return []
        
        movies = []
        seen_titles = set()
        
        logger.info(f"🎬 Fetching home movies (6/6)...")
        
        async for msg in User.get_chat_history(Config.MAIN_CHANNEL_ID, limit=12):
            if msg is not None and msg.text and len(msg.text) > 20:
                title = extract_title_smart(msg.text)
                
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    
                    # Extract year
                    year_match = re.search(r'\b(19|20)\d{2}\b', title)
                    year = year_match.group() if year_match else ""
                    
                    # Clean title
                    clean_title = re.sub(r'\s+\(\d{4}\)$', '', title)
                    clean_title = re.sub(r'\s+\d{4}$', '', clean_title)
                    
                    # Format content
                    post_content = msg.text
                    formatted_content = format_post(msg.text, max_length=500)
                    
                    movie_data = {
                        'title': clean_title,
                        'original_title': title,
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
                        'quality_options': {},
                        'is_video_file': False
                    }
                    
                    movies.append(movie_data)
                    
                    if len(movies) >= limit:
                        break
        
        # Fetch posters for all movies in batch
        if movies:
            movies_with_posters = await get_posters_for_movies_batch(movies)
            logger.info(f"✅ Fetched {len(movies_with_posters)} home movies")
            return movies_with_posters[:limit]
        else:
            logger.warning("⚠️ No movies found for home page")
            return []
        
    except Exception as e:
        logger.error(f"❌ Home movies error: {e}")
        return []

# ============================================================================
# ✅ MAIN INITIALIZATION
# ============================================================================

@performance_monitor.measure("system_init")
async def init_system():
    start_time = time.time()
    
    try:
        logger.info("🚀 Starting SK4FiLM v8.2 - COMPLETE INTEGRATED SYSTEM...")
        
        # Initialize MongoDB
        mongo_ok = await init_mongodb()
        if not mongo_ok:
            logger.warning("⚠️ MongoDB connection failed")
        
        # Initialize Cache Manager
        global cache_manager, verification_system, premium_system, poster_fetcher
        cache_manager = CacheManager(Config)
        redis_ok = await cache_manager.init_redis()
        if redis_ok:
            logger.info("✅ Cache Manager initialized")
            await cache_manager.start_cleanup_task()
        
        # Initialize Verification System
        verification_system = VerificationSystem(Config, mongo_client)
        logger.info("✅ Verification System initialized")
        
        # Initialize Premium System
        premium_system = PremiumSystem(Config, mongo_client)
        logger.info("✅ Premium System initialized")
        
        # Initialize Poster Fetcher
        poster_fetcher = PosterFetcher(Config, cache_manager)
        logger.info("✅ Poster Fetcher initialized")
        
        # Initialize Telegram DUAL Sessions
        if PYROGRAM_AVAILABLE:
            telegram_ok = await init_telegram_sessions()
            if not telegram_ok:
                logger.warning("⚠️ Telegram sessions failed")
        else:
            logger.warning("⚠️ Pyrogram not available")
        
        # Start background tasks
        if bot_session_ready and files_col is not None:
            asyncio.create_task(index_files_background_smart())
            logger.info("✅ Started BOT indexing")
            await channel_sync_manager.start_sync_monitoring()
            logger.info("✅ Started BOT sync monitoring")
        
        init_time = time.time() - start_time
        logger.info(f"⚡ SK4FiLM Started in {init_time:.2f}s")
        
        logger.info("🔧 INTEGRATED FEATURES:")
        logger.info(f"   • Cache System: {'✅ ENABLED' if cache_manager else '❌ DISABLED'}")
        logger.info(f"   • Verification: {'✅ ENABLED'}")
        logger.info(f"   • Premium System: {'✅ ENABLED'}")
        logger.info(f"   • Poster Fetcher: {'✅ ENABLED'}")
        logger.info(f"   • Quality Merging: ✅ ENABLED")
        logger.info(f"   • Dual Sessions: {'✅ ENABLED' if user_session_ready or bot_session_ready else '❌ DISABLED'}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ System initialization failed: {e}")
        return False

# ============================================================================
# ✅ API ROUTES
# ============================================================================

@app.route('/')
@performance_monitor.measure("root_endpoint")
async def root():
    # Check files_col is not None
    if files_col is not None:
        tf = await files_col.count_documents({})
        video_files = await files_col.count_documents({'is_video_file': True})
    else:
        tf = 0
        video_files = 0
    
    return jsonify({
        'status': 'healthy',
        'service': 'SK4FiLM v8.2 - COMPLETE INTEGRATED SYSTEM',
        'sessions': {
            'user_session': {
                'ready': user_session_ready,
                'channels': Config.TEXT_CHANNEL_IDS
            },
            'bot_session': {
                'ready': bot_session_ready,
                'channel': Config.FILE_CHANNEL_ID
            }
        },
        'components': {
            'cache': cache_manager is not None,
            'verification': verification_system is not None,
            'premium': premium_system is not None,
            'poster_fetcher': poster_fetcher is not None,
            'database': files_col is not None
        },
        'features': {
            'quality_merging': True,
            'home_movies_6_6': True,
            'hevc_support': True
        },
        'response_time': f"{time.perf_counter():.3f}s"
    })

@app.route('/health')
@performance_monitor.measure("health_endpoint")
async def health():
    return jsonify({
        'status': 'ok',
        'sessions': {
            'user': user_session_ready,
            'bot': bot_session_ready
        },
        'components': {
            'cache': cache_manager is not None,
            'verification': verification_system is not None,
            'premium': premium_system is not None,
            'poster_fetcher': poster_fetcher is not None
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/movies', methods=['GET'])
@performance_monitor.measure("movies_endpoint")
async def api_movies():
    try:
        # Get 6 home movies
        movies = await get_home_movies(limit=6)
        
        return jsonify({
            'status': 'success' if movies else 'empty',
            'movies': movies,
            'total': len(movies),
            'limit': 6,
            'source': 'telegram',
            'poster_fetcher': poster_fetcher is not None,
            'session_used': 'user',
            'channel_id': Config.MAIN_CHANNEL_ID,
            'timestamp': datetime.now().isoformat(),
            'feature': 'home_movies_6_6'
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
    try:
        query = request.args.get('query', '').strip()
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 12))
        
        if len(query) < 2:
            return jsonify({
                'status': 'error',
                'message': 'Query must be at least 2 characters'
            }), 400
        
        result_data = await search_movies_multi_channel_merged(query, limit, page)
        
        return jsonify({
            'status': 'success',
            'query': query,
            'results': result_data['results'],
            'pagination': result_data['pagination'],
            'search_metadata': {
                **result_data.get('search_metadata', {}),
                'feature': 'quality_merged_search',
                'quality_priority': Config.QUALITY_PRIORITY,
            },
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Search API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/poster', methods=['GET'])
@performance_monitor.measure("poster_endpoint")
async def api_poster():
    try:
        title = request.args.get('title', '').strip()
        year = request.args.get('year', '')
        quality = request.args.get('quality', '')
        
        if not title:
            return jsonify({
                'status': 'error',
                'message': 'Title is required'
            }), 400
        
        # Get poster
        poster_data = await get_poster_for_movie(title, year, quality)
        
        return jsonify({
            'status': 'success',
            'poster': poster_data,
            'timestamp': datetime.now().isoformat()
        })
                
    except Exception as e:
        logger.error(f"Poster API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/verification/status', methods=['GET'])
async def api_verification_status():
    try:
        user_id = int(request.args.get('user_id', 0))
        
        if user_id <= 0:
            return jsonify({
                'status': 'error',
                'message': 'Valid user_id required'
            }), 400
        
        if verification_system:
            is_verified, message = await verification_system.check_user_verified(user_id, premium_system)
            info = await verification_system.get_user_verification_info(user_id)
            
            return jsonify({
                'status': 'success',
                'user_id': user_id,
                'is_verified': is_verified,
                'message': message,
                'info': info,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Verification system not available'
            }), 500
    except Exception as e:
        logger.error(f"Verification status API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/premium/status', methods=['GET'])
async def api_premium_status():
    try:
        user_id = int(request.args.get('user_id', 0))
        
        if user_id <= 0:
            return jsonify({
                'status': 'error',
                'message': 'Valid user_id required'
            }), 400
        
        if premium_system:
            is_premium = await premium_system.is_premium_user(user_id)
            tier = await premium_system.get_user_tier(user_id)
            details = await premium_system.get_subscription_details(user_id)
            
            return jsonify({
                'status': 'success',
                'user_id': user_id,
                'is_premium': is_premium,
                'tier': tier.value if hasattr(tier, 'value') else tier,
                'details': details,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Premium system not available'
            }), 500
    except Exception as e:
        logger.error(f"Premium status API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/stats', methods=['GET'])
async def api_stats():
    """Get performance statistics"""
    try:
        perf_stats = performance_monitor.get_stats()
        
        if poster_fetcher and hasattr(poster_fetcher, 'get_stats'):
            poster_stats = poster_fetcher.get_stats()
        else:
            poster_stats = {}
        
        return jsonify({
            'status': 'success',
            'performance': perf_stats,
            'poster_fetcher': poster_stats,
            'cache': {
                'redis_enabled': cache_manager.redis_enabled if cache_manager else False
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
# ✅ STARTUP AND SHUTDOWN
# ============================================================================

app_start_time = time.time()

@app.before_serving
async def startup():
    await init_system()

@app.after_serving
async def shutdown():
    logger.info("🛑 Shutting down SK4FiLM v8.2...")
    
    shutdown_tasks = []
    
    await channel_sync_manager.stop_sync_monitoring()
    
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
    
    logger.info(f"🌐 Starting SK4FiLM v8.2 on port {Config.WEB_SERVER_PORT}...")
    logger.info("🎯 Features: Complete Integrated System with all modules")
    
    asyncio.run(serve(app, config))
