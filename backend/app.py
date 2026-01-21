# ============================================================================
# 🚀 SK4FiLM v9.0 - COMPLETE STREAMING & DOWNLOAD SYSTEM
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
    # Fallback VerificationSystem
    class VerificationSystem:
        def __init__(self, config, mongo_client):
            self.config = config
            self.mongo_client = mongo_client
        async def check_user_verified(self, user_id, premium_system):
            return True, "User verified"
        async def get_user_verification_info(self, user_id):
            return {"verified": True}
        async def stop(self): pass

try:
    from premium import PremiumSystem, PremiumTier
    logger.debug("✅ Premium module imported")
except ImportError as e:
    logger.error(f"❌ Premium module import error: {e}")
    PremiumSystem = None
    PremiumTier = None
    # Fallback PremiumSystem
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

try:
    from poster_fetching import PosterFetcher, PosterSource
    logger.debug("✅ Poster fetching module imported")
except ImportError as e:
    logger.error(f"❌ Poster fetching module import error: {e}")
    PosterFetcher = None
    PosterSource = None
    # Fallback PosterSource
    class PosterSource:
        TMDB = "tmdb"
        OMDB = "omdb"
        CUSTOM = "custom"
        FALLBACK = "fallback"

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
        # Basic normalization
        title = title.lower().strip()
        # Remove common suffixes
        title = re.sub(r'\s*\([^)]*\)$', '', title)
        title = re.sub(r'\s*\[[^\]]*\]$', '', title)
        title = re.sub(r'\s*\d{4}$', '', title)
        return title
    
    def extract_title_smart(text):
        if not text:
            return ""
        # Extract first line or first 100 chars
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if len(line) > 10 and not line.startswith('http'):
                return line[:100]
        return text[:50] if text else ""
    
    def extract_title_from_file(filename, caption=None):
        if filename:
            # Remove extension and clean
            name = os.path.splitext(filename)[0]
            # Remove quality indicators
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
        # Clean up text
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
        # Consider new if within last 7 days
        return (datetime.now() - date).days < 7

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
# ✅ CONFIGURATION - STREAMING & DOWNLOAD
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
    
    # URL Shortener
    SHORTLINK_API = os.environ.get("SHORTLINK_API", "")
    CUTTLY_API = os.environ.get("CUTTLY_API", "")
    
    # UPI IDs
    UPI_ID_BASIC = os.environ.get("UPI_ID_BASIC", "cf.sk4film@cashfreensdlpb")
    UPI_ID_PREMIUM = os.environ.get("UPI_ID_PREMIUM", "cf.sk4film@cashfreensdlpb")
    UPI_ID_GOLD = os.environ.get("UPI_ID_GOLD", "cf.sk4film@cashfreensdlpb")
    UPI_ID_DIAMOND = os.environ.get("UPI_ID_DIAMOND", "cf.sk4film@cashfreensdlpb")
    
    # Verification
    VERIFICATION_REQUIRED = os.environ.get("VERIFICATION_REQUIRED", "true").lower() == "true"
    VERIFICATION_DURATION = 6 * 60 * 60  # 6 hours
    
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
    
    # Fallback Poster
    FALLBACK_POSTER = "https://iili.io/fAeIwv9.th.png"
    
    # Thumbnail Settings
    THUMBNAIL_EXTRACT_TIMEOUT = 10
    THUMBNAIL_CACHE_DURATION = 24 * 60 * 60
    
    # 🔥 FILE CHANNEL INDEXING SETTINGS
    AUTO_INDEX_INTERVAL = int(os.environ.get("AUTO_INDEX_INTERVAL", "3600"))  # 1 hour
    BATCH_INDEX_SIZE = int(os.environ.get("BATCH_INDEX_SIZE", "100"))  # Smaller batches
    MAX_INDEX_LIMIT = int(os.environ.get("MAX_INDEX_LIMIT", "1000"))  # Limit indexing
    INDEX_ALL_HISTORY = os.environ.get("INDEX_ALL_HISTORY", "false").lower() == "true"  # Disable complete history
    INSTANT_AUTO_INDEX = os.environ.get("INSTANT_AUTO_INDEX", "true").lower() == "true"
    
    # 🔥 SEARCH SETTINGS
    SEARCH_MIN_QUERY_LENGTH = 2
    SEARCH_RESULTS_PER_PAGE = 10  # Reduced for performance
    SEARCH_CACHE_TTL = 300  # 5 minutes
    MAX_SEARCH_RESULTS = 50  # Limit total results
    
    # 🔥 STREAMING SETTINGS
    STREAMING_ENABLED = os.environ.get("STREAMING_ENABLED", "false").lower() == "true"  # Disabled by default
    STREAMING_PROXY_URL = os.environ.get("STREAMING_PROXY_URL", "https://stream.sk4film.workers.dev")
    STREAMING_TIMEOUT = int(os.environ.get("STREAMING_TIMEOUT", "30"))
    
    # 🔥 DOWNLOAD SETTINGS
    DIRECT_DOWNLOAD_ENABLED = os.environ.get("DIRECT_DOWNLOAD_ENABLED", "true").lower() == "true"
    TELEGRAM_CDN_BASE = "https://cdn5.telegram-cdn.org/file"
    TELEGRAM_DOWNLOAD_URL = "https://t.me/{bot_username}?start={file_id}"
    
    # 🔥 QUALITY PRIORITY FOR STREAMING
    STREAMING_QUALITY_PRIORITY = ['2160p', '1080p', '720p', '480p', '360p']
    
    # 🔥 PERFORMANCE OPTIMIZATIONS
    POSTER_FETCH_TIMEOUT = int(os.environ.get("POSTER_FETCH_TIMEOUT", "2"))  # 2 seconds
    POSTER_FETCH_BATCH_SIZE = int(os.environ.get("POSTER_FETCH_BATCH_SIZE", "5"))  # 5 at a time
    ENABLE_TEXT_SEARCH = os.environ.get("ENABLE_TEXT_SEARCH", "false").lower() == "true"  # Disable initially
    USE_MONGODB_TEXT_INDEX = os.environ.get("USE_MONGODB_TEXT_INDEX", "true").lower() == "true"
    
    # 🔥 TELEGRAM BOT FILE FORMAT
    TELEGRAM_FILE_FORMAT = "{channel_id}_{message_id}_{quality}"  # Format: -1001768249569_16066_480p

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
    response.headers['X-SK4FiLM-Version'] = '9.0-STREAMING-DOWNLOAD'
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
bot_handler = None

# Indexing State
is_indexing = False
last_index_time = None
indexing_task = None

# ============================================================================
# ✅ STREAMING PROXY MANAGER WITH FALLBACK
# ============================================================================

class StreamingProxyManager:
    """Manage streaming through proxy with fallback"""
    
    def __init__(self):
        self.proxy_url = Config.STREAMING_PROXY_URL
        self.session = None
    
    async def get_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=Config.STREAMING_TIMEOUT)
            )
        return self.session
    
    async def get_stream_url(self, file_id: str, quality: str = "auto") -> Optional[str]:
        """Get streaming URL for a file with fallback"""
        if not Config.STREAMING_ENABLED:
            return None
        
        try:
            session = await self.get_session()
            
            # Build proxy URL
            proxy_params = {
                "file_id": file_id,
                "quality": quality,
                "format": "stream"
            }
            
            stream_url = f"{self.proxy_url}/stream"
            
            try:
                async with session.get(stream_url, params=proxy_params, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("stream_url")
                    else:
                        logger.warning(f"Stream proxy error: {response.status}")
                        return None
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(f"Stream proxy connection failed: {e}")
                return None
                    
        except Exception as e:
            logger.error(f"Stream URL error: {e}")
            return None
    
    async def get_direct_download_url(self, file_id: str) -> Optional[Dict]:
        """Get direct download URL using Telegram format"""
        try:
            # Parse file_id format: channelId_messageId_quality
            parts = file_id.split('_')
            if len(parts) < 2:
                return None
            
            channel_id = parts[0]
            message_id = parts[1]
            quality = parts[2] if len(parts) > 2 else "480p"
            
            # Generate Telegram bot download link
            telegram_bot_url = f"https://t.me/{Config.BOT_USERNAME}?start={file_id}"
            
            # Get file info from database
            file_info = await self.get_file_info(file_id)
            
            if file_info:
                return {
                    "telegram_bot_url": telegram_bot_url,
                    "quality": quality,
                    "file_name": file_info.get('file_name', 'video.mp4'),
                    "file_size": file_info.get('file_size', 0),
                    "size_formatted": format_size(file_info.get('file_size', 0)),
                    "direct_download": False,  # For now, only bot download
                    "download_instructions": f"Click the link to download via Telegram bot"
                }
            else:
                return {
                    "telegram_bot_url": telegram_bot_url,
                    "quality": quality,
                    "file_name": "video.mp4",
                    "file_size": 0,
                    "size_formatted": "Unknown",
                    "direct_download": False,
                    "download_instructions": f"Click the link to download via Telegram bot"
                }
            
        except Exception as e:
            logger.error(f"Direct download URL error: {e}")
            return None
    
    async def get_file_info(self, file_id: str) -> Optional[Dict]:
        """Get detailed file information"""
        try:
            parts = file_id.split('_')
            if len(parts) < 2:
                return None
            
            channel_id = int(parts[0])
            message_id = int(parts[1])
            quality = parts[2] if len(parts) > 2 else "480p"
            
            if files_col is None:
                return None
            
            file_doc = await files_col.find_one({
                "channel_id": channel_id,
                "message_id": message_id
            })
            
            if not file_doc:
                return None
            
            # Format duration
            duration = file_doc.get('duration', 0)
            duration_formatted = self.format_duration(duration)
            
            return {
                "title": file_doc.get('title', ''),
                "file_name": file_doc.get('file_name', ''),
                "file_size": file_doc.get('file_size', 0),
                "quality": file_doc.get('quality', quality),
                "duration": duration,
                "duration_formatted": duration_formatted,
                "thumbnail_url": file_doc.get('thumbnail_url'),
                "telegram_file_id": file_doc.get('telegram_file_id'),
                "channel_id": channel_id,
                "message_id": message_id,
                "is_video_file": file_doc.get('is_video_file', False),
                "caption": file_doc.get('caption', ''),
                "date": file_doc.get('date'),
                "year": file_doc.get('year', '')
            }
            
        except Exception as e:
            logger.error(f"Get file info error: {e}")
            return None
    
    def format_duration(self, seconds: int) -> str:
        """Format duration in seconds to HH:MM:SS"""
        if not seconds or seconds <= 0:
            return "Unknown"
        
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    async def close(self):
        if self.session:
            await self.session.close()

streaming_proxy = StreamingProxyManager()

# ============================================================================
# ✅ FILE INFO ENHANCEMENT FOR VIEW PAGE
# ============================================================================

async def get_enhanced_file_info(file_id: str) -> Dict[str, Any]:
    """
    Get enhanced file information for view page
    Includes all quality options, download links, and streaming info
    """
    try:
        # Parse file_id format
        parts = file_id.split('_')
        if len(parts) < 2:
            return {"error": "Invalid file ID format"}
        
        channel_id = int(parts[0])
        message_id = int(parts[1])
        quality = parts[2] if len(parts) > 2 else ""
        
        # Get base file info
        if files_col is None:
            return {"error": "Database not available"}
        
        # Get the specific file
        file_doc = await files_col.find_one({
            "channel_id": channel_id,
            "message_id": message_id
        })
        
        if not file_doc:
            return {"error": "File not found"}
        
        # Get normalized title
        normalized_title = file_doc.get('normalized_title')
        if not normalized_title:
            normalized_title = normalize_title(file_doc.get('title', ''))
        
        # Find ALL files with same normalized title (all qualities)
        all_files_cursor = files_col.find({
            "normalized_title": normalized_title,
            "status": "active",
            "is_duplicate": False
        }, {
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
            'telegram_file_id': 1,
            'thumbnail_url': 1,
            'year': 1,
            'duration': 1,
            '_id': 0
        }).limit(20)
        
        all_files = await all_files_cursor.to_list(length=20)
        
        # Organize by quality
        quality_options = {}
        selected_quality_info = None
        
        for file_data in all_files:
            file_quality = file_data.get('quality', 'Unknown')
            file_msg_id = file_data.get('message_id')
            file_unique_id = f"{file_data.get('channel_id', Config.FILE_CHANNEL_ID)}_{file_msg_id}_{file_quality}"
            
            # Create quality option
            quality_option = {
                'file_id': file_unique_id,
                'file_size': file_data.get('file_size', 0),
                'size_formatted': format_size(file_data.get('file_size', 0)),
                'file_name': file_data.get('file_name', ''),
                'is_video': file_data.get('is_video_file', False),
                'channel_id': file_data.get('channel_id'),
                'message_id': file_msg_id,
                'quality': file_quality,
                'thumbnail_url': file_data.get('thumbnail_url'),
                'has_thumbnail': file_data.get('thumbnail_url') is not None,
                'date': file_data.get('date'),
                'telegram_file_id': file_data.get('telegram_file_id'),
                'duration': file_data.get('duration', 0),
                'duration_formatted': streaming_proxy.format_duration(file_data.get('duration', 0))
            }
            
            # Check if this is the selected quality
            if file_unique_id == file_id:
                selected_quality_info = quality_option
            
            # Add to quality options
            quality_options[file_quality] = quality_option
        
        # If no specific quality selected, use first one
        if not selected_quality_info and quality_options:
            first_quality = list(quality_options.keys())[0]
            selected_quality_info = quality_options[first_quality]
        
        # Get streaming URL for selected quality
        stream_url = None
        if Config.STREAMING_ENABLED and selected_quality_info:
            stream_url = await streaming_proxy.get_stream_url(
                selected_quality_info['file_id'], 
                selected_quality_info['quality']
            )
        
        # Get download info
        download_info = await streaming_proxy.get_direct_download_url(
            selected_quality_info['file_id'] if selected_quality_info else file_id
        )
        
        # Get poster for the movie
        poster_data = await get_poster_for_movie_quick(
            file_doc.get('title', ''),
            file_doc.get('year', '')
        )
        
        # Prepare quality list sorted by priority
        qualities_list = list(quality_options.keys())
        
        def get_quality_priority(q):
            base_q = q.replace(' HEVC', '')
            if base_q in Config.QUALITY_PRIORITY:
                return Config.QUALITY_PRIORITY.index(base_q)
            return 999
        
        qualities_list.sort(key=get_quality_priority)
        
        # Format duration
        duration_formatted = streaming_proxy.format_duration(file_doc.get('duration', 0))
        
        return {
            'status': 'success',
            'file_info': {
                'title': file_doc.get('title', ''),
                'original_title': file_doc.get('title', ''),
                'year': file_doc.get('year', ''),
                'caption': file_doc.get('caption', ''),
                'content': format_post(file_doc.get('caption', ''), max_length=1000),
                'post_content': file_doc.get('caption', ''),
                'date': file_doc.get('date'),
                'is_new': is_new(file_doc.get('date')),
                'poster_url': poster_data['poster_url'],
                'poster_source': poster_data['source'],
                'thumbnail_url': file_doc.get('thumbnail_url') or poster_data['poster_url'],
                'has_thumbnail': bool(file_doc.get('thumbnail_url')),
                'has_poster': True,
                'channel_id': file_doc.get('channel_id'),
                'channel_name': f"Channel {file_doc.get('channel_id')}",
                'message_id': file_doc.get('message_id'),
                'has_file': True,
                'has_post': bool(file_doc.get('caption')),
                'is_video_file': file_doc.get('is_video_file', False),
                'result_type': 'file' if not file_doc.get('caption') else 'both',
                'bot_username': Config.BOT_USERNAME,
                'streaming_enabled': Config.STREAMING_ENABLED,
                'direct_download_enabled': Config.DIRECT_DOWNLOAD_ENABLED
            },
            'streaming': {
                'enabled': Config.STREAMING_ENABLED,
                'stream_url': stream_url,
                'selected_quality': selected_quality_info['quality'] if selected_quality_info else '',
                'selected_file_id': selected_quality_info['file_id'] if selected_quality_info else file_id
            },
            'download': {
                'enabled': Config.DIRECT_DOWNLOAD_ENABLED,
                'info': download_info,
                'telegram_bot_url': f"https://t.me/{Config.BOT_USERNAME}?start={file_id}"
            },
            'quality_options': {
                'available': qualities_list,
                'selected': selected_quality_info['quality'] if selected_quality_info else '',
                'details': quality_options
            },
            'media_info': {
                'duration': file_doc.get('duration', 0),
                'duration_formatted': duration_formatted,
                'file_size': selected_quality_info['file_size'] if selected_quality_info else 0,
                'size_formatted': selected_quality_info['size_formatted'] if selected_quality_info else 'Unknown',
                'quality': selected_quality_info['quality'] if selected_quality_info else '',
                'file_name': selected_quality_info['file_name'] if selected_quality_info else '',
                'is_hevc': 'HEVC' in (selected_quality_info['quality'] if selected_quality_info else '')
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Enhanced file info error: {e}")
        return {
            'status': 'error',
            'message': str(e)
        }

# ============================================================================
# ✅ BOT HANDLER MODULE
# ============================================================================

class BotHandler:
    """Bot handler for Telegram bot operations"""
    
    def __init__(self, bot_token=None, api_id=None, api_hash=None):
        self.bot_token = bot_token or Config.BOT_TOKEN
        self.api_id = api_id or Config.API_ID
        self.api_hash = api_hash or Config.API_HASH
        self.bot = None
        self.initialized = False
        self.last_update = None
        
    async def initialize(self):
        """Initialize bot handler"""
        if not self.bot_token or not self.api_id or not self.api_hash:
            logger.error("❌ Bot token or API credentials not configured")
            return False
        
        try:
            # Reuse existing Bot session if available
            global Bot, bot_session_ready
            if Bot is not None and bot_session_ready:
                self.bot = Bot
                logger.info("✅ Bot Handler using existing Bot session")
                self.initialized = True
                self.last_update = datetime.now()
                return True
            
            # Otherwise create new session
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
            logger.info(f"✅ Bot Handler Ready: @{bot_info.username}")
            self.initialized = True
            self.last_update = datetime.now()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Bot handler initialization error: {e}")
            return False
    
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
    
    async def get_bot_status(self):
        """Get bot handler status"""
        return {
            'initialized': self.initialized,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'bot_username': (await self.bot.get_me()).username if self.initialized else None
        }
    
    async def shutdown(self):
        """Shutdown bot handler"""
        if self.bot and not bot_session_ready:
            await self.bot.stop()
        self.initialized = False

bot_handler = BotHandler()

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
# ✅ VIDEO THUMBNAIL EXTRACTOR
# ============================================================================

class VideoThumbnailExtractor:
    """Extract thumbnails from video files"""
    
    def __init__(self):
        self.extraction_lock = asyncio.Lock()
    
    async def extract_thumbnail(self, channel_id: int, message_id: int) -> Optional[str]:
        """
        Extract thumbnail from video file
        Returns base64 data URL or None
        """
        try:
            # Use bot handler to extract thumbnail
            if bot_handler and bot_handler.initialized:
                try:
                    message = await bot_handler.bot.get_messages(channel_id, message_id)
                    if not message:
                        return None
                    
                    thumbnail_data = None
                    
                    if message.video:
                        if hasattr(message.video, 'thumbnail') and message.video.thumbnail:
                            thumbnail_file_id = message.video.thumbnail.file_id
                            download_path = await bot_handler.bot.download_media(thumbnail_file_id, in_memory=True)
                            
                            if download_path:
                                if isinstance(download_path, bytes):
                                    thumbnail_data = download_path
                    
                    if thumbnail_data:
                        base64_data = base64.b64encode(thumbnail_data).decode('utf-8')
                        return f"data:image/jpeg;base64,{base64_data}"
                    
                except Exception as e:
                    logger.debug(f"Bot handler thumbnail extraction error: {e}")
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Thumbnail extraction failed: {e}")
            return None

thumbnail_extractor = VideoThumbnailExtractor()

# ============================================================================
# ✅ DUPLICATE PREVENTION SYSTEM
# ============================================================================

class DuplicatePreventionSystem:
    """Advanced duplicate detection and prevention"""
    
    def __init__(self):
        self.file_hashes = set()
        self.title_cache = defaultdict(set)
        self.lock = asyncio.Lock()
    
    async def initialize_from_database(self):
        """Load existing hashes from database"""
        if files_col is None:
            return
        
        try:
            async with self.lock:
                # Clear existing data
                self.file_hashes.clear()
                self.title_cache.clear()
                
                # Load file hashes
                cursor = files_col.find(
                    {"file_hash": {"$ne": None}},
                    {"file_hash": 1, "normalized_title": 1, "_id": 0}
                )
                
                async for doc in cursor:
                    file_hash = doc.get('file_hash')
                    normalized_title = doc.get('normalized_title')
                    
                    if file_hash:
                        self.file_hashes.add(file_hash)
                    
                    if normalized_title:
                        if file_hash:
                            self.title_cache[normalized_title].add(file_hash)
                
                logger.info(f"✅ Loaded {len(self.file_hashes)} file hashes from database")
                logger.info(f"✅ Loaded {len(self.title_cache)} unique titles from database")
                
        except Exception as e:
            logger.error(f"❌ Error initializing duplicate prevention: {e}")
    
    async def is_duplicate_file(self, file_hash, normalized_title=None):
        """
        Check if file is a duplicate
        Returns: (is_duplicate, reason)
        """
        if not file_hash:
            return False, "no_hash"
        
        async with self.lock:
            # Check if hash already exists
            if file_hash in self.file_hashes:
                return True, "same_hash"
            
            return False, "unique"
    
    async def add_file_hash(self, file_hash, normalized_title=None):
        """Add new file hash to tracking"""
        if not file_hash:
            return
        
        async with self.lock:
            self.file_hashes.add(file_hash)
            
            if normalized_title:
                self.title_cache[normalized_title].add(file_hash)
    
    async def remove_file_hash(self, file_hash, normalized_title=None):
        """Remove file hash from tracking"""
        if not file_hash:
            return
        
        async with self.lock:
            if file_hash in self.file_hashes:
                self.file_hashes.remove(file_hash)
            
            if normalized_title and normalized_title in self.title_cache:
                if file_hash in self.title_cache[normalized_title]:
                    self.title_cache[normalized_title].remove(file_hash)
                
                # Clean up empty sets
                if not self.title_cache[normalized_title]:
                    del self.title_cache[normalized_title]
    
    async def get_duplicate_stats(self):
        """Get duplicate statistics"""
        async with self.lock:
            return {
                'total_unique_hashes': len(self.file_hashes),
                'total_unique_titles': len(self.title_cache),
                'files_per_title': {
                    title: len(hashes) 
                    for title, hashes in list(self.title_cache.items())[:10]
                }
            }

duplicate_prevention = DuplicatePreventionSystem()

# ============================================================================
# ✅ FILE CHANNEL INDEXING MANAGER - OPTIMIZED
# ============================================================================

class FileChannelIndexingManager:
    """Optimized file channel indexing manager"""
    
    def __init__(self):
        self.is_running = False
        self.indexing_task = None
        self.last_run = None
        self.next_run = None
        self.total_indexed = 0
        self.total_duplicates = 0
        
        self.indexing_stats = {
            'total_runs': 0,
            'total_files_processed': 0,
            'total_indexed': 0,
            'total_duplicates': 0,
            'total_errors': 0,
            'last_success': None
        }
        
        self.is_first_run = True
    
    async def start_indexing(self):
        """Start file channel indexing"""
        if self.is_running:
            logger.warning("⚠️ File indexing already running")
            return
        
        logger.info("🚀 Starting FILE CHANNEL INDEXING...")
        self.is_running = True
        
        # Initialize duplicate prevention
        await duplicate_prevention.initialize_from_database()
        
        # Run immediate indexing
        asyncio.create_task(self._run_optimized_indexing())
        
        # Start periodic loop
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
    
    async def _run_optimized_indexing(self):
        """Run optimized indexing of file channel"""
        logger.info("🔥 RUNNING OPTIMIZED FILE CHANNEL INDEXING...")
        
        try:
            # Get last indexed message from database
            last_doc = await files_col.find_one(
                {"channel_id": Config.FILE_CHANNEL_ID},
                sort=[("message_id", -1)]
            )
            
            last_indexed_id = last_doc.get('message_id', 0) if last_doc else 0
            logger.info(f"📊 Last indexed message ID: {last_indexed_id}")
            
            # Fetch messages in batches starting from last indexed
            batch_size = Config.BATCH_INDEX_SIZE
            total_indexed = 0
            batch_count = 0
            
            while self.is_running:
                try:
                    # Fetch batch of messages
                    messages = []
                    async for msg in User.get_chat_history(
                        Config.FILE_CHANNEL_ID,
                        limit=batch_size,
                        offset_id=last_indexed_id
                    ):
                        if msg and (msg.document or msg.video):
                            messages.append(msg)
                    
                    if not messages:
                        logger.info("✅ No more messages to index")
                        break
                    
                    # Process batch
                    batch_indexed = 0
                    for msg in messages:
                        success = await index_single_file_smart(msg)
                        if success:
                            batch_indexed += 1
                            total_indexed += 1
                        
                        # Update last processed ID
                        if msg.id > last_indexed_id:
                            last_indexed_id = msg.id
                    
                    batch_count += 1
                    logger.info(f"📦 Batch {batch_count}: Processed {len(messages)} messages, indexed {batch_indexed} files")
                    
                    # Check if we reached the limit
                    if Config.MAX_INDEX_LIMIT > 0 and total_indexed >= Config.MAX_INDEX_LIMIT:
                        logger.info(f"⚠️ Reached max indexing limit: {Config.MAX_INDEX_LIMIT}")
                        break
                    
                    # Small delay to avoid rate limiting
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"❌ Batch processing error: {e}")
                    break
            
            logger.info(f"✅ Optimized indexing complete. Total indexed: {total_indexed}")
            self.total_indexed = total_indexed
            
        except Exception as e:
            logger.error(f"❌ Optimized indexing error: {e}")
    
    async def _indexing_loop(self):
        """Main indexing loop"""
        while self.is_running:
            try:
                # Wait for next run
                if self.next_run and self.next_run > datetime.now():
                    wait_seconds = (self.next_run - datetime.now()).total_seconds()
                    if wait_seconds > 30:
                        logger.info(f"⏰ Next index in {wait_seconds:.0f}s")
                    await asyncio.sleep(min(wait_seconds, 30))
                    continue
                
                # Run indexing cycle
                await self._run_indexing_cycle()
                
                # Schedule next run
                self.next_run = datetime.now() + timedelta(seconds=Config.AUTO_INDEX_INTERVAL)
                self.last_run = datetime.now()
                
                # Sleep before checking again
                await asyncio.sleep(10)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Indexing loop error: {e}")
                await asyncio.sleep(60)
    
    async def _run_indexing_cycle(self):
        """Run one indexing cycle"""
        logger.info("=" * 50)
        logger.info("🔄 FILE INDEXING CYCLE")
        logger.info("=" * 50)
        
        start_time = time.time()
        cycle_stats = {
            'processed': 0,
            'indexed': 0,
            'duplicates': 0,
            'errors': 0
        }
        
        try:
            # Get last indexed message
            last_indexed = await files_col.find_one(
                {"channel_id": Config.FILE_CHANNEL_ID}, 
                sort=[('message_id', -1)],
                projection={'message_id': 1}
            )
            
            last_message_id = last_indexed['message_id'] if last_indexed else 0
            
            logger.info(f"📊 Last indexed message ID: {last_message_id}")
            
            # Fetch new messages
            messages_to_index = []
            fetched_count = 0
            
            try:
                # Fetch recent messages
                async for msg in User.get_chat_history(Config.FILE_CHANNEL_ID, limit=Config.BATCH_INDEX_SIZE):
                    fetched_count += 1
                    
                    # Stop if we reach already indexed messages
                    if msg.id <= last_message_id:
                        break
                    
                    # Only index file messages
                    if msg and (msg.document or msg.video):
                        messages_to_index.append(msg)
                
                logger.info(f"📥 Fetched {fetched_count} messages, found {len(messages_to_index)} new files")
                
            except Exception as e:
                logger.error(f"❌ Error fetching messages: {e}")
                return
            
            # Process messages
            if messages_to_index:
                batch_stats = await self._process_indexing_batch(messages_to_index)
                cycle_stats.update(batch_stats)
            
            # Update stats
            self.indexing_stats['total_runs'] += 1
            self.indexing_stats['total_files_processed'] += cycle_stats['processed']
            self.indexing_stats['total_indexed'] += cycle_stats['indexed']
            self.indexing_stats['total_duplicates'] += cycle_stats['duplicates']
            self.indexing_stats['total_errors'] += cycle_stats['errors']
            self.indexing_stats['last_success'] = datetime.now()
            
            elapsed = time.time() - start_time
            
            logger.info("=" * 50)
            logger.info("📊 INDEXING CYCLE COMPLETE")
            logger.info("=" * 50)
            logger.info(f"⏱️  Time: {elapsed:.2f}s")
            logger.info(f"📥 Fetched: {fetched_count} messages")
            logger.info(f"📄 Processed: {cycle_stats['processed']} files")
            logger.info(f"✅ Indexed: {cycle_stats['indexed']} new files")
            logger.info(f"🔄 Duplicates: {cycle_stats['duplicates']} skipped")
            logger.info(f"❌ Errors: {cycle_stats['errors']}")
            logger.info(f"📈 Total Indexed: {self.indexing_stats['total_indexed']}")
            logger.info("=" * 50)
            
            # Update counts
            self.total_indexed += cycle_stats['indexed']
            self.total_duplicates += cycle_stats['duplicates']
            
        except Exception as e:
            logger.error(f"❌ Indexing cycle failed: {e}")
            self.indexing_stats['total_errors'] += 1
    
    async def _process_indexing_batch(self, messages):
        """Process a batch of messages"""
        batch_stats = {
            'processed': len(messages),
            'indexed': 0,
            'duplicates': 0,
            'errors': 0
        }
        
        for msg in messages:
            try:
                # Check if already indexed by message ID
                existing = await files_col.find_one({
                    'channel_id': Config.FILE_CHANNEL_ID,
                    'message_id': msg.id
                }, {'_id': 1})
                
                if existing:
                    logger.debug(f"📝 Already indexed: {msg.id}")
                    batch_stats['duplicates'] += 1
                    continue
                
                # Index the file
                success = await index_single_file_smart(msg)
                
                if success:
                    batch_stats['indexed'] += 1
                else:
                    batch_stats['duplicates'] += 1
                    
            except Exception as e:
                logger.error(f"❌ Error processing message {msg.id}: {e}")
                batch_stats['errors'] += 1
                continue
        
        logger.info(f"📦 Batch stats: {batch_stats}")
        return batch_stats
    
    async def get_indexing_status(self):
        """Get current indexing status"""
        return {
            'is_running': self.is_running,
            'is_first_run': self.is_first_run,
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'next_run': self.next_run.isoformat() if self.next_run else None,
            'total_indexed': self.total_indexed,
            'total_duplicates': self.total_duplicates,
            'stats': self.indexing_stats
        }

# Initialize file indexing manager
file_indexing_manager = FileChannelIndexingManager()

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
        """Start sync monitoring"""
        if self.is_monitoring:
            return
        
        logger.info("👁️ Starting sync monitoring...")
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
        """Sync deletions from Telegram"""
        try:
            if files_col is None:
                return
            
            current_time = time.time()
            if current_time - self.last_sync < 300:
                return
            
            self.last_sync = current_time
            
            # Get message IDs from MongoDB
            cursor = files_col.find(
                {"channel_id": Config.FILE_CHANNEL_ID},
                {"message_id": 1, "_id": 0, "file_hash": 1, "normalized_title": 1}
            )
            
            message_data = []
            async for doc in cursor:
                msg_id = doc.get('message_id')
                if msg_id:
                    message_data.append({
                        'message_id': msg_id,
                        'file_hash': doc.get('file_hash'),
                        'normalized_title': doc.get('normalized_title')
                    })
            
            if not message_data:
                return
            
            deleted_count = 0
            batch_size = 50
            
            for i in range(0, len(message_data), batch_size):
                batch = message_data[i:i + batch_size]
                message_ids = [item['message_id'] for item in batch]
                
                try:
                    # Check if messages exist using User session
                    messages = await User.get_messages(Config.FILE_CHANNEL_ID, message_ids)
                    
                    existing_ids = set()
                    if isinstance(messages, list):
                        for msg in messages:
                            if msg and hasattr(msg, 'id'):
                                existing_ids.add(msg.id)
                    elif messages is not None and hasattr(messages, 'id'):
                        existing_ids.add(messages.id)
                    
                    # Find deleted IDs
                    for item in batch:
                        if item['message_id'] not in existing_ids:
                            # Delete from MongoDB
                            await files_col.delete_one({
                                "channel_id": Config.FILE_CHANNEL_ID,
                                "message_id": item['message_id']
                            })
                            
                            # Remove from duplicate prevention
                            if item.get('file_hash'):
                                await duplicate_prevention.remove_file_hash(
                                    item['file_hash'],
                                    item.get('normalized_title')
                                )
                            
                            deleted_count += 1
                            self.deleted_count += 1
                
                except Exception as e:
                    logger.error(f"❌ Batch check error: {e}")
                    continue
            
            if deleted_count > 0:
                logger.info(f"✅ Sync: {deleted_count} files deleted")
            
        except Exception as e:
            logger.error(f"❌ Sync deletions error: {e}")

channel_sync_manager = ChannelSyncManager()

# ============================================================================
# ✅ FILE INDEXING FUNCTIONS - OPTIMIZED
# ============================================================================

async def generate_file_hash(message):
    """Generate unique hash for file to detect duplicates"""
    try:
        hash_parts = []
        
        if message.document:
            file_attrs = message.document
            # Use file_id for hash
            hash_parts.append(f"doc_{file_attrs.file_id}")
            if file_attrs.file_name:
                name_hash = hashlib.md5(file_attrs.file_name.encode()).hexdigest()[:16]
                hash_parts.append(f"name_{name_hash}")
            if file_attrs.file_size:
                hash_parts.append(f"size_{file_attrs.file_size}")
        elif message.video:
            file_attrs = message.video
            hash_parts.append(f"vid_{file_attrs.file_id}")
            if file_attrs.file_name:
                name_hash = hashlib.md5(file_attrs.file_name.encode()).hexdigest()[:16]
                hash_parts.append(f"name_{name_hash}")
            if file_attrs.file_size:
                hash_parts.append(f"size_{file_attrs.file_size}")
            if hasattr(file_attrs, 'duration'):
                hash_parts.append(f"dur_{file_attrs.duration}")
        else:
            return None
        
        # Add caption hash only if exists
        if message.caption:
            caption_hash = hashlib.md5(message.caption.encode()).hexdigest()[:12]
            hash_parts.append(f"cap_{caption_hash}")
        
        # Final hash
        final_hash = hashlib.sha256("_".join(hash_parts).encode()).hexdigest()
        return final_hash
        
    except Exception as e:
        logger.debug(f"Hash generation error: {e}")
        return None

async def extract_title_improved(filename, caption):
    """Improved title extraction"""
    # Try filename first
    if filename:
        # Clean filename
        name = os.path.splitext(filename)[0]
        
        # Remove common patterns
        patterns_to_remove = [
            r'\b(480p|720p|1080p|2160p|4k|uhd|hd|hevc|x265|x264|h264|h265)\b',
            r'\b(web|webrip|webdl|bluray|brrip|dvdrip|hdtv)\b',
            r'\b(hindi|english|tamil|telugu|malayalam|bengali)\b',
            r'\b(dual|multi)\b',
            r'\b(ac3|aac|dd5\.1|dts)\b',
            r'\b(\d{3,4}p)\b',
            r'[._]'
        ]
        
        for pattern in patterns_to_remove:
            name = re.sub(pattern, ' ', name, flags=re.IGNORECASE)
        
        # Clean up
        name = re.sub(r'\s+', ' ', name)
        name = name.strip()
        
        # Extract year if present
        year_match = re.search(r'\b(19|20)\d{2}\b', name)
        if year_match:
            year = year_match.group()
            # Remove year from name
            name = re.sub(r'\s*\b(19|20)\d{2}\b', '', name)
            name = f"{name.strip()} ({year})"
        
        if name and len(name) > 3:
            return name
    
    # Try caption
    if caption:
        # Extract first meaningful line
        lines = caption.split('\n')
        for line in lines:
            line = line.strip()
            if len(line) > 10 and not line.startswith('http'):
                # Clean the line
                line = re.sub(r'📥.*', '', line)  # Remove download indicators
                line = re.sub(r'🎬.*', '', line)  # Remove movie indicators
                line = re.sub(r'⚡.*', '', line)  # Remove speed indicators
                line = re.sub(r'✅.*', '', line)  # Remove check indicators
                line = re.sub(r'[⭐🌟]+', '', line)  # Remove stars
                line = line.strip()
                
                if line and len(line) > 5:
                    return line[:200]  # Limit length
    
    # Fallback to filename
    if filename:
        return os.path.splitext(filename)[0][:100]
    
    return "Unknown File"

async def index_single_file_smart(message):
    """Index single file with improved logic and Telegram file format"""
    try:
        if files_col is None:
            logger.error("❌ Database not ready for indexing")
            return False
        
        if not message or (not message.document and not message.video):
            logger.debug(f"❌ Not a file message: {message.id}")
            return False
        
        # Extract title
        caption = message.caption if hasattr(message, 'caption') else None
        file_name = None
        
        if message.document:
            file_name = message.document.file_name
        elif message.video:
            file_name = message.video.file_name
        
        title = await extract_title_improved(file_name, caption)
        if not title or title == "Unknown File":
            logger.debug(f"📝 Skipping - No valid title: {message.id}")
            return False
        
        normalized_title = normalize_title(title)
        
        # Check if already exists by message ID
        existing_by_id = await files_col.find_one({
            'channel_id': Config.FILE_CHANNEL_ID,
            'message_id': message.id
        }, {'_id': 1})
        
        if existing_by_id:
            logger.debug(f"📝 Already indexed: {title[:50]}... (ID: {message.id})")
            return False
        
        # Generate file hash for duplicate detection
        file_hash = await generate_file_hash(message)
        
        # Check for duplicates using hash
        if file_hash:
            is_duplicate, reason = await duplicate_prevention.is_duplicate_file(
                file_hash, normalized_title
            )
            
            if is_duplicate:
                logger.info(f"🔄 DUPLICATE: {title[:50]}... - Reason: {reason}")
                return False
        
        # Extract thumbnail if video file (do this later to avoid blocking)
        thumbnail_url = None
        is_video = False
        
        if message.video or (message.document and is_video_file(file_name or '')):
            is_video = True
            # We'll extract thumbnail separately to avoid blocking
        
        # Extract year from title
        year_match = re.search(r'\b(19|20)\d{2}\b', title)
        year = year_match.group() if year_match else ""
        
        # Extract quality
        quality = detect_quality_enhanced(file_name or "")
        
        # Create document with Telegram file format
        doc = {
            'channel_id': Config.FILE_CHANNEL_ID,
            'message_id': message.id,
            'title': title,
            'normalized_title': normalized_title,
            'date': message.date,
            'indexed_at': datetime.now(),
            'last_checked': datetime.now(),
            'is_video_file': is_video,
            'file_size': 0,
            'file_hash': file_hash,
            'thumbnail_url': thumbnail_url,
            'thumbnail_extracted': thumbnail_url is not None,
            'status': 'active',
            'is_duplicate': False,
            'quality': quality,
            'year': year,
            'telegram_file_format': f"{Config.FILE_CHANNEL_ID}_{message.id}_{quality}"  # Store file format
        }
        
        # Add file-specific data
        if message.document:
            doc.update({
                'file_name': message.document.file_name or '',
                'is_video_file': is_video_file(message.document.file_name or ''),
                'caption': caption or '',
                'mime_type': message.document.mime_type or '',
                'telegram_file_id': message.document.file_id,
                'file_size': message.document.file_size or 0
            })
            if hasattr(message.document, 'duration'):
                doc['duration'] = message.document.duration
        elif message.video:
            doc.update({
                'file_name': message.video.file_name or 'video.mp4',
                'is_video_file': True,
                'caption': caption or '',
                'duration': message.video.duration if hasattr(message.video, 'duration') else 0,
                'width': message.video.width if hasattr(message.video, 'width') else 0,
                'height': message.video.height if hasattr(message.video, 'height') else 0,
                'telegram_file_id': message.video.file_id,
                'file_size': message.video.file_size or 0
            })
        else:
            return False
        
        # Insert into MongoDB
        try:
            await files_col.insert_one(doc)
            
            # Add to duplicate prevention system
            if file_hash:
                await duplicate_prevention.add_file_hash(file_hash, normalized_title)
            
            # Log success
            file_type = "📹 Video" if doc['is_video_file'] else "📄 File"
            size_str = format_size(doc['file_size']) if doc['file_size'] > 0 else "Unknown"
            
            logger.info(f"✅ INDEXED: {title[:60]}...")
            logger.info(f"   📊 Message ID: {message.id} | Size: {size_str} | Quality: {quality}")
            logger.info(f"   📁 File Format: {doc['telegram_file_format']}")
            
            # Schedule thumbnail extraction for video files (non-blocking)
            if is_video and Config.FILE_CHANNEL_ID and message.id:
                asyncio.create_task(extract_and_update_thumbnail(Config.FILE_CHANNEL_ID, message.id))
            
            return True
            
        except Exception as e:
            if "duplicate key error" in str(e).lower():
                logger.debug(f"📝 Duplicate key error: {message.id}")
                return False
            else:
                logger.error(f"❌ Insert error: {e}")
                return False
        
    except Exception as e:
        logger.error(f"❌ Indexing error for message {message.id}: {e}")
        return False

async def extract_and_update_thumbnail(channel_id, message_id):
    """Extract and update thumbnail for a video file"""
    try:
        thumbnail_url = await thumbnail_extractor.extract_thumbnail(channel_id, message_id)
        if thumbnail_url:
            await files_col.update_one(
                {"channel_id": channel_id, "message_id": message_id},
                {"$set": {"thumbnail_url": thumbnail_url, "thumbnail_extracted": True}}
            )
            logger.debug(f"✅ Thumbnail extracted and saved: {channel_id}/{message_id}")
    except Exception as e:
        logger.debug(f"❌ Thumbnail extraction failed: {e}")

async def initial_indexing():
    """Initial indexing on startup"""
    if User is None or files_col is None or not user_session_ready:
        logger.warning("⚠️ User session not ready for initial indexing")
        return
    
    logger.info("=" * 60)
    logger.info("🚀 STARTING OPTIMIZED FILE CHANNEL INDEXING")
    logger.info("=" * 60)
    
    try:
        # Setup indexes
        await setup_database_indexes()
        
        # Start file indexing
        await file_indexing_manager.start_indexing()
        
        # Start sync monitoring
        await channel_sync_manager.start_sync_monitoring()
        
    except Exception as e:
        logger.error(f"❌ Initial indexing error: {e}")

async def setup_database_indexes():
    """Setup all required database indexes"""
    if files_col is None:
        return
    
    try:
        # Unique index for channel + message
        await files_col.create_index(
            [("channel_id", 1), ("message_id", 1)],
            unique=True,
            name="channel_message_unique",
            background=True
        )
        
        # Text search index for better search performance
        await files_col.create_index(
            [("normalized_title", "text"), ("title", "text")],
            name="title_text_search",
            background=True,
            default_language="english"
        )
        
        # Quality index
        await files_col.create_index(
            [("quality", 1)],
            name="quality_index",
            background=True
        )
        
        # Date index
        await files_col.create_index(
            [("date", -1)],
            name="date_index",
            background=True
        )
        
        # Normalized title index for quick lookups
        await files_col.create_index(
            [("normalized_title", 1)],
            name="normalized_title_index",
            background=True
        )
        
        logger.info("✅ Created database indexes")
        
    except Exception as e:
        logger.warning(f"⚠️ Index creation error: {e}")

# ============================================================================
# ✅ POSTER FETCHING FUNCTIONS - OPTIMIZED
# ============================================================================

async def get_poster_for_movie_quick(title: str, year: str = "") -> Dict[str, Any]:
    """Quick poster fetch with timeout"""
    if not title:
        return {
            'poster_url': Config.FALLBACK_POSTER,
            'source': 'fallback',
            'rating': '0.0',
            'year': year,
            'title': title
        }
    
    # Clean title for search
    search_title = title
    # Remove year from title if present
    search_title = re.sub(r'\s*\(\d{4}\)$', '', search_title)
    search_title = re.sub(r'\s*\d{4}$', '', search_title)
    
    try:
        # Try TMDB first
        if Config.TMDB_API_KEY:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2)) as session:
                search_url = "https://api.themoviedb.org/3/search/movie"
                params = {
                    'api_key': Config.TMDB_API_KEY,
                    'query': search_title[:50],
                    'year': year,
                    'page': 1,
                    'include_adult': 'false'
                }
                
                async with session.get(search_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('results'):
                            movie = data['results'][0]
                            poster_path = movie.get('poster_path')
                            rating = movie.get('vote_average', 0)
                            
                            if poster_path:
                                return {
                                    'poster_url': f"https://image.tmdb.org/t/p/w500{poster_path}",
                                    'source': 'tmdb',
                                    'rating': f"{rating:.1f}" if rating else '0.0',
                                    'year': movie.get('release_date', '')[:4] if movie.get('release_date') else year,
                                    'title': movie.get('title', title)
                                }
    except:
        pass
    
    # Fallback
    return {
        'poster_url': Config.FALLBACK_POSTER,
        'source': 'fallback',
        'rating': '0.0',
        'year': year,
        'title': title
    }

async def get_posters_for_movies_batch_quick(movies: List[Dict]) -> List[Dict]:
    """Get posters for multiple movies in batch (optimized)"""
    results = []
    
    # Limit batch size for performance
    limited_movies = movies[:Config.POSTER_FETCH_BATCH_SIZE]
    
    # Create tasks for all movies
    tasks = []
    for movie in limited_movies:
        title = movie.get('title', '')
        year = movie.get('year', '')
        
        task = asyncio.create_task(get_poster_for_movie_quick(title, year))
        tasks.append((movie, task))
    
    # Process results with timeout
    for movie, task in tasks:
        try:
            poster_data = await asyncio.wait_for(task, timeout=Config.POSTER_FETCH_TIMEOUT)
            
            # Update movie with poster data
            movie_with_poster = movie.copy()
            movie_with_poster.update({
                'poster_url': poster_data['poster_url'],
                'poster_source': poster_data['source'],
                'poster_rating': poster_data['rating'],
                'thumbnail': poster_data['poster_url'],
                'thumbnail_source': poster_data['source'],
                'has_poster': True,
                'has_thumbnail': True
            })
            
            results.append(movie_with_poster)
            
        except (asyncio.TimeoutError, Exception) as e:
            # Add movie with fallback
            movie_with_fallback = movie.copy()
            movie_with_fallback.update({
                'poster_url': Config.FALLBACK_POSTER,
                'poster_source': 'fallback',
                'poster_rating': '0.0',
                'thumbnail': Config.FALLBACK_POSTER,
                'thumbnail_source': 'fallback',
                'has_poster': True,
                'has_thumbnail': True
            })
            
            results.append(movie_with_fallback)
    
    # Add remaining movies with fallback
    for movie in movies[Config.POSTER_FETCH_BATCH_SIZE:]:
        movie_with_fallback = movie.copy()
        movie_with_fallback.update({
            'poster_url': Config.FALLBACK_POSTER,
            'poster_source': 'fallback',
            'poster_rating': '0.0',
            'thumbnail': Config.FALLBACK_POSTER,
            'thumbnail_source': 'fallback',
            'has_poster': True,
            'has_thumbnail': True
        })
        results.append(movie_with_fallback)
    
    return results

# ============================================================================
# ✅ DUAL SESSION INITIALIZATION
# ============================================================================

@performance_monitor.measure("telegram_init")
async def init_telegram_sessions():
    """Initialize Telegram sessions"""
    global User, Bot, user_session_ready, bot_session_ready
    
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
                sleep_threshold=30,
                in_memory=True,
                no_updates=True
            )
            
            await User.start()
            me = await User.get_me()
            logger.info(f"✅ USER Session Ready: {me.first_name}")
            
            # Test channel access
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
            if User is not None:
                try:
                    await User.stop()
                except:
                    pass
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
            if Bot is not None:
                try:
                    await Bot.stop()
                except:
                    pass
            Bot = None
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 SESSION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"USER Session: {'✅ READY' if user_session_ready else '❌ NOT READY'}")
    logger.info(f"BOT Session: {'✅ READY' if bot_session_ready else '❌ NOT READY'}")
    logger.info(f"Bot Handler: {'✅ INITIALIZED' if bot_handler.initialized else '❌ NOT READY'}")
    
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
# ✅ MAIN INITIALIZATION
# ============================================================================

@performance_monitor.measure("system_init")
async def init_system():
    start_time = time.time()
    
    try:
        logger.info("=" * 60)
        logger.info("🚀 SK4FiLM v9.0 - OPTIMIZED STREAMING & DOWNLOAD SYSTEM")
        logger.info("=" * 60)
        
        # Initialize MongoDB
        mongo_ok = await init_mongodb()
        if not mongo_ok:
            logger.error("❌ MongoDB connection failed")
            return False
        
        # Get current file count
        if files_col is not None:
            file_count = await files_col.count_documents({})
            logger.info(f"📊 Current files in database: {file_count}")
        
        # Initialize Bot Handler
        bot_handler_ok = await bot_handler.initialize()
        if bot_handler_ok:
            logger.info("✅ Bot Handler initialized")
        
        # Initialize Cache Manager
        global cache_manager, verification_system, premium_system, poster_fetcher
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
        
        # Initialize Poster Fetcher (fallback to our quick version)
        if PosterFetcher is not None:
            poster_fetcher = PosterFetcher(Config, cache_manager)
            logger.info("✅ Poster Fetcher initialized")
        else:
            logger.info("✅ Using optimized poster fetcher")
        
        # Initialize Telegram Sessions
        if PYROGRAM_AVAILABLE:
            telegram_ok = await init_telegram_sessions()
            if not telegram_ok:
                logger.warning("⚠️ Telegram sessions failed")
        
        # Start initial indexing
        if user_session_ready and files_col is not None:
            logger.info("🔄 Starting optimized file channel indexing...")
            asyncio.create_task(initial_indexing())
        
        init_time = time.time() - start_time
        logger.info(f"⚡ SK4FiLM Started in {init_time:.2f}s")
        logger.info("=" * 60)
        
        logger.info("🔧 INTEGRATED FEATURES:")
        logger.info(f"   • Telegram File Format: ✅ {Config.TELEGRAM_FILE_FORMAT}")
        logger.info(f"   • Real Message IDs: ✅ ENABLED")
        logger.info(f"   • File Channel Indexing: ✅ ENABLED")
        logger.info(f"   • Complete History: {'✅ ENABLED' if Config.INDEX_ALL_HISTORY else '❌ DISABLED'}")
        logger.info(f"   • Duplicate Prevention: ✅ ENABLED")
        logger.info(f"   • Cache System: {'✅ ENABLED' if cache_manager else '❌ DISABLED'}")
        logger.info(f"   • Poster Fetcher: ✅ OPTIMIZED")
        logger.info(f"   • Quality Merging: ✅ ENABLED")
        logger.info(f"   • User Session: {'✅ READY' if user_session_ready else '❌ NOT READY'}")
        logger.info(f"   • Bot Session: {'✅ READY' if bot_session_ready else '❌ NOT READY'}")
        logger.info(f"   • Bot Handler: {'✅ READY' if bot_handler.initialized else '❌ NOT READY'}")
        logger.info(f"   • Streaming: {'✅ ENABLED' if Config.STREAMING_ENABLED else '❌ DISABLED'}")
        logger.info(f"   • Direct Download: {'✅ ENABLED' if Config.DIRECT_DOWNLOAD_ENABLED else '❌ DISABLED'}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ System initialization failed: {e}")
        return False

# ============================================================================
# ✅ OPTIMIZED SEARCH FUNCTION
# ============================================================================

def channel_name_cached(cid):
    return f"Channel {cid}"

@performance_monitor.measure("multi_channel_search_merged")
@async_cache_with_ttl(maxsize=500, ttl=Config.SEARCH_CACHE_TTL)
async def search_movies_multi_channel_merged(query, limit=10, page=1):
    """OPTIMIZED: Fast search with MongoDB text index and pagination"""
    offset = (page - 1) * limit
    
    # Validate query
    query = query.strip()
    if len(query) < Config.SEARCH_MIN_QUERY_LENGTH:
        return {
            'results': [],
            'pagination': {'current_page': page, 'total_pages': 0, 'total_results': 0},
            'search_metadata': {'query': query, 'stats': {}}
        }
    
    # Try cache first
    cache_key = f"search_merged:{query}:{page}:{limit}"
    if cache_manager is not None and cache_manager.redis_enabled:
        cached_data = await cache_manager.get(cache_key)
        if cached_data:
            logger.info(f"✅ Cache HIT for: {query}")
            return cached_data
    
    logger.info(f"🔍 SEARCHING for: {query} (page {page}, limit {limit})")
    
    start_time = time.time()
    query_lower = query.lower()
    files_dict = {}
    
    # ============================================================================
    # ✅ 1. SEARCH FILE CHANNEL DATABASE USING TEXT INDEX
    # ============================================================================
    if files_col is not None:
        try:
            # Use MongoDB text search for better performance
            if Config.USE_MONGODB_TEXT_INDEX:
                search_query = {
                    "$text": {"$search": query},
                    "status": "active",
                    "is_duplicate": False
                }
                
                # Get total count for pagination
                total_count = await files_col.count_documents(search_query)
                total_count = min(total_count, Config.MAX_SEARCH_RESULTS)
                
                # Get paginated results
                cursor = files_col.find(
                    search_query,
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
                        'telegram_file_id': 1,
                        'thumbnail_url': 1,
                        'thumbnail_extracted': 1,
                        'year': 1,
                        'duration': 1,
                        '_id': 0
                    }
                ).skip(offset).limit(limit)
                
                file_count = 0
                
                async for doc in cursor:
                    file_count += 1
                    try:
                        title = doc.get('title', 'Unknown')
                        norm_title = normalize_title(title)
                        
                        # Extract quality
                        quality = doc.get('quality', detect_quality_enhanced(doc.get('file_name', '')))
                        
                        # Create file_id in Telegram format
                        file_id = f"{doc.get('channel_id', Config.FILE_CHANNEL_ID)}_{doc.get('message_id')}_{quality}"
                        
                        # Get thumbnail URL
                        thumbnail_url = doc.get('thumbnail_url')
                        
                        # Create quality option
                        quality_option = {
                            'file_id': file_id,
                            'file_size': doc.get('file_size', 0),
                            'file_name': doc.get('file_name', ''),
                            'is_video': doc.get('is_video_file', False),
                            'channel_id': doc.get('channel_id'),
                            'message_id': doc.get('message_id'),
                            'quality': quality,
                            'thumbnail_url': thumbnail_url,
                            'has_thumbnail': thumbnail_url is not None,
                            'date': doc.get('date'),
                            'telegram_file_id': doc.get('telegram_file_id'),
                            'duration': doc.get('duration', 0),
                            'duration_formatted': streaming_proxy.format_duration(doc.get('duration', 0))
                        }
                        
                        # Add to files_dict
                        if norm_title not in files_dict:
                            files_dict[norm_title] = {
                                'title': title,
                                'original_title': title,
                                'normalized_title': norm_title,
                                'content': format_post(doc.get('caption', ''), max_length=200),
                                'post_content': doc.get('caption', ''),
                                'quality_options': {quality: quality_option},
                                'quality_list': [quality],
                                'date': doc.get('date'),
                                'is_new': is_new(doc.get('date')),
                                'is_video_file': doc.get('is_video_file', False),
                                'channel_id': doc.get('channel_id'),
                                'channel_name': channel_name_cached(doc.get('channel_id')),
                                'has_file': True,
                                'has_post': bool(doc.get('caption')),
                                'file_caption': doc.get('caption', ''),
                                'year': doc.get('year', ''),
                                'quality': quality,
                                'has_thumbnail': thumbnail_url is not None,
                                'thumbnail_url': thumbnail_url,
                                'real_message_id': doc.get('message_id'),
                                'search_score': 3 if query_lower in title.lower() else 2,
                                'result_type': 'file' if not doc.get('caption') else 'both',
                                'total_files': 1,
                                'file_sizes': [doc.get('file_size', 0)],
                                'bot_username': Config.BOT_USERNAME,
                                'streaming_enabled': Config.STREAMING_ENABLED,
                                'direct_download_enabled': Config.DIRECT_DOWNLOAD_ENABLED,
                                'duration': doc.get('duration', 0),
                                'duration_formatted': streaming_proxy.format_duration(doc.get('duration', 0)),
                                'best_file_id': file_id,
                                'streaming_file_id': file_id,
                                'view_page_url': f"/view/{file_id}"
                            }
                        else:
                            # Merge qualities
                            existing = files_dict[norm_title]
                            if quality not in existing['quality_options']:
                                existing['quality_options'][quality] = quality_option
                                existing['quality_list'].append(quality)
                                existing['total_files'] += 1
                                existing['file_sizes'].append(doc.get('file_size', 0))
                            
                            # Update best file_id for streaming
                            existing['best_file_id'] = file_id
                            existing['streaming_file_id'] = file_id
                            existing['view_page_url'] = f"/view/{file_id}"
                            
                    except Exception as e:
                        logger.debug(f"File processing error: {e}")
                        continue
                
                logger.info(f"✅ Found {file_count} files using text index")
                
            else:
                # Fallback to regex search
                search_query = {
                    "$or": [
                        {"normalized_title": {"$regex": query, "$options": "i"}},
                        {"title": {"$regex": query, "$options": "i"}}
                    ],
                    "status": "active",
                    "is_duplicate": False
                }
                
                total_count = await files_col.count_documents(search_query)
                total_count = min(total_count, Config.MAX_SEARCH_RESULTS)
                
                cursor = files_col.find(
                    search_query,
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
                        'telegram_file_id': 1,
                        'thumbnail_url': 1,
                        'thumbnail_extracted': 1,
                        'year': 1,
                        'duration': 1,
                        '_id': 0
                    }
                ).skip(offset).limit(limit)
                
                file_count = 0
                
                async for doc in cursor:
                    file_count += 1
                    try:
                        title = doc.get('title', 'Unknown')
                        norm_title = normalize_title(title)
                        
                        # Extract quality
                        quality = doc.get('quality', detect_quality_enhanced(doc.get('file_name', '')))
                        
                        # Create file_id in Telegram format
                        file_id = f"{doc.get('channel_id', Config.FILE_CHANNEL_ID)}_{doc.get('message_id')}_{quality}"
                        
                        # Get thumbnail URL
                        thumbnail_url = doc.get('thumbnail_url')
                        
                        # Create quality option
                        quality_option = {
                            'file_id': file_id,
                            'file_size': doc.get('file_size', 0),
                            'file_name': doc.get('file_name', ''),
                            'is_video': doc.get('is_video_file', False),
                            'channel_id': doc.get('channel_id'),
                            'message_id': doc.get('message_id'),
                            'quality': quality,
                            'thumbnail_url': thumbnail_url,
                            'has_thumbnail': thumbnail_url is not None,
                            'date': doc.get('date'),
                            'telegram_file_id': doc.get('telegram_file_id'),
                            'duration': doc.get('duration', 0),
                            'duration_formatted': streaming_proxy.format_duration(doc.get('duration', 0))
                        }
                        
                        # Add to files_dict
                        if norm_title not in files_dict:
                            files_dict[norm_title] = {
                                'title': title,
                                'original_title': title,
                                'normalized_title': norm_title,
                                'content': format_post(doc.get('caption', ''), max_length=200),
                                'post_content': doc.get('caption', ''),
                                'quality_options': {quality: quality_option},
                                'quality_list': [quality],
                                'date': doc.get('date'),
                                'is_new': is_new(doc.get('date')),
                                'is_video_file': doc.get('is_video_file', False),
                                'channel_id': doc.get('channel_id'),
                                'channel_name': channel_name_cached(doc.get('channel_id')),
                                'has_file': True,
                                'has_post': bool(doc.get('caption')),
                                'file_caption': doc.get('caption', ''),
                                'year': doc.get('year', ''),
                                'quality': quality,
                                'has_thumbnail': thumbnail_url is not None,
                                'thumbnail_url': thumbnail_url,
                                'real_message_id': doc.get('message_id'),
                                'search_score': 3 if query_lower in title.lower() else 2,
                                'result_type': 'file' if not doc.get('caption') else 'both',
                                'total_files': 1,
                                'file_sizes': [doc.get('file_size', 0)],
                                'bot_username': Config.BOT_USERNAME,
                                'streaming_enabled': Config.STREAMING_ENABLED,
                                'direct_download_enabled': Config.DIRECT_DOWNLOAD_ENABLED,
                                'duration': doc.get('duration', 0),
                                'duration_formatted': streaming_proxy.format_duration(doc.get('duration', 0)),
                                'best_file_id': file_id,
                                'streaming_file_id': file_id,
                                'view_page_url': f"/view/{file_id}"
                            }
                        else:
                            # Merge qualities
                            existing = files_dict[norm_title]
                            if quality not in existing['quality_options']:
                                existing['quality_options'][quality] = quality_option
                                existing['quality_list'].append(quality)
                                existing['total_files'] += 1
                                existing['file_sizes'].append(doc.get('file_size', 0))
                            
                            # Update best file_id for streaming
                            existing['best_file_id'] = file_id
                            existing['streaming_file_id'] = file_id
                            existing['view_page_url'] = f"/view/{file_id}"
                            
                    except Exception as e:
                        logger.debug(f"File processing error: {e}")
                        continue
                
                logger.info(f"✅ Found {file_count} files using regex search")
            
        except Exception as e:
            logger.error(f"❌ File search error: {e}")
            total_count = 0
    
    # ============================================================================
    # ✅ 2. PROCESS QUALITY OPTIONS FOR EACH TITLE
    # ============================================================================
    results_list = list(files_dict.values())
    
    # Sort results
    results_list.sort(key=lambda x: (
        x.get('has_file', False),
        len(x.get('quality_options', {})),
        x.get('search_score', 0),
        x.get('is_new', False)
    ), reverse=True)
    
    # ============================================================================
    # ✅ 3. FETCH POSTERS FOR RESULTS
    # ============================================================================
    if results_list:
        # Only fetch posters for visible results
        results_with_posters = await get_posters_for_movies_batch_quick(results_list)
        results_list = results_with_posters
    
    # ============================================================================
    # ✅ 4. ENHANCE RESULTS FOR DISPLAY
    # ============================================================================
    enhanced_results = []
    for result in results_list:
        enhanced_result = result.copy()
        
        # Determine display type based on what's available
        if result.get('has_file') and result.get('has_post'):
            # ✅ Post with File (Same Title)
            enhanced_result['display_type'] = 'post_with_file'
            enhanced_result['download_message'] = 'Download in Player'
            enhanced_result['stream_button_enabled'] = True
            enhanced_result['download_button_enabled'] = False
            enhanced_result['view_page_required'] = True
            
        elif result.get('has_file') and not result.get('has_post'):
            # ✅ File Only (No Post)
            enhanced_result['display_type'] = 'file_only'
            enhanced_result['download_message'] = 'Download in Player'
            enhanced_result['stream_button_enabled'] = True
            enhanced_result['download_button_enabled'] = False
            enhanced_result['view_page_required'] = True
            
        elif result.get('has_post') and not result.get('has_file'):
            # ✅ Post Only (No File)
            enhanced_result['display_type'] = 'post_only'
            enhanced_result['download_message'] = 'Post Only - No File'
            enhanced_result['stream_button_enabled'] = False
            enhanced_result['download_button_enabled'] = False
            enhanced_result['view_page_required'] = False
            
        else:
            # Fallback
            enhanced_result['display_type'] = 'unknown'
            enhanced_result['download_message'] = 'No file available'
            enhanced_result['stream_button_enabled'] = False
            enhanced_result['download_button_enabled'] = False
            enhanced_result['view_page_required'] = False
        
        enhanced_results.append(enhanced_result)
    
    # Calculate pagination
    total_pages = math.ceil(total_count / limit) if total_count > 0 else 1
    
    # Final data structure
    result_data = {
        'results': enhanced_results,
        'pagination': {
            'current_page': page,
            'total_pages': total_pages,
            'total_results': total_count,
            'per_page': limit,
            'has_next': page < total_pages if total_count > 0 else False,
            'has_previous': page > 1
        },
        'search_metadata': {
            'query': query,
            'stats': {
                'total': total_count,
                'with_files': len([r for r in enhanced_results if r.get('has_file')]),
                'with_posts': len([r for r in enhanced_results if r.get('has_post')]),
                'streaming_enabled': Config.STREAMING_ENABLED,
                'direct_download_enabled': Config.DIRECT_DOWNLOAD_ENABLED
            },
            'quality_merging': True,
            'real_message_ids': True,
            'cache_hit': False,
            'streaming_enabled': Config.STREAMING_ENABLED,
            'direct_download_enabled': Config.DIRECT_DOWNLOAD_ENABLED,
            'bot_username': Config.BOT_USERNAME
        },
        'bot_username': Config.BOT_USERNAME
    }
    
    # Cache results
    if cache_manager is not None:
        await cache_manager.set(cache_key, result_data, expire_seconds=Config.SEARCH_CACHE_TTL)
    
    elapsed = time.time() - start_time
    logger.info(f"✅ Search complete: {len(enhanced_results)} results in {elapsed:.2f}s")
    
    return result_data

# ============================================================================
# ✅ HOME MOVIES - OPTIMIZED
# ============================================================================

@performance_monitor.measure("home_movies")
@async_cache_with_ttl(maxsize=1, ttl=60)
async def get_home_movies(limit=20):
    """Get home movies - optimized"""
    try:
        if User is None or not user_session_ready:
            return []
        
        movies = []
        seen_titles = set()
        
        logger.info(f"🎬 Fetching home movies ({limit})...")
        
        async for msg in User.get_chat_history(Config.MAIN_CHANNEL_ID, limit=limit * 2):
            if msg is not None and msg.text and len(msg.text) > 25:
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
                    formatted_content = format_post(msg.text, max_length=200)
                    
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
                        'is_video_file': False,
                        'streaming_enabled': Config.STREAMING_ENABLED,
                        'direct_download_enabled': Config.DIRECT_DOWNLOAD_ENABLED,
                        'display_type': 'post_only',
                        'download_message': 'Post Only - No File',
                        'stream_button_enabled': False,
                        'download_button_enabled': False,
                        'view_page_required': False
                    }
                    
                    movies.append(movie_data)
                    
                    if len(movies) >= limit:
                        break
        
        # Fetch posters
        if movies:
            movies_with_posters = await get_posters_for_movies_batch_quick(movies)
            logger.info(f"✅ Fetched {len(movies_with_posters)} home movies")
            return movies_with_posters[:limit]
        else:
            logger.warning("⚠️ No movies found for home page")
            return []
        
    except Exception as e:
        logger.error(f"❌ Home movies error: {e}")
        return []

# ============================================================================
# ✅ API ROUTES
# ============================================================================

@app.route('/')
@performance_monitor.measure("root_endpoint")
async def root():
    if files_col is not None:
        tf = await files_col.count_documents({})
        video_files = await files_col.count_documents({'is_video_file': True})
        thumbnails_extracted = await files_col.count_documents({'thumbnail_extracted': True})
    else:
        tf = 0
        video_files = 0
        thumbnails_extracted = 0
    
    # Get indexing status
    indexing_status = await file_indexing_manager.get_indexing_status()
    
    # Get bot handler status
    bot_status = await bot_handler.get_bot_status() if bot_handler else None
    
    return jsonify({
        'status': 'healthy',
        'service': 'SK4FiLM v9.0 - OPTIMIZED STREAMING & DOWNLOAD',
        'sessions': {
            'user_session': {
                'ready': user_session_ready,
                'channels': Config.TEXT_CHANNEL_IDS
            },
            'bot_session': {
                'ready': bot_session_ready,
                'channel': Config.FILE_CHANNEL_ID
            },
            'bot_handler': bot_status
        },
        'components': {
            'cache': cache_manager is not None,
            'verification': verification_system is not None,
            'premium': premium_system is not None,
            'database': files_col is not None,
            'bot_handler': bot_handler is not None and bot_handler.initialized,
            'streaming_proxy': Config.STREAMING_ENABLED
        },
        'features': {
            'telegram_file_format': Config.TELEGRAM_FILE_FORMAT,
            'real_message_ids': True,
            'file_channel_indexing': True,
            'complete_history': Config.INDEX_ALL_HISTORY,
            'instant_indexing': Config.INSTANT_AUTO_INDEX,
            'duplicate_prevention': True,
            'quality_merging': True,
            'thumbnail_extraction': True,
            'streaming': Config.STREAMING_ENABLED,
            'direct_download': Config.DIRECT_DOWNLOAD_ENABLED
        },
        'stats': {
            'total_files': tf,
            'video_files': video_files,
            'thumbnails_extracted': thumbnails_extracted
        },
        'indexing': indexing_status,
        'response_time': f"{time.perf_counter():.3f}s"
    })

@app.route('/health')
@performance_monitor.measure("health_endpoint")
async def health():
    indexing_status = await file_indexing_manager.get_indexing_status()
    bot_status = await bot_handler.get_bot_status() if bot_handler else None
    
    return jsonify({
        'status': 'ok',
        'sessions': {
            'user': user_session_ready,
            'bot': bot_session_ready,
            'bot_handler': bot_status.get('initialized') if bot_status else False
        },
        'indexing': {
            'running': indexing_status['is_running'],
            'last_run': indexing_status['last_run']
        },
        'streaming': {
            'enabled': Config.STREAMING_ENABLED,
            'proxy_url': Config.STREAMING_PROXY_URL
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/movies', methods=['GET'])
@performance_monitor.measure("movies_endpoint")
async def api_movies():
    try:
        # Get home movies
        movies = await get_home_movies(limit=20)
        
        return jsonify({
            'status': 'success' if movies else 'empty',
            'movies': movies,
            'total': len(movies),
            'limit': 20,
            'source': 'telegram',
            'poster_fetcher': True,
            'session_used': 'user',
            'channel_id': Config.MAIN_CHANNEL_ID,
            'streaming_enabled': Config.STREAMING_ENABLED,
            'direct_download_enabled': Config.DIRECT_DOWNLOAD_ENABLED,
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
    try:
        query = request.args.get('query', '').strip()
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', Config.SEARCH_RESULTS_PER_PAGE))
        
        if len(query) < Config.SEARCH_MIN_QUERY_LENGTH:
            return jsonify({
                'status': 'error',
                'message': f'Query must be at least {Config.SEARCH_MIN_QUERY_LENGTH} characters'
            }), 400
        
        result_data = await search_movies_multi_channel_merged(query, limit, page)
        
        return jsonify({
            'status': 'success',
            'query': query,
            'results': result_data['results'],
            'pagination': result_data['pagination'],
            'search_metadata': result_data['search_metadata'],
            'bot_username': Config.BOT_USERNAME,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Search API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/view/<file_id>', methods=['GET'])
@performance_monitor.measure("view_endpoint")
async def api_view(file_id):
    """Get detailed file information for view page"""
    try:
        logger.info(f"🔍 Getting view page data for: {file_id}")
        
        # Get enhanced file info
        file_info = await get_enhanced_file_info(file_id)
        
        if file_info.get('status') == 'error':
            return jsonify({
                'status': 'error',
                'message': file_info.get('message', 'File not found')
            }), 404
        
        return jsonify({
            'status': 'success',
            'file_info': file_info,
            'streaming_enabled': Config.STREAMING_ENABLED,
            'direct_download_enabled': Config.DIRECT_DOWNLOAD_ENABLED,
            'bot_username': Config.BOT_USERNAME,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"View API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/stream/info/<file_id>', methods=['GET'])
@performance_monitor.measure("stream_info")
async def get_stream_info(file_id):
    """Get streaming information for a file"""
    try:
        logger.info(f"🔍 Getting stream info for: {file_id}")
        
        # Get file info from database
        file_info = await streaming_proxy.get_file_info(file_id)
        if not file_info:
            return jsonify({
                'status': 'error',
                'message': 'File not found'
            }), 404
        
        # Get streaming URL
        stream_url = None
        if Config.STREAMING_ENABLED and file_info.get('is_video_file'):
            quality = file_info.get('quality', 'auto')
            stream_url = await streaming_proxy.get_stream_url(file_id, quality)
        
        # Get download URL
        download_info = await streaming_proxy.get_direct_download_url(file_id)
        
        return jsonify({
            'status': 'success',
            'file_id': file_id,
            'title': file_info.get('title', ''),
            'file_name': file_info.get('file_name', ''),
            'file_size': file_info.get('file_size', 0),
            'size_formatted': format_size(file_info.get('file_size', 0)),
            'quality': file_info.get('quality', '480p'),
            'duration': file_info.get('duration', 0),
            'duration_formatted': file_info.get('duration_formatted', 'Unknown'),
            'thumbnail_url': file_info.get('thumbnail_url'),
            'streaming_enabled': Config.STREAMING_ENABLED,
            'stream_url': stream_url,
            'download_enabled': Config.DIRECT_DOWNLOAD_ENABLED,
            'download_info': download_info,
            'telegram_bot_url': f"https://t.me/{Config.BOT_USERNAME}?start={file_id}",
            'bot_username': Config.BOT_USERNAME,
            'telegram_file_format': file_id  # This is the format: -1001768249569_16066_480p
        })
        
    except Exception as e:
        logger.error(f"❌ Stream info error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/stream/url/<file_id>', methods=['GET'])
@performance_monitor.measure("stream_url")
async def get_stream_url(file_id):
    """Get direct streaming URL"""
    try:
        quality = request.args.get('quality', 'auto')
        
        if not Config.STREAMING_ENABLED:
            return jsonify({
                'status': 'error',
                'message': 'Streaming is disabled'
            }), 403
        
        stream_url = await streaming_proxy.get_stream_url(file_id, quality)
        
        if stream_url:
            return jsonify({
                'status': 'success',
                'stream_url': stream_url,
                'quality': quality,
                'file_id': file_id
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Could not get streaming URL'
            }), 404
            
    except Exception as e:
        logger.error(f"❌ Stream URL error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/download/info/<file_id>', methods=['GET'])
@performance_monitor.measure("download_info")
async def get_download_info(file_id):
    """Get download information"""
    try:
        if not Config.DIRECT_DOWNLOAD_ENABLED:
            return jsonify({
                'status': 'error',
                'message': 'Direct download is disabled'
            }), 403
        
        download_info = await streaming_proxy.get_direct_download_url(file_id)
        
        if download_info:
            return jsonify({
                'status': 'success',
                'download_info': download_info,
                'file_id': file_id
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Could not get download URL'
            }), 404
            
    except Exception as e:
        logger.error(f"❌ Download info error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# ============================================================================
# ✅ ADMIN API ROUTES
# ============================================================================

@app.route('/api/admin/reindex', methods=['POST'])
async def api_admin_reindex():
    """Admin endpoint to trigger reindexing"""
    try:
        auth_token = request.headers.get('X-Admin-Token')
        if not auth_token or auth_token != os.environ.get('ADMIN_TOKEN', 'sk4film_admin_123'):
            return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401
        
        # Trigger reindexing
        asyncio.create_task(initial_indexing())
        
        return jsonify({
            'status': 'success',
            'message': 'File channel reindexing started',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Admin reindex error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/admin/indexing-status', methods=['GET'])
async def api_admin_indexing_status():
    """Check indexing status"""
    try:
        indexing_status = await file_indexing_manager.get_indexing_status()
        
        # Get database stats
        if files_col is not None:
            total_files = await files_col.count_documents({})
        else:
            total_files = 0
        
        return jsonify({
            'status': 'success',
            'indexing': indexing_status,
            'database_files': total_files,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Indexing status error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/admin/clear-cache', methods=['POST'])
async def api_admin_clear_cache():
    """Clear all cache"""
    try:
        auth_token = request.headers.get('X-Admin-Token')
        if not auth_token or auth_token != os.environ.get('ADMIN_TOKEN', 'sk4film_admin_123'):
            return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401
        
        if cache_manager and cache_manager.redis_enabled:
            try:
                # Clear all cache keys starting with "search_"
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
    
    # Stop indexing
    await file_indexing_manager.stop_indexing()
    await channel_sync_manager.stop_sync_monitoring()
    
    # Shutdown bot handler
    if bot_handler:
        await bot_handler.shutdown()
    
    # Close streaming proxy session
    await streaming_proxy.close()
    
    # Close poster fetcher session
    if poster_fetcher is not None and hasattr(poster_fetcher, 'close'):
        try:
            await poster_fetcher.close()
        except:
            pass
    
    # Close Telegram sessions
    if User is not None:
        shutdown_tasks.append(User.stop())
    
    if Bot is not None:
        shutdown_tasks.append(Bot.stop())
    
    # Close cache manager
    if cache_manager is not None:
        shutdown_tasks.append(cache_manager.stop())
    
    # Close verification system
    if verification_system is not None:
        shutdown_tasks.append(verification_system.stop())
    
    # Close premium system
    if premium_system is not None and hasattr(premium_system, 'stop_cleanup_task'):
        shutdown_tasks.append(premium_system.stop_cleanup_task())
    
    # Execute all shutdown tasks
    if shutdown_tasks:
        await asyncio.gather(*shutdown_tasks, return_exceptions=True)
    
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
    
    logger.info(f"🌐 Starting SK4FiLM v9.0 on port {Config.WEB_SERVER_PORT}...")
    logger.info("🎯 FEATURES: OPTIMIZED STREAMING & DOWNLOAD SYSTEM")
    logger.info(f"   • File Channel ID: {Config.FILE_CHANNEL_ID}")
    logger.info(f"   • Telegram File Format: {Config.TELEGRAM_FILE_FORMAT}")
    logger.info(f"   • Example: -1001768249569_16066_480p")
    logger.info(f"   • Real Message IDs: ✅ ENABLED")
    logger.info(f"   • Streaming: {'✅ ENABLED' if Config.STREAMING_ENABLED else '❌ DISABLED'}")
    logger.info(f"   • Direct Download: {'✅ ENABLED' if Config.DIRECT_DOWNLOAD_ENABLED else '❌ DISABLED'}")
    logger.info(f"   • Streaming Proxy: {Config.STREAMING_PROXY_URL}")
    logger.info(f"   • Bot Username: @{Config.BOT_USERNAME}")
    logger.info(f"   • Multi-Quality Merging: ✅ FIXED")
    logger.info(f"   • Single Title Results: ✅ ENABLED")
    logger.info(f"   • View Page API: ✅ READY")
    logger.info(f"   • Enhanced Search Results: ✅ IMPLEMENTED")
    logger.info(f"   • Search Results per Page: {Config.SEARCH_RESULTS_PER_PAGE}")
    logger.info(f"   • Poster Fetch Timeout: {Config.POSTER_FETCH_TIMEOUT}s")
    logger.info(f"   • Optimized Indexing: ✅ ENABLED")
    
    asyncio.run(serve(app, config))
