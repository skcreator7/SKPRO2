# ============================================================================
# 🚀 SK4FiLM v8.3 - COMPLETE INTEGRATED SYSTEM WITH AUTO INDEXING & DUPLICATE PREVENTION
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
    CacheManager = None

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
    def normalize_title(title): return title.lower().strip() if title else ""
    def extract_title_smart(text): return text[:50] if text else ""
    def extract_title_from_file(filename, caption=None): return filename or "Unknown"
    def format_size(size): return f"{size/1024/1024:.1f} MB" if size else "Unknown"
    def detect_quality(filename): return "480p"
    def is_video_file(filename): return bool(filename)
    def format_post(text, max_length=None): 
        if max_length and text and len(text) > max_length:
            text = text[:max_length] + "..."
        return text or ""
    def is_new(date): return False

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
    
    # Thumbnail Settings
    THUMBNAIL_EXTRACT_TIMEOUT = 10  # seconds
    THUMBNAIL_CACHE_DURATION = 24 * 60 * 60  # 24 hours
    
    # Auto Indexing Settings
    AUTO_INDEX_INTERVAL = int(os.environ.get("AUTO_INDEX_INTERVAL", "3600"))  # 1 hour
    BATCH_INDEX_SIZE = int(os.environ.get("BATCH_INDEX_SIZE", "100"))  # Messages per batch
    MAX_INDEX_LIMIT = int(os.environ.get("MAX_INDEX_LIMIT", "1000"))  # Max messages to check

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
    response.headers['X-SK4FiLM-Version'] = '8.3-AUTO-INDEXING-DUPLICATE-PREVENTION'
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
indexing_col = None  # New collection for indexing state

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

# Indexing State
is_indexing = False
last_index_time = None
indexing_task = None

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
                'message_id': option.get('message_id'),
                'thumbnail_url': option.get('thumbnail_url')  # ✅ Add thumbnail URL
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
# ✅ VIDEO THUMBNAIL EXTRACTOR (ACTUAL EXTRACTION FROM TELEGRAM)
# ============================================================================

class VideoThumbnailExtractor:
    """Extract thumbnails directly from Telegram video files"""
    
    def __init__(self):
        self.thumbnail_cache = {}  # Cache for extracted thumbnails
        self.extraction_lock = asyncio.Lock()
    
    async def extract_thumbnail_from_message(self, channel_id, message_id, file_id=None):
        """
        Extract thumbnail from Telegram video message
        Returns direct Telegram thumbnail URL or None
        """
        try:
            if Bot is None or not bot_session_ready:
                logger.warning("❌ Bot session not ready for thumbnail extraction")
                return None
            
            # Create cache key
            cache_key = f"thumbnail_{channel_id}_{message_id}"
            
            # Check cache first
            if cache_manager and cache_manager.redis_enabled:
                cached = await cache_manager.get(cache_key)
                if cached:
                    logger.debug(f"✅ Thumbnail cache hit: {cache_key}")
                    return cached
            
            logger.info(f"🔍 Extracting thumbnail from: {channel_id}/{message_id}")
            
            # Get message using BOT session
            try:
                message = await Bot.get_messages(channel_id, message_id)
                
                if not message:
                    logger.warning(f"❌ Message not found: {channel_id}/{message_id}")
                    return None
                
                # Check if message has video or document
                thumbnail_url = None
                
                if message.video:
                    # Video messages have thumbnails
                    if hasattr(message.video, 'thumbnail') and message.video.thumbnail:
                        # Get thumbnail file ID
                        thumbnail_file_id = message.video.thumbnail.file_id
                        thumbnail_url = await self._get_telegram_file_url(thumbnail_file_id)
                    
                    # If no thumbnail in video object, try to download video and extract frame
                    if not thumbnail_url:
                        thumbnail_url = await self._extract_frame_from_video(message.video.file_id)
                
                elif message.document and is_video_file(message.document.file_name or ''):
                    # Video document - try to get thumbnail
                    if hasattr(message.document, 'thumbnail') and message.document.thumbnail:
                        thumbnail_file_id = message.document.thumbnail.file_id
                        thumbnail_url = await self._get_telegram_file_url(thumbnail_file_id)
                    
                    # If no thumbnail, try to download and extract frame
                    if not thumbnail_url:
                        thumbnail_url = await self._extract_frame_from_video(message.document.file_id)
                
                # Cache the result
                if thumbnail_url and cache_manager and cache_manager.redis_enabled:
                    await cache_manager.set(
                        cache_key, 
                        thumbnail_url, 
                        expire_seconds=Config.THUMBNAIL_CACHE_DURATION
                    )
                    logger.debug(f"✅ Thumbnail cached: {cache_key}")
                
                return thumbnail_url
                
            except Exception as e:
                logger.error(f"❌ Error extracting thumbnail: {e}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Thumbnail extraction failed: {e}")
            return None
    
    async def _get_telegram_file_url(self, file_id):
        """
        Get direct URL for Telegram file (thumbnail)
        """
        try:
            # Get file from Telegram
            file = await Bot.get_file(file_id)
            
            if not file:
                return None
            
            # Construct download path
            file_path = file.file_path
            
            if not file_path:
                # Generate path if not provided
                file_path = f"downloads/{file_id}"
            
            # Download file
            download_path = await Bot.download_media(file, in_memory=True)
            
            if not download_path:
                return None
            
            # For now, return a placeholder or process the thumbnail
            # In production, you would upload this to a CDN or return base64
            
            # Convert to base64 for API response
            if isinstance(download_path, bytes):
                # If it's bytes (in_memory download)
                thumbnail_data = download_path
            else:
                # If it's file path
                with open(download_path, 'rb') as f:
                    thumbnail_data = f.read()
            
            # Convert to base64
            base64_data = base64.b64encode(thumbnail_data).decode('utf-8')
            
            # Return data URL
            return f"data:image/jpeg;base64,{base64_data}"
            
        except Exception as e:
            logger.error(f"❌ Error getting Telegram file URL: {e}")
            return None
    
    async def _extract_frame_from_video(self, file_id, time_offset=5):
        """
        Extract a frame from video file at specific time offset
        """
        try:
            # Download video file temporarily
            temp_path = f"temp_{file_id}_{int(time.time())}.mp4"
            
            # Download using Bot session
            download_path = await Bot.download_media(
                file_id,
                file_name=temp_path
            )
            
            if not download_path:
                return None
            
            # Use ffmpeg to extract frame
            import subprocess
            import tempfile
            
            # Create temporary file for thumbnail
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                thumbnail_path = temp_file.name
            
            # Extract frame using ffmpeg
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', download_path,
                '-ss', str(time_offset),  # Time offset in seconds
                '-vframes', '1',
                '-q:v', '2',
                '-y',  # Overwrite output file
                thumbnail_path
            ]
            
            try:
                # Run ffmpeg
                result = subprocess.run(
                    ffmpeg_cmd,
                    capture_output=True,
                    timeout=Config.THUMBNAIL_EXTRACT_TIMEOUT
                )
                
                if result.returncode == 0 and os.path.exists(thumbnail_path):
                    # Read thumbnail
                    with open(thumbnail_path, 'rb') as f:
                        thumbnail_data = f.read()
                    
                    # Convert to base64
                    base64_data = base64.b64encode(thumbnail_data).decode('utf-8')
                    
                    # Cleanup
                    try:
                        os.remove(download_path)
                        os.remove(thumbnail_path)
                    except:
                        pass
                    
                    return f"data:image/jpeg;base64,{base64_data}"
                else:
                    logger.error(f"❌ FFmpeg failed: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                logger.error("❌ FFmpeg timeout")
            
            # Cleanup on error
            try:
                if os.path.exists(download_path):
                    os.remove(download_path)
                if os.path.exists(thumbnail_path):
                    os.remove(thumbnail_path)
            except:
                pass
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Frame extraction error: {e}")
            return None
    
    async def extract_thumbnails_batch(self, file_entries):
        """
        Extract thumbnails for multiple files in batch
        """
        results = {}
        
        # Group by channel_id and message_id
        extraction_tasks = []
        
        for entry in file_entries:
            channel_id = entry.get('channel_id')
            message_id = entry.get('message_id')
            file_id = entry.get('file_id')
            
            if channel_id and message_id:
                task = self.extract_thumbnail_from_message(channel_id, message_id, file_id)
                extraction_tasks.append((entry, task))
        
        # Process with concurrency limit
        semaphore = asyncio.Semaphore(5)  # Limit concurrent extractions
        
        async def process_with_semaphore(entry, task):
            async with semaphore:
                thumbnail_url = await task
                return entry, thumbnail_url
        
        # Create tasks with semaphore
        tasks_with_semaphore = [
            process_with_semaphore(entry, task) 
            for entry, task in extraction_tasks
        ]
        
        # Execute all tasks
        for entry, task in extraction_tasks:
            try:
                thumbnail_url = await task
                if thumbnail_url:
                    # Create unique key for this file
                    key = f"{entry.get('channel_id')}_{entry.get('message_id')}"
                    results[key] = thumbnail_url
            except Exception as e:
                logger.error(f"❌ Batch thumbnail error: {e}")
                continue
        
        return results

# Initialize thumbnail extractor
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
            
            # Check for similar files with same title
            if normalized_title and normalized_title in self.title_cache:
                # We have other files with same title, but different hash
                # This is okay - different quality versions
                pass
            
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

# Initialize duplicate prevention
duplicate_prevention = DuplicatePreventionSystem()

# ============================================================================
# ✅ AUTO INDEXING MANAGER
# ============================================================================

class AutoIndexingManager:
    """Automatic batch indexing with duplicate prevention"""
    
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
    
    async def start_auto_indexing(self):
        """Start automatic indexing"""
        if self.is_running:
            logger.warning("⚠️ Auto indexing already running")
            return
        
        logger.info("🚀 Starting AUTO INDEXING system...")
        self.is_running = True
        
        # Initialize duplicate prevention
        await duplicate_prevention.initialize_from_database()
        
        # Start the main loop
        self.indexing_task = asyncio.create_task(self._indexing_loop())
    
    async def stop_auto_indexing(self):
        """Stop automatic indexing"""
        self.is_running = False
        if self.indexing_task:
            self.indexing_task.cancel()
            try:
                await self.indexing_task
            except asyncio.CancelledError:
                pass
        
        logger.info("🛑 Auto indexing stopped")
    
    async def _indexing_loop(self):
        """Main indexing loop"""
        while self.is_running:
            try:
                # Wait for next run
                if self.next_run and self.next_run > datetime.now():
                    wait_seconds = (self.next_run - datetime.now()).total_seconds()
                    logger.info(f"⏰ Next auto index in {wait_seconds:.0f} seconds")
                    await asyncio.sleep(min(wait_seconds, 60))  # Check every minute
                    continue
                
                # Run indexing
                await self._run_indexing_cycle()
                
                # Schedule next run
                self.next_run = datetime.now() + timedelta(seconds=Config.AUTO_INDEX_INTERVAL)
                self.last_run = datetime.now()
                
                # Sleep a bit before checking again
                await asyncio.sleep(10)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Indexing loop error: {e}")
                await asyncio.sleep(60)  # Wait a minute on error
    
    async def _run_indexing_cycle(self):
        """Run one indexing cycle"""
        logger.info("=" * 60)
        logger.info("🔄 STARTING AUTO INDEXING CYCLE")
        logger.info("=" * 60)
        
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
            logger.info(f"📡 Fetching up to {Config.MAX_INDEX_LIMIT} new messages...")
            
            # Fetch new messages in batches
            messages_to_index = []
            fetched_count = 0
            
            async for msg in User.get_chat_history(
                Config.FILE_CHANNEL_ID, 
                limit=Config.MAX_INDEX_LIMIT
            ):
                fetched_count += 1
                
                # Stop if we reach already indexed messages
                if msg.id <= last_message_id:
                    logger.info(f"✅ Reached last indexed message {last_message_id}")
                    break
                
                # Only index file messages
                if msg and (msg.document or msg.video):
                    messages_to_index.append(msg)
                
                # Process in batches
                if len(messages_to_index) >= Config.BATCH_INDEX_SIZE:
                    batch_stats = await self._process_indexing_batch(messages_to_index)
                    cycle_stats['processed'] += batch_stats['processed']
                    cycle_stats['indexed'] += batch_stats['indexed']
                    cycle_stats['duplicates'] += batch_stats['duplicates']
                    cycle_stats['errors'] += batch_stats['errors']
                    
                    messages_to_index = []  # Clear batch
                    await asyncio.sleep(1)  # Small delay between batches
            
            # Process remaining messages
            if messages_to_index:
                batch_stats = await self._process_indexing_batch(messages_to_index)
                cycle_stats['processed'] += batch_stats['processed']
                cycle_stats['indexed'] += batch_stats['indexed']
                cycle_stats['duplicates'] += batch_stats['duplicates']
                cycle_stats['errors'] += batch_stats['errors']
            
            # Update global stats
            self.indexing_stats['total_runs'] += 1
            self.indexing_stats['total_files_processed'] += cycle_stats['processed']
            self.indexing_stats['total_indexed'] += cycle_stats['indexed']
            self.indexing_stats['total_duplicates'] += cycle_stats['duplicates']
            self.indexing_stats['total_errors'] += cycle_stats['errors']
            self.indexing_stats['last_success'] = datetime.now()
            
            elapsed = time.time() - start_time
            
            logger.info("=" * 60)
            logger.info("📊 INDEXING CYCLE COMPLETE")
            logger.info("=" * 60)
            logger.info(f"⏱️  Time: {elapsed:.2f}s")
            logger.info(f"📥 Fetched: {fetched_count} messages")
            logger.info(f"📄 Processed: {cycle_stats['processed']} files")
            logger.info(f"✅ Indexed: {cycle_stats['indexed']} new files")
            logger.info(f"🔄 Duplicates: {cycle_stats['duplicates']} skipped")
            logger.info(f"❌ Errors: {cycle_stats['errors']}")
            logger.info("=" * 60)
            
            # Update global counts
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
                    logger.debug(f"📝 Already indexed by message ID: {msg.id}")
                    batch_stats['duplicates'] += 1
                    continue
                
                # Index the file
                success = await self._index_single_file_with_duplicate_check(msg)
                
                if success:
                    batch_stats['indexed'] += 1
                else:
                    batch_stats['duplicates'] += 1
                    
            except Exception as e:
                logger.error(f"❌ Error processing message {msg.id}: {e}")
                batch_stats['errors'] += 1
                continue
        
        logger.info(f"📦 Batch processed: {batch_stats}")
        return batch_stats
    
    async def _index_single_file_with_duplicate_check(self, message):
        """Index single file with duplicate check"""
        try:
            if not message or (not message.document and not message.video):
                return False
            
            # Generate file hash for duplicate detection
            file_hash = await generate_file_hash(message)
            
            # Extract title for duplicate check
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
            
            # Check for duplicates
            if file_hash:
                is_duplicate, reason = await duplicate_prevention.is_duplicate_file(
                    file_hash, normalized_title
                )
                
                if is_duplicate:
                    logger.info(f"🔄 DUPLICATE SKIPPED: {title} - Reason: {reason}")
                    
                    # Still add to database as duplicate (for tracking)
                    await self._add_duplicate_record(message, title, normalized_title, file_hash, reason)
                    return False
            
            # Extract thumbnail if video file
            thumbnail_url = None
            is_video = False
            
            if message.video or (message.document and is_video_file(file_name or '')):
                is_video = True
                try:
                    thumbnail_url = await thumbnail_extractor.extract_thumbnail_from_message(
                        Config.FILE_CHANNEL_ID,
                        message.id,
                        message.video.file_id if message.video else message.document.file_id
                    )
                    
                    if thumbnail_url:
                        logger.debug(f"✅ Thumbnail extracted for: {title}")
                except Exception as e:
                    logger.error(f"❌ Thumbnail extraction error: {e}")
            
            # Create document
            doc = {
                'channel_id': Config.FILE_CHANNEL_ID,
                'message_id': message.id,
                'title': title,
                'normalized_title': normalized_title,
                'date': message.date,
                'indexed_at': datetime.now(),
                'last_checked': datetime.now(),
                'is_video_file': is_video,
                'file_id': None,
                'file_size': 0,
                'file_hash': file_hash,
                'thumbnail_url': thumbnail_url,
                'thumbnail_extracted': thumbnail_url is not None,
                'status': 'active',
                'is_duplicate': False,
                'duplicate_reason': None
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
                thumbnail_status = "✅" if thumbnail_url else "❌"
                
                logger.info(f"✅ {file_type} AUTO-INDEXED: {title}")
                logger.info(f"   📊 Size: {size_str} | Quality: {doc.get('quality', 'Unknown')}")
                logger.info(f"   🖼️ Thumbnail: {thumbnail_status}")
                logger.info(f"   🔑 Hash: {file_hash[:20] if file_hash else 'None'}")
                
                return True
                
            except Exception as e:
                if "duplicate key error" in str(e).lower():
                    logger.debug(f"📝 Duplicate key error: {message.id}")
                    return False
                else:
                    logger.error(f"❌ Insert error: {e}")
                    return False
        
        except Exception as e:
            logger.error(f"❌ Indexing error for message {message.id if message else 'unknown'}: {e}")
            return False
    
    async def _add_duplicate_record(self, message, title, normalized_title, file_hash, reason):
        """Add duplicate record to database for tracking"""
        try:
            duplicate_doc = {
                'channel_id': Config.FILE_CHANNEL_ID,
                'message_id': message.id,
                'title': title,
                'normalized_title': normalized_title,
                'date': message.date,
                'detected_at': datetime.now(),
                'file_hash': file_hash,
                'duplicate_reason': reason,
                'status': 'duplicate'
            }
            
            # Store in separate duplicates collection
            await db.duplicates.insert_one(duplicate_doc)
            
        except Exception as e:
            logger.error(f"❌ Error adding duplicate record: {e}")
    
    async def get_indexing_status(self):
        """Get current indexing status"""
        return {
            'is_running': self.is_running,
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'next_run': self.next_run.isoformat() if self.next_run else None,
            'total_indexed': self.total_indexed,
            'total_duplicates': self.total_duplicates,
            'stats': self.indexing_stats
        }
    
    async def run_manual_index(self, limit=100):
        """Run manual indexing"""
        logger.info(f"🔧 Running MANUAL indexing (limit: {limit})...")
        
        try:
            # Fetch messages
            messages_to_index = []
            
            async for msg in User.get_chat_history(
                Config.FILE_CHANNEL_ID, 
                limit=limit
            ):
                if msg and (msg.document or msg.video):
                    messages_to_index.append(msg)
            
            # Process batch
            stats = await self._process_indexing_batch(messages_to_index)
            
            logger.info(f"✅ Manual indexing complete: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"❌ Manual indexing error: {e}")
            return None

# Initialize auto indexing manager
auto_indexing_manager = AutoIndexingManager()

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
                    # Check if messages exist using BOT
                    messages = await Bot.get_messages(Config.FILE_CHANNEL_ID, message_ids)
                    
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
    
    async def manual_sync(self):
        await self.sync_deletions_from_telegram()

channel_sync_manager = ChannelSyncManager()

# ============================================================================
# ✅ FILE INDEXING FUNCTIONS - UPDATED WITH DUPLICATE PREVENTION
# ============================================================================

async def generate_file_hash(message):
    """Generate unique hash for file to detect duplicates - ENHANCED"""
    try:
        hash_parts = []
        
        if message.document:
            file_attrs = message.document
            hash_parts.append(f"doc_{file_attrs.file_unique_id}")
            if file_attrs.file_name:
                # Use file size and name for hash
                name_hash = hashlib.md5(file_attrs.file_name.encode()).hexdigest()[:16]
                hash_parts.append(f"name_{name_hash}")
            if file_attrs.file_size:
                hash_parts.append(f"size_{file_attrs.file_size}")
        elif message.video:
            file_attrs = message.video
            hash_parts.append(f"vid_{file_attrs.file_unique_id}")
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

async def index_single_file_smart(message):
    """Index single file using BOT session - Updated with duplicate prevention"""
    try:
        if files_col is None or Bot is None or not bot_session_ready:
            logger.error("❌ Bot session not ready for indexing")
            return False
        
        if not message or (not message.document and not message.video):
            logger.debug(f"❌ Not a file message: {message.id}")
            return False
        
        # Check if already exists by message ID
        existing_by_id = await files_col.find_one({
            'channel_id': Config.FILE_CHANNEL_ID,
            'message_id': message.id
        }, {'_id': 1})
        
        if existing_by_id:
            logger.debug(f"📝 Already indexed: {message.id}")
            return True
        
        # Generate file hash for duplicate detection
        file_hash = await generate_file_hash(message)
        
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
        
        # Check for duplicates using hash
        if file_hash:
            is_duplicate, reason = await duplicate_prevention.is_duplicate_file(
                file_hash, normalized_title
            )
            
            if is_duplicate:
                logger.info(f"🔄 DUPLICATE DETECTED: {title} - Reason: {reason}")
                
                # Add duplicate record for tracking
                await db.duplicates.insert_one({
                    'channel_id': Config.FILE_CHANNEL_ID,
                    'message_id': message.id,
                    'title': title,
                    'normalized_title': normalized_title,
                    'file_hash': file_hash,
                    'duplicate_reason': reason,
                    'detected_at': datetime.now(),
                    'status': 'duplicate'
                })
                
                return False
        
        # Extract thumbnail if video file
        thumbnail_url = None
        is_video = False
        
        if message.video or (message.document and is_video_file(file_name or '')):
            is_video = True
            try:
                thumbnail_url = await thumbnail_extractor.extract_thumbnail_from_message(
                    Config.FILE_CHANNEL_ID,
                    message.id,
                    message.video.file_id if message.video else message.document.file_id
                )
                
                if thumbnail_url:
                    logger.debug(f"✅ Thumbnail extracted for: {title}")
            except Exception as e:
                logger.error(f"❌ Thumbnail extraction error: {e}")
        
        # Create document
        doc = {
            'channel_id': Config.FILE_CHANNEL_ID,
            'message_id': message.id,
            'title': title,
            'normalized_title': normalized_title,
            'date': message.date,
            'indexed_at': datetime.now(),
            'last_checked': datetime.now(),
            'is_video_file': is_video,
            'file_id': None,
            'file_size': 0,
            'file_hash': file_hash,
            'thumbnail_url': thumbnail_url,
            'thumbnail_extracted': thumbnail_url is not None,
            'status': 'active',
            'is_duplicate': False,
            'duplicate_reason': None
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
            thumbnail_status = "✅" if thumbnail_url else "❌"
            
            logger.info(f"✅ {file_type} indexed: {title}")
            logger.info(f"   📊 Size: {size_str} | Quality: {doc.get('quality', 'Unknown')}")
            logger.info(f"   🖼️ Thumbnail: {thumbnail_status}")
            logger.info(f"   🔑 Hash: {file_hash[:20] if file_hash else 'None'}")
            
            return True
            
        except Exception as e:
            if "duplicate key error" in str(e).lower():
                logger.debug(f"📝 Duplicate key error: {message.id}")
                return True  # Already exists
            else:
                logger.error(f"❌ Insert error: {e}")
                return False
        
    except Exception as e:
        logger.error(f"❌ Indexing error for message {message.id if message else 'unknown'}: {e}")
        return False

async def index_files_background_smart():
    """Background indexing with duplicate prevention"""
    if User is None or files_col is None or not user_session_ready:
        logger.warning("⚠️ User session not ready for indexing")
        return
    
    logger.info("📁 Starting smart indexing with duplicate prevention...")
    
    try:
        # Setup indexes
        await setup_database_indexes()
        
        # Initialize duplicate prevention
        await duplicate_prevention.initialize_from_database()
        
        # Get last indexed message
        last_indexed = await files_col.find_one(
            {"channel_id": Config.FILE_CHANNEL_ID}, 
            sort=[('message_id', -1)],
            projection={'message_id': 1}
        )
        
        last_message_id = last_indexed['message_id'] if last_indexed else 0
        
        logger.info(f"🔄 Starting from message ID: {last_message_id}")
        logger.info(f"📡 Fetching messages from FILE channel: {Config.FILE_CHANNEL_ID}")
        
        # Fetch messages
        messages = []
        fetched_count = 0
        
        try:
            async for msg in User.get_chat_history(Config.FILE_CHANNEL_ID, limit=Config.MAX_INDEX_LIMIT):
                fetched_count += 1
                if msg.id <= last_message_id:
                    logger.info(f"✅ Reached last indexed message {last_message_id}")
                    break
                
                if msg is not None and (msg.document or msg.video):
                    messages.append(msg)
                    logger.debug(f"📥 Found file: {msg.id}")
        except Exception as e:
            logger.error(f"❌ Error fetching chat history: {e}")
            return
        
        logger.info(f"📥 Fetched {fetched_count} messages, found {len(messages)} new files to index")
        
        # Process messages in reverse order (oldest first)
        messages.reverse()
        
        # Process in batches
        batch_size = 20
        total_indexed = 0
        total_duplicates = 0
        
        for i in range(0, len(messages), batch_size):
            batch = messages[i:i + batch_size]
            logger.info(f"📦 Processing batch {i//batch_size + 1}/{(len(messages)-1)//batch_size + 1}")
            
            for msg in batch:
                try:
                    success = await index_single_file_smart(msg)
                    if success:
                        total_indexed += 1
                    else:
                        total_duplicates += 1
                    
                    # Small delay to avoid rate limiting
                    await asyncio.sleep(0.2)
                    
                except Exception as e:
                    logger.error(f"❌ Error processing message {msg.id}: {e}")
                    continue
            
            # Delay between batches
            await asyncio.sleep(2)
        
        if total_indexed > 0 or total_duplicates > 0:
            logger.info(f"✅ Smart indexing complete:")
            logger.info(f"   ✅ Indexed: {total_indexed} new files")
            logger.info(f"   🔄 Skipped: {total_duplicates} duplicates")
            
            # Get stats
            await log_database_stats()
        else:
            logger.info("✅ No new files to index")
        
        # Start auto indexing
        await auto_indexing_manager.start_auto_indexing()
        logger.info("✅ Started AUTO indexing system")
        
        # Start sync monitoring
        if bot_session_ready:
            await channel_sync_manager.start_sync_monitoring()
            logger.info("✅ Started BOT sync monitoring")
        
    except Exception as e:
        logger.error(f"❌ Background indexing error: {e}")

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
        logger.info("✅ Created unique index on channel_id + message_id")
    except Exception as e:
        logger.warning(f"⚠️ Index creation error (might already exist): {e}")
    
    try:
        # Text search index
        await files_col.create_index(
            [("normalized_title", "text"), ("title", "text")],
            name="text_search_index",
            default_language="english",
            background=True
        )
        logger.info("✅ Created text search index")
    except Exception as e:
        logger.warning(f"⚠️ Text index creation error: {e}")
    
    try:
        # File hash index for duplicate detection
        await files_col.create_index(
            [("file_hash", 1)],
            name="file_hash_index",
            background=True,
            sparse=True
        )
        logger.info("✅ Created file_hash index")
    except Exception as e:
        logger.warning(f"⚠️ File hash index creation error: {e}")
    
    try:
        # Quality index
        await files_col.create_index(
            [("quality", 1)],
            name="quality_index",
            background=True
        )
        logger.info("✅ Created quality index")
    except Exception as e:
        logger.warning(f"⚠️ Quality index creation error: {e}")
    
    try:
        # Date index for sorting
        await files_col.create_index(
            [("date", -1)],
            name="date_index",
            background=True
        )
        logger.info("✅ Created date index")
    except Exception as e:
        logger.warning(f"⚠️ Date index creation error: {e}")

async def log_database_stats():
    """Log database statistics"""
    if files_col is None:
        return
    
    try:
        total_files = await files_col.count_documents({})
        video_files = await files_col.count_documents({'is_video_file': True})
        file_channel_files = await files_col.count_documents({
            "channel_id": Config.FILE_CHANNEL_ID
        })
        
        # Count thumbnails
        thumbnails_extracted = await files_col.count_documents({'thumbnail_extracted': True})
        
        # Count duplicates
        total_duplicates = await db.duplicates.count_documents({}) if hasattr(db, 'duplicates') else 0
        
        # Get unique hashes
        unique_hashes = await files_col.distinct("file_hash")
        unique_hashes_count = len([h for h in unique_hashes if h])
        
        logger.info("📊 DATABASE STATISTICS:")
        logger.info(f"   • Total files: {total_files}")
        logger.info(f"   • Video files: {video_files}")
        logger.info(f"   • FILE channel files: {file_channel_files}")
        logger.info(f"   • Thumbnails extracted: {thumbnails_extracted}")
        logger.info(f"   • Unique file hashes: {unique_hashes_count}")
        logger.info(f"   • Tracked duplicates: {total_duplicates}")
        
    except Exception as e:
        logger.error(f"❌ Error getting database stats: {e}")

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
            'source': 'custom' if PosterSource is None else PosterSource.CUSTOM.value,
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
                'source': 'custom' if PosterSource is None else PosterSource.CUSTOM.value,
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
            'source': 'custom' if PosterSource is None else PosterSource.CUSTOM.value,
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
                'thumbnail': poster_data['poster_url'],  # Default to poster
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
                'poster_source': 'fallback',
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
        
        # Create duplicates collection if not exists
        if 'duplicates' not in await db.list_collection_names():
            await db.create_collection('duplicates')
        
        logger.info("✅ MongoDB OK")
        return True
        
    except asyncio.TimeoutError:
        logger.error("❌ MongoDB connection timeout")
        return False
    except Exception as e:
        logger.error(f"❌ MongoDB error: {e}")
        return False

# ============================================================================
# ✅ COMBINED SEARCH WITH QUALITY MERGING (WITH VIDEO THUMBNAILS)
# ============================================================================

@performance_monitor.measure("multi_channel_search_merged")
@async_cache_with_ttl(maxsize=500, ttl=300)
async def search_movies_multi_channel_merged(query, limit=12, page=1):
    """COMBINED search with QUALITY MERGING and VIDEO THUMBNAILS"""
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
    # ✅ 2. SEARCH FILE CHANNEL (BOT SESSION) - WITH QUALITY DETECTION & THUMBNAILS
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
                    ],
                    "status": "active",
                    "is_duplicate": False  # Exclude duplicates
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
                    'thumbnail_url': 1,
                    'thumbnail_extracted': 1,
                    '_id': 0
                }
            ).limit(limit * 3)  # Get more for quality merging
        
            async for doc in cursor:
                try:
                    norm_title = doc.get('normalized_title', normalize_title(doc['title']))
                    quality_info = extract_quality_info(doc.get('file_name', ''))
                    quality = quality_info['full']
                    
                    # Check if we have extracted thumbnail
                    thumbnail_url = doc.get('thumbnail_url')
                    thumbnail_extracted = doc.get('thumbnail_extracted', False)
                    
                    # If video file but no thumbnail extracted, extract now
                    if doc.get('is_video_file') and not thumbnail_extracted:
                        try:
                            thumbnail_url = await thumbnail_extractor.extract_thumbnail_from_message(
                                doc.get('channel_id'),
                                doc.get('message_id'),
                                doc.get('file_id')
                            )
                            
                            # Update database with extracted thumbnail
                            if thumbnail_url:
                                await files_col.update_one(
                                    {
                                        'channel_id': doc.get('channel_id'),
                                        'message_id': doc.get('message_id')
                                    },
                                    {
                                        '$set': {
                                            'thumbnail_url': thumbnail_url,
                                            'thumbnail_extracted': True
                                        }
                                    }
                                )
                        except Exception as e:
                            logger.error(f"❌ Thumbnail extraction in search: {e}")
                    
                    # Quality option
                    quality_option = {
                        'file_id': f"{doc.get('channel_id', Config.FILE_CHANNEL_ID)}_{doc.get('message_id')}_{quality}",
                        'file_size': doc.get('file_size', 0),
                        'file_name': doc.get('file_name', ''),
                        'is_video': doc.get('is_video_file', False),
                        'channel_id': doc.get('channel_id'),
                        'message_id': doc.get('message_id'),
                        'quality_info': quality_info,
                        'thumbnail_url': thumbnail_url,
                        'has_thumbnail': thumbnail_url is not None
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
                            'quality_options': {quality: quality_option},
                            'date': doc['date'].isoformat() if isinstance(doc['date'], datetime) else doc['date'],
                            'is_new': is_new(doc['date']) if doc.get('date') else False,
                            'is_video_file': doc.get('is_video_file', False),
                            'channel_id': doc.get('channel_id'),
                            'channel_name': channel_name_cached(doc.get('channel_id')),
                            'has_file': True,
                            'has_post': bool(doc.get('caption')),
                            'file_caption': doc.get('caption', ''),
                            'year': year,
                            'quality': quality,
                            'has_thumbnail': thumbnail_url is not None,
                            'thumbnail_source': 'video_extracted' if thumbnail_url else 'none'
                        }
                        
                        # If we have video thumbnail, use it as primary thumbnail
                        if thumbnail_url:
                            files_dict[norm_title]['thumbnail'] = thumbnail_url
                            files_dict[norm_title]['thumbnail_source'] = 'video_extracted'
                    else:
                        # Add quality option to existing entry
                        files_dict[norm_title]['quality_options'][quality] = quality_option
                        
                        # Update thumbnail if this quality has one
                        if thumbnail_url and not files_dict[norm_title].get('has_thumbnail'):
                            files_dict[norm_title]['thumbnail'] = thumbnail_url
                            files_dict[norm_title]['thumbnail_source'] = 'video_extracted'
                            files_dict[norm_title]['has_thumbnail'] = True
                        
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
            
            # Use video thumbnail if available
            if file_data.get('has_thumbnail') and file_data.get('thumbnail'):
                result['thumbnail'] = file_data['thumbnail']
                result['thumbnail_source'] = file_data['thumbnail_source']
                result['has_thumbnail'] = True
        
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
    # ✅ 5. FETCH POSTERS IN BATCH (Only for movies without video thumbnails)
    # ============================================================================
    logger.info(f"🎬 Fetching posters for {len(movies_for_posters)} movies...")
    
    # Separate movies with and without thumbnails
    movies_without_thumbnails = []
    movies_with_thumbnails = []
    
    for movie in movies_for_posters:
        if movie.get('has_thumbnail'):
            movies_with_thumbnails.append(movie)
        else:
            movies_without_thumbnails.append(movie)
    
    # Get posters only for movies without video thumbnails
    if movies_without_thumbnails:
        movies_with_posters = await get_posters_for_movies_batch(movies_without_thumbnails)
    else:
        movies_with_posters = []
    
    # Update merged dict with poster/thumbnail data
    all_movies = movies_with_thumbnails + movies_with_posters
    
    for movie in all_movies:
        norm_title = movie.get('normalized_title', normalize_title(movie['title']))
        if norm_title in merged:
            # If movie has video thumbnail, keep it
            if movie.get('has_thumbnail'):
                merged[norm_title].update({
                    'thumbnail': movie['thumbnail'],
                    'thumbnail_source': movie['thumbnail_source'],
                    'has_thumbnail': True,
                    'poster_url': movie.get('poster_url', Config.FALLBACK_POSTER),
                    'poster_source': movie.get('poster_source', 'fallback'),
                    'poster_rating': movie.get('poster_rating', '0.0'),
                    'has_poster': True
                })
            else:
                # Use poster as thumbnail
                merged[norm_title].update({
                    'poster_url': movie['poster_url'],
                    'poster_source': movie['poster_source'],
                    'poster_rating': movie['poster_rating'],
                    'thumbnail': movie['thumbnail'],
                    'thumbnail_source': movie['thumbnail_source'],
                    'has_poster': True,
                    'has_thumbnail': True
                })
    
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
        'video_files': sum(1 for r in results_list if r.get('is_video_file', False)),
        'with_video_thumbnails': sum(1 for r in results_list if r.get('thumbnail_source') == 'video_extracted')
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
            'video_thumbnails': True,
            'poster_fetcher': poster_fetcher is not None,
            'user_session_used': user_session_ready,
            'bot_session_used': bot_session_ready,
            'duplicate_prevention': True,
            'cache_hit': False
        }
    }
    
    # Cache results
    if cache_manager is not None:
        await cache_manager.set(cache_key, result_data, expire_seconds=600)
    
    logger.info(f"✅ QUALITY-MERGED search complete: {len(paginated)} results")
    logger.info(f"   📊 Stats: {stats}")
    logger.info(f"   🖼️ Video thumbnails: {stats['with_video_thumbnails']}")
    
    return result_data

# ============================================================================
# ✅ HOME MOVIES (6/6)
# ============================================================================

def channel_name_cached(cid):
    return f"Channel {cid}"

@performance_monitor.measure("home_movies")
@async_cache_with_ttl(maxsize=1, ttl=60)
async def get_home_movies(limit=20):
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
# ✅ THUMBNAIL API ENDPOINTS
# ============================================================================

@app.route('/api/thumbnail/extract', methods=['GET'])
async def api_extract_thumbnail():
    """Extract thumbnail from specific video file"""
    try:
        channel_id = int(request.args.get('channel_id', 0))
        message_id = int(request.args.get('message_id', 0))
        
        if not channel_id or not message_id:
            return jsonify({
                'status': 'error',
                'message': 'channel_id and message_id required'
            }), 400
        
        # Extract thumbnail
        thumbnail_url = await thumbnail_extractor.extract_thumbnail_from_message(channel_id, message_id)
        
        if thumbnail_url:
            return jsonify({
                'status': 'success',
                'thumbnail_url': thumbnail_url,
                'channel_id': channel_id,
                'message_id': message_id,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to extract thumbnail',
                'channel_id': channel_id,
                'message_id': message_id
            }), 404
            
    except Exception as e:
        logger.error(f"Thumbnail extraction API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/thumbnail/batch_extract', methods=['POST'])
async def api_batch_extract_thumbnails():
    """Extract thumbnails for multiple files in batch"""
    try:
        data = await request.get_json()
        if not data or 'files' not in data:
            return jsonify({
                'status': 'error',
                'message': 'files array required in JSON body'
            }), 400
        
        files = data['files']
        if not isinstance(files, list) or len(files) > 20:
            return jsonify({
                'status': 'error',
                'message': 'files must be array with max 20 items'
            }), 400
        
        results = {}
        extraction_tasks = []
        
        for file_info in files:
            channel_id = file_info.get('channel_id')
            message_id = file_info.get('message_id')
            
            if channel_id and message_id:
                task = thumbnail_extractor.extract_thumbnail_from_message(channel_id, message_id)
                extraction_tasks.append((f"{channel_id}_{message_id}", task))
        
        # Execute all tasks
        for key, task in extraction_tasks:
            try:
                thumbnail_url = await task
                results[key] = {
                    'thumbnail_url': thumbnail_url,
                    'success': thumbnail_url is not None
                }
            except Exception as e:
                logger.error(f"Batch thumbnail error for {key}: {e}")
                results[key] = {
                    'thumbnail_url': None,
                    'success': False,
                    'error': str(e)
                }
        
        return jsonify({
            'status': 'success',
            'results': results,
            'total': len(results),
            'successful': sum(1 for r in results.values() if r.get('success')),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Batch thumbnail API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/thumbnail/stats', methods=['GET'])
async def api_thumbnail_stats():
    """Get thumbnail extraction statistics"""
    try:
        if files_col is None:
            return jsonify({
                'status': 'error',
                'message': 'Database not available'
            }), 500
        
        total_videos = await files_col.count_documents({'is_video_file': True})
        thumbnails_extracted = await files_col.count_documents({'thumbnail_extracted': True})
        
        return jsonify({
            'status': 'success',
            'stats': {
                'total_video_files': total_videos,
                'thumbnails_extracted': thumbnails_extracted,
                'extraction_rate': f"{(thumbnails_extracted/total_videos*100):.1f}%" if total_videos > 0 else "0%",
                'thumbnail_extractor_ready': True
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Thumbnail stats API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# ============================================================================
# ✅ MAIN INITIALIZATION
# ============================================================================

@performance_monitor.measure("system_init")
async def init_system():
    start_time = time.time()
    
    try:
        logger.info("🚀 Starting SK4FiLM v8.3 - AUTO INDEXING WITH DUPLICATE PREVENTION...")
        
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
        if VerificationSystem is not None:
            verification_system = VerificationSystem(Config, mongo_client)
            logger.info("✅ Verification System initialized")
        else:
            verification_system = None
            logger.warning("⚠️ Verification System not available")
        
        # Initialize Premium System
        if PremiumSystem is not None:
            premium_system = PremiumSystem(Config, mongo_client)
            logger.info("✅ Premium System initialized")
        else:
            premium_system = None
            logger.warning("⚠️ Premium System not available")
        
        # Initialize Poster Fetcher
        if PosterFetcher is not None:
            poster_fetcher = PosterFetcher(Config, cache_manager)
            logger.info("✅ Poster Fetcher initialized")
        else:
            poster_fetcher = None
            logger.warning("⚠️ Poster Fetcher not available")
        
        # Initialize Telegram DUAL Sessions
        if PYROGRAM_AVAILABLE:
            telegram_ok = await init_telegram_sessions()
            if not telegram_ok:
                logger.warning("⚠️ Telegram sessions failed")
        else:
            logger.warning("⚠️ Pyrogram not available")
        
        # Start background indexing
        if user_session_ready and files_col is not None:
            asyncio.create_task(index_files_background_smart())
            logger.info("✅ Started AUTO indexing with duplicate prevention")
        
        init_time = time.time() - start_time
        logger.info(f"⚡ SK4FiLM Started in {init_time:.2f}s")
        
        logger.info("🔧 INTEGRATED FEATURES:")
        logger.info(f"   • Auto Indexing: ✅ ENABLED ({Config.AUTO_INDEX_INTERVAL}s interval)")
        logger.info(f"   • Duplicate Prevention: ✅ ENABLED")
        logger.info(f"   • Cache System: {'✅ ENABLED' if cache_manager else '❌ DISABLED'}")
        logger.info(f"   • Verification: {'✅ ENABLED' if verification_system else '❌ DISABLED'}")
        logger.info(f"   • Premium System: {'✅ ENABLED' if premium_system else '❌ DISABLED'}")
        logger.info(f"   • Poster Fetcher: {'✅ ENABLED' if poster_fetcher else '❌ DISABLED'}")
        logger.info(f"   • Quality Merging: ✅ ENABLED")
        logger.info(f"   • Video Thumbnail Extraction: ✅ ENABLED")
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
        thumbnails_extracted = await files_col.count_documents({'thumbnail_extracted': True})
        duplicates_count = await db.duplicates.count_documents({}) if hasattr(db, 'duplicates') else 0
    else:
        tf = 0
        video_files = 0
        thumbnails_extracted = 0
        duplicates_count = 0
    
    # Get indexing status
    indexing_status = await auto_indexing_manager.get_indexing_status()
    
    return jsonify({
        'status': 'healthy',
        'service': 'SK4FiLM v8.3 - AUTO INDEXING WITH DUPLICATE PREVENTION',
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
            'database': files_col is not None,
            'thumbnail_extractor': True,
            'auto_indexing': True,
            'duplicate_prevention': True
        },
        'features': {
            'quality_merging': True,
            'home_movies_6_6': True,
            'hevc_support': True,
            'video_thumbnail_extraction': True,
            'auto_indexing_interval': Config.AUTO_INDEX_INTERVAL,
            'duplicate_detection': True
        },
        'stats': {
            'total_files': tf,
            'video_files': video_files,
            'thumbnails_extracted': thumbnails_extracted,
            'duplicates_tracked': duplicates_count
        },
        'indexing': indexing_status,
        'response_time': f"{time.perf_counter():.3f}s"
    })

@app.route('/health')
@performance_monitor.measure("health_endpoint")
async def health():
    indexing_status = await auto_indexing_manager.get_indexing_status()
    
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
            'poster_fetcher': poster_fetcher is not None,
            'thumbnail_extractor': True,
            'auto_indexing': True
        },
        'indexing': {
            'running': indexing_status['is_running'],
            'last_run': indexing_status['last_run']
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
            'limit': 20,
            'source': 'telegram',
            'poster_fetcher': poster_fetcher is not None,
            'thumbnail_extractor': True,
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
                'video_thumbnails': True,
                'duplicate_prevention': True
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
        
        # Get database stats
        if files_col is not None:
            total_files = await files_col.count_documents({})
            video_files = await files_col.count_documents({'is_video_file': True})
            thumbnails_extracted = await files_col.count_documents({'thumbnail_extracted': True})
            duplicates_count = await db.duplicates.count_documents({}) if hasattr(db, 'duplicates') else 0
            
            # Get indexing stats
            indexing_status = await auto_indexing_manager.get_indexing_status()
            
            # Get duplicate stats
            duplicate_stats = await duplicate_prevention.get_duplicate_stats()
        else:
            total_files = 0
            video_files = 0
            thumbnails_extracted = 0
            duplicates_count = 0
            indexing_status = {}
            duplicate_stats = {}
        
        return jsonify({
            'status': 'success',
            'performance': perf_stats,
            'poster_fetcher': poster_stats,
            'database_stats': {
                'total_files': total_files,
                'video_files': video_files,
                'thumbnails_extracted': thumbnails_extracted,
                'duplicates_tracked': duplicates_count,
                'extraction_rate': f"{(thumbnails_extracted/video_files*100):.1f}%" if video_files > 0 else "0%"
            },
            'indexing_stats': indexing_status,
            'duplicate_stats': duplicate_stats,
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

@app.route('/api/admin/index', methods=['POST'])
async def api_admin_index():
    """Manual indexing endpoint (admin only)"""
    try:
        # Check admin
        auth_token = request.headers.get('Authorization')
        if not auth_token or auth_token != os.environ.get('ADMIN_TOKEN', 'sk4film_admin_2024'):
            return jsonify({
                'status': 'error',
                'message': 'Unauthorized'
            }), 401
        
        data = await request.get_json()
        limit = data.get('limit', 100) if data else 100
        
        # Run manual indexing
        stats = await auto_indexing_manager.run_manual_index(limit)
        
        if stats:
            return jsonify({
                'status': 'success',
                'message': 'Manual indexing completed',
                'stats': stats,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Manual indexing failed',
                'timestamp': datetime.now().isoformat()
            }), 500
    except Exception as e:
        logger.error(f"Admin index API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/admin/index_status', methods=['GET'])
async def api_admin_index_status():
    """Get indexing status"""
    try:
        if files_col is not None:
            total_files = await files_col.count_documents({})
            video_files = await files_col.count_documents({'is_video_file': True})
            file_channel_files = await files_col.count_documents({
                "channel_id": Config.FILE_CHANNEL_ID
            })
            thumbnails_extracted = await files_col.count_documents({'thumbnail_extracted': True})
            duplicates_count = await db.duplicates.count_documents({}) if hasattr(db, 'duplicates') else 0
            
            # Get latest indexed file
            latest_file = await files_col.find_one(
                {"channel_id": Config.FILE_CHANNEL_ID},
                sort=[('message_id', -1)],
                projection={'title': 1, 'message_id': 1, 'indexed_at': 1, 'thumbnail_extracted': 1, 'file_hash': 1}
            )
        else:
            total_files = 0
            video_files = 0
            file_channel_files = 0
            thumbnails_extracted = 0
            duplicates_count = 0
            latest_file = None
        
        # Get indexing status
        indexing_status = await auto_indexing_manager.get_indexing_status()
        
        # Get duplicate stats
        duplicate_stats = await duplicate_prevention.get_duplicate_stats()
        
        return jsonify({
            'status': 'success',
            'indexing': {
                'total_files': total_files,
                'video_files': video_files,
                'file_channel_files': file_channel_files,
                'thumbnails_extracted': thumbnails_extracted,
                'duplicates_tracked': duplicates_count,
                'thumbnail_extraction_rate': f"{(thumbnails_extracted/video_files*100):.1f}%" if video_files > 0 else "0%",
                'latest_file': latest_file,
                'auto_indexing_running': indexing_status['is_running'],
                'auto_indexing_last_run': indexing_status['last_run'],
                'auto_indexing_next_run': indexing_status['next_run'],
                'auto_indexing_stats': indexing_status['stats'],
                'user_session_ready': user_session_ready,
                'bot_session_ready': bot_session_ready,
                'file_channel_id': Config.FILE_CHANNEL_ID,
                'last_update': datetime.now().isoformat()
            },
            'duplicate_prevention': duplicate_stats,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Index status API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/admin/duplicates', methods=['GET'])
async def api_admin_duplicates():
    """Get duplicate files information"""
    try:
        # Check admin
        auth_token = request.headers.get('Authorization')
        if not auth_token or auth_token != os.environ.get('ADMIN_TOKEN', 'sk4film_admin_2024'):
            return jsonify({
                'status': 'error',
                'message': 'Unauthorized'
            }), 401
        
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 20))
        offset = (page - 1) * limit
        
        # Get duplicates from database
        cursor = db.duplicates.find(
            {},
            {
                'channel_id': 1,
                'message_id': 1,
                'title': 1,
                'normalized_title': 1,
                'file_hash': 1,
                'duplicate_reason': 1,
                'detected_at': 1,
                '_id': 0
            }
        ).sort('detected_at', -1).skip(offset).limit(limit)
        
        duplicates = []
        async for doc in cursor:
            duplicates.append(doc)
        
        total_duplicates = await db.duplicates.count_documents({})
        
        return jsonify({
            'status': 'success',
            'duplicates': duplicates,
            'pagination': {
                'current_page': page,
                'total_pages': math.ceil(total_duplicates / limit) if total_duplicates > 0 else 1,
                'total_results': total_duplicates,
                'per_page': limit
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Duplicates API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/admin/reindex', methods=['POST'])
async def api_admin_reindex():
    """Re-index specific file"""
    try:
        # Check admin
        auth_token = request.headers.get('Authorization')
        if not auth_token or auth_token != os.environ.get('ADMIN_TOKEN', 'sk4film_admin_2024'):
            return jsonify({
                'status': 'error',
                'message': 'Unauthorized'
            }), 401
        
        data = await request.get_json()
        channel_id = data.get('channel_id', Config.FILE_CHANNEL_ID)
        message_id = data.get('message_id')
        
        if not message_id:
            return jsonify({
                'status': 'error',
                'message': 'message_id required'
            }), 400
        
        # Get message
        try:
            if User is None:
                return jsonify({
                    'status': 'error',
                    'message': 'User session not available'
                }), 500
            
            message = await User.get_messages(channel_id, message_id)
            
            if not message:
                return jsonify({
                    'status': 'error',
                    'message': 'Message not found'
                }), 404
            
            # Delete existing record
            await files_col.delete_one({
                'channel_id': channel_id,
                'message_id': message_id
            })
            
            # Remove from duplicate prevention
            existing = await files_col.find_one({
                'channel_id': channel_id,
                'message_id': message_id
            }, {'file_hash': 1, 'normalized_title': 1})
            
            if existing:
                await duplicate_prevention.remove_file_hash(
                    existing.get('file_hash'),
                    existing.get('normalized_title')
                )
            
            # Re-index
            success = await index_single_file_smart(message)
            
            if success:
                return jsonify({
                    'status': 'success',
                    'message': 'File re-indexed successfully',
                    'channel_id': channel_id,
                    'message_id': message_id,
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'Failed to re-index file',
                    'channel_id': channel_id,
                    'message_id': message_id
                }), 500
                
        except Exception as e:
            logger.error(f"Re-index error: {e}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
            
    except Exception as e:
        logger.error(f"Re-index API error: {e}")
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
    logger.info("🛑 Shutting down SK4FiLM v8.3...")
    
    shutdown_tasks = []
    
    # Stop auto indexing
    await auto_indexing_manager.stop_auto_indexing()
    await channel_sync_manager.stop_sync_monitoring()
    
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
    
    logger.info(f"🌐 Starting SK4FiLM v8.3 on port {Config.WEB_SERVER_PORT}...")
    logger.info("🎯 Features: AUTO INDEXING WITH DUPLICATE PREVENTION")
    logger.info(f"   • Auto Index Interval: {Config.AUTO_INDEX_INTERVAL}s")
    logger.info(f"   • Batch Size: {Config.BATCH_INDEX_SIZE}")
    logger.info(f"   • Max Index Limit: {Config.MAX_INDEX_LIMIT}")
    
    asyncio.run(serve(app, config))
