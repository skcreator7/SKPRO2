# ============================================================================
# 🚀 SK4FiLM v9.0 - OPTIMIZED BACKEND WITH FLOOD WAIT PROTECTION
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
bot_handler = None
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
# ✅ THUMBNAIL EXTRACTION AND POSTER FETCHING
# ============================================================================

class EnhancedPosterFetcher:
    """Enhanced poster fetcher with multiple fallbacks"""
    
    def __init__(self, config, cache_manager=None):
        self.config = config
        self.cache_manager = cache_manager
        self.stats = {
            'total_requests': 0,
            'successful': 0,
            'failed': 0,
            'cache_hits': 0
        }
        
    async def fetch_poster(self, title, year=None, quality=None):
        """Fetch poster for movie with multiple sources"""
        self.stats['total_requests'] += 1
        
        # Clean title for searching
        clean_title = self._clean_title(title)
        
        # Check cache first
        if self.cache_manager and self.cache_manager.redis_enabled:
            cache_key = f"poster:{clean_title}"
            cached = await self.cache_manager.get(cache_key)
            if cached:
                self.stats['cache_hits'] += 1
                return cached
        
        poster_data = None
        
        # Try TMDB first
        if self.config.TMDB_API_KEY and self.config.TMDB_API_KEY != "e547e17d4e91f3e62a571655cd1ccaff":
            poster_data = await self._fetch_from_tmdb(clean_title, year)
        
        # Try OMDB if TMDB failed
        if not poster_data and self.config.OMDB_API_KEY and self.config.OMDB_API_KEY != "8265bd1c":
            poster_data = await self._fetch_from_omdb(clean_title, year)
        
        # Try backup TMDB API
        if not poster_data:
            poster_data = await self._fetch_from_backup_tmdb(clean_title, year)
        
        # If all fails, use fallback
        if not poster_data:
            poster_data = self._get_fallback_poster(title, quality)
        
        # Cache the result
        if poster_data and self.cache_manager and self.cache_manager.redis_enabled:
            await self.cache_manager.set(
                f"poster:{clean_title}", 
                poster_data, 
                expire_seconds=86400  # 24 hours
            )
        
        if poster_data and poster_data.get('poster_url') != self.config.FALLBACK_POSTER:
            self.stats['successful'] += 1
        else:
            self.stats['failed'] += 1
        
        return poster_data
    
    def _clean_title(self, title):
        """Clean title for searching"""
        if not title:
            return ""
        
        # Remove year
        title = re.sub(r'\s*\(\d{4}\)$', '', title)
        title = re.sub(r'\s+\d{4}$', '', title)
        
        # Remove quality indicators
        quality_patterns = [
            r'\b\d{3,4}p\b',
            r'\b4k\b',
            r'\buhd\b',
            r'\bhd\b',
            r'\bfullhd\b',
            r'\bfhd\b',
            r'\bhevc\b',
            r'\bx265\b',
            r'\bx264\b',
        ]
        
        for pattern in quality_patterns:
            title = re.sub(pattern, '', title, flags=re.IGNORECASE)
        
        # Remove brackets and extra spaces
        title = re.sub(r'[\[\]\(\)]', '', title)
        title = re.sub(r'\s+', ' ', title)
        
        return title.strip()
    
    async def _fetch_from_tmdb(self, title, year=None):
        """Fetch from TMDB API"""
        try:
            url = f"https://api.themoviedb.org/3/search/movie"
            params = {
                'api_key': self.config.TMDB_API_KEY,
                'query': title,
                'language': 'en-US',
                'page': 1,
                'include_adult': False
            }
            
            if year:
                params['year'] = year
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get('results') and len(data['results']) > 0:
                            movie = data['results'][0]
                            
                            poster_path = movie.get('poster_path')
                            if poster_path:
                                poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
                                
                                return {
                                    'poster_url': poster_url,
                                    'source': 'tmdb',
                                    'rating': str(movie.get('vote_average', '0.0')),
                                    'year': str(movie.get('release_date', '')[:4]),
                                    'title': movie.get('title', title),
                                    'quality': 'unknown'
                                }
        
        except Exception as e:
            logger.debug(f"TMDB fetch error: {e}")
        
        return None
    
    async def _fetch_from_omdb(self, title, year=None):
        """Fetch from OMDB API"""
        try:
            url = "https://www.omdbapi.com/"
            params = {
                't': title,
                'apikey': self.config.OMDB_API_KEY,
                'plot': 'short'
            }
            
            if year:
                params['y'] = year
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get('Poster') and data['Poster'] != 'N/A':
                            return {
                                'poster_url': data['Poster'],
                                'source': 'omdb',
                                'rating': data.get('imdbRating', '0.0'),
                                'year': data.get('Year', ''),
                                'title': data.get('Title', title),
                                'quality': 'unknown'
                            }
        
        except Exception as e:
            logger.debug(f"OMDB fetch error: {e}")
        
        return None
    
    async def _fetch_from_backup_tmdb(self, title, year=None):
        """Fetch from backup TMDB API (public key)"""
        try:
            url = "https://api.themoviedb.org/3/search/movie"
            params = {
                'api_key': "e547e17d4e91f3e62a571655cd1ccaff",  # Backup key
                'query': title,
                'language': 'en-US',
                'page': 1,
                'include_adult': False
            }
            
            if year:
                params['year'] = year
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get('results') and len(data['results']) > 0:
                            movie = data['results'][0]
                            
                            poster_path = movie.get('poster_path')
                            if poster_path:
                                poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
                                
                                return {
                                    'poster_url': poster_url,
                                    'source': 'tmdb_backup',
                                    'rating': str(movie.get('vote_average', '0.0')),
                                    'year': str(movie.get('release_date', '')[:4]),
                                    'title': movie.get('title', title),
                                    'quality': 'unknown'
                                }
        
        except Exception as e:
            logger.debug(f"Backup TMDB error: {e}")
        
        return None
    
    def _get_fallback_poster(self, title, quality=None):
        """Generate fallback poster"""
        # Create a colorful gradient based on title hash
        colors = [
            ('#667eea', '#764ba2'),
            ('#f093fb', '#f5576c'),
            ('#4facfe', '#00f2fe'),
            ('#43e97b', '#38f9d7'),
            ('#fa709a', '#fee140'),
            ('#30cfd0', '#330867')
        ]
        
        # Generate hash from title
        if not title:
            title_hash = 0
        else:
            title_hash = sum(ord(c) for c in title)
        
        color_pair = colors[title_hash % len(colors)]
        
        # Encode title for URL
        encoded_title = urllib.parse.quote(title[:30] if title else "Movie")
        
        # Create a gradient image URL (using a simple service)
        # Note: This is a placeholder. You might want to use a real service
        fallback_url = self.config.FALLBACK_POSTER
        
        return {
            'poster_url': fallback_url,
            'source': 'fallback',
            'rating': '0.0',
            'year': '',
            'title': title or 'Unknown',
            'quality': quality or 'unknown'
        }
    
    def get_stats(self):
        return self.stats

# ============================================================================
# ✅ OPTIMIZED FILE INDEXING MANAGER
# ============================================================================

class OptimizedFileIndexingManager:
    """Optimized file channel indexing manager - ULTRA FAST"""
    
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
        await asyncio.sleep(30)  # Initial delay
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
        """Main indexing loop - OPTIMIZED"""
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
        """Run one indexing cycle - OPTIMIZED"""
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
            
            # Fetch new messages (small batch)
            messages_to_index = []
            try:
                # Fetch only 20 messages to avoid flood wait
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
# ✅ OPTIMIZED SYNC MANAGEMENT
# ============================================================================

class OptimizedSyncManager:
    """Optimized sync manager with auto-delete"""
    
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
        self.monitoring_task = asyncio.create_task(self._monitor_loop())
    
    async def stop_sync_monitoring(self):
        """Stop sync monitoring"""
        self.is_monitoring = False
        if self.monitoring_task is not None:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except:
                pass
        logger.info("🛑 Sync monitoring stopped")
    
    async def _monitor_loop(self):
        """Monitor loop"""
        while self.is_monitoring:
            try:
                # Run sync check
                await self._check_deleted_files()
                
                # Wait for next check (10 minutes)
                await asyncio.sleep(600)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Sync error: {e}")
                await asyncio.sleep(300)
    
    async def _check_deleted_files(self):
        """Check and delete removed files"""
        try:
            if files_col is None or User is None or not user_session_ready:
                return
            
            current_time = time.time()
            if current_time - self.last_sync < 1800:  # 30 minutes minimum
                return
            
            self.last_sync = current_time
            
            logger.info("🔄 Checking for deleted files...")
            
            # Get a small batch of newest messages
            cursor = files_col.find(
                {"channel_id": Config.FILE_CHANNEL_ID},
                {"message_id": 1, "_id": 1}
            ).sort("message_id", -1).limit(50)
            
            message_ids = []
            db_ids = []
            
            async for doc in cursor:
                message_ids.append(doc['message_id'])
                db_ids.append(doc['_id'])
            
            if not message_ids:
                return
            
            try:
                # Check messages in batch
                messages = await User.get_messages(Config.FILE_CHANNEL_ID, message_ids)
                
                # Find existing message IDs
                existing_ids = set()
                if isinstance(messages, list):
                    for msg in messages:
                        if msg and hasattr(msg, 'id'):
                            existing_ids.add(msg.id)
                
                # Delete non-existing messages
                deleted = 0
                for msg_id, db_id in zip(message_ids, db_ids):
                    if msg_id not in existing_ids:
                        await files_col.delete_one({"_id": db_id})
                        deleted += 1
                
                if deleted > 0:
                    self.deleted_count += deleted
                    logger.info(f"✅ Auto-deleted {deleted} files")
                    
            except Exception as e:
                logger.error(f"❌ Error checking messages: {e}")
                
        except Exception as e:
            logger.error(f"❌ Check deleted files error: {e}")

# Initialize sync manager
sync_manager = OptimizedSyncManager()

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
    """Fast file indexing WITH POSTER FETCHING"""
    try:
        if files_col is None:
            return False
        
        if not message or (not message.document and not message.video):
            return False
        
        # Check if already indexed
        existing = await files_col.find_one({
            'channel_id': Config.FILE_CHANNEL_ID,
            'message_id': message.id
        }, {'_id': 1})
        
        if existing:
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
        
        # Get poster for this movie
        poster_data = None
        if poster_fetcher is not None:
            try:
                poster_data = await poster_fetcher.fetch_poster(title, year, quality)
            except Exception as e:
                logger.debug(f"Poster fetch error: {e}")
                poster_data = None
        
        # Create document
        doc = {
            'channel_id': Config.FILE_CHANNEL_ID,
            'message_id': message.id,
            'real_message_id': message.id,
            'title': title,
            'normalized_title': normalize_title(title),
            'date': message.date,
            'indexed_at': datetime.now(),
            'last_checked': datetime.now(),
            'is_video_file': True,
            'file_id': file_id,
            'file_size': file_size,
            'caption': caption or '',
            'quality': quality,
            'status': 'active',
            'thumbnail_extracted': False,
            'year': year,
            'thumbnail_url': poster_data['poster_url'] if poster_data else Config.FALLBACK_POSTER,
            'poster_url': poster_data['poster_url'] if poster_data else Config.FALLBACK_POSTER,
            'poster_source': poster_data['source'] if poster_data else 'fallback',
            'poster_rating': poster_data['rating'] if poster_data else '0.0'
        }
        
        # Insert into database
        await files_col.insert_one(doc)
        
        logger.info(f"✅ Indexed: {title[:50]}... (ID: {message.id})")
        
        return True
        
    except Exception as e:
        if "duplicate key error" in str(e).lower():
            return False
        logger.error(f"❌ Indexing error: {e}")
        return False

async def setup_database_indexes():
    """Setup minimal required database indexes"""
    if files_col is None:
        return
    
    try:
        # Only create essential indexes
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
        
        logger.info("✅ Created minimal database indexes")
        
    except Exception as e:
        logger.warning(f"⚠️ Index creation error: {e}")

# ============================================================================
# ✅ POSTER FETCHING FUNCTIONS
# ============================================================================

async def get_poster_for_movie(title: str, year: str = "", quality: str = "") -> Dict[str, Any]:
    """Get poster for movie"""
    global poster_fetcher
    
    # If poster_fetcher is not available, use fallback
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
        # Fetch poster with timeout
        poster_task = asyncio.create_task(poster_fetcher.fetch_poster(title, year, quality))
        
        try:
            poster_data = await asyncio.wait_for(poster_task, timeout=3.0)
            
            if poster_data and poster_data.get('poster_url'):
                logger.debug(f"✅ Poster fetched: {title} - {poster_data['source']}")
                return poster_data
            else:
                raise ValueError("Invalid poster data")
                
        except (asyncio.TimeoutError, ValueError, Exception) as e:
            logger.warning(f"⚠️ Poster fetch timeout/error for {title}: {e}")
            
            if not poster_task.done():
                poster_task.cancel()
            
            # Return fallback
            return {
                'poster_url': Config.FALLBACK_POSTER,
                'source': 'fallback',
                'rating': '0.0',
                'year': year,
                'title': title,
                'quality': quality or 'unknown'
            }
            
    except Exception as e:
        logger.error(f"❌ Unexpected error in get_poster_for_movie: {e}")
        return {
            'poster_url': Config.FALLBACK_POSTER,
            'source': 'fallback',
            'rating': '0.0',
            'year': year,
            'title': title,
            'quality': quality or 'unknown'
        }

async def get_posters_for_movies_batch(movies: List[Dict]) -> List[Dict]:
    """Get posters for multiple movies in batch"""
    results = []
    
    # Create tasks for all movies
    tasks = []
    for movie in movies:
        title = movie.get('title', '')
        year = movie.get('year', '')
        quality = movie.get('quality', '')
        
        task = asyncio.create_task(get_poster_for_movie(title, year, quality))
        tasks.append((movie, task))
    
    # Process results
    for movie, task in tasks:
        try:
            poster_data = await task
            
            # Update movie with poster data
            movie_with_poster = movie.copy()
            movie_with_poster.update({
                'poster_url': poster_data['poster_url'],
                'poster_source': poster_data['source'],
                'poster_rating': poster_data['rating'],
                'thumbnail': poster_data['poster_url'],  # ✅ FIXED: Add thumbnail
                'thumbnail_url': poster_data['poster_url'],  # ✅ FIXED: Add thumbnail_url
                'thumbnail_source': poster_data['source'],
                'has_poster': True,
                'has_thumbnail': True
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
                'thumbnail': Config.FALLBACK_POSTER,  # ✅ FIXED: Add fallback thumbnail
                'thumbnail_url': Config.FALLBACK_POSTER,  # ✅ FIXED: Add fallback thumbnail_url
                'thumbnail_source': 'fallback',
                'has_poster': True,
                'has_thumbnail': True
            })
            
            results.append(movie_with_fallback)
    
    return results

# ============================================================================
# ✅ OPTIMIZED TELEGRAM SESSION INITIALIZATION
# ============================================================================

@performance_monitor.measure("telegram_init")
async def init_telegram_sessions():
    """Initialize Telegram sessions with flood wait protection"""
    global User, Bot, user_session_ready, bot_session_ready
    
    logger.info("=" * 50)
    logger.info("🚀 TELEGRAM SESSION INITIALIZATION")
    logger.info("=" * 50)
    
    if not PYROGRAM_AVAILABLE:
        logger.error("❌ Pyrogram not installed!")
        return False
    
    # Configure Pyrogram for flood wait protection
    pyrogram_config = {
        'api_id': Config.API_ID,
        'api_hash': Config.API_HASH,
        'sleep_threshold': 120,  # Increased to 120 seconds
        'in_memory': True,
        'no_updates': True,
        'max_concurrent_transmissions': 1,  # Only 1 at a time
    }
    
    # Initialize USER Session
    if Config.API_ID > 0 and Config.API_HASH and Config.USER_SESSION_STRING:
        logger.info("\n👤 Initializing USER Session...")
        try:
            User = Client(
                "sk4film_user",
                session_string=Config.USER_SESSION_STRING,
                **pyrogram_config
            )
            
            await User.start()
            me = await User.get_me()
            logger.info(f"✅ USER Session Ready: {me.first_name}")
            user_session_ready = True
                
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
                bot_token=Config.BOT_TOKEN,
                **pyrogram_config
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
# ✅ OPTIMIZED INITIAL INDEXING
# ============================================================================

async def initial_indexing_simple():
    """Simple initial indexing"""
    if User is None or files_col is None or not user_session_ready:
        logger.warning("⚠️ User session not ready")
        return
    
    logger.info("🚀 Starting simple indexing...")
    
    try:
        # Setup database indexes
        await setup_database_indexes()
        
        # Start file indexing (delayed)
        await asyncio.sleep(10)
        await file_indexing_manager.start_indexing()
        
        # Start sync monitoring (delayed)
        await asyncio.sleep(5)
        await sync_manager.start_sync_monitoring()
        
    except Exception as e:
        logger.error(f"❌ Initial indexing error: {e}")

# ============================================================================
# ✅ OPTIMIZED SEARCH FUNCTION WITH PROPER THUMBNAILS
# ============================================================================

@performance_monitor.measure("enhanced_search")
@async_cache_with_ttl(maxsize=500, ttl=Config.SEARCH_CACHE_TTL)
async def search_movies_enhanced_fixed(query, limit=15, page=1):
    """Optimized search endpoint WITH THUMBNAILS FIX"""
    offset = (page - 1) * limit
    
    # Simple cache check
    cache_key = f"search:{query}:{page}:{limit}"
    if cache_manager is not None and cache_manager.redis_enabled:
        cached = await cache_manager.get(cache_key)
        if cached:
            return cached
    
    logger.info(f"🔍 Searching: {query}")
    
    # Simple search logic
    results = []
    
    # Search database
    if files_col is not None:
        try:
            search_query = {
                "$or": [
                    {"title": {"$regex": query, "$options": "i"}},
                    {"normalized_title": {"$regex": query, "$options": "i"}},
                ],
                "status": "active"
            }
            
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
                    'real_message_id': 1,
                    'date': 1,
                    'caption': 1,
                    'file_id': 1,
                    'thumbnail_url': 1,
                    'poster_url': 1,
                    'poster_source': 1,
                    'poster_rating': 1,
                    'thumbnail_extracted': 1,
                    'year': 1,
                    '_id': 0
                }
            ).limit(100).skip(offset)
            
            async for doc in cursor:
                try:
                    title = doc.get('title', 'Unknown')
                    norm_title = normalize_title(title)
                    
                    # Extract quality
                    quality = doc.get('quality', '480p')
                    
                    # Get thumbnail - FIXED: Use poster_url if available
                    thumbnail_url = doc.get('thumbnail_url') or doc.get('poster_url')
                    if not thumbnail_url:
                        # If no thumbnail in DB, get one dynamically
                        year = doc.get('year', '')
                        if poster_fetcher is not None:
                            try:
                                poster_data = await poster_fetcher.fetch_poster(title, year, quality)
                                thumbnail_url = poster_data['poster_url'] if poster_data else Config.FALLBACK_POSTER
                            except:
                                thumbnail_url = Config.FALLBACK_POSTER
                        else:
                            thumbnail_url = Config.FALLBACK_POSTER
                    
                    # Get message ID
                    message_id = doc.get('real_message_id') or doc.get('message_id')
                    
                    # Extract year
                    year = doc.get('year', '')
                    
                    # Create result
                    result = {
                        'title': title,
                        'normalized_title': norm_title,
                        'content': format_post(doc.get('caption', ''), max_length=300),
                        'post_content': doc.get('caption', ''),
                        'quality_options': {
                            quality: {
                                'quality': quality,
                                'file_size': doc.get('file_size', 0),
                                'message_id': message_id,
                                'real_message_id': message_id,
                                'is_video_file': doc.get('is_video_file', False)
                            }
                        },
                        'date': doc['date'].isoformat() if isinstance(doc['date'], datetime) else doc['date'],
                        'is_new': is_new(doc['date']) if doc.get('date') else False,
                        'is_video_file': doc.get('is_video_file', False),
                        'channel_id': doc.get('channel_id'),
                        'has_file': True,
                        'has_post': bool(doc.get('caption')),
                        'year': year,
                        'quality': quality,
                        'has_thumbnail': thumbnail_url is not None and thumbnail_url != Config.FALLBACK_POSTER,
                        'thumbnail': thumbnail_url,  # ✅ FIXED: Add thumbnail field
                        'thumbnail_url': thumbnail_url,  # ✅ FIXED: Add thumbnail_url field
                        'poster_url': thumbnail_url,  # ✅ FIXED: Add poster_url field
                        'poster_source': doc.get('poster_source', 'fallback'),
                        'poster_rating': doc.get('poster_rating', '0.0'),
                        'real_message_id': message_id,
                        'message_id': message_id,
                        'result_type': 'file_only',
                        'quality_count': 1,
                        'has_poster': thumbnail_url is not None and thumbnail_url != Config.FALLBACK_POSTER,
                    }
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Result processing error: {e}")
                    continue
        except Exception as e:
            logger.error(f"Database search error: {e}")
    
    # If no results from files, check posts
    if not results:
        try:
            if User is not None and user_session_ready:
                # Search in text channels
                for channel_id in Config.TEXT_CHANNEL_IDS:
                    async for msg in User.search_messages(channel_id, query=query, limit=10):
                        if msg and msg.text:
                            title = extract_title_smart(msg.text)
                            if title and len(title) > 10:
                                post_data = {
                                    'title': title,
                                    'content': format_post(msg.text, max_length=300),
                                    'post_content': msg.text,
                                    'date': msg.date.isoformat() if isinstance(msg.date, datetime) else str(msg.date),
                                    'is_new': is_new(msg.date),
                                    'channel_id': channel_id,
                                    'message_id': msg.id,
                                    'has_file': False,
                                    'has_post': True,
                                    'has_thumbnail': False,
                                    'result_type': 'post_only'
                                }
                                results.append(post_data)
        except Exception as e:
            logger.error(f"Text search error: {e}")
    
    # Sort results
    results.sort(key=lambda x: (
        x.get('is_new', False),
        x.get('date', '')
    ), reverse=True)
    
    total = len(results)
    paginated = results[offset:offset + limit]
    
    # Fetch posters for results if needed
    if paginated and poster_fetcher is not None:
        # Filter results that need posters
        results_needing_posters = []
        result_indices = []
        
        for i, result in enumerate(paginated):
            if not result.get('has_poster', False) or not result.get('thumbnail'):
                results_needing_posters.append(result)
                result_indices.append(i)
        
        if results_needing_posters:
            logger.info(f"📸 Fetching posters for {len(results_needing_posters)} results")
            updated_results = await get_posters_for_movies_batch(results_needing_posters)
            
            # Update the original list
            for idx, updated_result in zip(result_indices, updated_results):
                if idx < len(paginated):
                    paginated[idx] = updated_result
    
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
# ✅ HOME MOVIES FUNCTION WITH THUMBNAILS
# ============================================================================

@performance_monitor.measure("home_movies")
@async_cache_with_ttl(maxsize=1, ttl=60)
async def get_home_movies(limit=25):
    """Get home movies WITH THUMBNAILS"""
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
                        'channel': 'SK4FiLM',
                        'channel_id': Config.MAIN_CHANNEL_ID,
                        'message_id': msg.id,
                        'has_file': False,
                        'has_post': True,
                        'content': formatted_content,
                        'post_content': post_content,
                        'quality_options': {},
                        'is_video_file': False,
                        'result_type': 'post_only',
                        'search_score': 1,
                        'has_poster': False,
                        'has_thumbnail': False,
                        'thumbnail': None,
                        'thumbnail_url': None,
                        'poster_url': None
                    }
                    
                    movies.append(movie_data)
                    
                    if len(movies) >= limit:
                        break
        
        # Fetch posters for all movies in batch
        if movies:
            movies_with_posters = await get_posters_for_movies_batch(movies)
            logger.info(f"✅ Fetched {len(movies_with_posters)} home movies with posters")
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
        logger.info("=" * 60)
        logger.info("🚀 SK4FiLM v9.0 - FAST & STABLE")
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
        global cache_manager, poster_fetcher
        try:
            cache_manager = CacheManager(Config)
            redis_ok = await cache_manager.init_redis()
            if redis_ok:
                logger.info("✅ Cache Manager initialized")
        except:
            logger.warning("⚠️ Cache init failed")
            cache_manager = None
        
        # Initialize PosterFetcher
        poster_fetcher = EnhancedPosterFetcher(Config, cache_manager)
        logger.info("✅ Enhanced Poster Fetcher initialized")
        
        # Initialize Verification System
        if VerificationSystem is not None:
            verification_system = VerificationSystem(Config, mongo_client)
            logger.info("✅ Verification System initialized")
        
        # Initialize Premium System
        if PremiumSystem is not None:
            premium_system = PremiumSystem(Config, mongo_client)
            logger.info("✅ Premium System initialized")
        
        # Initialize Telegram sessions
        if PYROGRAM_AVAILABLE:
            await asyncio.sleep(2)
            telegram_ok = await init_telegram_sessions()
            if not telegram_ok:
                logger.warning("⚠️ Telegram sessions failed")
        
        # Start indexing (delayed)
        if user_session_ready and files_col is not None:
            logger.info("🔄 Starting indexing in 15 seconds...")
            asyncio.create_task(delayed_indexing_start())
        
        init_time = time.time() - start_time
        logger.info(f"⚡ System started in {init_time:.2f}s")
        logger.info("=" * 60)
        
        logger.info("🔧 OPTIMIZED FEATURES:")
        logger.info(f"   • Flood Wait Protection: ✅ ENABLED")
        logger.info(f"   • Fast Search: ✅ ENABLED")
        logger.info(f"   • Real Message IDs: ✅ ENABLED")
        logger.info(f"   • Auto Indexing: ✅ ENABLED")
        logger.info(f"   • Auto Delete: ✅ ENABLED")
        logger.info(f"   • Enhanced Poster Fetching: ✅ ENABLED")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ System initialization failed: {e}")
        return False

async def delayed_indexing_start():
    """Start indexing with delay"""
    await asyncio.sleep(15)
    await initial_indexing_simple()

# ============================================================================
# ✅ API ROUTES
# ============================================================================

@app.route('/')
@performance_monitor.measure("root_endpoint")
async def root():
    """Ultra fast root endpoint"""
    try:
        # Get basic stats (non-blocking)
        total_files = 0
        if files_col is not None:
            try:
                total_files = await asyncio.wait_for(
                    files_col.count_documents({}),
                    timeout=2
                )
            except:
                pass
        
        # Quick status
        indexing_status = {
            'running': file_indexing_manager.is_running,
            'last_run': file_indexing_manager.last_run.isoformat() 
                if file_indexing_manager.last_run else None
        }
        
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
            'indexing': indexing_status,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Root endpoint error: {e}")
        return jsonify({'status': 'error'}), 500

@app.route('/health')
@performance_monitor.measure("health_endpoint")
async def health():
    """Lightweight health check"""
    try:
        return jsonify({
            'status': 'ok',
            'sessions': {
                'user': user_session_ready,
                'bot': bot_session_ready
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Health endpoint error: {e}")
        return jsonify({'status': 'error'}), 500

@app.route('/api/movies', methods=['GET'])
@performance_monitor.measure("movies_endpoint")
async def api_movies():
    """Get movies for home page"""
    try:
        movies = await get_home_movies(limit=25)
        
        return jsonify({
            'status': 'success' if movies else 'empty',
            'movies': movies,
            'total': len(movies),
            'limit': 25,
            'source': 'telegram',
            'session_used': 'user',
            'channel_id': Config.MAIN_CHANNEL_ID,
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
        else:
            total_files = 0
            video_files = 0
        
        # Get indexing stats
        indexing_status = await file_indexing_manager.get_indexing_status()
        
        return jsonify({
            'status': 'success',
            'performance': perf_stats,
            'poster_fetcher': poster_stats,
            'database_stats': {
                'total_files': total_files,
                'video_files': video_files,
            },
            'indexing_stats': indexing_status,
            'sync_stats': {
                'running': sync_manager.is_monitoring,
                'deleted_count': sync_manager.deleted_count
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

@app.route('/api/admin/reindex', methods=['POST'])
async def api_admin_reindex():
    """Admin endpoint to trigger reindexing"""
    try:
        auth_token = request.headers.get('X-Admin-Token')
        if not auth_token or auth_token != os.environ.get('ADMIN_TOKEN', 'sk4film_admin_123'):
            return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401
        
        # Trigger reindexing
        asyncio.create_task(initial_indexing_simple())
        
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
    
    # Stop indexing
    await file_indexing_manager.stop_indexing()
    await sync_manager.stop_sync_monitoring()
    
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
    
    # Execute all shutdown tasks
    if shutdown_tasks:
        results = await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"❌ Shutdown task {i} failed: {result}")
    
    # Close MongoDB
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
    
    logger.info(f"🌐 Starting SK4FiLM on port {Config.WEB_SERVER_PORT}...")
    logger.info("🔧 Features: Fast Indexing • Quick Search • Real Message IDs")
    
    try:
        asyncio.run(serve(app, config))
    except KeyboardInterrupt:
        logger.info("👋 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
