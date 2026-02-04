# ============================================================================
# 🚀 SK4FiLM v9.0 - COMPLETE MOVIE SEARCH WITH POSTER FETCHING FIXED
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
    # Fallback CacheManager will be defined below

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
    
    # Application
    WEBSITE_URL = os.environ.get("WEBSITE_URL", "https://sk4film.vercel.app")
    BOT_USERNAME = os.environ.get("BOT_USERNAME", "sk4filmbot")
    ADMIN_IDS = [int(x) for x in os.environ.get("ADMIN_IDS", "123456789").split(",")]
    WEB_SERVER_PORT = int(os.environ.get("PORT", 8000))
    BACKEND_URL = os.environ.get("BACKEND_URL", "https://sk4film.koyeb.app")
    
    # ✅ POSTER FETCHING API KEYS - IMPORTANT!
    TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")
    OMDB_API_KEY = os.environ.get("OMDB_API_KEY", "")
    
    # Performance Settings
    MAX_CONCURRENT_REQUESTS = int(os.environ.get("MAX_CONCURRENT_REQUESTS", "50"))
    CACHE_TTL = int(os.environ.get("CACHE_TTL", "300"))
    REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "10"))
    
    # Sync Management Settings
    MONITOR_INTERVAL = int(os.environ.get("MONITOR_INTERVAL", "300"))
    
    # Quality Settings
    QUALITY_PRIORITY = ['2160p', '1080p', '720p', '480p', '360p']
    HEVC_VARIANTS = ['720p HEVC', '1080p HEVC', '2160p HEVC']
    
    # ✅ FALLBACK POSTER - MUST WORK!
    FALLBACK_POSTER = "https://image.tmdb.org/t/p/w500/wwemzKWzjKYJFfCeiB57q3r4Bcm.png"  # Netflix logo
    
    # FILE CHANNEL INDEXING SETTINGS
    AUTO_INDEX_INTERVAL = int(os.environ.get("AUTO_INDEX_INTERVAL", "120"))
    BATCH_INDEX_SIZE = int(os.environ.get("BATCH_INDEX_SIZE", "500"))
    MAX_INDEX_LIMIT = int(os.environ.get("MAX_INDEX_LIMIT", "0"))
    
    # SEARCH SETTINGS
    SEARCH_MIN_QUERY_LENGTH = 2
    SEARCH_RESULTS_PER_PAGE = 12
    SEARCH_CACHE_TTL = 600

# ============================================================================
# ✅ APP INITIALIZATION
# ============================================================================

app = Quart(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

@app.after_request
async def add_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['X-SK4FiLM-Version'] = '9.0-POSTER-FIXED'
    response.headers['X-Response-Time'] = f"{time.perf_counter():.3f}"
    return response

# ============================================================================
# ✅ GLOBAL COMPONENTS
# ============================================================================

mongo_client = None
db = None
files_col = None
verification_col = None

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

cache_manager = None
verification_system = None
premium_system = None
poster_fetcher = None
bot_handler = None
telegram_bot = None

is_indexing = False
last_index_time = None
indexing_task = None

# ============================================================================
# ✅ CACHE MANAGER - COMPLETE IMPLEMENTATION
# ============================================================================

class CacheManager:
    """Complete Cache Manager implementation"""
    
    def __init__(self, config):
        self.config = config
        self.redis_enabled = False
        self.redis_client = None
        self.cleanup_task = None
    
    async def init_redis(self):
        """Initialize Redis connection"""
        try:
            if not self.config.REDIS_URL:
                logger.warning("⚠️ Redis URL not configured")
                return False
            
            self.redis_client = redis.from_url(
                self.config.REDIS_URL,
                password=self.config.REDIS_PASSWORD or None,
                decode_responses=True,
                socket_keepalive=True,
                max_connections=10
            )
            
            await self.redis_client.ping()
            self.redis_enabled = True
            logger.info("✅ Redis connection established")
            return True
            
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            return False
    
    async def get(self, key):
        """Get value from cache"""
        if not self.redis_enabled or not self.redis_client:
            return None
        
        try:
            value = await self.redis_client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.debug(f"Cache get error: {e}")
            return None
    
    async def set(self, key, value, expire_seconds=0):
        """Set value in cache"""
        if not self.redis_enabled or not self.redis_client:
            return
        
        try:
            json_value = json.dumps(value)
            if expire_seconds > 0:
                await self.redis_client.setex(key, expire_seconds, json_value)
            else:
                await self.redis_client.set(key, json_value)
        except Exception as e:
            logger.debug(f"Cache set error: {e}")
    
    async def delete(self, key):
        """Delete key from cache"""
        if not self.redis_enabled or not self.redis_client:
            return
        
        try:
            await self.redis_client.delete(key)
        except:
            pass
    
    async def start_cleanup_task(self):
        """Start cache cleanup task"""
        if not self.redis_enabled:
            return
        
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(3600)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Cache cleanup error: {e}")
                    await asyncio.sleep(60)
        
        self.cleanup_task = asyncio.create_task(cleanup_loop())
        logger.info("✅ Cache cleanup task started")
    
    async def stop(self):
        """Stop cache manager"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self.redis_client:
            await self.redis_client.close()
            logger.info("✅ Redis connection closed")
        
        self.redis_enabled = False

# ============================================================================
# ✅ POSTER FETCHER - COMPLETE AND WORKING IMPLEMENTATION
# ============================================================================

class PosterFetcher:
    """Complete Poster Fetcher that actually works"""
    
    def __init__(self, config, cache_manager=None):
        self.config = config
        self.cache_manager = cache_manager
        self.session = None
        self.stats = {
            'tmdb_hits': 0,
            'omdb_hits': 0,
            'fallback_hits': 0,
            'total_requests': 0,
            'errors': 0
        }
    
    async def initialize(self):
        """Initialize poster fetcher"""
        try:
            # Create aiohttp session with timeout
            timeout = aiohttp.ClientTimeout(total=10)
            self.session = aiohttp.ClientSession(timeout=timeout)
            logger.info(f"✅ Poster Fetcher initialized | TMDB: {'✅' if self.config.TMDB_API_KEY else '❌'} | OMDB: {'✅' if self.config.OMDB_API_KEY else '❌'}")
            return True
        except Exception as e:
            logger.error(f"❌ Poster Fetcher initialization error: {e}")
            return False
    
    async def fetch_poster(self, title, year=""):
        """Fetch poster for movie - MAIN FUNCTION"""
        self.stats['total_requests'] += 1
        
        # Check cache first
        cache_key = f"poster:{title}:{year}"
        if self.cache_manager and self.cache_manager.redis_enabled:
            cached = await self.cache_manager.get(cache_key)
            if cached:
                logger.debug(f"✅ Cache HIT for poster: {title}")
                return cached
        
        # Clean title for search
        search_title = self._clean_title_for_search(title)
        
        logger.debug(f"🔍 Searching poster for: {search_title} (original: {title})")
        
        # Try TMDB first if API key is available
        tmdb_result = None
        if self.config.TMDB_API_KEY and self.config.TMDB_API_KEY.strip():
            tmdb_result = await self._fetch_from_tmdb(search_title, year)
            if tmdb_result and tmdb_result.get('poster_url'):
                self.stats['tmdb_hits'] += 1
                logger.info(f"✅ TMDB poster found: {title}")
                
                # Cache the result
                if self.cache_manager:
                    await self.cache_manager.set(cache_key, tmdb_result, expire_seconds=86400)  # 24 hours
                
                return tmdb_result
        
        # Try OMDB if API key is available
        omdb_result = None
        if self.config.OMDB_API_KEY and self.config.OMDB_API_KEY.strip():
            omdb_result = await self._fetch_from_omdb(search_title, year)
            if omdb_result and omdb_result.get('poster_url'):
                self.stats['omdb_hits'] += 1
                logger.info(f"✅ OMDB poster found: {title}")
                
                # Cache the result
                if self.cache_manager:
                    await self.cache_manager.set(cache_key, omdb_result, expire_seconds=86400)
                
                return omdb_result
        
        # Fallback to default poster
        self.stats['fallback_hits'] += 1
        logger.info(f"⚠️ Using fallback poster for: {title}")
        
        fallback_result = {
            'poster_url': self.config.FALLBACK_POSTER,
            'source': 'fallback',
            'rating': '0.0',
            'year': year or '',
            'title': title,
            'search_title': search_title
        }
        
        # Cache fallback too
        if self.cache_manager:
            await self.cache_manager.set(cache_key, fallback_result, expire_seconds=3600)  # 1 hour
        
        return fallback_result
    
    def _clean_title_for_search(self, title):
        """Clean title for API search"""
        if not title:
            return ""
        
        # Remove year in parentheses
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
            r'\bbluray\b',
            r'\bwebdl\b',
            r'\bwebrip\b',
            r'\bdvdrip\b',
            r'\[.*?\]',
            r'\(.*?\)',
            r'\s+-\s+.*$'  # Remove anything after dash
        ]
        
        for pattern in quality_patterns:
            title = re.sub(pattern, '', title, flags=re.IGNORECASE)
        
        # Clean up extra spaces
        title = re.sub(r'\s+', ' ', title)
        title = title.strip()
        
        # Remove common suffixes
        title = re.sub(r'(:.*)$', '', title)  # Remove after colon
        
        return title[:100]  # Limit length
    
    async def _fetch_from_tmdb(self, title, year=""):
        """Fetch from TMDB API"""
        try:
            # First, search for movie
            search_url = "https://api.themoviedb.org/3/search/movie"
            params = {
                'api_key': self.config.TMDB_API_KEY.strip(),
                'query': title,
                'language': 'en-US',
                'page': 1,
                'include_adult': 'false'
            }
            
            if year and year.isdigit():
                params['year'] = year
            
            logger.debug(f"🔍 TMDB Search: {title} (year: {year})")
            
            async with self.session.get(search_url, params=params, timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('results') and len(data['results']) > 0:
                        # Get the first result
                        movie = data['results'][0]
                        poster_path = movie.get('poster_path')
                        
                        if poster_path:
                            poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
                            
                            return {
                                'poster_url': poster_url,
                                'source': 'tmdb',
                                'rating': str(movie.get('vote_average', '0.0')),
                                'year': str(movie.get('release_date', '')[:4]) if movie.get('release_date') else year,
                                'title': movie.get('title', title),
                                'original_title': movie.get('original_title', title),
                                'overview': movie.get('overview', ''),
                                'search_title': title
                            }
                        else:
                            logger.debug(f"TMDB: No poster for {title}")
                    else:
                        logger.debug(f"TMDB: No results for {title}")
                else:
                    logger.debug(f"TMDB API error: {response.status}")
        
        except asyncio.TimeoutError:
            logger.warning(f"⏱️ TMDB timeout for: {title}")
        except Exception as e:
            logger.debug(f"TMDB error for {title}: {e}")
            self.stats['errors'] += 1
        
        return None
    
    async def _fetch_from_omdb(self, title, year=""):
        """Fetch from OMDB API"""
        try:
            search_url = "http://www.omdbapi.com/"
            params = {
                'apikey': self.config.OMDB_API_KEY.strip(),
                't': title,
                'plot': 'short',
                'r': 'json'
            }
            
            if year and year.isdigit():
                params['y'] = year
            
            logger.debug(f"🔍 OMDB Search: {title} (year: {year})")
            
            async with self.session.get(search_url, params=params, timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('Response') == 'True' and data.get('Poster') and data['Poster'] != 'N/A':
                        return {
                            'poster_url': data['Poster'],
                            'source': 'omdb',
                            'rating': data.get('imdbRating', '0.0'),
                            'year': data.get('Year', year),
                            'title': data.get('Title', title),
                            'original_title': data.get('Title', title),
                            'overview': data.get('Plot', ''),
                            'search_title': title
                        }
                    else:
                        logger.debug(f"OMDB: No results for {title}")
                else:
                    logger.debug(f"OMDB API error: {response.status}")
        
        except asyncio.TimeoutError:
            logger.warning(f"⏱️ OMDB timeout for: {title}")
        except Exception as e:
            logger.debug(f"OMDB error for {title}: {e}")
            self.stats['errors'] += 1
        
        return None
    
    def get_stats(self):
        """Get poster fetcher statistics"""
        stats = self.stats.copy()
        stats['success_rate'] = f"{((stats['tmdb_hits'] + stats['omdb_hits']) / stats['total_requests'] * 100):.1f}%" if stats['total_requests'] > 0 else "0%"
        return stats
    
    async def close(self):
        """Close poster fetcher"""
        if self.session:
            await self.session.close()
            logger.info("✅ Poster Fetcher session closed")

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
    """Enhanced quality detection"""
    if not filename:
        return "480p"
    
    filename_lower = filename.lower()
    
    # Check for HEVC variants
    is_hevc = any(re.search(pattern, filename_lower) for pattern in HEVC_PATTERNS)
    
    # Check quality
    for pattern, quality in QUALITY_PATTERNS:
        if re.search(pattern, filename_lower):
            if is_hevc and quality in ['720p', '1080p', '2160p']:
                return f"{quality} HEVC"
            return quality
    
    return "480p"

# ============================================================================
# ✅ QUALITY MERGER
# ============================================================================

class QualityMerger:
    """Merge multiple qualities for same title"""
    
    @staticmethod
    def merge_quality_options(quality_options_list):
        """Merge quality options from multiple files"""
        if not quality_options_list:
            return {}
        
        merged = {}
        
        for quality_option in quality_options_list:
            for quality, data in quality_option.items():
                if quality not in merged:
                    merged[quality] = data
                else:
                    # Keep the one with larger file size (better quality)
                    if data.get('file_size', 0) > merged[quality].get('file_size', 0):
                        merged[quality] = data
        
        return merged
    
    @staticmethod
    def create_combined_quality_options(file_list):
        """Create combined quality options from multiple files"""
        if not file_list:
            return {}
        
        all_quality_options = []
        
        for file_data in file_list:
            quality = file_data.get('quality', '480p')
            quality_options = {
                quality: {
                    'quality': quality,
                    'file_size': file_data.get('file_size', 0),
                    'message_id': file_data.get('message_id'),
                    'file_id': file_data.get('file_id'),
                    'telegram_file_id': file_data.get('telegram_file_id'),
                    'file_name': file_data.get('file_name', ''),
                    'is_video_file': file_data.get('is_video_file', False),
                    'real_message_id': file_data.get('real_message_id')
                }
            }
            all_quality_options.append(quality_options)
        
        return QualityMerger.merge_quality_options(all_quality_options)

# ============================================================================
# ✅ POSTER FETCHING FUNCTIONS
# ============================================================================

async def get_poster_for_movie(title: str, year: str = "") -> Dict[str, Any]:
    """Get poster for movie - SIMPLE WRAPPER"""
    global poster_fetcher
    
    if not poster_fetcher:
        return {
            'poster_url': Config.FALLBACK_POSTER,
            'source': 'fallback',
            'rating': '0.0',
            'year': year,
            'title': title
        }
    
    try:
        poster_data = await poster_fetcher.fetch_poster(title, year)
        return poster_data
    except Exception as e:
        logger.error(f"❌ Error fetching poster for {title}: {e}")
        return {
            'poster_url': Config.FALLBACK_POSTER,
            'source': 'error',
            'rating': '0.0',
            'year': year,
            'title': title
        }

async def get_posters_for_movies_batch(movies: List[Dict]) -> List[Dict]:
    """Get posters for multiple movies in batch"""
    if not poster_fetcher:
        logger.warning("⚠️ Poster fetcher not available, using fallback")
        results = []
        for movie in movies:
            movie_with_poster = movie.copy()
            movie_with_poster.update({
                'poster_url': Config.FALLBACK_POSTER,
                'poster_source': 'fallback',
                'poster_rating': '0.0',
                'has_poster': True,
                'has_thumbnail': True,
                'thumbnail': Config.FALLBACK_POSTER,
                'thumbnail_source': 'fallback'
            })
            results.append(movie_with_poster)
        return results
    
    results = []
    
    # Create tasks for all movies
    tasks = []
    for movie in movies:
        title = movie.get('title', '')
        year = movie.get('year', '')
        
        task = asyncio.create_task(get_poster_for_movie(title, year))
        tasks.append((movie, task))
    
    # Process results
    for movie, task in tasks:
        try:
            poster_data = await task
            
            # Update movie with poster data
            movie_with_poster = movie.copy()
            movie_with_poster.update({
                'poster_url': poster_data.get('poster_url', Config.FALLBACK_POSTER),
                'poster_source': poster_data.get('source', 'fallback'),
                'poster_rating': poster_data.get('rating', '0.0'),
                'has_poster': True,
                'has_thumbnail': True,
                'thumbnail': poster_data.get('poster_url', Config.FALLBACK_POSTER),
                'thumbnail_source': poster_data.get('source', 'fallback')
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
                'has_poster': True,
                'has_thumbnail': True,
                'thumbnail': Config.FALLBACK_POSTER,
                'thumbnail_source': 'fallback'
            })
            
            results.append(movie_with_fallback)
    
    logger.info(f"✅ Fetched posters for {len(results)} movies")
    return results

# ============================================================================
# ✅ ENHANCED SEARCH FUNCTION - WITH PROPER POSTER FETCHING
# ============================================================================

def channel_name_cached(cid):
    return f"Channel {cid}"

@performance_monitor.measure("enhanced_search_with_posters")
@async_cache_with_ttl(maxsize=500, ttl=Config.SEARCH_CACHE_TTL)
async def search_movies_enhanced_fixed(query, limit=15, page=1):
    """Search movies with proper poster fetching"""
    offset = (page - 1) * limit
    
    # Try cache first
    cache_key = f"search:{query}:{page}:{limit}"
    if cache_manager is not None and cache_manager.redis_enabled:
        cached_data = await cache_manager.get(cache_key)
        if cached_data:
            logger.info(f"✅ Cache HIT for: {query}")
            return cached_data
    
    logger.info(f"🔍 SEARCH for: {query} (page: {page})")
    
    query_lower = query.lower()
    all_entries = {}  # normalized_title -> {'posts': [], 'files': []}
    
    # ============================================================================
    # ✅ PHASE 1: SEARCH TEXT CHANNELS
    # ============================================================================
    if user_session_ready and User is not None:
        logger.info("📝 Searching text channels...")
        
        for channel_id in Config.TEXT_CHANNEL_IDS:
            try:
                cname = channel_name_cached(channel_id)
                async for msg in User.search_messages(channel_id, query=query, limit=10):
                    if msg and msg.text and len(msg.text) > 15:
                        title = extract_title_smart(msg.text)
                        if title and (query_lower in title.lower() or query_lower in msg.text.lower()):
                            norm_title = normalize_title(title)
                            
                            # Get year
                            year_match = re.search(r'\b(19|20)\d{2}\b', title)
                            year = year_match.group() if year_match else ""
                            
                            # Clean title
                            clean_title = re.sub(r'\s+\(\d{4}\)$', '', title)
                            clean_title = re.sub(r'\s+\d{4}$', '', clean_title)
                            
                            # Create post data
                            post_data = {
                                'title': clean_title,
                                'original_title': title,
                                'normalized_title': norm_title,
                                'content': format_post(msg.text, max_length=500),
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
                                'year': year,
                                'search_score': 3,
                                'result_type': 'post_only',
                                'thumbnail_url': None,
                                'has_thumbnail': False,
                                'bot_username': Config.BOT_USERNAME
                            }
                            
                            if norm_title not in all_entries:
                                all_entries[norm_title] = {'posts': [], 'files': []}
                            
                            all_entries[norm_title]['posts'].append(post_data)
                            
            except Exception as e:
                logger.error(f"Text search error in {channel_id}: {e}")
        
        logger.info(f"📝 Found {len([k for k, v in all_entries.items() if v['posts']])} posts")
    
    # ============================================================================
    # ✅ PHASE 2: SEARCH FILE CHANNEL
    # ============================================================================
    if files_col is not None:
        try:
            logger.info("📁 Searching file channel database...")
            
            search_query = {
                "$or": [
                    {"title": {"$regex": query, "$options": "i"}},
                    {"normalized_title": {"$regex": query, "$options": "i"}},
                    {"file_name": {"$regex": query, "$options": "i"}},
                    {"caption": {"$regex": query, "$options": "i"}}
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
                    'telegram_file_id': 1,
                    'thumbnail_url': 1,
                    'thumbnail_extracted': 1,
                    'year': 1,
                    '_id': 0
                }
            ).limit(100)
            
            async for doc in cursor:
                try:
                    title = doc.get('title', 'Unknown')
                    norm_title = normalize_title(title)
                    
                    quality = doc.get('quality', '480p')
                    if not quality or quality == 'unknown':
                        quality = detect_quality_enhanced(doc.get('file_name', ''))
                    
                    thumbnail_url = doc.get('thumbnail_url')
                    real_msg_id = doc.get('real_message_id') or doc.get('message_id')
                    year = doc.get('year', '')
                    
                    file_data = {
                        'title': title,
                        'original_title': title,
                        'normalized_title': norm_title,
                        'quality': quality,
                        'file_size': doc.get('file_size', 0),
                        'message_id': real_msg_id,
                        'real_message_id': real_msg_id,
                        'file_id': doc.get('file_id'),
                        'telegram_file_id': doc.get('telegram_file_id'),
                        'file_name': doc.get('file_name', ''),
                        'is_video_file': doc.get('is_video_file', False),
                        'caption': doc.get('caption', ''),
                        'thumbnail_url': thumbnail_url,
                        'channel_id': doc.get('channel_id'),
                        'channel_name': channel_name_cached(doc.get('channel_id')),
                        'year': year,
                        'date': doc['date'].isoformat() if isinstance(doc['date'], datetime) else doc['date'],
                        'has_file': True,
                        'has_post': bool(doc.get('caption')),
                        'post_content': doc.get('caption', ''),
                        'content': format_post(doc.get('caption', ''), max_length=300),
                        'has_thumbnail': thumbnail_url is not None,
                        'search_score': 2,
                        'result_type': 'file_only',
                        'bot_username': Config.BOT_USERNAME
                    }
                    
                    if norm_title not in all_entries:
                        all_entries[norm_title] = {'posts': [], 'files': []}
                    
                    all_entries[norm_title]['files'].append(file_data)
                    
                except Exception as e:
                    logger.error(f"File processing error: {e}")
                    continue
            
            logger.info(f"📁 Found {len([k for k, v in all_entries.items() if v['files']])} files")
            
        except Exception as e:
            logger.error(f"❌ File search error: {e}")
    
    # ============================================================================
    # ✅ PHASE 3: MERGE POSTS AND FILES
    # ============================================================================
    logger.info("🔄 Merging posts and files...")
    
    final_results = []
    
    for norm_title, data in all_entries.items():
        posts = data.get('posts', [])
        files = data.get('files', [])
        
        if posts and files:
            # POST + FILE COMBINED
            base_post = posts[0]
            combined_quality_options = QualityMerger.create_combined_quality_options(files)
            
            # Get best thumbnail
            best_thumbnail = None
            for file_data in files:
                if file_data.get('thumbnail_url'):
                    best_thumbnail = file_data.get('thumbnail_url')
                    break
            
            merged_result = {
                'title': base_post.get('title', ''),
                'original_title': base_post.get('original_title', ''),
                'normalized_title': norm_title,
                'content': base_post.get('content', ''),
                'post_content': base_post.get('post_content', ''),
                'channel': base_post.get('channel', ''),
                'channel_id': base_post.get('channel_id'),
                'message_id': base_post.get('message_id'),
                'date': base_post.get('date'),
                'is_new': base_post.get('is_new', False),
                'has_file': True,
                'has_post': True,
                'quality_options': combined_quality_options,
                'is_video_file': any(f.get('is_video_file', False) for f in files),
                'year': base_post.get('year', '') or (files[0].get('year', '') if files else ''),
                'search_score': 5,
                'result_type': 'post_and_file',
                'thumbnail_url': best_thumbnail or base_post.get('thumbnail_url'),
                'has_thumbnail': bool(best_thumbnail or base_post.get('thumbnail_url')),
                'real_message_id': files[0].get('real_message_id') if files else None,
                'bot_username': Config.BOT_USERNAME,
                'quality_count': len(combined_quality_options),
                'all_qualities': list(combined_quality_options.keys()),
                'combined': True,
                # Poster fields (will be filled later)
                'poster_url': None,
                'poster_source': None,
                'poster_rating': '0.0'
            }
            
            final_results.append(merged_result)
            
        elif posts and not files:
            # POST ONLY
            for post in posts:
                post_only_result = post.copy()
                post_only_result.update({
                    'has_file': False,
                    'result_type': 'post_only',
                    'search_score': 3,
                    'combined': False,
                    'poster_url': None,
                    'poster_source': None,
                    'poster_rating': '0.0'
                })
                final_results.append(post_only_result)
            
        elif not posts and files:
            # FILE ONLY
            combined_quality_options = QualityMerger.create_combined_quality_options(files)
            base_file = files[0]
            
            best_thumbnail = None
            for file_data in files:
                if file_data.get('thumbnail_url'):
                    best_thumbnail = file_data.get('thumbnail_url')
                    break
            
            file_only_result = {
                'title': base_file.get('title', ''),
                'original_title': base_file.get('original_title', ''),
                'normalized_title': norm_title,
                'content': base_file.get('content', ''),
                'post_content': base_file.get('post_content', ''),
                'channel': base_file.get('channel_name', ''),
                'channel_id': base_file.get('channel_id'),
                'message_id': base_file.get('message_id'),
                'date': base_file.get('date'),
                'is_new': is_new(base_file.get('date')) if base_file.get('date') else False,
                'has_file': True,
                'has_post': bool(base_file.get('post_content')),
                'quality_options': combined_quality_options,
                'is_video_file': any(f.get('is_video_file', False) for f in files),
                'year': base_file.get('year', ''),
                'search_score': 2,
                'result_type': 'file_only',
                'thumbnail_url': best_thumbnail,
                'has_thumbnail': bool(best_thumbnail),
                'real_message_id': base_file.get('real_message_id'),
                'bot_username': Config.BOT_USERNAME,
                'quality_count': len(combined_quality_options),
                'all_qualities': list(combined_quality_options.keys()),
                'combined': False,
                'poster_url': None,
                'poster_source': None,
                'poster_rating': '0.0'
            }
            
            final_results.append(file_only_result)
    
    # ============================================================================
    # ✅ PHASE 4: FETCH POSTERS - MOST IMPORTANT!
    # ============================================================================
    if final_results:
        logger.info(f"🎬 Fetching posters for {len(final_results)} results...")
        
        # Prepare movies for poster fetching
        movies_for_posters = []
        for result in final_results:
            movies_for_posters.append({
                'title': result.get('title', ''),
                'year': result.get('year', ''),
                'original_title': result.get('original_title', '')
            })
        
        # Fetch posters in batch
        movies_with_posters = await get_posters_for_movies_batch(movies_for_posters)
        
        # Update results with poster data
        for i, result in enumerate(final_results):
            if i < len(movies_with_posters):
                poster_data = movies_with_posters[i]
                
                # Update with poster info
                result.update({
                    'poster_url': poster_data.get('poster_url', Config.FALLBACK_POSTER),
                    'poster_source': poster_data.get('poster_source', 'fallback'),
                    'poster_rating': poster_data.get('poster_rating', '0.0'),
                    'has_poster': True,
                    'has_thumbnail': True,
                    'thumbnail': result.get('thumbnail_url') or poster_data.get('poster_url'),
                    'thumbnail_source': result.get('thumbnail_source') or poster_data.get('poster_source', 'fallback')
                })
            else:
                # Fallback
                result.update({
                    'poster_url': Config.FALLBACK_POSTER,
                    'poster_source': 'fallback',
                    'poster_rating': '0.0',
                    'has_poster': True,
                    'has_thumbnail': True,
                    'thumbnail': result.get('thumbnail_url') or Config.FALLBACK_POSTER,
                    'thumbnail_source': 'fallback'
                })
        
        logger.info(f"✅ Posters fetched successfully")
    
    # ============================================================================
    # ✅ PHASE 5: SORT AND PAGINATE
    # ============================================================================
    # Sort results
    final_results.sort(key=lambda x: (
        x.get('result_type') == 'post_and_file',
        x.get('result_type') == 'post_only',
        x.get('search_score', 0),
        x.get('is_new', False),
        x.get('date', '') if isinstance(x.get('date'), str) else ''
    ), reverse=True)
    
    # Pagination
    total = len(final_results)
    start_idx = offset
    end_idx = offset + limit
    paginated = final_results[start_idx:end_idx]
    
    # Statistics
    post_count = sum(1 for r in final_results if r.get('result_type') == 'post_only')
    file_count = sum(1 for r in final_results if r.get('result_type') == 'file_only')
    combined_count = sum(1 for r in final_results if r.get('result_type') == 'post_and_file')
    
    # Create final result
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
            'stats': {
                'total': total,
                'post_only': post_count,
                'file_only': file_count,
                'post_and_file': combined_count
            },
            'post_file_merged': True,
            'file_only_with_poster': True,
            'poster_fetcher': poster_fetcher is not None,
            'thumbnails_enabled': True,
            'real_message_ids': True,
            'search_logic': 'enhanced_with_posters'
        },
        'bot_username': Config.BOT_USERNAME
    }
    
    # Cache results
    if cache_manager is not None:
        await cache_manager.set(cache_key, result_data, expire_seconds=Config.SEARCH_CACHE_TTL)
    
    logger.info(f"✅ Search complete: {len(paginated)} results (page {page}) | Posters: ✅")
    
    return result_data

# ============================================================================
# ✅ BOT HANDLER
# ============================================================================

class BotHandler:
    """Simple bot handler"""
    
    def __init__(self, bot_token=None, api_id=None, api_hash=None):
        self.bot_token = bot_token or Config.BOT_TOKEN
        self.api_id = api_id or Config.API_ID
        self.api_hash = api_hash or Config.API_HASH
        self.bot = None
        self.initialized = False
        
    async def initialize(self):
        """Initialize bot handler"""
        if not self.bot_token or not self.api_id or not self.api_hash:
            logger.error("❌ Bot credentials not configured")
            return False
        
        try:
            from pyrogram import Client
            
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
            return True
            
        except Exception as e:
            logger.error(f"❌ Bot handler initialization error: {e}")
            return False
    
    async def get_bot_status(self):
        """Get bot handler status"""
        return {
            'initialized': self.initialized,
            'bot_username': self.bot_token[:10] + '...' if self.bot_token else None
        }
    
    async def shutdown(self):
        """Shutdown bot handler"""
        if self.bot:
            try:
                await self.bot.stop()
            except:
                pass
        self.initialized = False

bot_handler = BotHandler()

# ============================================================================
# ✅ SYNC MANAGER
# ============================================================================

class OptimizedSyncManager:
    """Sync manager"""
    
    def __init__(self):
        self.is_monitoring = False
        self.monitoring_task = None
    
    async def start_sync_monitoring(self):
        """Start sync monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        logger.info("👁️ Starting sync monitoring...")
    
    async def stop_sync_monitoring(self):
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
        logger.info("🛑 Sync monitoring stopped")

sync_manager = OptimizedSyncManager()

# ============================================================================
# ✅ FILE INDEXING MANAGER
# ============================================================================

class OptimizedFileIndexingManager:
    """File indexing manager"""
    
    def __init__(self):
        self.is_running = False
        self.indexing_task = None
    
    async def start_indexing(self):
        """Start indexing"""
        if self.is_running:
            return
        
        self.is_running = True
        logger.info("🚀 Starting file indexing...")
        
        # Simple indexing task
        async def indexing_loop():
            while self.is_running:
                try:
                    await asyncio.sleep(Config.AUTO_INDEX_INTERVAL)
                except asyncio.CancelledError:
                    break
        
        self.indexing_task = asyncio.create_task(indexing_loop())
    
    async def stop_indexing(self):
        """Stop indexing"""
        self.is_running = False
        if self.indexing_task:
            self.indexing_task.cancel()
        logger.info("🛑 File indexing stopped")
    
    async def get_indexing_status(self):
        """Get indexing status"""
        return {
            'is_running': self.is_running,
            'last_run': datetime.now().isoformat()
        }

file_indexing_manager = OptimizedFileIndexingManager()

# ============================================================================
# ✅ TELEGRAM BOT
# ============================================================================

class SK4FiLMBot:
    """Telegram Bot"""
    
    def __init__(self, config, db_manager=None):
        self.config = config
        self.db_manager = db_manager
        self.bot = None
        self.bot_started = False
        
    async def initialize(self):
        """Initialize bot"""
        try:
            if not self.config.BOT_TOKEN:
                return False
            
            from pyrogram import Client
            
            self.bot = Client(
                "sk4film_telegram_bot",
                api_id=self.config.API_ID,
                api_hash=self.config.API_HASH,
                bot_token=self.config.BOT_TOKEN,
                sleep_threshold=30,
                in_memory=True,
                no_updates=True
            )
            
            await self.bot.start()
            self.bot_started = True
            logger.info("✅ Telegram Bot started")
            return True
            
        except Exception as e:
            logger.error(f"❌ Telegram Bot error: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown bot"""
        if self.bot:
            await self.bot.stop()
        self.bot_started = False

# ============================================================================
# ✅ DATABASE INITIALIZATION
# ============================================================================

async def init_mongodb():
    global mongo_client, db, files_col, verification_col
    
    try:
        logger.info("🔌 MongoDB initialization...")
        
        mongo_client = AsyncIOMotorClient(
            Config.MONGODB_URI,
            serverSelectionTimeoutMS=10000,
            connectTimeoutMS=10000
        )
        
        await mongo_client.admin.command('ping')
        
        db = mongo_client.sk4film
        files_col = db.files
        verification_col = db.verifications
        
        logger.info("✅ MongoDB OK")
        return True
        
    except Exception as e:
        logger.error(f"❌ MongoDB error: {e}")
        return False

# ============================================================================
# ✅ TELEGRAM SESSIONS
# ============================================================================

async def init_telegram_sessions():
    global User, Bot, user_session_ready, bot_session_ready
    
    logger.info("🚀 Telegram session initialization")
    
    if not PYROGRAM_AVAILABLE:
        logger.error("❌ Pyrogram not installed!")
        return False
    
    # USER Session
    if Config.API_ID > 0 and Config.API_HASH and Config.USER_SESSION_STRING:
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
            user_session_ready = True
            logger.info("✅ USER Session Ready")
        except Exception as e:
            logger.error(f"❌ USER Session failed: {e}")
            user_session_ready = False
    
    # BOT Session
    if Config.BOT_TOKEN:
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
            bot_session_ready = True
            logger.info("✅ BOT Session Ready")
        except Exception as e:
            logger.error(f"❌ BOT Session failed: {e}")
            bot_session_ready = False
    
    return user_session_ready or bot_session_ready

# ============================================================================
# ✅ HOME MOVIES
# ============================================================================

@performance_monitor.measure("home_movies")
@async_cache_with_ttl(maxsize=1, ttl=60)
async def get_home_movies(limit=25):
    """Get home movies"""
    try:
        if User is None or not user_session_ready:
            return []
        
        movies = []
        seen_titles = set()
        
        async for msg in User.get_chat_history(Config.MAIN_CHANNEL_ID, limit=25):
            if msg and msg.text and len(msg.text) > 25:
                title = extract_title_smart(msg.text)
                
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    
                    year_match = re.search(r'\b(19|20)\d{2}\b', title)
                    year = year_match.group() if year_match else ""
                    
                    clean_title = re.sub(r'\s+\(\d{4}\)$', '', title)
                    clean_title = re.sub(r'\s+\d{4}$', '', clean_title)
                    
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
                        'content': format_post(msg.text, max_length=500),
                        'post_content': msg.text,
                        'quality_options': {},
                        'is_video_file': False,
                        'result_type': 'post_only',
                        'search_score': 1,
                        'has_poster': False,
                        'has_thumbnail': False
                    }
                    
                    movies.append(movie_data)
                    
                    if len(movies) >= limit:
                        break
        
        # Fetch posters
        if movies:
            movies_with_posters = await get_posters_for_movies_batch(movies)
            return movies_with_posters[:limit]
        else:
            return []
        
    except Exception as e:
        logger.error(f"❌ Home movies error: {e}")
        return []

# ============================================================================
# ✅ MAIN INITIALIZATION
# ============================================================================

async def init_system():
    start_time = time.time()
    
    try:
        logger.info("=" * 60)
        logger.info("🚀 SK4FiLM v9.0 - POSTER FETCHING FIXED")
        logger.info("=" * 60)
        
        # Initialize MongoDB
        mongo_ok = await init_mongodb()
        if not mongo_ok:
            logger.error("❌ MongoDB connection failed")
            return False
        
        # Initialize Cache Manager
        global cache_manager
        cache_manager = CacheManager(Config)
        redis_ok = await cache_manager.init_redis()
        if redis_ok:
            logger.info("✅ Cache Manager initialized")
            await cache_manager.start_cleanup_task()
        
        # ✅ POSTER FETCHER INITIALIZATION - MOST IMPORTANT!
        global poster_fetcher
        poster_fetcher = PosterFetcher(Config, cache_manager)
        poster_ok = await poster_fetcher.initialize()
        if poster_ok:
            logger.info("✅ Poster Fetcher initialized successfully")
        else:
            logger.warning("⚠️ Poster Fetcher failed to initialize, using fallback")
        
        # Initialize Telegram Sessions
        if PYROGRAM_AVAILABLE:
            telegram_ok = await init_telegram_sessions()
            if telegram_ok:
                logger.info("✅ Telegram sessions initialized")
        
        # Initialize other components
        global bot_handler, telegram_bot
        bot_handler = BotHandler()
        await bot_handler.initialize()
        
        telegram_bot = SK4FiLMBot(Config)
        await telegram_bot.initialize()
        
        # Start indexing
        await file_indexing_manager.start_indexing()
        await sync_manager.start_sync_monitoring()
        
        init_time = time.time() - start_time
        logger.info(f"⚡ SK4FiLM Started in {init_time:.2f}s")
        logger.info("=" * 60)
        
        logger.info("🎯 POSTER FETCHING STATUS:")
        logger.info(f"   • TMDB API Key: {'✅ CONFIGURED' if Config.TMDB_API_KEY and Config.TMDB_API_KEY.strip() else '❌ NOT CONFIGURED'}")
        logger.info(f"   • OMDB API Key: {'✅ CONFIGURED' if Config.OMDB_API_KEY and Config.OMDB_API_KEY.strip() else '❌ NOT CONFIGURED'}")
        logger.info(f"   • Fallback Poster: {Config.FALLBACK_POSTER}")
        logger.info(f"   • Poster Fetcher: {'✅ READY' if poster_fetcher and poster_ok else '❌ NOT READY'}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ System initialization failed: {e}")
        return False

# ============================================================================
# ✅ API ROUTES
# ============================================================================

@app.route('/')
async def root():
    if files_col is not None:
        tf = await files_col.count_documents({})
    else:
        tf = 0
    
    poster_stats = poster_fetcher.get_stats() if poster_fetcher else {}
    
    return jsonify({
        'status': 'healthy',
        'service': 'SK4FiLM v9.0 - POSTER FETCHING FIXED',
        'poster_fetching': {
            'enabled': poster_fetcher is not None,
            'stats': poster_stats,
            'tmdb_configured': bool(Config.TMDB_API_KEY and Config.TMDB_API_KEY.strip()),
            'omdb_configured': bool(Config.OMDB_API_KEY and Config.OMDB_API_KEY.strip()),
            'fallback_poster': Config.FALLBACK_POSTER
        },
        'sessions': {
            'user_session': user_session_ready,
            'bot_session': bot_session_ready
        },
        'stats': {
            'total_files': tf
        }
    })

@app.route('/health')
async def health():
    poster_stats = poster_fetcher.get_stats() if poster_fetcher else {}
    
    return jsonify({
        'status': 'ok',
        'poster_fetching': {
            'active': poster_fetcher is not None,
            'total_requests': poster_stats.get('total_requests', 0),
            'success_rate': poster_stats.get('success_rate', '0%'),
            'tmdb_hits': poster_stats.get('tmdb_hits', 0),
            'omdb_hits': poster_stats.get('omdb_hits', 0)
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/movies', methods=['GET'])
async def api_movies():
    try:
        movies = await get_home_movies(limit=25)
        
        # Log poster info
        if movies:
            poster_sources = [m.get('poster_source', 'unknown') for m in movies]
            logger.info(f"🎬 Home movies: {len(movies)} | Poster sources: {set(poster_sources)}")
        
        return jsonify({
            'status': 'success' if movies else 'empty',
            'movies': movies,
            'total': len(movies),
            'poster_stats': poster_fetcher.get_stats() if poster_fetcher else {}
        })
    except Exception as e:
        logger.error(f"Movies API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'movies': []
        }), 500

@app.route('/api/search', methods=['GET'])
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
        
        result_data = await search_movies_enhanced_fixed(query, limit, page)
        
        # Log poster info
        if result_data.get('results'):
            posters = [r.get('poster_source', 'unknown') for r in result_data['results']]
            logger.info(f"🔍 Search results: {len(result_data['results'])} | Poster sources: {set(posters)}")
        
        return jsonify({
            'status': 'success',
            'query': query,
            'results': result_data['results'],
            'pagination': result_data['pagination'],
            'poster_stats': poster_fetcher.get_stats() if poster_fetcher else {}
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
        
        poster_stats = poster_fetcher.get_stats() if poster_fetcher else {}
        
        # Get database stats
        if files_col is not None:
            total_files = await files_col.count_documents({})
        else:
            total_files = 0
        
        return jsonify({
            'status': 'success',
            'performance': perf_stats,
            'poster_fetcher': poster_stats,
            'database_stats': {
                'total_files': total_files
            },
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Stats API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/test-poster', methods=['GET'])
async def test_poster():
    """Test poster fetching endpoint"""
    try:
        movie_title = request.args.get('title', 'Avatar')
        movie_year = request.args.get('year', '2009')
        
        if not poster_fetcher:
            return jsonify({
                'status': 'error',
                'message': 'Poster fetcher not initialized'
            }), 500
        
        logger.info(f"🎬 Testing poster fetch for: {movie_title} ({movie_year})")
        
        # Fetch poster
        start_time = time.time()
        poster_data = await poster_fetcher.fetch_poster(movie_title, movie_year)
        elapsed = time.time() - start_time
        
        return jsonify({
            'status': 'success',
            'test_movie': {
                'title': movie_title,
                'year': movie_year
            },
            'poster_data': poster_data,
            'response_time': f"{elapsed:.2f}s",
            'poster_stats': poster_fetcher.get_stats()
        })
        
    except Exception as e:
        logger.error(f"Test poster error: {e}")
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
    
    # Shutdown components
    if poster_fetcher:
        await poster_fetcher.close()
    
    if cache_manager:
        await cache_manager.stop()
    
    if bot_handler:
        await bot_handler.shutdown()
    
    if telegram_bot:
        await telegram_bot.shutdown()
    
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
    
    logger.info(f"🌐 Starting SK4FiLM v9.0 on port {Config.WEB_SERVER_PORT}...")
    logger.info("🎯 POSTER FETCHING FIXED VERSION")
    logger.info("=" * 50)
    logger.info("IMPORTANT: Set these environment variables for poster fetching:")
    logger.info("")
    logger.info("1. TMDB API Key (Recommended):")
    logger.info("   Get from: https://www.themoviedb.org/settings/api")
    logger.info("   Set: TMDB_API_KEY=your_tmdb_api_key_here")
    logger.info("")
    logger.info("2. OMDB API Key (Alternative):")
    logger.info("   Get from: http://www.omdbapi.com/apikey.aspx")
    logger.info("   Set: OMDB_API_KEY=your_omdb_api_key_here")
    logger.info("")
    logger.info("3. Test poster fetching after startup:")
    logger.info(f"   GET /api/test-poster?title=Avatar&year=2009")
    logger.info("=" * 50)
    
    try:
        asyncio.run(serve(app, config))
    except KeyboardInterrupt:
        logger.info("👋 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
