# ============================================================================
# 🚀 SK4FiLM v8.1 - QUALITY MERGING & FAST POSTERS
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

# ✅ ULTRA-FAST POSTER CONFIGURATION
POSTER_SOURCES = [
    {
        'name': 'TMDB Fast',
        'url_template': 'https://image.tmdb.org/t/p/w500{poster_path}',
        'priority': 1,
        'fallback': False
    },
    {
        'name': 'OMDB Fast',
        'url_template': 'https://img.omdbapi.com/?apikey={api_key}&i={imdb_id}',
        'priority': 2,
        'fallback': False
    },
    {
        'name': 'Google Images',
        'url_template': 'https://www.googleapis.com/customsearch/v1?key={api_key}&cx={cx}&q={query}&searchType=image&num=1',
        'priority': 3,
        'fallback': False
    },
    {
        'name': 'Placeholder Service',
        'url_template': 'https://via.placeholder.com/300x450/{color}/ffffff?text={title}',
        'priority': 4,
        'fallback': True
    },
    {
        'name': 'Local Generated',
        'url_template': '{backend_url}/api/poster?title={title}&year={year}',
        'priority': 5,
        'fallback': True
    }
]

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
    
    # API Keys for FAST POSTERS
    TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "e547e17d4e91f3e62a571655cd1ccaff")
    OMDB_API_KEY = os.environ.get("OMDB_API_KEY", "8265bd1c")
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
    GOOGLE_CX = os.environ.get("GOOGLE_CX", "")
    
    # Performance Settings
    MAX_CONCURRENT_REQUESTS = int(os.environ.get("MAX_CONCURRENT_REQUESTS", "50"))
    CACHE_TTL = int(os.environ.get("CACHE_TTL", "300"))
    REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "10"))
    
    # Sync Management Settings
    MONITOR_INTERVAL = int(os.environ.get("MONITOR_INTERVAL", "300"))
    
    # Quality Settings
    QUALITY_PRIORITY = ['2160p', '1080p', '720p', '480p', '360p']
    HEVC_VARIANTS = ['720p HEVC', '1080p HEVC', '2160p HEVC']
    
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
    response.headers['X-SK4FiLM-Version'] = '8.1-QUALITY-MERGED-FAST-POSTERS'
    response.headers['X-Response-Time'] = f"{time.perf_counter():.3f}"
    return response

# ============================================================================
# ✅ DUAL SESSION ARCHITECTURE
# ============================================================================

# GLOBAL SESSIONS
User = None        # ✅ For TEXT channel searches (-1001891090100, -1002024811395)
Bot = None         # ✅ For FILE channel operations (-1001768249569)
user_session_ready = False
bot_session_ready = False

# Database
mongo_client = None
db = None
files_col = None
verification_col = None
poster_col = None

# Components
cache_manager = None
poster_fetcher = None

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
# ✅ FAST POSTER FETCHER
# ============================================================================

class FastPosterFetcher:
    def __init__(self, cache_manager):
        self.cache_manager = cache_manager
        self.http_session = None
        self.memory_cache = {}
    
    async def init_http_session(self):
        if self.http_session is None:
            self.http_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=3),
                connector=aiohttp.TCPConnector(limit=20)
            )
    
    async def fetch_fast_poster(self, title: str, year: str = ""):
        """Fetch poster from fastest available source"""
        cache_key = f"fast_poster:{title}:{year}"
        
        # Check memory cache first
        if cache_key in self.memory_cache:
            data, expiry = self.memory_cache[cache_key]
            if expiry > datetime.now():
                return data
        
        # Check Redis cache
        if self.cache_manager is not None and self.cache_manager.redis_enabled:
            cached_data = await self.cache_manager.get(cache_key)
            if cached_data:
                self.memory_cache[cache_key] = (cached_data, datetime.now() + timedelta(hours=24))
                return cached_data
        
        # Try multiple sources concurrently
        poster_data = await self._try_all_sources_concurrently(title, year)
        
        if poster_data:
            # Cache the result
            self.memory_cache[cache_key] = (poster_data, datetime.now() + timedelta(hours=24))
            if self.cache_manager is not None:
                await self.cache_manager.set(cache_key, poster_data, expire_seconds=7*24*3600)
        
        return poster_data
    
    async def _try_all_sources_concurrently(self, title: str, year: str):
        """Try all poster sources concurrently"""
        tasks = []
        
        # TMDB source
        if Config.TMDB_API_KEY:
            tasks.append(self._try_tmdb_source(title, year))
        
        # OMDB source
        if Config.OMDB_API_KEY:
            tasks.append(self._try_omdb_source(title, year))
        
        # Google source
        if Config.GOOGLE_API_KEY and Config.GOOGLE_CX:
            tasks.append(self._try_google_source(title, year))
        
        # Add fallback sources
        tasks.append(self._try_placeholder_source(title, year))
        tasks.append(self._try_local_source(title, year))
        
        # Run all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Return first successful result
        for result in results:
            if result and not isinstance(result, Exception):
                return result
        
        # If all failed, create basic poster
        return self._create_basic_poster(title, year)
    
    async def _try_tmdb_source(self, title: str, year: str):
        """Try TMDB API for poster"""
        try:
            await self.init_http_session()
            
            search_query = urllib.parse.quote(f"{title} {year}")
            url = f"https://api.themoviedb.org/3/search/movie?api_key={Config.TMDB_API_KEY}&query={search_query}&language=en-US"
            
            async with self.http_session.get(url, timeout=2) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('results') and len(data['results']) > 0:
                        movie = data['results'][0]
                        if movie.get('poster_path'):
                            poster_url = f"https://image.tmdb.org/t/p/w500{movie['poster_path']}"
                            return {
                                'url': poster_url,
                                'source': 'tmdb',
                                'rating': str(movie.get('vote_average', '0.0')),
                                'year': str(movie.get('release_date', ''))[:4],
                                'title': movie.get('title', title),
                                'backdrop_url': f"https://image.tmdb.org/t/p/original{movie.get('backdrop_path', '')}" if movie.get('backdrop_path') else None
                            }
        except:
            pass
        return None
    
    async def _try_omdb_source(self, title: str, year: str):
        """Try OMDB API for poster"""
        try:
            await self.init_http_session()
            
            search_query = urllib.parse.quote(title)
            url = f"http://www.omdbapi.com/?apikey={Config.OMDB_API_KEY}&t={search_query}"
            if year:
                url += f"&y={year}"
            
            async with self.http_session.get(url, timeout=2) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('Poster') and data['Poster'] != 'N/A':
                        return {
                            'url': data['Poster'],
                            'source': 'omdb',
                            'rating': data.get('imdbRating', '0.0'),
                            'year': data.get('Year', year),
                            'title': data.get('Title', title),
                            'plot': data.get('Plot', '')
                        }
        except:
            pass
        return None
    
    async def _try_google_source(self, title: str, year: str):
        """Try Google Custom Search for poster"""
        try:
            await self.init_http_session()
            
            search_query = urllib.parse.quote(f"{title} {year} movie poster")
            url = f"https://www.googleapis.com/customsearch/v1?key={Config.GOOGLE_API_KEY}&cx={Config.GOOGLE_CX}&q={search_query}&searchType=image&num=1"
            
            async with self.http_session.get(url, timeout=2) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('items') and len(data['items']) > 0:
                        image_url = data['items'][0].get('link')
                        if image_url:
                            return {
                                'url': image_url,
                                'source': 'google',
                                'rating': '0.0',
                                'year': year,
                                'title': title
                            }
        except:
            pass
        return None
    
    async def _try_placeholder_source(self, title: str, year: str):
        """Generate placeholder poster"""
        try:
            title_hash = hashlib.md5(title.encode()).hexdigest()
            color = f"#{title_hash[:6]}"
            
            encoded_title = urllib.parse.quote(title[:50])
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
        except:
            return None
    
    async def _try_local_source(self, title: str, year: str):
        """Try local poster generator"""
        try:
            encoded_title = urllib.parse.quote(title[:50])
            if year:
                poster_url = f"{Config.BACKEND_URL}/api/poster?title={encoded_title}&year={year}"
            else:
                poster_url = f"{Config.BACKEND_URL}/api/poster?title={encoded_title}"
            
            return {
                'url': poster_url,
                'source': 'local',
                'rating': '0.0',
                'year': year,
                'title': title
            }
        except:
            return None
    
    def _create_basic_poster(self, title: str, year: str):
        """Create basic poster as last resort"""
        clean_title = ''.join(c for c in title if c.isalnum() or c in ' _-')
        encoded_title = urllib.parse.quote(clean_title[:50])
        
        poster_url = f"https://via.placeholder.com/300x450/1a1a2e/00ccff?text={encoded_title}"
        if year:
            poster_url += f"&subtext={year}"
        
        return {
            'url': poster_url,
            'source': 'basic',
            'rating': '0.0',
            'year': year,
            'title': title
        }
    
    def clear_cache(self):
        self.memory_cache.clear()
    
    async def close(self):
        if self.http_session is not None:
            await self.http_session.close()

# ============================================================================
# ✅ FILE THUMBNAIL EXTRACTOR
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
                            norm_title = normalize_title_cached(title)
                            if norm_title not in channel_posts:
                                # Get FAST poster
                                year_match = re.search(r'\b(19|20)\d{2}\b', title)
                                year = year_match.group() if year_match else ""
                                
                                poster_data = await poster_fetcher.fetch_fast_poster(title, year) if poster_fetcher else None
                                poster_url = poster_data['url'] if poster_data else Config.get_poster(title, year)
                                
                                channel_posts[norm_title] = {
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
                                    'thumbnail': poster_url,
                                    'thumbnail_source': poster_data['source'] if poster_data else 'local',
                                    'poster_url': poster_url,
                                    'poster_source': poster_data['source'] if poster_data else 'local',
                                    'poster_rating': poster_data.get('rating', '0.0') if poster_data else '0.0',
                                    'is_video_file': False,
                                    'year': year
                                }
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
                    norm_title = doc.get('normalized_title', normalize_title_cached(doc['title']))
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
                        # Get poster for file
                        title = doc['title']
                        year_match = re.search(r'\b(19|20)\d{2}\b', title)
                        year = year_match.group() if year_match else ""
                        
                        poster_data = await poster_fetcher.fetch_fast_poster(title, year) if poster_fetcher else None
                        
                        # Generate video thumbnail
                        if doc.get('is_video_file'):
                            thumbnail_url = FileThumbnailExtractor.generate_video_thumbnail(title, quality)
                            thumbnail_source = 'video_file'
                        else:
                            thumbnail_url = poster_data['url'] if poster_data else Config.get_poster(title, year)
                            thumbnail_source = poster_data['source'] if poster_data else 'local'
                        
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
                            'thumbnail': thumbnail_url,
                            'thumbnail_source': thumbnail_source,
                            'poster_url': poster_data['url'] if poster_data else thumbnail_url,
                            'poster_source': poster_data['source'] if poster_data else thumbnail_source,
                            'poster_rating': poster_data.get('rating', '0.0') if poster_data else '0.0',
                            'file_caption': doc.get('caption', ''),
                            'year': year
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
            
            # Use better thumbnail if available
            if file_data.get('is_video_file'):
                result['thumbnail'] = file_data['thumbnail']
                result['thumbnail_source'] = file_data['thumbnail_source']
            
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
        
        # Ensure required fields
        if 'thumbnail' not in result or not result['thumbnail']:
            year_match = re.search(r'\b(19|20)\d{2}\b', result['title'])
            year = year_match.group() if year_match else result.get('year', '')
            
            poster_data = await poster_fetcher.fetch_fast_poster(result['title'], year) if poster_fetcher else None
            if poster_data:
                result['thumbnail'] = poster_data['url']
                result['thumbnail_source'] = poster_data['source']
                result['poster_url'] = poster_data['url']
                result['poster_source'] = poster_data['source']
                result['poster_rating'] = poster_data.get('rating', '0.0')
            else:
                result['thumbnail'] = Config.get_poster(result['title'], year)
                result['thumbnail_source'] = 'local'
        
        merged[norm_title] = result
    
    # ============================================================================
    # ✅ 5. SORT AND PAGINATE
    # ============================================================================
    results_list = list(merged.values())
    
    # Sort: Has files first, then new, then by date
    results_list.sort(key=lambda x: (
        x['has_file'],
        x.get('is_new', False),
        x['date'] if isinstance(x.get('date'), str) else ''
    ), reverse=True)
    
    total = len(results_list)
    paginated = results_list[offset:offset + limit]
    
    # Statistics
    stats = {
        'total': total,
        'with_files': sum(1 for r in results_list if r['has_file']),
        'with_posts': sum(1 for r in results_list if r['has_post']),
        'both': sum(1 for r in results_list if r['has_file'] and r['has_post']),
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
            'fast_posters': True,
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
# ✅ HOME MOVIES WITH FAST POSTERS (6/6)
# ============================================================================

@performance_monitor.measure("home_movies_fast")
@async_cache_with_ttl(maxsize=1, ttl=60)
async def get_home_movies_fast(limit=6):
    """Get home movies with FAST POSTERS (6/6)"""
    try:
        if User is None or not user_session_ready:
            return []
        
        movies = []
        seen_titles = set()
        
        logger.info(f"🎬 Fetching home movies (6/6) with FAST posters...")
        
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
                    
                    # ✅ GET FAST POSTER (concurrent)
                    poster_task = poster_fetcher.fetch_fast_poster(clean_title, year) if poster_fetcher else None
                    
                    # Format content
                    post_content = msg.text
                    formatted_content = format_post(msg.text, max_length=500)
                    
                    # Get poster result
                    poster_data = await poster_task if poster_task else None
                    
                    movie_data = {
                        'title': clean_title,
                        'original_title': title,
                        'year': year,
                        'date': msg.date.isoformat() if isinstance(msg.date, datetime) else str(msg.date),
                        'is_new': is_new(msg.date) if msg.date else False,
                        'channel': channel_name_cached(Config.MAIN_CHANNEL_ID),
                        'channel_id': Config.MAIN_CHANNEL_ID,
                        'message_id': msg.id,
                        'has_poster': True,
                        'poster_url': poster_data['url'] if poster_data else Config.get_poster(clean_title, year),
                        'poster_source': poster_data['source'] if poster_data else 'local',
                        'poster_rating': poster_data.get('rating', '0.0') if poster_data else '0.0',
                        'thumbnail': poster_data['url'] if poster_data else Config.get_poster(clean_title, year),
                        'thumbnail_source': poster_data['source'] if poster_data else 'local',
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
        
        logger.info(f"✅ Fetched {len(movies)} home movies with FAST posters")
        return movies[:limit]
        
    except Exception as e:
        logger.error(f"❌ Home movies error: {e}")
        return []

# ============================================================================
# ✅ API FUNCTIONS UPDATED
# ============================================================================

@performance_monitor.measure("search_api_merged")
async def search_movies_api_merged(query, limit=12, page=1):
    try:
        result_data = await search_movies_multi_channel_merged(query, limit, page)
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

# ============================================================================
# ✅ INITIALIZATION UPDATED
# ============================================================================

@performance_monitor.measure("system_init_enhanced")
async def init_system():
    start_time = time.time()
    
    try:
        logger.info("🚀 Starting SK4FiLM v8.1 - QUALITY MERGING & FAST POSTERS...")
        
        # Initialize MongoDB
        mongo_ok = await init_mongodb()
        if not mongo_ok:
            logger.warning("⚠️ MongoDB connection failed")
        
        # Initialize Cache Manager
        global cache_manager, poster_fetcher
        cache_manager = CacheManager(Config)
        redis_ok = await cache_manager.init_redis()
        if redis_ok:
            logger.info("✅ Cache Manager initialized")
            await cache_manager.start_cleanup_task()
        
        # Initialize FAST Poster Fetcher
        poster_fetcher = FastPosterFetcher(cache_manager)
        await poster_fetcher.init_http_session()
        logger.info("✅ FAST Poster Fetcher initialized")
        
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
        
        logger.info("🔧 ENHANCED FEATURES:")
        logger.info(f"   • Quality Merging: ✅ ENABLED")
        logger.info(f"   • FAST Posters: ✅ ENABLED")
        logger.info(f"   • Video Thumbnails: ✅ ENABLED")
        logger.info(f"   • Home Movies: 6/6 with FAST posters")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ System initialization failed: {e}")
        return False

# ============================================================================
# ✅ UPDATED API ROUTES
# ============================================================================

@app.route('/api/search', methods=['GET'])
@performance_monitor.measure("search_endpoint_enhanced")
async def api_search_enhanced():
    try:
        query = request.args.get('query', '').strip()
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 12))
        
        if len(query) < 2:
            return jsonify({
                'status': 'error',
                'message': 'Query must be at least 2 characters'
            }), 400
        
        result_data = await search_movies_api_merged(query, limit, page)
        
        return jsonify({
            'status': 'success',
            'query': query,
            'results': result_data['results'],
            'pagination': result_data['pagination'],
            'search_metadata': {
                **result_data.get('search_metadata', {}),
                'feature': 'quality_merged_fast_posters',
                'thumbnail_sources': THUMBNAIL_SOURCES,
                'quality_priority': Config.QUALITY_PRIORITY
            },
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Search API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/movies', methods=['GET'])
@performance_monitor.measure("movies_endpoint_fast")
async def api_movies_fast():
    try:
        # Get 6 home movies with FAST posters
        movies = await get_home_movies_fast(limit=6)
        
        return jsonify({
            'status': 'success' if movies else 'empty',
            'movies': movies,
            'total': len(movies),
            'limit': 6,
            'source': 'telegram_fast',
            'poster_sources': POSTER_SOURCES,
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

@app.route('/api/poster/fast', methods=['GET'])
async def api_poster_fast():
    try:
        title = request.args.get('title', '').strip()
        year = request.args.get('year', '')
        
        if not title:
            return jsonify({
                'status': 'error',
                'message': 'Title is required'
            }), 400
        
        if poster_fetcher is not None:
            poster_data = await poster_fetcher.fetch_fast_poster(title, year)
            if poster_data:
                return jsonify({
                    'status': 'success',
                    'poster': poster_data,
                    'sources_tried': POSTER_SOURCES,
                    'timestamp': datetime.now().isoformat()
                })
        
        # Fallback
        poster_url = Config.get_poster(title, year)
        return jsonify({
            'status': 'success',
            'poster': {
                'url': poster_url,
                'source': 'fallback',
                'rating': '0.0',
                'year': year,
                'title': title
            },
            'timestamp': datetime.now().isoformat()
        })
                
    except Exception as e:
        logger.error(f"Fast Poster API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# ============================================================================
# ✅ NEW ENDPOINTS FOR QUALITY INFO
# ============================================================================

@app.route('/api/quality/info', methods=['GET'])
async def api_quality_info():
    try:
        filename = request.args.get('filename', '')
        if not filename:
            return jsonify({
                'status': 'error',
                'message': 'Filename is required'
            }), 400
        
        quality_info = extract_quality_info(filename)
        
        return jsonify({
            'status': 'success',
            'filename': filename,
            'quality_info': quality_info,
            'quality_priority': Config.QUALITY_PRIORITY,
            'hevc_variants': Config.HEVC_VARIANTS
        })
    except Exception as e:
        logger.error(f"Quality info API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/thumbnail/generate', methods=['GET'])
async def api_thumbnail_generate():
    try:
        title = request.args.get('title', 'Movie')
        quality = request.args.get('quality', '1080p')
        file_type = request.args.get('type', 'video')
        
        if file_type == 'video':
            thumbnail_url = FileThumbnailExtractor.generate_video_thumbnail(title, quality)
        else:
            thumbnail_url = Config.get_poster(title)
        
        return jsonify({
            'status': 'success',
            'thumbnail_url': thumbnail_url,
            'title': title,
            'quality': quality,
            'file_type': file_type,
            'file_icon': FileThumbnailExtractor.get_file_type_icon(f"{title}.mp4")
        })
    except Exception as e:
        logger.error(f"Thumbnail generate API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# ============================================================================
# ✅ SHUTDOWN UPDATED
# ============================================================================

@app.after_serving
async def shutdown():
    logger.info("🛑 Shutting down SK4FiLM v8.1...")
    
    shutdown_tasks = []
    
    await channel_sync_manager.stop_sync_monitoring()
    
    if User is not None:
        shutdown_tasks.append(User.stop())
    
    if Bot is not None:
        shutdown_tasks.append(Bot.stop())
    
    if cache_manager is not None:
        shutdown_tasks.append(cache_manager.stop())
    
    if poster_fetcher is not None:
        shutdown_tasks.append(poster_fetcher.close())
    
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
    
    logger.info(f"🌐 Starting SK4FiLM v8.1 on port {Config.WEB_SERVER_PORT}...")
    logger.info("🎯 Features: Quality Merging + FAST Posters + 6/6 Home Movies")
    
    asyncio.run(serve(app, config))
