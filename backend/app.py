# ============================================================================
# 🚀 SK4FiLM v9.0 - ULTIMATE FIXED VERSION - SINGLE RESULT WITH MERGED QUALITIES
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
        # Advanced normalization
        title = title.lower().strip()
        # Remove year in parentheses or brackets
        title = re.sub(r'\s*\([^)]*\)', '', title)
        title = re.sub(r'\s*\[[^\]]*\]', '', title)
        # Remove quality indicators
        title = re.sub(r'\b(480p|720p|1080p|2160p|4k|hd|hevc|x265|x264|bluray|webrip|webdl|dvdrip)\b', '', title, flags=re.IGNORECASE)
        # Remove extra spaces
        title = re.sub(r'\s+', ' ', title)
        return title.strip()
    
    def extract_title_smart(text):
        if not text:
            return ""
        # Extract first meaningful line
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            # Skip empty lines and URLs
            if len(line) > 10 and not line.startswith('http') and not line.startswith('@'):
                # Remove common prefixes
                line = re.sub(r'^(Movie|Film|Watch|Download|Stream)\s*[:-]?\s*', '', line, flags=re.IGNORECASE)
                if len(line) > 5:
                    return line[:150]
        return text[:80] if text else ""
    
    def extract_title_from_file(filename, caption=None):
        if filename:
            # Remove extension
            name = os.path.splitext(filename)[0]
            # Remove common separators
            name = re.sub(r'[._\-]', ' ', name)
            # Remove quality indicators
            name = re.sub(r'\b(480p|720p|1080p|2160p|4k|hd|hevc|x265|x264|bluray|webrip|webdl|dvdrip|brrip|web)\b', '', name, flags=re.IGNORECASE)
            # Remove year
            name = re.sub(r'\b(19|20)\d{2}\b', '', name)
            # Clean up
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
# ✅ CONFIGURATION - ULTIMATE FIXED VERSION
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
    AUTO_INDEX_INTERVAL = int(os.environ.get("AUTO_INDEX_INTERVAL", "120"))
    BATCH_INDEX_SIZE = int(os.environ.get("BATCH_INDEX_SIZE", "500"))
    MAX_INDEX_LIMIT = int(os.environ.get("MAX_INDEX_LIMIT", "0"))
    INDEX_ALL_HISTORY = os.environ.get("INDEX_ALL_HISTORY", "true").lower() == "true"
    INSTANT_AUTO_INDEX = os.environ.get("INSTANT_AUTO_INDEX", "true").lower() == "true"
    
    # 🔥 SEARCH SETTINGS - ULTIMATE MERGING
    SEARCH_MIN_QUERY_LENGTH = 2
    SEARCH_RESULTS_PER_PAGE = 12
    SEARCH_CACHE_TTL = 600  # 10 minutes
    MERGE_SAME_TITLES = True  # ✅ FORCE MERGING
    MERGE_QUALITIES = True  # ✅ FORCE QUALITY MERGING

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
    response.headers['X-SK4FiLM-Version'] = '9.0-ULTIMATE-FIXED'
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
# ✅ IMPROVED TITLE NORMALIZATION
# ============================================================================

def advanced_normalize_title(title):
    """Advanced title normalization for better matching"""
    if not title:
        return ""
    
    # Convert to lowercase
    normalized = title.lower().strip()
    
    # Remove common prefixes
    prefixes = [
        r'^movie\s*[:-]?\s*',
        r'^film\s*[:-]?\s*', 
        r'^watch\s*[:-]?\s*',
        r'^download\s*[:-]?\s*',
        r'^stream\s*[:-]?\s*',
        r'^full\s*[:-]?\s*',
        r'^hd\s*[:-]?\s*'
    ]
    
    for prefix in prefixes:
        normalized = re.sub(prefix, '', normalized, flags=re.IGNORECASE)
    
    # Remove year in parentheses
    normalized = re.sub(r'\s*\(\s*(19|20)\d{2}\s*\)', '', normalized)
    normalized = re.sub(r'\s*\[\s*(19|20)\d{2}\s*\]', '', normalized)
    
    # Remove standalone year
    normalized = re.sub(r'\s+(19|20)\d{2}$', '', normalized)
    normalized = re.sub(r'\s+(19|20)\d{2}\s+', ' ', normalized)
    
    # Remove quality indicators
    quality_patterns = [
        r'\b(480p|720p|1080p|2160p|4k|uhd)\b',
        r'\b(hd|fullhd|fhd|bluray|webrip|webdl|dvdrip|brrip)\b',
        r'\b(hevc|x265|x264|h\.265|h\.264)\b',
        r'\b(dual audio|hindi dubbed|english)\b'
    ]
    
    for pattern in quality_patterns:
        normalized = re.sub(pattern, '', normalized, flags=re.IGNORECASE)
    
    # Remove special characters
    normalized = re.sub(r'[._\-]', ' ', normalized)
    
    # Remove extra spaces
    normalized = re.sub(r'\s+', ' ', normalized)
    
    return normalized.strip()

# ============================================================================
# ✅ ULTIMATE FIXED SEARCH FUNCTION - SINGLE RESULT WITH MERGED QUALITIES
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

@async_cache_with_ttl(maxsize=500, ttl=Config.SEARCH_CACHE_TTL)
async def search_movies_ultimate_fixed(query, limit=15, page=1):
    """ULTIMATE FIXED: Single result with merged qualities"""
    offset = (page - 1) * limit
    
    # Try cache first
    cache_key = f"search_ultimate:{query}:{page}:{limit}"
    if cache_manager is not None and cache_manager.redis_enabled:
        cached_data = await cache_manager.get(cache_key)
        if cached_data:
            logger.info(f"✅ Cache HIT for: {query}")
            return cached_data
    
    logger.info(f"🔍 ULTIMATE SEARCH for: {query}")
    logger.info(f"✅ MERGE_SAME_TITLES: {Config.MERGE_SAME_TITLES}")
    logger.info(f"✅ MERGE_QUALITIES: {Config.MERGE_QUALITIES}")
    
    query_lower = query.lower()
    all_results = []
    
    # ============================================================================
    # ✅ STEP 1: COLLECT ALL RAW DATA
    # ============================================================================
    
    # 1A. Collect posts from text channels
    posts_by_title = {}  # normalized_title -> list of posts
    if user_session_ready and User is not None:
        logger.info(f"📝 Searching TEXT CHANNELS...")
        
        async def search_text_channel(channel_id):
            try:
                async for msg in User.search_messages(channel_id, query=query, limit=15):
                    if msg and msg.text and len(msg.text) > 20:
                        title = extract_title_smart(msg.text)
                        if title and (query_lower in title.lower() or query_lower in msg.text.lower()):
                            norm_title = advanced_normalize_title(title)
                            
                            if norm_title not in posts_by_title:
                                posts_by_title[norm_title] = []
                            
                            posts_by_title[norm_title].append({
                                'title': title,
                                'normalized_title': norm_title,
                                'content': msg.text,
                                'channel_id': channel_id,
                                'message_id': msg.id,
                                'date': msg.date,
                                'source': 'post'
                            })
            except Exception as e:
                logger.error(f"Text search error: {e}")
        
        tasks = [search_text_channel(channel_id) for channel_id in Config.TEXT_CHANNEL_IDS]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info(f"📝 Found {sum(len(posts) for posts in posts_by_title.values())} posts for {len(posts_by_title)} titles")
    
    # 1B. Collect files from file channel
    files_by_title = {}  # normalized_title -> list of files
    if files_col is not None:
        try:
            logger.info(f"📁 Searching FILE CHANNEL database...")
            
            # Get matching files
            cursor = files_col.find(
                {
                    "$or": [
                        {"title": {"$regex": query, "$options": "i"}},
                        {"normalized_title": {"$regex": query, "$options": "i"}},
                        {"file_name": {"$regex": query, "$options": "i"}},
                        {"caption": {"$regex": query, "$options": "i"}}
                    ],
                    "status": "active"
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
                    'real_message_id': 1,
                    'date': 1,
                    'caption': 1,
                    'file_id': 1,
                    'telegram_file_id': 1,
                    'thumbnail_url': 1,
                    'year': 1,
                    '_id': 0
                }
            ).limit(200)
            
            async for doc in cursor:
                try:
                    title = doc.get('title', 'Unknown')
                    norm_title = advanced_normalize_title(title)
                    
                    if norm_title not in files_by_title:
                        files_by_title[norm_title] = []
                    
                    files_by_title[norm_title].append({
                        'title': title,
                        'normalized_title': norm_title,
                        'quality': doc.get('quality', '480p'),
                        'file_size': doc.get('file_size', 0),
                        'file_name': doc.get('file_name', ''),
                        'is_video_file': doc.get('is_video_file', False),
                        'channel_id': doc.get('channel_id'),
                        'message_id': doc.get('message_id'),
                        'real_message_id': doc.get('real_message_id') or doc.get('message_id'),
                        'caption': doc.get('caption', ''),
                        'file_id': doc.get('file_id'),
                        'telegram_file_id': doc.get('telegram_file_id'),
                        'thumbnail_url': doc.get('thumbnail_url'),
                        'year': doc.get('year', ''),
                        'date': doc.get('date'),
                        'source': 'file'
                    })
                except Exception as e:
                    logger.error(f"File processing error: {e}")
            
            logger.info(f"📁 Found {sum(len(files) for files in files_by_title.values())} files for {len(files_by_title)} titles")
            
        except Exception as e:
            logger.error(f"❌ File search error: {e}")
    
    # ============================================================================
    # ✅ STEP 2: MERGE INTO SINGLE RESULTS
    # ============================================================================
    
    logger.info(f"🔄 Merging data into single results...")
    
    # Get all unique titles
    all_titles = set(list(posts_by_title.keys()) + list(files_by_title.keys()))
    logger.info(f"📊 Unique titles found: {len(all_titles)}")
    
    for norm_title in all_titles:
        try:
            # Get posts and files for this title
            posts = posts_by_title.get(norm_title, [])
            files = files_by_title.get(norm_title, [])
            
            if not posts and not files:
                continue
            
            # Determine display title (use first available)
            display_title = ""
            original_title = ""
            
            if posts:
                display_title = posts[0]['title']
                original_title = posts[0]['title']
            elif files:
                # Try to get title from caption or filename
                for file in files:
                    if file.get('caption'):
                        display_title = extract_title_smart(file['caption'])
                        if display_title:
                            original_title = file['title']
                            break
                if not display_title and files:
                    display_title = files[0]['title']
                    original_title = files[0]['title']
            
            # Get latest date
            latest_date = None
            all_dates = []
            for post in posts:
                if post.get('date'):
                    all_dates.append(post['date'])
            for file in files:
                if file.get('date'):
                    all_dates.append(file['date'])
            if all_dates:
                latest_date = max(all_dates)
            
            # Get year
            year = ""
            year_match = re.search(r'\b(19|20)\d{2}\b', display_title)
            if year_match:
                year = year_match.group()
            
            # Clean title (remove year)
            clean_title = re.sub(r'\s+\(\d{4}\)$', '', display_title)
            clean_title = re.sub(r'\s+\d{4}$', '', clean_title)
            clean_title = clean_title.strip()
            
            # Get post content (use first post)
            post_content = ""
            if posts:
                post_content = posts[0].get('content', '')
            
            # Merge quality options from all files
            quality_options = {}
            if files:
                for file in files:
                    quality = file.get('quality', '480p')
                    if quality not in quality_options:
                        quality_options[quality] = {
                            'quality': quality,
                            'file_size': file.get('file_size', 0),
                            'message_id': file.get('real_message_id'),
                            'file_id': file.get('file_id'),
                            'telegram_file_id': file.get('telegram_file_id'),
                            'file_name': file.get('file_name', ''),
                            'is_video_file': file.get('is_video_file', False),
                            'thumbnail_url': file.get('thumbnail_url')
                        }
                    else:
                        # Keep the better quality (higher resolution)
                        existing_quality = quality_options[quality]['quality']
                        # Simple priority check
                        quality_priority = {'2160p': 4, '1080p': 3, '720p': 2, '480p': 1, '360p': 0}
                        existing_prio = quality_priority.get(existing_quality.split()[0], 0)
                        new_prio = quality_priority.get(quality.split()[0], 0)
                        if new_prio > existing_prio:
                            quality_options[quality] = {
                                'quality': quality,
                                'file_size': file.get('file_size', 0),
                                'message_id': file.get('real_message_id'),
                                'file_id': file.get('file_id'),
                                'telegram_file_id': file.get('telegram_file_id'),
                                'file_name': file.get('file_name', ''),
                                'is_video_file': file.get('is_video_file', False),
                                'thumbnail_url': file.get('thumbnail_url')
                            }
            
            # Get thumbnail (prefer from files, then posts)
            thumbnail_url = None
            for file in files:
                if file.get('thumbnail_url'):
                    thumbnail_url = file['thumbnail_url']
                    break
            
            # Determine result type
            has_post = len(posts) > 0
            has_file = len(files) > 0
            
            if has_post and has_file:
                result_type = 'post_and_file'
                search_score = 5  # Highest priority
            elif has_post:
                result_type = 'post_only'
                search_score = 3
            else:
                result_type = 'file_only'
                search_score = 2
            
            # Create merged result
            merged_result = {
                'title': clean_title,
                'original_title': original_title,
                'normalized_title': norm_title,
                'content': format_post(post_content, max_length=500),
                'post_content': post_content,
                'quality_options': quality_options,
                'date': latest_date.isoformat() if isinstance(latest_date, datetime) else str(latest_date) if latest_date else '',
                'is_new': is_new(latest_date) if latest_date else False,
                'is_video_file': any(f.get('is_video_file', False) for f in files),
                'channel_id': Config.FILE_CHANNEL_ID if files else (posts[0]['channel_id'] if posts else 0),
                'channel_name': 'File Channel' if files else 'Text Channel',
                'has_file': has_file,
                'has_post': has_post,
                'file_caption': files[0].get('caption', '') if files else '',
                'year': year,
                'quality': list(quality_options.keys())[0] if quality_options else '',
                'has_thumbnail': thumbnail_url is not None,
                'thumbnail_url': thumbnail_url,
                'real_message_id': files[0].get('real_message_id') if files else (posts[0].get('message_id') if posts else None),
                'search_score': search_score,
                'result_type': result_type,
                'quality_count': len(quality_options),
                'poster_url': None,
                'poster_source': None,
                'combined': True,  # Always true now
                'merged_qualities': len(quality_options),
                'total_files_merged': len(files),
                'total_posts_merged': len(posts)
            }
            
            all_results.append(merged_result)
            
            # Log merge details
            logger.debug(f"✅ MERGED: {clean_title[:40]}... | Type: {result_type} | Files: {len(files)} | Posts: {len(posts)} | Qualities: {len(quality_options)}")
            
        except Exception as e:
            logger.error(f"❌ Error merging title {norm_title}: {e}")
            continue
    
    # ============================================================================
    # ✅ STEP 3: SORT RESULTS
    # ============================================================================
    
    if all_results:
        # Sort by: combined first, then search score, then date
        all_results.sort(key=lambda x: (
            x.get('result_type') == 'post_and_file',  # Combined first
            x.get('search_score', 0),  # Higher score first
            x.get('is_new', False),  # New first
            x.get('date', '') if isinstance(x.get('date'), str) else ''  # Recent first
        ), reverse=True)
        
        logger.info(f"📊 After sorting: {len(all_results)} results")
    
    # ============================================================================
    # ✅ STEP 4: FETCH POSTERS
    # ============================================================================
    
    if all_results and poster_fetcher is not None:
        logger.info(f"🎬 Fetching posters for {len(all_results)} results...")
        
        # Prepare movies for poster fetching
        movies_for_posters = []
        for result in all_results:
            movies_for_posters.append({
                'title': result.get('title', ''),
                'year': result.get('year', ''),
                'original_title': result.get('original_title', ''),
                'result_type': result.get('result_type', 'unknown')
            })
        
        # Fetch posters in batch
        if movies_for_posters:
            try:
                movies_with_posters = await get_posters_for_movies_batch(movies_for_posters)
                
                # Update results with poster data
                poster_map = {movie['title']: movie for movie in movies_with_posters}
                for result in all_results:
                    title = result.get('title', '')
                    if title in poster_map:
                        poster_data = poster_map[title]
                        result.update({
                            'poster_url': poster_data.get('poster_url', Config.FALLBACK_POSTER),
                            'poster_source': poster_data.get('poster_source', 'fallback'),
                            'poster_rating': poster_data.get('poster_rating', '0.0'),
                            'has_poster': True,
                            'thumbnail': result.get('thumbnail_url') or poster_data.get('poster_url'),
                            'thumbnail_source': result.get('thumbnail_source') or poster_data.get('poster_source', 'fallback'),
                            'has_thumbnail': True
                        })
                    else:
                        result.update({
                            'poster_url': Config.FALLBACK_POSTER,
                            'poster_source': 'fallback',
                            'poster_rating': '0.0',
                            'has_poster': True,
                            'thumbnail': result.get('thumbnail_url') or Config.FALLBACK_POSTER,
                            'thumbnail_source': 'fallback',
                            'has_thumbnail': True
                        })
                
                logger.info(f"✅ Posters fetched successfully")
                
            except Exception as e:
                logger.error(f"❌ Poster fetching error: {e}")
                # Add fallback posters
                for result in all_results:
                    result.update({
                        'poster_url': Config.FALLBACK_POSTER,
                        'poster_source': 'fallback',
                        'poster_rating': '0.0',
                        'has_poster': True,
                        'thumbnail': result.get('thumbnail_url') or Config.FALLBACK_POSTER,
                        'thumbnail_source': 'fallback',
                        'has_thumbnail': True
                    })
    
    # ============================================================================
    # ✅ STEP 5: PAGINATION
    # ============================================================================
    
    total = len(all_results)
    start_idx = offset
    end_idx = offset + limit
    paginated = all_results[start_idx:end_idx]
    
    # Statistics
    post_count = sum(1 for r in all_results if r.get('result_type') == 'post_only')
    file_count = sum(1 for r in all_results if r.get('result_type') == 'file_only')
    combined_count = sum(1 for r in all_results if r.get('result_type') == 'post_and_file')
    
    stats = {
        'total': total,
        'post_only': post_count,
        'file_only': file_count,
        'post_and_file': combined_count,
        'titles_merged': len(all_titles),
        'qualities_merged': sum(r.get('merged_qualities', 0) for r in all_results)
    }
    
    # Log detailed results
    logger.info(f"📊 FINAL RESULTS:")
    logger.info(f"   • Total unique results: {total}")
    logger.info(f"   • Post-only: {post_count}")
    logger.info(f"   • File-only: {file_count}")
    logger.info(f"   • Post+File combined: {combined_count}")
    logger.info(f"   • Qualities merged: {stats['qualities_merged']}")
    
    # Show sample of results
    for i, result in enumerate(paginated[:5]):
        result_type = result.get('result_type', 'unknown')
        title = result.get('title', '')[:40]
        has_file = result.get('has_file', False)
        has_post = result.get('has_post', False)
        quality_count = len(result.get('quality_options', {}))
        
        logger.info(f"   📋 {i+1}. {result_type}: {title}... | File: {has_file} | Post: {has_post} | Qualities: {quality_count}")
    
    # ============================================================================
    # ✅ STEP 6: FINAL DATA STRUCTURE
    # ============================================================================
    
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
            'post_file_merged': True,
            'file_only_with_poster': True,
            'poster_fetcher': poster_fetcher is not None,
            'thumbnails_enabled': True,
            'real_message_ids': True,
            'search_logic': 'ultimate_fixed_merging',
            'merge_same_titles': Config.MERGE_SAME_TITLES,
            'merge_qualities': Config.MERGE_QUALITIES
        },
        'bot_username': Config.BOT_USERNAME
    }
    
    # Cache results
    if cache_manager is not None:
        await cache_manager.set(cache_key, result_data, expire_seconds=Config.SEARCH_CACHE_TTL)
    
    logger.info(f"✅ Ultimate search complete: {len(paginated)} results (page {page})")
    
    return result_data

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
            'source': 'custom',
            'rating': '0.0',
            'year': year,
            'title': title,
            'quality': quality or 'unknown'
        }
    
    try:
        # Fetch poster with timeout
        poster_task = asyncio.create_task(poster_fetcher.fetch_poster(title))
        
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
                'source': 'custom',
                'rating': '0.0',
                'year': year,
                'title': title,
                'quality': quality or 'unknown'
            }
            
    except Exception as e:
        logger.error(f"❌ Unexpected error in get_poster_for_movie: {e}")
        return {
            'poster_url': Config.FALLBACK_POSTER,
            'source': 'custom',
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
                'thumbnail': poster_data['poster_url'],
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
                'thumbnail': Config.FALLBACK_POSTER,
                'thumbnail_source': 'fallback',
                'has_poster': True,
                'has_thumbnail': True
            })
            
            results.append(movie_with_fallback)
    
    return results

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
# ✅ TELEGRAM INITIALIZATION
# ============================================================================

async def init_telegram_sessions():
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
    
    return user_session_ready or bot_session_ready

# ============================================================================
# ✅ MAIN INITIALIZATION
# ============================================================================

async def init_system():
    start_time = time.time()
    
    try:
        logger.info("=" * 60)
        logger.info("🚀 SK4FiLM v9.0 - ULTIMATE FIXED VERSION")
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
        
        # Initialize Cache Manager
        global cache_manager, verification_system, premium_system, poster_fetcher
        cache_manager = CacheManager(Config)
        redis_ok = await cache_manager.init_redis()
        if redis_ok:
            logger.info("✅ Cache Manager initialized")
        
        # Initialize Verification System
        if VerificationSystem is not None:
            verification_system = VerificationSystem(Config, mongo_client)
            logger.info("✅ Verification System initialized")
        
        # Initialize Premium System
        if PremiumSystem is not None:
            premium_system = PremiumSystem(Config, mongo_client)
            logger.info("✅ Premium System initialized")
        
        # Initialize Poster Fetcher
        if PosterFetcher is not None:
            poster_fetcher = PosterFetcher(Config, cache_manager)
            logger.info("✅ Poster Fetcher initialized")
        
        # Initialize Telegram Sessions
        if PYROGRAM_AVAILABLE:
            telegram_ok = await init_telegram_sessions()
            if not telegram_ok:
                logger.warning("⚠️ Telegram sessions failed")
        
        init_time = time.time() - start_time
        logger.info(f"⚡ SK4FiLM Started in {init_time:.2f}s")
        logger.info("=" * 60)
        
        logger.info("🔧 ULTIMATE FEATURES:")
        logger.info(f"   • Single Result Merging: ✅ ENABLED")
        logger.info(f"   • Quality Merging: ✅ ENABLED")
        logger.info(f"   • Advanced Title Matching: ✅ ENABLED")
        logger.info(f"   • Post+File Detection: ✅ ENABLED")
        logger.info(f"   • Real Message IDs: ✅ ENABLED")
        logger.info(f"   • File Channel ID: {Config.FILE_CHANNEL_ID}")
        
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
        video_files = await files_col.count_documents({'is_video_file': True})
    else:
        tf = 0
        video_files = 0
    
    return jsonify({
        'status': 'healthy',
        'service': 'SK4FiLM v9.0 - ULTIMATE FIXED',
        'features': {
            'single_result_merging': True,
            'quality_merging': True,
            'advanced_title_matching': True,
            'post_file_detection': True,
            'real_message_ids': True
        },
        'sessions': {
            'user_session': user_session_ready,
            'bot_session': bot_session_ready
        },
        'stats': {
            'total_files': tf,
            'video_files': video_files
        },
        'response_time': f"{time.perf_counter():.3f}s"
    })

@app.route('/health')
async def health():
    return jsonify({
        'status': 'ok',
        'optimized': True,
        'sessions': {
            'user': user_session_ready,
            'bot': bot_session_ready
        },
        'features': {
            'single_result': True,
            'merged_qualities': True
        },
        'timestamp': datetime.now().isoformat()
    })

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
        
        result_data = await search_movies_ultimate_fixed(query, limit, page)
        
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

@app.route('/api/movies', methods=['GET'])
async def api_movies():
    try:
        # Simple home movies endpoint
        if files_col is not None:
            # Get recent files
            cursor = files_col.find(
                {},
                {
                    'title': 1,
                    'normalized_title': 1,
                    'quality': 1,
                    'date': 1,
                    'thumbnail_url': 1,
                    'year': 1,
                    '_id': 0
                }
            ).sort('date', -1).limit(20)
            
            movies = []
            async for doc in cursor:
                movies.append({
                    'title': doc.get('title', 'Unknown'),
                    'quality': doc.get('quality', '480p'),
                    'date': doc['date'].isoformat() if isinstance(doc['date'], datetime) else str(doc['date']),
                    'thumbnail_url': doc.get('thumbnail_url', Config.FALLBACK_POSTER),
                    'year': doc.get('year', ''),
                    'has_file': True,
                    'has_post': False
                })
            
            return jsonify({
                'status': 'success',
                'movies': movies,
                'total': len(movies),
                'source': 'database'
            })
        else:
            return jsonify({
                'status': 'success',
                'movies': [],
                'total': 0
            })
            
    except Exception as e:
        logger.error(f"Movies API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'movies': [],
            'total': 0
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
    logger.info("🛑 Shutting down SK4FiLM v9.0...")
    
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
    
    logger.info(f"🌐 Starting SK4FiLM ULTIMATE FIXED on port {Config.WEB_SERVER_PORT}...")
    logger.info("🎯 FEATURES:")
    logger.info(f"   • Single Result Merging: ✅ ENABLED")
    logger.info(f"   • Quality Merging: ✅ ENABLED")  
    logger.info(f"   • Advanced Title Matching: ✅ ENABLED")
    logger.info(f"   • Post+File Detection: ✅ ENABLED")
    logger.info(f"   • File Channel ID: {Config.FILE_CHANNEL_ID}")
    
    try:
        asyncio.run(serve(app, config))
    except KeyboardInterrupt:
        logger.info("👋 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
