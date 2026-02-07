# ============================================================================
# 🚀 SK4FiLM v9.0 - FIXED THUMBNAIL EXTRACTION
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
from enum import Enum

# ✅ SETUP LOGGING FIRST
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
# ✅ CONFIGURATION
# ============================================================================

class Config:
    # API Configuration
    API_ID = int(os.environ.get("API_ID", "0"))
    API_HASH = os.environ.get("API_HASH", "")
    USER_SESSION_STRING = os.environ.get("USER_SESSION_STRING", "")
    BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
    
    # Database Configuration
    MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017/sk4film")
    REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
    
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
    
    # API Keys
    TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "e547e17d4e91f3e62a571655cd1ccaff")
    OMDB_API_KEY = os.environ.get("OMDB_API_KEY", "8265bd1c")
    
    # Performance Settings
    MAX_CONCURRENT_REQUESTS = 50
    CACHE_TTL = 300
    REQUEST_TIMEOUT = 10
    
    # Sync Settings
    MONITOR_INTERVAL = 300
    
    # Quality Settings
    QUALITY_PRIORITY = ['2160p', '1080p', '720p', '480p', '360p']
    
    # Fallback Poster
    FALLBACK_POSTER = "https://iili.io/fAeIwv9.th.png"
    
    # Thumbnail Settings
    THUMBNAIL_EXTRACT_TIMEOUT = 10
    THUMBNAIL_CACHE_DURATION = 24 * 60 * 60
    THUMBNAIL_EXTRACTION_ENABLED = True
    
    # Indexing Settings
    AUTO_INDEX_INTERVAL = 120
    BATCH_INDEX_SIZE = 500
    MAX_INDEX_LIMIT = 0
    INDEX_ALL_HISTORY = True
    INSTANT_AUTO_INDEX = True
    
    # Search Settings
    SEARCH_MIN_QUERY_LENGTH = 2
    SEARCH_RESULTS_PER_PAGE = 12
    SEARCH_CACHE_TTL = 600

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
            return async_wrapper
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

performance_monitor = PerformanceMonitor()

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
    response.headers['X-SK4FiLM-Version'] = '9.0-FIXED-THUMBNAILS'
    return response

# ============================================================================
# ✅ GLOBAL COMPONENTS
# ============================================================================

# Database
mongo_client = None
db = None
files_col = None
thumbnails_col = None

# Telegram
try:
    from pyrogram import Client
    from pyrogram.types import Message
    PYROGRAM_AVAILABLE = True
except ImportError:
    PYROGRAM_AVAILABLE = False

User = None
Bot = None
user_session_ready = False
bot_session_ready = False

# Systems
cache_manager = None
poster_fetcher = None
thumbnail_extractor = None
telegram_bot = None

# ============================================================================
# ✅ UTILITY FUNCTIONS
# ============================================================================

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

def format_post(text, max_length=None):
    if not text:
        return ""
    text = re.sub(r'\n\s*\n', '\n\n', text)
    if max_length and len(text) > max_length:
        text = text[:max_length] + "..."
    return text.strip()

def is_video_file(filename):
    if not filename:
        return False
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v']
    return any(filename.lower().endswith(ext) for ext in video_extensions)

def detect_quality_enhanced(filename):
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
    
    is_hevc = any(re.search(pattern, filename_lower) 
                  for pattern in [r'\bhevc\b', r'\bx265\b', r'\bh\.?265\b'])
    
    for pattern, quality in patterns:
        if re.search(pattern, filename_lower):
            if is_hevc and quality in ['720p', '1080p', '2160p']:
                return f"{quality} HEVC"
            return quality
    
    return "480p"

def is_new(date):
    if not date:
        return False
    if isinstance(date, str):
        try:
            date = datetime.fromisoformat(date.replace('Z', '+00:00'))
        except:
            return False
    return (datetime.now() - date).days < 7

def extract_year_from_title(title):
    """Extract year from title"""
    if not title:
        return ""
    
    # Look for year patterns like (2024) or 2024
    year_match = re.search(r'\((\d{4})\)', title)
    if year_match:
        return year_match.group(1)
    
    year_match = re.search(r'\b(19|20)\d{2}\b', title)
    if year_match:
        return year_match.group()
    
    return ""

def clean_title(title):
    """Clean title by removing year and other artifacts"""
    if not title:
        return ""
    
    # Remove year in parentheses
    title = re.sub(r'\s*\(\d{4}\)', '', title)
    
    # Remove year at end
    title = re.sub(r'\s+\d{4}$', '', title)
    
    # Remove quality indicators
    title = re.sub(r'\b(480p|720p|1080p|2160p|4k|hd|hevc|x265|x264|web-dl|webrip|hdrip|bluray|dvdrip|tc|ts|cam)\b', '', title, flags=re.IGNORECASE)
    
    # Clean extra spaces
    title = re.sub(r'\s+', ' ', title)
    
    return title.strip()

# ============================================================================
# ✅ POSTER FETCHER - SIMPLIFIED
# ============================================================================

class PosterFetcher:
    def __init__(self, config):
        self.config = config
        self.session = None
        
    async def fetch_poster(self, title, year=""):
        """Fetch movie poster"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Clean title
            clean_title = self._clean_title(title)
            
            # Try TMDB first
            if self.config.TMDB_API_KEY:
                tmdb_poster = await self._fetch_tmdb_poster(clean_title, year)
                if tmdb_poster:
                    return {
                        'poster_url': tmdb_poster,
                        'source': 'tmdb',
                        'rating': '0.0'
                    }
            
            # Try OMDB
            if self.config.OMDB_API_KEY:
                omdb_poster = await self._fetch_omdb_poster(clean_title)
                if omdb_poster:
                    return {
                        'poster_url': omdb_poster,
                        'source': 'omdb',
                        'rating': '0.0'
                    }
            
            # Fallback
            return {
                'poster_url': self.config.FALLBACK_POSTER,
                'source': 'fallback',
                'rating': '0.0'
            }
            
        except Exception as e:
            logger.error(f"Poster fetch error: {e}")
            return {
                'poster_url': self.config.FALLBACK_POSTER,
                'source': 'error',
                'rating': '0.0'
            }
    
    def _clean_title(self, title):
        """Clean title for search"""
        if not title:
            return ""
        
        patterns_to_remove = [
            r'\b\d{3,4}p\b',
            r'\bHD\b',
            r'\bHEVC\b',
            r'\bx265\b',
            r'\bx264\b',
            r'\bWEB-DL\b',
            r'\bWEBRip\b',
            r'\bHDRip\b',
            r'\bBluRay\b',
            r'\bDVDRip\b',
            r'\bTC\b',
            r'\bTS\b',
            r'\bCAM\b',
            r'\[.*?\]',
            r'\(.*?\)',
        ]
        
        for pattern in patterns_to_remove:
            title = re.sub(pattern, '', title, flags=re.IGNORECASE)
        
        title = re.sub(r'\s+', ' ', title)
        title = title.strip()
        title = re.sub(r'\s*\d{4}$', '', title)
        
        return title
    
    async def _fetch_tmdb_poster(self, title, year=""):
        """Fetch from TMDB"""
        try:
            search_url = "https://api.themoviedb.org/3/search/movie"
            params = {
                'api_key': self.config.TMDB_API_KEY,
                'query': title,
                'language': 'en-US',
                'page': 1
            }
            
            if year:
                params['year'] = year
            
            async with self.session.get(search_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('results') and len(data['results']) > 0:
                        poster_path = data['results'][0].get('poster_path')
                        if poster_path:
                            return f"https://image.tmdb.org/t/p/w500{poster_path}"
        except:
            pass
        return None
    
    async def _fetch_omdb_poster(self, title):
        """Fetch from OMDB"""
        try:
            url = f"https://www.omdbapi.com/?t={urllib.parse.quote(title)}&apikey={self.config.OMDB_API_KEY}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    poster = data.get('Poster')
                    if poster and poster != 'N/A':
                        return poster
        except:
            pass
        return None
    
    async def close(self):
        if self.session:
            await self.session.close()

# ============================================================================
# ✅ THUMBNAIL EXTRACTOR - FIXED VERSION
# ============================================================================

class ThumbnailExtractor:
    """Thumbnail extractor that actually works"""
    
    def __init__(self, mongo_client, config):
        self.mongo_client = mongo_client
        self.db = mongo_client.get_database()
        self.thumbnails_col = self.db.thumbnails
        self.config = config
        
        # In-memory cache
        self.cache = {}
        
        logger.info("🖼️ Thumbnail Extractor initialized")
    
    async def initialize(self):
        """Initialize thumbnail extractor"""
        try:
            # Create indexes
            await self.thumbnails_col.create_index(
                [("normalized_title", 1)],
                unique=True
            )
            
            # Index for channel + message
            await self.thumbnails_col.create_index(
                [("channel_id", 1), ("message_id", 1)]
            )
            
            logger.info("✅ Thumbnail indexes created")
            return True
        except Exception as e:
            logger.error(f"❌ Thumbnail init error: {e}")
            return False
    
    async def extract_from_telegram(self, channel_id: int, message_id: int, normalized_title: str) -> Optional[str]:
        """
        Extract thumbnail from Telegram message
        """
        try:
            # Check cache first
            cache_key = f"{channel_id}_{message_id}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Check database
            existing = await self.thumbnails_col.find_one({
                "channel_id": channel_id,
                "message_id": message_id
            })
            
            if existing and existing.get("thumbnail_url"):
                self.cache[cache_key] = existing["thumbnail_url"]
                return existing["thumbnail_url"]
            
            # Try to extract using Bot
            if Bot is not None:
                try:
                    message = await Bot.get_messages(channel_id, message_id)
                    if message:
                        # Check if it's a video
                        if message.video:
                            # Video messages have thumbnails
                            if hasattr(message.video, 'thumbnail'):
                                # Download thumbnail
                                download_path = await Bot.download_media(
                                    message.video.thumbnail.file_id,
                                    in_memory=True
                                )
                                
                                if download_path:
                                    if isinstance(download_path, bytes):
                                        thumbnail_data = download_path
                                    else:
                                        with open(download_path, 'rb') as f:
                                            thumbnail_data = f.read()
                                    
                                    # Convert to base64 data URL
                                    base64_data = base64.b64encode(thumbnail_data).decode('utf-8')
                                    thumbnail_url = f"data:image/jpeg;base64,{base64_data}"
                                    
                                    # Save to database
                                    await self.save_thumbnail(
                                        normalized_title=normalized_title,
                                        thumbnail_url=thumbnail_url,
                                        channel_id=channel_id,
                                        message_id=message_id,
                                        source="telegram"
                                    )
                                    
                                    self.cache[cache_key] = thumbnail_url
                                    logger.info(f"✅ Thumbnail extracted: {normalized_title}")
                                    return thumbnail_url
                        
                        # Check if it's a video document
                        elif message.document and is_video_file(message.document.file_name or ''):
                            if hasattr(message.document, 'thumbnail') and message.document.thumbnail:
                                # Download thumbnail
                                download_path = await Bot.download_media(
                                    message.document.thumbnail.file_id,
                                    in_memory=True
                                )
                                
                                if download_path:
                                    if isinstance(download_path, bytes):
                                        thumbnail_data = download_path
                                    else:
                                        with open(download_path, 'rb') as f:
                                            thumbnail_data = f.read()
                                    
                                    base64_data = base64.b64encode(thumbnail_data).decode('utf-8')
                                    thumbnail_url = f"data:image/jpeg;base64,{base64_data}"
                                    
                                    # Save to database
                                    await self.save_thumbnail(
                                        normalized_title=normalized_title,
                                        thumbnail_url=thumbnail_url,
                                        channel_id=channel_id,
                                        message_id=message_id,
                                        source="telegram"
                                    )
                                    
                                    self.cache[cache_key] = thumbnail_url
                                    logger.info(f"✅ Thumbnail extracted from document: {normalized_title}")
                                    return thumbnail_url
                
                except Exception as e:
                    logger.error(f"❌ Telegram thumbnail extraction error: {e}")
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Thumbnail extraction error: {e}")
            return None
    
    async def save_thumbnail(self, normalized_title: str, thumbnail_url: str, 
                           channel_id: int = None, message_id: int = None, 
                           source: str = "api"):
        """Save thumbnail to database"""
        try:
            thumbnail_doc = {
                "normalized_title": normalized_title,
                "thumbnail_url": thumbnail_url,
                "source": source,
                "updated_at": datetime.now()
            }
            
            if channel_id:
                thumbnail_doc["channel_id"] = channel_id
            if message_id:
                thumbnail_doc["message_id"] = message_id
            
            await self.thumbnails_col.update_one(
                {"normalized_title": normalized_title},
                {"$set": thumbnail_doc, "$setOnInsert": {"created_at": datetime.now()}},
                upsert=True
            )
            
            logger.debug(f"✅ Thumbnail saved: {normalized_title}")
            
        except Exception as e:
            logger.error(f"❌ Save thumbnail error: {e}")
    
    async def get_thumbnail(self, normalized_title: str) -> Optional[str]:
        """Get thumbnail URL for movie"""
        if not normalized_title:
            return None
        
        # Check cache
        if normalized_title in self.cache:
            return self.cache[normalized_title]
        
        # Check database
        try:
            thumb = await self.thumbnails_col.find_one(
                {"normalized_title": normalized_title},
                {"thumbnail_url": 1}
            )
            if thumb and thumb.get("thumbnail_url"):
                self.cache[normalized_title] = thumb["thumbnail_url"]
                return thumb["thumbnail_url"]
        except Exception as e:
            logger.error(f"❌ Thumbnail fetch error: {e}")
        
        return None
    
    async def get_stats(self):
        """Get thumbnail stats"""
        try:
            total = await self.thumbnails_col.count_documents({})
            telegram_count = await self.thumbnails_col.count_documents({"source": "telegram"})
            api_count = await self.thumbnails_col.count_documents({"source": {"$ne": "telegram"}})
            
            return {
                "total_thumbnails": total,
                "telegram_extracted": telegram_count,
                "api_fetched": api_count,
                "cache_size": len(self.cache)
            }
        except Exception as e:
            logger.error(f"❌ Thumbnail stats error: {e}")
            return {"total_thumbnails": 0}

# ============================================================================
# ✅ TELEGRAM SESSIONS
# ============================================================================

async def init_telegram_sessions():
    global User, Bot, user_session_ready, bot_session_ready
    
    logger.info("🚀 Initializing Telegram sessions...")
    
    if not PYROGRAM_AVAILABLE:
        logger.error("❌ Pyrogram not installed!")
        return False
    
    # User Session
    if Config.API_ID > 0 and Config.API_HASH and Config.USER_SESSION_STRING:
        try:
            User = Client(
                "sk4film_user",
                api_id=Config.API_ID,
                api_hash=Config.API_HASH,
                session_string=Config.USER_SESSION_STRING,
                in_memory=True,
                no_updates=True
            )
            await User.start()
            me = await User.get_me()
            logger.info(f"✅ User Session: {me.first_name}")
            user_session_ready = True
        except Exception as e:
            logger.error(f"❌ User session failed: {e}")
            user_session_ready = False
    
    # Bot Session
    if Config.BOT_TOKEN:
        try:
            Bot = Client(
                "sk4film_bot",
                api_id=Config.API_ID,
                api_hash=Config.API_HASH,
                bot_token=Config.BOT_TOKEN,
                in_memory=True,
                no_updates=True
            )
            await Bot.start()
            bot_info = await Bot.get_me()
            logger.info(f"✅ Bot Session: @{bot_info.username}")
            bot_session_ready = True
        except Exception as e:
            logger.error(f"❌ Bot session failed: {e}")
            bot_session_ready = False
    
    return user_session_ready or bot_session_ready

# ============================================================================
# ✅ MONGODB INITIALIZATION
# ============================================================================

async def init_mongodb():
    global mongo_client, db, files_col, thumbnails_col
    
    try:
        logger.info("🔌 Initializing MongoDB...")
        
        mongodb_uri = Config.MONGODB_URI
        
        mongo_client = AsyncIOMotorClient(
            mongodb_uri,
            serverSelectionTimeoutMS=10000,
            connectTimeoutMS=10000,
            socketTimeoutMS=15000
        )
        
        # Test connection
        await mongo_client.admin.command('ping')
        logger.info("✅ MongoDB connection test successful")
        
        # Get database name
        if "mongodb.net/" in mongodb_uri:
            parts = mongodb_uri.split("mongodb.net/")
            if len(parts) > 1:
                db_name_part = parts[1].split("?")[0]
                if db_name_part:
                    db_name = db_name_part
                else:
                    db_name = "sk4film"
            else:
                db_name = "sk4film"
        else:
            db_name = "sk4film"
        
        # Get database
        db = mongo_client[db_name]
        logger.info(f"✅ Using database: {db_name}")
        
        # Initialize collections
        files_col = db.files
        thumbnails_col = db.thumbnails
        
        # Create indexes
        try:
            # Files collection indexes
            await files_col.create_index(
                [("channel_id", 1), ("message_id", 1)],
                unique=True
            )
            await files_col.create_index([("normalized_title", "text")])
            
            logger.info("✅ Files collection indexes created")
            
        except Exception as e:
            logger.warning(f"⚠️ Index creation error: {e}")
        
        logger.info("✅ MongoDB initialization complete")
        return True
        
    except Exception as e:
        logger.error(f"❌ MongoDB error: {e}")
        return False

# ============================================================================
# ✅ ENHANCED SEARCH WITH THUMBNAIL EXTRACTION
# ============================================================================

async def search_movies_enhanced_fixed(query, limit=15, page=1):
    """Search with proper thumbnail extraction"""
    offset = (page - 1) * limit
    
    logger.info(f"🔍 SEARCH WITH THUMBNAILS for: {query}")
    
    query_lower = query.lower()
    all_results = []
    
    # ============================================================================
    # ✅ STEP 1: SEARCH TEXT CHANNELS FOR POSTS
    # ============================================================================
    
    post_results = []
    if user_session_ready and User is not None:
        logger.info(f"📝 Searching TEXT CHANNELS for posts...")
        
        for channel_id in Config.TEXT_CHANNEL_IDS:
            try:
                async for msg in User.search_messages(channel_id, query=query, limit=10):
                    if msg and msg.text and len(msg.text) > 15:
                        title = extract_title_smart(msg.text)
                        if title and (query_lower in title.lower() or query_lower in msg.text.lower()):
                            year = extract_year_from_title(title)
                            clean_title_val = clean_title(title)
                            norm_title = normalize_title(clean_title_val)
                            
                            post_data = {
                                'title': clean_title_val,
                                'original_title': title,
                                'normalized_title': norm_title,
                                'content': format_post(msg.text, max_length=500),
                                'post_content': msg.text,
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
                                'result_type': 'post_only'
                            }
                            
                            post_results.append(post_data)
                            
            except Exception as e:
                logger.error(f"Text search error in {channel_id}: {e}")
                continue
        
        logger.info(f"📝 Found {len(post_results)} POST results")
    
    # ============================================================================
    # ✅ STEP 2: SEARCH FILE CHANNEL DATABASE
    # ============================================================================
    
    file_results = []
    if files_col is not None:
        try:
            logger.info(f"📁 Searching FILE CHANNEL database...")
            
            search_query = {
                "$or": [
                    {"title": {"$regex": query, "$options": "i"}},
                    {"normalized_title": {"$regex": query, "$options": "i"}},
                    {"file_name": {"$regex": query, "$options": "i"}},
                    {"caption": {"$regex": query, "$options": "i"}}
                ]
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
                    'year': 1,
                    '_id': 0
                }
            ).limit(100)
            
            async for doc in cursor:
                title = doc.get('title', 'Unknown')
                norm_title = doc.get('normalized_title', '')
                quality = doc.get('quality', '480p')
                year = doc.get('year', '')
                channel_id = doc.get('channel_id')
                message_id = doc.get('real_message_id') or doc.get('message_id')
                is_video = doc.get('is_video_file', False)
                
                file_data = {
                    'title': title,
                    'normalized_title': norm_title,
                    'content': format_post(doc.get('caption', ''), max_length=300),
                    'post_content': doc.get('caption', ''),
                    'quality_options': {quality: {
                        'quality': quality,
                        'file_size': doc.get('file_size', 0),
                        'message_id': message_id,
                        'file_name': doc.get('file_name', ''),
                        'channel_id': channel_id
                    }},
                    'date': doc['date'].isoformat() if isinstance(doc['date'], datetime) else doc['date'],
                    'is_new': is_new(doc['date']) if doc.get('date') else False,
                    'is_video_file': is_video,
                    'channel_id': channel_id,
                    'has_file': True,
                    'has_post': bool(doc.get('caption')),
                    'quality': quality,
                    'real_message_id': message_id,
                    'result_type': 'file_only',
                    'year': year,
                    'search_score': 2
                }
                
                file_results.append(file_data)
            
            logger.info(f"📁 Found {len(file_results)} FILE results")
            
        except Exception as e:
            logger.error(f"❌ File search error: {e}")
    
    # ============================================================================
    # ✅ STEP 3: MERGE POST AND FILE RESULTS
    # ============================================================================
    
    logger.info("🔄 Merging post and file results...")
    
    merged_dict = {}
    
    # First add all post results
    for post in post_results:
        norm_title = post['normalized_title']
        if norm_title:
            merged_dict[norm_title] = post.copy()
    
    # Now merge file results
    for file_item in file_results:
        norm_title = file_item['normalized_title']
        
        if norm_title in merged_dict:
            # Merge file with existing post
            existing_item = merged_dict[norm_title]
            
            existing_item.update({
                'has_file': True,
                'is_video_file': file_item['is_video_file'],
                'real_message_id': file_item['real_message_id'],
                'channel_id': file_item['channel_id'],
                'result_type': 'post_and_file',
                'search_score': 5
            })
            
            if 'quality_options' not in existing_item:
                existing_item['quality_options'] = {}
            
            for quality, q_data in file_item['quality_options'].items():
                existing_item['quality_options'][quality] = q_data
            
            existing_item['quality_count'] = len(existing_item['quality_options'])
            
            logger.debug(f"✅ Merged file with post: {norm_title}")
            
        else:
            file_item['result_type'] = 'file_only'
            merged_dict[norm_title] = file_item
    
    # Convert to list
    all_results = list(merged_dict.values())
    
    # ============================================================================
    # ✅ STEP 4: ADD THUMBNAILS (FIXED)
    # ============================================================================
    
    logger.info(f"🖼️ Extracting thumbnails for {len(all_results)} results...")
    
    thumbnail_tasks = []
    
    for result in all_results:
        norm_title = result.get('normalized_title')
        channel_id = result.get('channel_id')
        message_id = result.get('real_message_id')
        title = result.get('title', '')
        year = result.get('year', '')
        
        # Task 1: Try to get existing thumbnail
        thumbnail_task = asyncio.create_task(
            thumbnail_extractor.get_thumbnail(norm_title) if thumbnail_extractor else None
        )
        
        # Task 2: Extract from Telegram if it's a video file
        extract_task = None
        if result.get('is_video_file') and channel_id and message_id:
            extract_task = asyncio.create_task(
                thumbnail_extractor.extract_from_telegram(channel_id, message_id, norm_title)
                if thumbnail_extractor else None
            )
        
        # Task 3: Get poster
        poster_task = asyncio.create_task(
            poster_fetcher.fetch_poster(title, year) if poster_fetcher else None
        )
        
        thumbnail_tasks.append((result, thumbnail_task, extract_task, poster_task))
    
    # Process all tasks
    for result, thumbnail_task, extract_task, poster_task in thumbnail_tasks:
        try:
            thumbnail_url = None
            
            # Try existing thumbnail first
            if thumbnail_task:
                try:
                    thumbnail_url = await asyncio.wait_for(thumbnail_task, timeout=1.0)
                except (asyncio.TimeoutError, Exception):
                    pass
            
            # If no thumbnail, try to extract from Telegram
            if not thumbnail_url and extract_task:
                try:
                    thumbnail_url = await asyncio.wait_for(extract_task, timeout=3.0)
                except (asyncio.TimeoutError, Exception):
                    pass
            
            # Get poster
            poster_data = None
            if poster_task:
                try:
                    poster_data = await asyncio.wait_for(poster_task, timeout=2.0)
                except (asyncio.TimeoutError, Exception):
                    poster_data = {
                        'poster_url': Config.FALLBACK_POSTER,
                        'source': 'fallback',
                        'rating': '0.0'
                    }
            else:
                poster_data = {
                    'poster_url': Config.FALLBACK_POSTER,
                    'source': 'fallback',
                    'rating': '0.0'
                }
            
            # Update result with poster data
            result.update({
                'poster_url': poster_data['poster_url'],
                'poster_source': poster_data['source'],
                'poster_rating': poster_data['rating'],
                'has_poster': True
            })
            
            # Set thumbnail - use extracted if available, otherwise use poster
            if thumbnail_url:
                result.update({
                    'thumbnail_url': thumbnail_url,
                    'has_thumbnail': True,
                    'thumbnail_source': 'extracted'
                })
            else:
                result.update({
                    'thumbnail_url': poster_data['poster_url'],
                    'has_thumbnail': True,
                    'thumbnail_source': 'poster'
                })
            
            # Add quality summary
            if 'quality_options' in result and result['quality_options']:
                qualities = list(result['quality_options'].keys())
                result['all_qualities'] = qualities
                result['quality_count'] = len(qualities)
                
                if len(qualities) <= 3:
                    result['quality_summary'] = " • ".join(qualities)
                else:
                    result['quality_summary'] = f"{qualities[0]} • {qualities[1]} • +{len(qualities)-2} more"
                    
        except Exception as e:
            logger.error(f"❌ Thumbnail processing error: {e}")
            # Fallback
            result.update({
                'poster_url': Config.FALLBACK_POSTER,
                'poster_source': 'fallback',
                'poster_rating': '0.0',
                'has_poster': True,
                'thumbnail_url': Config.FALLBACK_POSTER,
                'has_thumbnail': True,
                'thumbnail_source': 'fallback'
            })
    
    # ============================================================================
    # ✅ STEP 5: SORT AND PAGINATE
    # ============================================================================
    
    all_results.sort(key=lambda x: (
        x.get('result_type') == 'post_and_file',
        x.get('result_type') == 'post_only',
        x.get('search_score', 0),
        x.get('is_new', False),
    ), reverse=True)
    
    total = len(all_results)
    start_idx = offset
    end_idx = offset + limit
    paginated = all_results[start_idx:end_idx]
    
    # Statistics
    post_count = sum(1 for r in all_results if r.get('result_type') == 'post_only')
    file_count = sum(1 for r in all_results if r.get('result_type') == 'file_only')
    merged_count = sum(1 for r in all_results if r.get('result_type') == 'post_and_file')
    
    # Thumbnail stats
    thumbnail_stats = await thumbnail_extractor.get_stats() if thumbnail_extractor else {}
    
    # Log results
    logger.info("📊 FINAL RESULTS WITH THUMBNAILS:")
    logger.info(f"   • Total results: {total}")
    logger.info(f"   • Post-only: {post_count}")
    logger.info(f"   • File-only: {file_count}")
    logger.info(f"   • Post+File merged: {merged_count}")
    logger.info(f"   • Thumbnails in DB: {thumbnail_stats.get('total_thumbnails', 0)}")
    
    # Check thumbnail extraction
    extracted_count = 0
    for result in paginated[:3]:
        title = result.get('title', '')[:30]
        thumb_source = result.get('thumbnail_source', 'none')
        has_thumb = result.get('has_thumbnail', False)
        thumb_url = result.get('thumbnail_url', '')[:50] if result.get('thumbnail_url') else 'none'
        
        logger.info(f"   📋 {title}... | Thumb: {has_thumb} | Source: {thumb_source}")
        logger.info(f"      URL: {thumb_url}...")
        
        if thumb_source == 'extracted':
            extracted_count += 1
    
    if extracted_count > 0:
        logger.info(f"   ✅ {extracted_count} thumbnails extracted from Telegram")
    
    return {
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
                'post_and_file': merged_count
            },
            'thumbnail_stats': thumbnail_stats,
            'post_file_merged': True,
            'thumbnail_extraction': True,
            'telegram_session': bot_session_ready
        },
        'bot_username': Config.BOT_USERNAME
    }

# ============================================================================
# ✅ HOME MOVIES
# ============================================================================

async def get_home_movies(limit=25):
    """Get home movies"""
    movies = []
    
    if user_session_ready and User is not None:
        try:
            async for msg in User.get_chat_history(Config.MAIN_CHANNEL_ID, limit=limit):
                if msg and msg.text and len(msg.text) > 25:
                    title = extract_title_smart(msg.text)
                    if title:
                        year = extract_year_from_title(title)
                        clean_title_val = clean_title(title)
                        norm_title = normalize_title(clean_title_val)
                        
                        movie_data = {
                            'title': clean_title_val,
                            'year': year,
                            'normalized_title': norm_title,
                            'date': msg.date.isoformat() if isinstance(msg.date, datetime) else str(msg.date),
                            'channel_id': Config.MAIN_CHANNEL_ID,
                            'message_id': msg.id,
                            'has_file': False,
                            'has_post': True,
                            'content': format_post(msg.text, max_length=500),
                            'post_content': msg.text,
                            'result_type': 'post_only',
                            'quality_options': {},
                            'is_video_file': False
                        }
                        
                        # Get poster
                        poster_data = await poster_fetcher.fetch_poster(clean_title_val, year) if poster_fetcher else {
                            'poster_url': Config.FALLBACK_POSTER,
                            'source': 'fallback',
                            'rating': '0.0'
                        }
                        
                        movie_data.update({
                            'poster_url': poster_data['poster_url'],
                            'poster_source': poster_data['source'],
                            'poster_rating': poster_data['rating'],
                            'has_poster': True
                        })
                        
                        # Try to get thumbnail
                        thumbnail_url = None
                        if thumbnail_extractor:
                            thumbnail_url = await thumbnail_extractor.get_thumbnail(norm_title)
                        
                        if thumbnail_url:
                            movie_data.update({
                                'thumbnail_url': thumbnail_url,
                                'has_thumbnail': True,
                                'thumbnail_source': 'extracted'
                            })
                        else:
                            movie_data.update({
                                'thumbnail_url': poster_data['poster_url'],
                                'has_thumbnail': True,
                                'thumbnail_source': 'poster'
                            })
                        
                        movies.append(movie_data)
                        
                        if len(movies) >= limit:
                            break
            
            logger.info(f"✅ Fetched {len(movies)} home movies")
            
        except Exception as e:
            logger.error(f"❌ Home movies error: {e}")
    
    return movies

# ============================================================================
# ✅ SYSTEM INITIALIZATION
# ============================================================================

async def init_system():
    global poster_fetcher, thumbnail_extractor
    
    logger.info("=" * 60)
    logger.info("🚀 SK4FiLM v9.0 - FIXED THUMBNAIL EXTRACTION")
    logger.info("=" * 60)
    
    try:
        # Initialize MongoDB
        mongo_ok = await init_mongodb()
        if not mongo_ok:
            logger.error("❌ MongoDB failed")
            return False
        
        # Get current stats
        if files_col is not None:
            try:
                file_count = await files_col.count_documents({})
                logger.info(f"📊 Current files in database: {file_count}")
            except:
                logger.warning("⚠️ Could not count files")
        
        # Initialize Telegram Sessions
        telegram_ok = await init_telegram_sessions()
        
        # Initialize Poster Fetcher
        poster_fetcher = PosterFetcher(Config)
        logger.info("✅ Poster Fetcher initialized")
        
        # Initialize Thumbnail Extractor
        thumbnail_extractor = ThumbnailExtractor(mongo_client, Config)
        await thumbnail_extractor.initialize()
        logger.info("✅ Thumbnail Extractor initialized")
        
        logger.info("✅ SK4FiLM initialized successfully")
        logger.info("=" * 60)
        logger.info("🎯 FIXED THUMBNAIL FEATURES:")
        logger.info(f"   • Telegram Bot Session: {'✅ READY' if bot_session_ready else '❌ NOT READY'}")
        logger.info(f"   • Thumbnail Extraction: ✅ ENABLED")
        logger.info(f"   • Video File Detection: ✅ ENABLED")
        logger.info(f"   • Base64 Thumbnails: ✅ ENABLED")
        logger.info(f"   • File Channel ID: {Config.FILE_CHANNEL_ID}")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ System init error: {e}")
        return False

# ============================================================================
# ✅ API ROUTES
# ============================================================================

@app.route('/')
async def root():
    # Get stats
    file_count = 0
    if files_col is not None:
        try:
            file_count = await files_col.count_documents({})
        except:
            pass
    
    # Thumbnail stats
    thumb_stats = await thumbnail_extractor.get_stats() if thumbnail_extractor else {}
    
    return jsonify({
        'status': 'healthy',
        'service': 'SK4FiLM v9.0 - Fixed Thumbnails',
        'sessions': {
            'user': user_session_ready,
            'bot': bot_session_ready
        },
        'stats': {
            'total_files': file_count,
            'total_thumbnails': thumb_stats.get('total_thumbnails', 0),
            'telegram_extracted': thumb_stats.get('telegram_extracted', 0)
        },
        'features': {
            'post_file_merge': True,
            'poster_fetcher': poster_fetcher is not None,
            'thumbnail_extractor': thumbnail_extractor is not None,
            'telegram_thumbnails': bot_session_ready
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/health')
async def health():
    return jsonify({
        'status': 'ok',
        'post_file_merge': True,
        'thumbnail_extraction': thumbnail_extractor is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/movies', methods=['GET'])
async def api_movies():
    try:
        movies = await get_home_movies(limit=25)
        return jsonify({
            'status': 'success',
            'movies': movies,
            'total': len(movies),
            'poster_fetcher': poster_fetcher is not None,
            'thumbnail_extractor': thumbnail_extractor is not None
        })
    except Exception as e:
        logger.error(f"Movies API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
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
        
        return jsonify({
            'status': 'success',
            'query': query,
            'results': result_data['results'],
            'pagination': result_data['pagination'],
            'search_metadata': result_data['search_metadata'],
            'bot_username': Config.BOT_USERNAME
        })
    except Exception as e:
        logger.error(f"Search API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/stats', methods=['GET'])
async def api_stats():
    try:
        file_count = 0
        if files_col is not None:
            try:
                file_count = await files_col.count_documents({})
            except:
                pass
        
        # Get thumbnail stats
        thumb_stats = await thumbnail_extractor.get_stats() if thumbnail_extractor else {}
        
        return jsonify({
            'status': 'success',
            'database_files': file_count,
            'thumbnail_stats': thumb_stats,
            'sessions': {
                'user': user_session_ready,
                'bot': bot_session_ready
            },
            'features': {
                'post_file_merge': True,
                'thumbnail_extraction': True,
                'telegram_bot_ready': bot_session_ready
            }
        })
    except Exception as e:
        logger.error(f"Stats API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/admin/test-thumbnail', methods=['GET'])
async def api_admin_test_thumbnail():
    """Test thumbnail extraction"""
    try:
        auth_token = request.headers.get('X-Admin-Token')
        if not auth_token or auth_token != os.environ.get('ADMIN_TOKEN', 'sk4film_admin'):
            return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401
        
        # Test with a specific message
        channel_id = Config.FILE_CHANNEL_ID
        message_id = request.args.get('message_id', type=int)
        
        if not message_id:
            return jsonify({'status': 'error', 'message': 'message_id required'}), 400
        
        logger.info(f"🔍 Testing thumbnail extraction: {channel_id}/{message_id}")
        
        # Try to extract thumbnail
        thumbnail_url = None
        if thumbnail_extractor:
            thumbnail_url = await thumbnail_extractor.extract_from_telegram(
                channel_id, message_id, "test_movie"
            )
        
        if thumbnail_url:
            # Check if it's a data URL
            is_data_url = thumbnail_url.startswith('data:image/')
            thumbnail_type = 'data_url' if is_data_url else 'regular_url'
            thumbnail_preview = thumbnail_url[:100] + "..." if len(thumbnail_url) > 100 else thumbnail_url
            
            return jsonify({
                'status': 'success',
                'message': 'Thumbnail extracted successfully',
                'thumbnail_url': thumbnail_url,
                'thumbnail_type': thumbnail_type,
                'thumbnail_preview': thumbnail_preview,
                'channel_id': channel_id,
                'message_id': message_id
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to extract thumbnail',
                'possible_reasons': [
                    'Message not found',
                    'Message has no video/document',
                    'Video/document has no thumbnail',
                    'Telegram bot session not ready'
                ],
                'bot_session_ready': bot_session_ready,
                'thumbnail_extractor': thumbnail_extractor is not None
            }), 400
        
    except Exception as e:
        logger.error(f"Thumbnail test error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# ============================================================================
# ✅ STARTUP AND SHUTDOWN
# ============================================================================

@app.before_serving
async def startup():
    await init_system()

@app.after_serving
async def shutdown():
    logger.info("🛑 Shutting down SK4FiLM...")
    
    # Close poster fetcher
    if poster_fetcher is not None:
        await poster_fetcher.close()
    
    # Close Telegram sessions
    if User is not None:
        await User.stop()
    if Bot is not None:
        await Bot.stop()
    
    # Close MongoDB
    if mongo_client is not None:
        mongo_client.close()
    
    logger.info("👋 Shutdown complete")

# ============================================================================
# ✅ ERROR HANDLER
# ============================================================================

@app.errorhandler(Exception)
async def handle_error(error):
    logger.error(f"Unhandled error: {error}")
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500

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
    
    logger.info(f"🌐 Starting SK4FiLM on port {Config.WEB_SERVER_PORT}...")
    logger.info("=" * 60)
    logger.info("🎯 FIXED THUMBNAIL EXTRACTION SYSTEM")
    logger.info("=" * 60)
    logger.info(f"   • Telegram Bot: {'✅ REQUIRED' if Config.BOT_TOKEN else '❌ NOT CONFIGURED'}")
    logger.info(f"   • Thumbnail Extraction: ✅ ENABLED")
    logger.info(f"   • Base64 Thumbnails: ✅ ENABLED")
    logger.info(f"   • File Channel ID: {Config.FILE_CHANNEL_ID}")
    logger.info(f"   • Video File Detection: ✅ ENABLED")
    logger.info("=" * 60)
    
    try:
        asyncio.run(serve(app, config))
    except KeyboardInterrupt:
        logger.info("👋 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
