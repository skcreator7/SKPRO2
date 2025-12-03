"""
app.py - Main SK4FiLM Bot Web Server
"""
import asyncio
import os
import logging
import json
import re
import math
import html
import time
import secrets
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import defaultdict
from functools import wraps

import aiohttp
import urllib.parse
from quart import Quart, jsonify, request, Response
from hypercorn.asyncio import serve
from hypercorn.config import Config as HyperConfig
from motor.motor_asyncio import AsyncIOMotorClient
import redis.asyncio as redis

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('sk4film.log')
    ]
)
logger = logging.getLogger(__name__)

# Configuration class - Moved here to avoid circular imports
class Config:
    # Telegram API
    API_ID = int(os.environ.get("API_ID", "0"))
    API_HASH = os.environ.get("API_HASH", "")
    USER_SESSION_STRING = os.environ.get("USER_SESSION_STRING", "")
    BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
    
    # Database
    MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
    REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
    
    # Channels
    MAIN_CHANNEL_ID = -1001891090100
    TEXT_CHANNEL_IDS = [-1001891090100, -1002024811395]
    FILE_CHANNEL_ID = -1001768249569
    
    # URLs
    MAIN_CHANNEL_LINK = "https://t.me/sk4film"
    UPDATES_CHANNEL_LINK = "https://t.me/sk4film_Request"
    CHANNEL_USERNAME = "sk4film"
    WEBSITE_URL = os.environ.get("WEBSITE_URL", "https://sk4film.vercel.app")
    BACKEND_URL = os.environ.get("BACKEND_URL", "https://sk4film.koyeb.app")
    
    # Bot
    BOT_USERNAME = os.environ.get("BOT_USERNAME", "sk4filmbot")
    ADMIN_IDS = [int(x) for x in os.environ.get("ADMIN_IDS", "123456789").split(",")]
    AUTO_DELETE_TIME = int(os.environ.get("AUTO_DELETE_TIME", "300"))
    
    # Shortener & Verification
    SHORTLINK_API = os.environ.get("SHORTLINK_API", "")
    VERIFICATION_REQUIRED = os.environ.get("VERIFICATION_REQUIRED", "false").lower() == "true"
    VERIFICATION_DURATION = 6 * 60 * 60
    
    # Server
    WEB_SERVER_PORT = int(os.environ.get("PORT", 8000))
    
    # API Keys
    OMDB_KEYS = os.environ.get("OMDB_KEYS", "8265bd1c,b9bd48a6,3e7e1e9d").split(",")
    TMDB_KEYS = os.environ.get("TMDB_KEYS", "e547e17d4e91f3e62a571655cd1ccaff,8265bd1f").split(",")
    
    # UPI IDs for Premium
    UPI_ID_BASIC = os.environ.get("UPI_ID_BASIC", "sk4filmbot@ybl")
    UPI_ID_PREMIUM = os.environ.get("UPI_ID_PREMIUM", "sk4filmbot@ybl")
    UPI_ID_ULTIMATE = os.environ.get("UPI_ID_ULTIMATE", "sk4filmbot@ybl")
    UPI_ID_LIFETIME = os.environ.get("UPI_ID_LIFETIME", "sk4filmbot@ybl")
    
    # Admin
    ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "admin123")
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        errors = []
        
        if not cls.API_ID or cls.API_ID == 0:
            errors.append("API_ID is required")
        
        if not cls.API_HASH:
            errors.append("API_HASH is required")
        
        if not cls.BOT_TOKEN:
            errors.append("BOT_TOKEN is required")
        
        if not cls.ADMIN_IDS:
            errors.append("ADMIN_IDS is required")
        
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")

# Initialize Quart app
app = Quart(__name__)

@app.after_request
async def add_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

# Global instances - Will be set by main()
bot_instance = None
db_manager = None
files_col = None
User = None

# Rate limiters
api_rate_limiter = None
search_rate_limiter = None

# Enhanced search statistics
search_stats = {
    'redis_hits': 0,
    'redis_misses': 0,
    'multi_channel_searches': 0,
    'total_searches': 0
}

# Movie database cache
movie_db = {
    'search_cache': {},
    'poster_cache': {},
    'title_cache': {},
    'stats': {
        'redis_hits': 0,
        'redis_misses': 0,
        'multi_channel_searches': 0
    }
}

# Channel configuration
CHANNEL_CONFIG = {
    -1001891090100: {'name': 'SK4FiLM Main', 'type': 'text', 'search_priority': 1},
    -1002024811395: {'name': 'SK4FiLM Updates', 'type': 'text', 'search_priority': 2},
    -1001768249569: {'name': 'SK4FiLM Files', 'type': 'file', 'search_priority': 0}
}

# Redis cache handler
class RedisCache:
    def __init__(self, redis_url):
        self.redis_url = redis_url
        self.client = None
        self.enabled = False
    
    async def connect(self):
        try:
            if self.redis_url and self.redis_url != "redis://localhost:6379":
                self.client = await redis.from_url(self.redis_url, decode_responses=True)
                await self.client.ping()
                self.enabled = True
                logger.info("✅ Redis cache connected")
            else:
                logger.warning("Redis URL not configured, using memory cache")
        except Exception as e:
            logger.warning(f"Redis connection failed, using memory cache: {e}")
            self.enabled = False
    
    async def get(self, key):
        if not self.enabled or not self.client:
            return None
        try:
            data = await self.client.get(key)
            if data:
                search_stats['redis_hits'] += 1
                return json.loads(data)
            else:
                search_stats['redis_misses'] += 1
            return None
        except Exception:
            return None
    
    async def set(self, key, value, expire=3600):
        if not self.enabled or not self.client:
            return False
        try:
            await self.client.setex(key, expire, json.dumps(value))
            return True
        except Exception:
            return False
    
    async def clear_search_cache(self):
        if not self.enabled or not self.client:
            return False
        try:
            keys = await self.client.keys("search:*")
            if keys:
                await self.client.delete(*keys)
            return True
        except Exception:
            return False
    
    async def close(self):
        if self.client:
            await self.client.close()

redis_cache = RedisCache(Config.REDIS_URL)

# Utility functions
def normalize_title(title):
    """Normalize movie title for searching"""
    if not title:
        return ""
    normalized = title.lower().strip()
    
    normalized = re.sub(
        r'\b(19|20)\d{2}\b|\b(480p|720p|1080p|2160p|4k|hd|fhd|uhd|hevc|x264|x265|h264|h265|'
        r'bluray|webrip|hdrip|web-dl|hdtv|hdrip|webdl|hindi|english|tamil|telugu|'
        r'malayalam|kannada|punjabi|bengali|marathi|gujarati|movie|film|series|'
        r'complete|full|part|episode|season|hdrc|dvdscr|pre-dvd|p-dvd|pdc|'
        r'rarbg|yts|amzn|netflix|hotstar|prime|disney|hc-esub|esub|subs)\b',
        '', 
        normalized, 
        flags=re.IGNORECASE
    )
    
    normalized = re.sub(r'[\._\-]', ' ', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized

def extract_title_smart(text):
    """Extract movie title from text"""
    if not text or len(text) < 10:
        return None
    
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if not lines:
        return None
    
    first_line = lines[0]
    
    patterns = [
        (r'^([A-Za-z\s]{3,50}?)\s*(?:\(?\d{4}\)?|\b(?:480p|720p|1080p|2160p|4k|hd|fhd|uhd)\b)', 1),
        (r'🎬\s*([^\n\-\(]{3,60}?)\s*(?:\(\d{4}\)|$)', 1),
        (r'^([^\-\n]{3,60}?)\s*\-', 1),
        (r'^([^\(\n]{3,60}?)\s*\(\d{4}\)', 1),
        (r'^([A-Za-z\s]{3,50}?)\s*(?:\d{4}|Hindi|Movie|Film|HDTC|WebDL|X264|AAC|ESub)', 1),
    ]
    
    for pattern, group in patterns:
        match = re.search(pattern, first_line, re.IGNORECASE)
        if match:
            title = match.group(group).strip()
            title = re.sub(r'\s+', ' ', title)
            if 3 <= len(title) <= 60:
                return title
    
    return None

def extract_title_from_file(msg):
    """Extract title from file message"""
    try:
        if msg.caption:
            t = extract_title_smart(msg.caption)
            if t:
                return t
        
        fn = msg.document.file_name if msg.document else (msg.video.file_name if msg.video else None)
        if fn:
            name = fn.rsplit('.', 1)[0]
            name = re.sub(r'[\._\-]', ' ', name)
            
            name = re.sub(
                r'\b(720p|1080p|480p|2160p|4k|HDRip|WEBRip|WEB-DL|BluRay|BRRip|DVDRip|HDTV|HDTC|'
                r'X264|X265|HEVC|H264|H265|AAC|AC3|DD5\.1|DDP5\.1|HC|ESub|Subs|Hindi|English|'
                r'Dual|Multi|Complete|Full|Movie|Film|\d{4})\b',
                '', 
                name, 
                flags=re.IGNORECASE
            )
            
            name = re.sub(r'\s+', ' ', name).strip()
            name = re.sub(r'\s+\(\d{4}\)$', '', name)
            name = re.sub(r'\s+\d{4}$', '', name)
            
            if 4 <= len(name) <= 50:
                return name
    except Exception as e:
        logger.error(f"File title extraction error: {e}")
    return None

def format_size(size):
    """Format file size"""
    if not size:
        return "Unknown"
    if size < 1024*1024:
        return f"{size/1024:.1f} KB"
    elif size < 1024*1024*1024:
        return f"{size/(1024*1024):.1f} MB"
    else:
        return f"{size/(1024*1024*1024):.2f} GB"

def detect_quality(filename):
    """Detect video quality from filename"""
    if not filename:
        return "480p"
    fl = filename.lower()
    is_hevc = 'hevc' in fl or 'x265' in fl
    if '2160p' in fl or '4k' in fl:
        return "2160p HEVC" if is_hevc else "2160p"
    elif '1080p' in fl:
        return "1080p HEVC" if is_hevc else "1080p"
    elif '720p' in fl:
        return "720p HEVC" if is_hevc else "720p"
    elif '480p' in fl:
        return "480p HEVC" if is_hevc else "480p"
    return "480p"

def format_post(text):
    """Format post text for HTML"""
    if not text:
        return ""
    text = html.escape(text)
    text = re.sub(r'(https?://[^\s]+)', r'<a href="\1" target="_blank" style="color:#00ccff">\1</a>', text)
    return text.replace('\n', '<br>')

def channel_name(cid):
    """Get channel name from config"""
    return CHANNEL_CONFIG.get(cid, {}).get('name', f"Channel {cid}")

def is_new(date):
    """Check if date is within 48 hours"""
    try:
        if isinstance(date, str):
            date = datetime.fromisoformat(date.replace('Z', '+00:00'))
        hours = (datetime.now() - date.replace(tzinfo=None)).total_seconds() / 3600
        return hours <= 48
    except:
        return False

def is_video_file(file_name):
    """Check if file is video"""
    if not file_name:
        return False
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.mpg', '.mpeg']
    file_name_lower = file_name.lower()
    return any(file_name_lower.endswith(ext) for ext in video_extensions)

async def safe_telegram_operation(func, *args, **kwargs):
    """Safely execute Telegram operations with error handling"""
    try:
        return await func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Telegram operation error: {e}")
        return None

async def safe_telegram_generator(func, *args, **kwargs):
    """Safely iterate through Telegram generator"""
    try:
        async for item in func(*args, **kwargs):
            yield item
    except Exception as e:
        logger.error(f"Telegram generator error: {e}")

async def index_single_file(tg_message):
    """Index a single file from Telegram message"""
    try:
        logger.info(f"Indexing file from message {tg_message.id}")
        return True
    except Exception as e:
        logger.error(f"Indexing error: {e}")
        return False

async def auto_delete_file(message, delay_seconds):
    """Auto-delete file after specified delay"""
    try:
        await asyncio.sleep(delay_seconds)
        await message.delete()
        logger.info(f"Auto-deleted file message after {delay_seconds} seconds")
    except Exception as e:
        logger.error(f"Auto-delete error: {e}")

async def get_telegram_video_thumbnail(user_client, channel_id, message_id):
    """Get thumbnail from Telegram video"""
    try:
        msg = await safe_telegram_operation(
            user_client.get_messages,
            channel_id, 
            message_id
        )
        if msg and msg.video and msg.video.thumbs:
            # Get the largest thumbnail
            thumb = msg.video.thumbs[-1]
            file = await user_client.download_media(thumb.file_id)
            return file
    except Exception as e:
        logger.error(f"Error getting video thumbnail: {e}")
    return None

async def verify_user_with_url_shortener(user_id, verification_url):
    """Verify user with URL shortener"""
    try:
        if not Config.VERIFICATION_REQUIRED:
            return True, "Verification not required"
        
        # Implement actual verification logic here
        is_verified = True  # Placeholder
        message = "User verified successfully"
        
        return is_verified, message
    except Exception as e:
        return False, f"Verification error: {e}"

async def check_url_shortener_verification(user_id):
    """Check URL shortener verification"""
    try:
        if not Config.VERIFICATION_REQUIRED:
            return True, "Verification not required"
        
        # Implement actual check logic here
        is_verified = True  # Placeholder
        message = "User is verified"
        
        return is_verified, message
    except Exception as e:
        return False, f"Check error: {e}"

async def generate_verification_url(user_id):
    """Generate verification URL"""
    try:
        # Generate verification URL
        verification_url = f"https://t.me/{Config.BOT_USERNAME}?start=verify_{user_id}"
        return verification_url
    except Exception as e:
        logger.error(f"Error generating verification URL: {e}")
        return ""

async def get_home_movies_live():
    """Get home movies from database"""
    try:
        if files_col is None:
            return []
        
        # Get latest movies from database
        cursor = files_col.find(
            {'is_video_file': True}
        ).sort('date', -1).limit(20)
        
        movies = []
        async for doc in cursor:
            movies.append({
                'id': f"{doc.get('channel_id', Config.FILE_CHANNEL_ID)}_{doc.get('message_id')}",
                'title': doc.get('title', 'Unknown'),
                'quality': doc.get('quality', '480p'),
                'size': doc.get('formatted_size', 'Unknown'),
                'channel': doc.get('channel_name', 'Unknown'),
                'is_new': doc.get('is_new', False),
                'date': doc.get('date', datetime.now()).isoformat() if isinstance(doc.get('date'), datetime) else str(doc.get('date'))
            })
        
        return movies
    except Exception as e:
        logger.error(f"Error getting home movies: {e}")
        return []

async def search_movies_multi_channel(query, limit=12, page=1):
    """Search movies across multiple channels"""
    try:
        cache_key = f"search:{query}:{limit}:{page}"
        
        # Check Redis cache first
        cached_result = await redis_cache.get(cache_key)
        if cached_result:
            search_stats['redis_hits'] += 1
            movie_db['stats']['redis_hits'] += 1
            return cached_result
        
        search_stats['redis_misses'] += 1
        movie_db['stats']['redis_misses'] += 1
        
        # Search in database
        if files_col is None:
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
                    'source': 'database',
                    'channels_searched': 0
                }
            }
        
        normalized_query = normalize_title(query)
        
        # Build search query
        search_query = {
            'is_video_file': True,
            '$or': [
                {'title': {'$regex': normalized_query, '$options': 'i'}},
                {'normalized_title': {'$regex': normalized_query, '$options': 'i'}},
                {'file_name': {'$regex': normalized_query, '$options': 'i'}}
            ]
        }
        
        # Count total results
        total_results = await files_col.count_documents(search_query)
        total_pages = max(1, math.ceil(total_results / limit))
        
        # Get paginated results
        skip = (page - 1) * limit
        cursor = files_col.find(search_query).sort('date', -1).skip(skip).limit(limit)
        
        results = []
        async for doc in cursor:
            results.append({
                'id': f"{doc.get('channel_id', Config.FILE_CHANNEL_ID)}_{doc.get('message_id')}",
                'title': doc.get('title', 'Unknown'),
                'quality': doc.get('quality', '480p'),
                'size': doc.get('formatted_size', 'Unknown'),
                'channel': doc.get('channel_name', 'Unknown'),
                'is_new': doc.get('is_new', False),
                'date': doc.get('date', datetime.now()).isoformat() if isinstance(doc.get('date'), datetime) else str(doc.get('date')),
                'has_thumbnail': doc.get('thumbnail') is not None,
                'channel_id': doc.get('channel_id'),
                'message_id': doc.get('message_id')
            })
        
        result = {
            'results': results,
            'pagination': {
                'current_page': page,
                'total_pages': total_pages,
                'total_results': total_results,
                'per_page': limit,
                'has_next': page < total_pages,
                'has_previous': page > 1
            },
            'search_metadata': {
                'source': 'database',
                'channels_searched': len(set(doc.get('channel_id') for doc in results if doc.get('channel_id'))),
                'query_type': 'multi_channel',
                'cache_hit': False
            }
        }
        
        # Cache in Redis
        await redis_cache.set(cache_key, result, expire=300)
        
        search_stats['multi_channel_searches'] += 1
        movie_db['stats']['multi_channel_searches'] += 1
        
        return result
        
    except Exception as e:
        logger.error(f"Search error: {e}")
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
                'source': 'error',
                'channels_searched': 0,
                'error': str(e)
            }
        }

# Database Manager
class DatabaseManager:
    def __init__(self, uri, max_pool_size=10):
        self.uri = uri
        self.max_pool_size = max_pool_size
        self.client = None
        self.db = None
        self.files_col = None
        self.users_col = None
        self.premium_col = None
    
    async def connect(self):
        """Connect to MongoDB with connection pooling"""
        try:
            self.client = AsyncIOMotorClient(
                self.uri,
                maxPoolSize=self.max_pool_size,
                minPoolSize=5,
                maxIdleTimeMS=60000,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000
            )
            
            # Test connection
            await self.client.admin.command('ping')
            
            self.db = self.client['sk4film']
            self.files_col = self.db['files']
            self.users_col = self.db['users']
            self.premium_col = self.db['premium']
            
            # Create indexes with error handling
            try:
                await self.files_col.create_index([('normalized_title', 'text')])
            except Exception as e:
                logger.warning(f"Text index warning: {e}")
            
            try:
                await self.files_col.create_index([('channel_id', 1), ('message_id', 1)], unique=True)
                await self.files_col.create_index([('date', -1)])
                await self.files_col.create_index([('is_video_file', 1)])
            except Exception as e:
                logger.warning(f"Index creation warning: {e}")
            
            try:
                await self.premium_col.create_index([('user_id', 1)], unique=True)
                await self.premium_col.create_index([('expires_at', 1)])
            except Exception as e:
                logger.warning(f"Premium index creation warning: {e}")
            
            logger.info("✅ MongoDB connected with connection pooling")
            return True
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            return False
    
    async def close(self):
        """Close database connection"""
        if self.client:
            self.client.close()
            logger.info("✅ MongoDB connection closed")

# Rate Limiter
class RateLimiter:
    def __init__(self, max_requests, window_seconds):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
    
    async def is_allowed(self, key):
        now = time.time()
        window_start = now - self.window_seconds
        
        # Clean old requests
        self.requests[key] = [req_time for req_time in self.requests[key] 
                             if req_time > window_start]
        
        # Check limit
        if len(self.requests[key]) >= self.max_requests:
            return False
        
        # Add current request
        self.requests[key].append(now)
        return True

# Bot status variables
bot_started = False
user_session_ready = False

async def idle():
    """Keep the bot running"""
    while True:
        await asyncio.sleep(3600)

# OPTIMIZED API ROUTES (as per your request)
@app.route('/')
async def root():
    tf = await files_col.count_documents({}) if files_col is not None else 0
    video_files = await files_col.count_documents({'is_video_file': True}) if files_col is not None else 0
    thumbnails = await files_col.count_documents({'thumbnail': {'$ne': None}}) if files_col is not None else 0
    
    return jsonify({
        'status': 'healthy',
        'service': 'SK4FiLM v8.0 - MULTI-CHANNEL ENHANCED',
        'database': {
            'total_files': tf, 
            'video_files': video_files,
            'thumbnails': thumbnails,
            'mode': 'MULTI-CHANNEL ENHANCED'
        },
        'channels': {
            'text_channels': len(Config.TEXT_CHANNEL_IDS),
            'file_channels': 1,
            'total_channels': len(Config.TEXT_CHANNEL_IDS) + 1
        },
        'cache': {
            'redis_enabled': redis_cache.enabled,
            'redis_hits': movie_db['stats']['redis_hits'],
            'redis_misses': movie_db['stats']['redis_misses'],
            'multi_channel_searches': movie_db['stats']['multi_channel_searches']
        },
        'bot_status': 'online' if bot_started else 'starting',
        'user_session': 'ready' if user_session_ready else 'flood_wait',
        'features': 'MULTI-CHANNEL SEARCH + REDIS CACHE + BATCH THUMBNAILS'
    })

@app.route('/health')
async def health():
    return jsonify({
        'status': 'ok' if bot_started else 'starting',
        'user_session': user_session_ready,
        'redis_enabled': redis_cache.enabled,
        'channels_configured': len(Config.TEXT_CHANNEL_IDS),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/verify_user', methods=['POST'])
async def api_verify_user():
    try:
        data = await request.get_json()
        user_id = data.get('user_id')
        verification_url = data.get('verification_url')
        
        if not user_id:
            return jsonify({'status': 'error', 'message': 'User ID required'}), 400
        
        is_verified, message = await verify_user_with_url_shortener(user_id, verification_url)
        
        return jsonify({
            'status': 'success' if is_verified else 'error',
            'verified': is_verified,
            'message': message,
            'user_id': user_id
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/check_verification/<int:user_id>')
async def api_check_verification(user_id):
    try:
        is_verified, message = await check_url_shortener_verification(user_id)
        return jsonify({
            'status': 'success',
            'verified': is_verified,
            'message': message,
            'user_id': user_id,
            'verification_required': Config.VERIFICATION_REQUIRED
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/generate_verification_url/<int:user_id>')
async def api_generate_verification_url(user_id):
    try:
        verification_url = await generate_verification_url(user_id)
        return jsonify({
            'status': 'success',
            'verification_url': verification_url,
            'user_id': user_id
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/index_status')
async def api_index_status():
    try:
        if files_col is None:
            return jsonify({'status': 'error', 'message': 'Database not ready'}), 503
        
        total = await files_col.count_documents({})
        video_files = await files_col.count_documents({'is_video_file': True})
        video_thumbnails = await files_col.count_documents({'is_video_file': True, 'thumbnail': {'$ne': None}})
        total_thumbnails = await files_col.count_documents({'thumbnail': {'$ne': None}})
        
        latest = await files_col.find_one({}, sort=[('indexed_at', -1)])
        last_indexed = "Never"
        if latest and latest.get('indexed_at'):
            dt = latest['indexed_at']
            if isinstance(dt, datetime):
                mins_ago = int((datetime.now() - dt).total_seconds() / 60)
                last_indexed = f"{mins_ago} min ago" if mins_ago > 0 else "Just now"
        
        return jsonify({
            'status': 'success',
            'total_indexed': total,
            'video_files': video_files,
            'video_thumbnails': video_thumbnails,
            'total_thumbnails': total_thumbnails,
            'thumbnail_coverage': f"{(video_thumbnails/video_files*100):.1f}%" if video_files > 0 else "0%",
            'last_indexed': last_indexed,
            'bot_status': 'online' if bot_started else 'starting',
            'user_session': user_session_ready,
            'redis_enabled': redis_cache.enabled,
            'channels': {
                'text_channels': Config.TEXT_CHANNEL_IDS,
                'file_channel': Config.FILE_CHANNEL_ID
            },
            'cache_stats': {
                'redis_hits': movie_db['stats']['redis_hits'],
                'redis_misses': movie_db['stats']['redis_misses'],
                'multi_channel_searches': movie_db['stats']['multi_channel_searches']
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/movies')
async def api_movies():
    try:
        if not bot_started:
            return jsonify({'status': 'error', 'message': 'Starting...'}), 503
        
        movies = await get_home_movies_live()
        return jsonify({
            'status': 'success', 
            'movies': movies, 
            'total': len(movies), 
            'bot_username': Config.BOT_USERNAME,
            'mode': 'MULTI_CHANNEL_ENHANCED',
            'channels_searched': len(Config.TEXT_CHANNEL_IDS)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/search')
async def api_search():
    try:
        q = request.args.get('query', '').strip()
        p = int(request.args.get('page', 1))
        l = int(request.args.get('limit', 12))
        
        if not q:
            return jsonify({'status': 'error', 'message': 'Query required'}), 400
        if not bot_started:
            return jsonify({'status': 'error', 'message': 'Starting...'}), 503
        
        result = await search_movies_multi_channel(q, l, p)
        return jsonify({
            'status': 'success', 
            'query': q, 
            'results': result['results'], 
            'pagination': result['pagination'],
            'search_metadata': result.get('search_metadata', {}),
            'bot_username': Config.BOT_USERNAME,
            'cache': 'redis' if redis_cache.enabled else 'memory',
            'mode': 'MULTI_CHANNEL_ENHANCED'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/clear_cache')
async def api_clear_cache():
    try:
        # Clear Redis cache
        if redis_cache.enabled:
            await redis_cache.clear_search_cache()
        
        # Clear memory cache
        movie_db['search_cache'].clear()
        movie_db['poster_cache'].clear()
        movie_db['title_cache'].clear()
        
        movie_db['stats']['redis_hits'] = 0
        movie_db['stats']['redis_misses'] = 0
        movie_db['stats']['multi_channel_searches'] = 0
        
        return jsonify({
            'status': 'success',
            'message': 'All caches cleared successfully',
            'redis_cleared': redis_cache.enabled,
            'memory_cleared': True,
            'stats_reset': True
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/post')
async def api_post():
    try:
        channel_id = request.args.get('channel', '').strip()
        message_id = request.args.get('message', '').strip()
        
        if not channel_id or not message_id:
            return jsonify({'status':'error', 'message':'Missing channel or message parameter'}), 400
        
        if not bot_started or not User or not user_session_ready:
            return jsonify({'status':'error', 'message':'Bot not ready yet - Flood wait active'}), 503
        
        try:
            channel_id = int(channel_id)
            message_id = int(message_id)
        except ValueError:
            return jsonify({'status':'error', 'message':'Invalid channel or message ID'}), 400
        
        msg = await safe_telegram_operation(
            User.get_messages,
            channel_id, 
            message_id
        )
        if not msg or not msg.text:
            return jsonify({'status':'error', 'message':'Message not found or has no text content'}), 404
        
        title = extract_title_smart(msg.text)
        if not title:
            title = msg.text.split('\n')[0][:60] if msg.text else "Movie Post"
        
        normalized_title = normalize_title(title)
        quality_options = {}
        has_file = False
        thumbnail_url = None
        thumbnail_source = None
        
        if files_col is not None:
            cursor = files_col.find({'normalized_title': normalized_title})
            async for doc in cursor:
                quality = doc.get('quality', '480p')
                if quality not in quality_options:
                    file_name = doc.get('file_name', '').lower()
                    file_is_video = is_video_file(file_name)
                    
                    if file_is_video and not thumbnail_url:
                        thumbnail_url = doc.get('thumbnail')
                        thumbnail_source = doc.get('thumbnail_source', 'unknown')
                        
                        if not thumbnail_url and user_session_ready:
                            thumbnail_url = await get_telegram_video_thumbnail(User, doc['channel_id'], doc['message_id'])
                            if thumbnail_url:
                                thumbnail_source = 'video_direct'
                    
                    quality_options[quality] = {
                        'file_id': f"{doc.get('channel_id', Config.FILE_CHANNEL_ID)}_{doc.get('message_id')}_{quality}",
                        'file_size': doc.get('file_size', 0),
                        'file_name': doc.get('file_name', 'video.mp4'),
                        'is_video': file_is_video,
                        'channel_id': doc.get('channel_id'),
                        'message_id': doc.get('message_id')
                    }
                    has_file = True
        
        post_data = {
            'title': title,
            'content': format_post(msg.text),
            'channel': channel_name(channel_id),
            'channel_id': channel_id,
            'message_id': message_id,
            'date': msg.date.isoformat() if isinstance(msg.date, datetime) else str(msg.date),
            'is_new': is_new(msg.date) if msg.date else False,
            'has_file': has_file,
            'quality_options': quality_options,
            'views': getattr(msg, 'views', 0),
            'thumbnail': thumbnail_url,
            'thumbnail_source': thumbnail_source
        }
        
        return jsonify({'status': 'success', 'post': post_data, 'bot_username': Config.BOT_USERNAME})
    
    except Exception as e:
        return jsonify({'status':'error', 'message': str(e)}), 500

@app.route('/api/poster')
async def api_poster():
    try:
        t = request.args.get('title', 'Movie')
        y = request.args.get('year', '')
        
        d = t[:20] + "..." if len(t) > 20 else t
        
        color_schemes = [
            {'bg1': '#667eea', 'bg2': '#764ba2', 'text': '#ffffff'},
            {'bg1': '#f093fb', 'bg2': '#f5576c', 'text': '#ffffff'},
            {'bg1': '#4facfe', 'bg2': '#00f2fe', 'text': '#ffffff'},
            {'bg1': '#43e97b', 'bg2': '#38f9d7', 'text': '#ffffff'},
            {'bg1': '#fa709a', 'bg2': '#fee140', 'text': '#ffffff'},
        ]
        
        scheme = color_schemes[hash(t) % len(color_schemes)]
        text_color = scheme['text']
        bg1_color = scheme['bg1']
        bg2_color = scheme['bg2']
        
        year_text = f'<text x="150" y="305" text-anchor="middle" fill="{text_color}" font-size="14" font-family="Arial">{html.escape(y)}</text>' if y else ''
        
        svg = f'''<svg width="300" height="450" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:{bg1_color};stop-opacity:1"/>
                    <stop offset="100%" style="stop-color:{bg2_color};stop-opacity:1"/>
                </linearGradient>
            </defs>
            <rect width="100%" height="100%" fill="url(#bg)"/>
            <rect x="10" y="10" width="280" height="430" fill="none" stroke="{text_color}" stroke-width="2" stroke-opacity="0.3" rx="10"/>
            <circle cx="150" cy="180" r="60" fill="rgba(255,255,255,0.1)"/>
            <text x="150" y="185" text-anchor="middle" fill="{text_color}" font-size="60" font-family="Arial">🎬</text>
            <text x="150" y="280" text-anchor="middle" fill="{text_color}" font-size="16" font-weight="bold" font-family="Arial">{html.escape(d)}</text>
            {year_text}
            <rect x="50" y="380" width="200" height="40" rx="20" fill="rgba(0,0,0,0.3)"/>
            <text x="150" y="405" text-anchor="middle" fill="{text_color}" font-size="16" font-weight="bold" font-family="Arial">SK4FiLM</text>
        </svg>'''
        
        return Response(svg, mimetype='image/svg+xml', headers={
            'Cache-Control': 'public, max-age=86400',
            'Content-Type': 'image/svg+xml'
        })
    except Exception as e:
        simple_svg = '''<svg width="300" height="450" xmlns="http://www.w3.org/2000/svg">
            <rect width="100%" height="100%" fill="#667eea"/>
            <text x="150" y="225" text-anchor="middle" fill="white" font-size="18" font-family="Arial">SK4FiLM</text>
        </svg>'''
        return Response(simple_svg, mimetype='image/svg+xml')

async def main():
    """Main function to start the bot"""
    global bot_instance, db_manager, files_col, bot_started, user_session_ready, User
    
    try:
        # Initialize configuration
        config = Config()
        config.validate()
        
        # Initialize Redis cache
        await redis_cache.connect()
        
        # Initialize database if MONGODB_URI is provided
        if config.MONGODB_URI and config.MONGODB_URI != "mongodb://localhost:27017":
            db_manager = DatabaseManager(config.MONGODB_URI)
            await db_manager.connect()
            if db_manager.files_col:
                files_col = db_manager.files_col
        
        # Now import bot_handlers (after all dependencies are defined)
        try:
            from bot_handlers import SK4FiLMBot, setup_bot_handlers
            
            # Create bot instance
            bot_instance = SK4FiLMBot(config, db_manager)
            
            # Initialize bot
            await bot_instance.initialize()
            
            # Update global variables
            bot_started = bot_instance.bot_started
            user_session_ready = bot_instance.user_session_ready
            User = bot_instance.user_client
            
            # Setup bot handlers
            await setup_bot_handlers(bot_instance.bot, bot_instance)
            
            logger.info(f"🤖 Bot initialized successfully: @{config.BOT_USERNAME}")
            
        except ImportError as e:
            logger.warning(f"Bot handlers not available: {e}")
            bot_started = True  # Start web server anyway
        
        # Start web server
        hypercorn_config = HyperConfig()
        hypercorn_config.bind = [f"0.0.0.0:{config.WEB_SERVER_PORT}"]
        hypercorn_config.workers = 1
        
        logger.info(f"🚀 Starting web server on port {config.WEB_SERVER_PORT}")
        logger.info(f"🤖 Bot username: @{config.BOT_USERNAME}")
        logger.info(f"🌐 Website: {config.WEBSITE_URL}")
        
        # Run web server
        await serve(app, hypercorn_config)
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if bot_instance:
            await bot_instance.shutdown()
        if db_manager:
            await db_manager.close()
        await redis_cache.close()
        logger.info("Bot stopped")

if __name__ == "__main__":
    asyncio.run(main())
