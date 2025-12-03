"""
app.py - Complete SK4FiLM Web API System (Frontend Ready)
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
import base64
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

# Configuration class
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
    UPI_ID_GOLD = os.environ.get("UPI_ID_GOLD", "sk4filmbot@ybl")
    UPI_ID_DIAMOND = os.environ.get("UPI_ID_DIAMOND", "sk4filmbot@ybl")
    
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

# Global instances
bot_instance = None
db_manager = None
premium_system = None
verification_system = None
cache_manager = None
poster_fetcher = None

# Search statistics
search_stats = {
    'redis_hits': 0,
    'redis_misses': 0,
    'total_searches': 0,
    'api_requests': 0
}

# Channel configuration
CHANNEL_CONFIG = {
    -1001891090100: {'name': 'SK4FiLM Main', 'type': 'text', 'search_priority': 1},
    -1002024811395: {'name': 'SK4FiLM Updates', 'type': 'text', 'search_priority': 2},
    -1001768249569: {'name': 'SK4FiLM Files', 'type': 'file', 'search_priority': 0}
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
        self.verification_col = None
    
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
            self.verification_col = self.db['verifications']
            
            # Create indexes
            await self._create_indexes()
            
            logger.info("✅ MongoDB connected with connection pooling")
            return True
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            return False
    
    async def _create_indexes(self):
        """Create database indexes"""
        try:
            # Files collection indexes
            await self.files_col.create_index([('normalized_title', 'text')])
            await self.files_col.create_index([('channel_id', 1), ('message_id', 1)], unique=True)
            await self.files_col.create_index([('indexed_at', -1)])
            await self.files_col.create_index([('is_video_file', 1)])
            
            # Premium collection indexes
            await self.premium_col.create_index([('user_id', 1)], unique=True)
            await self.premium_col.create_index([('expires_at', 1)])
            
            # Verification collection indexes
            await self.verification_col.create_index([('user_id', 1)], unique=True)
            
            logger.info("✅ Database indexes created")
        except Exception as e:
            logger.warning(f"Index creation warning: {e}")
    
    async def get_file_count(self):
        """Get total file count"""
        try:
            return await self.files_col.count_documents({})
        except:
            return 0
    
    async def get_video_file_count(self):
        """Get video file count"""
        try:
            return await self.files_col.count_documents({'is_video_file': True})
        except:
            return 0
    
    async def get_thumbnail_count(self):
        """Get thumbnail count"""
        try:
            return await self.files_col.count_documents({'thumbnail': {'$ne': None}})
        except:
            return 0
    
    async def get_last_indexed(self):
        """Get last indexed file info"""
        try:
            return await self.files_col.find_one({}, sort=[('indexed_at', -1)])
        except:
            return None
    
    async def search_files(self, query, limit=12, skip=0):
        """Search files in database"""
        try:
            # Text search
            cursor = self.files_col.find(
                {'$text': {'$search': query}},
                {'score': {'$meta': 'textScore'}}
            ).sort([('score', {'$meta': 'textScore'})]).skip(skip).limit(limit)
            
            results = []
            async for doc in cursor:
                results.append(doc)
            
            return results
        except Exception as e:
            logger.error(f"Database search error: {e}")
            return []
    
    async def get_file_by_message(self, channel_id, message_id):
        """Get file by channel and message ID"""
        try:
            return await self.files_col.find_one({
                'channel_id': channel_id,
                'message_id': message_id
            })
        except:
            return None
    
    async def close(self):
        """Close database connection"""
        if self.client:
            self.client.close()
            logger.info("✅ MongoDB connection closed")

# Cache Manager (Simplified)
class CacheManager:
    def __init__(self):
        self.client = None
        self.redis_enabled = False
        
    async def init_redis(self):
        """Initialize Redis connection"""
        try:
            if Config.REDIS_URL:
                self.client = redis.from_url(
                    Config.REDIS_URL,
                    decode_responses=True,
                    encoding='utf-8'
                )
                await self.client.ping()
                self.redis_enabled = True
                logger.info("✅ Redis connected successfully")
                return True
        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed: {e}")
            self.redis_enabled = False
            return False
    
    async def get(self, key):
        """Get value from cache"""
        if not self.redis_enabled:
            return None
        try:
            return await self.client.get(key)
        except:
            return None
    
    async def set(self, key, value, expire_seconds=3600):
        """Set value in cache"""
        if not self.redis_enabled:
            return False
        try:
            await self.client.setex(key, expire_seconds, value)
            return True
        except:
            return False
    
    async def delete(self, key):
        """Delete key from cache"""
        if not self.redis_enabled:
            return False
        try:
            await self.client.delete(key)
            return True
        except:
            return False
    
    async def clear_search_cache(self):
        """Clear all search cache"""
        if not self.redis_enabled:
            return 0
        try:
            keys = await self.client.keys("search:*")
            if keys:
                await self.client.delete(*keys)
            return len(keys)
        except:
            return 0

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

# API Helper Functions
async def search_movies_api(query, limit=12, page=1):
    """Search movies API function"""
    offset = (page - 1) * limit
    search_stats['total_searches'] += 1
    
    # Try cache first
    cache_key = f"search:{query}:{page}:{limit}"
    if cache_manager and cache_manager.redis_enabled:
        cached = await cache_manager.get(cache_key)
        if cached:
            search_stats['redis_hits'] += 1
            try:
                return json.loads(cached)
            except:
                pass
        else:
            search_stats['redis_misses'] += 1
    
    # Sample search results (Replace with actual database search)
    sample_results = []
    
    # Generate sample data based on query
    if 'tere' in query.lower() or 'ishk' in query.lower():
        sample_results.append({
            'title': 'Tere Ishk Mein (2025)',
            'content': '🎬 <b>Tere Ishk Mein (2025)</b>\n📅 Release: 2025\n🎭 Genre: Romance, Drama\n⭐ Starring: New Cast\n\n📥 Download now from SK4FiLM!',
            'channel': 'SK4FiLM Main',
            'channel_id': -1001891090100,
            'message_id': 12345,
            'date': datetime.now().isoformat(),
            'is_new': True,
            'has_file': True,
            'has_post': True,
            'quality_options': {
                '1080p HEVC': {
                    'file_id': '-1001768249569_123456_1080p',
                    'file_size': 1500000000,
                    'file_name': 'Tere.Ishk.Mein.2025.1080p.HEVC.mkv',
                    'is_video': True
                },
                '720p': {
                    'file_id': '-1001768249569_123457_720p',
                    'file_size': 800000000,
                    'file_name': 'Tere.Ishk.Mein.2025.720p.mkv',
                    'is_video': True
                }
            },
            'thumbnail': None
        })
    
    if 'pushpa' in query.lower():
        sample_results.append({
            'title': 'Pushpa 2: The Rule (2024)',
            'content': '🎬 <b>Pushpa 2: The Rule (2024)</b>\n📅 Release: 2024\n🎭 Genre: Action, Drama\n⭐ Starring: Allu Arjun\n\n📥 Download now from SK4FiLM!',
            'channel': 'SK4FiLM Updates',
            'channel_id': -1002024811395,
            'message_id': 67890,
            'date': datetime.now().isoformat(),
            'is_new': True,
            'has_file': True,
            'has_post': True,
            'quality_options': {
                '4K': {
                    'file_id': '-1001768249569_234567_4k',
                    'file_size': 3500000000,
                    'file_name': 'Pushpa.2.The.Rule.2024.4K.mkv',
                    'is_video': True
                }
            },
            'thumbnail': None
        })
    
    if 'kalki' in query.lower():
        sample_results.append({
            'title': 'Kalki 2898 AD (2024)',
            'content': '🎬 <b>Kalki 2898 AD (2024)</b>\n📅 Release: 2024\n🎭 Genre: Sci-Fi, Action\n⭐ Starring: Prabhas, Amitabh Bachchan\n\n📥 Download now from SK4FiLM!',
            'channel': 'SK4FiLM Main',
            'channel_id': -1001891090100,
            'message_id': 34567,
            'date': (datetime.now() - timedelta(days=3)).isoformat(),
            'is_new': False,
            'has_file': True,
            'has_post': True,
            'quality_options': {
                '1080p': {
                    'file_id': '-1001768249569_345678_1080p',
                    'file_size': 2200000000,
                    'file_name': 'Kalki.2898.AD.2024.1080p.mkv',
                    'is_video': True
                }
            },
            'thumbnail': None
        })
    
    # Filter by query if no specific matches
    if not sample_results:
        sample_results = [
            {
                'title': 'Sample Movie (2024)',
                'content': f'🎬 <b>Sample Movie (2024)</b>\nSearch query: {query}\n\n📥 Download now from SK4FiLM!',
                'channel': 'SK4FiLM',
                'channel_id': -1001891090100,
                'message_id': 99999,
                'date': datetime.now().isoformat(),
                'is_new': True,
                'has_file': True,
                'has_post': True,
                'quality_options': {
                    '720p': {
                        'file_id': '-1001768249569_999999_720p',
                        'file_size': 1000000000,
                        'file_name': 'Sample.Movie.2024.720p.mkv',
                        'is_video': True
                    }
                },
                'thumbnail': None
            }
        ]
    
    # Apply pagination
    total = len(sample_results)
    paginated = sample_results[offset:offset + limit]
    
    # Enhance with posters
    if poster_fetcher:
        async with aiohttp.ClientSession() as session:
            for result in paginated:
                try:
                    poster_data = await poster_fetcher.fetch_poster(result['title'])
                    if poster_data and poster_data.get('poster_url'):
                        result['poster_url'] = poster_data['poster_url']
                        result['poster_source'] = poster_data['source']
                        result['poster_rating'] = poster_data.get('rating', '0.0')
                        result['has_poster'] = True
                    else:
                        result['poster_url'] = f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(result['title'])}"
                        result['poster_source'] = 'custom'
                        result['poster_rating'] = '0.0'
                        result['has_poster'] = False
                except:
                    result['poster_url'] = f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(result['title'])}"
                    result['poster_source'] = 'custom'
                    result['poster_rating'] = '0.0'
                    result['has_poster'] = False
    
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
            'channels_searched': len(Config.TEXT_CHANNEL_IDS),
            'channels_found': len(set(r.get('channel_id') for r in paginated if r.get('channel_id'))),
            'query': query
        }
    }
    
    # Cache the result
    if cache_manager and cache_manager.redis_enabled:
        await cache_manager.set(cache_key, json.dumps(result_data, default=str), expire_seconds=3600)
    
    return result_data

async def get_home_movies_api():
    """Get latest movies for homepage"""
    sample_movies = [
        {
            'title': 'Tere Ishk Mein (2025)',
            'date': datetime.now().isoformat(),
            'is_new': True,
            'channel': 'SK4FiLM Main',
            'channel_id': -1001891090100,
            'poster_url': f"{Config.BACKEND_URL}/api/poster?title=Tere+Ishk+Mein&year=2025",
            'poster_source': 'custom',
            'poster_rating': '0.0',
            'has_poster': True
        },
        {
            'title': 'Pushpa 2: The Rule (2024)',
            'date': datetime.now().isoformat(),
            'is_new': True,
            'channel': 'SK4FiLM Updates',
            'channel_id': -1002024811395,
            'poster_url': f"{Config.BACKEND_URL}/api/poster?title=Pushpa+2+The+Rule&year=2024",
            'poster_source': 'custom',
            'poster_rating': '8.5',
            'has_poster': True
        },
        {
            'title': 'Kalki 2898 AD (2024)',
            'date': (datetime.now() - timedelta(days=3)).isoformat(),
            'is_new': False,
            'channel': 'SK4FiLM Main',
            'channel_id': -1001891090100,
            'poster_url': f"{Config.BACKEND_URL}/api/poster?title=Kalki+2898+AD&year=2024",
            'poster_source': 'custom',
            'poster_rating': '8.0',
            'has_poster': True
        },
        {
            'title': 'Singham Again (2024)',
            'date': (datetime.now() - timedelta(days=5)).isoformat(),
            'is_new': False,
            'channel': 'SK4FiLM Updates',
            'channel_id': -1002024811395,
            'poster_url': f"{Config.BACKEND_URL}/api/poster?title=Singham+Again&year=2024",
            'poster_source': 'custom',
            'poster_rating': '7.5',
            'has_poster': True
        }
    ]
    
    # Enhance with posters from poster fetcher
    if poster_fetcher:
        async with aiohttp.ClientSession() as session:
            for movie in sample_movies:
                try:
                    poster_data = await poster_fetcher.fetch_poster(movie['title'])
                    if poster_data and poster_data.get('poster_url'):
                        movie['poster_url'] = poster_data['poster_url']
                        movie['poster_source'] = poster_data['source']
                        movie['poster_rating'] = poster_data.get('rating', '0.0')
                        movie['has_poster'] = True
                except:
                    pass
    
    return sample_movies

async def get_single_post_api(channel_id, message_id):
    """Get single movie/post details"""
    try:
        # Sample post data
        post_data = {
            'title': 'Tere Ishk Mein (2025)',
            'content': '''🎬 <b>Tere Ishk Mein (2025)</b>

📅 <b>Release Date:</b> 2025
🎭 <b>Genre:</b> Romance, Drama
⏱️ <b>Duration:</b> 2h 28m
⭐ <b>IMDb Rating:</b> 8.2/10

<b>🎥 Synopsis:</b>
A beautiful love story that transcends time and distance. Tere Ishk Mein is a heartwarming tale of love, sacrifice, and destiny.

<b>🌟 Star Cast:</b>
• New Romantic Lead
• Beautiful Heroine
• Supporting Cast

<b>🎵 Music:</b>
Original Soundtrack by Famous Music Director

<b>🎬 Director:</b>
Renowned Director

<b>📥 Download Links:</b>
Multiple quality options available below

✅ <b>Verified by SK4FiLM Team</b>
🎁 <b>High Quality Prints</b>
⚡ <b>Fast Download Speeds</b>''',
            'channel': channel_name(channel_id),
            'channel_id': channel_id,
            'message_id': message_id,
            'date': datetime.now().isoformat(),
            'is_new': True,
            'has_file': True,
            'has_post': True,
            'quality_options': {
                '2160p (4K)': {
                    'file_id': f'{channel_id}_{message_id}_2160p',
                    'file_size': 4500000000,
                    'file_name': 'Tere.Ishk.Mein.2025.2160p.4K.HEVC.mkv',
                    'is_video': True,
                    'quality': '2160p'
                },
                '1080p HEVC': {
                    'file_id': f'{channel_id}_{message_id}_1080p',
                    'file_size': 2200000000,
                    'file_name': 'Tere.Ishk.Mein.2025.1080p.HEVC.mkv',
                    'is_video': True,
                    'quality': '1080p'
                },
                '720p': {
                    'file_id': f'{channel_id}_{message_id}_720p',
                    'file_size': 1200000000,
                    'file_name': 'Tere.Ishk.Mein.2025.720p.mkv',
                    'is_video': True,
                    'quality': '720p'
                },
                '480p': {
                    'file_id': f'{channel_id}_{message_id}_480p',
                    'file_size': 800000000,
                    'file_name': 'Tere.Ishk.Mein.2025.480p.mkv',
                    'is_video': True,
                    'quality': '480p'
                }
            },
            'views': 1250,
            'downloads': 890,
            'thumbnail': None,
            'thumbnail_source': 'default'
        }
        
        # Add poster
        if poster_fetcher:
            try:
                poster_data = await poster_fetcher.fetch_poster(post_data['title'])
                if poster_data and poster_data.get('poster_url'):
                    post_data['poster_url'] = poster_data['poster_url']
                    post_data['poster_source'] = poster_data['source']
                    post_data['poster_rating'] = poster_data.get('rating', '0.0')
                else:
                    post_data['poster_url'] = f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(post_data['title'])}&year=2025"
                    post_data['poster_source'] = 'custom'
                    post_data['poster_rating'] = '8.2'
            except:
                post_data['poster_url'] = f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(post_data['title'])}&year=2025"
                post_data['poster_source'] = 'custom'
                post_data['poster_rating'] = '8.2'
        
        return post_data
    except Exception as e:
        logger.error(f"Single post API error: {e}")
        return None

async def verify_user_api(user_id, verification_url=None):
    """Verify user API function"""
    try:
        if verification_system:
            # Create verification link if not provided
            if not verification_url:
                verification_data = await verification_system.create_verification_link(user_id)
                return {
                    'verified': False,
                    'verification_url': verification_data['short_url'],
                    'message': 'Verification link created',
                    'service_name': verification_data['service_name']
                }
            
            # Check if user is already verified
            is_verified, message = await verification_system.check_user_verified(user_id, premium_system)
            if is_verified:
                return {
                    'verified': True,
                    'message': message,
                    'user_id': user_id
                }
            
            return {
                'verified': False,
                'message': 'Please complete verification',
                'verification_url': verification_url
            }
        else:
            # Fallback verification
            return {
                'verified': True,
                'message': 'Verification not required',
                'user_id': user_id
            }
    except Exception as e:
        logger.error(f"Verification API error: {e}")
        return {
            'verified': False,
            'message': f'Error: {str(e)}',
            'user_id': user_id
        }

async def get_index_status_api():
    """Get indexing status"""
    try:
        if db_manager:
            total_files = await db_manager.get_file_count()
            video_files = await db_manager.get_video_file_count()
            thumbnails = await db_manager.get_thumbnail_count()
            last_indexed = await db_manager.get_last_indexed()
            
            last_indexed_time = "Never"
            if last_indexed and last_indexed.get('indexed_at'):
                if isinstance(last_indexed['indexed_at'], datetime):
                    mins_ago = int((datetime.now() - last_indexed['indexed_at']).total_seconds() / 60)
                    last_indexed_time = f"{mins_ago} minutes ago"
                else:
                    last_indexed_time = str(last_indexed['indexed_at'])
            
            return {
                'total_indexed': total_files,
                'video_files': video_files,
                'total_thumbnails': thumbnails,
                'thumbnail_coverage': f"{(thumbnails/video_files*100):.1f}%" if video_files > 0 else "0%",
                'last_indexed': last_indexed_time,
                'bot_status': bot_instance.bot_started if bot_instance else False,
                'user_session': bot_instance.user_session_ready if bot_instance else False,
                'redis_enabled': cache_manager.redis_enabled if cache_manager else False,
                'cache_stats': search_stats
            }
        else:
            return {
                'total_indexed': 0,
                'video_files': 0,
                'total_thumbnails': 0,
                'thumbnail_coverage': "0%",
                'last_indexed': "Never",
                'bot_status': False,
                'user_session': False,
                'redis_enabled': False,
                'cache_stats': search_stats
            }
    except Exception as e:
        logger.error(f"Index status API error: {e}")
        return {
            'error': str(e),
            'status': 'error'
        }

# API Routes
@app.route('/')
async def root():
    """Root endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'SK4FiLM Complete Web API v8.0',
        'version': '8.0',
        'timestamp': datetime.now().isoformat(),
        'endpoints': {
            'health': '/health',
            'movies': '/api/movies',
            'search': '/api/search?query=movie&page=1&limit=12',
            'post': '/api/post?channel=-1001891090100&message=12345',
            'poster': '/api/poster?title=Movie+Name&year=2024',
            'verify_user': '/api/verify_user (POST)',
            'clear_cache': '/api/clear_cache',
            'index_status': '/api/index_status'
        },
        'website': Config.WEBSITE_URL,
        'bot_username': Config.BOT_USERNAME
    })

@app.route('/health')
async def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'service': 'SK4FiLM API',
        'bot_online': bot_instance.bot_started if bot_instance else False,
        'database': db_manager is not None,
        'redis': cache_manager.redis_enabled if cache_manager else False,
        'premium_system': premium_system is not None,
        'verification_system': verification_system is not None
    })

@app.route('/api/movies', methods=['GET'])
async def api_movies():
    """Home page movies"""
    try:
        search_stats['api_requests'] += 1
        movies = await get_home_movies_api()
        
        return jsonify({
            'status': 'success',
            'movies': movies,
            'total': len(movies),
            'timestamp': datetime.now().isoformat(),
            'bot_username': Config.BOT_USERNAME
        })
    except Exception as e:
        logger.error(f"Movies API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/search', methods=['GET'])
async def api_search():
    """Search movies"""
    try:
        # Get query parameters
        query = request.args.get('query', '').strip()
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 12))
        
        # Validate query
        if len(query) < 2:
            return jsonify({
                'status': 'error',
                'message': 'Query must be at least 2 characters'
            }), 400
        
        search_stats['api_requests'] += 1
        
        # Perform search
        result_data = await search_movies_api(query, limit, page)
        
        return jsonify({
            'status': 'success',
            'query': query,
            'results': result_data['results'],
            'pagination': result_data['pagination'],
            'search_metadata': result_data.get('search_metadata', {}),
            'timestamp': datetime.now().isoformat(),
            'bot_username': Config.BOT_USERNAME
        })
        
    except Exception as e:
        logger.error(f"Search API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/post', methods=['GET'])
async def api_post():
    """Single movie page"""
    try:
        # Get query parameters
        channel_id = request.args.get('channel', '').strip()
        message_id = request.args.get('message', '').strip()
        
        if not channel_id or not message_id:
            return jsonify({
                'status': 'error',
                'message': 'Channel and message parameters are required'
            }), 400
        
        try:
            channel_id = int(channel_id)
            message_id = int(message_id)
        except ValueError:
            return jsonify({
                'status': 'error',
                'message': 'Invalid channel or message ID'
            }), 400
        
        search_stats['api_requests'] += 1
        
        # Get single post
        post_data = await get_single_post_api(channel_id, message_id)
        
        if not post_data:
            return jsonify({
                'status': 'error',
                'message': 'Post not found'
            }), 404
        
        return jsonify({
            'status': 'success',
            'post': post_data,
            'timestamp': datetime.now().isoformat(),
            'bot_username': Config.BOT_USERNAME
        })
        
    except Exception as e:
        logger.error(f"Post API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/poster', methods=['GET'])
async def api_poster():
    """Custom poster generator"""
    try:
        title = request.args.get('title', 'Movie').strip()
        year = request.args.get('year', '').strip()
        
        if not title:
            return jsonify({
                'status': 'error',
                'message': 'Title is required'
            }), 400
        
        # Truncate title if too long
        display_title = title[:20] + "..." if len(title) > 20 else title
        
        # Color schemes
        color_schemes = [
            {'bg1': '#667eea', 'bg2': '#764ba2', 'text': '#ffffff'},
            {'bg1': '#f093fb', 'bg2': '#f5576c', 'text': '#ffffff'},
            {'bg1': '#4facfe', 'bg2': '#00f2fe', 'text': '#ffffff'},
            {'bg1': '#43e97b', 'bg2': '#38f9d7', 'text': '#ffffff'},
            {'bg1': '#fa709a', 'bg2': '#fee140', 'text': '#ffffff'},
        ]
        
        # Pick color scheme based on title hash
        scheme_index = hash(title) % len(color_schemes)
        scheme = color_schemes[scheme_index]
        
        text_color = scheme['text']
        bg1_color = scheme['bg1']
        bg2_color = scheme['bg2']
        
        # Year text if provided
        year_text = f'<text x="150" y="305" text-anchor="middle" fill="{text_color}" font-size="14" font-family="Arial">{year}</text>' if year else ''
        
        # Generate SVG
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
            <text x="150" y="280" text-anchor="middle" fill="{text_color}" font-size="16" font-weight="bold" font-family="Arial">{html.escape(display_title)}</text>
            {year_text}
            <rect x="50" y="380" width="200" height="40" rx="20" fill="rgba(0,0,0,0.3)"/>
            <text x="150" y="405" text-anchor="middle" fill="{text_color}" font-size="16" font-weight="bold" font-family="Arial">SK4FiLM</text>
        </svg>'''
        
        # Return as SVG image
        return Response(
            svg,
            mimetype='image/svg+xml',
            headers={
                'Cache-Control': 'public, max-age=86400',
                'Content-Type': 'image/svg+xml'
            }
        )
        
    except Exception as e:
        logger.error(f"Poster API error: {e}")
        # Return simple error SVG
        error_svg = '''<svg width="300" height="450" xmlns="http://www.w3.org/2000/svg">
            <rect width="100%" height="100%" fill="#667eea"/>
            <text x="150" y="225" text-anchor="middle" fill="white" font-size="18" font-family="Arial">SK4FiLM</text>
        </svg>'''
        return Response(error_svg, mimetype='image/svg+xml')

@app.route('/api/verify_user', methods=['POST'])
async def api_verify_user():
    """Verification system"""
    try:
        data = await request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No data provided'
            }), 400
        
        user_id = data.get('user_id')
        verification_url = data.get('verification_url')
        
        if not user_id:
            return jsonify({
                'status': 'error',
                'message': 'User ID is required'
            }), 400
        
        # Verify user
        verification_result = await verify_user_api(user_id, verification_url)
        
        return jsonify({
            'status': 'success',
            **verification_result,
            'timestamp': datetime.now().isoformat(),
            'bot_username': Config.BOT_USERNAME
        })
        
    except Exception as e:
        logger.error(f"Verify user API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/clear_cache', methods=['GET', 'POST'])
async def api_clear_cache():
    """Clear cache"""
    try:
        cleared_count = 0
        
        # Clear Redis cache
        if cache_manager:
            cleared_count = await cache_manager.clear_search_cache()
        
        # Reset search stats
        search_stats['redis_hits'] = 0
        search_stats['redis_misses'] = 0
        search_stats['total_searches'] = 0
        
        return jsonify({
            'status': 'success',
            'message': f'Cache cleared successfully',
            'cleared_keys': cleared_count,
            'cache_stats': search_stats,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Clear cache API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/index_status', methods=['GET'])
async def api_index_status():
    """Index report"""
    try:
        index_status = await get_index_status_api()
        
        return jsonify({
            'status': 'success',
            **index_status,
            'timestamp': datetime.now().isoformat(),
            'system': {
                'premium_system': premium_system is not None,
                'verification_system': verification_system is not None,
                'cache_system': cache_manager is not None,
                'poster_fetcher': poster_fetcher is not None
            }
        })
        
    except Exception as e:
        logger.error(f"Index status API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# Additional useful endpoints
@app.route('/api/stats', methods=['GET'])
async def api_stats():
    """Get system statistics"""
    try:
        total_premium = 0
        active_premium = 0
        total_revenue = 0
        
        if premium_system:
            admin_stats = await premium_system.get_admin_stats()
            total_premium = admin_stats.get('total_premium_users', 0)
            active_premium = admin_stats.get('active_premium_users', 0)
            total_revenue = admin_stats.get('total_revenue', 0)
        
        return jsonify({
            'status': 'success',
            'stats': {
                'premium': {
                    'total_users': total_premium,
                    'active_users': active_premium,
                    'total_revenue': total_revenue
                },
                'search': search_stats,
                'api_requests': search_stats['api_requests'],
                'timestamp': datetime.now().isoformat()
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/premium/plans', methods=['GET'])
async def api_premium_plans():
    """Get premium plans"""
    try:
        if premium_system:
            plans = await premium_system.get_all_plans()
        else:
            # Fallback plans
            plans = [
                {
                    'tier': 'basic',
                    'name': 'Basic Plan',
                    'icon': '🥉',
                    'price': 10,
                    'duration_days': 15,
                    'features': ['✅ All Quality (480p-4K)', '✅ Unlimited Downloads', '✅ No Verification Needed'],
                    'description': 'Perfect starter plan - Unlimited access for 15 days',
                    'color_code': '#4CAF50',
                    'per_day_cost': 0.67
                },
                {
                    'tier': 'premium',
                    'name': 'Premium Plan',
                    'icon': '🥈',
                    'price': 25,
                    'duration_days': 30,
                    'features': ['✅ All Quality (480p-4K)', '✅ Unlimited Downloads', '✅ No Verification Needed', '✅ Priority Support'],
                    'description': 'Best value - Unlimited access for 30 days',
                    'color_code': '#2196F3',
                    'per_day_cost': 0.83
                }
            ]
        
        return jsonify({
            'status': 'success',
            'plans': plans,
            'currency': 'INR',
            'bot_username': Config.BOT_USERNAME
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

async def main():
    """Main function to start the system"""
    global bot_instance, db_manager, premium_system, verification_system, cache_manager, poster_fetcher
    
    try:
        # Initialize configuration
        config = Config()
        config.validate()
        
        logger.info("🚀 Starting SK4FiLM Complete Web API System...")
        
        # Initialize database
        if config.MONGODB_URI:
            db_manager = DatabaseManager(config.MONGODB_URI)
            await db_manager.connect()
        
        # Initialize cache
        cache_manager = CacheManager()
        await cache_manager.init_redis()
        
        # Initialize other systems
        try:
            from verification import VerificationSystem
            from premium import PremiumSystem
            from poster_fetching import PosterFetcher
            
            verification_system = VerificationSystem(config, db_manager)
            premium_system = PremiumSystem(config, db_manager)
            poster_fetcher = PosterFetcher(config)
            
            logger.info("✅ All systems initialized")
        except Exception as e:
            logger.error(f"System initialization error: {e}")
        
        # Initialize bot if needed
        try:
            from bot_handlers import SK4FiLMBot, setup_bot_handlers
            
            bot_instance = SK4FiLMBot(config, db_manager)
            bot_instance.verification_system = verification_system
            bot_instance.premium_system = premium_system
            bot_instance.poster_fetcher = poster_fetcher
            bot_instance.cache_manager = cache_manager
            
            await bot_instance.initialize()
            await setup_bot_handlers(bot_instance.bot, bot_instance)
            
            logger.info("✅ Bot initialized successfully")
        except Exception as e:
            logger.error(f"Bot initialization error: {e}")
            bot_instance = None
        
        # Start web server
        hypercorn_config = HyperConfig()
        hypercorn_config.bind = [f"0.0.0.0:{config.WEB_SERVER_PORT}"]
        hypercorn_config.workers = 1
        
        logger.info("="*60)
        logger.info("🎬 SK4FiLM COMPLETE WEB API SYSTEM")
        logger.info(f"🌐 Server: http://localhost:{config.WEB_SERVER_PORT}")
        logger.info(f"🤖 Bot: @{config.BOT_USERNAME}")
        logger.info(f"🔗 Website: {config.WEBSITE_URL}")
        logger.info("="*60)
        logger.info("✅ API Endpoints:")
        logger.info("  GET  /health          - Health check")
        logger.info("  GET  /api/movies      - Home page movies")
        logger.info("  GET  /api/search      - Search movies")
        logger.info("  GET  /api/post        - Single movie page")
        logger.info("  GET  /api/poster      - Custom poster generator")
        logger.info("  POST /api/verify_user - Verification system")
        logger.info("  GET  /api/clear_cache - Clear cache")
        logger.info("  GET  /api/index_status- Index report")
        logger.info("="*60)
        
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
        logger.info("System stopped")

if __name__ == "__main__":
    asyncio.run(main())
