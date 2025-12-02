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
import base64
import secrets
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from io import BytesIO
from functools import wraps
from collections import defaultdict

import aiohttp
import urllib.parse
from quart import Quart, jsonify, request, Response
from hypercorn.asyncio import serve
from hypercorn.config import Config as HyperConfig
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, ValidationError
from bson import ObjectId
import redis.asyncio as redis

# Import bot handlers
from bot_handlers import setup_bot_handlers, SK4FiLMBot

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

# Pydantic Models for Validation
class SearchQuery(BaseModel):
    query: str
    page: int = 1
    limit: int = 12
    user_id: Optional[int] = None

class PremiumActivation(BaseModel):
    user_id: int
    tier: str
    duration_days: Optional[int] = 30

class BroadcastMessage(BaseModel):
    message: str
    user_type: str = 'premium'  # 'all', 'premium', 'free'

# Custom Exceptions
class BotError(Exception):
    """Base exception for bot errors"""
    pass

class SearchError(BotError):
    """Search-related errors"""
    pass

class DatabaseError(BotError):
    """Database-related errors"""
    pass

class RateLimitError(BotError):
    """Rate limit exceeded"""
    pass

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
            
            # Create indexes
            await self.files_col.create_index([('normalized_title', 'text')])
            await self.files_col.create_index([('channel_id', 1), ('message_id', 1)], unique=True)
            await self.files_col.create_index([('date', -1)])
            await self.files_col.create_index([('is_video_file', 1)])
            
            await self.premium_col.create_index([('user_id', 1)], unique=True)
            await self.premium_col.create_index([('expires_at', 1)])
            
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

# Task Manager
class TaskManager:
    def __init__(self):
        self.tasks = {}
        self.running = False
    
    async def start_task(self, name, coro_func, interval=3600):
        """Start a background task"""
        if name in self.tasks:
            self.tasks[name].cancel()
        
        async def task_wrapper():
            while self.running:
                try:
                    await coro_func()
                    await asyncio.sleep(interval)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Task {name} error: {e}")
                    await asyncio.sleep(60)  # Wait before retry
        
        task = asyncio.create_task(task_wrapper(), name=f"bg_{name}")
        self.tasks[name] = task
        logger.info(f"✅ Started background task: {name}")
    
    async def stop_all(self):
        """Stop all background tasks"""
        self.running = False
        for name, task in self.tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self.tasks.clear()

# Decorator for error handling
def handle_errors(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            raise BotError(f"{func.__name__} failed: {str(e)}")
    return wrapper

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

# Global instances
bot_instance = None
db_manager = None

# Rate limiters
api_rate_limiter = RateLimiter(max_requests=100, window_seconds=300)
search_rate_limiter = RateLimiter(max_requests=30, window_seconds=60)

# Enhanced search statistics
search_stats = {
    'redis_hits': 0,
    'redis_misses': 0,
    'multi_channel_searches': 0,
    'total_searches': 0
}

# Channel configuration
CHANNEL_CONFIG = {
    -1001891090100: {'name': 'SK4FiLM Main', 'type': 'text', 'search_priority': 1},
    -1002024811395: {'name': 'SK4FiLM Updates', 'type': 'text', 'search_priority': 2},
    -1001768249569: {'name': 'SK4FiLM Files', 'type': 'file', 'search_priority': 0}
}

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

# Flood wait protection
class FloodWaitProtection:
    def __init__(self):
        self.last_request_time = 0
        self.min_interval = 3
        self.request_count = 0
        self.reset_time = time.time()
    
    async def wait_if_needed(self):
        current_time = time.time()
        
        # Reset counter every 2 minutes
        if current_time - self.reset_time > 120:
            self.request_count = 0
            self.reset_time = current_time
        
        # Limit to 20 requests per 2 minutes
        if self.request_count >= 20:
            wait_time = 120 - (current_time - self.reset_time)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                self.request_count = 0
                self.reset_time = time.time()
        
        # Ensure minimum interval between requests
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_interval:
            await asyncio.sleep(self.min_interval - time_since_last)
        
        self.last_request_time = time.time()
        self.request_count += 1

flood_protection = FloodWaitProtection()

async def safe_telegram_operation(operation, *args, **kwargs):
    """Safely execute Telegram operations with flood wait protection"""
    max_retries = 2
    base_delay = 5
    
    for attempt in range(max_retries):
        try:
            await flood_protection.wait_if_needed()
            result = await operation(*args, **kwargs)
            return result
        except Exception as e:
            logger.error(f"Telegram operation failed: {e}")
            if attempt == max_retries - 1:
                raise e
            await asyncio.sleep(base_delay * (2 ** attempt))
    return None

async def safe_telegram_generator(operation, *args, limit=None, **kwargs):
    """Safely iterate over Telegram async generators"""
    max_retries = 2
    count = 0
    
    for attempt in range(max_retries):
        try:
            await flood_protection.wait_if_needed()
            async for item in operation(*args, **kwargs):
                yield item
                count += 1
                
                if count % 10 == 0:
                    await asyncio.sleep(1)
                    
                if limit and count >= limit:
                    break
            break
        except Exception as e:
            logger.error(f"Telegram generator failed: {e}")
            if attempt == max_retries - 1:
                raise e
            await asyncio.sleep(5 * (2 ** attempt))

# ==============================
# ENHANCED SEARCH FUNCTIONS
# ==============================

async def get_live_posts_multi_channel(limit_per_channel=10):
    """Get posts from multiple channels concurrently"""
    if not bot_instance or not bot_instance.user_session_ready:
        return []
    
    all_posts = []
    
    async def fetch_channel_posts(channel_id):
        posts = []
        try:
            async for msg in safe_telegram_generator(bot_instance.user_client.get_chat_history, channel_id, limit=limit_per_channel):
                if msg and msg.text and len(msg.text) > 15:
                    title = extract_title_smart(msg.text)
                    if title:
                        posts.append({
                            'title': title,
                            'normalized_title': normalize_title(title),
                            'content': msg.text,
                            'channel_name': channel_name(channel_id),
                            'channel_id': channel_id,
                            'message_id': msg.id,
                            'date': msg.date,
                            'is_new': is_new(msg.date) if msg.date else False
                        })
        except Exception as e:
            logger.error(f"Error getting posts from channel {channel_id}: {e}")
        return posts
    
    # Fetch from all text channels concurrently
    tasks = [fetch_channel_posts(channel_id) for channel_id in Config.TEXT_CHANNEL_IDS]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for result in results:
        if isinstance(result, list):
            all_posts.extend(result)
    
    # Sort by date (newest first) and remove duplicates
    seen_titles = set()
    unique_posts = []
    
    for post in sorted(all_posts, key=lambda x: x.get('date', datetime.min), reverse=True):
        if post['normalized_title'] not in seen_titles:
            seen_titles.add(post['normalized_title'])
            unique_posts.append(post)
    
    return unique_posts[:20]  # Return top 20 unique posts

@handle_errors
async def search_movies_multi_channel(query, limit=12, page=1):
    """Search across multiple channels with enhanced results"""
    offset = (page - 1) * limit
    search_stats['total_searches'] += 1
    search_stats['multi_channel_searches'] += 1
    
    # Try Redis cache first
    cache_key = f"search:enhanced:{query}:{page}:{limit}"
    if bot_instance and bot_instance.cache_manager and bot_instance.cache_manager.redis_enabled:
        cached_data = await bot_instance.cache_manager.get(cache_key)
        if cached_data:
            try:
                search_stats['redis_hits'] += 1
                logger.info(f"✅ Redis cache HIT for: {query}")
                return json.loads(cached_data)
            except Exception as e:
                logger.warning(f"Redis cache parse error: {e}")
    
    search_stats['redis_misses'] += 1
    logger.info(f"🔍 Multi-channel search for: {query}")
    
    query_lower = query.lower()
    posts_dict = {}
    files_dict = {}
    
    # Use MongoDB search primarily to avoid Telegram API calls
    try:
        if db_manager is not None and db_manager.files_col is not None:
            # Try text search first
            try:
                cursor = db_manager.files_col.find({'$text': {'$search': query}})
                async for doc in cursor:
                    try:
                        norm_title = doc.get('normalized_title', normalize_title(doc['title']))
                        quality = doc['quality']
                        
                        if norm_title not in files_dict:
                            file_name = doc.get('file_name', '').lower()
                            file_is_video = is_video_file(file_name)
                            
                            thumbnail_url = None
                            if file_is_video:
                                thumbnail_url = doc.get('thumbnail')
                            
                            files_dict[norm_title] = {
                                'title': doc['title'], 
                                'quality_options': {}, 
                                'date': doc['date'].isoformat() if isinstance(doc['date'], datetime) else doc['date'],
                                'thumbnail': thumbnail_url,
                                'is_video_file': file_is_video,
                                'thumbnail_source': doc.get('thumbnail_source', 'unknown'),
                                'channel_id': doc.get('channel_id'),
                                'channel_name': channel_name(doc.get('channel_id'))
                            }
                        
                        if quality not in files_dict[norm_title]['quality_options']:
                            files_dict[norm_title]['quality_options'][quality] = {
                                'file_id': f"{doc.get('channel_id', Config.FILE_CHANNEL_ID)}_{doc.get('message_id')}_{quality}",
                                'file_size': doc['file_size'],
                                'file_name': doc['file_name'],
                                'is_video': file_is_video,
                                'channel_id': doc.get('channel_id'),
                                'message_id': doc.get('message_id')
                            }
                    except Exception as e:
                        logger.error(f"Error processing search result: {e}")
                        continue
            except Exception as e:
                logger.error(f"Text search error: {e}")
            
            # If no results from text search, try regex search
            if not files_dict:
                regex_cursor = db_manager.files_col.find({
                    '$or': [
                        {'title': {'$regex': query, '$options': 'i'}},
                        {'normalized_title': {'$regex': query, '$options': 'i'}}
                    ]
                })
                
                async for doc in regex_cursor:
                    try:
                        norm_title = doc.get('normalized_title', normalize_title(doc['title']))
                        quality = doc['quality']
                        
                        if norm_title not in files_dict:
                            file_name = doc.get('file_name', '').lower()
                            file_is_video = is_video_file(file_name)
                            
                            thumbnail_url = None
                            if file_is_video:
                                thumbnail_url = doc.get('thumbnail')
                            
                            files_dict[norm_title] = {
                                'title': doc['title'], 
                                'quality_options': {}, 
                                'date': doc['date'].isoformat() if isinstance(doc['date'], datetime) else doc['date'],
                                'thumbnail': thumbnail_url,
                                'is_video_file': file_is_video,
                                'thumbnail_source': doc.get('thumbnail_source', 'unknown'),
                                'channel_id': doc.get('channel_id'),
                                'channel_name': channel_name(doc.get('channel_id'))
                            }
                        
                        if quality not in files_dict[norm_title]['quality_options']:
                            files_dict[norm_title]['quality_options'][quality] = {
                                'file_id': f"{doc.get('channel_id', Config.FILE_CHANNEL_ID)}_{doc.get('message_id')}_{quality}",
                                'file_size': doc['file_size'],
                                'file_name': doc['file_name'],
                                'is_video': file_is_video,
                                'channel_id': doc.get('channel_id'),
                                'message_id': doc.get('message_id')
                            }
                    except Exception as e:
                        logger.error(f"Error processing regex search result: {e}")
                        continue
    except Exception as e:
        logger.error(f"File search error: {e}")
    
    # Search Telegram channels if user session is ready
    if bot_instance and bot_instance.user_session_ready and bot_instance.user_client:
        channel_results = {}
        
        async def search_single_channel(channel_id):
            channel_posts = {}
            try:
                cname = channel_name(channel_id)
                async for msg in safe_telegram_generator(bot_instance.user_client.search_messages, channel_id, query=query, limit=20):
                    if msg and msg.text and len(msg.text) > 15:
                        title = extract_title_smart(msg.text)
                        if title and query_lower in title.lower():
                            norm_title = normalize_title(title)
                            if norm_title not in channel_posts:
                                channel_posts[norm_title] = {
                                    'title': title,
                                    'content': format_post(msg.text),
                                    'channel': cname,
                                    'channel_id': channel_id,
                                    'message_id': msg.id,
                                    'date': msg.date.isoformat() if isinstance(msg.date, datetime) else msg.date,
                                    'is_new': is_new(msg.date) if msg.date else False,
                                    'has_file': False,
                                    'has_post': True,
                                    'quality_options': {},
                                    'thumbnail': None
                                }
            except Exception as e:
                logger.error(f"Telegram search error in {channel_id}: {e}")
            return channel_posts
        
        # Search all text channels concurrently
        tasks = [search_single_channel(channel_id) for channel_id in Config.TEXT_CHANNEL_IDS]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Merge results from all channels
        for result in results:
            if isinstance(result, dict):
                posts_dict.update(result)
    
    # Merge posts and files data
    merged = {}
    
    # First add all posts
    for norm_title, post_data in posts_dict.items():
        merged[norm_title] = post_data
    
    # Then add/update with file information
    for norm_title, file_data in files_dict.items():
        if norm_title in merged:
            # Update existing post with file info
            merged[norm_title]['has_file'] = True
            merged[norm_title]['quality_options'] = file_data['quality_options']
            if file_data.get('is_video_file') and file_data.get('thumbnail'):
                merged[norm_title]['thumbnail'] = file_data['thumbnail']
                merged[norm_title]['thumbnail_source'] = file_data.get('thumbnail_source', 'unknown')
        else:
            # Create new entry for files without posts
            merged[norm_title] = {
                'title': file_data['title'],
                'content': f"<p>{file_data['title']}</p>",
                'channel': file_data.get('channel_name', 'SK4FiLM Files'),
                'date': file_data['date'],
                'is_new': False,
                'has_file': True,
                'has_post': False,
                'quality_options': file_data['quality_options'],
                'thumbnail': file_data.get('thumbnail') if file_data.get('is_video_file') else None,
                'thumbnail_source': file_data.get('thumbnail_source', 'unknown')
            }
    
    # Sort results: new posts first, then files, then by date
    results_list = list(merged.values())
    results_list.sort(key=lambda x: (
        not x.get('is_new', False),  # New posts first
        not x['has_file'],           # Files before posts only
        x['date']                    # Recent first
    ), reverse=True)
    
    total = len(results_list)
    paginated = results_list[offset:offset + limit]
    
    # Get posters for results
    if bot_instance and bot_instance.poster_fetcher and paginated:
        titles = [result['title'] for result in paginated]
        posters = await bot_instance.poster_fetcher.fetch_batch_posters(titles)
        
        for result in paginated:
            if result['title'] in posters:
                result['poster'] = posters[result['title']]
            else:
                result['poster'] = {
                    'poster_url': f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(result['title'])}",
                    'source': 'custom',
                    'rating': '0.0'
                }
    
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
            'query': query,
            'search_mode': 'multi_channel_enhanced',
            'cache_status': 'miss'
        }
    }
    
    # Cache in Redis with 1 hour expiration
    if bot_instance and bot_instance.cache_manager and bot_instance.cache_manager.redis_enabled:
        await bot_instance.cache_manager.set(cache_key, json.dumps(result_data), expire_seconds=3600)
    
    logger.info(f"✅ Multi-channel search completed: {len(paginated)} results from {len(set(r.get('channel_id') for r in paginated if r.get('channel_id')))} channels")
    
    return result_data

async def get_home_movies_live():
    """Get latest movies from multiple channels"""
    posts = await get_live_posts_multi_channel(limit_per_channel=15)
    
    movies = []
    seen = set()
    
    for post in posts:
        tk = post['title'].lower().strip()
        if tk not in seen:
            seen.add(tk)
            movies.append({
                'title': post['title'],
                'date': post['date'].isoformat() if isinstance(post['date'], datetime) else post['date'],
                'is_new': post.get('is_new', False),
                'channel': post.get('channel_name', 'SK4FiLM'),
                'channel_id': post.get('channel_id')
            })
            if len(movies) >= 20:
                break
    
    # Get posters
    if movies and bot_instance and bot_instance.poster_fetcher:
        titles = [movie['title'] for movie in movies]
        posters = await bot_instance.poster_fetcher.fetch_batch_posters(titles)
        
        for movie in movies:
            if movie['title'] in posters:
                movie.update(posters[movie['title']])
            else:
                movie.update({
                    'poster_url': f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(movie['title'])}",
                    'source': 'custom',
                    'rating': '0.0'
                })
    
    return movies

async def index_files_background():
    """Background file indexing"""
    if not bot_instance or not bot_instance.user_session_ready or not bot_instance.user_client:
        return
    
    try:
        logger.info("🔄 Starting manual background indexing...")
        
        total_indexed = 0
        async for message in safe_telegram_generator(
            bot_instance.user_client.get_chat_history,
            Config.FILE_CHANNEL_ID,
            limit=200
        ):
            if message and (message.document or message.video):
                await index_single_file(message)
                total_indexed += 1
        
        logger.info(f"✅ Manual indexing completed: {total_indexed} files indexed")
        
    except Exception as e:
        logger.error(f"Manual indexing error: {e}")

async def index_single_file(message):
    """Index a single file message"""
    title = extract_title_from_file(message)
    if not title:
        return
    
    file_id = message.document.file_id if message.document else message.video.file_id
    file_size = message.document.file_size if message.document else (message.video.file_size if message.video else 0)
    file_name = message.document.file_name if message.document else (message.video.file_name if message.video else 'video.mp4')
    quality = detect_quality(file_name)
    
    file_data = {
        'channel_id': Config.FILE_CHANNEL_ID,
        'message_id': message.id,
        'title': title,
        'normalized_title': normalize_title(title),
        'file_id': file_id,
        'quality': quality,
        'file_size': file_size,
        'file_name': file_name,
        'caption': message.caption or '',
        'date': message.date,
        'indexed_at': datetime.now(),
        'is_video_file': is_video_file(file_name)
    }
    
    # Try to extract thumbnail for video files
    if is_video_file(file_name) and hasattr(message, 'video') and message.video:
        try:
            # In a real implementation, you might extract thumbnail here
            # For now, we'll use a placeholder
            file_data['thumbnail'] = f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(title)}"
            file_data['thumbnail_source'] = 'auto_generated'
        except Exception as e:
            logger.error(f"Thumbnail extraction error: {e}")
    
    if db_manager is not None and db_manager.files_col is not None:
        await db_manager.files_col.update_one(
            {'channel_id': Config.FILE_CHANNEL_ID, 'message_id': message.id},
            {'$set': file_data},
            upsert=True
        )

async def auto_delete_file(message, delay_seconds):
    """Auto-delete file after specified delay"""
    await asyncio.sleep(delay_seconds)
    try:
        await message.delete()
        logger.info(f"Auto-deleted file for user {message.chat.id}")
    except Exception as e:
        logger.error(f"Failed to auto-delete file: {e}")

# API Routes
@app.route('/')
async def root():
    """Root endpoint with system status"""
    mongodb_count = 0
    if db_manager is not None and db_manager.files_col is not None:
        try:
            mongodb_count = await db_manager.files_col.count_documents({})
        except:
            pass
    
    bot_status = {
        'started': bot_instance.bot_started if bot_instance else False,
        'username': Config.BOT_USERNAME
    }
    
    user_session_status = bot_instance.user_session_ready if bot_instance else False
    
    redis_enabled = bot_instance.cache_manager.redis_enabled if bot_instance and bot_instance.cache_manager else False
    
    return jsonify({
        'status': 'healthy',
        'service': 'SK4FiLM v8.0 - Complete System with Enhanced Search',
        'version': '8.0',
        'timestamp': datetime.now().isoformat(),
        'modules': {
            'verification': bot_instance.verification_system is not None if bot_instance else False,
            'premium': bot_instance.premium_system is not None if bot_instance else False,
            'poster_fetcher': bot_instance.poster_fetcher is not None if bot_instance else False,
            'cache': bot_instance.cache_manager is not None if bot_instance else False
        },
        'bot': bot_status,
        'user_session': 'ready' if user_session_status else 'pending',
        'database': {
            'mongodb_connected': db_manager is not None,
            'total_files': mongodb_count
        },
        'channels': {
            'text_channels': len(Config.TEXT_CHANNEL_IDS),
            'file_channel': Config.FILE_CHANNEL_ID
        },
        'enhanced_search': {
            'enabled': True,
            'multi_channel': True,
            'redis_caching': redis_enabled,
            'search_stats': search_stats
        },
        'endpoints': {
            'api': '/api/*',
            'health': '/health',
            'verify': '/api/verify/{user_id}',
            'premium': '/api/premium/*',
            'poster': '/api/poster/{title}',
            'cache': '/api/cache/*',
            'search': '/api/search',
            'enhanced_search': '/api/search/enhanced',
            'search_stats': '/api/search/stats'
        }
    })

@app.route('/health')
async def health():
    """Health check endpoint"""
    mongodb_status = False
    if db_manager is not None and db_manager.client is not None:
        try:
            await db_manager.client.admin.command('ping')
            mongodb_status = True
        except:
            pass
    
    redis_status = False
    if bot_instance and bot_instance.cache_manager and bot_instance.cache_manager.redis_enabled and bot_instance.cache_manager.client:
        try:
            await bot_instance.cache_manager.client.ping()
            redis_status = True
        except:
            pass
    
    bot_started = bot_instance.bot_started if bot_instance else False
    user_session_ready = bot_instance.user_session_ready if bot_instance else False
    
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'services': {
            'bot': {
                'started': bot_started,
                'status': 'online' if bot_started else 'starting'
            },
            'user_session': user_session_ready,
            'mongodb': mongodb_status,
            'redis': redis_status,
            'web_server': True
        },
        'modules': {
            'verification': bot_instance.verification_system is not None if bot_instance else False,
            'premium': bot_instance.premium_system is not None if bot_instance else False,
            'poster_fetcher': bot_instance.poster_fetcher is not None if bot_instance else False,
            'cache': bot_instance.cache_manager is not None if bot_instance else False
        },
        'enhanced_search': {
            'enabled': True,
            'channels_ready': len(Config.TEXT_CHANNEL_IDS),
            'redis_ready': bot_instance.cache_manager.redis_enabled if bot_instance and bot_instance.cache_manager else False
        }
    })

@app.route('/metrics')
async def metrics():
    """Prometheus metrics endpoint"""
    metrics_data = [
        '# HELP sk4film_search_total Total search requests',
        '# TYPE sk4film_search_total counter',
        f'sk4film_search_total {search_stats["total_searches"]}',
        
        '# HELP sk4film_redis_hits_total Total Redis cache hits',
        '# TYPE sk4film_redis_hits_total counter',
        f'sk4film_redis_hits_total {search_stats["redis_hits"]}',
        
        '# HELP sk4film_redis_misses_total Total Redis cache misses',
        '# TYPE sk4film_redis_misses_total counter',
        f'sk4film_redis_misses_total {search_stats["redis_misses"]}',
        
        '# HELP sk4film_bot_online Bot online status',
        '# TYPE sk4film_bot_online gauge',
        f'sk4film_bot_online {1 if bot_instance and bot_instance.bot_started else 0}',
        
        '# HELP sk4film_user_session_online User session online status',
        '# TYPE sk4film_user_session_online gauge',
        f'sk4film_user_session_online {1 if bot_instance and bot_instance.user_session_ready else 0}',
    ]
    
    return Response('\n'.join(metrics_data), mimetype='text/plain')

@app.route('/api/search')
async def api_search():
    """Enhanced search for movies with multi-channel support"""
    try:
        # Rate limiting
        ip = request.remote_addr
        if not await api_rate_limiter.is_allowed(ip):
            return jsonify({
                'status': 'error',
                'message': 'Rate limit exceeded. Please try again later.'
            }), 429
        
        # Validate input
        search_data = SearchQuery(
            query=request.args.get('query', '').strip(),
            page=int(request.args.get('page', 1)),
            limit=int(request.args.get('limit', 12)),
            user_id=request.args.get('user_id', type=int)
        )
        
        # Validate query length
        if len(search_data.query) < 2:
            return jsonify({'status': 'error', 'message': 'Query too short'}), 400
        
        if len(search_data.query) > 100:
            return jsonify({'status': 'error', 'message': 'Query too long'}), 400
        
        # Check user access
        if search_data.user_id and bot_instance:
            # Check if premium user (bypasses verification)
            if bot_instance.premium_system is not None:
                is_premium = await bot_instance.premium_system.is_premium_user(search_data.user_id)
                if is_premium:
                    # Premium user, allow access
                    pass
                elif Config.VERIFICATION_REQUIRED and bot_instance.verification_system is not None:
                    # Check verification for non-premium users
                    is_verified, message = await bot_instance.verification_system.check_user_verified(search_data.user_id)
                    if not is_verified:
                        return jsonify({
                            'status': 'verification_required',
                            'message': 'User verification required',
                            'verification_url': f'/api/verify/{search_data.user_id}'
                        }), 403
        
        # Use enhanced search if available and requested
        search_mode = request.args.get('mode', 'enhanced')
        if search_mode == 'enhanced' and bot_instance and bot_instance.user_session_ready:
            try:
                result = await search_movies_multi_channel(search_data.query, search_data.limit, search_data.page)
                return jsonify({
                    'status': 'success',
                    'query': search_data.query,
                    'results': result['results'],
                    'pagination': result['pagination'],
                    'search_metadata': result.get('search_metadata', {}),
                    'bot_username': Config.BOT_USERNAME,
                    'search_mode': 'enhanced_multi_channel'
                })
            except Exception as e:
                logger.error(f"Enhanced search failed: {e}, falling back to basic search")
                # Fall back to basic search
        
        # Basic search (existing functionality)
        cache_key = f"search:basic:{search_data.query}:{search_data.page}:{search_data.limit}"
        if bot_instance and bot_instance.cache_manager:
            cached = await bot_instance.cache_manager.get(cache_key)
            if cached:
                return jsonify(json.loads(cached))
        
        results = []
        if db_manager is not None and db_manager.files_col is not None:
            # Search in MongoDB
            cursor = db_manager.files_col.find({
                '$or': [
                    {'title': {'$regex': search_data.query, '$options': 'i'}},
                    {'normalized_title': {'$regex': search_data.query, '$options': 'i'}}
                ]
            }).limit(search_data.limit).skip((search_data.page - 1) * search_data.limit)
            
            async for doc in cursor:
                results.append({
                    'title': doc.get('title'),
                    'normalized_title': doc.get('normalized_title'),
                    'quality': doc.get('quality', '480p'),
                    'file_size': doc.get('file_size'),
                    'file_name': doc.get('file_name'),
                    'date': doc.get('date'),
                    'channel_id': doc.get('channel_id'),
                    'message_id': doc.get('message_id'),
                    'file_id': doc.get('file_id'),
                    'is_new': is_new(doc.get('date')) if doc.get('date') else False,
                    'is_video_file': doc.get('is_video_file', False),
                    'thumbnail': doc.get('thumbnail'),
                    'thumbnail_source': doc.get('thumbnail_source', 'unknown')
                })
        
        total = len(results)
        response_data = {
            'status': 'success',
            'query': search_data.query,
            'results': results,
            'pagination': {
                'current_page': search_data.page,
                'total_pages': math.ceil(total / search_data.limit) if total > 0 else 1,
                'total_results': total,
                'per_page': search_data.limit,
                'has_next': search_data.page < math.ceil(total / search_data.limit) if total > 0 else False,
                'has_previous': search_data.page > 1
            },
            'search_mode': 'basic'
        }
        
        # Cache the results
        if bot_instance and bot_instance.cache_manager:
            await bot_instance.cache_manager.set(cache_key, json.dumps(response_data), expire_seconds=1800)
        
        return jsonify(response_data)
        
    except ValidationError as e:
        return jsonify({'status': 'error', 'message': 'Invalid parameters'}), 400
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/search/enhanced')
async def api_enhanced_search():
    """Force enhanced multi-channel search"""
    try:
        # Rate limiting
        ip = request.remote_addr
        if not await search_rate_limiter.is_allowed(ip):
            return jsonify({
                'status': 'error',
                'message': 'Search rate limit exceeded. Please try again later.'
            }), 429
        
        q = request.args.get('query', '').strip()
        p = int(request.args.get('page', 1))
        l = int(request.args.get('limit', 12))
        
        if not q:
            return jsonify({'status': 'error', 'message': 'Query required'}), 400
        
        if not bot_instance or not bot_instance.user_session_ready:
            return jsonify({
                'status': 'error',
                'message': 'Enhanced search requires user session. Try basic search.',
                'basic_search_url': f'/api/search?query={urllib.parse.quote(q)}&page={p}&limit={l}'
            }), 503
        
        result = await search_movies_multi_channel(q, l, p)
        return jsonify({
            'status': 'success',
            'query': q,
            'results': result['results'],
            'pagination': result['pagination'],
            'search_metadata': result.get('search_metadata', {}),
            'bot_username': Config.BOT_USERNAME,
            'search_mode': 'enhanced_multi_channel_forced'
        })
        
    except Exception as e:
        logger.error(f"Enhanced search error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/search/stats')
async def api_search_stats():
    """Get search statistics"""
    try:
        hit_rate = 0
        if search_stats['redis_hits'] + search_stats['redis_misses'] > 0:
            hit_rate = (search_stats['redis_hits'] / (search_stats['redis_hits'] + search_stats['redis_misses'])) * 100
        
        redis_enabled = bot_instance.cache_manager.redis_enabled if bot_instance and bot_instance.cache_manager else False
        user_session_ready = bot_instance.user_session_ready if bot_instance else False
        
        return jsonify({
            'status': 'success',
            'stats': {
                'total_searches': search_stats['total_searches'],
                'multi_channel_searches': search_stats['multi_channel_searches'],
                'redis_hits': search_stats['redis_hits'],
                'redis_misses': search_stats['redis_misses'],
                'redis_hit_rate': f"{hit_rate:.1f}%",
                'enhanced_search_available': user_session_ready,
                'channels_active': len(Config.TEXT_CHANNEL_IDS),
                'redis_enabled': redis_enabled
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/search/clear_cache', methods=['POST'])
async def api_clear_search_cache():
    """Clear search cache"""
    try:
        # Check admin key
        data = await request.get_json() or {}
        admin_key = data.get('admin_key') or request.args.get('admin_key')
        
        if admin_key != Config.ADMIN_API_KEY:
            return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401
        
        cleared = 0
        if bot_instance and bot_instance.cache_manager:
            cleared = await bot_instance.cache_manager.clear_pattern("search:*")
            
            # Reset search stats
            global search_stats
            search_stats = {
                'redis_hits': 0,
                'redis_misses': 0,
                'multi_channel_searches': 0,
                'total_searches': 0
            }
        
        return jsonify({
            'status': 'success',
            'message': f'Search cache cleared ({cleared} keys)',
            'cleared_keys': cleared,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/movies')
async def api_movies():
    """Get latest movies with enhanced multi-channel support"""
    try:
        if not bot_instance or not bot_instance.bot_started:
            return jsonify({'status': 'error', 'message': 'Starting...'}), 503
        
        # Use enhanced function to get movies
        movies = await get_home_movies_live()
        
        if not movies and db_manager is not None and db_manager.files_col is not None:
            # Fallback to database if no live posts
            return await api_movies_basic()
        
        return jsonify({
            'status': 'success', 
            'movies': movies, 
            'total': len(movies), 
            'bot_username': Config.BOT_USERNAME,
            'mode': 'multi_channel_enhanced',
            'channels_searched': len(Config.TEXT_CHANNEL_IDS),
            'live_posts': True
        })
    except Exception as e:
        logger.error(f"Movies error: {e}")
        # Fall back to basic approach
        return await api_movies_basic()

async def api_movies_basic():
    """Basic movies API fallback"""
    try:
        if db_manager is None or db_manager.files_col is None:
            return jsonify({'status': 'error', 'message': 'Database not ready'}), 503
        
        cursor = db_manager.files_col.find({}).sort('date', -1).limit(20)
        movies = []
        
        async for doc in cursor:
            movies.append({
                'title': doc.get('title'),
                'date': doc.get('date'),
                'quality': doc.get('quality', '480p'),
                'file_size': doc.get('file_size'),
                'is_new': is_new(doc.get('date')) if doc.get('date') else False,
                'is_video_file': doc.get('is_video_file', False),
                'thumbnail': doc.get('thumbnail'),
                'thumbnail_source': doc.get('thumbnail_source', 'unknown')
            })
        
        return jsonify({
            'status': 'success',
            'movies': movies,
            'total': len(movies),
            'mode': 'basic'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/background/index', methods=['POST'])
async def api_background_index():
    """Start background indexing (admin only)"""
    try:
        # Check admin key
        data = await request.get_json() or {}
        admin_key = data.get('admin_key') or request.args.get('admin_key')
        
        if admin_key != Config.ADMIN_API_KEY:
            return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401
        
        # Start indexing in background
        asyncio.create_task(index_files_background())
        
        return jsonify({
            'status': 'success',
            'message': 'Background indexing started',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/verify/<int:user_id>')
async def api_verify_user(user_id):
    """Get verification link for user"""
    try:
        if not bot_instance or not bot_instance.verification_system:
            return jsonify({'status': 'error', 'message': 'Verification system not available'}), 503
        
        verification_data = await bot_instance.verification_system.create_verification_link(user_id)
        
        return jsonify({
            'status': 'success',
            'user_id': user_id,
            'verification_url': verification_data['short_url'],
            'expires_in': '1 hour',
            'bot_username': Config.BOT_USERNAME
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/premium/plans')
async def api_premium_plans():
    """Get all premium plans"""
    try:
        if not bot_instance or not bot_instance.premium_system:
            return jsonify({'status': 'error', 'message': 'Premium system not available'}), 503
        
        plans = await bot_instance.premium_system.get_all_plans()
        
        return jsonify({
            'status': 'success',
            'plans': plans,
            'currency': 'INR'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/premium/status/<int:user_id>')
async def api_premium_status(user_id):
    """Get premium status for user"""
    try:
        if not bot_instance or not bot_instance.premium_system:
            return jsonify({'status': 'error', 'message': 'Premium system not available'}), 503
        
        details = await bot_instance.premium_system.get_subscription_details(user_id)
        
        return jsonify({
            'status': 'success',
            'user_id': user_id,
            'premium_details': details
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/poster')
async def api_poster():
    """Get poster for movie"""
    try:
        title = request.args.get('title', '').strip()
        if not title:
            return jsonify({'status': 'error', 'message': 'Title required'}), 400
        
        if not bot_instance or not bot_instance.poster_fetcher:
            # Return default poster
            return jsonify({
                'status': 'success',
                'poster_url': f"{Config.BACKEND_URL}/static/default_poster.jpg",
                'source': 'default',
                'rating': '0.0',
                'year': '2024'
            })
        
        poster_data = await bot_instance.poster_fetcher.fetch_poster(title)
        
        return jsonify({
            'status': 'success',
            'title': title,
            'poster_data': poster_data
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/system/stats')
async def api_system_stats():
    """Get system statistics (admin only)"""
    try:
        # Check admin key
        admin_key = request.args.get('admin_key')
        if admin_key != Config.ADMIN_API_KEY:
            return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401
        
        # Get MongoDB stats
        total_files = await db_manager.files_col.count_documents({}) if db_manager and db_manager.files_col else 0
        video_files = await db_manager.files_col.count_documents({'is_video_file': True}) if db_manager and db_manager.files_col else 0
        
        # Get verification stats
        verification_stats = {}
        if bot_instance and bot_instance.verification_system:
            verification_stats = await bot_instance.verification_system.get_user_stats()
        
        # Get premium stats
        premium_stats = {}
        if bot_instance and bot_instance.premium_system:
            premium_stats = await bot_instance.premium_system.get_admin_stats()
        
        # Get cache stats
        cache_stats = {}
        if bot_instance and bot_instance.cache_manager:
            cache_stats = await bot_instance.cache_manager.get_stats_summary()
        
        hit_rate = 0
        if search_stats['redis_hits'] + search_stats['redis_misses'] > 0:
            hit_rate = (search_stats['redis_hits'] / (search_stats['redis_hits'] + search_stats['redis_misses'])) * 100
        
        return jsonify({
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'database': {
                'total_files': total_files,
                'video_files': video_files,
                'non_video_files': total_files - video_files
            },
            'verification': verification_stats,
            'premium': premium_stats,
            'cache': cache_stats,
            'search': {
                **search_stats,
                'redis_hit_rate': f"{hit_rate:.1f}%"
            },
            'system': {
                'bot_online': bot_instance.bot_started if bot_instance else False,
                'user_session_online': bot_instance.user_session_ready if bot_instance else False,
                'channels_configured': len(CHANNEL_CONFIG),
                'text_channels': len(Config.TEXT_CHANNEL_IDS),
                'uptime': 'N/A'
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Main function
async def main():
    """Main function to start the bot"""
    global bot_instance, db_manager
    
    try:
        # Initialize configuration
        config = Config()
        config.validate()
        
        # Initialize database
        db_manager = DatabaseManager(config.MONGODB_URI)
        if not await db_manager.connect():
            raise DatabaseError("Failed to connect to database")
        
        # Create bot instance
        bot_instance = SK4FiLMBot(config, db_manager)
        
        # Initialize bot
        await bot_instance.initialize()
        
        # Setup bot handlers
        await setup_bot_handlers(bot_instance.bot, bot_instance)
        
        # Start web server
        hypercorn_config = HyperConfig()
        hypercorn_config.bind = [f"0.0.0.0:{config.WEB_SERVER_PORT}"]
        hypercorn_config.workers = 1  # Reduced for Koyeb
        
        logger.info(f"🚀 Starting web server on port {config.WEB_SERVER_PORT}")
        
        # Run both web server and bot
        await asyncio.gather(
            serve(app, hypercorn_config),
            idle()  # Keep bot running
        )
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        # Cleanup
        if bot_instance:
            await bot_instance.shutdown()
        if db_manager:
            await db_manager.close()
        logger.info("Bot stopped")

if __name__ == "__main__":
    asyncio.run(main())
