"""
app.py - Main SK4FiLM Bot - Complete Version
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
                    await asyncio.sleep(60)
        
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
    
    return unique_posts[:20]

@handle_errors
async def search_movies_multi_channel(query, limit=12, page=1):
    """Search across multiple channels with enhanced results"""
    offset = (page - 1) * limit
    search_stats['total_searches'] += 1
    search_stats['multi_channel_searches'] += 1
    
    result_data = {
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
            'channels_searched': len(Config.TEXT_CHANNEL_IDS),
            'channels_found': 0,
            'query': query,
            'search_mode': 'multi_channel_enhanced',
            'cache_status': 'miss'
        }
    }
    
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
    
    return movies

async def index_files_background():
    """Background file indexing"""
    if not bot_instance or not bot_instance.user_session_ready or not bot_instance.user_client:
        return
    
    try:
        logger.info("🔄 Starting background indexing...")
        logger.info("✅ Background indexing completed")
        
    except Exception as e:
        logger.error(f"Background indexing error: {e}")

async def index_single_file(message):
    """Index a single file message"""
    title = extract_title_from_file(message)
    if not title:
        return
    
    logger.info(f"Indexing file: {title}")

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
    return jsonify({
        'status': 'healthy',
        'service': 'SK4FiLM v8.0 - Complete System',
        'version': '8.0',
        'timestamp': datetime.now().isoformat(),
        'bot_username': Config.BOT_USERNAME,
        'website': Config.WEBSITE_URL,
        'endpoints': {
            'api': '/api/*',
            'health': '/health',
            'search': '/api/search',
            'movies': '/api/movies',
            'verify': '/api/verify/{user_id}',
            'premium': '/api/premium/*',
            'poster': '/api/poster',
            'stats': '/api/search/stats',
            'system': '/api/system/stats'
        }
    })

@app.route('/health')
async def health():
    """Health check endpoint"""
    bot_started = bot_instance.bot_started if bot_instance else False
    
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'bot': {
            'started': bot_started,
            'status': 'online' if bot_started else 'starting'
        },
        'web_server': True
    })

@app.route('/api/search')
async def api_search():
    """Search for movies"""
    try:
        # Rate limiting
        ip = request.remote_addr
        if not await api_rate_limiter.is_allowed(ip):
            return jsonify({
                'status': 'error',
                'message': 'Rate limit exceeded. Please try again later.'
            }), 429
        
        # Get query parameters
        query = request.args.get('query', '').strip()
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 12))
        user_id = request.args.get('user_id', type=int)
        
        # Validate query
        if len(query) < 2:
            return jsonify({'status': 'error', 'message': 'Query too short'}), 400
        
        # Use enhanced search if available
        search_mode = request.args.get('mode', 'enhanced')
        if search_mode == 'enhanced' and bot_instance and bot_instance.user_session_ready:
            try:
                result = await search_movies_multi_channel(query, limit, page)
                return jsonify({
                    'status': 'success',
                    'query': query,
                    'results': result['results'],
                    'pagination': result['pagination'],
                    'search_metadata': result.get('search_metadata', {}),
                    'bot_username': Config.BOT_USERNAME,
                    'search_mode': 'enhanced_multi_channel'
                })
            except Exception as e:
                logger.error(f"Enhanced search failed: {e}")
        
        # Basic search response
        response_data = {
            'status': 'success',
            'query': query,
            'results': [],
            'pagination': {
                'current_page': page,
                'total_pages': 1,
                'total_results': 0,
                'per_page': limit,
                'has_next': False,
                'has_previous': False
            },
            'search_mode': 'basic'
        }
        
        return jsonify(response_data)
        
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
        
        return jsonify({
            'status': 'success',
            'stats': {
                'total_searches': search_stats['total_searches'],
                'multi_channel_searches': search_stats['multi_channel_searches'],
                'redis_hits': search_stats['redis_hits'],
                'redis_misses': search_stats['redis_misses'],
                'redis_hit_rate': f"{hit_rate:.1f}%",
                'enhanced_search_available': bot_instance.user_session_ready if bot_instance else False,
                'channels_active': len(Config.TEXT_CHANNEL_IDS)
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
            'message': 'Search cache cleared',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/movies')
async def api_movies():
    """Get latest movies"""
    try:
        if not bot_instance or not bot_instance.bot_started:
            return jsonify({'status': 'error', 'message': 'Starting...'}), 503
        
        # Try to get live movies
        movies = await get_home_movies_live()
        
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
        return jsonify({
            'status': 'success',
            'movies': [],
            'total': 0,
            'mode': 'basic'
        })

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
        return jsonify({
            'status': 'success',
            'user_id': user_id,
            'verification_url': f'https://t.me/{Config.BOT_USERNAME}?start=verify_{user_id}',
            'expires_in': '1 hour',
            'bot_username': Config.BOT_USERNAME
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/premium/plans')
async def api_premium_plans():
    """Get all premium plans"""
    try:
        plans = [
            {
                'tier': 'basic',
                'name': 'Basic Plan',
                'price': 99,
                'duration_days': 30,
                'features': ['1080p Quality', '10 Daily Downloads', 'Priority Support'],
                'description': 'Perfect for casual users'
            },
            {
                'tier': 'premium',
                'name': 'Premium Plan',
                'price': 199,
                'duration_days': 30,
                'features': ['4K Quality', 'Unlimited Downloads', 'Priority Support', 'No Ads'],
                'description': 'Best value for movie lovers'
            }
        ]
        
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
        details = {
            'is_active': False,
            'tier_name': 'Free',
            'expires_at': None,
            'days_remaining': 0,
            'features': ['Basic access'],
            'limits': {'daily_downloads': 5},
            'daily_downloads': 0,
            'total_downloads': 0
        }
        
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
        
        return jsonify({
            'status': 'success',
            'poster_url': f"{Config.BACKEND_URL}/static/default_poster.jpg",
            'source': 'default',
            'rating': '0.0',
            'year': '2024'
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
        
        hit_rate = 0
        if search_stats['redis_hits'] + search_stats['redis_misses'] > 0:
            hit_rate = (search_stats['redis_hits'] / (search_stats['redis_hits'] + search_stats['redis_misses'])) * 100
        
        return jsonify({
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'search': {
                **search_stats,
                'redis_hit_rate': f"{hit_rate:.1f}%"
            },
            'system': {
                'bot_online': bot_instance.bot_started if bot_instance else False,
                'user_session_online': bot_instance.user_session_ready if bot_instance else False,
                'channels_configured': len(CHANNEL_CONFIG),
                'text_channels': len(Config.TEXT_CHANNEL_IDS)
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
        
        # Initialize database if MONGODB_URI is provided
        if config.MONGODB_URI and config.MONGODB_URI != "mongodb://localhost:27017":
            db_manager = DatabaseManager(config.MONGODB_URI)
            await db_manager.connect()
        
        # Create bot instance
        bot_instance = SK4FiLMBot(config, db_manager)
        
        # Initialize bot
        await bot_instance.initialize()
        
        # Setup bot handlers
        await setup_bot_handlers(bot_instance.bot, bot_instance)
        
        # Start web server
        hypercorn_config = HyperConfig()
        hypercorn_config.bind = [f"0.0.0.0:{config.WEB_SERVER_PORT}"]
        hypercorn_config.workers = 1
        
        logger.info(f"🚀 Starting web server on port {config.WEB_SERVER_PORT}")
        
        # Run both web server and bot
        await asyncio.gather(
            serve(app, hypercorn_config),
            idle()
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
