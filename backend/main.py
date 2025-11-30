import asyncio
import os
import logging
from datetime import datetime, timedelta
from pyrogram import Client, filters
from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from pyrogram.errors import UserNotParticipant, ChatAdminRequired, ChannelPrivate, FloodWait
from quart import Quart, jsonify, request, Response
from hypercorn.asyncio import serve
from hypercorn.config import Config as HyperConfig
from motor.motor_asyncio import AsyncIOMotorClient
import html
import re
import math
import aiohttp
import urllib.parse
import base64
from io import BytesIO
import time
import redis.asyncio as redis
import json

# FAST LOADING OPTIMIZATIONS
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger('asyncio').setLevel(logging.WARNING)
logging.getLogger('pyrogram').setLevel(logging.WARNING)
logging.getLogger('hypercorn').setLevel(logging.WARNING)

class Config:
    API_ID = int(os.environ.get("API_ID", "0"))
    API_HASH = os.environ.get("API_HASH", "")
    USER_SESSION_STRING = os.environ.get("USER_SESSION_STRING", "")
    BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
    MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
    REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
    
    # CHANNEL CONFIGURATION - MULTI-CHANNEL SUPPORT
    MAIN_CHANNEL_ID = -1001891090100
    TEXT_CHANNEL_IDS = [-1001891090100, -1002024811395]  # Both channels for search
    FILE_CHANNEL_ID = -1001768249569
    
    MAIN_CHANNEL_LINK = "https://t.me/sk4film"
    UPDATES_CHANNEL_LINK = "https://t.me/sk4film_Request"
    CHANNEL_USERNAME = "sk4film"
    
    SHORTLINK_URL = os.environ.get("SHORTLINK_URL", "gplinks.in")
    SHORTLINK_API = os.environ.get("SHORTLINK_API", "")
    VERIFICATION_REQUIRED = os.environ.get("VERIFICATION_REQUIRED", "false").lower() == "true"
    VERIFICATION_DURATION = 6 * 60 * 60  # 6 hours
    
    WEBSITE_URL = os.environ.get("WEBSITE_URL", "https://sk4film.vercel.app")
    BOT_USERNAME = os.environ.get("BOT_USERNAME", "sk4filmbot")
    ADMIN_IDS = [int(x) for x in os.environ.get("ADMIN_IDS", "123456789").split(",")]
    AUTO_DELETE_TIME = int(os.environ.get("AUTO_DELETE_TIME", "300"))
    WEB_SERVER_PORT = int(os.environ.get("PORT", 8000))
    BACKEND_URL = os.environ.get("BACKEND_URL", "https://sk4film.koyeb.app")
    
    OMDB_KEYS = ["8265bd1c", "b9bd48a6", "3e7e1e9d"]
    TMDB_KEYS = ["e547e17d4e91f3e62a571655cd1ccaff", "8265bd1f"]
    
    @classmethod
    def validate_config(cls):
        """Validate all required configuration settings"""
        required_vars = [
            "API_ID", "API_HASH", "BOT_TOKEN", 
            "MONGODB_URI", "MAIN_CHANNEL_ID"
        ]
        
        missing = []
        for var in required_vars:
            if not getattr(cls, var, None):
                missing.append(var)
        
        if missing:
            raise ValueError(f"Missing required config variables: {missing}")
        
        # Validate verification configuration
        if cls.VERIFICATION_REQUIRED:
            if not cls.SHORTLINK_API:
                logger.warning("⚠️ VERIFICATION_REQUIRED is True but SHORTLINK_API is not set")
            if not cls.SHORTLINK_URL:
                logger.warning("⚠️ VERIFICATION_REQUIRED is True but SHORTLINK_URL is not set")
        
        return True

# FAST INITIALIZATION
app = Quart(__name__)

@app.after_request
async def add_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

# GLOBAL VARIABLES - FAST ACCESS
mongo_client = None
db = None
files_col = None
verification_col = None
redis_client = None
User = None
bot = None
bot_started = False
user_session_ready = False

# Initialize verification system and poster fetcher
verification_system = None
poster_fetcher = None

# OPTIMIZED CACHE SYSTEM
movie_db = {
    'poster_cache': {},
    'title_cache': {},
    'search_cache': {},
    'stats': {
        'letterboxd': 0, 'imdb': 0, 'justwatch': 0, 'impawards': 0,
        'omdb': 0, 'tmdb': 0, 'custom': 0, 'cache_hits': 0, 'video_thumbnails': 0,
        'redis_hits': 0, 'redis_misses': 0, 'multi_channel_searches': 0
    }
}

# CHANNEL CONFIGURATION
CHANNEL_CONFIG = {
    -1001891090100: {
        'name': 'SK4FiLM Main',
        'type': 'text',
        'search_priority': 1
    },
    -1002024811395: {
        'name': 'SK4FiLM Updates', 
        'type': 'text',
        'search_priority': 2
    },
    -1001768249569: {
        'name': 'SK4FiLM Files',
        'type': 'file',
        'search_priority': 0  # Files channel - indexed separately
    }
}

# REDIS CACHE MANAGER
class RedisCache:
    def __init__(self):
        self.client = None
        self.enabled = False
    
    async def init_redis(self):
        try:
            # Use the exact working format from your example
            self.client = redis.Redis(
                host='redis-17119.c283.us-east-1-4.ec2.cloud.redislabs.com',
                port=17119,
                username="default",
                password="EjtnvQpIkLv5Z3g9Fr4FQDLfmLKZVqML",
                decode_responses=True,
                encoding='utf-8',
                socket_connect_timeout=10,
                socket_timeout=10,
                max_connections=10,
                health_check_interval=30
            )
            
            # Test connection
            await self.client.ping()
            self.enabled = True
            logger.info("✅ Redis connected successfully to Redis Labs!")
            
            # Test basic operations
            await self.client.set('connection_test', 'success', ex=60)
            test_result = await self.client.get('connection_test')
            if test_result == 'success':
                logger.info("✅ Redis basic operations test: PASSED")
            else:
                logger.warning("⚠️ Redis connected but operations test failed")
            
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed: {e}")
            self.enabled = False
            return False
    
    async def get(self, key):
        if not self.enabled or not self.client:
            return None
        try:
            data = await self.client.get(key)
            if data:
                movie_db['stats']['redis_hits'] += 1
            return data
        except Exception as e:
            logger.warning(f"Redis get error: {e}")
            return None
    
    async def set(self, key, value, expire=3600):
        if not self.enabled or not self.client:
            return False
        try:
            await self.client.setex(key, expire, value)
            return True
        except Exception as e:
            logger.warning(f"Redis set error: {e}")
            return False
    
    async def delete(self, key):
        if not self.enabled or not self.client:
            return False
        try:
            await self.client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Redis delete error: {e}")
            return False
    
    async def clear_search_cache(self):
        if not self.enabled or not self.client:
            return False
        try:
            keys = await self.client.keys("search:*")
            if keys:
                await self.client.delete(*keys)
                logger.info(f"🧹 Cleared {len(keys)} search cache keys")
            return True
        except Exception as e:
            logger.warning(f"Redis clear error: {e}")
            return False

# Initialize Redis cache
redis_cache = RedisCache()

# IMPROVED FLOOD WAIT PROTECTION
class FloodWaitProtection:
    def __init__(self):
        self.last_request_time = 0
        self.min_interval = 3  # Increased from 2 to 3 seconds
        self.request_count = 0
        self.reset_time = time.time()
        self.consecutive_waits = 0
        self.last_wait_time = 0
    
    async def wait_if_needed(self):
        current_time = time.time()
        
        # Reset counter every 2 minutes (increased from 1 minute)
        if current_time - self.reset_time > 120:
            self.request_count = 0
            self.reset_time = current_time
            self.consecutive_waits = 0
        
        # Limit to 20 requests per 2 minutes (reduced from 30 per minute)
        if self.request_count >= 20:
            wait_time = 120 - (current_time - self.reset_time)
            if wait_time > 0:
                self.consecutive_waits += 1
                extra_wait = self.consecutive_waits * 5  # Progressive waiting
                total_wait = wait_time + extra_wait
                
                logger.warning(f"⚠️ Rate limit reached, waiting {total_wait:.1f}s (consecutive: {self.consecutive_waits})")
                await asyncio.sleep(total_wait)
                
                self.request_count = 0
                self.reset_time = time.time()
        
        # Ensure minimum interval between requests
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_interval:
            wait_time = self.min_interval - time_since_last
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()
        self.request_count += 1

flood_protection = FloodWaitProtection()

# SAFE TELEGRAM OPERATIONS WITH IMPROVED FLOOD WAIT HANDLING
async def safe_telegram_operation(operation, *args, **kwargs):
    """Safely execute Telegram operations with enhanced flood wait protection"""
    max_retries = 2  # Reduced from 3 to minimize repeated attempts
    base_delay = 5   # Increased base delay
    
    for attempt in range(max_retries):
        try:
            await flood_protection.wait_if_needed()
            result = await operation(*args, **kwargs)
            
            # Reset consecutive waits on successful operation
            if flood_protection.consecutive_waits > 0:
                flood_protection.consecutive_waits = 0
                
            return result
        except FloodWait as e:
            wait_time = e.value + 10  # Extra buffer increased from 5 to 10
            logger.warning(f"⚠️ Flood wait detected: {e.value}s -> waiting {wait_time}s, attempt {attempt + 1}/{max_retries}")
            
            if attempt == max_retries - 1:
                logger.error(f"❌ Flood wait too long after {max_retries} attempts")
                # Store wait time for progressive backoff
                flood_protection.last_wait_time = wait_time
                raise e
            
            logger.info(f"🕒 Waiting {wait_time}s due to flood wait...")
            await asyncio.sleep(wait_time)
            
            # Progressive backoff for consecutive flood waits
            if attempt > 0:
                await asyncio.sleep(attempt * 10)
        except Exception as e:
            logger.error(f"❌ Telegram operation failed: {e}")
            if attempt == max_retries - 1:
                raise e
            await asyncio.sleep(base_delay * (2 ** attempt))  # Exponential backoff
    return None

# SAFE ASYNC ITERATOR FOR TELEGRAM GENERATORS
async def safe_telegram_generator(operation, *args, limit=None, **kwargs):
    """Safely iterate over Telegram async generators with flood wait protection"""
    max_retries = 2  # Reduced retries
    count = 0
    
    for attempt in range(max_retries):
        try:
            await flood_protection.wait_if_needed()
            async for item in operation(*args, **kwargs):
                yield item
                count += 1
                
                # Add small delay between items to reduce flood wait
                if count % 10 == 0:
                    await asyncio.sleep(1)
                    
                if limit and count >= limit:
                    break
            break  # Successfully completed iteration
        except FloodWait as e:
            wait_time = e.value + 10
            logger.warning(f"⚠️ Flood wait in generator: {e.value}s -> waiting {wait_time}s, attempt {attempt + 1}/{max_retries}")
            
            if attempt == max_retries - 1:
                logger.error(f"❌ Flood wait too long after {max_retries} attempts")
                raise e
            
            logger.info(f"🕒 Waiting {wait_time}s due to flood wait...")
            await asyncio.sleep(wait_time)
            
            # Progressive backoff
            if attempt > 0:
                await asyncio.sleep(attempt * 10)
        except Exception as e:
            logger.error(f"❌ Telegram generator failed: {e}")
            if attempt == max_retries - 1:
                raise e
            await asyncio.sleep(5 * (2 ** attempt))  # Increased base delay

# CACHE CLEANUP TASK
async def cache_cleanup():
    while True:
        await asyncio.sleep(3600)
        try:
            current_time = datetime.now()
            expired_keys = []
            for key, (data, timestamp) in movie_db['poster_cache'].items():
                if (current_time - timestamp).seconds > 3600:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del movie_db['poster_cache'][key]
            
            expired_search_keys = []
            for key, (data, timestamp) in movie_db['search_cache'].items():
                if (current_time - timestamp).seconds > 1800:
                    expired_search_keys.append(key)
            
            for key in expired_search_keys:
                del movie_db['search_cache'][key]
                
            logger.info(f"🧹 Cache cleaned: {len(expired_keys)} posters, {len(expired_search_keys)} searches")
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")

# OPTIMIZED MONGODB INIT WITH DUPLICATE KEY HANDLING
async def init_mongodb():
    global mongo_client, db, files_col, verification_col
    try:
        logger.info("🔌 MongoDB initialization...")
        mongo_client = AsyncIOMotorClient(Config.MONGODB_URI, serverSelectionTimeoutMS=5000, maxPoolSize=10)
        await mongo_client.admin.command('ping')
        
        db = mongo_client.sk4film
        files_col = db.files
        verification_col = db.verifications
        
        # Create indexes safely
        logger.info("🔧 Creating indexes...")
        
        existing_indexes = await files_col.index_information()
        
        if 'title_text' not in existing_indexes:
            try:
                await files_col.create_index([("title", "text")])
                logger.info("✅ Created title text index")
            except Exception as e:
                logger.warning(f"⚠️ Title text index creation failed: {e}")
        
        if 'normalized_title_1' not in existing_indexes:
            try:
                await files_col.create_index([("normalized_title", 1)])
                logger.info("✅ Created normalized_title index")
            except Exception as e:
                logger.warning(f"⚠️ Normalized title index creation failed: {e}")
        
        # Create non-unique index to avoid duplicate issues
        if 'msg_ch_idx' not in existing_indexes:
            try:
                await files_col.create_index(
                    [("message_id", 1), ("channel_id", 1)], 
                    name="msg_ch_idx"
                )
                logger.info("✅ Created message_channel index")
            except Exception as e:
                logger.warning(f"⚠️ Message channel index creation failed: {e}")
        
        if 'indexed_at_-1' not in existing_indexes:
            try:
                await files_col.create_index([("indexed_at", -1)])
                logger.info("✅ Created indexed_at index")
            except Exception as e:
                logger.warning(f"⚠️ Indexed_at index creation failed: {e}")
        
        # Verification collection indexes
        try:
            await verification_col.create_index([("user_id", 1)], unique=True)
            logger.info("✅ Created user_id index for verification")
        except Exception as e:
            logger.warning(f"⚠️ Verification index creation failed: {e}")
        
        logger.info("✅ MongoDB OK - Optimized and Cleaned")
        return True
    except Exception as e:
        logger.error(f"❌ MongoDB: {e}")
        return False

# IMPROVED TITLE NORMALIZATION FOR INDIAN MOVIES
def normalize_title(title):
    if not title:
        return ""
    normalized = title.lower().strip()
    
    # Remove years, quality indicators, and technical terms
    normalized = re.sub(
        r'\b(19|20)\d{2}\b|\b(480p|720p|1080p|2160p|4k|hd|fhd|uhd|hevc|x264|x265|h264|h265|bluray|webrip|hdrip|web-dl|hdtv|hdrip|webdl|hindi|english|tamil|telugu|malayalam|kannada|punjabi|bengali|marathi|gujarati|movie|film|series|complete|full|part|episode|season|hdrc|dvdscr|pre-dvd|p-dvd|pdc|rarbg|yts|amzn|netflix|hotstar|prime|disney|hc-esub|esub|subs)\b',
        '', 
        normalized, 
        flags=re.IGNORECASE
    )
    
    # Clean up extra spaces and special characters
    normalized = re.sub(r'[\._\-]', ' ', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized

# IMPROVED TITLE EXTRACTION FOR INDIAN MOVIE FORMATS
def extract_title_smart(text):
    if not text or len(text) < 10:
        return None
    
    text_hash = hash(text[:200])
    if text_hash in movie_db['title_cache']:
        return movie_db['title_cache'][text_hash]
    
    try:
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if not lines:
            return None
        
        first_line = lines[0]
        
        # IMPROVED PATTERNS FOR INDIAN MOVIE TITLES
        patterns = [
            # Pattern for: "Tere Ishk Mein 2025 1080p Hindi HDTC x264 AAC HC-ESub"
            (r'^([A-Za-z\s]{3,50}?)\s*(?:\(?\d{4}\)?|\b(?:480p|720p|1080p|2160p|4k|hd|fhd|uhd)\b)', 1),
            
            # Pattern for: "🎬 Tere Ishk Mein (2025)"
            (r'🎬\s*([^\n\-\(]{3,60}?)\s*(?:\(\d{4}\)|$)', 1),
            
            # Pattern for: "Tere Ishk Mein - 2025 Movie"
            (r'^([^\-\n]{3,60}?)\s*\-', 1),
            
            # Pattern for titles with year in parentheses: "Tere Ishk Mein (2025)"
            (r'^([^\(\n]{3,60}?)\s*\(\d{4}\)', 1),
            
            # Pattern for Hindi/Indian movie titles with common keywords
            (r'^([A-Za-z\s]{3,50}?)\s*(?:\d{4}|Hindi|Movie|Film|HDTC|WebDL|X264|AAC|ESub)', 1),
        ]
        
        for pattern, group in patterns:
            match = re.search(pattern, first_line, re.IGNORECASE)
            if match:
                title = match.group(group).strip()
                title = re.sub(r'\s+', ' ', title)
                if 3 <= len(title) <= 60:
                    movie_db['title_cache'][text_hash] = title
                    return title
        
        # FALLBACK: Extract first meaningful words before quality indicators
        if len(first_line) >= 3:
            # Remove common technical terms and quality indicators
            clean_title = re.sub(
                r'\b(480p|720p|1080p|2160p|4k|hd|fhd|uhd|hevc|x264|x265|h264|h265|bluray|webrip|hdrip|web-dl|hdtv|hdrip|webdl|hindi|english|tamil|telugu|malayalam|kannada|punjabi|bengali|marathi|gujarati|movie|film|series|complete|full|part|episode|season|\d{4}|hdrc|dvdscr|pre-dvd|p-dvd|pdc|rarbg|yts|amzn|netflix|hotstar|prime|disney|hc-esub|esub|subs)\b',
                '', 
                first_line, 
                flags=re.IGNORECASE
            )
            
            # Clean up extra spaces and special characters
            clean_title = re.sub(r'[\._\-]', ' ', clean_title)
            clean_title = re.sub(r'\s+', ' ', clean_title).strip()
            
            # Remove trailing year if any
            clean_title = re.sub(r'\s+\(\d{4}\)$', '', clean_title)
            clean_title = re.sub(r'\s+\d{4}$', '', clean_title)
            
            if 3 <= len(clean_title) <= 60:
                movie_db['title_cache'][text_hash] = clean_title
                return clean_title
                
    except Exception as e:
        logger.error(f"Title extraction error: {e}")
    
    movie_db['title_cache'][text_hash] = None
    return None

# IMPROVED FILE TITLE EXTRACTION
def extract_title_from_file(msg):
    try:
        if msg.caption:
            t = extract_title_smart(msg.caption)
            if t:
                return t
        
        fn = msg.document.file_name if msg.document else (msg.video.file_name if msg.video else None)
        if fn:
            # IMPROVED FILE NAME PARSING
            name = fn.rsplit('.', 1)[0]  # Remove extension
            
            # Replace separators with spaces
            name = re.sub(r'[\._\-]', ' ', name)
            
            # Remove quality and technical terms (IMPROVED LIST)
            name = re.sub(
                r'\b(720p|1080p|480p|2160p|4k|HDRip|WEBRip|WEB-DL|BluRay|BRRip|DVDRip|HDTV|HDTC|X264|X265|HEVC|H264|H265|AAC|AC3|DD5\.1|DDP5\.1|HC|ESub|Subs|Hindi|English|Dual|Multi|Complete|Full|Movie|Film|\d{4})\b',
                '', 
                name, 
                flags=re.IGNORECASE
            )
            
            # Clean up extra spaces
            name = re.sub(r'\s+', ' ', name).strip()
            
            # Remove trailing year if any
            name = re.sub(r'\s+\(\d{4}\)$', '', name)
            name = re.sub(r'\s+\d{4}$', '', name)
            
            if 4 <= len(name) <= 50:
                return name
    except Exception as e:
        logger.error(f"File title extraction error: {e}")
    return None

# FAST UTILITY FUNCTIONS
def format_size(size):
    if not size:
        return "Unknown"
    if size < 1024*1024:
        return f"{size/1024:.1f} KB"
    elif size < 1024*1024*1024:
        return f"{size/(1024*1024):.1f} MB"
    else:
        return f"{size/(1024*1024*1024):.2f} GB"

def detect_quality(filename):
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
    if not text:
        return ""
    text = html.escape(text)
    text = re.sub(r'(https?://[^\s]+)', r'<a href="\1" target="_blank" style="color:#00ccff">\1</a>', text)
    return text.replace('\n', '<br>')

def channel_name(cid):
    return CHANNEL_CONFIG.get(cid, {}).get('name', f"Channel {cid}")

def is_new(date):
    try:
        if isinstance(date, str):
            date = datetime.fromisoformat(date.replace('Z', '+00:00'))
        hours = (datetime.now() - date.replace(tzinfo=None)).total_seconds() / 3600
        return hours <= 48
    except:
        return False

def is_video_file(file_name):
    if not file_name:
        return False
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.mpg', '.mpeg']
    file_name_lower = file_name.lower()
    return any(file_name_lower.endswith(ext) for ext in video_extensions)

# ENHANCED VIDEO THUMBNAIL PROCESSING WITH BATCHING
async def process_thumbnail_batch(thumbnail_batch):
    """Process thumbnails in batches for better performance"""
    semaphore = asyncio.Semaphore(3)  # Limit concurrent thumbnail processing
    
    async def process_single(file_data):
        async with semaphore:
            try:
                thumbnail_url = await extract_video_thumbnail(User, file_data['message'])
                if thumbnail_url:
                    return file_data['message'].id, thumbnail_url, 'video_direct'
                
                # Fallback to poster
                title = extract_title_from_file(file_data['message'])
                if title:
                    async with aiohttp.ClientSession() as session:
                        poster_data = await poster_fetcher.get_poster_guaranteed(title, session)
                    if poster_data and poster_data.get('poster_url'):
                        return file_data['message'].id, poster_data['poster_url'], 'poster_api'
                
                return file_data['message'].id, None, 'none'
            except Exception as e:
                logger.error(f"Thumbnail processing failed for message {file_data['message'].id}: {e}")
                return file_data['message'].id, None, 'error'
    
    tasks = [process_single(file_data) for file_data in thumbnail_batch]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    thumbnails = {}
    for result in results:
        if isinstance(result, tuple) and len(result) == 3:
            message_id, thumbnail_url, source = result
            if thumbnail_url:
                thumbnails[message_id] = (thumbnail_url, source)
    
    return thumbnails

# IMPROVED VIDEO THUMBNAIL EXTRACTION
async def extract_video_thumbnail(user_client, message):
    """Extract thumbnail from video file with multiple fallback methods"""
    try:
        # Method 1: Get thumbnail from video message
        if message.video:
            # Get video thumbnail from Telegram
            thumbnail = message.video.thumbs[0] if message.video.thumbs else None
            if thumbnail:
                # Download thumbnail file
                thumbnail_path = await safe_telegram_operation(
                    user_client.download_media, 
                    thumbnail.file_id, 
                    in_memory=True
                )
                if thumbnail_path:
                    # Convert to base64 for web display
                    thumbnail_data = base64.b64encode(thumbnail_path.getvalue()).decode('utf-8')
                    thumbnail_url = f"data:image/jpeg;base64,{thumbnail_data}"
                    movie_db['stats']['video_thumbnails'] += 1
                    return thumbnail_url
        
        # Method 2: Try to get from document if it's a video file
        if message.document:
            file_name = message.document.file_name or ""
            if is_video_file(file_name):
                # Try to get document thumbnail
                thumbnail = message.document.thumbs[0] if message.document.thumbs else None
                if thumbnail:
                    thumbnail_path = await safe_telegram_operation(
                        user_client.download_media, 
                        thumbnail.file_id, 
                        in_memory=True
                    )
                    if thumbnail_path:
                        thumbnail_data = base64.b64encode(thumbnail_path.getvalue()).decode('utf-8')
                        thumbnail_url = f"data:image/jpeg;base64,{thumbnail_data}"
                        movie_db['stats']['video_thumbnails'] += 1
                        return thumbnail_url
        
        return None
        
    except Exception as e:
        logger.error(f"    ❌ Video thumbnail extraction failed: {e}")
        return None

async def get_telegram_video_thumbnail(user_client, channel_id, message_id):
    """Get thumbnail directly from Telegram video"""
    try:
        # Get the message
        msg = await safe_telegram_operation(
            user_client.get_messages,
            channel_id, 
            message_id
        )
        if not msg or (not msg.video and not msg.document):
            return None
        
        # Extract thumbnail
        thumbnail_url = await extract_video_thumbnail(user_client, msg)
        return thumbnail_url
        
    except Exception as e:
        logger.error(f"    ❌ Telegram video thumbnail error: {e}")
        return None

# SMART BACKGROUND INDEXING - ONLY NEW FILES WITH BATCH THUMBNAIL PROCESSING
async def index_files_background():
    if not User or files_col is None or not user_session_ready:
        logger.warning("⚠️ Cannot start indexing - User session not ready")
        return
    
    logger.info("📁 Starting SMART background indexing (NEW FILES ONLY)...")
    
    try:
        total_count = 0
        new_files_count = 0
        video_files_count = 0
        successful_thumbnails = 0
        batch = []
        batch_size = 15  # Smaller batch size for better performance
        thumbnail_batch = []
        
        # Get the last indexed message ID to start from there
        last_indexed = await files_col.find_one(
            {}, 
            sort=[('message_id', -1)]
        )
        
        last_message_id = last_indexed['message_id'] if last_indexed else 0
        logger.info(f"🔄 Starting from message ID: {last_message_id}")
        
        # Process only NEW messages (after last indexed)
        processed_count = 0
        new_messages_found = 0
        
        async for msg in safe_telegram_generator(User.get_chat_history, Config.FILE_CHANNEL_ID):
            processed_count += 1
            
            # Stop if we've processed too many old messages
            if msg.id <= last_message_id:
                if processed_count % 100 == 0:
                    logger.info(f"    ⏩ Skipped {processed_count} old messages...")
                continue
                
            new_messages_found += 1
            
            if new_messages_found % 20 == 0:
                logger.info(f"    📥 New files found: {new_messages_found}, processing...")
            
            if msg and (msg.document or msg.video):
                # Check if already exists in database (double check)
                existing = await files_col.find_one({
                    'channel_id': Config.FILE_CHANNEL_ID,
                    'message_id': msg.id
                })
                
                if existing:
                    if new_messages_found % 50 == 0:
                        logger.info(f"    ⏭️ Already indexed: {msg.id}, skipping...")
                    continue
                
                title = extract_title_from_file(msg)
                if title:
                    file_id = msg.document.file_id if msg.document else msg.video.file_id
                    file_size = msg.document.file_size if msg.document else (msg.video.file_size if msg.video else 0)
                    file_name = msg.document.file_name if msg.document else (msg.video.file_name if msg.video else 'video.mp4')
                    quality = detect_quality(file_name)
                    
                    file_is_video = is_video_file(file_name)
                    
                    # Prepare for batch thumbnail processing
                    if file_is_video:
                        video_files_count += 1
                        thumbnail_batch.append({
                            'message': msg,
                            'title': title,
                            'file_is_video': file_is_video
                        })
                    
                    # Add to database batch
                    batch.append({
                        'channel_id': Config.FILE_CHANNEL_ID,
                        'message_id': msg.id,
                        'title': title,
                        'normalized_title': normalize_title(title),
                        'file_id': file_id,
                        'quality': quality,
                        'file_size': file_size,
                        'file_name': file_name,
                        'caption': msg.caption or '',
                        'date': msg.date,
                        'indexed_at': datetime.now(),
                        'thumbnail': None,  # Will be updated after batch processing
                        'is_video_file': file_is_video,
                        'thumbnail_source': 'pending'
                    })
                    
                    total_count += 1
                    new_files_count += 1
                    
                    # Process thumbnail batch when it reaches size
                    if len(thumbnail_batch) >= 10:
                        logger.info(f"    🖼️ Processing batch of {len(thumbnail_batch)} thumbnails...")
                        thumbnails = await process_thumbnail_batch(thumbnail_batch)
                        
                        # Update batch with thumbnails
                        for doc in batch:
                            if doc['message_id'] in thumbnails:
                                thumbnail_url, source = thumbnails[doc['message_id']]
                                doc['thumbnail'] = thumbnail_url
                                doc['thumbnail_source'] = source
                                successful_thumbnails += 1
                        
                        thumbnail_batch = []
                    
                    # Process database batch
                    if len(batch) >= batch_size:
                        try:
                            for doc in batch:
                                await files_col.update_one(
                                    {
                                        'channel_id': doc['channel_id'], 
                                        'message_id': doc['message_id']
                                    },
                                    {'$set': doc},
                                    upsert=True
                                )
                            logger.info(f"    ✅ Batch processed: {new_files_count} new files, {successful_thumbnails} thumbnails")
                            batch = []
                            # Increased delay between batches to avoid flood
                            await asyncio.sleep(3)
                        except Exception as e:
                            logger.error(f"Batch error: {e}")
                            batch = []
        
        # Process remaining thumbnail batch
        if thumbnail_batch:
            logger.info(f"    🖼️ Processing final batch of {len(thumbnail_batch)} thumbnails...")
            thumbnails = await process_thumbnail_batch(thumbnail_batch)
            
            # Update batch with thumbnails
            for doc in batch:
                if doc['message_id'] in thumbnails:
                    thumbnail_url, source = thumbnails[doc['message_id']]
                    doc['thumbnail'] = thumbnail_url
                    doc['thumbnail_source'] = source
                    successful_thumbnails += 1
        
        # Process remaining database batch
        if batch:
            try:
                for doc in batch:
                    await files_col.update_one(
                        {
                            'channel_id': doc['channel_id'], 
                            'message_id': doc['message_id']
                        },
                        {'$set': doc},
                        upsert=True
                    )
            except Exception as e:
                logger.error(f"Final batch error: {e}")
        
        logger.info(f"✅ SMART indexing finished: {new_files_count} NEW files, {video_files_count} video files, {successful_thumbnails} thumbnails")
        logger.info(f"📊 Processed {processed_count} messages, found {new_messages_found} new messages")
        
        # Clear search cache after indexing new files
        await redis_cache.clear_search_cache()
        logger.info("🧹 Search cache cleared after indexing")
        
        # Update statistics
        total_in_db = await files_col.count_documents({})
        videos_in_db = await files_col.count_documents({'is_video_file': True})
        thumbnails_in_db = await files_col.count_documents({'thumbnail': {'$ne': None}})
        
        logger.info(f"📊 FINAL STATS: Total in DB: {total_in_db}, New Added: {new_files_count}, Videos: {videos_in_db}, Thumbnails: {thumbnails_in_db}")
        
    except Exception as e:
        logger.error(f"❌ Background indexing error: {e}")

# ENHANCED MULTI-CHANNEL LIVE POSTS FETCHING
async def get_live_posts_multi_channel(limit_per_channel=10):
    """Get posts from multiple channels concurrently"""
    if not User or not user_session_ready:
        return []
    
    all_posts = []
    
    async def fetch_channel_posts(channel_id):
        posts = []
        try:
            async for msg in safe_telegram_generator(User.get_chat_history, channel_id, limit=limit_per_channel):
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

# ENHANCED MULTI-CHANNEL SEARCH WITH REDIS CACHING
async def search_movies_multi_channel(query, limit=12, page=1):
    """Search across multiple channels with enhanced results"""
    offset = (page - 1) * limit
    
    # Try Redis cache first
    cache_key = f"search:{query}:{page}:{limit}"
    if redis_cache.enabled:
        cached_data = await redis_cache.get(cache_key)
        if cached_data:
            try:
                result_data = json.loads(cached_data)
                logger.info(f"✅ Redis cache HIT for: {query}")
                return result_data
            except Exception as e:
                logger.warning(f"Redis cache parse error: {e}")
    
    movie_db['stats']['redis_misses'] += 1
    movie_db['stats']['multi_channel_searches'] += 1
    logger.info(f"🔍 Multi-channel search for: {query}")
    
    query_lower = query.lower()
    posts_dict = {}
    files_dict = {}
    
    # Use MongoDB search primarily to avoid Telegram API calls
    try:
        if files_col is not None:
            cursor = files_col.find({'$text': {'$search': query}})
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
                except:
                    continue
    except Exception as e:
        logger.error(f"File search error: {e}")
    
    # Search Telegram channels if user session is ready
    if user_session_ready:
        channel_results = {}
        
        async def search_single_channel(channel_id):
            channel_posts = {}
            try:
                cname = channel_name(channel_id)
                async for msg in safe_telegram_generator(User.search_messages, channel_id, query=query, limit=20):
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
    
    # Cache in Redis with 1 hour expiration
    if redis_cache.enabled:
        await redis_cache.set(cache_key, json.dumps(result_data, default=str), expire=3600)
    
    # Also update in-memory cache
    movie_db['search_cache'][cache_key] = (result_data, datetime.now())
    
    logger.info(f"✅ Multi-channel search completed: {len(paginated)} results from {len(set(r.get('channel_id') for r in paginated if r.get('channel_id')))} channels")
    
    return result_data

# OPTIMIZED HOME MOVIES WITH MULTI-CHANNEL SUPPORT
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
    
    if movies:
        async with aiohttp.ClientSession() as session:
            tasks = [poster_fetcher.get_poster_guaranteed(movie['title'], session) for movie in movies]
            posters = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, (movie, poster_result) in enumerate(zip(movies, posters)):
                if isinstance(poster_result, dict):
                    movie['poster_url'] = poster_result['poster_url']
                    movie['poster_source'] = poster_result['source']
                    movie['poster_rating'] = poster_result.get('rating', '0.0')
                    movie['has_poster'] = True
                else:
                    movie['poster_url'] = f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(movie['title'])}"
                    movie['poster_source'] = 'CUSTOM'
                    movie['poster_rating'] = '0.0'
                    movie['has_poster'] = True
    
    return movies

# OPTIMIZED API ROUTES
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

# Verification API routes
@app.route('/api/verify_user', methods=['POST'])
async def api_verify_user():
    return await verification_system.api_verify_user(request)

@app.route('/api/check_verification/<int:user_id>')
async def api_check_verification(user_id):
    return await verification_system.api_check_verification(user_id)

@app.route('/api/generate_verification_url/<int:user_id>')
async def api_generate_verification_url(user_id):
    return await verification_system.api_generate_verification_url(user_id)

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
        
        # Clear poster fetcher cache
        poster_fetcher.clear_cache()
        
        movie_db['stats']['redis_hits'] = 0
        movie_db['stats']['redis_misses'] = 0
        movie_db['stats']['multi_channel_searches'] = 0
        
        return jsonify({
            'status': 'success',
            'message': 'All caches cleared successfully',
            'redis_cleared': redis_cache.enabled,
            'memory_cleared': True,
            'stats_reset': True,
            'poster_cache_cleared': True
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

# IMPROVED USER SESSION RECOVERY TASK
async def user_session_recovery():
    """Wait for flood wait to expire and then initialize user session"""
    global user_session_ready
    
    # Wait for initial flood wait to expire (30-40 minutes)
    logger.info("🕒 Waiting for flood wait to expire before initializing user session...")
    await asyncio.sleep(2400)  # Wait 40 minutes
    
    try:
        logger.info("🔄 Attempting to initialize user session after flood wait...")
        await User.start()
        user_session_ready = True
        logger.info("✅ User session initialized successfully!")
        
        # Start ENHANCED background indexing with user session (only new files)
        asyncio.create_task(index_files_background())
    except Exception as e:
        logger.error(f"❌ User session initialization failed: {e}")
        # Try again in 10 minutes
        await asyncio.sleep(600)
        asyncio.create_task(user_session_recovery())

# BOT HANDLERS SETUP (Now defined in main.py to avoid circular imports)
async def setup_bot_handlers():
    """Setup all bot handlers directly in main.py to avoid circular imports"""
    
    @bot.on_message(filters.command("start") & filters.private)
    async def start_handler(client, message):
        uid = message.from_user.id
        user_name = message.from_user.first_name or "User"
        
        if len(message.command) > 1:
            fid = message.command[1]
            
            if Config.VERIFICATION_REQUIRED:
                is_verified, status = await verification_system.check_verification(uid)
                
                if not is_verified:
                    verification_url = await verification_system.generate_verification_url(uid)
                    
                    keyboard = InlineKeyboardMarkup([
                        [InlineKeyboardButton("🔗 VERIFY NOW", url=verification_url)],
                        [InlineKeyboardButton("🔄 CHECK VERIFICATION", callback_data=f"check_verify_{uid}")],
                        [InlineKeyboardButton("📢 JOIN CHANNEL", url=Config.MAIN_CHANNEL_LINK)]
                    ])
                    
                    await message.reply_text(
                        f"👋 **Hello {user_name}!**\n\n"
                        "🔒 **Verification Required**\n"
                        "To download files, you need to complete URL verification.\n\n"
                        "🚀 **Quick Steps:**\n"
                        "1. Click **VERIFY NOW** below\n"
                        "2. Complete the verification process\n"
                        "3. Come back and click **CHECK VERIFICATION**\n"
                        "4. Start downloading!\n\n"
                        "⏰ **Verification valid for 6 hours**",
                        reply_markup=keyboard,
                        disable_web_page_preview=True
                    )
                    return
            
            try:
                parts = fid.split('_')
                if len(parts) >= 2:
                    channel_id = int(parts[0])
                    message_id = int(parts[1])
                    quality = parts[2] if len(parts) > 2 else "HD"
                    
                    pm = await message.reply_text(f"⏳ **Preparing your file...**\n\n📦 Quality: {quality}")
                    
                    file_message = await safe_telegram_operation(
                        bot.get_messages,
                        channel_id, 
                        message_id
                    )
                    
                    if not file_message or (not file_message.document and not file_message.video):
                        await pm.edit_text("❌ **File not found**\n\nThe file may have been deleted.")
                        return
                    
                    if file_message.document:
                        sent = await safe_telegram_operation(
                            bot.send_document,
                            uid, 
                            file_message.document.file_id, 
                            caption=f"♻ **Please forward this file/video to your saved messages**\n\n"
                                   f"📹 Quality: {quality}\n"
                                   f"📦 Size: {format_size(file_message.document.file_size)}\n\n"
                                   f"⚠️ Will auto-delete in {Config.AUTO_DELETE_TIME//60} minutes\n\n"
                                   f"@SK4FiLM 🍿"
                        )
                    else:
                        sent = await safe_telegram_operation(
                            bot.send_video,
                            uid, 
                            file_message.video.file_id, 

    
    @bot.on_message(filters.text & filters.private & ~filters.command(['start', 'stats', 'index', 'verify', 'clear_cache']))
    async def text_handler(client, message):
        user_name = message.from_user.first_name or "User"
        await message.reply_text(
            f"👋 **Hi {user_name}!**\n\n"
            "🔍 **Please Use Our Website To Search For Movies:**\n\n"
            f"{Config.WEBSITE_URL}\n\n"
            "This bot only handles file downloads via website links.",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)],
                [
                    InlineKeyboardButton("📢 Mᴀɪɴ Cʜᴀɴɴᴇʟ", url=Config.MAIN_CHANNEL_LINK),
                    InlineKeyboardButton("🔎 Mᴏᴠɪᴇꜱ Gʀᴏᴜᴘ", url=Config.UPDATES_CHANNEL_LINK)
                ]
            ]),
            disable_web_page_preview=True
        )
    
    @bot.on_message(filters.command("channel") & filters.private)
    async def channel_command(client, message):
        await message.reply_text(
            "📢 **SK4FiLM Channels**\n\n"
            "Join our channels for the latest movies and updates:\n\n"
            "🎬 **Main Channel:**\n"
            "• Latest movie releases\n"
            "• High quality files\n"
            "• Daily updates\n\n"
            "🔎 **Movies Group:**\n"
            "• Movie discussions\n"
            "• Requests & updates\n"
            "• Community interaction\n\n"
            "👇 **Click below to join:**",
            reply_markup=InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("🎬 MAIN CHANNEL", url=Config.MAIN_CHANNEL_LINK),
                    InlineKeyboardButton("🔎 Mᴏᴠɪᴇꜱ Gʀᴏᴜᴘ", url=Config.UPDATES_CHANNEL_LINK)
                ],
                [InlineKeyboardButton("🌐 WEBSITE", url=Config.WEBSITE_URL)]
            ]),
            disable_web_page_preview=True
        )
    
    @bot.on_message(filters.command("index") & filters.user(Config.ADMIN_IDS))
    async def index_handler(client, message):
        msg = await message.reply_text("🔄 **Starting ENHANCED background indexing (NEW FILES ONLY)...**")
        asyncio.create_task(index_files_background())
        await msg.edit_text("✅ **Enhanced indexing started in background!**\n\nOnly new files will be indexed with batch thumbnail processing. Check /stats for progress.")
    
    @bot.on_message(filters.command("clear_cache") & filters.user(Config.ADMIN_IDS))
    async def clear_cache_handler(client, message):
        msg = await message.reply_text("🧹 **Clearing all caches...**")
        
        # Clear Redis cache
        redis_cleared = await redis_cache.clear_search_cache()
        
        # Clear memory cache
        movie_db['search_cache'].clear()
        movie_db['poster_cache'].clear()
        movie_db['title_cache'].clear()
        
        # Clear poster fetcher cache
        poster_fetcher.clear_cache()
        
        movie_db['stats']['redis_hits'] = 0
        movie_db['stats']['redis_misses'] = 0
        movie_db['stats']['multi_channel_searches'] = 0
        
        await msg.edit_text(
            f"✅ **All caches cleared!**\n\n"
            f"• Redis cache: {'✅ Cleared' if redis_cleared else '❌ Failed'}\n"
            f"• Memory cache: ✅ Cleared\n"
            f"• Search cache: ✅ Cleared\n"
            f"• Poster cache: ✅ Cleared\n"
            f"• Poster fetcher cache: ✅ Cleared\n"
            f"• Multi-channel stats: ✅ Reset\n\n"
            f"Next search will be fresh from database."
        )
    
    @bot.on_message(filters.command("stats") & filters.user(Config.ADMIN_IDS))
    async def stats_handler(client, message):
        tf = await files_col.count_documents({}) if files_col is not None else 0
        video_files = await files_col.count_documents({'is_video_file': True}) if files_col is not None else 0
        video_thumbnails = await files_col.count_documents({'is_video_file': True, 'thumbnail': {'$ne': None}}) if files_col is not None else 0
        total_thumbnails = await files_col.count_documents({'thumbnail': {'$ne': None}}) if files_col is not None else 0
        
        thumbnail_coverage = f"{(video_thumbnails/video_files*100):.1f}%" if video_files > 0 else "0%"
        
        # Get last indexed file info
        last_indexed = await files_col.find_one({}, sort=[('message_id', -1)])
        last_msg_id = last_indexed['message_id'] if last_indexed else 'None'
        
        stats_text = (
            f"📊 **SK4FiLM MULTI-CHANNEL STATISTICS**\n\n"
            f"📁 **Total Files:** {tf}\n"
            f"🎥 **Video Files:** {video_files}\n"
            f"🖼️ **Video Thumbnails:** {video_thumbnails}\n"
            f"📸 **Total Thumbnails:** {total_thumbnails}\n"
            f"📈 **Coverage:** {thumbnail_coverage}\n"
            f"📨 **Last Message ID:** {last_msg_id}\n\n"
            f"🔴 **Live Posts:** Active\n"
            f"🤖 **Bot Status:** Online\n"
            f"👤 **User Session:** {'Ready' if User else 'Flood Wait'}\n"
            f"🔧 **Indexing Mode:** MULTI-CHANNEL ENHANCED\n"
            f"🔍 **Redis Cache:** {'✅ Enabled' if redis_cache.enabled else '❌ Disabled'}\n"
            f"📡 **Channels Active:** {len(Config.TEXT_CHANNEL_IDS)} text + 1 file\n\n"
            f"**🎨 Poster Sources:**\n"
            f"• Letterboxd: {movie_db['stats']['letterboxd']}\n"
            f"• IMDb: {movie_db['stats']['imdb']}\n"
            f"• JustWatch: {movie_db['stats']['justwatch']}\n"
            f"• IMPAwards: {movie_db['stats']['impawards']}\n"
            f"• OMDB: {movie_db['stats']['omdb']}\n"
            f"• TMDB: {movie_db['stats']['tmdb']}\n" 
            f"• Custom: {movie_db['stats']['custom']}\n"
            f"• Cache Hits: {movie_db['stats']['cache_hits']}\n"
            f"• Video Thumbnails: {movie_db['stats']['video_thumbnails']}\n\n"
            f"**🔍 Search Statistics:**\n"
            f"• Redis Hits: {movie_db['stats']['redis_hits']}\n"
            f"• Redis Misses: {movie_db['stats']['redis_misses']}\n"
            f"• Multi-channel Searches: {movie_db['stats']['multi_channel_searches']}\n"
            f"• Hit Rate: {(movie_db['stats']['redis_hits']/(movie_db['stats']['redis_hits'] + movie_db['stats']['redis_misses'])*100):.1f}%\n\n"
            f"**⚡ Enhanced Features:**\n"
            f"• ✅ Multi-channel search & posts\n"
            f"• ✅ Concurrent channel processing\n"
            f"• ✅ Enhanced file indexing (NEW ONLY)\n"
            f"• ✅ Batch thumbnail processing\n"
            f"• ✅ Redis search caching\n"
            f"• ✅ Enhanced flood protection\n\n"
            f"**🔗 Verification:** {'ENABLED (6 hours)' if Config.VERIFICATION_REQUIRED else 'DISABLED'}"
        )
        await message.reply_text(stats_text)

    logger.info("✅ Bot handlers setup completed!")

# OPTIMIZED INITIALIZATION WITH ENHANCED FLOOD PROTECTION
async def init():
    global User, bot, bot_started, user_session_ready, verification_system, poster_fetcher
    
    try:
        logger.info("🚀 INITIALIZING MULTI-CHANNEL SK4FiLM BOT...")
        
        # Validate configuration first
        Config.validate_config()
        
        # Initialize MongoDB first
        await init_mongodb()
        
        # Initialize Redis cache
        await redis_cache.init_redis()
        
        # Initialize verification system
        from verification_system import VerificationSystem
        verification_system = VerificationSystem(verification_col, Config)
        
        # Initialize poster fetcher
        from poster_fetcher import PosterFetcher
        poster_fetcher = PosterFetcher(Config, movie_db['stats'])
        
        # Initialize bot (this should work fine)
        bot = Client(
            "bot",
            api_id=Config.API_ID,
            api_hash=Config.API_HASH, 
            bot_token=Config.BOT_TOKEN
        )
        
        await bot.start()
        await setup_bot_handlers()
        
        me = await bot.get_me()
        logger.info(f"✅ BOT STARTED: @{me.username}")
        bot_started = True
        
        # Initialize user session but handle flood wait gracefully
        User = Client(
            "user_session", 
            api_id=Config.API_ID, 
            api_hash=Config.API_HASH, 
            session_string=Config.USER_SESSION_STRING,
            no_updates=True
        )
        
        try:
            # Try to start user session immediately
            await User.start()
            user_session_ready = True
            logger.info("✅ User session initialized immediately!")
            
            # Start ENHANCED background indexing (only new files)
            asyncio.create_task(index_files_background())
        except FloodWait as e:
            logger.warning(f"⚠️ User session flood wait detected: {e.value}s")
            logger.info("🔄 Starting user session recovery task...")
            # Start recovery task to initialize user session after flood wait
            asyncio.create_task(user_session_recovery())
        except Exception as e:
            logger.error(f"❌ User session initialization failed: {e}")
            user_session_ready = False
        
        # Start cache cleanup regardless
        asyncio.create_task(cache_cleanup())
        
        # Start verification cleanup task
        asyncio.create_task(verification_system.cleanup_expired_verifications())
        
        return True
    except Exception as e:
        logger.error(f"❌ INIT FAILED: {e}")
        return False

async def main():
    logger.info("="*60)
    logger.info("🎬 SK4FiLM v8.0 - MULTI-CHANNEL ENHANCED")
    logger.info("✅ Multi-Channel Search | Concurrent Processing")
    logger.info("✅ Redis Search Caching | Batch Thumbnail Processing")
    logger.info(f"✅ Channels: {len(Config.TEXT_CHANNEL_IDS)} text + 1 file")
    logger.info(f"✅ Redis: {'ENABLED' if redis_cache.enabled else 'DISABLED'}")
    logger.info(f"✅ Verification: {'ENABLED (6 hours)' if Config.VERIFICATION_REQUIRED else 'DISABLED'}")
    logger.info("="*60)
    
    success = await init()
    if not success:
        logger.error("❌ Failed to initialize multi-channel bot")
        return
    
    config = HyperConfig()
    config.bind = [f"0.0.0.0:{Config.WEB_SERVER_PORT}"]
    config.loglevel = "warning"
    
    logger.info(f"🌐 MULTI-CHANNEL web server starting on port {Config.WEB_SERVER_PORT}...")
    await serve(app, config)

if __name__ == "__main__":
    asyncio.run(main())
