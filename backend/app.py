"""
app.py - Complete SK4FiLM Web API System with Multi-Channel Search
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

import aiohttp
import urllib.parse
from quart import Quart, jsonify, request, Response
from hypercorn.asyncio import serve
from hypercorn.config import Config as HyperConfig
from motor.motor_asyncio import AsyncIOMotorClient
from pyrogram import Client, filters
from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from pyrogram.errors import FloodWait, UserNotParticipant, ChatAdminRequired, ChannelPrivate
import redis.asyncio as redis

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
    TEXT_CHANNEL_IDS = [-1001891090100, -1002024811395]
    FILE_CHANNEL_ID = -1001768249569
    
    MAIN_CHANNEL_LINK = "https://t.me/sk4film"
    UPDATES_CHANNEL_LINK = "https://t.me/sk4film_Request"
    CHANNEL_USERNAME = "sk4film"
    
    URL_SHORTENER_API = os.environ.get("URL_SHORTENER_API", "https://your-shortener-api.com/verify")
    URL_SHORTENER_KEY = os.environ.get("URL_SHORTENER_KEY", "")
    VERIFICATION_REQUIRED = os.environ.get("VERIFICATION_REQUIRED", "false").lower() == "true"
    VERIFICATION_DURATION = 6 * 60 * 60
    
    WEBSITE_URL = os.environ.get("WEBSITE_URL", "https://sk4film.vercel.app")
    BOT_USERNAME = os.environ.get("BOT_USERNAME", "sk4filmbot")
    ADMIN_IDS = [int(x) for x in os.environ.get("ADMIN_IDS", "123456789").split(",")]
    AUTO_DELETE_TIME = int(os.environ.get("AUTO_DELETE_TIME", "300"))
    WEB_SERVER_PORT = int(os.environ.get("PORT", 8000))
    BACKEND_URL = os.environ.get("BACKEND_URL", "https://sk4film.koyeb.app")
    
    OMDB_KEYS = ["8265bd1c", "b9bd48a6", "3e7e1e9d"]
    TMDB_KEYS = ["e547e17d4e91f3e62a571655cd1ccaff", "8265bd1f"]

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

# OPTIMIZED CACHE SYSTEM
movie_db = {
    'poster_cache': {},
    'title_cache': {},
    'search_cache': {},
    'stats': {
        'letterboxd': 0, 'imdb': 0, 'justwatch': 0, 'impawards': 0,
        'omdb': 0, 'tmdb': 0, 'custom': 0, 'cache_hits': 0, 'video_thumbnails': 0,
        'redis_hits': 0, 'redis_misses': 0, 'multi_channel_searches': 0,
        'total_searches': 0, 'api_requests': 0
    }
}

# CHANNEL CONFIGURATION
CHANNEL_CONFIG = {
    -1001891090100: {'name': 'SK4FiLM Main', 'type': 'text', 'search_priority': 1},
    -1002024811395: {'name': 'SK4FiLM Updates', 'type': 'text', 'search_priority': 2},
    -1001768249569: {'name': 'SK4FiLM Files', 'type': 'file', 'search_priority': 0}
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
            return 0
        try:
            keys = await self.client.keys("search:*")
            if keys:
                await self.client.delete(*keys)
                logger.info(f"🧹 Cleared {len(keys)} search cache keys")
            return len(keys)
        except Exception as e:
            logger.warning(f"Redis clear error: {e}")
            return 0

# Initialize Redis cache
redis_cache = RedisCache()

# IMPROVED FLOOD WAIT PROTECTION
class FloodWaitProtection:
    def __init__(self):
        self.last_request_time = 0
        self.min_interval = 3
        self.request_count = 0
        self.reset_time = time.time()
        self.consecutive_waits = 0
        self.last_wait_time = 0
    
    async def wait_if_needed(self):
        current_time = time.time()
        
        # Reset counter every 2 minutes
        if current_time - self.reset_time > 120:
            self.request_count = 0
            self.reset_time = current_time
            self.consecutive_waits = 0
        
        # Limit to 20 requests per 2 minutes
        if self.request_count >= 20:
            wait_time = 120 - (current_time - self.reset_time)
            if wait_time > 0:
                self.consecutive_waits += 1
                extra_wait = self.consecutive_waits * 5
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
    max_retries = 2
    base_delay = 5
    
    for attempt in range(max_retries):
        try:
            await flood_protection.wait_if_needed()
            result = await operation(*args, **kwargs)
            
            # Reset consecutive waits on successful operation
            if flood_protection.consecutive_waits > 0:
                flood_protection.consecutive_waits = 0
                
            return result
        except FloodWait as e:
            wait_time = e.value + 10
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
            await asyncio.sleep(base_delay * (2 ** attempt))
    return None

# SAFE ASYNC ITERATOR FOR TELEGRAM GENERATORS
async def safe_telegram_generator(operation, *args, limit=None, **kwargs):
    """Safely iterate over Telegram async generators with flood wait protection"""
    max_retries = 2
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
            break
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
            await asyncio.sleep(5 * (2 ** attempt))
    return

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
    
    normalized = re.sub(
        r'\b(19|20)\d{2}\b|\b(480p|720p|1080p|2160p|4k|hd|fhd|uhd|hevc|x264|x265|h264|h265|bluray|webrip|hdrip|web-dl|hdtv|hdrip|webdl|hindi|english|tamil|telugu|malayalam|kannada|punjabi|bengali|marathi|gujarati|movie|film|series|complete|full|part|episode|season|hdrc|dvdscr|pre-dvd|p-dvd|pdc|rarbg|yts|amzn|netflix|hotstar|prime|disney|hc-esub|esub|subs)\b',
        '', 
        normalized, 
        flags=re.IGNORECASE
    )
    
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
                    movie_db['title_cache'][text_hash] = title
                    return title
        
        # FALLBACK: Extract first meaningful words before quality indicators
        if len(first_line) >= 3:
            clean_title = re.sub(
                r'\b(480p|720p|1080p|2160p|4k|hd|fhd|uhd|hevc|x264|x265|h264|h265|bluray|webrip|hdrip|web-dl|hdtv|hdrip|webdl|hindi|english|tamil|telugu|malayalam|kannada|punjabi|bengali|marathi|gujarati|movie|film|series|complete|full|part|episode|season|\d{4}|hdrc|dvdscr|pre-dvd|p-dvd|pdc|rarbg|yts|amzn|netflix|hotstar|prime|disney|hc-esub|esub|subs)\b',
                '', 
                first_line, 
                flags=re.IGNORECASE
            )
            
            clean_title = re.sub(r'[\._\-]', ' ', clean_title)
            clean_title = re.sub(r'\s+', ' ', clean_title).strip()
            
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
            name = fn.rsplit('.', 1)[0]
            name = re.sub(r'[\._\-]', ' ', name)
            
            name = re.sub(
                r'\b(720p|1080p|480p|2160p|4k|HDRip|WEBRip|WEB-DL|BluRay|BRRip|DVDRip|HDTV|HDTC|X264|X265|HEVC|H264|H265|AAC|AC3|DD5\.1|DDP5\.1|HC|ESub|Subs|Hindi|English|Dual|Multi|Complete|Full|Movie|Film|\d{4})\b',
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
    semaphore = asyncio.Semaphore(3)
    
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
                        poster_data = await get_poster_guaranteed(title, session)
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

# OPTIMIZED VERIFICATION SYSTEM
async def check_url_shortener_verification(user_id):
    if not Config.VERIFICATION_REQUIRED:
        return True, "verification_not_required"
    
    try:
        verification = await verification_col.find_one({"user_id": user_id})
        if verification:
            verified_at = verification.get('verified_at')
            if isinstance(verified_at, datetime):
                time_elapsed = (datetime.now() - verified_at).total_seconds()
                if time_elapsed < Config.VERIFICATION_DURATION:
                    return True, "verified"
                else:
                    await verification_col.delete_one({"user_id": user_id})
                    return False, "expired"
        return False, "not_verified"
    except Exception as e:
        return False, "error"

async def verify_user_with_url_shortener(user_id, verification_url=None):
    if not Config.VERIFICATION_REQUIRED:
        return True, "verification_not_required"
    
    try:
        if not verification_url:
            verification_url = await generate_verification_url(user_id)
        
        async with aiohttp.ClientSession() as session:
            payload = {'user_id': user_id, 'verification_url': verification_url, 'api_key': Config.URL_SHORTENER_KEY}
            async with session.post(Config.URL_SHORTENER_API, json=payload, timeout=5) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get('verified') == True:
                        await verification_col.update_one(
                            {"user_id": user_id},
                            {"$set": {"verified_at": datetime.now(), "verification_url": verification_url, "verified_by": "url_shortener"}},
                            upsert=True
                        )
                        return True, "verified"
                    else:
                        return False, result.get('message', 'verification_failed')
                else:
                    return False, "api_error"
    except Exception as e:
        return False, "error"

async def generate_verification_url(user_id):
    base_url = Config.WEBSITE_URL or Config.BACKEND_URL
    verification_token = f"verify_{user_id}_{int(datetime.now().timestamp())}"
    return f"{base_url}/verify?token={verification_token}&user_id={user_id}"

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
        batch_size = 15
        thumbnail_batch = []
        
        # Get the last indexed message ID to start from there
        last_indexed = await files_col.find_one({}, sort=[('message_id', -1)])
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
                        'thumbnail': None,
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

# OPTIMIZED POSTER FETCHING WITH CONCURRENT REQUESTS
async def get_poster_letterboxd(title, session):
    try:
        clean_title = re.sub(r'[^\w\s]', '', title).strip()
        slug = clean_title.lower().replace(' ', '-')
        slug = re.sub(r'-+', '-', slug)
        
        patterns = [
            f"https://letterboxd.com/film/{slug}/",
            f"https://letterboxd.com/film/{slug}-2024/",
            f"https://letterboxd.com/film/{slug}-2023/",
        ]
        
        for url in patterns:
            try:
                async with session.get(url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'}) as r:
                    if r.status == 200:
                        html_content = await r.text()
                        poster_patterns = [
                            r'<meta property="og:image" content="([^"]+)"',
                            r'<img[^>]*class="[^"]*poster[^"]*"[^>]*src="([^"]+)"',
                        ]
                        
                        for pattern in poster_patterns:
                            poster_match = re.search(pattern, html_content)
                            if poster_match:
                                poster_url = poster_match.group(1)
                                if poster_url and poster_url.startswith('http'):
                                    if 'cloudfront.net' in poster_url:
                                        poster_url = poster_url.replace('-0-500-0-750', '-0-1000-0-1500')
                                    elif 's.ltrbxd.com' in poster_url:
                                        poster_url = poster_url.replace('/width/500/', '/width/1000/')
                                    
                                    rating_match = re.search(r'<meta name="twitter:data2" content="([^"]+)"', html_content)
                                    rating = rating_match.group(1) if rating_match else '0.0'
                                    
                                    res = {'poster_url': poster_url, 'source': 'Letterboxd', 'rating': rating}
                                    movie_db['stats']['letterboxd'] += 1
                                    return res
            except:
                continue
        return None
    except Exception as e:
        return None

async def get_poster_imdb(title, session):
    try:
        clean_title = re.sub(r'[^\w\s]', '', title).strip()
        search_url = f"https://v2.sg.media-imdb.com/suggestion/{clean_title[0].lower()}/{urllib.parse.quote(clean_title.replace(' ', '_'))}.json"
        
        async with session.get(search_url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'}) as r:
            if r.status == 200:
                data = await r.json()
                if data.get('d'):
                    for item in data['d']:
                        if item.get('i'):
                            poster_url = item['i'][0] if isinstance(item['i'], list) else item['i']
                            if poster_url and poster_url.startswith('http'):
                                poster_url = poster_url.replace('._V1_UX128_', '._V1_UX512_')
                                rating = str(item.get('yr', '0.0'))
                                res = {'poster_url': poster_url, 'source': 'IMDb', 'rating': rating}
                                movie_db['stats']['imdb'] += 1
                                return res
        return None
    except Exception as e:
        return None

async def get_poster_justwatch(title, session):
    try:
        clean_title = re.sub(r'[^\w\s]', '', title).strip()
        slug = clean_title.lower().replace(' ', '-')
        slug = re.sub(r'[^\w\-]', '', slug)
        
        domains = ['com', 'in', 'uk']
        
        for domain in domains:
            justwatch_url = f"https://www.justwatch.com/{domain}/movie/{slug}"
            try:
                async with session.get(justwatch_url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'}) as r:
                    if r.status == 200:
                        html_content = await r.text()
                        poster_match = re.search(r'<meta property="og:image" content="([^"]+)"', html_content)
                        if poster_match:
                            poster_url = poster_match.group(1)
                            if poster_url and poster_url.startswith('http'):
                                poster_url = poster_url.replace('http://', 'https://')
                                res = {'poster_url': poster_url, 'source': 'JustWatch', 'rating': '0.0'}
                                movie_db['stats']['justwatch'] += 1
                                return res
            except:
                continue
        return None
    except Exception as e:
        return None

async def get_poster_impawards(title, session):
    try:
        year_match = re.search(r'\b(19|20)\d{2}\b', title)
        if not year_match:
            return None
            
        year = year_match.group()
        clean_title = re.sub(r'\b(19|20)\d{2}\b', '', title).strip()
        clean_title = re.sub(r'[^\w\s]', '', clean_title).strip()
        slug = clean_title.lower().replace(' ', '_')
        
        formats = [
            f"https://www.impawards.com/{year}/posters/{slug}_xlg.jpg",
            f"https://www.impawards.com/{year}/posters/{slug}_ver7.jpg",
            f"https://www.impawards.com/{year}/posters/{slug}.jpg",
        ]
        
        for poster_url in formats:
            try:
                async with session.head(poster_url, timeout=3) as r:
                    if r.status == 200:
                        res = {'poster_url': poster_url, 'source': 'IMPAwards', 'rating': '0.0'}
                        movie_db['stats']['impawards'] += 1
                        return res
            except:
                continue
        return None
    except Exception as e:
        return None

async def get_poster_omdb_tmdb(title, session):
    try:
        for api_key in Config.OMDB_KEYS:
            try:
                url = f"http://www.omdbapi.com/?t={urllib.parse.quote(title)}&apikey={api_key}"
                async with session.get(url, timeout=5) as r:
                    if r.status == 200:
                        data = await r.json()
                        if data.get('Response') == 'True' and data.get('Poster') and data.get('Poster') != 'N/A':
                            poster_url = data['Poster'].replace('http://', 'https://')
                            res = {'poster_url': poster_url, 'source': 'OMDB', 'rating': data.get('imdbRating', '0.0')}
                            movie_db['stats']['omdb'] += 1
                            return res
            except:
                continue
        
        for api_key in Config.TMDB_KEYS:
            try:
                url = "https://api.themoviedb.org/3/search/movie"
                params = {'api_key': api_key, 'query': title}
                async with session.get(url, params=params, timeout=5) as r:
                    if r.status == 200:
                        data = await r.json()
                        if data.get('results') and len(data['results']) > 0:
                            result = data['results'][0]
                            poster_path = result.get('poster_path')
                            if poster_path:
                                poster_url = f"https://image.tmdb.org/t/p/w780{poster_path}"
                                res = {'poster_url': poster_url, 'source': 'TMDB', 'rating': str(result.get('vote_average', 0.0))}
                                movie_db['stats']['tmdb'] += 1
                                return res
            except:
                continue
        return None
    except Exception as e:
        return None

# OPTIMIZED POSTER FETCHING WITH CONCURRENT REQUESTS
async def get_poster_guaranteed(title, session):
    ck = title.lower().strip()
    
    if ck in movie_db['poster_cache']:
        c, ct = movie_db['poster_cache'][ck]
        if (datetime.now() - ct).seconds < 3600:
            movie_db['stats']['cache_hits'] += 1
            return c
    
    sources = [
        get_poster_letterboxd,
        get_poster_imdb, 
        get_poster_justwatch,
        get_poster_impawards,
        get_poster_omdb_tmdb,
    ]
    
    tasks = [source(title, session) for source in sources]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for result in results:
        if isinstance(result, dict) and result:
            movie_db['poster_cache'][ck] = (result, datetime.now())
            return result
    
    year_match = re.search(r'\b(19|20)\d{2}\b', title)
    year = year_match.group() if year_match else ""
    
    res = {
        'poster_url': f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(title)}&year={year}", 
        'source': 'CUSTOM', 
        'rating': '0.0'
    }
    movie_db['poster_cache'][ck] = (res, datetime.now())
    movie_db['stats']['custom'] += 1
    return res

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
    
    return unique_posts[:20]

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
        not x.get('is_new', False),
        not x['has_file'],
        x['date']
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
            tasks = [get_poster_guaranteed(movie['title'], session) for movie in movies]
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

# API Helper Functions
async def search_movies_api(query, limit=12, page=1):
    """Search movies API function"""
    offset = (page - 1) * limit
    movie_db['stats']['total_searches'] += 1
    
    # Try cache first
    cache_key = f"search:{query}:{page}:{limit}"
    if redis_cache.enabled:
        cached = await redis_cache.get(cache_key)
        if cached:
            try:
                return json.loads(cached)
            except:
                pass
    
    # Use the existing search function
    result_data = await search_movies_multi_channel(query, limit, page)
    
    # Enhance with posters
    async with aiohttp.ClientSession() as session:
        for result in result_data['results']:
            try:
                poster_data = await get_poster_guaranteed(result['title'], session)
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
    
    # Cache the result
    if redis_cache.enabled:
        await redis_cache.set(cache_key, json.dumps(result_data, default=str), expire=3600)
    
    return result_data

async def get_home_movies_api():
    """Get latest movies for homepage"""
    return await get_home_movies_live()

async def get_single_post_api(channel_id, message_id):
    """Get single movie/post details"""
    try:
        # Try to get the message
        if User and user_session_ready:
            msg = await safe_telegram_operation(
                User.get_messages,
                channel_id, 
                message_id
            )
            
            if msg and msg.text:
                title = extract_title_smart(msg.text)
                if not title:
                    title = msg.text.split('\n')[0][:60] if msg.text else "Movie Post"
                
                normalized_title = normalize_title(title)
                quality_options = {}
                has_file = False
                thumbnail_url = None
                thumbnail_source = None
                
                # Search for files with same title
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
                
                # Get poster
                poster_url = f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(title)}"
                async with aiohttp.ClientSession() as session:
                    poster_data = await get_poster_guaranteed(title, session)
                    if poster_data and poster_data.get('poster_url'):
                        poster_url = poster_data['poster_url']
                        poster_source = poster_data['source']
                        poster_rating = poster_data.get('rating', '0.0')
                    else:
                        poster_source = 'custom'
                        poster_rating = '0.0'
                
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
                    'thumbnail_source': thumbnail_source,
                    'poster_url': poster_url,
                    'poster_source': poster_source,
                    'poster_rating': poster_rating
                }
                
                return post_data
        
        # Fallback: return sample data
        return {
            'title': 'Sample Movie (2024)',
            'content': '🎬 <b>Sample Movie (2024)</b>\n📅 Release: 2024\n🎭 Genre: Action, Drama\n⭐ Starring: Popular Actors\n\n📥 Download now from SK4FiLM!',
            'channel': channel_name(channel_id),
            'channel_id': channel_id,
            'message_id': message_id,
            'date': datetime.now().isoformat(),
            'is_new': True,
            'has_file': True,
            'quality_options': {
                '1080p': {
                    'file_id': f'{channel_id}_{message_id}_1080p',
                    'file_size': 1500000000,
                    'file_name': 'Sample.Movie.2024.1080p.mkv',
                    'is_video': True
                }
            },
            'views': 1000,
            'thumbnail': None,
            'thumbnail_source': 'default',
            'poster_url': f"{Config.BACKEND_URL}/api/poster?title=Sample+Movie&year=2024",
            'poster_source': 'custom',
            'poster_rating': '7.5'
        }
        
    except Exception as e:
        logger.error(f"Single post API error: {e}")
        return None

async def verify_user_api(user_id, verification_url=None):
    """Verify user API function"""
    try:
        if Config.VERIFICATION_REQUIRED:
            if not verification_url:
                verification_url = await generate_verification_url(user_id)
                return {
                    'verified': False,
                    'verification_url': verification_url,
                    'message': 'Verification link created'
                }
            
            # Check if user is already verified
            is_verified, message = await check_url_shortener_verification(user_id)
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
            # Verification not required
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
        if files_col is not None:
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
            
            return {
                'total_indexed': total,
                'video_files': video_files,
                'video_thumbnails': video_thumbnails,
                'total_thumbnails': total_thumbnails,
                'thumbnail_coverage': f"{(video_thumbnails/video_files*100):.1f}%" if video_files > 0 else "0%",
                'last_indexed': last_indexed,
                'bot_status': bot_started,
                'user_session': user_session_ready,
                'redis_enabled': redis_cache.enabled,
                'cache_stats': movie_db['stats']
            }
        else:
            return {
                'total_indexed': 0,
                'video_files': 0,
                'video_thumbnails': 0,
                'total_thumbnails': 0,
                'thumbnail_coverage': "0%",
                'last_indexed': "Never",
                'bot_status': bot_started,
                'user_session': user_session_ready,
                'redis_enabled': redis_cache.enabled,
                'cache_stats': movie_db['stats']
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
    tf = await files_col.count_documents({}) if files_col is not None else 0
    video_files = await files_col.count_documents({'is_video_file': True}) if files_col is not None else 0
    thumbnails = await files_col.count_documents({'thumbnail': {'$ne': None}}) if files_col is not None else 0
    
    return jsonify({
        'status': 'healthy',
        'service': 'SK4FiLM v8.0 - COMPLETE API SYSTEM',
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
        'features': 'COMPLETE API SYSTEM + MULTI-CHANNEL SEARCH + REDIS CACHE',
        'api_endpoints': {
            'health': '/health',
            'movies': '/api/movies',
            'search': '/api/search?query=movie&page=1&limit=12',
            'post': '/api/post?channel=-1001891090100&message=12345',
            'poster': '/api/poster?title=Movie+Name&year=2024',
            'verify_user': '/api/verify_user (POST)',
            'clear_cache': '/api/clear_cache',
            'index_status': '/api/index_status'
        }
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

@app.route('/api/movies', methods=['GET'])
async def api_movies():
    """Home page movies"""
    try:
        movie_db['stats']['api_requests'] += 1
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
        
        movie_db['stats']['api_requests'] += 1
        
        # Perform search
        result_data = await search_movies_api(query, limit, page)
        
        return jsonify({
            'status': 'success',
            'query': query,
            'results': result_data['results'],
            'pagination': result_data['pagination'],
            'search_metadata': result_data.get('search_metadata', {}),
            'timestamp': datetime.now().isoformat(),
                        'bot_username': Config.BOT_USERNAME,
            'channels_searched': len(Config.TEXT_CHANNEL_IDS)
        })
    except Exception as e:
        logger.error(f"Search API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'query': query
        }), 500

@app.route('/api/post', methods=['GET'])
async def api_post():
    """Get single post details"""
    try:
        channel_id = int(request.args.get('channel', Config.MAIN_CHANNEL_ID))
        message_id = int(request.args.get('message', 0))
        
        if message_id <= 0:
            return jsonify({
                'status': 'error',
                'message': 'Invalid message ID'
            }), 400
        
        movie_db['stats']['api_requests'] += 1
        post_data = await get_single_post_api(channel_id, message_id)
        
        if post_data:
            return jsonify({
                'status': 'success',
                'post': post_data,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Post not found'
            }), 404
            
    except Exception as e:
        logger.error(f"Post API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/poster', methods=['GET'])
async def api_poster():
    """Get movie poster"""
    try:
        title = request.args.get('title', '').strip()
        year = request.args.get('year', '')
        
        if not title:
            return jsonify({
                'status': 'error',
                'message': 'Title is required'
            }), 400
        
        movie_db['stats']['api_requests'] += 1
        
        # Try to get poster from cache or external APIs
        async with aiohttp.ClientSession() as session:
            poster_data = await get_poster_guaranteed(title, session)
            
            if poster_data:
                return jsonify({
                    'status': 'success',
                    'poster': poster_data,
                    'title': title,
                    'year': year,
                    'timestamp': datetime.now().isoformat()
                })
            else:
                # Fallback to custom poster
                return jsonify({
                    'status': 'success',
                    'poster': {
                        'poster_url': f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(title)}&year={year}",
                        'source': 'CUSTOM',
                        'rating': '0.0'
                    },
                    'title': title,
                    'year': year,
                    'timestamp': datetime.now().isoformat()
                })
                
    except Exception as e:
        logger.error(f"Poster API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/verify_user', methods=['POST'])
async def api_verify_user():
    """Verify user with URL shortener"""
    try:
        data = await request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No JSON data provided'
            }), 400
        
        user_id = data.get('user_id')
        verification_url = data.get('verification_url')
        
        if not user_id:
            return jsonify({
                'status': 'error',
                'message': 'user_id is required'
            }), 400
        
        result = await verify_user_api(user_id, verification_url)
        
        return jsonify({
            'status': 'success',
            'verification': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Verify user API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/clear_cache', methods=['POST'])
async def api_clear_cache():
    """Clear all caches (admin only)"""
    try:
        # Check admin authentication
        data = await request.get_json()
        admin_key = data.get('admin_key') if data else request.headers.get('X-Admin-Key')
        
        if not admin_key or admin_key != os.environ.get('ADMIN_KEY', 'sk4film_admin_123'):
            return jsonify({
                'status': 'error',
                'message': 'Unauthorized'
            }), 401
        
        # Clear in-memory caches
        movie_db['poster_cache'].clear()
        movie_db['title_cache'].clear()
        movie_db['search_cache'].clear()
        
        # Clear Redis cache
        redis_cleared = await redis_cache.clear_search_cache()
        
        return jsonify({
            'status': 'success',
            'message': 'Cache cleared successfully',
            'caches_cleared': {
                'poster_cache': True,
                'title_cache': True,
                'search_cache': True,
                'redis_cache': redis_cleared
            },
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
    """Get indexing status"""
    try:
        status_data = await get_index_status_api()
        
        return jsonify({
            'status': 'success',
            'indexing': status_data,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Index status API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/stats', methods=['GET'])
async def api_stats():
    """Get system statistics"""
    try:
        # Calculate cache statistics
        cache_stats = movie_db['stats']
        
        # Get MongoDB statistics
        db_stats = {
            'total_files': await files_col.count_documents({}) if files_col else 0,
            'video_files': await files_col.count_documents({'is_video_file': True}) if files_col else 0,
            'files_with_thumbnails': await files_col.count_documents({'thumbnail': {'$ne': None}}) if files_col else 0,
            'unique_titles': len(await files_col.distinct('normalized_title')) if files_col else 0
        }
        
        # Calculate hit rates
        total_cache_access = cache_stats['redis_hits'] + cache_stats['redis_misses']
        redis_hit_rate = (cache_stats['redis_hits'] / total_cache_access * 100) if total_cache_access > 0 else 0
        
        # Poster source distribution
        poster_sources = {
            'letterboxd': cache_stats['letterboxd'],
            'imdb': cache_stats['imdb'],
            'justwatch': cache_stats['justwatch'],
            'impawards': cache_stats['impawards'],
            'omdb': cache_stats['omdb'],
            'tmdb': cache_stats['tmdb'],
            'custom': cache_stats['custom'],
            'cache_hits': cache_stats['cache_hits']
        }
        
        return jsonify({
            'status': 'success',
            'system': {
                'bot_status': bot_started,
                'user_session': user_session_ready,
                'redis_enabled': redis_cache.enabled,
                'channels_configured': len(Config.TEXT_CHANNEL_IDS),
                'uptime': time.time() - start_time if 'start_time' in globals() else 0
            },
            'database': db_stats,
            'cache': {
                'redis_hits': cache_stats['redis_hits'],
                'redis_misses': cache_stats['redis_misses'],
                'redis_hit_rate': f"{redis_hit_rate:.1f}%",
                'in_memory_hits': cache_stats['cache_hits'],
                'poster_cache_size': len(movie_db['poster_cache']),
                'title_cache_size': len(movie_db['title_cache']),
                'search_cache_size': len(movie_db['search_cache'])
            },
            'poster_sources': poster_sources,
            'search_stats': {
                'total_searches': cache_stats['total_searches'],
                'multi_channel_searches': cache_stats['multi_channel_searches'],
                'api_requests': cache_stats['api_requests'],
                'video_thumbnails': cache_stats['video_thumbnails']
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Stats API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# TELEGRAM BOT INITIALIZATION
async def init_bot():
    global bot, bot_started, User, user_session_ready
    start_time = time.time()
    
    try:
        # Initialize MongoDB first
        logger.info("🚀 Starting SK4FiLM v8.0 - MULTI-CHANNEL SYSTEM...")
        
        # Initialize Redis
        await redis_cache.init_redis()
        
        # Initialize MongoDB
        mongo_ok = await init_mongodb()
        if not mongo_ok:
            logger.error("❌ MongoDB initialization failed")
            return False
        
        # Initialize Pyrogram User Client
        logger.info("📱 Initializing User Session...")
        try:
            User = Client(
                name="user_session",
                api_id=Config.API_ID,
                api_hash=Config.API_HASH,
                session_string=Config.USER_SESSION_STRING,
                sleep_threshold=60,
                max_concurrent_transmissions=2
            )
            await User.start()
            logger.info("✅ User Session Started")
            
            # Verify channel access
            for channel_id in Config.TEXT_CHANNEL_IDS + [Config.FILE_CHANNEL_ID]:
                try:
                    chat = await User.get_chat(channel_id)
                    logger.info(f"✅ Access verified: {chat.title} ({channel_id})")
                    await asyncio.sleep(1)
                except Exception as e:
                    logger.warning(f"⚠️ Cannot access channel {channel_id}: {e}")
            
            user_session_ready = True
            
        except Exception as e:
            logger.error(f"❌ User Session Error: {e}")
            user_session_ready = False
        
        # Initialize Bot Client
        logger.info("🤖 Initializing Bot...")
        try:
            bot = Client(
                name="sk4film_bot",
                api_id=Config.API_ID,
                api_hash=Config.API_HASH,
                bot_token=Config.BOT_TOKEN,
                sleep_threshold=60
            )
            await bot.start()
            bot_started = True
            bot_info = await bot.get_me()
            logger.info(f"✅ Bot Started: @{bot_info.username}")
        except Exception as e:
            logger.error(f"❌ Bot Error: {e}")
            bot_started = False
        
        # Start background tasks
        asyncio.create_task(cache_cleanup())
        
        # Start indexing in background
        if user_session_ready:
            asyncio.create_task(index_files_background())
        
        logger.info(f"⚡ SK4FiLM Started in {time.time() - start_time:.1f}s")
        return True
        
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        return False

# STARTUP AND SHUTDOWN HANDLERS
@app.before_serving
async def startup():
    """Startup initialization"""
    await init_bot()

@app.after_serving
async def shutdown():
    """Clean shutdown"""
    logger.info("🛑 Shutting down SK4FiLM...")
    
    try:
        if User and user_session_ready:
            await User.stop()
            logger.info("✅ User Session Stopped")
    except:
        pass
    
    try:
        if bot and bot_started:
            await bot.stop()
            logger.info("✅ Bot Stopped")
    except:
        pass
    
    try:
        if mongo_client:
            mongo_client.close()
            logger.info("✅ MongoDB Connection Closed")
    except:
        pass
    
    try:
        if redis_cache.enabled and redis_cache.client:
            await redis_cache.client.close()
            logger.info("✅ Redis Connection Closed")
    except:
        pass
    
    logger.info("👋 Shutdown complete")

# MAIN ENTRY POINT
if __name__ == "__main__":
    config = HyperConfig()
    config.bind = [f"0.0.0.0:{Config.WEB_SERVER_PORT}"]
    config.worker_class = "asyncio"
    config.workers = 1
    config.accesslog = "-"
    config.errorlog = "-"
    config.loglevel = "info"
    
    logger.info(f"🌐 Starting Quart server on port {Config.WEB_SERVER_PORT}...")
    
    # Run the app
    asyncio.run(serve(app, config))
