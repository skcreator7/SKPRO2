# app.py - SK4FiLM v9.0 - Complete System with Premium, Verification & Cache

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

# Import custom modules
from premium import PremiumManager, setup_premium_handlers
from verification import VerificationManager, setup_verification_handlers
from cache import CacheManager

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
    
    # CHANNEL CONFIGURATION
    MAIN_CHANNEL_ID = -1001891090100
    TEXT_CHANNEL_IDS = [-1001891090100, -1002024811395]
    FILE_CHANNEL_ID = -1001768249569
    MAIN_CHANNEL_LINK = "https://t.me/sk4film"
    UPDATES_CHANNEL_LINK = "https://t.me/sk4film_Request"
    CHANNEL_USERNAME = "sk4film"
    
    # SHORTLINK & VERIFICATION
    SHORTLINK_API = os.environ.get("SHORTLINK_API", "")
    URL_SHORTENER_API = os.environ.get("URL_SHORTENER_API", "https://your-shortener-api.com/verify")
    URL_SHORTENER_KEY = os.environ.get("URL_SHORTENER_KEY", "")
    VERIFICATION_REQUIRED = os.environ.get("VERIFICATION_REQUIRED", "false").lower() == "true"
    VERIFICATION_DURATION = 6  # Hours
    
    # WEBSITE & BOT
    WEBSITE_URL = os.environ.get("WEBSITE_URL", "https://sk4film.vercel.app")
    BOT_USERNAME = os.environ.get("BOT_USERNAME", "sk4filmbot")
    ADMIN_IDS = [int(x) for x in os.environ.get("ADMIN_IDS", "123456789").split(",")]
    AUTO_DELETE_TIME = int(os.environ.get("AUTO_DELETE_TIME", "300"))
    WEB_SERVER_PORT = int(os.environ.get("PORT", 8000))
    BACKEND_URL = os.environ.get("BACKEND_URL", "https://sk4film.koyeb.app")
    
    # API KEYS
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

# GLOBAL VARIABLES
mongo_client = None
db = None
files_col = None
verification_col = None
premium_col = None
transactions_col = None
User = None
bot = None
bot_started = False
user_session_ready = False

# Module managers
premium_manager = None
verification_manager = None
cache_manager = None

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
        'search_priority': 0
    }
}

# REDIS CACHE MANAGER (Basic - Extended version in cache.py)
class RedisCache:
    def __init__(self):
        self.client = None
        self.enabled = False
    
    async def init_redis(self):
        try:
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
            
            await self.client.ping()
            self.enabled = True
            logger.info("✅ Redis connected successfully!")
            
            await self.client.set('connection_test', 'success', ex=60)
            test_result = await self.client.get('connection_test')
            
            if test_result == 'success':
                logger.info("✅ Redis operations test: PASSED")
            
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

redis_cache = RedisCache()

# FLOOD WAIT PROTECTION
class FloodWaitProtection:
    def __init__(self):
        self.last_request_time = 0
        self.min_interval = 3
        self.request_count = 0
        self.reset_time = time.time()
        self.consecutive_waits = 0
    
    async def wait_if_needed(self):
        current_time = time.time()
        
        if current_time - self.reset_time > 120:
            self.request_count = 0
            self.reset_time = current_time
            self.consecutive_waits = 0
        
        if self.request_count >= 20:
            wait_time = 120 - (current_time - self.reset_time)
            if wait_time > 0:
                self.consecutive_waits += 1
                extra_wait = self.consecutive_waits * 5
                total_wait = wait_time + extra_wait
                logger.warning(f"⚠️ Rate limit reached, waiting {total_wait:.1f}s")
                await asyncio.sleep(total_wait)
                self.request_count = 0
                self.reset_time = time.time()
        
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_interval:
            wait_time = self.min_interval - time_since_last
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()
        self.request_count += 1

flood_protection = FloodWaitProtection()

# SAFE TELEGRAM OPERATIONS
async def safe_telegram_operation(operation, *args, **kwargs):
    max_retries = 2
    base_delay = 5
    
    for attempt in range(max_retries):
        try:
            await flood_protection.wait_if_needed()
            result = await operation(*args, **kwargs)
            
            if flood_protection.consecutive_waits > 0:
                flood_protection.consecutive_waits = 0
            
            return result
        
        except FloodWait as e:
            wait_time = e.value + 10
            logger.warning(f"⚠️ Flood wait: {e.value}s -> waiting {wait_time}s")
            
            if attempt == max_retries - 1:
                logger.error(f"❌ Flood wait too long after {max_retries} attempts")
                raise e
            
            await asyncio.sleep(wait_time)
            
            if attempt > 0:
                await asyncio.sleep(attempt * 10)
        
        except Exception as e:
            logger.error(f"❌ Telegram operation failed: {e}")
            if attempt == max_retries - 1:
                raise e
            await asyncio.sleep(base_delay * (2 ** attempt))
    
    return None

async def safe_telegram_generator(operation, *args, limit=None, **kwargs):
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
        
        except FloodWait as e:
            wait_time = e.value + 10
            logger.warning(f"⚠️ Flood wait in generator: {e.value}s")
            
            if attempt == max_retries - 1:
                raise e
            
            await asyncio.sleep(wait_time)
            
            if attempt > 0:
                await asyncio.sleep(attempt * 10)
        
        except Exception as e:
            logger.error(f"❌ Telegram generator failed: {e}")
            if attempt == max_retries - 1:
                raise e
            await asyncio.sleep(5 * (2 ** attempt))

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

# MONGODB INITIALIZATION
async def init_mongodb():
    global mongo_client, db, files_col, verification_col, premium_col, transactions_col
    
    try:
        logger.info("🔌 MongoDB initialization...")
        
        mongo_client = AsyncIOMotorClient(
            Config.MONGODB_URI,
            serverSelectionTimeoutMS=5000,
            maxPoolSize=10
        )
        
        await mongo_client.admin.command('ping')
        
        db = mongo_client.sk4film
        files_col = db.files
        verification_col = db.verifications
        premium_col = db.premium_users
        transactions_col = db.transactions
        
        # Create indexes
        logger.info("🔧 Creating indexes...")
        
        existing_indexes = await files_col.index_information()
        
        if 'title_text' not in existing_indexes:
            try:
                await files_col.create_index([("title", "text")])
                logger.info("✅ Created title text index")
            except Exception as e:
                logger.warning(f"⚠️ Title index: {e}")
        
        if 'normalized_title_1' not in existing_indexes:
            try:
                await files_col.create_index([("normalized_title", 1)])
                logger.info("✅ Created normalized_title index")
            except Exception as e:
                logger.warning(f"⚠️ Normalized title index: {e}")
        
        if 'msg_ch_idx' not in existing_indexes:
            try:
                await files_col.create_index(
                    [("message_id", 1), ("channel_id", 1)],
                    name="msg_ch_idx"
                )
                logger.info("✅ Created message_channel index")
            except Exception as e:
                logger.warning(f"⚠️ Message channel index: {e}")
        
        if 'indexed_at_-1' not in existing_indexes:
            try:
                await files_col.create_index([("indexed_at", -1)])
                logger.info("✅ Created indexed_at index")
            except Exception as e:
                logger.warning(f"⚠️ Indexed_at index: {e}")
        
        logger.info("✅ MongoDB OK")
        return True
    
    except Exception as e:
        logger.error(f"❌ MongoDB error: {e}")
        return False

# TITLE UTILITIES
def normalize_title(title):
    if not title:
        return ""
    
    normalized = title.lower().strip()
    
    normalized = re.sub(
        r'\b(19|20)\d{2}\b|\b(480p|720p|1080p|2160p|4k|hd|fhd|uhd|hevc|x264|x265|h264|h265|bluray|webrip|hdrip|web-dl|hdtv|hindi|english|tamil|telugu|malayalam|kannada|punjabi|bengali|marathi|gujarati|movie|film|series|complete|full|part|episode|season|hdrc|dvdscr|rarbg|yts|amzn|netflix|hotstar|prime|disney|hc-esub|esub|subs)\b',
        '',
        normalized,
        flags=re.IGNORECASE
    )
    
    normalized = re.sub(r'[\._\-]', ' ', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized

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
        
        if len(first_line) >= 3:
            clean_title = re.sub(
                r'\b(480p|720p|1080p|2160p|4k|hd|fhd|uhd|hevc|x264|x265|h264|h265|bluray|webrip|hdrip|web-dl|hdtv|hindi|english|tamil|telugu|malayalam|kannada|punjabi|bengali|marathi|gujarati|movie|film|series|complete|full|part|episode|season|\d{4}|hdrc|dvdscr|rarbg|yts|amzn|netflix|hotstar|prime|disney|hc-esub|esub|subs)\b',
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

# UTILITY FUNCTIONS
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

# INDEXING SYSTEM
async def index_files_background():
    if not User or files_col is None or not user_session_ready:
        logger.warning("⚠️ Cannot start indexing - User session not ready")
        return
    
    logger.info("📁 Starting background indexing...")
    
    try:
        total_count = 0
        new_files_count = 0
        batch = []
        batch_size = 15
        
        last_indexed = await files_col.find_one({}, sort=[('message_id', -1)])
        last_message_id = last_indexed['message_id'] if last_indexed else 0
        
        logger.info(f"🔄 Starting from message ID: {last_message_id}")
        
        processed_count = 0
        
        async for msg in safe_telegram_generator(User.get_chat_history, Config.FILE_CHANNEL_ID):
            processed_count += 1
            
            if msg.id <= last_message_id:
                if processed_count % 100 == 0:
                    logger.info(f"⏩ Skipped {processed_count} old messages...")
                continue
            
            if msg and (msg.document or msg.video):
                existing = await files_col.find_one({
                    'channel_id': Config.FILE_CHANNEL_ID,
                    'message_id': msg.id
                })
                
                if existing:
                    continue
                
                title = extract_title_from_file(msg)
                
                if title:
                    file_id = msg.document.file_id if msg.document else msg.video.file_id
                    file_size = msg.document.file_size if msg.document else (msg.video.file_size if msg.video else 0)
                    file_name = msg.document.file_name if msg.document else (msg.video.file_name if msg.video else 'video.mp4')
                    quality = detect_quality(file_name)
                    file_is_video = is_video_file(file_name)
                    
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
                            
                            logger.info(f"✅ Batch processed: {new_files_count} new files")
                            batch = []
                            await asyncio.sleep(3)
                        
                        except Exception as e:
                            logger.error(f"Batch error: {e}")
                            batch = []
        
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
        
        logger.info(f"✅ Indexing finished: {new_files_count} NEW files")
        
        await redis_cache.clear_search_cache()
        logger.info("🧹 Search cache cleared")
    
    except Exception as e:
        logger.error(f"❌ Indexing error: {e}")

# POSTER FETCHING (Simplified - keep your original implementation)
async def get_poster_guaranteed(title, session):
    # Your existing poster fetching logic
    year_match = re.search(r'\b(19|20)\d{2}\b', title)
    year = year_match.group() if year_match else ""
    
    return {
        'poster_url': f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(title)}&year={year}",
        'source': 'CUSTOM',
        'rating': '0.0'
    }

# SEARCH SYSTEM
async def search_movies_multi_channel(query, limit=12, page=1):
    offset = (page - 1) * limit
    
    cache_key = f"search:{query}:{page}:{limit}"
    
    if redis_cache.enabled:
        cached_data = await redis_cache.get(cache_key)
        if cached_data:
            try:
                return json.loads(cached_data)
            except:
                pass
    
    movie_db['stats']['multi_channel_searches'] += 1
    logger.info(f"🔍 Search for: {query}")
    
    files_dict = {}
    
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
                        
                        files_dict[norm_title] = {
                            'title': doc['title'],
                            'quality_options': {},
                            'date': doc['date'].isoformat() if isinstance(doc['date'], datetime) else doc['date'],
                            'thumbnail': doc.get('thumbnail'),
                            'is_video_file': file_is_video,
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
    
    results_list = list(files_dict.values())
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
        }
    }
    
    if redis_cache.enabled:
        await redis_cache.set(cache_key, json.dumps(result_data, default=str), expire=3600)
    
    return result_data

# API ROUTES
@app.route('/')
async def root():
    tf = await files_col.count_documents({}) if files_col is not None else 0
    video_files = await files_col.count_documents({'is_video_file': True}) if files_col is not None else 0
    
    # Get premium stats
    premium_count = 0
    if premium_manager:
        premium_count = await premium_manager.get_premium_users_count()
    
    return jsonify({
        'status': 'healthy',
        'service': 'SK4FiLM v9.0 - Premium + Verification System',
        'database': {
            'total_files': tf,
            'video_files': video_files,
            'premium_users': premium_count
        },
        'features': [
            'Multi-Channel Search',
            'Premium Plans (5 tiers)',
            'Free Verification (6h)',
            'UPI Payment System',
            'Auto Cleanup',
            'Redis Cache',
            'Link Shortener Integration'
        ],
        'bot_status': 'online' if bot_started else 'starting',
        'user_session': 'ready' if user_session_ready else 'starting'
    })

@app.route('/health')
async def health():
    return jsonify({
        'status': 'ok' if bot_started else 'starting',
        'user_session': user_session_ready,
        'redis_enabled': redis_cache.enabled,
        'premium_enabled': premium_manager is not None,
        'verification_enabled': verification_manager is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/search', methods=['GET'])
async def api_search():
    try:
        query = request.args.get('query', '')
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 12))
        
        if not query:
            return jsonify({'error': 'Query parameter required'}), 400
        
        results = await search_movies_multi_channel(query, limit, page)
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/file/<file_id>', methods=['GET'])
async def get_file(file_id):
    try:
        user_id = request.args.get('user_id', type=int)
        
        if not user_id:
            return jsonify({'error': 'user_id required'}), 400
        
        # Check if user is verified or has premium
        is_verified = False
        access_type = 'none'
        
        if verification_manager:
            is_verified, status = await verification_manager.check_verification_status(user_id)
            if is_verified:
                access_type = status['type']  # 'premium' or 'verified'
        
        if not is_verified:
            return jsonify({
                'error': 'Verification required',
                'message': 'Please verify or buy premium to download files'
            }), 403
        
        # Parse file_id (format: channel_id_message_id_quality)
        parts = file_id.split('_')
        if len(parts) < 2:
            return jsonify({'error': 'Invalid file_id'}), 400
        
        channel_id = int(parts[0])
        message_id = int(parts[1])
        
        # Generate download link
        bot_username = Config.BOT_USERNAME
        download_url = f"https://t.me/{bot_username}?start=file_{channel_id}_{message_id}"
        
        return jsonify({
            'success': True,
            'download_url': download_url,
            'access_type': access_type,
            'file_id': file_id
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/premium/status/<int:user_id>', methods=['GET'])
async def premium_status(user_id):
    try:
        if not premium_manager:
            return jsonify({'error': 'Premium system not available'}), 503
        
        is_premium, data = await premium_manager.check_premium_status(user_id)
        
        if is_premium:
            return jsonify({
                'has_premium': True,
                'plan': data.get('plan_type'),
                'expires_at': data.get('expires_at').isoformat() if data.get('expires_at') else None
            })
        else:
            return jsonify({'has_premium': False})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/verification/status/<int:user_id>', methods=['GET'])
async def verification_status(user_id):
    try:
        if not verification_manager:
            return jsonify({'error': 'Verification system not available'}), 503
        
        is_verified, status = await verification_manager.check_verification_status(user_id)
        
        if is_verified:
            return jsonify({
                'verified': True,
                'type': status['type'],  # 'premium' or 'verified'
                'expires_at': status['data'].get('expires_at').isoformat() if status['data'].get('expires_at') else None
            })
        else:
            return jsonify({'verified': False})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# BOT HANDLERS
@bot.on_message(filters.command("start") & filters.private)
async def start_handler(client, message):
    # This will be overridden by verification.py setup_verification_handlers
    pass

@bot.on_message(filters.command("stats") & filters.user(Config.ADMIN_IDS))
async def stats_handler(client, message):
    tf = await files_col.count_documents({}) if files_col is not None else 0
    video_files = await files_col.count_documents({'is_video_file': True}) if files_col is not None else 0
    
    premium_count = 0
    if premium_manager:
        premium_count = await premium_manager.get_premium_users_count()
    
    stats_text = f"""
📊 **SK4FiLM STATISTICS**

**Database:**
📁 Total Files: {tf}
🎬 Video Files: {video_files}
💎 Premium Users: {premium_count}

**System:**
🤖 Bot Status: {'Online' if bot_started else 'Starting'}
👤 User Session: {'Ready' if user_session_ready else 'Flood Wait'}
💾 Redis Cache: {'Enabled' if redis_cache.enabled else 'Disabled'}

**Cache Stats:**
🔍 Multi-channel Searches: {movie_db['stats']['multi_channel_searches']}
✅ Redis Hits: {movie_db['stats']['redis_hits']}
❌ Redis Misses: {movie_db['stats']['redis_misses']}
"""
    
    await message.reply_text(stats_text)

@bot.on_message(filters.command("index") & filters.user(Config.ADMIN_IDS))
async def index_handler(client, message):
    msg = await message.reply_text("Starting background indexing...")
    asyncio.create_task(index_files_background())
    await msg.edit_text("✅ Indexing started in background!")

@bot.on_message(filters.command("clearcache") & filters.user(Config.ADMIN_IDS))
async def clearcache_handler(client, message):
    msg = await message.reply_text("Clearing all caches...")
    
    redis_cleared = await redis_cache.clear_search_cache()
    
    movie_db['search_cache'].clear()
    movie_db['poster_cache'].clear()
    movie_db['title_cache'].clear()
    movie_db['stats']['redis_hits'] = 0
    movie_db['stats']['redis_misses'] = 0
    
    await msg.edit_text(
        f"✅ **All caches cleared!**\n\n"
        f"Redis cache: {'Cleared' if redis_cleared else 'Failed'}\n"
        f"Memory cache: Cleared\n"
        f"Search cache: Cleared\n"
        f"Poster cache: Cleared"
    )

# INITIALIZATION
async def init():
    global User, bot, bot_started, user_session_ready
    global premium_manager, verification_manager, cache_manager
    
    try:
        logger.info("="*60)
        logger.info("SK4FiLM v9.0 - PREMIUM + VERIFICATION SYSTEM")
        logger.info("="*60)
        
        # Initialize MongoDB
        await init_mongodb()
        
        # Initialize Redis
        await redis_cache.init_redis()
        
        # Initialize Cache Manager
        cache_manager = CacheManager(Config)
        await cache_manager.init_redis()
        
        # Initialize Premium Manager
        premium_manager = PremiumManager(Config, db)
        await premium_manager.init_indexes()
        logger.info("✅ Premium Manager initialized")
        
        # Initialize Verification Manager
        verification_manager = VerificationManager(Config, db, premium_manager)
        await verification_manager.init_indexes()
        logger.info("✅ Verification Manager initialized")
        
        # Initialize Bot
        global bot
        bot = Client(
            "sk4film_bot",
            api_id=Config.API_ID,
            api_hash=Config.API_HASH,
            bot_token=Config.BOT_TOKEN,
            no_updates=False
        )
        
        await bot.start()
        bot_started = True
        logger.info("✅ Bot started")
        
        # Setup Premium Handlers
        await setup_premium_handlers(bot, premium_manager, Config)
        
        # Setup Verification Handlers
        await setup_verification_handlers(bot, verification_manager, premium_manager, Config)
        
        # Initialize User Session
        User = Client(
            "user_session",
            api_id=Config.API_ID,
            api_hash=Config.API_HASH,
            session_string=Config.USER_SESSION_STRING,
            no_updates=True
        )
        
        try:
            await User.start()
            user_session_ready = True
            logger.info("✅ User session ready")
        except FloodWait as e:
            logger.warning(f"⚠️ User session flood wait: {e.value}s")
            asyncio.create_task(user_session_recovery())
        except Exception as e:
            logger.error(f"❌ User session error: {e}")
        
        # Start auto cleanup
        await cache_manager.start_auto_cleanup(premium_manager, verification_manager)
        
        # Start cache cleanup
        asyncio.create_task(cache_cleanup())
        
        # Start background indexing
        if user_session_ready:
            asyncio.create_task(index_files_background())
        
        logger.info("="*60)
        logger.info("✅ SYSTEM FULLY INITIALIZED")
        logger.info(f"💎 Premium Plans: {len(premium_manager.PREMIUM_PLANS)}")
        logger.info(f"🔓 Verification: {'Enabled' if Config.VERIFICATION_REQUIRED else 'Disabled'}")
        logger.info(f"💾 Redis Cache: {'Enabled' if redis_cache.enabled else 'Disabled'}")
        logger.info("="*60)
        
        return True
    
    except Exception as e:
        logger.error(f"❌ INIT FAILED: {e}")
        return False

async def user_session_recovery():
    global User, user_session_ready
    
    logger.info("🔄 User session recovery task started...")
    await asyncio.sleep(2400)  # Wait 40 minutes
    
    try:
        logger.info("🔄 Attempting user session recovery...")
        await User.start()
        user_session_ready = True
        logger.info("✅ User session recovered!")
        
        # Start indexing after recovery
        asyncio.create_task(index_files_background())
    
    except Exception as e:
        logger.error(f"❌ User session recovery failed: {e}")

# MAIN
async def main():
    success = await init()
    
    if not success:
        logger.error("Failed to initialize system")
        return
    
    # Start web server
    config = HyperConfig()
    config.bind = f"0.0.0.0:{Config.WEB_SERVER_PORT}"
    config.loglevel = "warning"
    
    logger.info(f"🌐 Web server starting on port {Config.WEB_SERVER_PORT}...")
    
    await serve(app, config)

if __name__ == "__main__":
    asyncio.run(main())
