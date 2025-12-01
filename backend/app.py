# app.py - SK4FiLM v9.0 - Fixed Bot Initialization

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
from poster_fetching import init_poster_manager

# LOGGING SETUP
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

# QUART APP
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
poster_manager = None

# CACHE SYSTEM
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

# CHANNEL CONFIG
CHANNEL_CONFIG = {
    -1001891090100: {'name': 'SK4FiLM Main', 'type': 'text', 'search_priority': 1},
    -1002024811395: {'name': 'SK4FiLM Updates', 'type': 'text', 'search_priority': 2},
    -1001768249569: {'name': 'SK4FiLM Files', 'type': 'file', 'search_priority': 0}
}

# REDIS CACHE
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
            logger.info("✅ Redis connected!")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Redis failed: {e}")
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
        except:
            return None
    
    async def set(self, key, value, expire=3600):
        if not self.enabled or not self.client:
            return False
        try:
            await self.client.setex(key, expire, value)
            return True
        except:
            return False
    
    async def delete(self, key):
        if not self.enabled or not self.client:
            return False
        try:
            await self.client.delete(key)
            return True
        except:
            return False
    
    async def clear_search_cache(self):
        if not self.enabled or not self.client:
            return False
        try:
            keys = await self.client.keys("search:*")
            if keys:
                await self.client.delete(*keys)
                logger.info(f"🧹 Cleared {len(keys)} cache keys")
            return True
        except:
            return False

redis_cache = RedisCache()

# FLOOD PROTECTION
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
                total_wait = wait_time + (self.consecutive_waits * 5)
                logger.warning(f"⚠️ Rate limit, waiting {total_wait:.1f}s")
                await asyncio.sleep(total_wait)
                self.request_count = 0
                self.reset_time = time.time()
        
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_interval:
            await asyncio.sleep(self.min_interval - time_since_last)
        
        self.last_request_time = time.time()
        self.request_count += 1

flood_protection = FloodWaitProtection()

# SAFE TELEGRAM OPERATIONS
async def safe_telegram_operation(operation, *args, **kwargs):
    max_retries = 2
    for attempt in range(max_retries):
        try:
            await flood_protection.wait_if_needed()
            result = await operation(*args, **kwargs)
            return result
        except FloodWait as e:
            wait_time = e.value + 10
            logger.warning(f"⚠️ Flood wait: {wait_time}s")
            if attempt == max_retries - 1:
                raise e
            await asyncio.sleep(wait_time)
        except Exception as e:
            logger.error(f"❌ Operation failed: {e}")
            if attempt == max_retries - 1:
                raise e
            await asyncio.sleep(5 * (2 ** attempt))
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
            if attempt == max_retries - 1:
                raise e
            await asyncio.sleep(wait_time)
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            await asyncio.sleep(5 * (2 ** attempt))

# CACHE CLEANUP
async def cache_cleanup():
    while True:
        await asyncio.sleep(3600)
        try:
            current_time = datetime.now()
            
            expired = [k for k, (d, t) in movie_db['poster_cache'].items() 
                      if (current_time - t).seconds > 3600]
            for k in expired:
                del movie_db['poster_cache'][k]
            
            expired_search = [k for k, (d, t) in movie_db['search_cache'].items() 
                             if (current_time - t).seconds > 1800]
            for k in expired_search:
                del movie_db['search_cache'][k]
            
            logger.info(f"🧹 Cache cleaned: {len(expired)} posters, {len(expired_search)} searches")
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")

# MONGODB INIT
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
        
        logger.info("🔧 Creating indexes...")
        
        existing = await files_col.index_information()
        
        if 'title_text' not in existing:
            await files_col.create_index([("title", "text")])
        
        if 'normalized_title_1' not in existing:
            await files_col.create_index([("normalized_title", 1)])
        
        if 'msg_ch_idx' not in existing:
            await files_col.create_index([("message_id", 1), ("channel_id", 1)], name="msg_ch_idx")
        
        if 'indexed_at_-1' not in existing:
            await files_col.create_index([("indexed_at", -1)])
        
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
        r'\b(19|20)\d{2}\b|\b(480p|720p|1080p|2160p|4k|hd|fhd|uhd|hevc|x264|x265|bluray|webrip|hdrip|web-dl|hdtv|hindi|english|tamil|telugu|movie|film|series|complete|full|part|episode|season|rarbg|yts|amzn|netflix|hotstar|prime|disney|esub|subs)\b',
        '', normalized, flags=re.IGNORECASE
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
            (r'^([A-Za-z\s]{3,50}?)\s*(?:\(?\d{4}\)?|\b(?:480p|720p|1080p|2160p|4k)\b)', 1),
            (r'🎬\s*([^\n\-\(]{3,60}?)\s*(?:\(\d{4}\)|$)', 1),
            (r'^([^\-\n]{3,60}?)\s*\-', 1),
            (r'^([^\(\n]{3,60}?)\s*\(\d{4}\)', 1),
        ]
        
        for pattern, group in patterns:
            match = re.search(pattern, first_line, re.IGNORECASE)
            if match:
                title = match.group(group).strip()
                title = re.sub(r'\s+', ' ', title)
                if 3 <= len(title) <= 60:
                    movie_db['title_cache'][text_hash] = title
                    return title
        
        clean_title = re.sub(
            r'\b(480p|720p|1080p|2160p|4k|hd|hevc|x264|x265|bluray|webrip|hdrip|web-dl|hdtv|hindi|english|movie|film|series|\d{4}|rarbg|yts|netflix|hotstar|esub)\b',
            '', first_line, flags=re.IGNORECASE
        )
        clean_title = re.sub(r'[\._\-]', ' ', clean_title)
        clean_title = re.sub(r'\s+', ' ', clean_title).strip()
        clean_title = re.sub(r'\s+\d{4}$', '', clean_title)
        
        if 3 <= len(clean_title) <= 60:
            movie_db['title_cache'][text_hash] = clean_title
            return clean_title
    
    except Exception as e:
        logger.error(f"Title extraction error: {e}")
    
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
                r'\b(720p|1080p|480p|2160p|4k|HDRip|WEBRip|BluRay|HDTV|HDTC|X264|X265|HEVC|AAC|Hindi|English|Movie|Film|\d{4})\b',
                '', name, flags=re.IGNORECASE
            )
            name = re.sub(r'\s+', ' ', name).strip()
            name = re.sub(r'\s+\d{4}$', '', name)
            
            if 4 <= len(name) <= 50:
                return name
    
    except Exception as e:
        logger.error(f"File title error: {e}")
    
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
    return "480p"

def format_post(text):
    if not text:
        return ""
    text = html.escape(text)
    text = re.sub(r'(https?://[^\s]+)', r'<a href="\1">\1</a>', text)
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
    video_ext = ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v']
    return any(file_name.lower().endswith(ext) for ext in video_ext)

# INDEXING
async def index_files_background():
    if not User or files_col is None or not user_session_ready:
        logger.warning("⚠️ Cannot start indexing - User session not ready")
        return
    
    logger.info("📁 Starting background indexing...")
    
    try:
        new_files = 0
        batch = []
        batch_size = 15
        
        last_indexed = await files_col.find_one({}, sort=[('message_id', -1)])
        last_message_id = last_indexed['message_id'] if last_indexed else 0
        
        logger.info(f"🔄 Starting from message ID: {last_message_id}")
        
        async for msg in safe_telegram_generator(User.get_chat_history, Config.FILE_CHANNEL_ID):
            if msg.id <= last_message_id:
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
                    
                    batch.append({
                        'channel_id': Config.FILE_CHANNEL_ID,
                        'message_id': msg.id,
                        'title': title,
                        'normalized_title': normalize_title(title),
                        'file_id': file_id,
                        'quality': detect_quality(file_name),
                        'file_size': file_size,
                        'file_name': file_name,
                        'caption': msg.caption or '',
                        'date': msg.date,
                        'indexed_at': datetime.now(),
                        'is_video_file': is_video_file(file_name)
                    })
                    
                    new_files += 1
                    
                    if len(batch) >= batch_size:
                        for doc in batch:
                            await files_col.update_one(
                                {'channel_id': doc['channel_id'], 'message_id': doc['message_id']},
                                {'$set': doc},
                                upsert=True
                            )
                        logger.info(f"✅ Batch: {new_files} new files")
                        batch = []
                        await asyncio.sleep(3)
        
        if batch:
            for doc in batch:
                await files_col.update_one(
                    {'channel_id': doc['channel_id'], 'message_id': doc['message_id']},
                    {'$set': doc},
                    upsert=True
                )
        
        logger.info(f"✅ Indexing done: {new_files} NEW files")
        await redis_cache.clear_search_cache()
    
    except Exception as e:
        logger.error(f"❌ Indexing error: {e}")

# SEARCH
async def search_movies_multi_channel(query, limit=12, page=1):
    offset = (page - 1) * limit
    cache_key = f"search:{query}:{page}:{limit}"
    
    if redis_cache.enabled:
        cached = await redis_cache.get(cache_key)
        if cached:
            try:
                return json.loads(cached)
            except:
                pass
    
    movie_db['stats']['multi_channel_searches'] += 1
    files_dict = {}
    
    try:
        if files_col is not None:
            cursor = files_col.find({'$text': {'$search': query}})
            
            async for doc in cursor:
                try:
                    norm_title = doc.get('normalized_title', normalize_title(doc['title']))
                    quality = doc['quality']
                    
                    if norm_title not in files_dict:
                        files_dict[norm_title] = {
                            'title': doc['title'],
                            'quality_options': {},
                            'date': doc['date'].isoformat() if isinstance(doc['date'], datetime) else doc['date'],
                            'is_video_file': doc.get('is_video_file', False),
                            'channel_id': doc.get('channel_id'),
                            'channel_name': channel_name(doc.get('channel_id'))
                        }
                    
                    files_dict[norm_title]['quality_options'][quality] = {
                        'file_id': f"{doc.get('channel_id', Config.FILE_CHANNEL_ID)}_{doc.get('message_id')}_{quality}",
                        'file_size': doc['file_size'],
                        'file_name': doc['file_name'],
                        'channel_id': doc.get('channel_id'),
                        'message_id': doc.get('message_id')
                    }
                except:
                    continue
    except Exception as e:
        logger.error(f"Search error: {e}")
    
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
    tf = await files_col.count_documents({}) if files_col else 0
    premium_count = await premium_manager.get_premium_users_count() if premium_manager else 0
    
    return jsonify({
        'status': 'healthy',
        'service': 'SK4FiLM v9.0 - Premium System',
        'database': {'total_files': tf, 'premium_users': premium_count},
        'features': ['Multi-Channel', 'Premium Plans', 'Free Verification', 'UPI Payment', 'Link Shortener'],
        'bot_status': 'online' if bot_started else 'starting'
    })

@app.route('/health')
async def health():
    return jsonify({
        'status': 'ok' if bot_started else 'starting',
        'user_session': user_session_ready,
        'redis': redis_cache.enabled,
        'premium': premium_manager is not None,
        'verification': verification_manager is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/search', methods=['GET'])
async def api_search():
    try:
        query = request.args.get('query', '')
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 12))
        
        if not query:
            return jsonify({'error': 'Query required'}), 400
        
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
        
        is_verified = False
        access_type = 'none'
        
        if verification_manager:
            is_verified, status = await verification_manager.check_verification_status(user_id)
            if is_verified:
                access_type = status['type']
        
        if not is_verified:
            return jsonify({
                'error': 'Verification required',
                'message': 'Please verify or buy premium'
            }), 403
        
        parts = file_id.split('_')
        if len(parts) < 2:
            return jsonify({'error': 'Invalid file_id'}), 400
        
        channel_id = int(parts[0])
        message_id = int(parts[1])
        
        download_url = f"https://t.me/{Config.BOT_USERNAME}?start=file_{channel_id}_{message_id}"
        
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
            return jsonify({'error': 'Premium system unavailable'}), 503
        
        is_premium, data = await premium_manager.check_premium_status(user_id)
        
        if is_premium:
            return jsonify({
                'has_premium': True,
                'plan': data.get('plan_type'),
                'expires_at': data.get('expires_at').isoformat() if data.get('expires_at') else None
            })
        return jsonify({'has_premium': False})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# BOT COMMAND HANDLERS (After bot initialization)
def setup_bot_handlers():
    """Setup bot handlers after bot is initialized"""
    
    @bot.on_message(filters.command("stats") & filters.user(Config.ADMIN_IDS))
    async def stats_handler(client, message):
        tf = await files_col.count_documents({}) if files_col else 0
        premium_count = await premium_manager.get_premium_users_count() if premium_manager else 0
        
        stats_text = f"""
📊 **SK4FiLM STATISTICS**

**Database:**
📁 Files: {tf}
💎 Premium: {premium_count}

**System:**
🤖 Bot: {'Online' if bot_started else 'Starting'}
👤 Session: {'Ready' if user_session_ready else 'Wait'}
💾 Redis: {'Enabled' if redis_cache.enabled else 'Disabled'}

**Cache:**
🔍 Searches: {movie_db['stats']['multi_channel_searches']}
✅ Hits: {movie_db['stats']['redis_hits']}
❌ Misses: {movie_db['stats']['redis_misses']}
"""
        await message.reply_text(stats_text)
    
    @bot.on_message(filters.command("index") & filters.user(Config.ADMIN_IDS))
    async def index_handler(client, message):
        msg = await message.reply_text("Starting indexing...")
        asyncio.create_task(index_files_background())
        await msg.edit_text("✅ Indexing started!")
    
    @bot.on_message(filters.command("clearcache") & filters.user(Config.ADMIN_IDS))
    async def clearcache_handler(client, message):
        await redis_cache.clear_search_cache()
        movie_db['search_cache'].clear()
        movie_db['poster_cache'].clear()
        movie_db['title_cache'].clear()
        await message.reply_text("✅ All caches cleared!")
    
    logger.info("✅ Bot handlers registered")

# INITIALIZATION
async def init():
    global User, bot, bot_started, user_session_ready
    global premium_manager, verification_manager, cache_manager, poster_manager
    
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
        
        # Initialize Poster Manager
        poster_manager = init_poster_manager(Config, movie_db)
        
        # Initialize Premium Manager
        premium_manager = PremiumManager(Config, db)
        await premium_manager.init_indexes()
        logger.info("✅ Premium Manager initialized")
        
        # Initialize Verification Manager
        verification_manager = VerificationManager(Config, db, premium_manager)
        await verification_manager.init_indexes()
        logger.info("✅ Verification Manager initialized")
        
        # Initialize Bot
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
        
        # Setup bot handlers AFTER bot initialization
        setup_bot_handlers()
        
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
        
        # Start indexing
        if user_session_ready:
            asyncio.create_task(index_files_background())
        
        logger.info("="*60)
        logger.info("✅ SYSTEM FULLY INITIALIZED")
        logger.info(f"💎 Premium Plans: {len(premium_manager.PREMIUM_PLANS)}")
        logger.info(f"🔓 Verification: {'Enabled' if Config.VERIFICATION_REQUIRED else 'Disabled'}")
        logger.info("="*60)
        
        return True
    
    except Exception as e:
        logger.error(f"❌ INIT FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

async def user_session_recovery():
    global User, user_session_ready
    
    logger.info("🔄 User session recovery...")
    await asyncio.sleep(2400)
    
    try:
        await User.start()
        user_session_ready = True
        logger.info("✅ User session recovered!")
        asyncio.create_task(index_files_background())
    except Exception as e:
        logger.error(f"❌ Recovery failed: {e}")

# MAIN
async def main():
    success = await init()
    
    if not success:
        logger.error("Failed to initialize")
        return
    
    config = HyperConfig()
    config.bind = f"0.0.0.0:{Config.WEB_SERVER_PORT}"
    config.loglevel = "warning"
    
    logger.info(f"🌐 Web server starting on port {Config.WEB_SERVER_PORT}...")
    
    await serve(app, config)

if __name__ == "__main__":
    asyncio.run(main())
