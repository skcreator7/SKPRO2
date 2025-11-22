import asyncio
import os
import logging
from datetime import datetime, timedelta
from pyrogram import Client, filters
from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from pyrogram.errors import UserNotParticipant, ChatAdminRequired, ChannelPrivate
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
import asyncio
import aiofiles
from cachetools import TTLCache
import brotli
import gzip

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger('asyncio').setLevel(logging.WARNING)

class Config:
    API_ID = int(os.environ.get("API_ID", "0"))
    API_HASH = os.environ.get("API_HASH", "")
    USER_SESSION_STRING = os.environ.get("USER_SESSION_STRING", "")
    BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
    MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
    
    MAIN_CHANNEL_ID = -1001891090100
    TEXT_CHANNEL_IDS = [-1001891090100, -1002024811395]
    FILE_CHANNEL_ID = -1001768249569
    
    # Telegram Channel Links
    MAIN_CHANNEL_LINK = "https://t.me/sk4film"
    UPDATES_CHANNEL_LINK = "https://t.me/sk4film_Request"
    CHANNEL_USERNAME = "sk4film"  # Without @
    
    # URL Shortener Verification (6 hours validity)
    URL_SHORTENER_API = os.environ.get("URL_SHORTENER_API", "https://your-shortener-api.com/verify")
    URL_SHORTENER_KEY = os.environ.get("URL_SHORTENER_KEY", "")
    VERIFICATION_REQUIRED = os.environ.get("VERIFICATION_REQUIRED", "false").lower() == "true"
    VERIFICATION_DURATION = 6 * 60 * 60  # 6 hours in seconds
    
    WEBSITE_URL = os.environ.get("WEBSITE_URL", "https://sk4film.vercel.app")
    BOT_USERNAME = os.environ.get("BOT_USERNAME", "sk4filmbot")
    ADMIN_IDS = [int(x) for x in os.environ.get("ADMIN_IDS", "123456789").split(",")]
    AUTO_DELETE_TIME = int(os.environ.get("AUTO_DELETE_TIME", "300"))
    WEB_SERVER_PORT = int(os.environ.get("PORT", 8000))
    BACKEND_URL = os.environ.get("BACKEND_URL", "https://sk4film.koyeb.app")
    
    OMDB_KEYS = ["8265bd1c", "b9bd48a6", "3e7e1e9d"]
    TMDB_KEYS = ["e547e17d4e91f3e62a571655cd1ccaff", "8265bd1f"]

app = Quart(__name__)

# 🚀 SUPER FAST CACHING SYSTEM
class FastCache:
    def __init__(self):
        # Memory cache for frequent data (5 minutes TTL)
        self.memory_cache = TTLCache(maxsize=1000, ttl=300)
        # API response cache (2 minutes TTL)
        self.api_cache = TTLCache(maxsize=500, ttl=120)
        # Poster URL cache (1 hour TTL)
        self.poster_cache = TTLCache(maxsize=2000, ttl=3600)
        
    def get_memory(self, key):
        return self.memory_cache.get(key)
    
    def set_memory(self, key, value):
        self.memory_cache[key] = value
        
    def get_api(self, key):
        return self.api_cache.get(key)
    
    def set_api(self, key, value):
        self.api_cache[key] = value
        
    def get_poster(self, key):
        return self.poster_cache.get(key)
    
    def set_poster(self, key, value):
        self.poster_cache[key] = value

fast_cache = FastCache()

# 🚀 COMPRESSION MIDDLEWARE
@app.after_request
async def add_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    
    # Compression headers
    accept_encoding = request.headers.get('Accept-Encoding', '')
    response_data = await response.get_data()
    
    if 'br' in accept_encoding and len(response_data) > 1024:
        compressed = brotli.compress(response_data)
        response.set_data(compressed)
        response.headers['Content-Encoding'] = 'br'
        response.headers['Content-Length'] = len(compressed)
    elif 'gzip' in accept_encoding and len(response_data) > 1024:
        compressed = gzip.compress(response_data)
        response.set_data(compressed)
        response.headers['Content-Encoding'] = 'gzip'
        response.headers['Content-Length'] = len(compressed)
    
    # Cache headers
    if request.path.startswith('/api/'):
        response.headers['Cache-Control'] = 'public, max-age=60, stale-while-revalidate=300'
    elif request.path.startswith('/api/poster'):
        response.headers['Cache-Control'] = 'public, max-age=3600, immutable'
    
    return response

mongo_client = None
db = None
files_col = None
verification_col = None

async def init_mongodb():
    global mongo_client, db, files_col, verification_col
    try:
        logger.info("🔌 MongoDB (Files + Verification)...")
        mongo_client = AsyncIOMotorClient(Config.MONGODB_URI, serverSelectionTimeoutMS=5000)
        await mongo_client.admin.command('ping')
        
        db = mongo_client.sk4film
        files_col = db.files
        verification_col = db.verifications
        
        # Optimized indexes
        try:
            await files_col.create_index([("normalized_title", 1)])
            await files_col.create_index([("indexed_at", -1)])
            await files_col.create_index([("title", "text")])
            await files_col.create_index([("channel_id", 1), ("message_id", 1)], unique=True)
        except:
            pass
        
        try:
            await verification_col.create_index([("user_id", 1)], unique=True)
            await verification_col.create_index([("verified_at", 1)], expireAfterSeconds=Config.VERIFICATION_DURATION)
        except:
            pass
        
        logger.info("✅ MongoDB OK")
        return True
    except Exception as e:
        logger.error(f"❌ MongoDB: {e}")
        return False

User = None
bot = None
bot_started = False

# 🚀 OPTIMIZED MOVIE DB WITH FAST CACHE
movie_db = {
    'stats': {
        'letterboxd': 0, 'imdb': 0, 'justwatch': 0, 'impawards': 0,
        'omdb': 0, 'tmdb': 0, 'custom': 0, 'cache_hits': 0, 'video_thumbnails': 0,
        'memory_cache_hits': 0, 'api_cache_hits': 0
    }
}

def normalize_title(title):
    """Ultra-fast title normalization"""
    if not title:
        return ""
    
    # Cache normalized titles
    cache_key = f"norm_{hash(title)}"
    cached = fast_cache.get_memory(cache_key)
    if cached:
        return cached
    
    normalized = title.lower().strip()
    # Fast regex patterns
    normalized = re.sub(r'\b(19|20)\d{2}\b', '', normalized)
    normalized = re.sub(r'\b(480p|720p|1080p|2160p|4k|hd|fhd|uhd|hevc|x264|x265|h264|h265|bluray|webrip|hdrip|web-dl|hdtv)\b', '', normalized, flags=re.IGNORECASE)
    normalized = ' '.join(normalized.split()).strip()
    
    fast_cache.set_memory(cache_key, normalized)
    return normalized

def extract_title_smart(text):
    """Fast title extraction with caching"""
    if not text or len(text) < 10:
        return None
    
    # Cache extraction results
    cache_key = f"extract_{hash(text[:100])}"
    cached = fast_cache.get_memory(cache_key)
    if cached:
        return cached
    
    try:
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if not lines:
            return None
        
        first_line = lines[0]
        
        # Fast pattern matching
        patterns = [
            (r'🎬\s*([^\n\-\(]{3,60})', 1),
            (r'^([^\(\n]{3,60})\s*\(\d{4}\)', 1),
            (r'^([^\-\n]{3,60})\s*-', 1)
        ]
        
        for pattern, group in patterns:
            match = re.search(pattern, first_line)
            if match:
                title = match.group(group).strip()
                title = re.sub(r'\s+', ' ', title)
                if 3 <= len(title) <= 60:
                    fast_cache.set_memory(cache_key, title)
                    return title
        
        # Fallback: first line processing
        if len(first_line) >= 3 and len(first_line) <= 60:
            title = re.sub(r'\b(480p|720p|1080p|2160p|4k|hevc|x264|x265)\b', '', first_line, flags=re.IGNORECASE)
            title = re.sub(r'\s+', ' ', title).strip()
            if 3 <= len(title) <= 60:
                fast_cache.set_memory(cache_key, title)
                return title
    except:
        pass
    
    return None

def extract_title_from_file(msg):
    """Fast file title extraction"""
    try:
        # Check cache first
        cache_key = f"file_title_{msg.id}"
        cached = fast_cache.get_memory(cache_key)
        if cached:
            return cached
            
        if msg.caption:
            t = extract_title_smart(msg.caption)
            if t:
                fast_cache.set_memory(cache_key, t)
                return t
                
        fn = msg.document.file_name if msg.document else (msg.video.file_name if msg.video else None)
        if fn:
            name = fn.rsplit('.', 1)[0]
            name = re.sub(r'[\._\-]', ' ', name)
            name = re.sub(r'(720p|1080p|480p|2160p|HDRip|WEB|BluRay|x264|x265|HEVC)', '', name, flags=re.IGNORECASE)
            name = ' '.join(name.split()).strip()
            if 4 <= len(name) <= 50:
                fast_cache.set_memory(cache_key, name)
                return name
    except:
        pass
    return None

def format_size(size):
    """Fast size formatting"""
    if not size:
        return "Unknown"
    if size < 1024*1024:
        return f"{size/1024:.1f} KB"
    elif size < 1024*1024*1024:
        return f"{size/(1024*1024):.1f} MB"
    else:
        return f"{size/(1024*1024*1024):.2f} GB"

def detect_quality(filename):
    """Fast quality detection with caching"""
    if not filename:
        return "480p"
        
    cache_key = f"quality_{filename}"
    cached = fast_cache.get_memory(cache_key)
    if cached:
        return cached
        
    fl = filename.lower()
    is_hevc = 'hevc' in fl or 'x265' in fl
    
    if '2160p' in fl or '4k' in fl:
        result = "2160p HEVC" if is_hevc else "2160p"
    elif '1080p' in fl:
        result = "1080p HEVC" if is_hevc else "1080p"
    elif '720p' in fl:
        result = "720p HEVC" if is_hevc else "720p"
    elif '480p' in fl:
        result = "480p HEVC" if is_hevc else "480p"
    else:
        result = "480p"
    
    fast_cache.set_memory(cache_key, result)
    return result

def format_post(text):
    """Fast post formatting"""
    if not text:
        return ""
        
    cache_key = f"post_{hash(text[:200])}"
    cached = fast_cache.get_memory(cache_key)
    if cached:
        return cached
        
    text = html.escape(text)
    text = re.sub(r'(https?://[^\s]+)', r'<a href="\1" target="_blank" style="color:#00ccff">\1</a>', text)
    formatted = text.replace('\n', '<br>')
    
    fast_cache.set_memory(cache_key, formatted)
    return formatted

def channel_name(cid):
    """Fast channel name lookup"""
    names = {
        -1001891090100: "SK4FiLM Main", 
        -1002024811395: "SK4FiLM Updates", 
        -1001768249569: "SK4FiLM Files"
    }
    return names.get(cid, f"Channel {cid}")

def is_new(date):
    """Fast date checking"""
    try:
        if isinstance(date, str):
            date = datetime.fromisoformat(date.replace('Z', '+00:00'))
        hours = (datetime.now() - date.replace(tzinfo=None)).total_seconds() / 3600
        return hours <= 48
    except:
        return False

def is_video_file(file_name):
    """Fast video file detection"""
    if not file_name:
        return False
    
    cache_key = f"is_video_{file_name}"
    cached = fast_cache.get_memory(cache_key)
    if cached is not None:
        return cached
        
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v']
    file_name_lower = file_name.lower()
    result = any(file_name_lower.endswith(ext) for ext in video_extensions)
    
    fast_cache.set_memory(cache_key, result)
    return result

# 🚀 ASYNC HTTP CLIENT WITH CONNECTION POOL
class FastHttpClient:
    def __init__(self):
        self.session = None
        self.connector = None
        
    async def get_session(self):
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=5, connect=2)
            self.connector = aiohttp.TCPConnector(limit=100, limit_per_host=20, keepalive_timeout=30)
            self.session = aiohttp.ClientSession(
                connector=self.connector,
                timeout=timeout,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate, br'
                }
            )
        return self.session
        
    async def close(self):
        if self.session:
            await self.session.close()
            self.session = None

fast_http = FastHttpClient()

async def extract_video_thumbnail(user_client, message):
    """Fast video thumbnail extraction"""
    try:
        cache_key = f"thumb_{message.id}"
        cached = fast_cache.get_memory(cache_key)
        if cached:
            movie_db['stats']['memory_cache_hits'] += 1
            return cached
            
        if message.video and message.video.thumbs:
            thumbnail = message.video.thumbs[0]
            thumbnail_path = await user_client.download_media(thumbnail.file_id, in_memory=True)
            if thumbnail_path:
                thumbnail_data = base64.b64encode(thumbnail_path.getvalue()).decode('utf-8')
                thumbnail_url = f"data:image/jpeg;base64,{thumbnail_data}"
                movie_db['stats']['video_thumbnails'] += 1
                fast_cache.set_memory(cache_key, thumbnail_url)
                return thumbnail_url
    except Exception as e:
        logger.debug(f"Thumbnail extraction failed: {e}")
    return None

async def get_poster_fast(title, session):
    """Ultra-fast poster fetching with multiple fallbacks"""
    cache_key = f"poster_{title.lower()}"
    cached = fast_cache.get_poster(cache_key)
    if cached:
        movie_db['stats']['cache_hits'] += 1
        return cached
    
    # Try multiple sources concurrently
    tasks = []
    
    # Letterboxd (fastest)
    async def try_letterboxd():
        try:
            clean_title = re.sub(r'[^\w\s]', '', title).strip()
            slug = clean_title.lower().replace(' ', '-')
            slug = re.sub(r'-+', '-', slug)
            
            url = f"https://letterboxd.com/film/{slug}/"
            async with session.get(url, timeout=3) as r:
                if r.status == 200:
                    html_content = await r.text()
                    poster_match = re.search(r'<meta property="og:image" content="([^"]+)"', html_content)
                    if poster_match:
                        poster_url = poster_match.group(1)
                        if poster_url and poster_url.startswith('http'):
                            if 'cloudfront.net' in poster_url:
                                poster_url = poster_url.replace('-0-500-0-750', '-0-300-0-450')  # Smaller for speed
                            return {'poster_url': poster_url, 'source': 'Letterboxd'}
        except:
            pass
        return None
    
    # IMDb fast search
    async def try_imdb_fast():
        try:
            clean_title = re.sub(r'[^\w\s]', '', title).strip()
            search_url = f"https://v2.sg.media-imdb.com/suggestion/{clean_title[0].lower()}/{urllib.parse.quote(clean_title.replace(' ', '_'))}.json"
            async with session.get(search_url, timeout=3) as r:
                if r.status == 200:
                    data = await r.json()
                    if data.get('d'):
                        item = data['d'][0]
                        if item.get('i'):
                            poster_url = item['i'][0] if isinstance(item['i'], list) else item['i']
                            if poster_url:
                                poster_url = poster_url.replace('._V1_', '._V1_UX300_')  # Smaller size
                                return {'poster_url': poster_url, 'source': 'IMDb'}
        except:
            pass
        return None
    
    # JustWatch fast
    async def try_justwatch_fast():
        try:
            clean_title = re.sub(r'[^\w\s]', '', title).strip()
            slug = clean_title.lower().replace(' ', '-')
            url = f"https://www.justwatch.com/in/movie/{slug}"
            async with session.get(url, timeout=3) as r:
                if r.status == 200:
                    html_content = await r.text()
                    poster_match = re.search(r'<meta property="og:image" content="([^"]+)"', html_content)
                    if poster_match:
                        poster_url = poster_match.group(1)
                        return {'poster_url': poster_url, 'source': 'JustWatch'}
        except:
            pass
        return None
    
    tasks = [try_letterboxd(), try_imdb_fast(), try_justwatch_fast()]
    
    # Run all tasks concurrently and take first successful result
    for completed in asyncio.as_completed(tasks):
        result = await completed
        if result:
            fast_cache.set_poster(cache_key, result)
            if result['source'] == 'Letterboxd':
                movie_db['stats']['letterboxd'] += 1
            elif result['source'] == 'IMDb':
                movie_db['stats']['imdb'] += 1
            elif result['source'] == 'JustWatch':
                movie_db['stats']['justwatch'] += 1
            return result
    
    # Fallback to custom poster (fastest)
    year_match = re.search(r'\b(19|20)\d{2}\b', title)
    year = year_match.group() if year_match else ""
    result = {
        'poster_url': f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(title)}&year={year}", 
        'source': 'CUSTOM'
    }
    movie_db['stats']['custom'] += 1
    fast_cache.set_poster(cache_key, result)
    return result

async def index_files_background():
    """Fast background indexing"""
    if not User or files_col is None:
        return
    
    logger.info("📁 Starting FAST background indexing...")
    
    try:
        count = 0
        video_files_count = 0
        batch = []
        batch_size = 100  # Larger batch for speed
        
        session = await fast_http.get_session()
        
        async for msg in User.get_chat_history(Config.FILE_CHANNEL_ID, limit=1000):  # Limit for speed
            if msg.document or msg.video:
                title = extract_title_from_file(msg)
                if title:
                    file_id = msg.document.file_id if msg.document else msg.video.file_id
                    file_size = msg.document.file_size if msg.document else (msg.video.file_size if msg.video else 0)
                    file_name = msg.document.file_name if msg.document else (msg.video.file_name if msg.video else 'video.mp4')
                    quality = detect_quality(file_name)
                    
                    file_is_video = is_video_file(file_name)
                    
                    thumbnail_url = None
                    if file_is_video:
                        # Fast thumbnail extraction
                        video_thumbnail = await extract_video_thumbnail(User, msg)
                        if video_thumbnail:
                            thumbnail_url = video_thumbnail
                        else:
                            # Fast poster fetch
                            poster_data = await get_poster_fast(title, session)
                            thumbnail_url = poster_data['poster_url'] if poster_data else None
                        
                        video_files_count += 1
                    
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
                        'thumbnail': thumbnail_url,
                        'is_video_file': file_is_video,
                        'thumbnail_source': 'video_direct' if video_thumbnail else 'poster_api'
                    })
                    
                    count += 1
                    
                    if len(batch) >= batch_size:
                        try:
                            # Bulk insert for speed
                            await files_col.insert_many(batch, ordered=False)
                            logger.info(f"    ✅ Fast indexed {count} files...")
                            batch = []
                        except Exception as e:
                            # Individual inserts if bulk fails
                            for doc in batch:
                                try:
                                    await files_col.update_one(
                                        {'channel_id': doc['channel_id'], 'message_id': doc['message_id']},
                                        {'$set': doc},
                                        upsert=True
                                    )
                                except:
                                    pass
                            batch = []
        
        if batch:
            try:
                await files_col.insert_many(batch, ordered=False)
            except:
                for doc in batch:
                    try:
                        await files_col.update_one(
                            {'channel_id': doc['channel_id'], 'message_id': doc['message_id']},
                            {'$set': doc},
                            upsert=True
                        )
                    except:
                        pass
        
        logger.info(f"✅ FAST indexing complete: {count} files, {video_files_count} videos")
        
    except Exception as e:
        logger.error(f"❌ Fast indexing error: {e}")

# 🚀 SUPER FAST API ENDPOINTS
@app.route('/')
async def root():
    """Ultra-fast root endpoint"""
    cache_key = "root_response"
    cached = fast_cache.get_api(cache_key)
    if cached:
        movie_db['stats']['api_cache_hits'] += 1
        return jsonify(cached)
    
    tf = await files_col.count_documents({}) if files_col is not None else 0
    
    response = {
        'status': 'healthy',
        'service': 'SK4FiLM v7.0 - SUPER FAST',
        'database': {'total_files': tf, 'mode': 'ULTRA FAST'},
        'bot_status': 'online' if bot_started else 'starting',
        'cache_stats': {
            'memory_hits': movie_db['stats']['memory_cache_hits'],
            'api_hits': movie_db['stats']['api_cache_hits'],
            'poster_hits': movie_db['stats']['cache_hits']
        },
        'performance': 'SUPER FAST LOADING'
    }
    
    fast_cache.set_api(cache_key, response)
    return jsonify(response)

@app.route('/health')
async def health():
    """Fast health check"""
    return jsonify({'status': 'ok' if bot_started else 'starting', 'timestamp': datetime.now().isoformat()})

@app.route('/api/movies')
async def api_movies():
    """Fast movies endpoint with aggressive caching"""
    cache_key = "api_movies_home"
    cached = fast_cache.get_api(cache_key)
    if cached:
        movie_db['stats']['api_cache_hits'] += 1
        return jsonify(cached)
    
    try:
        if not bot_started:
            return jsonify({'status': 'error', 'message': 'Starting...'}), 503
        
        # Fast home movies (limited for speed)
        posts = await get_live_posts(Config.MAIN_CHANNEL_ID, limit=20)
        movies = []
        seen = set()
        
        for post in posts[:15]:  # Limit to 15 for speed
            tk = post['title'].lower().strip()
            if tk not in seen:
                seen.add(tk)
                movies.append({
                    'title': post['title'],
                    'date': post['date'].isoformat() if isinstance(post['date'], datetime) else post['date'],
                    'is_new': post.get('is_new', False),
                    'channel': post.get('channel_name', 'SK4FiLM Main')
                })
        
        # Fast poster fetching in parallel
        session = await fast_http.get_session()
        tasks = [get_poster_fast(movie['title'], session) for movie in movies]
        posters = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, (movie, poster_result) in enumerate(zip(movies, posters)):
            if isinstance(poster_result, dict):
                movie.update({
                    'poster_url': poster_result['poster_url'],
                    'poster_source': poster_result['source'],
                    'has_poster': True
                })
            else:
                movie.update({
                    'poster_url': f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(movie['title'])}",
                    'poster_source': 'CUSTOM',
                    'has_poster': True
                })
        
        response = {
            'status': 'success', 
            'movies': movies, 
            'total': len(movies), 
            'mode': 'ULTRA FAST',
            'cache': 'MEMORY + API CACHING'
        }
        
        fast_cache.set_api(cache_key, response)
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"API /movies error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/search')
async def api_search():
    """Fast search with multiple optimizations"""
    q = request.args.get('query', '').strip().lower()
    p = int(request.args.get('page', 1))
    l = min(int(request.args.get('limit', 12)), 50)  # Limit for speed
    
    if not q:
        return jsonify({'status': 'error', 'message': 'Query required'}), 400
    
    # Cache search results
    cache_key = f"search_{q}_{p}_{l}"
    cached = fast_cache.get_api(cache_key)
    if cached:
        movie_db['stats']['api_cache_hits'] += 1
        return jsonify(cached)
    
    try:
        if not bot_started:
            return jsonify({'status': 'error', 'message': 'Starting...'}), 503
        
        offset = (p - 1) * l
        
        # Fast MongoDB search with projection
        pipeline = [
            {'$match': {'$text': {'$search': q}}},
            {'$project': {
                'title': 1, 'normalized_title': 1, 'quality': 1, 
                'file_size': 1, 'file_name': 1, 'date': 1, 'thumbnail': 1,
                'is_video_file': 1, 'thumbnail_source': 1, 'channel_id': 1, 'message_id': 1
            }},
            {'$sort': {'date': -1}},
            {'$skip': offset},
            {'$limit': l}
        ]
        
        files_results = []
        if files_col is not None:
            cursor = files_col.aggregate(pipeline)
            async for doc in cursor:
                files_results.append(doc)
        
        # Process results quickly
        results = []
        for doc in files_results:
            quality_options = {
                doc['quality']: {
                    'file_id': f"{doc.get('channel_id', Config.FILE_CHANNEL_ID)}_{doc.get('message_id')}_{doc['quality']}",
                    'file_size': doc['file_size'],
                    'file_name': doc['file_name'],
                    'is_video': doc.get('is_video_file', False)
                }
            }
            
            results.append({
                'title': doc['title'],
                'content': f"<p>{doc['title']}</p>",
                'channel': 'SK4FiLM',
                'date': doc['date'].isoformat() if isinstance(doc['date'], datetime) else doc['date'],
                'is_new': is_new(doc['date']),
                'has_file': True,
                'has_post': False,
                'quality_options': quality_options,
                'thumbnail': doc.get('thumbnail') if doc.get('is_video_file') else None,
                'thumbnail_source': doc.get('thumbnail_source', 'unknown')
            })
        
        response = {
            'status': 'success', 
            'query': q, 
            'results': results, 
            'pagination': {
                'current_page': p,
                'total_pages': 1,  # Simplified for speed
                'total_results': len(results),
                'per_page': l,
                'has_next': len(results) == l,
                'has_previous': p > 1
            }, 
            'mode': 'ULTRA FAST SEARCH'
        }
        
        fast_cache.set_api(cache_key, response)
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"API /search error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/post')
async def api_post():
    """Fast post endpoint"""
    try:
        channel_id = request.args.get('channel', '').strip()
        message_id = request.args.get('message', '').strip()
        
        if not channel_id or not message_id:
            return jsonify({'status':'error', 'message':'Missing parameters'}), 400
        
        # Cache individual posts
        cache_key = f"post_{channel_id}_{message_id}"
        cached = fast_cache.get_api(cache_key)
        if cached:
            movie_db['stats']['api_cache_hits'] += 1
            return jsonify(cached)
        
        if not bot_started or not User:
            return jsonify({'status':'error', 'message':'Bot not ready'}), 503
        
        channel_id = int(channel_id)
        message_id = int(message_id)
        
        msg = await User.get_messages(channel_id, message_id)
        if not msg or not msg.text:
            return jsonify({'status':'error', 'message':'Message not found'}), 404
        
        title = extract_title_smart(msg.text) or msg.text.split('\n')[0][:60]
        normalized_title = normalize_title(title)
        
        # Fast file lookup
        quality_options = {}
        has_file = False
        thumbnail_url = None
        
        if files_col is not None:
            doc = await files_col.find_one({'normalized_title': normalized_title})
            if doc:
                quality = doc.get('quality', '480p')
                file_is_video = doc.get('is_video_file', False)
                
                if file_is_video:
                    thumbnail_url = doc.get('thumbnail')
                
                quality_options[quality] = {
                    'file_id': f"{doc.get('channel_id', Config.FILE_CHANNEL_ID)}_{doc.get('message_id')}_{quality}",
                    'file_size': doc.get('file_size', 0),
                    'file_name': doc.get('file_name', 'video.mp4'),
                    'is_video': file_is_video
                }
                has_file = True
        
        post_data = {
            'title': title,
            'content': format_post(msg.text),
            'channel': channel_name(channel_id),
            'channel_id': channel_id,
            'message_id': message_id,
            'date': msg.date.isoformat() if isinstance(msg.date, datetime) else str(msg.date),
            'is_new': is_new(msg.date),
            'has_file': has_file,
            'quality_options': quality_options,
            'thumbnail': thumbnail_url
        }
        
        response = {'status': 'success', 'post': post_data}
        fast_cache.set_api(cache_key, response)
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"API /post error: {e}")
        return jsonify({'status':'error', 'message': str(e)}), 500

@app.route('/api/poster')
async def api_poster():
    """Fast poster generator with aggressive caching"""
    t = request.args.get('title', 'Movie')
    y = request.args.get('year', '')
    
    cache_key = f"poster_svg_{t}_{y}"
    cached = fast_cache.get_memory(cache_key)
    if cached:
        return Response(cached, mimetype='image/svg+xml', headers={
            'Cache-Control': 'public, max-age=86400, immutable',
            'Content-Type': 'image/svg+xml'
        })
    
    d = t[:20] + "..." if len(t) > 20 else t
    
    color_schemes = [
        {'bg1': '#667eea', 'bg2': '#764ba2', 'text': '#ffffff'},
        {'bg1': '#f093fb', 'bg2': '#f5576c', 'text': '#ffffff'},
        {'bg1': '#4facfe', 'bg2': '#00f2fe', 'text': '#ffffff'},
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
    
    fast_cache.set_memory(cache_key, svg)
    return Response(svg, mimetype='image/svg+xml', headers={
        'Cache-Control': 'public, max-age=86400, immutable',
        'Content-Type': 'image/svg+xml'
    })

# 🚀 FAST BOT COMMANDS (Simplified for speed)
async def setup_bot():
    @bot.on_message(filters.command("start") & filters.private)
    async def start_handler(client, message):
        uid = message.from_user.id
        user_name = message.from_user.first_name or "User"
        
        if len(message.command) > 1:
            # Fast file handling
            fid = message.command[1]
            try:
                parts = fid.split('_')
                if len(parts) >= 2:
                    channel_id = int(parts[0])
                    message_id = int(parts[1])
                    
                    file_message = await bot.get_messages(channel_id, message_id)
                    if file_message:
                        if file_message.document:
                            await bot.send_document(uid, file_message.document.file_id)
                        elif file_message.video:
                            await bot.send_video(uid, file_message.video.file_id)
                return
            except Exception as e:
                await message.reply_text("❌ Download failed")
                return
        
        # Fast welcome message
        welcome_text = f"🎬 **Welcome to SK4FiLM, {user_name}!**\n\n🌐 **Website:** {Config.WEBSITE_URL}"
        
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🌐 VISIT WEBSITE", url=Config.WEBSITE_URL)],
            [
                InlineKeyboardButton("📢 MAIN CHANNEL", url=Config.MAIN_CHANNEL_LINK),
                InlineKeyboardButton("🔎 MOVIES GROUP", url=Config.UPDATES_CHANNEL_LINK)
            ]
        ])
        
        await message.reply_text(welcome_text, reply_markup=keyboard, disable_web_page_preview=True)

async def init():
    global User, bot, bot_started
    try:
        logger.info("🚀 INITIALIZING SUPER FAST SK4FiLM BOT...")
        await init_mongodb()
        
        User = Client("user_session", api_id=Config.API_ID, api_hash=Config.API_HASH, 
                     session_string=Config.USER_SESSION_STRING, no_updates=True)
        bot = Client("bot", api_id=Config.API_ID, api_hash=Config.API_HASH, bot_token=Config.BOT_TOKEN)
        
        await User.start()
        await bot.start()
        await setup_bot()
        
        me = await bot.get_me()
        logger.info(f"✅ BOT STARTED: @{me.username}")
        bot_started = True
        
        # Start fast indexing
        asyncio.create_task(index_files_background())
        
        return True
    except Exception as e:
        logger.error(f"❌ INIT FAILED: {e}")
        return False

async def main():
    logger.info("="*60)
    logger.info("🎬 SK4FiLM v7.0 - SUPER FAST LOADING")
    logger.info("✅ Performance: ULTRA FAST")
    logger.info("✅ Caching: MULTI-LEVEL")
    logger.info("✅ Compression: BROTLI + GZIP")
    logger.info("✅ Database: OPTIMIZED")
    logger.info("="*60)
    
    success = await init()
    if not success:
        logger.error("❌ Failed to initialize bot")
        return
    
    config = HyperConfig()
    config.bind = [f"0.0.0.0:{Config.WEB_SERVER_PORT}"]
    config.loglevel = "warning"
    
    logger.info(f"🌐 SUPER FAST server starting on port {Config.WEB_SERVER_PORT}...")
    await serve(app, config)

if __name__ == "__main__":
    asyncio.run(main())
