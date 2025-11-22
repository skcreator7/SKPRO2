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
User = None
bot = None
bot_started = False

# OPTIMIZED CACHE SYSTEM
movie_db = {
    'poster_cache': {},
    'title_cache': {},
    'search_cache': {},
    'stats': {
        'letterboxd': 0, 'imdb': 0, 'justwatch': 0, 'impawards': 0,
        'omdb': 0, 'tmdb': 0, 'custom': 0, 'cache_hits': 0, 'video_thumbnails': 0
    }
}

# CACHE CLEANUP TASK
async def cache_cleanup():
    while True:
        await asyncio.sleep(3600)  # Clean every hour
        try:
            current_time = datetime.now()
            expired_keys = []
            for key, (data, timestamp) in movie_db['poster_cache'].items():
                if (current_time - timestamp).seconds > 3600:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del movie_db['poster_cache'][key]
            
            # Clear search cache more frequently (30 minutes)
            expired_search_keys = []
            for key, (data, timestamp) in movie_db['search_cache'].items():
                if (current_time - timestamp).seconds > 1800:
                    expired_search_keys.append(key)
            
            for key in expired_search_keys:
                del movie_db['search_cache'][key]
                
            logger.info(f"🧹 Cache cleaned: {len(expired_keys)} posters, {len(expired_search_keys)} searches")
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")

# OPTIMIZED MONGODB INIT
async def init_mongodb():
    global mongo_client, db, files_col, verification_col
    try:
        logger.info("🔌 MongoDB initialization...")
        mongo_client = AsyncIOMotorClient(Config.MONGODB_URI, serverSelectionTimeoutMS=5000, maxPoolSize=10)
        await mongo_client.admin.command('ping')
        
        db = mongo_client.sk4film
        files_col = db.files
        verification_col = db.verifications
        
        # FAST INDEX CREATION - ONLY IF NOT EXISTS
        existing_indexes = await files_col.index_information()
        
        if 'title_text' not in existing_indexes:
            await files_col.create_index([("title", "text")])
        
        if 'normalized_title_1' not in existing_indexes:
            await files_col.create_index([("normalized_title", 1)])
        
        if 'msg_ch_unique_idx' not in existing_indexes:
            await files_col.create_index(
                [("message_id", 1), ("channel_id", 1)], 
                unique=True,
                name="msg_ch_unique_idx"
            )
        
        if 'indexed_at_-1' not in existing_indexes:
            await files_col.create_index([("indexed_at", -1)])
        
        if 'user_id_1' not in existing_indexes:
            await verification_col.create_index([("user_id", 1)], unique=True)
        
        logger.info("✅ MongoDB OK - Optimized")
        return True
    except Exception as e:
        logger.error(f"❌ MongoDB: {e}")
        return False

# OPTIMIZED TITLE NORMALIZATION
def normalize_title(title):
    if not title:
        return ""
    # FAST NORMALIZATION - FEWER REGEX OPERATIONS
    normalized = title.lower().strip()
    normalized = re.sub(r'\b(19|20)\d{2}\b', '', normalized)
    normalized = re.sub(r'\b(480p|720p|1080p|2160p|4k|hd|fhd|uhd|hevc|x264|x265|h264|h265|bluray|webrip|hdrip|web-dl|hdtv)\b', '', normalized, flags=re.IGNORECASE)
    return ' '.join(normalized.split()).strip()

# OPTIMIZED TITLE EXTRACTION
def extract_title_smart(text):
    if not text or len(text) < 10:
        return None
    
    # CACHE CHECK
    text_hash = hash(text[:200])  # First 200 chars for cache key
    if text_hash in movie_db['title_cache']:
        return movie_db['title_cache'][text_hash]
    
    try:
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if not lines:
            return None
        
        first_line = lines[0]
        
        # PRIORITIZED PATTERN MATCHING
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
                    movie_db['title_cache'][text_hash] = title
                    return title
        
        # FALLBACK
        if len(first_line) >= 3 and len(first_line) <= 60:
            title = re.sub(r'\b(480p|720p|1080p|2160p|4k|hevc|x264|x265)\b', '', first_line, flags=re.IGNORECASE)
            title = ' '.join(title.split()).strip()
            if 3 <= len(title) <= 60:
                movie_db['title_cache'][text_hash] = title
                return title
    except:
        pass
    
    movie_db['title_cache'][text_hash] = None
    return None

# OPTIMIZED FILE TITLE EXTRACTION
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
            name = re.sub(r'(720p|1080p|480p|2160p|HDRip|WEB|BluRay|x264|x265|HEVC)', '', name, flags=re.IGNORECASE)
            name = ' '.join(name.split()).strip()
            if 4 <= len(name) <= 50:
                return name
    except:
        pass
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
    names = {
        -1001891090100: "SK4FiLM Main", 
        -1002024811395: "SK4FiLM Updates", 
        -1001768249569: "SK4FiLM Files"
    }
    return names.get(cid, f"Channel {cid}")

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

# OPTIMIZED THUMBNAIL EXTRACTION
async def extract_video_thumbnail(user_client, message):
    try:
        if message.video:
            thumbnail = message.video.thumbs[0] if message.video.thumbs else None
            if thumbnail:
                thumbnail_path = await user_client.download_media(thumbnail.file_id, in_memory=True)
                if thumbnail_path:
                    thumbnail_data = base64.b64encode(thumbnail_path.getvalue()).decode('utf-8')
                    thumbnail_url = f"data:image/jpeg;base64,{thumbnail_data}"
                    movie_db['stats']['video_thumbnails'] += 1
                    return thumbnail_url
        return None
    except Exception as e:
        return None

async def get_telegram_video_thumbnail(user_client, channel_id, message_id):
    try:
        msg = await user_client.get_messages(channel_id, message_id)
        if not msg or (not msg.video and not msg.document):
            return None
        return await extract_video_thumbnail(user_client, msg)
    except Exception as e:
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

# OPTIMIZED BACKGROUND INDEXING
async def index_files_background():
    if not User or files_col is None:
        return
    
    logger.info("📁 Starting OPTIMIZED background indexing...")
    
    try:
        count = 0
        video_files_count = 0
        batch = []
        batch_size = 100  # Increased batch size for speed
        
        async for msg in User.get_chat_history(Config.FILE_CHANNEL_ID, limit=2000):  # Limit for speed
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
                        video_thumbnail = await extract_video_thumbnail(User, msg)
                        if video_thumbnail:
                            thumbnail_url = video_thumbnail
                        else:
                            async with aiohttp.ClientSession() as session:
                                poster_data = await get_poster_guaranteed(title, session)
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
                            for doc in batch:
                                await files_col.update_one(
                                    {'channel_id': doc['channel_id'], 'message_id': doc['message_id']},
                                    {'$set': doc},
                                    upsert=True
                                )
                            logger.info(f"    ✅ Batch indexed: {count} files")
                            batch = []
                        except Exception as e:
                            batch = []
        
        if batch:
            try:
                for doc in batch:
                    await files_col.update_one(
                        {'channel_id': doc['channel_id'], 'message_id': doc['message_id']},
                        {'$set': doc},
                        upsert=True
                    )
            except Exception as e:
                pass
        
        logger.info(f"✅ OPTIMIZED indexing complete: {count} files, {video_files_count} video files")
        
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
    
    # CONCURRENT REQUESTS FOR SPEED
    tasks = [source(title, session) for source in sources]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for result in results:
        if isinstance(result, dict) and result:
            movie_db['poster_cache'][ck] = (result, datetime.now())
            return result
    
    # FALLBACK
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

# OPTIMIZED LIVE POSTS FETCHING
async def get_live_posts(channel_id, limit=30):  # Reduced limit for speed
    if not User:
        return []
    
    posts = []
    count = 0
    
    try:
        async for msg in User.get_chat_history(channel_id, limit=limit):
            if msg.text and len(msg.text) > 15:
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
                    count += 1
    except Exception as e:
        pass
    
    return posts

# OPTIMIZED SEARCH WITH CACHE
async def search_movies_live(query, limit=12, page=1):
    offset = (page - 1) * limit
    
    # CACHE KEY
    cache_key = f"{query}_{page}_{limit}"
    if cache_key in movie_db['search_cache']:
        data, timestamp = movie_db['search_cache'][cache_key]
        if (datetime.now() - timestamp).seconds < 300:  # 5 minute cache
            return data
    
    query_lower = query.lower()
    posts_dict = {}
    files_dict = {}
    
    # FAST TEXT CHANNEL SEARCH
    for channel_id in Config.TEXT_CHANNEL_IDS:
        try:
            cname = channel_name(channel_id)
            async for msg in User.search_messages(channel_id, query=query, limit=100):  # Reduced limit
                if msg.text and len(msg.text) > 15:
                    title = extract_title_smart(msg.text)
                    if title and query_lower in title.lower():
                        norm_title = normalize_title(title)
                        if norm_title not in posts_dict:
                            posts_dict[norm_title] = {
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
            continue
    
    # FAST FILE SEARCH
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
                            if not thumbnail_url:
                                thumbnail_url = await get_telegram_video_thumbnail(User, doc['channel_id'], doc['message_id'])
                        
                        files_dict[norm_title] = {
                            'title': doc['title'], 
                            'quality_options': {}, 
                            'date': doc['date'].isoformat() if isinstance(doc['date'], datetime) else doc['date'],
                            'thumbnail': thumbnail_url,
                            'is_video_file': file_is_video,
                            'thumbnail_source': doc.get('thumbnail_source', 'unknown')
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
        pass
    
    # MERGE RESULTS
    merged = {}
    for norm_title, post_data in posts_dict.items():
        merged[norm_title] = post_data
    
    for norm_title, file_data in files_dict.items():
        if norm_title in merged:
            merged[norm_title]['has_file'] = True
            merged[norm_title]['quality_options'] = file_data['quality_options']
            if file_data.get('is_video_file') and file_data.get('thumbnail'):
                merged[norm_title]['thumbnail'] = file_data['thumbnail']
                merged[norm_title]['thumbnail_source'] = file_data.get('thumbnail_source', 'unknown')
        else:
            merged[norm_title] = {
                'title': file_data['title'],
                'content': f"<p>{file_data['title']}</p>",
                'channel': 'SK4FiLM',
                'date': file_data['date'],
                'is_new': False,
                'has_file': True,
                'has_post': False,
                'quality_options': file_data['quality_options'],
                'thumbnail': file_data.get('thumbnail') if file_data.get('is_video_file') else None,
                'thumbnail_source': file_data.get('thumbnail_source', 'unknown')
            }
    
    results_list = list(merged.values())
    results_list.sort(key=lambda x: (not x.get('is_new', False), not x['has_file'], x['date']), reverse=True)
    
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
    
    # CACHE THE RESULTS
    movie_db['search_cache'][cache_key] = (result_data, datetime.now())
    return result_data

# OPTIMIZED HOME MOVIES WITH CONCURRENT POSTER FETCHING
async def get_home_movies_live():
    posts = await get_live_posts(Config.MAIN_CHANNEL_ID, limit=30)
    
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
                'channel': post.get('channel_name', 'SK4FiLM Main')
            })
            if len(movies) >= 30:
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

# OPTIMIZED API ROUTES
@app.route('/')
async def root():
    tf = await files_col.count_documents({}) if files_col is not None else 0
    return jsonify({
        'status': 'healthy',
        'service': 'SK4FiLM v6.0 - OPTIMIZED',
        'database': {'total_files': tf, 'mode': 'FAST'},
        'bot_status': 'online' if bot_started else 'starting',
        'optimizations': 'CACHING + CONCURRENT REQUESTS + BATCH PROCESSING'
    })

@app.route('/health')
async def health():
    return jsonify({'status': 'ok' if bot_started else 'starting'})

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
        latest = await files_col.find_one({}, sort=[('indexed_at', -1)])
        last_indexed = "Never"
        if latest and latest.get('indexed_at'):
            dt = latest['indexed_at']
            if isinstance(dt, datetime):
                mins_ago = int((datetime.now() - dt).total_seconds() / 60)
                last_indexed = f"{mins_ago} min ago" if mins_ago > 0 else "Just now"
        
        video_files = await files_col.count_documents({'is_video_file': True})
        video_thumbnails = await files_col.count_documents({'is_video_file': True, 'thumbnail': {'$ne': None}})
        
        return jsonify({
            'status': 'success',
            'total_indexed': total,
            'video_files': video_files,
            'video_thumbnails': video_thumbnails,
            'last_indexed': last_indexed,
            'bot_status': 'online' if bot_started else 'starting',
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
            'mode': 'OPTIMIZED'
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
        
        result = await search_movies_live(q, l, p)
        return jsonify({
            'status': 'success', 
            'query': q, 
            'results': result['results'], 
            'pagination': result['pagination'], 
            'bot_username': Config.BOT_USERNAME,
            'mode': 'OPTIMIZED'
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
        
        if not bot_started or not User:
            return jsonify({'status':'error', 'message':'Bot not ready yet'}), 503
        
        try:
            channel_id = int(channel_id)
            message_id = int(message_id)
        except ValueError:
            return jsonify({'status':'error', 'message':'Invalid channel or message ID'}), 400
        
        msg = await User.get_messages(channel_id, message_id)
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
                        
                        if not thumbnail_url:
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

# OPTIMIZED BOT SETUP
async def setup_bot():
    @bot.on_message(filters.command("start") & filters.private)
    async def start_handler(client, message):
        uid = message.from_user.id
        user_name = message.from_user.first_name or "User"
        
        if len(message.command) > 1:
            fid = message.command[1]
            
            if Config.VERIFICATION_REQUIRED:
                is_verified, status = await check_url_shortener_verification(uid)
                
                if not is_verified:
                    verification_url = await generate_verification_url(uid)
                    
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
                    
                    file_message = await bot.get_messages(channel_id, message_id)
                    
                    if not file_message or (not file_message.document and not file_message.video):
                        await pm.edit_text("❌ **File not found**\n\nThe file may have been deleted.")
                        return
                    
                    if file_message.document:
                        sent = await bot.send_document(
                            uid, 
                            file_message.document.file_id, 
                            caption=f"♻ **Please forward this file/video to your saved messages**\n\n"
                                   f"📹 Quality: {quality}\n"
                                   f"📦 Size: {format_size(file_message.document.file_size)}\n\n"
                                   f"⚠️ Will auto-delete in {Config.AUTO_DELETE_TIME//60} minutes\n\n"
                                   f"@SK4FiLM 🍿"
                        )
                    else:
                        sent = await bot.send_video(
                            uid, 
                            file_message.video.file_id, 
                            caption=f"♻ **Please forward this file/video to your saved messages**\n\n"
                                   f"📹 Quality: {quality}\n" 
                                   f"📦 Size: {format_size(file_message.video.file_size)}\n\n"
                                   f"⚠️ Will auto-delete in {Config.AUTO_DELETE_TIME//60} minutes\n\n"
                                   f"@SK4FiLM 🍿"
                        )
                    
                    await pm.delete()
                    
                    if Config.AUTO_DELETE_TIME > 0:
                        async def auto_delete():
                            await asyncio.sleep(Config.AUTO_DELETE_TIME)
                            try:
                                await sent.delete()
                            except:
                                pass
                        asyncio.create_task(auto_delete())
                        
                else:
                    await message.reply_text("❌ **Invalid file link**\n\nPlease get a fresh link from the website.")
                    
            except Exception as e:
                try:
                    await message.reply_text(f"❌ **Download Failed**\n\nError: `{str(e)}`")
                except:
                    pass
            return
        
        welcome_text = (
            f"🎬 **Welcome to SK4FiLM, {user_name}!**\n\n"
            "🌐 **Use our website to browse and download movies:**\n"
            f"{Config.WEBSITE_URL}\n\n"
        )
        
        if Config.VERIFICATION_REQUIRED:
            welcome_text += "🔒 **URL Verification Required**\n• Complete one-time verification\n• Valid for 6 hours\n\n"
        
        welcome_text += (
            "✨ **Features:**\n"
            "• 🎥 Latest movies & shows\n" 
            "• 📺 Multiple quality options\n"
            "• ⚡ Fast downloads\n\n"
            "👇 **Get started below:**"
        )
        
        buttons = []
        if Config.VERIFICATION_REQUIRED:
            verification_url = await generate_verification_url(uid)
            buttons.append([InlineKeyboardButton("🔗 GET VERIFIED", url=verification_url)])
        
        buttons.extend([
            [InlineKeyboardButton("🌐 VISIT WEBSITE", url=Config.WEBSITE_URL)],
            [
                InlineKeyboardButton("📢 Mᴀɪɴ Cʜᴀɴɴᴇʟ", url=Config.MAIN_CHANNEL_LINK),
                InlineKeyboardButton("🔎 Mᴏᴠɪᴇꜱ Gʀᴏᴜᴘ", url=Config.UPDATES_CHANNEL_LINK)
            ]
        ])
        
        keyboard = InlineKeyboardMarkup(buttons)
        await message.reply_text(welcome_text, reply_markup=keyboard, disable_web_page_preview=True)
    
    @bot.on_callback_query(filters.regex(r"^check_verify_"))
    async def check_verify_callback(client, callback_query):
        user_id = callback_query.from_user.id
        try:
            is_verified, status = await check_url_shortener_verification(user_id)
            
            if is_verified:
                await callback_query.message.edit_text(
                    "✅ **Verification Successful!**\n\n"
                    "You Can Now Download Files From The Website.\n\n"
                    f"🌐 **Website:** {Config.WEBSITE_URL}\n\n"
                    "⏰ **Verification Valid For 6 Hours**",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)],
                        [
                            InlineKeyboardButton("📢 Mᴀɪɴ Cʜᴀɴɴᴇʟ", url=Config.MAIN_CHANNEL_LINK),
                            InlineKeyboardButton("🔎 Mᴏᴠɪᴇꜱ Gʀᴏᴜᴘ", url=Config.UPDATES_CHANNEL_LINK)
                        ]
                    ])
                )
            else:
                verification_url = await generate_verification_url(user_id)
                await callback_query.message.edit_text(
                    "❌ **Not Verified Yet**\n\n"
                    "Please complete the verification process first.\n\n"
                    f"🔗 **Verification URL:** `{verification_url}`",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("🔗 VERIFY NOW", url=verification_url)],
                        [InlineKeyboardButton("🔄 CHECK AGAIN", callback_data=f"check_verify_{user_id}")],
                        [
                            InlineKeyboardButton("📢 Mᴀɪɴ Cʜᴀɴɴᴇʟ", url=Config.MAIN_CHANNEL_LINK),
                            InlineKeyboardButton("🔎 Mᴏᴠɪᴇꜱ Gʀᴏᴜᴘ", url=Config.UPDATES_CHANNEL_LINK)
                        ]
                    ]),
                    disable_web_page_preview=True
                )
                
        except Exception as e:
            await callback_query.answer("Error checking verification", show_alert=True)
    
    @bot.on_message(filters.text & filters.private & ~filters.command(['start', 'stats', 'index', 'verify']))
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
    
    @bot.on_message(filters.command("verify") & filters.private)
    async def verify_command(client, message):
        user_id = message.from_user.id
        user_name = message.from_user.first_name or "User"
        
        if Config.VERIFICATION_REQUIRED:
            is_verified, status = await check_url_shortener_verification(user_id)
            
            if is_verified:
                await message.reply_text(
                    f"✅ **Already Verified, {user_name}!**\n\n"
                    f"Your verification is active and valid for 6 hours.\n\n"
                    "You can download files from the website now! 🎬",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)],
                        [
                            InlineKeyboardButton("📢 Mᴀɪɴ Cʜᴀɴɴᴇʟ", url=Config.MAIN_CHANNEL_LINK),
                            InlineKeyboardButton("🔎 Mᴏᴠɪᴇꜱ Gʀᴏᴜᴘ", url=Config.UPDATES_CHANNEL_LINK)
                        ]
                    ])
                )
            else:
                verification_url = await generate_verification_url(user_id)
                await message.reply_text(
                    f"🔗 **Verification Required, {user_name}**\n\n"
                    "To download files, please complete the URL verification:\n\n"
                    f"**Verification URL:** `{verification_url}`\n\n"
                    "⏰ **Valid for 6 hours after verification**",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("🔗 VERIFY NOW", url=verification_url)],
                        [
                            InlineKeyboardButton("📢 Mᴀɪɴ Cʜᴀɴɴᴇʟ", url=Config.MAIN_CHANNEL_LINK),
                            InlineKeyboardButton("🔎 Mᴏᴠɪᴇꜱ Gʀᴏᴜᴘ", url=Config.UPDATES_CHANNEL_LINK)
                        ]
                    ]),
                    disable_web_page_preview=True
                )
        else:
            await message.reply_text(
                "ℹ️ **Verification Not Required**\n\n"
                "URL shortener verification is currently disabled.\n"
                "You can download files directly from the website.",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)],
                    [
                        InlineKeyboardButton("📢 Mᴀɪɴ Cʜᴀɴɴᴇʟ", url=Config.MAIN_CHANNEL_LINK),
                        InlineKeyboardButton("🔎 Mᴏᴠɪᴇꜱ Gʀᴏᴜᴘ", url=Config.UPDATES_CHANNEL_LINK)
                    ]
                ])
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
        msg = await message.reply_text("🔄 **Starting background indexing...**")
        asyncio.create_task(index_files_background())
        await msg.edit_text("✅ **Indexing started in background!**\n\nCheck /stats for progress.")
    
    @bot.on_message(filters.command("stats") & filters.user(Config.ADMIN_IDS))
    async def stats_handler(client, message):
        tf = await files_col.count_documents({}) if files_col is not None else 0
        video_files = await files_col.count_documents({'is_video_file': True})
        video_thumbnails = await files_col.count_documents({'is_video_file': True, 'thumbnail': {'$ne': None}})
        
        stats_text = (
            f"📊 **SK4FiLM Statistics**\n\n"
            f"📁 **Total Files:** {tf}\n"
            f"🎥 **Video Files:** {video_files}\n"
            f"🖼️ **Video Thumbnails:** {video_thumbnails}\n\n"
            f"🔴 **Live Posts:** Active\n"
            f"🤖 **Bot Status:** Online\n\n"
            f"**🎨 Poster Sources:**\n"
            f"• Letterboxd: {movie_db['stats']['letterboxd']}\n"
            f"• IMDb: {movie_db['stats']['imdb']}\n"
            f"• JustWatch: {movie_db['stats']['justwatch']}\n"
            f"• IMPAwards: {movie_db['stats']['impawards']}\n"
            f"• OMDB: {movie_db['stats']['omdb']}\n"
            f"• TMDB: {movie_db['stats']['tmdb']}\n" 
            f"• Custom: {movie_db['stats']['custom']}\n"
            f"• Cache Hits: {movie_db['stats']['cache_hits']}\n\n"
            f"**⚡ Optimizations:**\n"
            f"• ✅ Smart caching\n"
            f"• ✅ Concurrent requests\n"
            f"• ✅ Batch processing\n\n"
            f"**🔗 Verification:** {'ENABLED (6 hours)' if Config.VERIFICATION_REQUIRED else 'DISABLED'}"
        )
        await message.reply_text(stats_text)

# OPTIMIZED INITIALIZATION
async def init():
    global User, bot, bot_started
    try:
        logger.info("🚀 INITIALIZING OPTIMIZED SK4FiLM BOT...")
        
        # PARALLEL INITIALIZATION
        mongo_success = await init_mongodb()
        
        User = Client(
            "user_session", 
            api_id=Config.API_ID, 
            api_hash=Config.API_HASH, 
            session_string=Config.USER_SESSION_STRING,
            no_updates=True
        )
        
        bot = Client(
            "bot",
            api_id=Config.API_ID,
            api_hash=Config.API_HASH, 
            bot_token=Config.BOT_TOKEN
        )
        
        await User.start()
        await bot.start()
        await setup_bot()
        
        me = await bot.get_me()
        logger.info(f"✅ OPTIMIZED BOT STARTED: @{me.username}")
        bot_started = True
        
        # START BACKGROUND TASKS
        asyncio.create_task(index_files_background())
        asyncio.create_task(cache_cleanup())
        
        return True
    except Exception as e:
        logger.error(f"❌ OPTIMIZED INIT FAILED: {e}")
        return False

async def main():
    logger.info("="*60)
    logger.info("🎬 SK4FiLM v6.0 - SUPER FAST OPTIMIZED VERSION")
    logger.info("✅ Smart Caching | Concurrent Requests | Batch Processing")
    logger.info(f"✅ Verification: {'ENABLED (6 hours)' if Config.VERIFICATION_REQUIRED else 'DISABLED'}")
    logger.info("="*60)
    
    success = await init()
    if not success:
        logger.error("❌ Failed to initialize optimized bot")
        return
    
    config = HyperConfig()
    config.bind = [f"0.0.0.0:{Config.WEB_SERVER_PORT}"]
    config.loglevel = "warning"
    
    logger.info(f"🌐 OPTIMIZED web server starting on port {Config.WEB_SERVER_PORT}...")
    await serve(app, config)

if __name__ == "__main__":
    asyncio.run(main())
