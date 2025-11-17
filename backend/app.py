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
import random

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
    FORCE_SUB_CHANNEL = -1002555323872
    
    WEBSITE_URL = os.environ.get("WEBSITE_URL", "https://sk4film.vercel.app")
    BOT_USERNAME = os.environ.get("BOT_USERNAME", "sk4filmbot")
    ADMIN_IDS = [int(x) for x in os.environ.get("ADMIN_IDS", "123456789").split(",")]
    AUTO_DELETE_TIME = int(os.environ.get("AUTO_DELETE_TIME", "300"))
    WEB_SERVER_PORT = int(os.environ.get("PORT", 8000))
    BACKEND_URL = os.environ.get("BACKEND_URL", "https://sk4film.koyeb.app")
    
    OMDB_KEYS = ["8265bd1c", "b9bd48a6", "3e7e1e9d"]
    TMDB_KEYS = ["e547e17d4e91f3e62a571655cd1ccaff", "8265bd1f"]

app = Quart(__name__)

@app.after_request
async def add_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

mongo_client = None
db = None
files_col = None

async def init_mongodb():
    global mongo_client, db, files_col
    try:
        logger.info("🔌 MongoDB (Files Only)...")
        mongo_client = AsyncIOMotorClient(Config.MONGODB_URI, serverSelectionTimeoutMS=10000)
        await mongo_client.admin.command('ping')
        
        db = mongo_client.sk4film
        files_col = db.files
        
        try:
            await files_col.create_index([("title", "text")])
        except:
            pass
        
        try:
            await files_col.create_index([("normalized_title", 1)])
        except:
            pass
        
        try:
            await files_col.drop_index("message_id_1_channel_id_1")
            logger.info("  Dropped old unique index")
        except:
            pass
        
        try:
            await files_col.create_index(
                [("message_id", 1), ("channel_id", 1)], 
                unique=True,
                name="msg_ch_unique_idx"
            )
        except:
            pass
        
        try:
            await files_col.create_index([("indexed_at", -1)])
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
movie_db = {
    'poster_cache': {},
    'stats': {
        'omdb': 0,
        'tmdb': 0,
        'impawards': 0,
        'justwatch': 0,
        'letterboxd': 0,
        'custom': 0,
        'cache_hits': 0
    }
}

# 🎬 EASTER EGG CONTENT 🎬
MOVIE_EXCUSES = [
    "🎭 OMDB is currently starring in 'Server Down: The Sequel'",
    "🎬 TMDB took a coffee break (they're on attempt #47 of making the perfect espresso)",
    "🎥 IMPAwards is being dramatic again... classic IMPAwards",
    "🍿 JustWatch is justifying why they need another retry",
    "📽️ Letterboxd wrote a 2000-word essay on why they can't fetch right now",
    "🎞️ All APIs are in a production meeting arguing about poster aspect ratios",
    "🎪 The internet is buffering... much like our patience",
    "🎨 Our custom SVG generator is practicing its art skills (it's getting REALLY good)",
]

RETRY_WISDOM = [
    "☕ Attempt #{n}: Because {n-1} wasn't enough apparently",
    "🔄 Attempt #{n}: Third time's a charm... or fourth... or fifth...",
    "⏰ Attempt #{n}: We're not stubborn, we're... persistent",
    "🎯 Attempt #{n}: Eventually, the server has to respond. It's the law.",
    "🚀 Attempt #{n}: Houston, we have a retry",
    "🎲 Attempt #{n}: Rolling the API dice again",
    "💪 Attempt #{n}: Never give up, never surrender!",
    "🎵 Attempt #{n}: 'Try, try again' - Ancient Developer Proverb",
]

POSTER_SUCCESS_STYLES = [
    "✨ LEGENDARY",
    "🔥 FIRE",
    "💎 DIAMOND TIER",
    "👑 ROYAL",
    "⚡ SPEED DEMON",
    "🌟 STELLAR",
    "🎯 BULLSEYE",
    "🏆 CHAMPION",
]

def get_retry_message(attempt, max_retries, source):
    """Inject personality into retry attempts"""
    if attempt == 0:
        return f"🎬 {source}: Rolling out the red carpet..."
    elif attempt == max_retries - 1:
        return f"⚠️ {source}: This is our last shot! 🎯"
    else:
        wisdom = random.choice(RETRY_WISDOM).replace("{n}", str(attempt + 1)).replace("{n-1}", str(attempt))
        return f"{wisdom} ({source})"

def get_failure_excuse():
    """Because every API failure deserves a dramatic excuse"""
    return random.choice(MOVIE_EXCUSES)

def get_success_style():
    """Celebrate those poster wins!"""
    return random.choice(POSTER_SUCCESS_STYLES)

def normalize_title(title):
    if not title:
        return ""
    normalized = title.lower().strip()
    normalized = re.sub(r'\b(19|20)\d{2}\b', '', normalized)
    normalized = re.sub(r'\b(480p|720p|1080p|2160p|4k|hd|fhd|uhd|hevc|x264|x265|h264|h265|bluray|webrip|hdrip|web-dl|hdtv)\b', '', normalized, flags=re.IGNORECASE)
    normalized = ' '.join(normalized.split()).strip()
    return normalized

def extract_title_smart(text):
    if not text or len(text) < 10:
        return None
    try:
        clean = re.sub(r'[^\w\s\(\)\-\.\n:]', ' ', text)
        lines = [l.strip() for l in clean.split('\n') if l.strip()]
        
        if not lines:
            return None
        
        first_line = lines[0]
        
        m = re.search(r'🎬\s*([^\n\-\(]{3,60})', first_line)
        if m:
            title = m.group(1).strip()
            title = re.sub(r'\s+', ' ', title)
            if 3 <= len(title) <= 60:
                return title
        
        m = re.search(r'^([^\(\n]{3,60})\s*\(\d{4}\)', first_line)
        if m:
            title = m.group(1).strip()
            if 3 <= len(title) <= 60:
                return title
        
        m = re.search(r'^([^\-\n]{3,60})\s*-', first_line)
        if m:
            title = m.group(1).strip()
            title = re.sub(r'\s+', ' ', title)
            if 3 <= len(title) <= 60:
                return title
        
        if len(first_line) >= 3 and len(first_line) <= 60:
            title = re.sub(r'\b(480p|720p|1080p|2160p|4k|hevc|x264|x265)\b', '', first_line, flags=re.IGNORECASE)
            title = re.sub(r'\s+', ' ', title).strip()
            if 3 <= len(title) <= 60:
                return title
    except:
        pass
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
            name = re.sub(r'(720p|1080p|480p|2160p|HDRip|WEB|BluRay|x264|x265|HEVC)', '', name, flags=re.IGNORECASE)
            name = ' '.join(name.split()).strip()
            if 4 <= len(name) <= 50:
                return name
    except:
        pass
    return None

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

async def check_force_sub(user_id, max_retries=3):
    """Enhanced force subscription check with retry mechanism"""
    for attempt in range(max_retries):
        try:
            logger.info(f"🔍 Checking subscription for user {user_id} (Attempt {attempt + 1}/{max_retries})")
            
            if attempt > 0:
                await asyncio.sleep(2)
            
            member = await bot.get_chat_member(Config.FORCE_SUB_CHANNEL, user_id)
            is_member = member.status in ["member", "administrator", "creator"]
            
            logger.info(f"  {'✅' if is_member else '❌'} Status: {member.status}")
            
            if is_member:
                return True
                
        except UserNotParticipant:
            logger.info(f"  ❌ User {user_id} not in channel (Attempt {attempt + 1})")
            if attempt == max_retries - 1:
                return False
        except (ChatAdminRequired, ChannelPrivate):
            logger.warning(f"  ⚠️ Bot permission issue - allowing access")
            return True
        except Exception as e:
            logger.error(f"  ❌ Force sub error (Attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return True
    
    return False

async def index_files_background():
    """Background file indexing - non-blocking"""
    if not User or files_col is None:
        logger.warning("⚠️ Cannot index in background")
        return
    
    logger.info("📁 Starting background file indexing...")
    
    try:
        count = 0
        batch = []
        batch_size = 50
        
        async for msg in User.get_chat_history(Config.FILE_CHANNEL_ID):
            if msg.document or msg.video:
                title = extract_title_from_file(msg)
                if title:
                    file_id = msg.document.file_id if msg.document else msg.video.file_id
                    file_size = msg.document.file_size if msg.document else (msg.video.file_size if msg.video else 0)
                    file_name = msg.document.file_name if msg.document else (msg.video.file_name if msg.video else 'video.mp4')
                    quality = detect_quality(file_name)
                    
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
                        'indexed_at': datetime.now()
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
                            logger.info(f"    ✅ Indexed {count} files...")
                            batch = []
                        except Exception as e:
                            logger.error(f"Batch error: {e}")
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
                logger.error(f"Final batch error: {e}")
        
        logger.info(f"✅ Background indexing complete: {count} files")
        
    except Exception as e:
        logger.error(f"❌ Background indexing error: {e}")

async def get_poster_guaranteed(title, session, max_retries=3):
    """100% GUARANTEED poster with retry mechanism and ALL sources (NOW WITH EASTER EGG!)"""
    ck = title.lower().strip()
    
    if ck in movie_db['poster_cache']:
        c, ct = movie_db['poster_cache'][ck]
        if (datetime.now() - ct).seconds < 3600:
            movie_db['stats']['cache_hits'] += 1
            logger.info(f"  📦 Cache hit: {title}")
            return c
    
    year_match = re.search(r'\b(19|20)\d{2}\b', title)
    year = year_match.group() if year_match else None
    clean_title = re.sub(r'\b(19|20)\d{2}\b', '', title).strip()
    
    logger.info(f"  🎨 Fetching poster: {title}")
    
    for attempt in range(max_retries):
        # OMDB
        for api_key in Config.OMDB_KEYS:
            try:
                logger.info(get_retry_message(attempt, max_retries, "OMDB"))
                url = f"http://www.omdbapi.com/?t={urllib.parse.quote(title)}&apikey={api_key}"
                async with session.get(url, timeout=8) as r:
                    if r.status == 200:
                        data = await r.json()
                        if data.get('Response') == 'True' and data.get('Poster') and data.get('Poster') != 'N/A':
                            poster_url = data['Poster'].replace('http://', 'https://')
                            res = {'poster_url': poster_url, 'source': 'OMDB', 'rating': data.get('imdbRating', '0.0')}
                            movie_db['poster_cache'][ck] = (res, datetime.now())
                            movie_db['stats']['omdb'] += 1
                            logger.info(f"    {get_success_style()} OMDB: {title}")
                            return res
            except Exception as e:
                logger.debug(f"    {get_failure_excuse()}")
                continue
        
        # TMDB
        for api_key in Config.TMDB_KEYS:
            try:
                logger.info(get_retry_message(attempt, max_retries, "TMDB"))
                url = "https://api.themoviedb.org/3/search/movie"
                params = {'api_key': api_key, 'query': title}
                async with session.get(url, params=params, timeout=8) as r:
                    if r.status == 200:
                        data = await r.json()
                        if data.get('results') and len(data['results']) > 0:
                            result = data['results'][0]
                            poster_path = result.get('poster_path')
                            if poster_path:
                                poster_url = f"https://image.tmdb.org/t/p/w780{poster_path}"
                                res = {'poster_url': poster_url, 'source': 'TMDB', 'rating': str(result.get('vote_average', 0.0))}
                                movie_db['poster_cache'][ck] = (res, datetime.now())
                                movie_db['stats']['tmdb'] += 1
                                logger.info(f"    {get_success_style()} TMDB: {title}")
                                return res
            except Exception as e:
                logger.debug(f"    {get_failure_excuse()}")
                continue
        
        # IMPAwards
        if year:
            try:
                logger.info(get_retry_message(attempt, max_retries, "IMPAwards"))
                slug = clean_title.lower()
                slug = re.sub(r'[^\w\s]', '', slug)
                slug = slug.replace(' ', '_')
                
                formats = [
                    f"https://www.impawards.com/{year}/posters/{slug}.jpg",
                    f"https://www.impawards.com/{year}/posters/{slug}_ver2.jpg",
                    f"https://www.impawards.com/{year}/posters/{slug}_xlg.jpg",
                    f"https://www.impawards.com/{year}/posters/{slug}_ver3.jpg"
                ]
                
                for poster_url in formats:
                    try:
                        async with session.head(poster_url, timeout=5) as r:
                            if r.status == 200:
                                res = {'poster_url': poster_url, 'source': 'IMPAwards', 'rating': '0.0'}
                                movie_db['poster_cache'][ck] = (res, datetime.now())
                                movie_db['stats']['impawards'] += 1
                                logger.info(f"    {get_success_style()} IMPAwards: {title}")
                                return res
                    except:
                        continue
            except Exception as e:
                logger.debug(f"    {get_failure_excuse()}")
        
        # JustWatch
        try:
            logger.info(get_retry_message(attempt, max_retries, "JustWatch"))
            slug = clean_title.lower().replace(' ', '-')
            slug = re.sub(r'[^\w\-]', '', slug)
            justwatch_url = f"https://www.justwatch.com/us/movie/{slug}"
            
            async with session.get(justwatch_url, timeout=8, headers={'User-Agent': 'Mozilla/5.0'}) as r:
                if r.status == 200:
                    html_content = await r.text()
                    poster_match = re.search(r'<meta property="og:image" content="([^"]+)"', html_content)
                    if poster_match:
                        poster_url = poster_match.group(1)
                        if poster_url and poster_url.startswith('http'):
                            res = {'poster_url': poster_url, 'source': 'JustWatch', 'rating': '0.0'}
                            movie_db['poster_cache'][ck] = (res, datetime.now())
                            movie_db['stats']['justwatch'] += 1
                            logger.info(f"    {get_success_style()} JustWatch: {title}")
                            return res
        except Exception as e:
            logger.debug(f"    {get_failure_excuse()}")
        
        # Letterboxd
        try:
            logger.info(get_retry_message(attempt, max_retries, "Letterboxd"))
            slug = clean_title.lower().replace(' ', '-')
            slug = re.sub(r'[^\w\-]', '', slug)
            if year:
                slug = f"{slug}-{year}"
            
            letterboxd_url = f"https://letterboxd.com/film/{slug}/"
            
            async with session.get(letterboxd_url, timeout=8, headers={'User-Agent': 'Mozilla/5.0'}) as r:
                if r.status == 200:
                    html_content = await r.text()
                    poster_match = re.search(r'<meta property="og:image" content="([^"]+)"', html_content)
                    if poster_match:
                        poster_url = poster_match.group(1)
                        if poster_url and poster_url.startswith('http'):
                            poster_url = poster_url.replace('-0-500-0-750', '-0-1000-0-1500')
                            res = {'poster_url': poster_url, 'source': 'Letterboxd', 'rating': '0.0'}
                            movie_db['poster_cache'][ck] = (res, datetime.now())
                            movie_db['stats']['letterboxd'] += 1
                            logger.info(f"    {get_success_style()} Letterboxd: {title}")
                            return res
        except Exception as e:
            logger.debug(f"    {get_failure_excuse()}")
        
        # TMDB HD
        for api_key in Config.TMDB_KEYS:
            try:
                url = "https://api.themoviedb.org/3/search/movie"
                params = {'api_key': api_key, 'query': title}
                async with session.get(url, params=params, timeout=8) as r:
                    if r.status == 200:
                        data = await r.json()
                        if data.get('results'):
                            for result in data['results'][:3]:
                                poster_path = result.get('poster_path')
                                if poster_path:
                                    poster_url = f"https://image.tmdb.org/t/p/original{poster_path}"
                                    res = {'poster_url': poster_url, 'source': 'TMDB-HD', 'rating': str(result.get('vote_average', 0.0))}
                                    movie_db['poster_cache'][ck] = (res, datetime.now())
                                    movie_db['stats']['tmdb'] += 1
                                    logger.info(f"    {get_success_style()} TMDB-HD: {title}")
                                    return res
            except Exception as e:
                logger.debug(f"    {get_failure_excuse()}")
                continue
        
        if attempt < max_retries - 1:
            await asyncio.sleep(1)
    
    # Custom fallback
    logger.info(f"    🎨 All sources exhausted, deploying custom SVG masterpiece: {title}")
    movie_db['stats']['custom'] += 1
    res = {'poster_url': f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(title)}", 'source': 'CUSTOM', 'rating': '0.0'}
    movie_db['poster_cache'][ck] = (res, datetime.now())
    return res

async def get_live_posts(channel_id, limit=50):
    if not User:
        return []
    
    logger.info(f"🔴 LIVE: {channel_name(channel_id)} (limit: {limit})")
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
        
        logger.info(f"  ✅ {count} posts")
    except Exception as e:
        logger.error(f"  ❌ Error: {e}")
    
    return posts

async def search_movies_live(query, limit=12, page=1):
    """Enhanced search with post availability tracking"""
    offset = (page - 1) * limit
    logger.info(f"🔴 SEARCH: '{query}' | Page: {page}")
    
    query_lower = query.lower()
    posts_dict = {}
    files_dict = {}
    
    for channel_id in Config.TEXT_CHANNEL_IDS:
        try:
            cname = channel_name(channel_id)
            logger.info(f"  🔴 {cname}...")
            count = 0
            
            try:
                async for msg in User.search_messages(channel_id, query=query, limit=200):
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
                                    'quality_options': {}
                                }
                                count += 1
            except Exception as e:
                logger.error(f"    ❌ Search error: {e}")
            
            logger.info(f"    ✅ {count} posts")
            
        except Exception as e:
            logger.error(f"    ❌ Channel error: {e}")
    
    try:
        logger.info("📁 Files...")
        count = 0
        
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
                            'date': doc['date'].isoformat() if isinstance(doc['date'], datetime) else doc['date']
                        }
                    
                    if quality not in files_dict[norm_title]['quality_options']:
                        files_dict[norm_title]['quality_options'][quality] = {
                            'file_id': f"{doc.get('channel_id', Config.FILE_CHANNEL_ID)}_{doc.get('message_id')}_{quality}",
                            'file_size': doc['file_size'],
                            'file_name': doc['file_name']
                        }
                        count += 1
                except Exception as e:
                    logger.debug(f"File processing error: {e}")
        
        logger.info(f"  ✅ {count} files")
        
    except Exception as e:
        logger.error(f"  ❌ Files error: {e}")
    
    # Merge results
    merged = {}
    for norm_title, post_data in posts_dict.items():
        merged[norm_title] = post_data
    
    for norm_title, file_data in files_dict.items():
        if norm_title in merged:
            merged[norm_title]['has_file'] = True
            merged[norm_title]['quality_options'] = file_data['quality_options']
        else:
            merged[norm_title] = {
                'title': file_data['title'],
                'content': f"<p>{file_data['title']}</p>",
                'channel': 'SK4FiLM',
                'date': file_data['date'],
                'is_new': False,
                'has_file': True,
                'has_post': False,
                'quality_options': file_data['quality_options']
            }
    
    results_list = list(merged.values())
    results_list.sort(key=lambda x: (not x.get('is_new', False), not x['has_file'], x['date']), reverse=True)
    
    total = len(results_list)
    paginated = results_list[offset:offset + limit]
    
    logger.info(f"✅ Total: {total} | Page: {len(paginated)}")
    
    return {
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

async def get_home_movies_live():
    logger.info("🏠 Fetching 30 movies with 100% poster guarantee...")
    
    posts = await get_live_posts(Config.MAIN_CHANNEL_ID, limit=50)
    
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
    
    logger.info(f"  ✓ {len(movies)} movies ready for poster fetch")
    
    if movies:
        async with aiohttp.ClientSession() as session:
            tasks = [get_poster_guaranteed(m['title'], session) for m in movies]
            posters = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, poster in enumerate(posters):
                if isinstance(poster, dict):
                    movies[i]['poster'] = poster['poster_url']
                    movies[i]['source'] = poster.get('source', 'Unknown')
                    movies[i]['rating'] = poster.get('rating', '0.0')
                else:
                    movies[i]['poster'] = f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(movies[i]['title'])}"
                    movies[i]['source'] = 'CUSTOM'
                    movies[i]['rating'] = '0.0'
    
    logger.info(f"✅ {len(movies)} movies with posters")
    return movies

# 🎬 EASTER EGG HANDLER 🎬
async def secret_poster_stats_animation():
    """Easter egg that shows poster fetching stats with maximum personality"""
    
    stats_copy = movie_db['stats'].copy()
    
    response = "🎬 <b>SK4FiLM POSTER FETCHING CHAMPIONSHIPS</b> 🏆\n\n"
    response += "<i>Calculating the impossible...</i>\n"
    await asyncio.sleep(0.5)
    
    response += "\n━━━━━━━━━━━━━━━━━━━━━━\n\n"
    
    total_attempts = sum([stats_copy.get(k, 0) for k in ['omdb', 'tmdb', 'impawards', 'justwatch', 'letterboxd', 'custom']])
    
    if total_attempts == 0:
        return response + "🤔 No posters fetched yet. Did someone break the bot again?\n\n<code>Hint: Try searching for some movies first!</code>"
    
    # Source rankings with personality
    sources = [
        ('OMDB', stats_copy.get('omdb', 0), '🥇'),
        ('TMDB', stats_copy.get('tmdb', 0), '🥈'),
        ('IMPAwards', stats_copy.get('impawards', 0), '🥉'),
        ('JustWatch', stats_copy.get('justwatch', 0), '🎪'),
        ('Letterboxd', stats_copy.get('letterboxd', 0), '📽️'),
        ('Custom SVG', stats_copy.get('custom', 0), '🎨'),
    ]
    
    response += "📊 <b>SOURCE LEADERBOARD:</b>\n\n"
    
    for name, count, emoji in sorted(sources, key=lambda x: x[1], reverse=True):
        if count > 0:
            percentage = (count / total_attempts * 100)
            bar_length = int(percentage / 5)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            
            if name == 'Custom SVG' and count > 0:
                comment = " <- Our safety net 🕸️"
            elif percentage > 50:
                comment = f" <- {random.choice(['CRUSHING IT', 'MVP', 'THE GOAT', 'UNSTOPPABLE'])}"
            elif percentage > 25:
                comment = " <- Solid effort"
            else:
                comment = " <- Trying its best"
            
            response += f"{emoji} <b>{name}</b>: {count} ({percentage:.1f}%)\n"
            response += f"   {bar}{comment}\n\n"
    
    # Cache stats
    cache_hits = stats_copy.get('cache_hits', 0)
    if cache_hits > 0:
        cache_percentage = (cache_hits / (total_attempts + cache_hits) * 100)
        response += f"💾 <b>Cache Hits:</b> {cache_hits} ({cache_percentage:.1f}%)\n"
        response += f"   Translation: We saved {cache_hits * 3} seconds of your life\n\n"
    
    # Fun facts
    response += "━━━━━━━━━━━━━━━━━━━━━━\n\n"
    response += "🎭 <b>FUN FACTS:</b>\n\n"
    
    if stats_copy.get('custom', 0) > 0:
        response += f"• Our custom SVG generator created {stats_copy['custom']} masterpieces\n"
        response += "  (Picasso is shaking)\n\n"
    
    if stats_copy.get('omdb', 0) > stats_copy.get('tmdb', 0):
        response += "• OMDB is winning the API popularity contest\n"
        response += "  TMDB is somewhere crying in the corner\n\n"
    elif stats_copy.get('tmdb', 0) > stats_copy.get('omdb', 0):
        response += "• TMDB is flexing on OMDB right now\n"
        response += "  OMDB needs to step up its game\n\n"
    
    success_rate = ((total_attempts - stats_copy.get('custom', 0)) / total_attempts * 100) if total_attempts > 0 else 0
    if success_rate > 95:
        response += f"• {success_rate:.1f}% success rate - We're basically wizards 🧙‍♂️\n\n"
    elif success_rate > 80:
        response += f"• {success_rate:.1f}% success rate - Not bad for a Monday\n\n"
    else:
        response += f"• {success_rate:.1f}% success rate - APIs are having a bad day\n\n"
    
    # Total attempts
    response += f"• Total poster fetches: {total_attempts}\n"
    response += f"  That's a lot of movie art! 🎨\n\n"
    
    # Final message
    response += "━━━━━━━━━━━━━━━━━━━━━━\n\n"
    response += "💪 <i>Remember: We don't give up on posters.</i>\n"
    response += "🎯 <i>We retry until the APIs surrender.</i>\n"
    response += "🎬 <i>That's the SK4FiLM way.</i>\n\n"
    response += f"<code>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</code>"
    
    return response

# 🎬 BOT INITIALIZATION AND HANDLERS 🎬

async def start_bot():
    global User, bot, bot_started
    
    if bot_started:
        logger.warning("⚠️ Bot already running")
        return
    
    try:
        logger.info("🤖 Starting User Client...")
        User = Client(
            "sk4film_user",
            api_id=Config.API_ID,
            api_hash=Config.API_HASH,
            session_string=Config.USER_SESSION_STRING
        )
        await User.start()
        me = await User.get_me()
        logger.info(f"✅ User Client: @{me.username}")
        
        logger.info("🤖 Starting Bot Client...")
        bot = Client(
            "sk4film_bot",
            api_id=Config.API_ID,
            api_hash=Config.API_HASH,
            bot_token=Config.BOT_TOKEN
        )
        await bot.start()
        bot_me = await bot.get_me()
        logger.info(f"✅ Bot Client: @{bot_me.username}")
        
        # Initialize MongoDB
        await init_mongodb()
        
        # Start background indexing
        asyncio.create_task(index_files_background())
        
        bot_started = True
        logger.info("✅ All systems operational!")
        
    except Exception as e:
        logger.error(f"❌ Bot startup failed: {e}")
        raise

# 🎬 EASTER EGG BOT HANDLER 🎬
@bot.on_message(filters.command(["debug_posters", "posterstats", "stats"]) & filters.private)
async def handle_poster_stats(client, message):
    """🎬 SECRET EASTER EGG: Shows poster fetching stats with style"""
    
    try:
        loading_msg = await message.reply("🎬 Accessing the SKret vault... 🔐")
        
        await asyncio.sleep(1)
        await loading_msg.edit("🎭 Waking up the APIs... ☕")
        
        await asyncio.sleep(1)
        await loading_msg.edit("📊 Crunching the numbers... 🔢")
        
        await asyncio.sleep(1)
        
        stats_message = await secret_poster_stats_animation()
        await loading_msg.edit(stats_message, parse_mode="html", disable_web_page_preview=True)
        
    except Exception as e:
        logger.error(f"Easter egg error: {e}")
        await message.reply("🎬 Oops! The stats got stage fright. Try again!")

# Regular bot handlers
@bot.on_message(filters.command("start") & filters.private)
async def start_handler(client, message):
    """Start command handler"""
    user_id = message.from_user.id
    
    # Check force subscription
    if not await check_force_sub(user_id):
        try:
            invite_link = await bot.create_chat_invite_link(Config.FORCE_SUB_CHANNEL)
            await message.reply(
                f"⚠️ Please join our channel first:\n\n{invite_link.invite_link}",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("Join Channel", url=invite_link.invite_link)],
                    [InlineKeyboardButton("Try Again", callback_data="check_sub")]
                ])
            )
            return
        except:
            pass
    
    welcome_text = (
        f"🎬 <b>Welcome to SK4FiLM Bot!</b>\n\n"
        f"🔍 Search for movies and get direct download links\n"
        f"🎥 High quality content in multiple formats\n\n"
        f"<b>Commands:</b>\n"
        f"/search - Search for movies\n"
        f"/help - Get help\n\n"
        f"<i>Just send me a movie name to search!</i>"
    )
    
    await message.reply(
        welcome_text,
        parse_mode="html",
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🌐 Visit Website", url=Config.WEBSITE_URL)],
            [InlineKeyboardButton("📢 Channel", url=f"https://t.me/{Config.BOT_USERNAME}")]
        ])
    )

@bot.on_message(filters.command("help") & filters.private)
async def help_handler(client, message):
    """Help command handler"""
    help_text = (
        "🎬 <b>SK4FiLM Bot Help</b>\n\n"
        "<b>How to use:</b>\n"
        "• Send movie name to search\n"
        "• Click on results to get files\n"
        "• Enjoy your movie!\n\n"
        "<b>Commands:</b>\n"
        "/start - Start the bot\n"
        "/search <movie> - Search for movies\n"
        "/help - Show this message\n\n"
        "<i>Need support? Contact @SK4FiLM</i>"
    )
    
    await message.reply(help_text, parse_mode="html")

# 🎬 WEB API ROUTES 🎬

@app.route("/")
async def home():
    return jsonify({
        "status": "online",
        "service": "SK4FiLM API",
        "version": "2.0",
        "easter_egg": "Try /api/stats for poster fetching statistics! 🎬"
    })

@app.route("/api/movies")
async def api_movies():
    """Get home movies"""
    try:
        movies = await get_home_movies_live()
        return jsonify({"success": True, "movies": movies, "count": len(movies)})
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/search")
async def api_search():
    """Search movies"""
    try:
        query = request.args.get("q", "")
        page = int(request.args.get("page", 1))
        limit = int(request.args.get("limit", 12))
        
        if not query:
            return jsonify({"success": False, "error": "Query required"}), 400
        
        results = await search_movies_live(query, limit=limit, page=page)
        return jsonify({"success": True, **results})
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/stats")
async def api_stats():
    """🎬 EASTER EGG: Poster fetching stats API endpoint"""
    try:
        stats_copy = movie_db['stats'].copy()
        total = sum([stats_copy.get(k, 0) for k in ['omdb', 'tmdb', 'impawards', 'justwatch', 'letterboxd', 'custom']])
        
        return jsonify({
            "success": True,
            "easter_egg": "🎬 You found the secret stats endpoint!",
            "stats": stats_copy,
            "total_fetches": total,
            "success_rate": f"{((total - stats_copy.get('custom', 0)) / total * 100):.1f}%" if total > 0 else "0%",
            "message": "We never give up on posters. That's the SK4FiLM way. 🎯"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/poster")
async def api_poster():
    """Generate custom SVG poster"""
    title = request.args.get("title", "Movie")
    
    svg = f'''<svg width="300" height="450" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#667eea;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#764ba2;stop-opacity:1" />
            </linearGradient>
        </defs>
        <rect width="300" height="450" fill="url(#grad)"/>
        <text x="150" y="200" font-family="Arial" font-size="24" fill="white" text-anchor="middle" font-weight="bold">
            🎬
        </text>
        <text x="150" y="240" font-family="Arial" font-size="16" fill="white" text-anchor="middle" style="word-wrap: break-word">
            {html.escape(title[:30])}
        </text>
        <text x="150" y="270" font-family="Arial" font-size="12" fill="rgba(255,255,255,0.7)" text-anchor="middle">
            SK4FiLM
        </text>
    </svg>'''
    
    return Response(svg, mimetype='image/svg+xml')

# 🎬 MAIN EXECUTION 🎬

async def main():
    """Main application entry point"""
    logger.info("🎬 SK4FiLM Bot Starting...")
    logger.info("🥚 Easter Egg: Type /debug_posters or /posterstats in bot for surprise!")
    
    # Start bot
    await start_bot()
    
    # Configure web server
    config = HyperConfig()
    config.bind = [f"0.0.0.0:{Config.WEB_SERVER_PORT}"]
    config.use_reloader = False
    
    logger.info(f"🌐 Web server starting on port {Config.WEB_SERVER_PORT}")
    logger.info(f"🎯 API Stats endpoint: {Config.BACKEND_URL}/api/stats")
    
    # Start web server
    await serve(app, config)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Shutting down gracefully...")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
