import asyncio
import os
import logging
import threading
from datetime import datetime, timedelta
from pyrogram import Client, filters
from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from flask import Flask, jsonify, request, Response
import html
import re
import math
import requests
import urllib.parse
import hashlib
import time
import json

# Logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s'
)
logger = logging.getLogger(__name__)

# MongoDB
try:
    from motor.motor_asyncio import AsyncIOMotorClient
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    logger.warning("MongoDB not available")

# ==================== CONFIG ====================
class Config:
    API_ID = int(os.environ.get("API_ID", "0"))
    API_HASH = os.environ.get("API_HASH", "")
    USER_SESSION_STRING = os.environ.get("USER_SESSION_STRING", "")
    BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
    
    TEXT_CHANNEL_IDS = [int(x) for x in os.environ.get("TEXT_CHANNEL_IDS", "-1001891090100,-1002024811395").split(",")]
    FILE_CHANNEL_ID = int(os.environ.get("FILE_CHANNEL_ID", "-1001768249569"))
    
    MONGODB_URI = os.environ.get("MONGODB_URI", "")
    DATABASE_NAME = os.environ.get("DATABASE_NAME", "sk4film_bot")
    
    FORCE_SUB_CHANNEL = int(os.environ.get("FORCE_SUB_CHANNEL", "-1001891090100"))
    WEBSITE_URL = os.environ.get("WEBSITE_URL", "https://sk4film.vercel.app")
    BOT_USERNAME = os.environ.get("BOT_USERNAME", "sk4film_bot")
    ADMIN_IDS = [int(x) for x in os.environ.get("ADMIN_IDS", "123456789").split(",")]
    AUTO_DELETE_TIME = int(os.environ.get("AUTO_DELETE_TIME", "300"))
    WEB_SERVER_PORT = int(os.environ.get("PORT", 8000))
    
    OMDB_KEYS = ["8265bd1c", "b9bd48a6", "2f2d1c8e", "c3e6f8d9"]
    TMDB_KEYS = ["e547e17d4e91f3e62a571655cd1ccaff", "2a9f7e1c3d4b5a6e"]
    BACKEND_URL = os.environ.get("BACKEND_URL", "https://sk4film.koyeb.app")

# ==================== FLASK APP ====================
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

# ==================== GLOBAL STATE ====================
bot_started = False
user_started = False
movie_db = {
    'home_movies': [],
    'last_update': None,
    'poster_cache': {},
    'updating': False,
    'stats': {'omdb_success': 0, 'tmdb_success': 0, 'custom': 0, 'omdb_fail': 0, 'tmdb_fail': 0}
}
file_registry = {}
loop = None
users_collection = None
stats_collection = None

# ==================== PYROGRAM ====================
bot = Client("sk4film_bot", api_id=Config.API_ID, api_hash=Config.API_HASH, bot_token=Config.BOT_TOKEN, workdir="/tmp", sleep_threshold=60)
User = Client("sk4film_user", api_id=Config.API_ID, api_hash=Config.API_HASH, session_string=Config.USER_SESSION_STRING, workdir="/tmp", sleep_threshold=60) if Config.USER_SESSION_STRING else None

# MongoDB
if MONGODB_AVAILABLE and Config.MONGODB_URI:
    try:
        mongo_client = AsyncIOMotorClient(Config.MONGODB_URI, serverSelectionTimeoutMS=5000)
        db = mongo_client[Config.DATABASE_NAME]
        users_collection = db.users
        stats_collection = db.stats
        logger.info("✅ MongoDB OK")
    except Exception as e:
        logger.error(f"MongoDB error: {e}")

# ==================== HELPERS ====================
def run_async(coro):
    """Execute async in event loop"""
    if loop and loop.is_running():
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        try:
            return future.result(timeout=30)
        except Exception as e:
            logger.error(f"Async exec error: {e}")
            return None
    logger.error("Event loop not running")
    return None

async def check_force_sub(user_id):
    try:
        member = await bot.get_chat_member(Config.FORCE_SUB_CHANNEL, user_id)
        return member.status in ["member", "administrator", "creator"]
    except:
        return True

def extract_title_smart(text):
    if not text or len(text) < 10:
        return None
    try:
        clean = re.sub(r'[^\w\s\(\)\-\.\u0900-\u097F]', ' ', text[:100])
        lines = [l.strip() for l in clean.split('\n') if l.strip()]
        if lines:
            first = lines[0]
            first = re.sub(r'^(Movie|Film|Watch|Download)[\s\:]+', '', first, flags=re.IGNORECASE)
            if 4 <= len(first) <= 50:
                return first.strip()
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
            name = re.sub(r'(720p|1080p|480p|HDRip|WEB|BluRay|x264|x265|HEVC|mkv|mp4)', '', name, flags=re.IGNORECASE)
            name = ' '.join(name.split()).strip()
            if 4 <= len(name) <= 50:
                return name
    except:
        pass
    return None

def format_size(size):
    if not size:
        return "Unknown"
    if size < 1024 * 1024:
        return f"{size/1024:.1f} KB"
    elif size < 1024 * 1024 * 1024:
        return f"{size/(1024*1024):.1f} MB"
    else:
        return f"{size/(1024*1024*1024):.2f} GB"

def detect_quality(fn):
    if not fn:
        return "480p"
    fl = fn.lower()
    if '1080p' in fl or 'fullhd' in fl or 'fhd' in fl:
        return "1080p"
    elif '720p' in fl or 'hd' in fl:
        return "720p"
    return "480p"

def format_post(text):
    if not text:
        return ""
    t = html.escape(text)
    t = re.sub(r'(https?://[^\s]+)', r'<a href="\1" target="_blank" style="color:#00ccff">\1</a>', t)
    return t.replace('\n', '<br>')

def channel_name(cid):
    return {-1001891090100: "SK4FiLM Main", -1002024811395: "SK4FiLM Updates", -1001768249569: "SK4FiLM Files"}.get(cid, "Channel")

def is_new(date):
    try:
        if isinstance(date, str):
            date = datetime.fromisoformat(date.replace('Z', '+00:00'))
        return (datetime.now() - date.replace(tzinfo=None)).seconds / 3600 <= 24
    except:
        return False

# ==================== POSTER (SYNC - WORKING) ====================
def get_poster_sync(title):
    """Synchronous poster fetching with detailed logging"""
    cache_key = title.lower().strip()
    
    # Cache check
    if cache_key in movie_db['poster_cache']:
        cached, cache_time = movie_db['poster_cache'][cache_key]
        if (datetime.now() - cache_time).seconds < 600:
            return cached
    
    logger.info(f"🎨 Fetching poster for: '{title}'")
    
    # Try OMDB with each key
    for i, api_key in enumerate(Config.OMDB_KEYS):
        try:
            url = f"http://www.omdbapi.com/?t={urllib.parse.quote(title)}&apikey={api_key}"
            logger.debug(f"  → OMDB attempt {i+1}/{len(Config.OMDB_KEYS)}")
            
            resp = requests.get(url, timeout=5)
            
            if resp.status_code == 200:
                data = resp.json()
                logger.debug(f"  → OMDB response: {data.get('Response')}")
                
                if data.get('Response') == 'True':
                    poster = data.get('Poster')
                    if poster and poster != 'N/A':
                        poster_url = poster.replace('http://', 'https://')
                        result = {
                            'poster_url': poster_url,
                            'title': data.get('Title', title),
                            'year': data.get('Year', ''),
                            'rating': data.get('imdbRating', ''),
                            'source': 'OMDB',
                            'success': True
                        }
                        movie_db['poster_cache'][cache_key] = (result, datetime.now())
                        movie_db['stats']['omdb_success'] += 1
                        logger.info(f"✅ OMDB poster found: {title}")
                        return result
        except Exception as e:
            logger.debug(f"  → OMDB error: {e}")
            movie_db['stats']['omdb_fail'] += 1
            continue
    
    # Try TMDB
    for i, api_key in enumerate(Config.TMDB_KEYS):
        try:
            url = "https://api.themoviedb.org/3/search/movie"
            params = {'api_key': api_key, 'query': title}
            logger.debug(f"  → TMDB attempt {i+1}/{len(Config.TMDB_KEYS)}")
            
            resp = requests.get(url, params=params, timeout=5)
            
            if resp.status_code == 200:
                data = resp.json()
                results = data.get('results', [])
                logger.debug(f"  → TMDB results: {len(results)}")
                
                if results:
                    movie = results[0]
                    poster_path = movie.get('poster_path')
                    
                    if poster_path:
                        result = {
                            'poster_url': f"https://image.tmdb.org/t/p/w500{poster_path}",
                            'title': movie.get('title', title),
                            'year': movie.get('release_date', '')[:4] if movie.get('release_date') else '',
                            'rating': f"{movie.get('vote_average', 0):.1f}",
                            'source': 'TMDB',
                            'success': True
                        }
                        movie_db['poster_cache'][cache_key] = (result, datetime.now())
                        movie_db['stats']['tmdb_success'] += 1
                        logger.info(f"✅ TMDB poster found: {title}")
                        return result
        except Exception as e:
            logger.debug(f"  → TMDB error: {e}")
            movie_db['stats']['tmdb_fail'] += 1
            continue
    
    # Fallback to custom
    logger.info(f"ℹ️ Using custom poster for: {title}")
    movie_db['stats']['custom'] += 1
    result = {
        'poster_url': f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(title)}",
        'title': title,
        'year': '',
        'rating': '',
        'source': 'CUSTOM',
        'success': True
    }
    movie_db['poster_cache'][cache_key] = (result, datetime.now())
    return result

# ==================== SEARCH ====================
async def search_movies_async(query, limit=12, page=1):
    """Complete search with detailed logging"""
    if not User or not user_started:
        logger.error("❌ User client not available for search")
        return {
            'results': [],
            'pagination': {
                'current_page': 1, 'total_pages': 1, 'total_results': 0,
                'per_page': limit, 'has_next': False, 'has_previous': False
            }
        }
    
    offset = (page - 1) * limit
    results = []
    seen = {}
    
    logger.info(f"🔍 SEARCH START: Query='{query}' | Page={page} | Limit={limit}")
    
    # Search text channels
    for cid in Config.TEXT_CHANNEL_IDS:
        try:
            logger.info(f"  📝 Searching text channel: {channel_name(cid)}")
            cnt = 0
            async for msg in User.search_messages(cid, query, limit=20):
                if cnt >= 20:
                    break
                
                if msg.text:
                    t = extract_title_smart(msg.text)
                    if t and t.lower() not in seen:
                        results.append({
                            'title': t,
                            'type': 'text_post',
                            'content': format_post(msg.text[:200]),
                            'date': msg.date.isoformat() if msg.date else datetime.now().isoformat(),
                            'channel': channel_name(cid),
                            'channel_id': cid,
                            'message_id': msg.id,
                            'is_new': is_new(msg.date) if msg.date else False,
                            'has_file': False,
                            'quality_options': {}
                        })
                        seen[t.lower()] = len(results) - 1
                        cnt += 1
                        logger.debug(f"    ✓ Found text: {t}")
            
            logger.info(f"  ✅ Text channel: {cnt} posts found")
        except Exception as e:
            logger.error(f"  ❌ Text channel {cid} error: {e}")
    
    # Search file channel
    try:
        logger.info(f"  📁 Searching file channel: {channel_name(Config.FILE_CHANNEL_ID)}")
        cnt = 0
        async for msg in User.search_messages(Config.FILE_CHANNEL_ID, query, limit=30):
            if cnt >= 30:
                break
            
            if msg.document or msg.video:
                t = extract_title_from_file(msg)
                if t:
                    fid = msg.document.file_id if msg.document else msg.video.file_id
                    fsz = msg.document.file_size if msg.document else (msg.video.file_size if msg.video else 0)
                    fnm = msg.document.file_name if msg.document else (msg.video.file_name if msg.video else 'video.mp4')
                    q = detect_quality(fnm)
                    uid = hashlib.md5(f"{fid}{time.time()}".encode()).hexdigest()
                    
                    file_registry[uid] = {
                        'file_id': fid,
                        'channel_id': Config.FILE_CHANNEL_ID,
                        'message_id': msg.id,
                        'quality': q,
                        'file_size': fsz,
                        'file_name': fnm,
                        'title': t,
                        'created_at': datetime.now()
                    }
                    
                    tk = t.lower()
                    if tk in seen:
                        idx = seen[tk]
                        results[idx]['has_file'] = True
                        results[idx]['type'] = 'with_file'
                        results[idx]['quality_options'][q] = {
                            'file_id': uid,
                            'file_size': fsz,
                            'file_name': fnm
                        }
                        logger.debug(f"    ✓ Added quality {q} to: {t}")
                    else:
                        results.append({
                            'title': t,
                            'type': 'with_file',
                            'content': format_post(msg.caption or t)[:200] if msg.caption else t,
                            'date': msg.date.isoformat() if msg.date else datetime.now().isoformat(),
                            'channel': channel_name(Config.FILE_CHANNEL_ID),
                            'channel_id': Config.FILE_CHANNEL_ID,
                            'message_id': msg.id,
                            'is_new': is_new(msg.date) if msg.date else False,
                            'has_file': True,
                            'quality_options': {
                                q: {
                                    'file_id': uid,
                                    'file_size': fsz,
                                    'file_name': fnm
                                }
                            }
                        })
                        seen[tk] = len(results) - 1
                        logger.debug(f"    ✓ New file: {t} ({q})")
                    cnt += 1
        
        logger.info(f"  ✅ File channel: {cnt} files found")
    except Exception as e:
        logger.error(f"  ❌ File channel error: {e}")
    
    # Sort: files first, then by date
    results.sort(key=lambda x: (x['has_file'], x['date']), reverse=True)
    
    total = len(results)
    paginated = results[offset:offset+limit]
    
    logger.info(f"🔍 SEARCH COMPLETE: Total={total} | Showing={len(paginated)} | Page={page}/{math.ceil(total/limit) if total else 1}")
    
    return {
        'results': paginated,
        'pagination': {
            'current_page': page,
            'total_pages': math.ceil(total/limit) if total else 1,
            'total_results': total,
            'per_page': limit,
            'has_next': page < math.ceil(total/limit) if total else False,
            'has_previous': page > 1
        }
    }

async def get_home_movies_async():
    """Homepage movies with posters"""
    if not User or not user_started:
        logger.error("❌ User not ready")
        return []
    
    logger.info("🏠 Loading homepage movies...")
    movies = []
    seen = set()
    
    for cid in Config.TEXT_CHANNEL_IDS:
        try:
            cnt = 0
            async for msg in User.get_chat_history(cid, limit=20):
                if cnt >= 20:
                    break
                if msg.text and msg.date:
                    t = extract_title_smart(msg.text)
                    if t and t.lower() not in seen:
                        movies.append({
                            'title': t,
                            'date': msg.date.isoformat(),
                            'is_new': is_new(msg.date)
                        })
                        seen.add(t.lower())
                        cnt += 1
            logger.info(f"  ✓ Channel {channel_name(cid)}: {cnt} movies")
        except Exception as e:
            logger.error(f"  ✗ Channel {cid} error: {e}")
    
    movies.sort(key=lambda x: x['date'], reverse=True)
    movies = movies[:24]
    
    logger.info(f"🎨 Fetching posters for {len(movies)} movies...")
    
    # Fetch posters
    for i, movie in enumerate(movies):
        try:
            poster = get_poster_sync(movie['title'])
            if poster and poster.get('success'):
                movie.update({
                    'poster_url': poster['poster_url'],
                    'poster_title': poster.get('title', movie['title']),
                    'poster_year': poster.get('year', ''),
                    'poster_rating': poster.get('rating', ''),
                    'poster_source': poster['source'],
                    'has_poster': True
                })
                logger.debug(f"  {i+1}/{len(movies)} ✓ {movie['title']} → {poster['source']}")
            else:
                movie.update({
                    'poster_url': f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(movie['title'])}",
                    'has_poster': True
                })
        except Exception as e:
            logger.error(f"  Poster error for {movie['title']}: {e}")
            movie.update({
                'poster_url': f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(movie['title'])}",
                'has_poster': True
            })
        
        # Small delay to avoid rate limiting
        if i % 5 == 0 and i > 0:
            time.sleep(0.5)
    
    logger.info(f"✅ Homepage complete: {len(movies)} movies with posters")
    return movies

# ==================== BOT HANDLERS ====================
@bot.on_message(filters.command("start") & filters.private)
async def start_handler(client, message):
    uid = message.from_user.id
    logger.info(f"👤 /start from user {uid}")
    
    if users_collection:
        try:
            await users_collection.update_one(
                {'user_id': uid},
                {'$set': {'first_name': message.from_user.first_name, 'last_seen': datetime.now()}, '$inc': {'start_count': 1}},
                upsert=True
            )
        except:
            pass
    
    # File delivery
    if len(message.command) > 1:
        fuid = message.command[1]
        logger.info(f"📥 File request: {fuid}")
        
        if not await check_force_sub(uid):
            try:
                ch = await bot.get_chat(Config.FORCE_SUB_CHANNEL)
                lk = f"https://t.me/{ch.username}" if ch.username else f"https://t.me/c/{str(Config.FORCE_SUB_CHANNEL)[4:]}/1"
            except:
                lk = f"https://t.me/c/{str(Config.FORCE_SUB_CHANNEL)[4:]}/1"
            
            await message.reply_text(
                "⚠️ **Access Denied**\n\nJoin channel first, then click download again.",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("📢 Join Now", url=lk)]])
            )
            logger.info(f"  ⚠️ Not subscribed")
            return
        
        fi = file_registry.get(fuid)
        if not fi:
            await message.reply_text(
                "❌ **File Not Found**\n\nLink expired. Search again.",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔍 Search", url=Config.WEBSITE_URL)]])
            )
            logger.warning(f"  ❌ File {fuid} not in registry")
            return
        
        try:
            pm = await message.reply_text(
                f"⏳ **Sending File...**\n\n"
                f"📁 {fi['file_name']}\n"
                f"📊 Quality: {fi['quality']}\n"
                f"📦 Size: {format_size(fi['file_size'])}"
            )
            
            if User:
                fm = await User.get_messages(fi['channel_id'], fi['message_id'])
                sent = await fm.copy(uid)
                await pm.delete()
                
                sm = await message.reply_text(
                    f"✅ **File Sent!**\n\n"
                    f"🎬 {fi['title']}\n"
                    f"📊 {fi['quality']}\n"
                    f"📦 {format_size(fi['file_size'])}\n\n"
                    f"⚠️ Will auto-delete in {Config.AUTO_DELETE_TIME//60} min"
                )
                
                logger.info(f"✅ File delivered: {fi['title']} ({fi['quality']}) → {uid}")
                
                # Stats
                if stats_collection:
                    try:
                        await stats_collection.insert_one({
                            'user_id': uid,
                            'file_id': fuid,
                            'quality': fi['quality'],
                            'title': fi['title'],
                            'timestamp': datetime.now(),
                            'type': 'download'
                        })
                    except:
                        pass
                
                # Auto-delete
                if Config.AUTO_DELETE_TIME > 0:
                    await asyncio.sleep(Config.AUTO_DELETE_TIME)
                    try:
                        await sent.delete()
                        await sm.edit_text("🗑️ **Auto-Deleted**\n\nFile removed for security.")
                        logger.info(f"🗑️ Auto-deleted file for user {uid}")
                    except:
                        pass
        except Exception as e:
            logger.error(f"❌ File delivery error: {e}")
            await message.reply_text("❌ Error sending file. Try again.")
        
        return
    
    # Normal start
    await message.reply_text(
        f"🎬 **Welcome to SK4FiLM!**\n\n"
        f"📌 **Bot Function:** File delivery only\n\n"
        f"**How to use:**\n"
        f"1️⃣ Visit website below\n"
        f"2️⃣ Search any movie\n"
        f"3️⃣ Select quality (480p/720p/1080p)\n"
        f"4️⃣ Bot sends file instantly\n\n"
        f"⚡ **Fast • Automated • Secure**",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🌐 Visit SK4FiLM Website", url=Config.WEBSITE_URL)]])
    )

@bot.on_message(filters.text & filters.private & ~filters.command(['start', 'stats', 'test']))
async def text_handler(client, message):
    logger.info(f"💬 Message from {message.from_user.id}: '{message.text[:30]}'")
    await message.reply_text(
        f"👋 Hi **{message.from_user.first_name}**!\n\n"
        f"🤖 Please use our website to search.\n\n"
        f"Bot delivers files only.",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🌐 Website", url=Config.WEBSITE_URL)]])
    )

@bot.on_message(filters.command("test") & filters.user(Config.ADMIN_IDS))
async def test_handler(client, message):
    """Test command for debugging"""
    test_results = []
    
    # Test user client
    if User and user_started:
        test_results.append("✅ User client: OK")
        try:
            me = await User.get_me()
            test_results.append(f"✅ User: {me.first_name}")
        except:
            test_results.append("❌ User get_me failed")
    else:
        test_results.append("❌ User client: Not started")
    
    # Test search
    try:
        result = await search_movies_async("test", 5, 1)
        test_results.append(f"✅ Search: {result['pagination']['total_results']} results")
    except Exception as e:
        test_results.append(f"❌ Search: {e}")
    
    # Test poster
    try:
        poster = get_poster_sync("Avengers")
        test_results.append(f"✅ Poster: {poster['source']}")
    except Exception as e:
        test_results.append(f"❌ Poster: {e}")
    
    await message.reply_text("\n".join(test_results))

@bot.on_message(filters.command("stats") & filters.user(Config.ADMIN_IDS))
async def stats_handler(client, message):
    try:
        tu = await users_collection.count_documents({}) if users_collection else 0
        td = await stats_collection.count_documents({'type': 'download'}) if stats_collection else 0
        
        await message.reply_text(
            f"📊 **SK4FiLM Stats**\n\n"
            f"👥 Users: `{tu}`\n"
            f"⬇️ Downloads: `{td}`\n"
            f"🎬 Movies: `{len(movie_db['home_movies'])}`\n"
            f"📁 Files: `{len(file_registry)}`\n"
            f"🖼️ Cache: `{len(movie_db['poster_cache'])}`\n\n"
            f"**Posters:**\n"
            f"✅ OMDB: `{movie_db['stats']['omdb_success']}`\n"
            f"✅ TMDB: `{movie_db['stats']['tmdb_success']}`\n"
            f"🎨 Custom: `{movie_db['stats']['custom']}`\n"
            f"❌ OMDB Fails: `{movie_db['stats']['omdb_fail']}`\n"
            f"❌ TMDB Fails: `{movie_db['stats']['tmdb_fail']}`\n\n"
            f"🤖 Bot: `{'✅' if bot_started else '❌'}`\n"
            f"👤 User: `{'✅' if user_started else '❌'}`"
        )
    except Exception as e:
        await message.reply_text(f"❌ Error: {e}")

# ==================== FLASK ROUTES ====================
@app.route('/')
def root():
    return jsonify({
        'status': 'healthy',
        'service': 'SK4FiLM Backend',
        'bot': f'@{Config.BOT_USERNAME}',
        'backend': Config.BACKEND_URL,
        'website': Config.WEBSITE_URL,
        'bot_running': bot_started,
        'user_running': user_started,
        'movies_loaded': len(movie_db['home_movies']),
        'files_in_registry': len(file_registry)
    })

@app.route('/health')
def health():
    is_healthy = bot_started and user_started
    return jsonify({
        'status': 'healthy' if is_healthy else 'starting',
        'bot': bot_started,
        'user': user_started,
        'movies': len(movie_db['home_movies']),
        'files': len(file_registry)
    }), 200 if is_healthy else 503

@app.route('/api/movies')
def api_movies():
    try:
        logger.info("📱 API: /api/movies called")
        
        if not user_started:
            logger.warning("⚠️ User not started")
            return jsonify({'status': 'error', 'message': 'Service starting'}), 503
        
        # Update if needed
        should_update = (
            not movie_db['home_movies'] or
            not movie_db['last_update'] or
            (datetime.now() - movie_db['last_update']).seconds > 300
        )
        
        if should_update and not movie_db['updating']:
            logger.info("🔄 Auto-updating homepage...")
            movie_db['updating'] = True
            
            movies = run_async(get_home_movies_async())
            if movies:
                movie_db['home_movies'] = movies
                movie_db['last_update'] = datetime.now()
                logger.info(f"✅ Updated: {len(movies)} movies")
            else:
                logger.error("❌ Update failed")
            
            movie_db['updating'] = False
        
        logger.info(f"📤 Sending {len(movie_db['home_movies'])} movies")
        
        return jsonify({
            'status': 'success',
            'movies': movie_db['home_movies'],
            'total': len(movie_db['home_movies']),
            'bot_username': Config.BOT_USERNAME,
            'last_update': movie_db['last_update'].isoformat() if movie_db['last_update'] else None
        })
    except Exception as e:
        logger.error(f"❌ API movies error: {e}")
        movie_db['updating'] = False
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/search')
def api_search():
    try:
        q = request.args.get('query', '').strip()
        p = int(request.args.get('page', 1))
        l = int(request.args.get('limit', 12))
        
        logger.info(f"📱 API: /api/search called | Query='{q}' | Page={p}")
        
        if not q:
            return jsonify({'status': 'error', 'message': 'Query required'}), 400
        
        if not user_started:
            logger.warning("⚠️ User not started")
            return jsonify({'status': 'error', 'message': 'Service starting'}), 503
        
        result = run_async(search_movies_async(q, l, p))
        
        if result:
            logger.info(f"📤 Sending {len(result['results'])} results")
            return jsonify({
                'status': 'success',
                'query': q,
                'results': result['results'],
                'pagination': result['pagination'],
                'bot_username': Config.BOT_USERNAME
            })
        else:
            logger.error("❌ Search returned None")
            return jsonify({'status': 'error', 'message': 'Search failed'}), 500
    except Exception as e:
        logger.error(f"❌ API search error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/poster')
def api_poster():
    """Custom SVG poster"""
    t = request.args.get('title', 'Movie')
    d = t[:18] + "..." if len(t) > 18 else t
    
    colors = [
        ('#667eea','#764ba2'), ('#f093fb','#f5576c'), ('#4facfe','#00f2fe'),
        ('#43e97b','#38f9d7'), ('#fa709a','#fee140'), ('#30cfd0','#330867'),
        ('#a8edea','#fed6e3'), ('#ff9a9e','#fecfef')
    ]
    c = colors[hash(t) % len(colors)]
    
    svg = f'''<svg width="300" height="450" xmlns="http://www.w3.org/2000/svg">
        <defs><linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" style="stop-color:{c[0]}"/>
        <stop offset="100%" style="stop-color:{c[1]}"/>
        </linearGradient></defs>
        <rect width="100%" height="100%" fill="url(#bg)" rx="20"/>
        <circle cx="150" cy="180" r="50" fill="rgba(255,255,255,0.2)"/>
        <text x="50%" y="200" text-anchor="middle" fill="#fff" font-size="50">🎬</text>
        <text x="50%" y="270" text-anchor="middle" fill="#fff" font-size="18" font-weight="bold">{html.escape(d)}</text>
        <rect x="50" y="380" width="200" height="40" rx="20" fill="rgba(0,0,0,0.3)"/>
        <text x="50%" y="405" text-anchor="middle" fill="#fff" font-size="18" font-weight="700">SK4FiLM</text>
    </svg>'''
    
    return Response(svg, mimetype='image/svg+xml', headers={'Cache-Control': 'public, max-age=3600'})

@app.route('/api/debug')
def api_debug():
    """Debug endpoint"""
    return jsonify({
        'bot_started': bot_started,
        'user_started': user_started,
        'movies_count': len(movie_db['home_movies']),
        'file_registry': len(file_registry),
        'poster_cache': len(movie_db['poster_cache']),
        'poster_stats': movie_db['stats'],
        'last_update': movie_db['last_update'].isoformat() if movie_db['last_update'] else None,
        'config': {
            'text_channels': len(Config.TEXT_CHANNEL_IDS),
            'backend_url': Config.BACKEND_URL,
            'website_url': Config.WEBSITE_URL
        }
    })

# ==================== STARTUP ====================
async def start_bot_client():
    global bot_started
    try:
        logger.info("🤖 Starting bot...")
        await bot.start()
        me = await bot.get_me()
        logger.info(f"✅ Bot online: @{me.username}")
        bot_started = True
    except Exception as e:
        logger.error(f"❌ Bot error: {e}")

async def start_user_client():
    global user_started
    if not User:
        logger.error("❌ USER_SESSION_STRING missing")
        return
    
    try:
        logger.info("👤 Starting user client...")
        await User.start()
        me = await User.get_me()
        logger.info(f"✅ User online: {me.first_name}")
        user_started = True
        
        # Initial load
        logger.info("📥 Loading initial data...")
        movie_db['home_movies'] = await get_home_movies_async()
        movie_db['last_update'] = datetime.now()
        logger.info("✅ Initial load complete")
    except Exception as e:
        logger.error(f"❌ User error: {e}")

def run_flask():
    """Flask server thread"""
    logger.info(f"🌐 Flask starting on 0.0.0.0:{Config.WEB_SERVER_PORT}")
    app.run(host='0.0.0.0', port=Config.WEB_SERVER_PORT, debug=False, use_reloader=False, threaded=True)

async def run_bots():
    """Pyrogram clients"""
    global loop
    loop = asyncio.get_event_loop()
    
    await start_bot_client()
    await start_user_client()
    
    logger.info("✅ All services running")
    logger.info(f"🌐 Backend: {Config.BACKEND_URL}")
    logger.info(f"🎬 Website: {Config.WEBSITE_URL}")
    logger.info(f"🤖 Bot: @{Config.BOT_USERNAME}")
    
    # Keep alive
    try:
        await asyncio.Event().wait()
    except:
        pass

def main():
    logger.info("=" * 70)
    logger.info("🚀 SK4FiLM - Complete Movie Platform")
    logger.info("=" * 70)
    logger.info(f"🤖 Bot Username: @{Config.BOT_USERNAME}")
    logger.info(f"🌐 Website: {Config.WEBSITE_URL}")
    logger.info(f"📡 Backend: {Config.BACKEND_URL}")
    logger.info(f"🔌 Port: {Config.WEB_SERVER_PORT}")
    logger.info(f"📝 Text Channels: {len(Config.TEXT_CHANNEL_IDS)}")
    logger.info(f"📁 File Channel: 1")
    logger.info("=" * 70)
    
    # Flask in thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    logger.info("✅ Flask thread started")
    
    time.sleep(3)  # Let Flask start
    
    # Bots in main thread
    try:
        asyncio.run(run_bots())
    except KeyboardInterrupt:
        logger.info("\n⚠️ Stopped")
    except Exception as e:
        logger.error(f"❌ Fatal: {e}")

if __name__ == "__main__":
    main()
