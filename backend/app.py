import asyncio
import os
import logging
import signal
from datetime import datetime, timedelta
from pyrogram import Client, filters, idle
from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from pyrogram.errors import FloodWait
from quart import Quart, jsonify, request, Response
from hypercorn.asyncio import serve
from hypercorn.config import Config as HyperConfig
import html
import re
import math
import aiohttp
import urllib.parse
import hashlib
import time

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress asyncio warnings
logging.getLogger('asyncio').setLevel(logging.WARNING)

# MongoDB
try:
    from motor.motor_asyncio import AsyncIOMotorClient
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False

# ==================== CONFIG ====================
class Config:
    API_ID = int(os.environ.get("API_ID", "0"))
    API_HASH = os.environ.get("API_HASH", "")
    USER_SESSION_STRING = os.environ.get("USER_SESSION_STRING", "")
    BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
    
    TEXT_CHANNEL_IDS = [int(x) for x in os.environ.get("TEXT_CHANNEL_IDS", "-1001891090100,-1002024811395").split(",")]
    FILE_CHANNEL_ID = int(os.environ.get("FILE_CHANNEL_ID", "-1001768249569"))
    
    QUALITY_PATTERNS = {
        "1080p": [r"1080p", r"1080P", r"FullHD", r"FHD"],
        "720p": [r"720p", r"720P", r"HD"],
        "480p": [r"480p", r"480P", r"SD"]
    }
    
    MONGODB_URI = os.environ.get("MONGODB_URI", "")
    DATABASE_NAME = os.environ.get("DATABASE_NAME", "sk4film_bot")
    
    FORCE_SUB_CHANNEL = int(os.environ.get("FORCE_SUB_CHANNEL", "-1001891090100"))
    WEBSITE_URL = os.environ.get("WEBSITE_URL", "https://sk4film.vercel.app")
    BOT_USERNAME = os.environ.get("BOT_USERNAME", "sk4filmbot")
    ADMIN_IDS = [int(x) for x in os.environ.get("ADMIN_IDS", "123456789").split(",")]
    AUTO_DELETE_TIME = int(os.environ.get("AUTO_DELETE_TIME", "300"))
    WEB_SERVER_PORT = int(os.environ.get("PORT", 8000))
    
    OMDB_KEYS = ["8265bd1c", "b9bd48a6", "2f2d1c8e"]
    TMDB_KEYS = ["e547e17d4e91f3e62a571655cd1ccaff"]
    AUTO_UPDATE_INTERVAL = 180
    BACKEND_URL = os.environ.get("BACKEND_URL", f"http://localhost:{WEB_SERVER_PORT}")

if Config.API_ID == 0 or not Config.BOT_TOKEN:
    logger.error("❌ Missing API_ID or BOT_TOKEN")
    exit(1)

# ==================== QUART APP (FIXED) ====================
app = Quart(__name__)

# Configure app BEFORE any routes
app.config.update(
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,
    JSON_SORT_KEYS=False,
    TEMPLATES_AUTO_RELOAD=False
)

@app.before_serving
async def startup():
    logger.info("🌐 Web server starting...")

@app.after_serving
async def shutdown():
    logger.info("🌐 Web server stopped")

@app.after_request
async def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Cache-Control'] = 'no-cache'
    return response

# ==================== PYROGRAM CLIENTS ====================
bot = Client(
    "sk4film_bot",
    api_id=Config.API_ID,
    api_hash=Config.API_HASH,
    bot_token=Config.BOT_TOKEN,
    workdir="/tmp",
    sleep_threshold=60
)

User = Client(
    "sk4film_user",
    api_id=Config.API_ID,
    api_hash=Config.API_HASH,
    session_string=Config.USER_SESSION_STRING,
    workdir="/tmp",
    sleep_threshold=60
) if Config.USER_SESSION_STRING else None

# ==================== GLOBAL STATE ====================
bot_started = False
user_started = False
shutdown_event = asyncio.Event()
movie_db = {
    'home_movies': [],
    'last_update': None,
    'poster_cache': {},
    'updating': False,
    'stats': {'omdb': 0, 'tmdb': 0, 'custom': 0}
}
file_registry = {}
users_collection = None
stats_collection = None

# MongoDB
if MONGODB_AVAILABLE and Config.MONGODB_URI:
    try:
        mongo_client = AsyncIOMotorClient(Config.MONGODB_URI, serverSelectionTimeoutMS=5000)
        db = mongo_client[Config.DATABASE_NAME]
        users_collection = db.users
        stats_collection = db.stats
        logger.info("✅ MongoDB connected")
    except Exception as e:
        logger.error(f"MongoDB error: {e}")

# ==================== HELPER FUNCTIONS ====================
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
        clean = re.sub(r'[^\w\s\(\)\-\.]', ' ', text[:100])
        lines = [l.strip() for l in clean.split('\n') if l.strip()]
        if not lines:
            return None
        
        first_line = lines[0]
        patterns = [
            r'^([A-Z][a-zA-Z\s]{3,40})',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, first_line)
            if match:
                title = match.group(1).strip()
                if 4 <= len(title) <= 50:
                    return title
        
        if 4 <= len(first_line) <= 50:
            return first_line
        
        return None
    except:
        return None

def extract_title_from_file(message):
    try:
        if message.caption:
            title = extract_title_smart(message.caption)
            if title:
                return title
        
        filename = None
        if message.document:
            filename = message.document.file_name
        elif message.video:
            filename = message.video.file_name
        
        if filename:
            name = filename.rsplit('.', 1)[0]
            name = re.sub(r'[\._\-]', ' ', name)
            name = re.sub(r'(720p|1080p|480p|HDRip|WEB|BluRay|x264|x265|HEVC)', '', name, flags=re.IGNORECASE)
            name = ' '.join(name.split()).strip()
            if 4 <= len(name) <= 50:
                return name
        
        return None
    except:
        return None

def format_size(size_bytes):
    if not size_bytes:
        return "Unknown"
    try:
        if size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"
    except:
        return "Unknown"

def detect_quality(filename):
    if not filename:
        return "480p"
    
    filename_lower = filename.lower()
    
    if any(p in filename_lower for p in ['1080p', 'fullhd', 'fhd']):
        return "1080p"
    elif any(p in filename_lower for p in ['720p', 'hd']):
        return "720p"
    else:
        return "480p"

def format_original_post(text):
    if not text:
        return ""
    formatted = html.escape(text)
    formatted = re.sub(r'(https?://[^\s]+)', r'<a href="\1" target="_blank" style="color: #00ccff;">\1</a>', formatted)
    formatted = formatted.replace('\n', '<br>')
    return formatted

def get_channel_name(channel_id):
    names = {
        -1001891090100: "SK4FiLM Main",
        -1002024811395: "SK4FiLM Updates",
        -1001768249569: "SK4FiLM Files"
    }
    return names.get(channel_id, "Channel")

def is_new_post(post_date):
    try:
        if isinstance(post_date, str):
            post_date = datetime.fromisoformat(post_date.replace('Z', '+00:00'))
        delta = datetime.now() - post_date.replace(tzinfo=None)
        return delta.total_seconds() / 3600 <= 24
    except:
        return False

# ==================== POSTER FUNCTIONS ====================
async def fetch_poster_omdb(title, session):
    for api_key in Config.OMDB_KEYS:
        try:
            url = f"http://www.omdbapi.com/?t={urllib.parse.quote(title)}&apikey={api_key}"
            async with session.get(url, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get('Response') == 'True' and data.get('Poster') != 'N/A':
                        poster_url = data['Poster'].replace('http://', 'https://')
                        movie_db['stats']['omdb'] += 1
                        return {
                            'poster_url': poster_url,
                            'title': data.get('Title', title),
                            'year': data.get('Year', ''),
                            'rating': data.get('imdbRating', ''),
                            'source': 'OMDB',
                            'success': True
                        }
        except:
            continue
    return None

async def fetch_poster_tmdb(title, session):
    for api_key in Config.TMDB_KEYS:
        try:
            url = "https://api.themoviedb.org/3/search/movie"
            params = {'api_key': api_key, 'query': title}
            
            async with session.get(url, params=params, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get('results') and len(data['results']) > 0:
                        movie = data['results'][0]
                        poster_path = movie.get('poster_path')
                        
                        if poster_path:
                            movie_db['stats']['tmdb'] += 1
                            return {
                                'poster_url': f"https://image.tmdb.org/t/p/w500{poster_path}",
                                'title': movie.get('title', title),
                                'year': movie.get('release_date', '')[:4] if movie.get('release_date') else '',
                                'rating': f"{movie.get('vote_average', 0):.1f}",
                                'source': 'TMDB',
                                'success': True
                            }
        except:
            continue
    return None

async def get_poster(title, session=None):
    cache_key = title.lower().strip()
    if cache_key in movie_db['poster_cache']:
        cached, cache_time = movie_db['poster_cache'][cache_key]
        if (datetime.now() - cache_time).seconds < 600:
            return cached
    
    should_close = False
    if not session:
        session = aiohttp.ClientSession()
        should_close = True
    
    try:
        result = await fetch_poster_omdb(title, session)
        if result:
            movie_db['poster_cache'][cache_key] = (result, datetime.now())
            return result
        
        result = await fetch_poster_tmdb(title, session)
        if result:
            movie_db['poster_cache'][cache_key] = (result, datetime.now())
            return result
        
        movie_db['stats']['custom'] += 1
        result = {
            'poster_url': f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(title)}",
            'title': title,
            'source': 'CUSTOM',
            'success': True
        }
        movie_db['poster_cache'][cache_key] = (result, datetime.now())
        return result
    
    except Exception as e:
        logger.error(f"Poster error: {e}")
        return {
            'poster_url': f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(title)}",
            'title': title,
            'source': 'CUSTOM',
            'success': True
        }
    finally:
        if should_close:
            await session.close()

# ==================== SEARCH FUNCTION ====================
async def search_movies(query, limit=12, page=1):
    if not User:
        return {'results': [], 'pagination': {'current_page': 1, 'total_pages': 1, 'total_results': 0, 'per_page': limit, 'has_next': False, 'has_previous': False}}
    
    offset = (page - 1) * limit
    results = []
    seen = {}
    
    logger.info(f"🔍 Search: '{query}' | Page: {page}")
    
    # Search text channels
    for channel_id in Config.TEXT_CHANNEL_IDS:
        try:
            count = 0
            async for msg in User.search_messages(channel_id, query, limit=20):
                if count >= 20:
                    break
                
                if msg.text:
                    title = extract_title_smart(msg.text)
                    if title and title.lower() not in seen:
                        results.append({
                            'title': title,
                            'type': 'text_post',
                            'content': format_original_post(msg.text),
                            'date': msg.date.isoformat() if msg.date else datetime.now().isoformat(),
                            'channel': get_channel_name(channel_id),
                            'channel_id': channel_id,
                            'message_id': msg.id,
                            'is_new': is_new_post(msg.date) if msg.date else False,
                            'has_file': False,
                            'quality_options': {}
                        })
                        seen[title.lower()] = len(results) - 1
                        count += 1
        except Exception as e:
            logger.error(f"Text channel {channel_id} error: {e}")
    
    # Search file channel
    try:
        count = 0
        async for msg in User.search_messages(Config.FILE_CHANNEL_ID, query, limit=30):
            if count >= 30:
                break
            
            if msg.document or msg.video:
                title = extract_title_from_file(msg)
                if title:
                    file_id = msg.document.file_id if msg.document else msg.video.file_id
                    file_size = msg.document.file_size if msg.document else (msg.video.file_size if msg.video else 0)
                    file_name = msg.document.file_name if msg.document else (msg.video.file_name if msg.video else 'video.mp4')
                    
                    quality = detect_quality(file_name)
                    unique_id = hashlib.md5(f"{file_id}{time.time()}".encode()).hexdigest()
                    
                    file_registry[unique_id] = {
                        'file_id': file_id,
                        'channel_id': Config.FILE_CHANNEL_ID,
                        'message_id': msg.id,
                        'quality': quality,
                        'file_size': file_size,
                        'file_name': file_name,
                        'title': title,
                        'created_at': datetime.now()
                    }
                    
                    title_key = title.lower()
                    if title_key in seen:
                        idx = seen[title_key]
                        results[idx]['has_file'] = True
                        results[idx]['type'] = 'with_file'
                        results[idx]['quality_options'][quality] = {
                            'file_id': unique_id,
                            'file_size': file_size,
                            'file_name': file_name
                        }
                    else:
                        results.append({
                            'title': title,
                            'type': 'with_file',
                            'content': format_original_post(msg.caption or title),
                            'date': msg.date.isoformat() if msg.date else datetime.now().isoformat(),
                            'channel': get_channel_name(Config.FILE_CHANNEL_ID),
                            'channel_id': Config.FILE_CHANNEL_ID,
                            'message_id': msg.id,
                            'is_new': is_new_post(msg.date) if msg.date else False,
                            'has_file': True,
                            'quality_options': {
                                quality: {
                                    'file_id': unique_id,
                                    'file_size': file_size,
                                    'file_name': file_name
                                }
                            }
                        })
                        seen[title_key] = len(results) - 1
                    count += 1
    except Exception as e:
        logger.error(f"File channel error: {e}")
    
    results.sort(key=lambda x: (x['has_file'], x['date']), reverse=True)
    
    total = len(results)
    paginated = results[offset:offset + limit]
    
    logger.info(f"✅ Found {total} | Showing {len(paginated)}")
    
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

async def get_homepage_movies():
    if not User:
        return []
    
    movies = []
    seen = set()
    
    for channel_id in Config.TEXT_CHANNEL_IDS:
        try:
            count = 0
            async for msg in User.get_chat_history(channel_id, limit=20):
                if count >= 20:
                    break
                
                if msg.text and msg.date:
                    title = extract_title_smart(msg.text)
                    if title and title.lower() not in seen:
                        movies.append({
                            'title': title,
                            'date': msg.date.isoformat(),
                            'is_new': is_new_post(msg.date)
                        })
                        seen.add(title.lower())
                        count += 1
        except Exception as e:
            logger.error(f"Homepage channel error: {e}")
    
    movies.sort(key=lambda x: x['date'], reverse=True)
    movies = movies[:24]
    
    # Fetch posters
    async with aiohttp.ClientSession() as session:
        for i in range(0, len(movies), 5):
            batch = movies[i:i+5]
            poster_tasks = [get_poster(m['title'], session) for m in batch]
            posters = await asyncio.gather(*poster_tasks, return_exceptions=True)
            
            for movie, poster in zip(batch, posters):
                if isinstance(poster, dict) and poster.get('success'):
                    movie.update({
                        'poster_url': poster['poster_url'],
                        'poster_title': poster.get('title', movie['title']),
                        'poster_year': poster.get('year', ''),
                        'poster_rating': poster.get('rating', ''),
                        'poster_source': poster['source'],
                        'has_poster': True
                    })
                else:
                    movie.update({
                        'poster_url': f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(movie['title'])}",
                        'has_poster': True
                    })
            
            await asyncio.sleep(0.2)
    
    logger.info(f"✅ Homepage: {len(movies)} movies")
    return movies

# ==================== BOT HANDLERS ====================
@bot.on_message(filters.command("start") & filters.private)
async def start_handler(client, message):
    user_id = message.from_user.id
    
    if users_collection:
        try:
            await users_collection.update_one(
                {'user_id': user_id},
                {'$set': {'first_name': message.from_user.first_name, 'last_seen': datetime.now()}, '$inc': {'start_count': 1}},
                upsert=True
            )
        except:
            pass
    
    if len(message.command) > 1:
        unique_id = message.command[1]
        
        if not await check_force_sub(user_id):
            try:
                channel = await bot.get_chat(Config.FORCE_SUB_CHANNEL)
                link = f"https://t.me/{channel.username}" if channel.username else f"https://t.me/c/{str(Config.FORCE_SUB_CHANNEL)[4:]}/1"
            except:
                link = f"https://t.me/c/{str(Config.FORCE_SUB_CHANNEL)[4:]}/1"
            
            await message.reply_text(
                "⚠️ **Join Channel First**\n\nClick join, then download again.",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("📢 Join", url=link)]])
            )
            return
        
        file_info = file_registry.get(unique_id)
        if not file_info:
            await message.reply_text(
                "❌ **File Not Found**\n\nLink expired. Search again.",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🌐 Search", url=Config.WEBSITE_URL)]])
            )
            return
        
        try:
            processing = await message.reply_text(f"⏳ Sending...\n\n📁 {file_info['file_name']}\n📊 {file_info['quality']}")
            
            if User:
                file_msg = await User.get_messages(file_info['channel_id'], file_info['message_id'])
                sent = await file_msg.copy(user_id)
                
                await processing.delete()
                
                success = await message.reply_text(
                    f"✅ **File Sent!**\n\n🎬 {file_info['title']}\n📊 {file_info['quality']}\n\n⚠️ Auto-delete in {Config.AUTO_DELETE_TIME//60} min"
                )
                
                logger.info(f"📥 File: {file_info['title']} → {user_id}")
                
                if Config.AUTO_DELETE_TIME > 0:
                    await asyncio.sleep(Config.AUTO_DELETE_TIME)
                    try:
                        await sent.delete()
                        await success.edit_text("🗑️ Auto-deleted")
                    except:
                        pass
            else:
                await processing.edit_text("❌ Unavailable")
        
        except Exception as e:
            logger.error(f"File error: {e}")
            await message.reply_text("❌ Error. Try again.")
        
        return
    
    await message.reply_text(
        f"🎬 **SK4FiLM Bot**\n\n📌 File delivery only\n\n**Usage:**\n1. Visit website\n2. Search\n3. Select quality\n4. Get file\n\n⚡ Automated",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🌐 Website", url=Config.WEBSITE_URL)]])
    )

@bot.on_message(filters.text & filters.private & ~filters.command(['start', 'stats']))
async def text_handler(client, message):
    await message.reply_text(
        f"👋 Hi!\n\n🤖 Use website to search.\n\nBot delivers files only.",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🌐 Website", url=Config.WEBSITE_URL)]])
    )

@bot.on_message(filters.command("stats") & filters.user(Config.ADMIN_IDS))
async def stats_handler(client, message):
    try:
        total_users = await users_collection.count_documents({}) if users_collection else 0
        
        await message.reply_text(
            f"📊 **Stats**\n\n"
            f"👥 Users: `{total_users}`\n"
            f"🎬 Movies: `{len(movie_db['home_movies'])}`\n"
            f"📁 Files: `{len(file_registry)}`\n"
            f"🖼️ Posters: OMDB:{movie_db['stats']['omdb']} TMDB:{movie_db['stats']['tmdb']} Custom:{movie_db['stats']['custom']}\n"
            f"🤖 Bot: `{'✅' if bot_started else '❌'}`\n"
            f"👤 User: `{'✅' if user_started else '❌'}`"
        )
    except Exception as e:
        await message.reply_text(f"❌ Error: {e}")

# ==================== API ROUTES ====================
@app.route('/')
async def root():
    return jsonify({
        'status': 'healthy',
        'service': 'SK4FiLM',
        'bot': f'@{Config.BOT_USERNAME}',
        'website': Config.WEBSITE_URL
    })

@app.route('/api/movies')
async def api_movies():
    try:
        if not user_started:
            return jsonify({'status': 'error', 'message': 'Starting'}), 503
        
        if not movie_db['home_movies'] or not movie_db['last_update'] or (datetime.now() - movie_db['last_update']).seconds > 300:
            if not movie_db['updating']:
                movie_db['updating'] = True
                movie_db['home_movies'] = await get_homepage_movies()
                movie_db['last_update'] = datetime.now()
                movie_db['updating'] = False
        
        return jsonify({
            'status': 'success',
            'movies': movie_db['home_movies'],
            'total': len(movie_db['home_movies']),
            'bot_username': Config.BOT_USERNAME
        })
    except Exception as e:
        logger.error(f"API movies error: {e}")
        movie_db['updating'] = False
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/search')
async def api_search():
    try:
        query = request.args.get('query', '').strip()
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 12))
        
        if not query:
            return jsonify({'status': 'error', 'message': 'Query required'}), 400
        
        if not user_started:
            return jsonify({'status': 'error', 'message': 'Starting'}), 503
        
        result = await search_movies(query, limit, page)
        
        return jsonify({
            'status': 'success',
            'query': query,
            'results': result['results'],
            'pagination': result['pagination'],
            'bot_username': Config.BOT_USERNAME
        })
    except Exception as e:
        logger.error(f"API search error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/poster')
async def api_poster():
    title = request.args.get('title', 'Movie')
    display = title[:18] + "..." if len(title) > 18 else title
    
    colors = [
        ('#667eea', '#764ba2'),
        ('#f093fb', '#f5576c'),
        ('#4facfe', '#00f2fe'),
        ('#43e97b', '#38f9d7'),
        ('#fa709a', '#fee140')
    ]
    color_set = colors[hash(title) % len(colors)]
    
    svg = f'''<svg width="300" height="450" xmlns="http://www.w3.org/2000/svg">
        <defs><linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" style="stop-color:{color_set[0]}"/>
        <stop offset="100%" style="stop-color:{color_set[1]}"/>
        </linearGradient></defs>
        <rect width="100%" height="100%" fill="url(#bg)" rx="20"/>
        <circle cx="150" cy="180" r="50" fill="rgba(255,255,255,0.2)"/>
        <text x="50%" y="200" text-anchor="middle" fill="#fff" font-size="50">🎬</text>
        <text x="50%" y="270" text-anchor="middle" fill="#fff" font-size="18" font-weight="bold">{html.escape(display)}</text>
        <rect x="50" y="380" width="200" height="40" rx="20" fill="rgba(0,0,0,0.3)"/>
        <text x="50%" y="405" text-anchor="middle" fill="#fff" font-size="18" font-weight="700">SK4FiLM</text>
    </svg>'''
    
    return Response(svg, mimetype='image/svg+xml', headers={'Cache-Control': 'public, max-age=3600'})

# ==================== STARTUP ====================
async def start_bot_client():
    global bot_started
    try:
        logger.info("🤖 Starting bot...")
        await bot.start()
        me = await bot.get_me()
        logger.info(f"✅ Bot: @{me.username}")
        bot_started = True
    except Exception as e:
        logger.error(f"❌ Bot error: {e}")

async def start_user_client():
    global user_started
    if not User:
        logger.error("❌ User session missing")
        return
    
    try:
        logger.info("👤 Starting user...")
        await User.start()
        me = await User.get_me()
        logger.info(f"✅ User: {me.first_name}")
        user_started = True
        
        movie_db['home_movies'] = await get_homepage_movies()
        movie_db['last_update'] = datetime.now()
    except Exception as e:
        logger.error(f"❌ User error: {e}")

async def start_web_server():
    try:
        config = HyperConfig()
        config.bind = [f"0.0.0.0:{Config.WEB_SERVER_PORT}"]
        config.loglevel = "warning"
        config.graceful_timeout = 2
        logger.info(f"🌐 Web server on port {Config.WEB_SERVER_PORT}")
        await serve(app, config, shutdown_trigger=shutdown_event.wait)
    except Exception as e:
        logger.error(f"❌ Web error: {e}")

async def cleanup():
    """Cleanup on shutdown"""
    logger.info("🔄 Cleaning up...")
    
    if bot_started:
        try:
            await bot.stop()
            logger.info("✅ Bot stopped")
        except:
            pass
    
    if user_started and User:
        try:
            await User.stop()
            logger.info("✅ User stopped")
        except:
            pass
    
    shutdown_event.set()
    logger.info("✅ Cleanup complete")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"\n⚠️ Received signal {signum}")
    asyncio.create_task(cleanup())

async def main():
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("=" * 60)
    logger.info("🚀 SK4FiLM Starting")
    logger.info("=" * 60)
    logger.info(f"🤖 Bot: @{Config.BOT_USERNAME}")
    logger.info(f"🌐 Website: {Config.WEBSITE_URL}")
    logger.info("=" * 60)
    
    try:
        await asyncio.gather(
            start_bot_client(),
            start_user_client(),
            start_web_server()
        )
    except KeyboardInterrupt:
        logger.info("\n⚠️ Keyboard interrupt")
    except Exception as e:
        logger.error(f"❌ Fatal: {e}")
    finally:
        await cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n⚠️ Stopped")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
