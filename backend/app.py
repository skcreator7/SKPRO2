import asyncio
import os
import logging
from datetime import datetime, timedelta
from pyrogram import Client, filters, errors
from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
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
import uuid

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MongoDB (optional)
try:
    from motor.motor_asyncio import AsyncIOMotorClient
    import ssl
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    logger.warning("⚠️ MongoDB not available")

# ==================== CONFIG ====================
class Config:
    # Telegram
    API_ID = int(os.environ.get("API_ID", "0"))
    API_HASH = os.environ.get("API_HASH", "")
    USER_SESSION_STRING = os.environ.get("USER_SESSION_STRING", "")
    BOT_TOKEN = os.environ.get("BOT_TOKEN", ""))
    
    # Channels
    TEXT_CHANNEL_IDS = [int(x) for x in os.environ.get("TEXT_CHANNEL_IDS", "-1001891090100,-1002024811395").split(",")]
    FILE_CHANNEL_480P = int(os.environ.get("FILE_CHANNEL_480P", "-1001768249569"))
    FILE_CHANNEL_720P = int(os.environ.get("FILE_CHANNEL_720P", "-1001768249569"))
    FILE_CHANNEL_1080P = int(os.environ.get("FILE_CHANNEL_1080P", "-1001768249569"))
    FILE_CHANNELS = {"480p": FILE_CHANNEL_480P, "720p": FILE_CHANNEL_720P, "1080p": FILE_CHANNEL_1080P}
    
    # MongoDB
    MONGODB_URI = os.environ.get("MONGODB_URI", "")
    DATABASE_NAME = os.environ.get("DATABASE_NAME", "sk4film_bot")
    
    # Settings
    FORCE_SUB_CHANNEL = int(os.environ.get("FORCE_SUB_CHANNEL", "-1002555323872"))
    SHORTENER_API = os.environ.get("SHORTENER_API", "")
    SHORTENER_DOMAIN = os.environ.get("SHORTENER_DOMAIN", "https://gplinks.in/api")
    WEBSITE_URL = os.environ.get("WEBSITE_URL", "https://sk4film.vercel.app")
    BOT_USERNAME = os.environ.get("BOT_USERNAME", "sk4filmbot")
    ADMIN_IDS = [int(x) for x in os.environ.get("ADMIN_IDS", "5928972764").split(",")]
    AUTO_DELETE_TIME = int(os.environ.get("AUTO_DELETE_TIME", "300"))
    WEB_SERVER_PORT = int(os.environ.get("PORT", 8000))
    
    # APIs
    OMDB_KEYS = ["8265bd1c", "b9bd48a6", "2f2d1c8e"]
    TMDB_KEYS = ["e547e17d4e91f3e62a571655cd1ccaff"]
    AUTO_UPDATE_INTERVAL = 180

# ==================== QUART APP ====================
app = Quart(__name__)
app.config['PROVIDE_AUTOMATIC_OPTIONS'] = True
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['JSON_SORT_KEYS'] = False

@app.after_request
async def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

# ==================== PYROGRAM CLIENTS ====================
bot = Client("sk4film_bot", api_id=Config.API_ID, api_hash=Config.API_HASH, bot_token=Config.BOT_TOKEN, workdir="/tmp")
User = Client("sk4film_user", api_id=Config.API_ID, api_hash=Config.API_HASH, session_string=Config.USER_SESSION_STRING, workdir="/tmp")

# ==================== GLOBAL VARIABLES ====================
bot_started = False
user_started = False
movie_db = {
    'home_movies': [],
    'last_update': None,
    'poster_cache': {},
    'updating': False,
    'stats': {'omdb': 0, 'tmdb': 0, 'custom': 0}
}
files_collection = None
users_collection = None
stats_collection = None

# File tracking for quality selection
file_registry = {}  # {unique_id: {channel_id, message_id, quality, file_id}}

# MongoDB setup
if MONGODB_AVAILABLE and Config.MONGODB_URI:
    try:
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        mongo_client = AsyncIOMotorClient(
            Config.MONGODB_URI,
            serverSelectionTimeoutMS=30000,
            connectTimeoutMS=30000,
            socketTimeoutMS=30000,
            tls=True,
            tlsAllowInvalidCertificates=True,
            ssl_cert_reqs=ssl.CERT_NONE
        )
        db = mongo_client[Config.DATABASE_NAME]
        files_collection = db.files
        users_collection = db.users
        stats_collection = db.stats
        logger.info("✅ MongoDB initialized")
    except Exception as e:
        logger.error(f"MongoDB error: {e}")

# ==================== HELPER FUNCTIONS ====================
async def check_force_sub(user_id):
    """Check if user is subscribed"""
    try:
        member = await bot.get_chat_member(Config.FORCE_SUB_CHANNEL, user_id)
        return member.status in ["member", "administrator", "creator"]
    except:
        return True

async def shorten_link(url):
    """Shorten link using GPLinks"""
    if not Config.SHORTENER_API:
        return url
    try:
        api_url = f"{Config.SHORTENER_DOMAIN}?api={Config.SHORTENER_API}&url={url}"
        async with aiohttp.ClientSession() as session:
            async with session.get(api_url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("status") == "success":
                        return data.get("shortenedUrl", url)
        return url
    except:
        return url

def extract_title_smart(text):
    """Smart title extraction"""
    if not text or len(text) < 15:
        return None
    try:
        clean_text = re.sub(r'[^\w\s\(\)\-\.\n\u0900-\u097F]', ' ', text)
        first_line = clean_text.split('\n')[0].strip()
        patterns = [
            r'🎬\s*([^-\n]{4,45})(?:\s*-|\n|$)',
            r'^([^(]{4,45})\s*\(\d{4}\)',
            r'^([^-]{4,45})\s*-\s*(?:Hindi|English|Tamil|Telugu|20\d{2})',
            r'^([A-Z][a-z]+(?:\s+[A-Za-z]+){1,4})'
        ]
        for pattern in patterns:
            match = re.search(pattern, first_line, re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                title = re.sub(r'\s+', ' ', title)
                if 4 <= len(title) <= 45:
                    return title
        return None
    except:
        return None

def extract_title_from_file(message):
    """Extract title from file message"""
    try:
        if message.caption:
            title = extract_title_smart(message.caption)
            if title:
                return title
        filename = message.document.file_name if message.document else (message.video.file_name if message.video else None)
        if filename:
            title = filename.rsplit('.', 1)[0]
            title = re.sub(r'[\._\-]', ' ', title)
            title = re.sub(r'(720p|1080p|480p|HDRip|WEB-DL|BluRay|x264|x265)', '', title, flags=re.IGNORECASE)
            title = ' '.join(title.split()).strip()[:50]
            if len(title) >= 4:
                return title
        return None
    except:
        return None

def format_size(size_bytes):
    """Format file size"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"

async def get_poster(title, session):
    """Get poster from multiple sources"""
    cache_key = title.lower().strip()
    if cache_key in movie_db['poster_cache']:
        cached, cache_time = movie_db['poster_cache'][cache_key]
        if datetime.now() - cache_time < timedelta(minutes=12):
            return cached
    
    try:
        # OMDB
        for api_key in Config.OMDB_KEYS:
            try:
                url = f"http://www.omdbapi.com/?t={urllib.parse.quote(title)}&apikey={api_key}"
                async with session.get(url, timeout=7) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get('Response') == 'True' and data.get('Poster') != 'N/A':
                            result = {
                                'poster_url': data['Poster'],
                                'title': data.get('Title', title),
                                'year': data.get('Year', ''),
                                'rating': data.get('imdbRating', ''),
                                'source': 'OMDB',
                                'success': True
                            }
                            movie_db['poster_cache'][cache_key] = (result, datetime.now())
                            movie_db['stats']['omdb'] += 1
                            return result
            except:
                continue
        
        # TMDB
        for tmdb_key in Config.TMDB_KEYS:
            try:
                url = "https://api.themoviedb.org/3/search/movie"
                async with session.get(url, params={'api_key': tmdb_key, 'query': title}, timeout=8) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get('results'):
                            movie = data['results'][0]
                            poster_path = movie.get('poster_path')
                            if poster_path:
                                result = {
                                    'poster_url': f"https://image.tmdb.org/t/p/w780{poster_path}",
                                    'title': movie.get('title', title),
                                    'year': movie.get('release_date', '')[:4] if movie.get('release_date') else '',
                                    'rating': f"{movie.get('vote_average', 0):.1f}",
                                    'source': 'TMDB',
                                    'success': True
                                }
                                movie_db['poster_cache'][cache_key] = (result, datetime.now())
                                movie_db['stats']['tmdb'] += 1
                                return result
            except:
                continue
        
        # Custom fallback
        result = {
            'poster_url': f"/api/poster?title={urllib.parse.quote(title)}",
            'title': title,
            'source': 'CUSTOM',
            'success': True
        }
        movie_db['poster_cache'][cache_key] = (result, datetime.now())
        movie_db['stats']['custom'] += 1
        return result
    except:
        return {
            'poster_url': f"/api/poster?title={urllib.parse.quote(title)}",
            'title': title,
            'source': 'CUSTOM',
            'success': True
        }

def is_new_post(post_date):
    """Check if post is within 24 hours"""
    try:
        if isinstance(post_date, str):
            post_date = datetime.fromisoformat(post_date.replace('Z', '+00:00'))
        return (datetime.now() - post_date.replace(tzinfo=None)).total_seconds() / 3600 <= 24
    except:
        return False

def format_original_post(text):
    """Format post content"""
    if not text:
        return ""
    formatted = html.escape(text)
    formatted = re.sub(r'(https?://[^\s]+)', r'<a href="\1" target="_blank" style="color: #00ccff;">\1</a>', formatted)
    formatted = formatted.replace('\n', '<br>')
    return formatted

# ==================== SEARCH FUNCTIONS ====================
async def search_movies(query, limit=12, page=1):
    """Search both text and file channels"""
    offset = (page - 1) * limit
    results = []
    seen = {}
    
    # Search text channels
    for channel_id in Config.TEXT_CHANNEL_IDS:
        try:
            async for msg in User.search_messages(channel_id, query, limit=30):
                if msg.text:
                    title = extract_title_smart(msg.text)
                    if title and title.lower() not in seen:
                        results.append({
                            'title': title,
                            'type': 'text_post',
                            'content': format_original_post(msg.text),
                            'date': msg.date.isoformat() if msg.date else datetime.now().isoformat(),
                            'channel_id': channel_id,
                            'message_id': msg.id,
                            'is_new': is_new_post(msg.date) if msg.date else False,
                            'has_file': False,
                            'quality_options': {}
                        })
                        seen[title.lower()] = len(results) - 1
        except Exception as e:
            logger.warning(f"Text channel {channel_id} search error: {e}")
    
    # Search file channels
    for quality, channel_id in Config.FILE_CHANNELS.items():
        try:
            async for msg in User.search_messages(channel_id, query, limit=20):
                if msg.document or msg.video:
                    title = extract_title_from_file(msg)
                    if title:
                        file_id = msg.document.file_id if msg.document else msg.video.file_id
                        file_size = msg.document.file_size if msg.document else msg.video.file_size
                        file_name = msg.document.file_name if msg.document else (msg.video.file_name or 'video.mp4')
                        
                        # Generate unique ID
                        unique_id = hashlib.md5(f"{file_id}{time.time()}".encode()).hexdigest()
                        
                        # Store in registry
                        file_registry[unique_id] = {
                            'file_id': file_id,
                            'channel_id': channel_id,
                            'message_id': msg.id,
                            'quality': quality,
                            'file_size': file_size,
                            'file_name': file_name
                        }
                        
                        title_key = title.lower()
                        if title_key in seen:
                            idx = seen[title_key]
                            results[idx]['has_file'] = True
                            results[idx]['type'] = 'with_file'
                            results[idx]['quality_options'][quality] = {
                                'unique_id': unique_id,
                                'file_size': file_size,
                                'file_size_formatted': format_size(file_size),
                                'file_name': file_name
                            }
                        else:
                            results.append({
                                'title': title,
                                'type': 'with_file',
                                'content': format_original_post(msg.caption or title),
                                'date': msg.date.isoformat() if msg.date else datetime.now().isoformat(),
                                'channel_id': channel_id,
                                'message_id': msg.id,
                                'is_new': is_new_post(msg.date) if msg.date else False,
                                'has_file': True,
                                'quality_options': {
                                    quality: {
                                        'unique_id': unique_id,
                                        'file_size': file_size,
                                        'file_size_formatted': format_size(file_size),
                                        'file_name': file_name
                                    }
                                }
                            })
                            seen[title_key] = len(results) - 1
        except Exception as e:
            logger.warning(f"File channel {quality} search error: {e}")
    
    # Sort: files first, then by date
    results.sort(key=lambda x: (x['has_file'], x['date']), reverse=True)
    
    total = len(results)
    paginated = results[offset:offset+limit]
    
    return {
        'results': paginated,
        'pagination': {
            'current_page': page,
            'total_pages': math.ceil(total/limit) if total > 0 else 1,
            'total_results': total,
            'per_page': limit,
            'has_next': page < math.ceil(total/limit),
            'has_previous': page > 1
        }
    }

async def background_update():
    """Background update for home page"""
    if movie_db['updating']:
        return
    
    try:
        movie_db['updating'] = True
        logger.info("🔄 Background update starting...")
        
        all_posts = []
        for channel_id in Config.TEXT_CHANNEL_IDS:
            try:
                async for msg in User.get_chat_history(channel_id, limit=30):
                    if msg.text and msg.date:
                        title = extract_title_smart(msg.text)
                        if title:
                            all_posts.append({
                                'title': title,
                                'date': msg.date,
                                'is_new': is_new_post(msg.date)
                            })
            except Exception as e:
                logger.warning(f"Channel {channel_id} error: {e}")
        
        all_posts.sort(key=lambda x: x['date'], reverse=True)
        
        unique = []
        seen = set()
        for post in all_posts:
            if post['title'].lower() not in seen:
                seen.add(post['title'].lower())
                post['date'] = post['date'].isoformat()
                unique.append(post)
        
        # Fetch posters
        async with aiohttp.ClientSession() as session:
            for i in range(0, min(len(unique), 24), 4):
                batch = unique[i:i+4]
                posters = await asyncio.gather(
                    *[get_poster(m['title'], session) for m in batch],
                    return_exceptions=True
                )
                for movie, poster in zip(batch, posters):
                    if isinstance(poster, dict) and poster.get('success'):
                        movie.update({
                            'poster_url': poster['poster_url'],
                            'poster_title': poster['title'],
                            'poster_year': poster.get('year', ''),
                            'poster_rating': poster.get('rating', ''),
                            'poster_source': poster['source'],
                            'has_poster': True
                        })
                await asyncio.sleep(0.3)
        
        movie_db['home_movies'] = unique[:24]
        movie_db['last_update'] = datetime.now()
        
        logger.info(f"✅ Updated: {len(movie_db['home_movies'])} movies")
    except Exception as e:
        logger.error(f"Background update error: {e}")
    finally:
        movie_db['updating'] = False

# ==================== BOT HANDLERS ====================
@bot.on_message(filters.command("start") & filters.private)
async def start_cmd(client, message):
    user_id = message.from_user.id
    
    # Check force subscribe
    if not await check_force_sub(user_id):
        try:
            channel = await bot.get_chat(Config.FORCE_SUB_CHANNEL)
            channel_link = f"https://t.me/{channel.username}" if channel.username else f"https://t.me/c/{str(Config.FORCE_SUB_CHANNEL)[4:]}/1"
        except:
            channel_link = f"https://t.me/c/{str(Config.FORCE_SUB_CHANNEL)[4:]}/1"
        
        await message.reply_text(
            "⚠️ **Access Restricted**\n\n"
            "Please join our channel first to use this bot.\n\n"
            "After joining, click /start again.",
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("📢 Join Channel", url=channel_link)
            ]])
        )
        return
    
    # Track user
    if users_collection:
        try:
            await users_collection.update_one(
                {'user_id': user_id},
                {'$set': {
                    'first_name': message.from_user.first_name,
                    'username': message.from_user.username,
                    'last_seen': datetime.now()
                }, '$inc': {'start_count': 1}},
                upsert=True
            )
        except:
            pass
    
    await message.reply_text(
        f"👋 **Welcome to SK4FiLM Bot!**\n\n"
        f"🎬 Search and download movies with quality options\n\n"
        f"**How to use:**\n"
        f"• Send any movie name to search\n"
        f"• Choose quality: 480p/720p/1080p\n"
        f"• Get direct download link\n\n"
        f"🌐 **Website:** {Config.WEBSITE_URL}\n\n"
        f"⚡ Powered by SK4FiLM",
        reply_markup=InlineKeyboardMarkup([[
            InlineKeyboardButton("🌐 Visit Website", url=Config.WEBSITE_URL),
            InlineKeyboardButton("📢 Channel", url=f"https://t.me/c/{str(Config.FORCE_SUB_CHANNEL)[4:]}/1")
        ]])
    )

@bot.on_message(filters.text & filters.private & ~filters.command(['start', 'help', 'about']))
async def search_cmd(client, message):
    user_id = message.from_user.id
    
    # Check force subscribe
    if not await check_force_sub(user_id):
        await message.reply_text("❌ Please join our channel first. Click /start")
        return
    
    query = message.text.strip()
    search_url = f"{Config.WEBSITE_URL}/search.html?query={urllib.parse.quote(query)}"
    shortened = await shorten_link(search_url)
    
    # Track search
    if stats_collection:
        try:
            await stats_collection.insert_one({
                'user_id': user_id,
                'query': query,
                'timestamp': datetime.now(),
                'type': 'search'
            })
        except:
            pass
    
    await message.reply_text(
        f"🔍 **Searching for:** {query}\n\n"
        f"👉 Click below to view all results with quality options\n\n"
        f"📱 Direct link: {shortened}",
        reply_markup=InlineKeyboardMarkup([[
            InlineKeyboardButton("📱 View Results", url=shortened)
        ]])
    )

@bot.on_callback_query(filters.regex(r"^file_"))
async def handle_file_request(client, callback_query: CallbackQuery):
    """Handle file download requests"""
    try:
        unique_id = callback_query.data.split("_")[1]
        user_id = callback_query.from_user.id
        
        # Check force subscribe
        if not await check_force_sub(user_id):
            await callback_query.answer("❌ Please join our channel first!", show_alert=True)
            return
        
        # Get file info
        file_info = file_registry.get(unique_id)
        if not file_info:
            await callback_query.answer("❌ File not found or expired. Please search again.", show_alert=True)
            return
        
        # Get the file message
        try:
            file_msg = await User.get_messages(file_info['channel_id'], file_info['message_id'])
            
            # Forward file to user
            sent_msg = await file_msg.copy(user_id)
            
            # Track download
            if stats_collection:
                try:
                    await stats_collection.insert_one({
                        'user_id': user_id,
                        'file_id': unique_id,
                        'quality': file_info['quality'],
                        'timestamp': datetime.now(),
                        'type': 'download'
                    })
                except:
                    pass
            
            await callback_query.answer(f"✅ File sent! Quality: {file_info['quality']}", show_alert=False)
            
            # Auto-delete after configured time
            if Config.AUTO_DELETE_TIME > 0:
                await asyncio.sleep(Config.AUTO_DELETE_TIME)
                try:
                    await sent_msg.delete()
                except:
                    pass
        
        except Exception as e:
            logger.error(f"File forward error: {e}")
            await callback_query.answer("❌ Error sending file. Please try again.", show_alert=True)
    
    except Exception as e:
        logger.error(f"Callback error: {e}")
        await callback_query.answer("❌ An error occurred", show_alert=True)

@bot.on_message(filters.command("stats") & filters.user(Config.ADMIN_IDS))
async def stats_cmd(client, message):
    """Admin stats command"""
    try:
        total_users = await users_collection.count_documents({}) if users_collection else 0
        total_searches = await stats_collection.count_documents({'type': 'search'}) if stats_collection else 0
        total_downloads = await stats_collection.count_documents({'type': 'download'}) if stats_collection else 0
        
        stats_text = (
            f"📊 **Bot Statistics**\n\n"
            f"👥 Total Users: `{total_users}`\n"
            f"🔍 Total Searches: `{total_searches}`\n"
            f"⬇️ Total Downloads: `{total_downloads}`\n"
            f"🎬 Home Movies: `{len(movie_db['home_movies'])}`\n"
            f"🖼️ Cached Posters: `{len(movie_db['poster_cache'])}`\n"
            f"📁 File Registry: `{len(file_registry)}`\n\n"
            f"**Poster Stats:**\n"
            f"• OMDB: `{movie_db['stats']['omdb']}`\n"
            f"• TMDB: `{movie_db['stats']['tmdb']}`\n"
            f"• Custom: `{movie_db['stats']['custom']}`\n\n"
            f"🕒 Last Update: `{movie_db['last_update'].strftime('%Y-%m-%d %H:%M:%S') if movie_db['last_update'] else 'Never'}`"
        )
        
        await message.reply_text(stats_text)
    except Exception as e:
        await message.reply_text(f"❌ Error: {e}")

# ==================== WEB ROUTES ====================
@app.route('/')
async def home():
    return jsonify({
        'status': 'healthy' if bot_started and user_started else 'starting',
        'service': 'SK4FiLM Complete System',
        'features': ['bot', 'api', 'search', 'quality_selection', 'file_delivery', 'stats'],
        'bot_username': f'@{Config.BOT_USERNAME}',
        'channels': {
            'text': len(Config.TEXT_CHANNEL_IDS),
            'file': len(Config.FILE_CHANNELS)
        },
        'stats': {
            'movies': len(movie_db['home_movies']),
            'cached_posters': len(movie_db['poster_cache']),
            'file_registry': len(file_registry)
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/health')
async def health():
    return jsonify({
        'status': 'healthy',
        'bot_running': bot_started,
        'user_running': user_started,
        'mongodb': files_collection is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/movies')
async def api_movies():
    """Home movies API"""
    if not user_started:
        return jsonify({'status': 'error', 'message': 'Service starting'}), 503
    
    return jsonify({
        'status': 'success',
        'movies': movie_db['home_movies'],
        'total': len(movie_db['home_movies']),
        'last_update': movie_db['last_update'].isoformat() if movie_db['last_update'] else None,
        'bot_username': Config.BOT_USERNAME,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/search')
async def api_search():
    """Search API"""
    query = request.args.get('query', '').strip()
    page = int(request.args.get('page', 1))
    limit = int(request.args.get('limit', 12))
    
    if not query:
        return jsonify({'status': 'error', 'message': 'Query required'}), 400
    
    if not user_started:
        return jsonify({'status': 'error', 'message': 'Service starting'}), 503
    
    try:
        result = await search_movies(query, limit, page)
        return jsonify({
            'status': 'success',
            'query': query,
            'results': result['results'],
            'pagination': result['pagination'],
            'bot_username': Config.BOT_USERNAME,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/file/<unique_id>')
async def api_file_info(unique_id):
    """Get file info for download"""
    file_info = file_registry.get(unique_id)
    if not file_info:
        return jsonify({'status': 'error', 'message': 'File not found'}), 404
    
    return jsonify({
        'status': 'success',
        'file': {
            'unique_id': unique_id,
            'quality': file_info['quality'],
            'file_size': file_info['file_size'],
            'file_size_formatted': format_size(file_info['file_size']),
            'file_name': file_info['file_name'],
            'bot_username': Config.BOT_USERNAME,
            'download_url': f"https://t.me/{Config.BOT_USERNAME}?start=file_{unique_id}"
        }
    })

@app.route('/api/poster')
async def api_poster():
    """Custom poster generator"""
    title = request.args.get('title', 'Movie')
    display_title = title[:20] + "..." if len(title) > 20 else title
    
    svg = f'''<svg width="300" height="450" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#667eea"/>
                <stop offset="100%" style="stop-color:#764ba2"/>
            </linearGradient>
        </defs>
        <rect width="100%" height="100%" fill="url(#bg)" rx="18"/>
        <circle cx="150" cy="180" r="45" fill="rgba(255,255,255,0.15)"/>
        <text x="50%" y="195" text-anchor="middle" fill="#fff" font-size="44">🎬</text>
        <text x="50%" y="250" text-anchor="middle" fill="#fff" font-size="16" font-weight="bold">{html.escape(display_title)}</text>
        <text x="50%" y="410" text-anchor="middle" fill="#fff" font-size="16" font-weight="700">SK4FiLM</text>
    </svg>'''
    
    return Response(svg, mimetype='image/svg+xml')

@app.route('/api/stats')
async def api_stats():
    """Public stats API"""
    return jsonify({
        'status': 'success',
        'stats': {
            'movies': len(movie_db['home_movies']),
            'cached_posters': len(movie_db['poster_cache']),
            'file_registry': len(file_registry),
            'poster_sources': movie_db['stats'],
            'last_update': movie_db['last_update'].isoformat() if movie_db['last_update'] else None
        }
    })

# ==================== STARTUP ====================
async def start_bot():
    """Start Telegram bot"""
    global bot_started
    try:
        logger.info("🤖 Starting bot...")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                await bot.start()
                me = await bot.get_me()
                logger.info(f"✅ Bot: @{me.username}")
                bot_started = True
                return
            except FloodWait as e:
                wait_time = min(e.value, 300)
                logger.warning(f"FloodWait: {wait_time}s")
                await asyncio.sleep(wait_time)
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.error(f"Bot start attempt {attempt+1} failed: {e}")
                    await asyncio.sleep(5)
                else:
                    raise
    except Exception as e:
        logger.error(f"Bot start error: {e}")

async def start_user():
    """Start user client"""
    global user_started
    try:
        logger.info("👤 Starting user client...")
        await User.start()
        me = await User.get_me()
        logger.info(f"✅ User: {me.first_name}")
        user_started = True
        
        # Initial update
        await background_update()
        
        # Start auto-update loop
        asyncio.create_task(auto_update_loop())
    except Exception as e:
        logger.error(f"User client error: {e}")

async def auto_update_loop():
    """Auto-update loop"""
    while True:
        try:
            await asyncio.sleep(Config.AUTO_UPDATE_INTERVAL)
            await background_update()
        except Exception as e:
            logger.error(f"Auto-update error: {e}")

async def start_web():
    """Start web server"""
    try:
        config = HyperConfig()
        config.bind = [f"0.0.0.0:{Config.WEB_SERVER_PORT}"]
        config.loglevel = "info"
        logger.info(f"🌐 Starting web server on port {Config.WEB_SERVER_PORT}")
        await serve(app, config)
    except Exception as e:
        logger.error(f"Web server error: {e}")

async def main():
    """Main startup"""
    logger.info("🚀 SK4FiLM - Starting Complete System")
    logger.info(f"📊 Config: {len(Config.TEXT_CHANNEL_IDS)} text channels, {len(Config.FILE_CHANNELS)} file channels")
    logger.info(f"💾 MongoDB: {'Enabled' if MONGODB_AVAILABLE and Config.MONGODB_URI else 'Disabled'}")
    
    await asyncio.gather(
        start_bot(),
        start_user(),
        start_web()
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("⚠️ Stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
