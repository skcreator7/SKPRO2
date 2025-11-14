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
    BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
    
    # Channels
    TEXT_CHANNEL_IDS = [int(x) for x in os.environ.get("TEXT_CHANNEL_IDS", "-1001891090100,-1002024811395").split(",")]
    FILE_CHANNEL_ID = int(os.environ.get("FILE_CHANNEL_ID", "-1001768249569"))
    
    # Quality patterns
    QUALITY_PATTERNS = {
        "1080p": [r"1080p", r"1080P", r"FullHD", r"FHD"],
        "720p": [r"720p", r"720P", r"HD", r"720p\.HEVC", r"720pHEVC"],
        "480p": [r"480p", r"480P", r"SD"]
    }
    
    # MongoDB
    MONGODB_URI = os.environ.get("MONGODB_URI", "")
    DATABASE_NAME = os.environ.get("DATABASE_NAME", "sk4film_bot")
    
    # Settings
    FORCE_SUB_CHANNEL = int(os.environ.get("FORCE_SUB_CHANNEL", "-1001891090100"))
    SHORTENER_API = os.environ.get("SHORTENER_API", "")
    SHORTENER_DOMAIN = os.environ.get("SHORTENER_DOMAIN", "https://gplinks.in/api")
    WEBSITE_URL = os.environ.get("WEBSITE_URL", "https://sk4film.vercel.app")
    BOT_USERNAME = os.environ.get("BOT_USERNAME", "sk4filmbot")
    ADMIN_IDS = [int(x) for x in os.environ.get("ADMIN_IDS", "123456789").split(",")]
    AUTO_DELETE_TIME = int(os.environ.get("AUTO_DELETE_TIME", "300"))
    WEB_SERVER_PORT = int(os.environ.get("PORT", 8000))
    
    # APIs
    OMDB_KEYS = ["8265bd1c", "b9bd48a6", "2f2d1c8e"]
    TMDB_KEYS = ["e547e17d4e91f3e62a571655cd1ccaff"]
    AUTO_UPDATE_INTERVAL = 180

# ==================== QUART APP ====================
app = Quart(__name__, static_folder=None, static_url_path=None)

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
file_registry = {}

# MongoDB setup
if MONGODB_AVAILABLE and Config.MONGODB_URI:
    try:
        mongo_client = AsyncIOMotorClient(
            Config.MONGODB_URI,
            serverSelectionTimeoutMS=30000,
            connectTimeoutMS=30000,
            socketTimeoutMS=30000,
            tls=True,
            tlsAllowInvalidCertificates=True
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
        api_url = f"{Config.SHORTENER_DOMAIN}?api={Config.SHORTENER_API}&url={urllib.parse.quote(url)}"
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
            title = re.sub(r'(720p|1080p|480p|HDRip|WEB-DL|BluRay|x264|x265|HEVC)', '', title, flags=re.IGNORECASE)
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

def detect_quality(filename):
    """Detect quality from filename"""
    filename_lower = filename.lower()
    
    for quality, patterns in Config.QUALITY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, filename_lower):
                return quality
    
    return "480p"

async def get_poster(title, session):
    """Get poster from multiple sources"""
    cache_key = title.lower().strip()
    if cache_key in movie_db['poster_cache']:
        cached, cache_time = movie_db['poster_cache'][cache_key]
        if datetime.now() - cache_time < timedelta(minutes=12):
            return cached
    
    try:
        # Try OMDB
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
        
        # Try TMDB
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
        
        # Custom poster
        result = {
            'poster_url': f"{Config.WEBSITE_URL}/api/poster?title={urllib.parse.quote(title)}",
            'title': title,
            'source': 'CUSTOM',
            'success': True
        }
        movie_db['poster_cache'][cache_key] = (result, datetime.now())
        movie_db['stats']['custom'] += 1
        return result
    except:
        return {
            'poster_url': f"{Config.WEBSITE_URL}/api/poster?title={urllib.parse.quote(title)}",
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
    """Format post content for HTML display"""
    if not text:
        return ""
    formatted = html.escape(text)
    formatted = re.sub(r'(https?://[^\s]+)', r'<a href="\1" target="_blank" style="color: #00ccff;">\1</a>', formatted)
    formatted = formatted.replace('\n', '<br>')
    return formatted

def get_channel_name(channel_id):
    """Get friendly channel name"""
    channel_names = {
        -1001891090100: "SK4FiLM Main",
        -1002024811395: "SK4FiLM Updates",
        -1001768249569: "SK4FiLM Files"
    }
    return channel_names.get(channel_id, f"Channel {channel_id}")

# ==================== SEARCH FUNCTIONS ====================
async def search_movies(query, limit=12, page=1):
    """Search text channels and file channel with quality options"""
    offset = (page - 1) * limit
    results = []
    seen = {}
    
    logger.info(f"🔍 Search started: '{query}' | Page: {page}")
    
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
                            'channel': get_channel_name(channel_id),
                            'channel_id': channel_id,
                            'message_id': msg.id,
                            'is_new': is_new_post(msg.date) if msg.date else False,
                            'has_file': False,
                            'quality_options': {}
                        })
                        seen[title.lower()] = len(results) - 1
        except Exception as e:
            logger.warning(f"Text channel {channel_id} search error: {e}")
    
    # Search file channel
    try:
        async for msg in User.search_messages(Config.FILE_CHANNEL_ID, query, limit=50):
            if msg.document or msg.video:
                title = extract_title_from_file(msg)
                if title:
                    file_id = msg.document.file_id if msg.document else msg.video.file_id
                    file_size = msg.document.file_size if msg.document else msg.video.file_size
                    file_name = msg.document.file_name if msg.document else (msg.video.file_name or 'video.mp4')
                    
                    detected_quality = detect_quality(file_name)
                    unique_id = hashlib.md5(f"{file_id}{time.time()}".encode()).hexdigest()
                    
                    # Store in registry
                    file_registry[unique_id] = {
                        'file_id': file_id,
                        'channel_id': Config.FILE_CHANNEL_ID,
                        'message_id': msg.id,
                        'quality': detected_quality,
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
                        results[idx]['quality_options'][detected_quality] = {
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
                                detected_quality: {
                                    'file_id': unique_id,
                                    'file_size': file_size,
                                    'file_name': file_name
                                }
                            }
                        })
                        seen[title_key] = len(results) - 1
    except Exception as e:
        logger.warning(f"File channel search error: {e}")
    
    # Sort results
    results.sort(key=lambda x: (x['has_file'], x['date']), reverse=True)
    total = len(results)
    paginated = results[offset:offset+limit]
    
    logger.info(f"✅ Search complete: {total} results | Showing: {len(paginated)}")
    
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
    """Background update for homepage"""
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

async def cleanup_old_files():
    """Cleanup old file registry entries"""
    try:
        cutoff = datetime.now() - timedelta(hours=2)
        to_remove = [
            uid for uid, info in file_registry.items()
            if info.get('created_at', datetime.now()) < cutoff
        ]
        for uid in to_remove:
            del file_registry[uid]
        if to_remove:
            logger.info(f"🗑️ Cleaned up {len(to_remove)} old file entries")
    except Exception as e:
        logger.error(f"Cleanup error: {e}")

# ==================== BOT HANDLERS ====================
@bot.on_message(filters.command("start") & filters.private)
async def start_cmd(client, message):
    """Bot start command - Only for file delivery"""
    user_id = message.from_user.id
    
    # Handle file request with start parameter
    if len(message.command) > 1:
        param = message.command[1]
        
        # Check subscription first
        if not await check_force_sub(user_id):
            try:
                channel = await bot.get_chat(Config.FORCE_SUB_CHANNEL)
                channel_link = f"https://t.me/{channel.username}" if channel.username else f"https://t.me/c/{str(Config.FORCE_SUB_CHANNEL)[4:]}/1"
            except:
                channel_link = f"https://t.me/c/{str(Config.FORCE_SUB_CHANNEL)[4:]}/1"
            
            await message.reply_text(
                "⚠️ **Access Restricted**\n\n"
                "Please join our channel first to download files.\n\n"
                "After joining, click the download link again.",
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton("📢 Join Channel", url=channel_link)
                ]])
            )
            return
        
        # Handle file request
        unique_id = param
        file_info = file_registry.get(unique_id)
        
        if not file_info:
            await message.reply_text(
                "❌ **File Not Found**\n\n"
                "This file link has expired or is invalid.\n\n"
                "Please search again on our website.",
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton("🌐 Search on Website", url=Config.WEBSITE_URL)
                ]])
            )
            return
        
        try:
            # Send processing message
            processing_msg = await message.reply_text(
                f"⏳ **Processing Your Request**\n\n"
                f"📁 File: `{file_info['file_name']}`\n"
                f"🎬 Quality: `{file_info['quality']}`\n"
                f"📦 Size: `{format_size(file_info['file_size'])}`\n\n"
                f"Please wait..."
            )
            
            # Forward file from channel
            file_msg = await User.get_messages(file_info['channel_id'], file_info['message_id'])
            sent_msg = await file_msg.copy(user_id)
            
            # Delete processing message
            await processing_msg.delete()
            
            # Send success message
            success_msg = await message.reply_text(
                f"✅ **File Sent Successfully!**\n\n"
                f"🎬 {file_info['title']}\n"
                f"📊 Quality: `{file_info['quality']}`\n"
                f"📦 Size: `{format_size(file_info['file_size'])}`\n\n"
                f"⚠️ This file will auto-delete in {Config.AUTO_DELETE_TIME//60} minutes.\n\n"
                f"🔍 Search more movies: {Config.WEBSITE_URL}"
            )
            
            # Track stats
            if stats_collection:
                try:
                    await stats_collection.insert_one({
                        'user_id': user_id,
                        'file_id': unique_id,
                        'quality': file_info['quality'],
                        'title': file_info['title'],
                        'timestamp': datetime.now(),
                        'type': 'download'
                    })
                except:
                    pass
            
            logger.info(f"📥 File delivered: {file_info['title']} ({file_info['quality']}) to {user_id}")
            
            # Auto-delete after configured time
            if Config.AUTO_DELETE_TIME > 0:
                await asyncio.sleep(Config.AUTO_DELETE_TIME)
                try:
                    await sent_msg.delete()
                    await success_msg.edit_text(
                        f"🗑️ **File Auto-Deleted**\n\n"
                        f"The file has been automatically deleted for security.\n\n"
                        f"🔄 Search again: {Config.WEBSITE_URL}"
                    )
                except:
                    pass
        
        except Exception as e:
            logger.error(f"File forward error: {e}")
            await message.reply_text(
                "❌ **Error Sending File**\n\n"
                "An error occurred while processing your request.\n\n"
                "Please try again or contact support.",
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton("🔄 Try Again", url=Config.WEBSITE_URL)
                ]])
            )
        
        return
    
    # Regular start - Redirect to website
    if users_collection:
        try:
            await users_collection.update_one(
                {'user_id': user_id},
                {
                    '$set': {
                        'first_name': message.from_user.first_name,
                        'username': message.from_user.username,
                        'last_seen': datetime.now()
                    },
                    '$inc': {'start_count': 1}
                },
                upsert=True
            )
        except:
            pass
    
    await message.reply_text(
        f"🎬 **Welcome to SK4FiLM Bot!**\n\n"
        f"📌 This bot is only for **file delivery**.\n\n"
        f"🔍 To search movies and download:\n"
        f"👉 Visit our website\n\n"
        f"**How it works:**\n"
        f"1️⃣ Search movie on website\n"
        f"2️⃣ Select quality (480p/720p/1080p)\n"
        f"3️⃣ Click download - Bot will send file\n\n"
        f"⚡ Fast • Simple • Automated",
        reply_markup=InlineKeyboardMarkup([[
            InlineKeyboardButton("🌐 Visit Website", url=Config.WEBSITE_URL)
        ]])
    )

@bot.on_message(filters.text & filters.private & ~filters.command(['start', 'stats', 'broadcast']))
async def handle_any_message(client, message):
    """Handle any text message - Redirect to website"""
    user_id = message.from_user.id
    
    await message.reply_text(
        f"👋 Hi **{message.from_user.first_name}**!\n\n"
        f"🤖 This bot doesn't accept direct messages.\n\n"
        f"🔍 **To search movies:**\n"
        f"Please visit our website below\n\n"
        f"📌 All searches and downloads are done through the website.\n"
        f"This bot only delivers files automatically.",
        reply_markup=InlineKeyboardMarkup([[
            InlineKeyboardButton("🌐 Visit SK4FiLM Website", url=Config.WEBSITE_URL)
        ]])
    )
    
    logger.info(f"💬 Message from {user_id}: '{message.text[:50]}' - Redirected to website")

@bot.on_message(filters.command("stats") & filters.user(Config.ADMIN_IDS))
async def stats_cmd(client, message):
    """Admin stats command"""
    try:
        total_users = await users_collection.count_documents({}) if users_collection else 0
        total_searches = await stats_collection.count_documents({'type': 'search'}) if stats_collection else 0
        total_downloads = await stats_collection.count_documents({'type': 'download'}) if stats_collection else 0
        
        # Recent activity
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        today_downloads = await stats_collection.count_documents({'type': 'download', 'timestamp': {'$gte': today}}) if stats_collection else 0
        
        stats_text = (
            f"📊 **SK4FiLM Bot Statistics**\n\n"
            f"**Overall:**\n"
            f"👥 Total Users: `{total_users}`\n"
            f"🔍 Total Searches: `{total_searches}`\n"
            f"⬇️ Total Downloads: `{total_downloads}`\n\n"
            f"**Today:**\n"
            f"⬇️ Downloads: `{today_downloads}`\n\n"
            f"**Database:**\n"
            f"🎬 Movies: `{len(movie_db['home_movies'])}`\n"
            f"🖼️ Cached Posters: `{len(movie_db['poster_cache'])}`\n"
            f"📁 File Registry: `{len(file_registry)}`\n\n"
            f"**Poster Sources:**\n"
            f"• OMDB: `{movie_db['stats']['omdb']}`\n"
            f"• TMDB: `{movie_db['stats']['tmdb']}`\n"
            f"• Custom: `{movie_db['stats']['custom']}`\n\n"
            f"🕒 Last Update: `{movie_db['last_update'].strftime('%H:%M:%S') if movie_db['last_update'] else 'Never'}`\n"
            f"🤖 Bot: `{'✅ Running' if bot_started else '❌ Stopped'}`\n"
            f"👤 User Client: `{'✅ Running' if user_started else '❌ Stopped'}`"
        )
        await message.reply_text(stats_text)
    except Exception as e:
        await message.reply_text(f"❌ Error: {e}")

@bot.on_message(filters.command("broadcast") & filters.user(Config.ADMIN_IDS))
async def broadcast_cmd(client, message):
    """Admin broadcast command"""
    if not message.reply_to_message:
        await message.reply_text("❌ Reply to a message to broadcast")
        return
    
    if not users_collection:
        await message.reply_text("❌ MongoDB not available")
        return
    
    try:
        users = await users_collection.find({}).to_list(length=None)
        total = len(users)
        success = 0
        failed = 0
        
        status_msg = await message.reply_text(f"📢 Broadcasting to {total} users...")
        
        for user in users:
            try:
                await message.reply_to_message.copy(user['user_id'])
                success += 1
                
                if success % 50 == 0:
                    await status_msg.edit_text(f"📢 Progress: {success}/{total}")
                
                await asyncio.sleep(0.05)
            except:
                failed += 1
        
        await status_msg.edit_text(
            f"✅ **Broadcast Complete**\n\n"
            f"✅ Success: `{success}`\n"
            f"❌ Failed: `{failed}`\n"
            f"📊 Total: `{total}`"
        )
    except Exception as e:
        await message.reply_text(f"❌ Error: {e}")

# ==================== WEB API ROUTES ====================
@app.route('/')
async def home():
    return jsonify({
        'status': 'healthy' if bot_started and user_started else 'starting',
        'service': 'SK4FiLM API',
        'version': '2.0',
        'bot_username': f'@{Config.BOT_USERNAME}',
        'website': Config.WEBSITE_URL,
        'bot_mode': 'File Delivery Only',
        'channels': {
            'text': len(Config.TEXT_CHANNEL_IDS),
            'file': 1
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
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/movies')
async def api_movies():
    """Home movies API"""
    if not user_started:
        return jsonify({'status': 'error', 'message': 'Service starting'}), 503
    
    logger.info("📱 API: Homepage movies requested")
    
    return jsonify({
        'status': 'success',
        'movies': movie_db['home_movies'],
        'total': len(movie_db['home_movies']),
        'last_update': movie_db['last_update'].isoformat() if movie_db['last_update'] else None,
        'bot_username': Config.BOT_USERNAME
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
        logger.info(f"📱 API: Search request - '{query}' | Page: {page}")
        result = await search_movies(query, limit, page)
        
        return jsonify({
            'status': 'success',
            'query': query,
            'results': result['results'],
            'pagination': result['pagination'],
            'bot_username': Config.BOT_USERNAME
        })
    except Exception as e:
        logger.error(f"Search API error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/file/<unique_id>')
async def api_file_info(unique_id):
    """Get file info"""
    file_info = file_registry.get(unique_id)
    if not file_info:
        return jsonify({'status': 'error', 'message': 'File not found'}), 404
    
    logger.info(f"📱 API: File info requested - {unique_id}")
    
    return jsonify({
        'status': 'success',
        'file': {
            'unique_id': unique_id,
            'title': file_info['title'],
            'quality': file_info['quality'],
            'file_size': file_info['file_size'],
            'file_size_formatted': format_size(file_info['file_size']),
            'file_name': file_info['file_name'],
            'bot_username': Config.BOT_USERNAME,
            'download_url': f"https://t.me/{Config.BOT_USERNAME}?start={unique_id}"
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
    """Public stats"""
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
        logger.info("🤖 Starting Telegram bot...")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                await bot.start()
                me = await bot.get_me()
                logger.info(f"✅ Bot started: @{me.username}")
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
        logger.error(f"❌ Bot start error: {e}")

async def start_user():
    """Start user client"""
    global user_started
    try:
        logger.info("👤 Starting user client...")
        await User.start()
        me = await User.get_me()
        logger.info(f"✅ User client started: {me.first_name}")
        user_started = True
        
        # Initial homepage update
        await background_update()
        
        # Start background tasks
        asyncio.create_task(auto_update_loop())
        asyncio.create_task(cleanup_loop())
    except Exception as e:
        logger.error(f"❌ User client error: {e}")

async def auto_update_loop():
    """Auto-update homepage movies"""
    while True:
        try:
            await asyncio.sleep(Config.AUTO_UPDATE_INTERVAL)
            await background_update()
        except Exception as e:
            logger.error(f"Auto-update error: {e}")

async def cleanup_loop():
    """Cleanup old file registry entries"""
    while True:
        try:
            await asyncio.sleep(3600)  # Every hour
            await cleanup_old_files()
        except Exception as e:
            logger.error(f"Cleanup loop error: {e}")

async def start_web():
    """Start web API server"""
    try:
        config = HyperConfig()
        config.bind = [f"0.0.0.0:{Config.WEB_SERVER_PORT}"]
        config.loglevel = "info"
        logger.info(f"🌐 Starting web API server on port {Config.WEB_SERVER_PORT}")
        await serve(app, config)
    except Exception as e:
        logger.error(f"❌ Web server error: {e}")

async def main():
    """Main startup function"""
    logger.info("=" * 70)
    logger.info("🚀 SK4FiLM - Movie Bot & API System")
    logger.info("=" * 70)
    logger.info(f"📌 Bot Mode: FILE DELIVERY ONLY")
    logger.info(f"📊 Configuration:")
    logger.info(f"   • Text Channels: {len(Config.TEXT_CHANNEL_IDS)}")
    logger.info(f"   • File Channel: 1")
    logger.info(f"   • Force Subscribe: ✅ Enabled")
    logger.info(f"   • Auto Delete: {Config.AUTO_DELETE_TIME}s")
    logger.info(f"   • Auto Update: {Config.AUTO_UPDATE_INTERVAL}s")
    logger.info(f"   • Web Port: {Config.WEB_SERVER_PORT}")
    logger.info(f"💾 MongoDB: {'✅ Enabled' if MONGODB_AVAILABLE and Config.MONGODB_URI else '❌ Disabled'}")
    logger.info(f"🌐 Website: {Config.WEBSITE_URL}")
    logger.info(f"🤖 Bot: @{Config.BOT_USERNAME}")
    logger.info("=" * 70)
    logger.info("🔄 Starting all services...")
    
    # Start all services concurrently
    await asyncio.gather(
        start_bot(),      # Telegram bot - file delivery only
        start_user(),     # User client - channel access
        start_web()       # Web API server
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n⚠️ Stopped by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
