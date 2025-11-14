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

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bot.log')
    ]
)
logger = logging.getLogger(__name__)

# MongoDB (optional)
try:
    from motor.motor_asyncio import AsyncIOMotorClient
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
        "720p": [r"720p", r"720P", r"HD"],
        "480p": [r"480p", r"480P", r"SD"]
    }
    
    # MongoDB
    MONGODB_URI = os.environ.get("MONGODB_URI", "")
    DATABASE_NAME = os.environ.get("DATABASE_NAME", "sk4film_bot")
    
    # Settings
    FORCE_SUB_CHANNEL = int(os.environ.get("FORCE_SUB_CHANNEL", "-1001891090100"))
    WEBSITE_URL = os.environ.get("WEBSITE_URL", "https://sk4film.vercel.app")
    BOT_USERNAME = os.environ.get("BOT_USERNAME", "sk4film_bot")
    ADMIN_IDS = [int(x) for x in os.environ.get("ADMIN_IDS", "123456789").split(",")]
    AUTO_DELETE_TIME = int(os.environ.get("AUTO_DELETE_TIME", "300"))
    WEB_SERVER_PORT = int(os.environ.get("PORT", 8000))
    
    # APIs
    OMDB_KEYS = ["8265bd1c", "b9bd48a6", "2f2d1c8e"]
    TMDB_KEYS = ["e547e17d4e91f3e62a571655cd1ccaff"]
    AUTO_UPDATE_INTERVAL = 180

# Validate config
if Config.API_ID == 0 or not Config.API_HASH or not Config.BOT_TOKEN:
    logger.error("❌ Missing required config: API_ID, API_HASH, or BOT_TOKEN")
    exit(1)

# ==================== QUART APP ====================
app = Quart(__name__)

@app.after_request
async def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

# ==================== PYROGRAM CLIENTS ====================
bot = Client(
    "sk4film_bot",
    api_id=Config.API_ID,
    api_hash=Config.API_HASH,
    bot_token=Config.BOT_TOKEN,
    workdir="/tmp"
)

User = Client(
    "sk4film_user",
    api_id=Config.API_ID,
    api_hash=Config.API_HASH,
    session_string=Config.USER_SESSION_STRING,
    workdir="/tmp"
) if Config.USER_SESSION_STRING else None

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
file_registry = {}
users_collection = None
stats_collection = None

# MongoDB setup
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
    """Check if user is subscribed"""
    try:
        member = await bot.get_chat_member(Config.FORCE_SUB_CHANNEL, user_id)
        return member.status in ["member", "administrator", "creator"]
    except:
        return True

def extract_title_smart(text):
    """Smart title extraction"""
    if not text or len(text) < 10:
        return None
    try:
        # Remove emojis and special chars
        clean = re.sub(r'[^\w\s\(\)\-\.]', ' ', text[:100])
        lines = [l.strip() for l in clean.split('\n') if l.strip()]
        if not lines:
            return None
        
        first_line = lines[0]
        
        # Try patterns
        patterns = [
            r'^([A-Z][a-zA-Z\s]{3,40})',  # Title starting with capital
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4})',  # Multiple capitalized words
        ]
        
        for pattern in patterns:
            match = re.search(pattern, first_line)
            if match:
                title = match.group(1).strip()
                if 4 <= len(title) <= 50:
                    return title
        
        # Fallback: first line if reasonable
        if 4 <= len(first_line) <= 50:
            return first_line
        
        return None
    except:
        return None

def extract_title_from_file(message):
    """Extract title from file"""
    try:
        # Try caption first
        if message.caption:
            title = extract_title_smart(message.caption)
            if title:
                return title
        
        # Try filename
        filename = None
        if message.document:
            filename = message.document.file_name
        elif message.video:
            filename = message.video.file_name
        
        if filename:
            # Remove extension
            name = filename.rsplit('.', 1)[0]
            # Clean up
            name = re.sub(r'[\._\-]', ' ', name)
            name = re.sub(r'(720p|1080p|480p|HDRip|WEB|BluRay|x264|x265|HEVC|mkv|mp4)', '', name, flags=re.IGNORECASE)
            name = ' '.join(name.split()).strip()
            if 4 <= len(name) <= 50:
                return name
        
        return None
    except:
        return None

def format_size(size_bytes):
    """Format file size"""
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
    """Detect quality from filename"""
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
    """Format post content"""
    if not text:
        return ""
    formatted = html.escape(text)
    formatted = re.sub(r'(https?://[^\s]+)', r'<a href="\1" target="_blank" style="color: #00ccff;">\1</a>', formatted)
    formatted = formatted.replace('\n', '<br>')
    return formatted

def get_channel_name(channel_id):
    """Get channel name"""
    names = {
        -1001891090100: "SK4FiLM Main",
        -1002024811395: "SK4FiLM Updates",
        -1001768249569: "SK4FiLM Files"
    }
    return names.get(channel_id, "Channel")

def is_new_post(post_date):
    """Check if post is new (within 24 hours)"""
    try:
        if isinstance(post_date, str):
            post_date = datetime.fromisoformat(post_date.replace('Z', '+00:00'))
        delta = datetime.now() - post_date.replace(tzinfo=None)
        return delta.total_seconds() / 3600 <= 24
    except:
        return False

# ==================== SEARCH FUNCTION ====================
async def search_movies(query, limit=12, page=1):
    """Search movies with pagination"""
    if not User:
        logger.error("❌ User client not available")
        return {'results': [], 'pagination': {'current_page': 1, 'total_pages': 1, 'total_results': 0, 'per_page': limit, 'has_next': False, 'has_previous': False}}
    
    offset = (page - 1) * limit
    results = []
    seen = {}
    
    logger.info(f"🔍 Searching: '{query}' | Page: {page}")
    
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
                    
                    # Store file info
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
                        # Add quality to existing result
                        idx = seen[title_key]
                        results[idx]['has_file'] = True
                        results[idx]['type'] = 'with_file'
                        results[idx]['quality_options'][quality] = {
                            'file_id': unique_id,
                            'file_size': file_size,
                            'file_name': file_name
                        }
                    else:
                        # New result with file
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
    
    # Sort: files first, then by date
    results.sort(key=lambda x: (x['has_file'], x['date']), reverse=True)
    
    total = len(results)
    paginated = results[offset:offset + limit]
    
    logger.info(f"✅ Found {total} results | Showing {len(paginated)}")
    
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
    """Get latest movies for homepage"""
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
                            'is_new': is_new_post(msg.date),
                            'poster_url': f"{Config.WEBSITE_URL}/api/poster?title={urllib.parse.quote(title)}",
                            'has_poster': True
                        })
                        seen.add(title.lower())
                        count += 1
        except Exception as e:
            logger.error(f"Homepage channel {channel_id} error: {e}")
    
    # Sort by date
    movies.sort(key=lambda x: x['date'], reverse=True)
    return movies[:24]

# ==================== BOT HANDLERS ====================
@bot.on_message(filters.command("start") & filters.private)
async def start_handler(client, message):
    """Handle /start command"""
    user_id = message.from_user.id
    
    # Track user
    if users_collection:
        try:
            await users_collection.update_one(
                {'user_id': user_id},
                {'$set': {'first_name': message.from_user.first_name, 'last_seen': datetime.now()}, '$inc': {'start_count': 1}},
                upsert=True
            )
        except:
            pass
    
    # Check for file delivery
    if len(message.command) > 1:
        unique_id = message.command[1]
        
        # Check subscription
        if not await check_force_sub(user_id):
            try:
                channel = await bot.get_chat(Config.FORCE_SUB_CHANNEL)
                link = f"https://t.me/{channel.username}" if channel.username else f"https://t.me/c/{str(Config.FORCE_SUB_CHANNEL)[4:]}/1"
            except:
                link = f"https://t.me/c/{str(Config.FORCE_SUB_CHANNEL)[4:]}/1"
            
            await message.reply_text(
                "⚠️ **Please Join Our Channel First**\n\nAfter joining, click the download link again.",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("📢 Join Channel", url=link)]])
            )
            return
        
        # Get file info
        file_info = file_registry.get(unique_id)
        if not file_info:
            await message.reply_text(
                "❌ **File Not Found**\n\nLink expired. Please search again.",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🌐 Search Again", url=Config.WEBSITE_URL)]])
            )
            return
        
        try:
            # Send file
            processing = await message.reply_text(f"⏳ Sending file...\n\n📁 {file_info['file_name']}\n📊 {file_info['quality']}\n📦 {format_size(file_info['file_size'])}")
            
            if User:
                file_msg = await User.get_messages(file_info['channel_id'], file_info['message_id'])
                sent = await file_msg.copy(user_id)
                
                await processing.delete()
                
                success = await message.reply_text(
                    f"✅ **File Sent!**\n\n🎬 {file_info['title']}\n📊 {file_info['quality']}\n\n⚠️ Auto-delete in {Config.AUTO_DELETE_TIME//60} min\n\n🔍 More: {Config.WEBSITE_URL}"
                )
                
                logger.info(f"📥 File sent: {file_info['title']} to {user_id}")
                
                # Auto-delete
                if Config.AUTO_DELETE_TIME > 0:
                    await asyncio.sleep(Config.AUTO_DELETE_TIME)
                    try:
                        await sent.delete()
                        await success.edit_text("🗑️ File auto-deleted.")
                    except:
                        pass
            else:
                await processing.edit_text("❌ Service unavailable")
        
        except Exception as e:
            logger.error(f"File send error: {e}")
            await message.reply_text("❌ Error sending file. Try again.")
        
        return
    
    # Regular start
    await message.reply_text(
        f"🎬 **Welcome to SK4FiLM!**\n\n"
        f"📌 This bot delivers files automatically.\n\n"
        f"**How to use:**\n"
        f"1️⃣ Visit website\n"
        f"2️⃣ Search movie\n"
        f"3️⃣ Select quality\n"
        f"4️⃣ Get file instantly\n\n"
        f"⚡ Fast & Automated",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🌐 Visit Website", url=Config.WEBSITE_URL)]])
    )

@bot.on_message(filters.text & filters.private & ~filters.command(['start', 'stats']))
async def text_handler(client, message):
    """Handle text messages"""
    await message.reply_text(
        f"👋 Hi **{message.from_user.first_name}**!\n\n"
        f"🤖 Please use our website to search movies.\n\n"
        f"This bot only delivers files automatically.",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🌐 Visit Website", url=Config.WEBSITE_URL)]])
    )

@bot.on_message(filters.command("stats") & filters.user(Config.ADMIN_IDS))
async def stats_handler(client, message):
    """Admin stats"""
    try:
        total_users = await users_collection.count_documents({}) if users_collection else 0
        
        stats_text = (
            f"📊 **Bot Statistics**\n\n"
            f"👥 Users: `{total_users}`\n"
            f"🎬 Movies: `{len(movie_db['home_movies'])}`\n"
            f"📁 Files: `{len(file_registry)}`\n"
            f"🤖 Bot: `{'✅' if bot_started else '❌'}`\n"
            f"👤 User: `{'✅' if user_started else '❌'}`"
        )
        await message.reply_text(stats_text)
    except Exception as e:
        await message.reply_text(f"❌ Error: {e}")

# ==================== WEB ROUTES ====================
@app.route('/')
async def root():
    return jsonify({
        'status': 'healthy',
        'service': 'SK4FiLM API',
        'bot': f'@{Config.BOT_USERNAME}',
        'website': Config.WEBSITE_URL
    })

@app.route('/api/movies')
async def api_movies():
    """Homepage movies"""
    try:
        if not user_started:
            return jsonify({'status': 'error', 'message': 'Service starting'}), 503
        
        # Update if needed
        if not movie_db['home_movies'] or not movie_db['last_update'] or (datetime.now() - movie_db['last_update']).seconds > 300:
            movie_db['home_movies'] = await get_homepage_movies()
            movie_db['last_update'] = datetime.now()
        
        return jsonify({
            'status': 'success',
            'movies': movie_db['home_movies'],
            'total': len(movie_db['home_movies']),
            'bot_username': Config.BOT_USERNAME
        })
    except Exception as e:
        logger.error(f"API movies error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/search')
async def api_search():
    """Search API"""
    try:
        query = request.args.get('query', '').strip()
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 12))
        
        if not query:
            return jsonify({'status': 'error', 'message': 'Query required'}), 400
        
        if not user_started:
            return jsonify({'status': 'error', 'message': 'Service starting'}), 503
        
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
    """Custom poster"""
    title = request.args.get('title', 'Movie')
    display = title[:20] + "..." if len(title) > 20 else title
    
    svg = f'''<svg width="300" height="450" xmlns="http://www.w3.org/2000/svg">
        <defs><linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" style="stop-color:#667eea"/>
        <stop offset="100%" style="stop-color:#764ba2"/>
        </linearGradient></defs>
        <rect width="100%" height="100%" fill="url(#bg)" rx="18"/>
        <circle cx="150" cy="180" r="45" fill="rgba(255,255,255,0.15)"/>
        <text x="50%" y="195" text-anchor="middle" fill="#fff" font-size="44">🎬</text>
        <text x="50%" y="250" text-anchor="middle" fill="#fff" font-size="16" font-weight="bold">{html.escape(display)}</text>
        <text x="50%" y="410" text-anchor="middle" fill="#fff" font-size="16" font-weight="700">SK4FiLM</text>
    </svg>'''
    
    return Response(svg, mimetype='image/svg+xml')

# ==================== STARTUP ====================
async def start_bot_client():
    """Start bot"""
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
    """Start user client"""
    global user_started
    if not User:
        logger.error("❌ User session not configured")
        return
    
    try:
        logger.info("👤 Starting user client...")
        await User.start()
        me = await User.get_me()
        logger.info(f"✅ User: {me.first_name}")
        user_started = True
        
        # Initial load
        movie_db['home_movies'] = await get_homepage_movies()
        movie_db['last_update'] = datetime.now()
    except Exception as e:
        logger.error(f"❌ User error: {e}")

async def start_web_server():
    """Start web server"""
    try:
        config = HyperConfig()
        config.bind = [f"0.0.0.0:{Config.WEB_SERVER_PORT}"]
        config.loglevel = "warning"
        logger.info(f"🌐 Starting web server on port {Config.WEB_SERVER_PORT}")
        await serve(app, config)
    except Exception as e:
        logger.error(f"❌ Web error: {e}")

async def main():
    """Main"""
    logger.info("=" * 60)
    logger.info("🚀 SK4FiLM - Starting System")
    logger.info("=" * 60)
    logger.info(f"🤖 Bot: @{Config.BOT_USERNAME}")
    logger.info(f"🌐 Website: {Config.WEBSITE_URL}")
    logger.info(f"📊 Channels: {len(Config.TEXT_CHANNEL_IDS) + 1}")
    logger.info("=" * 60)
    
    await asyncio.gather(
        start_bot_client(),
        start_user_client(),
        start_web_server()
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n⚠️ Stopped")
    except Exception as e:
        logger.error(f"❌ Fatal: {e}")
