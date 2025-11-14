import asyncio
import os
import logging
import hashlib
import time
from datetime import datetime, timedelta
from pyrogram import Client, filters
from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from quart import Quart, jsonify, request, Response
from hypercorn.asyncio import serve
from hypercorn.config import Config as HyperConfig
import html
import re
import math
import aiohttp
import urllib.parse
import uuid

# ==================== CONFIG ====================
class Config:
    API_ID = int(os.environ.get("API_ID", "0"))
    API_HASH = os.environ.get("API_HASH", "")
    USER_SESSION_STRING = os.environ.get("USER_SESSION_STRING", "")
    BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
    
    TEXT_CHANNEL_IDS = [-1001891090100, -1002024811395]
    FILE_CHANNEL_ID = int(os.environ.get("FILE_CHANNEL_ID", "-1001768249569"))
    
    SECRET_KEY = os.environ.get("SECRET_KEY", "sk4film-secret-key-2024")
    WEB_SERVER_PORT = int(os.environ.get("PORT", 8000))
    
    OMDB_KEYS = ["8265bd1c", "b9bd48a6", "2f2d1c8e", "c3e6f8d9"]
    TMDB_KEYS = ["e547e17d4e91f3e62a571655cd1ccaff", "a1b2c3d4e5f6g7h8"]
    
    FORCE_SUB_CHANNEL = int(os.environ.get("FORCE_SUB_CHANNEL", "-1001891090100"))
    WEBSITE_URL = os.environ.get("WEBSITE_URL", "https://sk4film.vercel.app")
    BOT_USERNAME = os.environ.get("BOT_USERNAME", "sk4filmbot")
    ADMIN_IDS = [int(x) for x in os.environ.get("ADMIN_IDS", "123456789").split(",")]
    AUTO_DELETE_TIME = int(os.environ.get("AUTO_DELETE_TIME", "300"))
    BACKEND_URL = os.environ.get("BACKEND_URL", "https://web-bot.koyeb.app")
    
    AUTO_UPDATE_INTERVAL = 180

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Quart(__name__)
User = None
bot = None
bot_started = False
auto_update_task = None

# ==================== GLOBAL STATE ====================
movie_db = {
    'all_movies': [],
    'home_movies': [],
    'last_update': None,
    'poster_cache': {},
    'updating': False,
    'stats': {'omdb': 0, 'tmdb_hq': 0, 'justwatch_hq': 0, 'letterboxd_hq': 0, 'imdb_hq': 0, 'custom': 0}
}
file_registry = {}

# ==================== HELPER FUNCTIONS ====================
def extract_title_smart(text):
    if not text or len(text) < 15:
        return None
    
    try:
        clean_text = re.sub(r'[^\w\s\(\)\-\.\n\u0900-\u097F]', ' ', text)
        first_line = clean_text.split('\n')[0].strip()
        
        patterns = [
            r'🎬\s*([^-\n]{4,45})(?:\s*-|\n|$)',
            r'^([^(]{4,45})\s*\(\d{4}\)',
            r'^([^-]{4,45})\s*-\s*(?:Hindi|English|Tamil|Telugu|20\d{2})',
            r'^([A-Z][a-z]+(?:\s+[A-Za-z]+){1,4})',
            r'"([^"]{4,35})"'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, first_line, re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                title = re.sub(r'\s+', ' ', title)
                
                if validate_title_smart(title):
                    return title
        
        return None
        
    except:
        return None

def validate_title_smart(title):
    if not title or len(title) < 4 or len(title) > 45:
        return False
    
    bad_words = ['size', 'quality', 'download', 'link', 'channel', 'mb', 'gb', 'file']
    if any(word in title.lower() for word in bad_words):
        return False
    
    return bool(re.search(r'[a-zA-Z]', title))

def extract_title_from_file(msg):
    """Extract title from file message"""
    try:
        # Try caption first
        if msg.caption:
            t = extract_title_smart(msg.caption)
            if t:
                return t
        
        # Try filename
        fn = None
        if msg.document:
            fn = msg.document.file_name
        elif msg.video:
            fn = msg.video.file_name
        
        if fn:
            name = fn.rsplit('.', 1)[0]
            name = re.sub(r'[\._\-]', ' ', name)
            name = re.sub(r'(720p|1080p|480p|HDRip|WEB|BluRay|x264|x265|HEVC)', '', name, flags=re.IGNORECASE)
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

def is_new_post(post_date):
    try:
        if isinstance(post_date, str):
            post_date = datetime.fromisoformat(post_date.replace('Z', '+00:00'))
        
        hours_ago = (datetime.now() - post_date.replace(tzinfo=None)).total_seconds() / 3600
        return hours_ago <= 24
    except:
        return False

def format_original_post(text):
    if not text:
        return ""
    
    formatted = html.escape(text)
    formatted = re.sub(
        r'(https?://[^\s]+)', 
        r'<a href="\1" target="_blank" style="color: #00ccff; text-decoration: underline;">\1</a>', 
        formatted
    )
    formatted = formatted.replace('\n', '<br>')
    return formatted

def get_channel_name(cid):
    return {-1001891090100: "SK4FiLM Main", -1002024811395: "SK4FiLM Updates", -1001768249569: "SK4FiLM Files"}.get(cid, "Channel")

# ==================== BOT FORCE SUB ====================
async def check_force_sub(user_id):
    """Check if user subscribed"""
    try:
        member = await bot.get_chat_member(Config.FORCE_SUB_CHANNEL, user_id)
        return member.status in ["member", "administrator", "creator"]
    except:
        return True

# ==================== POSTER FUNCTIONS ====================
async def get_high_quality_poster(title, session):
    """HIGH QUALITY poster fetching"""
    cache_key = title.lower().strip()
    
    if cache_key in movie_db['poster_cache']:
        cached, cache_time = movie_db['poster_cache'][cache_key]
        if datetime.now() - cache_time < timedelta(minutes=12):
            return cached
    
    try:
        logger.info(f"🔍 Fetching HQ poster: {title}")
        
        # SOURCE 1: OMDB
        for api_key in Config.OMDB_KEYS:
            try:
                omdb_url = f"http://www.omdbapi.com/?t={urllib.parse.quote(title)}&apikey={api_key}&plot=short"
                
                async with session.get(omdb_url, timeout=7) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if (data.get('Response') == 'True' and 
                            data.get('Poster') and 
                            data['Poster'] != 'N/A' and
                            data['Poster'].startswith('http')):
                            
                            result = {
                                'poster_url': data['Poster'].replace('http://', 'https://'),
                                'title': data.get('Title', title),
                                'year': data.get('Year', ''),
                                'rating': data.get('imdbRating', ''),
                                'source': 'OMDB',
                                'quality': 'HIGH',
                                'success': True
                            }
                            
                            movie_db['poster_cache'][cache_key] = (result, datetime.now())
                            movie_db['stats']['omdb'] += 1
                            
                            logger.info(f"✅ OMDB poster: {title}")
                            return result
                
                await asyncio.sleep(0.1)
                
            except:
                continue
        
        # SOURCE 2: TMDB HIGH QUALITY
        for tmdb_key in Config.TMDB_KEYS:
            try:
                tmdb_search_url = "https://api.themoviedb.org/3/search/movie"
                tmdb_params = {'api_key': tmdb_key, 'query': title}
                
                async with session.get(tmdb_search_url, params=tmdb_params, timeout=8) as response:
                    if response.status == 200:
                        tmdb_data = await response.json()
                        
                        if tmdb_data.get('results') and len(tmdb_data['results']) > 0:
                            movie = tmdb_data['results'][0]
                            poster_path = movie.get('poster_path')
                            
                            if poster_path:
                                hq_poster_url = f"https://image.tmdb.org/t/p/w780{poster_path}"
                                
                                result = {
                                    'poster_url': hq_poster_url,
                                    'title': movie.get('title', title),
                                    'year': movie.get('release_date', '')[:4] if movie.get('release_date') else '',
                                    'rating': f"{movie.get('vote_average', 0):.1f}",
                                    'source': 'TMDB',
                                    'quality': 'HIGH',
                                    'success': True
                                }
                                
                                movie_db['poster_cache'][cache_key] = (result, datetime.now())
                                movie_db['stats']['tmdb_hq'] += 1
                                
                                logger.info(f"✅ TMDB poster: {title}")
                                return result
                
            except Exception as e:
                logger.warning(f"TMDB error: {e}")
        
        # Fallback: Custom poster
        logger.info(f"ℹ️ Using custom poster: {title}")
        custom_result = {
            'poster_url': f"{Config.BACKEND_URL}/api/enhanced_poster?title={urllib.parse.quote(title)}",
            'title': title,
            'source': 'CUSTOM',
            'quality': 'CUSTOM',
            'success': True
        }
        
        movie_db['poster_cache'][cache_key] = (custom_result, datetime.now())
        movie_db['stats']['custom'] += 1
        
        return custom_result
        
    except:
        return {
            'poster_url': f"{Config.BACKEND_URL}/api/enhanced_poster?title={urllib.parse.quote(title)}",
            'title': title,
            'source': 'CUSTOM',
            'quality': 'CUSTOM',
            'success': True
        }

# ==================== SEARCH WITH FILE SUPPORT ====================
async def search_with_files_and_pagination(query, limit=12, page=1):
    """Complete search with file support and pagination"""
    try:
        if not User:
            return {'results': [], 'pagination': {'current_page': 1, 'total_pages': 1, 'total_results': 0, 'per_page': limit, 'has_next': False, 'has_previous': False}}
        
        offset = (page - 1) * limit
        results = []
        seen = {}
        
        logger.info(f"🔍 Search: '{query}' | Page {page}")
        
        # Search text channels
        for channel_id in Config.TEXT_CHANNEL_IDS:
            try:
                cnt = 0
                async for message in User.search_messages(channel_id, query, limit=20):
                    if cnt >= 20:
                        break
                    
                    if message.text:
                        title = extract_title_smart(message.text)
                        
                        if title and title.lower() not in seen:
                            original_content = format_original_post(message.text)
                            
                            results.append({
                                'title': title,
                                'type': 'text_post',
                                'content': original_content,
                                'date': message.date.isoformat() if message.date else datetime.now().isoformat(),
                                'channel': get_channel_name(channel_id),
                                'channel_id': channel_id,
                                'message_id': message.id,
                                'is_new': is_new_post(message.date) if message.date else False,
                                'has_file': False,
                                'quality_options': {}
                            })
                            seen[title.lower()] = len(results) - 1
                            cnt += 1
                
                logger.info(f"  ✓ Text channel: {cnt} results")
            except Exception as e:
                logger.error(f"Text channel error: {e}")
        
        # Search file channel
        try:
            cnt = 0
            async for message in User.search_messages(Config.FILE_CHANNEL_ID, query, limit=30):
                if cnt >= 30:
                    break
                
                if message.document or message.video:
                    title = extract_title_from_file(message)
                    
                    if title:
                        file_id = message.document.file_id if message.document else message.video.file_id
                        file_size = message.document.file_size if message.document else (message.video.file_size if message.video else 0)
                        file_name = message.document.file_name if message.document else (message.video.file_name if message.video else 'video.mp4')
                        
                        quality = detect_quality(file_name)
                        unique_id = hashlib.md5(f"{file_id}{time.time()}".encode()).hexdigest()
                        
                        # Store file in registry
                        file_registry[unique_id] = {
                            'file_id': file_id,
                            'channel_id': Config.FILE_CHANNEL_ID,
                            'message_id': message.id,
                            'quality': quality,
                            'file_size': file_size,
                            'file_name': file_name,
                            'title': title,
                            'created_at': datetime.now()
                        }
                        
                        title_key = title.lower()
                        
                        if title_key in seen:
                            # Add quality option to existing result
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
                                'content': format_original_post(message.caption or title),
                                'date': message.date.isoformat() if message.date else datetime.now().isoformat(),
                                'channel': get_channel_name(Config.FILE_CHANNEL_ID),
                                'channel_id': Config.FILE_CHANNEL_ID,
                                'message_id': message.id,
                                'is_new': is_new_post(message.date) if message.date else False,
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
                        cnt += 1
            
            logger.info(f"  ✓ File channel: {cnt} files")
        except Exception as e:
            logger.error(f"File channel error: {e}")
        
        # Sort: files first, then by date
        results.sort(key=lambda x: (x['has_file'], x['date']), reverse=True)
        
        total_results = len(results)
        paginated = results[offset:offset + limit]
        total_pages = math.ceil(total_results / limit) if total_results > 0 else 1
        
        logger.info(f"✅ Search complete: {total_results} total | {len(paginated)} on page")
        
        return {
            "results": paginated,
            "pagination": {
                "current_page": page,
                "total_pages": total_pages,
                "total_results": total_results,
                "per_page": limit,
                "has_next": page < total_pages,
                "has_previous": page > 1
            }
        }
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return {"results": [], "pagination": {"current_page": 1, "total_pages": 1, "total_results": 0}}

# ==================== BACKGROUND UPDATE ====================
async def background_update_hq():
    """Background update with HIGH QUALITY posters"""
    if not User or not bot_started or movie_db['updating']:
        return
    
    try:
        movie_db['updating'] = True
        logger.info("🔄 Background update starting...")
        
        all_posts = []
        
        for channel_id in Config.TEXT_CHANNEL_IDS:
            try:
                count = 0
                async for message in User.get_chat_history(channel_id, limit=30):
                    if message.text and len(message.text) > 40 and message.date:
                        title = extract_title_smart(message.text)
                        
                        if title:
                            all_posts.append({
                                'title': title,
                                'original_text': message.text,
                                'date': message.date,
                                'date_iso': message.date.isoformat(),
                                'channel': get_channel_name(channel_id),
                                'message_id': message.id,
                                'is_new': is_new_post(message.date)
                            })
                            count += 1
                
                logger.info(f"  ✓ Channel: {count} posts")
            except Exception as e:
                logger.warning(f"Channel error: {e}")
        
        all_posts.sort(key=lambda x: x['date'], reverse=True)
        
        # Remove duplicates
        unique_movies = []
        seen = set()
        
        for post in all_posts:
            title_key = post['title'].lower()
            if title_key not in seen:
                seen.add(title_key)
                post['date'] = post['date_iso']
                del post['date_iso']
                unique_movies.append(post)
        
        # Add HIGH QUALITY posters
        async with aiohttp.ClientSession() as session:
            batch_size = 4
            
            for i in range(0, min(len(unique_movies), 60), batch_size):
                batch = unique_movies[i:i + batch_size]
                
                poster_tasks = [get_high_quality_poster(movie['title'], session) for movie in batch]
                poster_results = await asyncio.gather(*poster_tasks, return_exceptions=True)
                
                for movie, poster_data in zip(batch, poster_results):
                    if isinstance(poster_data, dict) and poster_data.get('success'):
                        movie.update({
                            'poster_url': poster_data['poster_url'],
                            'poster_title': poster_data['title'],
                            'poster_year': poster_data.get('year', ''),
                            'poster_rating': poster_data.get('rating', ''),
                            'poster_source': poster_data['source'],
                            'poster_quality': poster_data.get('quality', 'STANDARD'),
                            'has_poster': True
                        })
                
                await asyncio.sleep(0.3)
        
        movie_db['all_movies'] = unique_movies
        movie_db['home_movies'] = unique_movies[:24]
        movie_db['last_update'] = datetime.now()
        
        logger.info(f"✅ Updated: {len(unique_movies)} movies | Home: {len(movie_db['home_movies'])}")
        logger.info(f"📊 Poster stats: {movie_db['stats']}")
        
    except Exception as e:
        logger.error(f"Background update error: {e}")
    finally:
        movie_db['updating'] = False

async def start_hidden_hq_update():
    """Auto-update loop"""
    global auto_update_task
    
    async def hq_update_loop():
        while bot_started:
            try:
                await asyncio.sleep(Config.AUTO_UPDATE_INTERVAL)
                logger.info("🔄 Auto-update triggered")
                await background_update_hq()
            except Exception as e:
                logger.error(f"Auto-update error: {e}")
    
    auto_update_task = asyncio.create_task(hq_update_loop())
    logger.info(f"✅ Auto-updates started (every {Config.AUTO_UPDATE_INTERVAL//60} min)")

# ==================== BOT HANDLERS ====================
@bot.on_message(filters.command("start") & filters.private)
async def start_handler(client, message):
    uid = message.from_user.id
    logger.info(f"👤 /start from {uid}")
    
    # File delivery
    if len(message.command) > 1:
        file_id = message.command[1]
        logger.info(f"📥 File request: {file_id}")
        
        # Check subscription
        if not await check_force_sub(uid):
            try:
                ch = await bot.get_chat(Config.FORCE_SUB_CHANNEL)
                link = f"https://t.me/{ch.username}" if ch.username else f"https://t.me/c/{str(Config.FORCE_SUB_CHANNEL)[4:]}/1"
            except:
                link = f"https://t.me/c/{str(Config.FORCE_SUB_CHANNEL)[4:]}/1"
            
            await message.reply_text(
                "⚠️ **Join Channel First**\n\nAfter joining, click download again.",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("📢 Join", url=link)]])
            )
            return
        
        # Get file info
        fi = file_registry.get(file_id)
        if not fi:
            await message.reply_text(
                "❌ **File Not Found**\n\nLink expired.",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔍 Search", url=Config.WEBSITE_URL)]])
            )
            return
        
        try:
            pm = await message.reply_text(
                f"⏳ **Sending...**\n\n"
                f"📁 {fi['file_name']}\n"
                f"📊 {fi['quality']}\n"
                f"📦 {format_size(fi['file_size'])}"
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
                    f"⚠️ Auto-delete in {Config.AUTO_DELETE_TIME//60} min"
                )
                
                logger.info(f"✅ File delivered: {fi['title']} → {uid}")
                
                # Auto-delete
                if Config.AUTO_DELETE_TIME > 0:
                    await asyncio.sleep(Config.AUTO_DELETE_TIME)
                    try:
                        await sent.delete()
                        await sm.edit_text("🗑️ Auto-deleted")
                    except:
                        pass
        except Exception as e:
            logger.error(f"File error: {e}")
            await message.reply_text("❌ Error")
        
        return
    
    # Normal start
    await message.reply_text(
        f"🎬 **Welcome to SK4FiLM!**\n\n"
        f"📌 File delivery bot\n\n"
        f"**Usage:**\n"
        f"1. Visit website\n"
        f"2. Search movie\n"
        f"3. Select quality\n"
        f"4. Get file",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🌐 Website", url=Config.WEBSITE_URL)]])
    )

@bot.on_message(filters.text & filters.private & ~filters.command(['start', 'stats']))
async def text_handler(client, message):
    await message.reply_text(
        f"👋 Hi!\n\n🤖 Use website to search.",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🌐 Website", url=Config.WEBSITE_URL)]])
    )

@bot.on_message(filters.command("stats") & filters.user(Config.ADMIN_IDS))
async def stats_handler(client, message):
    await message.reply_text(
        f"📊 **Stats**\n\n"
        f"🎬 Movies: `{len(movie_db['home_movies'])}`\n"
        f"📁 Files: `{len(file_registry)}`\n"
        f"🖼️ Posters:\n"
        f"  • OMDB: `{movie_db['stats']['omdb']}`\n"
        f"  • TMDB: `{movie_db['stats']['tmdb_hq']}`\n"
        f"  • Custom: `{movie_db['stats']['custom']}`\n"
        f"🤖 Bot: `{'✅' if bot_started else '❌'}`"
    )

# ==================== API ROUTES ====================
@app.after_request
async def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

@app.route('/')
async def home():
    return jsonify({
        "status": "healthy" if bot_started else "starting",
        "service": "SK4FiLM - With Bot & File Sharing",
        "bot": f"@{Config.BOT_USERNAME}",
        "stats": movie_db['stats'],
        "home_movies": len(movie_db['home_movies']),
        "total_movies": len(movie_db['all_movies']),
        "files": len(file_registry)
    })

@app.route('/api/movies')
async def api_movies():
    """Movies API"""
    try:
        page = request.args.get('page')
        
        if not bot_started:
            return jsonify({"status": "starting"}), 503
        
        if page:
            # With pagination
            page = int(page)
            limit = int(request.args.get('limit', 8))
            
            total = len(movie_db['all_movies'])
            offset = (page - 1) * limit
            paginated = movie_db['all_movies'][offset:offset + limit]
            total_pages = math.ceil(total / limit) if total > 0 else 1
            
            return jsonify({
                "status": "success",
                "movies": paginated,
                "pagination": {
                    "current_page": page,
                    "total_pages": total_pages,
                    "total_movies": total,
                    "per_page": limit,
                    "has_next": page < total_pages,
                    "has_previous": page > 1
                },
                "bot_username": Config.BOT_USERNAME
            })
        else:
            # Homepage - no pagination
            return jsonify({
                "status": "success",
                "movies": movie_db['home_movies'],
                "total_movies": len(movie_db['home_movies']),
                "bot_username": Config.BOT_USERNAME
            })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/search')
async def api_search():
    """Search API with file support"""
    try:
        query = request.args.get('query', '').strip()
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 12))
        
        if not query:
            return jsonify({"status": "error", "message": "Query required"}), 400
        
        if not bot_started:
            return jsonify({"status": "error", "message": "Service starting"}), 503
        
        result = await search_with_files_and_pagination(query, limit, page)
        
        return jsonify({
            "status": "success",
            "query": query,
            "results": result["results"],
            "pagination": result["pagination"],
            "bot_username": Config.BOT_USERNAME
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/poster')
async def proxy_hq_poster():
    """Poster proxy"""
    try:
        poster_url = request.args.get('url', '').strip()
        
        if not poster_url or not poster_url.startswith('http'):
            title = request.args.get('title', 'Movie')
            return create_enhanced_poster_svg(title)
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'image/*',
            'Referer': 'https://www.themoviedb.org/'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(poster_url, headers=headers, timeout=10) as response:
                if response.status == 200:
                    image_data = await response.read()
                    content_type = response.headers.get('content-type', 'image/jpeg')
                    
                    return Response(
                        image_data,
                        mimetype=content_type,
                        headers={'Cache-Control': 'public, max-age=7200', 'Access-Control-Allow-Origin': '*'}
                    )
        
        return create_enhanced_poster_svg("Error")
        
    except:
        return create_enhanced_poster_svg("Error")

@app.route('/api/enhanced_poster')
async def enhanced_poster_api():
    title = request.args.get('title', 'Movie')
    return create_enhanced_poster_svg(title)

def create_enhanced_poster_svg(title):
    """Enhanced custom poster"""
    display_title = title[:20] + "..." if len(title) > 20 else title
    
    themes = [
        {'bg': ['#667eea', '#764ba2'], 'text': '#ffffff', 'accent': '#f093fb'},
        {'bg': ['#f093fb', '#f5576c'], 'text': '#ffffff', 'accent': '#4facfe'},
        {'bg': ['#43e97b', '#38f9d7'], 'text': '#2c3e50', 'accent': '#667eea'},
        {'bg': ['#fa709a', '#fee140'], 'text': '#2c3e50', 'accent': '#667eea'},
        {'bg': ['#a8edea', '#fed6e3'], 'text': '#2c3e50', 'accent': '#d299c2'}
    ]
    
    theme = themes[hash(title) % len(themes)]
    
    svg = f'''<svg width="300" height="450" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:{theme['bg'][0]}"/>
                <stop offset="100%" style="stop-color:{theme['bg'][1]}"/>
            </linearGradient>
            <filter id="shadow">
                <feDropShadow dx="2" dy="2" stdDeviation="4" flood-opacity="0.5"/>
            </filter>
        </defs>
        
        <rect width="100%" height="100%" fill="url(#bg)" rx="18"/>
        <rect x="25" y="60" width="250" height="320" fill="rgba(255,255,255,0.1)" rx="15" stroke="rgba(255,255,255,0.3)" stroke-width="1"/>
        <circle cx="150" cy="180" r="45" fill="rgba(255,255,255,0.15)" stroke="rgba(255,255,255,0.3)" stroke-width="2"/>
        <text x="50%" y="195" text-anchor="middle" fill="{theme['text']}" font-size="44" filter="url(#shadow)">🎬</text>
        <text x="50%" y="250" text-anchor="middle" fill="{theme['text']}" font-size="16" font-weight="bold" filter="url(#shadow)">{html.escape(display_title)}</text>
        <text x="50%" y="410" text-anchor="middle" fill="{theme['text']}" font-size="16" font-weight="700" filter="url(#shadow)">SK4FiLM</text>
    </svg>'''
    
    return Response(svg, mimetype='image/svg+xml', headers={'Cache-Control': 'public, max-age=1800', 'Access-Control-Allow-Origin': '*'})

# ==================== INITIALIZATION ====================
async def init_telegram_hq():
    """Initialize Telegram clients"""
    global User, bot, bot_started
    
    try:
        logger.info("🔄 Initializing Telegram...")
        
        # User client
        session_name = f"sk4film_{uuid.uuid4().hex[:8]}"
        User = Client(
            session_name,
            api_id=Config.API_ID,
            api_hash=Config.API_HASH,
            session_string=Config.USER_SESSION_STRING,
            workdir="/tmp",
            sleep_threshold=60
        )
        
        await User.start()
        me = await User.get_me()
        logger.info(f"✅ User: {me.first_name}")
        
        # Bot client
        bot = Client(
            "sk4film_bot",
            api_id=Config.API_ID,
            api_hash=Config.API_HASH,
            bot_token=Config.BOT_TOKEN,
            workdir="/tmp",
            sleep_threshold=60
        )
        
        await bot.start()
        bot_me = await bot.get_me()
        logger.info(f"✅ Bot: @{bot_me.username}")
        
        # Verify channels
        working = []
        for cid in Config.TEXT_CHANNEL_IDS:
            try:
                chat = await User.get_chat(cid)
                logger.info(f"✅ Channel: {chat.title}")
                working.append(cid)
            except Exception as e:
                logger.error(f"❌ Channel {cid}: {e}")
        
        if working:
            Config.TEXT_CHANNEL_IDS = working
            bot_started = True
            
            # Initial load
            await background_update_hq()
            
            # Start auto-updates
            await start_hidden_hq_update()
            
            logger.info(f"🎉 System ready! {len(movie_db['home_movies'])} movies")
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Init error: {e}")
        return False

async def run_hq_server():
    try:
        logger.info("=" * 70)
        logger.info("🚀 SK4FiLM - Complete System")
        logger.info("=" * 70)
        logger.info(f"🤖 Bot: @{Config.BOT_USERNAME}")
        logger.info(f"🌐 Website: {Config.WEBSITE_URL}")
        logger.info(f"📡 Backend: {Config.BACKEND_URL}")
        logger.info("=" * 70)
        
        success = await init_telegram_hq()
        
        if success:
            logger.info("🎉 All systems operational!")
        
        config = HyperConfig()
        config.bind = [f"0.0.0.0:{Config.WEB_SERVER_PORT}"]
        config.loglevel = "warning"
        
        await serve(app, config)
        
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        if auto_update_task:
            auto_update_task.cancel()
        if User:
            try:
                await User.stop()
            except:
                pass
        if bot:
            try:
                await bot.stop()
            except:
                pass

if __name__ == "__main__":
    asyncio.run(run_hq_server())
