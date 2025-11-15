import asyncio
import os
import logging
from datetime import datetime
from pyrogram import Client, filters
from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from quart import Quart, jsonify, request, Response
from hypercorn.asyncio import serve
from hypercorn.config import Config as HyperConfig
from motor.motor_asyncio import AsyncIOMotorClient
import html
import re
import math
import aiohttp
import urllib.parse

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger('asyncio').setLevel(logging.WARNING)

# Config
class Config:
    API_ID = int(os.environ.get("API_ID", "0"))
    API_HASH = os.environ.get("API_HASH", "")
    USER_SESSION_STRING = os.environ.get("USER_SESSION_STRING", "")
    BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
    MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
    
    TEXT_CHANNEL_IDS = [-1001891090100, -1002024811395]
    FILE_CHANNEL_ID = -1001768249569
    FORCE_SUB_CHANNEL = -1002555323872
    
    WEBSITE_URL = os.environ.get("WEBSITE_URL", "https://sk4film.vercel.app")
    BOT_USERNAME = os.environ.get("BOT_USERNAME", "sk4filmbot")
    ADMIN_IDS = [int(x) for x in os.environ.get("ADMIN_IDS", "123456789").split(",")]
    AUTO_DELETE_TIME = int(os.environ.get("AUTO_DELETE_TIME", "300"))
    WEB_SERVER_PORT = int(os.environ.get("PORT", 8000))
    BACKEND_URL = os.environ.get("BACKEND_URL", "https://sk4film.koyeb.app")
    
    OMDB_KEYS = ["8265bd1c", "b9bd48a6"]
    TMDB_KEYS = ["e547e17d4e91f3e62a571655cd1ccaff"]
    AUTO_INDEX_INTERVAL = 600

# Quart
app = Quart(__name__)

@app.after_request
async def add_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

# MongoDB
mongo_client = None
db = None
posts_col = None
files_col = None

async def init_mongodb():
    global mongo_client, db, posts_col, files_col
    try:
        logger.info("🔌 MongoDB...")
        mongo_client = AsyncIOMotorClient(Config.MONGODB_URI, serverSelectionTimeoutMS=10000)
        await mongo_client.admin.command('ping')
        
        db = mongo_client.sk4film
        posts_col = db.posts
        files_col = db.files
        
        await posts_col.create_index([("title", "text"), ("content", "text")])
        await posts_col.create_index([("date", -1)])
        await files_col.create_index([("title", "text")])
        await files_col.create_index([("file_id", 1)], unique=True)
        
        logger.info("✅ MongoDB OK")
        return True
    except Exception as e:
        logger.error(f"❌ MongoDB: {e}")
        return False

# Global
User = None
bot = None
bot_started = False
movie_db = {'poster_cache': {}, 'stats': {'omdb': 0, 'tmdb': 0, 'custom': 0}}
auto_index_task = None

# Helpers
def normalize_title(title):
    """Normalize title for better matching"""
    if not title:
        return ""
    # Remove special chars, convert to lowercase
    normalized = title.lower().strip()
    # Remove year (4 digits)
    normalized = re.sub(r'\b(19|20)\d{2}\b', '', normalized)
    # Remove quality tags
    normalized = re.sub(r'\b(480p|720p|1080p|2160p|4k|hd|fhd|uhd|hevc|x264|x265|h264|h265|bluray|webrip|hdrip|web-dl|hdtv)\b', '', normalized, flags=re.IGNORECASE)
    # Remove extra spaces
    normalized = ' '.join(normalized.split()).strip()
    return normalized

def extract_title_smart(text):
    if not text or len(text) < 15:
        return None
    try:
        clean = re.sub(r'[^\w\s\(\)\-\.\n]', ' ', text)
        first = clean.split('\n')[0].strip()
        patterns = [
            r'🎬\s*([^-\n]{4,50})',
            r'^([^(]{4,50})\s*\(\d{4}\)',
            r'^([^-]{4,50})\s*-',
            r'^([A-Z][a-z]+(?:\s+[A-Za-z]+){1,5})'
        ]
        for p in patterns:
            m = re.search(p, first)
            if m:
                t = m.group(1).strip()
                if 4 <= len(t) <= 50:
                    return t
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
    return "480p HEVC" if is_hevc else "480p"

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
    return names.get(cid, "Channel")

def is_new(date):
    try:
        if isinstance(date, str):
            date = datetime.fromisoformat(date.replace('Z', '+00:00'))
        hours = (datetime.now() - date.replace(tzinfo=None)).seconds / 3600
        return hours <= 24
    except:
        return False

async def check_force_sub(user_id):
    try:
        member = await bot.get_chat_member(Config.FORCE_SUB_CHANNEL, user_id)
        return member.status in ["member", "administrator", "creator"]
    except:
        return True

# INDEXING
async def index_all_files():
    if not User or not bot_started or posts_col is None or files_col is None:
        logger.warning("⚠️ Cannot index")
        return
    
    logger.info("📥 Starting indexing...")
    indexed_posts = 0
    indexed_files = 0
    
    try:
        # Index POSTS
        for channel_id in Config.TEXT_CHANNEL_IDS:
            try:
                count = 0
                logger.info(f"📝 Indexing {channel_name(channel_id)}...")
                async for msg in User.get_chat_history(channel_id, limit=100):
                    if msg.text:
                        title = extract_title_smart(msg.text)
                        if title:
                            try:
                                await posts_col.update_one(
                                    {'channel_id': channel_id, 'message_id': msg.id},
                                    {'$set': {
                                        'title': title,
                                        'normalized_title': normalize_title(title),
                                        'content': msg.text,
                                        'channel_id': channel_id,
                                        'channel_name': channel_name(channel_id),
                                        'message_id': msg.id,
                                        'date': msg.date,
                                        'is_new': is_new(msg.date) if msg.date else False,
                                        'indexed_at': datetime.now()
                                    }},
                                    upsert=True
                                )
                                count += 1
                                indexed_posts += 1
                            except:
                                pass
                logger.info(f"  ✓ Indexed {count} posts")
            except Exception as e:
                logger.error(f"  ✗ Error: {e}")
        
        # Index FILES
        try:
            count = 0
            logger.info(f"📁 Indexing {channel_name(Config.FILE_CHANNEL_ID)}...")
            async for msg in User.get_chat_history(Config.FILE_CHANNEL_ID, limit=200):
                if msg.document or msg.video:
                    title = extract_title_from_file(msg)
                    if title:
                        file_id = msg.document.file_id if msg.document else msg.video.file_id
                        file_size = msg.document.file_size if msg.document else (msg.video.file_size if msg.video else 0)
                        file_name = msg.document.file_name if msg.document else (msg.video.file_name if msg.video else 'video.mp4')
                        quality = detect_quality(file_name)
                        
                        try:
                            await files_col.update_one(
                                {'file_id': file_id},
                                {'$set': {
                                    'title': title,
                                    'normalized_title': normalize_title(title),
                                    'file_id': file_id,
                                    'channel_id': Config.FILE_CHANNEL_ID,
                                    'message_id': msg.id,
                                    'quality': quality,
                                    'file_size': file_size,
                                    'file_name': file_name,
                                    'caption': msg.caption or '',
                                    'date': msg.date,
                                    'indexed_at': datetime.now()
                                }},
                                upsert=True
                            )
                            count += 1
                            indexed_files += 1
                        except:
                            pass
            logger.info(f"  ✓ Indexed {count} files")
        except Exception as e:
            logger.error(f"  ✗ Error: {e}")
        
        logger.info(f"✅ Indexing complete: {indexed_posts} posts, {indexed_files} files")
        
    except Exception as e:
        logger.error(f"❌ Critical error: {e}")

async def auto_index_loop():
    while True:
        try:
            await asyncio.sleep(Config.AUTO_INDEX_INTERVAL)
            logger.info("🔄 Auto-index triggered")
            await index_all_files()
        except Exception as e:
            logger.error(f"Loop error: {e}")
            await asyncio.sleep(60)

# Posters
async def get_poster_multi(title, session):
    ck = title.lower().strip()
    if ck in movie_db['poster_cache']:
        c, ct = movie_db['poster_cache'][ck]
        if (datetime.now() - ct).seconds < 600:
            return c
    
    # OMDB
    for k in Config.OMDB_KEYS:
        try:
            async with session.get(f"http://www.omdbapi.com/?t={urllib.parse.quote(title)}&apikey={k}", timeout=5) as r:
                if r.status == 200:
                    d = await r.json()
                    if d.get('Response') == 'True' and d.get('Poster') != 'N/A':
                        res = {'poster_url': d['Poster'].replace('http://', 'https://'), 'source': 'OMDB', 'success': True}
                        movie_db['poster_cache'][ck] = (res, datetime.now())
                        movie_db['stats']['omdb'] += 1
                        return res
        except:
            continue
    
    # TMDB
    for k in Config.TMDB_KEYS:
        try:
            async with session.get("https://api.themoviedb.org/3/search/movie", params={'api_key': k, 'query': title}, timeout=5) as r:
                if r.status == 200:
                    d = await r.json()
                    if d.get('results'):
                        m = d['results'][0]
                        p = m.get('poster_path')
                        if p:
                            res = {'poster_url': f"https://image.tmdb.org/t/p/w780{p}", 'source': 'TMDB', 'success': True}
                            movie_db['poster_cache'][ck] = (res, datetime.now())
                            movie_db['stats']['tmdb'] += 1
                            return res
        except:
            continue
    
    # Custom
    movie_db['stats']['custom'] += 1
    res = {'poster_url': f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(title)}", 'source': 'CUSTOM', 'success': True}
    movie_db['poster_cache'][ck] = (res, datetime.now())
    return res

# IMPROVED SEARCH - Merge Posts + Files by Normalized Title
async def search_movies(query, limit=12, page=1):
    """
    Search and merge posts with files by normalized title
    """
    if posts_col is None or files_col is None:
        return {
            'results': [],
            'pagination': {
                'current_page': 1,
                'total_pages': 1,
                'total_results': 0,
                'per_page': limit,
                'has_next': False,
                'has_previous': False
            }
        }
    
    offset = (page - 1) * limit
    logger.info(f"🔍 Search: '{query}' Page {page}")
    
    # 1. Search POSTS (all channels)
    posts_dict = {}
    try:
        cursor = posts_col.find({'$text': {'$search': query}}).sort('date', -1).limit(100)
        async for doc in cursor:
            norm_title = doc.get('normalized_title', normalize_title(doc['title']))
            
            if norm_title not in posts_dict:
                posts_dict[norm_title] = {
                    'title': doc['title'],  # Original title from post
                    'content': format_post(doc['content']),
                    'channel': doc['channel_name'],
                    'date': doc['date'].isoformat() if isinstance(doc['date'], datetime) else doc['date'],
                    'is_new': doc.get('is_new', False),
                    'has_file': False,
                    'quality_options': {},
                    'normalized_title': norm_title
                }
        logger.info(f"  ✓ Posts: {len(posts_dict)}")
    except Exception as e:
        logger.error(f"  ✗ Posts error: {e}")
    
    # 2. Search FILES
    files_dict = {}
    try:
        cursor = files_col.find({'$text': {'$search': query}}).limit(300)
        
        async for doc in cursor:
            norm_title = doc.get('normalized_title', normalize_title(doc['title']))
            quality = doc['quality']
            
            if norm_title not in files_dict:
                files_dict[norm_title] = {
                    'title': doc['title'],
                    'quality_options': {},
                    'date': doc['date'].isoformat() if isinstance(doc['date'], datetime) else doc['date'],
                    'normalized_title': norm_title
                }
            
            # Add quality option
            files_dict[norm_title]['quality_options'][quality] = {
                'file_id': doc['file_id'],
                'file_size': doc['file_size'],
                'file_name': doc['file_name']
            }
        
        logger.info(f"  ✓ Files: {len(files_dict)} unique titles")
    except Exception as e:
        logger.error(f"  ✗ Files error: {e}")
    
    # 3. MERGE posts with files by normalized title
    merged_results = {}
    
    # Start with posts
    for norm_title, post_data in posts_dict.items():
        merged_results[norm_title] = post_data
    
    # Merge files into posts
    for norm_title, file_data in files_dict.items():
        if norm_title in merged_results:
            # Post exists - add file info
            merged_results[norm_title]['has_file'] = True
            merged_results[norm_title]['quality_options'] = file_data['quality_options']
            logger.info(f"  ✓ Merged: {merged_results[norm_title]['title']} ({len(file_data['quality_options'])} qualities)")
        else:
            # No post - create file-only entry
            merged_results[norm_title] = {
                'title': file_data['title'],
                'content': f"<p style='text-align:center;'>{file_data['title']}</p>",
                'channel': 'SK4FiLM',
                'date': file_data['date'],
                'is_new': False,
                'has_file': True,
                'quality_options': file_data['quality_options'],
                'normalized_title': norm_title
            }
            logger.info(f"  ✓ File-only: {file_data['title']} ({len(file_data['quality_options'])} qualities)")
    
    # 4. Sort and paginate
    results_list = list(merged_results.values())
    results_list.sort(key=lambda x: (x['has_file'], x['date']), reverse=True)
    
    total = len(results_list)
    paginated = results_list[offset:offset + limit]
    
    logger.info(f"✅ Total: {total} | Page {page}: {len(paginated)}")
    
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

async def get_home_movies():
    if posts_col is None:
        return []
    
    logger.info("🏠 Loading home movies...")
    movies = []
    seen = set()
    
    try:
        cursor = posts_col.find().sort('date', -1).limit(50)
        async for doc in cursor:
            tk = doc['title'].lower().strip()
            if tk in seen:
                continue
            seen.add(tk)
            movies.append({
                'title': doc['title'],
                'date': doc['date'].isoformat() if isinstance(doc['date'], datetime) else doc['date'],
                'is_new': doc.get('is_new', False)
            })
            if len(movies) >= 24:
                break
    except:
        pass
    
    logger.info(f"🎨 Fetching posters...")
    async with aiohttp.ClientSession() as s:
        for i in range(0, len(movies), 5):
            batch = movies[i:i + 5]
            posters = await asyncio.gather(*[get_poster_multi(m['title'], s) for m in batch], return_exceptions=True)
            for m, p in zip(batch, posters):
                if isinstance(p, dict) and p.get('success'):
                    m.update({'poster_url': p['poster_url'], 'poster_source': p['source'], 'has_poster': True})
                else:
                    m['poster_url'] = f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(m['title'])}"
                    m['has_poster'] = True
            await asyncio.sleep(0.2)
    
    logger.info(f"✅ {len(movies)} movies ready")
    return movies

# API
@app.route('/')
async def root():
    tp = await posts_col.count_documents({}) if posts_col is not None else 0
    tf = await files_col.count_documents({}) if files_col is not None else 0
    return jsonify({
        'status': 'healthy',
        'service': 'SK4FiLM v2.0',
        'database': {'posts': tp, 'files': tf},
        'bot_status': 'online' if bot_started else 'starting'
    })

@app.route('/health')
async def health():
    return jsonify({'status': 'ok' if bot_started else 'starting'})

@app.route('/api/movies')
async def api_movies():
    try:
        if not bot_started:
            return jsonify({'status': 'error', 'message': 'Starting...'}), 503
        movies = await get_home_movies()
        return jsonify({
            'status': 'success',
            'movies': movies,
            'total': len(movies),
            'bot_username': Config.BOT_USERNAME
        })
    except Exception as e:
        logger.error(f"❌ Movies: {e}")
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
        
        result = await search_movies(q, l, p)
        return jsonify({
            'status': 'success',
            'query': q,
            'results': result['results'],
            'pagination': result['pagination'],
            'bot_username': Config.BOT_USERNAME
        })
    except Exception as e:
        logger.error(f"❌ Search: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/poster')
async def api_poster():
    url = request.args.get('url', '').strip()
    if url and url.startswith('http'):
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'}) as r:
                    if r.status == 200:
                        return Response(await r.read(), mimetype=r.headers.get('content-type', 'image/jpeg'), headers={'Cache-Control': 'public, max-age=7200'})
        except:
            pass
    
    t = request.args.get('title', 'Movie')
    d = t[:18] + "..." if len(t) > 18 else t
    colors = [('#667eea','#764ba2'),('#f093fb','#f5576c'),('#4facfe','#00f2fe'),('#43e97b','#38f9d7')]
    c = colors[hash(t) % len(colors)]
    svg = f'''<svg width="300" height="450" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" style="stop-color:{c[0]}"/><stop offset="100%" style="stop-color:{c[1]}"/></linearGradient></defs><rect width="100%" height="100%" fill="url(#bg)" rx="20"/><circle cx="150" cy="180" r="50" fill="rgba(255,255,255,0.2)"/><text x="50%" y="200" text-anchor="middle" fill="#fff" font-size="50">🎬</text><text x="50%" y="270" text-anchor="middle" fill="#fff" font-size="18" font-weight="bold">{html.escape(d)}</text><rect x="50" y="380" width="200" height="40" rx="20" fill="rgba(0,0,0,0.3)"/><text x="50%" y="405" text-anchor="middle" fill="#fff" font-size="18" font-weight="700">SK4FiLM</text></svg>'''
    return Response(svg, mimetype='image/svg+xml', headers={'Cache-Control': 'public, max-age=3600'})

# BOT
async def setup_bot():
    @bot.on_message(filters.command("start") & filters.private)
    async def start_handler(client, message):
        uid = message.from_user.id
        
        if len(message.command) > 1:
            fid = message.command[1]
            logger.info(f"📥 File request: {uid} | {fid}")
            
            if not await check_force_sub(uid):
                try:
                    ch = await bot.get_chat(Config.FORCE_SUB_CHANNEL)
                    lk = f"https://t.me/{ch.username}" if ch.username else f"https://t.me/c/{str(Config.FORCE_SUB_CHANNEL)[4:]}/1"
                except:
                    lk = f"https://t.me/c/{str(Config.FORCE_SUB_CHANNEL)[4:]}/1"
                await message.reply_text("⚠️ Join channel first", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("📢 Join Channel", url=lk)]]))
                return
            
            fd = await files_col.find_one({'file_id': fid}) if files_col is not None else None
            
            if not fd:
                logger.error(f"❌ File not found: {fid}")
                await message.reply_text("❌ File Not Found", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔍 Search", url=Config.WEBSITE_URL)]]))
                return
            
            try:
                pm = await message.reply_text(f"⏳ Sending file...\n\n📁 {fd['file_name']}\n📊 {fd['quality']}\n📦 {format_size(fd['file_size'])}")
                
                if User:
                    fm = await User.get_messages(fd['channel_id'], fd['message_id'])
                    sent = await fm.copy(uid)
                    await pm.delete()
                    
                    sm = await message.reply_text(f"✅ File sent!\n\n🎬 {fd['title']}\n📊 {fd['quality']}\n📦 {format_size(fd['file_size'])}\n\n⚠️ File will auto-delete in {Config.AUTO_DELETE_TIME//60} minutes")
                    
                    logger.info(f"✅ Sent: {fd['title']} → {uid}")
                    
                    if Config.AUTO_DELETE_TIME > 0:
                        await asyncio.sleep(Config.AUTO_DELETE_TIME)
                        try:
                            await sent.delete()
                            await sm.edit_text("🗑️ File deleted")
                        except:
                            pass
            except Exception as e:
                logger.error(f"❌ Send error: {e}")
                await message.reply_text("❌ Error sending file")
            return
        
        await message.reply_text("🎬 SK4FiLM\n\nSearch & download movies", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🌐 Open Website", url=Config.WEBSITE_URL)]]))
    
    @bot.on_message(filters.text & filters.private & ~filters.command(['start', 'stats', 'index']))
    async def text_handler(client, message):
        await message.reply_text("👋 Use website to search", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🌐 Go to Website", url=Config.WEBSITE_URL)]]))
    
    @bot.on_message(filters.command("index") & filters.user(Config.ADMIN_IDS))
    async def index_handler(client, message):
        await message.reply_text("🔄 Starting indexing...")
        await index_all_files()
        tp = await posts_col.count_documents({}) if posts_col is not None else 0
        tf = await files_col.count_documents({}) if files_col is not None else 0
        await message.reply_text(f"✅ Indexing complete\n\n📝 Posts: {tp}\n📁 Files: {tf}")
    
    @bot.on_message(filters.command("stats") & filters.user(Config.ADMIN_IDS))
    async def stats_handler(client, message):
        tp = await posts_col.count_documents({}) if posts_col is not None else 0
        tf = await files_col.count_documents({}) if files_col is not None else 0
        await message.reply_text(f"📊 Statistics\n\n📝 Posts: {tp}\n📁 Files: {tf}\n\n🖼️ Posters:\nOMDB: {movie_db['stats']['omdb']}\nTMDB: {movie_db['stats']['tmdb']}\nCustom: {movie_db['stats']['custom']}")

# INIT
async def init():
    global User, bot, bot_started, auto_index_task
    try:
        logger.info("🔄 Initializing...")
        await init_mongodb()
        
        User = Client("sk4film_user", api_id=Config.API_ID, api_hash=Config.API_HASH, session_string=Config.USER_SESSION_STRING, workdir="/tmp", sleep_threshold=60)
        bot = Client("sk4film_bot", api_id=Config.API_ID, api_hash=Config.API_HASH, bot_token=Config.BOT_TOKEN, workdir="/tmp", sleep_threshold=60)
        
        await User.start()
        await bot.start()
        await setup_bot()
        
        me = await bot.get_me()
        logger.info(f"✅ Bot: @{me.username}")
        bot_started = True
        
        await index_all_files()
        auto_index_task = asyncio.create_task(auto_index_loop())
        
        return True
    except Exception as e:
        logger.error(f"❌ Init error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    logger.info("="*70)
    logger.info("🚀 SK4FiLM v2.0 - Complete System")
    logger.info("="*70)
    await init()
    cfg = HyperConfig()
    cfg.bind = [f"0.0.0.0:{Config.WEB_SERVER_PORT}"]
    cfg.loglevel = "warning"
    logger.info(f"🌐 Server: Port {Config.WEB_SERVER_PORT}")
    await serve(app, cfg)

if __name__ == "__main__":
    asyncio.run(main())
