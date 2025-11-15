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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger('asyncio').setLevel(logging.WARNING)

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

app = Quart(__name__)

@app.after_request
async def add_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

mongo_client = None
db = None
posts_col = None
files_col = None

async def init_mongodb():
    global mongo_client, db, posts_col, files_col
    try:
        logger.info("🔌 Connecting to MongoDB...")
        mongo_client = AsyncIOMotorClient(Config.MONGODB_URI, serverSelectionTimeoutMS=10000)
        await mongo_client.admin.command('ping')
        
        db = mongo_client.sk4film
        posts_col = db.posts
        files_col = db.files
        
        await posts_col.create_index([("title", "text"), ("content", "text")])
        await posts_col.create_index([("date", -1)])
        await files_col.create_index([("title", "text")])
        await files_col.create_index([("file_id", 1)], unique=True)
        
        logger.info("✅ MongoDB connected successfully")
        return True
    except Exception as e:
        logger.error(f"❌ MongoDB connection failed: {e}")
        return False

User = None
bot = None
bot_started = False
movie_db = {'poster_cache': {}, 'stats': {'omdb': 0, 'tmdb': 0, 'custom': 0}}
auto_index_task = None

def normalize_title(title):
    if not title:
        return ""
    normalized = title.lower().strip()
    normalized = re.sub(r'\b(19|20)\d{2}\b', '', normalized)
    normalized = re.sub(r'\b(480p|720p|1080p|2160p|4k|hd|fhd|uhd|hevc|x264|x265|h264|h265|bluray|webrip|hdrip|web-dl|hdtv)\b', '', normalized, flags=re.IGNORECASE)
    normalized = ' '.join(normalized.split()).strip()
    return normalized

def extract_title_smart(text):
    if not text or len(text) < 15:
        return None
    try:
        clean = re.sub(r'[^\w\s\(\)\-\.\n]', ' ', text)
        first = clean.split('\n')[0].strip()
        patterns = [r'🎬\s*([^-\n]{4,50})', r'^([^(]{4,50})\s*\(\d{4}\)', r'^([^-]{4,50})\s*-', r'^([A-Z][a-z]+(?:\s+[A-Za-z]+){1,5})']
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
    names = {-1001891090100: "SK4FiLM Main", -1002024811395: "SK4FiLM Updates", -1001768249569: "SK4FiLM Files"}
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

async def index_all_files():
    if not User or not bot_started or posts_col is None or files_col is None:
        logger.warning("⚠️ Cannot index - components not ready")
        return
    
    logger.info("📥 Starting indexing process...")
    indexed_posts = 0
    indexed_files = 0
    
    try:
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
                logger.info(f"  ✓ Indexed {count} posts from {channel_name(channel_id)}")
            except Exception as e:
                logger.error(f"  ✗ Error indexing {channel_name(channel_id)}: {e}")
        
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
            logger.error(f"  ✗ Error indexing files: {e}")
        
        logger.info(f"✅ Indexing complete: {indexed_posts} posts, {indexed_files} files")
        
    except Exception as e:
        logger.error(f"❌ Indexing error: {e}")

async def auto_index_loop():
    while True:
        try:
            await asyncio.sleep(Config.AUTO_INDEX_INTERVAL)
            logger.info("🔄 Auto-index triggered")
            await index_all_files()
        except Exception as e:
            logger.error(f"Auto-index loop error: {e}")
            await asyncio.sleep(60)

async def get_poster_multi(title, session):
    ck = title.lower().strip()
    if ck in movie_db['poster_cache']:
        c, ct = movie_db['poster_cache'][ck]
        if (datetime.now() - ct).seconds < 600:
            return c
    
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
    
    movie_db['stats']['custom'] += 1
    res = {'poster_url': f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(title)}", 'source': 'CUSTOM', 'success': True}
    movie_db['poster_cache'][ck] = (res, datetime.now())
    return res

async def direct_search_channels(query, limit=50):
    if not User:
        return {'posts': [], 'files': []}
    
    logger.info(f"🔍 Direct search in channels: '{query}'")
    query_lower = query.lower()
    posts_found = []
    files_found = []
    
    try:
        for channel_id in Config.TEXT_CHANNEL_IDS:
            try:
                count = 0
                async for msg in User.search_messages(channel_id, query=query, limit=limit):
                    if msg.text:
                        title = extract_title_smart(msg.text)
                        if title and query_lower in title.lower():
                            posts_found.append({
                                'title': title,
                                'normalized_title': normalize_title(title),
                                'content': msg.text,
                                'channel_name': channel_name(channel_id),
                                'date': msg.date
                            })
                            count += 1
                logger.info(f"  ✓ {channel_name(channel_id)}: {count} posts")
            except Exception as e:
                logger.error(f"  ✗ {channel_name(channel_id)}: {e}")
        
        try:
            count = 0
            async for msg in User.search_messages(Config.FILE_CHANNEL_ID, query=query, limit=limit):
                if msg.document or msg.video:
                    title = extract_title_from_file(msg)
                    if title and query_lower in title.lower():
                        file_id = msg.document.file_id if msg.document else msg.video.file_id
                        file_size = msg.document.file_size if msg.document else (msg.video.file_size if msg.video else 0)
                        file_name = msg.document.file_name if msg.document else (msg.video.file_name if msg.video else 'video.mp4')
                        quality = detect_quality(file_name)
                        
                        files_found.append({
                            'title': title,
                            'normalized_title': normalize_title(title),
                            'file_id': file_id,
                            'quality': quality,
                            'file_size': file_size,
                            'file_name': file_name,
                            'message_id': msg.id,
                            'channel_id': Config.FILE_CHANNEL_ID,
                            'date': msg.date
                        })
                        count += 1
            logger.info(f"  ✓ Files: {count}")
        except Exception as e:
            logger.error(f"  ✗ Files: {e}")
        
        logger.info(f"✅ Direct search: {len(posts_found)} posts, {len(files_found)} files")
        
    except Exception as e:
        logger.error(f"❌ Direct search error: {e}")
    
    return {'posts': posts_found, 'files': files_found}

async def search_movies(query, limit=12, page=1):
    offset = (page - 1) * limit
    logger.info(f"🔍 Search query: '{query}' | Page: {page}")
    
    posts_dict = {}
    files_dict = {}
    
    if posts_col is not None and files_col is not None:
        try:
            cursor = posts_col.find({'$text': {'$search': query}}).sort('date', -1).limit(200)
            async for doc in cursor:
                norm_title = doc.get('normalized_title', normalize_title(doc['title']))
                
                if norm_title not in posts_dict:
                    posts_dict[norm_title] = {
                        'title': doc['title'],
                        'content': format_post(doc['content']),
                        'channel': doc['channel_name'],
                        'date': doc['date'].isoformat() if isinstance(doc['date'], datetime) else doc['date'],
                        'is_new': doc.get('is_new', False),
                        'has_file': False,
                        'quality_options': {}
                    }
            
            cursor = files_col.find({'$text': {'$search': query}}).limit(300)
            async for doc in cursor:
                norm_title = doc.get('normalized_title', normalize_title(doc['title']))
                quality = doc['quality']
                
                if norm_title not in files_dict:
                    files_dict[norm_title] = {
                        'title': doc['title'], 
                        'quality_options': {}, 
                        'date': doc['date'].isoformat() if isinstance(doc['date'], datetime) else doc['date']
                    }
                
                files_dict[norm_title]['quality_options'][quality] = {
                    'file_id': doc['file_id'],
                    'file_size': doc['file_size'],
                    'file_name': doc['file_name'],
                    'message_id': doc.get('message_id'),
                    'channel_id': doc.get('channel_id', Config.FILE_CHANNEL_ID)
                }
            
            logger.info(f"  ✓ MongoDB: {len(posts_dict)} unique posts, {len(files_dict)} files")
        except Exception as e:
            logger.error(f"  ✗ MongoDB error: {e}")
    
    if len(posts_dict) == 0 and len(files_dict) == 0:
        logger.info("  🔄 Fallback to direct channel search...")
        direct = await direct_search_channels(query, limit=50)
        
        for post in direct['posts']:
            norm_title = post['normalized_title']
            if norm_title not in posts_dict:
                posts_dict[norm_title] = {
                    'title': post['title'],
                    'content': format_post(post['content']),
                    'channel': post['channel_name'],
                    'date': post['date'].isoformat() if isinstance(post['date'], datetime) else post['date'],
                    'is_new': False,
                    'has_file': False,
                    'quality_options': {}
                }
        
        for file in direct['files']:
            norm_title = file['normalized_title']
            quality = file['quality']
            
            if norm_title not in files_dict:
                files_dict[norm_title] = {
                    'title': file['title'], 
                    'quality_options': {}, 
                    'date': file['date'].isoformat() if isinstance(file['date'], datetime) else file['date']
                }
            
            files_dict[norm_title]['quality_options'][quality] = {
                'file_id': file['file_id'],
                'file_size': file['file_size'],
                'file_name': file['file_name'],
                'message_id': file.get('message_id'),
                'channel_id': file.get('channel_id', Config.FILE_CHANNEL_ID)
            }
    
    merged = {}
    for norm_title, post_data in posts_dict.items():
        merged[norm_title] = post_data
    
    for norm_title, file_data in files_dict.items():
        if norm_title in merged:
            merged[norm_title]['has_file'] = True
            merged[norm_title]['quality_options'] = file_data['quality_options']
            logger.info(f"  ✓ Merged: {merged[norm_title]['title']} ({len(file_data['quality_options'])} qualities)")
        else:
            merged[norm_title] = {
                'title': file_data['title'],
                'content': f"<p style='text-align:center;'>{file_data['title']}</p>",
                'channel': 'SK4FiLM',
                'date': file_data['date'],
                'is_new': False,
                'has_file': True,
                'quality_options': file_data['quality_options']
            }
    
    results_list = list(merged.values())
    results_list.sort(key=lambda x: (x['has_file'], x['date']), reverse=True)
    
    total = len(results_list)
    paginated = results_list[offset:offset + limit]
    
    logger.info(f"✅ Search complete: {total} unique results | Showing page {page}: {len(paginated)} items")
    
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
    
    logger.info(f"🎨 Fetching posters for {len(movies)} movies...")
    async with aiohttp.ClientSession() as s:
        for i in range(0, len(movies), 5):
            batch = movies[i:i + 5]
            posters = await asyncio.gather(*[get_poster_multi(m['title'], s) for m in batch], return_exceptions=True)
            for m, p in zip(batch, posters):
                if isinstance(p, dict) and p.get('success'):
                    m.update({'poster_url': p['poster_url'], 'has_poster': True})
                else:
                    m['poster_url'] = f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(m['title'])}"
                    m['has_poster'] = True
            await asyncio.sleep(0.2)
    
    logger.info(f"✅ Loaded {len(movies)} movies with posters")
    return movies

@app.route('/')
async def root():
    tp = await posts_col.count_documents({}) if posts_col is not None else 0
    tf = await files_col.count_documents({}) if files_col is not None else 0
    return jsonify({
        'status': 'healthy',
        'service': 'SK4FiLM v2.0 Final',
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
            return jsonify({'status': 'error', 'message': 'Bot is starting, please wait...'}), 503
        movies = await get_home_movies()
        return jsonify({
            'status': 'success',
            'movies': movies,
            'total': len(movies),
            'bot_username': Config.BOT_USERNAME
        })
    except Exception as e:
        logger.error(f"API /movies error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/search')
async def api_search():
    try:
        q = request.args.get('query', '').strip()
        p = int(request.args.get('page', 1))
        l = int(request.args.get('limit', 12))
        
        if not q:
            return jsonify({'status': 'error', 'message': 'Query parameter required'}), 400
        if not bot_started:
            return jsonify({'status': 'error', 'message': 'Bot is starting, please wait...'}), 503
        
        result = await search_movies(q, l, p)
        return jsonify({
            'status': 'success',
            'query': q,
            'results': result['results'],
            'pagination': result['pagination'],
            'bot_username': Config.BOT_USERNAME
        })
    except Exception as e:
        logger.error(f"API /search error: {e}")
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

async def setup_bot():
    @bot.on_message(filters.command("start") & filters.private)
    async def start_handler(client, message):
        uid = message.from_user.id
        
        if len(message.command) > 1:
            fid = message.command[1]
            logger.info(f"📥 File request: User {uid} | File ID: {fid}")
            
            if not await check_force_sub(uid):
                try:
                    ch = await bot.get_chat(Config.FORCE_SUB_CHANNEL)
                    lk = f"https://t.me/{ch.username}" if ch.username else f"https://t.me/c/{str(Config.FORCE_SUB_CHANNEL)[4:]}/1"
                except:
                    lk = f"https://t.me/c/{str(Config.FORCE_SUB_CHANNEL)[4:]}/1"
                
                await message.reply_text(
                    "⚠️ **Join channel first to download files**",
                    reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("📢 Join Channel", url=lk)]])
                )
                return
            
            if files_col is None:
                logger.error("❌ Database not initialized")
                await message.reply_text("❌ **Database error**")
                return
            
            fd = await files_col.find_one({'file_id': fid})
            
            if not fd:
                logger.error(f"❌ File not found in DB: {fid}")
                await message.reply_text(
                    "❌ **File Not Found**\n\nPlease search again",
                    reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔍 Search", url=Config.WEBSITE_URL)]])
                )
                return
            
            logger.info(f"✅ Found: {fd.get('title')} | {fd.get('quality')}")
            
            try:
                pm = await message.reply_text(
                    f"⏳ **Sending file...**\n\n"
                    f"📁 `{fd.get('file_name', 'Movie')}`\n"
                    f"📊 {fd.get('quality', 'Unknown')}\n"
                    f"📦 {format_size(fd.get('file_size', 0))}"
                )
                
                channel_id = fd.get('channel_id', Config.FILE_CHANNEL_ID)
                message_id = fd.get('message_id')
                
                if not message_id:
                    logger.error(f"❌ No message_id for file: {fid}")
                    await pm.edit_text("❌ **Error: File info incomplete**")
                    return
                
                logger.info(f"📤 Copying from channel {channel_id}, message {message_id}")
                
                if not User:
                    logger.error("❌ User client not initialized")
                    await pm.edit_text("❌ **Bot error**")
                    return
                
                try:
                    file_message = await User.get_messages(channel_id, message_id)
                except Exception as e:
                    logger.error(f"❌ Failed to get message: {e}")
                    await pm.edit_text("❌ **Error fetching file**")
                    return
                
                if not file_message:
                    logger.error(f"❌ Message not found: {message_id}")
                    await pm.edit_text("❌ **File not found in channel**")
                    return
                
                try:
                    sent_message = await file_message.copy(uid)
                    await pm.delete()
                    
                    success_msg = await message.reply_text(
                        f"✅ **File sent successfully!**\n\n"
                        f"🎬 {fd.get('title', 'Movie')}\n"
                        f"📊 {fd.get('quality', 'Unknown')}\n"
                        f"📦 {format_size(fd.get('file_size', 0))}\n\n"
                        f"⚠️ **File will auto-delete in {Config.AUTO_DELETE_TIME//60} minutes**\n"
                        f"💾 Download immediately!"
                    )
                    
                    logger.info(f"✅ File sent: {fd.get('title')} → User {uid}")
                    
                    if Config.AUTO_DELETE_TIME > 0:
                        await asyncio.sleep(Config.AUTO_DELETE_TIME)
                        try:
                            await sent_message.delete()
                            await success_msg.edit_text(
                                "🗑️ **File deleted**\n\n"
                                "File was automatically deleted for security."
                            )
                            logger.info(f"🗑️ Auto-deleted for user {uid}")
                        except Exception as e:
                            logger.error(f"❌ Auto-delete failed: {e}")
                
                except Exception as e:
                    logger.error(f"❌ Copy failed: {e}")
                    await pm.edit_text(f"❌ **Error sending file**\n\n`{str(e)}`")
                    
            except Exception as e:
                logger.error(f"❌ Handler error: {e}")
                await message.reply_text(f"❌ **Error**\n\n`{str(e)}`")
            
            return
        
        await message.reply_text(
            "🎬 **Welcome to SK4FiLM**\n\n"
            "Search and download movies easily!\n\n"
            "Click below to start:",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🌐 Open Website", url=Config.WEBSITE_URL)]
            ])
        )
    
    @bot.on_message(filters.text & filters.private & ~filters.command(['start', 'stats', 'index']))
    async def text_handler(client, message):
        await message.reply_text(
            "👋 Use website to search",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🌐 Go to Website", url=Config.WEBSITE_URL)]])
        )
    
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
        await message.reply_text(f"📊 **Statistics**\n\n📝 Posts: {tp}\n📁 Files: {tf}")

async def init():
    global User, bot, bot_started, auto_index_task
    try:
        logger.info("🔄 Initializing system...")
        await init_mongodb()
        
        User = Client("sk4film_user", api_id=Config.API_ID, api_hash=Config.API_HASH, session_string=Config.USER_SESSION_STRING, workdir="/tmp", sleep_threshold=60)
        bot = Client("sk4film_bot", api_id=Config.API_ID, api_hash=Config.API_HASH, bot_token=Config.BOT_TOKEN, workdir="/tmp", sleep_threshold=60)
        
        await User.start()
        await bot.start()
        await setup_bot()
        
        me = await bot.get_me()
        logger.info(f"✅ Bot started: @{me.username}")
        bot_started = True
        
        await index_all_files()
        auto_index_task = asyncio.create_task(auto_index_loop())
        
        return True
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    logger.info("="*70)
    logger.info("🚀 SK4FiLM v2.0 - Production Ready")
    logger.info("="*70)
    await init()
    cfg = HyperConfig()
    cfg.bind = [f"0.0.0.0:{Config.WEB_SERVER_PORT}"]
    cfg.loglevel = "warning"
    logger.info(f"🌐 Web server starting on port {Config.WEB_SERVER_PORT}")
    await serve(app, cfg)

if __name__ == "__main__":
    asyncio.run(main())
