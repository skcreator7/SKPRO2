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
        logger.info("🔌 MongoDB...")
        mongo_client = AsyncIOMotorClient(Config.MONGODB_URI, serverSelectionTimeoutMS=10000)
        await mongo_client.admin.command('ping')
        
        db = mongo_client.sk4film
        posts_col = db.posts
        files_col = db.files
        
        await posts_col.create_index([("title", "text"), ("content", "text")])
        await posts_col.create_index([("date", -1)])
        await posts_col.create_index([("is_new", -1)])
        await files_col.create_index([("title", "text")])
        await files_col.create_index([("file_id", 1)], unique=True)
        
        logger.info("✅ MongoDB OK")
        return True
    except Exception as e:
        logger.error(f"❌ MongoDB: {e}")
        return False

User = None
bot = None
bot_started = False
movie_db = {'poster_cache': {}, 'stats': {'omdb': 0, 'tmdb': 0, 'impawards': 0, 'justwatch': 0, 'letterboxd': 0, 'custom': 0}}
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
        hours = (datetime.now() - date.replace(tzinfo=None)).total_seconds() / 3600
        return hours <= 48
    except:
        return False

async def check_force_sub(user_id):
    try:
        member = await bot.get_chat_member(Config.FORCE_SUB_CHANNEL, user_id)
        return member.status in ["member", "administrator", "creator"]
    except:
        return True

async def index_all_files():
    """User client ONLY for indexing posts"""
    if not User or not bot_started or posts_col is None or files_col is None:
        logger.warning("⚠️ Cannot index")
        return
    
    logger.info("📥 Indexing...")
    indexed_posts = 0
    indexed_files = 0
    
    try:
        # User client for POST indexing
        for channel_id in Config.TEXT_CHANNEL_IDS:
            try:
                count = 0
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
                logger.info(f"  ✓ {channel_name(channel_id)}: {count}")
            except Exception as e:
                logger.error(f"  ✗ {channel_name(channel_id)}: {e}")
        
        # Bot for FILE indexing (faster, more reliable)
        try:
            count = 0
            async for msg in bot.get_chat_history(Config.FILE_CHANNEL_ID, limit=200):
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
            logger.info(f"  ✓ Files: {count}")
        except Exception as e:
            logger.error(f"  ✗ Files: {e}")
        
        logger.info(f"✅ Indexed: {indexed_posts} posts, {indexed_files} files")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")

async def auto_index_loop():
    while True:
        try:
            await asyncio.sleep(Config.AUTO_INDEX_INTERVAL)
            logger.info("🔄 Auto-index")
            await index_all_files()
        except Exception as e:
            logger.error(f"Loop: {e}")
            await asyncio.sleep(60)

async def get_poster_multi(title, session):
    """Enhanced poster from multiple sources"""
    ck = title.lower().strip()
    if ck in movie_db['poster_cache']:
        c, ct = movie_db['poster_cache'][ck]
        if (datetime.now() - ct).seconds < 600:
            return c
    
    # 1. OMDB
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
    
    # 2. TMDB
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
    
    # 3. IMPAwards
    try:
        search_title = title.replace(' ', '_').lower()
        year_match = re.search(r'\b(19|20)\d{2}\b', title)
        if year_match:
            year = year_match.group()
            poster_url = f"https://www.impawards.com/{year}/posters/{search_title}.jpg"
            async with session.head(poster_url, timeout=3) as r:
                if r.status == 200:
                    res = {'poster_url': poster_url, 'source': 'IMPAwards', 'success': True}
                    movie_db['poster_cache'][ck] = (res, datetime.now())
                    movie_db['stats']['impawards'] += 1
                    return res
    except:
        pass
    
    # 4. JustWatch
    try:
        slug = title.lower().replace(' ', '-').replace(':', '').replace('\'', '')
        slug = re.sub(r'[^\w\-]', '', slug)
        poster_url = f"https://images.justwatch.com/poster/{slug}/s718"
        async with session.head(poster_url, timeout=3) as r:
            if r.status == 200:
                res = {'poster_url': poster_url, 'source': 'JustWatch', 'success': True}
                movie_db['poster_cache'][ck] = (res, datetime.now())
                movie_db['stats']['justwatch'] += 1
                return res
    except:
        pass
    
    # 5. Letterboxd
    try:
        slug = title.lower().replace(' ', '-').replace(':', '').replace('\'', '')
        slug = re.sub(r'[^\w\-]', '', slug)
        if len(slug) >= 2:
            poster_url = f"<https://a.ltrbxd.com/resized/film-poster/{slug>[0]}/{slug[1] if len(slug) > 1 else 'x'}/{slug}/{slug}-0-230-0-345-crop.jpg"
            async with session.head(poster_url, timeout=3) as r:
                if r.status == 200:
                    res = {'poster_url': poster_url, 'source': 'Letterboxd', 'success': True}
                    movie_db['poster_cache'][ck] = (res, datetime.now())
                    movie_db['stats']['letterboxd'] += 1
                    return res
    except:
        pass
    
    # 6. Custom fallback
    movie_db['stats']['custom'] += 1
    res = {'poster_url': f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(title)}", 'source': 'CUSTOM', 'success': True}
    movie_db['poster_cache'][ck] = (res, datetime.now())
    return res

async def direct_search_channels(query, limit=50):
    """User client for direct post search"""
    if not User:
        return {'posts': [], 'files': []}
    
    logger.info(f"🔍 Direct search: '{query}'")
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
                                'date': msg.date,
                                'is_new': is_new(msg.date) if msg.date else False
                            })
                            count += 1
                logger.info(f"  ✓ {channel_name(channel_id)}: {count}")
            except Exception as e:
                logger.error(f"  ✗ {channel_name(channel_id)}: {e}")
        
        # Bot for file search
        try:
            count = 0
            async for msg in bot.search_messages(Config.FILE_CHANNEL_ID, query=query, limit=limit):
                if msg.document or msg.video:
                    title = extract_title_from_file(msg)
                    if title and query_lower in title.lower():
                        file_size = msg.document.file_size if msg.document else (msg.video.file_size if msg.video else 0)
                        file_name = msg.document.file_name if msg.document else (msg.video.file_name if msg.video else 'video.mp4')
                        quality = detect_quality(file_name)
                        
                        files_found.append({
                            'title': title,
                            'normalized_title': normalize_title(title),
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
        
        logger.info(f"✅ Direct: {len(posts_found)} posts, {len(files_found)} files")
        
    except Exception as e:
        logger.error(f"❌ Direct error: {e}")
    
    return {'posts': posts_found, 'files': files_found}

async def search_movies(query, limit=12, page=1):
    offset = (page - 1) * limit
    logger.info(f"🔍 Search: '{query}' Page {page}")
    
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
                    'file_id': f"{doc.get('channel_id', Config.FILE_CHANNEL_ID)}_{doc.get('message_id')}_{quality}",
                    'file_size': doc['file_size'],
                    'file_name': doc['file_name']
                }
            
            logger.info(f"  ✓ MongoDB: {len(posts_dict)} posts, {len(files_dict)} files")
        except Exception as e:
            logger.error(f"  ✗ MongoDB: {e}")
    
    if len(posts_dict) == 0 and len(files_dict) == 0:
        logger.info("  🔄 Direct search...")
        direct = await direct_search_channels(query, limit=50)
        
        for post in direct['posts']:
            norm_title = post['normalized_title']
            if norm_title not in posts_dict:
                posts_dict[norm_title] = {
                    'title': post['title'],
                    'content': format_post(post['content']),
                    'channel': post['channel_name'],
                    'date': post['date'].isoformat() if isinstance(post['date'], datetime) else post['date'],
                    'is_new': post.get('is_new', False),
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
                'file_id': f"{file.get('channel_id', Config.FILE_CHANNEL_ID)}_{file.get('message_id')}_{quality}",
                'file_size': file['file_size'],
                'file_name': file['file_name']
            }
    
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
                'quality_options': file_data['quality_options']
            }
    
    results_list = list(merged.values())
    
    # Sort: NEW first, then files, then date
    results_list.sort(key=lambda x: (
        not x.get('is_new', False),
        not x['has_file'],
        x['date']
    ), reverse=True)
    
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
    
    logger.info("🏠 Loading...")
    movies = []
    seen = set()
    
    try:
        # Get NEW posts first
        cursor = posts_col.find({'is_new': True}).sort('date', -1).limit(10)
        async for doc in cursor:
            tk = doc['title'].lower().strip()
            if tk not in seen:
                seen.add(tk)
                movies.append({
                    'title': doc['title'], 
                    'date': doc['date'].isoformat() if isinstance(doc['date'], datetime) else doc['date'], 
                    'is_new': True
                })
        
        # Then regular posts
        cursor = posts_col.find().sort('date', -1).limit(50)
        async for doc in cursor:
            tk = doc['title'].lower().strip()
            if tk not in seen:
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
    
    logger.info(f"🎨 Posters...")
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
    
    logger.info(f"✅ {len(movies)}")
    return movies

@app.route('/')
async def root():
    tp = await posts_col.count_documents({}) if posts_col is not None else 0
    tf = await files_col.count_documents({}) if files_col is not None else 0
    tn = await posts_col.count_documents({'is_new': True}) if posts_col is not None else 0
    return jsonify({
        'status': 'healthy',
        'service': 'SK4FiLM v2.3',
        'database': {'posts': tp, 'files': tf, 'new_posts': tn},
        'bot_status': 'online' if bot_started else 'starting',
        'poster_stats': movie_db['stats']
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
        logger.error(f"Movies: {e}")
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
        logger.error(f"Search: {e}")
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
            logger.info(f"="*60)
            logger.info(f"📥 User {uid} | ID: {fid}")
            
            if not await check_force_sub(uid):
                try:
                    ch = await bot.get_chat(Config.FORCE_SUB_CHANNEL)
                    lk = f"https://t.me/{ch.username}" if ch.username else f"https://t.me/c/{str(Config.FORCE_SUB_CHANNEL)[4:]}/1"
                except:
                    lk = f"https://t.me/c/{str(Config.FORCE_SUB_CHANNEL)[4:]}/1"
                
                await message.reply_text(
                    "⚠️ **Join first**",
                    reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("📢 Join", url=lk)]])
                )
                return
            
            try:
                parts = fid.split('_')
                if len(parts) >= 2:
                    channel_id = int(parts[0])
                    message_id = int(parts[1])
                    quality = parts[2] if len(parts) > 2 else "Unknown"
                    logger.info(f"✅ Parsed: Ch {channel_id} | Msg {message_id} | {quality}")
                else:
                    raise ValueError("Invalid")
            except Exception as e:
                logger.error(f"❌ Parse failed: {e}")
                await message.reply_text(
                    "❌ **Invalid ID**",
                    reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔍 Search", url=Config.WEBSITE_URL)]])
                )
                return
            
            try:
                pm = await message.reply_text(f"⏳ **Sending [{quality}]...**")
                
                file_message = await bot.get_messages(channel_id, message_id)
                
                if not file_message or (not file_message.document and not file_message.video):
                    raise Exception("No file")
                
                if file_message.document:
                    file_name = file_message.document.file_name
                    file_size = file_message.document.file_size
                    file_id = file_message.document.file_id
                    sent = await bot.send_document(uid, file_id, caption=f"🎬 {quality}\n📦 {format_size(file_size)}")
                else:
                    file_name = file_message.video.file_name or "video.mp4"
                    file_size = file_message.video.file_size
                    file_id = file_message.video.file_id
                    sent = await bot.send_video(uid, file_id, caption=f"🎬 {quality}\n📦 {format_size(file_size)}")
                
                await pm.delete()
                
                success = await message.reply_text(
                    f"✅ **Sent!**\n\n"
                    f"📁 `{file_name}`\n"
                    f"📊 {quality}\n"
                    f"📦 {format_size(file_size)}\n\n"
                    f"⚠️ **Auto-delete in {Config.AUTO_DELETE_TIME//60} min**"
                )
                
                logger.info(f"✅ SUCCESS: {file_name}")
                logger.info(f"="*60)
                
                if Config.AUTO_DELETE_TIME > 0:
                    await asyncio.sleep(Config.AUTO_DELETE_TIME)
                    try:
                        await sent.delete()
                        await success.edit_text("🗑️ **Deleted**")
                    except:
                        pass
                    
            except Exception as e:
                logger.error(f"❌ Error: {e}")
                await pm.edit_text(f"❌ **Failed**\n\n`{str(e)}`")
            
            return
        
        await message.reply_text(
            "🎬 **SK4FiLM**\n\nSearch movies",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🌐 Website", url=Config.WEBSITE_URL)]])
        )
    
    @bot.on_message(filters.text & filters.private & ~filters.command(['start', 'stats', 'index']))
    async def text_handler(client, message):
        await message.reply_text(
            "👋 Use website",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🌐 Go", url=Config.WEBSITE_URL)]])
        )
    
    @bot.on_message(filters.command("index") & filters.user(Config.ADMIN_IDS))
    async def index_handler(client, message):
        await message.reply_text("🔄 Indexing...")
        await index_all_files()
        tp = await posts_col.count_documents({}) if posts_col is not None else 0
        tf = await files_col.count_documents({}) if files_col is not None else 0
        tn = await posts_col.count_documents({'is_new': True}) if posts_col is not None else 0
        await message.reply_text(f"✅ Done\n\n📝 {tp}\n📁 {tf}\n🆕 {tn}")
    
    @bot.on_message(filters.command("stats") & filters.user(Config.ADMIN_IDS))
    async def stats_handler(client, message):
        tp = await posts_col.count_documents({}) if posts_col is not None else 0
        tf = await files_col.count_documents({}) if files_col is not None else 0
        tn = await posts_col.count_documents({'is_new': True}) if posts_col is not None else 0
        
        stats_text = f"📊 **Stats**\n\n"
        stats_text += f"📝 Posts: {tp}\n"
        stats_text += f"📁 Files: {tf}\n"
        stats_text += f"🆕 New: {tn}\n\n"
        stats_text += f"**Poster Sources:**\n"
        stats_text += f"OMDB: {movie_db['stats']['omdb']}\n"
        stats_text += f"TMDB: {movie_db['stats']['tmdb']}\n"
        stats_text += f"IMPAwards: {movie_db['stats']['impawards']}\n"
        stats_text += f"JustWatch: {movie_db['stats']['justwatch']}\n"
        stats_text += f"Letterboxd: {movie_db['stats']['letterboxd']}\n"
        stats_text += f"Custom: {movie_db['stats']['custom']}"
        
        await message.reply_text(stats_text)

async def init():
    global User, bot, bot_started, auto_index_task
    try:
        logger.info("🔄 Init...")
        await init_mongodb()
        
        User = Client("user", api_id=Config.API_ID, api_hash=Config.API_HASH, session_string=Config.USER_SESSION_STRING, workdir="/tmp", sleep_threshold=60)
        bot = Client("bot", api_id=Config.API_ID, api_hash=Config.API_HASH, bot_token=Config.BOT_TOKEN, workdir="/tmp", sleep_threshold=60)
        
        await User.start()
        await bot.start()
        await setup_bot()
        
        me = await bot.get_me()
        logger.info(f"✅ @{me.username}")
        bot_started = True
        
        await index_all_files()
        auto_index_task = asyncio.create_task(auto_index_loop())
        
        return True
    except Exception as e:
        logger.error(f"❌ {e}")
        return False

async def main():
    logger.info("="*70)
    logger.info("🚀 SK4FiLM v2.3 Final - Bot-only + Better Posters + NEW Tag")
    logger.info("="*70)
    await init()
    cfg = HyperConfig()
    cfg.bind = [f"0.0.0.0:{Config.WEB_SERVER_PORT}"]
    cfg.loglevel = "warning"
    logger.info(f"🌐 Port {Config.WEB_SERVER_PORT}")
    await serve(app, cfg)

if __name__ == "__main__":
    asyncio.run(main())
