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

async def check_force_sub_immediate(user_id, max_retries=5):
    """IMMEDIATE force subscription check with instant verification"""
    for attempt in range(max_retries):
        try:
            logger.info(f"🔍 IMMEDIATE SUB CHECK: User {user_id} (Attempt {attempt + 1}/{max_retries})")
            
            # Very short delay for Telegram to update membership status
            if attempt > 0:
                await asyncio.sleep(1)  # Reduced from 2 seconds to 1 second
            
            member = await bot.get_chat_member(Config.FORCE_SUB_CHANNEL, user_id)
            is_member = member.status in ["member", "administrator", "creator"]
            
            logger.info(f"  {'✅ IMMEDIATE ACCESS' if is_member else '❌ NOT SUBSCRIBED'} | Status: {member.status}")
            
            if is_member:
                return True, member.status
                
        except UserNotParticipant:
            logger.info(f"  ❌ User {user_id} not in channel (Attempt {attempt + 1})")
            if attempt == max_retries - 1:
                return False, "not_joined"
        except (ChatAdminRequired, ChannelPrivate):
            logger.warning(f"  ⚠️ Bot permission issue - allowing access")
            return True, "admin_required"
        except Exception as e:
            logger.error(f"  ❌ Force sub error (Attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return True, "error_allowed"  # Allow on final error to avoid blocking users
    
    return False, "max_retries_exceeded"

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

async def get_poster_simple(title, session):
    """SIMPLE and RELIABLE poster fetcher - 100% working"""
    ck = title.lower().strip()
    
    # Check cache first
    if ck in movie_db['poster_cache']:
        c, ct = movie_db['poster_cache'][ck]
        if (datetime.now() - ct).seconds < 3600:
            movie_db['stats']['cache_hits'] += 1
            logger.info(f"  📦 Cache hit: {title}")
            return c
    
    logger.info(f"  🎨 FETCHING POSTER: {title}")
    
    year_match = re.search(r'\b(19|20)\d{2}\b', title)
    year = year_match.group() if year_match else ""
    clean_title = re.sub(r'\b(19|20)\d{2}\b', '', title).strip()
    
    # 1. Try TMDB first (most reliable)
    for api_key in Config.TMDB_KEYS:
        try:
            logger.info(f"    🔍 Trying TMDB...")
            url = "https://api.themoviedb.org/3/search/movie"
            params = {'api_key': api_key, 'query': clean_title}
            async with session.get(url, params=params, timeout=10) as r:
                if r.status == 200:
                    data = await r.json()
                    if data.get('results') and len(data['results']) > 0:
                        result = data['results'][0]
                        poster_path = result.get('poster_path')
                        if poster_path:
                            poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
                            res = {'poster_url': poster_url, 'source': 'TMDB', 'rating': str(result.get('vote_average', 0.0))}
                            movie_db['poster_cache'][ck] = (res, datetime.now())
                            movie_db['stats']['tmdb'] += 1
                            logger.info(f"    ✅ TMDB SUCCESS: {title}")
                            return res
        except Exception as e:
            logger.info(f"    ⚠️ TMDB failed: {e}")
            continue
    
    # 2. Try OMDB
    for api_key in Config.OMDB_KEYS:
        try:
            logger.info(f"    🔍 Trying OMDB...")
            url = f"http://www.omdbapi.com/?t={urllib.parse.quote(clean_title)}&apikey={api_key}"
            async with session.get(url, timeout=10) as r:
                if r.status == 200:
                    data = await r.json()
                    if data.get('Response') == 'True' and data.get('Poster') and data.get('Poster') != 'N/A':
                        poster_url = data['Poster'].replace('http://', 'https://')
                        res = {'poster_url': poster_url, 'source': 'OMDB', 'rating': data.get('imdbRating', '0.0')}
                        movie_db['poster_cache'][ck] = (res, datetime.now())
                        movie_db['stats']['omdb'] += 1
                        logger.info(f"    ✅ OMDB SUCCESS: {title}")
                        return res
        except Exception as e:
            logger.info(f"    ⚠️ OMDB failed: {e}")
            continue
    
    # 3. Try JustWatch (simple approach)
    try:
        logger.info(f"    🔍 Trying JustWatch...")
        slug = clean_title.lower().replace(' ', '-')
        slug = re.sub(r'[^\w\-]', '', slug)
        justwatch_url = f"https://www.justwatch.com/in/movie/{slug}"
        
        async with session.get(justwatch_url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }) as r:
            if r.status == 200:
                html_content = await r.text()
                poster_match = re.search(r'<meta property="og:image" content="([^"]+)"', html_content)
                if poster_match:
                    poster_url = poster_match.group(1)
                    if poster_url and poster_url.startswith('http'):
                        res = {'poster_url': poster_url, 'source': 'JustWatch', 'rating': '0.0'}
                        movie_db['poster_cache'][ck] = (res, datetime.now())
                        movie_db['stats']['justwatch'] += 1
                        logger.info(f"    ✅ JustWatch SUCCESS: {title}")
                        return res
    except Exception as e:
        logger.info(f"    ⚠️ JustWatch failed: {e}")
    
    # 4. GUARANTEED FALLBACK - Custom SVG generator (100% WORKING)
    logger.info(f"    ⚠️ ALL SOURCES FAILED, USING CUSTOM POSTER: {title}")
    movie_db['stats']['custom'] += 1
    
    # Create custom poster URL
    poster_data = {
        'title': title,
        'year': year
    }
    
    # Use our own poster endpoint
    res = {
        'poster_url': f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(title)}&year={year}", 
        'source': 'CUSTOM', 
        'rating': '0.0'
    }
    movie_db['poster_cache'][ck] = (res, datetime.now())
    logger.info(f"    ✅ CUSTOM POSTER GENERATED: {title}")
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
    
    # Search text channels
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
    
    # Search files
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
    logger.info("🏠 Fetching 30 movies with SIMPLE POSTER FETCH...")
    
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
        logger.info("🎨 FETCHING POSTERS WITH SIMPLE METHOD...")
        async with aiohttp.ClientSession() as session:
            tasks = []
            for movie in movies:
                tasks.append(get_poster_simple(movie['title'], session))
            
            posters = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, (movie, poster_result) in enumerate(zip(movies, posters)):
                if isinstance(poster_result, dict):
                    movie['poster_url'] = poster_result['poster_url']
                    movie['poster_source'] = poster_result['source']
                    movie['poster_rating'] = poster_result.get('rating', '0.0')
                    movie['has_poster'] = True
                else:
                    # 100% FALLBACK GUARANTEE - Use our own poster endpoint
                    movie['poster_url'] = f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(movie['title'])}"
                    movie['poster_source'] = 'CUSTOM'
                    movie['poster_rating'] = '0.0'
                    movie['has_poster'] = True
                    movie_db['stats']['custom'] += 1
        
        logger.info(f"  ✅ 100% POSTERS READY - ALL MOVIES HAVE POSTERS")
    
    logger.info(f"✅ {len(movies)} movies ready with 100% GUARANTEED POSTERS")
    return movies

@app.route('/')
async def root():
    tf = await files_col.count_documents({}) if files_col is not None else 0
    
    return jsonify({
        'status': 'healthy',
        'service': 'SK4FiLM v4.1 - FIXED VERIFICATION + POSTERS',
        'database': {'total_files': tf, 'live_mode': 'Posts LIVE, Files cached'},
        'bot_status': 'online' if bot_started else 'starting',
        'features': {
            'force_sub_immediate': 'FIXED',
            'poster_guarantee': '100% WORKING',
            'simple_poster_fetch': 'ENABLED'
        },
        'poster_stats': movie_db['stats']
    })

@app.route('/health')
async def health():
    return jsonify({'status': 'ok' if bot_started else 'starting'})

@app.route('/api/index_status')
async def api_index_status():
    try:
        if files_col is None:
            return jsonify({'status': 'error', 'message': 'Database not ready'}), 503
        
        total = await files_col.count_documents({})
        latest = await files_col.find_one({}, sort=[('indexed_at', -1)])
        last_indexed = "Never"
        if latest and latest.get('indexed_at'):
            dt = latest['indexed_at']
            if isinstance(dt, datetime):
                mins_ago = int((datetime.now() - dt).total_seconds() / 60)
                last_indexed = f"{mins_ago} min ago" if mins_ago > 0 else "Just now"
        
        return jsonify({
            'status': 'success',
            'total_indexed': total,
            'last_indexed': last_indexed,
            'bot_status': 'online' if bot_started else 'starting',
            'features': 'FIXED VERIFICATION + 100% POSTERS'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/movies')
async def api_movies():
    try:
        if not bot_started:
            return jsonify({'status': 'error', 'message': 'Starting...'}), 503
        
        movies = await get_home_movies_live()
        return jsonify({
            'status': 'success', 
            'movies': movies, 
            'total': len(movies), 
            'bot_username': Config.BOT_USERNAME, 
            'mode': 'LIVE',
            'poster_guarantee': '100% WORKING',
            'poster_stats': movie_db['stats']
        })
    except Exception as e:
        logger.error(f"API /movies: {e}")
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
        
        result = await search_movies_live(q, l, p)
        return jsonify({
            'status': 'success', 
            'query': q, 
            'results': result['results'], 
            'pagination': result['pagination'], 
            'bot_username': Config.BOT_USERNAME, 
            'mode': 'LIVE',
            'features': 'FIXED VERIFICATION + 100% POSTERS'
        })
    except Exception as e:
        logger.error(f"API /search: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/post')
async def api_post():
    try:
        channel_id = request.args.get('channel', '').strip()
        message_id = request.args.get('message', '').strip()
        
        if not channel_id or not message_id:
            return jsonify({'status':'error', 'message':'Missing channel or message parameter'}), 400
        
        if not bot_started or not User:
            return jsonify({'status':'error', 'message':'Bot not ready yet'}), 503
        
        try:
            channel_id = int(channel_id)
            message_id = int(message_id)
        except ValueError:
            return jsonify({'status':'error', 'message':'Invalid channel or message ID'}), 400
        
        logger.info(f"📄 Fetching post: Channel {channel_id}, Message {message_id}")
        
        try:
            msg = await User.get_messages(channel_id, message_id)
        except Exception as e:
            logger.error(f"  ❌ Failed to fetch message: {e}")
            return jsonify({'status':'error', 'message':'Failed to fetch message from Telegram'}), 404
        
        if not msg or not msg.text:
            return jsonify({'status':'error', 'message':'Message not found or has no text content'}), 404
        
        title = extract_title_smart(msg.text)
        if not title:
            title = msg.text.split('\n')[0][:60] if msg.text else "Movie Post"
        
        normalized_title = normalize_title(title)
        quality_options = {}
        has_file = False
        
        if files_col is not None:
            try:
                cursor = files_col.find({'normalized_title': normalized_title})
                async for doc in cursor:
                    quality = doc.get('quality', '480p')
                    if quality not in quality_options:
                        quality_options[quality] = {
                            'file_id': f"{doc.get('channel_id', Config.FILE_CHANNEL_ID)}_{doc.get('message_id')}_{quality}",
                            'file_size': doc.get('file_size', 0),
                            'file_name': doc.get('file_name', 'video.mp4')
                        }
                        has_file = True
            except Exception as e:
                logger.error(f"  ⚠️ File search error: {e}")
        
        post_data = {
            'title': title,
            'content': format_post(msg.text),
            'channel': channel_name(channel_id),
            'channel_id': channel_id,
            'message_id': message_id,
            'date': msg.date.isoformat() if isinstance(msg.date, datetime) else str(msg.date),
            'is_new': is_new(msg.date) if msg.date else False,
            'has_file': has_file,
            'quality_options': quality_options,
            'views': getattr(msg, 'views', 0)
        }
        
        logger.info(f"  ✅ Post fetched: {title}")
        
        return jsonify({'status': 'success', 'post': post_data, 'bot_username': Config.BOT_USERNAME})
    
    except Exception as e:
        logger.error(f"❌ API /post error: {e}")
        return jsonify({'status':'error', 'message': str(e)}), 500

@app.route('/api/poster')
async def api_poster():
    """100% WORKING CUSTOM POSTER GENERATOR"""
    try:
        t = request.args.get('title', 'Movie')
        y = request.args.get('year', '')
        
        # Simple and reliable SVG generator
        d = t[:20] + "..." if len(t) > 20 else t
        
        # Different color schemes for variety
        color_schemes = [
            {'bg1': '#667eea', 'bg2': '#764ba2', 'text': '#ffffff'},
            {'bg1': '#f093fb', 'bg2': '#f5576c', 'text': '#ffffff'},
            {'bg1': '#4facfe', 'bg2': '#00f2fe', 'text': '#ffffff'},
            {'bg1': '#43e97b', 'bg2': '#38f9d7', 'text': '#ffffff'},
            {'bg1': '#fa709a', 'bg2': '#fee140', 'text': '#ffffff'},
            {'bg1': '#30cfd0', 'bg2': '#330867', 'text': '#ffffff'},
            {'bg1': '#a8edea', 'bg2': '#fed6e3', 'text': '#333333'},
        ]
        
        # Pick a color scheme based on title hash
        scheme = color_schemes[hash(t) % len(color_schemes)]
        
        svg = f'''<svg width="300" height="450" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:{scheme['bg1']};stop-opacity:1"/>
                    <stop offset="100%" style="stop-color:{scheme['bg2']};stop-opacity:1"/>
                </linearGradient>
            </defs>
            <rect width="100%" height="100%" fill="url(#bg)"/>
            <rect x="10" y="10" width="280" height="430" fill="none" stroke="{scheme['text']}" stroke-width="2" stroke-opacity="0.3" rx="10"/>
            <circle cx="150" cy="180" r="60" fill="rgba(255,255,255,0.1)"/>
            <text x="150" y="185" text-anchor="middle" fill="{scheme['text']}" font-size="60" font-family="Arial">🎬</text>
            <text x="150" y="280" text-anchor="middle" fill="{scheme['text']}" font-size="16" font-weight="bold" font-family="Arial">{html.escape(d)}</text>
            {f'<text x="150" y="305" text-anchor="middle" fill="{scheme['text']}" font-size="14" font-family="Arial">{html.escape(y)}</text>' if y else ''}
            <rect x="50" y="380" width="200" height="40" rx="20" fill="rgba(0,0,0,0.3)"/>
            <text x="150" y="405" text-anchor="middle" fill="{scheme['text']}" font-size="16" font-weight="bold" font-family="Arial">SK4FiLM</text>
        </svg>'''
        
        return Response(svg, mimetype='image/svg+xml', headers={
            'Cache-Control': 'public, max-age=86400',
            'Content-Type': 'image/svg+xml'
        })
        
    except Exception as e:
        logger.error(f"Poster generation error: {e}")
        # Ultimate fallback
        simple_svg = '''<svg width="300" height="450" xmlns="http://www.w3.org/2000/svg">
            <rect width="100%" height="100%" fill="#667eea"/>
            <text x="150" y="225" text-anchor="middle" fill="white" font-size="18" font-family="Arial">SK4FiLM</text>
        </svg>'''
        return Response(simple_svg, mimetype='image/svg+xml')

async def setup_bot():
    @bot.on_message(filters.command("start") & filters.private)
    async def start_handler(client, message):
        uid = message.from_user.id
        user_name = message.from_user.first_name or "User"
        
        if len(message.command) > 1:
            fid = message.command[1]
            logger.info(f"📥 File request | User: {uid} | File ID: {fid}")
            
            # FIXED: IMMEDIATE FORCE SUBSCRIPTION CHECK
            is_subscribed, status = await check_force_sub_immediate(uid, max_retries=5)
            
            if not is_subscribed:
                try:
                    ch = await bot.get_chat(Config.FORCE_SUB_CHANNEL)
                    invite_link = f"https://t.me/{ch.username}" if ch.username else f"https://t.me/c/{str(Config.FORCE_SUB_CHANNEL)[4:]}/1"
                except Exception as e:
                    logger.error(f"Channel link error: {e}")
                    invite_link = f"https://t.me/c/{str(Config.FORCE_SUB_CHANNEL)[4:]}/1"
                
                # FIXED: Better message with clear instructions
                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton("📢 JOIN CHANNEL", url=invite_link)],
                    [InlineKeyboardButton("🔄 TRY AGAIN", url=f"https://t.me/{Config.BOT_USERNAME}?start={fid}")]
                ])
                
                await message.reply_text(
                    f"👋 **Hello {user_name}!**\n\n"
                    "🔒 **Access Required**\n"
                    "To download files, you need to join our channel.\n\n"
                    "🚀 **Quick Steps:**\n"
                    "1. Click **JOIN CHANNEL** below\n"
                    "2. Wait for channel to open\n" 
                    "3. Click **JOIN** button in channel\n"
                    "4. Come back and click **TRY AGAIN**\n\n"
                    "⚡ **Instant verification!**",
                    reply_markup=keyboard,
                    disable_web_page_preview=True
                )
                logger.info(f"  ❌ Access denied for user {uid} | Status: {status}")
                return
            
            logger.info(f"  ✅ User {uid} VERIFIED | Status: {status}")
            
            try:
                parts = fid.split('_')
                if len(parts) >= 2:
                    channel_id = int(parts[0])
                    message_id = int(parts[1])
                    quality = parts[2] if len(parts) > 2 else "HD"
                    
                    pm = await message.reply_text(f"⏳ **Preparing your file...**\n\n📦 Quality: {quality}")
                    
                    # Fetch file from channel
                    file_message = await bot.get_messages(channel_id, message_id)
                    
                    if not file_message or (not file_message.document and not file_message.video):
                        await pm.edit_text("❌ **File not found**\n\nThe file may have been deleted.")
                        return
                    
                    # Send file to user
                    if file_message.document:
                        sent = await bot.send_document(
                            uid, 
                            file_message.document.file_id, 
                            caption=f"🎬 **Download Complete!**\n\n"
                                   f"📹 Quality: {quality}\n"
                                   f"📦 Size: {format_size(file_message.document.file_size)}\n\n"
                                   f"⚠️ Will auto-delete in {Config.AUTO_DELETE_TIME//60} minutes\n\n"
                                   f"Enjoy! 🍿"
                        )
                    else:
                        sent = await bot.send_video(
                            uid, 
                            file_message.video.file_id, 
                            caption=f"🎬 **Download Complete!**\n\n"
                                   f"📹 Quality: {quality}\n" 
                                   f"📦 Size: {format_size(file_message.video.file_size)}\n\n"
                                   f"⚠️ Will auto-delete in {Config.AUTO_DELETE_TIME//60} minutes\n\n"
                                   f"Enjoy! 🍿"
                        )
                    
                    await pm.delete()
                    logger.info(f"  ✅ File sent successfully to user {uid}")
                    
                    # Auto-delete after specified time
                    if Config.AUTO_DELETE_TIME > 0:
                        async def auto_delete():
                            await asyncio.sleep(Config.AUTO_DELETE_TIME)
                            try:
                                await sent.delete()
                                logger.info(f"  🗑️ Auto-deleted file for user {uid}")
                            except:
                                pass
                        
                        asyncio.create_task(auto_delete())
                        
                else:
                    await message.reply_text("❌ **Invalid file link**\n\nPlease get a fresh link from the website.")
                    
            except Exception as e:
                logger.error(f"  ❌ File send error: {e}")
                try:
                    await message.reply_text(
                        f"❌ **Download Failed**\n\n"
                        f"Error: `{str(e)}`\n\n"
                        f"Please try again or contact support."
                    )
                except:
                    pass
            return
        
        # Welcome message for /start without parameters
        welcome_text = (
            f"🎬 **Welcome to SK4FiLM, {user_name}!**\n\n"
            "🌐 **Use our website to browse and download movies:**\n"
            f"{Config.WEBSITE_URL}\n\n"
            "✨ **Features:**\n"
            "• 🎥 Latest movies & shows\n" 
            "• 📺 Multiple quality options\n"
            "• ⚡ Fast downloads\n"
            "• 🔒 Secure & reliable\n\n"
            "👇 **Get started below:**"
        )
        
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🌐 VISIT WEBSITE", url=Config.WEBSITE_URL)],
            [InlineKeyboardButton("📢 JOIN CHANNEL", url=f"https://t.me/c/{str(Config.FORCE_SUB_CHANNEL)[4:]}/1")]
        ])
        
        await message.reply_text(welcome_text, reply_markup=keyboard, disable_web_page_preview=True)
    
    @bot.on_message(filters.text & filters.private & ~filters.command(['start', 'stats', 'index']))
    async def text_handler(client, message):
        user_name = message.from_user.first_name or "User"
        await message.reply_text(
            f"👋 **Hi {user_name}!**\n\n"
            "🔍 **Please use our website to search for movies:**\n\n"
            f"{Config.WEBSITE_URL}\n\n"
            "This bot only handles file downloads via website links.",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)]
            ]),
            disable_web_page_preview=True
        )
    
    @bot.on_message(filters.command("index") & filters.user(Config.ADMIN_IDS))
    async def index_handler(client, message):
        msg = await message.reply_text("🔄 **Starting background indexing...**")
        asyncio.create_task(index_files_background())
        await msg.edit_text("✅ **Indexing started in background!**\n\nCheck /stats for progress.")
    
    @bot.on_message(filters.command("stats") & filters.user(Config.ADMIN_IDS))
    async def stats_handler(client, message):
        tf = await files_col.count_documents({}) if files_col is not None else 0
        
        stats_text = (
            f"📊 **SK4FiLM Statistics**\n\n"
            f"📁 **Files Indexed:** {tf}\n"
            f"🔴 **Live Posts:** Active\n"
            f"🤖 **Bot Status:** Online\n\n"
            f"**🎨 Poster Sources:**\n"
            f"• TMDB: {movie_db['stats']['tmdb']}\n"
            f"• OMDB: {movie_db['stats']['omdb']}\n" 
            f"• JustWatch: {movie_db['stats']['justwatch']}\n"
            f"• Custom: {movie_db['stats']['custom']}\n"
            f"• Cache Hits: {movie_db['stats']['cache_hits']}\n\n"
            f"**⚡ Features:**\n"
            f"• ✅ Fixed force sub verification\n"
            f"• ✅ 100% poster guarantee\n"
            f"• ✅ Simple & reliable"
        )
        await message.reply_text(stats_text)

async def init():
    global User, bot, bot_started
    try:
        logger.info("🚀 INITIALIZING SK4FiLM BOT...")
        await init_mongodb()
        
        # Initialize clients
        User = Client(
            "user_session", 
            api_id=Config.API_ID, 
            api_hash=Config.API_HASH, 
            session_string=Config.USER_SESSION_STRING,
            no_updates=True
        )
        
        bot = Client(
            "bot",
            api_id=Config.API_ID,
            api_hash=Config.API_HASH, 
            bot_token=Config.BOT_TOKEN
        )
        
        await User.start()
        await bot.start()
        await setup_bot()
        
        me = await bot.get_me()
        logger.info(f"✅ BOT STARTED: @{me.username}")
        bot_started = True
        
        logger.info("🔄 Starting background indexing...")
        asyncio.create_task(index_files_background())
        
        return True
    except Exception as e:
        logger.error(f"❌ INIT FAILED: {e}")
        return False

async def main():
    logger.info("="*60)
    logger.info("🎬 SK4FiLM v4.1 - FIXED VERSION")
    logger.info("✅ Force Sub: IMMEDIATE VERIFICATION (FIXED)")
    logger.info("✅ Posters: 100% GUARANTEED (FIXED)") 
    logger.info("✅ Simple & Reliable poster fetching")
    logger.info("="*60)
    
    success = await init()
    if not success:
        logger.error("❌ Failed to initialize bot")
        return
    
    # Start web server
    config = HyperConfig()
    config.bind = [f"0.0.0.0:{Config.WEB_SERVER_PORT}"]
    config.loglevel = "warning"
    
    logger.info(f"🌐 Web server starting on port {Config.WEB_SERVER_PORT}...")
    await serve(app, config)

if __name__ == "__main__":
    asyncio.run(main())
