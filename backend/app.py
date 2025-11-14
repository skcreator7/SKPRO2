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
from motor.motor_asyncio import AsyncIOMotorClient
import html
import re
import math
import aiohttp
import urllib.parse

# ==================== LOGGING ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logging.getLogger('asyncio').setLevel(logging.WARNING)

# ==================== CONFIG ====================
class Config:
    # Telegram
    API_ID = int(os.environ.get("API_ID", "0"))
    API_HASH = os.environ.get("API_HASH", "")
    USER_SESSION_STRING = os.environ.get("USER_SESSION_STRING", "")
    BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
    
    # MongoDB
    MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
    
    # Channels
    TEXT_CHANNEL_IDS = [-1001891090100, -1002024811395]
    FILE_CHANNEL_ID = int(os.environ.get("FILE_CHANNEL_ID", "-1001768249569"))
    FORCE_SUB_CHANNEL = int(os.environ.get("FORCE_SUB_CHANNEL", "-1001891090100"))
    
    # Bot Config
    WEBSITE_URL = os.environ.get("WEBSITE_URL", "https://sk4film.vercel.app")
    BOT_USERNAME = os.environ.get("BOT_USERNAME", "sk4film_bot")
    ADMIN_IDS = [int(x) for x in os.environ.get("ADMIN_IDS", "123456789").split(",")]
    AUTO_DELETE_TIME = int(os.environ.get("AUTO_DELETE_TIME", "300"))
    WEB_SERVER_PORT = int(os.environ.get("PORT", 8000))
    BACKEND_URL = os.environ.get("BACKEND_URL", "https://sk4film.koyeb.app")
    
    # Poster API Keys (Multiple for rate limiting)
    OMDB_KEYS = ["8265bd1c", "b9bd48a6", "2f2d1c8e", "c3e6f8d9", "3e7e7ac8"]
    TMDB_KEYS = ["e547e17d4e91f3e62a571655cd1ccaff", "8265bd1c", "2f2d1c8e"]
    
    # Auto-indexing
    AUTO_INDEX_INTERVAL = 600  # 10 minutes

# ==================== QUART APP ====================
app = Quart(__name__)
app.config.update({
    'JSON_SORT_KEYS': False,
    'PROVIDE_AUTOMATIC_OPTIONS': True,
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024
})

@app.after_request
async def add_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

# ==================== MONGODB ====================
mongo_client = None
db = None
posts_col = None
files_col = None

async def init_mongodb():
    global mongo_client, db, posts_col, files_col
    try:
        logger.info("🔌 Connecting MongoDB...")
        mongo_client = AsyncIOMotorClient(Config.MONGODB_URI, serverSelectionTimeoutMS=5000)
        
        # Test connection
        await mongo_client.admin.command('ping')
        
        db = mongo_client.sk4film
        posts_col = db.posts
        files_col = db.files
        
        # Create text indexes for search
        await posts_col.create_index([("title", "text"), ("content", "text")])
        await posts_col.create_index([("date", -1)])
        await posts_col.create_index([("channel_id", 1), ("message_id", 1)], unique=True)
        
        await files_col.create_index([("title", "text")])
        await files_col.create_index([("file_id", 1)], unique=True)
        await files_col.create_index([("date", -1)])
        
        logger.info("✅ MongoDB connected")
        return True
    except Exception as e:
        logger.error(f"❌ MongoDB error: {e}")
        return False

# ==================== GLOBAL ====================
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
        'failed': 0
    }
}
auto_index_task = None

# ==================== HELPERS ====================
def extract_title_smart(text):
    """Extract movie title from text post"""
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
        
        for pattern in patterns:
            match = re.search(pattern, first)
            if match:
                title = match.group(1).strip()
                if 4 <= len(title) <= 50:
                    return title
    except:
        pass
    return None

def extract_title_from_file(msg):
    """Extract movie title from file message"""
    try:
        # Try caption first
        if msg.caption:
            title = extract_title_smart(msg.caption)
            if title:
                return title
        
        # Extract from filename
        filename = msg.document.file_name if msg.document else (msg.video.file_name if msg.video else None)
        if filename:
            name = filename.rsplit('.', 1)[0]
            name = re.sub(r'[\._\-]', ' ', name)
            name = re.sub(r'(720p|1080p|480p|HDRip|WEB|BluRay|x264|x265|HEVC|HDTV)', '', name, flags=re.IGNORECASE)
            name = ' '.join(name.split()).strip()
            if 4 <= len(name) <= 50:
                return name
    except:
        pass
    return None

def format_size(size):
    """Format file size"""
    if not size:
        return "Unknown"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    elif size < 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    else:
        return f"{size / (1024 * 1024 * 1024):.2f} GB"

def detect_quality(filename):
    """Detect video quality from filename"""
    if not filename:
        return "480p"
    fl = filename.lower()
    if '2160p' in fl or '4k' in fl or 'uhd' in fl:
        return "2160p"
    elif '1080p' in fl or 'fhd' in fl:
        return "1080p"
    elif '720p' in fl or 'hd' in fl:
        return "720p"
    return "480p"

def format_post(text):
    """Format post content for HTML"""
    if not text:
        return ""
    text = html.escape(text)
    text = re.sub(r'(https?://[^\s]+)', r'<a href="\1" target="_blank" style="color:#00ccff">\1</a>', text)
    return text.replace('\n', '<br>')

def channel_name(channel_id):
    """Get channel name from ID"""
    channels = {
        -1001891090100: "SK4FiLM Main",
        -1002024811395: "SK4FiLM Updates",
        -1001768249569: "SK4FiLM Files"
    }
    return channels.get(channel_id, "Channel")

def is_new(date):
    """Check if post is new (within 24 hours)"""
    try:
        if isinstance(date, str):
            date = datetime.fromisoformat(date.replace('Z', '+00:00'))
        hours_diff = (datetime.now() - date.replace(tzinfo=None)).seconds / 3600
        return hours_diff <= 24
    except:
        return False

async def check_force_sub(user_id):
    """Check if user is subscribed to force sub channel"""
    try:
        member = await bot.get_chat_member(Config.FORCE_SUB_CHANNEL, user_id)
        return member.status in ["member", "administrator", "creator"]
    except:
        return True

# ==================== AUTO FILE INDEXING ====================
async def index_all_files():
    """Auto-index all files and posts from channels"""
    if not User or not bot_started or posts_col is None or files_col is None:
        logger.warning("⚠️ Cannot index - services not ready")
        return
    
    logger.info("📥 Starting auto-indexing...")
    indexed_count = 0
    
    try:
        # Index TEXT posts (with duplicate prevention)
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
                                indexed_count += 1
                            except:
                                pass  # Skip duplicates
                
                logger.info(f"  ✓ {channel_name(channel_id)}: {count} posts")
            except Exception as e:
                logger.error(f"  ✗ {channel_name(channel_id)}: {e}")
        
        # Index FILES
        try:
            count = 0
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
                            indexed_count += 1
                        except:
                            pass  # Skip duplicates
            
            logger.info(f"  ✓ {channel_name(Config.FILE_CHANNEL_ID)}: {count} files")
        except Exception as e:
            logger.error(f"  ✗ Files: {e}")
        
        logger.info(f"✅ Auto-indexing complete: {indexed_count} items")
        
    except Exception as e:
        logger.error(f"❌ Indexing error: {e}")

async def auto_index_loop():
    """Background task for periodic auto-indexing"""
    while True:
        try:
            await asyncio.sleep(Config.AUTO_INDEX_INTERVAL)
            logger.info("🔄 Periodic auto-index triggered...")
            await index_all_files()
        except Exception as e:
            logger.error(f"Auto-index loop error: {e}")
            await asyncio.sleep(60)

# ==================== MULTI-SOURCE POSTER FETCHING ====================
async def get_poster_multi(title, session):
    """
    Fetch poster from multiple sources with fallback
    Sources: OMDB → TMDB → IMPAwards → JustWatch → Letterboxd → Custom
    """
    cache_key = title.lower().strip()
    
    # Check cache first
    if cache_key in movie_db['poster_cache']:
        cached, cached_time = movie_db['poster_cache'][cache_key]
        if (datetime.now() - cached_time).seconds < 600:  # 10 min cache
            return cached
    
    logger.info(f"🎨 Fetching poster: {title}")
    
    # 1. OMDB (Best quality, most reliable)
    for api_key in Config.OMDB_KEYS:
        try:
            url = f"http://www.omdbapi.com/?t={urllib.parse.quote(title)}&apikey={api_key}"
            async with session.get(url, timeout=5) as r:
                if r.status == 200:
                    data = await r.json()
                    if data.get('Response') == 'True' and data.get('Poster') != 'N/A':
                        result = {
                            'poster_url': data['Poster'].replace('http://', 'https://'),
                            'title': data.get('Title', title),
                            'year': data.get('Year', ''),
                            'rating': data.get('imdbRating', ''),
                            'source': 'OMDB',
                            'success': True
                        }
                        movie_db['poster_cache'][cache_key] = (result, datetime.now())
                        movie_db['stats']['omdb'] += 1
                        logger.info(f"✅ OMDB: {title}")
                        return result
        except Exception as e:
            logger.debug(f"OMDB failed: {e}")
            continue
    
    # 2. TMDB (Good quality, large database)
    for api_key in Config.TMDB_KEYS:
        try:
            url = "https://api.themoviedb.org/3/search/movie"
            params = {'api_key': api_key, 'query': title}
            async with session.get(url, params=params, timeout=5) as r:
                if r.status == 200:
                    data = await r.json()
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
                            logger.info(f"✅ TMDB: {title}")
                            return result
        except Exception as e:
            logger.debug(f"TMDB failed: {e}")
            continue
    
    # 3. IMPAwards (High quality movie posters)
    try:
        search_url = f"https://www.impawards.com/search/?q={title.replace(' ', '+')}"
        async with session.get(search_url, timeout=6) as r:
            if r.status == 200:
                html_content = await r.text()
                match = re.search(r'<img src="([^"]+posters[^"]+\.jpg)"', html_content)
                if match:
                    poster_url = match.group(1)
                    if not poster_url.startswith('http'):
                        poster_url = 'https://www.impawards.com' + poster_url
                    
                    result = {
                        'poster_url': poster_url,
                        'title': title,
                        'source': 'IMPAwards',
                        'success': True
                    }
                    movie_db['poster_cache'][cache_key] = (result, datetime.now())
                    movie_db['stats']['impawards'] += 1
                    logger.info(f"✅ IMPAwards: {title}")
                    return result
    except Exception as e:
        logger.debug(f"IMPAwards failed: {e}")
    
    # 4. JustWatch (Streaming service posters)
    try:
        slug = title.lower().replace(' ', '-').replace(':', '').replace("'", '')
        jw_url = f"https://apis.justwatch.com/content/titles/movie/{slug}/locale/en_IN"
        async with session.get(jw_url, timeout=5) as r:
            if r.status == 200:
                data = await r.json()
                if data.get('poster'):
                    result = {
                        'poster_url': f"https://images.justwatch.com{data['poster']}",
                        'title': data.get('title', title),
                        'source': 'JustWatch',
                        'success': True
                    }
                    movie_db['poster_cache'][cache_key] = (result, datetime.now())
                    movie_db['stats']['justwatch'] += 1
                    logger.info(f"✅ JustWatch: {title}")
                    return result
    except Exception as e:
        logger.debug(f"JustWatch failed: {e}")
    
    # 5. Letterboxd (Film community posters)
    try:
        slug = title.lower().replace(' ', '-').replace(':', '').replace("'", '')
        lb_url = f"https://letterboxd.com/film/{slug}/"
        async with session.get(lb_url, timeout=6, headers={'User-Agent': 'Mozilla/5.0'}) as r:
            if r.status == 200:
                html_content = await r.text()
                match = re.search(r'<meta property="og:image" content="([^"]+)"', html_content)
                if match:
                    result = {
                        'poster_url': match.group(1),
                        'title': title,
                        'source': 'Letterboxd',
                        'success': True
                    }
                    movie_db['poster_cache'][cache_key] = (result, datetime.now())
                    movie_db['stats']['letterboxd'] += 1
                    logger.info(f"✅ Letterboxd: {title}")
                    return result
    except Exception as e:
        logger.debug(f"Letterboxd failed: {e}")
    
    # 6. Custom SVG fallback
    movie_db['stats']['custom'] += 1
    logger.info(f"ℹ️ Custom poster: {title}")
    
    result = {
        'poster_url': f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(title)}",
        'title': title,
        'source': 'CUSTOM',
        'success': True
    }
    movie_db['poster_cache'][cache_key] = (result, datetime.now())
    return result

# ==================== SEARCH FROM MONGODB ====================
async def search_movies(query, limit=12, page=1):
    """Search movies from MongoDB with pagination"""
    if posts_col is None or files_col is None:
        logger.warning("⚠️ Database not ready")
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
    logger.info(f"🔍 MongoDB Search: '{query}' Page {page}")
    
    # Search posts
    post_results = []
    try:
        cursor = posts_col.find({'$text': {'$search': query}}).sort('date', -1).limit(50)
        async for doc in cursor:
            post_results.append({
                'title': doc['title'],
                'content': format_post(doc['content']),
                'channel': doc['channel_name'],
                'channel_id': doc['channel_id'],
                'message_id': doc['message_id'],
                'date': doc['date'].isoformat() if isinstance(doc['date'], datetime) else doc['date'],
                'is_new': doc.get('is_new', False),
                'has_file': False,
                'quality_options': {}
            })
        logger.info(f"  ✓ Posts: {len(post_results)}")
    except Exception as e:
        logger.error(f"  ✗ Posts search error: {e}")
    
    # Search files
    file_results = {}
    try:
        cursor = files_col.find({'$text': {'$search': query}}).limit(50)
        async for doc in cursor:
            title_key = doc['title'].lower()
            quality = doc['quality']
            
            if title_key not in file_results:
                file_results[title_key] = {
                    'title': doc['title'],
                    'channel': channel_name(doc['channel_id']),
                    'channel_id': doc['channel_id'],
                    'message_id': doc['message_id'],
                    'date': doc['date'].isoformat() if isinstance(doc['date'], datetime) else doc['date'],
                    'is_new': is_new(doc['date']) if doc.get('date') else False,
                    'has_file': True,
                    'quality_options': {},
                    'content': format_post(doc.get('caption', '')) or doc['title']
                }
            
            file_results[title_key]['quality_options'][quality] = {
                'file_id': doc['file_id'],
                'file_size': doc['file_size'],
                'file_name': doc['file_name'],
                'message_id': doc['message_id']
            }
        
        logger.info(f"  ✓ Files: {len(file_results)}")
    except Exception as e:
        logger.error(f"  ✗ Files search error: {e}")
    
    # Merge results
    results = {}
    
    for post in post_results:
        title_key = post['title'].lower()
        results[title_key] = post
    
    for title_key, file_data in file_results.items():
        if title_key in results:
            results[title_key]['has_file'] = True
            results[title_key]['quality_options'] = file_data['quality_options']
        else:
            results[title_key] = file_data
    
    # Sort and paginate
    final_results = list(results.values())
    final_results.sort(key=lambda x: (x['has_file'], x['date']), reverse=True)
    
    total = len(final_results)
    paginated = final_results[offset:offset + limit]
    
    logger.info(f"✅ Total: {total} | Showing: {len(paginated)}")
    
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
    """
    Get homepage movies with posters
    FIXED: No duplicates + All 5 poster sources
    """
    if posts_col is None:
        logger.warning("⚠️ Database not ready")
        return []
    
    logger.info("🏠 Loading homepage movies...")
    movies = []
    seen_titles = set()  # FIX: Track unique titles
    
    try:
        cursor = posts_col.find().sort('date', -1).limit(50)  # Get extra, then deduplicate
        
        async for doc in cursor:
            title_key = doc['title'].lower().strip()
            
            # FIX: Skip duplicates
            if title_key in seen_titles:
                logger.debug(f"⚠️ Skipping duplicate: {doc['title']}")
                continue
            
            seen_titles.add(title_key)
            
            movies.append({
                'title': doc['title'],
                'date': doc['date'].isoformat() if isinstance(doc['date'], datetime) else doc['date'],
                'is_new': doc.get('is_new', False)
            })
            
            # Stop at 24 unique movies
            if len(movies) >= 24:
                break
                
    except Exception as e:
        logger.error(f"❌ Load movies error: {e}")
    
    logger.info(f"📋 Unique movies loaded: {len(movies)}")
    logger.info(f"🎨 Fetching posters from multiple sources...")
    
    # Fetch posters in batches (5 at a time)
    async with aiohttp.ClientSession() as session:
        for i in range(0, len(movies), 5):
            batch = movies[i:i + 5]
            
            # Fetch posters concurrently
            posters = await asyncio.gather(
                *[get_poster_multi(movie['title'], session) for movie in batch],
                return_exceptions=True
            )
            
            # Assign posters to movies
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
                    # Fallback to custom poster
                    movie['poster_url'] = f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(movie['title'])}"
                    movie['poster_source'] = 'CUSTOM'
                    movie['has_poster'] = True
                    movie_db['stats']['failed'] += 1
            
            # Small delay between batches
            await asyncio.sleep(0.2)
    
    logger.info(f"✅ Loaded {len(movies)} unique movies | Stats: {movie_db['stats']}")
    return movies

# ==================== API ROUTES ====================
@app.route('/')
async def root():
    """Health check and stats"""
    total_posts = await posts_col.count_documents({}) if posts_col is not None else 0
    total_files = await files_col.count_documents({}) if files_col is not None else 0
    
    return jsonify({
        'status': 'healthy',
        'service': 'SK4FiLM - MongoDB Auto-Index System',
        'version': '2.0',
        'bot': f'@{Config.BOT_USERNAME}',
        'poster_sources': ['OMDB', 'TMDB', 'IMPAwards', 'JustWatch', 'Letterboxd', 'Custom'],
        'poster_stats': movie_db['stats'],
        'database': {
            'posts_indexed': total_posts,
            'files_indexed': total_files
        },
        'bot_status': 'online' if bot_started else 'starting'
    })

@app.route('/health')
async def health():
    """Simple health check"""
    return jsonify({
        'status': 'ok' if bot_started else 'starting',
        'bot_started': bot_started
    })

@app.route('/api/movies')
async def api_movies():
    """Get homepage movies with posters"""
    try:
        if not bot_started:
            return jsonify({'status': 'error', 'message': 'Service starting...'}), 503
        
        movies = await get_home_movies()
        
        return jsonify({
            'status': 'success',
            'movies': movies,
            'total': len(movies),
            'bot_username': Config.BOT_USERNAME,
            'poster_stats': movie_db['stats']
        })
        
    except Exception as e:
        logger.error(f"❌ API movies error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/search')
async def api_search():
    """Search movies with pagination"""
    try:
        query = request.args.get('query', '').strip()
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 12))
        
        logger.info(f"📱 API Search: query='{query}' page={page} limit={limit}")
        
        if not query:
            return jsonify({'status': 'error', 'message': 'Query parameter required'}), 400
        
        if not bot_started:
            return jsonify({'status': 'error', 'message': 'Service starting...'}), 503
        
        result = await search_movies(query, limit, page)
        
        logger.info(f"📤 Returning {len(result['results'])} results")
        
        return jsonify({
            'status': 'success',
            'query': query,
            'results': result['results'],
            'pagination': result['pagination'],
            'bot_username': Config.BOT_USERNAME
        })
        
    except Exception as e:
        logger.error(f"❌ API search error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/poster')
async def api_poster():
    """
    Proxy poster images or generate custom SVG
    """
    # Proxy mode
    url = request.args.get('url', '').strip()
    if url and url.startswith('http'):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'}) as r:
                    if r.status == 200:
                        img_data = await r.read()
                        content_type = r.headers.get('content-type', 'image/jpeg')
                        return Response(
                            img_data,
                            mimetype=content_type,
                            headers={'Cache-Control': 'public, max-age=7200'}
                        )
        except:
            pass
    
    # Custom SVG generation
    title = request.args.get('title', 'Movie')
    display_title = title[:18] + "..." if len(title) > 18 else title
    
    # Color schemes
    colors = [
        ('#667eea', '#764ba2'),
        ('#f093fb', '#f5576c'),
        ('#4facfe', '#00f2fe'),
        ('#43e97b', '#38f9d7'),
        ('#fa709a', '#fee140')
    ]
    color = colors[hash(title) % len(colors)]
    
    svg = f'''<svg width="300" height="450" xmlns="http://www.w3.org/2000/svg">
    <defs>
        <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" style="stop-color:{color[0]}"/>
            <stop offset="100%" style="stop-color:{color[1]}"/>
        </linearGradient>
    </defs>
    <rect width="100%" height="100%" fill="url(#bg)" rx="20"/>
    <circle cx="150" cy="180" r="50" fill="rgba(255,255,255,0.2)"/>
    <text x="50%" y="200" text-anchor="middle" fill="#fff" font-size="50">🎬</text>
    <text x="50%" y="270" text-anchor="middle" fill="#fff" font-size="18" font-weight="bold">{html.escape(display_title)}</text>
    <rect x="50" y="380" width="200" height="40" rx="20" fill="rgba(0,0,0,0.3)"/>
    <text x="50%" y="405" text-anchor="middle" fill="#fff" font-size="18" font-weight="700">SK4FiLM</text>
</svg>'''
    
    return Response(
        svg,
        mimetype='image/svg+xml',
        headers={'Cache-Control': 'public, max-age=3600'}
    )

# ==================== BOT HANDLERS ====================
async def setup_bot():
    """Setup bot command handlers"""
    
    @bot.on_message(filters.command("start") & filters.private)
    async def start_handler(client, message):
        user_id = message.from_user.id
        
        # File delivery mode
        if len(message.command) > 1:
            file_id = message.command[1]
            
            # Check force subscribe
            if not await check_force_sub(user_id):
                try:
                    channel = await bot.get_chat(Config.FORCE_SUB_CHANNEL)
                    link = f"https://t.me/{channel.username}" if channel.username else f"https://t.me/c/{str(Config.FORCE_SUB_CHANNEL)[4:]}/1"
                except:
                    link = f"https://t.me/c/{str(Config.FORCE_SUB_CHANNEL)[4:]}/1"
                
                await message.reply_text(
                    "⚠️ **Join Our Channel First**",
                    reply_markup=InlineKeyboardMarkup([[
                        InlineKeyboardButton("📢 Join Channel", url=link)
                    ]])
                )
                return
            
            # Get file from database
            file_doc = await files_col.find_one({'file_id': file_id}) if files_col is not None else None
            
            if not file_doc:
                await message.reply_text(
                    "❌ **File Not Found**\n\nThe file may have been removed or is unavailable.",
                    reply_markup=InlineKeyboardMarkup([[
                        InlineKeyboardButton("🔍 Search Again", url=Config.WEBSITE_URL)
                    ]])
                )
                return
            
            try:
                # Send processing message
                progress_msg = await message.reply_text(
                    f"⏳ **Sending File...**\n\n"
                    f"📁 {file_doc['file_name']}\n"
                    f"📊 Quality: {file_doc['quality']}\n"
                    f"📦 Size: {format_size(file_doc['file_size'])}"
                )
                
                # Forward file from channel
                if User:
                    file_message = await User.get_messages(
                        file_doc['channel_id'],
                        file_doc['message_id']
                    )
                    
                    sent_file = await file_message.copy(user_id)
                    await progress_msg.delete()
                    
                    # Send success message
                    success_msg = await message.reply_text(
                        f"✅ **File Sent Successfully!**\n\n"
                        f"🎬 {file_doc['title']}\n"
                        f"📊 Quality: {file_doc['quality']}\n"
                        f"📦 Size: {format_size(file_doc['file_size'])}\n\n"
                        f"⚠️ **This file will be automatically deleted in {Config.AUTO_DELETE_TIME // 60} minutes**"
                    )
                    
                    logger.info(f"✅ File delivered: {file_doc['title']} ({file_doc['quality']}) → User {user_id}")
                    
                    # Auto-delete after timeout
                    if Config.AUTO_DELETE_TIME > 0:
                        await asyncio.sleep(Config.AUTO_DELETE_TIME)
                        try:
                            await sent_file.delete()
                            await success_msg.edit_text(
                                "🗑️ **File Deleted**\n\n"
                                "This file has been automatically deleted for privacy and storage management."
                            )
                        except:
                            pass
                        
            except Exception as e:
                logger.error(f"❌ File delivery error: {e}")
                await message.reply_text(
                    "❌ **Error Sending File**\n\n"
                    "Please try again later or contact support."
                )
            
            return
        
        # Welcome message
        await message.reply_text(
            "🎬 **Welcome to SK4FiLM Bot**\n\n"
            "📌 **How to use:**\n"
            "1. Visit our website\n"
            "2. Search for movies\n"
            "3. Select quality\n"
            "4. Get file here\n\n"
            "⚡ **Features:**\n"
            "• Multiple quality options\n"
            "• Fast delivery\n"
            "• 5 poster sources\n"
            "• MongoDB powered search",
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("🌐 Visit Website", url=Config.WEBSITE_URL)
            ]])
        )
    
    @bot.on_message(filters.text & filters.private & ~filters.command(['start', 'stats', 'index']))
    async def text_handler(client, message):
        await message.reply_text(
            "👋 **Please use our website to search for movies**",
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("🌐 Search Movies", url=Config.WEBSITE_URL)
            ]])
        )
    
    @bot.on_message(filters.command("index") & filters.user(Config.ADMIN_IDS))
    async def index_handler(client, message):
        await message.reply_text("🔄 **Starting manual indexing...**")
        
        await index_all_files()
        
        total_posts = await posts_col.count_documents({}) if posts_col is not None else 0
        total_files = await files_col.count_documents({}) if files_col is not None else 0
        
        await message.reply_text(
            f"✅ **Indexing Complete**\n\n"
            f"📝 Posts: {total_posts}\n"
            f"📁 Files: {total_files}"
        )
    
    @bot.on_message(filters.command("stats") & filters.user(Config.ADMIN_IDS))
    async def stats_handler(client, message):
        total_posts = await posts_col.count_documents({}) if posts_col is not None else 0
        total_files = await files_col.count_documents({}) if files_col is not None else 0
        
        await message.reply_text(
            f"📊 **SK4FiLM Statistics**\n\n"
            f"📝 **Posts Indexed:** {total_posts}\n"
            f"📁 **Files Indexed:** {total_files}\n\n"
            f"🖼️ **Poster Sources:**\n"
            f"  • OMDB: {movie_db['stats']['omdb']}\n"
            f"  • TMDB: {movie_db['stats']['tmdb']}\n"
            f"  • IMPAwards: {movie_db['stats']['impawards']}\n"
            f"  • JustWatch: {movie_db['stats']['justwatch']}\n"
            f"  • Letterboxd: {movie_db['stats']['letterboxd']}\n"
            f"  • Custom: {movie_db['stats']['custom']}\n"
            f"  • Failed: {movie_db['stats']['failed']}\n\n"
            f"🤖 **Bot Status:** {'✅ Online' if bot_started else '❌ Offline'}"
        )

# ==================== INITIALIZATION ====================
async def init():
    """Initialize all services"""
    global User, bot, bot_started, auto_index_task
    
    try:
        logger.info("🔄 Initializing services...")
        
        # Initialize MongoDB
        await init_mongodb()
        
        # Initialize Pyrogram clients
        logger.info("🔌 Starting Telegram clients...")
        
        User = Client(
            "sk4film_user",
            api_id=Config.API_ID,
            api_hash=Config.API_HASH,
            session_string=Config.USER_SESSION_STRING,
            workdir="/tmp",
            sleep_threshold=60
        )
        
        bot = Client(
            "sk4film_bot",
            api_id=Config.API_ID,
            api_hash=Config.API_HASH,
            bot_token=Config.BOT_TOKEN,
            workdir="/tmp",
            sleep_threshold=60
        )
        
        await User.start()
        user_me = await User.get_me()
        logger.info(f"✅ User client: {user_me.first_name}")
        
        await bot.start()
        bot_me = await bot.get_me()
        logger.info(f"✅ Bot client: @{bot_me.username}")
        
        # Setup bot handlers
        await setup_bot()
        
        bot_started = True
        
        # Initial indexing
        logger.info("📥 Running initial indexing...")
        await index_all_files()
        
        # Start auto-index background task
        auto_index_task = asyncio.create_task(auto_index_loop())
        logger.info("✅ Auto-index loop started")
        
        logger.info("✅ All services initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Initialization error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main entry point"""
    logger.info("=" * 70)
    logger.info("🚀 SK4FiLM - Complete System")
    logger.info("=" * 70)
    logger.info(f"🌐 Website: {Config.WEBSITE_URL}")
    logger.info(f"📡 Backend: {Config.BACKEND_URL}")
    logger.info(f"🤖 Bot: @{Config.BOT_USERNAME}")
    logger.info(f"🖼️ Poster Sources: 5 (OMDB, TMDB, IMPAwards, JustWatch, Letterboxd)")
    logger.info(f"🗄️ Database: MongoDB")
    logger.info("=" * 70)
    
    # Initialize services
    await init()
    
    # Configure Hypercorn
    config = HyperConfig()
    config.bind = [f"0.0.0.0:{Config.WEB_SERVER_PORT}"]
    config.loglevel = "warning"
    config.accesslog = "-"
    
    logger.info(f"🌐 Starting web server on port {Config.WEB_SERVER_PORT}...")
    
    # Start server
    await serve(app, config)

if __name__ == "__main__":
    asyncio.run(main())
