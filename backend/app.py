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
    
    # PRIORITY: Main channel first for recent posts
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
        await posts_col.create_index([("indexed_at", -1)])
        await posts_col.create_index([("is_new", -1)])
        await posts_col.create_index([("normalized_title", 1)])
        await posts_col.create_index([("channel_id", 1), ("message_id", 1)], unique=True)
        
        await files_col.create_index([("title", "text")])
        await files_col.create_index([("normalized_title", 1)])
        await files_col.create_index([("message_id", 1), ("channel_id", 1)], unique=True)
        
        logger.info("✅ MongoDB connected and indexed")
        return True
    except Exception as e:
        logger.error(f"❌ MongoDB error: {e}")
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
    """Enhanced title extraction"""
    if not text or len(text) < 10:
        return None
    try:
        # Clean text
        clean = re.sub(r'[^\w\s\(\)\-\.\n:]', ' ', text)
        lines = [l.strip() for l in clean.split('\n') if l.strip()]
        
        if not lines:
            return None
        
        first_line = lines[0]
        
        # Pattern 1: 🎬 Movie Name
        m = re.search(r'🎬\s*([^\n\-\(]{3,60})', first_line)
        if m:
            title = m.group(1).strip()
            title = re.sub(r'\s+', ' ', title)
            if 3 <= len(title) <= 60:
                return title
        
        # Pattern 2: Movie Name (2024)
        m = re.search(r'^([^\(\n]{3,60})\s*\(\d{4}\)', first_line)
        if m:
            title = m.group(1).strip()
            if 3 <= len(title) <= 60:
                return title
        
        # Pattern 3: Movie Name - Quality
        m = re.search(r'^([^\-\n]{3,60})\s*-', first_line)
        if m:
            title = m.group(1).strip()
            title = re.sub(r'\s+', ' ', title)
            if 3 <= len(title) <= 60:
                return title
        
        # Pattern 4: First line if looks like title
        if len(first_line) >= 3 and len(first_line) <= 60:
            # Remove quality markers
            title = re.sub(r'\b(480p|720p|1080p|2160p|4k|hevc|x264|x265)\b', '', first_line, flags=re.IGNORECASE)
            title = re.sub(r'\s+', ' ', title).strip()
            if 3 <= len(title) <= 60:
                return title
    except Exception as e:
        logger.error(f"Title extraction error: {e}")
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

async def check_force_sub(user_id):
    try:
        logger.info(f"🔍 Checking subscription for user {user_id}")
        member = await bot.get_chat_member(Config.FORCE_SUB_CHANNEL, user_id)
        status = member.status in ["member", "administrator", "creator"]
        logger.info(f"  {'✅' if status else '❌'} User status: {member.status}")
        return status
    except UserNotParticipant:
        logger.info(f"  ❌ User not in channel")
        return False
    except (ChatAdminRequired, ChannelPrivate):
        logger.warning(f"  ⚠️ Bot doesn't have permission")
        return True
    except Exception as e:
        logger.error(f"  ❌ Force sub error: {e}")
        return True

async def index_all_files():
    """FIXED: Proper indexing with priority to main channel"""
    if not User or not bot_started or posts_col is None or files_col is None:
        logger.warning("⚠️ Cannot index - services not ready")
        return
    
    logger.info("="*70)
    logger.info("📥 STARTING FULL INDEXING...")
    logger.info("="*70)
    
    indexed_posts = 0
    indexed_files = 0
    
    try:
        # Index TEXT channels (BOTH) with more messages
        for channel_id in Config.TEXT_CHANNEL_IDS:
            try:
                count = 0
                channel_name_str = channel_name(channel_id)
                logger.info(f"📝 Indexing {channel_name_str} ({channel_id})...")
                
                # Increased limit to 200 for better coverage
                async for msg in User.get_chat_history(channel_id, limit=200):
                    if msg.text and len(msg.text) > 15:  # Minimum text length
                        title = extract_title_smart(msg.text)
                        if title:
                            try:
                                result = await posts_col.update_one(
                                    {'channel_id': channel_id, 'message_id': msg.id},
                                    {'$set': {
                                        'title': title,
                                        'normalized_title': normalize_title(title),
                                        'content': msg.text[:1000],  # Limit content size
                                        'channel_id': channel_id,
                                        'channel_name': channel_name_str,
                                        'message_id': msg.id,
                                        'date': msg.date,
                                        'is_new': is_new(msg.date) if msg.date else False,
                                        'indexed_at': datetime.now()
                                    }},
                                    upsert=True
                                )
                                count += 1
                                indexed_posts += 1
                                
                                if count % 50 == 0:
                                    logger.info(f"    ⏳ Progress: {count} posts...")
                                    
                            except Exception as e:
                                logger.error(f"    ❌ Post error: {e}")
                
                logger.info(f"  ✅ Indexed {count} posts from {channel_name_str}")
                
            except Exception as e:
                logger.error(f"  ❌ Channel error: {e}")
        
        # Index FILES channel
        try:
            count = 0
            logger.info(f"📁 Indexing Files Channel ({Config.FILE_CHANNEL_ID})...")
            
            async for msg in bot.get_chat_history(Config.FILE_CHANNEL_ID, limit=400):
                if msg.document or msg.video:
                    title = extract_title_from_file(msg)
                    if title:
                        file_id = msg.document.file_id if msg.document else msg.video.file_id
                        file_size = msg.document.file_size if msg.document else (msg.video.file_size if msg.video else 0)
                        file_name = msg.document.file_name if msg.document else (msg.video.file_name if msg.video else 'video.mp4')
                        quality = detect_quality(file_name)
                        
                        try:
                            await files_col.update_one(
                                {'channel_id': Config.FILE_CHANNEL_ID, 'message_id': msg.id},
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
                            
                            if count % 100 == 0:
                                logger.info(f"    ⏳ Progress: {count} files...")
                                
                        except Exception as e:
                            logger.error(f"    ❌ File error: {e}")
            
            logger.info(f"  ✅ Indexed {count} files")
            
        except Exception as e:
            logger.error(f"  ❌ Files error: {e}")
        
        logger.info("="*70)
        logger.info(f"✅ INDEXING COMPLETE")
        logger.info(f"📝 Posts indexed: {indexed_posts}")
        logger.info(f"📁 Files indexed: {indexed_files}")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"❌ Indexing failed: {e}")

async def auto_index_loop():
    """Auto-index loop"""
    while True:
        try:
            await asyncio.sleep(Config.AUTO_INDEX_INTERVAL)
            logger.info("🔄 Auto-indexing triggered...")
            await index_all_files()
        except Exception as e:
            logger.error(f"Auto-index error: {e}")
            await asyncio.sleep(60)

async def get_poster_aggressive(title, session):
    """Aggressive poster fetching"""
    ck = title.lower().strip()
    
    if ck in movie_db['poster_cache']:
        c, ct = movie_db['poster_cache'][ck]
        if (datetime.now() - ct).seconds < 600:
            return c
    
    # OMDB
    for api_key in Config.OMDB_KEYS:
        try:
            url = f"http://www.omdbapi.com/?t={urllib.parse.quote(title)}&apikey={api_key}"
            async with session.get(url, timeout=5) as r:
                if r.status == 200:
                    data = await r.json()
                    if data.get('Response') == 'True' and data.get('Poster') != 'N/A':
                        poster_url = data['Poster'].replace('http://', 'https://')
                        res = {'poster_url': poster_url, 'source': 'OMDB', 'success': True}
                        movie_db['poster_cache'][ck] = (res, datetime.now())
                        movie_db['stats']['omdb'] += 1
                        return res
        except:
            continue
    
    # TMDB
    for api_key in Config.TMDB_KEYS:
        try:
            url = "https://api.themoviedb.org/3/search/movie"
            params = {'api_key': api_key, 'query': title}
            async with session.get(url, params=params, timeout=5) as r:
                if r.status == 200:
                    data = await r.json()
                    if data.get('results') and len(data['results']) > 0:
                        poster_path = data['results'][0].get('poster_path')
                        if poster_path:
                            poster_url = f"https://image.tmdb.org/t/p/w780{poster_path}"
                            res = {'poster_url': poster_url, 'source': 'TMDB', 'success': True}
                            movie_db['poster_cache'][ck] = (res, datetime.now())
                            movie_db['stats']['tmdb'] += 1
                            return res
        except:
            continue
    
    # Fallback
    movie_db['stats']['custom'] += 1
    res = {'poster_url': f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(title)}", 'source': 'CUSTOM', 'success': True}
    movie_db['poster_cache'][ck] = (res, datetime.now())
    return res

async def direct_search_channels(query, limit=100):
    """Enhanced direct search - BOTH channels"""
    if not User or not bot:
        return {'posts': [], 'files': []}
    
    logger.info(f"🔍 Direct search: '{query}'")
    query_lower = query.lower()
    posts_found = []
    files_found = []
    
    # Search BOTH text channels with increased limit
    for channel_id in Config.TEXT_CHANNEL_IDS:
        try:
            cname = channel_name(channel_id)
            logger.info(f"  📝 Searching {cname}...")
            count = 0
            
            async for msg in User.search_messages(channel_id, query=query, limit=limit):
                if msg.text and len(msg.text) > 15:
                    title = extract_title_smart(msg.text)
                    if title and query_lower in title.lower():
                        posts_found.append({
                            'title': title,
                            'normalized_title': normalize_title(title),
                            'content': msg.text,
                            'channel_name': cname,
                            'channel_id': channel_id,
                            'message_id': msg.id,
                            'date': msg.date,
                            'is_new': is_new(msg.date) if msg.date else False
                        })
                        count += 1
            
            logger.info(f"    ✅ Found {count} posts in {cname}")
            
        except Exception as e:
            logger.error(f"    ❌ Error: {e}")
    
    # Search files
    try:
        logger.info(f"  📁 Searching files...")
        count = 0
        
        async for msg in bot.search_messages(Config.FILE_CHANNEL_ID, query=query, limit=limit*2):
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
        
        logger.info(f"    ✅ Found {count} files")
        
    except Exception as e:
        logger.error(f"    ❌ Error: {e}")
    
    logger.info(f"✅ Search complete: {len(posts_found)} posts, {len(files_found)} files")
    return {'posts': posts_found, 'files': files_found}

async def search_movies(query, limit=12, page=1):
    """Enhanced search with proper merging"""
    offset = (page - 1) * limit
    logger.info(f"🔍 SEARCH: '{query}' | Page: {page}")
    
    posts_dict = {}
    files_dict = {}
    
    # 1. Direct Pyrogram search (most reliable)
    logger.info("📡 Direct Pyrogram search...")
    direct = await direct_search_channels(query, limit=150)
    
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
        
        if quality not in files_dict[norm_title]['quality_options']:
            files_dict[norm_title]['quality_options'][quality] = {
                'file_id': f"{file.get('channel_id', Config.FILE_CHANNEL_ID)}_{file.get('message_id')}_{quality}",
                'file_size': file['file_size'],
                'file_name': file['file_name']
            }
    
    # 2. MongoDB backup
    if posts_col is not None and files_col is not None:
        try:
            logger.info("💾 MongoDB backup search...")
            
            cursor = posts_col.find({'$text': {'$search': query}}).sort('date', -1).limit(200)
            mongo_posts = 0
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
                    mongo_posts += 1
            
            cursor = files_col.find({'$text': {'$search': query}}).limit(300)
            mongo_files = 0
            async for doc in cursor:
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
                    mongo_files += 1
            
            logger.info(f"  ✓ MongoDB: +{mongo_posts} posts, +{mongo_files} files")
        except Exception as e:
            logger.error(f"  ✗ MongoDB error: {e}")
    
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
                'quality_options': file_data['quality_options']
            }
    
    results_list = list(merged.values())
    results_list.sort(key=lambda x: (not x.get('is_new', False), not x['has_file'], x['date']), reverse=True)
    
    total = len(results_list)
    paginated = results_list[offset:offset + limit]
    
    logger.info(f"✅ Total: {total} | Page {page}: {len(paginated)} items")
    
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
    """FIXED: Load recent movies with PRIORITY from main channel"""
    if posts_col is None:
        return []
    
    logger.info("🏠 Loading recent movies for home page...")
    movies = []
    seen = set()
    
    try:
        # PRIORITY: Get from main channel first (-1001891090100)
        main_channel_id = -1001891090100
        logger.info(f"  📌 Priority: {channel_name(main_channel_id)}")
        
        cursor = posts_col.find({'channel_id': main_channel_id}).sort('indexed_at', -1).limit(40)
        main_count = 0
        async for doc in cursor:
            tk = doc['title'].lower().strip()
            if tk not in seen:
                seen.add(tk)
                movies.append({
                    'title': doc['title'], 
                    'date': doc['date'].isoformat() if isinstance(doc['date'], datetime) else doc['date'], 
                    'is_new': doc.get('is_new', False),
                    'channel': doc.get('channel_name', 'SK4FiLM Main')
                })
                main_count += 1
        
        logger.info(f"  ✅ Main channel: {main_count} movies")
        
        # Then get from other channels if needed
        if len(movies) < 24:
            logger.info(f"  📝 Getting more from other channels...")
            cursor = posts_col.find({'channel_id': {'$ne': main_channel_id}}).sort('indexed_at', -1).limit(30)
            other_count = 0
            async for doc in cursor:
                tk = doc['title'].lower().strip()
                if tk not in seen:
                    seen.add(tk)
                    movies.append({
                        'title': doc['title'], 
                        'date': doc['date'].isoformat() if isinstance(doc['date'], datetime) else doc['date'], 
                        'is_new': doc.get('is_new', False),
                        'channel': doc.get('channel_name', 'SK4FiLM')
                    })
                    other_count += 1
                    if len(movies) >= 24:
                        break
            
            logger.info(f"  ✅ Other channels: {other_count} movies")
        
        logger.info(f"  ✓ Total unique movies: {len(movies)}")
        
    except Exception as e:
        logger.error(f"  ✗ Error: {e}")
    
    # Fetch posters
    if movies:
        logger.info(f"🎨 Fetching posters for {len(movies)} movies...")
        async with aiohttp.ClientSession() as session:
            for i in range(0, len(movies), 3):
                batch = movies[i:i + 3]
                tasks = [get_poster_aggressive(m['title'], session) for m in batch]
                posters = await asyncio.gather(*tasks, return_exceptions=True)
                
                for movie, poster_result in zip(batch, posters):
                    if isinstance(poster_result, dict) and poster_result.get('success'):
                        movie['poster_url'] = poster_result['poster_url']
                        movie['poster_source'] = poster_result['source']
                        movie['has_poster'] = True
                    else:
                        movie['poster_url'] = f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(movie['title'])}"
                        movie['poster_source'] = 'FALLBACK'
                        movie['has_poster'] = True
                
                await asyncio.sleep(0.1)
    
    logger.info(f"✅ Home page ready with {len(movies)} movies")
    return movies

@app.route('/')
async def root():
    tp = await posts_col.count_documents({}) if posts_col is not None else 0
    tf = await files_col.count_documents({}) if files_col is not None else 0
    tn = await posts_col.count_documents({'is_new': True}) if posts_col is not None else 0
    
    ch_stats = {}
    for cid in Config.TEXT_CHANNEL_IDS:
        cnt = await posts_col.count_documents({'channel_id': cid}) if posts_col is not None else 0
        ch_stats[channel_name(cid)] = cnt
    
    return jsonify({
        'status': 'healthy',
        'service': 'SK4FiLM v2.6 - Fixed Channel Priority',
        'database': {
            'total_posts': tp, 
            'total_files': tf, 
            'new_posts': tn, 
            'channel_wise': ch_stats
        },
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
            return jsonify({'status': 'error', 'message': 'Bot starting...'}), 503
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
            return jsonify({'status': 'error', 'message': 'Query required'}), 400
        if not bot_started:
            return jsonify({'status': 'error', 'message': 'Bot starting...'}), 503
        
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
            logger.info(f"="*60)
            logger.info(f"📥 File request | User: {uid} | ID: {fid}")
            
            is_subscribed = await check_force_sub(uid)
            
            if not is_subscribed:
                try:
                    ch = await bot.get_chat(Config.FORCE_SUB_CHANNEL)
                    if ch.username:
                        invite_link = f"https://t.me/{ch.username}"
                    else:
                        invite_link = await bot.export_chat_invite_link(Config.FORCE_SUB_CHANNEL)
                except:
                    invite_link = f"https://t.me/c/{str(Config.FORCE_SUB_CHANNEL)[4:]}/1"
                
                await message.reply_text(
                    "⚠️ **Join channel first**\n\nAfter joining, try again",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("📢 Join Channel", url=invite_link)],
                        [InlineKeyboardButton("🔄 Try Again", url=f"https://t.me/{Config.BOT_USERNAME}?start={fid}")]
                    ])
                )
                return
            
            try:
                parts = fid.split('_')
                if len(parts) >= 2:
                    channel_id = int(parts[0])
                    message_id = int(parts[1])
                    quality = parts[2] if len(parts) > 2 else "Unknown"
                    logger.info(f"✅ Parsed | Channel: {channel_id} | Msg: {message_id} | Quality: {quality}")
                else:
                    raise ValueError("Invalid file ID format")
            except Exception as e:
                logger.error(f"❌ Parse error: {e}")
                await message.reply_text(
                    "❌ **Invalid file ID**",
                    reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔍 Search Movies", url=Config.WEBSITE_URL)]])
                )
                return
            
            try:
                pm = await message.reply_text(f"⏳ **Sending [{quality}]...**")
                
                file_message = await bot.get_messages(channel_id, message_id)
                
                if not file_message or (not file_message.document and not file_message.video):
                    raise Exception("No file found")
                
                if file_message.document:
                    file_name = file_message.document.file_name
                    file_size = file_message.document.file_size
                    file_id = file_message.document.file_id
                    sent = await bot.send_document(
                        uid, file_id, 
                        caption=f"🎬 **{quality}**\n📦 {format_size(file_size)}"
                    )
                else:
                    file_name = file_message.video.file_name or "video.mp4"
                    file_size = file_message.video.file_size
                    file_id = file_message.video.file_id
                    sent = await bot.send_video(
                        uid, file_id, 
                        caption=f"🎬 **{quality}**\n📦 {format_size(file_size)}"
                    )
                
                await pm.delete()
                
                success = await message.reply_text(
                    f"✅ **File sent successfully!**\n\n"
                    f"📁 `{file_name}`\n"
                    f"📊 {quality}\n"
                    f"📦 {format_size(file_size)}\n\n"
                    f"⚠️ **Auto-delete in {Config.AUTO_DELETE_TIME//60} minutes**"
                )
                
                logger.info(f"✅ SUCCESS: {file_name}")
                logger.info(f"="*60)
                
                if Config.AUTO_DELETE_TIME > 0:
                    await asyncio.sleep(Config.AUTO_DELETE_TIME)
                    try:
                        await sent.delete()
                        await success.edit_text("🗑️ **File deleted after timeout**")
                    except:
                        pass
                    
            except Exception as e:
                logger.error(f"❌ Send error: {e}")
                await pm.edit_text(f"❌ **Failed to send file**\n\n`{str(e)}`")
            
            return
        
        await message.reply_text(
            "🎬 **Welcome to SK4FiLM Bot**\n\n"
            "Search and download movies from our website",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🌐 Visit Website", url=Config.WEBSITE_URL)]])
        )
    
    @bot.on_message(filters.text & filters.private & ~filters.command(['start', 'stats', 'index']))
    async def text_handler(client, message):
        await message.reply_text(
            "👋 Please use our website to search movies",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🌐 Go to Website", url=Config.WEBSITE_URL)]])
        )
    
    @bot.on_message(filters.command("index") & filters.user(Config.ADMIN_IDS))
    async def index_handler(client, message):
        msg = await message.reply_text("🔄 **Starting indexing...**")
        await index_all_files()
        
        tp = await posts_col.count_documents({}) if posts_col is not None else 0
        tf = await files_col.count_documents({}) if files_col is not None else 0
        tn = await posts_col.count_documents({'is_new': True}) if posts_col is not None else 0
        
        ch_stats = []
        for cid in Config.TEXT_CHANNEL_IDS:
            cnt = await posts_col.count_documents({'channel_id': cid}) if posts_col is not None else 0
            ch_stats.append(f"• {channel_name(cid)}: **{cnt}** posts")
        
        await msg.edit_text(
            f"✅ **Indexing Complete!**\n\n"
            f"📝 Total Posts: **{tp}**\n"
            f"📁 Total Files: **{tf}**\n"
            f"🆕 New Posts: **{tn}**\n\n"
            f"**Channel Breakdown:**\n" + "\n".join(ch_stats)
        )
    
    @bot.on_message(filters.command("stats") & filters.user(Config.ADMIN_IDS))
    async def stats_handler(client, message):
        tp = await posts_col.count_documents({}) if posts_col is not None else 0
        tf = await files_col.count_documents({}) if files_col is not None else 0
        tn = await posts_col.count_documents({'is_new': True}) if posts_col is not None else 0
        
        ch_stats = []
        for cid in Config.TEXT_CHANNEL_IDS:
            cnt = await posts_col.count_documents({'channel_id': cid}) if posts_col is not None else 0
            ch_stats.append(f"• {channel_name(cid)}: **{cnt}** posts")
        
        qualities = {}
        if files_col is not None:
            async for doc in files_col.find({}, {'quality': 1}):
                q = doc.get('quality', 'Unknown')
                qualities[q] = qualities.get(q, 0) + 1
        
        quality_stats = [f"• {q}: **{count}** files" for q, count in sorted(qualities.items())]
        
        stats_text = (
            f"📊 **System Statistics**\n\n"
            f"📝 **Total Posts:** {tp}\n"
            f"📁 **Total Files:** {tf}\n"
            f"🆕 **New Posts:** {tn}\n\n"
            f"**Channel Breakdown:**\n" + "\n".join(ch_stats) + "\n\n"
            f"**Quality Distribution:**\n" + "\n".join(quality_stats) + "\n\n"
            f"**Poster Stats:**\n"
            f"• OMDB: **{movie_db['stats']['omdb']}**\n"
            f"• TMDB: **{movie_db['stats']['tmdb']}**\n"
            f"• Custom: **{movie_db['stats']['custom']}**"
        )
        
        await message.reply_text(stats_text)

async def init():
    global User, bot, bot_started, auto_index_task
    try:
        logger.info("🔄 Initializing SK4FiLM Bot...")
        await init_mongodb()
        
        User = Client(
            "user", 
            api_id=Config.API_ID, 
            api_hash=Config.API_HASH, 
            session_string=Config.USER_SESSION_STRING, 
            workdir="/tmp", 
            sleep_threshold=60
        )
        
        bot = Client(
            "bot", 
            api_id=Config.API_ID, 
            api_hash=Config.API_HASH, 
            bot_token=Config.BOT_TOKEN, 
            workdir="/tmp", 
            sleep_threshold=60
        )
        
        await User.start()
        await bot.start()
        await setup_bot()
        
        me = await bot.get_me()
        logger.info(f"✅ Bot started: @{me.username}")
        bot_started = True
        
        logger.info("📥 Starting initial indexing...")
        await index_all_files()
        
        logger.info("🔄 Starting auto-index loop...")
        auto_index_task = asyncio.create_task(auto_index_loop())
        
        return True
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        return False

async def main():
    logger.info("="*70)
    logger.info("🚀 SK4FiLM v2.6 - Fixed Channel Priority + Enhanced Search")
    logger.info("="*70)
    
    await init()
    
    cfg = HyperConfig()
    cfg.bind = [f"0.0.0.0:{Config.WEB_SERVER_PORT}"]
    cfg.loglevel = "warning"
    
    logger.info(f"🌐 Web server starting on port {Config.WEB_SERVER_PORT}")
    logger.info("="*70)
    
    await serve(app, cfg)

if __name__ == "__main__":
    asyncio.run(main())
