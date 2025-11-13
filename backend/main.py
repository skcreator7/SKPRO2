import asyncio
import os
import logging
from pyrogram import Client, errors
from quart import Quart, jsonify, request, Response, redirect
from hypercorn.asyncio import serve
from hypercorn.config import Config as HyperConfig
import html
import re
from datetime import datetime, timedelta
import math
import aiohttp
import urllib.parse
import json
import time
import uuid
from motor.motor_asyncio import AsyncIOMotorClient

class Config:
    API_ID = int(os.environ.get("API_ID", "0"))
    API_HASH = os.environ.get("API_HASH", "")
    USER_SESSION_STRING = os.environ.get("USER_SESSION_STRING", "")
    
    # TEXT CHANNELS - For text posts
    TEXT_CHANNEL_IDS = [
        int(x) for x in os.environ.get("TEXT_CHANNEL_IDS", "-1001891090100,-1002024811395").split(",")
    ]
    
    # FILE CHANNELS - For files with different qualities
    FILE_CHANNEL_480P = int(os.environ.get("FILE_CHANNEL_480P", "-1001768249569"))
    FILE_CHANNEL_720P = int(os.environ.get("FILE_CHANNEL_720P", "-1001768249569"))
    FILE_CHANNEL_1080P = int(os.environ.get("FILE_CHANNEL_1080P", "-1001768249569"))
    
    FILE_CHANNELS = {
        "480p": FILE_CHANNEL_480P,
        "720p": FILE_CHANNEL_720P,
        "1080p": FILE_CHANNEL_1080P
    }
    
    # MongoDB
    MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
    DATABASE_NAME = os.environ.get("DATABASE_NAME", "sk4film_bot")
    
    # Bot
    BOT_USERNAME = os.environ.get("BOT_USERNAME", "sk4filmbot")
    
    SECRET_KEY = os.environ.get("SECRET_KEY", "sk4film-secret-key-2024")
    WEB_SERVER_PORT = int(os.environ.get("PORT", 8000))
    
    # Poster sources
    OMDB_KEYS = ["8265bd1c", "b9bd48a6", "2f2d1c8e"]
    TMDB_KEYS = ["e547e17d4e91f3e62a571655cd1ccaff"]
    
    AUTO_UPDATE_INTERVAL = 180

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Quart(__name__)
User = None
bot_started = False
auto_update_task = None

# MongoDB Setup
mongo_client = AsyncIOMotorClient(Config.MONGODB_URI)
db = mongo_client[Config.DATABASE_NAME]
files_collection = db.files

# Movie database
movie_db = {
    'all_movies': [],
    'home_movies': [],
    'last_update': None,
    'poster_cache': {},
    'updating': False,
    'stats': {'omdb': 0, 'tmdb': 0, 'custom': 0}
}

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
            r'^([A-Z][a-z]+(?:\s+[A-Za-z]+){1,4})',
            r'"([^"]{4,35})"'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, first_line, re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                title = re.sub(r'\s+', ' ', title)
                
                if len(title) >= 4 and len(title) <= 45:
                    return title
        
        return None
    except:
        return None

def extract_title_from_file(message):
    """Extract title from file message"""
    try:
        # Try caption first
        if message.caption:
            title = extract_title_smart(message.caption)
            if title:
                return title
        
        # Try filename
        if message.document:
            filename = message.document.file_name
        elif message.video:
            filename = message.video.file_name
        else:
            return None
        
        if filename:
            # Remove extension and clean
            title = filename.rsplit('.', 1)[0]
            title = re.sub(r'[\._\-]', ' ', title)
            title = re.sub(r'\d{3,4}p', '', title, flags=re.IGNORECASE)
            title = re.sub(r'(720p|1080p|480p|HDRip|WEB-DL|BluRay|x264|x265)', '', title, flags=re.IGNORECASE)
            title = ' '.join(title.split())
            title = title.strip()[:50]
            
            if len(title) >= 4:
                return title
        
        return None
    except:
        return None

async def get_high_quality_poster(title, session):
    """Get high quality poster"""
    cache_key = title.lower().strip()
    
    if cache_key in movie_db['poster_cache']:
        cached, cache_time = movie_db['poster_cache'][cache_key]
        if datetime.now() - cache_time < timedelta(minutes=12):
            return cached
    
    try:
        # OMDB
        for api_key in Config.OMDB_KEYS:
            try:
                omdb_url = f"http://www.omdbapi.com/?t={urllib.parse.quote(title)}&apikey={api_key}"
                
                async with session.get(omdb_url, timeout=7) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get('Response') == 'True' and data.get('Poster') and data['Poster'] != 'N/A':
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
                tmdb_url = "https://api.themoviedb.org/3/search/movie"
                params = {'api_key': tmdb_key, 'query': title}
                
                async with session.get(tmdb_url, params=params, timeout=8) as response:
                    if response.status == 200:
                        tmdb_data = await response.json()
                        
                        if tmdb_data.get('results'):
                            movie = tmdb_data['results'][0]
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
        
        # Custom
        custom_result = {
            'poster_url': f"/api/enhanced_poster?title={urllib.parse.quote(title)}",
            'title': title,
            'source': 'CUSTOM',
            'success': True
        }
        movie_db['poster_cache'][cache_key] = (custom_result, datetime.now())
        movie_db['stats']['custom'] += 1
        return custom_result
        
    except:
        return {
            'poster_url': f"/api/enhanced_poster?title={urllib.parse.quote(title)}",
            'title': title,
            'source': 'CUSTOM',
            'success': True
        }

def is_new_post(post_date):
    """Check if post is within 24 hours"""
    try:
        if isinstance(post_date, str):
            post_date = datetime.fromisoformat(post_date.replace('Z', '+00:00'))
        hours_ago = (datetime.now() - post_date.replace(tzinfo=None)).total_seconds() / 3600
        return hours_ago <= 24
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

async def auto_index_to_mongodb(message, title, quality):
    """Auto-index files to MongoDB"""
    try:
        if message.document:
            file_id = message.document.file_id
            file_name = message.document.file_name
            file_size = message.document.file_size
        elif message.video:
            file_id = message.video.file_id
            file_name = message.video.file_name or f"video_{message.video.file_unique_id}.mp4"
            file_size = message.video.file_size
        else:
            return
        
        import hashlib
        unique_id = hashlib.md5(f"{file_id}{time.time()}".encode()).hexdigest()
        
        file_data = {
            "file_id": file_id,
            "unique_id": unique_id,
            "title": title,
            "file_name": file_name,
            "file_size": file_size,
            "quality": quality,
            "caption": message.caption,
            "message_id": message.id,
            "chat_id": message.chat.id,
            "indexed_at": datetime.now(),
            "downloads": 0
        }
        
        await files_collection.update_one(
            {"file_id": file_id},
            {"$set": file_data},
            upsert=True
        )
        
        logger.info(f"📁 Auto-indexed [{quality}]: {title}")
    except Exception as e:
        logger.error(f"Auto index error: {e}")

async def search_text_and_files(query, limit=12, page=1):
    """Search both text channels and file channels"""
    try:
        offset = (page - 1) * limit
        results = []
        seen_titles = {}
        
        # SEARCH TEXT CHANNELS
        for channel_id in Config.TEXT_CHANNEL_IDS:
            try:
                async for message in User.search_messages(channel_id, query, limit=30):
                    if message.text:
                        title = extract_title_smart(message.text)
                        
                        if title:
                            title_key = title.lower()
                            
                            if title_key not in seen_titles:
                                results.append({
                                    'title': title,
                                    'type': 'text_post',
                                    'content': format_original_post(message.text),
                                    'date': message.date.isoformat() if message.date else datetime.now().isoformat(),
                                    'channel': 'Movies Link' if channel_id == -1001891090100 else 'DISKWALA MOVIES',
                                    'channel_id': channel_id,
                                    'message_id': message.id,
                                    'is_new': is_new_post(message.date) if message.date else False,
                                    'has_file': False,
                                    'quality_options': {}
                                })
                                seen_titles[title_key] = len(results) - 1
            except Exception as e:
                logger.warning(f"Text channel search error: {e}")
        
        # SEARCH FILE CHANNELS
        for quality, channel_id in Config.FILE_CHANNELS.items():
            try:
                async for message in User.search_messages(channel_id, query, limit=20):
                    if message.document or message.video:
                        title = extract_title_from_file(message)
                        
                        if title:
                            title_key = title.lower()
                            
                            # Check if file is indexed in MongoDB
                            file_id = message.document.file_id if message.document else message.video.file_id
                            file_doc = await files_collection.find_one({"file_id": file_id})
                            unique_id = file_doc['unique_id'] if file_doc else None
                            
                            # If not indexed, index it now
                            if not unique_id:
                                await auto_index_to_mongodb(message, title, quality)
                                file_doc = await files_collection.find_one({"file_id": file_id})
                                unique_id = file_doc['unique_id'] if file_doc else None
                            
                            file_size = message.document.file_size if message.document else message.video.file_size
                            file_name = message.document.file_name if message.document else (message.video.file_name or 'video.mp4')
                            
                            # Check if title already exists
                            if title_key in seen_titles:
                                # Add quality option to existing result
                                idx = seen_titles[title_key]
                                results[idx]['has_file'] = True
                                results[idx]['type'] = 'with_file'
                                results[idx]['quality_options'][quality] = {
                                    'file_id': unique_id,
                                    'channel_id': channel_id,
                                    'message_id': message.id,
                                    'file_size': file_size,
                                    'file_name': file_name
                                }
                            else:
                                # Create new result
                                results.append({
                                    'title': title,
                                    'type': 'with_file',
                                    'content': format_original_post(message.caption or title),
                                    'date': message.date.isoformat() if message.date else datetime.now().isoformat(),
                                    'channel': f'File Channel ({quality})',
                                    'channel_id': channel_id,
                                    'message_id': message.id,
                                    'is_new': is_new_post(message.date) if message.date else False,
                                    'has_file': True,
                                    'quality_options': {
                                        quality: {
                                            'file_id': unique_id,
                                            'channel_id': channel_id,
                                            'message_id': message.id,
                                            'file_size': file_size,
                                            'file_name': file_name
                                        }
                                    }
                                })
                                seen_titles[title_key] = len(results) - 1
            except Exception as e:
                logger.warning(f"File channel [{quality}] search error: {e}")
        
        # Sort by file availability and date
        results.sort(key=lambda x: (x['has_file'], x['date']), reverse=True)
        
        total_results = len(results)
        paginated = results[offset:offset + limit]
        total_pages = math.ceil(total_results / limit) if total_results > 0 else 1
        
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

async def background_update_system():
    """Background update for home page"""
    if not User or not bot_started or movie_db['updating']:
        return
    
    try:
        movie_db['updating'] = True
        logger.info("🔄 Background update starting...")
        
        all_posts = []
        
        for channel_id in Config.TEXT_CHANNEL_IDS:
            try:
                async for message in User.get_chat_history(channel_id, limit=30):
                    if message.text and message.date:
                        title = extract_title_smart(message.text)
                        
                        if title:
                            all_posts.append({
                                'title': title,
                                'date': message.date,
                                'date_iso': message.date.isoformat(),
                                'is_new': is_new_post(message.date)
                            })
            except Exception as e:
                logger.warning(f"Channel error: {e}")
        
        all_posts.sort(key=lambda x: x['date'], reverse=True)
        
        unique_movies = []
        seen = set()
        
        for post in all_posts:
            title_key = post['title'].lower()
            if title_key not in seen:
                seen.add(title_key)
                post['date'] = post['date_iso']
                del post['date_iso']
                unique_movies.append(post)
        
        # Add posters
        async with aiohttp.ClientSession() as session:
            for i in range(0, min(len(unique_movies), 24), 4):
                batch = unique_movies[i:i + 4]
                poster_tasks = [get_high_quality_poster(m['title'], session) for m in batch]
                poster_results = await asyncio.gather(*poster_tasks, return_exceptions=True)
                
                for movie, poster_data in zip(batch, poster_results):
                    if isinstance(poster_data, dict) and poster_data.get('success'):
                        movie.update({
                            'poster_url': poster_data['poster_url'],
                            'poster_title': poster_data['title'],
                            'poster_year': poster_data.get('year', ''),
                            'poster_rating': poster_data.get('rating', ''),
                            'poster_source': poster_data['source'],
                            'has_poster': True
                        })
                await asyncio.sleep(0.3)
        
        movie_db['home_movies'] = unique_movies[:24]
        movie_db['last_update'] = datetime.now()
        
        logger.info(f"✅ Updated: {len(movie_db['home_movies'])} movies")
    except Exception as e:
        logger.error(f"Background update error: {e}")
    finally:
        movie_db['updating'] = False

async def start_auto_update():
    """Start auto update loop"""
    global auto_update_task
    
    async def update_loop():
        while bot_started:
            try:
                await asyncio.sleep(Config.AUTO_UPDATE_INTERVAL)
                await background_update_system()
            except Exception as e:
                logger.error(f"Auto update error: {e}")
    
    auto_update_task = asyncio.create_task(update_loop())

async def init_telegram_system():
    """Initialize Telegram system"""
    global User, bot_started
    
    try:
        logger.info("🔄 Initializing system...")
        
        User = Client(
            f"sk4film_user_{uuid.uuid4().hex[:8]}",
            api_id=Config.API_ID,
            api_hash=Config.API_HASH,
            session_string=Config.USER_SESSION_STRING,
            workdir="/tmp"
        )
        
        await User.start()
        me = await User.get_me()
        logger.info(f"✅ User: {me.first_name}")
        
        # MongoDB indexes
        await files_collection.create_index([("file_id", 1)], unique=True)
        await files_collection.create_index([("unique_id", 1)], unique=True)
        await files_collection.create_index([("title", "text")])
        logger.info("✅ MongoDB ready")
        
        bot_started = True
        
        # Initial load
        await background_update_system()
        
        # Start auto update
        await start_auto_update()
        
        logger.info("🎉 SYSTEM READY!")
        return True
    except Exception as e:
        logger.error(f"Init error: {e}")
        return False

@app.after_request
async def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST')
    return response

@app.route('/')
async def home():
    return jsonify({
        "status": "healthy" if bot_started else "starting",
        "service": "SK4FiLM Complete System",
        "features": ["web", "bot", "mongodb", "text_search", "file_search", "quality_selection"],
        "bot_username": f"@{Config.BOT_USERNAME}",
        "channels": {
            "text": len(Config.TEXT_CHANNEL_IDS),
            "file": len(Config.FILE_CHANNELS)
        },
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/movies')
async def api_movies():
    """Home movies API"""
    try:
        if not bot_started:
            return jsonify({"status": "starting"}), 503
        
        return jsonify({
            "status": "success",
            "movies": movie_db['home_movies'],
            "total": len(movie_db['home_movies']),
            "bot_username": Config.BOT_USERNAME,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/search')
async def api_search():
    """Search API - Text + File channels"""
    try:
        query = request.args.get('query', '').strip()
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 12))
        
        if not query:
            return jsonify({"status": "error", "message": "Query required"}), 400
        
        result = await search_text_and_files(query, limit, page)
        
        return jsonify({
            "status": "success",
            "query": query,
            "results": result["results"],
            "pagination": result["pagination"],
            "bot_username": Config.BOT_USERNAME,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/poster')
async def proxy_poster():
    """Poster proxy"""
    try:
        poster_url = request.args.get('url', '').strip()
        
        if not poster_url or poster_url.startswith('/api/'):
            title = request.args.get('title', 'Movie')
            return create_poster_svg(title)
        
        async with aiohttp.ClientSession() as session:
            headers = {'User-Agent': 'Mozilla/5.0'}
            async with session.get(poster_url, headers=headers, timeout=15) as response:
                if response.status == 200:
                    image_data = await response.read()
                    return Response(image_data, mimetype='image/jpeg')
        
        return create_poster_svg('Movie')
    except:
        return create_poster_svg('Error')

@app.route('/api/enhanced_poster')
async def enhanced_poster():
    title = request.args.get('title', 'Movie')
    return create_poster_svg(title)

def create_poster_svg(title):
    """Create custom poster SVG"""
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
        <text x="50%" y="250" text-anchor="middle" fill="#fff" font-size="16" font-weight="bold">
            {html.escape(display_title)}
        </text>
        <text x="50%" y="410" text-anchor="middle" fill="#fff" font-size="16" font-weight="700">SK4FiLM</text>
    </svg>'''
    
    return Response(svg, mimetype='image/svg+xml')

async def run_server():
    try:
        logger.info("🚀 SK4FiLM - COMPLETE SYSTEM")
        
        success = await init_telegram_system()
        
        if success:
            logger.info("🎉 ALL SYSTEMS OPERATIONAL!")
        
        config = HyperConfig()
        config.bind = [f"0.0.0.0:{Config.WEB_SERVER_PORT}"]
        await serve(app, config)
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        if auto_update_task:
            auto_update_task.cancel()
        if User:
            await User.stop()

if __name__ == "__main__":
    asyncio.run(run_server())
