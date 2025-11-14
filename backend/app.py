import asyncio
import os
import logging
import hashlib
import time
import json
from datetime import datetime, timedelta
from pyrogram import Client, filters
from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from flask import Flask, jsonify, request, Response
import html
import re
import math
import requests
import urllib.parse

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger('asyncio').setLevel(logging.WARNING)

# ==================== CONFIG ====================
class Config:
    API_ID = int(os.environ.get("API_ID", "0"))
    API_HASH = os.environ.get("API_HASH", "")
    USER_SESSION_STRING = os.environ.get("USER_SESSION_STRING", "")
    BOT_TOKEN = os.environ.get("BOT_TOKEN", "8368221226:AAEOwAiBl_XXWAhOfSLsltw5a7SEaN4uiMo")
    
    TEXT_CHANNEL_IDS = [int(x) for x in os.environ.get("TEXT_CHANNEL_IDS", "-1001891090100,-1002024811395").split(",")]
    FILE_CHANNEL_ID = int(os.environ.get("FILE_CHANNEL_ID", "-1001768249569"))
    
    FORCE_SUB_CHANNEL = int(os.environ.get("FORCE_SUB_CHANNEL", "-1001891090100"))
    WEBSITE_URL = os.environ.get("WEBSITE_URL", "https://sk4film.vercel.app")
    BOT_USERNAME = os.environ.get("BOT_USERNAME", "sk4film_bot")
    ADMIN_IDS = [int(x) for x in os.environ.get("ADMIN_IDS", "123456789").split(",")]
    AUTO_DELETE_TIME = int(os.environ.get("AUTO_DELETE_TIME", "300"))
    WEB_SERVER_PORT = int(os.environ.get("PORT", 8000))
    BACKEND_URL = os.environ.get("BACKEND_URL", "https://sk4film.koyeb.app")
    
    # Multiple poster sources
    OMDB_KEYS = ["8265bd1c", "b9bd48a6", "2f2d1c8e", "c3e6f8d9"]
    TMDB_KEYS = ["e547e17d4e91f3e62a571655cd1ccaff"]

if not Config.BOT_TOKEN:
    logger.error("❌ BOT_TOKEN missing")
    exit(1)

# ==================== INITIALIZE PYROGRAM ====================
logger.info("🔧 Initializing Pyrogram...")

bot = Client(
    "sk4film_bot",
    api_id=Config.API_ID,
    api_hash=Config.API_HASH,
    bot_token=Config.BOT_TOKEN,
    workdir="/tmp",
    sleep_threshold=60
)

User = Client(
    "sk4film_user",
    api_id=Config.API_ID,
    api_hash=Config.API_HASH,
    session_string=Config.USER_SESSION_STRING,
    workdir="/tmp",
    sleep_threshold=60
) if Config.USER_SESSION_STRING else None

logger.info("✅ Pyrogram initialized")

# ==================== FLASK APP (NOT QUART - FIX) ====================
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

@app.after_request
def add_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

# ==================== GLOBAL STATE ====================
bot_started = False
user_started = False
movie_db = {
    'home_movies': [],
    'last_update': None,
    'poster_cache': {},
    'updating': False,
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
file_registry = {}
loop = None

# ==================== HELPERS ====================
def extract_title_smart(text):
    if not text or len(text) < 10:
        return None
    try:
        clean = re.sub(r'[^\w\s\(\)\-\.]', ' ', text[:100])
        lines = [l.strip() for l in clean.split('\n') if l.strip()]
        if lines:
            first = lines[0]
            first = re.sub(r'^(Movie|Film|Watch)[\s\:]+', '', first, flags=re.IGNORECASE)
            if 4 <= len(first) <= 50:
                return first.strip()
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
    if size < 1024*1024:
        return f"{size/1024:.1f} KB"
    elif size < 1024*1024*1024:
        return f"{size/(1024*1024):.1f} MB"
    else:
        return f"{size/(1024*1024*1024):.2f} GB"

def detect_quality(fn):
    if not fn:
        return "480p"
    fl = fn.lower()
    if '1080p' in fl or 'fhd' in fl:
        return "1080p"
    elif '720p' in fl or 'hd' in fl:
        return "720p"
    return "480p"

def format_post(text):
    if not text:
        return ""
    t = html.escape(text)
    t = re.sub(r'(https?://[^\s]+)', r'<a href="\1" style="color:#00ccff">\1</a>', t)
    return t.replace('\n', '<br>')

def channel_name(cid):
    return {-1001891090100: "SK4FiLM Main", -1002024811395: "SK4FiLM Updates", -1001768249569: "SK4FiLM Files"}.get(cid, "Channel")

def is_new(date):
    try:
        if isinstance(date, str):
            date = datetime.fromisoformat(date.replace('Z', '+00:00'))
        return (datetime.now() - date.replace(tzinfo=None)).seconds / 3600 <= 24
    except:
        return False

def run_async(coro):
    """Run async from sync context"""
    if loop and loop.is_running():
        import concurrent.futures
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        try:
            return future.result(timeout=30)
        except:
            return None
    return None

async def check_force_sub(user_id):
    try:
        member = await bot.get_chat_member(Config.FORCE_SUB_CHANNEL, user_id)
        return member.status in ["member", "administrator", "creator"]
    except:
        return True

# ==================== ADVANCED POSTER SOURCES ====================
def get_poster_multi_source(title):
    """Multiple poster sources - SYNCHRONOUS for Flask"""
    ck = title.lower().strip()
    
    # Check cache
    if ck in movie_db['poster_cache']:
        c, ct = movie_db['poster_cache'][ck]
        if (datetime.now() - ct).seconds < 600:
            return c
    
    logger.info(f"🎨 Fetching poster: {title}")
    
    # SOURCE 1: OMDB (High Quality)
    for k in Config.OMDB_KEYS:
        try:
            r = requests.get(f"http://www.omdbapi.com/?t={urllib.parse.quote(title)}&apikey={k}", timeout=5)
            if r.status_code == 200:
                d = r.json()
                if d.get('Response') == 'True' and d.get('Poster') != 'N/A':
                    res = {
                        'poster_url': d['Poster'].replace('http://', 'https://'),
                        'title': d.get('Title', title),
                        'year': d.get('Year', ''),
                        'rating': d.get('imdbRating', ''),
                        'source': 'OMDB',
                        'quality': 'HIGH',
                        'success': True
                    }
                    movie_db['poster_cache'][ck] = (res, datetime.now())
                    movie_db['stats']['omdb'] += 1
                    logger.info(f"✅ OMDB: {title}")
                    return res
        except:
            continue
    
    # SOURCE 2: TMDB (High Resolution)
    for k in Config.TMDB_KEYS:
        try:
            r = requests.get("https://api.themoviedb.org/3/search/movie", params={'api_key': k, 'query': title}, timeout=5)
            if r.status_code == 200:
                d = r.json()
                if d.get('results'):
                    m = d['results'][0]
                    p = m.get('poster_path')
                    if p:
                        res = {
                            'poster_url': f"https://image.tmdb.org/t/p/w780{p}",  # Higher resolution
                            'title': m.get('title', title),
                            'year': m.get('release_date', '')[:4] if m.get('release_date') else '',
                            'rating': f"{m.get('vote_average', 0):.1f}",
                            'source': 'TMDB',
                            'quality': 'HIGH',
                            'success': True
                        }
                        movie_db['poster_cache'][ck] = (res, datetime.now())
                        movie_db['stats']['tmdb'] += 1
                        logger.info(f"✅ TMDB: {title}")
                        return res
        except:
            continue
    
    # SOURCE 3: IMPAwards (Alternative)
    try:
        # IMPAwards search
        search_title = title.replace(' ', '+')
        imp_url = f"https://www.impawards.com/search/?q={search_title}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml'
        }
        
        r = requests.get(imp_url, headers=headers, timeout=7)
        if r.status_code == 200:
            # Simple regex to find poster URLs
            poster_match = re.search(r'<img src="([^"]+posters[^"]+\.jpg)"', r.text)
            if poster_match:
                poster_url = poster_match.group(1)
                if not poster_url.startswith('http'):
                    poster_url = 'https://www.impawards.com' + poster_url
                
                res = {
                    'poster_url': poster_url,
                    'title': title,
                    'source': 'IMPAwards',
                    'quality': 'MEDIUM',
                    'success': True
                }
                movie_db['poster_cache'][ck] = (res, datetime.now())
                movie_db['stats']['impawards'] += 1
                logger.info(f"✅ IMPAwards: {title}")
                return res
    except Exception as e:
        logger.debug(f"IMPAwards failed: {e}")
    
    # SOURCE 4: JustWatch (Streaming service posters)
    try:
        # JustWatch API (simplified - they have restrictions)
        jw_search = title.lower().replace(' ', '-')
        jw_url = f"https://apis.justwatch.com/content/titles/movie/{jw_search}/locale/en_IN"
        
        r = requests.get(jw_url, timeout=5)
        if r.status_code == 200:
            d = r.json()
            poster_path = d.get('poster')
            if poster_path:
                poster_url = f"https://images.justwatch.com{poster_path}"
                
                res = {
                    'poster_url': poster_url,
                    'title': d.get('title', title),
                    'year': d.get('original_release_year', ''),
                    'source': 'JustWatch',
                    'quality': 'HIGH',
                    'success': True
                }
                movie_db['poster_cache'][ck] = (res, datetime.now())
                movie_db['stats']['justwatch'] += 1
                logger.info(f"✅ JustWatch: {title}")
                return res
    except:
        pass
    
    # SOURCE 5: Letterboxd (Film community)
    try:
        # Letterboxd search
        lb_search = title.lower().replace(' ', '-')
        lb_url = f"https://letterboxd.com/film/{lb_search}/"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        r = requests.get(lb_url, headers=headers, timeout=6)
        if r.status_code == 200:
            # Find Open Graph image
            og_match = re.search(r'<meta property="og:image" content="([^"]+)"', r.text)
            if og_match:
                poster_url = og_match.group(1)
                
                res = {
                    'poster_url': poster_url,
                    'title': title,
                    'source': 'Letterboxd',
                    'quality': 'HIGH',
                    'success': True
                }
                movie_db['poster_cache'][ck] = (res, datetime.now())
                movie_db['stats']['letterboxd'] += 1
                logger.info(f"✅ Letterboxd: {title}")
                return res
    except:
        pass
    
    # FALLBACK: Custom SVG
    logger.info(f"ℹ️ Custom poster: {title}")
    movie_db['stats']['custom'] += 1
    res = {
        'poster_url': f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(title)}",
        'title': title,
        'source': 'CUSTOM',
        'quality': 'CUSTOM',
        'success': True
    }
    movie_db['poster_cache'][ck] = (res, datetime.now())
    return res

# ==================== SEARCH ====================
async def search_movies(query, limit=12, page=1):
    if not User or not user_started:
        return {'results': [], 'pagination': {'current_page': 1, 'total_pages': 1, 'total_results': 0, 'per_page': limit}}
    
    offset = (page - 1) * limit
    results = []
    seen = {}
    
    logger.info(f"🔍 Search: '{query}' | Page {page}")
    
    # Text channels
    for cid in Config.TEXT_CHANNEL_IDS:
        try:
            cnt = 0
            async for msg in User.search_messages(cid, query, limit=20):
                if cnt >= 20:
                    break
                if msg.text:
                    t = extract_title_smart(msg.text)
                    if t and t.lower() not in seen:
                        results.append({
                            'title': t,
                            'type': 'text_post',
                            'content': format_post(msg.text[:200]),
                            'date': msg.date.isoformat() if msg.date else datetime.now().isoformat(),
                            'channel': channel_name(cid),
                            'is_new': is_new(msg.date) if msg.date else False,
                            'has_file': False,
                            'quality_options': {}
                        })
                        seen[t.lower()] = len(results) - 1
                        cnt += 1
            logger.info(f"  ✓ Text: {cnt}")
        except Exception as e:
            logger.error(f"Text error: {e}")
    
    # File channel
    try:
        cnt = 0
        async for msg in User.search_messages(Config.FILE_CHANNEL_ID, query, limit=30):
            if cnt >= 30:
                break
            if msg.document or msg.video:
                t = extract_title_from_file(msg)
                if t:
                    fid = msg.document.file_id if msg.document else msg.video.file_id
                    fsz = msg.document.file_size if msg.document else (msg.video.file_size if msg.video else 0)
                    fnm = msg.document.file_name if msg.document else (msg.video.file_name if msg.video else 'video.mp4')
                    q = detect_quality(fnm)
                    uid = hashlib.md5(f"{fid}{time.time()}".encode()).hexdigest()
                    
                    file_registry[uid] = {
                        'file_id': fid,
                        'channel_id': Config.FILE_CHANNEL_ID,
                        'message_id': msg.id,
                        'quality': q,
                        'file_size': fsz,
                        'file_name': fnm,
                        'title': t,
                        'created_at': datetime.now()
                    }
                    
                    tk = t.lower()
                    if tk in seen:
                        idx = seen[tk]
                        results[idx]['has_file'] = True
                        results[idx]['type'] = 'with_file'
                        results[idx]['quality_options'][q] = {'file_id': uid, 'file_size': fsz, 'file_name': fnm}
                    else:
                        results.append({
                            'title': t,
                            'type': 'with_file',
                            'content': format_post(msg.caption or t)[:200] if msg.caption else t,
                            'date': msg.date.isoformat() if msg.date else datetime.now().isoformat(),
                            'channel': channel_name(Config.FILE_CHANNEL_ID),
                            'is_new': is_new(msg.date) if msg.date else False,
                            'has_file': True,
                            'quality_options': {q: {'file_id': uid, 'file_size': fsz, 'file_name': fnm}}
                        })
                        seen[tk] = len(results) - 1
                    cnt += 1
        logger.info(f"  ✓ Files: {cnt}")
    except Exception as e:
        logger.error(f"File error: {e}")
    
    results.sort(key=lambda x: (x['has_file'], x['date']), reverse=True)
    total = len(results)
    paginated = results[offset:offset+limit]
    
    logger.info(f"✅ Total: {total} | Show: {len(paginated)}")
    
    return {
        'results': paginated,
        'pagination': {
            'current_page': page,
            'total_pages': math.ceil(total/limit) if total else 1,
            'total_results': total,
            'per_page': limit,
            'has_next': page < math.ceil(total/limit) if total else False,
            'has_previous': page > 1
        }
    }

async def get_home_movies():
    if not User or not user_started:
        return []
    
    logger.info("🏠 Loading homepage...")
    movies = []
    seen = set()
    
    for cid in Config.TEXT_CHANNEL_IDS:
        try:
            cnt = 0
            async for msg in User.get_chat_history(cid, limit=20):
                if cnt >= 20:
                    break
                if msg.text and msg.date:
                    t = extract_title_smart(msg.text)
                    if t and t.lower() not in seen:
                        movies.append({'title': t, 'date': msg.date.isoformat(), 'is_new': is_new(msg.date)})
                        seen.add(t.lower())
                        cnt += 1
            logger.info(f"  ✓ {channel_name(cid)}: {cnt}")
        except Exception as e:
            logger.error(f"Channel error: {e}")
    
    movies.sort(key=lambda x: x['date'], reverse=True)
    movies = movies[:24]
    
    logger.info(f"🎨 Fetching posters from multiple sources...")
    
    # Fetch posters synchronously
    for i, movie in enumerate(movies):
        try:
            poster = get_poster_multi_source(movie['title'])
            if poster and poster.get('success'):
                movie.update({
                    'poster_url': poster['poster_url'],
                    'poster_title': poster.get('title', movie['title']),
                    'poster_year': poster.get('year', ''),
                    'poster_rating': poster.get('rating', ''),
                    'poster_source': poster['source'],
                    'poster_quality': poster.get('quality', 'STANDARD'),
                    'has_poster': True
                })
            else:
                movie['poster_url'] = f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(movie['title'])}"
                movie['has_poster'] = True
            
            # Rate limit
            if i % 5 == 0 and i > 0:
                time.sleep(0.3)
        except Exception as e:
            logger.error(f"Poster error: {e}")
            movie['poster_url'] = f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(movie['title'])}"
            movie['has_poster'] = True
    
    logger.info(f"✅ Loaded {len(movies)} movies | Stats: {movie_db['stats']}")
    return movies

# ==================== BOT HANDLERS ====================
@bot.on_message(filters.command("start") & filters.private)
async def start_handler(client, message):
    uid = message.from_user.id
    logger.info(f"👤 /start: {uid}")
    
    # File delivery
    if len(message.command) > 1:
        fid = message.command[1]
        logger.info(f"📥 File: {fid}")
        
        if not await check_force_sub(uid):
            try:
                ch = await bot.get_chat(Config.FORCE_SUB_CHANNEL)
                lk = f"https://t.me/{ch.username}" if ch.username else f"https://t.me/c/{str(Config.FORCE_SUB_CHANNEL)[4:]}/1"
            except:
                lk = f"https://t.me/c/{str(Config.FORCE_SUB_CHANNEL)[4:]}/1"
            
            await message.reply_text(
                "⚠️ **Join First**",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("📢 Join", url=lk)]])
            )
            return
        
        fi = file_registry.get(fid)
        if not fi:
            await message.reply_text(
                "❌ **Not Found**",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔍 Search", url=Config.WEBSITE_URL)]])
            )
            return
        
        try:
            pm = await message.reply_text(f"⏳ Sending...\n\n📁 {fi['file_name']}\n📊 {fi['quality']}")
            
            if User:
                fm = await User.get_messages(fi['channel_id'], fi['message_id'])
                sent = await fm.copy(uid)
                await pm.delete()
                
                sm = await message.reply_text(f"✅ **Sent!**\n\n🎬 {fi['title']}\n📊 {fi['quality']}\n\n⚠️ Auto-delete in {Config.AUTO_DELETE_TIME//60}min")
                
                logger.info(f"✅ Delivered: {fi['title']} → {uid}")
                
                if Config.AUTO_DELETE_TIME > 0:
                    await asyncio.sleep(Config.AUTO_DELETE_TIME)
                    try:
                        await sent.delete()
                        await sm.edit_text("🗑️ Deleted")
                    except:
                        pass
        except Exception as e:
            logger.error(f"Delivery error: {e}")
            await message.reply_text("❌ Error")
        
        return
    
    # Normal start
    await message.reply_text(
        "🎬 **SK4FiLM Bot**\n\n📌 File delivery\n\n**Usage:**\n1. Visit website\n2. Search\n3. Get file",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🌐 Website", url=Config.WEBSITE_URL)]])
    )

@bot.on_message(filters.text & filters.private & ~filters.command(['start', 'stats']))
async def text_handler(client, message):
    await message.reply_text(
        "👋 Use website to search.",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🌐 Go", url=Config.WEBSITE_URL)]])
    )

@bot.on_message(filters.command("stats") & filters.user(Config.ADMIN_IDS))
async def stats_handler(client, message):
    await message.reply_text(
        f"📊 **Stats**\n\n"
        f"🎬 Movies: {len(movie_db['home_movies'])}\n"
        f"📁 Files: {len(file_registry)}\n"
        f"🖼️ Poster Sources:\n"
        f"  • OMDB: {movie_db['stats']['omdb']}\n"
        f"  • TMDB: {movie_db['stats']['tmdb']}\n"
        f"  • IMPAwards: {movie_db['stats']['impawards']}\n"
        f"  • JustWatch: {movie_db['stats']['justwatch']}\n"
        f"  • Letterboxd: {movie_db['stats']['letterboxd']}\n"
        f"  • Custom: {movie_db['stats']['custom']}\n"
        f"🤖 Bot: {'✅' if bot_started else '❌'}\n"
        f"👤 User: {'✅' if user_started else '❌'}"
    )

# ==================== API ====================
@app.route('/')
def root():
    return jsonify({
        'status': 'healthy',
        'service': 'SK4FiLM - Multi-Source Posters',
        'bot': f'@{Config.BOT_USERNAME}',
        'poster_sources': ['OMDB', 'TMDB', 'IMPAwards', 'JustWatch', 'Letterboxd', 'Custom'],
        'stats': movie_db['stats']
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok' if (bot_started and user_started) else 'starting',
        'bot': bot_started,
        'user': user_started
    })

@app.route('/api/movies')
def api_movies():
    try:
        if not user_started:
            return jsonify({'status': 'error', 'message': 'Starting'}), 503
        
        if not movie_db['home_movies'] or not movie_db['last_update'] or (datetime.now() - movie_db['last_update']).seconds > 300:
            if not movie_db['updating']:
                movie_db['updating'] = True
                movies = run_async(get_home_movies())
                if movies:
                    movie_db['home_movies'] = movies
                    movie_db['last_update'] = datetime.now()
                movie_db['updating'] = False
        
        return jsonify({
            'status': 'success',
            'movies': movie_db['home_movies'],
            'total': len(movie_db['home_movies']),
            'bot_username': Config.BOT_USERNAME,
            'poster_stats': movie_db['stats']
        })
    except Exception as e:
        logger.error(f"API movies: {e}")
        movie_db['updating'] = False
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/search')
def api_search():
    try:
        q = request.args.get('query', '').strip()
        p = int(request.args.get('page', 1))
        l = int(request.args.get('limit', 12))
        
        if not q:
            return jsonify({'status': 'error', 'message': 'Query required'}), 400
        
        if not user_started:
            return jsonify({'status': 'error', 'message': 'Starting'}), 503
        
        result = run_async(search_movies(q, l, p))
        if result:
            return jsonify({
                'status': 'success',
                'query': q,
                'results': result['results'],
                'pagination': result['pagination'],
                'bot_username': Config.BOT_USERNAME
            })
        else:
            return jsonify({'status': 'error', 'message': 'Search failed'}), 500
    except Exception as e:
        logger.error(f"API search: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/poster')
def api_poster():
    t = request.args.get('title', 'Movie')
    d = t[:18] + "..." if len(t) > 18 else t
    colors = [('#667eea','#764ba2'),('#f093fb','#f5576c'),('#4facfe','#00f2fe'),('#43e97b','#38f9d7'),('#fa709a','#fee140'),('#30cfd0','#330867'),('#a8edea','#fed6e3')]
    c = colors[hash(t) % len(colors)]
    svg = f'''<svg width="300" height="450" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" style="stop-color:{c[0]}"/><stop offset="100%" style="stop-color:{c[1]}"/></linearGradient></defs><rect width="100%" height="100%" fill="url(#bg)" rx="20"/><circle cx="150" cy="180" r="50" fill="rgba(255,255,255,0.2)"/><text x="50%" y="200" text-anchor="middle" fill="#fff" font-size="50">🎬</text><text x="50%" y="270" text-anchor="middle" fill="#fff" font-size="18" font-weight="bold">{html.escape(d)}</text><rect x="50" y="380" width="200" height="40" rx="20" fill="rgba(0,0,0,0.3)"/><text x="50%" y="405" text-anchor="middle" fill="#fff" font-size="18" font-weight="700">SK4FiLM</text></svg>'''
    return Response(svg, mimetype='image/svg+xml', headers={'Cache-Control': 'public, max-age=3600'})

# ==================== STARTUP ====================
async def start_clients():
    global bot_started, user_started, loop
    loop = asyncio.get_event_loop()
    
    try:
        logger.info("🤖 Starting bot...")
        await bot.start()
        me = await bot.get_me()
        logger.info(f"✅ Bot: @{me.username}")
        bot_started = True
    except Exception as e:
        logger.error(f"Bot error: {e}")
    
    if User:
        try:
            logger.info("👤 Starting user...")
            await User.start()
            me = await User.get_me()
            logger.info(f"✅ User: {me.first_name}")
            user_started = True
            
            # Initial load
            movie_db['home_movies'] = await get_home_movies()
            movie_db['last_update'] = datetime.now()
        except Exception as e:
            logger.error(f"User error: {e}")

def run_flask():
    """Flask in thread"""
    logger.info(f"🌐 Flask on 0.0.0.0:{Config.WEB_SERVER_PORT}")
    app.run(host='0.0.0.0', port=Config.WEB_SERVER_PORT, debug=False, use_reloader=False, threaded=True)

async def run_bot():
    """Pyrogram in main thread"""
    await start_clients()
    logger.info("✅ All systems running")
    try:
        await asyncio.Event().wait()
    except:
        pass

def main():
    import threading
    
    logger.info("=" * 70)
    logger.info("🚀 SK4FiLM - Multi-Source Poster System")
    logger.info("=" * 70)
    logger.info(f"🤖 @{Config.BOT_USERNAME}")
    logger.info(f"🌐 {Config.WEBSITE_URL}")
    logger.info(f"📡 {Config.BACKEND_URL}")
    logger.info(f"🎨 Poster Sources: 5+")
    logger.info("=" * 70)
    
    # Flask in thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    time.sleep(2)
    
    # Pyrogram in main
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        logger.info("\n⚠️ Stopped")

if __name__ == "__main__":
    main()
