import asyncio
import os
import logging
import signal
from datetime import datetime, timedelta
from pyrogram import Client, filters
from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton
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

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger('asyncio').setLevel(logging.WARNING)

# MongoDB
try:
    from motor.motor_asyncio import AsyncIOMotorClient
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False

# ==================== CONFIG ====================
class Config:
    API_ID = int(os.environ.get("API_ID", "0"))
    API_HASH = os.environ.get("API_HASH", "")
    USER_SESSION_STRING = os.environ.get("USER_SESSION_STRING", "")
    BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
    
    TEXT_CHANNEL_IDS = [int(x) for x in os.environ.get("TEXT_CHANNEL_IDS", "-1001891090100,-1002024811395").split(",")]
    FILE_CHANNEL_ID = int(os.environ.get("FILE_CHANNEL_ID", "-1001768249569"))
    
    MONGODB_URI = os.environ.get("MONGODB_URI", "")
    DATABASE_NAME = os.environ.get("DATABASE_NAME", "sk4film_bot")
    
    FORCE_SUB_CHANNEL = int(os.environ.get("FORCE_SUB_CHANNEL", "-1001891090100"))
    WEBSITE_URL = os.environ.get("WEBSITE_URL", "https://sk4film.vercel.app")
    BOT_USERNAME = os.environ.get("BOT_USERNAME", "sk4film_bot")
    ADMIN_IDS = [int(x) for x in os.environ.get("ADMIN_IDS", "123456789").split(",")]
    AUTO_DELETE_TIME = int(os.environ.get("AUTO_DELETE_TIME", "300"))
    WEB_SERVER_PORT = int(os.environ.get("PORT", 8000))
    
    OMDB_KEYS = ["8265bd1c", "b9bd48a6", "2f2d1c8e"]
    TMDB_KEYS = ["e547e17d4e91f3e62a571655cd1ccaff"]
    BACKEND_URL = os.environ.get("BACKEND_URL", f"http://localhost:{WEB_SERVER_PORT}")

if not Config.BOT_TOKEN:
    logger.error("❌ BOT_TOKEN missing")
    exit(1)

# ==================== GLOBAL STATE ====================
bot_started = False
user_started = False
shutdown_event = asyncio.Event()
movie_db = {'home_movies': [], 'last_update': None, 'poster_cache': {}, 'updating': False, 'stats': {'omdb': 0, 'tmdb': 0, 'custom': 0}}
file_registry = {}
users_collection = None
stats_collection = None

# ==================== QUART APP (MINIMAL CONFIG) ====================
# Create app WITHOUT any initial config to avoid KeyError
app = Quart(__name__)

# Set config AFTER creation
@app.before_first_request
async def configure_app():
    app.config['JSON_SORT_KEYS'] = False

@app.after_request
async def add_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

# ==================== PYROGRAM CLIENTS ====================
bot = Client("sk4film_bot", api_id=Config.API_ID, api_hash=Config.API_HASH, bot_token=Config.BOT_TOKEN, workdir="/tmp", sleep_threshold=60)
User = Client("sk4film_user", api_id=Config.API_ID, api_hash=Config.API_HASH, session_string=Config.USER_SESSION_STRING, workdir="/tmp", sleep_threshold=60) if Config.USER_SESSION_STRING else None

# MongoDB
if MONGODB_AVAILABLE and Config.MONGODB_URI:
    try:
        mongo_client = AsyncIOMotorClient(Config.MONGODB_URI, serverSelectionTimeoutMS=5000)
        db = mongo_client[Config.DATABASE_NAME]
        users_collection = db.users
        stats_collection = db.stats
        logger.info("✅ MongoDB connected")
    except Exception as e:
        logger.error(f"MongoDB: {e}")

# ==================== HELPERS ====================
async def check_force_sub(user_id):
    try:
        member = await bot.get_chat_member(Config.FORCE_SUB_CHANNEL, user_id)
        return member.status in ["member", "administrator", "creator"]
    except:
        return True

def extract_title_smart(text):
    if not text or len(text) < 10:
        return None
    try:
        clean = re.sub(r'[^\w\s\(\)\-\.]', ' ', text[:100])
        lines = [l.strip() for l in clean.split('\n') if l.strip()]
        if lines:
            first = lines[0]
            if 4 <= len(first) <= 50:
                return first
    except:
        pass
    return None

def extract_title_from_file(msg):
    try:
        if msg.caption:
            t = extract_title_smart(msg.caption)
            if t:
                return t
        
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
        return f"{size/1024:.1f}KB"
    elif size < 1024 * 1024 * 1024:
        return f"{size/(1024*1024):.1f}MB"
    else:
        return f"{size/(1024*1024*1024):.2f}GB"

def detect_quality(fn):
    if not fn:
        return "480p"
    fl = fn.lower()
    if any(p in fl for p in ['1080p', 'fullhd', 'fhd']):
        return "1080p"
    elif any(p in fl for p in ['720p', 'hd']):
        return "720p"
    return "480p"

def format_post(text):
    if not text:
        return ""
    t = html.escape(text)
    t = re.sub(r'(https?://[^\s]+)', r'<a href="\1" target="_blank" style="color:#00ccff">\1</a>', t)
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

# ==================== POSTER ====================
async def get_poster_omdb(title, s):
    for k in Config.OMDB_KEYS:
        try:
            async with s.get(f"http://www.omdbapi.com/?t={urllib.parse.quote(title)}&apikey={k}", timeout=5) as r:
                if r.status == 200:
                    d = await r.json()
                    if d.get('Response') == 'True' and d.get('Poster') != 'N/A':
                        movie_db['stats']['omdb'] += 1
                        return {'poster_url': d['Poster'].replace('http://', 'https://'), 'title': d.get('Title', title), 'year': d.get('Year', ''), 'rating': d.get('imdbRating', ''), 'source': 'OMDB', 'success': True}
        except:
            continue
    return None

async def get_poster_tmdb(title, s):
    for k in Config.TMDB_KEYS:
        try:
            async with s.get("https://api.themoviedb.org/3/search/movie", params={'api_key': k, 'query': title}, timeout=5) as r:
                if r.status == 200:
                    d = await r.json()
                    if d.get('results'):
                        m = d['results'][0]
                        p = m.get('poster_path')
                        if p:
                            movie_db['stats']['tmdb'] += 1
                            return {'poster_url': f"https://image.tmdb.org/t/p/w500{p}", 'title': m.get('title', title), 'year': m.get('release_date', '')[:4] if m.get('release_date') else '', 'rating': f"{m.get('vote_average', 0):.1f}", 'source': 'TMDB', 'success': True}
        except:
            continue
    return None

async def get_poster(title, s=None):
    ck = title.lower().strip()
    if ck in movie_db['poster_cache']:
        c, ct = movie_db['poster_cache'][ck]
        if (datetime.now() - ct).seconds < 600:
            return c
    
    close = False
    if not s:
        s = aiohttp.ClientSession()
        close = True
    
    try:
        r = await get_poster_omdb(title, s)
        if r:
            movie_db['poster_cache'][ck] = (r, datetime.now())
            return r
        
        r = await get_poster_tmdb(title, s)
        if r:
            movie_db['poster_cache'][ck] = (r, datetime.now())
            return r
        
        movie_db['stats']['custom'] += 1
        r = {'poster_url': f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(title)}", 'title': title, 'source': 'CUSTOM', 'success': True}
        movie_db['poster_cache'][ck] = (r, datetime.now())
        return r
    except:
        return {'poster_url': f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(title)}", 'title': title, 'source': 'CUSTOM', 'success': True}
    finally:
        if close:
            await s.close()

# ==================== SEARCH ====================
async def search_movies(query, limit=12, page=1):
    if not User:
        return {'results': [], 'pagination': {'current_page': 1, 'total_pages': 1, 'total_results': 0, 'per_page': limit, 'has_next': False, 'has_previous': False}}
    
    offset = (page - 1) * limit
    results = []
    seen = {}
    
    logger.info(f"🔍 Search: '{query}' | Page {page}")
    
    for cid in Config.TEXT_CHANNEL_IDS:
        try:
            cnt = 0
            async for msg in User.search_messages(cid, query, limit=20):
                if cnt >= 20:
                    break
                if msg.text:
                    t = extract_title_smart(msg.text)
                    if t and t.lower() not in seen:
                        results.append({'title': t, 'type': 'text_post', 'content': format_post(msg.text), 'date': msg.date.isoformat() if msg.date else datetime.now().isoformat(), 'channel': channel_name(cid), 'is_new': is_new(msg.date) if msg.date else False, 'has_file': False, 'quality_options': {}})
                        seen[t.lower()] = len(results) - 1
                        cnt += 1
        except Exception as e:
            logger.error(f"Text ch {cid}: {e}")
    
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
                    
                    file_registry[uid] = {'file_id': fid, 'channel_id': Config.FILE_CHANNEL_ID, 'message_id': msg.id, 'quality': q, 'file_size': fsz, 'file_name': fnm, 'title': t, 'created_at': datetime.now()}
                    
                    tk = t.lower()
                    if tk in seen:
                        idx = seen[tk]
                        results[idx]['has_file'] = True
                        results[idx]['type'] = 'with_file'
                        results[idx]['quality_options'][q] = {'file_id': uid, 'file_size': fsz, 'file_name': fnm}
                    else:
                        results.append({'title': t, 'type': 'with_file', 'content': format_post(msg.caption or t), 'date': msg.date.isoformat() if msg.date else datetime.now().isoformat(), 'channel': channel_name(Config.FILE_CHANNEL_ID), 'is_new': is_new(msg.date) if msg.date else False, 'has_file': True, 'quality_options': {q: {'file_id': uid, 'file_size': fsz, 'file_name': fnm}}})
                        seen[tk] = len(results) - 1
                    cnt += 1
    except Exception as e:
        logger.error(f"File ch: {e}")
    
    results.sort(key=lambda x: (x['has_file'], x['date']), reverse=True)
    total = len(results)
    paginated = results[offset:offset+limit]
    
    logger.info(f"✅ Found {total} | Show {len(paginated)}")
    
    return {'results': paginated, 'pagination': {'current_page': page, 'total_pages': math.ceil(total/limit) if total else 1, 'total_results': total, 'per_page': limit, 'has_next': page < math.ceil(total/limit) if total else False, 'has_previous': page > 1}}

async def get_home_movies():
    if not User:
        return []
    
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
        except Exception as e:
            logger.error(f"Home ch {cid}: {e}")
    
    movies.sort(key=lambda x: x['date'], reverse=True)
    movies = movies[:24]
    
    async with aiohttp.ClientSession() as s:
        for i in range(0, len(movies), 5):
            batch = movies[i:i+5]
            posters = await asyncio.gather(*[get_poster(m['title'], s) for m in batch], return_exceptions=True)
            for m, p in zip(batch, posters):
                if isinstance(p, dict) and p.get('success'):
                    m.update({'poster_url': p['poster_url'], 'poster_title': p.get('title', m['title']), 'poster_year': p.get('year', ''), 'poster_rating': p.get('rating', ''), 'poster_source': p['source'], 'has_poster': True})
                else:
                    m.update({'poster_url': f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(m['title'])}", 'has_poster': True})
            await asyncio.sleep(0.2)
    
    logger.info(f"✅ Home: {len(movies)} movies")
    return movies

# ==================== BOT ====================
@bot.on_message(filters.command("start") & filters.private)
async def start_h(c, m):
    uid = m.from_user.id
    
    if users_collection:
        try:
            await users_collection.update_one({'user_id': uid}, {'$set': {'first_name': m.from_user.first_name, 'last_seen': datetime.now()}, '$inc': {'start_count': 1}}, upsert=True)
        except:
            pass
    
    if len(m.command) > 1:
        fuid = m.command[1]
        
        if not await check_force_sub(uid):
            try:
                ch = await bot.get_chat(Config.FORCE_SUB_CHANNEL)
                lk = f"https://t.me/{ch.username}" if ch.username else f"https://t.me/c/{str(Config.FORCE_SUB_CHANNEL)[4:]}/1"
            except:
                lk = f"https://t.me/c/{str(Config.FORCE_SUB_CHANNEL)[4:]}/1"
            await m.reply_text("⚠️ **Join Channel First**", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("📢 Join", url=lk)]]))
            return
        
        fi = file_registry.get(fuid)
        if not fi:
            await m.reply_text("❌ **File Not Found**\n\nExpired.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🌐 Search", url=Config.WEBSITE_URL)]]))
            return
        
        try:
            pm = await m.reply_text(f"⏳ Sending...\n\n📁 {fi['file_name']}\n📊 {fi['quality']}")
            
            if User:
                fm = await User.get_messages(fi['channel_id'], fi['message_id'])
                sent = await fm.copy(uid)
                await pm.delete()
                sm = await m.reply_text(f"✅ **Sent!**\n\n🎬 {fi['title']}\n📊 {fi['quality']}\n\n⚠️ Auto-delete in {Config.AUTO_DELETE_TIME//60}min")
                logger.info(f"📥 File: {fi['title']} → {uid}")
                
                if Config.AUTO_DELETE_TIME > 0:
                    await asyncio.sleep(Config.AUTO_DELETE_TIME)
                    try:
                        await sent.delete()
                        await sm.edit_text("🗑️ Deleted")
                    except:
                        pass
            else:
                await pm.edit_text("❌ Unavailable")
        except Exception as e:
            logger.error(f"File err: {e}")
            await m.reply_text("❌ Error")
        return
    
    await m.reply_text("🎬 **SK4FiLM Bot**\n\n📌 File delivery only\n\n**Usage:**\n1. Visit website\n2. Search\n3. Select quality\n4. Get file", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🌐 Website", url=Config.WEBSITE_URL)]]))

@bot.on_message(filters.text & filters.private & ~filters.command(['start', 'stats']))
async def text_h(c, m):
    await m.reply_text("👋 Hi!\n\n🤖 Use website.\n\nBot delivers files only.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🌐 Website", url=Config.WEBSITE_URL)]]))

@bot.on_message(filters.command("stats") & filters.user(Config.ADMIN_IDS))
async def stats_h(c, m):
    try:
        tu = await users_collection.count_documents({}) if users_collection else 0
        await m.reply_text(f"📊 **Stats**\n\n👥 Users: `{tu}`\n🎬 Movies: `{len(movie_db['home_movies'])}`\n📁 Files: `{len(file_registry)}`\n🖼️ OMDB:{movie_db['stats']['omdb']} TMDB:{movie_db['stats']['tmdb']} Custom:{movie_db['stats']['custom']}\n🤖 Bot: `{'✅' if bot_started else '❌'}`\n👤 User: `{'✅' if user_started else '❌'}`")
    except Exception as e:
        await m.reply_text(f"❌ Error: {e}")

# ==================== API ====================
@app.route('/')
async def root():
    return jsonify({'status': 'healthy', 'service': 'SK4FiLM', 'bot': f'@{Config.BOT_USERNAME}'})

@app.route('/api/movies')
async def api_movies():
    try:
        if not user_started:
            return jsonify({'status': 'error', 'message': 'Starting'}), 503
        
        if not movie_db['home_movies'] or not movie_db['last_update'] or (datetime.now() - movie_db['last_update']).seconds > 300:
            if not movie_db['updating']:
                movie_db['updating'] = True
                movie_db['home_movies'] = await get_home_movies()
                movie_db['last_update'] = datetime.now()
                movie_db['updating'] = False
        
        return jsonify({'status': 'success', 'movies': movie_db['home_movies'], 'total': len(movie_db['home_movies']), 'bot_username': Config.BOT_USERNAME})
    except Exception as e:
        logger.error(f"API movies: {e}")
        movie_db['updating'] = False
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/search')
async def api_search():
    try:
        q = request.args.get('query', '').strip()
        p = int(request.args.get('page', 1))
        l = int(request.args.get('limit', 12))
        
        if not q:
            return jsonify({'status': 'error', 'message': 'Query required'}), 400
        
        if not user_started:
            return jsonify({'status': 'error', 'message': 'Starting'}), 503
        
        r = await search_movies(q, l, p)
        return jsonify({'status': 'success', 'query': q, 'results': r['results'], 'pagination': r['pagination'], 'bot_username': Config.BOT_USERNAME})
    except Exception as e:
        logger.error(f"API search: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/poster')
async def api_poster():
    t = request.args.get('title', 'Movie')
    d = t[:18] + "..." if len(t) > 18 else t
    colors = [('#667eea','#764ba2'),('#f093fb','#f5576c'),('#4facfe','#00f2fe'),('#43e97b','#38f9d7'),('#fa709a','#fee140')]
    c = colors[hash(t) % len(colors)]
    svg = f'''<svg width="300" height="450" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" style="stop-color:{c[0]}"/><stop offset="100%" style="stop-color:{c[1]}"/></linearGradient></defs><rect width="100%" height="100%" fill="url(#bg)" rx="20"/><circle cx="150" cy="180" r="50" fill="rgba(255,255,255,0.2)"/><text x="50%" y="200" text-anchor="middle" fill="#fff" font-size="50">🎬</text><text x="50%" y="270" text-anchor="middle" fill="#fff" font-size="18" font-weight="bold">{html.escape(d)}</text><rect x="50" y="380" width="200" height="40" rx="20" fill="rgba(0,0,0,0.3)"/><text x="50%" y="405" text-anchor="middle" fill="#fff" font-size="18" font-weight="700">SK4FiLM</text></svg>'''
    return Response(svg, mimetype='image/svg+xml', headers={'Cache-Control': 'public, max-age=3600'})

# ==================== STARTUP ====================
async def start_bot():
    global bot_started
    try:
        logger.info("🤖 Starting bot...")
        await bot.start()
        me = await bot.get_me()
        logger.info(f"✅ Bot: @{me.username}")
        bot_started = True
    except Exception as e:
        logger.error(f"❌ Bot: {e}")

async def start_user():
    global user_started
    if not User:
        logger.error("❌ User session missing")
        return
    try:
        logger.info("👤 Starting user...")
        await User.start()
        me = await User.get_me()
        logger.info(f"✅ User: {me.first_name}")
        user_started = True
        movie_db['home_movies'] = await get_home_movies()
        movie_db['last_update'] = datetime.now()
    except Exception as e:
        logger.error(f"❌ User: {e}")

async def start_web():
    try:
        cfg = HyperConfig()
        cfg.bind = [f"0.0.0.0:{Config.WEB_SERVER_PORT}"]
        cfg.loglevel = "warning"
        cfg.graceful_timeout = 2
        logger.info(f"🌐 Web on port {Config.WEB_SERVER_PORT}")
        await serve(app, cfg, shutdown_trigger=shutdown_event.wait)
    except Exception as e:
        logger.error(f"❌ Web: {e}")

async def cleanup():
    logger.info("🔄 Cleaning...")
    if bot_started:
        try:
            await bot.stop()
            logger.info("✅ Bot stopped")
        except:
            pass
    if user_started and User:
        try:
            await User.stop()
            logger.info("✅ User stopped")
        except:
            pass
    shutdown_event.set()
    logger.info("✅ Cleanup done")

def sig_handler(signum, frame):
    logger.info(f"\n⚠️ Signal {signum}")
    asyncio.create_task(cleanup())

async def main():
    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)
    
    logger.info("=" * 60)
    logger.info("🚀 SK4FiLM Starting")
    logger.info("=" * 60)
    logger.info(f"🤖 Bot: @{Config.BOT_USERNAME}")
    logger.info(f"🌐 Website: {Config.WEBSITE_URL}")
    logger.info("=" * 60)
    
    try:
        await asyncio.gather(start_bot(), start_user(), start_web())
    except KeyboardInterrupt:
        logger.info("\n⚠️ Keyboard interrupt")
    except Exception as e:
        logger.error(f"❌ Fatal: {e}")
    finally:
        await cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n⚠️ Stopped")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
