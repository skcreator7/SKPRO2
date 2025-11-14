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

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger('asyncio').setLevel(logging.WARNING)

# ==================== CONFIG ====================
class Config:
    API_ID = int(os.environ.get("API_ID", "0"))
    API_HASH = os.environ.get("API_HASH", "")
    USER_SESSION_STRING = os.environ.get("USER_SESSION_STRING", "")
    BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
    
    TEXT_CHANNEL_IDS = [-1001891090100, -1002024811395]
    FILE_CHANNEL_ID = int(os.environ.get("FILE_CHANNEL_ID", "-1001768249569"))
    
    FORCE_SUB_CHANNEL = int(os.environ.get("FORCE_SUB_CHANNEL", "-1001891090100"))
    WEBSITE_URL = os.environ.get("WEBSITE_URL", "https://sk4film.vercel.app")
    BOT_USERNAME = os.environ.get("BOT_USERNAME", "sk4filmbot")
    ADMIN_IDS = [int(x) for x in os.environ.get("ADMIN_IDS", "123456789").split(",")]
    AUTO_DELETE_TIME = int(os.environ.get("AUTO_DELETE_TIME", "300"))
    WEB_SERVER_PORT = int(os.environ.get("PORT", 8000))
    BACKEND_URL = os.environ.get("BACKEND_URL", "https://sk4film.koyeb.app")
    
    OMDB_KEYS = ["8265bd1c", "b9bd48a6", "2f2d1c8e", "c3e6f8d9"]
    TMDB_KEYS = ["e547e17d4e91f3e62a571655cd1ccaff"]

# ==================== QUART APP (FIX APPLIED) ====================
# FIX: Initialize with proper config BEFORE any routes
app = Quart(__name__)

# Set config BEFORE using app
app.config.update({
    'JSON_SORT_KEYS': False,
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,
    'PROVIDE_AUTOMATIC_OPTIONS': True  # FIX for Quart 0.19.4 + Flask 3.1 compatibility
})

@app.after_request
async def add_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

# ==================== GLOBAL ====================
User = None
bot = None
bot_started = False
movie_db = {'home_movies': [], 'last_update': None, 'poster_cache': {}, 'updating': False, 'stats': {'omdb': 0, 'tmdb': 0, 'impawards': 0, 'justwatch': 0, 'letterboxd': 0, 'custom': 0}}
file_registry = {}

# ==================== HELPERS ====================
def extract_title_smart(text):
    if not text or len(text) < 15:
        return None
    try:
        clean = re.sub(r'[^\w\s\(\)\-\.\n]', ' ', text)
        first = clean.split('\n')[0].strip()
        patterns = [r'🎬\s*([^-\n]{4,45})', r'^([^(]{4,45})\s*\(\d{4}\)', r'^([^-]{4,45})\s*-', r'^([A-Z][a-z]+(?:\s+[A-Za-z]+){1,4})']
        for p in patterns:
            m = re.search(p, first)
            if m:
                t = m.group(1).strip()
                if 4 <= len(t) <= 45:
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

async def check_force_sub(user_id):
    try:
        member = await bot.get_chat_member(Config.FORCE_SUB_CHANNEL, user_id)
        return member.status in ["member", "administrator", "creator"]
    except:
        return True

# ==================== MULTI-SOURCE POSTERS ====================
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
                        res = {'poster_url': d['Poster'].replace('http://', 'https://'), 'title': d.get('Title', title), 'year': d.get('Year', ''), 'rating': d.get('imdbRating', ''), 'source': 'OMDB', 'success': True}
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
                            res = {'poster_url': f"https://image.tmdb.org/t/p/w780{p}", 'title': m.get('title', title), 'year': m.get('release_date', '')[:4] if m.get('release_date') else '', 'rating': f"{m.get('vote_average', 0):.1f}", 'source': 'TMDB', 'success': True}
                            movie_db['poster_cache'][ck] = (res, datetime.now())
                            movie_db['stats']['tmdb'] += 1
                            return res
        except:
            continue
    
    # IMPAwards
    try:
        async with session.get(f"https://www.impawards.com/search/?q={title.replace(' ', '+')}", timeout=6) as r:
            if r.status == 200:
                text = await r.text()
                match = re.search(r'<img src="([^"]+posters[^"]+\.jpg)"', text)
                if match:
                    url = match.group(1)
                    if not url.startswith('http'):
                        url = 'https://www.impawards.com' + url
                    res = {'poster_url': url, 'title': title, 'source': 'IMPAwards', 'success': True}
                    movie_db['poster_cache'][ck] = (res, datetime.now())
                    movie_db['stats']['impawards'] += 1
                    return res
    except:
        pass
    
    # JustWatch
    try:
        async with session.get(f"https://apis.justwatch.com/content/titles/movie/{title.lower().replace(' ', '-')}/locale/en_IN", timeout=5) as r:
            if r.status == 200:
                d = await r.json()
                if d.get('poster'):
                    res = {'poster_url': f"https://images.justwatch.com{d['poster']}", 'title': d.get('title', title), 'source': 'JustWatch', 'success': True}
                    movie_db['poster_cache'][ck] = (res, datetime.now())
                    movie_db['stats']['justwatch'] += 1
                    return res
    except:
        pass
    
    # Letterboxd
    try:
        async with session.get(f"https://letterboxd.com/film/{title.lower().replace(' ', '-')}/", timeout=6) as r:
            if r.status == 200:
                text = await r.text()
                match = re.search(r'<meta property="og:image" content="([^"]+)"', text)
                if match:
                    res = {'poster_url': match.group(1), 'title': title, 'source': 'Letterboxd', 'success': True}
                    movie_db['poster_cache'][ck] = (res, datetime.now())
                    movie_db['stats']['letterboxd'] += 1
                    return res
    except:
        pass
    
    # Custom
    movie_db['stats']['custom'] += 1
    res = {'poster_url': f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(title)}", 'title': title, 'source': 'CUSTOM', 'success': True}
    movie_db['poster_cache'][ck] = (res, datetime.now())
    return res

# ==================== SEARCH ====================
async def search_movies(query, limit=12, page=1):
    if not User or not bot_started:
        return {'results': [], 'pagination': {'current_page': 1, 'total_pages': 1, 'total_results': 0, 'per_page': limit}}
    
    offset = (page - 1) * limit
    results = []
    seen = {}
    
    logger.info(f"🔍 '{query}' P{page}")
    
    for cid in Config.TEXT_CHANNEL_IDS:
        try:
            cnt = 0
            async for msg in User.search_messages(cid, query, limit=20):
                if cnt >= 20:
                    break
                if msg.text:
                    t = extract_title_smart(msg.text)
                    if t and t.lower() not in seen:
                        results.append({'title': t, 'type': 'text_post', 'content': format_post(msg.text), 'date': msg.date.isoformat() if msg.date else datetime.now().isoformat(), 'channel': channel_name(cid), 'channel_id': cid, 'message_id': msg.id, 'is_new': is_new(msg.date) if msg.date else False, 'has_file': False, 'quality_options': {}})
                        seen[t.lower()] = len(results) - 1
                        cnt += 1
        except:
            pass
    
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
                        results.append({'title': t, 'type': 'with_file', 'content': format_post(msg.caption or t) if msg.caption else t, 'date': msg.date.isoformat() if msg.date else datetime.now().isoformat(), 'channel': channel_name(Config.FILE_CHANNEL_ID), 'channel_id': Config.FILE_CHANNEL_ID, 'message_id': msg.id, 'is_new': is_new(msg.date) if msg.date else False, 'has_file': True, 'quality_options': {q: {'file_id': uid, 'file_size': fsz, 'file_name': fnm}}})
                        seen[tk] = len(results) - 1
                    cnt += 1
    except:
        pass
    
    results.sort(key=lambda x: (x['has_file'], x['date']), reverse=True)
    total = len(results)
    paginated = results[offset:offset+limit]
    
    return {'results': paginated, 'pagination': {'current_page': page, 'total_pages': math.ceil(total/limit) if total else 1, 'total_results': total, 'per_page': limit, 'has_next': page < math.ceil(total/limit) if total else False, 'has_previous': page > 1}}

async def get_home_movies():
    if not User or not bot_started:
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
        except:
            pass
    
    movies.sort(key=lambda x: x['date'], reverse=True)
    movies = movies[:24]
    
    async with aiohttp.ClientSession() as s:
        for i in range(0, len(movies), 5):
            batch = movies[i:i+5]
            posters = await asyncio.gather(*[get_poster_multi(m['title'], s) for m in batch], return_exceptions=True)
            for m, p in zip(batch, posters):
                if isinstance(p, dict) and p.get('success'):
                    m.update({'poster_url': p['poster_url'], 'poster_title': p.get('title', m['title']), 'poster_year': p.get('year', ''), 'poster_rating': p.get('rating', ''), 'poster_source': p['source'], 'has_poster': True})
                else:
                    m['poster_url'] = f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(m['title'])}"
                    m['has_poster'] = True
            await asyncio.sleep(0.2)
    
    logger.info(f"✅ {len(movies)} | {movie_db['stats']}")
    return movies

# ==================== API ====================
@app.route('/')
async def root():
    return jsonify({'status': 'healthy', 'service': 'SK4FiLM', 'bot': f'@{Config.BOT_USERNAME}', 'sources': ['OMDB','TMDB','IMPAwards','JustWatch','Letterboxd'], 'stats': movie_db['stats'], 'movies': len(movie_db['home_movies']), 'files': len(file_registry)})

@app.route('/health')
async def health():
    return jsonify({'status': 'ok' if bot_started else 'starting', 'bot': bot_started, 'movies': len(movie_db['home_movies'])})

@app.route('/api/movies')
async def api_movies():
    try:
        if not bot_started:
            return jsonify({'status': 'error', 'message': 'Starting'}), 503
        if not movie_db['home_movies'] or not movie_db['last_update'] or (datetime.now() - movie_db['last_update']).seconds > 300:
            if not movie_db['updating']:
                movie_db['updating'] = True
                movie_db['home_movies'] = await get_home_movies()
                movie_db['last_update'] = datetime.now()
                movie_db['updating'] = False
        return jsonify({'status': 'success', 'movies': movie_db['home_movies'], 'total': len(movie_db['home_movies']), 'bot_username': Config.BOT_USERNAME})
    except Exception as e:
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
        if not bot_started:
            return jsonify({'status': 'error', 'message': 'Starting'}), 503
        result = await search_movies(q, l, p)
        return jsonify({'status': 'success', 'query': q, 'results': result['results'], 'pagination': result['pagination'], 'bot_username': Config.BOT_USERNAME})
    except Exception as e:
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
    colors = [('#667eea','#764ba2'),('#f093fb','#f5576c'),('#4facfe','#00f2fe'),('#43e97b','#38f9d7'),('#fa709a','#fee140')]
    c = colors[hash(t) % len(colors)]
    svg = f'''<svg width="300" height="450" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" style="stop-color:{c[0]}"/><stop offset="100%" style="stop-color:{c[1]}"/></linearGradient></defs><rect width="100%" height="100%" fill="url(#bg)" rx="20"/><circle cx="150" cy="180" r="50" fill="rgba(255,255,255,0.2)"/><text x="50%" y="200" text-anchor="middle" fill="#fff" font-size="50">🎬</text><text x="50%" y="270" text-anchor="middle" fill="#fff" font-size="18" font-weight="bold">{html.escape(d)}</text><rect x="50" y="380" width="200" height="40" rx="20" fill="rgba(0,0,0,0.3)"/><text x="50%" y="405" text-anchor="middle" fill="#fff" font-size="18" font-weight="700">SK4FiLM</text></svg>'''
    return Response(svg, mimetype='image/svg+xml', headers={'Cache-Control': 'public, max-age=3600'})

# ==================== BOT SETUP ====================
async def setup_bot():
    @bot.on_message(filters.command("start") & filters.private)
    async def start_handler(client, message):
        uid = message.from_user.id
        if len(message.command) > 1:
            fid = message.command[1].split('_')[0]
            if not await check_force_sub(uid):
                try:
                    ch = await bot.get_chat(Config.FORCE_SUB_CHANNEL)
                    lk = f"https://t.me/{ch.username}" if ch.username else f"https://t.me/c/{str(Config.FORCE_SUB_CHANNEL)[4:]}/1"
                except:
                    lk = f"https://t.me/c/{str(Config.FORCE_SUB_CHANNEL)[4:]}/1"
                await message.reply_text("⚠️ Join", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("📢 Join", url=lk)]]))
                return
            fi = file_registry.get(fid)
            if not fi:
                await message.reply_text("❌ Not Found", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔍 Search", url=Config.WEBSITE_URL)]]))
                return
            try:
                pm = await message.reply_text(f"⏳ Sending\n\n📁 {fi['file_name']}\n📊 {fi['quality']}\n📦 {format_size(fi['file_size'])}")
                if User:
                    fm = await User.get_messages(fi['channel_id'], fi['message_id'])
                    sent = await fm.copy(uid)
                    await pm.delete()
                    sm = await message.reply_text(f"✅ Sent!\n\n🎬 {fi['title']}\n📊 {fi['quality']}\n📦 {format_size(fi['file_size'])}\n\n⚠️ Delete in {Config.AUTO_DELETE_TIME//60}min")
                    logger.info(f"✅ {fi['title']} → {uid}")
                    if Config.AUTO_DELETE_TIME > 0:
                        await asyncio.sleep(Config.AUTO_DELETE_TIME)
                        try:
                            await sent.delete()
                            await sm.edit_text("🗑️ Deleted")
                        except:
                            pass
            except Exception as e:
                logger.error(f"Delivery: {e}")
                await message.reply_text("❌ Error")
            return
        await message.reply_text("🎬 **SK4FiLM**\n\n1. Visit\n2. Search\n3. Get", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🌐 Go", url=Config.WEBSITE_URL)]]))
    
    @bot.on_message(filters.text & filters.private & ~filters.command(['start', 'stats']))
    async def text_handler(client, message):
        await message.reply_text("👋 Use website", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🌐 Go", url=Config.WEBSITE_URL)]]))
    
    @bot.on_message(filters.command("stats") & filters.user(Config.ADMIN_IDS))
    async def stats_handler(client, message):
        await message.reply_text(f"📊 **Stats**\n\n🎬 Movies: {len(movie_db['home_movies'])}\n📁 Files: {len(file_registry)}\n🖼️ Posters:\n  OMDB: {movie_db['stats']['omdb']}\n  TMDB: {movie_db['stats']['tmdb']}\n  IMPAwards: {movie_db['stats']['impawards']}\n  JustWatch: {movie_db['stats']['justwatch']}\n  Letterboxd: {movie_db['stats']['letterboxd']}\n  Custom: {movie_db['stats']['custom']}\n\n🤖 Bot: {'✅' if bot_started else '❌'}")

# ==================== INIT ====================
async def init():
    global User, bot, bot_started
    try:
        logger.info("🔄 Initializing...")
        User = Client("sk4film_user", api_id=Config.API_ID, api_hash=Config.API_HASH, session_string=Config.USER_SESSION_STRING, workdir="/tmp", sleep_threshold=60)
        bot = Client("sk4film_bot", api_id=Config.API_ID, api_hash=Config.API_HASH, bot_token=Config.BOT_TOKEN, workdir="/tmp", sleep_threshold=60)
        await User.start()
        await bot.start()
        await setup_bot()
        me = await bot.get_me()
        logger.info(f"✅ Bot: @{me.username}")
        bot_started = True
        movie_db['home_movies'] = await get_home_movies()
        movie_db['last_update'] = datetime.now()
        logger.info("✅ System ready!")
        return True
    except Exception as e:
        logger.error(f"Init error: {e}")
        return False

async def main():
    logger.info("="*70)
    logger.info("🚀 SK4FiLM - Complete System with 5 Poster Sources")
    logger.info("="*70)
    logger.info(f"🌐 Website: {Config.WEBSITE_URL}")
    logger.info(f"📡 Backend: {Config.BACKEND_URL}")
    logger.info(f"🤖 Bot: @{Config.BOT_USERNAME}")
    logger.info("="*70)
    
    await init()
    
    cfg = HyperConfig()
    cfg.bind = [f"0.0.0.0:{Config.WEB_SERVER_PORT}"]
    cfg.loglevel = "warning"
    cfg.accesslog = "-"
    
    logger.info(f"🌐 Server starting on port {Config.WEB_SERVER_PORT}")
    await serve(app, cfg)

if __name__ == "__main__":
    asyncio.run(main())
