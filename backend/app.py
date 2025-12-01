"""
app.py - Main SK4FiLM Bot with all features
"""
import asyncio
import os
import logging
import json
import re
import math
import html
import time
import base64
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from io import BytesIO

import aiohttp
import urllib.parse
from pyrogram import Client, filters, idle
from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton, Message
from pyrogram.errors import FloodWait, UserNotParticipant, ChannelPrivate
from quart import Quart, jsonify, request, Response
from hypercorn.asyncio import serve
from hypercorn.config import Config as HyperConfig
from motor.motor_asyncio import AsyncIOMotorClient

# Import our modules
from verification import VerificationSystem
from premium import PremiumSystem, PremiumTier, PremiumPlan
from poster_fetching import PosterFetcher, PosterSource
from cache import CacheManager

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Config:
    # Telegram API
    API_ID = int(os.environ.get("API_ID", "0"))
    API_HASH = os.environ.get("API_HASH", "")
    USER_SESSION_STRING = os.environ.get("USER_SESSION_STRING", "")
    BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
    
    # Database
    MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
    REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
    
    # Channels
    MAIN_CHANNEL_ID = -1001891090100
    TEXT_CHANNEL_IDS = [-1001891090100, -1002024811395]
    FILE_CHANNEL_ID = -1001768249569
    
    # URLs
    MAIN_CHANNEL_LINK = "https://t.me/sk4film"
    UPDATES_CHANNEL_LINK = "https://t.me/sk4film_Request"
    CHANNEL_USERNAME = "sk4film"
    WEBSITE_URL = os.environ.get("WEBSITE_URL", "https://sk4film.vercel.app")
    BACKEND_URL = os.environ.get("BACKEND_URL", "https://sk4film.koyeb.app")
    
    # Bot
    BOT_USERNAME = os.environ.get("BOT_USERNAME", "sk4filmbot")
    ADMIN_IDS = [int(x) for x in os.environ.get("ADMIN_IDS", "123456789").split(",")]
    AUTO_DELETE_TIME = int(os.environ.get("AUTO_DELETE_TIME", "300"))
    
    # Shortener & Verification
    SHORTLINK_API = os.environ.get("SHORTLINK_API", "")
    VERIFICATION_REQUIRED = os.environ.get("VERIFICATION_REQUIRED", "false").lower() == "true"
    VERIFICATION_DURATION = 6 * 60 * 60
    
    # Server
    WEB_SERVER_PORT = int(os.environ.get("PORT", 8000))
    
    # API Keys
    OMDB_KEYS = os.environ.get("OMDB_KEYS", "8265bd1c,b9bd48a6,3e7e1e9d").split(",")
    TMDB_KEYS = os.environ.get("TMDB_KEYS", "e547e17d4e91f3e62a571655cd1ccaff,8265bd1f").split(",")
    
    # UPI IDs for Premium
    UPI_ID_BASIC = os.environ.get("UPI_ID_BASIC", "sk4filmbot@ybl")
    UPI_ID_PREMIUM = os.environ.get("UPI_ID_PREMIUM", "sk4filmbot@ybl")
    UPI_ID_ULTIMATE = os.environ.get("UPI_ID_ULTIMATE", "sk4filmbot@ybl")
    UPI_ID_LIFETIME = os.environ.get("UPI_ID_LIFETIME", "sk4filmbot@ybl")
    
    # Admin
    ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "admin123")

# Initialize Quart app
app = Quart(__name__)

@app.after_request
async def add_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

# Global instances
bot = None
user_client = None
mongo_client = None
db = None
files_col = None
verification_col = None

# Module instances
verification_system = None
premium_system = None
poster_fetcher = None
cache_manager = None

# Status flags
bot_started = False
user_session_ready = False

# Channel configuration
CHANNEL_CONFIG = {
    -1001891090100: {'name': 'SK4FiLM Main', 'type': 'text', 'search_priority': 1},
    -1002024811395: {'name': 'SK4FiLM Updates', 'type': 'text', 'search_priority': 2},
    -1001768249569: {'name': 'SK4FiLM Files', 'type': 'file', 'search_priority': 0}
}

# Flood wait protection
class FloodWaitProtection:
    def __init__(self):
        self.last_request_time = 0
        self.min_interval = 3
        self.request_count = 0
        self.reset_time = time.time()
    
    async def wait_if_needed(self):
        current_time = time.time()
        
        # Reset counter every 2 minutes
        if current_time - self.reset_time > 120:
            self.request_count = 0
            self.reset_time = current_time
        
        # Limit to 20 requests per 2 minutes
        if self.request_count >= 20:
            wait_time = 120 - (current_time - self.reset_time)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                self.request_count = 0
                self.reset_time = time.time()
        
        # Ensure minimum interval between requests
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_interval:
            await asyncio.sleep(self.min_interval - time_since_last)
        
        self.last_request_time = time.time()
        self.request_count += 1

flood_protection = FloodWaitProtection()

# Utility functions
def normalize_title(title):
    """Normalize movie title for searching"""
    if not title:
        return ""
    normalized = title.lower().strip()
    
    normalized = re.sub(
        r'\b(19|20)\d{2}\b|\b(480p|720p|1080p|2160p|4k|hd|fhd|uhd|hevc|x264|x265|h264|h265|'
        r'bluray|webrip|hdrip|web-dl|hdtv|hdrip|webdl|hindi|english|tamil|telugu|'
        r'malayalam|kannada|punjabi|bengali|marathi|gujarati|movie|film|series|'
        r'complete|full|part|episode|season|hdrc|dvdscr|pre-dvd|p-dvd|pdc|'
        r'rarbg|yts|amzn|netflix|hotstar|prime|disney|hc-esub|esub|subs)\b',
        '', 
        normalized, 
        flags=re.IGNORECASE
    )
    
    normalized = re.sub(r'[\._\-]', ' ', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized

def extract_title_smart(text):
    """Extract movie title from text"""
    if not text or len(text) < 10:
        return None
    
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if not lines:
        return None
    
    first_line = lines[0]
    
    patterns = [
        (r'^([A-Za-z\s]{3,50}?)\s*(?:\(?\d{4}\)?|\b(?:480p|720p|1080p|2160p|4k|hd|fhd|uhd)\b)', 1),
        (r'🎬\s*([^\n\-\(]{3,60}?)\s*(?:\(\d{4}\)|$)', 1),
        (r'^([^\-\n]{3,60}?)\s*\-', 1),
        (r'^([^\(\n]{3,60}?)\s*\(\d{4}\)', 1),
        (r'^([A-Za-z\s]{3,50}?)\s*(?:\d{4}|Hindi|Movie|Film|HDTC|WebDL|X264|AAC|ESub)', 1),
    ]
    
    for pattern, group in patterns:
        match = re.search(pattern, first_line, re.IGNORECASE)
        if match:
            title = match.group(group).strip()
            title = re.sub(r'\s+', ' ', title)
            if 3 <= len(title) <= 60:
                return title
    
    return None

def extract_title_from_file(msg):
    """Extract title from file message"""
    try:
        if msg.caption:
            t = extract_title_smart(msg.caption)
            if t:
                return t
        
        fn = msg.document.file_name if msg.document else (msg.video.file_name if msg.video else None)
        if fn:
            name = fn.rsplit('.', 1)[0]
            name = re.sub(r'[\._\-]', ' ', name)
            
            name = re.sub(
                r'\b(720p|1080p|480p|2160p|4k|HDRip|WEBRip|WEB-DL|BluRay|BRRip|DVDRip|HDTV|HDTC|'
                r'X264|X265|HEVC|H264|H265|AAC|AC3|DD5\.1|DDP5\.1|HC|ESub|Subs|Hindi|English|'
                r'Dual|Multi|Complete|Full|Movie|Film|\d{4})\b',
                '', 
                name, 
                flags=re.IGNORECASE
            )
            
            name = re.sub(r'\s+', ' ', name).strip()
            name = re.sub(r'\s+\(\d{4}\)$', '', name)
            name = re.sub(r'\s+\d{4}$', '', name)
            
            if 4 <= len(name) <= 50:
                return name
    except Exception as e:
        logger.error(f"File title extraction error: {e}")
    return None

def format_size(size):
    """Format file size"""
    if not size:
        return "Unknown"
    if size < 1024*1024:
        return f"{size/1024:.1f} KB"
    elif size < 1024*1024*1024:
        return f"{size/(1024*1024):.1f} MB"
    else:
        return f"{size/(1024*1024*1024):.2f} GB"

def detect_quality(filename):
    """Detect video quality from filename"""
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
    """Format post text for HTML"""
    if not text:
        return ""
    text = html.escape(text)
    text = re.sub(r'(https?://[^\s]+)', r'<a href="\1" target="_blank" style="color:#00ccff">\1</a>', text)
    return text.replace('\n', '<br>')

def channel_name(cid):
    """Get channel name from config"""
    return CHANNEL_CONFIG.get(cid, {}).get('name', f"Channel {cid}")

def is_new(date):
    """Check if date is within 48 hours"""
    try:
        if isinstance(date, str):
            date = datetime.fromisoformat(date.replace('Z', '+00:00'))
        hours = (datetime.now() - date.replace(tzinfo=None)).total_seconds() / 3600
        return hours <= 48
    except:
        return False

def is_video_file(file_name):
    """Check if file is video"""
    if not file_name:
        return False
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.mpg', '.mpeg']
    file_name_lower = file_name.lower()
    return any(file_name_lower.endswith(ext) for ext in video_extensions)

async def safe_telegram_operation(operation, *args, **kwargs):
    """Safely execute Telegram operations with flood wait protection"""
    max_retries = 2
    base_delay = 5
    
    for attempt in range(max_retries):
        try:
            await flood_protection.wait_if_needed()
            result = await operation(*args, **kwargs)
            return result
        except FloodWait as e:
            wait_time = e.value + 10
            logger.warning(f"Flood wait detected: {e.value}s -> waiting {wait_time}s, attempt {attempt + 1}/{max_retries}")
            
            if attempt == max_retries - 1:
                raise e
            
            await asyncio.sleep(wait_time)
        except Exception as e:
            logger.error(f"Telegram operation failed: {e}")
            if attempt == max_retries - 1:
                raise e
            await asyncio.sleep(base_delay * (2 ** attempt))
    return None

async def safe_telegram_generator(operation, *args, limit=None, **kwargs):
    """Safely iterate over Telegram async generators"""
    max_retries = 2
    count = 0
    
    for attempt in range(max_retries):
        try:
            await flood_protection.wait_if_needed()
            async for item in operation(*args, **kwargs):
                yield item
                count += 1
                
                if count % 10 == 0:
                    await asyncio.sleep(1)
                    
                if limit and count >= limit:
                    break
            break
        except FloodWait as e:
            wait_time = e.value + 10
            logger.warning(f"Flood wait in generator: {e.value}s -> waiting {wait_time}s, attempt {attempt + 1}/{max_retries}")
            
            if attempt == max_retries - 1:
                raise e
            
            await asyncio.sleep(wait_time)
        except Exception as e:
            logger.error(f"Telegram generator failed: {e}")
            if attempt == max_retries - 1:
                raise e
            await asyncio.sleep(5 * (2 ** attempt))

# API Routes
@app.route('/')
async def root():
    """Root endpoint with system status"""
    mongodb_count = 0
    if files_col is not None:
        try:
            mongodb_count = await files_col.count_documents({})
        except:
            pass
    
    return jsonify({
        'status': 'healthy',
        'service': 'SK4FiLM v8.0 - Complete System',
        'version': '8.0',
        'timestamp': datetime.now().isoformat(),
        'modules': {
            'verification': verification_system is not None,
            'premium': premium_system is not None,
            'poster_fetcher': poster_fetcher is not None,
            'cache': cache_manager is not None
        },
        'bot': {
            'started': bot_started,
            'username': Config.BOT_USERNAME
        },
        'user_session': 'ready' if user_session_ready else 'pending',
        'database': {
            'mongodb_connected': mongo_client is not None,
            'total_files': mongodb_count
        },
        'channels': {
            'text_channels': len(Config.TEXT_CHANNEL_IDS),
            'file_channel': Config.FILE_CHANNEL_ID
        },
        'endpoints': {
            'api': '/api/*',
            'health': '/health',
            'verify': '/api/verify/{user_id}',
            'premium': '/api/premium/*',
            'poster': '/api/poster/{title}',
            'cache': '/api/cache/*',
            'search': '/api/search'
        }
    })

@app.route('/health')
async def health():
    """Health check endpoint"""
    mongodb_status = False
    if mongo_client is not None:
        try:
            await mongo_client.admin.command('ping')
            mongodb_status = True
        except:
            pass
    
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'services': {
            'bot': {
                'started': bot_started,
                'status': 'online' if bot_started else 'starting'
            },
            'user_session': user_session_ready,
            'mongodb': mongodb_status,
            'redis': cache_manager.redis_enabled if cache_manager is not None else False,
            'web_server': True
        },
        'modules': {
            'verification': verification_system is not None,
            'premium': premium_system is not None,
            'poster_fetcher': poster_fetcher is not None,
            'cache': cache_manager is not None
        }
    })

@app.route('/api/verify/<int:user_id>', methods=['POST'])
async def api_verify_user(user_id):
    """Create verification link for user"""
    if verification_system is None:
        return jsonify({'status': 'error', 'message': 'Verification system not initialized'}), 500
    
    try:
        verification_data = await verification_system.create_verification_link(user_id)
        return jsonify({
            'status': 'success',
            'verification_url': verification_data['short_url'],
            'token': verification_data['token'],
            'user_id': user_id
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/verify/check/<int:user_id>')
async def api_check_verification(user_id):
    """Check user verification status"""
    if verification_system is None:
        return jsonify({'status': 'error', 'message': 'Verification system not initialized'}), 500
    
    try:
        is_verified, message = await verification_system.check_user_verified(user_id)
        return jsonify({
            'status': 'success',
            'verified': is_verified,
            'message': message,
            'user_id': user_id
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/premium/plans')
async def api_get_premium_plans():
    """Get all premium plans"""
    if premium_system is None:
        return jsonify({'status': 'error', 'message': 'Premium system not initialized'}), 500
    
    try:
        plans = await premium_system.get_all_plans()
        return jsonify({
            'status': 'success',
            'plans': plans
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/premium/user/<int:user_id>')
async def api_get_user_premium(user_id):
    """Get user premium status"""
    if premium_system is None:
        return jsonify({'status': 'error', 'message': 'Premium system not initialized'}), 500
    
    try:
        details = await premium_system.get_subscription_details(user_id)
        return jsonify({
            'status': 'success',
            'subscription': details
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/premium/create-payment', methods=['POST'])
async def api_create_payment():
    """Create payment request"""
    if premium_system is None:
        return jsonify({'status': 'error', 'message': 'Premium system not initialized'}), 500
    
    try:
        data = await request.get_json()
        user_id = data.get('user_id')
        tier_str = data.get('tier')
        
        if not user_id or not tier_str:
            return jsonify({'status': 'error', 'message': 'User ID and tier required'}), 400
        
        try:
            tier = PremiumTier(tier_str)
        except ValueError:
            return jsonify({'status': 'error', 'message': 'Invalid tier'}), 400
        
        if tier == PremiumTier.FREE:
            return jsonify({'status': 'error', 'message': 'Cannot create payment for free tier'}), 400
        
        payment_data = await premium_system.create_payment_request(user_id, tier)
        
        return jsonify({
            'status': 'success',
            'payment': payment_data,
            'instructions': 'Please pay using the QR code and send screenshot to bot'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/premium/check-access/<int:user_id>')
async def api_check_premium_access(user_id):
    """Check if user has premium access"""
    if premium_system is None:
        return jsonify({'status': 'error', 'message': 'Premium system not initialized'}), 500
    
    try:
        is_premium = await premium_system.is_premium_user(user_id)
        subscription = await premium_system.get_subscription_details(user_id)
        
        return jsonify({
            'status': 'success',
            'is_premium': is_premium,
            'subscription': subscription,
            'has_verification_bypass': is_premium
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/premium/can-download/<int:user_id>')
async def api_can_download(user_id):
    """Check if user can download"""
    if premium_system is None:
        return jsonify({'status': 'error', 'message': 'Premium system not initialized'}), 500
    
    try:
        can_download, message, details = await premium_system.can_user_download(user_id)
        
        return jsonify({
            'status': 'success',
            'can_download': can_download,
            'message': message,
            'details': details
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/poster/<path:title>')
async def api_get_poster(title):
    """Get movie poster"""
    if poster_fetcher is None:
        return jsonify({'status': 'error', 'message': 'Poster fetcher not initialized'}), 500
    
    try:
        poster_data = await poster_fetcher.fetch_poster(title)
        return jsonify({
            'status': 'success',
            'poster': poster_data
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/cache/stats')
async def api_cache_stats():
    """Get cache statistics"""
    if cache_manager is None:
        return jsonify({'status': 'error', 'message': 'Cache manager not initialized'}), 500
    
    try:
        stats = await cache_manager.get_stats_summary()
        return jsonify({
            'status': 'success',
            'stats': stats
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/cache/clear', methods=['POST'])
async def api_clear_cache():
    """Clear cache"""
    if cache_manager is None:
        return jsonify({'status': 'error', 'message': 'Cache manager not initialized'}), 500
    
    try:
        await cache_manager.clear_all()
        return jsonify({
            'status': 'success',
            'message': 'Cache cleared successfully'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/search')
async def api_search():
    """Search for movies"""
    try:
        q = request.args.get('query', '').strip()
        p = int(request.args.get('page', 1))
        l = int(request.args.get('limit', 12))
        
        if not q:
            return jsonify({'status': 'error', 'message': 'Query required'}), 400
        
        # Check cache first
        cache_key = f"search:{q}:{p}:{l}"
        if cache_manager is not None:
            cached = await cache_manager.get(cache_key)
            if cached:
                return jsonify(cached)
        
        # Check user access
        user_id = request.args.get('user_id', type=int)
        if user_id:
            # Check if premium user (bypasses verification)
            if premium_system is not None:
                is_premium = await premium_system.is_premium_user(user_id)
                if is_premium:
                    # Premium user, allow access
                    pass
                elif Config.VERIFICATION_REQUIRED and verification_system is not None:
                    # Check verification for non-premium users
                    is_verified, message = await verification_system.check_user_verified(user_id)
                    if not is_verified:
                        return jsonify({
                            'status': 'verification_required',
                            'message': 'User verification required',
                            'verification_url': f'/api/verify/{user_id}'
                        }), 403
        
        # Perform search
        results = []
        if files_col is not None:
            # Search in MongoDB
            cursor = files_col.find({
                '$or': [
                    {'title': {'$regex': q, '$options': 'i'}},
                    {'normalized_title': {'$regex': q, '$options': 'i'}}
                ]
            }).limit(l).skip((p - 1) * l)
            
            async for doc in cursor:
                results.append({
                    'title': doc.get('title'),
                    'normalized_title': doc.get('normalized_title'),
                    'quality': doc.get('quality', '480p'),
                    'file_size': doc.get('file_size'),
                    'file_name': doc.get('file_name'),
                    'date': doc.get('date'),
                    'channel_id': doc.get('channel_id'),
                    'message_id': doc.get('message_id'),
                    'file_id': doc.get('file_id'),
                    'is_new': is_new(doc.get('date')) if doc.get('date') else False,
                    'is_video_file': doc.get('is_video_file', False),
                    'thumbnail': doc.get('thumbnail'),
                    'thumbnail_source': doc.get('thumbnail_source', 'unknown')
                })
        
        total = len(results)
        response_data = {
            'status': 'success',
            'query': q,
            'results': results,
            'pagination': {
                'current_page': p,
                'total_pages': math.ceil(total / l) if total > 0 else 1,
                'total_results': total,
                'per_page': l,
                'has_next': p < math.ceil(total / l) if total > 0 else False,
                'has_previous': p > 1
            }
        }
        
        # Cache the results
        if cache_manager is not None:
            await cache_manager.set(cache_key, response_data, expire_seconds=1800)
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/post')
async def api_post():
    """Get post details"""
    try:
        channel_id = request.args.get('channel', '').strip()
        message_id = request.args.get('message', '').strip()
        
        if not channel_id or not message_id:
            return jsonify({'status':'error', 'message':'Missing channel or message parameter'}), 400
        
        if not bot_started or not user_client or not user_session_ready:
            return jsonify({'status':'error', 'message':'Bot not ready yet'}), 503
        
        try:
            channel_id = int(channel_id)
            message_id = int(message_id)
        except ValueError:
            return jsonify({'status':'error', 'message':'Invalid channel or message ID'}), 400
        
        msg = await safe_telegram_operation(
            user_client.get_messages,
            channel_id, 
            message_id
        )
        
        if not msg or not msg.text:
            return jsonify({'status':'error', 'message':'Message not found or has no text content'}), 404
        
        title = extract_title_smart(msg.text)
        if not title:
            title = msg.text.split('\n')[0][:60] if msg.text else "Movie Post"
        
        normalized_title = normalize_title(title)
        quality_options = {}
        has_file = False
        
        if files_col is not None:
            cursor = files_col.find({'normalized_title': normalized_title})
            async for doc in cursor:
                quality = doc.get('quality', '480p')
                if quality not in quality_options:
                    quality_options[quality] = {
                        'file_id': f"{doc.get('channel_id', Config.FILE_CHANNEL_ID)}_{doc.get('message_id')}_{quality}",
                        'file_size': doc.get('file_size', 0),
                        'file_name': doc.get('file_name', 'video.mp4'),
                        'is_video': doc.get('is_video_file', False),
                        'channel_id': doc.get('channel_id'),
                        'message_id': doc.get('message_id')
                    }
                    has_file = True
        
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
        
        return jsonify({'status': 'success', 'post': post_data})
    
    except Exception as e:
        logger.error(f"Post error: {e}")
        return jsonify({'status':'error', 'message': str(e)}), 500

@app.route('/api/movies')
async def api_movies():
    """Get latest movies"""
    try:
        if not bot_started:
            return jsonify({'status': 'error', 'message': 'Starting...'}), 503
        
        # Get posts from channels
        posts = []
        if user_session_ready:
            for channel_id in Config.TEXT_CHANNEL_IDS:
                try:
                    async for msg in safe_telegram_generator(
                        user_client.get_chat_history, 
                        channel_id, 
                        limit=10
                    ):
                        if msg and msg.text and len(msg.text) > 15:
                            title = extract_title_smart(msg.text)
                            if title:
                                posts.append({
                                    'title': title,
                                    'channel_name': channel_name(channel_id),
                                    'channel_id': channel_id,
                                    'message_id': msg.id,
                                    'date': msg.date.isoformat() if isinstance(msg.date, datetime) else str(msg.date),
                                    'is_new': is_new(msg.date) if msg.date else False
                                })
                except Exception as e:
                    logger.error(f"Error getting posts from channel {channel_id}: {e}")
        
        # Remove duplicates
        seen = set()
        unique_posts = []
        for post in posts:
            if post['title'] not in seen:
                seen.add(post['title'])
                unique_posts.append(post)
        
        # Get posters
        movies = unique_posts[:20]  # Limit to 20
        if poster_fetcher is not None:
            titles = [movie['title'] for movie in movies]
            posters = await poster_fetcher.fetch_batch_posters(titles)
            
            for movie in movies:
                if movie['title'] in posters:
                    movie.update(posters[movie['title']])
                else:
                    movie.update({
                        'poster_url': f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(movie['title'])}",
                        'source': 'custom',
                        'rating': '0.0'
                    })
        
        return jsonify({
            'status': 'success', 
            'movies': movies, 
            'total': len(movies), 
            'bot_username': Config.BOT_USERNAME
        })
    except Exception as e:
        logger.error(f"Movies error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/poster')
async def api_poster_svg():
    """Generate SVG poster"""
    try:
        t = request.args.get('title', 'Movie')
        y = request.args.get('year', '')
        
        d = t[:20] + "..." if len(t) > 20 else t
        
        color_schemes = [
            {'bg1': '#667eea', 'bg2': '#764ba2', 'text': '#ffffff'},
            {'bg1': '#f093fb', 'bg2': '#f5576c', 'text': '#ffffff'},
            {'bg1': '#4facfe', 'bg2': '#00f2fe', 'text': '#ffffff'},
            {'bg1': '#43e97b', 'bg2': '#38f9d7', 'text': '#ffffff'},
            {'bg1': '#fa709a', 'bg2': '#fee140', 'text': '#ffffff'},
        ]
        
        scheme = color_schemes[hash(t) % len(color_schemes)]
        text_color = scheme['text']
        bg1_color = scheme['bg1']
        bg2_color = scheme['bg2']
        
        year_text = f'<text x="150" y="305" text-anchor="middle" fill="{text_color}" font-size="14" font-family="Arial">{html.escape(y)}</text>' if y else ''
        
        svg = f'''<svg width="300" height="450" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:{bg1_color};stop-opacity:1"/>
                    <stop offset="100%" style="stop-color:{bg2_color};stop-opacity:1"/>
                </linearGradient>
            </defs>
            <rect width="100%" height="100%" fill="url(#bg)"/>
            <rect x="10" y="10" width="280" height="430" fill="none" stroke="{text_color}" stroke-width="2" stroke-opacity="0.3" rx="10"/>
            <circle cx="150" cy="180" r="60" fill="rgba(255,255,255,0.1)"/>
            <text x="150" y="185" text-anchor="middle" fill="{text_color}" font-size="60" font-family="Arial">🎬</text>
            <text x="150" y="280" text-anchor="middle" fill="{text_color}" font-size="16" font-weight="bold" font-family="Arial">{html.escape(d)}</text>
            {year_text}
            <rect x="50" y="380" width="200" height="40" rx="20" fill="rgba(0,0,0,0.3)"/>
            <text x="150" y="405" text-anchor="middle" fill="{text_color}" font-size="16" font-weight="bold" font-family="Arial">SK4FiLM</text>
        </svg>'''
        
        return Response(svg, mimetype='image/svg+xml', headers={
            'Cache-Control': 'public, max-age=86400',
            'Content-Type': 'image/svg+xml'
        })
    except Exception as e:
        logger.error(f"Poster SVG error: {e}")
        simple_svg = '''<svg width="300" height="450" xmlns="http://www.w3.org/2000/svg">
            <rect width="100%" height="100%" fill="#667eea"/>
            <text x="150" y="225" text-anchor="middle" fill="white" font-size="18" font-family="Arial">SK4FiLM</text>
        </svg>'''
        return Response(simple_svg, mimetype='image/svg+xml')

# Bot handlers
async def setup_bot():
    """Setup bot commands and handlers"""
    
    @bot.on_message(filters.command("start") & filters.private)
    async def start_handler(client, message):
        user_id = message.from_user.id
        user_name = message.from_user.first_name or "User"
        
        # Check if this is a verification token
        if len(message.command) > 1:
            command_arg = message.command[1]
            
            if command_arg.startswith('verify_'):
                token = command_arg[7:]
                
                # Verify token
                if verification_system is not None:
                    is_verified, verified_user_id, verify_message = await verification_system.verify_user_token(token)
                    
                    if is_verified and verified_user_id == user_id:
                        await message.reply_text(
                            f"✅ **Verification Successful, {user_name}!**\n\n"
                            "You are now verified and can download files.\n\n"
                            f"🌐 **Website:** {Config.WEBSITE_URL}\n"
                            f"⏰ **Verification valid for 6 hours**",
                            reply_markup=InlineKeyboardMarkup([
                                [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)]
                            ])
                        )
                        return
                    else:
                        await message.reply_text(
                            "❌ **Verification Failed**\n\n"
                            f"Error: {verify_message}\n\n"
                            "Please generate a new verification link."
                        )
                        return
        
        # Regular start command
        welcome_text = (
            f"🎬 **Welcome to SK4FiLM, {user_name}!**\n\n"
            "🌐 **Use our website to browse and download movies:**\n"
            f"{Config.WEBSITE_URL}\n\n"
        )
        
        # Check premium status
        is_premium = False
        premium_details = {}
        if premium_system is not None:
            is_premium = await premium_system.is_premium_user(user_id)
            premium_details = await premium_system.get_subscription_details(user_id)
        
        if is_premium:
            welcome_text += f"🌟 **Premium Status:** {premium_details.get('tier_name', 'Premium')}\n"
            welcome_text += f"📅 **Days Remaining:** {premium_details.get('days_remaining', 0)}\n\n"
            welcome_text += "✅ **You have full access to all features!**\n\n"
            
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)],
                [InlineKeyboardButton("📊 PREMIUM STATUS", callback_data=f"premium_status_{user_id}")]
            ])
        elif Config.VERIFICATION_REQUIRED and verification_system is not None:
            # Check if user is verified
            is_verified, status = await verification_system.check_user_verified(user_id)
            
            if not is_verified:
                # Create verification link
                verification_data = await verification_system.create_verification_link(user_id)
                
                welcome_text += (
                    "🔒 **Verification Required**\n"
                    "Please complete verification to download files:\n\n"
                    f"🔗 **Verification Link:** {verification_data['short_url']}\n\n"
                    "Click the link above and then click 'Start' in the bot.\n"
                    "⏰ **Valid for 1 hour**\n\n"
                    "✨ **Or upgrade to Premium for instant access!**"
                )
                
                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔗 VERIFY NOW", url=verification_data['short_url'])],
                    [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")],
                    [InlineKeyboardButton("🔄 CHECK VERIFICATION", callback_data=f"check_verify_{user_id}")]
                ])
            else:
                welcome_text += "✅ **You are verified!**\nYou can download files from the website.\n\n"
                welcome_text += "✨ **Upgrade to Premium for more features!**"
                
                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)],
                    [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")]
                ])
        else:
            welcome_text += "✨ **Start browsing movies now!**\n\n"
            welcome_text += "⭐ **Upgrade to Premium for:**\n• Higher quality\n• More downloads\n• Faster speeds"
            
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)],
                [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")]
            ])
        
        await message.reply_text(welcome_text, reply_markup=keyboard, disable_web_page_preview=True)
    
    @bot.on_callback_query(filters.regex(r"^check_verify_"))
    async def check_verify_callback(client, callback_query):
        user_id = int(callback_query.data.split('_')[2])
        
        if verification_system is None:
            await callback_query.answer("Verification system not available", show_alert=True)
            return
        
        is_verified, message = await verification_system.check_user_verified(user_id)
        
        if is_verified:
            await callback_query.message.edit_text(
                "✅ **Verification Successful!**\n\n"
                "You can now download files from the website.\n\n"
                f"🌐 **Website:** {Config.WEBSITE_URL}",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)]
                ])
            )
        else:
            verification_data = await verification_system.create_verification_link(user_id)
            await callback_query.message.edit_text(
                "❌ **Not Verified Yet**\n\n"
                "Please complete the verification process:\n\n"
                f"🔗 **Verification Link:** {verification_data['short_url']}\n\n"
                "Click the link above and then click 'Start' in the bot.",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔗 VERIFY NOW", url=verification_data['short_url'])],
                    [InlineKeyboardButton("🔄 CHECK AGAIN", callback_data=f"check_verify_{user_id}")]
                ]),
                disable_web_page_preview=True
            )
    
    @bot.on_callback_query(filters.regex(r"^buy_premium$"))
    async def buy_premium_callback(client, callback_query):
        """Show premium plans"""
        if premium_system is None:
            await callback_query.answer("Premium system not available", show_alert=True)
            return
        
        user_id = callback_query.from_user.id
        plans = await premium_system.get_all_plans()
        
        text = "⭐ **SK4FiLM PREMIUM PLANS** ⭐\n\n"
        text += "Upgrade for better quality, more downloads and faster speeds!\n\n"
        
        keyboard = []
        for plan in plans:
            text += f"**{plan['name']}**\n"
            text += f"💰 **Price:** ₹{plan['price']}\n"
            text += f"⏰ **Duration:** {plan['duration_days']} days\n"
            text += "**Features:**\n"
            for feature in plan['features'][:3]:  # Show only 3 features
                text += f"• {feature}\n"
            text += "\n"
            
            keyboard.append([InlineKeyboardButton(
                f"{plan['name']} - ₹{plan['price']}", 
                callback_data=f"select_plan_{plan['tier']}"
            )])
        
        text += "\n**How to purchase:**\n1. Select a plan\n2. Pay using UPI\n3. Send screenshot\n4. Get activated!"
        
        keyboard.append([InlineKeyboardButton("🔙 BACK", callback_data="back_to_start")])
        
        await callback_query.message.edit_text(
            text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            disable_web_page_preview=True
        )
    
    @bot.on_callback_query(filters.regex(r"^select_plan_"))
    async def select_plan_callback(client, callback_query):
        """Select premium plan and show payment details"""
        if premium_system is None:
            await callback_query.answer("Premium system not available", show_alert=True)
            return
        
        tier_str = callback_query.data.split('_')[2]
        user_id = callback_query.from_user.id
        
        try:
            tier = PremiumTier(tier_str)
        except ValueError:
            await callback_query.answer("Invalid plan", show_alert=True)
            return
        
        # Create payment request
        payment_data = await premium_system.create_payment_request(user_id, tier)
        
        text = f"💰 **Payment for {payment_data['tier_name']}**\n\n"
        text += f"**Amount:** ₹{payment_data['amount']}\n"
        text += f"**UPI ID:** `{payment_data['upi_id']}`\n\n"
        text += "**Payment Instructions:**\n"
        text += "1. Scan the QR code below OR\n"
        text += "2. Send ₹{amount} to UPI ID: `{upi_id}`\n".format(**payment_data)
        text += "3. Take screenshot of payment\n"
        text += "4. Send screenshot to this bot\n\n"
        text += "⏰ **Payment valid for 1 hour**\n"
        text += "✅ **Admin will activate within 24 hours**"
        
        keyboard = [
            [InlineKeyboardButton("📸 SEND SCREENSHOT", callback_data=f"send_screenshot_{payment_data['payment_id']}")],
            [InlineKeyboardButton("🔙 BACK TO PLANS", callback_data="buy_premium")]
        ]
        
        await callback_query.message.delete()
        
        # Send QR code if available
        if payment_data.get('qr_code'):
            # For QR code display, we'd need to handle it differently
            # For now, just send text
            await callback_query.message.reply_text(
                text,
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
        else:
            await callback_query.message.reply_text(
                text,
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
    
    @bot.on_callback_query(filters.regex(r"^premium_status_"))
    async def premium_status_callback(client, callback_query):
        """Show premium status"""
        if premium_system is None:
            await callback_query.answer("Premium system not available", show_alert=True)
            return
        
        user_id = int(callback_query.data.split('_')[2])
        details = await premium_system.get_subscription_details(user_id)
        
        text = f"⭐ **PREMIUM STATUS**\n\n"
        text += f"**Plan:** {details['tier_name']}\n"
        text += f"**Status:** {'✅ Active' if details['is_active'] else '❌ Inactive'}\n"
        
        if details['expires_at']:
            expires = datetime.fromisoformat(details['expires_at']) if isinstance(details['expires_at'], str) else details['expires_at']
            text += f"**Expires:** {expires.strftime('%d %b %Y')}\n"
            text += f"**Days Remaining:** {details['days_remaining']}\n"
        
        text += f"\n**Features:**\n"
        for feature in details['features'][:5]:  # Show only 5 features
            text += f"• {feature}\n"
        
        text += f"\n**Downloads Today:** {details.get('daily_downloads', 0)}/{details['limits']['daily_downloads']}\n"
        text += f"**Total Downloads:** {details.get('total_downloads', 0)}\n"
        
        keyboard = [
            [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)],
            [InlineKeyboardButton("🔙 BACK", callback_data="back_to_start")]
        ]
        
        await callback_query.message.edit_text(
            text,
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    @bot.on_message(filters.command("premium") & filters.private)
    async def premium_command(client, message):
        """Premium command"""
        if premium_system is None:
            await message.reply_text("Premium system is not available.")
            return
        
        user_id = message.from_user.id
        is_premium = await premium_system.is_premium_user(user_id)
        
        if is_premium:
            details = await premium_system.get_subscription_details(user_id)
            text = f"⭐ **You are a Premium User!**\n\n"
            text += f"**Plan:** {details['tier_name']}\n"
            text += f"**Days Remaining:** {details['days_remaining']}\n\n"
            text += "✅ **You have full access to all features!**"
            
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)]
            ])
        else:
            text = "⭐ **Upgrade to Premium!**\n\n"
            text += "Get access to:\n"
            text += "• Higher quality (1080p/4K)\n"
            text += "• More daily downloads\n"
            text += "• Faster download speeds\n"
            text += "• No verification required\n"
            text += "• Priority support\n\n"
            text += "Click below to view plans:"
            
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("⭐ VIEW PLANS", callback_data="buy_premium")],
                [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)]
            ])
        
        await message.reply_text(text, reply_markup=keyboard)
    
    @bot.on_message(filters.command("verify") & filters.private)
    async def verify_command(client, message):
        """Verification command"""
        if verification_system is None:
            await message.reply_text("Verification system is not available.")
            return
        
        user_id = message.from_user.id
        user_name = message.from_user.first_name or "User"
        
        # Check if premium user (bypass verification)
        if premium_system is not None:
            is_premium = await premium_system.is_premium_user(user_id)
            if is_premium:
                await message.reply_text(
                    f"✅ **Premium User Detected!**\n\n"
                    f"As a premium user, you don't need verification.\n"
                    f"You have full access to all features, {user_name}! 🎬"
                )
                return
        
        is_verified, status = await verification_system.check_user_verified(user_id)
        
        if is_verified:
            await message.reply_text(
                f"✅ **Already Verified, {user_name}!**\n\n"
                f"Your verification is active and valid for 6 hours.\n"
                "You can download files from the website now! 🎬"
            )
        else:
            verification_data = await verification_system.create_verification_link(user_id)
            await message.reply_text(
                f"🔗 **Verification Required, {user_name}**\n\n"
                "To download files, please complete the URL verification:\n\n"
                f"**Verification URL:** {verification_data['short_url']}\n\n"
                "⏰ **Valid for 1 hour**\n\n"
                "Click the link above and then click 'Start' in the bot.",
                disable_web_page_preview=True
            )
    
    @bot.on_message(filters.command("premiumuser") & filters.user(Config.ADMIN_IDS))
    async def premium_user_admin(client, message):
        """Admin command to activate premium for user"""
        if premium_system is None:
            await message.reply_text("Premium system is not available.")
            return
        
        try:
            parts = message.text.split()
            if len(parts) < 2:
                await message.reply_text(
                    "Usage: /premiumuser <user_id> [plan]\n\n"
                    "Plans: basic, premium, ultimate, lifetime\n"
                    "Example: /premiumuser 123456789 premium"
                )
                return
            
            user_id = int(parts[1])
            tier_str = parts[2] if len(parts) > 2 else "premium"
            
            try:
                tier = PremiumTier(tier_str)
            except ValueError:
                await message.reply_text(
                    "Invalid plan. Available plans:\n"
                    "- basic\n- premium\n- ultimate\n- lifetime"
                )
                return
            
            # Activate premium
            subscription = await premium_system.activate_premium(
                admin_id=message.from_user.id,
                user_id=user_id,
                tier=tier
            )
            
            await message.reply_text(
                f"✅ **Premium Activated!**\n\n"
                f"**User:** {user_id}\n"
                f"**Plan:** {subscription['tier_name']}\n"
                f"**Expires:** {subscription['expires_at'].strftime('%d %b %Y')}\n"
                f"**Days:** {subscription['duration_days']}\n\n"
                f"User will receive a notification."
            )
            
        except Exception as e:
            await message.reply_text(f"❌ Error: {e}")
    
    @bot.on_message(filters.command("stats") & filters.user(Config.ADMIN_IDS))
    async def stats_command(client, message):
        """Admin stats command"""
        try:
            # Get MongoDB stats
            total_files = await files_col.count_documents({}) if files_col is not None else 0
            video_files = await files_col.count_documents({'is_video_file': True}) if files_col is not None else 0
            
            # Get verification stats
            verification_stats = {}
            if verification_system is not None:
                verification_stats = await verification_system.get_user_stats()
            
            # Get premium stats
            premium_stats = {}
            if premium_system is not None:
                premium_stats = await premium_system.get_admin_stats()
            
            # Get cache stats
            cache_stats = {}
            if cache_manager is not None:
                cache_stats = await cache_manager.get_stats_summary()
            
            text = "📊 **SK4FiLM STATISTICS**\n\n"
            text += f"📁 **Total Files:** {total_files}\n"
            text += f"🎥 **Video Files:** {video_files}\n"
            text += f"🔐 **Pending Verifications:** {verification_stats.get('pending_verifications', 0)}\n"
            text += f"✅ **Verified Users:** {verification_stats.get('verified_users', 0)}\n"
            text += f"⭐ **Premium Users:** {premium_stats.get('active_premium_users', 0)}\n"
            text += f"💰 **Total Revenue:** ₹{premium_stats.get('total_revenue', 0)}\n"
            text += f"🔧 **Redis Enabled:** {cache_stats.get('redis_enabled', False)}\n"
            text += f"📡 **Bot Status:** {'✅ Online' if bot_started else '⏳ Starting'}\n"
            text += f"👤 **User Session:** {'✅ Ready' if user_session_ready else '⏳ Pending'}\n\n"
            text += "⚡ **All systems operational!**"
            
            await message.reply_text(text)
            
        except Exception as e:
            await message.reply_text(f"❌ Error getting stats: {e}")
    
    @bot.on_message(filters.command("broadcast") & filters.user(Config.ADMIN_IDS))
    async def broadcast_command(client, message):
        """Broadcast to premium users"""
        if premium_system is None:
            await message.reply_text("Premium system is not available.")
            return
        
        try:
            # Get message from reply or command
            if message.reply_to_message:
                broadcast_text = message.reply_to_message.text or message.reply_to_message.caption
            else:
                parts = message.text.split(' ', 1)
                if len(parts) < 2:
                    await message.reply_text("Usage: /broadcast <message> or reply to a message")
                    return
                broadcast_text = parts[1]
            
            if not broadcast_text:
                await message.reply_text("No message to broadcast")
                return
            
            result = await premium_system.broadcast_to_premium_users(broadcast_text)
            
            await message.reply_text(
                f"📢 **Broadcast Scheduled**\n\n"
                f"**Message:** {broadcast_text[:50]}...\n"
                f"**Users:** {result.get('user_count', 0)}\n"
                f"**Status:** {result.get('status', 'unknown')}\n\n"
                f"Messages will be sent shortly."
            )
            
        except Exception as e:
            await message.reply_text(f"❌ Error: {e}")
    
    @bot.on_message(filters.command("index") & filters.user(Config.ADMIN_IDS))
    async def index_command(client, message):
        """Index files from channel"""
        if not user_session_ready:
            await message.reply_text("User session not ready. Cannot index files.")
            return
        
        msg = await message.reply_text("🔄 **Starting file indexing...**")
        
        try:
            total = 0
            async for message in safe_telegram_generator(
                user_client.get_chat_history,
                Config.FILE_CHANNEL_ID,
                limit=100
            ):
                if message and (message.document or message.video):
                    title = extract_title_from_file(message)
                    if title:
                        file_id = message.document.file_id if message.document else message.video.file_id
                        file_size = message.document.file_size if message.document else (message.video.file_size if message.video else 0)
                        file_name = message.document.file_name if message.document else (message.video.file_name if message.video else 'video.mp4')
                        quality = detect_quality(file_name)
                        
                        file_data = {
                            'channel_id': Config.FILE_CHANNEL_ID,
                            'message_id': message.id,
                            'title': title,
                            'normalized_title': normalize_title(title),
                            'file_id': file_id,
                            'quality': quality,
                            'file_size': file_size,
                            'file_name': file_name,
                            'caption': message.caption or '',
                            'date': message.date,
                            'indexed_at': datetime.now(),
                            'is_video_file': is_video_file(file_name)
                        }
                        
                        if files_col is not None:
                            await files_col.update_one(
                                {'channel_id': Config.FILE_CHANNEL_ID, 'message_id': message.id},
                                {'$set': file_data},
                                upsert=True
                            )
                        
                        total += 1
                        if total % 10 == 0:
                            await msg.edit_text(f"🔄 **Indexing...** {total} files processed")
            
            await msg.edit_text(f"✅ **Indexing Complete!**\n\n**Total files indexed:** {total}")
            
        except Exception as e:
            await msg.edit_text(f"❌ **Indexing Failed:** {e}")
    
    @bot.on_message(filters.text & filters.private & ~filters.command(['start', 'stats', 'premium', 'verify', 'index', 'broadcast', 'premiumuser']))
    async def text_handler(client, message):
        """Handle file download links"""
        user_id = message.from_user.id
        user_name = message.from_user.first_name or "User"
        
        # Check if message contains a file link
        if message.text and '_' in message.text and message.text.replace('_', '').isdigit():
            # This could be a file link format: channel_message_quality
            try:
                parts = message.text.split('_')
                if len(parts) >= 2:
                    channel_id = int(parts[0])
                    message_id = int(parts[1])
                    quality = parts[2] if len(parts) > 2 else "HD"
                    
                    # Check user access
                    can_download = True
                    message_text = "Access granted"
                    
                    if premium_system is not None:
                        # Premium users bypass all checks
                        is_premium = await premium_system.is_premium_user(user_id)
                        if not is_premium:
                            # Check download limits for free users
                            can_download, message_text, details = await premium_system.can_user_download(user_id)
                    
                    if not can_download:
                        await message.reply_text(
                            f"❌ **Download Failed**\n\n"
                            f"{message_text}\n\n"
                            f"⭐ **Upgrade to Premium for unlimited downloads!**",
                            reply_markup=InlineKeyboardMarkup([
                                [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")]
                            ])
                        )
                        return
                    
                    processing_msg = await message.reply_text(f"⏳ **Preparing your file...**\n\n📦 Quality: {quality}")
                    
                    # Get file from channel
                    file_message = await safe_telegram_operation(
                        bot.get_messages,
                        channel_id, 
                        message_id
                    )
                    
                    if not file_message or (not file_message.document and not file_message.video):
                        await processing_msg.edit_text("❌ **File not found**\n\nThe file may have been deleted.")
                        return
                    
                    # Send file to user
                    if file_message.document:
                        sent = await safe_telegram_operation(
                            bot.send_document,
                            user_id, 
                            file_message.document.file_id, 
                            caption=f"♻ **Please forward this file/video to your saved messages**\n\n"
                                   f"📹 Quality: {quality}\n"
                                   f"📦 Size: {format_size(file_message.document.file_size)}\n\n"
                                   f"⚠️ Will auto-delete in {Config.AUTO_DELETE_TIME//60} minutes\n\n"
                                   f"@SK4FiLM 🍿"
                        )
                    else:
                        sent = await safe_telegram_operation(
                            bot.send_video,
                            user_id, 
                            file_message.video.file_id, 
                            caption=f"♻ **Please forward this file/video to your saved messages**\n\n"
                                   f"📹 Quality: {quality}\n" 
                                   f"📦 Size: {format_size(file_message.video.file_size)}\n\n"
                                   f"⚠️ Will auto-delete in {Config.AUTO_DELETE_TIME//60} minutes\n\n"
                                   f"@SK4FiLM 🍿"
                        )
                    
                    await processing_msg.delete()
                    
                    # Record download for user
                    if premium_system is not None:
                        await premium_system.record_download(user_id)
                    
                    # Auto-delete if configured
                    if Config.AUTO_DELETE_TIME > 0:
                        async def auto_delete():
                            await asyncio.sleep(Config.AUTO_DELETE_TIME)
                            try:
                                await sent.delete()
                            except:
                                pass
                        asyncio.create_task(auto_delete())
                    
                    return
                    
            except Exception as e:
                logger.error(f"File download error: {e}")
        
        # Generic text response
        welcome_text = (
            f"👋 **Hi {user_name}!**\n\n"
            "🔍 **Please use our website to search for movies:**\n\n"
            f"{Config.WEBSITE_URL}\n\n"
            "This bot handles file downloads from website links.\n\n"
            "**Commands:**\n"
            "/start - Start the bot\n"
            "/premium - Premium plans\n"
            "/verify - Check verification\n"
            "/help - Help information"
        )
        
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)],
            [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")]
        ])
        
        await message.reply_text(welcome_text, reply_markup=keyboard, disable_web_page_preview=True)

# Background tasks
async def cache_cleanup():
    """Cleanup expired cache"""
    while True:
        await asyncio.sleep(3600)
        if cache_manager is not None:
            await cache_manager.clear_pattern("search:*")
            logger.info("🧹 Search cache cleaned")

async def delayed_bot_start(delay: int):
    """Start bot after delay"""
    global bot_started
    
    logger.info(f"⏳ Waiting {delay} seconds before starting bot...")
    await asyncio.sleep(delay)
    
    try:
        await bot.start()
        await setup_bot()
        
        me = await bot.get_me()
        logger.info(f"✅ Bot started after delay: @{me.username}")
        bot_started = True
        
    except Exception as e:
        logger.error(f"❌ Delayed bot start failed: {e}")
        await asyncio.sleep(300)
        asyncio.create_task(delayed_bot_start(60))

async def delayed_user_session_start(delay: int):
    """Start user session after delay"""
    global user_session_ready
    
    await asyncio.sleep(delay)
    
    try:
        await user_client.start()
        user_session_ready = True
        logger.info("✅ User session started after delay")
    except Exception as e:
        logger.error(f"❌ Delayed user session failed: {e}")

# Initialization
async def init_modules():
    """Initialize all modules"""
    global verification_system, premium_system, poster_fetcher, cache_manager
    
    logger.info("🔄 Initializing modules...")
    
    # Initialize Cache Manager
    cache_manager = CacheManager(Config)
    await cache_manager.init_redis()
    await cache_manager.start_cleanup_task()
    logger.info("✅ Cache manager initialized")
    
    # Initialize Verification System
    verification_system = VerificationSystem(Config)
    await verification_system.start_cleanup_task()
    logger.info("✅ Verification system initialized")
    
    # Initialize Premium System
    premium_system = PremiumSystem(Config)
    await premium_system.start_cleanup_task()
    logger.info("✅ Premium system initialized")
    
    # Initialize Poster Fetcher
    poster_fetcher = PosterFetcher(Config, cache_manager)
    logger.info("✅ Poster fetcher initialized")
    
    return True

async def init_mongodb():
    """Initialize MongoDB"""
    global mongo_client, db, files_col, verification_col
    
    try:
        mongo_client = AsyncIOMotorClient(Config.MONGODB_URI)
        await mongo_client.admin.command('ping')
        
        db = mongo_client.sk4film
        files_col = db.files
        verification_col = db.verifications
        
        # Create indexes
        await files_col.create_index([("title", "text")])
        await files_col.create_index([("normalized_title", 1)])
        await files_col.create_index([("message_id", 1), ("channel_id", 1)], name="msg_ch_idx")
        await files_col.create_index([("indexed_at", -1)])
        
        await verification_col.create_index([("user_id", 1)], unique=True)
        
        logger.info("✅ MongoDB initialized")
        return True
    except Exception as e:
        logger.error(f"❌ MongoDB initialization failed: {e}")
        return False

async def init_bot():
    """Initialize Telegram bot and user client"""
    global bot, user_client, bot_started, user_session_ready
    
    try:
        # Initialize bot
        bot = Client(
            "bot",
            api_id=Config.API_ID,
            api_hash=Config.API_HASH, 
            bot_token=Config.BOT_TOKEN
        )
        
        # Try to start bot with flood wait handling
        try:
            await bot.start()
            await setup_bot()
            
            me = await bot.get_me()
            logger.info(f"✅ Bot started: @{me.username}")
            bot_started = True
            
        except FloodWait as e:
            logger.warning(f"⚠️ Bot flood wait: {e.value}s")
            logger.info(f"⏳ Bot will start automatically in {e.value} seconds...")
            
            asyncio.create_task(delayed_bot_start(e.value))
            bot_started = False
            return True
        
        except Exception as e:
            logger.error(f"❌ Bot initialization failed: {e}")
            return False
        
        # Initialize user client (for file access)
        if Config.USER_SESSION_STRING:
            user_client = Client(
                "user_session", 
                api_id=Config.API_ID, 
                api_hash=Config.API_HASH, 
                session_string=Config.USER_SESSION_STRING,
                no_updates=True
            )
            
            try:
                await user_client.start()
                user_session_ready = True
                logger.info("✅ User session initialized")
            except FloodWait as e:
                logger.warning(f"⚠️ User session flood wait: {e.value}s")
                asyncio.create_task(delayed_user_session_start(e.value + 10))
            except Exception as e:
                logger.error(f"❌ User session failed: {e}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Bot initialization failed: {e}")
        return False

async def main():
    """Main application entry point"""
    logger.info("="*60)
    logger.info("🎬 SK4FiLM v8.0 - Complete System")
    logger.info("✅ Dual Access: Free + Premium")
    logger.info("✅ Admin Control Panel")
    logger.info("✅ Redis Caching")
    logger.info("✅ MongoDB Storage")
    logger.info("✅ Multi-Channel Support")
    logger.info("="*60)
    
    # Initialize modules
    await init_modules()
    
    # Initialize MongoDB
    await init_mongodb()
    
    # Initialize bot
    bot_success = await init_bot()
    
    if not bot_success:
        logger.warning("⚠️ Bot initialization had issues, but web server will start")
    
    # Start cleanup tasks
    asyncio.create_task(cache_cleanup())
    
    # Start web server
    config = HyperConfig()
    config.bind = [f"0.0.0.0:{Config.WEB_SERVER_PORT}"]
    config.loglevel = "warning"
    
    logger.info(f"🌐 Web server starting on port {Config.WEB_SERVER_PORT}...")
    await serve(app, config)

if __name__ == "__main__":
    asyncio.run(main())
