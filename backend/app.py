"""
app.py - Main SK4FiLM Bot Web Server
"""
import asyncio
import os
import logging
import json
import re
import math
import html
import time
import secrets
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import defaultdict
from functools import wraps

import aiohttp
import urllib.parse
from quart import Quart, jsonify, request, Response
from hypercorn.asyncio import serve
from hypercorn.config import Config as HyperConfig
from motor.motor_asyncio import AsyncIOMotorClient
import redis.asyncio as redis

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('sk4film.log')
    ]
)
logger = logging.getLogger(__name__)

# Configuration class - Moved here to avoid circular imports
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
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        errors = []
        
        if not cls.API_ID or cls.API_ID == 0:
            errors.append("API_ID is required")
        
        if not cls.API_HASH:
            errors.append("API_HASH is required")
        
        if not cls.BOT_TOKEN:
            errors.append("BOT_TOKEN is required")
        
        if not cls.ADMIN_IDS:
            errors.append("ADMIN_IDS is required")
        
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")

# Initialize Quart app
app = Quart(__name__)

@app.after_request
async def add_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

# Global instances - Will be set by main()
bot_instance = None
db_manager = None

# Rate limiters
api_rate_limiter = None
search_rate_limiter = None

# Enhanced search statistics
search_stats = {
    'redis_hits': 0,
    'redis_misses': 0,
    'multi_channel_searches': 0,
    'total_searches': 0
}

# Channel configuration
CHANNEL_CONFIG = {
    -1001891090100: {'name': 'SK4FiLM Main', 'type': 'text', 'search_priority': 1},
    -1002024811395: {'name': 'SK4FiLM Updates', 'type': 'text', 'search_priority': 2},
    -1001768249569: {'name': 'SK4FiLM Files', 'type': 'file', 'search_priority': 0}
}

# Utility functions - These are standalone functions used by both modules
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

# Database Manager (moved here to avoid circular imports)
class DatabaseManager:
    def __init__(self, uri, max_pool_size=10):
        self.uri = uri
        self.max_pool_size = max_pool_size
        self.client = None
        self.db = None
        self.files_col = None
        self.users_col = None
        self.premium_col = None
    
    async def connect(self):
        """Connect to MongoDB with connection pooling"""
        try:
            self.client = AsyncIOMotorClient(
                self.uri,
                maxPoolSize=self.max_pool_size,
                minPoolSize=5,
                maxIdleTimeMS=60000,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000
            )
            
            # Test connection
            await self.client.admin.command('ping')
            
            self.db = self.client['sk4film']
            self.files_col = self.db['files']
            self.users_col = self.db['users']
            self.premium_col = self.db['premium']
            
            # Create indexes
            await self.files_col.create_index([('normalized_title', 'text')])
            await self.files_col.create_index([('channel_id', 1), ('message_id', 1)], unique=True)
            await self.files_col.create_index([('date', -1)])
            await self.files_col.create_index([('is_video_file', 1)])
            
            await self.premium_col.create_index([('user_id', 1)], unique=True)
            await self.premium_col.create_index([('expires_at', 1)])
            
            logger.info("✅ MongoDB connected with connection pooling")
            return True
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            return False
    
    async def close(self):
        """Close database connection"""
        if self.client:
            self.client.close()
            logger.info("✅ MongoDB connection closed")

# Rate Limiter (moved here)
class RateLimiter:
    def __init__(self, max_requests, window_seconds):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
    
    async def is_allowed(self, key):
        now = time.time()
        window_start = now - self.window_seconds
        
        # Clean old requests
        self.requests[key] = [req_time for req_time in self.requests[key] 
                             if req_time > window_start]
        
        # Check limit
        if len(self.requests[key]) >= self.max_requests:
            return False
        
        # Add current request
        self.requests[key].append(now)
        return True

# API Routes
@app.route('/')
async def root():
    """Root endpoint with system status"""
    bot_started = bot_instance.bot_started if bot_instance else False
    user_session_ready = bot_instance.user_session_ready if bot_instance else False
    
    return jsonify({
        'status': 'healthy' if bot_started else 'starting',
        'service': 'SK4FiLM v8.0 - Complete System',
        'version': '8.0',
        'timestamp': datetime.now().isoformat(),
        'bot': {
            'started': bot_started,
            'user_session_ready': user_session_ready,
            'username': Config.BOT_USERNAME
        },
        'website': Config.WEBSITE_URL,
        'endpoints': {
            'api': '/api/*',
            'health': '/health',
            'search': '/api/search',
            'movies': '/api/movies',
            'verify': '/api/verify/{user_id}',
            'premium': '/api/premium/*',
            'poster': '/api/poster',
            'stats': '/api/search/stats',
            'system': '/api/system/stats'
        }
    })

@app.route('/health')
async def health():
    """Health check endpoint"""
    bot_started = bot_instance.bot_started if bot_instance else False
    
    return jsonify({
        'status': 'ok' if bot_started else 'starting',
        'timestamp': datetime.now().isoformat(),
        'bot': {
            'started': bot_started,
            'status': 'online' if bot_started else 'starting'
        },
        'web_server': True
    })

@app.route('/api/search')
async def api_search():
    """Search for movies"""
    try:
        # Get query parameters
        query = request.args.get('query', '').strip()
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 12))
        
        # Validate query
        if len(query) < 2:
            return jsonify({'status': 'error', 'message': 'Query too short'}), 400
        
        # Basic search response
        response_data = {
            'status': 'success',
            'query': query,
            'results': [],
            'pagination': {
                'current_page': page,
                'total_pages': 1,
                'total_results': 0,
                'per_page': limit,
                'has_next': False,
                'has_previous': False
            },
            'search_mode': 'basic',
            'bot_username': Config.BOT_USERNAME
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/movies')
async def api_movies():
    """Get latest movies"""
    try:
        return jsonify({
            'status': 'success',
            'movies': [],
            'total': 0,
            'mode': 'basic',
            'bot_username': Config.BOT_USERNAME
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/verify/<int:user_id>')
async def api_verify_user(user_id):
    """Get verification link for user"""
    try:
        return jsonify({
            'status': 'success',
            'user_id': user_id,
            'verification_url': f'https://t.me/{Config.BOT_USERNAME}?start=verify_{user_id}',
            'expires_in': '1 hour',
            'bot_username': Config.BOT_USERNAME
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/premium/plans')
async def api_premium_plans():
    """Get all premium plans"""
    try:
        plans = [
            {
                'tier': 'basic',
                'name': 'Basic Plan',
                'price': 99,
                'duration_days': 30,
                'features': ['1080p Quality', '10 Daily Downloads', 'Priority Support'],
                'description': 'Perfect for casual users'
            },
            {
                'tier': 'premium',
                'name': 'Premium Plan',
                'price': 199,
                'duration_days': 30,
                'features': ['4K Quality', 'Unlimited Downloads', 'Priority Support', 'No Ads'],
                'description': 'Best value for movie lovers'
            }
        ]
        
        return jsonify({
            'status': 'success',
            'plans': plans,
            'currency': 'INR'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/poster')
async def api_poster():
    """Get poster for movie"""
    try:
        title = request.args.get('title', '').strip()
        if not title:
            return jsonify({'status': 'error', 'message': 'Title required'}), 400
        
        return jsonify({
            'status': 'success',
            'poster_url': f"{Config.BACKEND_URL}/static/default_poster.jpg",
            'source': 'default',
            'rating': '0.0',
            'year': '2024'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Import bot_handlers AFTER defining all utility functions
# This is done in main() function to avoid circular imports

async def main():
    """Main function to start the bot"""
    global bot_instance, db_manager, api_rate_limiter, search_rate_limiter
    
    try:
        # Initialize configuration
        config = Config()
        config.validate()
        
        # Initialize rate limiters
        api_rate_limiter = RateLimiter(max_requests=100, window_seconds=300)
        search_rate_limiter = RateLimiter(max_requests=30, window_seconds=60)
        
        # Initialize database if MONGODB_URI is provided
        if config.MONGODB_URI and config.MONGODB_URI != "mongodb://localhost:27017":
            db_manager = DatabaseManager(config.MONGODB_URI)
            await db_manager.connect()
        
        # Now import bot_handlers (after all dependencies are defined)
        from bot_handlers import SK4FiLMBot, setup_bot_handlers
        
        # Create bot instance
        bot_instance = SK4FiLMBot(config, db_manager)
        
        # Initialize bot
        await bot_instance.initialize()
        
        # Setup bot handlers
        await setup_bot_handlers(bot_instance.bot, bot_instance)
        
        # Start web server
        hypercorn_config = HyperConfig()
        hypercorn_config.bind = [f"0.0.0.0:{config.WEB_SERVER_PORT}"]
        hypercorn_config.workers = 1
        
        logger.info(f"🚀 Starting web server on port {config.WEB_SERVER_PORT}")
        logger.info(f"🤖 Bot username: @{config.BOT_USERNAME}")
        logger.info(f"🌐 Website: {config.WEBSITE_URL}")
        
        # Run both web server and bot
        await asyncio.gather(
            serve(app, hypercorn_config),
            idle()
        )
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if bot_instance:
            await bot_instance.shutdown()
        if db_manager:
            await db_manager.close()
        logger.info("Bot stopped")

if __name__ == "__main__":
    asyncio.run(main())
