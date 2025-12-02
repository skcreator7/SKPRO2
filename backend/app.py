"""
app.py - Main SK4FiLM Bot Web Server - UPDATED
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

# Global instances
bot_instance = None
db_manager = None
api_rate_limiter = None
search_rate_limiter = None

# Configuration class
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
    VERIFICATION_REQUIRED = os.environ.get("VERIFICATION_REQUIRED", "true").lower() == "true"
    VERIFICATION_DURATION = 6 * 60 * 60
    
    # Server
    WEB_SERVER_PORT = int(os.environ.get("PORT", 8000))
    
    # API Keys
    OMDB_KEYS = os.environ.get("OMDB_KEYS", "8265bd1c,b9bd48a6,3e7e1e9d").split(",")
    TMDB_KEYS = os.environ.get("TMDB_KEYS", "e547e17d4e91f3e62a571655cd1ccaff,8265bd1f").split(",")
    
    # UPI IDs for Premium
    UPI_ID_BASIC = os.environ.get("UPI_ID_BASIC", "sk4filmbot@ybl")
    UPI_ID_PREMIUM = os.environ.get("UPI_ID_PREMIUM", "sk4filmbot@ybl")
    
    # Admin
    ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "admin123")
    
    # Website Integration
    WEBSITE_SECRET_KEY = os.environ.get("WEBSITE_SECRET_KEY", "sk4film_secret_key")
    
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
app.secret_key = Config.WEBSITE_SECRET_KEY

@app.after_request
async def add_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

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

def is_video_file(file_name):
    """Check if file is video"""
    if not file_name:
        return False
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.mpg', '.mpeg']
    file_name_lower = file_name.lower()
    return any(file_name_lower.endswith(ext) for ext in video_extensions)

async def safe_telegram_operation(func, *args, **kwargs):
    """Safely execute Telegram operations with error handling"""
    try:
        return await func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Telegram operation error: {e}")
        return None

async def auto_delete_file(message, delay_seconds):
    """Auto-delete file after specified delay"""
    try:
        await asyncio.sleep(delay_seconds)
        await message.delete()
        logger.info(f"Auto-deleted file message after {delay_seconds} seconds")
    except Exception as e:
        logger.error(f"Auto-delete error: {e}")

# Database Manager
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

# Rate Limiter
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

# Auth decorator for website
def require_website_auth(f):
    @wraps(f)
    async def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'status': 'error', 'message': 'Authorization required'}), 401
        
        token = auth_header.split(' ')[1]
        if token != Config.WEBSITE_SECRET_KEY:
            return jsonify({'status': 'error', 'message': 'Invalid token'}), 403
        
        return await f(*args, **kwargs)
    return decorated_function

async def idle():
    """Keep the bot running"""
    while True:
        await asyncio.sleep(3600)

# API Routes for Website Integration
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
            'verify': '/api/verify',
            'download': '/api/download',
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

@app.route('/api/verify', methods=['POST'])
@require_website_auth
async def api_verify():
    """Get verification link for user (website calls this)"""
    try:
        data = await request.get_json()
        if not data or 'user_id' not in data:
            return jsonify({'status': 'error', 'message': 'user_id required'}), 400
        
        user_id = int(data['user_id'])
        
        if bot_instance and bot_instance.verification_system:
            verification_link = await bot_instance.verification_system.get_verification_link_for_user(user_id)
            
            return jsonify({
                'status': 'success',
                'user_id': user_id,
                'verification_url': verification_link['short_url'],
                'expires_in': '1 hour',
                'bot_username': Config.BOT_USERNAME
            })
        else:
            return jsonify({'status': 'error', 'message': 'Verification system not available'}), 500
            
    except Exception as e:
        logger.error(f"Verification API error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/download', methods=['POST'])
@require_website_auth
async def api_download():
    """Request file download (website calls this)"""
    try:
        data = await request.get_json()
        if not data or 'user_id' not in data or 'channel_id' not in data or 'message_id' not in data:
            return jsonify({'status': 'error', 'message': 'Missing required fields'}), 400
        
        user_id = int(data['user_id'])
        channel_id = int(data['channel_id'])
        message_id = int(data['message_id'])
        quality = data.get('quality', 'HD')
        
        # Check if bot instance is available
        if not bot_instance or not bot_instance.bot:
            return jsonify({'status': 'error', 'message': 'Bot not ready'}), 500
        
        # Check user access
        has_access = False
        access_message = ""
        
        if bot_instance.premium_system:
            try:
                is_premium = await bot_instance.premium_system.is_premium_user(user_id)
                if is_premium:
                    has_access = True
                    access_message = "Premium user"
            except Exception as e:
                logger.error(f"Premium check error: {e}")
        
        if not has_access and bot_instance.verification_system:
            is_verified, verify_msg = await bot_instance.verification_system.check_user_verified(user_id)
            if is_verified:
                has_access = True
                access_message = "Verified user"
        
        if not has_access:
            return jsonify({
                'status': 'verification_required',
                'message': 'Verification required',
                'user_id': user_id
            }), 403
        
        # Get file from Telegram
        file_message = await safe_telegram_operation(
            bot_instance.bot.get_messages,
            channel_id, 
            message_id
        )
        
        if not file_message or (not file_message.document and not file_message.video):
            return jsonify({'status': 'error', 'message': 'File not found'}), 404
        
        # Send file to user via bot
        try:
            if file_message.document:
                sent = await safe_telegram_operation(
                    bot_instance.bot.send_document,
                    user_id, 
                    file_message.document.file_id,
                    caption=f"✅ **File from SK4FiLM**\n\n📦 Quality: {quality}\n📊 Size: {format_size(file_message.document.file_size)}\n⏰ Auto-delete in {Config.AUTO_DELETE_TIME//60} minutes"
                )
            else:
                sent = await safe_telegram_operation(
                    bot_instance.bot.send_video,
                    user_id, 
                    file_message.video.file_id,
                    caption=f"✅ **File from SK4FiLM**\n\n📦 Quality: {quality}\n📊 Size: {format_size(file_message.video.file_size)}\n⏰ Auto-delete in {Config.AUTO_DELETE_TIME//60} minutes"
                )
            
            if sent:
                # Auto-delete file after specified time
                if Config.AUTO_DELETE_TIME > 0:
                    asyncio.create_task(auto_delete_file(sent, Config.AUTO_DELETE_TIME))
                
                # Record download
                if bot_instance.premium_system:
                    try:
                        await bot_instance.premium_system.record_download(user_id)
                    except:
                        pass
                
                logger.info(f"File sent to user {user_id} via website API")
                
                return jsonify({
                    'status': 'success',
                    'message': 'File sent successfully',
                    'user_id': user_id,
                    'access_type': access_message,
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return jsonify({'status': 'error', 'message': 'Failed to send file'}), 500
                
        except Exception as e:
            logger.error(f"Error sending file: {e}")
            return jsonify({'status': 'error', 'message': f'Error sending file: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Download API error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/check-access', methods=['POST'])
@require_website_auth
async def api_check_access():
    """Check user access status (website calls this)"""
    try:
        data = await request.get_json()
        if not data or 'user_id' not in data:
            return jsonify({'status': 'error', 'message': 'user_id required'}), 400
        
        user_id = int(data['user_id'])
        
        has_access = False
        access_type = "none"
        access_message = ""
        verification_link = None
        
        if bot_instance.premium_system:
            try:
                is_premium = await bot_instance.premium_system.is_premium_user(user_id)
                if is_premium:
                    has_access = True
                    access_type = "premium"
                    tier = await bot_instance.premium_system.get_user_tier(user_id)
                    access_message = f"Premium ({tier.value})"
            except Exception as e:
                logger.error(f"Premium check error: {e}")
        
        if not has_access and bot_instance.verification_system:
            is_verified, verify_msg = await bot_instance.verification_system.check_user_verified(user_id)
            if is_verified:
                has_access = True
                access_type = "verified"
                access_message = verify_msg
            else:
                # Get verification link
                verification_link_data = await bot_instance.verification_system.get_verification_link_for_user(user_id)
                if verification_link_data:
                    verification_link = verification_link_data['short_url']
        
        return jsonify({
            'status': 'success',
            'user_id': user_id,
            'has_access': has_access,
            'access_type': access_type,
            'access_message': access_message,
            'verification_required': not has_access,
            'verification_link': verification_link,
            'bot_username': Config.BOT_USERNAME
        })
            
    except Exception as e:
        logger.error(f"Check access API error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/send-message', methods=['POST'])
@require_website_auth
async def api_send_message():
    """Send message to user (website calls this)"""
    try:
        data = await request.get_json()
        if not data or 'user_id' not in data or 'message' not in data:
            return jsonify({'status': 'error', 'message': 'user_id and message required'}), 400
        
        user_id = int(data['user_id'])
        message_text = data['message']
        
        if not bot_instance or not bot_instance.bot:
            return jsonify({'status': 'error', 'message': 'Bot not ready'}), 500
        
        # Send message to user
        sent = await safe_telegram_operation(
            bot_instance.bot.send_message,
            user_id,
            message_text
        )
        
        if sent:
            return jsonify({
                'status': 'success',
                'message': 'Message sent',
                'user_id': user_id
            })
        else:
            return jsonify({'status': 'error', 'message': 'Failed to send message'}), 500
            
    except Exception as e:
        logger.error(f"Send message API error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

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
        
        # Import bot_handlers
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
        logger.info(f"🔗 Backend API: {config.BACKEND_URL}")
        
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
