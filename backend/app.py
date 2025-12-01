"""
app.py - Main application with all modules integrated
"""
import asyncio
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from pyrogram import Client, filters
from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from pyrogram.errors import UserNotParticipant, ChatAdminRequired, ChannelPrivate, FloodWait

from quart import Quart, jsonify, request, Response
from hypercorn.asyncio import serve
from hypercorn.config import Config as HyperConfig

from motor.motor_asyncio import AsyncIOMotorClient

# Import our modules
from verification import VerificationSystem
from premium import PremiumSystem, PremiumTier
from poster_fetching import PosterFetcher
from cache import CacheManager

# Import other utilities
import html
import re
import math
import aiohttp
import urllib.parse
import base64
from io import BytesIO
import time
import json

# FAST LOADING OPTIMIZATIONS
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger('asyncio').setLevel(logging.WARNING)
logging.getLogger('pyrogram').setLevel(logging.WARNING)
logging.getLogger('hypercorn').setLevel(logging.WARNING)

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
    
    # Shortener & Verification
    SHORTLINK_API = os.environ.get("SHORTLINK_API", "")
    VERIFICATION_REQUIRED = os.environ.get("VERIFICATION_REQUIRED", "false").lower() == "true"
    VERIFICATION_DURATION = 6 * 60 * 60
    
    # Website & Bot
    WEBSITE_URL = os.environ.get("WEBSITE_URL", "https://sk4film.vercel.app")
    BOT_USERNAME = os.environ.get("BOT_USERNAME", "sk4filmbot")
    ADMIN_IDS = [int(x) for x in os.environ.get("ADMIN_IDS", "123456789").split(",")]
    AUTO_DELETE_TIME = int(os.environ.get("AUTO_DELETE_TIME", "300"))
    WEB_SERVER_PORT = int(os.environ.get("PORT", 8000))
    BACKEND_URL = os.environ.get("BACKEND_URL", "https://sk4film.koyeb.app")
    
    # API Keys
    OMDB_KEYS = ["8265bd1c", "b9bd48a6", "3e7e1e9d"]
    TMDB_KEYS = ["e547e17d4e91f3e62a571655cd1ccaff", "8265bd1f"]

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

# API Routes
@app.route('/')
async def root():
    """Root endpoint with system status"""
    mongodb_count = 0
    if files_col:
        try:
            mongodb_count = await files_col.count_documents({})
        except:
            pass
    
    return jsonify({
        'status': 'healthy',
        'service': 'SK4FiLM v8.0 - Modular Architecture',
        'version': '8.0',
        'timestamp': datetime.now().isoformat(),
        'modules': {
            'verification': verification_system is not None,
            'premium': premium_system is not None,
            'poster_fetcher': poster_fetcher is not None,
            'cache': cache_manager is not None
        },
        'bot': {
            'status': 'online' if bot_started else 'starting',
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
            'cache': '/api/cache/*'
        }
    })

@app.route('/health')
async def health():
    """Health check endpoint"""
    mongodb_status = False
    if mongo_client:
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
            'redis': cache_manager.redis_enabled if cache_manager else False,
            'web_server': True
        },
        'modules': {
            'verification': verification_system is not None,
            'premium': premium_system is not None,
            'poster_fetcher': poster_fetcher is not None,
            'cache': cache_manager is not None
        }
    })

@app.route('/api/status')
async def api_status():
    """Get detailed system status"""
    # Get verification stats
    verification_stats = await verification_system.get_user_stats() if verification_system else {}
    
    # Get cache stats
    cache_stats = await cache_manager.get_stats_summary() if cache_manager else {}
    
    # Get poster fetcher stats
    poster_stats = poster_fetcher.get_stats() if poster_fetcher else {}
    
    return jsonify({
        'status': 'success',
        'timestamp': datetime.now().isoformat(),
        'services': {
            'bot': {
                'started': bot_started,
                'username': Config.BOT_USERNAME,
                'user_session_ready': user_session_ready
            },
            'mongodb': mongo_client is not None,
            'redis': cache_manager.redis_enabled if cache_manager else False,
            'web_server': True
        },
        'statistics': {
            'verification': verification_stats,
            'cache': cache_stats,
            'poster_fetcher': poster_stats
        },
        'configuration': {
            'verification_required': Config.VERIFICATION_REQUIRED,
            'premium_enabled': premium_system is not None,
            'channels': len(Config.TEXT_CHANNEL_IDS),
            'auto_delete_minutes': Config.AUTO_DELETE_TIME // 60
        }
    })

@app.route('/api/verify/<int:user_id>', methods=['POST'])
async def api_verify_user(user_id):
    """Create verification link for user"""
    if not verification_system:
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
    if not verification_system:
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

@app.route('/api/premium/tiers')
async def api_get_premium_tiers():
    """Get available premium tiers"""
    if not premium_system:
        return jsonify({'status': 'error', 'message': 'Premium system not initialized'}), 500
    
    try:
        tiers = await premium_system.get_all_tiers()
        return jsonify({
            'status': 'success',
            'tiers': tiers
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/premium/user/<int:user_id>')
async def api_get_user_premium(user_id):
    """Get user premium status"""
    if not premium_system:
        return jsonify({'status': 'error', 'message': 'Premium system not initialized'}), 500
    
    try:
        details = await premium_system.get_subscription_details(user_id)
        return jsonify({
            'status': 'success',
            'subscription': details
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/poster/<path:title>')
async def api_get_poster(title):
    """Get movie poster"""
    if not poster_fetcher:
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
    if not cache_manager:
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
    if not cache_manager:
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
        
        # Check if user is verified if verification is required
        if Config.VERIFICATION_REQUIRED:
            user_id = request.args.get('user_id', type=int)
            if user_id:
                if verification_system:
                    is_verified, message = await verification_system.check_user_verified(user_id)
                    if not is_verified:
                        return jsonify({
                            'status': 'verification_required',
                            'message': 'User verification required',
                            'verification_url': f'/api/verify/{user_id}'
                        }), 403
        
        # Simple search logic (you can enhance this later)
        results = []
        if files_col:
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
                    'quality': doc.get('quality', '480p'),
                    'file_size': doc.get('file_size'),
                    'date': doc.get('date'),
                    'is_new': is_new(doc.get('date')) if doc.get('date') else False
                })
        
        total = len(results)
        
        return jsonify({
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
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

def is_new(date):
    """Check if date is within 48 hours"""
    try:
        if isinstance(date, str):
            date = datetime.fromisoformat(date.replace('Z', '+00:00'))
        hours = (datetime.now() - date.replace(tzinfo=None)).total_seconds() / 3600
        return hours <= 48
    except:
        return False

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
        
        if Config.VERIFICATION_REQUIRED:
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
                    "⏰ **Valid for 1 hour**"
                )
                
                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔗 VERIFY NOW", url=verification_data['short_url'])],
                    [InlineKeyboardButton("🔄 CHECK VERIFICATION", callback_data=f"check_verify_{user_id}")]
                ])
            else:
                welcome_text += "✅ **You are verified!**\nYou can download files from the website.\n\n"
                keyboard = InlineKeyboardMarkup([
                    [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)]
                ])
        else:
            welcome_text += "✨ **Start browsing movies now!**\n\n"
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)]
            ])
        
        # Add premium info if available
        if premium_system:
            tier = await premium_system.get_user_tier(user_id)
            if tier != PremiumTier.FREE:
                welcome_text += f"🌟 **Premium Status:** {tier.value.upper()}\n"
        
        await message.reply_text(welcome_text, reply_markup=keyboard, disable_web_page_preview=True)
    
    @bot.on_callback_query(filters.regex(r"^check_verify_"))
    async def check_verify_callback(client, callback_query):
        user_id = int(callback_query.data.split('_')[2])
        
        if not verification_system:
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
    
    @bot.on_message(filters.command("stats") & filters.user(Config.ADMIN_IDS))
    async def stats_handler(client, message):
        """Admin stats command"""
        try:
            # Get MongoDB stats
            total_files = await files_col.count_documents({}) if files_col else 0
            video_files = await files_col.count_documents({'is_video_file': True}) if files_col else 0
            
            # Get verification stats
            verification_stats = await verification_system.get_user_stats() if verification_system else {}
            
            # Get cache stats
            cache_stats = await cache_manager.get_stats_summary() if cache_manager else {}
            
            stats_text = (
                f"📊 **SK4FiLM STATISTICS**\n\n"
                f"📁 **Total Files:** {total_files}\n"
                f"🎥 **Video Files:** {video_files}\n"
                f"🔐 **Pending Verifications:** {verification_stats.get('pending_verifications', 0)}\n"
                f"✅ **Verified Users:** {verification_stats.get('verified_users', 0)}\n"
                f"🔧 **Redis Enabled:** {cache_stats.get('redis_enabled', False)}\n"
                f"📡 **Bot Status:** {'✅ Online' if bot_started else '⏳ Starting'}\n"
                f"👤 **User Session:** {'✅ Ready' if user_session_ready else '⏳ Pending'}\n\n"
                f"⚡ **All systems operational!**"
            )
            
            await message.reply_text(stats_text)
            
        except Exception as e:
            await message.reply_text(f"❌ Error getting stats: {e}")
    
    @bot.on_message(filters.command("help") & filters.private)
    async def help_handler(client, message):
        """Help command"""
        help_text = (
            "🎬 **SK4FiLM Bot Help**\n\n"
            "**Available Commands:**\n"
            "/start - Start the bot\n"
            "/help - Show this help message\n"
            "\n"
            "**How to use:**\n"
            "1. Browse movies on our website\n"
            "2. Click on any movie to get download link\n"
            "3. Send the link to this bot\n"
            "4. Bot will send you the file\n"
            "\n"
            "**Website:** {Config.WEBSITE_URL}\n"
            "\n"
            "**Support:** @SK4FiLM_Support"
        )
        
        await message.reply_text(help_text, disable_web_page_preview=True)
    
    @bot.on_message(filters.text & filters.private & ~filters.command(['start', 'stats', 'help']))
    async def text_handler(client, message):
        """Handle text messages"""
        user_name = message.from_user.first_name or "User"
        
        # Check if message contains a file link
        if 't.me' in message.text and ('/' in message.text):
            # This could be a file link
            await message.reply_text(
                f"🔗 **File Link Detected**\n\n"
                f"Processing your request, {user_name}...\n\n"
                f"⏳ Please wait while we fetch the file.",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)]
                ])
            )
            # TODO: Implement file link processing
        else:
            # Generic text response
            await message.reply_text(
                f"👋 **Hi {user_name}!**\n\n"
                "🔍 **Please use our website to search for movies:**\n\n"
                f"{Config.WEBSITE_URL}\n\n"
                "This bot handles file downloads from website links.",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)]
                ]),
                disable_web_page_preview=True
            )

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
        # Try again in 5 minutes
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
    global mongo_client, db, files_col
    
    try:
        mongo_client = AsyncIOMotorClient(Config.MONGODB_URI)
        await mongo_client.admin.command('ping')
        
        db = mongo_client.sk4film
        files_col = db.files
        
        # Create indexes
        await files_col.create_index([("title", "text")])
        await files_col.create_index([("normalized_title", 1)])
        await files_col.create_index([("message_id", 1), ("channel_id", 1)], name="msg_ch_idx")
        await files_col.create_index([("indexed_at", -1)])
        
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
            
            # Schedule bot to start after flood wait
            asyncio.create_task(delayed_bot_start(e.value))
            
            # Mark bot as "starting" but not fully started
            bot_started = False
            return True  # Return success so web server can start
        
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
                # Schedule retry
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
    logger.info("🎬 SK4FiLM v8.0 - Modular Architecture")
    logger.info("="*60)
    
    # Initialize modules
    await init_modules()
    
    # Initialize MongoDB
    await init_mongodb()
    
    # Initialize bot
    bot_success = await init_bot()
    
    if not bot_success:
        logger.warning("⚠️ Bot initialization had issues, but web server will start")
    
    # Start web server
    config = HyperConfig()
    config.bind = [f"0.0.0.0:{Config.WEB_SERVER_PORT}"]
    config.loglevel = "warning"
    
    logger.info(f"🌐 Web server starting on port {Config.WEB_SERVER_PORT}...")
    await serve(app, config)

if __name__ == "__main__":
    asyncio.run(main())
