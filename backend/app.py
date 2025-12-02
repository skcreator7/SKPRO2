"""
app.py - Main SK4FiLM Bot with all features - Complete Version
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
import secrets
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from io import BytesIO
from functools import wraps
from collections import defaultdict

import aiohttp
import urllib.parse
from pyrogram import Client, filters, idle
from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton, Message, CallbackQuery
from pyrogram.errors import FloodWait, UserNotParticipant, ChannelPrivate, BadRequest
from quart import Quart, jsonify, request, Response
from hypercorn.asyncio import serve
from hypercorn.config import Config as HyperConfig
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, ValidationError
from bson import ObjectId
import redis.asyncio as redis

# Import our modules
try:
    from verification import VerificationSystem
    from premium import PremiumSystem, PremiumTier, PremiumPlan
    from poster_fetching import PosterFetcher, PosterSource
    from cache import CacheManager
except ImportError:
    # Create dummy classes if modules are not available
    class VerificationSystem:
        def __init__(self, config): 
            self.config = config
            self.logger = logging.getLogger(__name__)
        
        async def create_verification_link(self, user_id): 
            token = secrets.token_urlsafe(32)
            short_url = f"https://t.me/{self.config.BOT_USERNAME}?start=verify_{token}"
            return {'short_url': short_url, 'token': token}
        
        async def check_user_verified(self, user_id): 
            return False, 'not_verified'
        
        async def verify_user_token(self, token): 
            return False, 0, 'invalid'
        
        async def start_cleanup_task(self): 
            pass
        
        async def get_user_stats(self): 
            return {'pending_verifications': 0, 'verified_users': 0}
    
    class PremiumTier:
        FREE = 'free'
        BASIC = 'basic'
        PREMIUM = 'premium'
        ULTIMATE = 'ultimate'
        LIFETIME = 'lifetime'
    
    class PremiumSystem:
        def __init__(self, config): 
            self.config = config
            self.logger = logging.getLogger(__name__)
        
        async def is_premium_user(self, user_id): 
            return False
        
        async def get_subscription_details(self, user_id): 
            return {
                'is_active': False,
                'tier_name': 'Free',
                'expires_at': None,
                'days_remaining': 0,
                'features': ['Basic access'],
                'limits': {'daily_downloads': 5},
                'daily_downloads': 0,
                'total_downloads': 0
            }
        
        async def get_all_plans(self): 
            return [
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
        
        async def create_payment_request(self, user_id, tier): 
            upi_id = getattr(self.config, f'UPI_ID_{tier.upper()}', 'sk4filmbot@ybl')
            payment_id = secrets.token_hex(8)
            return {
                'payment_id': payment_id,
                'upi_id': upi_id,
                'amount': 199 if tier == 'premium' else 99,
                'tier_name': tier.capitalize() + ' Plan',
                'qr_code': None
            }
        
        async def can_user_download(self, user_id): 
            return True, 'ok', {}
        
        async def record_download(self, user_id): 
            pass
        
        async def activate_premium(self, admin_id, user_id, tier): 
            return {
                'user_id': user_id,
                'tier_name': tier.capitalize(),
                'expires_at': datetime.now() + timedelta(days=30),
                'duration_days': 30,
                'activated_by': admin_id
            }
        
        async def get_admin_stats(self): 
            return {'active_premium_users': 0, 'total_revenue': 0}
        
        async def broadcast_to_premium_users(self, message): 
            return {'user_count': 0, 'status': 'completed'}
        
        async def start_cleanup_task(self): 
            pass
    
    class PosterSource:
        LETTERBOXD = 'letterboxd'
        IMDB = 'imdb'
        JUSTWATCH = 'justwatch'
        IMPAWARDS = 'impawards'
        OMDB = 'omdb'
        TMDB = 'tmdb'
        CUSTOM = 'custom'
    
    class PosterFetcher:
        def __init__(self, config, cache_manager): 
            self.config = config
            self.cache_manager = cache_manager
            self.logger = logging.getLogger(__name__)
        
        async def fetch_poster(self, title): 
            return {
                'poster_url': f"{self.config.BACKEND_URL}/api/poster?title={urllib.parse.quote(title)}",
                'source': 'custom',
                'rating': '0.0',
                'year': '2024'
            }
        
        async def fetch_batch_posters(self, titles): 
            return {title: await self.fetch_poster(title) for title in titles}
    
    class CacheManager:
        def __init__(self, config): 
            self.config = config
            self.redis_enabled = False
            self.client = None
            self.logger = logging.getLogger(__name__)
        
        async def init_redis(self): 
            try:
                if self.config.REDIS_URL and self.config.REDIS_URL != "redis://localhost:6379":
                    self.client = redis.from_url(self.config.REDIS_URL, decode_responses=True)
                    await self.client.ping()
                    self.redis_enabled = True
                    self.logger.info("✅ Redis connected successfully")
                    return True
            except Exception as e:
                self.logger.warning(f"Redis connection failed: {e}")
            return False
        
        async def start_cleanup_task(self): 
            pass
        
        async def get(self, key): 
            if self.redis_enabled and self.client:
                try:
                    return await self.client.get(key)
                except Exception as e:
                    self.logger.error(f"Redis get error: {e}")
            return None
        
        async def set(self, key, value, expire_seconds=3600): 
            if self.redis_enabled and self.client:
                try:
                    await self.client.setex(key, expire_seconds, value)
                    return True
                except Exception as e:
                    self.logger.error(f"Redis set error: {e}")
            return False
        
        async def clear_pattern(self, pattern): 
            if self.redis_enabled and self.client:
                try:
                    keys = await self.client.keys(pattern)
                    if keys:
                        await self.client.delete(*keys)
                    return len(keys)
                except Exception as e:
                    self.logger.error(f"Redis clear pattern error: {e}")
            return 0
        
        async def clear_all(self): 
            if self.redis_enabled and self.client:
                try:
                    await self.client.flushdb()
                    return True
                except Exception as e:
                    self.logger.error(f"Redis flush error: {e}")
            return False
        
        async def get_stats_summary(self): 
            return {
                'redis_enabled': self.redis_enabled,
                'keys_count': 0
            }

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

# Pydantic Models for Validation
class SearchQuery(BaseModel):
    query: str
    page: int = 1
    limit: int = 12
    user_id: Optional[int] = None

class PremiumActivation(BaseModel):
    user_id: int
    tier: str
    duration_days: Optional[int] = 30

class BroadcastMessage(BaseModel):
    message: str
    user_type: str = 'premium'  # 'all', 'premium', 'free'

# Custom Exceptions
class BotError(Exception):
    """Base exception for bot errors"""
    pass

class SearchError(BotError):
    """Search-related errors"""
    pass

class DatabaseError(BotError):
    """Database-related errors"""
    pass

class RateLimitError(BotError):
    """Rate limit exceeded"""
    pass

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

# Task Manager
class TaskManager:
    def __init__(self):
        self.tasks = {}
        self.running = False
    
    async def start_task(self, name, coro_func, interval=3600):
        """Start a background task"""
        if name in self.tasks:
            self.tasks[name].cancel()
        
        async def task_wrapper():
            while self.running:
                try:
                    await coro_func()
                    await asyncio.sleep(interval)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Task {name} error: {e}")
                    await asyncio.sleep(60)  # Wait before retry
        
        task = asyncio.create_task(task_wrapper(), name=f"bg_{name}")
        self.tasks[name] = task
        logger.info(f"✅ Started background task: {name}")
    
    async def stop_all(self):
        """Stop all background tasks"""
        self.running = False
        for name, task in self.tasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self.tasks.clear()

# Decorator for error handling
def handle_errors(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except FloodWait as e:
            logger.warning(f"Flood wait in {func.__name__}: {e.value}s")
            raise
        except ValidationError as e:
            logger.error(f"Validation error in {func.__name__}: {e}")
            raise BotError(f"Invalid input: {e}")
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            raise BotError(f"{func.__name__} failed: {str(e)}")
    return wrapper

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

# Global instances
bot = None
user_client = None
db_manager = None

# Module instances
verification_system = None
premium_system = None
poster_fetcher = None
cache_manager = None
task_manager = None

# Rate limiters
api_rate_limiter = RateLimiter(max_requests=100, window_seconds=300)
search_rate_limiter = RateLimiter(max_requests=30, window_seconds=60)

# Status flags
bot_started = False
user_session_ready = False

# Channel configuration
CHANNEL_CONFIG = {
    -1001891090100: {'name': 'SK4FiLM Main', 'type': 'text', 'search_priority': 1},
    -1002024811395: {'name': 'SK4FiLM Updates', 'type': 'text', 'search_priority': 2},
    -1001768249569: {'name': 'SK4FiLM Files', 'type': 'file', 'search_priority': 0}
}

# Enhanced search statistics
search_stats = {
    'redis_hits': 0,
    'redis_misses': 0,
    'multi_channel_searches': 0,
    'total_searches': 0
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

# Main Application Class
class SK4FiLMBot:
    def __init__(self, config):
        self.config = config
        self.bot = None
        self.user_client = None
        self.db_manager = None
        self.task_manager = TaskManager()
        
        # Initialize systems
        self.cache_manager = None
        self.verification_system = None
        self.premium_system = None
        self.poster_fetcher = None
        
        # Status flags
        self.bot_started = False
        self.user_session_ready = False
    
    async def initialize(self):
        """Initialize all components"""
        logger.info("🚀 Initializing SK4FiLM Bot v8.0...")
        
        # Validate configuration
        self.config.validate()
        
        # Initialize database
        self.db_manager = DatabaseManager(self.config.MONGODB_URI)
        if not await self.db_manager.connect():
            raise DatabaseError("Failed to connect to database")
        
        # Initialize cache
        self.cache_manager = CacheManager(self.config)
        await self.cache_manager.init_redis()
        
        # Initialize other systems
        self.verification_system = VerificationSystem(self.config)
        self.premium_system = PremiumSystem(self.config)
        self.poster_fetcher = PosterFetcher(self.config, self.cache_manager)
        
        # Initialize Telegram clients
        await self.initialize_telegram()
        
        # Start background tasks
        await self.start_background_tasks()
        
        logger.info("✅ SK4FiLM Bot initialized successfully")
        
        # Update global references
        global bot, user_client, db_manager, verification_system, premium_system
        global poster_fetcher, cache_manager, task_manager, bot_started, user_session_ready
        
        bot = self.bot
        user_client = self.user_client
        db_manager = self.db_manager
        verification_system = self.verification_system
        premium_system = self.premium_system
        poster_fetcher = self.poster_fetcher
        cache_manager = self.cache_manager
        task_manager = self.task_manager
        bot_started = self.bot_started
        user_session_ready = self.user_session_ready
    
    async def initialize_telegram(self):
        """Initialize Telegram clients"""
        try:
            # Initialize bot
            self.bot = Client(
                "bot",
                api_id=self.config.API_ID,
                api_hash=self.config.API_HASH,
                bot_token=self.config.BOT_TOKEN,
                workers=20
            )
            
            # Initialize user client if session string is provided
            if self.config.USER_SESSION_STRING:
                self.user_client = Client(
                    "user",
                    api_id=self.config.API_ID,
                    api_hash=self.config.API_HASH,
                    session_string=self.config.USER_SESSION_STRING
                )
                await self.user_client.start()
                self.user_session_ready = True
                logger.info("✅ User session started successfully")
            
            # Start bot
            await self.bot.start()
            self.bot_started = True
            logger.info("✅ Bot started successfully")
            
        except Exception as e:
            logger.error(f"❌ Telegram initialization failed: {e}")
            raise
    
    async def start_background_tasks(self):
        """Start all background tasks"""
        self.task_manager.running = True
        
        # Start cleanup tasks
        if self.verification_system:
            await self.task_manager.start_task(
                'verification_cleanup',
                self.verification_system.start_cleanup_task,
                interval=3600
            )
        
        if self.premium_system:
            await self.task_manager.start_task(
                'premium_cleanup',
                self.premium_system.start_cleanup_task,
                interval=7200
            )
        
        if self.cache_manager:
            await self.task_manager.start_task(
                'cache_cleanup',
                self.cache_manager.start_cleanup_task,
                interval=1800
            )
        
        # Start file indexing task
        await self.task_manager.start_task(
            'file_indexing',
            self.index_files_background,
            interval=86400  # Run once per day
        )
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("🛑 Shutting down SK4FiLM Bot...")
        
        # Stop all background tasks
        await self.task_manager.stop_all()
        
        # Close database connections
        if self.db_manager:
            await self.db_manager.close()
        
        # Stop Telegram clients
        if self.bot and self.bot_started:
            await self.bot.stop()
        if self.user_client and self.user_session_ready:
            await self.user_client.stop()
        
        logger.info("✅ Shutdown complete")
    
    async def index_files_background(self):
        """Background file indexing task"""
        if not self.user_session_ready:
            return
        
        try:
            logger.info("🔄 Starting background file indexing...")
            
            total_indexed = 0
            async for message in safe_telegram_generator(
                self.user_client.get_chat_history,
                self.config.FILE_CHANNEL_ID,
                limit=500
            ):
                if message and (message.document or message.video):
                    await self.index_single_file(message)
                    total_indexed += 1
            
            logger.info(f"✅ Background indexing completed: {total_indexed} files indexed")
            
        except Exception as e:
            logger.error(f"Background indexing error: {e}")
    
    async def index_single_file(self, message):
        """Index a single file message"""
        title = extract_title_from_file(message)
        if not title:
            return
        
        file_id = message.document.file_id if message.document else message.video.file_id
        file_size = message.document.file_size if message.document else (message.video.file_size if message.video else 0)
        file_name = message.document.file_name if message.document else (message.video.file_name if message.video else 'video.mp4')
        quality = detect_quality(file_name)
        
        file_data = {
            'channel_id': self.config.FILE_CHANNEL_ID,
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
        
        # Try to extract thumbnail for video files
        if is_video_file(file_name) and hasattr(message, 'video') and message.video:
            try:
                # In a real implementation, you might extract thumbnail here
                # For now, we'll use a placeholder
                file_data['thumbnail'] = f"{self.config.BACKEND_URL}/api/poster?title={urllib.parse.quote(title)}"
                file_data['thumbnail_source'] = 'auto_generated'
            except Exception as e:
                logger.error(f"Thumbnail extraction error: {e}")
        
        if self.db_manager and self.db_manager.files_col:
            await self.db_manager.files_col.update_one(
                {'channel_id': self.config.FILE_CHANNEL_ID, 'message_id': message.id},
                {'$set': file_data},
                upsert=True
            )

# ==============================
# ENHANCED SEARCH FUNCTIONS
# ==============================

async def get_live_posts_multi_channel(limit_per_channel=10):
    """Get posts from multiple channels concurrently"""
    if not user_client or not user_session_ready:
        return []
    
    all_posts = []
    
    async def fetch_channel_posts(channel_id):
        posts = []
        try:
            async for msg in safe_telegram_generator(user_client.get_chat_history, channel_id, limit=limit_per_channel):
                if msg and msg.text and len(msg.text) > 15:
                    title = extract_title_smart(msg.text)
                    if title:
                        posts.append({
                            'title': title,
                            'normalized_title': normalize_title(title),
                            'content': msg.text,
                            'channel_name': channel_name(channel_id),
                            'channel_id': channel_id,
                            'message_id': msg.id,
                            'date': msg.date,
                            'is_new': is_new(msg.date) if msg.date else False
                        })
        except Exception as e:
            logger.error(f"Error getting posts from channel {channel_id}: {e}")
        return posts
    
    # Fetch from all text channels concurrently
    tasks = [fetch_channel_posts(channel_id) for channel_id in Config.TEXT_CHANNEL_IDS]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for result in results:
        if isinstance(result, list):
            all_posts.extend(result)
    
    # Sort by date (newest first) and remove duplicates
    seen_titles = set()
    unique_posts = []
    
    for post in sorted(all_posts, key=lambda x: x.get('date', datetime.min), reverse=True):
        if post['normalized_title'] not in seen_titles:
            seen_titles.add(post['normalized_title'])
            unique_posts.append(post)
    
    return unique_posts[:20]  # Return top 20 unique posts

@handle_errors
async def search_movies_multi_channel(query, limit=12, page=1):
    """Search across multiple channels with enhanced results"""
    offset = (page - 1) * limit
    search_stats['total_searches'] += 1
    search_stats['multi_channel_searches'] += 1
    
    # Try Redis cache first
    cache_key = f"search:enhanced:{query}:{page}:{limit}"
    if cache_manager is not None and cache_manager.redis_enabled:
        cached_data = await cache_manager.get(cache_key)
        if cached_data:
            try:
                search_stats['redis_hits'] += 1
                logger.info(f"✅ Redis cache HIT for: {query}")
                return json.loads(cached_data)
            except Exception as e:
                logger.warning(f"Redis cache parse error: {e}")
    
    search_stats['redis_misses'] += 1
    logger.info(f"🔍 Multi-channel search for: {query}")
    
    query_lower = query.lower()
    posts_dict = {}
    files_dict = {}
    
    # Use MongoDB search primarily to avoid Telegram API calls
    try:
        if db_manager is not None and db_manager.files_col is not None:
            # Try text search first
            try:
                cursor = db_manager.files_col.find({'$text': {'$search': query}})
                async for doc in cursor:
                    try:
                        norm_title = doc.get('normalized_title', normalize_title(doc['title']))
                        quality = doc['quality']
                        
                        if norm_title not in files_dict:
                            file_name = doc.get('file_name', '').lower()
                            file_is_video = is_video_file(file_name)
                            
                            thumbnail_url = None
                            if file_is_video:
                                thumbnail_url = doc.get('thumbnail')
                            
                            files_dict[norm_title] = {
                                'title': doc['title'], 
                                'quality_options': {}, 
                                'date': doc['date'].isoformat() if isinstance(doc['date'], datetime) else doc['date'],
                                'thumbnail': thumbnail_url,
                                'is_video_file': file_is_video,
                                'thumbnail_source': doc.get('thumbnail_source', 'unknown'),
                                'channel_id': doc.get('channel_id'),
                                'channel_name': channel_name(doc.get('channel_id'))
                            }
                        
                        if quality not in files_dict[norm_title]['quality_options']:
                            files_dict[norm_title]['quality_options'][quality] = {
                                'file_id': f"{doc.get('channel_id', Config.FILE_CHANNEL_ID)}_{doc.get('message_id')}_{quality}",
                                'file_size': doc['file_size'],
                                'file_name': doc['file_name'],
                                'is_video': file_is_video,
                                'channel_id': doc.get('channel_id'),
                                'message_id': doc.get('message_id')
                            }
                    except Exception as e:
                        logger.error(f"Error processing search result: {e}")
                        continue
            except Exception as e:
                logger.error(f"Text search error: {e}")
            
            # If no results from text search, try regex search
            if not files_dict:
                regex_cursor = db_manager.files_col.find({
                    '$or': [
                        {'title': {'$regex': query, '$options': 'i'}},
                        {'normalized_title': {'$regex': query, '$options': 'i'}}
                    ]
                })
                
                async for doc in regex_cursor:
                    try:
                        norm_title = doc.get('normalized_title', normalize_title(doc['title']))
                        quality = doc['quality']
                        
                        if norm_title not in files_dict:
                            file_name = doc.get('file_name', '').lower()
                            file_is_video = is_video_file(file_name)
                            
                            thumbnail_url = None
                            if file_is_video:
                                thumbnail_url = doc.get('thumbnail')
                            
                            files_dict[norm_title] = {
                                'title': doc['title'], 
                                'quality_options': {}, 
                                'date': doc['date'].isoformat() if isinstance(doc['date'], datetime) else doc['date'],
                                'thumbnail': thumbnail_url,
                                'is_video_file': file_is_video,
                                'thumbnail_source': doc.get('thumbnail_source', 'unknown'),
                                'channel_id': doc.get('channel_id'),
                                'channel_name': channel_name(doc.get('channel_id'))
                            }
                        
                        if quality not in files_dict[norm_title]['quality_options']:
                            files_dict[norm_title]['quality_options'][quality] = {
                                'file_id': f"{doc.get('channel_id', Config.FILE_CHANNEL_ID)}_{doc.get('message_id')}_{quality}",
                                'file_size': doc['file_size'],
                                'file_name': doc['file_name'],
                                'is_video': file_is_video,
                                'channel_id': doc.get('channel_id'),
                                'message_id': doc.get('message_id')
                            }
                    except Exception as e:
                        logger.error(f"Error processing regex search result: {e}")
                        continue
    except Exception as e:
        logger.error(f"File search error: {e}")
    
    # Search Telegram channels if user session is ready
    if user_session_ready and user_client:
        channel_results = {}
        
        async def search_single_channel(channel_id):
            channel_posts = {}
            try:
                cname = channel_name(channel_id)
                async for msg in safe_telegram_generator(user_client.search_messages, channel_id, query=query, limit=20):
                    if msg and msg.text and len(msg.text) > 15:
                        title = extract_title_smart(msg.text)
                        if title and query_lower in title.lower():
                            norm_title = normalize_title(title)
                            if norm_title not in channel_posts:
                                channel_posts[norm_title] = {
                                    'title': title,
                                    'content': format_post(msg.text),
                                    'channel': cname,
                                    'channel_id': channel_id,
                                    'message_id': msg.id,
                                    'date': msg.date.isoformat() if isinstance(msg.date, datetime) else msg.date,
                                    'is_new': is_new(msg.date) if msg.date else False,
                                    'has_file': False,
                                    'has_post': True,
                                    'quality_options': {},
                                    'thumbnail': None
                                }
            except Exception as e:
                logger.error(f"Telegram search error in {channel_id}: {e}")
            return channel_posts
        
        # Search all text channels concurrently
        tasks = [search_single_channel(channel_id) for channel_id in Config.TEXT_CHANNEL_IDS]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Merge results from all channels
        for result in results:
            if isinstance(result, dict):
                posts_dict.update(result)
    
    # Merge posts and files data
    merged = {}
    
    # First add all posts
    for norm_title, post_data in posts_dict.items():
        merged[norm_title] = post_data
    
    # Then add/update with file information
    for norm_title, file_data in files_dict.items():
        if norm_title in merged:
            # Update existing post with file info
            merged[norm_title]['has_file'] = True
            merged[norm_title]['quality_options'] = file_data['quality_options']
            if file_data.get('is_video_file') and file_data.get('thumbnail'):
                merged[norm_title]['thumbnail'] = file_data['thumbnail']
                merged[norm_title]['thumbnail_source'] = file_data.get('thumbnail_source', 'unknown')
        else:
            # Create new entry for files without posts
            merged[norm_title] = {
                'title': file_data['title'],
                'content': f"<p>{file_data['title']}</p>",
                'channel': file_data.get('channel_name', 'SK4FiLM Files'),
                'date': file_data['date'],
                'is_new': False,
                'has_file': True,
                'has_post': False,
                'quality_options': file_data['quality_options'],
                'thumbnail': file_data.get('thumbnail') if file_data.get('is_video_file') else None,
                'thumbnail_source': file_data.get('thumbnail_source', 'unknown')
            }
    
    # Sort results: new posts first, then files, then by date
    results_list = list(merged.values())
    results_list.sort(key=lambda x: (
        not x.get('is_new', False),  # New posts first
        not x['has_file'],           # Files before posts only
        x['date']                    # Recent first
    ), reverse=True)
    
    total = len(results_list)
    paginated = results_list[offset:offset + limit]
    
    # Get posters for results
    if poster_fetcher is not None and paginated:
        titles = [result['title'] for result in paginated]
        posters = await poster_fetcher.fetch_batch_posters(titles)
        
        for result in paginated:
            if result['title'] in posters:
                result['poster'] = posters[result['title']]
            else:
                result['poster'] = {
                    'poster_url': f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(result['title'])}",
                    'source': 'custom',
                    'rating': '0.0'
                }
    
    result_data = {
        'results': paginated,
        'pagination': {
            'current_page': page,
            'total_pages': math.ceil(total / limit) if total > 0 else 1,
            'total_results': total,
            'per_page': limit,
            'has_next': page < math.ceil(total / limit) if total > 0 else False,
            'has_previous': page > 1
        },
        'search_metadata': {
            'channels_searched': len(Config.TEXT_CHANNEL_IDS),
            'channels_found': len(set(r.get('channel_id') for r in paginated if r.get('channel_id'))),
            'query': query,
            'search_mode': 'multi_channel_enhanced',
            'cache_status': 'miss'
        }
    }
    
    # Cache in Redis with 1 hour expiration
    if cache_manager is not None and cache_manager.redis_enabled:
        await cache_manager.set(cache_key, json.dumps(result_data), expire_seconds=3600)
    
    logger.info(f"✅ Multi-channel search completed: {len(paginated)} results from {len(set(r.get('channel_id') for r in paginated if r.get('channel_id')))} channels")
    
    return result_data

async def get_home_movies_live():
    """Get latest movies from multiple channels"""
    posts = await get_live_posts_multi_channel(limit_per_channel=15)
    
    movies = []
    seen = set()
    
    for post in posts:
        tk = post['title'].lower().strip()
        if tk not in seen:
            seen.add(tk)
            movies.append({
                'title': post['title'],
                'date': post['date'].isoformat() if isinstance(post['date'], datetime) else post['date'],
                'is_new': post.get('is_new', False),
                'channel': post.get('channel_name', 'SK4FiLM'),
                'channel_id': post.get('channel_id')
            })
            if len(movies) >= 20:
                break
    
    # Get posters
    if movies and poster_fetcher is not None:
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
    
    return movies

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
    if db_manager is not None and db_manager.files_col is not None:
        try:
            mongodb_count = await db_manager.files_col.count_documents({})
        except:
            pass
    
    return jsonify({
        'status': 'healthy',
        'service': 'SK4FiLM v8.0 - Complete System with Enhanced Search',
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
            'mongodb_connected': db_manager is not None,
            'total_files': mongodb_count
        },
        'channels': {
            'text_channels': len(Config.TEXT_CHANNEL_IDS),
            'file_channel': Config.FILE_CHANNEL_ID
        },
        'enhanced_search': {
            'enabled': True,
            'multi_channel': True,
            'redis_caching': cache_manager.redis_enabled if cache_manager is not None else False,
            'search_stats': search_stats
        },
        'endpoints': {
            'api': '/api/*',
            'health': '/health',
            'verify': '/api/verify/{user_id}',
            'premium': '/api/premium/*',
            'poster': '/api/poster/{title}',
            'cache': '/api/cache/*',
            'search': '/api/search',
            'enhanced_search': '/api/search/enhanced',
            'search_stats': '/api/search/stats'
        }
    })

@app.route('/health')
async def health():
    """Health check endpoint"""
    mongodb_status = False
    if db_manager is not None and db_manager.client is not None:
        try:
            await db_manager.client.admin.command('ping')
            mongodb_status = True
        except:
            pass
    
    redis_status = False
    if cache_manager is not None and cache_manager.redis_enabled and cache_manager.client:
        try:
            await cache_manager.client.ping()
            redis_status = True
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
            'redis': redis_status,
            'web_server': True
        },
        'modules': {
            'verification': verification_system is not None,
            'premium': premium_system is not None,
            'poster_fetcher': poster_fetcher is not None,
            'cache': cache_manager is not None
        },
        'enhanced_search': {
            'enabled': True,
            'channels_ready': len(Config.TEXT_CHANNEL_IDS),
            'redis_ready': cache_manager.redis_enabled if cache_manager is not None else False
        }
    })

@app.route('/metrics')
async def metrics():
    """Prometheus metrics endpoint"""
    metrics_data = [
        '# HELP sk4film_search_total Total search requests',
        '# TYPE sk4film_search_total counter',
        f'sk4film_search_total {search_stats["total_searches"]}',
        
        '# HELP sk4film_redis_hits_total Total Redis cache hits',
        '# TYPE sk4film_redis_hits_total counter',
        f'sk4film_redis_hits_total {search_stats["redis_hits"]}',
        
        '# HELP sk4film_redis_misses_total Total Redis cache misses',
        '# TYPE sk4film_redis_misses_total counter',
        f'sk4film_redis_misses_total {search_stats["redis_misses"]}',
        
        '# HELP sk4film_bot_online Bot online status',
        '# TYPE sk4film_bot_online gauge',
        f'sk4film_bot_online {1 if bot_started else 0}',
        
        '# HELP sk4film_user_session_online User session online status',
        '# TYPE sk4film_user_session_online gauge',
        f'sk4film_user_session_online {1 if user_session_ready else 0}',
    ]
    
    return Response('\n'.join(metrics_data), mimetype='text/plain')

# Update the existing /api/search route to use enhanced search
@app.route('/api/search')
async def api_search():
    """Enhanced search for movies with multi-channel support"""
    try:
        # Rate limiting
        ip = request.remote_addr
        if not await api_rate_limiter.is_allowed(ip):
            return jsonify({
                'status': 'error',
                'message': 'Rate limit exceeded. Please try again later.'
            }), 429
        
        # Validate input
        search_data = SearchQuery(
            query=request.args.get('query', '').strip(),
            page=int(request.args.get('page', 1)),
            limit=int(request.args.get('limit', 12)),
            user_id=request.args.get('user_id', type=int)
        )
        
        # Validate query length
        if len(search_data.query) < 2:
            return jsonify({'status': 'error', 'message': 'Query too short'}), 400
        
        if len(search_data.query) > 100:
            return jsonify({'status': 'error', 'message': 'Query too long'}), 400
        
        # Check user access
        if search_data.user_id:
            # Check if premium user (bypasses verification)
            if premium_system is not None:
                is_premium = await premium_system.is_premium_user(search_data.user_id)
                if is_premium:
                    # Premium user, allow access
                    pass
                elif Config.VERIFICATION_REQUIRED and verification_system is not None:
                    # Check verification for non-premium users
                    is_verified, message = await verification_system.check_user_verified(search_data.user_id)
                    if not is_verified:
                        return jsonify({
                            'status': 'verification_required',
                            'message': 'User verification required',
                            'verification_url': f'/api/verify/{search_data.user_id}'
                        }), 403
        
        # Use enhanced search if available and requested
        search_mode = request.args.get('mode', 'enhanced')
        if search_mode == 'enhanced' and user_session_ready:
            try:
                result = await search_movies_multi_channel(search_data.query, search_data.limit, search_data.page)
                return jsonify({
                    'status': 'success',
                    'query': search_data.query,
                    'results': result['results'],
                    'pagination': result['pagination'],
                    'search_metadata': result.get('search_metadata', {}),
                    'bot_username': Config.BOT_USERNAME,
                    'search_mode': 'enhanced_multi_channel'
                })
            except Exception as e:
                logger.error(f"Enhanced search failed: {e}, falling back to basic search")
                # Fall back to basic search
        
        # Basic search (existing functionality)
        cache_key = f"search:basic:{search_data.query}:{search_data.page}:{search_data.limit}"
        if cache_manager is not None:
            cached = await cache_manager.get(cache_key)
            if cached:
                return jsonify(json.loads(cached))
        
        results = []
        if db_manager is not None and db_manager.files_col is not None:
            # Search in MongoDB
            cursor = db_manager.files_col.find({
                '$or': [
                    {'title': {'$regex': search_data.query, '$options': 'i'}},
                    {'normalized_title': {'$regex': search_data.query, '$options': 'i'}}
                ]
            }).limit(search_data.limit).skip((search_data.page - 1) * search_data.limit)
            
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
            'query': search_data.query,
            'results': results,
            'pagination': {
                'current_page': search_data.page,
                'total_pages': math.ceil(total / search_data.limit) if total > 0 else 1,
                'total_results': total,
                'per_page': search_data.limit,
                'has_next': search_data.page < math.ceil(total / search_data.limit) if total > 0 else False,
                'has_previous': search_data.page > 1
            },
            'search_mode': 'basic'
        }
        
        # Cache the results
        if cache_manager is not None:
            await cache_manager.set(cache_key, json.dumps(response_data), expire_seconds=1800)
        
        return jsonify(response_data)
        
    except ValidationError as e:
        return jsonify({'status': 'error', 'message': 'Invalid parameters'}), 400
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/search/enhanced')
async def api_enhanced_search():
    """Force enhanced multi-channel search"""
    try:
        # Rate limiting
        ip = request.remote_addr
        if not await search_rate_limiter.is_allowed(ip):
            return jsonify({
                'status': 'error',
                'message': 'Search rate limit exceeded. Please try again later.'
            }), 429
        
        q = request.args.get('query', '').strip()
        p = int(request.args.get('page', 1))
        l = int(request.args.get('limit', 12))
        
        if not q:
            return jsonify({'status': 'error', 'message': 'Query required'}), 400
        
        if not user_session_ready:
            return jsonify({
                'status': 'error',
                'message': 'Enhanced search requires user session. Try basic search.',
                'basic_search_url': f'/api/search?query={urllib.parse.quote(q)}&page={p}&limit={l}'
            }), 503
        
        result = await search_movies_multi_channel(q, l, p)
        return jsonify({
            'status': 'success',
            'query': q,
            'results': result['results'],
            'pagination': result['pagination'],
            'search_metadata': result.get('search_metadata', {}),
            'bot_username': Config.BOT_USERNAME,
            'search_mode': 'enhanced_multi_channel_forced'
        })
        
    except Exception as e:
        logger.error(f"Enhanced search error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/search/stats')
async def api_search_stats():
    """Get search statistics"""
    try:
        hit_rate = 0
        if search_stats['redis_hits'] + search_stats['redis_misses'] > 0:
            hit_rate = (search_stats['redis_hits'] / (search_stats['redis_hits'] + search_stats['redis_misses'])) * 100
        
        return jsonify({
            'status': 'success',
            'stats': {
                'total_searches': search_stats['total_searches'],
                'multi_channel_searches': search_stats['multi_channel_searches'],
                'redis_hits': search_stats['redis_hits'],
                'redis_misses': search_stats['redis_misses'],
                'redis_hit_rate': f"{hit_rate:.1f}%",
                'enhanced_search_available': user_session_ready,
                'channels_active': len(Config.TEXT_CHANNEL_IDS),
                'redis_enabled': cache_manager.redis_enabled if cache_manager is not None else False
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/search/clear_cache', methods=['POST'])
async def api_clear_search_cache():
    """Clear search cache"""
    try:
        # Check admin key
        data = await request.get_json() or {}
        admin_key = data.get('admin_key') or request.args.get('admin_key')
        
        if admin_key != Config.ADMIN_API_KEY:
            return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401
        
        cleared = 0
        if cache_manager is not None:
            cleared = await cache_manager.clear_pattern("search:*")
            
            # Reset search stats
            global search_stats
            search_stats = {
                'redis_hits': 0,
                'redis_misses': 0,
                'multi_channel_searches': 0,
                'total_searches': 0
            }
        
        return jsonify({
            'status': 'success',
            'message': f'Search cache cleared ({cleared} keys)',
            'cleared_keys': cleared,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Update the /api/movies route to use enhanced function
@app.route('/api/movies')
async def api_movies():
    """Get latest movies with enhanced multi-channel support"""
    try:
        if not bot_started:
            return jsonify({'status': 'error', 'message': 'Starting...'}), 503
        
        # Use enhanced function to get movies
        movies = await get_home_movies_live()
        
        if not movies and db_manager is not None and db_manager.files_col is not None:
            # Fallback to database if no live posts
            return await api_movies_basic()
        
        return jsonify({
            'status': 'success', 
            'movies': movies, 
            'total': len(movies), 
            'bot_username': Config.BOT_USERNAME,
            'mode': 'multi_channel_enhanced',
            'channels_searched': len(Config.TEXT_CHANNEL_IDS),
            'live_posts': True
        })
    except Exception as e:
        logger.error(f"Movies error: {e}")
        # Fall back to basic approach
        return await api_movies_basic()

async def api_movies_basic():
    """Basic movies API fallback"""
    try:
        if db_manager is None or db_manager.files_col is None:
            return jsonify({'status': 'error', 'message': 'Database not ready'}), 503
        
        cursor = db_manager.files_col.find({}).sort('date', -1).limit(20)
        movies = []
        
        async for doc in cursor:
            movies.append({
                'title': doc.get('title'),
                'date': doc.get('date'),
                'quality': doc.get('quality', '480p'),
                'file_size': doc.get('file_size'),
                'is_new': is_new(doc.get('date')) if doc.get('date') else False,
                'is_video_file': doc.get('is_video_file', False),
                'thumbnail': doc.get('thumbnail'),
                'thumbnail_source': doc.get('thumbnail_source', 'unknown')
            })
        
        return jsonify({
            'status': 'success',
            'movies': movies,
            'total': len(movies),
            'mode': 'basic'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/background/index', methods=['POST'])
async def api_background_index():
    """Start background indexing (admin only)"""
    try:
        # Check admin key
        data = await request.get_json() or {}
        admin_key = data.get('admin_key') or request.args.get('admin_key')
        
        if admin_key != Config.ADMIN_API_KEY:
            return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401
        
        # Start indexing in background
        asyncio.create_task(index_files_background())
        
        return jsonify({
            'status': 'success',
            'message': 'Background indexing started',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

async def index_files_background():
    """Background file indexing"""
    if not user_session_ready or not user_client:
        return
    
    try:
        logger.info("🔄 Starting manual background indexing...")
        
        total_indexed = 0
        async for message in safe_telegram_generator(
            user_client.get_chat_history,
            Config.FILE_CHANNEL_ID,
            limit=200
        ):
            if message and (message.document or message.video):
                await index_single_file(message)
                total_indexed += 1
        
        logger.info(f"✅ Manual indexing completed: {total_indexed} files indexed")
        
    except Exception as e:
        logger.error(f"Manual indexing error: {e}")

async def index_single_file(message):
    """Index a single file message"""
    title = extract_title_from_file(message)
    if not title:
        return
    
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
    
    # Try to extract thumbnail for video files
    if is_video_file(file_name) and hasattr(message, 'video') and message.video:
        try:
            # In a real implementation, you might extract thumbnail here
            # For now, we'll use a placeholder
            file_data['thumbnail'] = f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(title)}"
            file_data['thumbnail_source'] = 'auto_generated'
        except Exception as e:
            logger.error(f"Thumbnail extraction error: {e}")
    
    if db_manager is not None and db_manager.files_col is not None:
        await db_manager.files_col.update_one(
            {'channel_id': Config.FILE_CHANNEL_ID, 'message_id': message.id},
            {'$set': file_data},
            upsert=True
        )

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
        except (AttributeError, ValueError):
            await callback_query.answer("Invalid plan", show_alert=True)
            return
        
        # Create payment request
        payment_data = await premium_system.create_payment_request(user_id, tier)
        
        text = f"💰 **Payment for {payment_data['tier_name']}**\n\n"
        text += f"**Amount:** ₹{payment_data['amount']}\n"
        text += f"**UPI ID:** `{payment_data['upi_id']}`\n\n"
        text += "**Payment Instructions:**\n"
        text += "1. Scan the QR code below OR\n"
        text += f"2. Send ₹{payment_data['amount']} to UPI ID: `{payment_data['upi_id']}`\n"
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
            except (AttributeError, ValueError):
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
            total_files = await db_manager.files_col.count_documents({}) if db_manager and db_manager.files_col else 0
            video_files = await db_manager.files_col.count_documents({'is_video_file': True}) if db_manager and db_manager.files_col else 0
            
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
            async for tg_message in safe_telegram_generator(
                user_client.get_chat_history,
                Config.FILE_CHANNEL_ID,
                limit=100
            ):
                if tg_message and (tg_message.document or tg_message.video):
                    await index_single_file(tg_message)
                    total += 1
            
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
                    
                    # Auto-delete file after specified time
                    if Config.AUTO_DELETE_TIME > 0:
                        asyncio.create_task(auto_delete_file(sent, Config.AUTO_DELETE_TIME))
                    
                    logger.info(f"File sent to user {user_id}: {quality} quality")
                    
                    # Send success message
                    await message.reply_text(
                        f"✅ **File sent successfully!**\n\n"
                        f"📦 **Quality:** {quality}\n"
                        f"⏰ **Auto-delete in:** {Config.AUTO_DELETE_TIME//60} minutes\n\n"
                        f"♻ **Please forward to saved messages**\n"
                        f"⭐ **Consider upgrading to Premium for better features!**",
                        reply_markup=InlineKeyboardMarkup([
                            [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")]
                        ])
                    )
                    
                    return
        
        # If not a file link, show help
        await message.reply_text(
            "🎬 **SK4FiLM File Download**\n\n"
            "To download a file, use the website:\n"
            f"🌐 **Website:** {Config.WEBSITE_URL}\n\n"
            "Find your movie and click download to get the file link.\n"
            "Then paste the link here and I'll send you the file! 🍿"
        )
    
    @bot.on_callback_query(filters.regex(r"^back_to_start$"))
    async def back_to_start_callback(client, callback_query):
        """Go back to start"""
        user_id = callback_query.from_user.id
        user_name = callback_query.from_user.first_name or "User"
        
        welcome_text = (
            f"🎬 **Welcome back to SK4FiLM, {user_name}!**\n\n"
            "🌐 **Use our website to browse and download movies:**\n"
            f"{Config.WEBSITE_URL}\n\n"
        )
        
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)],
            [InlineKeyboardButton("⭐ BUY PREMIUM", callback_data="buy_premium")]
        ])
        
        await callback_query.message.edit_text(welcome_text, reply_markup=keyboard, disable_web_page_preview=True)
    
    @bot.on_callback_query(filters.regex(r"^send_screenshot_"))
    async def send_screenshot_callback(client, callback_query):
        """Handle screenshot sending"""
        payment_id = callback_query.data.split('_')[2]
        
        await callback_query.answer(
            "Now please send the payment screenshot to this chat.\n"
            "Make sure your payment details are visible in the screenshot.",
            show_alert=True
        )
        
        await callback_query.message.edit_text(
            "📸 **Send Payment Screenshot**\n\n"
            "Please send the payment screenshot to this chat.\n"
            "Make sure:\n"
            "1. Payment amount is visible\n"
            "2. UPI ID is visible\n"
            "3. Transaction ID is visible\n\n"
            "⚠️ **Send the screenshot now...**"
        )

async def auto_delete_file(message, delay_seconds):
    """Auto-delete file after specified delay"""
    await asyncio.sleep(delay_seconds)
    try:
        await message.delete()
        logger.info(f"Auto-deleted file for user {message.chat.id}")
    except Exception as e:
        logger.error(f"Failed to auto-delete file: {e}")

# Add more API routes
@app.route('/api/verify/<int:user_id>')
async def api_verify_user(user_id):
    """Get verification link for user"""
    try:
        if verification_system is None:
            return jsonify({'status': 'error', 'message': 'Verification system not available'}), 503
        
        verification_data = await verification_system.create_verification_link(user_id)
        
        return jsonify({
            'status': 'success',
            'user_id': user_id,
            'verification_url': verification_data['short_url'],
            'expires_in': '1 hour',
            'bot_username': Config.BOT_USERNAME
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/premium/plans')
async def api_premium_plans():
    """Get all premium plans"""
    try:
        if premium_system is None:
            return jsonify({'status': 'error', 'message': 'Premium system not available'}), 503
        
        plans = await premium_system.get_all_plans()
        
        return jsonify({
            'status': 'success',
            'plans': plans,
            'currency': 'INR'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/premium/status/<int:user_id>')
async def api_premium_status(user_id):
    """Get premium status for user"""
    try:
        if premium_system is None:
            return jsonify({'status': 'error', 'message': 'Premium system not available'}), 503
        
        details = await premium_system.get_subscription_details(user_id)
        
        return jsonify({
            'status': 'success',
            'user_id': user_id,
            'premium_details': details
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
        
        if poster_fetcher is None:
            # Return default poster
            return jsonify({
                'status': 'success',
                'poster_url': f"{Config.BACKEND_URL}/static/default_poster.jpg",
                'source': 'default',
                'rating': '0.0',
                'year': '2024'
            })
        
        poster_data = await poster_fetcher.fetch_poster(title)
        
        return jsonify({
            'status': 'success',
            'title': title,
            'poster_data': poster_data
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/system/stats')
async def api_system_stats():
    """Get system statistics (admin only)"""
    try:
        # Check admin key
        admin_key = request.args.get('admin_key')
        if admin_key != Config.ADMIN_API_KEY:
            return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401
        
        # Get MongoDB stats
        total_files = await db_manager.files_col.count_documents({}) if db_manager and db_manager.files_col else 0
        video_files = await db_manager.files_col.count_documents({'is_video_file': True}) if db_manager and db_manager.files_col else 0
        
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
        
        hit_rate = 0
        if search_stats['redis_hits'] + search_stats['redis_misses'] > 0:
            hit_rate = (search_stats['redis_hits'] / (search_stats['redis_hits'] + search_stats['redis_misses'])) * 100
        
        return jsonify({
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'database': {
                'total_files': total_files,
                'video_files': video_files,
                'non_video_files': total_files - video_files
            },
            'verification': verification_stats,
            'premium': premium_stats,
            'cache': cache_stats,
            'search': {
                **search_stats,
                'redis_hit_rate': f"{hit_rate:.1f}%"
            },
            'system': {
                'bot_online': bot_started,
                'user_session_online': user_session_ready,
                'channels_configured': len(CHANNEL_CONFIG),
                'text_channels': len(Config.TEXT_CHANNEL_IDS),
                'uptime': 'N/A'  # You can add uptime tracking
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Main function
async def main():
    """Main function to start the bot"""
    try:
        # Initialize configuration
        config = Config()
        
        # Create bot instance
        sk4film_bot = SK4FiLMBot(config)
        
        # Initialize bot
        await sk4film_bot.initialize()
        
        # Setup bot handlers
        await setup_bot()
        
        # Start web server
        hypercorn_config = HyperConfig()
        hypercorn_config.bind = [f"0.0.0.0:{config.WEB_SERVER_PORT}"]
        hypercorn_config.workers = 2
        
        logger.info(f"🚀 Starting web server on port {config.WEB_SERVER_PORT}")
        
        # Run both web server and bot
        await asyncio.gather(
            serve(app, hypercorn_config),
            idle()  # Keep bot running
        )
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        # Cleanup
        if 'sk4film_bot' in locals():
            await sk4film_bot.shutdown()
        logger.info("Bot stopped")

if __name__ == "__main__":
    asyncio.run(main())
