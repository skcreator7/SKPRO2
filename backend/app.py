"""
app.py - Complete SK4FiLM Web API System with Multi-Channel Search
FIXED: No circular imports - All modules integrated
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
import base64
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import defaultdict

import aiohttp
import urllib.parse
from quart import Quart, jsonify, request, Response
from hypercorn.asyncio import serve
from hypercorn.config import Config as HyperConfig
from motor.motor_asyncio import AsyncIOMotorClient
import redis.asyncio as redis

# ✅ ADD THESE PYROGRAM IMPORTS
from pyrogram import Client
from pyrogram.errors import FloodWait

# Import modular components
from cache import CacheManager
from verification import VerificationSystem
from premium import PremiumSystem, PremiumTier
from poster_fetching import PosterFetcher, PosterSource

# Import shared utilities
from utils import (
    normalize_title,
    extract_title_smart,
    extract_title_from_file,
    format_size,
    detect_quality,
    is_video_file,
    format_post,
    is_new
)

# Import bot_handlers AFTER all other imports
from bot_handlers import SK4FiLMBot, setup_bot_handlers

# FAST LOADING OPTIMIZATIONS
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger('asyncio').setLevel(logging.WARNING)
logging.getLogger('pyrogram').setLevel(logging.WARNING)
logging.getLogger('hypercorn').setLevel(logging.WARNING)

class Config:
    API_ID = int(os.environ.get("API_ID", "0"))
    API_HASH = os.environ.get("API_HASH", "")
    USER_SESSION_STRING = os.environ.get("USER_SESSION_STRING", "")
    BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
    MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
    REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
    
    # CHANNEL CONFIGURATION - MULTI-CHANNEL SUPPORT
    MAIN_CHANNEL_ID = -1001891090100
    TEXT_CHANNEL_IDS = [-1001891090100, -1002024811395]
    FILE_CHANNEL_ID = -1001768249569
    
    MAIN_CHANNEL_LINK = "https://t.me/sk4film"
    UPDATES_CHANNEL_LINK = "https://t.me/sk4film_Request"
    CHANNEL_USERNAME = "sk4film"
    
    # URL Shortener Configuration
    SHORTLINK_API = os.environ.get("SHORTLINK_API", "")
    CUTTLY_API = os.environ.get("CUTTLY_API", "")
    
    # UPI IDs for Premium
    UPI_ID_BASIC = os.environ.get("UPI_ID_BASIC", "sk4filmbot@ybl")
    UPI_ID_PREMIUM = os.environ.get("UPI_ID_PREMIUM", "sk4filmbot@ybl")
    UPI_ID_GOLD = os.environ.get("UPI_ID_GOLD", "sk4filmbot@ybl")
    UPI_ID_DIAMOND = os.environ.get("UPI_ID_DIAMOND", "sk4filmbot@ybl")
    
    VERIFICATION_REQUIRED = os.environ.get("VERIFICATION_REQUIRED", "true").lower() == "true"
    VERIFICATION_DURATION = 6 * 60 * 60
    
    WEBSITE_URL = os.environ.get("WEBSITE_URL", "https://sk4film.vercel.app")
    BOT_USERNAME = os.environ.get("BOT_USERNAME", "sk4filmbot")
    ADMIN_IDS = [int(x) for x in os.environ.get("ADMIN_IDS", "123456789").split(",")]
    AUTO_DELETE_TIME = int(os.environ.get("AUTO_DELETE_TIME", "300"))
    WEB_SERVER_PORT = int(os.environ.get("PORT", 8000))
    BACKEND_URL = os.environ.get("BACKEND_URL", "https://sk4film.koyeb.app")
    
    OMDB_KEYS = ["8265bd1c", "b9bd48a6", "3e7e1e9d"]
    TMDB_KEYS = ["e547e17d4e91f3e62a571655cd1ccaff", "8265bd1f"]

# FAST INITIALIZATION
app = Quart(__name__)

@app.after_request
async def add_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

# GLOBAL VARIABLES - FAST ACCESS
mongo_client = None
db = None
files_col = None
verification_col = None

# MODULAR COMPONENTS
cache_manager = None
verification_system = None
premium_system = None
poster_fetcher = None
sk4film_bot = None

# Telegram clients
User = None
bot = None
bot_started = False
user_session_ready = False

# CHANNEL CONFIGURATION
CHANNEL_CONFIG = {
    -1001891090100: {'name': 'SK4FiLM Main', 'type': 'text', 'search_priority': 1},
    -1002024811395: {'name': 'SK4FiLM Updates', 'type': 'text', 'search_priority': 2},
    -1001768249569: {'name': 'SK4FiLM Files', 'type': 'file', 'search_priority': 0}
}

# IMPROVED FLOOD WAIT PROTECTION
class FloodWaitProtection:
    def __init__(self):
        self.last_request_time = 0
        self.min_interval = 3
        self.request_count = 0
        self.reset_time = time.time()
        self.consecutive_waits = 0
        self.last_wait_time = 0
    
    async def wait_if_needed(self):
        current_time = time.time()
        
        # Reset counter every 2 minutes
        if current_time - self.reset_time > 120:
            self.request_count = 0
            self.reset_time = current_time
            self.consecutive_waits = 0
        
        # Limit to 20 requests per 2 minutes
        if self.request_count >= 20:
            wait_time = 120 - (current_time - self.reset_time)
            if wait_time > 0:
                self.consecutive_waits += 1
                extra_wait = self.consecutive_waits * 5
                total_wait = wait_time + extra_wait
                
                logger.warning(f"⚠️ Rate limit reached, waiting {total_wait:.1f}s (consecutive: {self.consecutive_waits})")
                await asyncio.sleep(total_wait)
                
                self.request_count = 0
                self.reset_time = time.time()
        
        # Ensure minimum interval between requests
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_interval:
            wait_time = self.min_interval - time_since_last
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()
        self.request_count += 1

flood_protection = FloodWaitProtection()

# SAFE TELEGRAM OPERATIONS
async def safe_telegram_operation(operation, *args, **kwargs):
    """Safely execute Telegram operations with enhanced flood wait protection"""
    max_retries = 2
    base_delay = 5
    
    for attempt in range(max_retries):
        try:
            await flood_protection.wait_if_needed()
            result = await operation(*args, **kwargs)
            
            if flood_protection.consecutive_waits > 0:
                flood_protection.consecutive_waits = 0
                
            return result
        except Exception as e:
            logger.error(f"Telegram operation failed: {e}")
            if attempt == max_retries - 1:
                raise e
            await asyncio.sleep(base_delay * (2 ** attempt))
    return None

# SAFE ASYNC ITERATOR FOR TELEGRAM GENERATORS
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
        except Exception as e:
            logger.error(f"Telegram generator failed: {e}")
            if attempt == max_retries - 1:
                raise e
            await asyncio.sleep(5 * (2 ** attempt))
    return

# CACHE CLEANUP TASK
async def cache_cleanup():
    while True:
        await asyncio.sleep(3600)
        try:
            if poster_fetcher:
                await poster_fetcher.cleanup_expired_cache()
            
            if cache_manager:
                # Cleanup expired cache entries
                await cache_manager.clear_pattern("temp:")
                
            logger.info("🧹 Cache cleanup completed")
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")

# UTILITY FUNCTION FOR EXTRACTING TITLE FROM TELEGRAM MESSAGE
async def extract_title_from_telegram_msg(msg):
    """Extract title from Telegram message"""
    try:
        caption = msg.caption if hasattr(msg, 'caption') else None
        file_name = None
        
        if msg.document:
            file_name = msg.document.file_name
        elif msg.video:
            file_name = msg.video.file_name
            
        return extract_title_from_file(file_name, caption)
    except Exception as e:
        logger.error(f"File title extraction error: {e}")
        return None

def channel_name(cid):
    return CHANNEL_CONFIG.get(cid, {}).get('name', f"Channel {cid}")

# VIDEO THUMBNAIL PROCESSING
async def extract_video_thumbnail(user_client, message):
    """Extract thumbnail from video file"""
    try:
        if message.video:
            thumbnail = message.video.thumbs[0] if message.video.thumbs else None
            if thumbnail:
                thumbnail_path = await safe_telegram_operation(
                    user_client.download_media, 
                    thumbnail.file_id, 
                    in_memory=True
                )
                if thumbnail_path:
                    thumbnail_data = base64.b64encode(thumbnail_path.getvalue()).decode('utf-8')
                    thumbnail_url = f"data:image/jpeg;base64,{thumbnail_data}"
                    return thumbnail_url
        
        if message.document:
            file_name = message.document.file_name or ""
            if is_video_file(file_name):
                thumbnail = message.document.thumbs[0] if message.document.thumbs else None
                if thumbnail:
                    thumbnail_path = await safe_telegram_operation(
                        user_client.download_media, 
                        thumbnail.file_id, 
                        in_memory=True
                    )
                    if thumbnail_path:
                        thumbnail_data = base64.b64encode(thumbnail_path.getvalue()).decode('utf-8')
                        thumbnail_url = f"data:image/jpeg;base64,{thumbnail_data}"
                        return thumbnail_url
        
        return None
        
    except Exception as e:
        logger.error(f"Video thumbnail extraction failed: {e}")
        return None

async def get_telegram_video_thumbnail(user_client, channel_id, message_id):
    """Get thumbnail directly from Telegram video"""
    try:
        msg = await safe_telegram_operation(
            user_client.get_messages,
            channel_id, 
            message_id
        )
        if not msg or (not msg.video and not msg.document):
            return None
        
        thumbnail_url = await extract_video_thumbnail(user_client, msg)
        return thumbnail_url
        
    except Exception as e:
        logger.error(f"Telegram video thumbnail error: {e}")
        return None

# MONGODB INITIALIZATION
async def init_mongodb():
    global mongo_client, db, files_col, verification_col
    try:
        logger.info("🔌 MongoDB initialization...")
        mongo_client = AsyncIOMotorClient(Config.MONGODB_URI, serverSelectionTimeoutMS=5000, maxPoolSize=10)
        await mongo_client.admin.command('ping')
        
        db = mongo_client.sk4film
        files_col = db.files
        verification_col = db.verifications
        
        # Create indexes
        logger.info("🔧 Creating indexes...")
        
        existing_indexes = await files_col.index_information()
        
        if 'title_text' not in existing_indexes:
            try:
                await files_col.create_index([("title", "text")])
                logger.info("✅ Created title text index")
            except Exception as e:
                logger.warning(f"⚠️ Title text index creation failed: {e}")
        
        if 'normalized_title_1' not in existing_indexes:
            try:
                await files_col.create_index([("normalized_title", 1)])
                logger.info("✅ Created normalized_title index")
            except Exception as e:
                logger.warning(f"⚠️ Normalized title index creation failed: {e}")
        
        if 'msg_ch_idx' not in existing_indexes:
            try:
                await files_col.create_index(
                    [("message_id", 1), ("channel_id", 1)], 
                    name="msg_ch_idx"
                )
                logger.info("✅ Created message_channel index")
            except Exception as e:
                logger.warning(f"⚠️ Message channel index creation failed: {e}")
        
        if 'indexed_at_-1' not in existing_indexes:
            try:
                await files_col.create_index([("indexed_at", -1)])
                logger.info("✅ Created indexed_at index")
            except Exception as e:
                logger.warning(f"⚠️ Indexed_at index creation failed: {e}")
        
        # Verification collection indexes
        try:
            await verification_col.create_index([("user_id", 1)], unique=True)
            logger.info("✅ Created user_id index for verification")
        except Exception as e:
            logger.warning(f"⚠️ Verification index creation failed: {e}")
        
        logger.info("✅ MongoDB OK - Optimized and Cleaned")
        return True
    except Exception as e:
        logger.error(f"❌ MongoDB: {e}")
        return False

# FILE INDEXING
async def index_single_file(message):
    """Index a single file into database"""
    try:
        if not files_col:
            return False
        
        # Check if already exists
        existing = await files_col.find_one({
            'channel_id': Config.FILE_CHANNEL_ID,
            'message_id': message.id
        })
        
        if existing:
            return True
        
        title = await extract_title_from_telegram_msg(message)
        if not title:
            return False
        
        file_id = message.document.file_id if message.document else message.video.file_id
        file_size = message.document.file_size if message.document else (message.video.file_size if message.video else 0)
        file_name = message.document.file_name if message.document else (message.video.file_name if message.video else 'video.mp4')
        quality = detect_quality(file_name)
        file_is_video = is_video_file(file_name)
        
        # Extract thumbnail if video
        thumbnail_url = None
        thumbnail_source = 'none'
        
        if file_is_video and User and user_session_ready:
            thumbnail_url = await extract_video_thumbnail(User, message)
            if thumbnail_url:
                thumbnail_source = 'video_direct'
        
        # Prepare document
        doc = {
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
            'thumbnail': thumbnail_url,
            'is_video_file': file_is_video,
            'thumbnail_source': thumbnail_source
        }
        
        await files_col.update_one(
            {
                'channel_id': Config.FILE_CHANNEL_ID,
                'message_id': message.id
            },
            {'$set': doc},
            upsert=True
        )
        
        logger.debug(f"Indexed file: {title}")
        return True
        
    except Exception as e:
        logger.error(f"Error indexing file: {e}")
        return False

async def auto_delete_file(message, delay_seconds):
    """Auto delete file after specified time"""
    try:
        await asyncio.sleep(delay_seconds)
        await message.delete()
        logger.info(f"Auto-deleted file after {delay_seconds} seconds")
    except Exception as e:
        logger.error(f"Auto-delete error: {e}")

async def index_files_background():
    """Background indexing of files"""
    if not User or files_col is None or not user_session_ready:
        logger.warning("⚠️ Cannot start indexing - User session not ready")
        return
    
    logger.info("📁 Starting SMART background indexing...")
    
    try:
        total_count = 0
        new_files_count = 0
        
        # Get the last indexed message ID
        last_indexed = await files_col.find_one({}, sort=[('message_id', -1)])
        last_message_id = last_indexed['message_id'] if last_indexed else 0
        logger.info(f"🔄 Starting from message ID: {last_message_id}")
        
        processed_count = 0
        new_messages_found = 0
        
        async for msg in safe_telegram_generator(User.get_chat_history, Config.FILE_CHANNEL_ID):
            processed_count += 1
            
            if msg.id <= last_message_id:
                if processed_count % 100 == 0:
                    logger.info(f"    ⏩ Skipped {processed_count} old messages...")
                continue
                
            new_messages_found += 1
            
            if new_messages_found % 20 == 0:
                logger.info(f"    📥 New files found: {new_messages_found}, processing...")
            
            if msg and (msg.document or msg.video):
                await index_single_file(msg)
                new_files_count += 1
                total_count += 1
                
                # Small delay to avoid flood
                if new_files_count % 10 == 0:
                    await asyncio.sleep(3)
        
        logger.info(f"✅ Background indexing finished: {new_files_count} NEW files")
        
        # Clear search cache after indexing
        if cache_manager:
            await cache_manager.clear_search_cache()
            logger.info("🧹 Search cache cleared after indexing")
        
        # Update statistics
        total_in_db = await files_col.count_documents({})
        videos_in_db = await files_col.count_documents({'is_video_file': True})
        thumbnails_in_db = await files_col.count_documents({'thumbnail': {'$ne': None}})
        
        logger.info(f"📊 FINAL STATS: Total in DB: {total_in_db}, New Added: {new_files_count}, Videos: {videos_in_db}, Thumbnails: {thumbnails_in_db}")
        
    except Exception as e:
        logger.error(f"❌ Background indexing error: {e}")

# POSTER FETCHING USING MODULAR COMPONENT
async def get_poster_guaranteed(title):
    """Get poster using PosterFetcher module"""
    if poster_fetcher:
        return await poster_fetcher.fetch_poster(title)
    
    # Fallback to custom poster
    year_match = re.search(r'\b(19|20)\d{2}\b', title)
    year = year_match.group() if year_match else ""
    
    return {
        'poster_url': f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(title)}&year={year}",
        'source': PosterSource.CUSTOM.value,
        'rating': '0.0',
        'year': year,
        'title': title
    }

# SEARCH FUNCTION USING MODULAR COMPONENTS
async def search_movies_multi_channel(query, limit=12, page=1):
    """Search across multiple channels with enhanced results"""
    offset = (page - 1) * limit
    
    # Try cache first using cache manager
    cache_key = f"search:{query}:{page}:{limit}"
    if cache_manager and cache_manager.redis_enabled:
        cached_data = await cache_manager.get_search_results(query, page, limit)
        if cached_data:
            logger.info(f"✅ Cache HIT for: {query}")
            return cached_data
    
    logger.info(f"🔍 Multi-channel search for: {query}")
    
    query_lower = query.lower()
    posts_dict = {}
    files_dict = {}
    
    # Use MongoDB search
    try:
        if files_col is not None:
            cursor = files_col.find({'$text': {'$search': query}})
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
                except:
                    continue
    except Exception as e:
        logger.error(f"File search error: {e}")
    
    # Search Telegram channels
    if user_session_ready:
        channel_results = {}
        
        async def search_single_channel(channel_id):
            channel_posts = {}
            try:
                cname = channel_name(channel_id)
                async for msg in safe_telegram_generator(User.search_messages, channel_id, query=query, limit=20):
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
        
        # Merge results
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
    
    # Sort results
    results_list = list(merged.values())
    results_list.sort(key=lambda x: (
        not x.get('is_new', False),
        not x['has_file'],
        x['date']
    ), reverse=True)
    
    total = len(results_list)
    paginated = results_list[offset:offset + limit]
    
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
            'query': query
        }
    }
    
    # Cache using cache manager
    if cache_manager:
        await cache_manager.cache_search_results(query, page, limit, result_data)
    
    logger.info(f"✅ Multi-channel search completed: {len(paginated)} results from {len(set(r.get('channel_id') for r in paginated if r.get('channel_id')))} channels")
    
    return result_data

async def get_live_posts_multi_channel(limit_per_channel=10):
    """Get posts from multiple channels concurrently"""
    if not User or not user_session_ready:
        return []
    
    all_posts = []
    
    async def fetch_channel_posts(channel_id):
        posts = []
        try:
            async for msg in safe_telegram_generator(User.get_chat_history, channel_id, limit=limit_per_channel):
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
    
    # Sort by date and remove duplicates
    seen_titles = set()
    unique_posts = []
    
    for post in sorted(all_posts, key=lambda x: x.get('date', datetime.min), reverse=True):
        if post['normalized_title'] not in seen_titles:
            seen_titles.add(post['normalized_title'])
            unique_posts.append(post)
    
    return unique_posts[:20]

# API FUNCTIONS USING MODULAR COMPONENTS
async def search_movies_api(query, limit=12, page=1):
    """Search movies API function"""
    result_data = await search_movies_multi_channel(query, limit, page)
    
    # Enhance with posters using poster fetcher
    if poster_fetcher:
        # Get posters for all results in batch
        titles = [result['title'] for result in result_data['results']]
        posters = await poster_fetcher.fetch_batch_posters(titles)
        
        for result in result_data['results']:
            if result['title'] in posters:
                poster_data = posters[result['title']]
                result['poster_url'] = poster_data['poster_url']
                result['poster_source'] = poster_data['source']
                result['poster_rating'] = poster_data.get('rating', '0.0')
                result['has_poster'] = True
            else:
                # Fallback
                result['poster_url'] = f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(result['title'])}"
                result['poster_source'] = 'custom'
                result['poster_rating'] = '0.0'
                result['has_poster'] = False
    else:
        # Fallback to individual fetching
        async with aiohttp.ClientSession() as session:
            for result in result_data['results']:
                try:
                    poster_data = await get_poster_guaranteed(result['title'])
                    if poster_data and poster_data.get('poster_url'):
                        result['poster_url'] = poster_data['poster_url']
                        result['poster_source'] = poster_data['source']
                        result['poster_rating'] = poster_data.get('rating', '0.0')
                        result['has_poster'] = True
                    else:
                        result['poster_url'] = f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(result['title'])}"
                        result['poster_source'] = 'custom'
                        result['poster_rating'] = '0.0'
                        result['has_poster'] = False
                except:
                    result['poster_url'] = f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(result['title'])}"
                    result['poster_source'] = 'custom'
                    result['poster_rating'] = '0.0'
                    result['has_poster'] = False
    
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
    
    if movies:
        if poster_fetcher:
            # Batch fetch posters
            titles = [movie['title'] for movie in movies]
            posters = await poster_fetcher.fetch_batch_posters(titles)
            
            for movie in movies:
                if movie['title'] in posters:
                    poster_data = posters[movie['title']]
                    movie['poster_url'] = poster_data['poster_url']
                    movie['poster_source'] = poster_data['source']
                    movie['poster_rating'] = poster_data.get('rating', '0.0')
                    movie['has_poster'] = True
                else:
                    # Fallback
                    movie['poster_url'] = f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(movie['title'])}"
                    movie['poster_source'] = 'custom'
                    movie['poster_rating'] = '0.0'
                    movie['has_poster'] = True
        else:
            # Fallback to individual fetching
            for movie in movies:
                try:
                    poster_data = await get_poster_guaranteed(movie['title'])
                    if poster_data and poster_data.get('poster_url'):
                        movie['poster_url'] = poster_data['poster_url']
                        movie['poster_source'] = poster_data['source']
                        movie['poster_rating'] = poster_data.get('rating', '0.0')
                        movie['has_poster'] = True
                    else:
                        movie['poster_url'] = f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(movie['title'])}"
                        movie['poster_source'] = 'custom'
                        movie['poster_rating'] = '0.0'
                        movie['has_poster'] = True
                except:
                    movie['poster_url'] = f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(movie['title'])}"
                    movie['poster_source'] = 'custom'
                    movie['poster_rating'] = '0.0'
                    movie['has_poster'] = True
    
    return movies

async def get_single_post_api(channel_id, message_id):
    """Get single movie/post details"""
    try:
        # Try to get the message
        if User and user_session_ready:
            msg = await safe_telegram_operation(
                User.get_messages,
                channel_id, 
                message_id
            )
            
            if msg and msg.text:
                title = extract_title_smart(msg.text)
                if not title:
                    title = msg.text.split('\n')[0][:60] if msg.text else "Movie Post"
                
                normalized_title = normalize_title(title)
                quality_options = {}
                has_file = False
                thumbnail_url = None
                thumbnail_source = None
                
                # Search for files with same title
                if files_col is not None:
                    cursor = files_col.find({'normalized_title': normalized_title})
                    async for doc in cursor:
                        quality = doc.get('quality', '480p')
                        if quality not in quality_options:
                            file_name = doc.get('file_name', '').lower()
                            file_is_video = is_video_file(file_name)
                            
                            if file_is_video and not thumbnail_url:
                                thumbnail_url = doc.get('thumbnail')
                                thumbnail_source = doc.get('thumbnail_source', 'unknown')
                                
                                if not thumbnail_url and user_session_ready:
                                    thumbnail_url = await get_telegram_video_thumbnail(User, doc['channel_id'], doc['message_id'])
                                    if thumbnail_url:
                                        thumbnail_source = 'video_direct'
                            
                            quality_options[quality] = {
                                'file_id': f"{doc.get('channel_id', Config.FILE_CHANNEL_ID)}_{doc.get('message_id')}_{quality}",
                                'file_size': doc.get('file_size', 0),
                                'file_name': doc.get('file_name', 'video.mp4'),
                                'is_video': file_is_video,
                                'channel_id': doc.get('channel_id'),
                                'message_id': doc.get('message_id')
                            }
                            has_file = True
                
                # Get poster
                poster_url = f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(title)}"
                poster_source = 'custom'
                poster_rating = '0.0'
                
                if poster_fetcher:
                    poster_data = await poster_fetcher.fetch_poster(title)
                    if poster_data:
                        poster_url = poster_data['poster_url']
                        poster_source = poster_data['source']
                        poster_rating = poster_data.get('rating', '0.0')
                
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
                    'views': getattr(msg, 'views', 0),
                    'thumbnail': thumbnail_url,
                    'thumbnail_source': thumbnail_source,
                    'poster_url': poster_url,
                    'poster_source': poster_source,
                    'poster_rating': poster_rating
                }
                
                return post_data
        
        # Fallback: return sample data
        return {
            'title': 'Sample Movie (2024)',
            'content': '🎬 <b>Sample Movie (2024)</b>\n📅 Release: 2024\n🎭 Genre: Action, Drama\n⭐ Starring: Popular Actors\n\n📥 Download now from SK4FiLM!',
            'channel': channel_name(channel_id),
            'channel_id': channel_id,
            'message_id': message_id,
            'date': datetime.now().isoformat(),
            'is_new': True,
            'has_file': True,
            'quality_options': {
                '1080p': {
                    'file_id': f'{channel_id}_{message_id}_1080p',
                    'file_size': 1500000000,
                    'file_name': 'Sample.Movie.2024.1080p.mkv',
                    'is_video': True
                }
            },
            'views': 1000,
            'thumbnail': None,
            'thumbnail_source': 'default',
            'poster_url': f"{Config.BACKEND_URL}/api/poster?title=Sample+Movie&year=2024",
            'poster_source': 'custom',
            'poster_rating': '7.5'
        }
        
    except Exception as e:
        logger.error(f"Single post API error: {e}")
        return None

async def verify_user_api(user_id, verification_url=None):
    """Verify user API function using verification system"""
    try:
        if not Config.VERIFICATION_REQUIRED:
            return {
                'verified': True,
                'message': 'Verification not required',
                'user_id': user_id
            }
        
        if verification_system:
            # Check if user is already verified
            is_verified, message = await verification_system.check_user_verified(user_id, premium_system)
            if is_verified:
                return {
                    'verified': True,
                    'message': message,
                    'user_id': user_id
                }
            
            # Create new verification link
            if not verification_url:
                verification_data = await verification_system.create_verification_link(user_id)
                verification_url = verification_data['short_url']
                
                return {
                    'verified': False,
                    'verification_url': verification_url,
                    'service_name': verification_data['service_name'],
                    'valid_for_hours': verification_data['valid_for_hours'],
                    'message': 'Verification link created'
                }
            
            return {
                'verified': False,
                'message': 'Please complete verification',
                'verification_url': verification_url
            }
        else:
            # Fallback
            return {
                'verified': False,
                'message': 'Verification system not available',
                'user_id': user_id
            }
    except Exception as e:
        logger.error(f"Verification API error: {e}")
        return {
            'verified': False,
            'message': f'Error: {str(e)}',
            'user_id': user_id
        }

async def get_index_status_api():
    """Get indexing status"""
    try:
        if files_col is not None:
            total = await files_col.count_documents({})
            video_files = await files_col.count_documents({'is_video_file': True})
            video_thumbnails = await files_col.count_documents({'is_video_file': True, 'thumbnail': {'$ne': None}})
            total_thumbnails = await files_col.count_documents({'thumbnail': {'$ne': None}})
            
            latest = await files_col.find_one({}, sort=[('indexed_at', -1)])
            last_indexed = "Never"
            if latest and latest.get('indexed_at'):
                dt = latest['indexed_at']
                if isinstance(dt, datetime):
                    mins_ago = int((datetime.now() - dt).total_seconds() / 60)
                    last_indexed = f"{mins_ago} min ago" if mins_ago > 0 else "Just now"
            
            return {
                'total_indexed': total,
                'video_files': video_files,
                'video_thumbnails': video_thumbnails,
                'total_thumbnails': total_thumbnails,
                'thumbnail_coverage': f"{(video_thumbnails/video_files*100):.1f}%" if video_files > 0 else "0%",
                'last_indexed': last_indexed,
                'bot_status': bot_started,
                'user_session': user_session_ready,
                'redis_enabled': cache_manager.redis_enabled if cache_manager else False,
                'cache_stats': poster_fetcher.get_stats() if poster_fetcher else {}
            }
        else:
            return {
                'total_indexed': 0,
                'video_files': 0,
                'video_thumbnails': 0,
                'total_thumbnails': 0,
                'thumbnail_coverage': "0%",
                'last_indexed': "Never",
                'bot_status': bot_started,
                'user_session': user_session_ready,
                'redis_enabled': cache_manager.redis_enabled if cache_manager else False,
                'cache_stats': {}
            }
    except Exception as e:
        logger.error(f"Index status API error: {e}")
        return {
            'error': str(e),
            'status': 'error'
        }

# TELEGRAM BOT INITIALIZATION
async def init_telegram_clients():
    """Initialize Telegram clients"""
    global User, bot, bot_started, user_session_ready
    
    try:
        # Initialize Pyrogram User Client
        logger.info("📱 Initializing User Session...")
        try:
            User = Client(
                name="user_session",
                api_id=Config.API_ID,
                api_hash=Config.API_HASH,
                session_string=Config.USER_SESSION_STRING,
                sleep_threshold=60,
                max_concurrent_transmissions=2
            )
            await User.start()
            
            # Verify channel access
            for channel_id in Config.TEXT_CHANNEL_IDS + [Config.FILE_CHANNEL_ID]:
                try:
                    chat = await User.get_chat(channel_id)
                    logger.info(f"✅ Access verified: {chat.title} ({channel_id})")
                    await asyncio.sleep(1)
                except Exception as e:
                    logger.warning(f"⚠️ Cannot access channel {channel_id}: {e}")
            
            user_session_ready = True
            logger.info("✅ User Session Started")
            
        except Exception as e:
            logger.error(f"❌ User Session Error: {e}")
            user_session_ready = False
        
        # Initialize Bot Client
        logger.info("🤖 Initializing Bot...")
        try:
            bot = Client(
                name="sk4film_bot",
                api_id=Config.API_ID,
                api_hash=Config.API_HASH,
                bot_token=Config.BOT_TOKEN,
                sleep_threshold=60
            )
            await bot.start()
            bot_started = True
            bot_info = await bot.get_me()
            logger.info(f"✅ Bot Started: @{bot_info.username}")
        except Exception as e:
            logger.error(f"❌ Bot Error: {e}")
            bot_started = False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Telegram clients initialization failed: {e}")
        return False

# MAIN INITIALIZATION
async def init_system():
    """Initialize complete system"""
    start_time = time.time()
    
    try:
        logger.info("🚀 Starting SK4FiLM v8.0 - MODULAR SYSTEM...")
        
        # Initialize MongoDB
        mongo_ok = await init_mongodb()
        if not mongo_ok:
            logger.error("❌ MongoDB initialization failed")
            return False
        
        # Initialize modular components
        global cache_manager, verification_system, premium_system, poster_fetcher, sk4film_bot
        
        # Initialize Cache Manager
        cache_manager = CacheManager(Config)
        redis_ok = await cache_manager.init_redis()
        if redis_ok:
            logger.info("✅ Cache Manager initialized")
            await cache_manager.start_cleanup_task()
        else:
            logger.warning("⚠️ Cache Manager - Redis not available")
        
        # Initialize Verification System
        verification_system = VerificationSystem(Config, db)
        await verification_system.start_cleanup_task()
        logger.info("✅ Verification System initialized")
        
        # Initialize Premium System
        premium_system = PremiumSystem(Config, db)
        await premium_system.start_cleanup_task()
        logger.info("✅ Premium System initialized")
        
        # Initialize Poster Fetcher
        poster_fetcher = PosterFetcher(Config, cache_manager)
        logger.info("✅ Poster Fetcher initialized")
        
        # Initialize SK4FiLMBot
        sk4film_bot = SK4FiLMBot(Config, db)
        await sk4film_bot.initialize()
        logger.info("✅ SK4FiLMBot initialized")
        
        # Initialize Telegram clients
        telegram_ok = await init_telegram_clients()
        if not telegram_ok:
            logger.warning("⚠️ Telegram clients not initialized")
        
        # Setup bot handlers
        if sk4film_bot and sk4film_bot.bot_started and bot:
            await setup_bot_handlers(bot, sk4film_bot)
            logger.info("✅ Bot handlers setup complete")
        
        # Start background tasks
        asyncio.create_task(cache_cleanup())
        
        # Start indexing in background
        if user_session_ready:
            asyncio.create_task(index_files_background())
        
        logger.info(f"⚡ SK4FiLM Started in {time.time() - start_time:.1f}s")
        return True
        
    except Exception as e:
        logger.error(f"❌ System initialization failed: {e}")
        return False

# API ROUTES
@app.route('/')
async def root():
    tf = await files_col.count_documents({}) if files_col is not None else 0
    video_files = await files_col.count_documents({'is_video_file': True}) if files_col is not None else 0
    thumbnails = await files_col.count_documents({'thumbnail': {'$ne': None}}) if files_col is not None else 0
    
    return jsonify({
        'status': 'healthy',
        'service': 'SK4FiLM v8.0 - MODULAR API SYSTEM',
        'database': {
            'total_files': tf, 
            'video_files': video_files,
            'thumbnails': thumbnails,
            'mode': 'MULTI-CHANNEL MODULAR'
        },
        'modules': {
            'cache_manager': cache_manager is not None,
            'verification_system': verification_system is not None,
            'premium_system': premium_system is not None,
            'poster_fetcher': poster_fetcher is not None,
            'bot_system': sk4film_bot is not None
        },
        'channels': {
            'text_channels': len(Config.TEXT_CHANNEL_IDS),
            'file_channels': 1,
            'total_channels': len(Config.TEXT_CHANNEL_IDS) + 1
        },
        'bot_status': 'online' if bot_started else 'starting',
        'user_session': 'ready' if user_session_ready else 'flood_wait',
        'features': 'COMPLETE MODULAR SYSTEM',
        'api_endpoints': {
            'health': '/health',
            'movies': '/api/movies',
            'search': '/api/search?query=movie&page=1&limit=12',
            'post': '/api/post?channel=-1001891090100&message=12345',
            'poster': '/api/poster?title=Movie+Name&year=2024',
            'verify_user': '/api/verify_user (POST)',
            'premium_plans': '/api/premium/plans',
            'user_status': '/api/user/status?user_id=123',
            'clear_cache': '/api/clear_cache',
            'index_status': '/api/index_status',
            'system_stats': '/api/stats'
        }
    })

@app.route('/health')
async def health():
    return jsonify({
        'status': 'ok' if bot_started else 'starting',
        'user_session': user_session_ready,
        'modules': {
            'cache': cache_manager.redis_enabled if cache_manager else False,
            'verification': verification_system is not None,
            'premium': premium_system is not None,
            'poster': poster_fetcher is not None,
            'bot': sk4film_bot.bot_started if sk4film_bot else False
        },
        'channels_configured': len(Config.TEXT_CHANNEL_IDS),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/movies', methods=['GET'])
async def api_movies():
    """Home page movies"""
    try:
        movies = await get_home_movies_live()
        
        return jsonify({
            'status': 'success',
            'movies': movies,
            'total': len(movies),
            'timestamp': datetime.now().isoformat(),
            'bot_username': Config.BOT_USERNAME
        })
    except Exception as e:
        logger.error(f"Movies API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/search', methods=['GET'])
async def api_search():
    """Search movies"""
    try:
        query = request.args.get('query', '').strip()
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 12))
        
        if len(query) < 2:
            return jsonify({
                'status': 'error',
                'message': 'Query must be at least 2 characters'
            }), 400
        
        result_data = await search_movies_api(query, limit, page)
        
        return jsonify({
            'status': 'success',
            'query': query,
            'results': result_data['results'],
            'pagination': result_data['pagination'],
            'search_metadata': result_data.get('search_metadata', {}),
            'timestamp': datetime.now().isoformat(),
            'bot_username': Config.BOT_USERNAME,
            'channels_searched': len(Config.TEXT_CHANNEL_IDS)
        })
    except Exception as e:
        logger.error(f"Search API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'query': query
        }), 500

@app.route('/api/post', methods=['GET'])
async def api_post():
    """Get single post details"""
    try:
        channel_id = int(request.args.get('channel', Config.MAIN_CHANNEL_ID))
        message_id = int(request.args.get('message', 0))
        
        if message_id <= 0:
            return jsonify({
                'status': 'error',
                'message': 'Invalid message ID'
            }), 400
        
        post_data = await get_single_post_api(channel_id, message_id)
        
        if post_data:
            return jsonify({
                'status': 'success',
                'post': post_data,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Post not found'
            }), 404
            
    except Exception as e:
        logger.error(f"Post API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/poster', methods=['GET'])
async def api_poster():
    """Get movie poster"""
    try:
        title = request.args.get('title', '').strip()
        year = request.args.get('year', '')
        
        if not title:
            return jsonify({
                'status': 'error',
                'message': 'Title is required'
            }), 400
        
        if poster_fetcher:
            poster_data = await poster_fetcher.fetch_poster(title)
            
            if poster_data:
                return jsonify({
                    'status': 'success',
                    'poster': poster_data,
                    'title': title,
                    'year': year,
                    'timestamp': datetime.now().isoformat()
                })
        
        # Fallback
        return jsonify({
            'status': 'success',
            'poster': {
                'poster_url': f"{Config.BACKEND_URL}/api/poster?title={urllib.parse.quote(title)}&year={year}",
                'source': PosterSource.CUSTOM.value,
                'rating': '0.0'
            },
            'title': title,
            'year': year,
            'timestamp': datetime.now().isoformat()
        })
                
    except Exception as e:
        logger.error(f"Poster API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/verify_user', methods=['POST'])
async def api_verify_user():
    """Verify user with URL shortener"""
    try:
        data = await request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No JSON data provided'
            }), 400
        
        user_id = data.get('user_id')
        verification_url = data.get('verification_url')
        
        if not user_id:
            return jsonify({
                'status': 'error',
                'message': 'user_id is required'
            }), 400
        
        result = await verify_user_api(user_id, verification_url)
        
        return jsonify({
            'status': 'success',
            'verification': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Verify user API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/premium/plans', methods=['GET'])
async def api_premium_plans():
    """Get premium plans"""
    try:
        if premium_system:
            plans = await premium_system.get_all_plans()
            
            return jsonify({
                'status': 'success',
                'plans': plans,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Premium system not available'
            }), 503
    except Exception as e:
        logger.error(f"Premium plans API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/user/status', methods=['GET'])
async def api_user_status():
    """Get user status (premium/verification)"""
    try:
        user_id = int(request.args.get('user_id', 0))
        
        if user_id <= 0:
            return jsonify({
                'status': 'error',
                'message': 'Valid user_id is required'
            }), 400
        
        result = {}
        
        # Get premium status
        if premium_system:
            premium_status = await premium_system.get_subscription_details(user_id)
            result['premium'] = premium_status
        
        # Get verification status
        if verification_system:
            is_verified, message = await verification_system.check_user_verified(user_id, premium_system)
            result['verification'] = {
                'is_verified': is_verified,
                'message': message,
                'needs_verification': not is_verified and Config.VERIFICATION_REQUIRED
            }
        
        # Get download permissions
        if premium_system:
            can_download, download_message, download_details = await premium_system.can_user_download(user_id)
            result['download'] = {
                'can_download': can_download,
                'message': download_message,
                'details': download_details
            }
        
        return jsonify({
            'status': 'success',
            'user_id': user_id,
            'user_status': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"User status API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/clear_cache', methods=['POST'])
async def api_clear_cache():
    """Clear all caches (admin only)"""
    try:
        data = await request.get_json()
        admin_key = data.get('admin_key') if data else request.headers.get('X-Admin-Key')
        
        if not admin_key or admin_key != os.environ.get('ADMIN_KEY', 'sk4film_admin_123'):
            return jsonify({
                'status': 'error',
                'message': 'Unauthorized'
            }), 401
        
        cleared = {
            'cache_manager': False,
            'poster_fetcher': False,
            'search_cache': 0
        }
        
        if cache_manager:
            await cache_manager.clear_all()
            cleared['cache_manager'] = True
        
        if poster_fetcher:
            poster_fetcher.clear_cache()
            cleared['poster_fetcher'] = True
        
        if cache_manager:
            cleared['search_cache'] = await cache_manager.clear_search_cache()
        
        return jsonify({
            'status': 'success',
            'message': 'Cache cleared successfully',
            'cleared': cleared,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Clear cache API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/index_status', methods=['GET'])
async def api_index_status():
    """Get indexing status"""
    try:
        status_data = await get_index_status_api()
        
        return jsonify({
            'status': 'success',
            'indexing': status_data,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Index status API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/stats', methods=['GET'])
async def api_stats():
    """Get system statistics"""
    try:
        stats = {}
        
        # Database stats
        if files_col:
            stats['database'] = {
                'total_files': await files_col.count_documents({}),
                'video_files': await files_col.count_documents({'is_video_file': True}),
                'files_with_thumbnails': await files_col.count_documents({'thumbnail': {'$ne': None}}),
                'unique_titles': len(await files_col.distinct('normalized_title'))
            }
        
        # Cache stats
        if cache_manager:
            cache_stats = await cache_manager.get_stats_summary()
            stats['cache'] = cache_stats
        
        # Poster fetcher stats
        if poster_fetcher:
            poster_stats = poster_fetcher.get_stats()
            stats['poster'] = poster_stats
        
        # Premium system stats
        if premium_system:
            premium_stats = await premium_system.get_admin_stats()
            stats['premium'] = premium_stats
        
        # Verification system stats
        if verification_system:
            verification_stats = await verification_system.get_user_stats()
            stats['verification'] = verification_stats
        
        # System info
        stats['system'] = {
            'bot_status': bot_started,
            'user_session': user_session_ready,
            'channels_configured': len(Config.TEXT_CHANNEL_IDS),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify({
            'status': 'success',
            'stats': stats,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Stats API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# STARTUP AND SHUTDOWN HANDLERS
@app.before_serving
async def startup():
    """Startup initialization"""
    await init_system()

@app.after_serving
async def shutdown():
    """Clean shutdown"""
    logger.info("🛑 Shutting down SK4FiLM...")
    
    try:
        if User and user_session_ready:
            await User.stop()
            logger.info("✅ User Session Stopped")
    except:
        pass
    
    try:
        if bot and bot_started:
            await bot.stop()
            logger.info("✅ Bot Stopped")
    except:
        pass
    
    try:
        if sk4film_bot:
            await sk4film_bot.shutdown()
            logger.info("✅ SK4FiLMBot Stopped")
    except:
        pass
    
    try:
        if cache_manager:
            await cache_manager.stop()
            logger.info("✅ Cache Manager Stopped")
    except:
        pass
    
    try:
        if verification_system:
            await verification_system.stop()
            logger.info("✅ Verification System Stopped")
    except:
        pass
    
    try:
        if premium_system:
            await premium_system.stop_cleanup_task()
            logger.info("✅ Premium System Stopped")
    except:
        pass
    
    try:
        if mongo_client:
            mongo_client.close()
            logger.info("✅ MongoDB Connection Closed")
    except:
        pass
    
    logger.info("👋 Shutdown complete")

# MAIN ENTRY POINT
if __name__ == "__main__":
    config = HyperConfig()
    config.bind = [f"0.0.0.0:{Config.WEB_SERVER_PORT}"]
    config.worker_class = "asyncio"
    config.workers = 1
    config.accesslog = "-"
    config.errorlog = "-"
    config.loglevel = "info"
    
    logger.info(f"🌐 Starting Quart server on port {Config.WEB_SERVER_PORT}...")
    
    # Run the app
    asyncio.run(serve(app, config))
