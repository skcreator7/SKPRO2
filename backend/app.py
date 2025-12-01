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
        if files_col is not None:
            # Try text search first
            try:
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
                    except Exception as e:
                        logger.error(f"Error processing search result: {e}")
                        continue
            except Exception as e:
                logger.error(f"Text search error: {e}")
            
            # If no results from text search, try regex search
            if not files_dict:
                regex_cursor = files_col.find({
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

async def extract_video_thumbnail(user_client, message):
    """Extract thumbnail from video file"""
    try:
        # Method 1: Get thumbnail from video message
        if message.video:
            # Get video thumbnail from Telegram
            thumbnail = message.video.thumbs[0] if message.video.thumbs else None
            if thumbnail:
                # Download thumbnail file
                thumbnail_path = await safe_telegram_operation(
                    user_client.download_media, 
                    thumbnail.file_id, 
                    in_memory=True
                )
                if thumbnail_path:
                    # Convert to base64 for web display
                    thumbnail_data = base64.b64encode(thumbnail_path.getvalue()).decode('utf-8')
                    thumbnail_url = f"data:image/jpeg;base64,{thumbnail_data}"
                    return thumbnail_url
        
        # Method 2: Try to get from document if it's a video file
        if message.document:
            file_name = message.document.file_name or ""
            if is_video_file(file_name):
                # Try to get document thumbnail
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
        # Get the message
        msg = await safe_telegram_operation(
            user_client.get_messages,
            channel_id, 
            message_id
        )
        if not msg or (not msg.video and not msg.document):
            return None
        
        # Extract thumbnail
        thumbnail_url = await extract_video_thumbnail(user_client, msg)
        return thumbnail_url
        
    except Exception as e:
        logger.error(f"Telegram video thumbnail error: {e}")
        return None

async def process_thumbnail_batch(thumbnail_batch):
    """Process thumbnails in batches for better performance"""
    semaphore = asyncio.Semaphore(3)  # Limit concurrent thumbnail processing
    
    async def process_single(file_data):
        async with semaphore:
            try:
                thumbnail_url = await extract_video_thumbnail(user_client, file_data['message'])
                if thumbnail_url:
                    return file_data['message'].id, thumbnail_url, 'video_direct'
                
                # Fallback to poster
                title = extract_title_from_file(file_data['message'])
                if title and poster_fetcher is not None:
                    poster_data = await poster_fetcher.fetch_poster(title)
                    if poster_data and poster_data.get('poster_url'):
                        return file_data['message'].id, poster_data['poster_url'], 'poster_api'
                
                return file_data['message'].id, None, 'none'
            except Exception as e:
                logger.error(f"Thumbnail processing failed for message {file_data['message'].id}: {e}")
                return file_data['message'].id, None, 'error'
    
    tasks = [process_single(file_data) for file_data in thumbnail_batch]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    thumbnails = {}
    for result in results:
        if isinstance(result, tuple) and len(result) == 3:
            message_id, thumbnail_url, source = result
            if thumbnail_url:
                thumbnails[message_id] = (thumbnail_url, source)
    
    return thumbnails

async def index_files_background():
    """Smart background indexing - only new files with batch thumbnail processing"""
    if not user_client or files_col is None or not user_session_ready:
        logger.warning("⚠️ Cannot start indexing - User session not ready")
        return
    
    logger.info("📁 Starting SMART background indexing (NEW FILES ONLY)...")
    
    try:
        total_count = 0
        new_files_count = 0
        video_files_count = 0
        successful_thumbnails = 0
        batch = []
        batch_size = 15  # Smaller batch size for better performance
        thumbnail_batch = []
        
        # Get the last indexed message ID to start from there
        last_indexed = await files_col.find_one(
            {}, 
            sort=[('message_id', -1)]
        )
        
        last_message_id = last_indexed['message_id'] if last_indexed else 0
        logger.info(f"🔄 Starting from message ID: {last_message_id}")
        
        # Process only NEW messages (after last indexed)
        processed_count = 0
        new_messages_found = 0
        
        async for msg in safe_telegram_generator(user_client.get_chat_history, Config.FILE_CHANNEL_ID):
            processed_count += 1
            
            # Stop if we've processed too many old messages
            if msg.id <= last_message_id:
                if processed_count % 100 == 0:
                    logger.info(f"    ⏩ Skipped {processed_count} old messages...")
                continue
                
            new_messages_found += 1
            
            if new_messages_found % 20 == 0:
                logger.info(f"    📥 New files found: {new_messages_found}, processing...")
            
            if msg and (msg.document or msg.video):
                # Check if already exists in database (double check)
                existing = await files_col.find_one({
                    'channel_id': Config.FILE_CHANNEL_ID,
                    'message_id': msg.id
                })
                
                if existing:
                    if new_messages_found % 50 == 0:
                        logger.info(f"    ⏭️ Already indexed: {msg.id}, skipping...")
                    continue
                
                title = extract_title_from_file(msg)
                if title:
                    file_id = msg.document.file_id if msg.document else msg.video.file_id
                    file_size = msg.document.file_size if msg.document else (msg.video.file_size if msg.video else 0)
                    file_name = msg.document.file_name if msg.document else (msg.video.file_name if msg.video else 'video.mp4')
                    quality = detect_quality(file_name)
                    
                    file_is_video = is_video_file(file_name)
                    
                    # Prepare for batch thumbnail processing
                    if file_is_video:
                        video_files_count += 1
                        thumbnail_batch.append({
                            'message': msg,
                            'title': title,
                            'file_is_video': file_is_video
                        })
                    
                    # Add to database batch
                    batch.append({
                        'channel_id': Config.FILE_CHANNEL_ID,
                        'message_id': msg.id,
                        'title': title,
                        'normalized_title': normalize_title(title),
                        'file_id': file_id,
                        'quality': quality,
                        'file_size': file_size,
                        'file_name': file_name,
                        'caption': msg.caption or '',
                        'date': msg.date,
                        'indexed_at': datetime.now(),
                        'thumbnail': None,  # Will be updated after batch processing
                        'is_video_file': file_is_video,
                        'thumbnail_source': 'pending'
                    })
                    
                    total_count += 1
                    new_files_count += 1
                    
                    # Process thumbnail batch when it reaches size
                    if len(thumbnail_batch) >= 10:
                        logger.info(f"    🖼️ Processing batch of {len(thumbnail_batch)} thumbnails...")
                        thumbnails = await process_thumbnail_batch(thumbnail_batch)
                        
                        # Update batch with thumbnails
                        for doc in batch:
                            if doc['message_id'] in thumbnails:
                                thumbnail_url, source = thumbnails[doc['message_id']]
                                doc['thumbnail'] = thumbnail_url
                                doc['thumbnail_source'] = source
                                successful_thumbnails += 1
                        
                        thumbnail_batch = []
                    
                    # Process database batch
                    if len(batch) >= batch_size:
                        try:
                            for doc in batch:
                                await files_col.update_one(
                                    {
                                        'channel_id': doc['channel_id'], 
                                        'message_id': doc['message_id']
                                    },
                                    {'$set': doc},
                                    upsert=True
                                )
                            logger.info(f"    ✅ Batch processed: {new_files_count} new files, {successful_thumbnails} thumbnails")
                            batch = []
                            # Increased delay between batches to avoid flood
                            await asyncio.sleep(3)
                        except Exception as e:
                            logger.error(f"Batch error: {e}")
                            batch = []
        
        # Process remaining thumbnail batch
        if thumbnail_batch:
            logger.info(f"    🖼️ Processing final batch of {len(thumbnail_batch)} thumbnails...")
            thumbnails = await process_thumbnail_batch(thumbnail_batch)
            
            # Update batch with thumbnails
            for doc in batch:
                if doc['message_id'] in thumbnails:
                    thumbnail_url, source = thumbnails[doc['message_id']]
                    doc['thumbnail'] = thumbnail_url
                    doc['thumbnail_source'] = source
                    successful_thumbnails += 1
        
        # Process remaining database batch
        if batch:
            try:
                for doc in batch:
                    await files_col.update_one(
                        {
                            'channel_id': doc['channel_id'], 
                            'message_id': doc['message_id']
                        },
                        {'$set': doc},
                        upsert=True
                    )
            except Exception as e:
                logger.error(f"Final batch error: {e}")
        
        logger.info(f"✅ SMART indexing finished: {new_files_count} NEW files, {video_files_count} video files, {successful_thumbnails} thumbnails")
        logger.info(f"📊 Processed {processed_count} messages, found {new_messages_found} new messages")
        
        # Clear search cache after indexing new files
        if cache_manager is not None:
            await cache_manager.clear_pattern("search:*")
        logger.info("🧹 Search cache cleared after indexing")
        
        # Update statistics
        total_in_db = await files_col.count_documents({})
        videos_in_db = await files_col.count_documents({'is_video_file': True})
        thumbnails_in_db = await files_col.count_documents({'thumbnail': {'$ne': None}})
        
        logger.info(f"📊 FINAL STATS: Total in DB: {total_in_db}, New Added: {new_files_count}, Videos: {videos_in_db}, Thumbnails: {thumbnails_in_db}")
        
    except Exception as e:
        logger.error(f"❌ Background indexing error: {e}")

# Utility functions (keep your existing ones)
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
            'mongodb_connected': mongo_client is not None,
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
        },
        'enhanced_search': {
            'enabled': True,
            'channels_ready': len(Config.TEXT_CHANNEL_IDS),
            'redis_ready': cache_manager.redis_enabled if cache_manager is not None else False
        }
    })

# Update the existing /api/search route to use enhanced search
@app.route('/api/search')
async def api_search():
    """Enhanced search for movies with multi-channel support"""
    try:
        q = request.args.get('query', '').strip()
        p = int(request.args.get('page', 1))
        l = int(request.args.get('limit', 12))
        search_mode = request.args.get('mode', 'enhanced')  # 'enhanced' or 'basic'
        
        if not q:
            return jsonify({'status': 'error', 'message': 'Query required'}), 400
        
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
        
        # Use enhanced search if available and requested
        if search_mode == 'enhanced' and user_session_ready:
            try:
                result = await search_movies_multi_channel(q, l, p)
                return jsonify({
                    'status': 'success',
                    'query': q,
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
        cache_key = f"search:basic:{q}:{p}:{l}"
        if cache_manager is not None:
            cached = await cache_manager.get(cache_key)
            if cached:
                return jsonify(json.loads(cached))
        
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
            },
            'search_mode': 'basic'
        }
        
        # Cache the results
        if cache_manager is not None:
            await cache_manager.set(cache_key, json.dumps(response_data), expire_seconds=1800)
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/search/enhanced')
async def api_enhanced_search():
    """Force enhanced multi-channel search"""
    try:
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
        
        if cache_manager is not None:
            await cache_manager.clear_pattern("search:*")
            
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
            'message': 'Search cache cleared and stats reset',
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
        
        if not movies and files_col is not None:
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
        if files_col is None:
            return jsonify({'status': 'error', 'message': 'Database not ready'}), 503
        
        cursor = files_col.find({}).sort('date', -1).limit(20)
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

# Add enhanced search command to bot handlers
async def setup_bot():
    """Setup bot commands and handlers"""import asyncio
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
        if files_col is not None:
            # Try text search first
            try:
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
                    except Exception as e:
                        logger.error(f"Error processing search result: {e}")
                        continue
            except Exception as e:
                logger.error(f"Text search error: {e}")
            
            # If no results from text search, try regex search
            if not files_dict:
                regex_cursor = files_col.find({
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

async def extract_video_thumbnail(user_client, message):
    """Extract thumbnail from video file"""
    try:
        # Method 1: Get thumbnail from video message
        if message.video:
            # Get video thumbnail from Telegram
            thumbnail = message.video.thumbs[0] if message.video.thumbs else None
            if thumbnail:
                # Download thumbnail file
                thumbnail_path = await safe_telegram_operation(
                    user_client.download_media, 
                    thumbnail.file_id, 
                    in_memory=True
                )
                if thumbnail_path:
                    # Convert to base64 for web display
                    thumbnail_data = base64.b64encode(thumbnail_path.getvalue()).decode('utf-8')
                    thumbnail_url = f"data:image/jpeg;base64,{thumbnail_data}"
                    return thumbnail_url
        
        # Method 2: Try to get from document if it's a video file
        if message.document:
            file_name = message.document.file_name or ""
            if is_video_file(file_name):
                # Try to get document thumbnail
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
        # Get the message
        msg = await safe_telegram_operation(
            user_client.get_messages,
            channel_id, 
            message_id
        )
        if not msg or (not msg.video and not msg.document):
            return None
        
        # Extract thumbnail
        thumbnail_url = await extract_video_thumbnail(user_client, msg)
        return thumbnail_url
        
    except Exception as e:
        logger.error(f"Telegram video thumbnail error: {e}")
        return None

async def process_thumbnail_batch(thumbnail_batch):
    """Process thumbnails in batches for better performance"""
    semaphore = asyncio.Semaphore(3)  # Limit concurrent thumbnail processing
    
    async def process_single(file_data):
        async with semaphore:
            try:
                thumbnail_url = await extract_video_thumbnail(user_client, file_data['message'])
                if thumbnail_url:
                    return file_data['message'].id, thumbnail_url, 'video_direct'
                
                # Fallback to poster
                title = extract_title_from_file(file_data['message'])
                if title and poster_fetcher is not None:
                    poster_data = await poster_fetcher.fetch_poster(title)
                    if poster_data and poster_data.get('poster_url'):
                        return file_data['message'].id, poster_data['poster_url'], 'poster_api'
                
                return file_data['message'].id, None, 'none'
            except Exception as e:
                logger.error(f"Thumbnail processing failed for message {file_data['message'].id}: {e}")
                return file_data['message'].id, None, 'error'
    
    tasks = [process_single(file_data) for file_data in thumbnail_batch]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    thumbnails = {}
    for result in results:
        if isinstance(result, tuple) and len(result) == 3:
            message_id, thumbnail_url, source = result
            if thumbnail_url:
                thumbnails[message_id] = (thumbnail_url, source)
    
    return thumbnails

async def index_files_background():
    """Smart background indexing - only new files with batch thumbnail processing"""
    if not user_client or files_col is None or not user_session_ready:
        logger.warning("⚠️ Cannot start indexing - User session not ready")
        return
    
    logger.info("📁 Starting SMART background indexing (NEW FILES ONLY)...")
    
    try:
        total_count = 0
        new_files_count = 0
        video_files_count = 0
        successful_thumbnails = 0
        batch = []
        batch_size = 15  # Smaller batch size for better performance
        thumbnail_batch = []
        
        # Get the last indexed message ID to start from there
        last_indexed = await files_col.find_one(
            {}, 
            sort=[('message_id', -1)]
        )
        
        last_message_id = last_indexed['message_id'] if last_indexed else 0
        logger.info(f"🔄 Starting from message ID: {last_message_id}")
        
        # Process only NEW messages (after last indexed)
        processed_count = 0
        new_messages_found = 0
        
        async for msg in safe_telegram_generator(user_client.get_chat_history, Config.FILE_CHANNEL_ID):
            processed_count += 1
            
            # Stop if we've processed too many old messages
            if msg.id <= last_message_id:
                if processed_count % 100 == 0:
                    logger.info(f"    ⏩ Skipped {processed_count} old messages...")
                continue
                
            new_messages_found += 1
            
            if new_messages_found % 20 == 0:
                logger.info(f"    📥 New files found: {new_messages_found}, processing...")
            
            if msg and (msg.document or msg.video):
                # Check if already exists in database (double check)
                existing = await files_col.find_one({
                    'channel_id': Config.FILE_CHANNEL_ID,
                    'message_id': msg.id
                })
                
                if existing:
                    if new_messages_found % 50 == 0:
                        logger.info(f"    ⏭️ Already indexed: {msg.id}, skipping...")
                    continue
                
                title = extract_title_from_file(msg)
                if title:
                    file_id = msg.document.file_id if msg.document else msg.video.file_id
                    file_size = msg.document.file_size if msg.document else (msg.video.file_size if msg.video else 0)
                    file_name = msg.document.file_name if msg.document else (msg.video.file_name if msg.video else 'video.mp4')
                    quality = detect_quality(file_name)
                    
                    file_is_video = is_video_file(file_name)
                    
                    # Prepare for batch thumbnail processing
                    if file_is_video:
                        video_files_count += 1
                        thumbnail_batch.append({
                            'message': msg,
                            'title': title,
                            'file_is_video': file_is_video
                        })
                    
                    # Add to database batch
                    batch.append({
                        'channel_id': Config.FILE_CHANNEL_ID,
                        'message_id': msg.id,
                        'title': title,
                        'normalized_title': normalize_title(title),
                        'file_id': file_id,
                        'quality': quality,
                        'file_size': file_size,
                        'file_name': file_name,
                        'caption': msg.caption or '',
                        'date': msg.date,
                        'indexed_at': datetime.now(),
                        'thumbnail': None,  # Will be updated after batch processing
                        'is_video_file': file_is_video,
                        'thumbnail_source': 'pending'
                    })
                    
                    total_count += 1
                    new_files_count += 1
                    
                    # Process thumbnail batch when it reaches size
                    if len(thumbnail_batch) >= 10:
                        logger.info(f"    🖼️ Processing batch of {len(thumbnail_batch)} thumbnails...")
                        thumbnails = await process_thumbnail_batch(thumbnail_batch)
                        
                        # Update batch with thumbnails
                        for doc in batch:
                            if doc['message_id'] in thumbnails:
                                thumbnail_url, source = thumbnails[doc['message_id']]
                                doc['thumbnail'] = thumbnail_url
                                doc['thumbnail_source'] = source
                                successful_thumbnails += 1
                        
                        thumbnail_batch = []
                    
                    # Process database batch
                    if len(batch) >= batch_size:
                        try:
                            for doc in batch:
                                await files_col.update_one(
                                    {
                                        'channel_id': doc['channel_id'], 
                                        'message_id': doc['message_id']
                                    },
                                    {'$set': doc},
                                    upsert=True
                                )
                            logger.info(f"    ✅ Batch processed: {new_files_count} new files, {successful_thumbnails} thumbnails")
                            batch = []
                            # Increased delay between batches to avoid flood
                            await asyncio.sleep(3)
                        except Exception as e:
                            logger.error(f"Batch error: {e}")
                            batch = []
        
        # Process remaining thumbnail batch
        if thumbnail_batch:
            logger.info(f"    🖼️ Processing final batch of {len(thumbnail_batch)} thumbnails...")
            thumbnails = await process_thumbnail_batch(thumbnail_batch)
            
            # Update batch with thumbnails
            for doc in batch:
                if doc['message_id'] in thumbnails:
                    thumbnail_url, source = thumbnails[doc['message_id']]
                    doc['thumbnail'] = thumbnail_url
                    doc['thumbnail_source'] = source
                    successful_thumbnails += 1
        
        # Process remaining database batch
        if batch:
            try:
                for doc in batch:
                    await files_col.update_one(
                        {
                            'channel_id': doc['channel_id'], 
                            'message_id': doc['message_id']
                        },
                        {'$set': doc},
                        upsert=True
                    )
            except Exception as e:
                logger.error(f"Final batch error: {e}")
        
        logger.info(f"✅ SMART indexing finished: {new_files_count} NEW files, {video_files_count} video files, {successful_thumbnails} thumbnails")
        logger.info(f"📊 Processed {processed_count} messages, found {new_messages_found} new messages")
        
        # Clear search cache after indexing new files
        if cache_manager is not None:
            await cache_manager.clear_pattern("search:*")
        logger.info("🧹 Search cache cleared after indexing")
        
        # Update statistics
        total_in_db = await files_col.count_documents({})
        videos_in_db = await files_col.count_documents({'is_video_file': True})
        thumbnails_in_db = await files_col.count_documents({'thumbnail': {'$ne': None}})
        
        logger.info(f"📊 FINAL STATS: Total in DB: {total_in_db}, New Added: {new_files_count}, Videos: {videos_in_db}, Thumbnails: {thumbnails_in_db}")
        
    except Exception as e:
        logger.error(f"❌ Background indexing error: {e}")

# Utility functions (keep your existing ones)
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
            'mongodb_connected': mongo_client is not None,
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
        },
        'enhanced_search': {
            'enabled': True,
            'channels_ready': len(Config.TEXT_CHANNEL_IDS),
            'redis_ready': cache_manager.redis_enabled if cache_manager is not None else False
        }
    })

# Update the existing /api/search route to use enhanced search
@app.route('/api/search')
async def api_search():
    """Enhanced search for movies with multi-channel support"""
    try:
        q = request.args.get('query', '').strip()
        p = int(request.args.get('page', 1))
        l = int(request.args.get('limit', 12))
        search_mode = request.args.get('mode', 'enhanced')  # 'enhanced' or 'basic'
        
        if not q:
            return jsonify({'status': 'error', 'message': 'Query required'}), 400
        
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
        
        # Use enhanced search if available and requested
        if search_mode == 'enhanced' and user_session_ready:
            try:
                result = await search_movies_multi_channel(q, l, p)
                return jsonify({
                    'status': 'success',
                    'query': q,
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
        cache_key = f"search:basic:{q}:{p}:{l}"
        if cache_manager is not None:
            cached = await cache_manager.get(cache_key)
            if cached:
                return jsonify(json.loads(cached))
        
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
            },
            'search_mode': 'basic'
        }
        
        # Cache the results
        if cache_manager is not None:
            await cache_manager.set(cache_key, json.dumps(response_data), expire_seconds=1800)
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/search/enhanced')
async def api_enhanced_search():
    """Force enhanced multi-channel search"""
    try:
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
        
        if cache_manager is not None:
            await cache_manager.clear_pattern("search:*")
            
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
            'message': 'Search cache cleared and stats reset',
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
        
        if not movies and files_col is not None:
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
        if files_col is None:
            return jsonify({'status': 'error', 'message': 'Database not ready'}), 503
        
        cursor = files_col.find({}).sort('date', -1).limit(20)
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

# Add enhanced search command to bot handlers
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
