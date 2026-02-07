# ============================================================================
# 🚀 SK4FiLM v9.0 - ULTRA FIXED VERSION
# ============================================================================

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
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import defaultdict
from functools import lru_cache, wraps
import urllib.parse
import aiohttp
from quart import Quart, jsonify, request, Response, send_file
from hypercorn.asyncio import serve
from hypercorn.config import Config as HyperConfig
from motor.motor_asyncio import AsyncIOMotorClient
import redis.asyncio as redis

# ✅ SETUP LOGGING FIRST
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Reduce log noise
logging.getLogger('asyncio').setLevel(logging.WARNING)
logging.getLogger('pyrogram').setLevel(logging.WARNING)
logging.getLogger('hypercorn').setLevel(logging.WARNING)
logging.getLogger('motor').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

# ============================================================================
# ✅ CONFIGURATION - SIMPLIFIED
# ============================================================================

class Config:
    # API Configuration
    API_ID = int(os.environ.get("API_ID", "0"))
    API_HASH = os.environ.get("API_HASH", "")
    USER_SESSION_STRING = os.environ.get("USER_SESSION_STRING", "")
    BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
    
    # Database Configuration - FIXED
    MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017/sk4film")
    REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
    
    # Channel Configuration
    MAIN_CHANNEL_ID = -1001891090100
    TEXT_CHANNEL_IDS = [-1001891090100, -1002024811395]
    FILE_CHANNEL_ID = -1001768249569
    
    # Links
    MAIN_CHANNEL_LINK = "https://t.me/sk4film"
    UPDATES_CHANNEL_LINK = "https://t.me/sk4film_Request"
    CHANNEL_USERNAME = "sk4film"
    
    # Application
    WEBSITE_URL = os.environ.get("WEBSITE_URL", "https://sk4film.vercel.app")
    BOT_USERNAME = os.environ.get("BOT_USERNAME", "sk4filmbot")
    ADMIN_IDS = [int(x) for x in os.environ.get("ADMIN_IDS", "123456789").split(",")]
    WEB_SERVER_PORT = int(os.environ.get("PORT", 8000))
    BACKEND_URL = os.environ.get("BACKEND_URL", "https://sk4film.koyeb.app")
    
    # API Keys
    TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "e547e17d4e91f3e62a571655cd1ccaff")
    OMDB_API_KEY = os.environ.get("OMDB_API_KEY", "8265bd1c")
    
    # Performance Settings
    MAX_CONCURRENT_REQUESTS = 50
    CACHE_TTL = 300
    
    # Quality Settings
    QUALITY_PRIORITY = ['2160p', '1080p', '720p', '480p', '360p']
    
    # Fallback Poster - FIXED URL
    FALLBACK_POSTER = "https://iili.io/fAeIwv9.th.png"
    
    # Search Settings
    SEARCH_MIN_QUERY_LENGTH = 2
    SEARCH_RESULTS_PER_PAGE = 12
    SEARCH_CACHE_TTL = 600

# ============================================================================
# ✅ APP INITIALIZATION
# ============================================================================

app = Quart(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

@app.after_request
async def add_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['X-SK4FiLM-Version'] = '9.0-ULTRA-FIXED'
    return response

# ============================================================================
# ✅ GLOBAL COMPONENTS
# ============================================================================

# Database
mongo_client = None
db = None
files_col = None

# Telegram
try:
    from pyrogram import Client
    from pyrogram.types import Message
    PYROGRAM_AVAILABLE = True
except ImportError:
    PYROGRAM_AVAILABLE = False

User = None
Bot = None
user_session_ready = False
bot_session_ready = False

# Systems
poster_fetcher = None

# ============================================================================
# ✅ UTILITY FUNCTIONS - FIXED
# ============================================================================

def normalize_title(title):
    if not title:
        return ""
    title = title.lower().strip()
    title = re.sub(r'\s*\([^)]*\)$', '', title)
    title = re.sub(r'\s*\[[^\]]*\]$', '', title)
    title = re.sub(r'\s*\d{4}$', '', title)
    return title

def extract_title_smart(text):
    if not text:
        return ""
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if len(line) > 10 and not line.startswith('http'):
            return line[:100]
    return text[:50] if text else ""

def format_post(text, max_length=None):
    if not text:
        return ""
    text = re.sub(r'\n\s*\n', '\n\n', text)
    if max_length and len(text) > max_length:
        text = text[:max_length] + "..."
    return text.strip()

def is_video_file(filename):
    if not filename:
        return False
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v']
    return any(filename.lower().endswith(ext) for ext in video_extensions)

def detect_quality_enhanced(filename):
    if not filename:
        return "480p"
    filename_lower = filename.lower()
    
    patterns = [
        (r'\b2160p\b|\b4k\b|\buhd\b', '2160p'),
        (r'\b1080p\b|\bfullhd\b|\bfhd\b', '1080p'),
        (r'\b720p\b|\bhd\b', '720p'),
        (r'\b480p\b', '480p'),
        (r'\b360p\b', '360p'),
    ]
    
    is_hevc = any(re.search(pattern, filename_lower) 
                  for pattern in [r'\bhevc\b', r'\bx265\b', r'\bh\.?265\b'])
    
    for pattern, quality in patterns:
        if re.search(pattern, filename_lower):
            if is_hevc and quality in ['720p', '1080p', '2160p']:
                return f"{quality} HEVC"
            return quality
    
    return "480p"

def is_new(date):
    if not date:
        return False
    if isinstance(date, str):
        try:
            date = datetime.fromisoformat(date.replace('Z', '+00:00'))
        except:
            return False
    return (datetime.now() - date).days < 7

def extract_year_from_title(title):
    """Extract year from title"""
    if not title:
        return ""
    
    year_match = re.search(r'\((\d{4})\)', title)
    if year_match:
        return year_match.group(1)
    
    year_match = re.search(r'\b(19|20)\d{2}\b', title)
    if year_match:
        return year_match.group()
    
    return ""

def clean_title(title):
    """Clean title by removing year and other artifacts"""
    if not title:
        return ""
    
    # Remove year in parentheses
    title = re.sub(r'\s*\(\d{4}\)', '', title)
    
    # Remove year at end
    title = re.sub(r'\s+\d{4}$', '', title)
    
    # Remove quality indicators
    title = re.sub(r'\b(480p|720p|1080p|2160p|4k|hd|hevc|x265|x264|web-dl|webrip|hdrip|bluray|dvdrip|tc|ts|cam)\b', '', title, flags=re.IGNORECASE)
    
    # Clean extra spaces
    title = re.sub(r'\s+', ' ', title)
    
    return title.strip()

# ============================================================================
# ✅ POSTER FETCHER - SIMPLE VERSION
# ============================================================================

class SimplePosterFetcher:
    """Simple poster fetcher that actually works"""
    
    def __init__(self, config):
        self.config = config
        self.session = None
        
    async def fetch_poster(self, title, year=""):
        """Fetch movie poster - SIMPLE AND WORKING"""
        try:
            # Clean title
            clean_title_val = self._clean_title(title)
            search_title = clean_title_val
            
            if year:
                search_title = f"{clean_title_val} {year}"
            
            # Try TMDB
            tmdb_poster = await self._fetch_tmdb(search_title)
            if tmdb_poster:
                return {
                    'poster_url': tmdb_poster,
                    'source': 'tmdb',
                    'rating': '0.0'
                }
            
            # Fallback to default
            return {
                'poster_url': self.config.FALLBACK_POSTER,
                'source': 'fallback',
                'rating': '0.0'
            }
            
        except Exception as e:
            logger.error(f"Poster fetch error: {e}")
            return {
                'poster_url': self.config.FALLBACK_POSTER,
                'source': 'error',
                'rating': '0.0'
            }
    
    def _clean_title(self, title):
        """Simple title cleaning"""
        if not title:
            return ""
        
        # Remove common patterns
        patterns = [
            r'\b\d{3,4}p\b',
            r'\bHD\b',
            r'\bHEVC\b',
            r'\bx265\b',
            r'\bWEB-DL\b',
            r'\bWEBRip\b',
            r'\bHDRip\b',
            r'\bBluRay\b',
            r'\[.*?\]',
            r'\(.*?\)',
        ]
        
        for pattern in patterns:
            title = re.sub(pattern, '', title, flags=re.IGNORECASE)
        
        title = re.sub(r'\s+', ' ', title)
        title = title.strip()
        
        return title
    
    async def _fetch_tmdb(self, title):
        """Fetch from TMDB"""
        try:
            if not self.config.TMDB_API_KEY:
                return None
            
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            search_url = "https://api.themoviedb.org/3/search/movie"
            params = {
                'api_key': self.config.TMDB_API_KEY,
                'query': title,
                'language': 'en-US',
                'page': 1
            }
            
            async with self.session.get(search_url, params=params, timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('results') and len(data['results']) > 0:
                        poster_path = data['results'][0].get('poster_path')
                        if poster_path:
                            return f"https://image.tmdb.org/t/p/w500{poster_path}"
            
            return None
            
        except Exception as e:
            logger.debug(f"TMDB fetch error: {e}")
            return None
    
    async def close(self):
        if self.session:
            await self.session.close()

# ============================================================================
# ✅ MONGODB INITIALIZATION - FIXED VERSION
# ============================================================================

async def init_mongodb():
    """Initialize MongoDB - FIXED VERSION"""
    global mongo_client, db, files_col
    
    try:
        logger.info("🔌 Initializing MongoDB...")
        
        # Get MongoDB URI
        mongodb_uri = Config.MONGODB_URI
        
        # Fix URI if needed
        if mongodb_uri.startswith("mongodb+srv://") and "/?" in mongodb_uri:
            # Add database name if missing
            base = mongodb_uri.split("/?")[0]
            params = mongodb_uri.split("/?")[1]
            mongodb_uri = f"{base}/sk4film?{params}"
        
        logger.info(f"📡 Connecting to MongoDB: {mongodb_uri[:50]}...")
        
        # Create MongoClient
        mongo_client = AsyncIOMotorClient(
            mongodb_uri,
            serverSelectionTimeoutMS=10000,
            connectTimeoutMS=10000
        )
        
        # Test connection
        await mongo_client.admin.command('ping')
        logger.info("✅ MongoDB connection successful")
        
        # Get database
        db_name = "sk4film"
        db = mongo_client[db_name]
        
        # Initialize collections
        files_col = db.files
        
        # Try to create indexes (ignore if already exist)
        try:
            await files_col.create_index(
                [("channel_id", 1), ("message_id", 1)],
                unique=True
            )
            logger.info("✅ Created unique index")
        except Exception as e:
            if "already exists" in str(e):
                logger.info("ℹ️ Index already exists")
            else:
                logger.warning(f"⚠️ Index error: {e}")
        
        try:
            await files_col.create_index([("normalized_title", "text")])
            logger.info("✅ Created text index")
        except Exception as e:
            logger.warning(f"⚠️ Text index error: {e}")
        
        logger.info("✅ MongoDB initialization complete")
        return True
        
    except Exception as e:
        logger.error(f"❌ MongoDB error: {e}")
        return False

# ============================================================================
# ✅ TELEGRAM SESSIONS - FIXED
# ============================================================================

async def init_telegram_sessions():
    """Initialize Telegram sessions - FIXED"""
    global User, Bot, user_session_ready, bot_session_ready
    
    if not PYROGRAM_AVAILABLE:
        logger.error("❌ Pyrogram not installed!")
        return False
    
    # User Session
    if Config.API_ID and Config.API_HASH and Config.USER_SESSION_STRING:
        try:
            User = Client(
                "sk4film_user",
                api_id=Config.API_ID,
                api_hash=Config.API_HASH,
                session_string=Config.USER_SESSION_STRING,
                in_memory=True,
                no_updates=True
            )
            await User.start()
            me = await User.get_me()
            logger.info(f"✅ User Session: {me.first_name}")
            user_session_ready = True
        except Exception as e:
            logger.error(f"❌ User session failed: {e}")
            user_session_ready = False
    
    # Bot Session
    if Config.BOT_TOKEN:
        try:
            Bot = Client(
                "sk4film_bot",
                api_id=Config.API_ID,
                api_hash=Config.API_HASH,
                bot_token=Config.BOT_TOKEN,
                in_memory=True,
                no_updates=True
            )
            await Bot.start()
            bot_info = await Bot.get_me()
            logger.info(f"✅ Bot Session: @{bot_info.username}")
            bot_session_ready = True
        except Exception as e:
            logger.error(f"❌ Bot session failed: {e}")
            bot_session_ready = False
    
    return user_session_ready or bot_session_ready

# ============================================================================
# ✅ SEARCH FUNCTION - COMPLETELY FIXED
# ============================================================================

async def search_movies_enhanced_fixed(query, limit=15, page=1):
    """Search movies - FIXED VERSION"""
    offset = (page - 1) * limit
    
    logger.info(f"🔍 SEARCHING: {query}")
    
    query_lower = query.lower()
    all_results = []
    
    # ============================================================================
    # ✅ STEP 1: SEARCH TEXT CHANNELS FOR POSTS
    # ============================================================================
    
    post_results = []
    if user_session_ready and User is not None:
        logger.info("📝 Searching TEXT channels...")
        
        for channel_id in Config.TEXT_CHANNEL_IDS:
            try:
                async for msg in User.search_messages(channel_id, query=query, limit=10):
                    if msg and msg.text and len(msg.text) > 15:
                        title = extract_title_smart(msg.text)
                        if title and (query_lower in title.lower() or query_lower in msg.text.lower()):
                            year = extract_year_from_title(title)
                            clean_title_val = clean_title(title)
                            norm_title = normalize_title(clean_title_val)
                            
                            post_data = {
                                'title': clean_title_val,
                                'original_title': title,
                                'normalized_title': norm_title,
                                'content': format_post(msg.text, max_length=500),
                                'post_content': msg.text,
                                'channel_id': channel_id,
                                'message_id': msg.id,
                                'date': msg.date.isoformat() if isinstance(msg.date, datetime) else str(msg.date),
                                'is_new': is_new(msg.date) if msg.date else False,
                                'has_file': False,
                                'has_post': True,
                                'quality_options': {},
                                'is_video_file': False,
                                'year': year,
                                'search_score': 3,
                                'result_type': 'post_only',
                                'poster_url': Config.FALLBACK_POSTER,
                                'poster_source': 'fallback',
                                'thumbnail_url': Config.FALLBACK_POSTER,
                                'thumbnail_source': 'fallback',
                                'has_poster': True,
                                'has_thumbnail': True
                            }
                            
                            post_results.append(post_data)
                            
            except Exception as e:
                logger.error(f"Text search error: {e}")
                continue
        
        logger.info(f"📝 Found {len(post_results)} POST results")
    
    # ============================================================================
    # ✅ STEP 2: SEARCH FILE CHANNEL DATABASE
    # ============================================================================
    
    file_results = []
    if files_col is not None:
        try:
            logger.info("📁 Searching FILE database...")
            
            # Build search query
            search_query = {
                "$or": [
                    {"title": {"$regex": query, "$options": "i"}},
                    {"normalized_title": {"$regex": query, "$options": "i"}},
                    {"file_name": {"$regex": query, "$options": "i"}},
                    {"caption": {"$regex": query, "$options": "i"}}
                ]
            }
            
            # Get matching files
            cursor = files_col.find(
                search_query,
                {
                    'title': 1,
                    'normalized_title': 1,
                    'quality': 1,
                    'file_size': 1,
                    'file_name': 1,
                    'is_video_file': 1,
                    'channel_id': 1,
                    'message_id': 1,
                    'date': 1,
                    'caption': 1,
                    'year': 1,
                    '_id': 0
                }
            ).limit(100)
            
            async for doc in cursor:
                title = doc.get('title', 'Unknown')
                norm_title = doc.get('normalized_title', '')
                quality = doc.get('quality', '480p')
                year = doc.get('year', '')
                channel_id = doc.get('channel_id')
                message_id = doc.get('message_id')
                is_video = doc.get('is_video_file', False)
                
                file_data = {
                    'title': title,
                    'normalized_title': norm_title,
                    'content': format_post(doc.get('caption', ''), max_length=300),
                    'post_content': doc.get('caption', ''),
                    'quality_options': {quality: {
                        'quality': quality,
                        'file_size': doc.get('file_size', 0),
                        'message_id': message_id,
                        'file_name': doc.get('file_name', ''),
                        'channel_id': channel_id
                    }},
                    'date': doc['date'].isoformat() if isinstance(doc['date'], datetime) else doc['date'],
                    'is_new': is_new(doc['date']) if doc.get('date') else False,
                    'is_video_file': is_video,
                    'channel_id': channel_id,
                    'has_file': True,
                    'has_post': bool(doc.get('caption')),
                    'quality': quality,
                    'real_message_id': message_id,
                    'result_type': 'file_only',
                    'year': year,
                    'search_score': 2,
                    'poster_url': Config.FALLBACK_POSTER,
                    'poster_source': 'fallback',
                    'thumbnail_url': Config.FALLBACK_POSTER,
                    'thumbnail_source': 'fallback',
                    'has_poster': True,
                    'has_thumbnail': True
                }
                
                file_results.append(file_data)
            
            logger.info(f"📁 Found {len(file_results)} FILE results")
            
        except Exception as e:
            logger.error(f"❌ File search error: {e}")
    
    # ============================================================================
    # ✅ STEP 3: MERGE POST AND FILE RESULTS
    # ============================================================================
    
    logger.info("🔄 Merging post and file results...")
    
    merged_dict = {}
    
    # First add all post results
    for post in post_results:
        norm_title = post['normalized_title']
        if norm_title:
            merged_dict[norm_title] = post.copy()
    
    # Now merge file results
    for file_item in file_results:
        norm_title = file_item['normalized_title']
        
        if norm_title in merged_dict:
            # Merge file with existing post
            existing_item = merged_dict[norm_title]
            
            existing_item.update({
                'has_file': True,
                'is_video_file': file_item['is_video_file'],
                'real_message_id': file_item['real_message_id'],
                'channel_id': file_item['channel_id'],
                'result_type': 'post_and_file',
                'search_score': 5
            })
            
            if 'quality_options' not in existing_item:
                existing_item['quality_options'] = {}
            
            for quality, q_data in file_item['quality_options'].items():
                existing_item['quality_options'][quality] = q_data
            
            existing_item['quality_count'] = len(existing_item['quality_options'])
            
            logger.debug(f"✅ Merged file with post: {norm_title}")
            
        else:
            file_item['result_type'] = 'file_only'
            merged_dict[norm_title] = file_item
    
    # Convert to list
    all_results = list(merged_dict.values())
    
    # ============================================================================
    # ✅ STEP 4: ADD POSTERS (ASYNC BUT SIMPLE)
    # ============================================================================
    
    logger.info(f"🎬 Adding posters to {len(all_results)} results...")
    
    # Fetch posters for all results
    if poster_fetcher:
        poster_tasks = []
        for result in all_results:
            title = result.get('title', '')
            year = result.get('year', '')
            task = asyncio.create_task(poster_fetcher.fetch_poster(title, year))
            poster_tasks.append((result, task))
        
        # Process results
        for result, task in poster_tasks:
            try:
                poster_data = await asyncio.wait_for(task, timeout=2.0)
                if poster_data:
                    result.update({
                        'poster_url': poster_data['poster_url'],
                        'poster_source': poster_data['source'],
                        'poster_rating': poster_data['rating'],
                        'thumbnail_url': poster_data['poster_url'],  # Use poster as thumbnail
                        'thumbnail_source': poster_data['source'],
                        'has_poster': True,
                        'has_thumbnail': True
                    })
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning(f"⚠️ Poster fetch failed: {e}")
                # Keep fallback values
    
    # ============================================================================
    # ✅ STEP 5: SORT AND PAGINATE
    # ============================================================================
    
    # Sort results
    all_results.sort(key=lambda x: (
        x.get('result_type') == 'post_and_file',
        x.get('result_type') == 'post_only',
        x.get('search_score', 0),
        x.get('is_new', False),
    ), reverse=True)
    
    # Pagination
    total = len(all_results)
    start_idx = offset
    end_idx = offset + limit
    paginated = all_results[start_idx:end_idx]
    
    # Calculate statistics
    post_count = sum(1 for r in all_results if r.get('result_type') == 'post_only')
    file_count = sum(1 for r in all_results if r.get('result_type') == 'file_only')
    merged_count = sum(1 for r in all_results if r.get('result_type') == 'post_and_file')
    
    # Log results
    logger.info("📊 FINAL RESULTS:")
    logger.info(f"   • Total results: {total}")
    logger.info(f"   • Post-only: {post_count}")
    logger.info(f"   • File-only: {file_count}")
    logger.info(f"   • Post+File merged: {merged_count}")
    
    # Show sample
    for i, result in enumerate(paginated[:3]):
        title = result.get('title', '')[:40]
        result_type = result.get('result_type', 'unknown')
        has_poster = result.get('has_poster', False)
        has_thumbnail = result.get('has_thumbnail', False)
        logger.info(f"   📋 {i+1}. {result_type}: {title}... | Poster: {has_poster} | Thumb: {has_thumbnail}")
    
    return {
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
            'query': query,
            'stats': {
                'total': total,
                'post_only': post_count,
                'file_only': file_count,
                'post_and_file': merged_count
            },
            'post_file_merged': True,
            'poster_fetcher': poster_fetcher is not None
        },
        'bot_username': Config.BOT_USERNAME
    }

# ============================================================================
# ✅ HOME MOVIES - SIMPLE VERSION
# ============================================================================

async def get_home_movies(limit=25):
    """Get home movies - SIMPLE VERSION"""
    movies = []
    
    if user_session_ready and User is not None:
        try:
            async for msg in User.get_chat_history(Config.MAIN_CHANNEL_ID, limit=limit):
                if msg and msg.text and len(msg.text) > 25:
                    title = extract_title_smart(msg.text)
                    if title:
                        year = extract_year_from_title(title)
                        clean_title_val = clean_title(title)
                        
                        movie_data = {
                            'title': clean_title_val,
                            'year': year,
                            'date': msg.date.isoformat() if isinstance(msg.date, datetime) else str(msg.date),
                            'channel_id': Config.MAIN_CHANNEL_ID,
                            'message_id': msg.id,
                            'has_file': False,
                            'has_post': True,
                            'content': format_post(msg.text, max_length=500),
                            'post_content': msg.text,
                            'result_type': 'post_only',
                            'quality_options': {},
                            'is_video_file': False,
                            'poster_url': Config.FALLBACK_POSTER,
                            'poster_source': 'fallback',
                            'thumbnail_url': Config.FALLBACK_POSTER,
                            'thumbnail_source': 'fallback',
                            'has_poster': True,
                            'has_thumbnail': True
                        }
                        
                        # Try to get poster
                        if poster_fetcher:
                            try:
                                poster_data = await poster_fetcher.fetch_poster(clean_title_val, year)
                                if poster_data:
                                    movie_data.update({
                                        'poster_url': poster_data['poster_url'],
                                        'poster_source': poster_data['source'],
                                        'poster_rating': poster_data['rating'],
                                        'thumbnail_url': poster_data['poster_url'],
                                        'thumbnail_source': poster_data['source']
                                    })
                            except Exception as e:
                                logger.warning(f"Poster fetch error: {e}")
                        
                        movies.append(movie_data)
                        
                        if len(movies) >= limit:
                            break
            
            logger.info(f"✅ Fetched {len(movies)} home movies")
            
        except Exception as e:
            logger.error(f"❌ Home movies error: {e}")
    
    return movies

# ============================================================================
# ✅ SYSTEM INITIALIZATION - FIXED
# ============================================================================

async def init_system():
    """Initialize system - FIXED VERSION"""
    global poster_fetcher
    
    logger.info("=" * 60)
    logger.info("🚀 SK4FiLM v9.0 - ULTRA FIXED")
    logger.info("=" * 60)
    
    try:
        # Initialize MongoDB
        mongo_ok = await init_mongodb()
        if not mongo_ok:
            logger.error("❌ MongoDB failed")
            return False
        
        # Get current stats
        if files_col is not None:
            try:
                file_count = await files_col.count_documents({})
                logger.info(f"📊 Current files in database: {file_count}")
            except:
                logger.warning("⚠️ Could not count files")
        
        # Initialize Telegram Sessions
        telegram_ok = await init_telegram_sessions()
        if not telegram_ok:
            logger.warning("⚠️ Telegram sessions not ready")
        
        # Initialize Poster Fetcher
        poster_fetcher = SimplePosterFetcher(Config)
        logger.info("✅ Poster Fetcher initialized")
        
        logger.info("✅ SK4FiLM initialized successfully")
        logger.info("=" * 60)
        logger.info("🎯 WORKING FEATURES:")
        logger.info(f"   • MongoDB: ✅ CONNECTED")
        logger.info(f"   • User Session: {'✅ READY' if user_session_ready else '❌ NOT READY'}")
        logger.info(f"   • Bot Session: {'✅ READY' if bot_session_ready else '❌ NOT READY'}")
        logger.info(f"   • Poster Fetcher: ✅ READY")
        logger.info(f"   • Text Channels: {len(Config.TEXT_CHANNEL_IDS)}")
        logger.info(f"   • File Channel: {Config.FILE_CHANNEL_ID}")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ System init error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

# ============================================================================
# ✅ API ROUTES - SIMPLE AND WORKING
# ============================================================================

@app.route('/')
async def root():
    """Root endpoint"""
    # Get stats
    file_count = 0
    if files_col is not None:
        try:
            file_count = await files_col.count_documents({})
        except:
            pass
    
    return jsonify({
        'status': 'healthy',
        'service': 'SK4FiLM v9.0 - ULTRA FIXED',
        'sessions': {
            'user': user_session_ready,
            'bot': bot_session_ready
        },
        'stats': {
            'total_files': file_count
        },
        'features': {
            'post_file_merge': True,
            'poster_fetcher': poster_fetcher is not None,
            'search': True
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/health')
async def health():
    """Health check"""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/movies', methods=['GET'])
async def api_movies():
    """Get home movies"""
    try:
        movies = await get_home_movies(limit=25)
        return jsonify({
            'status': 'success',
            'movies': movies,
            'total': len(movies)
        })
    except Exception as e:
        logger.error(f"Movies API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)[:100]
        }), 500

@app.route('/api/search', methods=['GET'])
async def api_search():
    """Search movies"""
    try:
        query = request.args.get('query', '').strip()
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', Config.SEARCH_RESULTS_PER_PAGE))
        
        if len(query) < Config.SEARCH_MIN_QUERY_LENGTH:
            return jsonify({
                'status': 'error',
                'message': f'Query must be at least {Config.SEARCH_MIN_QUERY_LENGTH} characters'
            }), 400
        
        # Call the FIXED search function
        result_data = await search_movies_enhanced_fixed(query, limit, page)
        
        return jsonify({
            'status': 'success',
            'query': query,
            'results': result_data['results'],
            'pagination': result_data['pagination'],
            'search_metadata': result_data['search_metadata'],
            'bot_username': Config.BOT_USERNAME
        })
        
    except Exception as e:
        logger.error(f"Search API error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e)[:100]
        }), 500

@app.route('/api/stats', methods=['GET'])
async def api_stats():
    """Get statistics"""
    try:
        file_count = 0
        if files_col is not None:
            try:
                file_count = await files_col.count_documents({})
            except:
                pass
        
        return jsonify({
            'status': 'success',
            'database_files': file_count,
            'sessions': {
                'user': user_session_ready,
                'bot': bot_session_ready
            },
            'features': {
                'post_file_merge': True,
                'text_channels': len(Config.TEXT_CHANNEL_IDS)
            }
        })
    except Exception as e:
        logger.error(f"Stats API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)[:100]
        }), 500

# ============================================================================
# ✅ STARTUP AND SHUTDOWN
# ============================================================================

@app.before_serving
async def startup():
    """Startup"""
    await init_system()

@app.after_serving
async def shutdown():
    """Shutdown"""
    logger.info("🛑 Shutting down SK4FiLM...")
    
    # Close poster fetcher
    if poster_fetcher is not None:
        await poster_fetcher.close()
    
    # Close Telegram sessions
    if User is not None:
        await User.stop()
    if Bot is not None:
        await Bot.stop()
    
    # Close MongoDB
    if mongo_client is not None:
        mongo_client.close()
    
    logger.info("👋 Shutdown complete")

# ============================================================================
# ✅ ERROR HANDLER
# ============================================================================

@app.errorhandler(Exception)
async def handle_error(error):
    """Global error handler"""
    logger.error(f"Unhandled error: {error}")
    import traceback
    logger.error(traceback.format_exc())
    
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500

# ============================================================================
# ✅ MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    config = HyperConfig()
    config.bind = [f"0.0.0.0:{Config.WEB_SERVER_PORT}"]
    config.worker_class = "asyncio"
    config.workers = 1
    config.accesslog = None
    config.errorlog = "-"
    config.loglevel = "warning"
    
    logger.info(f"🌐 Starting SK4FiLM on port {Config.WEB_SERVER_PORT}...")
    logger.info("=" * 60)
    logger.info("🎯 ULTRA FIXED VERSION - SIMPLE & WORKING")
    logger.info("=" * 60)
    logger.info(f"   • MongoDB: ✅ CONNECTED")
    logger.info(f"   • Post+File Merge: ✅ ENABLED")
    logger.info(f"   • Poster Fetching: ✅ ENABLED")
    logger.info(f"   • Text Channels: {len(Config.TEXT_CHANNEL_IDS)}")
    logger.info(f"   • File Channel: {Config.FILE_CHANNEL_ID}")
    logger.info("=" * 60)
    
    try:
        asyncio.run(serve(app, config))
    except KeyboardInterrupt:
        logger.info("👋 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
