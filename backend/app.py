# ============================================================================
# 🚀 SK4FiLM v10.0 - COMPLETE API WITH ALL FEATURES
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
from quart import Quart, jsonify, request, Response
from hypercorn.asyncio import serve
from hypercorn.config import Config as HyperConfig
from motor.motor_asyncio import AsyncIOMotorClient
import redis.asyncio as redis

# ============================================================================
# ✅ LOGGING SETUP
# ============================================================================

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

# ============================================================================
# ✅ CONFIGURATION
# ============================================================================

class Config:
    # API Configuration
    API_ID = int(os.environ.get("API_ID", "0"))
    API_HASH = os.environ.get("API_HASH", "")
    USER_SESSION_STRING = os.environ.get("USER_SESSION_STRING", "")
    BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
    
    # Database Configuration
    MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
    REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
    
    # Channel Configuration
    MAIN_CHANNEL_ID = -1001891090100
    TEXT_CHANNEL_IDS = [-1001891090100, -1002024811395]  # ✅ Both text channels
    FILE_CHANNEL_ID = -1001768249569  # ✅ FILE CHANNEL
    
    # Application
    WEBSITE_URL = os.environ.get("WEBSITE_URL", "https://sk4film.vercel.app")
    BOT_USERNAME = os.environ.get("BOT_USERNAME", "sk4filmbot")
    WEB_SERVER_PORT = int(os.environ.get("PORT", 8000))
    
    # API Keys for POSTERS
    TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "e547e17d4e91f3e62a571655cd1ccaff")
    OMDB_API_KEY = os.environ.get("OMDB_API_KEY", "8265bd1c")
    
    # Performance Settings
    CACHE_TTL = int(os.environ.get("CACHE_TTL", "300"))
    REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "10"))
    
    # Quality Settings
    QUALITY_PRIORITY = ['2160p', '1080p', '720p', '480p', '360p']
    HEVC_VARIANTS = ['720p HEVC', '1080p HEVC', '2160p HEVC']
    
    # Fallback Poster
    FALLBACK_POSTER = "https://iili.io/fAeIwv9.th.png"
    
    # 🔥 SEARCH SETTINGS
    SEARCH_MIN_QUERY_LENGTH = 2
    SEARCH_RESULTS_PER_PAGE = 12
    SEARCH_CACHE_TTL = 600  # 10 minutes
    
    # 🔥 FEATURE SETTINGS
    SEPARATE_POSTS_FOR_SAME_TITLE = True  # ✅ Dono text channels ke alag alag posts
    ENABLE_STREAMING = True
    ENABLE_DIRECT_DOWNLOAD = True

# ============================================================================
# ✅ QUART APP SETUP
# ============================================================================

app = Quart(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# CORS headers
@app.after_request
async def add_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['X-SK4FiLM-Version'] = '10.0-COMPLETE'
    return response

# ============================================================================
# ✅ GLOBAL COMPONENTS
# ============================================================================

# Database
mongo_client = None
db = None
files_col = None

# Telegram Sessions
try:
    from pyrogram import Client
    PYROGRAM_AVAILABLE = True
    
    User = None
    Bot = None
    user_session_ready = False
    bot_session_ready = False
except ImportError:
    PYROGRAM_AVAILABLE = False
    User = None
    Bot = None

# ============================================================================
# ✅ HELPER FUNCTIONS
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

def format_size(size):
    if not size:
        return "Unknown"
    if size < 1024:
        return f"{size} B"
    elif size < 1024*1024:
        return f"{size/1024:.1f} KB"
    elif size < 1024*1024*1024:
        return f"{size/1024/1024:.1f} MB"
    else:
        return f"{size/1024/1024/1024:.2f} GB"

def is_video_file(filename):
    if not filename:
        return False
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v']
    return any(filename.lower().endswith(ext) for ext in video_extensions)

def format_post(text, max_length=None):
    if not text:
        return ""
    text = re.sub(r'\n\s*\n', '\n\n', text)
    if max_length and len(text) > max_length:
        text = text[:max_length] + "..."
    return text.strip()

def is_new(date):
    if not date:
        return False
    if isinstance(date, str):
        try:
            date = datetime.fromisoformat(date.replace('Z', '+00:00'))
        except:
            return False
    return (datetime.now() - date).days < 7

QUALITY_PATTERNS = [
    (r'\b2160p\b|\b4k\b|\buhd\b', '2160p'),
    (r'\b1080p\b|\bfullhd\b|\bfhd\b', '1080p'),
    (r'\b720p\b|\bhd\b', '720p'),
    (r'\b480p\b', '480p'),
    (r'\b360p\b', '360p'),
]

HEVC_PATTERNS = [
    r'\bhevc\b',
    r'\bx265\b',
    r'\bh\.?265\b',
]

def detect_quality_enhanced(filename):
    """Enhanced quality detection with HEVC support"""
    if not filename:
        return "480p"
    
    filename_lower = filename.lower()
    
    # First check for HEVC variants
    is_hevc = any(re.search(pattern, filename_lower) for pattern in HEVC_PATTERNS)
    
    # Then check quality
    for pattern, quality in QUALITY_PATTERNS:
        if re.search(pattern, filename_lower):
            if is_hevc and quality in ['720p', '1080p', '2160p']:
                return f"{quality} HEVC"
            return quality
    
    return "480p"

def extract_quality_info(filename):
    """Extract detailed quality info"""
    quality = detect_quality_enhanced(filename)
    
    # Parse quality components
    base_quality = quality
    is_hevc = 'HEVC' in quality
    
    if is_hevc:
        base_quality = quality.replace(' HEVC', '')
    
    return {
        'full': quality,
        'base': base_quality,
        'is_hevc': is_hevc,
        'priority': Config.QUALITY_PRIORITY.index(base_quality) if base_quality in Config.QUALITY_PRIORITY else 999
    }

# ============================================================================
# ✅ QUALITY MERGER
# ============================================================================

class QualityMerger:
    """Merge multiple qualities for same title"""
    
    @staticmethod
    def merge_quality_options(quality_options_dict):
        """Merge quality options from multiple sources"""
        if not quality_options_dict:
            return {}
        
        merged = {}
        
        # Sort by priority
        for quality, option in quality_options_dict.items():
            base_quality = quality.replace(' HEVC', '')
            
            if base_quality not in merged:
                merged[base_quality] = {
                    'qualities': [],
                    'best_option': None,
                    'total_size': 0,
                    'file_count': 0
                }
            
            # Add quality variant
            merged[base_quality]['qualities'].append({
                'full_quality': quality,
                'is_hevc': 'HEVC' in quality,
                'file_id': option.get('file_id'),
                'file_size': option.get('file_size', 0),
                'file_name': option.get('file_name', ''),
                'is_video': option.get('is_video', False),
                'channel_id': option.get('channel_id'),
                'message_id': option.get('message_id'),
                'real_message_id': option.get('real_message_id'),
                'thumbnail_url': option.get('thumbnail_url'),
                'telegram_file_id': option.get('telegram_file_id')
            })
            
            merged[base_quality]['total_size'] += option.get('file_size', 0)
            merged[base_quality]['file_count'] += 1
            
            # Set best option (highest quality, smallest size)
            if merged[base_quality]['best_option'] is None:
                merged[base_quality]['best_option'] = quality
            else:
                current_priority = Config.QUALITY_PRIORITY.index(base_quality) if base_quality in Config.QUALITY_PRIORITY else 999
                best_base = merged[base_quality]['best_option'].replace(' HEVC', '')
                best_priority = Config.QUALITY_PRIORITY.index(best_base) if best_base in Config.QUALITY_PRIORITY else 999
                
                if current_priority < best_priority:
                    merged[base_quality]['best_option'] = quality
        
        # Sort by quality priority
        sorted_merged = {}
        for quality in Config.QUALITY_PRIORITY:
            if quality in merged:
                sorted_merged[quality] = merged[quality]
        
        # Add any remaining qualities
        for quality in merged:
            if quality not in sorted_merged:
                sorted_merged[quality] = merged[quality]
        
        return sorted_merged

# ============================================================================
# ✅ ASYNC CACHE DECORATOR
# ============================================================================

def async_cache_with_ttl(maxsize=128, ttl=300):
    """Async cache decorator with TTL"""
    cache = {}
    cache_lock = asyncio.Lock()
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            key_parts = [func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
            key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
            
            now = time.time()
            
            async with cache_lock:
                if key in cache:
                    value, timestamp = cache[key]
                    if now - timestamp < ttl:
                        return value
            
            result = await func(*args, **kwargs)
            
            async with cache_lock:
                cache[key] = (result, now)
                if len(cache) > maxsize:
                    oldest_key = min(cache.keys(), key=lambda k: cache[k][1])
                    del cache[oldest_key]
            
            return result
        return wrapper
    return decorator

# ============================================================================
# ✅ POSTER FETCHING
# ============================================================================

async def get_poster_for_movie(title: str, year: str = "", quality: str = "") -> Dict[str, Any]:
    """Get poster for movie"""
    try:
        # Try TMDB first
        if Config.TMDB_API_KEY:
            async with aiohttp.ClientSession() as session:
                url = f"https://api.themoviedb.org/3/search/movie?api_key={Config.TMDB_API_KEY}&query={urllib.parse.quote(title)}"
                async with session.get(url, timeout=3) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('results') and data['results'][0] and data['results'][0].get('poster_path'):
                            poster_url = f"https://image.tmdb.org/t/p/w500{data['results'][0]['poster_path']}"
                            return {
                                'poster_url': poster_url,
                                'source': 'tmdb',
                                'rating': str(data['results'][0].get('vote_average', '0.0')),
                                'year': year,
                                'title': title,
                                'quality': quality or 'unknown'
                            }
        
        # Try OMDB as fallback
        if Config.OMDB_API_KEY:
            async with aiohttp.ClientSession() as session:
                url = f"https://www.omdbapi.com/?t={urllib.parse.quote(title)}&apikey={Config.OMDB_API_KEY}"
                async with session.get(url, timeout=3) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('Poster') and data['Poster'] != 'N/A':
                            return {
                                'poster_url': data['Poster'],
                                'source': 'omdb',
                                'rating': data.get('imdbRating', '0.0'),
                                'year': data.get('Year', year),
                                'title': title,
                                'quality': quality or 'unknown'
                            }
    except Exception as e:
        logger.warning(f"Poster fetch error for {title}: {e}")
    
    # Fallback
    return {
        'poster_url': Config.FALLBACK_POSTER,
        'source': 'fallback',
        'rating': '0.0',
        'year': year,
        'title': title,
        'quality': quality or 'unknown'
    }

async def get_posters_for_movies_batch(movies: List[Dict]) -> List[Dict]:
    """Get posters for multiple movies in batch"""
    if not movies:
        return []
    
    results = []
    
    # Create tasks for all movies
    tasks = []
    for movie in movies:
        title = movie.get('title', '')
        year = movie.get('year', '')
        quality = movie.get('quality', '')
        
        task = asyncio.create_task(get_poster_for_movie(title, year, quality))
        tasks.append((movie, task))
    
    # Process results
    for movie, task in tasks:
        try:
            poster_data = await task
            
            # Update movie with poster data
            movie_with_poster = movie.copy()
            movie_with_poster.update({
                'poster_url': poster_data['poster_url'],
                'poster_source': poster_data['source'],
                'poster_rating': poster_data['rating'],
                'thumbnail': poster_data['poster_url'],
                'thumbnail_source': poster_data['source'],
                'has_poster': True
            })
            
            results.append(movie_with_poster)
            
        except Exception as e:
            logger.warning(f"Poster error for {movie.get('title')}: {e}")
            
            # Add movie with fallback
            movie_with_fallback = movie.copy()
            movie_with_fallback.update({
                'poster_url': Config.FALLBACK_POSTER,
                'poster_source': 'fallback',
                'poster_rating': '0.0',
                'thumbnail': Config.FALLBACK_POSTER,
                'thumbnail_source': 'fallback',
                'has_poster': True
            })
            
            results.append(movie_with_fallback)
    
    return results

# ============================================================================
# ✅ CHANNEL NAME HELPER
# ============================================================================

def channel_name_cached(cid):
    """Get channel name"""
    if cid == Config.MAIN_CHANNEL_ID:
        return "SK4FiLM Main"
    elif cid == Config.FILE_CHANNEL_ID:
        return "SK4FiLM Files"
    elif cid == -1002024811395:
        return "SK4FiLM Updates"
    else:
        return f"Channel {cid}"

# ============================================================================
# ✅ MAIN SEARCH FUNCTION - DONO TEXT CHANNELS KE ALAG ALAG POSTS
# ============================================================================

@async_cache_with_ttl(maxsize=500, ttl=Config.SEARCH_CACHE_TTL)
async def search_movies_multi_channel_separate_posts(query, limit=15, page=1):
    """
    ✅ DONO TEXT CHANNELS KE ALAG ALAG POSTS
    ✅ SAME TITLE SE FILE RESULTS POST KE SAATH
    ✅ SIRF FILE RESULTS -> THUMBNAIL + TITLE + STREAM NOW
    """
    offset = (page - 1) * limit
    
    logger.info(f"🔍 SEARCHING: {query} - Page: {page}")
    
    query_lower = query.lower()
    all_results = []  # Store all results
    posts_by_channel = defaultdict(list)  # Posts grouped by channel
    files_by_title = defaultdict(list)  # Files grouped by normalized title
    
    # ============================================================================
    # ✅ 1. SEARCH DONO TEXT CHANNELS - ALAG ALAG POSTS
    # ============================================================================
    if user_session_ready and User is not None:
        async def search_text_channel(channel_id):
            channel_posts = []
            try:
                cname = channel_name_cached(channel_id)
                logger.info(f"📝 Searching in channel: {cname} ({channel_id})")
                
                async for msg in User.search_messages(channel_id, query=query, limit=30):
                    if msg is not None and msg.text and len(msg.text) > 15:
                        title = extract_title_smart(msg.text)
                        
                        # Check if query matches
                        if title and (query_lower in title.lower() or query_lower in msg.text.lower()):
                            norm_title = normalize_title(title)
                            
                            # Get year
                            year_match = re.search(r'\b(19|20)\d{2}\b', title)
                            year = year_match.group() if year_match else ""
                            
                            # Create unique ID for post
                            post_id = f"post_{channel_id}_{msg.id}"
                            
                            # ✅ Create movie data - ALWAYS SHOW SEPARATELY
                            movie_data = {
                                'id': post_id,
                                'title': title,
                                'original_title': title,
                                'normalized_title': norm_title,
                                'content': format_post(msg.text, max_length=1000),
                                'post_content': msg.text,
                                'channel': cname,
                                'channel_id': channel_id,
                                'message_id': msg.id,
                                'date': msg.date.isoformat() if isinstance(msg.date, datetime) else str(msg.date),
                                'is_new': is_new(msg.date) if msg.date else False,
                                'has_file': False,  # Initially no file
                                'has_post': True,
                                'quality_options': {},
                                'is_video_file': False,
                                'year': year,
                                'search_score': 3 if query_lower in title.lower() else 2,
                                'result_type': 'post',
                                'post_channel_name': cname,
                                'unique_key': post_id,
                                'show_separate': True,  # ✅ ALWAYS SHOW SEPARATELY
                                'has_thumbnail': False,
                                'thumbnail_url': None,
                                'stream_now_available': False
                            }
                            
                            channel_posts.append(movie_data)
                            
                            logger.debug(f"✅ Found post in {cname}: {title[:50]}...")
                            
            except Exception as e:
                logger.error(f"❌ Text search error in {channel_id}: {e}")
            return channel_posts
        
        # Search both text channels concurrently
        tasks = [search_text_channel(channel_id) for channel_id in Config.TEXT_CHANNEL_IDS]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results from each channel
        for channel_idx, result in enumerate(results):
            if isinstance(result, list):
                channel_id = Config.TEXT_CHANNEL_IDS[channel_idx]
                posts_by_channel[channel_id] = result
                all_results.extend(result)
        
        logger.info(f"📝 Found {len(all_results)} posts in text channels")
    
    # ============================================================================
    # ✅ 2. SEARCH FILE CHANNEL - GROUP FILES BY TITLE
    # ============================================================================
    if files_col is not None:
        try:
            logger.info(f"🔍 Searching FILE CHANNEL database for: {query}")
            
            # Build search query
            search_query = {
                "$or": [
                    {"title": {"$regex": query, "$options": "i"}},
                    {"normalized_title": {"$regex": query, "$options": "i"}},
                    {"file_name": {"$regex": query, "$options": "i"}},
                    {"caption": {"$regex": query, "$options": "i"}}
                ],
                "status": "active",
                "is_duplicate": False
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
                    'real_message_id': 1,
                    'date': 1,
                    'caption': 1,
                    'file_id': 1,
                    'telegram_file_id': 1,
                    'thumbnail_url': 1,
                    'thumbnail_extracted': 1,
                    'year': 1,
                    '_id': 0
                }
            ).limit(1000)
            
            file_count = 0
            
            async for doc in cursor:
                file_count += 1
                try:
                    title = doc.get('title', 'Unknown')
                    norm_title = normalize_title(title)
                    
                    # Extract quality info
                    quality_info = extract_quality_info(doc.get('file_name', ''))
                    quality = quality_info['full']
                    
                    # Get thumbnail URL
                    thumbnail_url = doc.get('thumbnail_url')
                    
                    # Get REAL message ID
                    real_msg_id = doc.get('real_message_id') or doc.get('message_id')
                    
                    # Create quality option
                    quality_option = {
                        'file_id': f"{doc.get('channel_id', Config.FILE_CHANNEL_ID)}_{real_msg_id}_{quality}",
                        'file_size': doc.get('file_size', 0),
                        'file_name': doc.get('file_name', ''),
                        'is_video': doc.get('is_video_file', False),
                        'channel_id': doc.get('channel_id'),
                        'message_id': real_msg_id,
                        'real_message_id': real_msg_id,
                        'quality': quality,
                        'base_quality': quality_info['base'],
                        'is_hevc': quality_info['is_hevc'],
                        'priority': quality_info['priority'],
                        'thumbnail_url': thumbnail_url,
                        'has_thumbnail': thumbnail_url is not None,
                        'date': doc.get('date'),
                        'telegram_file_id': doc.get('telegram_file_id'),
                        'file_size_formatted': format_size(doc.get('file_size', 0))
                    }
                    
                    # Group files by normalized title
                    files_by_title[norm_title].append({
                        'title': title,
                        'quality_option': quality_option,
                        'year': doc.get('year', ''),
                        'caption': doc.get('caption', ''),
                        'date': doc.get('date'),
                        'has_thumbnail': thumbnail_url is not None,
                        'thumbnail_url': thumbnail_url,
                        'norm_title': norm_title
                    })
                    
                except Exception as e:
                    logger.error(f"File processing error: {e}")
                    continue
            
            logger.info(f"✅ Found {file_count} files in database for query: {query}")
            logger.info(f"📦 Grouped into {len(files_by_title)} unique titles")
            
        except Exception as e:
            logger.error(f"❌ File search error: {e}")
    
    # ============================================================================
    # ✅ 3. COMBINE POSTS AND FILES - SAME TITLE SE FILES POSTS KE SAATH
    # ============================================================================
    final_results = []
    
    # First, process all posts (dono channels ke alag alag)
    processed_titles = set()
    
    for channel_id, posts in posts_by_channel.items():
        for post in posts:
            norm_title = post['normalized_title']
            processed_titles.add(norm_title)
            
            # ✅ Check if we have files with same title
            if norm_title in files_by_title:
                files_for_title = files_by_title[norm_title]
                
                # Merge quality options from all files
                quality_options = {}
                for file_data in files_for_title:
                    quality_opt = file_data['quality_option']
                    quality = quality_opt['quality']
                    quality_options[quality] = quality_opt
                
                # Merge quality options using QualityMerger
                if quality_options:
                    merged_options = QualityMerger.merge_quality_options(quality_options)
                    
                    # ✅ Update post with file information
                    post.update({
                        'has_file': True,
                        'stream_now_available': True,  # ✅ Stream Now available
                        'quality_options': merged_options,
                        'quality_list': list(merged_options.keys()),
                        'quality_count': len(merged_options),
                        'best_quality': list(merged_options.keys())[0] if merged_options else '',
                        'total_size': sum(opt.get('total_size', 0) for opt in merged_options.values()),
                        'file_count': sum(opt.get('file_count', 0) for opt in merged_options.values()),
                        'is_video_file': any(file_data['quality_option']['is_video'] for file_data in files_for_title),
                        'has_thumbnail': any(file_data.get('has_thumbnail') for file_data in files_for_title),
                        'thumbnail_url': next((file_data['thumbnail_url'] for file_data in files_for_title if file_data.get('thumbnail_url')), None),
                        'thumbnail': next((file_data['thumbnail_url'] for file_data in files_for_title if file_data.get('thumbnail_url')), Config.FALLBACK_POSTER)
                    })
                
                # Remove from files_by_title to avoid duplication
                del files_by_title[norm_title]
            
            final_results.append(post)
    
    # ============================================================================
    # ✅ 4. ADD FILES WITHOUT POSTS - SIRF FILE RESULTS
    # ============================================================================
    for norm_title, files_list in files_by_title.items():
        if files_list:
            first_file = files_list[0]
            
            # Merge quality options
            quality_options = {}
            for file_data in files_list:
                quality_opt = file_data['quality_option']
                quality = quality_opt['quality']
                quality_options[quality] = quality_opt
            
            merged_options = QualityMerger.merge_quality_options(quality_options)
            
            # ✅ Create file-only result - THUMBNAIL + TITLE + STREAM NOW
            file_result = {
                'id': f"file_{norm_title}_{hashlib.md5(norm_title.encode()).hexdigest()[:8]}",
                'title': first_file['title'],
                'original_title': first_file['title'],
                'normalized_title': norm_title,
                'content': first_file.get('caption', ''),
                'post_content': first_file.get('caption', ''),
                'channel': channel_name_cached(Config.FILE_CHANNEL_ID),
                'channel_id': Config.FILE_CHANNEL_ID,
                'date': first_file.get('date'),
                'is_new': is_new(first_file.get('date')) if first_file.get('date') else False,
                'has_file': True,
                'has_post': False,
                'stream_now_available': True,  # ✅ Always show Stream Now for files
                'quality_options': merged_options,
                'quality_list': list(merged_options.keys()),
                'quality_count': len(merged_options),
                'best_quality': list(merged_options.keys())[0] if merged_options else '',
                'total_size': sum(opt.get('total_size', 0) for opt in merged_options.values()),
                'file_count': sum(opt.get('file_count', 0) for opt in merged_options.values()),
                'is_video_file': any(file_data['quality_option']['is_video'] for file_data in files_list),
                'has_thumbnail': any(file_data.get('has_thumbnail') for file_data in files_list),
                'thumbnail_url': next((file_data['thumbnail_url'] for file_data in files_list if file_data.get('thumbnail_url')), None),
                'thumbnail': next((file_data['thumbnail_url'] for file_data in files_list if file_data.get('thumbnail_url')), Config.FALLBACK_POSTER),
                'year': first_file.get('year', ''),
                'search_score': 2,
                'result_type': 'file_only',
                'unique_key': f"file_{norm_title}",
                'show_separate': True  # ✅ Always show separately
            }
            
            final_results.append(file_result)
    
    # ============================================================================
    # ✅ 5. FETCH POSTERS FOR ALL RESULTS
    # ============================================================================
    if final_results:
        logger.info(f"🎬 Fetching posters for {len(final_results)} results...")
        
        # Fetch posters in batch
        movies_with_posters = await get_posters_for_movies_batch(final_results)
        
        # Update results with poster data
        for i, result in enumerate(movies_with_posters):
            if i < len(final_results):
                # Use fetched poster or existing thumbnail
                poster_url = result.get('poster_url', Config.FALLBACK_POSTER)
                thumbnail_url = final_results[i].get('thumbnail_url') or poster_url
                
                final_results[i].update({
                    'poster_url': poster_url,
                    'poster_source': result.get('poster_source', 'fallback'),
                    'poster_rating': result.get('poster_rating', '0.0'),
                    'thumbnail': thumbnail_url,
                    'thumbnail_source': result.get('thumbnail_source', 'fallback'),
                    'has_poster': True,
                    'has_thumbnail': True
                })
    
    # ============================================================================
    # ✅ 6. SORT AND PAGINATE
    # ============================================================================
    # Enhanced sorting:
    # 1. Results with posts + files (highest priority)
    # 2. Results with stream_now_available
    # 3. Results with multiple qualities
    # 4. New results first
    # 5. Higher search score
    final_results.sort(key=lambda x: (
        x.get('has_post', False) and x.get('has_file', False),  # Posts with files first
        x.get('stream_now_available', False),  # Stream Now available
        x.get('quality_count', 0),  # More qualities first
        x.get('is_new', False),  # New first
        x.get('search_score', 0),  # Search relevance
        x.get('date', '') if isinstance(x.get('date'), str) else ''  # Recent first
    ), reverse=True)
    
    total = len(final_results)
    paginated = final_results[offset:offset + limit]
    
    # Statistics
    stats = {
        'total': total,
        'with_posts': sum(1 for r in final_results if r.get('has_post', False)),
        'with_files': sum(1 for r in final_results if r.get('has_file', False)),
        'both': sum(1 for r in final_results if r.get('has_post', False) and r.get('has_file', False)),
        'stream_now_available': sum(1 for r in final_results if r.get('stream_now_available', False)),
        'video_files': sum(1 for r in final_results if r.get('is_video_file', False)),
        'multi_quality': sum(1 for r in final_results if r.get('quality_count', 0) > 1),
        'separate_posts': Config.SEPARATE_POSTS_FOR_SAME_TITLE
    }
    
    # Log detailed info
    logger.info(f"📊 FINAL RESULTS STATS:")
    logger.info(f"   • Total results: {total}")
    logger.info(f"   • Separate posts: {stats['separate_posts']}")
    logger.info(f"   • Posts with files: {stats['both']}")
    logger.info(f"   • Stream Now available: {stats['stream_now_available']}")
    logger.info(f"   • Multi-quality files: {stats['multi_quality']}")
    
    # Show examples
    for i, result in enumerate(paginated[:3]):
        source = f"{result.get('channel', 'Unknown')}"
        if result.get('has_file'):
            source += f" + {result.get('quality_count', 0)} qualities"
        logger.info(f"   📦 Example {i+1}: {result.get('title', '')[:40]}... - {source}")
    
    # Final data structure
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
            'query': query,
            'stats': stats,
            'separate_posts': Config.SEPARATE_POSTS_FOR_SAME_TITLE,
            'streaming_support': Config.ENABLE_STREAMING,
            'quality_merging': True,
            'real_message_ids': True,
            'feature': 'enhanced_search_separate_posts_with_streaming'
        },
        'bot_username': Config.BOT_USERNAME,
        'file_channel_id': Config.FILE_CHANNEL_ID
    }
    
    logger.info(f"✅ Search complete: {len(paginated)} results (page {page})")
    
    return result_data

# ============================================================================
# ✅ GET HOME MOVIES
# ============================================================================

@async_cache_with_ttl(maxsize=1, ttl=60)
async def get_home_movies(limit=25):
    """Get home movies for homepage"""
    try:
        if User is None or not user_session_ready:
            return []
        
        movies = []
        seen_titles = set()
        
        logger.info(f"🎬 Fetching home movies ({limit})...")
        
        async for msg in User.get_chat_history(Config.MAIN_CHANNEL_ID, limit=30):
            if msg is not None and msg.text and len(msg.text) > 25:
                title = extract_title_smart(msg.text)
                
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    
                    # Extract year
                    year_match = re.search(r'\b(19|20)\d{2}\b', title)
                    year = year_match.group() if year_match else ""
                    
                    # Clean title
                    clean_title = re.sub(r'\s+\(\d{4}\)$', '', title)
                    clean_title = re.sub(r'\s+\d{4}$', '', clean_title)
                    
                    # Format content
                    post_content = msg.text
                    formatted_content = format_post(msg.text, max_length=500)
                    
                    movie_data = {
                        'title': clean_title,
                        'original_title': title,
                        'year': year,
                        'date': msg.date.isoformat() if isinstance(msg.date, datetime) else str(msg.date),
                        'is_new': is_new(msg.date) if msg.date else False,
                        'channel': channel_name_cached(Config.MAIN_CHANNEL_ID),
                        'channel_id': Config.MAIN_CHANNEL_ID,
                        'message_id': msg.id,
                        'has_file': False,
                        'has_post': True,
                        'content': formatted_content,
                        'post_content': post_content,
                        'quality_options': {},
                        'is_video_file': False,
                        'search_score': 5,
                        'result_type': 'home_movie'
                    }
                    
                    movies.append(movie_data)
                    
                    if len(movies) >= limit:
                        break
        
        # Fetch posters for all movies in batch
        if movies:
            movies_with_posters = await get_posters_for_movies_batch(movies)
            logger.info(f"✅ Fetched {len(movies_with_posters)} home movies with posters")
            return movies_with_posters[:limit]
        else:
            logger.warning("⚠️ No movies found for home page")
            return []
        
    except Exception as e:
        logger.error(f"❌ Home movies error: {e}")
        return []

# ============================================================================
# ✅ GET FILE DETAILS
# ============================================================================

async def get_file_details(channel_id, message_id):
    """Get detailed file information"""
    try:
        if files_col is None:
            return None
        
        file_doc = await files_col.find_one({
            'channel_id': channel_id,
            'message_id': message_id
        })
        
        if not file_doc:
            return None
        
        # Format response
        response = {
            'title': file_doc.get('title', ''),
            'file_name': file_doc.get('file_name', ''),
            'file_size': file_doc.get('file_size', 0),
            'file_size_formatted': format_size(file_doc.get('file_size', 0)),
            'quality': file_doc.get('quality', '480p'),
            'is_video': file_doc.get('is_video_file', False),
            'channel_id': file_doc.get('channel_id'),
            'message_id': file_doc.get('message_id'),
            'real_message_id': file_doc.get('real_message_id', message_id),
            'date': file_doc.get('date'),
            'caption': file_doc.get('caption', ''),
            'thumbnail_url': file_doc.get('thumbnail_url'),
            'has_thumbnail': file_doc.get('thumbnail_extracted', False),
            'year': file_doc.get('year', ''),
            'telegram_file_id': file_doc.get('telegram_file_id'),
            'duration': file_doc.get('duration', 0) if file_doc.get('is_video_file') else None,
            'width': file_doc.get('width', 0) if file_doc.get('is_video_file') else None,
            'height': file_doc.get('height', 0) if file_doc.get('is_video_file') else None
        }
        
        return response
        
    except Exception as e:
        logger.error(f"File details error: {e}")
        return None

# ============================================================================
# ✅ GENERATE STREAMING URL
# ============================================================================

async def generate_streaming_url(telegram_file_id, quality="480p"):
    """Generate streaming URL from Telegram file ID"""
    if not Config.BOT_TOKEN:
        return None
    
    try:
        async with aiohttp.ClientSession() as session:
            # Get file path from Telegram
            url = f"https://api.telegram.org/bot{Config.BOT_TOKEN}/getFile?file_id={telegram_file_id}"
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('ok'):
                        file_path = data['result']['file_path']
                        
                        # Create streaming URL
                        streaming_url = f"https://api.telegram.org/file/bot{Config.BOT_TOKEN}/{file_path}"
                        return streaming_url
        
        return None
        
    except Exception as e:
        logger.error(f"Streaming URL generation error: {e}")
        return None

# ============================================================================
# ✅ GET ALL QUALITIES FOR TITLE
# ============================================================================

async def get_all_qualities_for_title(normalized_title):
    """Get all qualities available for a title"""
    try:
        if files_col is None:
            return {}
        
        # Find all files with same normalized title
        cursor = files_col.find({
            'normalized_title': normalized_title,
            'status': 'active',
            'is_duplicate': False
        }, {
            'quality': 1,
            'telegram_file_id': 1,
            'file_size': 1,
            'file_name': 1,
            'message_id': 1,
            'real_message_id': 1,
            'title': 1,
            '_id': 0
        })
        
        qualities = {}
        
        async for doc in cursor:
            quality = doc.get('quality', '480p')
            telegram_file_id = doc.get('telegram_file_id')
            
            if telegram_file_id:
                qualities[quality] = {
                    'telegram_file_id': telegram_file_id,
                    'file_size': doc.get('file_size', 0),
                    'file_size_formatted': format_size(doc.get('file_size', 0)),
                    'file_name': doc.get('file_name', ''),
                    'message_id': doc.get('message_id'),
                    'real_message_id': doc.get('real_message_id'),
                    'title': doc.get('title', ''),
                    'quality': quality
                }
        
        return qualities
        
    except Exception as e:
        logger.error(f"Get qualities error: {e}")
        return {}

# ============================================================================
# ✅ MONGODB INITIALIZATION
# ============================================================================

async def init_mongodb():
    global mongo_client, db, files_col
    
    try:
        logger.info("🔌 MongoDB initialization...")
        
        mongo_client = AsyncIOMotorClient(
            Config.MONGODB_URI,
            serverSelectionTimeoutMS=10000,
            connectTimeoutMS=10000,
            socketTimeoutMS=15000,
            maxPoolSize=20,
            minPoolSize=5,
            retryWrites=True,
            retryReads=True
        )
        
        await asyncio.wait_for(mongo_client.admin.command('ping'), timeout=5)
        
        db = mongo_client.sk4film
        files_col = db.files
        
        logger.info("✅ MongoDB OK")
        return True
        
    except asyncio.TimeoutError:
        logger.error("❌ MongoDB connection timeout")
        return False
    except Exception as e:
        logger.error(f"❌ MongoDB error: {e}")
        return False

# ============================================================================
# ✅ TELEGRAM SESSION INITIALIZATION
# ============================================================================

async def init_telegram_sessions():
    """Initialize Telegram sessions"""
    global User, Bot, user_session_ready, bot_session_ready
    
    logger.info("=" * 50)
    logger.info("🚀 TELEGRAM SESSION INITIALIZATION")
    logger.info("=" * 50)
    
    if not PYROGRAM_AVAILABLE:
        logger.error("❌ Pyrogram not installed!")
        return False
    
    # Initialize USER Session
    if Config.API_ID > 0 and Config.API_HASH and Config.USER_SESSION_STRING:
        logger.info("\n👤 Initializing USER Session...")
        try:
            User = Client(
                "sk4film_user",
                api_id=Config.API_ID,
                api_hash=Config.API_HASH,
                session_string=Config.USER_SESSION_STRING,
                sleep_threshold=30,
                in_memory=True,
                no_updates=True
            )
            
            await User.start()
            me = await User.get_me()
            logger.info(f"✅ USER Session Ready: {me.first_name}")
            
            # Test channel access
            try:
                for channel_id in Config.TEXT_CHANNEL_IDS:
                    chat = await User.get_chat(channel_id)
                    logger.info(f"✅ Channel Access: {chat.title} ({channel_id})")
                
                user_session_ready = True
            except Exception as e:
                logger.error(f"❌ Channel access failed: {e}")
                user_session_ready = False
                
        except Exception as e:
            logger.error(f"❌ USER Session failed: {e}")
            user_session_ready = False
            if User is not None:
                try:
                    await User.stop()
                except:
                    pass
            User = None
    
    # Initialize BOT Session
    if Config.BOT_TOKEN:
        logger.info("\n🤖 Initializing BOT Session...")
        try:
            Bot = Client(
                "sk4film_bot",
                api_id=Config.API_ID,
                api_hash=Config.API_HASH,
                bot_token=Config.BOT_TOKEN,
                sleep_threshold=30,
                in_memory=True,
                no_updates=True
            )
            
            await Bot.start()
            bot_info = await Bot.get_me()
            logger.info(f"✅ BOT Session Ready: @{bot_info.username}")
            bot_session_ready = True
                
        except Exception as e:
            logger.error(f"❌ BOT Session failed: {e}")
            bot_session_ready = False
            if Bot is not None:
                try:
                    await Bot.stop()
                except:
                    pass
            Bot = None
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 SESSION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"USER Session: {'✅ READY' if user_session_ready else '❌ NOT READY'}")
    logger.info(f"BOT Session: {'✅ READY' if bot_session_ready else '❌ NOT READY'}")
    
    return user_session_ready or bot_session_ready

# ============================================================================
# ✅ SYSTEM INITIALIZATION
# ============================================================================

async def init_system():
    start_time = time.time()
    
    try:
        logger.info("=" * 60)
        logger.info("🚀 SK4FiLM v10.0 - COMPLETE API WITH ALL FEATURES")
        logger.info("=" * 60)
        
        # Initialize MongoDB
        mongo_ok = await init_mongodb()
        if not mongo_ok:
            logger.error("❌ MongoDB connection failed")
            return False
        
        # Get current file count
        if files_col is not None:
            file_count = await files_col.count_documents({})
            logger.info(f"📊 Current files in database: {file_count}")
        
        # Initialize Telegram Sessions
        if PYROGRAM_AVAILABLE:
            telegram_ok = await init_telegram_sessions()
            if not telegram_ok:
                logger.warning("⚠️ Telegram sessions failed")
        
        init_time = time.time() - start_time
        logger.info(f"⚡ SK4FiLM Started in {init_time:.2f}s")
        logger.info("=" * 60)
        
        logger.info("🔧 ENHANCED FEATURES:")
        logger.info(f"   • Separate Posts: ✅ ENABLED (Dono text channels ke alag alag posts)")
        logger.info(f"   • Streaming Support: {'✅ ENABLED' if Config.ENABLE_STREAMING else '❌ DISABLED'}")
        logger.info(f"   • File Channel Indexing: ✅ ENABLED")
        logger.info(f"   • Real Message IDs: ✅ ENABLED")
        logger.info(f"   • User Session: {'✅ READY' if user_session_ready else '❌ NOT READY'}")
        logger.info(f"   • Text Channels: {len(Config.TEXT_CHANNEL_IDS)} channels")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ System initialization failed: {e}")
        return False

# ============================================================================
# ✅ API ROUTES - ALL REQUIRED ENDPOINTS
# ============================================================================

@app.route('/')
async def root():
    """Root endpoint - API status"""
    if files_col is not None:
        tf = await files_col.count_documents({})
        video_files = await files_col.count_documents({'is_video_file': True})
        thumbnails_extracted = await files_col.count_documents({'thumbnail_extracted': True})
    else:
        tf = 0
        video_files = 0
        thumbnails_extracted = 0
    
    return jsonify({
        'status': 'healthy',
        'service': 'SK4FiLM v10.0 - COMPLETE API',
        'sessions': {
            'user_session': {
                'ready': user_session_ready,
                'channels': Config.TEXT_CHANNEL_IDS
            },
            'bot_session': {
                'ready': bot_session_ready,
                'username': Config.BOT_USERNAME
            }
        },
        'features': {
            'separate_posts': Config.SEPARATE_POSTS_FOR_SAME_TITLE,
            'streaming': Config.ENABLE_STREAMING,
            'direct_download': Config.ENABLE_DIRECT_DOWNLOAD,
            'real_message_ids': True,
            'file_channel_indexing': True
        },
        'stats': {
            'total_files': tf,
            'video_files': video_files,
            'thumbnails_extracted': thumbnails_extracted
        },
        'endpoints': {
            'search': '/api/search',
            'movies': '/api/movies',
            'stream_links': '/api/stream/links',
            'stream_info': '/api/stream/info',
            'file_info': '/api/file/info',
            'qualities': '/api/qualities',
            'health': '/health',
            'stats': '/api/stats'
        }
    })

@app.route('/health')
async def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'sessions': {
            'user': user_session_ready,
            'bot': bot_session_ready
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/movies', methods=['GET'])
async def api_movies():
    """Get home movies for homepage"""
    try:
        limit = int(request.args.get('limit', 25))
        
        movies = await get_home_movies(limit)
        
        return jsonify({
            'status': 'success' if movies else 'empty',
            'movies': movies,
            'total': len(movies),
            'limit': limit,
            'source': 'telegram',
            'channel_id': Config.MAIN_CHANNEL_ID,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Movies API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'movies': [],
            'total': 0
        }), 500

@app.route('/api/search', methods=['GET'])
async def api_search():
    """Main search endpoint - DONO TEXT CHANNELS KE ALAG ALAG POSTS"""
    try:
        query = request.args.get('query', '').strip()
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', Config.SEARCH_RESULTS_PER_PAGE))
        
        if len(query) < Config.SEARCH_MIN_QUERY_LENGTH:
            return jsonify({
                'status': 'error',
                'message': f'Query must be at least {Config.SEARCH_MIN_QUERY_LENGTH} characters'
            }), 400
        
        logger.info(f"🔍 API SEARCH: {query} - Page: {page}")
        
        # Use enhanced search with separate posts
        result_data = await search_movies_multi_channel_separate_posts(query, limit, page)
        
        return jsonify({
            'status': 'success',
            'query': query,
            'results': result_data['results'],
            'pagination': result_data['pagination'],
            'search_metadata': {
                **result_data.get('search_metadata', {}),
                'feature': 'enhanced_search_separate_posts',
                'quality_priority': Config.QUALITY_PRIORITY,
                'real_message_ids': True,
                'streaming_enabled': Config.ENABLE_STREAMING
            },
            'bot_username': Config.BOT_USERNAME,
            'file_channel_id': Config.FILE_CHANNEL_ID,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Search API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/stream/links', methods=['GET'])
async def api_stream_links():
    """Get streaming links for a file - VIEW.HTML KE LIYE"""
    try:
        channel_id = int(request.args.get('channel_id', Config.FILE_CHANNEL_ID))
        message_id = int(request.args.get('message_id', 0))
        quality = request.args.get('quality', '480p')
        
        if not message_id:
            return jsonify({'status': 'error', 'message': 'Message ID required'}), 400
        
        # Get file info from database
        if files_col is None:
            return jsonify({'status': 'error', 'message': 'Database not available'}), 500
        
        file_doc = await files_col.find_one({
            'channel_id': channel_id,
            'message_id': message_id
        })
        
        if not file_doc:
            return jsonify({'status': 'error', 'message': 'File not found'}), 404
        
        # Get Telegram file ID
        telegram_file_id = file_doc.get('telegram_file_id')
        if not telegram_file_id:
            return jsonify({'status': 'error', 'message': 'File ID not available'}), 404
        
        # Generate streaming URL
        streaming_url = await generate_streaming_url(telegram_file_id, quality)
        
        if not streaming_url:
            return jsonify({'status': 'error', 'message': 'Could not generate streaming URL'}), 500
        
        return jsonify({
            'status': 'success',
            'streaming': {
                'url': streaming_url,
                'type': 'telegram_stream',
                'quality': quality,
                'title': file_doc.get('title', ''),
                'file_name': file_doc.get('file_name', ''),
                'file_size': file_doc.get('file_size', 0),
                'file_size_formatted': format_size(file_doc.get('file_size', 0)),
                'is_video': file_doc.get('is_video_file', False),
                'supports_range': True,
                'mime_type': 'video/mp4'
            },
            'file_info': {
                'channel_id': channel_id,
                'message_id': message_id,
                'real_message_id': file_doc.get('real_message_id', message_id),
                'quality': quality,
                'bot_username': Config.BOT_USERNAME,
                'title': file_doc.get('title', '')
            },
            'direct_download': streaming_url if Config.ENABLE_DIRECT_DOWNLOAD else None,
            'bot_download_url': f"https://t.me/{Config.BOT_USERNAME}?start={channel_id}_{message_id}_{quality}",
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Stream links API error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/stream/info', methods=['GET'])
async def api_stream_info():
    """Get streaming information for multiple qualities - QUALITY DROPDOWN KE LIYE"""
    try:
        channel_id = int(request.args.get('channel_id', Config.FILE_CHANNEL_ID))
        message_id = int(request.args.get('message_id', 0))
        
        if not message_id:
            return jsonify({'status': 'error', 'message': 'Message ID required'}), 400
        
        # Get file info from database
        if files_col is None:
            return jsonify({'status': 'error', 'message': 'Database not available'}), 500
        
        file_doc = await files_col.find_one({
            'channel_id': channel_id,
            'message_id': message_id
        })
        
        if not file_doc:
            return jsonify({'status': 'error', 'message': 'File not found'}), 404
        
        # Get normalized title
        normalized_title = file_doc.get('normalized_title')
        if not normalized_title:
            return jsonify({'status': 'error', 'message': 'Title not found'}), 404
        
        # Get all files with same title (different qualities)
        same_title_files = await files_col.find({
            'normalized_title': normalized_title,
            'status': 'active',
            'is_duplicate': False
        }, {
            'quality': 1,
            'telegram_file_id': 1,
            'file_size': 1,
            'file_name': 1,
            'message_id': 1,
            'real_message_id': 1,
            'title': 1,
            '_id': 0
        }).to_list(length=20)
        
        streaming_info = {}
        
        for file_data in same_title_files:
            quality = file_data.get('quality', '480p')
            telegram_file_id = file_data.get('telegram_file_id')
            
            if telegram_file_id:
                streaming_url = await generate_streaming_url(telegram_file_id, quality)
                
                if streaming_url:
                    streaming_info[quality] = {
                        'url': streaming_url,
                        'file_size': file_data.get('file_size', 0),
                        'file_size_formatted': format_size(file_data.get('file_size', 0)),
                        'file_name': file_data.get('file_name', ''),
                        'message_id': file_data.get('message_id'),
                        'real_message_id': file_data.get('real_message_id'),
                        'quality': quality,
                        'title': file_data.get('title', ''),
                        'bot_download_url': f"https://t.me/{Config.BOT_USERNAME}?start={channel_id}_{file_data.get('message_id')}_{quality}",
                        'direct_download': streaming_url if Config.ENABLE_DIRECT_DOWNLOAD else None
                    }
        
        return jsonify({
            'status': 'success',
            'title': file_doc.get('title', ''),
            'streaming_qualities': streaming_info,
            'available_qualities': list(streaming_info.keys()),
            'best_quality': list(streaming_info.keys())[0] if streaming_info else '480p',
            'has_streaming': len(streaming_info) > 0,
            'bot_username': Config.BOT_USERNAME,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Stream info API error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/file/info', methods=['GET'])
async def api_file_info():
    """Get detailed file information"""
    try:
        channel_id = int(request.args.get('channel_id', Config.FILE_CHANNEL_ID))
        message_id = int(request.args.get('message_id', 0))
        
        if not message_id:
            return jsonify({'status': 'error', 'message': 'Message ID required'}), 400
        
        # Get file info from database
        file_info = await get_file_details(channel_id, message_id)
        
        if not file_info:
            return jsonify({'status': 'error', 'message': 'File not found'}), 404
        
        return jsonify({
            'status': 'success',
            'file_info': file_info,
            'download_links': {
                'telegram_bot': f"https://t.me/{Config.BOT_USERNAME}?start={channel_id}_{message_id}_480p",
                'direct_download': Config.ENABLE_DIRECT_DOWNLOAD
            },
            'streaming_support': Config.ENABLE_STREAMING,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"File info API error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/qualities', methods=['GET'])
async def api_qualities():
    """Get all available qualities for a title - DROPDOWN KE LIYE"""
    try:
        title = request.args.get('title', '').strip()
        normalized_title = request.args.get('normalized_title', '').strip()
        
        if not title and not normalized_title:
            return jsonify({'status': 'error', 'message': 'Title or normalized_title required'}), 400
        
        if not normalized_title:
            normalized_title = normalize_title(title)
        
        # Get all qualities for this title
        qualities = await get_all_qualities_for_title(normalized_title)
        
        return jsonify({
            'status': 'success',
            'title': title,
            'normalized_title': normalized_title,
            'qualities': qualities,
            'available_qualities': list(qualities.keys()),
            'quality_count': len(qualities),
            'has_qualities': len(qualities) > 0,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Qualities API error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/bot/download', methods=['GET'])
async def api_bot_download():
    """Generate Telegram bot download link - QUALITY DROPDOWN KE LIYE"""
    try:
        channel_id = int(request.args.get('channel_id', Config.FILE_CHANNEL_ID))
        message_id = int(request.args.get('message_id', 0))
        quality = request.args.get('quality', '480p')
        
        if not message_id:
            return jsonify({'status': 'error', 'message': 'Message ID required'}), 400
        
        # Get file info to verify
        file_info = await get_file_details(channel_id, message_id)
        if not file_info:
            return jsonify({'status': 'error', 'message': 'File not found'}), 404
        
        # Create bot download link
        bot_url = f"https://t.me/{Config.BOT_USERNAME}?start={channel_id}_{message_id}_{quality}"
        
        return jsonify({
            'status': 'success',
            'bot_url': bot_url,
            'quality': quality,
            'file_info': {
                'title': file_info.get('title', ''),
                'file_name': file_info.get('file_name', ''),
                'file_size_formatted': file_info.get('file_size_formatted', '')
            },
            'instructions': 'Click the link to open Telegram and send /start to the bot',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Bot download API error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
async def api_stats():
    """Get system statistics"""
    try:
        if files_col is not None:
            total_files = await files_col.count_documents({})
            video_files = await files_col.count_documents({'is_video_file': True})
            thumbnails_extracted = await files_col.count_documents({'thumbnail_extracted': True})
            
            # Get quality distribution
            pipeline = [
                {"$group": {"_id": "$quality", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]
            quality_dist = await files_col.aggregate(pipeline).to_list(length=10)
            
            # Get recent files
            recent = await files_col.find({}, {
                'title': 1, 
                'date': 1, 
                'real_message_id': 1,
                'quality': 1,
                '_id': 0
            }).sort('date', -1).limit(5).to_list(length=5)
        else:
            total_files = 0
            video_files = 0
            thumbnails_extracted = 0
            quality_dist = []
            recent = []
        
        return jsonify({
            'status': 'success',
            'database_stats': {
                'total_files': total_files,
                'video_files': video_files,
                'thumbnails_extracted': thumbnails_extracted,
                'extraction_rate': f"{(thumbnails_extracted/video_files*100):.1f}%" if video_files > 0 else "0%"
            },
            'quality_distribution': quality_dist,
            'recent_files': recent,
            'sessions': {
                'user': user_session_ready,
                'bot': bot_session_ready
            },
            'features': {
                'separate_posts': Config.SEPARATE_POSTS_FOR_SAME_TITLE,
                'streaming': Config.ENABLE_STREAMING,
                'direct_download': Config.ENABLE_DIRECT_DOWNLOAD
            },
            'channels': {
                'text_channels': Config.TEXT_CHANNEL_IDS,
                'file_channel': Config.FILE_CHANNEL_ID
            },
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Stats API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/debug/search-test', methods=['GET'])
async def api_debug_search_test():
    """Debug endpoint to test search with separate posts"""
    try:
        query = request.args.get('query', 'test')
        
        # Get results
        result_data = await search_movies_multi_channel_separate_posts(query, 5, 1)
        
        # Analyze results
        analysis = {
            'total_results': len(result_data['results']),
            'posts': sum(1 for r in result_data['results'] if r.get('has_post', False)),
            'files': sum(1 for r in result_data['results'] if r.get('has_file', False)),
            'both': sum(1 for r in result_data['results'] if r.get('has_post', False) and r.get('has_file', False)),
            'stream_now': sum(1 for r in result_data['results'] if r.get('stream_now_available', False)),
            'channels': set(r.get('channel_id') for r in result_data['results']),
            'separate_posts': Config.SEPARATE_POSTS_FOR_SAME_TITLE
        }
        
        # Sample results
        samples = []
        for i, result in enumerate(result_data['results'][:3]):
            samples.append({
                'title': result.get('title', '')[:50],
                'type': 'post+file' if result.get('has_post') and result.get('has_file') else ('post' if result.get('has_post') else 'file'),
                'channel': result.get('channel', 'Unknown'),
                'channel_id': result.get('channel_id'),
                'qualities': result.get('quality_count', 0),
                'stream_now': result.get('stream_now_available', False)
            })
        
        return jsonify({
            'status': 'success',
            'query': query,
            'analysis': analysis,
            'sample_results': samples,
            'features_active': {
                'separate_posts': Config.SEPARATE_POSTS_FOR_SAME_TITLE,
                'text_channels': Config.TEXT_CHANNEL_IDS
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Debug search test error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# ============================================================================
# ✅ STARTUP AND SHUTDOWN
# ============================================================================

app_start_time = time.time()

@app.before_serving
async def startup():
    await init_system()

@app.after_serving
async def shutdown():
    logger.info("🛑 Shutting down SK4FiLM v10.0...")
    
    # Close Telegram sessions
    if User is not None:
        await User.stop()
    
    if Bot is not None:
        await Bot.stop()
    
    # Close MongoDB
    if mongo_client is not None:
        mongo_client.close()
    
    logger.info(f"👋 Shutdown complete. Uptime: {time.time() - app_start_time:.1f}s")

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
    config.http2 = True
    config.keep_alive_timeout = 30
    
    logger.info(f"🌐 Starting SK4FiLM v10.0 on port {Config.WEB_SERVER_PORT}...")
    logger.info("🎯 COMPLETE FEATURES:")
    logger.info(f"   • Separate Posts: ✅ ENABLED (Dono text channels ke alag alag posts)")
    logger.info(f"   • Streaming Support: ✅ ENABLED")
    logger.info(f"   • Direct Download: {'✅ ENABLED' if Config.ENABLE_DIRECT_DOWNLOAD else '❌ DISABLED'}")
    logger.info(f"   • Text Channels: {len(Config.TEXT_CHANNEL_IDS)} channels")
    logger.info(f"   • File Channel: {Config.FILE_CHANNEL_ID}")
    logger.info(f"   • Real Message IDs: ✅ ENABLED")
    logger.info("📡 Available API Endpoints:")
    logger.info("   • /api/search - Main search with separate posts")
    logger.info("   • /api/movies - Home movies")
    logger.info("   • /api/stream/links - Streaming links")
    logger.info("   • /api/stream/info - Quality info for dropdown")
    logger.info("   • /api/file/info - File details")
    logger.info("   • /api/qualities - All qualities for title")
    logger.info("   • /api/bot/download - Telegram bot download")
    logger.info("   • /api/stats - System statistics")
    
    asyncio.run(serve(app, config))
