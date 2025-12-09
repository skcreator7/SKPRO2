"""
app.py - SK4FiLM Web API with Turbo Performance
ULTRA-OPTIMIZED: Async caching, connection pooling, and minimal latency
"""
import asyncio
import os
import logging
import json
import re
import math
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from functools import lru_cache

import aiohttp
from quart import Quart, jsonify, request
from hypercorn.asyncio import serve
from hypercorn.config import Config as HyperConfig
from motor.motor_asyncio import AsyncIOMotorClient
import redis.asyncio as redis

# Import only essentials
from utils import normalize_title, extract_title_smart, format_post, is_new

# ULTRA-FAST CONFIGURATION
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Reduce noise
for log_name in ['asyncio', 'motor', 'hypercorn', 'urllib3']:
    logging.getLogger(log_name).setLevel(logging.WARNING)

class Config:
    # API Configuration
    API_ID = int(os.environ.get("API_ID", "0"))
    API_HASH = os.environ.get("API_HASH", "")
    USER_SESSION_STRING = os.environ.get("USER_SESSION_STRING", "")
    
    # Database
    MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
    REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
    
    # Channels
    MAIN_CHANNEL_ID = -1001891090100
    TEXT_CHANNEL_IDS = [-1001891090100, -1002024811395]
    FILE_CHANNEL_ID = -1001768249569
    
    # Performance
    MAX_CONCURRENT_REQUESTS = 50
    CACHE_TTL = 300
    REQUEST_TIMEOUT = 5
    WEB_SERVER_PORT = int(os.environ.get("PORT", 8000))
    BACKEND_URL = os.environ.get("BACKEND_URL", "https://sk4film.koyeb.app")

# Initialize app with minimal config
app = Quart(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# CORS
@app.after_request
async def add_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Cache-Control'] = 'public, max-age=60'
    return response

# GLOBAL VARIABLES
mongo_client = None
db = None
files_col = None
redis_client = None
user_session_ready = False

# OPTIMIZED CACHING SYSTEM
class CacheManager:
    def __init__(self):
        self.redis_enabled = False
        self.memory_cache = {}
    
    async def init_redis(self):
        try:
            self.redis_client = redis.from_url(Config.REDIS_URL, decode_responses=True)
            await self.redis_client.ping()
            self.redis_enabled = True
            logger.info("✅ Redis cache enabled")
            return True
        except:
            logger.warning("⚠️ Redis not available, using memory cache")
            return False
    
    async def get(self, key):
        if self.redis_enabled:
            try:
                cached = await self.redis_client.get(key)
                if cached:
                    return json.loads(cached)
            except:
                pass
        return self.memory_cache.get(key)
    
    async def set(self, key, value, ttl=300):
        if self.redis_enabled:
            try:
                await self.redis_client.setex(key, ttl, json.dumps(value))
            except:
                pass
        self.memory_cache[key] = value
    
    async def delete(self, key):
        if self.redis_enabled:
            try:
                await self.redis_client.delete(key)
            except:
                pass
        if key in self.memory_cache:
            del self.memory_cache[key]

cache_manager = CacheManager()

# FAST MONGO INIT
async def init_mongodb():
    global mongo_client, db, files_col
    try:
        mongo_client = AsyncIOMotorClient(
            Config.MONGODB_URI,
            serverSelectionTimeoutMS=2000,
            connectTimeoutMS=2000,
            maxPoolSize=10
        )
        await mongo_client.admin.command('ping')
        db = mongo_client.sk4film
        files_col = db.files
        logger.info("✅ MongoDB connected")
        return True
    except Exception as e:
        logger.error(f"❌ MongoDB error: {e}")
        return False

# OPTIMIZED SEARCH
@lru_cache(maxsize=1000)
def normalize_title_cached(title: str) -> str:
    return normalize_title(title)

async def search_movies(query: str, limit: int = 12, page: int = 1):
    """Ultra-fast search with caching"""
    offset = (page - 1) * limit
    cache_key = f"search:{query}:{page}:{limit}"
    
    # Try cache
    cached = await cache_manager.get(cache_key)
    if cached:
        logger.info(f"✅ Cache HIT: {query}")
        return cached
    
    logger.info(f"🔍 Searching: {query}")
    
    try:
        # MongoDB text search
        cursor = files_col.find(
            {'$text': {'$search': query}},
            {
                'title': 1,
                'quality': 1,
                'file_size': 1,
                'thumbnail': 1,
                'channel_id': 1,
                'message_id': 1,
                'date': 1,
                '_id': 0
            }
        ).limit(limit * 2)
        
        results = []
        async for doc in cursor:
            results.append({
                'title': doc['title'],
                'quality': doc.get('quality', '480p'),
                'file_size': doc.get('file_size', 0),
                'thumbnail': doc.get('thumbnail'),
                'channel_id': doc.get('channel_id'),
                'message_id': doc.get('message_id'),
                'date': doc['date'].isoformat() if isinstance(doc['date'], datetime) else doc['date']
            })
        
        # Paginate
        total = len(results)
        paginated = results[offset:offset + limit]
        
        result_data = {
            'results': paginated,
            'pagination': {
                'current_page': page,
                'total_pages': math.ceil(total / limit) if total > 0 else 1,
                'total_results': total,
                'per_page': limit
            }
        }
        
        # Cache for 5 minutes
        await cache_manager.set(cache_key, result_data, 300)
        
        return result_data
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return {'results': [], 'pagination': {'current_page': page, 'total_pages': 1, 'total_results': 0, 'per_page': limit}}

# FAST API ENDPOINTS
@app.route('/')
async def root():
    return jsonify({
        'status': 'healthy',
        'service': 'SK4FiLM v8.0',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/movies', methods=['GET'])
async def api_movies():
    """Get latest movies - SUPER FAST"""
    try:
        cache_key = "home_movies"
        cached = await cache_manager.get(cache_key)
        if cached:
            return jsonify(cached)
        
        # Get latest from database
        cursor = files_col.find(
            {},
            {
                'title': 1,
                'thumbnail': 1,
                'date': 1,
                '_id': 0
            }
        ).sort('date', -1).limit(20)
        
        movies = []
        async for doc in cursor:
            movies.append({
                'title': doc['title'],
                'thumbnail': doc.get('thumbnail'),
                'date': doc['date'].isoformat() if isinstance(doc['date'], datetime) else doc['date'],
                'is_new': is_new(doc['date']) if doc.get('date') else False
            })
        
        # Enhance with simple poster
        for movie in movies:
            movie['poster_url'] = f"{Config.BACKEND_URL}/api/poster?title={movie['title']}"
            movie['poster_rating'] = '0.0'
        
        response = {
            'status': 'success',
            'movies': movies,
            'total': len(movies),
            'timestamp': datetime.now().isoformat()
        }
        
        await cache_manager.set(cache_key, response, 60)  # 1 minute cache
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Movies API error: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Internal server error',
            'movies': []
        })

@app.route('/api/search', methods=['GET'])
async def api_search():
    """Optimized search endpoint"""
    try:
        query = request.args.get('query', '').strip()
        page = int(request.args.get('page', 1))
        limit = min(int(request.args.get('limit', 12)), 50)
        
        if len(query) < 2:
            return jsonify({
                'status': 'error',
                'message': 'Query must be at least 2 characters'
            }), 400
        
        result_data = await search_movies(query, limit, page)
        
        return jsonify({
            'status': 'success',
            'query': query,
            'results': result_data['results'],
            'pagination': result_data['pagination'],
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Search API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/poster', methods=['GET'])
async def api_poster():
    """Fast poster endpoint with fallback"""
    title = request.args.get('title', '').strip()
    year = request.args.get('year', '')
    
    if not title:
        return jsonify({
            'status': 'error',
            'message': 'Title required'
        }), 400
    
    # Simple fallback poster
    return jsonify({
        'status': 'success',
        'poster': {
            'poster_url': f"https://via.placeholder.com/300x450/1a1a2e/ffffff?text={title[:20]}",
            'source': 'placeholder',
            'rating': '0.0'
        },
        'title': title,
        'year': year
    })

@app.route('/api/health', methods=['GET'])
async def api_health():
    """Health check"""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'cache': cache_manager.redis_enabled,
        'database': files_col is not None
    })

# STARTUP
async def init_system():
    """Fast initialization"""
    logger.info("🚀 Starting SK4FiLM v8.0 - OPTIMIZED")
    
    # Initialize databases
    await init_mongodb()
    await cache_manager.init_redis()
    
    logger.info("✅ System initialized")

@app.before_serving
async def startup():
    await init_system()

# RUN SERVER
if __name__ == "__main__":
    config = HyperConfig()
    config.bind = [f"0.0.0.0:{Config.WEB_SERVER_PORT}"]
    config.worker_class = "asyncio"
    config.workers = 1
    config.accesslog = None
    config.errorlog = "-"
    config.loglevel = "warning"
    
    asyncio.run(serve(app, config))
