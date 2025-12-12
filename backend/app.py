"""
SK4FiLM Web API - Complete with Telegram Session Generator
"""
import os
import sys
import asyncio
import logging
import json
import re
import math
import time
import base64
import io
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from functools import lru_cache, wraps

import aiohttp
import qrcode
from PIL import Image
from quart import Quart, jsonify, request, Response, render_template_string
from hypercorn.asyncio import serve
from hypercorn.config import Config as HyperConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ========== CONFIGURATION ==========
class Config:
    # Telegram Configuration
    API_ID = int(os.environ.get("API_ID", "0"))
    API_HASH = os.environ.get("API_HASH", "")
    USER_SESSION_STRING = os.environ.get("USER_SESSION_STRING", "")
    BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
    
    # Database Configuration
    MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
    REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
    
    # Channel Configuration
    MAIN_CHANNEL_ID = -1001891090100
    TEXT_CHANNEL_IDS = [-1001891090100, -1002024811395]
    FILE_CHANNEL_ID = -1001768249569
    
    # Website Links
    WEBSITE_URL = os.environ.get("WEBSITE_URL", "https://sk4film.vercel.app")
    MAIN_CHANNEL_LINK = "https://t.me/sk4film"
    UPDATES_CHANNEL_LINK = "https://t.me/sk4film_Request"
    
    # Admin Configuration
    ADMIN_IDS = [int(x) for x in os.environ.get("ADMIN_IDS", "123456789").split(",")]
    ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "sk4film_admin")
    
    # Server Configuration
    WEB_SERVER_PORT = int(os.environ.get("PORT", 8000))
    BACKEND_URL = os.environ.get("BACKEND_URL", "https://sk4film.koyeb.app")
    
    # API Keys
    OMDB_KEYS = ["8265bd1c", "b9bd48a6", "3e7e1e9d"]
    TMDB_KEYS = ["e547e17d4e91f3e62a571655cd1ccaff"]
    
    # Session Generator Configuration
    SESSION_EXPIRY_HOURS = int(os.environ.get("SESSION_EXPIRY_HOURS", "24"))
    MAX_SESSIONS_PER_IP = int(os.environ.get("MAX_SESSIONS_PER_IP", "3"))

# Create Quart app
app = Quart(__name__)

# CORS headers
@app.after_request
async def add_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    response.headers['X-SK4FiLM-Version'] = '2.0-SESSION-GEN'
    return response

# ========== DATABASE SETUP ==========
from motor.motor_asyncio import AsyncIOMotorClient
import redis.asyncio as redis

mongo_client = None
db = None
files_col = None
sessions_col = None
users_col = None

async def init_mongodb():
    """Initialize MongoDB"""
    global mongo_client, db, files_col, sessions_col, users_col
    
    try:
        logger.info("🔌 Initializing MongoDB...")
        mongo_client = AsyncIOMotorClient(Config.MONGODB_URI)
        db = mongo_client.sk4film
        
        # Collections
        files_col = db.files
        sessions_col = db.sessions
        users_col = db.users
        
        # Create indexes
        await files_col.create_index([("normalized_title", 1)])
        await files_col.create_index([("title", "text")])
        await sessions_col.create_index([("session_string", 1)], unique=True)
        await sessions_col.create_index([("created_at", 1)], expireAfterSeconds=Config.SESSION_EXPIRY_HOURS * 3600)
        await users_col.create_index([("user_id", 1)], unique=True)
        
        logger.info("✅ MongoDB initialized")
        return True
    except Exception as e:
        logger.error(f"❌ MongoDB error: {e}")
        return False

# Redis connection
redis_client = None

async def init_redis():
    """Initialize Redis"""
    global redis_client
    
    try:
        logger.info("🔌 Initializing Redis...")
        redis_client = await redis.from_url(Config.REDIS_URL, decode_responses=True)
        await redis_client.ping()
        logger.info("✅ Redis initialized")
        return True
    except Exception as e:
        logger.warning(f"⚠️ Redis error: {e}")
        return False

# ========== TELEGRAM SESSION GENERATOR ==========
class TelegramSessionGenerator:
    """Built-in Telegram session generator"""
    
    def __init__(self):
        self.api_id = Config.API_ID
        self.api_hash = Config.API_HASH
        self.sessions = {}
        self.pending_auth = {}
    
    async def generate_qr_code(self, session_id: str) -> str:
        """Generate QR code for session login"""
        try:
            # Create login URL for Telegram
            login_url = f"tg://login?token={session_id}"
            
            # Generate QR code
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4,
            )
            qr.add_data(login_url)
            qr.make(fit=True)
            
            # Create image
            img = qr.make_image(fill_color="black", back_color="white")
            
            # Convert to base64
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return f"data:image/png;base64,{img_str}"
        except Exception as e:
            logger.error(f"QR code generation error: {e}")
            return None
    
    async def create_session(self, user_ip: str, user_agent: str) -> Dict[str, Any]:
        """Create a new session request"""
        try:
            # Check rate limit
            session_count = await sessions_col.count_documents({
                "user_ip": user_ip,
                "created_at": {"$gte": datetime.now() - timedelta(hours=24)}
            })
            
            if session_count >= Config.MAX_SESSIONS_PER_IP:
                return {
                    "success": False,
                    "error": "Rate limit exceeded. Maximum 3 sessions per IP per day."
                }
            
            # Generate session ID
            session_id = hashlib.sha256(f"{user_ip}{time.time()}".encode()).hexdigest()[:32]
            
            # Store session request
            session_data = {
                "session_id": session_id,
                "user_ip": user_ip,
                "user_agent": user_agent[:200],
                "status": "pending",
                "created_at": datetime.now(),
                "expires_at": datetime.now() + timedelta(hours=Config.SESSION_EXPIRY_HOURS),
                "api_id": self.api_id,
                "api_hash": self.api_hash
            }
            
            await sessions_col.insert_one(session_data)
            
            # Generate QR code
            qr_code = await self.generate_qr_code(session_id)
            
            return {
                "success": True,
                "session_id": session_id,
                "qr_code": qr_code,
                "expires_in": Config.SESSION_EXPIRY_HOURS,
                "instructions": "Scan QR code with Telegram to generate session"
            }
        except Exception as e:
            logger.error(f"Session creation error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def check_session_status(self, session_id: str) -> Dict[str, Any]:
        """Check session status"""
        try:
            session = await sessions_col.find_one({"session_id": session_id})
            
            if not session:
                return {
                    "success": False,
                    "error": "Session not found"
                }
            
            if session["status"] == "completed":
                return {
                    "success": True,
                    "status": "completed",
                    "session_string": session["session_string"],
                    "user_id": session["user_id"],
                    "phone_number": session.get("phone_number"),
                    "username": session.get("username"),
                    "expires_at": session["expires_at"].isoformat()
                }
            elif session["status"] == "pending":
                return {
                    "success": True,
                    "status": "pending",
                    "created_at": session["created_at"].isoformat(),
                    "expires_at": session["expires_at"].isoformat()
                }
            else:
                return {
                    "success": False,
                    "status": session["status"],
                    "error": session.get("error", "Unknown error")
                }
        except Exception as e:
            logger.error(f"Session status check error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def process_session_callback(self, session_id: str, user_data: Dict[str, Any]) -> bool:
        """Process callback from Telegram (to be called by webhook)"""
        try:
            session = await sessions_col.find_one({"session_id": session_id})
            
            if not session:
                logger.error(f"Session {session_id} not found for callback")
                return False
            
            # Update session with user data
            update_data = {
                "status": "completed",
                "user_id": user_data.get("id"),
                "phone_number": user_data.get("phone"),
                "username": user_data.get("username"),
                "first_name": user_data.get("first_name"),
                "last_name": user_data.get("last_name"),
                "session_string": user_data.get("session_string"),
                "completed_at": datetime.now()
            }
            
            await sessions_col.update_one(
                {"session_id": session_id},
                {"$set": update_data}
            )
            
            logger.info(f"✅ Session {session_id} completed for user {user_data.get('id')}")
            return True
            
        except Exception as e:
            logger.error(f"Session callback error: {e}")
            return False

# Create session generator instance
session_generator = TelegramSessionGenerator()

# ========== TELEGRAM CLIENT SETUP ==========
try:
    from pyrogram import Client
    from pyrogram.errors import (
        SessionPasswordNeeded, PhoneCodeInvalid, 
        PhoneCodeExpired, FloodWait
    )
    PYROGRAM_AVAILABLE = True
except ImportError:
    logger.error("❌ Pyrogram not installed. Please install: pip install pyrogram")
    PYROGRAM_AVAILABLE = False
    Client = None

User = None
Bot = None
user_session_ready = False
bot_started = False

async def init_telegram_clients():
    """Initialize Telegram clients"""
    global User, Bot, user_session_ready, bot_started
    
    if not PYROGRAM_AVAILABLE:
        return False
    
    logger.info("📱 Initializing Telegram clients...")
    
    # Initialize Bot
    if Config.BOT_TOKEN:
        try:
            logger.info("🤖 Starting bot...")
            Bot = Client(
                name="sk4film_bot",
                api_id=Config.API_ID,
                api_hash=Config.API_HASH,
                bot_token=Config.BOT_TOKEN,
                workers=10
            )
            await Bot.start()
            bot_info = await Bot.get_me()
            bot_started = True
            logger.info(f"✅ Bot started: @{bot_info.username}")
        except Exception as e:
            logger.error(f"❌ Bot error: {e}")
    else:
        logger.warning("⚠️ No BOT_TOKEN configured")
    
    # Initialize User Client from session string
    if Config.USER_SESSION_STRING:
        try:
            logger.info("👤 Starting user session from string...")
            User = Client(
                name="user_session",
                api_id=Config.API_ID,
                api_hash=Config.API_HASH,
                session_string=Config.USER_SESSION_STRING,
                in_memory=True
            )
            await User.start()
            user_info = await User.get_me()
            user_session_ready = True
            logger.info(f"✅ User session started: @{user_info.username}")
        except Exception as e:
            logger.error(f"❌ User session error: {e}")
    else:
        logger.info("ℹ️ No USER_SESSION_STRING, will use generated sessions")
    
    return user_session_ready or bot_started

async def get_telegram_client_for_session(session_string: str) -> Optional[Client]:
    """Get Telegram client for a specific session"""
    try:
        # Create temporary client
        temp_client = Client(
            name=f"temp_session_{hash(session_string)}",
            api_id=Config.API_ID,
            api_hash=Config.API_HASH,
            session_string=session_string,
            in_memory=True
        )
        
        await temp_client.start()
        return temp_client
    except Exception as e:
        logger.error(f"Error creating client for session: {e}")
        return None

# ========== MOVIE DATABASE ==========
# Real movie data with proper structure
MOVIE_DATABASE = [
    {
        "id": "avatar_2022",
        "title": "Avatar: The Way of Water",
        "year": "2022",
        "rating": "7.6",
        "duration": "192 min",
        "genres": ["Action", "Adventure", "Fantasy"],
        "description": "Jake Sully lives with his newfound family formed on the extrasolar moon Pandora.",
        "poster_url": "https://m.media-amazon.com/images/M/MV5BYjhiNjBlODctY2ZiOC00YjVlLWFlNzAtNTVhNzM1YjI1NzMxXkEyXkFqcGdeQXVyMjQxNTE1MDA@._V1_SX300.jpg",
        "backdrop_url": "https://image.tmdb.org/t/p/original/s16H6tpK2utvwDtzZ8Qy4qm5Emw.jpg",
        "channel": "SK4FiLM Main",
        "channel_id": Config.MAIN_CHANNEL_ID,
        "date": "2023-10-15T10:30:00",
        "is_new": True,
        "qualities": [
            {
                "quality": "480p",
                "size": "1.2 GB",
                "file_id": f"{Config.FILE_CHANNEL_ID}_16066_480p",
                "available": True
            },
            {
                "quality": "720p",
                "size": "2.5 GB",
                "file_id": f"{Config.FILE_CHANNEL_ID}_16067_720p",
                "available": True
            },
            {
                "quality": "1080p",
                "size": "4.8 GB",
                "file_id": f"{Config.FILE_CHANNEL_ID}_16068_1080p",
                "available": True
            }
        ],
        "related": ["avatar_2009", "avatar_3"]
    },
    {
        "id": "john_wick_4",
        "title": "John Wick 4",
        "year": "2023",
        "rating": "7.9",
        "duration": "169 min",
        "genres": ["Action", "Crime", "Thriller"],
        "description": "John Wick uncovers a path to defeating The High Table.",
        "poster_url": "https://m.media-amazon.com/images/M/MV5BMDExZGMyOTMtMDgyYi00NGIwLWJhMTEtOTdkZGFjNmZiMTEwXkEyXkFqcGdeQXVyMjM4NTM5NDY@._V1_SX300.jpg",
        "backdrop_url": "https://image.tmdb.org/t/p/original/vZloFAK7NmvMGKE7VkF5UHaz0I.jpg",
        "channel": "SK4FiLM Main",
        "channel_id": Config.MAIN_CHANNEL_ID,
        "date": "2023-10-18T14:20:00",
        "is_new": True,
        "qualities": [
            {
                "quality": "480p",
                "size": "1.0 GB",
                "file_id": f"{Config.FILE_CHANNEL_ID}_17001_480p",
                "available": True
            },
            {
                "quality": "720p",
                "size": "2.1 GB",
                "file_id": f"{Config.FILE_CHANNEL_ID}_17002_720p",
                "available": True
            },
            {
                "quality": "1080p",
                "size": "4.2 GB",
                "file_id": f"{Config.FILE_CHANNEL_ID}_17003_1080p",
                "available": True
            }
        ],
        "related": ["john_wick_1", "john_wick_2", "john_wick_3"]
    },
    {
        "id": "mission_impossible_7",
        "title": "Mission: Impossible - Dead Reckoning Part One",
        "year": "2023",
        "rating": "7.8",
        "duration": "163 min",
        "genres": ["Action", "Adventure", "Thriller"],
        "description": "Ethan Hunt and his IMF team must track down a dangerous weapon.",
        "poster_url": "https://m.media-amazon.com/images/M/MV5BYzFiZjc1YzctMDQ3Zi00ZGJkLWI3NWItM2RkNjA1NjY4YTM4XkEyXkFqcGdeQXVyMTUzMTg2ODkz._V1_SX300.jpg",
        "backdrop_url": "https://image.tmdb.org/t/p/original/jt3IhXjSlcDZgsiCv9k8GYCOo1o.jpg",
        "channel": "SK4FiLM Updates",
        "channel_id": -1002024811395,
        "date": "2023-10-20T16:45:00",
        "is_new": True,
        "qualities": [
            {
                "quality": "480p",
                "size": "950 MB",
                "file_id": f"{Config.FILE_CHANNEL_ID}_18001_480p",
                "available": True
            },
            {
                "quality": "720p",
                "size": "2.0 GB",
                "file_id": f"{Config.FILE_CHANNEL_ID}_18002_720p",
                "available": True
            },
            {
                "quality": "1080p",
                "size": "4.0 GB",
                "file_id": f"{Config.FILE_CHANNEL_ID}_18003_1080p",
                "available": True
            }
        ],
        "related": ["mission_impossible_1", "mission_impossible_6"]
    },
    {
        "id": "spiderman_nowayhome",
        "title": "Spider-Man: No Way Home",
        "year": "2021",
        "rating": "8.2",
        "duration": "148 min",
        "genres": ["Action", "Adventure", "Fantasy"],
        "description": "With Spider-Man identity now revealed, Peter asks Doctor Strange for help.",
        "poster_url": "https://m.media-amazon.com/images/M/MV5BZWMyYzFjYTYtNTRjYi00OGExLWE2YzgtOGRmYjAxZTU3NzBiXkEyXkFqcGdeQXVyMzQ0MzA0NTM@._V1_SX300.jpg",
        "backdrop_url": "https://image.tmdb.org/t/p/original/14QbnygCuTO0vl7CAFmPf1fgZfV.jpg",
        "channel": "SK4FiLM Main",
        "channel_id": Config.MAIN_CHANNEL_ID,
        "date": "2023-10-12T11:15:00",
        "is_new": False,
        "qualities": [
            {
                "quality": "480p",
                "size": "1.1 GB",
                "file_id": f"{Config.FILE_CHANNEL_ID}_15001_480p",
                "available": True
            },
            {
                "quality": "720p",
                "size": "2.3 GB",
                "file_id": f"{Config.FILE_CHANNEL_ID}_15002_720p",
                "available": True
            },
            {
                "quality": "1080p",
                "size": "4.5 GB",
                "file_id": f"{Config.FILE_CHANNEL_ID}_15003_1080p",
                "available": True
            }
        ],
        "related": ["spiderman_homecoming", "spiderman_farfromhome"]
    },
    {
        "id": "oppenheimer",
        "title": "Oppenheimer",
        "year": "2023",
        "rating": "8.3",
        "duration": "180 min",
        "genres": ["Biography", "Drama", "History"],
        "description": "The story of American scientist J. Robert Oppenheimer and his role in the atomic bomb.",
        "poster_url": "https://m.media-amazon.com/images/M/MV5BMDBmYTZjNjUtN2M1MS00MTQ2LTk2ODgtNzc2M2QyZGE5NTVjXkEyXkFqcGdeQXVyNzAwMjU2MTY@._V1_SX300.jpg",
        "backdrop_url": "https://image.tmdb.org/t/p/original/8b8R8l88Qje9dn9OE8PY05Nx1S8.jpg",
        "channel": "SK4FiLM Main",
        "channel_id": Config.MAIN_CHANNEL_ID,
        "date": "2023-10-25T09:45:00",
        "is_new": True,
        "qualities": [
            {
                "quality": "480p",
                "size": "1.3 GB",
                "file_id": f"{Config.FILE_CHANNEL_ID}_19001_480p",
                "available": True
            },
            {
                "quality": "720p",
                "size": "2.7 GB",
                "file_id": f"{Config.FILE_CHANNEL_ID}_19002_720p",
                "available": True
            },
            {
                "quality": "1080p",
                "size": "5.2 GB",
                "file_id": f"{Config.FILE_CHANNEL_ID}_19003_1080p",
                "available": True
            }
        ],
        "related": ["dunkirk", "interstellar", "tenet"]
    }
]

# ========== UTILITY FUNCTIONS ==========
def normalize_title(title: str) -> str:
    """Normalize movie title for search"""
    if not title:
        return ""
    title = title.lower()
    title = re.sub(r'[^\w\s]', '', title)
    title = re.sub(r'\s+', ' ', title).strip()
    return title

@lru_cache(maxsize=1000)
def normalize_title_cached(title: str) -> str:
    return normalize_title(title)

def get_client_ip() -> str:
    """Get client IP address"""
    if request:
        return request.remote_addr or "127.0.0.1"
    return "127.0.0.1"

# ========== API ENDPOINTS ==========
@app.route('/')
async def root():
    """Root endpoint with system info"""
    total_movies = len(MOVIE_DATABASE)
    new_movies = len([m for m in MOVIE_DATABASE if m.get("is_new", False)])
    
    return jsonify({
        "status": "healthy",
        "service": "SK4FiLM API v2.0",
        "version": "2.0-SESSION-GEN",
        "endpoints": {
            "movies": "/api/movies",
            "search": "/api/search?query=movie+name",
            "movie_detail": "/api/movie/{id}",
            "poster": "/api/poster?title=movie+name",
            "system": "/api/system/status",
            "session_generate": "/api/session/generate",
            "session_status": "/api/session/status/{session_id}",
            "telegram_login": "/api/telegram/login"
        },
        "statistics": {
            "total_movies": total_movies,
            "new_movies": new_movies,
            "channels": len(Config.TEXT_CHANNEL_IDS),
            "telegram": {
                "user_session": user_session_ready,
                "bot": bot_started
            }
        },
        "timestamp": datetime.now().isoformat()
    })

@app.route('/health')
async def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "uptime": time.time() - app_start_time if 'app_start_time' in globals() else 0
    })

@app.route('/api/movies', methods=['GET'])
async def api_movies():
    """Get movies with pagination"""
    try:
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 12))
        sort = request.args.get('sort', 'date')  # date, title, rating, new
        
        # Sort movies
        movies = MOVIE_DATABASE.copy()
        
        if sort == 'date':
            movies.sort(key=lambda x: x.get('date', ''), reverse=True)
        elif sort == 'title':
            movies.sort(key=lambda x: x.get('title', '').lower())
        elif sort == 'rating':
            movies.sort(key=lambda x: float(x.get('rating', 0)), reverse=True)
        elif sort == 'new':
            movies.sort(key=lambda x: (x.get('is_new', False), x.get('date', '')), reverse=True)
        
        # Paginate
        total = len(movies)
        total_pages = math.ceil(total / limit)
        start = (page - 1) * limit
        end = start + limit
        
        paginated_movies = movies[start:end]
        
        # Add download stats
        for movie in paginated_movies:
            movie['download_stats'] = {
                'total_downloads': 1000 + hash(movie['id']) % 5000,
                'last_week': 100 + hash(movie['id']) % 400
            }
        
        return jsonify({
            "status": "success",
            "movies": paginated_movies,
            "pagination": {
                "current_page": page,
                "total_pages": total_pages,
                "total_results": total,
                "per_page": limit,
                "has_next": page < total_pages,
                "has_previous": page > 1
            },
            "sorting": sort,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Movies API error: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/search', methods=['GET'])
async def api_search():
    """Search movies"""
    try:
        query = request.args.get('query', '').strip().lower()
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 12))
        
        if not query or len(query) < 2:
            return jsonify({
                "status": "error",
                "message": "Query must be at least 2 characters"
            }), 400
        
        # Search in movies
        results = []
        for movie in MOVIE_DATABASE:
            title_lower = movie['title'].lower()
            genres_lower = ' '.join(movie['genres']).lower() if movie['genres'] else ''
            
            if (query in title_lower or 
                query in genres_lower or
                any(query in genre.lower() for genre in movie['genres'])):
                results.append(movie)
        
        # Also search in normalized titles
        for movie in MOVIE_DATABASE:
            if movie not in results:
                norm_title = normalize_title_cached(movie['title'])
                if query in norm_title:
                    results.append(movie)
        
        # Remove duplicates
        seen_ids = set()
        unique_results = []
        for movie in results:
            if movie['id'] not in seen_ids:
                seen_ids.add(movie['id'])
                unique_results.append(movie)
        
        # Sort by relevance
        def relevance_score(movie):
            score = 0
            title_lower = movie['title'].lower()
            
            if query == title_lower:
                score += 100
            elif title_lower.startswith(query):
                score += 50
            elif query in title_lower:
                score += 30
            
            if movie.get('is_new', False):
                score += 20
            
            return score
        
        unique_results.sort(key=relevance_score, reverse=True)
        
        # Paginate
        total = len(unique_results)
        total_pages = math.ceil(total / limit) if total > 0 else 1
        start = (page - 1) * limit
        end = start + limit
        
        paginated_results = unique_results[start:end]
        
        return jsonify({
            "status": "success",
            "query": query,
            "results": paginated_results,
            "pagination": {
                "current_page": page,
                "total_pages": total_pages,
                "total_results": total,
                "per_page": limit,
                "has_next": page < total_pages,
                "has_previous": page > 1
            },
            "search_metadata": {
                "channels_searched": len(Config.TEXT_CHANNEL_IDS),
                "cache_hit": False
            },
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Search API error: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/movie/<movie_id>', methods=['GET'])
async def api_movie_detail(movie_id):
    """Get detailed movie information"""
    try:
        movie = next((m for m in MOVIE_DATABASE if m['id'] == movie_id), None)
        
        if not movie:
            return jsonify({
                "status": "error",
                "message": "Movie not found"
            }), 404
        
        # Get similar movies
        similar_movies = []
        for m in MOVIE_DATABASE:
            if m['id'] != movie_id and m['id'] in movie.get('related', []):
                similar_movies.append({
                    'id': m['id'],
                    'title': m['title'],
                    'year': m['year'],
                    'rating': m['rating'],
                    'poster_url': m['poster_url']
                })
        
        # If no related movies, get movies with same genres
        if not similar_movies:
            movie_genres = set(movie['genres'])
            for m in MOVIE_DATABASE:
                if m['id'] != movie_id and set(m['genres']) & movie_genres:
                    similar_movies.append({
                        'id': m['id'],
                        'title': m['title'],
                        'year': m['year'],
                        'rating': m['rating'],
                        'poster_url': m['poster_url']
                    })
                if len(similar_movies) >= 5:
                    break
        
        # Add download instructions
        download_info = {
            "instructions": "Click on any quality to download via Telegram bot",
            "bot_username": "sk4filmbot" if Config.BOT_TOKEN else "Not configured",
            "session_required": not user_session_ready,
            "session_generator": f"{Config.BACKEND_URL}/api/session/generate"
        }
        
        response_data = {
            "status": "success",
            "movie": movie,
            "similar_movies": similar_movies[:5],
            "download_info": download_info,
            "system_info": {
                "telegram_ready": user_session_ready,
                "bot_ready": bot_started,
                "session_generator": True
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Movie detail API error: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/poster', methods=['GET'])
async def api_poster():
    """Get movie poster information"""
    try:
        title = request.args.get('title', '').strip()
        year = request.args.get('year', '')
        
        if not title:
            return jsonify({
                "status": "error",
                "message": "Title is required"
            }), 400
        
        # Find movie in database
        movie = None
        for m in MOVIE_DATABASE:
            if title.lower() in m['title'].lower():
                movie = m
                break
        
        if movie:
            poster_data = {
                "poster_url": movie['poster_url'],
                "backdrop_url": movie['backdrop_url'],
                "source": "database",
                "rating": movie['rating'],
                "year": movie['year'],
                "title": movie['title'],
                "genres": movie['genres']
            }
        else:
            # Generate placeholder
            year_match = re.search(r'\b(19|20)\d{2}\b', title)
            detected_year = year_match.group() if year_match else year
            
            poster_data = {
                "poster_url": f"https://via.placeholder.com/300x450/1a1a1a/ffffff?text={title.replace(' ', '+')}",
                "backdrop_url": f"https://via.placeholder.com/1280x720/2a2a2a/ffffff?text={title.replace(' ', '+')}",
                "source": "placeholder",
                "rating": "0.0",
                "year": detected_year,
                "title": title,
                "genres": ["Movie"]
            }
        
        return jsonify({
            "status": "success",
            "poster": poster_data,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Poster API error: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# ========== SESSION GENERATOR ENDPOINTS ==========
@app.route('/api/session/generate', methods=['GET', 'POST'])
async def api_session_generate():
    """Generate new Telegram session"""
    try:
        user_ip = get_client_ip()
        user_agent = request.headers.get('User-Agent', 'Unknown')
        
        # Generate session
        result = await session_generator.create_session(user_ip, user_agent)
        
        if not result.get('success'):
            return jsonify({
                "status": "error",
                "message": result.get('error', 'Failed to generate session')
            }), 400
        
        return jsonify({
            "status": "success",
            "session": {
                "id": result['session_id'],
                "qr_code": result['qr_code'],
                "expires_in_hours": result['expires_in'],
                "instructions": result['instructions'],
                "created_at": datetime.now().isoformat()
            },
            "next_step": f"Check status at: /api/session/status/{result['session_id']}",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Session generation error: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/session/status/<session_id>', methods=['GET'])
async def api_session_status(session_id):
    """Check session status"""
    try:
        result = await session_generator.check_session_status(session_id)
        
        if not result.get('success'):
            return jsonify({
                "status": "error",
                "message": result.get('error', 'Session not found')
            }), 404
        
        response_data = {
            "status": "success",
            "session": {
                "id": session_id,
                "status": result['status'],
                "created_at": result.get('created_at'),
                "expires_at": result.get('expires_at')
            }
        }
        
        # Add session string if completed
        if result['status'] == 'completed':
            response_data['session']['session_string'] = result['session_string']
            response_data['session']['user_info'] = {
                "user_id": result.get('user_id'),
                "phone": result.get('phone_number'),
                "username": result.get('username')
            }
            response_data['usage_instructions'] = {
                "api_usage": f"Use this session string with Pyrogram client",
                "validity": f"Valid for {Config.SESSION_EXPIRY_HOURS} hours",
                "rate_limit": f"Max {Config.MAX_SESSIONS_PER_IP} sessions per IP per day"
            }
        
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Session status error: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/telegram/login', methods=['GET'])
async def api_telegram_login_page():
    """HTML page for Telegram login"""
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SK4FiLM - Telegram Login</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }
            
            .container {
                background: white;
                border-radius: 20px;
                padding: 40px;
                max-width: 500px;
                width: 100%;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            }
            
            .logo {
                text-align: center;
                margin-bottom: 30px;
            }
            
            .logo h1 {
                color: #333;
                font-size: 28px;
                margin-bottom: 5px;
            }
            
            .logo p {
                color: #666;
                font-size: 16px;
            }
            
            .qr-container {
                text-align: center;
                margin: 30px 0;
            }
            
            #qrCode {
                max-width: 300px;
                margin: 0 auto;
                border: 10px solid #f5f5f5;
                border-radius: 10px;
            }
            
            .instructions {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
            }
            
            .instructions h3 {
                color: #333;
                margin-bottom: 10px;
            }
            
            .instructions ol {
                padding-left: 20px;
                color: #555;
            }
            
            .instructions li {
                margin-bottom: 8px;
            }
            
            .status {
                text-align: center;
                margin: 20px 0;
                padding: 15px;
                border-radius: 10px;
                font-weight: bold;
            }
            
            .status.pending {
                background: #fff3cd;
                color: #856404;
                border: 1px solid #ffeaa7;
            }
            
            .status.success {
                background: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }
            
            .status.error {
                background: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }
            
            .buttons {
                display: flex;
                gap: 10px;
                margin-top: 20px;
            }
            
            .btn {
                flex: 1;
                padding: 12px;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                text-align: center;
                text-decoration: none;
            }
            
            .btn-primary {
                background: #4f46e5;
                color: white;
            }
            
            .btn-primary:hover {
                background: #4338ca;
                transform: translateY(-2px);
            }
            
            .btn-secondary {
                background: #6c757d;
                color: white;
            }
            
            .btn-secondary:hover {
                background: #5a6268;
                transform: translateY(-2px);
            }
            
            .loading {
                text-align: center;
                color: #666;
                font-size: 14px;
            }
            
            .session-info {
                background: #e8f4fd;
                padding: 15px;
                border-radius: 8px;
                margin-top: 20px;
                font-family: monospace;
                font-size: 12px;
                word-break: break-all;
                display: none;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="logo">
                <h1>🎬 SK4FiLM</h1>
                <p>Telegram Session Generator</p>
            </div>
            
            <div id="status" class="status pending">
                Generating session... Please wait
            </div>
            
            <div class="qr-container">
                <img id="qrCode" alt="QR Code will appear here">
            </div>
            
            <div class="instructions">
                <h3>📱 How to Login:</h3>
                <ol>
                    <li>Open Telegram on your phone</li>
                    <li>Go to Settings → Devices → Link Desktop Device</li>
                    <li>Scan the QR code above</li>
                    <li>Wait for confirmation below</li>
                    <li>Your session will be generated automatically</li>
                </ol>
            </div>
            
            <div class="buttons">
                <button class="btn btn-primary" onclick="generateSession()">
                    🔄 Generate New Session
                </button>
                <a href="/" class="btn btn-secondary">
                    🏠 Back to Home
                </a>
            </div>
            
            <div id="sessionInfo" class="session-info">
                <strong>Session String:</strong><br>
                <span id="sessionString"></span>
            </div>
            
            <div class="loading" id="loading">
                Checking session status...
            </div>
        </div>
        
        <script>
            let sessionId = null;
            let checkInterval = null;
            
            async function generateSession() {
                try {
                    // Show loading
                    document.getElementById('status').className = 'status pending';
                    document.getElementById('status').innerHTML = 'Generating session... Please wait';
                    document.getElementById('qrCode').src = '';
                    document.getElementById('sessionInfo').style.display = 'none';
                    document.getElementById('loading').style.display = 'block';
                    
                    // Generate new session
                    const response = await fetch('/api/session/generate', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        }
                    });
                    
                    const data = await response.json();
                    
                    if (data.status === 'success') {
                        sessionId = data.session.id;
                        
                        // Show QR code
                        document.getElementById('qrCode').src = data.session.qr_code;
                        document.getElementById('status').innerHTML = '✅ Session generated! Scan QR code with Telegram';
                        
                        // Start checking status
                        startStatusCheck();
                    } else {
                        document.getElementById('status').className = 'status error';
                        document.getElementById('status').innerHTML = '❌ Error: ' + (data.message || 'Failed to generate session');
                    }
                } catch (error) {
                    document.getElementById('status').className = 'status error';
                    document.getElementById('status').innerHTML = '❌ Network error. Please try again.';
                }
            }
            
            async function checkSessionStatus() {
                if (!sessionId) return;
                
                try {
                    const response = await fetch(`/api/session/status/${sessionId}`);
                    const data = await response.json();
                    
                    if (data.status === 'success') {
                        if (data.session.status === 'completed') {
                            // Session completed!
                            clearInterval(checkInterval);
                            document.getElementById('status').className = 'status success';
                            document.getElementById('status').innerHTML = '✅ Session generated successfully!';
                            
                            // Show session string
                            document.getElementById('sessionString').textContent = data.session.session_string;
                            document.getElementById('sessionInfo').style.display = 'block';
                            
                            document.getElementById('loading').innerHTML = `
                                <strong>🎉 Success!</strong><br>
                                Your Telegram session has been generated.<br>
                                Copy the session string above for API use.
                            `;
                        } else if (data.session.status === 'pending') {
                            document.getElementById('loading').innerHTML = 'Waiting for you to scan QR code...';
                        }
                    } else {
                        clearInterval(checkInterval);
                        document.getElementById('status').className = 'status error';
                        document.getElementById('status').innerHTML = '❌ Error: ' + (data.message || 'Session check failed');
                    }
                } catch (error) {
                    console.error('Status check error:', error);
                }
            }
            
            function startStatusCheck() {
                // Clear any existing interval
                if (checkInterval) {
                    clearInterval(checkInterval);
                }
                
                // Check every 3 seconds
                checkInterval = setInterval(checkSessionStatus, 3000);
                
                // First check immediately
                checkSessionStatus();
            }
            
            // Auto-generate session on page load
            window.onload = generateSession;
        </script>
    </body>
    </html>
    """
    
    return await render_template_string(html_template)

@app.route('/api/telegram/callback', methods=['POST'])
async def api_telegram_callback():
    """Webhook for Telegram login callback (called by external service)"""
    try:
        data = await request.get_json()
        
        if not data:
            return jsonify({
                "status": "error",
                "message": "No data received"
            }), 400
        
        session_id = data.get('session_id')
        user_data = data.get('user_data', {})
        
        if not session_id or not user_data:
            return jsonify({
                "status": "error",
                "message": "Missing session_id or user_data"
            }), 400
        
        # Process the session callback
        success = await session_generator.process_session_callback(session_id, user_data)
        
        if success:
            return jsonify({
                "status": "success",
                "message": "Session updated successfully"
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to update session"
            }), 400
            
    except Exception as e:
        logger.error(f"Telegram callback error: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/system/status', methods=['GET'])
async def api_system_status():
    """Get complete system status"""
    try:
        # Count sessions
        total_sessions = await sessions_col.count_documents({}) if sessions_col else 0
        active_sessions = await sessions_col.count_documents({"status": "completed"}) if sessions_col else 0
        
        # Count movies by channel
        channel_stats = {}
        for movie in MOVIE_DATABASE:
            channel = movie.get('channel', 'Unknown')
            channel_stats[channel] = channel_stats.get(channel, 0) + 1
        
        system_status = {
            "database": {
                "mongodb": db is not None,
                "redis": redis_client is not None,
                "collections": {
                    "movies": files_col is not None,
                    "sessions": sessions_col is not None,
                    "users": users_col is not None
                }
            },
            "telegram": {
                "user_client": user_session_ready,
                "bot_client": bot_started,
                "api_configured": bool(Config.API_ID and Config.API_HASH),
                "session_generator": True
            },
            "sessions": {
                "total_generated": total_sessions,
                "active_sessions": active_sessions,
                "max_per_ip": Config.MAX_SESSIONS_PER_IP,
                "expiry_hours": Config.SESSION_EXPIRY_HOURS
            },
            "movies": {
                "total": len(MOVIE_DATABASE),
                "new": len([m for m in MOVIE_DATABASE if m.get('is_new', False)]),
                "by_channel": channel_stats,
                "channels_monitored": len(Config.TEXT_CHANNEL_IDS)
            },
            "performance": {
                "start_time": app_start_time if 'app_start_time' in globals() else 0,
                "uptime": time.time() - app_start_time if 'app_start_time' in globals() else 0
            }
        }
        
        return jsonify({
            "status": "success",
            "system": system_status,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"System status error: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/channels', methods=['GET'])
async def api_channels():
    """Get channel information"""
    try:
        channels = [
            {
                "id": Config.MAIN_CHANNEL_ID,
                "name": "SK4FiLM Main",
                "type": "movie",
                "link": Config.MAIN_CHANNEL_LINK,
                "movie_count": len([m for m in MOVIE_DATABASE if m.get('channel') == "SK4FiLM Main"])
            },
            {
                "id": -1002024811395,
                "name": "SK4FiLM Updates",
                "type": "updates",
                "link": Config.UPDATES_CHANNEL_LINK,
                "movie_count": len([m for m in MOVIE_DATABASE if m.get('channel') == "SK4FiLM Updates"])
            },
            {
                "id": Config.FILE_CHANNEL_ID,
                "name": "SK4FiLM Files",
                "type": "files",
                "link": "https://t.me/SK4FiLM_Files",
                "movie_count": "Unknown"
            }
        ]
        
        return jsonify({
            "status": "success",
            "channels": channels,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Channels API error: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/download/instructions', methods=['GET'])
async def api_download_instructions():
    """Get download instructions"""
    try:
        instructions = {
            "method_1": {
                "title": "Using Telegram Bot",
                "steps": [
                    "1. Visit our website and search for movies",
                    "2. Click on download button for desired quality",
                    "3. File will be sent via Telegram bot automatically"
                ],
                "requirements": "Telegram account"
            },
            "method_2": {
                "title": "Using Session Generator",
                "steps": [
                    "1. Generate Telegram session using /api/session/generate",
                    "2. Scan QR code with Telegram app",
                    "3. Use session string with Pyrogram client",
                    "4. Access files directly from Telegram channels"
                ],
                "requirements": "Python/Pyrogram knowledge"
            },
            "method_3": {
                "title": "Direct Channel Access",
                "steps": [
                    f"1. Join our channel: {Config.MAIN_CHANNEL_LINK}",
                    "2. Browse movie posts",
                    "3. Use built-in Telegram download"
                ],
                "requirements": "Telegram account, channel membership"
            }
        }
        
        return jsonify({
            "status": "success",
            "instructions": instructions,
            "bot_username": "sk4filmbot" if Config.BOT_TOKEN else "Not configured",
            "session_generator": f"{Config.BACKEND_URL}/api/telegram/login",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Download instructions error: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# ========== ERROR HANDLERS ==========
@app.errorhandler(404)
async def not_found(error):
    return jsonify({
        "status": "error",
        "message": "Endpoint not found",
        "timestamp": datetime.now().isoformat()
    }), 404

@app.errorhandler(500)
async def server_error(error):
    return jsonify({
        "status": "error",
        "message": "Internal server error",
        "timestamp": datetime.now().isoformat()
    }), 500

# ========== STARTUP & SHUTDOWN ==========
app_start_time = 0

async def startup():
    """Initialize the application"""
    global app_start_time
    app_start_time = time.time()
    
    logger.info("🚀 Starting SK4FiLM API v2.0 with Session Generator...")
    logger.info("📱 Features: Telegram Session Generator, Movie API, Bot Integration")
    
    # Initialize databases
    await init_mongodb()
    await init_redis()
    
    # Initialize Telegram
    telegram_ok = await init_telegram_clients()
    
    logger.info("✅ SK4FiLM API Ready!")
    logger.info(f"📊 Statistics:")
    logger.info(f"   - Movies in database: {len(MOVIE_DATABASE)}")
    logger.info(f"   - Telegram User Session: {'✅ Ready' if user_session_ready else '❌ Not ready'}")
    logger.info(f"   - Telegram Bot: {'✅ Started' if bot_started else '❌ Not started'}")
    logger.info(f"   - Session Generator: ✅ Active")
    logger.info(f"   - API Endpoints: ✅ Ready")

async def shutdown():
    """Clean shutdown"""
    logger.info("🛑 Shutting down SK4FiLM API...")
    
    # Stop Telegram clients
    global User, Bot
    
    if user_session_ready and User:
        try:
            await User.stop()
            logger.info("✅ User session stopped")
        except:
            pass
    
    if bot_started and Bot:
        try:
            await Bot.stop()
            logger.info("✅ Bot stopped")
        except:
            pass
    
    # Close Redis
    if redis_client:
        try:
            await redis_client.close()
            logger.info("✅ Redis connection closed")
        except:
            pass
    
    logger.info("👋 Shutdown complete")

# ========== MAIN ENTRY POINT ==========
async def main():
    """Main async entry point"""
    await startup()
    
    try:
        # Configure Hypercorn
        config = HyperConfig()
        config.bind = [f"0.0.0.0:{Config.WEB_SERVER_PORT}"]
        config.workers = 1
        
        logger.info(f"🌐 Server starting on port {Config.WEB_SERVER_PORT}...")
        
        # Run server
        shutdown_event = asyncio.Event()
        await serve(app, config, shutdown_trigger=shutdown_event.wait)
        
    except KeyboardInterrupt:
        logger.info("🛑 Received shutdown signal")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
    finally:
        await shutdown()

if __name__ == "__main__":
    # Check environment
    logger.info("🔍 Environment Check:")
    logger.info(f"   - API_ID: {'✅ Configured' if Config.API_ID and Config.API_ID != 0 else '❌ Not configured'}")
    logger.info(f"   - API_HASH: {'✅ Configured' if Config.API_HASH else '❌ Not configured'}")
    logger.info(f"   - BOT_TOKEN: {'✅ Configured' if Config.BOT_TOKEN else '❌ Not configured'}")
    logger.info(f"   - USER_SESSION_STRING: {'✅ Configured' if Config.USER_SESSION_STRING else '❌ Not configured'}")
    logger.info(f"   - MONGODB_URI: {'✅ Configured' if Config.MONGODB_URI else '❌ Not configured'}")
    logger.info(f"   - PORT: {Config.WEB_SERVER_PORT}")
    logger.info(f"   - BACKEND_URL: {Config.BACKEND_URL}")
    
    # Run application
    asyncio.run(main())
