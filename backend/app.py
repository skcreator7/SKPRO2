"""
SK4FiLM - Main Application
Flask Web Server + Telegram Bot Integration
FIXED: No circular imports - all utilities moved to utils.py
"""

import os
import logging
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any, List
import re

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from motor.motor_asyncio import AsyncIOMotorClient
import redis.asyncio as aioredis

# Import from config and utils (NO circular imports!)
from config import Config
from utils import (
    normalize_title, extract_title_from_file, extract_title_smart,
    format_size, detect_quality, is_video_file,
    safe_telegram_operation, safe_telegram_generator,
    auto_delete_file, index_single_file
)

# Import bot components
from bot_handlers import SK4FiLMBot, setup_bot_handlers

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== FLASK APP INITIALIZATION ====================

app = Flask(__name__)
CORS(app)

# Global variables
db_client: Optional[AsyncIOMotorClient] = None
db = None
redis_client: Optional[aioredis.Redis] = None
bot_instance: Optional[SK4FiLMBot] = None


# ==================== DATABASE MANAGER CLASS ====================

class DatabaseManager:
    """Manages MongoDB connections and operations"""
    
    def __init__(self, mongodb_uri: str, database_name: str):
        self.mongodb_uri = mongodb_uri
        self.database_name = database_name
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None
        self.collections = {}
    
    async def connect(self):
        """Connect to MongoDB"""
        try:
            self.client = AsyncIOMotorClient(self.mongodb_uri)
            self.db = self.client[self.database_name]
            
            # Initialize collections
            self.collections = {
                'files': self.db.files,
                'users': self.db.users,
                'premium': self.db.premium_users,
                'verification': self.db.verification_tokens,
                'downloads': self.db.download_logs,
                'search_cache': self.db.search_cache
            }
            
            # Test connection
            await self.client.admin.command('ping')
            logger.info("✅ MongoDB connected successfully")
            
            # Create indexes
            await self._create_indexes()
            
            return True
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            return False
    
    async def _create_indexes(self):
        """Create database indexes for better performance"""
        try:
            # Files collection indexes
            await self.collections['files'].create_index("file_id", unique=True)
            await self.collections['files'].create_index("normalized_title")
            await self.collections['files'].create_index([("title", "text")])
            await self.collections['files'].create_index("quality")
            
            # Users collection indexes
            await self.collections['users'].create_index("user_id", unique=True)
            
            # Premium users indexes
            await self.collections['premium'].create_index("user_id", unique=True)
            await self.collections['premium'].create_index("expiry_date")
            
            # Verification tokens indexes
            await self.collections['verification'].create_index("token", unique=True)
            await self.collections['verification'].create_index("expiry_date")
            
            # Download logs indexes
            await self.collections['downloads'].create_index([("user_id", 1), ("timestamp", -1)])
            
            logger.info("✅ Database indexes created")
        except Exception as e:
            logger.error(f"Index creation error: {e}")
    
    async def store_file(self, file_data: Dict[str, Any]) -> bool:
        """Store file metadata in database"""
        try:
            await self.collections['files'].update_one(
                {'file_id': file_data['file_id']},
                {'$set': file_data},
                upsert=True
            )
            return True
        except Exception as e:
            logger.error(f"Error storing file: {e}")
            return False
    
    async def search_files(self, query: str, limit: int = 50) -> List[Dict]:
        """Search files by title"""
        try:
            normalized_query = normalize_title(query)
            
            # Text search
            results = await self.collections['files'].find(
                {'$text': {'$search': query}},
                {'score': {'$meta': 'textScore'}}
            ).sort([('score', {'$meta': 'textScore'})]).limit(limit).to_list(length=limit)
            
            # If no results, try normalized title match
            if not results:
                results = await self.collections['files'].find(
                    {'normalized_title': {'$regex': normalized_query, '$options': 'i'}}
                ).limit(limit).to_list(length=limit)
            
            return results
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    async def get_file_by_id(self, file_id: str) -> Optional[Dict]:
        """Get file by file_id"""
        try:
            return await self.collections['files'].find_one({'file_id': file_id})
        except Exception as e:
            logger.error(f"Error getting file: {e}")
            return None
    
    async def get_user(self, user_id: int) -> Optional[Dict]:
        """Get user data"""
        try:
            return await self.collections['users'].find_one({'user_id': user_id})
        except Exception as e:
            logger.error(f"Error getting user: {e}")
            return None
    
    async def update_user(self, user_id: int, user_data: Dict) -> bool:
        """Update user data"""
        try:
            await self.collections['users'].update_one(
                {'user_id': user_id},
                {'$set': user_data},
                upsert=True
            )
            return True
        except Exception as e:
            logger.error(f"Error updating user: {e}")
            return False
    
    async def log_download(self, user_id: int, file_id: str, file_name: str) -> bool:
        """Log download activity"""
        try:
            log_entry = {
                'user_id': user_id,
                'file_id': file_id,
                'file_name': file_name,
                'timestamp': datetime.utcnow(),
                'ip_address': request.remote_addr if request else None
            }
            await self.collections['downloads'].insert_one(log_entry)
            return True
        except Exception as e:
            logger.error(f"Error logging download: {e}")
            return False
    
    async def close(self):
        """Close database connection"""
        if self.client:
            self.client.close()
            logger.info("✅ MongoDB connection closed")


# ==================== CACHE MANAGER CLASS ====================

class CacheManager:
    """Manages Redis cache operations"""
    
    def __init__(self, host: str, port: int, password: str = None, db: int = 0):
        self.host = host
        self.port = port
        self.password = password
        self.db = db
        self.client: Optional[aioredis.Redis] = None
        self.redis_enabled = False
    
    async def connect(self):
        """Connect to Redis"""
        try:
            self.client = await aioredis.from_url(
                f"redis://{self.host}:{self.port}/{self.db}",
                password=self.password if self.password else None,
                encoding="utf-8",
                decode_responses=True
            )
            
            # Test connection
            await self.client.ping()
            self.redis_enabled = True
            logger.info("✅ Redis connected successfully")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed (continuing without cache): {e}")
            self.redis_enabled = False
            return False
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from cache"""
        if not self.redis_enabled:
            return None
        try:
            return await self.client.get(key)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    async def set(self, key: str, value: str, expire: int = None) -> bool:
        """Set value in cache"""
        if not self.redis_enabled:
            return False
        try:
            if expire:
                await self.client.setex(key, expire, value)
            else:
                await self.client.set(key, value)
            return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self.redis_enabled:
            return False
        try:
            await self.client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        if not self.redis_enabled:
            return False
        try:
            return await self.client.exists(key) > 0
        except Exception as e:
            logger.error(f"Cache exists error: {e}")
            return False
    
    async def close(self):
        """Close Redis connection"""
        if self.client:
            await self.client.close()
            logger.info("✅ Redis connection closed")


# ==================== APPLICATION INITIALIZATION ====================

async def initialize_app():
    """Initialize all application components"""
    global db_client, db, redis_client, bot_instance
    
    try:
        logger.info("🚀 Starting SK4FiLM Application...")
        
        # Validate configuration
        Config.validate()
        logger.info("✅ Configuration validated")
        
        # Initialize database
        db_manager = DatabaseManager(Config.MONGODB_URI, Config.DATABASE_NAME)
        if await db_manager.connect():
            db_client = db_manager.client
            db = db_manager.db
            logger.info("✅ Database initialized")
        else:
            logger.error("❌ Database initialization failed")
            return False
        
        # Initialize cache
        cache_manager = CacheManager(
            Config.REDIS_HOST,
            Config.REDIS_PORT,
            Config.REDIS_PASSWORD,
            Config.REDIS_DB
        )
        await cache_manager.connect()
        redis_client = cache_manager.client
        
        # Initialize bot
        bot_instance = SK4FiLMBot(Config, db_manager)
        bot_instance.cache_manager = cache_manager
        
        if await bot_instance.initialize():
            logger.info("✅ Bot initialized")
        else:
            logger.error("❌ Bot initialization failed")
            return False
        
        # Setup bot handlers
        await setup_bot_handlers(bot_instance.bot, bot_instance)
        logger.info("✅ Bot handlers registered")
        
        logger.info("🎉 Application started successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Application initialization failed: {e}")
        return False


async def shutdown_app():
    """Cleanup on shutdown"""
    global db_client, redis_client, bot_instance
    
    try:
        logger.info("🛑 Shutting down application...")
        
        if bot_instance:
            await bot_instance.shutdown()
        
        if redis_client:
            await redis_client.close()
        
        if db_client:
            db_client.close()
        
        logger.info("✅ Application shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# ==================== FLASK ROUTES ====================

@app.route('/')
def home():
    """Home page"""
    return jsonify({
        'status': 'running',
        'service': 'SK4FiLM API',
        'version': '1.0.0',
        'endpoints': {
            'search': '/api/search?q=<query>',
            'file': '/api/file/<file_id>',
            'health': '/health'
        }
    })


@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'bot': bot_instance.bot_started if bot_instance else False,
        'database': db is not None,
        'cache': redis_client is not None,
        'timestamp': datetime.utcnow().isoformat()
    })


@app.route('/api/search')
def search_api():
    """Search API endpoint"""
    query = request.args.get('q', '').strip()
    
    if not query:
        return jsonify({'error': 'Query parameter required'}), 400
    
    try:
        # Run async search in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Search in database
        db_manager = DatabaseManager(Config.MONGODB_URI, Config.DATABASE_NAME)
        results = loop.run_until_complete(db_manager.search_files(query, limit=50))
        
        return jsonify({
            'success': True,
            'query': query,
            'results': results,
            'count': len(results)
        })
    
    except Exception as e:
        logger.error(f"Search API error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/file/<file_id>')
def get_file_api(file_id):
    """Get file details API endpoint"""
    try:
        # Run async get in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        db_manager = DatabaseManager(Config.MONGODB_URI, Config.DATABASE_NAME)
        file_data = loop.run_until_complete(db_manager.get_file_by_id(file_id))
        
        if not file_data:
            return jsonify({'error': 'File not found'}), 404
        
        return jsonify({
            'success': True,
            'file': file_data
        })
    
    except Exception as e:
        logger.error(f"Get file API error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/verify/<token>')
def verify_token_api(token):
    """Verify user token"""
    # TODO: Implement verification logic
    return jsonify({
        'success': True,
        'message': 'Token verified',
        'valid_until': datetime.utcnow().isoformat()
    })


@app.route('/api/stats')
def stats_api():
    """Get system statistics"""
    try:
        return jsonify({
            'success': True,
            'stats': {
                'bot_status': bot_instance.bot_started if bot_instance else False,
                'user_session': bot_instance.user_session_ready if bot_instance else False,
                'database': db is not None,
                'cache': redis_client is not None,
                'uptime': 'N/A'  # TODO: Track actual uptime
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== UTILITY FUNCTIONS (Keep all existing) ====================

def create_verification_link(user_id: int) -> str:
    """Create verification link for user"""
    token = secrets.token_urlsafe(16)
    # TODO: Store token in database
    return f"https://t.me/{Config.BOT_USERNAME}?start=verify_{token}"


def validate_file_request(user_id: int, file_id: str) -> tuple[bool, str]:
    """Validate if user can download file"""
    # TODO: Implement validation logic
    return True, "Access granted"


async def get_file_from_channel(channel_id: int, message_id: int):
    """Get file from Telegram channel"""
    if not bot_instance or not bot_instance.user_session_ready:
        return None
    
    try:
        message = await safe_telegram_operation(
            bot_instance.user_client.get_messages,
            channel_id,
            message_id
        )
        return message
    except Exception as e:
        logger.error(f"Error getting file from channel: {e}")
        return None


async def send_file_to_user(user_id: int, file_message, quality: str = "HD"):
    """Send file to user"""
    if not bot_instance or not bot_instance.bot_started:
        return None
    
    try:
        caption = (
            f"♻ **Please forward to saved messages**\n\n"
            f"📹 Quality: {quality}\n"
            f"⚠️ Auto-delete in {Config.AUTO_DELETE_TIME//60} minutes\n\n"
            f"@{Config.BOT_USERNAME} 🍿"
        )
        
        if file_message.document:
            sent = await safe_telegram_operation(
                bot_instance.bot.send_document,
                user_id,
                file_message.document.file_id,
                caption=caption
            )
        elif file_message.video:
            sent = await safe_telegram_operation(
                bot_instance.bot.send_video,
                user_id,
                file_message.video.file_id,
                caption=caption
            )
        else:
            return None
        
        # Auto-delete after specified time
        if Config.AUTO_DELETE_TIME > 0:
            asyncio.create_task(auto_delete_file(sent, Config.AUTO_DELETE_TIME))
        
        return sent
    
    except Exception as e:
        logger.error(f"Error sending file to user: {e}")
        return None


def parse_file_link(link: str) -> Optional[Dict[str, Any]]:
    """Parse file download link"""
    try:
        # Format: channel_message_quality
        parts = link.strip().split('_')
        if len(parts) >= 2:
            return {
                'channel_id': int(parts[0]),
                'message_id': int(parts[1]),
                'quality': parts[2] if len(parts) > 2 else 'HD'
            }
    except Exception as e:
        logger.error(f"Error parsing file link: {e}")
    return None


# ==================== MAIN ENTRY POINT ====================

if __name__ == '__main__':
    # Initialize app
    loop = asyncio.get_event_loop()
    success = loop.run_until_complete(initialize_app())
    
    if not success:
        logger.error("Failed to initialize application")
        exit(1)
    
    # Run Flask app
    try:
        app.run(
            host='0.0.0.0',
            port=Config.PORT,
            debug=False
        )
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        loop.run_until_complete(shutdown_app())
