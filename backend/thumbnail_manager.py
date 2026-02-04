# ============================================================================
# thumbnail_manager.py - AUTOMATIC THUMBNAIL EXTRACTION & MANAGEMENT
# ============================================================================

import asyncio
import base64
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import hashlib
import aiohttp
from motor.motor_asyncio import AsyncIOMotorCollection

logger = logging.getLogger(__name__)

class ThumbnailManager:
    """Automatic thumbnail extraction and management system"""
    
    def __init__(self, mongo_client, config, bot_handler=None):
        self.mongo_client = mongo_client
        self.db = mongo_client.sk4film
        self.thumbnails_col = self.db.thumbnails
        self.config = config
        self.bot_handler = bot_handler
        
        # Thumbnail cache for fast access
        self.thumbnail_cache = {}
        self.cache_ttl = 3600  # 1 hour
        
        # Statistics
        self.stats = {
            'total_extracted': 0,
            'total_cached': 0,
            'total_fetched_from_api': 0,
            'total_fallback': 0,
            'cleanup_count': 0
        }
        
        # Start cleanup task
        self.cleanup_task = None
    
    async def initialize(self):
        """Initialize thumbnail manager"""
        logger.info("🖼️ Initializing Thumbnail Manager...")
        
        # Create indexes
        await self.create_indexes()
        
        # Start cleanup task
        self.cleanup_task = asyncio.create_task(self.periodic_cleanup())
        
        logger.info("✅ Thumbnail Manager initialized")
        return True
    
    async def create_indexes(self):
        """Create MongoDB indexes for thumbnails collection"""
        try:
            # Create TTL index for automatic deletion
            await self.thumbnails_col.create_index(
                [("last_accessed", 1)],
                expireAfterSeconds=30 * 24 * 60 * 60,  # 30 days
                name="ttl_index",
                background=True
            )
            
            # Create unique index for message_id + channel_id
            await self.thumbnails_col.create_index(
                [("channel_id", 1), ("message_id", 1)],
                unique=True,
                name="message_unique",
                background=True
            )
            
            # Create index for normalized_title
            await self.thumbnails_col.create_index(
                [("normalized_title", 1)],
                name="title_index",
                background=True
            )
            
            logger.info("✅ Thumbnail indexes created")
            
        except Exception as e:
            logger.error(f"❌ Error creating thumbnail indexes: {e}")
    
    async def extract_thumbnail_from_telegram(self, channel_id: int, message_id: int) -> Optional[str]:
        """
        Extract thumbnail directly from Telegram message
        Returns base64 data URL or None
        """
        try:
            logger.info(f"📸 Extracting thumbnail from Telegram: {channel_id}/{message_id}")
            
            # Try bot handler first
            if self.bot_handler and self.bot_handler.initialized:
                thumbnail_url = await self.bot_handler.extract_thumbnail(channel_id, message_id)
                if thumbnail_url:
                    logger.info(f"✅ Thumbnail extracted via bot handler: {channel_id}/{message_id}")
                    return thumbnail_url
            
            # Try direct Telegram API
            try:
                from pyrogram import Client
                
                # Get global Bot session
                global Bot
                if Bot is not None:
                    message = await Bot.get_messages(channel_id, message_id)
                    if not message:
                        return None
                    
                    thumbnail_data = None
                    
                    if message.video:
                        if hasattr(message.video, 'thumbnail') and message.video.thumbnail:
                            thumbnail_file_id = message.video.thumbnail.file_id
                            download_path = await Bot.download_media(thumbnail_file_id, in_memory=True)
                            
                            if download_path:
                                if isinstance(download_path, bytes):
                                    thumbnail_data = download_path
                                else:
                                    with open(download_path, 'rb') as f:
                                        thumbnail_data = f.read()
                    
                    elif message.document and self.is_video_file(message.document.file_name or ''):
                        if hasattr(message.document, 'thumbnail') and message.document.thumbnail:
                            thumbnail_file_id = message.document.thumbnail.file_id
                            download_path = await Bot.download_media(thumbnail_file_id, in_memory=True)
                            
                            if download_path:
                                if isinstance(download_path, bytes):
                                    thumbnail_data = download_path
                                else:
                                    with open(download_path, 'rb') as f:
                                        thumbnail_data = f.read()
                    
                    if thumbnail_data:
                        base64_data = base64.b64encode(thumbnail_data).decode('utf-8')
                        thumbnail_url = f"data:image/jpeg;base64,{base64_data}"
                        logger.info(f"✅ Telegram thumbnail extracted: {channel_id}/{message_id}")
                        return thumbnail_url
                    
            except Exception as e:
                logger.error(f"❌ Telegram thumbnail extraction error: {e}")
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Thumbnail extraction failed: {e}")
            return None
    
    def is_video_file(self, filename: str) -> bool:
        """Check if file is video"""
        if not filename:
            return False
        video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v']
        return any(filename.lower().endswith(ext) for ext in video_extensions)
    
    async def get_thumbnail_for_movie(self, title: str, channel_id: int = None, message_id: int = None) -> Dict[str, Any]:
        """
        Get thumbnail for movie with fallback system
        1. Check MongoDB for existing thumbnail
        2. Extract from Telegram if message_id provided
        3. Fetch from TMDB/OMDB
        4. Return fallback image
        """
        try:
            # Generate cache key
            cache_key = f"{title}_{channel_id}_{message_id}"
            
            # Check in-memory cache first
            if cache_key in self.thumbnail_cache:
                cached_data = self.thumbnail_cache[cache_key]
                if time.time() - cached_data['timestamp'] < self.cache_ttl:
                    logger.debug(f"✅ Thumbnail cache hit: {title}")
                    self.stats['total_cached'] += 1
                    return cached_data['data']
            
            normalized_title = self.normalize_title(title)
            
            # 1. Check MongoDB for existing thumbnail
            thumbnail_data = await self._get_from_mongodb(normalized_title, channel_id, message_id)
            if thumbnail_data:
                # Update cache
                self.thumbnail_cache[cache_key] = {
                    'data': thumbnail_data,
                    'timestamp': time.time()
                }
                return thumbnail_data
            
            # 2. Try to extract from Telegram if message_id is provided
            if channel_id and message_id:
                telegram_thumbnail = await self.extract_thumbnail_from_telegram(channel_id, message_id)
                if telegram_thumbnail:
                    # Save to MongoDB
                    await self._save_to_mongodb(
                        normalized_title=normalized_title,
                        thumbnail_url=telegram_thumbnail,
                        source='telegram',
                        channel_id=channel_id,
                        message_id=message_id
                    )
                    
                    thumbnail_data = {
                        'thumbnail_url': telegram_thumbnail,
                        'source': 'telegram',
                        'has_thumbnail': True,
                        'extracted': True
                    }
                    
                    # Update cache
                    self.thumbnail_cache[cache_key] = {
                        'data': thumbnail_data,
                        'timestamp': time.time()
                    }
                    
                    self.stats['total_extracted'] += 1
                    logger.info(f"✅ Thumbnail extracted and saved: {title}")
                    return thumbnail_data
            
            # 3. Try to fetch from TMDB/OMDB
            api_thumbnail = await self._fetch_from_api(title)
            if api_thumbnail:
                # Save to MongoDB
                await self._save_to_mongodb(
                    normalized_title=normalized_title,
                    thumbnail_url=api_thumbnail['url'],
                    source=api_thumbnail['source'],
                    channel_id=channel_id,
                    message_id=message_id
                )
                
                thumbnail_data = {
                    'thumbnail_url': api_thumbnail['url'],
                    'source': api_thumbnail['source'],
                    'has_thumbnail': True,
                    'extracted': True
                }
                
                # Update cache
                self.thumbnail_cache[cache_key] = {
                    'data': thumbnail_data,
                    'timestamp': time.time()
                }
                
                self.stats['total_fetched_from_api'] += 1
                logger.info(f"✅ Thumbnail fetched from API: {title} - {api_thumbnail['source']}")
                return thumbnail_data
            
            # 4. Return fallback image
            fallback_data = {
                'thumbnail_url': self.config.FALLBACK_POSTER,
                'source': 'fallback',
                'has_thumbnail': True,
                'extracted': False
            }
            
            # Update cache with fallback
            self.thumbnail_cache[cache_key] = {
                'data': fallback_data,
                'timestamp': time.time()
            }
            
            self.stats['total_fallback'] += 1
            logger.info(f"⚠️ Using fallback thumbnail for: {title}")
            return fallback_data
            
        except Exception as e:
            logger.error(f"❌ Error getting thumbnail for {title}: {e}")
            return {
                'thumbnail_url': self.config.FALLBACK_POSTER,
                'source': 'fallback',
                'has_thumbnail': True,
                'extracted': False
            }
    
    async def _get_from_mongodb(self, normalized_title: str, channel_id: int = None, message_id: int = None) -> Optional[Dict]:
        """Get thumbnail from MongoDB"""
        try:
            query = {"normalized_title": normalized_title}
            
            # Prioritize by message_id if provided
            if channel_id and message_id:
                query["channel_id"] = channel_id
                query["message_id"] = message_id
            
            # Find thumbnail
            thumbnail_doc = await self.thumbnails_col.find_one(query)
            
            if thumbnail_doc:
                # Update last_accessed timestamp
                await self.thumbnails_col.update_one(
                    {"_id": thumbnail_doc["_id"]},
                    {"$set": {"last_accessed": datetime.now()}}
                )
                
                return {
                    'thumbnail_url': thumbnail_doc.get('thumbnail_url'),
                    'source': thumbnail_doc.get('source', 'unknown'),
                    'has_thumbnail': True,
                    'extracted': thumbnail_doc.get('extracted', False)
                }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ MongoDB thumbnail fetch error: {e}")
            return None
    
    async def _save_to_mongodb(self, normalized_title: str, thumbnail_url: str, 
                              source: str, channel_id: int = None, message_id: int = None):
        """Save thumbnail to MongoDB"""
        try:
            thumbnail_doc = {
                'normalized_title': normalized_title,
                'thumbnail_url': thumbnail_url,
                'source': source,
                'extracted': True,
                'created_at': datetime.now(),
                'last_accessed': datetime.now()
            }
            
            if channel_id:
                thumbnail_doc['channel_id'] = channel_id
            if message_id:
                thumbnail_doc['message_id'] = message_id
            
            # Create unique key for message-based thumbnails
            if channel_id and message_id:
                await self.thumbnails_col.update_one(
                    {
                        'channel_id': channel_id,
                        'message_id': message_id
                    },
                    {
                        '$set': thumbnail_doc,
                        '$setOnInsert': {
                            'first_seen': datetime.now()
                        }
                    },
                    upsert=True
                )
            else:
                # Title-based thumbnail
                await self.thumbnails_col.update_one(
                    {
                        'normalized_title': normalized_title,
                        'source': source
                    },
                    {
                        '$set': thumbnail_doc,
                        '$setOnInsert': {
                            'first_seen': datetime.now()
                        }
                    },
                    upsert=True
                )
            
            logger.debug(f"✅ Thumbnail saved to MongoDB: {normalized_title}")
            
        except Exception as e:
            logger.error(f"❌ Error saving thumbnail to MongoDB: {e}")
    
    async def _fetch_from_api(self, title: str) -> Optional[Dict]:
        """Fetch thumbnail from TMDB/OMDB APIs"""
        try:
            # Clean title for API search
            clean_title = self.clean_title_for_api(title)
            
            # Try TMDB first
            tmdb_thumbnail = await self._fetch_from_tmdb(clean_title)
            if tmdb_thumbnail:
                return {
                    'url': tmdb_thumbnail,
                    'source': 'tmdb'
                }
            
            # Try OMDB as fallback
            omdb_thumbnail = await self._fetch_from_omdb(clean_title)
            if omdb_thumbnail:
                return {
                    'url': omdb_thumbnail,
                    'source': 'omdb'
                }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ API thumbnail fetch error: {e}")
            return None
    
    async def _fetch_from_tmdb(self, title: str) -> Optional[str]:
        """Fetch poster from TMDB"""
        try:
            if not self.config.TMDB_API_KEY:
                return None
            
            async with aiohttp.ClientSession() as session:
                url = f"https://api.themoviedb.org/3/search/movie"
                params = {
                    'api_key': self.config.TMDB_API_KEY,
                    'query': title,
                    'language': 'en-US',
                    'page': 1
                }
                
                async with session.get(url, params=params, timeout=5) as response:
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
    
    async def _fetch_from_omdb(self, title: str) -> Optional[str]:
        """Fetch poster from OMDB"""
        try:
            if not self.config.OMDB_API_KEY:
                return None
            
            async with aiohttp.ClientSession() as session:
                url = "http://www.omdbapi.com/"
                params = {
                    't': title,
                    'apikey': self.config.OMDB_API_KEY,
                    'plot': 'short'
                }
                
                async with session.get(url, params=params, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('Poster') and data['Poster'] != 'N/A':
                            return data['Poster']
            
            return None
            
        except Exception as e:
            logger.debug(f"OMDB fetch error: {e}")
            return None
    
    def normalize_title(self, title: str) -> str:
        """Normalize title for consistent matching"""
        if not title:
            return ""
        
        # Convert to lowercase
        title = title.lower().strip()
        
        # Remove year in parentheses
        title = re.sub(r'\s*\(\d{4}\)', '', title)
        
        # Remove quality indicators
        title = re.sub(r'\b(480p|720p|1080p|2160p|4k|hd|hevc|x265|x264|web|bluray|dvd|hdtv)\b', '', title, flags=re.IGNORECASE)
        
        # Remove special characters and extra spaces
        title = re.sub(r'[^\w\s]', ' ', title)
        title = re.sub(r'\s+', ' ', title)
        
        return title.strip()
    
    def clean_title_for_api(self, title: str) -> str:
        """Clean title for API search"""
        if not title:
            return ""
        
        # Remove year if present
        title = re.sub(r'\s*\(\d{4}\)', '', title)
        
        # Remove quality and format info
        title = re.sub(r'\b(720p|1080p|2160p|4k|hd|hevc|bluray|webrip|hdtv)\b', '', title, flags=re.IGNORECASE)
        
        # Remove special characters but keep spaces
        title = re.sub(r'[^\w\s\-]', '', title)
        title = re.sub(r'\s+', ' ', title)
        
        return title.strip()
    
    async def get_thumbnails_batch(self, movies: List[Dict]) -> List[Dict]:
        """Get thumbnails for multiple movies in batch"""
        try:
            # Prepare tasks for all movies
            tasks = []
            for movie in movies:
                title = movie.get('title', '')
                channel_id = movie.get('channel_id')
                message_id = movie.get('message_id')
                
                task = asyncio.create_task(
                    self.get_thumbnail_for_movie(title, channel_id, message_id)
                )
                tasks.append((movie, task))
            
            # Process results
            results = []
            for movie, task in tasks:
                try:
                    thumbnail_data = await task
                    
                    # Merge thumbnail data with movie data
                    movie_with_thumbnail = movie.copy()
                    movie_with_thumbnail.update(thumbnail_data)
                    
                    # Ensure thumbnail_url is always present
                    if not movie_with_thumbnail.get('thumbnail_url'):
                        movie_with_thumbnail['thumbnail_url'] = self.config.FALLBACK_POSTER
                        movie_with_thumbnail['source'] = 'fallback'
                    
                    results.append(movie_with_thumbnail)
                    
                except Exception as e:
                    logger.error(f"❌ Batch thumbnail error for {movie.get('title')}: {e}")
                    
                    # Add fallback
                    movie_with_fallback = movie.copy()
                    movie_with_fallback.update({
                        'thumbnail_url': self.config.FALLBACK_POSTER,
                        'source': 'fallback',
                        'has_thumbnail': True,
                        'extracted': False
                    })
                    results.append(movie_with_fallback)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Batch thumbnails error: {e}")
            return movies
    
    async def periodic_cleanup(self):
        """Periodic cleanup of orphaned thumbnails"""
        while True:
            try:
                await asyncio.sleep(24 * 60 * 60)  # Run daily
                await self.cleanup_orphaned_thumbnails()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Cleanup task error: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour on error
    
    async def cleanup_orphaned_thumbnails(self):
        """Remove orphaned thumbnails (no corresponding file in files collection)"""
        try:
            logger.info("🧹 Cleaning up orphaned thumbnails...")
            
            # Get all thumbnails with channel_id and message_id
            cursor = self.thumbnails_col.find(
                {
                    "channel_id": {"$exists": True},
                    "message_id": {"$exists": True}
                },
                {"channel_id": 1, "message_id": 1, "_id": 1}
            )
            
            orphaned_count = 0
            async for doc in cursor:
                channel_id = doc.get('channel_id')
                message_id = doc.get('message_id')
                
                # Check if file exists in files collection
                from app import files_col
                if files_col:
                    file_exists = await files_col.find_one({
                        "channel_id": channel_id,
                        "message_id": message_id
                    })
                    
                    # If file doesn't exist, delete thumbnail
                    if not file_exists:
                        await self.thumbnails_col.delete_one({"_id": doc["_id"]})
                        orphaned_count += 1
            
            self.stats['cleanup_count'] += orphaned_count
            
            if orphaned_count > 0:
                logger.info(f"✅ Cleaned up {orphaned_count} orphaned thumbnails")
            else:
                logger.info("✅ No orphaned thumbnails found")
                
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get thumbnail manager statistics"""
        try:
            total_thumbnails = await self.thumbnails_col.count_documents({})
            
            # Get source distribution
            pipeline = [
                {"$group": {"_id": "$source", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]
            source_dist = await self.thumbnails_col.aggregate(pipeline).to_list(length=10)
            
            # Get recent thumbnails
            recent = await self.thumbnails_col.find(
                {},
                {"normalized_title": 1, "source": 1, "created_at": 1, "_id": 0}
            ).sort("created_at", -1).limit(5).to_list(length=5)
            
            return {
                'total_thumbnails': total_thumbnails,
                'source_distribution': source_dist,
                'recent_thumbnails': recent,
                'cache_stats': {
                    'cache_size': len(self.thumbnail_cache),
                    'total_extracted': self.stats['total_extracted'],
                    'total_cached': self.stats['total_cached'],
                    'total_api_fetches': self.stats['total_fetched_from_api'],
                    'total_fallback': self.stats['total_fallback'],
                    'total_cleanup': self.stats['cleanup_count']
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Stats error: {e}")
            return {}
    
    async def shutdown(self):
        """Shutdown thumbnail manager"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("✅ Thumbnail Manager shutdown complete")

# Helper function to check if file is video
def is_video_file(filename: str) -> bool:
    """Check if file is video"""
    if not filename:
        return False
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v']
    return any(filename.lower().endswith(ext) for ext in video_extensions)
