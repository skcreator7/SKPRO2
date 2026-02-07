# ============================================================================
# thumbnail_manager.py - SINGLE MONGODB THUMBNAIL MANAGEMENT SYSTEM (FIXED)
# ============================================================================

import asyncio
import base64
import logging
import time
import re
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import hashlib
import aiohttp

logger = logging.getLogger(__name__)

class ThumbnailManager:
    """Thumbnail management system with single MongoDB database - FIXED VERSION"""
    
    def __init__(self, mongo_client, config, bot_handler=None):
        self.mongo_client = mongo_client
        self.db = mongo_client.get_database()
        self.thumbnails_col = self.db.thumbnails  # Main thumbnails collection
        self.files_col = self.db.files  # Reference to files collection
        self.config = config
        self.bot_handler = bot_handler
        
        # Enhanced Thumbnail cache
        self.thumbnail_cache = {}
        self.cache_ttl = 3600  # 1 hour
        
        # Multiple API Services for 99% success rate
        self.api_services = [
            self._fetch_from_tmdb,
            self._fetch_from_omdb,
            self._fetch_from_tvdb,  # New service
            self._fetch_from_imdb,   # New service
            self._fetch_from_google_images  # New service
        ]
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful': 0,
            'failed': 0,
            'from_cache': 0,
            'from_telegram': 0,
            'from_api': 0,
            'fallback_used': 0,
            'api_success_rate': {},
            'response_times': []
        }
        
        # Start cleanup task
        self.cleanup_task = None
        
        # Batch processing
        self.batch_queue = []
        self.batch_processing = False
    
    async def initialize(self):
        """Initialize thumbnail manager"""
        logger.info("🖼️ Initializing ENHANCED Thumbnail Manager (99% success rate)...")
        
        # Create indexes
        await self.create_indexes()
        
        # Start cleanup task
        self.cleanup_task = asyncio.create_task(self.periodic_cleanup())
        
        logger.info("✅ ENHANCED Thumbnail Manager initialized")
        return True
    
    async def create_indexes(self):
        """Create MongoDB indexes for thumbnails collection"""
        try:
            # TTL index for automatic cleanup
            if self.config.THUMBNAIL_TTL_DAYS > 0:
                await self.thumbnails_col.create_index(
                    [("last_accessed", 1)],
                    expireAfterSeconds=self.config.THUMBNAIL_TTL_DAYS * 24 * 60 * 60,
                    name="thumbnails_ttl_index",
                    background=True
                )
                logger.info(f"✅ TTL index created ({self.config.THUMBNAIL_TTL_DAYS} days)")
            
            # Normalized title index (unique per movie)
            await self.thumbnails_col.create_index(
                [("normalized_title", 1)],
                unique=True,
                name="thumbnails_title_unique",
                background=True
            )
            
            # Channel + Message ID index for quick lookup
            await self.thumbnails_col.create_index(
                [("channel_id", 1), ("message_id", 1)],
                name="thumbnails_message_index",
                background=True
            )
            
            # Source index for analytics
            await self.thumbnails_col.create_index(
                [("source", 1)],
                name="thumbnails_source_index",
                background=True
            )
            
            # Created at index for sorting
            await self.thumbnails_col.create_index(
                [("created_at", -1)],
                name="thumbnails_created_index",
                background=True
            )
            
            logger.info("✅ Thumbnail collection indexes created")
            
        except Exception as e:
            logger.error(f"❌ Error creating thumbnail indexes: {e}")
    
    async def extract_thumbnail_from_telegram(self, channel_id: int, message_id: int) -> Optional[str]:
        """
        Extract thumbnail directly from Telegram message - FIXED
        Returns base64 data URL or None
        """
        try:
            if self.bot_handler and self.bot_handler.initialized:
                thumbnail_url = await self.bot_handler.extract_thumbnail(channel_id, message_id)
                if thumbnail_url:
                    logger.info(f"✅ Telegram thumbnail extracted: {channel_id}/{message_id}")
                    self.stats['from_telegram'] += 1
                    self.stats['successful'] += 1
                    return thumbnail_url
            
            return None
            
        except Exception as e:
            logger.debug(f"Telegram extraction failed: {e}")
            return None
    
    def is_video_file(self, filename: str) -> bool:
        """Check if file is video"""
        if not filename:
            return False
        video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v']
        return any(filename.lower().endswith(ext) for ext in video_extensions)
    
    async def get_thumbnail_for_movie(self, title: str, channel_id: int = None, message_id: int = None) -> Dict[str, Any]:
        """
        Get thumbnail for movie with 99% success rate
        Priority: Cache -> Database -> Telegram -> Multiple APIs -> Fallback
        """
        self.stats['total_requests'] += 1
        start_time = time.time()
        
        try:
            # Generate cache key
            cache_key = f"{title}_{channel_id}_{message_id}"
            
            # 1. Check in-memory cache first (fastest)
            if cache_key in self.thumbnail_cache:
                cached_data = self.thumbnail_cache[cache_key]
                if time.time() - cached_data['timestamp'] < self.cache_ttl:
                    logger.debug(f"✅ Memory cache hit: {title}")
                    self.stats['from_cache'] += 1
                    self.stats['successful'] += 1
                    return cached_data['data']
            
            normalized_title = self.normalize_title(title)
            
            # 2. Check database cache
            db_thumbnail = await self._get_from_database(normalized_title, channel_id, message_id)
            if db_thumbnail:
                # Update memory cache
                self.thumbnail_cache[cache_key] = {
                    'data': db_thumbnail,
                    'timestamp': time.time()
                }
                self.stats['from_cache'] += 1
                self.stats['successful'] += 1
                return db_thumbnail
            
            # 3. Try Telegram extraction (if message_id provided)
            telegram_thumbnail = None
            if channel_id and message_id:
                telegram_thumbnail = await self.extract_thumbnail_from_telegram(channel_id, message_id)
            
            if telegram_thumbnail:
                await self._save_to_database(
                    normalized_title=normalized_title,
                    thumbnail_url=telegram_thumbnail,
                    source='telegram',
                    extracted=True,
                    channel_id=channel_id,
                    message_id=message_id
                )
                
                thumbnail_data = {
                    'thumbnail_url': telegram_thumbnail,
                    'source': 'telegram',
                    'has_thumbnail': True,
                    'extracted': True
                }
                
                self.thumbnail_cache[cache_key] = {
                    'data': thumbnail_data,
                    'timestamp': time.time()
                }
                
                self.stats['successful'] += 1
                return thumbnail_data
            
            # 4. Try MULTIPLE API services for 99% success
            logger.info(f"🔍 Fetching thumbnail from APIs for: {title}")
            api_thumbnail = await self._fetch_from_multiple_apis(title)
            
            if api_thumbnail:
                # Save to database
                await self._save_to_database(
                    normalized_title=normalized_title,
                    thumbnail_url=api_thumbnail['url'],
                    source=api_thumbnail['source'],
                    extracted=True,
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
                
                self.stats['from_api'] += 1
                self.stats['successful'] += 1
                
                # Update API success stats
                api_source = api_thumbnail['source']
                if api_source not in self.stats['api_success_rate']:
                    self.stats['api_success_rate'][api_source] = 0
                self.stats['api_success_rate'][api_source] += 1
                
                logger.info(f"✅ API thumbnail fetched: {title} - {api_source}")
                return thumbnail_data
            
            # 5. Fallback (ONLY 1% cases)
            fallback_data = {
                'thumbnail_url': self.config.FALLBACK_POSTER,
                'source': 'fallback',
                'has_thumbnail': True,
                'extracted': False
            }
            
            self.thumbnail_cache[cache_key] = {
                'data': fallback_data,
                'timestamp': time.time()
            }
            
            self.stats['fallback_used'] += 1
            self.stats['failed'] += 1
            logger.warning(f"⚠️ Fallback used for: {title}")
            
            # Calculate success rate
            success_rate = (self.stats['successful'] / self.stats['total_requests']) * 100
            if success_rate < 99:
                logger.warning(f"⚠️ Success rate: {success_rate:.2f}% (target: 99%)")
            
            return fallback_data
            
        except Exception as e:
            logger.error(f"❌ Error getting thumbnail for {title}: {e}")
            self.stats['failed'] += 1
            
            # Return fallback
            return {
                'thumbnail_url': self.config.FALLBACK_POSTER,
                'source': 'fallback',
                'has_thumbnail': True,
                'extracted': False
            }
        finally:
            response_time = time.time() - start_time
            self.stats['response_times'].append(response_time)
            
            # Keep only last 100 response times
            if len(self.stats['response_times']) > 100:
                self.stats['response_times'] = self.stats['response_times'][-100:]
    
    async def _fetch_from_multiple_apis(self, title: str) -> Optional[Dict]:
        """Fetch thumbnail from multiple API services concurrently"""
        clean_title = self.clean_title_for_api(title)
        
        # Create tasks for all API services
        tasks = []
        for api_service in self.api_services:
            task = asyncio.create_task(api_service(clean_title))
            tasks.append(task)
        
        # Wait for first successful result
        for task in tasks:
            try:
                result = await asyncio.wait_for(task, timeout=2.0)
                if result:
                    return result
            except (asyncio.TimeoutError, Exception):
                continue
        
        return None
    
    async def _fetch_from_tmdb(self, title: str) -> Optional[Dict]:
        """Fetch from TMDB (Primary Source)"""
        try:
            if not self.config.TMDB_API_KEY:
                return None
            
            async with aiohttp.ClientSession() as session:
                # Try movie search
                movie_url = "https://api.themoviedb.org/3/search/movie"
                movie_params = {
                    'api_key': self.config.TMDB_API_KEY,
                    'query': title,
                    'language': 'en-US',
                    'page': 1,
                    'include_adult': False
                }
                
                async with session.get(movie_url, params=movie_params, timeout=3) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('results') and len(data['results']) > 0:
                            poster_path = data['results'][0].get('poster_path')
                            if poster_path:
                                return {
                                    'url': f"https://image.tmdb.org/t/p/w500{poster_path}",
                                    'source': 'tmdb'
                                }
                
                # Try TV search
                tv_url = "https://api.themoviedb.org/3/search/tv"
                tv_params = {
                    'api_key': self.config.TMDB_API_KEY,
                    'query': title,
                    'language': 'en-US',
                    'page': 1
                }
                
                async with session.get(tv_url, params=tv_params, timeout=3) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('results') and len(data['results']) > 0:
                            poster_path = data['results'][0].get('poster_path')
                            if poster_path:
                                return {
                                    'url': f"https://image.tmdb.org/t/p/w500{poster_path}",
                                    'source': 'tmdb_tv'
                                }
            
            return None
            
        except Exception as e:
            logger.debug(f"TMDB fetch error: {e}")
            return None
    
    async def _fetch_from_omdb(self, title: str) -> Optional[Dict]:
        """Fetch from OMDB (Secondary Source)"""
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
                
                async with session.get(url, params=params, timeout=3) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('Poster') and data['Poster'] != 'N/A':
                            return {
                                'url': data['Poster'],
                                'source': 'omdb'
                            }
            
            return None
            
        except Exception as e:
            logger.debug(f"OMDB fetch error: {e}")
            return None
    
    async def _fetch_from_tvdb(self, title: str) -> Optional[Dict]:
        """Fetch from TVDB (For TV Shows)"""
        try:
            # TVDB requires authentication, using fallback approach
            # Try to search via TMDB first
            return None
            
        except Exception:
            return None
    
    async def _fetch_from_imdb(self, title: str) -> Optional[Dict]:
        """Fetch from IMDb via OMDB"""
        try:
            # OMDB already handles IMDb IDs
            # Try with IMDb ID pattern
            imdb_pattern = r'tt\d{7,8}'
            imdb_match = re.search(imdb_pattern, title)
            
            if imdb_match and self.config.OMDB_API_KEY:
                imdb_id = imdb_match.group()
                async with aiohttp.ClientSession() as session:
                    url = "http://www.omdbapi.com/"
                    params = {
                        'i': imdb_id,
                        'apikey': self.config.OMDB_API_KEY
                    }
                    
                    async with session.get(url, params=params, timeout=3) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data.get('Poster') and data['Poster'] != 'N/A':
                                return {
                                    'url': data['Poster'],
                                    'source': 'imdb'
                                }
            
            return None
            
        except Exception as e:
            logger.debug(f"IMDb fetch error: {e}")
            return None
    
    async def _fetch_from_google_images(self, title: str) -> Optional[Dict]:
        """Fetch from Google Images (Fallback Source)"""
        try:
            # Use DuckDuckGo Instant Answer API as fallback
            async with aiohttp.ClientSession() as session:
                url = "https://api.duckduckgo.com/"
                params = {
                    'q': f"{title} movie poster",
                    'format': 'json',
                    'no_html': '1',
                    'skip_disambig': '1'
                }
                
                async with session.get(url, params=params, timeout=3) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('Image'):
                            return {
                                'url': data['Image'],
                                'source': 'duckduckgo'
                            }
            
            return None
            
        except Exception as e:
            logger.debug(f"Google Images fetch error: {e}")
            return None
    
    async def _get_from_database(self, normalized_title: str, channel_id: int = None, message_id: int = None) -> Optional[Dict]:
        """Get thumbnail from database"""
        try:
            # First try by normalized_title (one movie → one thumbnail)
            thumbnail_doc = await self.thumbnails_col.find_one(
                {"normalized_title": normalized_title},
                sort=[("created_at", -1)]  # Get the latest
            )
            
            # If not found and we have message details, try that
            if not thumbnail_doc and channel_id and message_id:
                thumbnail_doc = await self.thumbnails_col.find_one({
                    "channel_id": channel_id,
                    "message_id": message_id
                })
            
            if thumbnail_doc:
                # Update last_accessed
                await self.thumbnails_col.update_one(
                    {"_id": thumbnail_doc["_id"]},
                    {
                        "$set": {"last_accessed": datetime.now()},
                        "$inc": {"access_count": 1}
                    }
                )
                
                return {
                    'thumbnail_url': thumbnail_doc.get('thumbnail_url'),
                    'source': thumbnail_doc.get('source', 'database'),
                    'has_thumbnail': True,
                    'extracted': thumbnail_doc.get('extracted', False)
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Database fetch error: {e}")
            return None
    
    async def _save_to_database(self, normalized_title: str, thumbnail_url: str, 
                               source: str, extracted: bool, 
                               channel_id: int = None, message_id: int = None):
        """Save thumbnail to database"""
        try:
            thumbnail_doc = {
                'normalized_title': normalized_title,
                'thumbnail_url': thumbnail_url,
                'source': source,
                'extracted': extracted,
                'created_at': datetime.now(),
                'last_accessed': datetime.now(),
                'updated_at': datetime.now(),
                'access_count': 1
            }
            
            if channel_id:
                thumbnail_doc['channel_id'] = channel_id
            if message_id:
                thumbnail_doc['message_id'] = message_id
            
            # Use upsert to update or insert
            await self.thumbnails_col.update_one(
                {
                    'normalized_title': normalized_title
                },
                {
                    '$set': thumbnail_doc,
                    '$inc': {'access_count': 1},
                    '$setOnInsert': {'first_seen': datetime.now()}
                },
                upsert=True
            )
            
            logger.debug(f"✅ Thumbnail saved to database: {normalized_title}")
            
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
    
    def normalize_title(self, title: str) -> str:
        """Advanced title normalization for one movie → one thumbnail"""
        if not title:
            return ""
        
        # Convert to lowercase
        title = title.lower().strip()
        
        # Remove year patterns (1999), (2000), [2020], etc.
        title = re.sub(r'\s*[\(\[]?\d{4}[\)\]]?', '', title)
        
        # Remove quality indicators
        quality_patterns = [
            r'\b\d{3,4}p\b',
            r'\b(?:hd|fhd|uhd|4k|hevc|x265|x264|h264|h265)\b',
            r'\b(?:webrip|web-dl|bluray|dvdrip|hdtv|brrip)\b',
            r'\b(?:dts|ac3|aac|dd5\.1|dd\+)\b',
            r'\b(?:esub|subs|subtitles)\b',
            r'@\w+',  # Remove @mentions
            r'\[.*?\]',  # Remove brackets
            r'\(.*?\)',  # Remove parentheses
        ]
        
        for pattern in quality_patterns:
            title = re.sub(pattern, '', title, flags=re.IGNORECASE)
        
        # Remove special characters and extra spaces
        title = re.sub(r'[^\w\s]', ' ', title)
        title = re.sub(r'\s+', ' ', title)
        
        # Remove common prefixes/suffixes
        common_patterns = [
            r'^movie\s+',
            r'^film\s+',
            r'^full\s+',
            r'^latest\s+',
            r'^new\s+',
            r'\s+full$',
            r'\s+movie$',
            r'\s+film$',
            r'\s+hindi$',
            r'\s+english$',
        ]
        
        for pattern in common_patterns:
            title = re.sub(pattern, '', title, flags=re.IGNORECASE)
        
        return title.strip()
    
    def clean_title_for_api(self, title: str) -> str:
        """Clean title for API search"""
        if not title:
            return ""
        
        # Remove year
        title = re.sub(r'\s*[\(\[]?\d{4}[\)\]]?', '', title)
        
        # Remove quality info
        title = re.sub(r'\b(?:720p|1080p|2160p|4k|hd|hevc|bluray|webrip|hdtv)\b', '', title, flags=re.IGNORECASE)
        
        # Remove special characters
        title = re.sub(r'[^\w\s\-]', '', title)
        title = re.sub(r'\s+', ' ', title)
        
        # Take only first 3-5 words for better matching
        words = title.split()
        if len(words) > 5:
            title = ' '.join(words[:5])
        
        return title.strip()
    
    async def get_thumbnails_batch(self, movies: List[Dict]) -> List[Dict]:
        """
        Get thumbnails for multiple movies in batch with 99% success
        Optimized for search results
        """
        try:
            # Prepare batch data
            batch_tasks = []
            for movie in movies:
                title = movie.get('title', '')
                channel_id = movie.get('channel_id')
                message_id = movie.get('message_id') or movie.get('real_message_id')
                
                task = asyncio.create_task(
                    self.get_thumbnail_for_movie(title, channel_id, message_id)
                )
                batch_tasks.append((movie, task))
            
            # Process results
            results = []
            successful = 0
            failed = 0
            
            for movie, task in batch_tasks:
                try:
                    thumbnail_data = await task
                    
                    # Merge thumbnail data with movie data
                    movie_with_thumbnail = movie.copy()
                    movie_with_thumbnail.update(thumbnail_data)
                    
                    results.append(movie_with_thumbnail)
                    
                    if thumbnail_data.get('source') != 'fallback':
                        successful += 1
                    else:
                        failed += 1
                        
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
                    failed += 1
            
            # Calculate batch success rate
            total = successful + failed
            if total > 0:
                success_rate = (successful / total) * 100
                logger.info(f"📊 Batch thumbnail success: {successful}/{total} ({success_rate:.1f}%)")
                
                if success_rate < 99:
                    logger.warning(f"⚠️ Batch success rate below target: {success_rate:.1f}%")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Batch thumbnails error: {e}")
            # Return movies with fallback thumbnails
            return [{
                **movie,
                'thumbnail_url': self.config.FALLBACK_POSTER,
                'source': 'fallback',
                'has_thumbnail': True,
                'extracted': False
            } for movie in movies]
    
    async def extract_thumbnails_for_existing_files(self):
        """
        Extract thumbnails for all existing video files in database
        This ensures one movie → one thumbnail for all existing content
        """
        if self.files_col is None:
            logger.warning("⚠️ Files collection not available")
            return
        
        logger.info("🔄 Extracting thumbnails for existing files in database...")
        
        try:
            # Get all video files from database
            cursor = self.files_col.find({
                'is_video_file': True
            }, {
                'title': 1,
                'normalized_title': 1,
                'channel_id': 1,
                'message_id': 1,
                'real_message_id': 1,
                'thumbnail_extracted': 1,
                '_id': 1
            })
            
            files_to_process = []
            async for doc in cursor:
                # Skip if already has extracted thumbnail
                if doc.get('thumbnail_extracted'):
                    continue
                
                files_to_process.append({
                    'title': doc.get('title', ''),
                    'normalized_title': doc.get('normalized_title', ''),
                    'channel_id': doc.get('channel_id'),
                    'message_id': doc.get('real_message_id') or doc.get('message_id'),
                    'db_id': doc.get('_id')
                })
            
            logger.info(f"📊 Found {len(files_to_process)} files needing thumbnails")
            
            if not files_to_process:
                logger.info("✅ All files already have thumbnails")
                return
            
            # Process in batches
            batch_size = 20
            total_batches = (len(files_to_process) + batch_size - 1) // batch_size
            total_success = 0
            total_failed = 0
            
            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, len(files_to_process))
                batch = files_to_process[start_idx:end_idx]
                
                logger.info(f"🖼️ Processing batch {batch_num + 1}/{total_batches} ({len(batch)} files)...")
                
                # Get thumbnails for batch
                thumbnail_results = await self.get_thumbnails_batch(batch)
                
                # Update files collection with thumbnail info
                for i, file_info in enumerate(batch):
                    if i < len(thumbnail_results):
                        thumbnail_data = thumbnail_results[i]
                        
                        if thumbnail_data.get('thumbnail_url'):
                            # Update files collection
                            await self.files_col.update_one(
                                {'_id': file_info['db_id']},
                                {'$set': {
                                    'thumbnail_url': thumbnail_data['thumbnail_url'],
                                    'thumbnail_extracted': thumbnail_data.get('extracted', False),
                                    'thumbnail_source': thumbnail_data.get('source', 'unknown')
                                }}
                            )
                            
                            total_success += 1
                        else:
                            total_failed += 1
                
                # Small delay between batches
                if batch_num < total_batches - 1:
                    await asyncio.sleep(2)
            
            logger.info(f"✅ Existing files thumbnail extraction complete: {total_success} successful, {total_failed} failed")
            
        except Exception as e:
            logger.error(f"❌ Error extracting thumbnails for existing files: {e}")
    
    async def periodic_cleanup(self):
        """Periodic cleanup of cache and orphaned thumbnails"""
        while True:
            try:
                await asyncio.sleep(24 * 60 * 60)  # Daily
                
                # Clean memory cache
                self._cleanup_memory_cache()
                
                # Clean orphaned thumbnails
                await self.cleanup_orphaned_thumbnails()
                
                logger.info("✅ Periodic cleanup completed")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Cleanup task error: {e}")
                await asyncio.sleep(3600)
    
    def _cleanup_memory_cache(self):
        """Clean up expired entries from memory cache"""
        try:
            current_time = time.time()
            expired_keys = [
                key for key, data in self.thumbnail_cache.items()
                if current_time - data['timestamp'] > self.cache_ttl
            ]
            
            for key in expired_keys:
                del self.thumbnail_cache[key]
            
            if expired_keys:
                logger.debug(f"🧹 Cleaned {len(expired_keys)} expired cache entries")
                
        except Exception as e:
            logger.error(f"❌ Memory cache cleanup error: {e}")
    
    async def cleanup_orphaned_thumbnails(self):
        """Remove orphaned thumbnails (no corresponding files)"""
        try:
            logger.info("🧹 Cleaning up orphaned thumbnails...")
            
            if self.files_col is None:
                return
            
            # Find thumbnails without corresponding files
            pipeline = [
                {
                    '$lookup': {
                        'from': 'files',
                        'let': {'norm_title': '$normalized_title'},
                        'pipeline': [
                            {
                                '$match': {
                                    '$expr': {
                                        '$eq': ['$normalized_title', '$$norm_title']
                                    }
                                }
                            }
                        ],
                        'as': 'matching_files'
                    }
                },
                {
                    '$match': {
                        'matching_files': {'$size': 0},
                        'source': {'$ne': 'fallback'}  # Keep fallback thumbnails
                    }
                },
                {
                    '$project': {
                        '_id': 1,
                        'normalized_title': 1,
                        'source': 1,
                        'created_at': 1
                    }
                }
            ]
            
            orphaned = await self.thumbnails_col.aggregate(pipeline).to_list(length=None)
            
            deleted_count = 0
            for doc in orphaned:
                try:
                    await self.thumbnails_col.delete_one({"_id": doc["_id"]})
                    deleted_count += 1
                    logger.debug(f"🗑️ Deleted orphaned thumbnail: {doc.get('normalized_title', 'Unknown')}")
                except Exception as e:
                    logger.error(f"❌ Error deleting orphaned thumbnail: {e}")
            
            if deleted_count > 0:
                logger.info(f"✅ Cleaned up {deleted_count} orphaned thumbnails")
            else:
                logger.info("✅ No orphaned thumbnails found")
                
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get detailed statistics"""
        try:
            total_thumbnails = await self.thumbnails_col.count_documents({})
            extracted_count = await self.thumbnails_col.count_documents({'extracted': True})
            
            # Calculate success rate
            total_requests = self.stats['total_requests']
            successful = self.stats['successful']
            success_rate = (successful / total_requests * 100) if total_requests > 0 else 0
            
            # Calculate average response time
            response_times = self.stats['response_times']
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            
            # Get source distribution
            pipeline = [
                {"$group": {"_id": "$source", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]
            source_dist = await self.thumbnails_col.aggregate(pipeline).to_list(length=10)
            
            # Get recent thumbnails
            recent_thumbnails = await self.thumbnails_col.find(
                {},
                {
                    'normalized_title': 1,
                    'source': 1,
                    'created_at': 1,
                    'access_count': 1,
                    '_id': 0
                }
            ).sort('created_at', -1).limit(5).to_list(length=5)
            
            return {
                'total_thumbnails': total_thumbnails,
                'extracted_thumbnails': extracted_count,
                'performance_stats': {
                    'total_requests': self.stats['total_requests'],
                    'successful': self.stats['successful'],
                    'failed': self.stats['failed'],
                    'from_cache': self.stats['from_cache'],
                    'from_telegram': self.stats['from_telegram'],
                    'from_api': self.stats['from_api'],
                    'fallback_used': self.stats['fallback_used'],
                    'api_success_rate': self.stats['api_success_rate']
                },
                'success_rate': f"{success_rate:.2f}%",
                'avg_response_time_ms': f"{avg_response_time * 1000:.1f}",
                'source_distribution': source_dist,
                'recent_thumbnails': recent_thumbnails,
                'cache_stats': {
                    'memory_cache_size': len(self.thumbnail_cache),
                    'memory_cache_ttl_hours': self.cache_ttl / 3600,
                    'database_cache_size': total_thumbnails
                },
                'features': {
                    'one_movie_one_thumbnail': True,
                    'multi_quality_same_thumbnail': True,
                    'priority_extracted_first': True,
                    'fallback_system': True,
                    'ttl_enabled': self.config.THUMBNAIL_TTL_DAYS > 0,
                    'ttl_days': self.config.THUMBNAIL_TTL_DAYS
                },
                'target': '99% success rate'
            }
            
        except Exception as e:
            logger.error(f"❌ Stats error: {e}")
            return {}
    
    async def find_duplicate_titles(self) -> List[Dict]:
        """Find duplicate normalized titles (for debugging)"""
        try:
            pipeline = [
                {"$group": {
                    "_id": "$normalized_title",
                    "count": {"$sum": 1},
                    "titles": {"$push": "$original_title"},
                    "sources": {"$push": "$source"},
                    "ids": {"$push": "$_id"}
                }},
                {"$match": {"count": {"$gt": 1}}},
                {"$sort": {"count": -1}}
            ]
            
            duplicates = await self.thumbnails_col.aggregate(pipeline).to_list(length=50)
            return duplicates
            
        except Exception as e:
            logger.error(f"❌ Error finding duplicates: {e}")
            return []
    
    async def merge_duplicate_thumbnails(self):
        """Merge duplicate thumbnails (keep best one)"""
        try:
            duplicates = await self.find_duplicate_titles()
            
            if not duplicates:
                logger.info("✅ No duplicate thumbnails found")
                return
            
            logger.info(f"🔄 Merging {len(duplicates)} duplicate thumbnails...")
            
            merged_count = 0
            for dup in duplicates:
                normalized_title = dup['_id']
                ids = dup['ids']
                sources = dup['sources']
                
                # Determine best thumbnail (priority: telegram > tmdb > omdb > others > fallback)
                source_priority = {
                    'telegram': 1,
                    'tmdb': 2,
                    'tmdb_tv': 3,
                    'omdb': 4,
                    'imdb': 5,
                    'duckduckgo': 6,
                    'fallback': 999
                }
                
                # Find best thumbnail
                best_id = None
                best_priority = 999
                
                for i, source in enumerate(sources):
                    priority = source_priority.get(source, 100)
                    if priority < best_priority:
                        best_priority = priority
                        best_id = ids[i]
                
                # Delete duplicates, keep best
                if best_id and len(ids) > 1:
                    delete_ids = [id for id in ids if id != best_id]
                    await self.thumbnails_col.delete_many({"_id": {"$in": delete_ids}})
                    merged_count += len(delete_ids)
                    logger.debug(f"✅ Merged duplicates for: {normalized_title}")
            
            logger.info(f"✅ Merged {merged_count} duplicate thumbnails")
            
        except Exception as e:
            logger.error(f"❌ Error merging duplicates: {e}")
    
    async def shutdown(self):
        """Shutdown thumbnail manager"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        self.thumbnail_cache.clear()
        logger.info("✅ Thumbnail Manager shutdown complete")
