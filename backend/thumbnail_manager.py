# ============================================================================
# thumbnail_manager.py - COMPLETE THUMBNAIL SYSTEM - ONE MOVIE, ONE THUMBNAIL
# ============================================================================

import asyncio
import base64
import logging
import time
import re
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import hashlib
import aiohttp

logger = logging.getLogger(__name__)

class ThumbnailManager:
    """COMPLETE Thumbnail management system - ONE MOVIE → ONE THUMBNAIL"""
    
    def __init__(self, mongo_client, config, bot_handler=None):
        self.mongo_client = mongo_client
        self.db_name = "sk4film"
        self.db = mongo_client[self.db_name]
        self.thumbnails_col = self.db.thumbnails
        self.files_col = self.db.files
        self.config = config
        self.bot_handler = bot_handler
        
        # Enhanced Thumbnail cache
        self.thumbnail_cache = {}
        self.cache_ttl = 3600  # 1 hour
        
        # Multiple API Services for 99% success rate
        self.api_services = [
            self._fetch_from_tmdb,
            self._fetch_from_omdb,
            self._fetch_from_imdb,
            self._fetch_from_google_images,
            self._fetch_from_letterboxd
        ]
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful': 0,
            'failed': 0,
            'from_cache': 0,
            'from_telegram': 0,
            'from_api': 0,
            'no_thumbnail': 0,
            'api_success_rate': {},
            'response_times': []
        }
        
        # Start cleanup task
        self.cleanup_task = None
        
        # Batch processing
        self.batch_queue = []
        self.batch_processing = False
        
        # API Keys
        self.tmdb_api_key = getattr(config, 'TMDB_API_KEY', '')
        self.omdb_api_key = getattr(config, 'OMDB_API_KEY', '')
    
    async def initialize(self):
        """Initialize thumbnail manager"""
        logger.info("🖼️ Initializing COMPLETE Thumbnail Manager (ONE MOVIE → ONE THUMBNAIL)...")
        
        # Test database connection
        try:
            await self.db.command('ping')
            logger.info(f"✅ Database connection verified: {self.db_name}")
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False
        
        # Create indexes
        await self.create_indexes()
        
        # Start cleanup task
        self.cleanup_task = asyncio.create_task(self.periodic_cleanup())
        
        logger.info("✅ COMPLETE Thumbnail Manager initialized")
        logger.info("🎯 FEATURES:")
        logger.info("   • One Movie → One Thumbnail")
        logger.info("   • 4 Qualities → Same Thumbnail")
        logger.info("   • Existing Files Auto-Extract")
        logger.info("   • New Files Auto-Extract")
        logger.info("   • Priority: Extracted → Poster → Empty")
        logger.info("   • No Default/Fallback Image")
        logger.info("   • Success Rate Target: 99%")
        
        return True
    
    async def create_indexes(self):
        """Create MongoDB indexes for thumbnails collection"""
        try:
            # TTL index for automatic cleanup (30 days)
            ttl_days = getattr(self.config, 'THUMBNAIL_TTL_DAYS', 30)
            if ttl_days > 0:
                await self.thumbnails_col.create_index(
                    [("last_accessed", 1)],
                    expireAfterSeconds=ttl_days * 24 * 60 * 60,
                    name="thumbnails_ttl_index",
                    background=True
                )
                logger.info(f"✅ TTL index created ({ttl_days} days)")
            
            # Normalized title index (unique per movie) - MOST IMPORTANT
            try:
                await self.thumbnails_col.create_index(
                    [("normalized_title", 1)],
                    unique=True,
                    name="thumbnails_title_unique",
                    background=True
                )
                logger.info("✅ Normalized title unique index created (One Movie → One Thumbnail)")
            except Exception as e:
                logger.warning(f"⚠️ Could not create unique index (may already exist): {e}")
            
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
            
            logger.info("✅ Thumbnail collection indexes created/verified")
            
        except Exception as e:
            logger.error(f"❌ Error creating thumbnail indexes: {e}")
    
    async def extract_thumbnail_from_telegram(self, channel_id: int, message_id: int) -> Optional[str]:
        """
        Extract thumbnail directly from Telegram message
        Returns base64 data URL or None
        """
        try:
            if self.bot_handler and self.bot_handler.initialized:
                # First try bot_handler method
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
        Priority: Cache → Database → Telegram → Multiple APIs → Empty (No Fallback)
        Returns empty string if no thumbnail found
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
                    return cached_data['data']
            
            # Normalize title for ONE MOVIE → ONE THUMBNAIL
            normalized_title = self.normalize_title(title)
            logger.debug(f"🔍 Looking for thumbnail: '{title}' → '{normalized_title}'")
            
            # 2. Check database cache (One Movie → One Thumbnail)
            db_thumbnail = await self._get_from_database(normalized_title)
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
                # Save to database with normalized title (One Movie → One Thumbnail)
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
                
                # Update cache
                self.thumbnail_cache[cache_key] = {
                    'data': thumbnail_data,
                    'timestamp': time.time()
                }
                
                self.stats['successful'] += 1
                logger.info(f"✅ Telegram thumbnail saved for: {title}")
                return thumbnail_data
            
            # 4. Try MULTIPLE API services for 99% success
            logger.info(f"🔍 Fetching thumbnail from APIs for: {title}")
            api_thumbnail = await self._fetch_from_multiple_apis(title)
            
            if api_thumbnail and api_thumbnail.get('url'):
                # Save to database with normalized title (One Movie → One Thumbnail)
                await self._save_to_database(
                    normalized_title=normalized_title,
                    thumbnail_url=api_thumbnail['url'],
                    source=api_thumbnail['source'],
                    extracted=False,
                    channel_id=channel_id,
                    message_id=message_id
                )
                
                thumbnail_data = {
                    'thumbnail_url': api_thumbnail['url'],
                    'source': api_thumbnail['source'],
                    'has_thumbnail': True,
                    'extracted': False
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
            
            # 5. NO FALLBACK - Return empty (as requested)
            empty_data = {
                'thumbnail_url': '',
                'source': 'none',
                'has_thumbnail': False,
                'extracted': False
            }
            
            # Still save to database to avoid repeated API calls
            await self._save_to_database(
                normalized_title=normalized_title,
                thumbnail_url='',
                source='none',
                extracted=False,
                channel_id=channel_id,
                message_id=message_id
            )
            
            self.thumbnail_cache[cache_key] = {
                'data': empty_data,
                'timestamp': time.time()
            }
            
            self.stats['no_thumbnail'] += 1
            self.stats['failed'] += 1
            logger.warning(f"⚠️ No thumbnail found for: {title}")
            
            # Calculate success rate
            success_rate = (self.stats['successful'] / self.stats['total_requests']) * 100
            if success_rate < 99:
                logger.warning(f"⚠️ Success rate: {success_rate:.2f}% (target: 99%)")
            
            return empty_data
            
        except Exception as e:
            logger.error(f"❌ Error getting thumbnail for {title}: {e}")
            self.stats['failed'] += 1
            
            # Return empty on error too
            return {
                'thumbnail_url': '',
                'source': 'error',
                'has_thumbnail': False,
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
        
        if not clean_title:
            return None
        
        # Create tasks for all API services
        tasks = []
        for api_service in self.api_services:
            task = asyncio.create_task(api_service(clean_title))
            tasks.append(task)
        
        # Wait for first successful result
        for task in tasks:
            try:
                result = await asyncio.wait_for(task, timeout=3.0)
                if result and result.get('url'):
                    return result
            except (asyncio.TimeoutError, Exception):
                continue
        
        return None
    
    async def _fetch_from_tmdb(self, title: str) -> Optional[Dict]:
        """Fetch from TMDB (Primary Source)"""
        try:
            if not self.tmdb_api_key:
                return None
            
            async with aiohttp.ClientSession() as session:
                # Try movie search
                movie_url = "https://api.themoviedb.org/3/search/movie"
                movie_params = {
                    'api_key': self.tmdb_api_key,
                    'query': title,
                    'language': 'en-US',
                    'page': 1,
                    'include_adult': False
                }
                
                async with session.get(movie_url, params=movie_params, timeout=5) as response:
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
                    'api_key': self.tmdb_api_key,
                    'query': title,
                    'language': 'en-US',
                    'page': 1
                }
                
                async with session.get(tv_url, params=tv_params, timeout=5) as response:
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
            if not self.omdb_api_key:
                return None
            
            async with aiohttp.ClientSession() as session:
                url = "http://www.omdbapi.com/"
                params = {
                    't': title,
                    'apikey': self.omdb_api_key,
                    'plot': 'short'
                }
                
                async with session.get(url, params=params, timeout=5) as response:
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
    
    async def _fetch_from_imdb(self, title: str) -> Optional[Dict]:
        """Fetch from IMDb (via OMDB with IMDb ID)"""
        try:
            # Extract IMDb ID from title if present
            imdb_pattern = r'tt\d{7,8}'
            imdb_match = re.search(imdb_pattern, title)
            
            if imdb_match and self.omdb_api_key:
                imdb_id = imdb_match.group()
                async with aiohttp.ClientSession() as session:
                    url = "http://www.omdbapi.com/"
                    params = {
                        'i': imdb_id,
                        'apikey': self.omdb_api_key
                    }
                    
                    async with session.get(url, params=params, timeout=5) as response:
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
        """Fetch from Google Images via DuckDuckGo"""
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://api.duckduckgo.com/"
                params = {
                    'q': f"{title} movie poster",
                    'format': 'json',
                    'no_html': '1',
                    'skip_disambig': '1'
                }
                
                async with session.get(url, params=params, timeout=5) as response:
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
    
    async def _fetch_from_letterboxd(self, title: str) -> Optional[Dict]:
        """Fetch from Letterboxd"""
        try:
            # Create letterboxd slug
            slug = re.sub(r'[^\w\s-]', '', title).strip().lower()
            slug = re.sub(r'[-\s]+', '-', slug)
            
            async with aiohttp.ClientSession() as session:
                url = f"https://letterboxd.com/film/{slug}/"
                async with session.get(url, timeout=5) as response:
                    if response.status == 200:
                        html = await response.text()
                        # Look for og:image meta tag
                        og_image_match = re.search(r'property="og:image" content="([^"]+)"', html)
                        if og_image_match:
                            return {
                                'url': og_image_match.group(1),
                                'source': 'letterboxd'
                            }
            
            return None
            
        except Exception as e:
            logger.debug(f"Letterboxd fetch error: {e}")
            return None
    
    async def _get_from_database(self, normalized_title: str) -> Optional[Dict]:
        """Get thumbnail from database using normalized title (One Movie → One Thumbnail)"""
        try:
            thumbnail_doc = await self.thumbnails_col.find_one(
                {"normalized_title": normalized_title}
            )
            
            if thumbnail_doc:
                # Update last_accessed
                await self.thumbnails_col.update_one(
                    {"_id": thumbnail_doc["_id"]},
                    {
                        "$set": {"last_accessed": datetime.now()},
                        "$inc": {"access_count": 1}
                    }
                )
                
                # Check if thumbnail_url is empty
                thumbnail_url = thumbnail_doc.get('thumbnail_url', '')
                has_thumbnail = bool(thumbnail_url)
                
                return {
                    'thumbnail_url': thumbnail_url,
                    'source': thumbnail_doc.get('source', 'database'),
                    'has_thumbnail': has_thumbnail,
                    'extracted': thumbnail_doc.get('extracted', False)
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Database fetch error: {e}")
            return None
    
    async def _save_to_database(self, normalized_title: str, thumbnail_url: str, 
                               source: str, extracted: bool, 
                               channel_id: int = None, message_id: int = None):
        """Save thumbnail to database with normalized title (One Movie → One Thumbnail)"""
        try:
            # First, try to find existing document
            existing_doc = await self.thumbnails_col.find_one(
                {'normalized_title': normalized_title}
            )
            
            thumbnail_doc = {
                'normalized_title': normalized_title,
                'thumbnail_url': thumbnail_url,
                'source': source,
                'extracted': extracted,
                'updated_at': datetime.now(),
                'last_accessed': datetime.now(),
                'access_count': 1
            }
            
            if channel_id:
                thumbnail_doc['channel_id'] = channel_id
            if message_id:
                thumbnail_doc['message_id'] = message_id
            
            if existing_doc:
                # Update existing document
                thumbnail_doc['created_at'] = existing_doc.get('created_at', datetime.now())
                thumbnail_doc['access_count'] = existing_doc.get('access_count', 0) + 1
                
                await self.thumbnails_col.update_one(
                    {'normalized_title': normalized_title},
                    {'$set': thumbnail_doc}
                )
            else:
                # Insert new document
                thumbnail_doc['created_at'] = datetime.now()
                thumbnail_doc['first_seen'] = datetime.now()
                
                await self.thumbnails_col.insert_one(thumbnail_doc)
            
            logger.debug(f"✅ Thumbnail saved to database: {normalized_title} ({source})")
            
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
    
    def normalize_title(self, title: str) -> str:
        """
        Advanced title normalization for ONE MOVIE → ONE THUMBNAIL
        This ensures 4 qualities → same thumbnail
        """
        if not title:
            return ""
        
        # Store original for debugging
        original_title = title
        
        # Convert to lowercase
        title = title.lower().strip()
        
        # Remove @mentions and hashtags at start
        title = re.sub(r'^@\w+\s*', '', title)
        title = re.sub(r'^#\w+\s*', '', title)
        
        # Remove common prefixes
        prefixes = [
            r'^@ap\s+files\s+',
            r'^@cszmovies\s+',
            r'^@\w+\s+files\s+',
            r'^@\w+\s+movie[s]?\s+',
            r'^latest\s+movie[s]?\s+',
            r'^new\s+movie[s]?\s+',
            r'^movie\s+',
            r'^film\s+',
            r'^full\s+',
        ]
        
        for prefix in prefixes:
            title = re.sub(prefix, '', title, flags=re.IGNORECASE)
        
        # Remove year patterns (1999), (2000), [2020], etc.
        title = re.sub(r'\s*[\(\[]?\d{4}[\)\]]?', '', title)
        
        # Remove quality indicators - EXPANDED LIST FOR HINDI MOVIES
        quality_patterns = [
            # Resolutions
            r'\b\d{3,4}p\b',
            r'\b(?:hd|fhd|uhd|4k|2160p|1080p|720p|480p|360p)\b',
            
            # Codecs
            r'\b(?:hevc|x265|x264|h264|h265|10bit|av1|avc|av1\s+10bit)\b',
            
            # Sources
            r'\b(?:webrip|web-dl|webdl|bluray|dvdrip|hdtv|brrip|camrip|hdrip|tc|ts|pre\d*|org|original)\b',
            r'\b(?:amzn|netflix|nf|zee5|zee|hotstar|prime|disney\+?|ds4k)\b',
            
            # Audio
            r'\b(?:dts|ac3|aac|dd5\.1|dd\+|ddp|he-aac|atmos|dolby|aac2\.0|aac5\.1|dd2\.0|dd5\.1|dpp5\.1|he-aac2\.0)\b',
            
            # Languages
            r'\b(?:hindi|english|tamil|telugu|malayalam|kannada|bengali|marathi|gujarati|punjabi|odia|assamese)\b',
            r'\b(?:hin|eng|ta|te|ml|kn|bn|mr|gu|pa|or|as)\b',
            r'\b(?:dub|dubbed|sub|subbed|subtitle[s]?|esub|hsub|msub|clean\s+audio)\b',
            r'\b(?:dual\s+audio|multi\s+audio|multi\s+language|multi\s+sub)\b',
            
            # Editions
            r'\b(?:clean|uncut|theatrical|director\'?s\s+cut|extended|unrated|remastered)\b',
            
            # File info
            r'\b(?:part\d+|cd\d+|vol\d+|chapter\s+\d+)\b',
            r'\b(?:sample|trailer|teaser|promo)\b',
            
            # Brackets and parentheses
            r'\[.*?\]',
            r'\(.*?\)',
            r'\{.*?\}',
            
            # Special patterns
            r'\b(?:hc|esub|hcesub|hc-esub)\b',
            r'\b(?:line\s+audio|li?ne)\b',
            r'\b(?:exclusive|uploaded\s+by|first\s+on\s+net)\b',
            
            # Numbers at end
            r'\s+\d+$',
            r'\s+\d+\s*gb$',
            r'\s+\d+\s*mb$',
        ]
        
        for pattern in quality_patterns:
            title = re.sub(pattern, '', title, flags=re.IGNORECASE)
        
        # Remove special characters and extra spaces
        title = re.sub(r'[^\w\s]', ' ', title)
        title = re.sub(r'\s+', ' ', title)
        
        # Remove common suffixes
        suffixes = [
            r'\s+full\s+movie$',
            r'\s+movie$',
            r'\s+film$',
            r'\s+hd$',
            r'\s+version$',
            r'\s+edit$',
            r'\s+print$',
            r'\s+rip$',
        ]
        
        for suffix in suffixes:
            title = re.sub(suffix, '', title, flags=re.IGNORECASE)
        
        # Trim and clean
        title = title.strip()
        
        # If title is too short after cleaning, use smarter extraction
        if len(title) < 3:
            # Try to extract just the movie name
            words = original_title.split()
            movie_words = []
            for word in words:
                word_lower = word.lower()
                if (len(word) > 2 and 
                    not re.match(r'^\d{3,4}p$', word_lower) and
                    not re.match(r'^\d+$', word_lower) and
                    word_lower not in ['hd', 'full', 'movie', 'film', 'hindi', 'english'] and
                    not re.match(r'^x\d+$', word_lower) and
                    not re.match(r'^h\.?\d+$', word_lower)):
                    movie_words.append(word)
            
            if movie_words:
                title = ' '.join(movie_words[:5])
        
        # Final cleanup
        title = re.sub(r'\s+', ' ', title).strip()
        
        # Debug log
        if original_title != title:
            logger.debug(f"🧹 Title normalized: '{original_title}' → '{title}'")
        
        return title
    
    def clean_title_for_api(self, title: str) -> str:
        """Clean title for API search - Better version"""
        if not title:
            return ""
        
        # First normalize
        title = self.normalize_title(title)
        
        # Remove any remaining numbers at end
        title = re.sub(r'\s+\d+$', '', title)
        
        # Take only first 3-5 words for better matching
        words = title.split()
        if len(words) > 5:
            # Try to keep important words (usually movie names are 1-3 words)
            title = ' '.join(words[:3])
        
        return title.strip()
    
    async def get_thumbnails_batch(self, movies: List[Dict]) -> List[Dict]:
        """
        Get thumbnails for multiple movies in batch
        Optimized for search results - ONE MOVIE → ONE THUMBNAIL
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
            no_thumbnail = 0
            
            for movie, task in batch_tasks:
                try:
                    thumbnail_data = await task
                    
                    # Merge thumbnail data with movie data
                    movie_with_thumbnail = movie.copy()
                    movie_with_thumbnail.update(thumbnail_data)
                    
                    results.append(movie_with_thumbnail)
                    
                    if thumbnail_data.get('has_thumbnail'):
                        successful += 1
                    else:
                        no_thumbnail += 1
                        
                except Exception as e:
                    logger.error(f"❌ Batch thumbnail error for {movie.get('title')}: {e}")
                    
                    # Add empty thumbnail
                    movie_with_empty = movie.copy()
                    movie_with_empty.update({
                        'thumbnail_url': '',
                        'source': 'error',
                        'has_thumbnail': False,
                        'extracted': False
                    })
                    results.append(movie_with_empty)
                    no_thumbnail += 1
            
            # Calculate batch success rate
            total = successful + no_thumbnail
            if total > 0:
                success_rate = (successful / total) * 100
                logger.info(f"📊 Batch thumbnail success: {successful}/{total} ({success_rate:.1f}%)")
                
                if success_rate < 99:
                    logger.warning(f"⚠️ Batch success rate below target: {success_rate:.1f}%")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Batch thumbnails error: {e}")
            # Return movies with empty thumbnails
            return [{
                **movie,
                'thumbnail_url': '',
                'source': 'error',
                'has_thumbnail': False,
                'extracted': False
            } for movie in movies]
    
    async def extract_thumbnails_for_existing_files(self):
        """
        Extract thumbnails for all existing video files in database
        This ensures ONE MOVIE → ONE THUMBNAIL for all existing content
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
                '_id': 1
            })
            
            files_to_process = []
            async for doc in cursor:
                files_to_process.append({
                    'title': doc.get('title', ''),
                    'normalized_title': doc.get('normalized_title', ''),
                    'channel_id': doc.get('channel_id'),
                    'message_id': doc.get('real_message_id') or doc.get('message_id'),
                    'db_id': doc.get('_id')
                })
            
            logger.info(f"📊 Found {len(files_to_process)} video files to process")
            
            if not files_to_process:
                logger.info("✅ No files found")
                return
            
            # Process in batches
            batch_size = 20
            total_batches = (len(files_to_process) + batch_size - 1) // batch_size
            total_success = 0
            total_no_thumbnail = 0
            
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
                        
                        # Update files collection
                        await self.files_col.update_one(
                            {'_id': file_info['db_id']},
                            {'$set': {
                                'thumbnail_url': thumbnail_data.get('thumbnail_url', ''),
                                'thumbnail_extracted': thumbnail_data.get('extracted', False),
                                'thumbnail_source': thumbnail_data.get('source', 'none')
                            }}
                        )
                        
                        if thumbnail_data.get('has_thumbnail'):
                            total_success += 1
                        else:
                            total_no_thumbnail += 1
                
                # Small delay between batches
                if batch_num < total_batches - 1:
                    await asyncio.sleep(2)
            
            logger.info(f"✅ Existing files thumbnail extraction complete: {total_success} successful, {total_no_thumbnail} no thumbnail")
            
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
                
                # Merge duplicate thumbnails
                await self.merge_duplicate_thumbnails()
                
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
                        'thumbnail_url': {'$ne': ''}  # Keep empty thumbnails
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
            thumbnails_with_image = await self.thumbnails_col.count_documents({'thumbnail_url': {'$ne': ''}})
            
            # Calculate success rate
            total_requests = self.stats['total_requests']
            successful = self.stats['successful']
            success_rate = (successful / total_requests * 100) if total_requests > 0 else 0
            
            # Calculate average response time
            response_times = self.stats['response_times']
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            
            # Get source distribution
            pipeline = [
                {"$match": {"source": {"$ne": "none"}}},
                {"$group": {"_id": "$source", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]
            source_dist = await self.thumbnails_col.aggregate(pipeline).to_list(length=10)
            
            # Get recent thumbnails
            recent_thumbnails = await self.thumbnails_col.find(
                {"thumbnail_url": {"$ne": ""}},
                {
                    'normalized_title': 1,
                    'source': 1,
                    'created_at': 1,
                    'access_count': 1,
                    '_id': 0
                }
            ).sort('created_at', -1).limit(5).to_list(length=5)
            
            ttl_days = getattr(self.config, 'THUMBNAIL_TTL_DAYS', 30)
            
            return {
                'total_thumbnails': total_thumbnails,
                'thumbnails_with_image': thumbnails_with_image,
                'thumbnails_empty': total_thumbnails - thumbnails_with_image,
                'performance_stats': {
                    'total_requests': self.stats['total_requests'],
                    'successful': self.stats['successful'],
                    'failed': self.stats['failed'],
                    'from_cache': self.stats['from_cache'],
                    'from_telegram': self.stats['from_telegram'],
                    'from_api': self.stats['from_api'],
                    'no_thumbnail': self.stats['no_thumbnail'],
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
                    'existing_files_auto_extract': True,
                    'new_files_auto_extract': True,
                    'priority_extracted_first': True,
                    'no_fallback_image': True,
                    'ttl_enabled': ttl_days > 0,
                    'ttl_days': ttl_days
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
                
                # Determine best thumbnail (priority: telegram > tmdb > omdb > others > empty)
                source_priority = {
                    'telegram': 1,
                    'tmdb': 2,
                    'tmdb_tv': 3,
                    'omdb': 4,
                    'imdb': 5,
                    'duckduckgo': 6,
                    'letterboxd': 7,
                    'database': 8,
                    'none': 999,
                    'error': 999
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
