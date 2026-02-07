# ============================================================================
# thumbnail_manager.py - COMPLETE FIXED VERSION WITH ALL METHODS
# ============================================================================

import asyncio
import base64
import logging
import time
import re
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, Callable
import hashlib
import aiohttp
import io

logger = logging.getLogger(__name__)

class ThumbnailManager:
    """COMPLETE Thumbnail management with all API methods included"""
    
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
        
        # Telegram extraction methods
        self.telegram_extraction_methods = [
            self._extract_from_video_thumbnail,
            self._extract_from_document_thumbnail,
            self._extract_from_preview,
            self._extract_from_message_media
        ]
        
        # API Services - COMPLETE LIST
        self.api_services = [
            self._fetch_from_tmdb,
            self._fetch_from_omdb,
            self._fetch_from_imdb,
            self._fetch_from_letterboxd,
            self._fetch_from_wikipedia,
            self._fetch_from_youtube,
            self._fetch_from_google_images,
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
            'telegram_extraction_stats': {},
            'response_times': []
        }
        
        # API Keys
        self.tmdb_api_key = getattr(config, 'TMDB_API_KEY', '')
        self.omdb_api_key = getattr(config, 'OMDB_API_KEY', '')
        self.youtube_api_key = getattr(config, 'YOUTUBE_API_KEY', '')
    
    async def initialize(self):
        """Initialize thumbnail manager"""
        logger.info("🖼️ Initializing Thumbnail Manager...")
        
        try:
            await self.db.command('ping')
            logger.info(f"✅ Database connection verified: {self.db_name}")
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False
        
        # Create indexes
        await self.create_indexes()
        
        logger.info("✅ Thumbnail Manager initialized")
        logger.info(f"🎯 API Services: {len(self.api_services)}")
        logger.info(f"🎯 Telegram Methods: {len(self.telegram_extraction_methods)}")
        
        return True
    
    async def create_indexes(self):
        """Create optimized MongoDB indexes"""
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
            
            # Normalized title unique index (Most Important)
            try:
                await self.thumbnails_col.create_index(
                    [("normalized_title", 1)],
                    unique=True,
                    name="thumbnails_title_unique",
                    background=True
                )
                logger.info("✅ Normalized title unique index created")
            except Exception as e:
                logger.warning(f"⚠️ Unique index error: {e}")
            
            # Channel + Message ID index (sparse for non-null values)
            try:
                await self.thumbnails_col.create_index(
                    [("channel_id", 1), ("message_id", 1)],
                    sparse=True,
                    name="thumbnails_message_index",
                    background=True
                )
                logger.info("✅ Sparse message index created")
            except Exception as e:
                logger.warning(f"⚠️ Message index error: {e}")
            
            # Source index for analytics
            await self.thumbnails_col.create_index(
                [("source", 1)],
                name="thumbnails_source_index",
                background=True
            )
            
            logger.info("✅ All indexes created/verified")
            
        except Exception as e:
            logger.error(f"❌ Error creating indexes: {e}")
    
    async def extract_thumbnail_with_fallbacks(self, channel_id: int, message_id: int) -> Optional[str]:
        """
        Extract thumbnail using multiple Telegram methods
        Returns base64 data URL or None
        """
        if not channel_id or not message_id:
            logger.debug("Invalid channel_id or message_id")
            return None
        
        # Initialize stats
        if "telegram_fallbacks" not in self.stats['telegram_extraction_stats']:
            self.stats['telegram_extraction_stats']["telegram_fallbacks"] = {
                'attempts': 0,
                'success': 0,
                'failures': 0
            }
        
        self.stats['telegram_extraction_stats']["telegram_fallbacks"]['attempts'] += 1
        
        for method in self.telegram_extraction_methods:
            method_name = method.__name__
            
            # Initialize stats for this specific method
            if method_name not in self.stats['telegram_extraction_stats']:
                self.stats['telegram_extraction_stats'][method_name] = {
                    'attempts': 0,
                    'success': 0,
                    'failures': 0
                }
            
            self.stats['telegram_extraction_stats'][method_name]['attempts'] += 1
            
            try:
                thumbnail = await method(channel_id, message_id)
                if thumbnail:
                    logger.debug(f"✅ {method_name} succeeded for {channel_id}/{message_id}")
                    self.stats['telegram_extraction_stats'][method_name]['success'] += 1
                    self.stats['from_telegram'] += 1
                    return thumbnail
                else:
                    self.stats['telegram_extraction_stats'][method_name]['failures'] += 1
                    
            except Exception as e:
                self.stats['telegram_extraction_stats'][method_name]['failures'] += 1
                logger.debug(f"❌ {method_name} failed: {e}")
                continue
        
        logger.debug(f"❌ All Telegram extraction methods failed for {channel_id}/{message_id}")
        return None
    
    async def _extract_from_video_thumbnail(self, channel_id: int, message_id: int) -> Optional[str]:
        """Extract from video thumbnail"""
        try:
            if not self.bot_handler:
                return None
            
            # Get message
            message = await self.bot_handler.get_message(channel_id, message_id)
            if not message:
                return None
            
            # Check for video
            if message.video:
                if hasattr(message.video, 'thumbs') and message.video.thumbs:
                    # Convert to base64
                    for thumb in message.video.thumbs:
                        if hasattr(thumb, 'bytes'):
                            return f"data:image/jpeg;base64,{base64.b64encode(thumb.bytes).decode()}"
            
            return None
            
        except Exception as e:
            logger.debug(f"Video thumbnail extraction failed: {e}")
            return None
    
    async def _extract_from_document_thumbnail(self, channel_id: int, message_id: int) -> Optional[str]:
        """Extract from document thumbnail (for video files sent as document)"""
        try:
            if not self.bot_handler:
                return None
            
            message = await self.bot_handler.get_message(channel_id, message_id)
            if not message:
                return None
            
            if message.document:
                # Check if it's a video file
                mime_type = getattr(message.document, 'mime_type', '')
                if mime_type and 'video' in mime_type:
                    if hasattr(message.document, 'thumbs') and message.document.thumbs:
                        for thumb in message.document.thumbs:
                            if hasattr(thumb, 'bytes'):
                                return f"data:image/jpeg;base64,{base64.b64encode(thumb.bytes).decode()}"
            
            return None
            
        except Exception as e:
            logger.debug(f"Document thumbnail extraction failed: {e}")
            return None
    
    async def _extract_from_preview(self, channel_id: int, message_id: int) -> Optional[str]:
        """Extract from message preview"""
        try:
            if not self.bot_handler:
                return None
            
            # Some messages have photo previews even for videos
            message = await self.bot_handler.get_message(channel_id, message_id)
            if not message:
                return None
            
            # Check for photo in message
            if message.photo:
                # Download photo
                file = await self.bot_handler.download_media(message.photo)
                if file:
                    return f"data:image/jpeg;base64,{base64.b64encode(file).decode()}"
            
            return None
            
        except Exception as e:
            logger.debug(f"Preview extraction failed: {e}")
            return None
    
    async def _extract_from_message_media(self, channel_id: int, message_id: int) -> Optional[str]:
        """Extract from any media in message"""
        try:
            if not self.bot_handler:
                return None
            
            # Try to get any media from the message
            message = await self.bot_handler.get_message(channel_id, message_id)
            if not message:
                return None
            
            # Try different media types
            media_types = ['photo', 'video', 'document', 'animation']
            for media_type in media_types:
                media = getattr(message, media_type, None)
                if media:
                    # Try to download thumbnail
                    if hasattr(media, 'thumbs') and media.thumbs:
                        for thumb in media.thumbs:
                            if hasattr(thumb, 'bytes'):
                                return f"data:image/jpeg;base64,{base64.b64encode(thumb.bytes).decode()}"
            
            return None
            
        except Exception as e:
            logger.debug(f"Message media extraction failed: {e}")
            return None
    
    # ============================================================================
    # API METHODS - ALL INCLUDED
    # ============================================================================
    
    async def _fetch_from_tmdb(self, title: str) -> Optional[Dict]:
        """Fetch from TMDB"""
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
        """Fetch from OMDB"""
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
    
    async def _fetch_from_letterboxd(self, title: str) -> Optional[Dict]:
        """Fetch from Letterboxd"""
        try:
            # Create letterboxd slug
            slug = re.sub(r'[^\w\s-]', '', title).strip().lower()
            slug = re.sub(r'[-\s]+', '-', slug)
            
            async with aiohttp.ClientSession() as session:
                url = f"https://letterboxd.com/film/{slug}/"
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                async with session.get(url, headers=headers, timeout=5) as response:
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
    
    async def _fetch_from_wikipedia(self, title: str) -> Optional[Dict]:
        """Fetch from Wikipedia"""
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://en.wikipedia.org/w/api.php"
                params = {
                    'action': 'query',
                    'format': 'json',
                    'titles': f"{title} (film)",
                    'prop': 'pageimages',
                    'pithumbsize': 500
                }
                
                async with session.get(url, params=params, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        pages = data.get('query', {}).get('pages', {})
                        for page in pages.values():
                            if 'thumbnail' in page:
                                return {
                                    'url': page['thumbnail']['source'],
                                    'source': 'wikipedia'
                                }
        
            return None
        except Exception as e:
            logger.debug(f"Wikipedia fetch error: {e}")
            return None
    
    async def _fetch_from_youtube(self, title: str) -> Optional[Dict]:
        """Fetch trailer thumbnail from YouTube"""
        try:
            if not self.youtube_api_key:
                return None
            
            async with aiohttp.ClientSession() as session:
                url = "https://www.googleapis.com/youtube/v3/search"
                params = {
                    'q': f"{title} official trailer",
                    'part': 'snippet',
                    'maxResults': 1,
                    'type': 'video',
                    'key': self.youtube_api_key
                }
                
                async with session.get(url, params=params, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        items = data.get('items', [])
                        if items:
                            thumbnails = items[0]['snippet'].get('thumbnails', {})
                            if 'high' in thumbnails:
                                return {
                                    'url': thumbnails['high']['url'],
                                    'source': 'youtube'
                                }
        
            return None
        except Exception as e:
            logger.debug(f"YouTube fetch error: {e}")
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
    
    # ============================================================================
    # MAIN THUMBNAIL FETCHING LOGIC
    # ============================================================================
    
    async def get_thumbnail_for_movie(self, title: str, channel_id: int = None, message_id: int = None) -> Dict[str, Any]:
        """
        Get thumbnail with high success rate
        """
        self.stats['total_requests'] += 1
        start_time = time.time()
        
        try:
            # Generate cache key
            cache_key = f"{title}_{channel_id}_{message_id}"
            
            # 1. Check in-memory cache
            if cache_key in self.thumbnail_cache:
                cached_data = self.thumbnail_cache[cache_key]
                if time.time() - cached_data['timestamp'] < self.cache_ttl:
                    self.stats['from_cache'] += 1
                    return cached_data['data']
            
            # 2. Normalize title
            normalized_title = self.normalize_title(title)
            
            # 3. Check database
            db_thumbnail = await self._get_from_database(normalized_title)
            if db_thumbnail:
                self.thumbnail_cache[cache_key] = {
                    'data': db_thumbnail,
                    'timestamp': time.time()
                }
                self.stats['from_cache'] += 1
                self.stats['successful'] += 1
                return db_thumbnail
            
            # 4. Try Telegram extraction
            telegram_thumbnail = None
            if channel_id and message_id:
                telegram_thumbnail = await self.extract_thumbnail_with_fallbacks(channel_id, message_id)
            
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
            
            # 5. Try APIs
            api_thumbnail = await self._fetch_from_multiple_apis(title)
            
            if api_thumbnail and api_thumbnail.get('url'):
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
                
                self.thumbnail_cache[cache_key] = {
                    'data': thumbnail_data,
                    'timestamp': time.time()
                }
                
                self.stats['from_api'] += 1
                self.stats['successful'] += 1
                
                # Update API stats
                api_source = api_thumbnail['source']
                self.stats['api_success_rate'][api_source] = self.stats['api_success_rate'].get(api_source, 0) + 1
                
                return thumbnail_data
            
            # 6. No thumbnail found
            empty_data = {
                'thumbnail_url': '',
                'source': 'none',
                'has_thumbnail': False,
                'extracted': False
            }
            
            # Save empty result to avoid repeated searches
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
            logger.debug(f"❌ No thumbnail found for: {title}")
            
            return empty_data
            
        except Exception as e:
            logger.error(f"❌ Error getting thumbnail for {title}: {e}")
            self.stats['failed'] += 1
            
            return {
                'thumbnail_url': '',
                'source': 'error',
                'has_thumbnail': False,
                'extracted': False
            }
        finally:
            response_time = time.time() - start_time
            self.stats['response_times'].append(response_time)
            
            if len(self.stats['response_times']) > 100:
                self.stats['response_times'] = self.stats['response_times'][-100:]
    
    async def _fetch_from_multiple_apis(self, title: str) -> Optional[Dict]:
        """Fetch thumbnail from multiple API services"""
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
    
    async def _get_from_database(self, normalized_title: str) -> Optional[Dict]:
        """Get thumbnail from database"""
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
        """Save thumbnail to database"""
        try:
            thumbnail_doc = {
                'normalized_title': normalized_title,
                'thumbnail_url': thumbnail_url,
                'source': source,
                'extracted': extracted,
                'updated_at': datetime.now(),
                'last_accessed': datetime.now(),
                'access_count': 1
            }
            
            # Only add if not None
            if channel_id is not None:
                thumbnail_doc['channel_id'] = channel_id
            if message_id is not None:
                thumbnail_doc['message_id'] = message_id
            
            # Upsert operation
            await self.thumbnails_col.update_one(
                {'normalized_title': normalized_title},
                {
                    '$set': thumbnail_doc,
                    '$setOnInsert': {
                        'created_at': datetime.now(),
                        'first_seen': datetime.now()
                    },
                    '$inc': {'access_count': 1}
                },
                upsert=True
            )
            
        except Exception as e:
            logger.error(f"❌ Save error: {e}")
    
    def normalize_title(self, title: str) -> str:
        """Normalize movie title"""
        if not title:
            return ""
        
        # Convert to lowercase
        title = title.lower().strip()
        
        # Remove special characters
        title = re.sub(r'[^\w\s]', ' ', title)
        
        # Remove common prefixes
        prefixes = [
            r'^@ap\s+files\s+',
            r'^@cszmovies\s+',
            r'^latest\s+movie[s]?\s+',
            r'^new\s+movie[s]?\s+',
            r'^movie\s+',
            r'^film\s+',
        ]
        
        for prefix in prefixes:
            title = re.sub(prefix, '', title, flags=re.IGNORECASE)
        
        # Remove year patterns
        title = re.sub(r'\s*[\(\[]?\d{4}[\)\]]?', '', title)
        
        # Remove quality indicators
        quality_patterns = [
            r'\b\d{3,4}p\b',
            r'\b(?:hd|fhd|uhd|4k|2160p|1080p|720p|480p|360p)\b',
            r'\b(?:hevc|x265|x264|h264|h265|10bit|av1)\b',
            r'\b(?:webrip|web-dl|webdl|bluray|dvdrip|hdtv|brrip|camrip|hdrip|tc|ts)\b',
            r'\b(?:amzn|netflix|nf|zee5|hotstar|prime|disney\+?)\b',
            r'\b(?:dts|ac3|aac|dd5\.1|dd\+|ddp|atmos|dolby)\b',
            r'\b(?:hindi|english|tamil|telugu|malayalam|kannada|bengali)\b',
            r'\[.*?\]',
            r'\(.*?\)',
        ]
        
        for pattern in quality_patterns:
            title = re.sub(pattern, '', title, flags=re.IGNORECASE)
        
        # Clean up
        title = re.sub(r'\s+', ' ', title).strip()
        
        return title
    
    def clean_title_for_api(self, title: str) -> str:
        """Clean title for API search"""
        if not title:
            return ""
        
        # First normalize
        title = self.normalize_title(title)
        
        # Remove numbers at end
        title = re.sub(r'\s+\d+$', '', title)
        
        # Take only first 3-5 words for better matching
        words = title.split()
        if len(words) > 5:
            title = ' '.join(words[:4])
        
        return title.strip()
    
    # ============================================================================
    # BATCH PROCESSING
    # ============================================================================
    
    async def get_thumbnails_batch(self, movies: List[Dict]) -> List[Dict]:
        """Get thumbnails for multiple movies in batch"""
        try:
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
                    logger.error(f"❌ Batch thumbnail error: {e}")
                    
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
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Batch thumbnails error: {e}")
            return [{
                **movie,
                'thumbnail_url': '',
                'source': 'error',
                'has_thumbnail': False,
                'extracted': False
            } for movie in movies]
    
    async def extract_thumbnails_for_existing_files(self):
        """Extract thumbnails for all existing video files"""
        if self.files_col is None:
            logger.warning("⚠️ Files collection not available")
            return
        
        logger.info("🔄 Extracting thumbnails for existing files...")
        
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
            batch_size = 10
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
                
                # Update files collection
                for i, file_info in enumerate(batch):
                    if i < len(thumbnail_results):
                        thumbnail_data = thumbnail_results[i]
                        
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
                    await asyncio.sleep(1)
            
            logger.info(f"✅ Existing files thumbnail extraction complete: {total_success} successful, {total_no_thumbnail} no thumbnail")
            
        except Exception as e:
            logger.error(f"❌ Error extracting thumbnails: {e}")
    
    # ============================================================================
    # STATISTICS
    # ============================================================================
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get detailed statistics"""
        try:
            total_thumbnails = await self.thumbnails_col.count_documents({})
            thumbnails_with_image = await self.thumbnails_col.count_documents({'thumbnail_url': {'$ne': ''}})
            
            total_requests = self.stats['total_requests']
            successful = self.stats['successful']
            success_rate = (successful / total_requests * 100) if total_requests > 0 else 0
            
            # Telegram extraction stats
            telegram_stats = self.stats['telegram_extraction_stats']
            
            # API success rates
            api_stats = []
            for source, count in self.stats['api_success_rate'].items():
                api_stats.append({
                    'source': source,
                    'success_count': count
                })
            
            # Sort by success count
            api_stats.sort(key=lambda x: x['success_count'], reverse=True)
            
            return {
                'overall': {
                    'total_requests': total_requests,
                    'successful': successful,
                    'failed': self.stats['failed'],
                    'success_rate': f"{success_rate:.2f}%",
                    'target_rate': "99%"
                },
                'sources': {
                    'from_cache': self.stats['from_cache'],
                    'from_telegram': self.stats['from_telegram'],
                    'from_api': self.stats['from_api'],
                    'no_thumbnail': self.stats['no_thumbnail']
                },
                'database': {
                    'total_thumbnails': total_thumbnails,
                    'thumbnails_with_image': thumbnails_with_image,
                    'thumbnails_empty': total_thumbnails - thumbnails_with_image,
                    'coverage_rate': f"{(thumbnails_with_image/total_thumbnails*100):.1f}%" if total_thumbnails > 0 else "0%"
                },
                'telegram_extraction': telegram_stats,
                'api_success_distribution': api_stats[:5],
                'features': {
                    'one_movie_one_thumbnail': True,
                    'telegram_extraction_methods': len(self.telegram_extraction_methods),
                    'api_services': len(self.api_services)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Stats error: {e}")
            return {}
    
    async def shutdown(self):
        """Shutdown thumbnail manager"""
        self.thumbnail_cache.clear()
        logger.info("✅ Thumbnail Manager shutdown complete")
