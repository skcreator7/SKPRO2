# ============================================================================
# thumbnail_manager.py - EXTRACTED ONLY STORAGE SYSTEM
# ============================================================================
# ✅ Ek movie → ek thumbnail (sirf extracted store hoga)
# ✅ 4 qualities → same thumbnail  
# ✅ Old files auto migrate (ek baar extract)
# ✅ New files auto extract
# ✅ Search priority system
#     1. Extracted thumbnail (database se)
#     2. Poster sources API Services (live fetch, no store)
#     3. Last Fallback poster https://iili.io/fAeIwv9.th.png
# ============================================================================

import asyncio
import base64
import logging
import time
import re
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, List
import aiohttp

logger = logging.getLogger(__name__)

class ThumbnailManager:
    """EXTRACTED ONLY STORAGE - Sirf Telegram extracted thumbnails store honge"""
    
    # Fallback poster URL (LAST RESORT)
    FALLBACK_POSTER = "https://iili.io/fAeIwv9.th.png"
    
    def __init__(self, mongo_client, config, bot_handler=None):
        self.mongo_client = mongo_client
        self.db = mongo_client["sk4film"]
        self.extracted_thumbnails_col = self.db.extracted_thumbnails  # Sirf extracted store
        self.files_col = self.db.files
        self.config = config
        self.bot_handler = bot_handler
        
        # Memory cache for API results (temporary, database mein nahi)
        self.api_cache = {}
        self.api_cache_ttl = 300  # 5 minutes only
        
        # API Services - Direct fetch no store
        self.api_services = [
            self._fetch_from_tmdb,
            self._fetch_from_omdb,
            self._fetch_from_imdb,
            self._fetch_from_letterboxd,
            self._fetch_from_wikipedia,
            self._fetch_from_youtube,
            self._fetch_from_google_images,
        ]
        
        # API Keys
        self.tmdb_api_key = getattr(config, 'TMDB_API_KEY', '')
        self.omdb_api_key = getattr(config, 'OMDB_API_KEY', '')
        self.youtube_api_key = getattr(config, 'YOUTUBE_API_KEY', '')
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'from_extracted_db': 0,
            'from_telegram_live': 0,
            'from_api_live': 0,
            'from_fallback': 0
        }
    
    async def initialize(self):
        """Initialize thumbnail manager"""
        logger.info("🖼️ Initializing EXTRACTED-ONLY Storage System...")
        
        # Test database
        try:
            await self.db.command('ping')
            logger.info("✅ Database connected")
        except Exception as e:
            logger.error(f"❌ Database error: {e}")
            return False
        
        # Create indexes for extracted thumbnails only
        await self._create_extracted_indexes()
        
        logger.info("✅ Thumbnail System initialized")
        logger.info("🎯 STORAGE POLICY:")
        logger.info("   • MongoDB: SIRF extracted Telegram thumbnails")
        logger.info("   • API results: NO STORAGE (live fetch only)")
        logger.info("   • Fallback: Always available")
        
        return True
    
    async def _create_extracted_indexes(self):
        """Create indexes for extracted thumbnails collection only"""
        try:
            # Unique index for "Ek movie → Ek extracted thumbnail"
            await self.extracted_thumbnails_col.create_index(
                [("movie_id", 1)],
                unique=True,
                name="extracted_movie_id_unique"
            )
            logger.info("✅ Created extracted thumbnails unique index")
            
            # TTL index for extracted thumbnails (90 days)
            await self.extracted_thumbnails_col.create_index(
                [("last_accessed", 1)],
                expireAfterSeconds=90 * 24 * 3600,
                name="extracted_ttl_index"
            )
            logger.info("✅ Created extracted TTL index (90 days)")
            
        except Exception as e:
            logger.warning(f"⚠️ Index creation warning: {e}")
    
    def _generate_movie_id(self, title: str) -> str:
        """
        Generate unique movie ID for "Ek movie → Ek extracted thumbnail"
        Same ID for 4 qualities of same movie
        """
        # Normalize title first
        normalized = self._normalize_title(title)
        
        # Create hash for consistent ID
        movie_hash = hashlib.md5(normalized.encode()).hexdigest()[:12]
        
        return f"movie_{movie_hash}"
    
    def _normalize_title(self, title: str) -> str:
        """
        Normalize title for "4 qualities → same thumbnail"
        Removes quality indicators, years, languages, etc.
        """
        if not title:
            return ""
        
        # Convert to lowercase
        title = title.lower().strip()
        
        # Remove @mentions and hashtags
        title = re.sub(r'^[@#]\w+\s*', '', title)
        
        # Remove common prefixes
        prefixes = [
            r'^@ap\s+files\s+',
            r'^@cszmovies\s+',
            r'^latest\s+movie[s]?\s+',
            r'^new\s+movie[s]?\s+',
            r'^movie\s+',
            r'^film\s+',
            r'^full\s+movie\s+',
        ]
        
        for prefix in prefixes:
            title = re.sub(prefix, '', title, flags=re.IGNORECASE)
        
        # Remove year patterns
        title = re.sub(r'\s*[\(\[]?\d{4}[\)\]]?', '', title)
        
        # Remove ALL quality indicators (4 qualities → same thumbnail)
        quality_patterns = [
            # Resolutions
            r'\b\d{3,4}p\b',
            r'\b(?:hd|fhd|uhd|4k|2160p|1080p|720p|480p|360p)\b',
            
            # Codecs
            r'\b(?:hevc|x265|x264|h264|h265|10bit|av1)\b',
            
            # Sources
            r'\b(?:webrip|web-dl|webdl|bluray|dvdrip|hdtv|brrip|camrip|hdrip|tc|ts)\b',
            r'\b(?:amzn|netflix|nf|zee5|hotstar|prime|disney\+?)\b',
            
            # Audio
            r'\b(?:dts|ac3|aac|dd5\.1|dd\+|ddp|atmos|dolby)\b',
            
            # Languages
            r'\b(?:hindi|english|tamil|telugu|malayalam|kannada|bengali)\b',
            r'\b(?:dual\s+audio|multi\s+audio|multi\s+language)\b',
            
            # Editions
            r'\b(?:uncut|theatrical|director\'?s\s+cut|extended|unrated)\b',
            
            # File info
            r'\[.*?\]',
            r'\(.*?\)',
            r'\b(?:part\d+|cd\d+|vol\d+)\b',
        ]
        
        for pattern in quality_patterns:
            title = re.sub(pattern, '', title, flags=re.IGNORECASE)
        
        # Clean up
        title = re.sub(r'\s+', ' ', title).strip()
        
        return title.strip()
    
    def _clean_title_for_api(self, title: str) -> str:
        """Clean title for API search"""
        title = self._normalize_title(title)
        
        # Remove numbers at end
        title = re.sub(r'\s+\d+$', '', title)
        
        # Take first 3-4 words for better matching
        words = title.split()
        if len(words) > 5:
            title = ' '.join(words[:4])
        
        return title.strip()
    
    async def get_thumbnail_for_movie(self, title: str, channel_id: int = None, message_id: int = None) -> Dict[str, Any]:
        """
        Get thumbnail with PRIORITY SYSTEM:
        1. Extracted thumbnail (database se - SIRF YAHI STORE HOTA HAI)
        2. Poster sources API Services (live fetch, NO STORE)
        3. Last Fallback poster
        """
        self.stats['total_requests'] += 1
        
        try:
            # Generate movie ID
            movie_id = self._generate_movie_id(title)
            
            # =============================================
            # PRIORITY 1: EXTRACTED THUMBNAIL (DATABASE)
            # =============================================
            extracted_thumbnail = await self._get_extracted_from_database(movie_id)
            if extracted_thumbnail:
                self.stats['from_extracted_db'] += 1
                logger.info(f"✅ Extracted thumbnail found in DB: {title}")
                return extracted_thumbnail
            
            # =============================================
            # CHECK IF WE CAN EXTRACT FROM TELEGRAM NOW
            # =============================================
            if channel_id and message_id:
                # Try to extract thumbnail from Telegram
                telegram_thumbnail = await self._extract_from_telegram(channel_id, message_id)
                
                if telegram_thumbnail:
                    # ✅ SIRF EXTRACTED THUMBNAIL DATABASE MEIN STORE HOTA HAI
                    await self._save_extracted_to_database(
                        movie_id=movie_id,
                        thumbnail_url=telegram_thumbnail,
                        title=title,
                        channel_id=channel_id,
                        message_id=message_id
                    )
                    
                    self.stats['from_telegram_live'] += 1
                    logger.info(f"✅ New Telegram thumbnail extracted & stored: {title}")
                    
                    return {
                        'thumbnail_url': telegram_thumbnail,
                        'source': 'telegram_extracted',
                        'has_thumbnail': True,
                        'extracted': True,
                        'movie_id': movie_id,
                        'stored_in_db': True  # Yeh database mein save hua hai
                    }
            
            # =============================================
            # PRIORITY 2: API SERVICES (LIVE FETCH, NO STORE)
            # =============================================
            logger.info(f"🔍 Live API search for: {title}")
            
            # Check temporary memory cache first
            api_cache_key = f"api_{movie_id}"
            if api_cache_key in self.api_cache:
                cached = self.api_cache[api_cache_key]
                if time.time() - cached['timestamp'] < self.api_cache_ttl:
                    self.stats['from_api_live'] += 1
                    logger.info(f"✅ API cache hit: {title}")
                    return {
                        'thumbnail_url': cached['url'],
                        'source': cached['source'],
                        'has_thumbnail': True,
                        'extracted': False,
                        'movie_id': movie_id,
                        'stored_in_db': False,  # IMPORTANT: API result NOT stored in DB
                        'from_cache': True
                    }
            
            # Live API fetch
            api_thumbnail = await self._fetch_from_apis(title)
            
            if api_thumbnail:
                # Store in temporary memory cache (5 minutes only)
                self.api_cache[api_cache_key] = {
                    'url': api_thumbnail['url'],
                    'source': api_thumbnail['source'],
                    'timestamp': time.time()
                }
                
                self.stats['from_api_live'] += 1
                logger.info(f"✅ Live API result: {title} ({api_thumbnail['source']})")
                
                return {
                    'thumbnail_url': api_thumbnail['url'],
                    'source': api_thumbnail['source'],
                    'has_thumbnail': True,
                    'extracted': False,
                    'movie_id': movie_id,
                    'stored_in_db': False,  # IMPORTANT: API result NOT stored in DB
                    'from_cache': False
                }
            
            # =============================================
            # PRIORITY 3: LAST FALLBACK POSTER
            # =============================================
            logger.warning(f"⚠️ No thumbnail found, using fallback: {title}")
            
            self.stats['from_fallback'] += 1
            return {
                'thumbnail_url': self.FALLBACK_POSTER,
                'source': 'fallback',
                'has_thumbnail': True,
                'extracted': False,
                'movie_id': movie_id,
                'stored_in_db': False,
                'is_fallback': True
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting thumbnail for {title}: {e}")
            
            # Return fallback on error
            return {
                'thumbnail_url': self.FALLBACK_POSTER,
                'source': 'error_fallback',
                'has_thumbnail': True,
                'extracted': False,
                'stored_in_db': False,
                'is_fallback': True
            }
    
    async def _get_extracted_from_database(self, movie_id: str) -> Optional[Dict]:
        """
        Get ONLY extracted thumbnail from database
        API results database mein nahi hote
        """
        try:
            doc = await self.extracted_thumbnails_col.find_one({"movie_id": movie_id})
            
            if doc and doc.get('thumbnail_url'):
                # Update last accessed
                await self.extracted_thumbnails_col.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"last_accessed": datetime.now()}}
                )
                
                return {
                    'thumbnail_url': doc['thumbnail_url'],
                    'source': 'telegram_extracted_db',
                    'has_thumbnail': True,
                    'extracted': True,
                    'movie_id': movie_id,
                    'stored_in_db': True,
                    'extracted_at': doc.get('created_at')
                }
            
            return None
        except Exception as e:
            logger.debug(f"Extracted DB fetch error: {e}")
            return None
    
    async def _save_extracted_to_database(self, movie_id: str, thumbnail_url: str, 
                                        title: str, channel_id: int = None, message_id: int = None):
        """
        Save ONLY extracted thumbnail to database
        API results ismein save nahi hote
        """
        try:
            doc = {
                'movie_id': movie_id,
                'thumbnail_url': thumbnail_url,
                'title': title,
                'normalized_title': self._normalize_title(title),
                'source': 'telegram_extracted',
                'last_accessed': datetime.now(),
                'updated_at': datetime.now(),
                'extracted': True
            }
            
            if channel_id:
                doc['channel_id'] = channel_id
            if message_id:
                doc['message_id'] = message_id
            
            # Upsert for "Ek movie → Ek extracted thumbnail"
            await self.extracted_thumbnails_col.update_one(
                {'movie_id': movie_id},
                {'$set': doc, '$setOnInsert': {'created_at': datetime.now()}},
                upsert=True
            )
            
            logger.debug(f"✅ Extracted thumbnail saved to DB: {title}")
            
        except Exception as e:
            logger.error(f"❌ Extracted DB save error: {e}")
    
    async def _extract_from_telegram(self, channel_id: int, message_id: int) -> Optional[str]:
        """Extract thumbnail from Telegram message"""
        try:
            if not self.bot_handler:
                return None
            
            # Try multiple extraction methods
            methods = [
                self._extract_video_thumbnail,
                self._extract_document_thumbnail,
                self._extract_photo_preview
            ]
            
            for method in methods:
                try:
                    thumbnail = await method(channel_id, message_id)
                    if thumbnail:
                        return thumbnail
                except Exception:
                    continue
            
            return None
            
        except Exception as e:
            logger.debug(f"Telegram extraction failed: {e}")
            return None
    
    async def _extract_video_thumbnail(self, channel_id: int, message_id: int) -> Optional[str]:
        """Extract from video thumbnail"""
        try:
            message = await self.bot_handler.get_message(channel_id, message_id)
            if not message or not hasattr(message, 'video'):
                return None
            
            if message.video and hasattr(message.video, 'thumbs'):
                for thumb in message.video.thumbs:
                    if hasattr(thumb, 'bytes'):
                        return f"data:image/jpeg;base64,{base64.b64encode(thumb.bytes).decode()}"
            
            return None
        except Exception:
            return None
    
    async def _extract_document_thumbnail(self, channel_id: int, message_id: int) -> Optional[str]:
        """Extract from document thumbnail"""
        try:
            message = await self.bot_handler.get_message(channel_id, message_id)
            if not message or not hasattr(message, 'document'):
                return None
            
            if message.document:
                mime_type = getattr(message.document, 'mime_type', '')
                if 'video' in mime_type and hasattr(message.document, 'thumbs'):
                    for thumb in message.document.thumbs:
                        if hasattr(thumb, 'bytes'):
                            return f"data:image/jpeg;base64,{base64.b64encode(thumb.bytes).decode()}"
            
            return None
        except Exception:
            return None
    
    async def _extract_photo_preview(self, channel_id: int, message_id: int) -> Optional[str]:
        """Extract from photo preview"""
        try:
            message = await self.bot_handler.get_message(channel_id, message_id)
            if not message or not hasattr(message, 'photo'):
                return None
            
            if message.photo:
                # Download smallest photo for thumbnail
                file = await self.bot_handler.download_media(message.photo)
                if file:
                    return f"data:image/jpeg;base64,{base64.b64encode(file).decode()}"
            
            return None
        except Exception:
            return None
    
    async def _fetch_from_apis(self, title: str) -> Optional[Dict]:
        """Fetch from all API services in parallel - NO DATABASE STORAGE"""
        clean_title = self._clean_title_for_api(title)
        
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
    
    # ============================================================================
    # API SERVICE METHODS - LIVE FETCH ONLY, NO STORAGE
    # ============================================================================
    
    async def _fetch_from_tmdb(self, title: str) -> Optional[Dict]:
        """Fetch from TMDB - No storage"""
        try:
            if not self.tmdb_api_key:
                return None
            
            async with aiohttp.ClientSession() as session:
                url = "https://api.themoviedb.org/3/search/movie"
                params = {
                    'api_key': self.tmdb_api_key,
                    'query': title,
                    'language': 'en-US',
                    'page': 1
                }
                
                async with session.get(url, params=params, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('results') and len(data['results']) > 0:
                            poster = data['results'][0].get('poster_path')
                            if poster:
                                return {
                                    'url': f"https://image.tmdb.org/t/p/w500{poster}",
                                    'source': 'tmdb_live'
                                }
            
            return None
        except Exception:
            return None
    
    async def _fetch_from_omdb(self, title: str) -> Optional[Dict]:
        """Fetch from OMDB - No storage"""
        try:
            if not self.omdb_api_key:
                return None
            
            async with aiohttp.ClientSession() as session:
                url = "http://www.omdbapi.com/"
                params = {
                    't': title,
                    'apikey': self.omdb_api_key
                }
                
                async with session.get(url, params=params, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('Poster') and data['Poster'] != 'N/A':
                            return {
                                'url': data['Poster'],
                                'source': 'omdb_live'
                            }
            
            return None
        except Exception:
            return None
    
    async def _fetch_from_imdb(self, title: str) -> Optional[Dict]:
        """Fetch from IMDb - No storage"""
        try:
            # Extract IMDb ID if present
            imdb_match = re.search(r'tt\d{7,8}', title)
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
                                    'source': 'imdb_live'
                                }
            
            return None
        except Exception:
            return None
    
    async def _fetch_from_letterboxd(self, title: str) -> Optional[Dict]:
        """Fetch from Letterboxd - No storage"""
        try:
            # Create slug
            slug = re.sub(r'[^\w\s-]', '', title).strip().lower()
            slug = re.sub(r'[-\s]+', '-', slug)
            
            async with aiohttp.ClientSession() as session:
                url = f"https://letterboxd.com/film/{slug}/"
                headers = {'User-Agent': 'Mozilla/5.0'}
                
                async with session.get(url, headers=headers, timeout=5) as response:
                    if response.status == 200:
                        html = await response.text()
                        match = re.search(r'property="og:image" content="([^"]+)"', html)
                        if match:
                            return {
                                'url': match.group(1),
                                'source': 'letterboxd_live'
                            }
            
            return None
        except Exception:
            return None
    
    async def _fetch_from_wikipedia(self, title: str) -> Optional[Dict]:
        """Fetch from Wikipedia - No storage"""
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
                                    'source': 'wikipedia_live'
                                }
            
            return None
        except Exception:
            return None
    
    async def _fetch_from_youtube(self, title: str) -> Optional[Dict]:
        """Fetch from YouTube - No storage"""
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
                            thumbs = items[0]['snippet'].get('thumbnails', {})
                            if 'high' in thumbs:
                                return {
                                    'url': thumbs['high']['url'],
                                    'source': 'youtube_live'
                                }
            
            return None
        except Exception:
            return None
    
    async def _fetch_from_google_images(self, title: str) -> Optional[Dict]:
        """Fetch from Google Images - No storage"""
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://api.duckduckgo.com/"
                params = {
                    'q': f"{title} movie poster",
                    'format': 'json',
                    'no_html': '1'
                }
                
                async with session.get(url, params=params, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('Image'):
                            return {
                                'url': data['Image'],
                                'source': 'duckduckgo_live'
                            }
            
            return None
        except Exception:
            return None
    
    async def get_thumbnails_batch(self, movies: List[Dict]) -> List[Dict]:
        """Get thumbnails for batch of movies"""
        try:
            results = []
            
            for movie in movies:
                title = movie.get('title', '')
                channel_id = movie.get('channel_id')
                message_id = movie.get('message_id') or movie.get('real_message_id')
                
                thumbnail = await self.get_thumbnail_for_movie(title, channel_id, message_id)
                
                # Merge with movie data
                movie_with_thumb = movie.copy()
                movie_with_thumb.update(thumbnail)
                results.append(movie_with_thumb)
            
            # Calculate stats
            extracted_count = sum(1 for r in results if r.get('source', '').startswith('telegram'))
            api_count = sum(1 for r in results if r.get('source', '').endswith('_live'))
            fallback_count = sum(1 for r in results if r.get('is_fallback'))
            
            logger.info(f"📊 Batch results: {extracted_count} extracted, {api_count} API, {fallback_count} fallback")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Batch error: {e}")
            return [{
                **movie,
                'thumbnail_url': self.FALLBACK_POSTER,
                'source': 'error_fallback',
                'has_thumbnail': True,
                'extracted': False,
                'stored_in_db': False,
                'is_fallback': True
            } for movie in movies]
    
    async def extract_thumbnails_for_existing_files(self):
        """OLD FILES AUTO MIGRATE - Sirf extracted save honge database mein"""
        if not self.files_col:
            return
        
        logger.info("🔄 Extracting thumbnails for OLD FILES (sirf extracted store honge)...")
        
        try:
            # Get all video files
            cursor = self.files_col.find(
                {'is_video_file': True},
                {'title': 1, 'channel_id': 1, 'real_message_id': 1, '_id': 1}
            )
            
            files = []
            async for doc in cursor:
                files.append({
                    'title': doc.get('title', ''),
                    'channel_id': doc.get('channel_id'),
                    'message_id': doc.get('real_message_id'),
                    'db_id': doc.get('_id')
                })
            
            logger.info(f"📊 Found {len(files)} old files to process")
            
            if not files:
                logger.info("✅ No old files found")
                return
            
            # Process in batches
            batch_size = 10
            extracted_count = 0
            
            for i in range(0, len(files), batch_size):
                batch = files[i:i + batch_size]
                
                for file_info in batch:
                    # Try to extract thumbnail from Telegram
                    if file_info['channel_id'] and file_info['message_id']:
                        thumbnail = await self._extract_from_telegram(
                            file_info['channel_id'], 
                            file_info['message_id']
                        )
                        
                        if thumbnail:
                            # Generate movie ID
                            movie_id = self._generate_movie_id(file_info['title'])
                            
                            # ✅ SIRF EXTRACTED THUMBNAIL DATABASE MEIN STORE HOTA HAI
                            await self._save_extracted_to_database(
                                movie_id=movie_id,
                                thumbnail_url=thumbnail,
                                title=file_info['title'],
                                channel_id=file_info['channel_id'],
                                message_id=file_info['message_id']
                            )
                            
                            # Update files collection
                            await self.files_col.update_one(
                                {'_id': file_info['db_id']},
                                {'$set': {
                                    'thumbnail_url': thumbnail,
                                    'thumbnail_source': 'telegram_extracted',
                                    'has_thumbnail': True
                                }}
                            )
                            
                            extracted_count += 1
                            logger.debug(f"✅ Extracted & stored: {file_info['title']}")
                
                logger.info(f"🔄 Processed {min(i + batch_size, len(files))}/{len(files)} files")
                
                # Small delay
                if i + batch_size < len(files):
                    await asyncio.sleep(1)
            
            logger.info(f"✅ OLD FILES migration complete: {extracted_count}/{len(files)} extracted & stored")
            
        except Exception as e:
            logger.error(f"❌ Old files migration error: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            total_extracted = await self.extracted_thumbnails_col.count_documents({})
            
            return {
                'storage_policy': 'EXTRACTED ONLY - API results not stored',
                'performance': {
                    'total_requests': self.stats['total_requests'],
                    'from_extracted_db': self.stats['from_extracted_db'],
                    'from_telegram_live': self.stats['from_telegram_live'],
                    'from_api_live': self.stats['from_api_live'],
                    'from_fallback': self.stats['from_fallback']
                },
                'database': {
                    'extracted_thumbnails_count': total_extracted,
                    'api_results_stored': 0,  # API results database mein nahi hote
                    'extracted_storage_size_mb': total_extracted * 0.01  # Approximate
                },
                'features': {
                    'ek_movie_ek_thumbnail': True,
                    'multi_quality_same_thumbnail': True,
                    'extracted_only_storage': True,
                    'api_no_storage': True,
                    'old_files_auto_migrate': True,
                    'new_files_auto_extract': True,
                    'priority_system': 'Extracted (DB) → API (Live) → Fallback'
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Stats error: {e}")
            return {'status': 'error'}
    
    async def cleanup_api_cache(self):
        """Cleanup temporary API cache"""
        current_time = time.time()
        expired_keys = [
            key for key, data in self.api_cache.items()
            if current_time - data['timestamp'] > self.api_cache_ttl
        ]
        
        for key in expired_keys:
            del self.api_cache[key]
        
        if expired_keys:
            logger.debug(f"🧹 Cleaned {len(expired_keys)} expired API cache entries")
    
    async def shutdown(self):
        """Clean shutdown"""
        self.api_cache.clear()
        logger.info("✅ Thumbnail Manager shutdown")
