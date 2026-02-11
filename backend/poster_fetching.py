import asyncio
import re
import base64
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum
import aiohttp
import urllib.parse
import json
import logging

logger = logging.getLogger(__name__)

TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w780"
CUSTOM_POSTER_URL = "https://iili.io/fAeIwv9.th.png"
CACHE_TTL = 3600  # 1 hour


class PosterSource(Enum):
    TMDB = "tmdb"
    OMDB = "omdb"
    LETTERBOXD = "letterboxd"
    IMDB = "imdb"
    JUSTWATCH = "justwatch"
    IMPAWARDS = "impawards"
    CUSTOM = "custom"
    TELEGRAM_EXTRACTED = "telegram_extracted"
    TELEGRAM_EXTRACTED_DB = "telegram_extracted_db"
    FALLBACK = "fallback"


class PosterFetcher:
    def __init__(self, config, redis=None, bot_handler=None, mongo_client=None):
        self.config = config
        self.redis = redis
        self.bot_handler = bot_handler
        self.mongo_client = mongo_client
        
        # Fallback poster URL (LAST RESORT)
        self.FALLBACK_POSTER = getattr(config, 'FALLBACK_POSTER', "https://iili.io/fAeIwv9.th.png")
        
        if mongo_client:
            self.db = mongo_client["sk4film"]
            self.extracted_thumbnails_col = self.db.extracted_thumbnails
            self.files_col = self.db.files
        else:
            self.extracted_thumbnails_col = None
            self.files_col = None

        self.tmdb_keys = getattr(config, "TMDB_KEYS", [])
        self.omdb_keys = getattr(config, "OMDB_KEYS", [])
        self.tmdb_api_key = getattr(config, 'TMDB_API_KEY', '')
        self.omdb_api_key = getattr(config, 'OMDB_API_KEY', '')
        self.youtube_api_key = getattr(config, 'YOUTUBE_API_KEY', '')

        self.poster_cache: Dict[str, tuple] = {}
        self.api_cache = {}
        self.api_cache_ttl = 300  # 5 minutes only
        self.http_session: Optional[aiohttp.ClientSession] = None
        self.lock = asyncio.Lock()

        # Statistics
        self.stats = {
            "tmdb": 0,
            "imdb": 0,
            "letterboxd": 0,
            "justwatch": 0,
            "impawards": 0,
            "omdb": 0,
            "custom": 0,
            "telegram_extracted": 0,
            "cache_hits": 0,
            "total_requests": 0,
            "from_extracted_db": 0,
            "from_telegram_live": 0,
            "from_api_live": 0,
            "from_fallback": 0
        }
        
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

    # -------------------------------------------------
    # IMPROVED TITLE NORMALIZATION
    # -------------------------------------------------
    def _generate_movie_id(self, title: str) -> str:
        """Generate unique movie ID"""
        normalized = self._normalize_title(title)
        movie_hash = hashlib.md5(normalized.encode()).hexdigest()[:12]
        return f"movie_{movie_hash}"

    def _normalize_title(self, title: str) -> str:
        """Normalize title for movie matching"""
        if not title:
            return ""

        title_lower = title.lower().strip()
        
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
            title_lower = re.sub(prefix, '', title_lower, flags=re.IGNORECASE)
        
        # Remove quality indicators
        quality_patterns = [
            r'\s*\d{3,4}p\b',
            r'\s*(?:hd|fhd|uhd|4k|2160p|1080p|720p|480p|360p)\b',
            r'\s*(?:hevc|x265|x264|h264|h265|10bit|av1|avc|aac|dd|dts|ac3)\b',
            r'\s*(?:webrip|web-dl|webdl|bluray|dvdrip|hdtv|brrip|camrip|hdrip|tc|ts|hdtc|hdts|web)\b',
            r'\s*(?:amzn|netflix|nf|zee5|hotstar|prime|disney\+?)\b',
            r'\s*(?:dd5\.1|dd\+|ddp|atmos|dolby|dual\s+audio|multi\s+audio)\b',
            r'\s*\[.*?\]',
            r'\s*\(.*?\)',
            r'\s*\b(?:part\d+|cd\d+|vol\d+|org|uncut|theatrical|director\'?s\s+cut)\b',
            r'\s*\b(?:no\s+ads|with\s+ads|ads\s+free)\b',
        ]
        
        for pattern in quality_patterns:
            title_lower = re.sub(pattern, '', title_lower, flags=re.IGNORECASE)
        
        # Remove years
        title_lower = re.sub(r'\s+\(\s*\d{4}\s*\)$', '', title_lower)
        title_lower = re.sub(r'\s*\d{4}\s*$', '', title_lower)
        
        # Clean up
        title_lower = re.sub(r'\s+', ' ', title_lower).strip()
        
        if len(title_lower) < 3:
            movie_match = re.match(r'^([^\(\[]+?)\s*[\(\[]', title)
            if movie_match:
                return movie_match.group(1).strip().lower()
            return title[:50].strip().lower()
        
        return title_lower

    def _clean_title_for_api(self, title: str) -> str:
        """Clean title for API search"""
        clean_title = self._normalize_title(title)
        
        suffixes = [
            r'\s+full\s+movie$',
            r'\s+complete$',
            r'\s+hd$',
            r'\s+download$',
            r'\s+watch\s+online$',
        ]
        
        for suffix in suffixes:
            clean_title = re.sub(suffix, '', clean_title, flags=re.IGNORECASE)
        
        clean_title = re.sub(r'[^\w\s\-\(\)]', ' ', clean_title)
        clean_title = re.sub(r'\s+', ' ', clean_title).strip()
        
        if len(clean_title) < 10:
            pattern = r'^([^\(\[]+?)\s*[\(\[]'
            match = re.search(pattern, title)
            if match:
                clean_title = match.group(1).strip()
        
        if len(clean_title) < 3:
            clean_title = title[:30].strip()
        
        return clean_title

    # -------------------------------------------------
    # DUAL PRIORITY SYSTEM
    # -------------------------------------------------
    async def get_thumbnail_for_movie_search(self, title: str, channel_id: int = None, message_id: int = None) -> Dict[str, Any]:
        """
        SEARCH RESULTS PRIORITY:
        1. Extracted thumbnail (Telegram files से)
        2. API Services (TMDB, Letterboxd, etc.)
        3. Fallback poster
        """
        self.stats['total_requests'] += 1
        
        try:
            movie_id = self._generate_movie_id(title)
            
            logger.debug(f"🔍 [SEARCH] Looking for thumbnail: {title}")
            
            # PRIORITY 1: EXTRACTED THUMBNAIL
            extracted_thumbnail = await self._get_extracted_from_database(movie_id)
            if extracted_thumbnail and extracted_thumbnail.get('poster_url'):
                self.stats['from_extracted_db'] += 1
                logger.info(f"✅ [SEARCH] Extracted thumbnail found in DB: {title}")
                return self.normalize_poster(extracted_thumbnail, title)
            
            # EXTRACT FROM TELEGRAM IF POSSIBLE
            if channel_id and message_id and self.bot_handler:
                telegram_thumbnail = await self._extract_from_telegram(channel_id, message_id)
                
                if telegram_thumbnail:
                    await self._save_extracted_to_database(
                        movie_id=movie_id,
                        thumbnail_url=telegram_thumbnail,
                        title=title,
                        channel_id=channel_id,
                        message_id=message_id
                    )
                    
                    self.stats['from_telegram_live'] += 1
                    logger.info(f"✅ [SEARCH] New Telegram thumbnail extracted: {title}")
                    
                    return {
                        'poster_url': telegram_thumbnail,
                        'source': PosterSource.TELEGRAM_EXTRACTED.value,
                        'title': title,
                        'year': '',
                        'rating': '0.0',
                        'has_thumbnail': True,
                        'extracted': True,
                        'movie_id': movie_id,
                        'stored_in_db': True,
                        'is_fallback': False
                    }
            
            # PRIORITY 2: API SERVICES
            api_result = await self._get_poster_from_api(title)
            if api_result:
                return api_result
            
            # PRIORITY 3: FALLBACK POSTER
            logger.warning(f"⚠️ [SEARCH] No thumbnail found, using fallback: {title}")
            
            self.stats['from_fallback'] += 1
            return self._create_fallback_response(title, movie_id)
            
        except Exception as e:
            logger.error(f"❌ [SEARCH] Error getting thumbnail for {title}: {e}")
            return self._create_fallback_response(title, "", is_error=True)

    async def get_thumbnail_for_movie_home(self, title: str) -> Dict[str, Any]:
        """
        HOME MOVIES PRIORITY (POSTS ONLY):
        1. API Services (TMDB, Letterboxd, etc.)
        2. Extracted thumbnail (if exists)
        3. Fallback poster
        """
        self.stats['total_requests'] += 1
        
        try:
            movie_id = self._generate_movie_id(title)
            
            logger.debug(f"🔍 [HOME] Looking for thumbnail: {title}")
            
            # PRIORITY 1: API SERVICES
            api_result = await self._get_poster_from_api(title)
            if api_result:
                return api_result
            
            # PRIORITY 2: EXTRACTED THUMBNAIL
            extracted_thumbnail = await self._get_extracted_from_database(movie_id)
            if extracted_thumbnail and extracted_thumbnail.get('poster_url'):
                self.stats['from_extracted_db'] += 1
                logger.info(f"✅ [HOME] Extracted thumbnail found in DB: {title}")
                return self.normalize_poster(extracted_thumbnail, title)
            
            # PRIORITY 3: FALLBACK POSTER
            logger.warning(f"⚠️ [HOME] No thumbnail found, using fallback: {title}")
            
            self.stats['from_fallback'] += 1
            return self._create_fallback_response(title, movie_id)
            
        except Exception as e:
            logger.error(f"❌ [HOME] Error getting thumbnail for {title}: {e}")
            return self._create_fallback_response(title, "", is_error=True)

    # -------------------------------------------------
    # HELPER METHODS
    # -------------------------------------------------
    async def _get_poster_from_api(self, title: str) -> Optional[Dict[str, Any]]:
        """Get poster from API services"""
        clean_title = self._clean_title_for_api(title)
        
        if len(clean_title) < 3:
            original_parts = title.split()
            if len(original_parts) > 2:
                clean_title = ' '.join(original_parts[:3])
            else:
                clean_title = title[:30]
        
        logger.debug(f"🔍 [API] Searching for: {clean_title}")
        
        import time
        api_cache_key = f"api_{hashlib.md5(clean_title.encode()).hexdigest()[:12]}"
        if api_cache_key in self.api_cache:
            cached = self.api_cache[api_cache_key]
            if time.time() - cached['timestamp'] < self.api_cache_ttl:
                self.stats['from_api_live'] += 1
                logger.debug(f"✅ [API] Cache hit: {clean_title}")
                return {
                    'poster_url': cached['url'],
                    'source': cached['source'],
                    'title': title,
                    'year': cached.get('year', ''),
                    'rating': cached.get('rating', '0.0'),
                    'has_thumbnail': True,
                    'extracted': False,
                    'stored_in_db': False,
                    'from_cache': True,
                    'is_fallback': False
                }
        
        api_thumbnail = await self._fetch_from_apis(clean_title)
        
        if api_thumbnail and api_thumbnail.get('url'):
            self.api_cache[api_cache_key] = {
                'url': api_thumbnail['url'],
                'source': api_thumbnail['source'],
                'year': api_thumbnail.get('year', ''),
                'rating': api_thumbnail.get('rating', '0.0'),
                'timestamp': time.time()
            }
            
            self.stats['from_api_live'] += 1
            logger.info(f"✅ [API] Live result: {title} ({api_thumbnail['source']})")
            
            return {
                'poster_url': api_thumbnail['url'],
                'source': api_thumbnail['source'],
                'title': title,
                'year': api_thumbnail.get('year', ''),
                'rating': api_thumbnail.get('rating', '0.0'),
                'has_thumbnail': True,
                'extracted': False,
                'movie_id': self._generate_movie_id(title),
                'stored_in_db': False,
                'from_cache': False,
                'is_fallback': False
            }
        
        return None

    def _create_fallback_response(self, title: str, movie_id: str = "", is_error: bool = False) -> Dict[str, Any]:
        """Create fallback response"""
        return {
            'poster_url': self.FALLBACK_POSTER,
            'source': 'error_fallback' if is_error else PosterSource.FALLBACK.value,
            'title': title,
            'year': '',
            'rating': '0.0',
            'has_thumbnail': True,
            'extracted': False,
            'movie_id': movie_id or self._generate_movie_id(title),
            'stored_in_db': False,
            'is_fallback': True
        }

    # -------------------------------------------------
    # DATABASE OPERATIONS
    # -------------------------------------------------
    async def _get_extracted_from_database(self, movie_id: str) -> Optional[Dict]:
        """Get extracted thumbnail from database"""
        try:
            if self.extracted_thumbnails_col is None:
                return None
                
            doc = await self.extracted_thumbnails_col.find_one({"movie_id": movie_id})
            
            if doc and doc.get('thumbnail_url'):
                await self.extracted_thumbnails_col.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"last_accessed": datetime.now()}}
                )
                
                return {
                    'poster_url': doc['thumbnail_url'],
                    'source': PosterSource.TELEGRAM_EXTRACTED_DB.value,
                    'title': doc.get('title', ''),
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
        """Save extracted thumbnail to database"""
        try:
            if self.extracted_thumbnails_col is None:
                return
                
            doc = {
                'movie_id': movie_id,
                'thumbnail_url': thumbnail_url,
                'title': title,
                'normalized_title': self._normalize_title(title),
                'source': PosterSource.TELEGRAM_EXTRACTED.value,
                'last_accessed': datetime.now(),
                'updated_at': datetime.now(),
                'extracted': True
            }
            
            if channel_id:
                doc['channel_id'] = channel_id
            if message_id:
                doc['message_id'] = message_id
            
            await self.extracted_thumbnails_col.update_one(
                {'movie_id': movie_id},
                {'$set': doc, '$setOnInsert': {'created_at': datetime.now()}},
                upsert=True
            )
            
            logger.debug(f"✅ Extracted thumbnail saved to DB: {title}")
            
        except Exception as e:
            logger.error(f"❌ Extracted DB save error: {e}")

    # -------------------------------------------------
    # TELEGRAM THUMBNAIL EXTRACTION
    # -------------------------------------------------
    async def _extract_from_telegram(self, channel_id: int, message_id: int) -> Optional[str]:
        """Extract thumbnail from Telegram message"""
        try:
            if not self.bot_handler:
                return None
            
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
                file = await self.bot_handler.download_media(message.photo)
                if file:
                    return f"data:image/jpeg;base64,{base64.b64encode(file).decode()}"
            
            return None
        except Exception:
            return None

    # -------------------------------------------------
    # BATCH OPERATIONS
    # -------------------------------------------------
    async def get_thumbnails_batch(self, movies: List[Dict], mode: str = "search") -> List[Dict]:
        """Get thumbnails for batch of movies"""
        try:
            results = []
            
            for movie in movies:
                title = movie.get('title', '')
                channel_id = movie.get('channel_id')
                message_id = movie.get('message_id') or movie.get('real_message_id')
                
                if mode == "home":
                    thumbnail = await self.get_thumbnail_for_movie_home(title)
                else:
                    thumbnail = await self.get_thumbnail_for_movie_search(title, channel_id, message_id)
                
                movie_with_thumb = movie.copy()
                movie_with_thumb.update(thumbnail)
                results.append(movie_with_thumb)
            
            # Calculate stats
            extracted_count = sum(1 for r in results if r.get('source', '').startswith('telegram'))
            api_count = sum(1 for r in results if r.get('source', '').endswith('_live'))
            fallback_count = sum(1 for r in results if r.get('is_fallback'))
            
            logger.info(f"📊 [{mode.upper()}] Batch results: {extracted_count} extracted, {api_count} API, {fallback_count} fallback")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ [{mode.upper()}] Batch error: {e}")
            return [{
                **movie,
                'poster_url': self.FALLBACK_POSTER,
                'source': 'error_fallback',
                'has_thumbnail': True,
                'extracted': False,
                'stored_in_db': False,
                'is_fallback': True
            } for movie in movies]

    # -------------------------------------------------
    # API FETCHING METHODS
    # -------------------------------------------------
    async def _fetch_from_apis(self, title: str) -> Optional[Dict]:
        """Fetch from all API services"""
        if not title or len(title) < 3:
            return None
        
        title_variations = self._generate_title_variations(title)
        
        logger.debug(f"🔍 Trying API variations for: {title}")
        
        tasks = []
        for api_service in self.api_services:
            for title_var in title_variations[:3]:
                task = asyncio.create_task(api_service(title_var))
                tasks.append(task)
        
        try:
            done, pending = await asyncio.wait(tasks, timeout=5.0, return_when=asyncio.FIRST_COMPLETED)
            
            for task in pending:
                task.cancel()
            
            for task in done:
                try:
                    result = await task
                    if result and result.get('url'):
                        logger.debug(f"✅ API success with: {result.get('source')}")
                        return result
                except Exception as e:
                    logger.debug(f"API task error: {e}")
                    continue
                    
        except asyncio.TimeoutError:
            logger.debug(f"⏱️ API timeout for: {title}")
        
        return None

    def _generate_title_variations(self, title: str) -> List[str]:
        """Generate multiple title variations"""
        variations = []
        variations.append(title)
        
        title_no_year = re.sub(r'\s+\(\d{4}\)$', '', title)
        title_no_year = re.sub(r'\s+\d{4}$', '', title_no_year)
        if title_no_year != title:
            variations.append(title_no_year)
        
        common_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        words = title.split()
        filtered_words = [w for w in words if w.lower() not in common_words]
        if len(filtered_words) > 1:
            variations.append(' '.join(filtered_words))
        
        if any(word.lower() in ['hindi', 'tamil', 'telugu', 'malayalam', 'kannada', 'bengali'] for word in words):
            lang_pattern = r'\b(hindi|tamil|telugu|malayalam|kannada|bengali|english)\b'
            title_no_lang = re.sub(lang_pattern, '', title, flags=re.IGNORECASE)
            title_no_lang = re.sub(r'\s+', ' ', title_no_lang).strip()
            if title_no_lang and len(title_no_lang) > 3:
                variations.append(title_no_lang)
        
        title_clean = re.sub(r'[^\w\s\-]', ' ', title)
        title_clean = re.sub(r'\s+', ' ', title_clean).strip()
        if title_clean != title and len(title_clean) > 3:
            variations.append(title_clean)
        
        unique_variations = []
        seen = set()
        for var in variations:
            if var and var not in seen:
                seen.add(var)
                unique_variations.append(var)
        
        return unique_variations[:5]

    async def _fetch_from_tmdb(self, title: str) -> Optional[Dict]:
        """Fetch from TMDB"""
        try:
            if not self.tmdb_api_key:
                return None
            
            session = await self.get_http_session()
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
                                'source': 'tmdb_live',
                                'year': (data['results'][0].get('release_date') or '')[:4],
                                'rating': str(data['results'][0].get('vote_average', '0.0'))
                            }
            
            return None
        except Exception:
            return None

    async def _fetch_from_omdb(self, title: str) -> Optional[Dict]:
        """Fetch from OMDB"""
        try:
            if not self.omdb_api_key:
                return None
            
            session = await self.get_http_session()
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
                            'source': 'omdb_live',
                            'year': data.get('Year', ''),
                            'rating': data.get('imdbRating', '0.0')
                        }
            
            return None
        except Exception:
            return None

    async def _fetch_from_imdb(self, title: str) -> Optional[Dict]:
        """Fetch from IMDb"""
        try:
            imdb_match = re.search(r'tt\d{7,8}', title)
            if imdb_match and self.omdb_api_key:
                imdb_id = imdb_match.group()
                
                session = await self.get_http_session()
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
                                'source': 'imdb_live',
                                'year': data.get('Year', ''),
                                'rating': data.get('imdbRating', '0.0')
                            }
            
            return None
        except Exception:
            return None

    async def _fetch_from_letterboxd(self, title: str) -> Optional[Dict]:
        """Fetch from Letterboxd"""
        try:
            slug = re.sub(r'[^\w\s\-]', '', title).strip().lower()
            slug = re.sub(r'[-\s]+', '-', slug)
            slug = re.sub(r'-\d{4}$', '', slug)
            
            session = await self.get_http_session()
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
        """Fetch from Wikipedia"""
        try:
            session = await self.get_http_session()
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
        """Fetch from YouTube"""
        try:
            if not self.youtube_api_key:
                return None
            
            session = await self.get_http_session()
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
        """Fetch from Google Images"""
        try:
            session = await self.get_http_session()
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

    # -------------------------------------------------
    # ORIGINAL POSTER FETCHING (FOR COMPATIBILITY)
    # -------------------------------------------------
    async def fetch_poster(self, title: str) -> Dict[str, Any]:
        """Original poster fetching function"""
        key = f"poster:{title.lower().strip()}"

        cached = await self.redis_get(key)
        if cached:
            return cached

        if key in self.poster_cache:
            data, ts = self.poster_cache[key]
            import time
            if time.time() - ts < CACHE_TTL:
                self.stats["cache_hits"] += 1
                return data

        sources = [
            self.fetch_from_tmdb,
            self.fetch_from_omdb,
            self.fetch_from_letterboxd,
            self.fetch_from_imdb,
            self.fetch_from_justwatch,
            self.fetch_from_impawards,
        ]

        results = await asyncio.gather(*sources, return_exceptions=True)

        for r in results:
            if isinstance(r, dict) and r.get("poster_url", "").startswith("http"):
                normalized = self.normalize_poster(r, title)
                self.poster_cache[key] = (normalized, datetime.now())
                await self.redis_set(key, normalized)
                return normalized

        custom = await self.create_custom_poster(title)
        normalized = self.normalize_poster(custom, title)
        self.poster_cache[key] = (normalized, datetime.now())
        await self.redis_set(key, normalized)
        return normalized

    # Original API methods for compatibility
    async def fetch_from_tmdb(self, title: str):
        session = await self.get_http_session()
        for key in self.tmdb_keys:
            try:
                async with session.get(
                    "https://api.themoviedb.org/3/search/movie",
                    params={"api_key": key, "query": title},
                ) as r:
                    if r.status != 200:
                        continue
                    data = await r.json()
                    if not data.get("results"):
                        continue

                    m = data["results"][0]
                    if not m.get("poster_path"):
                        continue

                    self.stats["tmdb"] += 1
                    return {
                        "poster_url": f"{TMDB_IMAGE_BASE}{m['poster_path']}",
                        "source": PosterSource.TMDB.value,
                        "rating": str(m.get("vote_average", "0.0")),
                        "year": (m.get("release_date") or "")[:4],
                        "title": m.get("title", title),
                    }
            except Exception as e:
                logger.error(f"TMDB error: {e}")
        return None

    async def fetch_from_omdb(self, title: str):
        session = await self.get_http_session()
        for key in self.omdb_keys:
            try:
                async with session.get(
                    f"https://www.omdbapi.com/?t={urllib.parse.quote(title)}&apikey={key}"
                ) as r:
                    data = await r.json()
                    poster = data.get("Poster")
                    if poster and poster.startswith("http"):
                        self.stats["omdb"] += 1
                        return {
                            "poster_url": poster,
                            "source": PosterSource.OMDB.value,
                            "rating": data.get("imdbRating", "0.0"),
                            "year": data.get("Year", ""),
                            "title": data.get("Title", title),
                        }
            except Exception:
                pass
        return None

    async def fetch_from_imdb(self, title: str):
        session = await self.get_http_session()
        try:
            clean = re.sub(r"[^\w\s]", "", title).strip()
            if not clean:
                return None
            url = f"https://v2.sg.media-imdb.com/suggestion/{clean[0].lower()}/{urllib.parse.quote(clean.replace(' ', '_'))}.json"
            async with session.get(url) as r:
                data = await r.json()
                for item in data.get("d", []):
                    img = item.get("i")
                    poster = (
                        img.get("imageUrl")
                        if isinstance(img, dict)
                        else img[0]
                        if isinstance(img, list)
                        else img
                        if isinstance(img, str)
                        else ""
                    )
                    if poster.startswith("http"):
                        self.stats["imdb"] += 1
                        return {
                            "poster_url": poster,
                            "source": PosterSource.IMDB.value,
                            "year": str(item.get("yr", "")),
                            "title": item.get("l", title),
                            "rating": "0.0",
                        }
        except Exception:
            pass
        return None

    async def fetch_from_letterboxd(self, title: str):
        session = await self.get_http_session()
        slug = re.sub(r"[^\w\s]", "", title).lower().replace(" ", "-")
        try:
            async with session.get(f"https://letterboxd.com/film/{slug}/") as r:
                html = await r.text()
                m = re.search(r'property="og:image" content="([^"]+)"', html)
                if m:
                    self.stats["letterboxd"] += 1
                    return {
                        "poster_url": m.group(1),
                        "source": PosterSource.LETTERBOXD.value,
                        "title": title,
                        "rating": "0.0",
                    }
        except Exception:
            pass
        return None

    async def fetch_from_justwatch(self, title: str):
        session = await self.get_http_session()
        slug = re.sub(r"[^\w\s]", "", title).lower().replace(" ", "-")
        try:
            async with session.get(f"https://www.justwatch.com/in/movie/{slug}") as r:
                html = await r.text()
                m = re.search(r'property="og:image" content="([^"]+)"', html)
                if m:
                    self.stats["justwatch"] += 1
                    return {
                        "poster_url": m.group(1),
                        "source": PosterSource.JUSTWATCH.value,
                        "title": title,
                        "rating": "0.0",
                    }
        except Exception:
            pass
        return None

    async def fetch_from_impawards(self, title: str):
        session = await self.get_http_session()
        year = re.search(r"\b(19|20)\d{2}\b", title)
        if not year:
            return None
        clean = re.sub(r"\b(19|20)\d{2}\b", "", title).strip().replace(" ", "_")
        url = f"https://www.impawards.com/{year.group()}/posters/{clean}.jpg"
        try:
            async with session.head(url) as r:
                if r.status == 200:
                    self.stats["impawards"] += 1
                    return {
                        "poster_url": url,
                        "source": PosterSource.IMPAWARDS.value,
                        "title": title,
                        "year": year.group(),
                        "rating": "0.0",
                    }
        except Exception:
            pass
        return None

    async def create_custom_poster(self, title: str):
        self.stats["custom"] += 1
        return {
            "poster_url": CUSTOM_POSTER_URL,
            "source": PosterSource.CUSTOM.value,
            "title": title,
            "year": "",
            "rating": "0.0",
        }

    async def fetch_batch_posters(self, titles: List[str]) -> Dict[str, Dict]:
        posters = await asyncio.gather(*(self.fetch_poster(t) for t in titles))
        return dict(zip(titles, posters))

    # -------------------------------------------------
    # UTILITY METHODS
    # -------------------------------------------------
    async def get_http_session(self):
        async with self.lock:
            if not self.http_session or self.http_session.closed:
                self.http_session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=10),
                    headers={"User-Agent": "Mozilla/5.0"},
                )
            return self.http_session

    async def redis_get(self, key: str):
        if not self.redis:
            return None
        try:
            data = await self.redis.get(key)
            if data:
                self.stats["cache_hits"] += 1
                return json.loads(data)
        except Exception:
            pass
        return None

    async def redis_set(self, key: str, value: Dict[str, Any]):
        if not self.redis:
            return
        try:
            await self.redis.setex(key, CACHE_TTL, json.dumps(value))
        except Exception:
            pass

    def normalize_poster(self, poster: Dict[str, Any], title: str) -> Dict[str, Any]:
        return {
            "poster_url": poster.get("poster_url", ""),
            "source": poster.get("source", PosterSource.CUSTOM.value),
            "title": poster.get("title", title),
            "year": poster.get("year", ""),
            "rating": poster.get("rating", "0.0"),
            "has_thumbnail": poster.get("has_thumbnail", False),
            "extracted": poster.get("extracted", False),
            "stored_in_db": poster.get("stored_in_db", False),
            "is_fallback": poster.get("is_fallback", False)
        }

    async def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            total_extracted = 0
            if self.extracted_thumbnails_col is not None:
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
                'poster_stats': {
                    'tmdb': self.stats['tmdb'],
                    'imdb': self.stats['imdb'],
                    'letterboxd': self.stats['letterboxd'],
                    'justwatch': self.stats['justwatch'],
                    'impawards': self.stats['impawards'],
                    'omdb': self.stats['omdb'],
                    'custom': self.stats['custom'],
                    'telegram_extracted': self.stats['telegram_extracted'],
                    'cache_hits': self.stats['cache_hits']
                },
                'database': {
                    'extracted_thumbnails_count': total_extracted,
                    'api_results_stored': 0,
                    'extracted_storage_size_mb': total_extracted * 0.01
                },
                'features': {
                    'ek_movie_ek_thumbnail': True,
                    'multi_quality_same_thumbnail': True,
                    'extracted_only_storage': True,
                    'api_no_storage': True,
                    'old_files_auto_migrate': True,
                    'new_files_auto_extract': True,
                    'dual_priority_system': True,
                    'home_priority': 'Sources → Thumbnail → Fallback',
                    'search_priority': 'Thumbnail → Sources → Fallback',
                    'fallback_image': True
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Stats error: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'stats_available': {
                    'total_requests': self.stats['total_requests'],
                    'from_api_live': self.stats['from_api_live'],
                    'from_fallback': self.stats['from_fallback']
                }
            }

    async def cleanup_api_cache(self):
        """Cleanup temporary API cache"""
        import time
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
        if self.http_session and not self.http_session.closed:
            await self.http_session.close()
        logger.info("✅ PosterFetcher shutdown")
