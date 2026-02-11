# ============================================================================
# poster_fetching.py - DUAL PRIORITY THUMBNAIL & POSTER SYSTEM
# ============================================================================
# ✅ HOME MODE: Sources → Thumbnail → Fallback
# ✅ SEARCH MODE: Thumbnail → Sources → Fallback
# ✅ Extracted Thumbnails from Telegram
# ✅ TMDB/OMDB API Posters
# ✅ MongoDB Caching with TTL
# ✅ Batch Processing Support
# ✅ 99% Success Rate Target
# ============================================================================

import asyncio
import aiohttp
import logging
import time
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import quote
import base64

logger = logging.getLogger(__name__)

class PosterSource:
    """Source constants for poster/thumbnail origin"""
    EXTRACTED = "extracted"  # From Telegram video file
    TMDB = "tmdb"           # From The Movie Database
    OMDB = "omdb"          # From Open Movie Database
    CUSTOM = "custom"      # Custom uploaded thumbnail
    FALLBACK = "fallback"  # Default fallback image
    ERROR = "error"        # Error fallback
    CACHE = "cache"        # From cache


class PosterFetcher:
    """
    Dual Priority Thumbnail & Poster System
    
    HOME MODE: Sources → Thumbnail → Fallback
        - First check poster sources (TMDB/OMDB)
        - Then check extracted thumbnails
        - Finally use fallback image
    
    SEARCH MODE: Thumbnail → Sources → Fallback
        - First check extracted thumbnails
        - Then check poster sources (TMDB/OMDB)
        - Finally use fallback image
    """
    
    def __init__(self, config, cache_manager=None, bot_handler=None, mongo_client=None):
        self.config = config
        self.cache_manager = cache_manager
        self.bot_handler = bot_handler
        self.mongo_client = mongo_client
        
        # MongoDB collections
        self.thumbnails_col = None
        self.poster_cache_col = None
        
        # Statistics tracking
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'extracted_hits': 0,
            'tmdb_hits': 0,
            'omdb_hits': 0,
            'fallback_hits': 0,
            'errors': 0,
            'home_mode_calls': 0,
            'search_mode_calls': 0
        }
        
        # HTTP session
        self.session = None
        self.session_lock = asyncio.Lock()
        
        # TTL settings
        self.poster_cache_ttl = 7 * 24 * 60 * 60  # 7 days
        self.thumbnail_cache_ttl = 30 * 24 * 60 * 60  # 30 days
        
        # Rate limiting
        self.tmdb_last_call = 0
        self.omdb_last_call = 0
        self.rate_limit_delay = 0.25  # 250ms between API calls
        
        # Fallback URL
        self.fallback_url = getattr(config, 'FALLBACK_POSTER', 'https://iili.io/fAeIwv9.th.png')
        
        logger.info("🎬 PosterFetcher initialized with Dual Priority Mode")
    
    async def _ensure_collections(self):
        """Ensure MongoDB collections exist"""
        if self.mongo_client and self.thumbnails_col is None:
            db = self.mongo_client.sk4film
            self.thumbnails_col = db.thumbnails
            self.poster_cache_col = db.poster_cache
            
            # Create indexes
            try:
                # Thumbnails collection - one per normalized title
                await self.thumbnails_col.create_index(
                    "normalized_title",
                    unique=True,
                    name="title_unique"
                )
                await self.thumbnails_col.create_index(
                    "expires_at",
                    expireAfterSeconds=0,
                    name="ttl_cleanup"
                )
                
                # Poster cache collection
                await self.poster_cache_col.create_index(
                    "cache_key",
                    unique=True,
                    name="cache_key_unique"
                )
                await self.poster_cache_col.create_index(
                    "expires_at",
                    expireAfterSeconds=0,
                    name="cache_ttl"
                )
                
                logger.info("✅ PosterFetcher collections initialized with indexes")
            except Exception as e:
                logger.warning(f"⚠️ Index creation error: {e}")
    
    async def _get_session(self):
        """Get or create aiohttp session"""
        async with self.session_lock:
            if self.session is None or self.session.closed:
                self.session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=10),
                    headers={
                        'User-Agent': 'SK4FiLM/9.0 (PosterFetcher)'
                    }
                )
            return self.session
    
    async def close(self):
        """Close HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("✅ PosterFetcher session closed")
    
    # ========================================================================
    # HOME MODE: Sources → Thumbnail → Fallback
    # ========================================================================
    
    async def get_thumbnail_for_movie_home(self, title: str, year: str = "") -> Dict[str, Any]:
        """
        HOME MODE: Sources → Thumbnail → Fallback
        
        Priority Order:
        1. TMDB/OMDB Posters (API sources)
        2. Extracted Thumbnail (if available)
        3. Fallback Image (always works)
        """
        self.stats['total_requests'] += 1
        self.stats['home_mode_calls'] += 1
        
        start_time = time.time()
        
        try:
            # Check cache first
            cached = await self._get_cached_poster(title, year)
            if cached:
                self.stats['cache_hits'] += 1
                cached['source'] = PosterSource.CACHE
                cached['priority_mode'] = 'home'
                cached['response_time'] = time.time() - start_time
                return cached
            
            # STEP 1: Try TMDB/OMDB Sources
            logger.debug(f"[HOME-MODE] Fetching poster from sources: {title}")
            poster_data = await self._fetch_from_sources(title, year)
            
            if poster_data and poster_data.get('poster_url'):
                # Cache the result
                await self._cache_poster(title, year, poster_data)
                
                poster_data['priority_mode'] = 'home'
                poster_data['response_time'] = time.time() - start_time
                poster_data['has_thumbnail'] = True
                poster_data['is_fallback'] = False
                
                logger.info(f"✅ [HOME-MODE] Poster found from {poster_data['source']}: {title[:30]}...")
                return poster_data
            
            # STEP 2: Try Extracted Thumbnail
            logger.debug(f"[HOME-MODE] No poster, trying extracted thumbnail: {title}")
            thumbnail_data = await self._get_extracted_thumbnail(title)
            
            if thumbnail_data and thumbnail_data.get('thumbnail_url'):
                result = {
                    'poster_url': thumbnail_data['thumbnail_url'],
                    'source': PosterSource.EXTRACTED,
                    'extracted': True,
                    'priority_mode': 'home',
                    'has_thumbnail': True,
                    'is_fallback': False,
                    'response_time': time.time() - start_time
                }
                
                self.stats['extracted_hits'] += 1
                logger.info(f"✅ [HOME-MODE] Extracted thumbnail found: {title[:30]}...")
                return result
            
            # STEP 3: Fallback Image (always works)
            logger.debug(f"[HOME-MODE] Using fallback image: {title[:30]}...")
            self.stats['fallback_hits'] += 1
            
            return {
                'poster_url': self.fallback_url,
                'source': PosterSource.FALLBACK,
                'priority_mode': 'home',
                'has_thumbnail': True,
                'is_fallback': True,
                'response_time': time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"❌ [HOME-MODE] Error for {title}: {e}")
            self.stats['errors'] += 1
            
            return {
                'poster_url': self.fallback_url,
                'source': PosterSource.ERROR,
                'priority_mode': 'home',
                'has_thumbnail': True,
                'is_fallback': True,
                'error': str(e),
                'response_time': time.time() - start_time
            }
    
    # ========================================================================
    # SEARCH MODE: Thumbnail → Sources → Fallback
    # ========================================================================
    
    async def get_thumbnail_for_movie_search(self, title: str, year: str = "", 
                                           channel_id: int = None, message_id: int = None) -> Dict[str, Any]:
        """
        SEARCH MODE: Thumbnail → Sources → Fallback
        
        Priority Order:
        1. Extracted Thumbnail (from Telegram video)
        2. TMDB/OMDB Posters (API sources)
        3. Fallback Image (always works)
        """
        self.stats['total_requests'] += 1
        self.stats['search_mode_calls'] += 1
        
        start_time = time.time()
        
        try:
            # Check cache first
            cached = await self._get_cached_poster(title, year)
            if cached:
                self.stats['cache_hits'] += 1
                cached['source'] = PosterSource.CACHE
                cached['priority_mode'] = 'search'
                cached['response_time'] = time.time() - start_time
                return cached
            
            # STEP 1: Try Extracted Thumbnail
            logger.debug(f"[SEARCH-MODE] Fetching extracted thumbnail: {title}")
            
            # If we have channel_id and message_id, try to extract now
            if channel_id and message_id and self.bot_handler:
                thumbnail_data = await self._extract_thumbnail_now(title, channel_id, message_id)
                if thumbnail_data:
                    result = {
                        'poster_url': thumbnail_data['thumbnail_url'],
                        'source': PosterSource.EXTRACTED,
                        'extracted': True,
                        'priority_mode': 'search',
                        'has_thumbnail': True,
                        'is_fallback': False,
                        'response_time': time.time() - start_time
                    }
                    
                    # Cache it
                    await self._cache_poster(title, year, result)
                    
                    self.stats['extracted_hits'] += 1
                    logger.info(f"✅ [SEARCH-MODE] Live thumbnail extracted: {title[:30]}...")
                    return result
            else:
                # Check existing thumbnail in database
                thumbnail_data = await self._get_extracted_thumbnail(title)
                if thumbnail_data and thumbnail_data.get('thumbnail_url'):
                    result = {
                        'poster_url': thumbnail_data['thumbnail_url'],
                        'source': PosterSource.EXTRACTED,
                        'extracted': thumbnail_data.get('extracted', False),
                        'priority_mode': 'search',
                        'has_thumbnail': True,
                        'is_fallback': False,
                        'response_time': time.time() - start_time
                    }
                    
                    self.stats['extracted_hits'] += 1
                    logger.debug(f"✅ [SEARCH-MODE] Cached thumbnail found: {title[:30]}...")
                    return result
            
            # STEP 2: Try TMDB/OMDB Sources
            logger.debug(f"[SEARCH-MODE] No thumbnail, trying poster sources: {title}")
            poster_data = await self._fetch_from_sources(title, year)
            
            if poster_data and poster_data.get('poster_url'):
                await self._cache_poster(title, year, poster_data)
                
                poster_data['priority_mode'] = 'search'
                poster_data['response_time'] = time.time() - start_time
                poster_data['has_thumbnail'] = True
                poster_data['is_fallback'] = False
                
                logger.info(f"✅ [SEARCH-MODE] Poster found from {poster_data['source']}: {title[:30]}...")
                return poster_data
            
            # STEP 3: Fallback Image
            logger.debug(f"[SEARCH-MODE] Using fallback image: {title[:30]}...")
            self.stats['fallback_hits'] += 1
            
            return {
                'poster_url': self.fallback_url,
                'source': PosterSource.FALLBACK,
                'priority_mode': 'search',
                'has_thumbnail': True,
                'is_fallback': True,
                'response_time': time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"❌ [SEARCH-MODE] Error for {title}: {e}")
            self.stats['errors'] += 1
            
            return {
                'poster_url': self.fallback_url,
                'source': PosterSource.ERROR,
                'priority_mode': 'search',
                'has_thumbnail': True,
                'is_fallback': True,
                'error': str(e),
                'response_time': time.time() - start_time
            }
    
    # ========================================================================
    # BATCH PROCESSING
    # ========================================================================
    
    async def get_thumbnails_batch(self, movies: List[Dict], mode: str = "search") -> List[Dict]:
        """
        Get thumbnails for multiple movies in batch
        
        Args:
            movies: List of movie dicts with 'title', optional 'year', 'channel_id', 'message_id'
            mode: 'home' or 'search' - determines priority order
        
        Returns:
            List of movie dicts with thumbnail data added
        """
        results = []
        
        for movie in movies:
            title = movie.get('title', '')
            year = movie.get('year', '')
            channel_id = movie.get('channel_id')
            message_id = movie.get('message_id') or movie.get('real_message_id')
            
            if mode == "home":
                thumbnail_data = await self.get_thumbnail_for_movie_home(title, year)
            else:  # search mode
                thumbnail_data = await self.get_thumbnail_for_movie_search(
                    title, year, channel_id, message_id
                )
            
            # Merge with original movie data
            movie_with_thumbnail = movie.copy()
            movie_with_thumbnail.update({
                'poster_url': thumbnail_data.get('poster_url'),
                'thumbnail_url': thumbnail_data.get('poster_url'),  # Alias
                'source': thumbnail_data.get('source'),
                'priority_mode': thumbnail_data.get('priority_mode'),
                'has_thumbnail': thumbnail_data.get('has_thumbnail', False),
                'is_fallback': thumbnail_data.get('is_fallback', True),
                'extracted': thumbnail_data.get('extracted', False),
                'response_time': thumbnail_data.get('response_time', 0)
            })
            
            results.append(movie_with_thumbnail)
        
        # Update stats
        logger.debug(f"✅ Batch processed {len(results)} movies in {mode} mode")
        
        return results
    
    # ========================================================================
    # EXTRACTED THUMBNAIL HANDLING
    # ========================================================================
    
    async def _get_extracted_thumbnail(self, title: str) -> Optional[Dict]:
        """Get existing extracted thumbnail from database"""
        try:
            await self._ensure_collections()
            
            if not self.thumbnails_col:
                return None
            
            normalized_title = self._normalize_title(title)
            
            thumbnail = await self.thumbnails_col.find_one({
                'normalized_title': normalized_title
            })
            
            if thumbnail and thumbnail.get('thumbnail_url'):
                # Check if expired
                expires_at = thumbnail.get('expires_at')
                if expires_at and datetime.now() > expires_at:
                    # Expired, delete it
                    await self.thumbnails_col.delete_one({'_id': thumbnail['_id']})
                    return None
                
                return {
                    'thumbnail_url': thumbnail['thumbnail_url'],
                    'extracted': thumbnail.get('extracted', True),
                    'source': PosterSource.EXTRACTED,
                    'stored_at': thumbnail.get('stored_at')
                }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting extracted thumbnail: {e}")
            return None
    
    async def _extract_thumbnail_now(self, title: str, channel_id: int, message_id: int) -> Optional[Dict]:
        """Extract thumbnail from Telegram message NOW"""
        try:
            if not self.bot_handler or not self.bot_handler.initialized:
                logger.warning("⚠️ Bot handler not available for thumbnail extraction")
                return None
            
            # Extract thumbnail using bot handler
            thumbnail_base64 = await self.bot_handler.extract_thumbnail(channel_id, message_id)
            
            if thumbnail_base64:
                # Store in database
                await self._store_extracted_thumbnail(title, thumbnail_base64, channel_id, message_id)
                
                return {
                    'thumbnail_url': thumbnail_base64,
                    'extracted': True,
                    'channel_id': channel_id,
                    'message_id': message_id
                }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error extracting thumbnail: {e}")
            return None
    
    async def _store_extracted_thumbnail(self, title: str, thumbnail_base64: str, 
                                       channel_id: int, message_id: int) -> bool:
        """Store extracted thumbnail in database"""
        try:
            await self._ensure_collections()
            
            if not self.thumbnails_col:
                return False
            
            normalized_title = self._normalize_title(title)
            
            # Calculate expiry (30 days from now)
            expires_at = datetime.now() + timedelta(days=self.config.THUMBNAIL_TTL_DAYS)
            
            # Upsert thumbnail
            await self.thumbnails_col.update_one(
                {'normalized_title': normalized_title},
                {'$set': {
                    'title': title,
                    'normalized_title': normalized_title,
                    'thumbnail_url': thumbnail_base64,
                    'channel_id': channel_id,
                    'message_id': message_id,
                    'extracted': True,
                    'stored_at': datetime.now(),
                    'expires_at': expires_at,
                    'updated_at': datetime.now()
                }},
                upsert=True
            )
            
            logger.info(f"✅ Stored extracted thumbnail for: {title[:30]}...")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error storing thumbnail: {e}")
            return False
    
    # ========================================================================
    # POSTER SOURCES (TMDB/OMDB)
    # ========================================================================
    
    async def _fetch_from_sources(self, title: str, year: str = "") -> Optional[Dict]:
        """Fetch poster from TMDB or OMDB"""
        
        # Clean title for API
        clean_title = self._clean_title_for_api(title)
        
        # Try TMDB first
        tmdb_result = await self._fetch_tmdb_poster(clean_title, year)
        if tmdb_result:
            self.stats['tmdb_hits'] += 1
            return tmdb_result
        
        # Try OMDB second
        omdb_result = await self._fetch_omdb_poster(clean_title, year)
        if omdb_result:
            self.stats['omdb_hits'] += 1
            return omdb_result
        
        return None
    
    async def _fetch_tmdb_poster(self, title: str, year: str = "") -> Optional[Dict]:
        """Fetch poster from TMDB API"""
        try:
            # Rate limiting
            now = time.time()
            if now - self.tmdb_last_call < self.rate_limit_delay:
                await asyncio.sleep(self.rate_limit_delay)
            self.tmdb_last_call = now
            
            session = await self._get_session()
            
            # Search for movie
            search_url = "https://api.themoviedb.org/3/search/movie"
            params = {
                'api_key': self.config.TMDB_API_KEY,
                'query': title,
                'include_adult': False
            }
            
            if year:
                params['year'] = year
            
            async with session.get(search_url, params=params) as response:
                if response.status != 200:
                    return None
                
                data = await response.json()
                
                if not data.get('results'):
                    return None
                
                # Get first result
                movie = data['results'][0]
                poster_path = movie.get('poster_path')
                
                if not poster_path:
                    return None
                
                poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
                
                return {
                    'poster_url': poster_url,
                    'source': PosterSource.TMDB,
                    'title': movie.get('title', title),
                    'year': movie.get('release_date', '')[:4] if movie.get('release_date') else year,
                    'rating': str(movie.get('vote_average', '0.0')),
                    'has_thumbnail': True,
                    'is_fallback': False
                }
                
        except Exception as e:
            logger.debug(f"TMDB error for {title}: {e}")
            return None
    
    async def _fetch_omdb_poster(self, title: str, year: str = "") -> Optional[Dict]:
        """Fetch poster from OMDB API"""
        try:
            # Rate limiting
            now = time.time()
            if now - self.omdb_last_call < self.rate_limit_delay:
                await asyncio.sleep(self.rate_limit_delay)
            self.omdb_last_call = now
            
            session = await self._get_session()
            
            # Search for movie
            params = {
                'apikey': self.config.OMDB_API_KEY,
                't': title,
                'plot': 'short'
            }
            
            if year:
                params['y'] = year
            
            async with session.get("http://www.omdbapi.com/", params=params) as response:
                if response.status != 200:
                    return None
                
                data = await response.json()
                
                if data.get('Response') != 'True':
                    return None
                
                poster_url = data.get('Poster')
                
                # OMDB returns "N/A" if no poster
                if not poster_url or poster_url == 'N/A':
                    return None
                
                return {
                    'poster_url': poster_url,
                    'source': PosterSource.OMDB,
                    'title': data.get('Title', title),
                    'year': data.get('Year', ''),
                    'rating': data.get('imdbRating', '0.0'),
                    'has_thumbnail': True,
                    'is_fallback': False
                }
                
        except Exception as e:
            logger.debug(f"OMDB error for {title}: {e}")
            return None
    
    # ========================================================================
    # CACHE MANAGEMENT
    # ========================================================================
    
    async def _get_cached_poster(self, title: str, year: str = "") -> Optional[Dict]:
        """Get cached poster data"""
        try:
            await self._ensure_collections()
            
            if not self.poster_cache_col:
                return None
            
            cache_key = self._make_cache_key(title, year)
            
            cached = await self.poster_cache_col.find_one({'cache_key': cache_key})
            
            if cached and cached.get('poster_url'):
                # Check if expired
                expires_at = cached.get('expires_at')
                if expires_at and datetime.now() > expires_at:
                    await self.poster_cache_col.delete_one({'_id': cached['_id']})
                    return None
                
                return {
                    'poster_url': cached['poster_url'],
                    'source': cached.get('source', PosterSource.CACHE),
                    'title': cached.get('title', title),
                    'year': cached.get('year', year),
                    'rating': cached.get('rating', '0.0'),
                    'has_thumbnail': True,
                    'is_fallback': False
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    async def _cache_poster(self, title: str, year: str, poster_data: Dict) -> bool:
        """Cache poster data in MongoDB"""
        try:
            await self._ensure_collections()
            
            if not self.poster_cache_col:
                return False
            
            cache_key = self._make_cache_key(title, year)
            expires_at = datetime.now() + timedelta(seconds=self.poster_cache_ttl)
            
            await self.poster_cache_col.update_one(
                {'cache_key': cache_key},
                {'$set': {
                    'cache_key': cache_key,
                    'title': title,
                    'year': year,
                    'poster_url': poster_data.get('poster_url'),
                    'source': poster_data.get('source'),
                    'rating': poster_data.get('rating', '0.0'),
                    'cached_at': datetime.now(),
                    'expires_at': expires_at
                }},
                upsert=True
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    # ========================================================================
    # UTILITY FUNCTIONS
    # ========================================================================
    
    def _normalize_title(self, title: str) -> str:
        """Normalize title for consistent lookup"""
        if not title:
            return ""
        
        # Convert to lowercase
        title = title.lower()
        
        # Remove year in parentheses
        title = re.sub(r'\s*\(\d{4}\)$', '', title)
        title = re.sub(r'\s+\d{4}$', '', title)
        
        # Remove quality tags
        title = re.sub(r'\b(480p|720p|1080p|2160p|4k|hd|hevc|x265|x264)\b', '', title)
        
        # Remove special characters
        title = re.sub(r'[^\w\s]', ' ', title)
        
        # Remove extra spaces
        title = re.sub(r'\s+', ' ', title)
        
        return title.strip()
    
    def _clean_title_for_api(self, title: str) -> str:
        """Clean title for API search"""
        if not title:
            return ""
        
        # Remove year in parentheses
        title = re.sub(r'\s*\(\d{4}\)$', '', title)
        title = re.sub(r'\s+\d{4}$', '', title)
        
        # Remove quality indicators
        title = re.sub(r'\b(480p|720p|1080p|2160p|4k|hd|hevc|x265|x264|bluray|webrip|hdtv)\b', '', title, flags=re.IGNORECASE)
        
        # Remove extra spaces
        title = re.sub(r'\s+', ' ', title)
        
        return title.strip()
    
    def _make_cache_key(self, title: str, year: str = "") -> str:
        """Create cache key from title and year"""
        normalized = self._normalize_title(title)
        if year:
            return f"poster:{normalized}:{year}"
        return f"poster:{normalized}"
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        try:
            await self._ensure_collections()
            
            # Count thumbnails in database
            thumbnail_count = 0
            if self.thumbnails_col:
                thumbnail_count = await self.thumbnails_col.count_documents({})
            
            # Count cached posters
            cache_count = 0
            if self.poster_cache_col:
                cache_count = await self.poster_cache_col.count_documents({})
            
            total_requests = self.stats['total_requests'] or 1  # Avoid division by zero
            
            return {
                'dual_priority_system': {
                    'home_mode': 'Sources → Thumbnail → Fallback',
                    'search_mode': 'Thumbnail → Sources → Fallback',
                    'fallback_enabled': True,
                    'fallback_url': self.fallback_url
                },
                'requests': {
                    'total': self.stats['total_requests'],
                    'home_mode': self.stats['home_mode_calls'],
                    'search_mode': self.stats['search_mode_calls'],
                    'cache_hits': self.stats['cache_hits'],
                    'extracted_hits': self.stats['extracted_hits'],
                    'tmdb_hits': self.stats['tmdb_hits'],
                    'omdb_hits': self.stats['omdb_hits'],
                    'fallback_hits': self.stats['fallback_hits'],
                    'errors': self.stats['errors']
                },
                'hit_rates': {
                    'cache': f"{(self.stats['cache_hits'] / total_requests * 100):.1f}%",
                    'extracted': f"{(self.stats['extracted_hits'] / total_requests * 100):.1f}%",
                    'tmdb': f"{(self.stats['tmdb_hits'] / total_requests * 100):.1f}%",
                    'omdb': f"{(self.stats['omdb_hits'] / total_requests * 100):.1f}%",
                    'fallback': f"{(self.stats['fallback_hits'] / total_requests * 100):.1f}%",
                    'success': f"{((total_requests - self.stats['fallback_hits'] - self.stats['errors']) / total_requests * 100):.1f}%"
                },
                'database': {
                    'extracted_thumbnails': thumbnail_count,
                    'cached_posters': cache_count,
                    'ttl_days': self.config.THUMBNAIL_TTL_DAYS
                },
                'target': '99% success rate',
                'status': 'operational'
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {
                'error': str(e),
                'dual_priority_system': True,
                'status': 'degraded'
            }
