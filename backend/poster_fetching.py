# ============================================================================
# poster_fetching.py - DUAL PRIORITY THUMBNAIL & POSTER SYSTEM - ULTIMATE FIX
# ============================================================================
# ✅ HOME MODE: Sources → Thumbnail → Fallback
# ✅ SEARCH MODE: Thumbnail → Sources → Fallback  
# ✅ FIXED: User session for thumbnail extraction
# ✅ FIXED: Channel access with proper error handling
# ✅ FIXED: MongoDB collection checks
# ============================================================================

import asyncio
import aiohttp
import logging
import time
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import base64
import os

logger = logging.getLogger(__name__)

class PosterSource:
    """Source constants for poster/thumbnail origin"""
    EXTRACTED = "extracted"
    TMDB = "tmdb"
    OMDB = "omdb"
    FALLBACK = "fallback"
    CACHE = "cache"
    ERROR = "error"


class PosterFetcher:
    """
    DUAL PRIORITY THUMBNAIL SYSTEM - ULTIMATE FIX
    HOME MODE: Sources → Thumbnail → Fallback
    SEARCH MODE: Thumbnail → Sources → Fallback
    """
    
    def __init__(self, config, cache_manager=None, bot_handler=None, mongo_client=None, user_client=None):
        self.config = config
        self.cache_manager = cache_manager
        self.bot_handler = bot_handler
        self.user_client = user_client  # ✅ CRITICAL: User session for thumbnails
        self.mongo_client = mongo_client
        
        # MongoDB collections
        self.thumbnails_col = None
        self.poster_cache_col = None
        self._collections_initialized = False
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'extracted_hits': 0,
            'tmdb_hits': 0,
            'omdb_hits': 0,
            'fallback_hits': 0,
            'errors': 0,
            'home_mode_calls': 0,
            'search_mode_calls': 0,
            'channel_invalid_errors': 0
        }
        
        # HTTP session
        self.session = None
        self.session_lock = asyncio.Lock()
        
        # TTL settings
        self.poster_cache_ttl = 7 * 24 * 60 * 60
        self.thumbnail_cache_ttl = 30 * 24 * 60 * 60
        
        # Rate limiting
        self.tmdb_last_call = 0
        self.omdb_last_call = 0
        self.rate_limit_delay = 0.25
        
        # Fallback URL
        self.fallback_url = getattr(config, 'FALLBACK_POSTER', 'https://iili.io/fAeIwv9.th.png')
        
        logger.info("🎬 PosterFetcher initialized with User Session for thumbnails")
    
    async def _ensure_collections(self):
        """Ensure MongoDB collections exist"""
        if self._collections_initialized:
            return True
        
        try:
            if self.mongo_client is None:
                return False
            
            db = self.mongo_client.sk4film
            self.thumbnails_col = db.thumbnails
            self.poster_cache_col = db.poster_cache
            
            # Create indexes (ignore duplicate errors)
            try:
                await self.thumbnails_col.create_index(
                    "normalized_title", unique=True, background=True
                )
            except:
                pass
            
            try:
                await self.thumbnails_col.create_index(
                    "expires_at", expireAfterSeconds=0, background=True
                )
            except:
                pass
            
            try:
                await self.poster_cache_col.create_index(
                    "cache_key", unique=True, background=True
                )
            except:
                pass
            
            try:
                await self.poster_cache_col.create_index(
                    "expires_at", expireAfterSeconds=0, background=True
                )
            except:
                pass
            
            self._collections_initialized = True
            return True
        except Exception as e:
            logger.error(f"❌ Collection init error: {e}")
            return False
    
    async def _get_session(self):
        """Get HTTP session"""
        async with self.session_lock:
            if self.session is None or self.session.closed:
                self.session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=10),
                    headers={'User-Agent': 'SK4FiLM/9.0'}
                )
            return self.session
    
    # ========================================================================
    # ✅ ULTIMATE FIX: Thumbnail Extraction with User Session
    # ========================================================================
    
    async def _extract_thumbnail_from_user(self, channel_id: int, message_id: int) -> Optional[str]:
        """
        Extract thumbnail using USER session - 100% Working
        Bot doesn't have access, User does!
        """
        try:
            # Get the global User client
            from __main__ import User as user_client
            
            if user_client is None:
                logger.error("❌ User client not available")
                return None
            
            logger.debug(f"📸 User extracting thumbnail: {channel_id}/{message_id}")
            
            # Get message
            message = await user_client.get_messages(channel_id, message_id)
            if not message:
                logger.debug(f"❌ Message not found: {message_id}")
                return None
            
            # Find thumbnail file_id
            thumbnail_file_id = None
            
            # Video with thumbnail
            if message.video and message.video.thumbnail:
                thumbnail_file_id = message.video.thumbnail.file_id
                logger.debug(f"✅ Video thumbnail found")
            
            # Document with thumbnail (video file)
            elif message.document and message.document.thumbnail:
                # Check if it's video
                is_video = False
                if message.document.mime_type and 'video' in message.document.mime_type:
                    is_video = True
                elif message.document.file_name and self._is_video_file(message.document.file_name):
                    is_video = True
                
                if is_video:
                    thumbnail_file_id = message.document.thumbnail.file_id
                    logger.debug(f"✅ Document thumbnail found")
            
            if not thumbnail_file_id:
                logger.debug(f"❌ No thumbnail in message {message_id}")
                return None
            
            # Download thumbnail
            logger.debug(f"⬇️ Downloading thumbnail...")
            download_path = await user_client.download_media(
                thumbnail_file_id,
                in_memory=True
            )
            
            if not download_path:
                logger.debug(f"❌ Download failed")
                return None
            
            # Convert to base64
            if isinstance(download_path, bytes):
                thumbnail_bytes = download_path
            else:
                try:
                    with open(download_path, 'rb') as f:
                        thumbnail_bytes = f.read()
                    # Clean up temp file
                    try:
                        os.remove(download_path)
                    except:
                        pass
                except Exception as e:
                    logger.error(f"❌ File read error: {e}")
                    return None
            
            base64_data = base64.b64encode(thumbnail_bytes).decode('utf-8')
            result = f"data:image/jpeg;base64,{base64_data}"
            
            logger.info(f"✅✅✅ Thumbnail EXTRACTED via USER: {message_id}")
            return result
            
        except Exception as e:
            error_str = str(e).lower()
            if 'channel_invalid' in error_str:
                self.stats['channel_invalid_errors'] += 1
                logger.error(f"❌ CHANNEL_INVALID: User also can't access? Check permissions")
            else:
                logger.error(f"❌ Extraction error: {e}")
            return None
    
    def _is_video_file(self, filename):
        """Check if file is video"""
        if not filename:
            return False
        video_ext = ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v']
        return any(filename.lower().endswith(ext) for ext in video_ext)
    
    # ========================================================================
    # ✅ SEARCH MODE: Thumbnail FIRST
    # ========================================================================
    
    async def get_thumbnail_for_movie_search(self, title: str, year: str = "",
                                           channel_id: int = None, message_id: int = None) -> Dict[str, Any]:
        """
        SEARCH MODE: Thumbnail → Sources → Fallback
        1. Try extracted thumbnail (from DB)
        2. Try live extraction (from Telegram via USER)
        3. Try TMDB/OMDB
        4. Fallback image
        """
        self.stats['total_requests'] += 1
        self.stats['search_mode_calls'] += 1
        start_time = time.time()
        
        try:
            # STEP 0: Check cache
            cached = await self._get_cached_poster(title, year)
            if cached:
                self.stats['cache_hits'] += 1
                cached.update({
                    'source': PosterSource.CACHE,
                    'priority_mode': 'search',
                    'response_time': time.time() - start_time,
                    'has_thumbnail': True,
                    'is_fallback': False
                })
                return cached
            
            # STEP 1: Check database for existing thumbnail
            logger.debug(f"[SEARCH] DB lookup: {title[:30]}...")
            db_thumb = await self._get_extracted_thumbnail(title)
            if db_thumb and db_thumb.get('thumbnail_url'):
                self.stats['extracted_hits'] += 1
                logger.info(f"✅ [SEARCH] DB thumbnail found: {title[:30]}...")
                return {
                    'poster_url': db_thumb['thumbnail_url'],
                    'source': PosterSource.EXTRACTED,
                    'extracted': True,
                    'priority_mode': 'search',
                    'has_thumbnail': True,
                    'is_fallback': False,
                    'response_time': time.time() - start_time
                }
            
            # STEP 2: Try LIVE extraction with USER session
            if channel_id and message_id and self.user_client:
                logger.debug(f"[SEARCH] Live extraction: {channel_id}/{message_id}")
                live_thumb = await self._extract_thumbnail_from_user(channel_id, message_id)
                if live_thumb:
                    # Store in database
                    await self._store_extracted_thumbnail(title, live_thumb, channel_id, message_id)
                    self.stats['extracted_hits'] += 1
                    logger.info(f"✅✅ [SEARCH] LIVE thumbnail extracted: {title[:30]}...")
                    return {
                        'poster_url': live_thumb,
                        'source': PosterSource.EXTRACTED,
                        'extracted': True,
                        'priority_mode': 'search',
                        'has_thumbnail': True,
                        'is_fallback': False,
                        'response_time': time.time() - start_time
                    }
            
            # STEP 3: Try TMDB/OMDB
            logger.debug(f"[SEARCH] API fallback: {title[:30]}...")
            poster = await self._fetch_from_sources(title, year)
            if poster:
                await self._cache_poster(title, year, poster)
                poster.update({
                    'priority_mode': 'search',
                    'response_time': time.time() - start_time,
                    'has_thumbnail': True,
                    'is_fallback': False
                })
                logger.info(f"✅ [SEARCH] API poster: {poster['source']} - {title[:30]}...")
                return poster
            
            # STEP 4: Fallback
            self.stats['fallback_hits'] += 1
            logger.debug(f"➖ [SEARCH] Fallback: {title[:30]}...")
            return {
                'poster_url': self.fallback_url,
                'source': PosterSource.FALLBACK,
                'priority_mode': 'search',
                'has_thumbnail': True,
                'is_fallback': True,
                'response_time': time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"❌ [SEARCH] Error: {e}")
            self.stats['errors'] += 1
            return {
                'poster_url': self.fallback_url,
                'source': PosterSource.ERROR,
                'priority_mode': 'search',
                'has_thumbnail': True,
                'is_fallback': True,
                'response_time': time.time() - start_time
            }
    
    # ========================================================================
    # ✅ HOME MODE: Sources FIRST
    # ========================================================================
    
    async def get_thumbnail_for_movie_home(self, title: str, year: str = "") -> Dict[str, Any]:
        """HOME MODE: Sources → Thumbnail → Fallback"""
        self.stats['total_requests'] += 1
        self.stats['home_mode_calls'] += 1
        start_time = time.time()
        
        try:
            # Check cache
            cached = await self._get_cached_poster(title, year)
            if cached:
                self.stats['cache_hits'] += 1
                cached.update({
                    'source': PosterSource.CACHE,
                    'priority_mode': 'home',
                    'response_time': time.time() - start_time,
                    'has_thumbnail': True,
                    'is_fallback': False
                })
                return cached
            
            # Try API sources
            poster = await self._fetch_from_sources(title, year)
            if poster:
                await self._cache_poster(title, year, poster)
                poster.update({
                    'priority_mode': 'home',
                    'response_time': time.time() - start_time,
                    'has_thumbnail': True,
                    'is_fallback': False
                })
                return poster
            
            # Try DB thumbnail
            db_thumb = await self._get_extracted_thumbnail(title)
            if db_thumb:
                self.stats['extracted_hits'] += 1
                return {
                    'poster_url': db_thumb['thumbnail_url'],
                    'source': PosterSource.EXTRACTED,
                    'extracted': True,
                    'priority_mode': 'home',
                    'has_thumbnail': True,
                    'is_fallback': False,
                    'response_time': time.time() - start_time
                }
            
            # Fallback
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
            logger.error(f"❌ [HOME] Error: {e}")
            self.stats['errors'] += 1
            return {
                'poster_url': self.fallback_url,
                'source': PosterSource.ERROR,
                'priority_mode': 'home',
                'has_thumbnail': True,
                'is_fallback': True,
                'response_time': time.time() - start_time
            }
    
    # ========================================================================
    # ✅ BATCH PROCESSING
    # ========================================================================
    
    async def get_thumbnails_batch(self, movies: List[Dict], mode: str = "search") -> List[Dict]:
        """Batch thumbnail fetching"""
        results = []
        for movie in movies:
            try:
                title = movie.get('title', '')
                if not title:
                    movie['poster_url'] = self.fallback_url
                    movie['source'] = PosterSource.FALLBACK
                    movie['has_thumbnail'] = True
                    movie['is_fallback'] = True
                    results.append(movie)
                    continue
                
                if mode == "home":
                    thumb = await self.get_thumbnail_for_movie_home(
                        title, movie.get('year', '')
                    )
                else:
                    thumb = await self.get_thumbnail_for_movie_search(
                        title,
                        movie.get('year', ''),
                        movie.get('channel_id'),
                        movie.get('message_id') or movie.get('real_message_id')
                    )
                
                movie.update({
                    'poster_url': thumb.get('poster_url', self.fallback_url),
                    'thumbnail_url': thumb.get('poster_url', self.fallback_url),
                    'source': thumb.get('source', PosterSource.FALLBACK),
                    'has_thumbnail': thumb.get('has_thumbnail', True),
                    'is_fallback': thumb.get('is_fallback', True),
                    'extracted': thumb.get('extracted', False)
                })
                results.append(movie)
                
            except Exception as e:
                logger.error(f"❌ Batch error: {e}")
                movie['poster_url'] = self.fallback_url
                movie['source'] = PosterSource.ERROR
                movie['has_thumbnail'] = True
                movie['is_fallback'] = True
                results.append(movie)
        
        return results
    
    # ========================================================================
    # ✅ DATABASE OPERATIONS
    # ========================================================================
    
    async def _get_extracted_thumbnail(self, title: str) -> Optional[Dict]:
        """Get thumbnail from database"""
        try:
            await self._ensure_collections()
            if not self.thumbnails_col:
                return None
            
            norm_title = self._normalize_title(title)
            thumb = await self.thumbnails_col.find_one({'normalized_title': norm_title})
            
            if thumb and thumb.get('thumbnail_url'):
                # Check expiry
                if thumb.get('expires_at') and datetime.now() > thumb['expires_at']:
                    await self.thumbnails_col.delete_one({'_id': thumb['_id']})
                    return None
                return {
                    'thumbnail_url': thumb['thumbnail_url'],
                    'extracted': True
                }
            return None
        except Exception as e:
            logger.error(f"❌ DB get error: {e}")
            return None
    
    async def _store_extracted_thumbnail(self, title: str, thumbnail_base64: str,
                                       channel_id: int, message_id: int) -> bool:
        """Store thumbnail in database"""
        try:
            await self._ensure_collections()
            if not self.thumbnails_col:
                return False
            
            norm_title = self._normalize_title(title)
            expires_at = datetime.now() + timedelta(days=self.config.THUMBNAIL_TTL_DAYS)
            
            await self.thumbnails_col.update_one(
                {'normalized_title': norm_title},
                {'$set': {
                    'title': title,
                    'normalized_title': norm_title,
                    'thumbnail_url': thumbnail_base64,
                    'channel_id': channel_id,
                    'message_id': message_id,
                    'extracted': True,
                    'stored_at': datetime.now(),
                    'expires_at': expires_at
                }},
                upsert=True
            )
            logger.info(f"✅ Stored thumbnail: {title[:30]}...")
            return True
        except Exception as e:
            logger.error(f"❌ DB store error: {e}")
            return False
    
    async def _get_cached_poster(self, title: str, year: str = "") -> Optional[Dict]:
        """Get cached poster"""
        try:
            await self._ensure_collections()
            if not self.poster_cache_col:
                return None
            
            cache_key = self._make_cache_key(title, year)
            cached = await self.poster_cache_col.find_one({'cache_key': cache_key})
            
            if cached and cached.get('poster_url'):
                if cached.get('expires_at') and datetime.now() > cached['expires_at']:
                    await self.poster_cache_col.delete_one({'_id': cached['_id']})
                    return None
                return {
                    'poster_url': cached['poster_url'],
                    'source': cached.get('source', PosterSource.CACHE),
                    'title': cached.get('title', title),
                    'year': cached.get('year', year)
                }
            return None
        except Exception:
            return None
    
    async def _cache_poster(self, title: str, year: str, poster_data: Dict) -> bool:
        """Cache poster"""
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
                    'poster_url': poster_data['poster_url'],
                    'source': poster_data['source'],
                    'cached_at': datetime.now(),
                    'expires_at': expires_at
                }},
                upsert=True
            )
            return True
        except Exception:
            return False
    
    # ========================================================================
    # ✅ API SOURCES
    # ========================================================================
    
    async def _fetch_from_sources(self, title: str, year: str = "") -> Optional[Dict]:
        """Fetch from TMDB/OMDB"""
        clean_title = self._clean_title_for_api(title)
        
        # Try TMDB
        tmdb = await self._fetch_tmdb_poster(clean_title, year)
        if tmdb:
            self.stats['tmdb_hits'] += 1
            return tmdb
        
        # Try OMDB
        omdb = await self._fetch_omdb_poster(clean_title, year)
        if omdb:
            self.stats['omdb_hits'] += 1
            return omdb
        
        return None
    
    async def _fetch_tmdb_poster(self, title: str, year: str = "") -> Optional[Dict]:
        """TMDB API"""
        try:
            # Rate limit
            now = time.time()
            if now - self.tmdb_last_call < self.rate_limit_delay:
                await asyncio.sleep(self.rate_limit_delay)
            self.tmdb_last_call = now
            
            session = await self._get_session()
            url = "https://api.themoviedb.org/3/search/movie"
            params = {
                'api_key': self.config.TMDB_API_KEY,
                'query': title,
                'include_adult': False
            }
            if year:
                params['year'] = year
            
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                if not data.get('results'):
                    return None
                
                movie = data['results'][0]
                if not movie.get('poster_path'):
                    return None
                
                return {
                    'poster_url': f"https://image.tmdb.org/t/p/w500{movie['poster_path']}",
                    'source': PosterSource.TMDB,
                    'title': movie.get('title', title),
                    'year': movie.get('release_date', '')[:4] if movie.get('release_date') else year
                }
        except Exception as e:
            logger.debug(f"TMDB error: {e}")
            return None
    
    async def _fetch_omdb_poster(self, title: str, year: str = "") -> Optional[Dict]:
        """OMDB API"""
        try:
            now = time.time()
            if now - self.omdb_last_call < self.rate_limit_delay:
                await asyncio.sleep(self.rate_limit_delay)
            self.omdb_last_call = now
            
            session = await self._get_session()
            params = {
                'apikey': self.config.OMDB_API_KEY,
                't': title,
                'plot': 'short'
            }
            if year:
                params['y'] = year
            
            async with session.get("http://www.omdbapi.com/", params=params) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                if data.get('Response') != 'True':
                    return None
                
                poster = data.get('Poster')
                if not poster or poster == 'N/A':
                    return None
                
                return {
                    'poster_url': poster,
                    'source': PosterSource.OMDB,
                    'title': data.get('Title', title),
                    'year': data.get('Year', year)
                }
        except Exception as e:
            logger.debug(f"OMDB error: {e}")
            return None
    
    # ========================================================================
    # ✅ UTILITY
    # ========================================================================
    
    def _normalize_title(self, title: str) -> str:
        if not title: return ""
        title = title.lower().strip()
        title = re.sub(r'\s*\(\d{4}\)$', '', title)
        title = re.sub(r'\s+\d{4}$', '', title)
        title = re.sub(r'\b(480p|720p|1080p|2160p|4k|hd|hevc|x265|x264)\b', '', title)
        title = re.sub(r'[^\w\s]', ' ', title)
        title = re.sub(r'\s+', ' ', title)
        return title.strip()
    
    def _clean_title_for_api(self, title: str) -> str:
        if not title: return ""
        title = re.sub(r'\s*\(\d{4}\)$', '', title)
        title = re.sub(r'\s+\d{4}$', '', title)
        title = re.sub(r'\b(480p|720p|1080p|2160p|4k|hd|hevc|x265|x264|bluray|webrip|hdtv)\b', '', title, flags=re.IGNORECASE)
        title = re.sub(r'\s+', ' ', title)
        return title.strip()
    
    def _make_cache_key(self, title: str, year: str = "") -> str:
        norm = self._normalize_title(title)
        return f"poster:{norm}:{year}" if year else f"poster:{norm}"
    
    async def close(self):
        """Close session"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        try:
            await self._ensure_collections()
            thumb_count = await self.thumbnails_col.count_documents({}) if self.thumbnails_col else 0
            cache_count = await self.poster_cache_col.count_documents({}) if self.poster_cache_col else 0
            total = self.stats['total_requests'] or 1
            
            return {
                'dual_priority_system': {
                    'home_mode': 'Sources → Thumbnail → Fallback',
                    'search_mode': 'Thumbnail → Sources → Fallback',
                    'extraction_method': 'USER_SESSION',
                    'fallback_url': self.fallback_url
                },
                'requests': self.stats,
                'hit_rates': {
                    'cache': f"{(self.stats['cache_hits'] / total * 100):.1f}%",
                    'extracted': f"{(self.stats['extracted_hits'] / total * 100):.1f}%",
                    'tmdb': f"{(self.stats['tmdb_hits'] / total * 100):.1f}%",
                    'omdb': f"{(self.stats['omdb_hits'] / total * 100):.1f}%",
                    'fallback': f"{(self.stats['fallback_hits'] / total * 100):.1f}%",
                    'success': f"{((total - self.stats['fallback_hits'] - self.stats['errors']) / total * 100):.1f}%"
                },
                'database': {
                    'extracted_thumbnails': thumb_count,
                    'cached_posters': cache_count
                }
            }
        except Exception as e:
            logger.error(f"❌ Stats error: {e}")
            return {'status': 'degraded', 'error': str(e)}
