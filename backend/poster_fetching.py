# ============================================================================
# poster_fetching.py - METHOD NAME FIX
# ============================================================================
# ✅ FIXED: Method names match what app.py expects
# ============================================================================

import asyncio
import aiohttp
import logging
import time
import re
import base64
import os
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from urllib.parse import quote, urlencode

logger = logging.getLogger(__name__)

class PosterSource:
    EXTRACTED = "extracted"
    TMDB = "tmdb"
    OMDB = "omdb"
    IMDB = "imdb"
    LETTERBOXD = "letterboxd"
    WIKIPEDIA = "wikipedia"
    YOUTUBE = "youtube"
    GOOGLE = "google_images"
    FALLBACK = "fallback"
    CACHE = "cache"
    ERROR = "error"


class PosterFetcher:
    def __init__(self, config, cache_manager=None, bot_handler=None, mongo_client=None, user_client=None):
        self.config = config
        self.cache_manager = cache_manager
        self.bot_handler = bot_handler
        self.user_client = user_client
        self.mongo_client = mongo_client
        
        # MongoDB collections
        self.thumbnails_col = None
        self.poster_cache_col = None
        self._collections_initialized = False
        
        # Statistics
        self.stats = {
            'total_requests': 0, 'cache_hits': 0, 'extracted_hits': 0,
            'tmdb_hits': 0, 'omdb_hits': 0, 'imdb_hits': 0,
            'letterboxd_hits': 0, 'wikipedia_hits': 0, 'youtube_hits': 0,
            'google_hits': 0, 'fallback_hits': 0, 'errors': 0,
            'home_mode_calls': 0, 'search_mode_calls': 0
        }
        
        # HTTP session
        self.session = None
        self.session_lock = asyncio.Lock()
        
        # TTL settings
        self.poster_cache_ttl = 7 * 24 * 60 * 60
        self.thumbnail_cache_ttl = 30 * 24 * 60 * 60
        
        # Rate limiting
        self.api_last_call = {}
        self.rate_limit_delay = 0.25
        
        # Fallback URL
        self.fallback_url = getattr(config, 'FALLBACK_POSTER', 'https://iili.io/fAeIwv9.th.png')
        
        # Headers for different APIs
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        logger.info("🎬 PosterFetcher initialized with 7 API Sources")
        logger.info("   • TMDB, OMDB, IMDb, Letterboxd, Wikipedia, YouTube, Google Images")
    
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
            
            # Create indexes
            try:
                await self.thumbnails_col.create_index("normalized_title", unique=True, background=True)
                await self.thumbnails_col.create_index("expires_at", expireAfterSeconds=0, background=True)
                await self.poster_cache_col.create_index("cache_key", unique=True, background=True)
                await self.poster_cache_col.create_index("expires_at", expireAfterSeconds=0, background=True)
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
                    timeout=aiohttp.ClientTimeout(total=15),
                    headers=self.headers
                )
            return self.session
    
    # ========================================================================
    # ✅ CRITICAL FIX: Method names must match what app.py expects
    # ========================================================================
    
    async def get_thumbnail_for_movie_search(self, title: str, year: str = "",
                                           channel_id: int = None, message_id: int = None) -> Dict[str, Any]:
        """
        SEARCH MODE: Thumbnail → 7 API Sources → Fallback
        ✅ This method name is called by app.py
        """
        self.stats['total_requests'] += 1
        self.stats['search_mode_calls'] += 1
        start_time = time.time()
        
        try:
            # STEP 1: Check cache
            cached = await self._get_cached_poster(title, year)
            if cached is not None:
                self.stats['cache_hits'] += 1
                cached.update({
                    'source': PosterSource.CACHE,
                    'priority_mode': 'search',
                    'response_time': time.time() - start_time,
                    'has_thumbnail': True,
                    'is_fallback': False
                })
                return cached
            
            # STEP 2: Check database for extracted thumbnail
            db_thumb = await self._get_extracted_thumbnail(title)
            if db_thumb is not None and db_thumb.get('thumbnail_url'):
                self.stats['extracted_hits'] += 1
                logger.info(f"✅ [SEARCH] DB thumbnail: {title[:30]}...")
                return {
                    'poster_url': db_thumb['thumbnail_url'],
                    'source': PosterSource.EXTRACTED,
                    'extracted': True,
                    'priority_mode': 'search',
                    'has_thumbnail': True,
                    'is_fallback': False,
                    'response_time': time.time() - start_time
                }
            
            # STEP 3: Try LIVE extraction with USER session
            if channel_id is not None and message_id is not None and self.user_client is not None:
                live_thumb = await self._extract_thumbnail_from_user(channel_id, message_id)
                if live_thumb is not None:
                    await self._store_extracted_thumbnail(title, live_thumb, channel_id, message_id)
                    self.stats['extracted_hits'] += 1
                    logger.info(f"✅✅ [SEARCH] LIVE thumbnail: {title[:30]}...")
                    return {
                        'poster_url': live_thumb,
                        'source': PosterSource.EXTRACTED,
                        'extracted': True,
                        'priority_mode': 'search',
                        'has_thumbnail': True,
                        'is_fallback': False,
                        'response_time': time.time() - start_time
                    }
            
            # STEP 4: Try ALL 7 API SOURCES IN PARALLEL
            logger.info(f"🔄 [SEARCH] Fetching from 7 APIs: {title[:30]}...")
            poster = await self._fetch_from_all_sources_parallel(title, year)
            
            if poster is not None:
                await self._cache_poster(title, year, poster)
                poster.update({
                    'priority_mode': 'search',
                    'response_time': time.time() - start_time,
                    'has_thumbnail': True,
                    'is_fallback': False
                })
                logger.info(f"✅ [SEARCH] Poster from {poster['source']}: {title[:30]}...")
                return poster
            
            # STEP 5: Fallback
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
    
    async def get_thumbnail_for_movie_home(self, title: str, year: str = "") -> Dict[str, Any]:
        """
        HOME MODE: Sources → Thumbnail → Fallback
        ✅ This method name is called by app.py
        """
        self.stats['total_requests'] += 1
        self.stats['home_mode_calls'] += 1
        start_time = time.time()
        
        try:
            # STEP 1: Check cache
            cached = await self._get_cached_poster(title, year)
            if cached is not None:
                self.stats['cache_hits'] += 1
                cached.update({
                    'source': PosterSource.CACHE,
                    'priority_mode': 'home',
                    'response_time': time.time() - start_time,
                    'has_thumbnail': True,
                    'is_fallback': False
                })
                return cached
            
            # STEP 2: Try API sources
            poster = await self._fetch_from_all_sources_parallel(title, year)
            if poster is not None:
                await self._cache_poster(title, year, poster)
                poster.update({
                    'priority_mode': 'home',
                    'response_time': time.time() - start_time,
                    'has_thumbnail': True,
                    'is_fallback': False
                })
                logger.info(f"✅ [HOME] Poster from {poster['source']}: {title[:30]}...")
                return poster
            
            # STEP 3: Try DB thumbnail
            db_thumb = await self._get_extracted_thumbnail(title)
            if db_thumb is not None and db_thumb.get('thumbnail_url'):
                self.stats['extracted_hits'] += 1
                logger.info(f"✅ [HOME] DB thumbnail: {title[:30]}...")
                return {
                    'poster_url': db_thumb['thumbnail_url'],
                    'source': PosterSource.EXTRACTED,
                    'extracted': True,
                    'priority_mode': 'home',
                    'has_thumbnail': True,
                    'is_fallback': False,
                    'response_time': time.time() - start_time
                }
            
            # STEP 4: Fallback
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
    
    async def get_thumbnails_batch(self, movies: List[Dict], mode: str = "search") -> List[Dict]:
        """
        Batch thumbnail fetching
        ✅ This method name is called by app.py
        """
        results = []
        
        for movie in movies:
            try:
                title = movie.get('title', '')
                if not title:
                    movie_copy = movie.copy()
                    movie_copy.update({
                        'poster_url': self.fallback_url,
                        'thumbnail_url': self.fallback_url,
                        'source': PosterSource.FALLBACK,
                        'has_thumbnail': True,
                        'is_fallback': True,
                        'extracted': False
                    })
                    results.append(movie_copy)
                    continue
                
                year = movie.get('year', '')
                channel_id = movie.get('channel_id')
                message_id = movie.get('message_id') or movie.get('real_message_id')
                
                if mode == "home":
                    thumb = await self.get_thumbnail_for_movie_home(title, year)
                else:  # search mode
                    thumb = await self.get_thumbnail_for_movie_search(title, year, channel_id, message_id)
                
                movie_copy = movie.copy()
                movie_copy.update({
                    'poster_url': thumb.get('poster_url', self.fallback_url),
                    'thumbnail_url': thumb.get('poster_url', self.fallback_url),
                    'source': thumb.get('source', PosterSource.FALLBACK),
                    'has_thumbnail': thumb.get('has_thumbnail', True),
                    'is_fallback': thumb.get('is_fallback', True),
                    'extracted': thumb.get('extracted', False)
                })
                results.append(movie_copy)
                
            except Exception as e:
                logger.error(f"❌ Batch error: {e}")
                movie_copy = movie.copy()
                movie_copy.update({
                    'poster_url': self.fallback_url,
                    'thumbnail_url': self.fallback_url,
                    'source': PosterSource.ERROR,
                    'has_thumbnail': True,
                    'is_fallback': True,
                    'extracted': False
                })
                results.append(movie_copy)
        
        return results
    
    # ========================================================================
    # ✅ PARALLEL API FETCHING - ALL 7 SOURCES
    # ========================================================================
    
    async def _fetch_from_all_sources_parallel(self, title: str, year: str = "") -> Optional[Dict]:
        """Fetch from ALL 7 sources in parallel"""
        clean_title = self._clean_title_for_api(title)
        
        # Create all fetch tasks
        tasks = [
            self._fetch_from_tmdb(clean_title, year),
            self._fetch_from_omdb(clean_title, year),
            self._fetch_from_imdb(clean_title, year),
            self._fetch_from_letterboxd(clean_title, year),
            self._fetch_from_wikipedia(clean_title, year),
            self._fetch_from_youtube(clean_title, year),
            self._fetch_from_google_images(clean_title, year)
        ]
        
        # Run all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Return first successful result
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                continue
            if result is not None and result.get('poster_url'):
                # Update stats based on source
                source = result.get('source')
                if source == PosterSource.TMDB:
                    self.stats['tmdb_hits'] += 1
                elif source == PosterSource.OMDB:
                    self.stats['omdb_hits'] += 1
                elif source == PosterSource.IMDB:
                    self.stats['imdb_hits'] += 1
                elif source == PosterSource.LETTERBOXD:
                    self.stats['letterboxd_hits'] += 1
                elif source == PosterSource.WIKIPEDIA:
                    self.stats['wikipedia_hits'] += 1
                elif source == PosterSource.YOUTUBE:
                    self.stats['youtube_hits'] += 1
                elif source == PosterSource.GOOGLE:
                    self.stats['google_hits'] += 1
                
                return result
        
        return None
    
    # ========================================================================
    # ✅ API 1: TMDB
    # ========================================================================
    
    async def _fetch_from_tmdb(self, title: str, year: str = "") -> Optional[Dict]:
        """Fetch poster from TMDB API"""
        try:
            await self._rate_limit('tmdb')
            
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
                
                poster_url = f"https://image.tmdb.org/t/p/w500{movie['poster_path']}"
                
                return {
                    'poster_url': poster_url,
                    'source': PosterSource.TMDB,
                    'title': movie.get('title', title),
                    'year': movie.get('release_date', '')[:4] if movie.get('release_date') else year
                }
        except Exception as e:
            logger.debug(f"TMDB error: {e}")
            return None
    
    # ========================================================================
    # ✅ API 2: OMDB
    # ========================================================================
    
    async def _fetch_from_omdb(self, title: str, year: str = "") -> Optional[Dict]:
        """Fetch poster from OMDB API"""
        try:
            await self._rate_limit('omdb')
            
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
    # ✅ API 3: IMDb (via Scraping)
    # ========================================================================
    
    async def _fetch_from_imdb(self, title: str, year: str = "") -> Optional[Dict]:
        """Fetch poster from IMDb"""
        try:
            await self._rate_limit('imdb')
            
            session = await self._get_session()
            search_url = f"https://www.imdb.com/find?q={quote(title)}&s=tt&ttype=ft&ref_=fn_ft"
            
            async with session.get(search_url) as resp:
                if resp.status != 200:
                    return None
                
                html = await resp.text()
                
                # Extract first movie URL
                movie_url_match = re.search(r'/title/(tt\d+)/', html)
                if not movie_url_match:
                    return None
                
                movie_id = movie_url_match.group(1)
                poster_url = f"https://img.omdbapi.com/?i={movie_id}&apikey={self.config.OMDB_API_KEY}"
                
                # Verify poster exists
                async with session.get(poster_url) as img_resp:
                    if img_resp.status == 200 and img_resp.headers.get('content-type', '').startswith('image'):
                        return {
                            'poster_url': poster_url,
                            'source': PosterSource.IMDB,
                            'title': title,
                            'year': year
                        }
            
            return None
        except Exception as e:
            logger.debug(f"IMDb error: {e}")
            return None
    
    # ========================================================================
    # ✅ API 4: Letterboxd
    # ========================================================================
    
    async def _fetch_from_letterboxd(self, title: str, year: str = "") -> Optional[Dict]:
        """Fetch poster from Letterboxd"""
        try:
            await self._rate_limit('letterboxd')
            
            session = await self._get_session()
            
            # Create search query
            search_title = title.lower().replace(' ', '-')
            search_title = re.sub(r'[^a-z0-9-]', '', search_title)
            
            if year:
                search_url = f"https://letterboxd.com/film/{search_title}-{year}/"
            else:
                search_url = f"https://letterboxd.com/film/{search_title}/"
            
            async with session.get(search_url) as resp:
                if resp.status != 200:
                    return None
                
                html = await resp.text()
                
                # Extract poster URL
                poster_match = re.search(r'<img[^>]*src="([^"]+)"[^>]*class="[^"]*poster[^"]*"', html)
                if not poster_match:
                    poster_match = re.search(r'<img[^>]*src="([^"]+)"[^>]*alt="[^"]*poster[^"]*"', html)
                
                if poster_match:
                    poster_url = poster_match.group(1)
                    if poster_url.startswith('//'):
                        poster_url = 'https:' + poster_url
                    
                    return {
                        'poster_url': poster_url,
                        'source': PosterSource.LETTERBOXD,
                        'title': title,
                        'year': year
                    }
            
            return None
        except Exception as e:
            logger.debug(f"Letterboxd error: {e}")
            return None
    
    # ========================================================================
    # ✅ API 5: Wikipedia
    # ========================================================================
    
    async def _fetch_from_wikipedia(self, title: str, year: str = "") -> Optional[Dict]:
        """Fetch poster from Wikipedia"""
        try:
            await self._rate_limit('wikipedia')
            
            session = await self._get_session()
            
            # Search for the film
            search_url = "https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'list': 'search',
                'srsearch': f"{title} film",
                'format': 'json'
            }
            
            if year:
                params['srsearch'] = f"{title} {year} film"
            
            async with session.get(search_url, params=params) as resp:
                if resp.status != 200:
                    return None
                
                data = await resp.json()
                search_results = data.get('query', {}).get('search', [])
                
                if not search_results:
                    return None
                
                page_title = search_results[0]['title']
                
                # Get page image
                image_url = "https://en.wikipedia.org/w/api.php"
                image_params = {
                    'action': 'query',
                    'titles': page_title,
                    'prop': 'pageimages',
                    'pithumbsize': 500,
                    'format': 'json'
                }
                
                async with session.get(image_url, params=image_params) as img_resp:
                    img_data = await img_resp.json()
                    pages = img_data.get('query', {}).get('pages', {})
                    
                    for page_id, page_info in pages.items():
                        if page_id != '-1' and 'thumbnail' in page_info:
                            poster_url = page_info['thumbnail']['source']
                            return {
                                'poster_url': poster_url,
                                'source': PosterSource.WIKIPEDIA,
                                'title': title,
                                'year': year
                            }
            
            return None
        except Exception as e:
            logger.debug(f"Wikipedia error: {e}")
            return None
    
    # ========================================================================
    # ✅ API 6: YouTube (Video Thumbnail)
    # ========================================================================
    
    async def _fetch_from_youtube(self, title: str, year: str = "") -> Optional[Dict]:
        """Fetch thumbnail from YouTube trailer"""
        try:
            await self._rate_limit('youtube')
            
            session = await self._get_session()
            
            # Search for movie trailer
            search_query = f"{title} {year} official trailer" if year else f"{title} official trailer"
            search_url = "https://www.youtube.com/results"
            params = {
                'search_query': search_query
            }
            
            async with session.get(search_url, params=params) as resp:
                if resp.status != 200:
                    return None
                
                html = await resp.text()
                
                # Extract video ID
                video_id_match = re.search(r'"videoId":"([^"]+)"', html)
                if not video_id_match:
                    return None
                
                video_id = video_id_match.group(1)
                
                # YouTube thumbnail URLs
                thumbnails = [
                    f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
                    f"https://img.youtube.com/vi/{video_id}/sddefault.jpg",
                    f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
                    f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg"
                ]
                
                # Try each thumbnail size
                for thumb_url in thumbnails:
                    async with session.get(thumb_url) as thumb_resp:
                        if thumb_resp.status == 200:
                            return {
                                'poster_url': thumb_url,
                                'source': PosterSource.YOUTUBE,
                                'title': title,
                                'year': year,
                                'video_id': video_id
                            }
            
            return None
        except Exception as e:
            logger.debug(f"YouTube error: {e}")
            return None
    
    # ========================================================================
    # ✅ API 7: Google Images
    # ========================================================================
    
    async def _fetch_from_google_images(self, title: str, year: str = "") -> Optional[Dict]:
        """Fetch poster from Google Images"""
        try:
            await self._rate_limit('google')
            
            session = await self._get_session()
            
            # Search query
            search_query = f"{title} {year} movie poster" if year else f"{title} movie poster"
            
            # Try DuckDuckGo as fallback
            ddg_url = f"https://duckduckgo.com/?q={quote(search_query)}&iax=images&ia=images"
            
            async with session.get(ddg_url) as resp:
                if resp.status == 200:
                    html = await resp.text()
                    
                    # Extract image URL from DDG
                    img_match = re.search(r'https?://[^\s"\']+\.(?:jpg|jpeg|png|webp)[^\s"\']*', html)
                    if img_match:
                        poster_url = img_match.group(0)
                        return {
                            'poster_url': poster_url,
                            'source': PosterSource.GOOGLE,
                            'title': title,
                            'year': year
                        }
            
            return None
        except Exception as e:
            logger.debug(f"Google Images error: {e}")
            return None
    
    # ========================================================================
    # ✅ Rate Limiting
    # ========================================================================
    
    async def _rate_limit(self, api_name: str):
        """Rate limiting for APIs"""
        now = time.time()
        last_call = self.api_last_call.get(api_name, 0)
        
        if now - last_call < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay)
        
        self.api_last_call[api_name] = time.time()
    
    # ========================================================================
    # ✅ Thumbnail Extraction from Telegram
    # ========================================================================
    
    async def _extract_thumbnail_from_user(self, channel_id: int, message_id: int) -> Optional[str]:
        """Extract thumbnail using USER session"""
        try:
            from __main__ import User as user_client
            
            if user_client is None:
                return None
            
            message = await user_client.get_messages(channel_id, message_id)
            if not message:
                return None
            
            thumbnail_file_id = None
            
            # Pyrogram v2+ uses .thumbs[0]
            if message.video and message.video.thumbs:
                thumbnail_file_id = message.video.thumbs[0].file_id
            
            elif message.document and message.document.thumbs:
                is_video = False
                if message.document.mime_type and 'video' in message.document.mime_type:
                    is_video = True
                elif message.document.file_name and self._is_video_file(message.document.file_name):
                    is_video = True
                
                if is_video and message.document.thumbs:
                    thumbnail_file_id = message.document.thumbs[0].file_id
            
            if not thumbnail_file_id:
                return None
            
            download_path = await user_client.download_media(
                thumbnail_file_id,
                in_memory=True
            )
            
            if not download_path:
                return None
            
            if isinstance(download_path, bytes):
                thumbnail_bytes = download_path
            else:
                with open(download_path, 'rb') as f:
                    thumbnail_bytes = f.read()
                try:
                    os.remove(download_path)
                except:
                    pass
            
            base64_data = base64.b64encode(thumbnail_bytes).decode('utf-8')
            return f"data:image/jpeg;base64,{base64_data}"
            
        except Exception as e:
            logger.error(f"❌ Extraction error: {e}")
            return None
    
    def _is_video_file(self, filename):
        """Check if file is video"""
        if not filename:
            return False
        video_ext = ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v']
        return any(filename.lower().endswith(ext) for ext in video_ext)
    
    # ========================================================================
    # ✅ Database Operations
    # ========================================================================
    
    async def _get_extracted_thumbnail(self, title: str) -> Optional[Dict]:
        """Get thumbnail from database"""
        try:
            await self._ensure_collections()
            if self.thumbnails_col is None:
                return None
            
            norm_title = self._normalize_title(title)
            thumb = await self.thumbnails_col.find_one({'normalized_title': norm_title})
            
            if thumb is not None and thumb.get('thumbnail_url'):
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
            if self.thumbnails_col is None:
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
            return True
        except Exception as e:
            logger.error(f"❌ DB store error: {e}")
            return False
    
    async def _get_cached_poster(self, title: str, year: str = "") -> Optional[Dict]:
        """Get cached poster"""
        try:
            await self._ensure_collections()
            if self.poster_cache_col is None:
                return None
            
            cache_key = self._make_cache_key(title, year)
            cached = await self.poster_cache_col.find_one({'cache_key': cache_key})
            
            if cached is not None and cached.get('poster_url'):
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
            if self.poster_cache_col is None:
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
    # ✅ Utility Functions
    # ========================================================================
    
    def _normalize_title(self, title: str) -> str:
        if not title:
            return ""
        title = title.lower().strip()
        title = re.sub(r'\s*\(\d{4}\)$', '', title)
        title = re.sub(r'\s+\d{4}$', '', title)
        title = re.sub(r'\b(480p|720p|1080p|2160p|4k|hd|hevc|x265|x264)\b', '', title)
        title = re.sub(r'[^\w\s]', ' ', title)
        title = re.sub(r'\s+', ' ', title)
        return title.strip()
    
    def _clean_title_for_api(self, title: str) -> str:
        if not title:
            return ""
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
        if self.session is not None and not self.session.closed:
            await self.session.close()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        try:
            await self._ensure_collections()
            
            thumb_count = 0
            if self.thumbnails_col is not None:
                thumb_count = await self.thumbnails_col.count_documents({})
            
            cache_count = 0
            if self.poster_cache_col is not None:
                cache_count = await self.poster_cache_col.count_documents({})
            
            total = self.stats['total_requests'] or 1
            
            return {
                'dual_priority_system': {
                    'home_mode': 'Sources → Thumbnail → Fallback',
                    'search_mode': 'Thumbnail → 7 API Sources → Fallback',
                    'extraction_method': 'USER_SESSION',
                    'fallback_url': self.fallback_url
                },
                'requests': self.stats,
                'hit_rates': {
                    'cache': f"{(self.stats['cache_hits'] / total * 100):.1f}%",
                    'extracted': f"{(self.stats['extracted_hits'] / total * 100):.1f}%",
                    'tmdb': f"{(self.stats['tmdb_hits'] / total * 100):.1f}%",
                    'omdb': f"{(self.stats['omdb_hits'] / total * 100):.1f}%",
                    'imdb': f"{(self.stats['imdb_hits'] / total * 100):.1f}%",
                    'letterboxd': f"{(self.stats['letterboxd_hits'] / total * 100):.1f}%",
                    'wikipedia': f"{(self.stats['wikipedia_hits'] / total * 100):.1f}%",
                    'youtube': f"{(self.stats['youtube_hits'] / total * 100):.1f}%",
                    'google': f"{(self.stats['google_hits'] / total * 100):.1f}%",
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
