# ============================================================================
# poster_fetching.py - COMPLETE FIXED VERSION v3.0
# ============================================================================
# 🎯 PRIORITY ORDER:
# 1. MongoDB Cache (Fastest)
# 2. TMDB (The Movie Database)
# 3. OMDB (Open Movie Database)
# 4. Letterboxd
# 5. IMDB
# 6. JustWatch
# 7. IMPAwards
# 8. TELEGRAM (Main Channel Search) ⬅️ NEW
# 9. CUSTOM / FALLBACK (Last Resort)
# ============================================================================

import asyncio
import re
import base64
import json
import logging
import urllib.parse
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import aiohttp
import hashlib

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w780"
CUSTOM_POSTER_URL = "https://iili.io/fAeIwv9.th.png"
CACHE_TTL = 86400  # 24 hours
MEMORY_CACHE_TTL = 3600  # 1 hour

# ============================================================================
# POSTER SOURCE ENUM
# ============================================================================

class PosterSource(Enum):
    MONGODB = "mongodb"          # 1️⃣ Highest priority
    TMDB = "tmdb"                # 2️⃣
    OMDB = "omdb"                # 3️⃣
    LETTERBOXD = "letterboxd"    # 4️⃣
    IMDB = "imdb"                # 5️⃣
    JUSTWATCH = "justwatch"      # 6️⃣
    IMPAWARDS = "impawards"      # 7️⃣
    TELEGRAM = "telegram"        # 8️⃣ NEW
    CUSTOM = "custom"            # 9️⃣ Fallback

# ============================================================================
# POSTER FETCHER - COMPLETE VERSION
# ============================================================================

class PosterFetcher:
    """
    🎬 COMPLETE POSTER FETCHER v3.0
    Fetches movie posters from multiple sources with priority order.
    Now includes TELEGRAM channel search!
    """
    
    def __init__(self, config, mongo_client=None, redis_client=None):
        self.config = config
        self.mongo_client = mongo_client
        self.redis_client = redis_client
        
        # API Keys
        self.tmdb_api_key = getattr(config, "TMDB_API_KEY", "e547e17d4e91f3e62a571655cd1ccaff")
        self.omdb_api_key = getattr(config, "OMDB_API_KEY", "8265bd1c")
        
        # Channel IDs
        self.main_channel_id = getattr(config, "MAIN_CHANNEL_ID", -1001767371495)
        self.file_channel_id = getattr(config, "FILE_CHANNEL_ID", -1001768249569)
        
        # Telegram clients (set later)
        self.user_client = None
        self.bot_client = None
        
        # Cache
        self.poster_cache: Dict[str, Tuple[Dict, datetime]] = {}
        self.cache_lock = asyncio.Lock()
        
        # HTTP Session
        self.http_session: Optional[aiohttp.ClientSession] = None
        self.session_lock = asyncio.Lock()
        
        # Statistics
        self.stats = {
            "mongodb_hits": 0,
            "tmdb_hits": 0,
            "omdb_hits": 0,
            "letterboxd_hits": 0,
            "imdb_hits": 0,
            "justwatch_hits": 0,
            "impawards_hits": 0,
            "telegram_hits": 0,
            "custom_hits": 0,
            "cache_hits": 0,
            "total_requests": 0,
            "failed_requests": 0
        }
        
        # Initialize database collections
        self.db = None
        self.posters_col = None
        
        logger.info("🎬 PosterFetcher v3.0 initialized")

    # ============================================================================
    # INITIALIZATION
    # ============================================================================
    
    async def initialize(self):
        """Initialize HTTP session and database connections"""
        try:
            # Create HTTP session
            timeout = aiohttp.ClientTimeout(total=10, connect=5)
            self.http_session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "application/json, text/html, */*"
                }
            )
            
            # Initialize MongoDB
            if self.mongo_client:
                self.db = self.mongo_client.sk4film
                self.posters_col = self.db.posters
                
                # Create indexes
                try:
                    await self.posters_col.create_index("cache_key", unique=True)
                    await self.posters_col.create_index("normalized_title")
                    await self.posters_col.create_index("cached_at")
                except Exception as e:
                    logger.debug(f"Index creation: {e}")
            
            logger.info("✅ PosterFetcher initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ PosterFetcher initialization error: {e}")
            return False
    
    # ============================================================================
    # TELEGRAM CLIENT SETUP
    # ============================================================================
    
    def set_telegram_clients(self, user_client=None, bot_client=None):
        """Set Telegram clients for poster fetching"""
        self.user_client = user_client
        self.bot_client = bot_client
        logger.info("✅ Telegram clients set for poster fetcher")
        
        # Log which clients are available
        if user_client:
            logger.info("   ✅ User client available for poster search")
        if bot_client:
            logger.info("   ✅ Bot client available for poster search")
        if not user_client and not bot_client:
            logger.warning("   ⚠️ No Telegram clients available for poster fetching")

    # ============================================================================
    # HTTP SESSION GETTER
    # ============================================================================
    
    async def _get_session(self):
        """Get or create HTTP session"""
        async with self.session_lock:
            if self.http_session is None or self.http_session.closed:
                timeout = aiohttp.ClientTimeout(total=10, connect=5)
                self.http_session = aiohttp.ClientSession(
                    timeout=timeout,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    }
                )
            return self.http_session

    # ============================================================================
    # CACHE HELPERS
    # ============================================================================
    
    def _get_cache_key(self, title: str, year: str = "") -> str:
        """Generate cache key from title and year"""
        normalized = self._normalize_title(title)
        return f"poster:{normalized}:{year}" if year else f"poster:{normalized}"
    
    def _normalize_title(self, title: str) -> str:
        """Normalize title for caching"""
        if not title:
            return ""
        title = title.lower().strip()
        title = re.sub(r'\s*\([^)]*\)$', '', title)
        title = re.sub(r'\s*\[[^\]]*\]$', '', title)
        title = re.sub(r'\s*\d{4}$', '', title)
        title = re.sub(r'[^a-z0-9]+', '_', title)
        title = re.sub(r'_+', '_', title)
        return title.strip('_')
    
    async def _get_from_mongodb(self, cache_key: str) -> Optional[Dict]:
        """Get poster from MongoDB cache"""
        try:
            if not self.posters_col:
                return None
            
            doc = await self.posters_col.find_one({
                'cache_key': cache_key,
                'found': True
            })
            
            if doc and doc.get('poster_url'):
                # Check age
                cached_at = doc.get('cached_at')
                if cached_at:
                    if isinstance(cached_at, datetime):
                        age = (datetime.now() - cached_at).total_seconds()
                        if age < CACHE_TTL:
                            self.stats['mongodb_hits'] += 1
                            return {
                                'poster_url': doc['poster_url'],
                                'source': PosterSource.MONGODB.value,
                                'rating': doc.get('rating', '0.0'),
                                'year': doc.get('year', ''),
                                'title': doc.get('title', ''),
                                'found': True,
                                'from_cache': True
                            }
            return None
            
        except Exception as e:
            logger.debug(f"MongoDB cache error: {e}")
            return None
    
    async def _store_in_mongodb(self, cache_key: str, poster_data: Dict):
        """Store poster in MongoDB cache"""
        try:
            if not self.posters_col:
                return
            
            await self.posters_col.update_one(
                {'cache_key': cache_key},
                {'$set': {
                    'cache_key': cache_key,
                    'normalized_title': self._normalize_title(poster_data.get('title', '')),
                    'title': poster_data.get('title', ''),
                    'year': poster_data.get('year', ''),
                    'poster_url': poster_data.get('poster_url'),
                    'source': poster_data.get('source', PosterSource.TMDB.value),
                    'rating': poster_data.get('rating', '0.0'),
                    'found': True,
                    'cached_at': datetime.now()
                }},
                upsert=True
            )
            
        except Exception as e:
            logger.debug(f"MongoDB store error: {e}")
    
    async def _get_from_redis(self, key: str) -> Optional[Dict]:
        """Get poster from Redis cache"""
        try:
            if not self.redis_client:
                return None
            
            data = await self.redis_client.get(key)
            if data:
                self.stats['cache_hits'] += 1
                return json.loads(data)
            return None
            
        except Exception as e:
            logger.debug(f"Redis error: {e}")
            return None
    
    async def _store_in_redis(self, key: str, data: Dict):
        """Store poster in Redis cache"""
        try:
            if not self.redis_client:
                return
            
            await self.redis_client.setex(
                key,
                CACHE_TTL,
                json.dumps(data)
            )
            
        except Exception as e:
            logger.debug(f"Redis store error: {e}")
    
    async def _get_from_memory(self, key: str) -> Optional[Dict]:
        """Get poster from memory cache"""
        async with self.cache_lock:
            if key in self.poster_cache:
                data, cached_at = self.poster_cache[key]
                if (datetime.now() - cached_at).total_seconds() < MEMORY_CACHE_TTL:
                    self.stats['cache_hits'] += 1
                    return data
                else:
                    del self.poster_cache[key]
            return None
    
    async def _store_in_memory(self, key: str, data: Dict):
        """Store poster in memory cache"""
        async with self.cache_lock:
            self.poster_cache[key] = (data, datetime.now())
            
            # Limit cache size
            if len(self.poster_cache) > 500:
                # Remove oldest entries
                sorted_keys = sorted(
                    self.poster_cache.keys(),
                    key=lambda k: self.poster_cache[k][1]
                )
                for old_key in sorted_keys[:100]:
                    del self.poster_cache[old_key]

    # ============================================================================
    # TITLE CLEANING & VARIATIONS
    # ============================================================================
    
    def _clean_title_for_search(self, title: str) -> str:
        """Clean title for search - PRESERVES NUMBERS"""
        if not title:
            return ""
        
        # Remove special characters but keep numbers
        clean = re.sub(r'[^\w\s\-]', ' ', title)
        clean = re.sub(r'\s+', ' ', clean).strip()
        
        # Remove common quality tags but keep numbers
        quality_patterns = [
            r'\b(480p|720p|1080p|2160p|4k|hd|hevc|x265|x264)\b',
            r'\b(web-dl|webrip|bluray|hdtv|dvdrip|cam|ts|tc|screener)\b',
            r'\b(hindi|english|tamil|telugu|malayalam|kannada|dual|audio)\b'
        ]
        for pattern in quality_patterns:
            clean = re.sub(pattern, '', clean, flags=re.IGNORECASE)
        
        clean = re.sub(r'\s+', ' ', clean).strip()
        
        # Remove year at end (but keep for search)
        clean = re.sub(r'\s+(19|20)\d{2}$', '', clean)
        
        return clean
    
    def _generate_search_variations(self, title: str, year: str = "") -> List[str]:
        """Generate search variations for better matching"""
        variations = []
        clean_title = self._clean_title_for_search(title)
        
        if not clean_title:
            return []
        
        # 1. Original clean title
        variations.append(clean_title)
        
        # 2. With year
        if year:
            variations.append(f"{clean_title} {year}")
        
        # 3. Remove common suffixes (season, part, volume, chapter)
        cleaned = re.sub(r'\s*(season|saison|part|episode|vol|volume|chapter)\s*\d+.*$', '', clean_title, flags=re.IGNORECASE)
        if cleaned.strip() and cleaned.strip() != clean_title:
            variations.append(cleaned.strip())
            if year:
                variations.append(f"{cleaned.strip()} {year}")
        
        # 4. For numbered movies (Fast & Furious 2, Don 2, etc.)
        numbered_match = re.search(r'^(.*?)\s+(\d+)$', clean_title)
        if numbered_match:
            base = numbered_match.group(1).strip()
            num = numbered_match.group(2)
            variations.append(base)
            variations.append(f"{base} {num}")
            if year:
                variations.append(f"{base} {year}")
        
        # 5. Remove "The" prefix
        if clean_title.lower().startswith('the '):
            without_the = clean_title[4:]
            variations.append(without_the)
            if year:
                variations.append(f"{without_the} {year}")
        
        # 6. Common aliases
        aliases = {
            'money heist': ['La Casa De Papel'],
            'the boys': ['Boys'],
            'fast and furious': ['Fast & Furious', 'Furious'],
            'game of thrones': ['GoT', 'Thrones'],
            'breaking bad': ['BrBa'],
            'stranger things': ['Stranger'],
            'the walking dead': ['Walking Dead', 'TWD']
        }
        
        title_lower = clean_title.lower()
        for key, alias_list in aliases.items():
            if key in title_lower:
                for alias in alias_list:
                    variations.append(alias)
                    if year:
                        variations.append(f"{alias} {year}")
        
        # 7. Remove special characters
        clean2 = re.sub(r'[^\w\s]', ' ', clean_title)
        clean2 = re.sub(r'\s+', ' ', clean2).strip()
        if clean2 != clean_title:
            variations.append(clean2)
            if year:
                variations.append(f"{clean2} {year}")
        
        # 8. Original title (as fallback)
        variations.append(title)
        if year:
            variations.append(f"{title} {year}")
        
        # Deduplicate and limit
        seen = set()
        unique_variations = []
        for v in variations:
            v_clean = v.strip()
            if v_clean and v_clean.lower() not in seen:
                seen.add(v_clean.lower())
                unique_variations.append(v_clean)
        
        return unique_variations[:15]

    # ============================================================================
    # 1️⃣ MONGODB CACHE
    # ============================================================================
    
    async def _fetch_from_mongodb(self, title: str, year: str = "") -> Optional[Dict]:
        """Fetch poster from MongoDB cache"""
        cache_key = self._get_cache_key(title, year)
        return await self._get_from_mongodb(cache_key)

    # ============================================================================
    # 2️⃣ TMDB - THE MOVIE DATABASE
    # ============================================================================
    
    async def _fetch_from_tmdb(self, title: str, year: str = "") -> Optional[Dict]:
        """Fetch poster from TMDB"""
        try:
            session = await self._get_session()
            clean_title = self._clean_title_for_search(title)
            
            if not clean_title:
                return None
            
            # Try with year first if available
            search_terms = []
            if year:
                search_terms.append(f"{clean_title} {year}")
            search_terms.append(clean_title)
            
            for search_term in search_terms:
                params = {
                    'api_key': self.tmdb_api_key,
                    'query': search_term,
                    'language': 'en-US',
                    'page': 1,
                    'include_adult': False
                }
                
                async with session.get(
                    'https://api.themoviedb.org/3/search/movie',
                    params=params
                ) as response:
                    if response.status != 200:
                        continue
                    
                    data = await response.json()
                    results = data.get('results', [])
                    
                    if not results:
                        continue
                    
                    # Get first result
                    movie = results[0]
                    poster_path = movie.get('poster_path')
                    
                    if not poster_path:
                        continue
                    
                    poster_url = f"{TMDB_IMAGE_BASE}{poster_path}"
                    rating = movie.get('vote_average', 0)
                    release_date = movie.get('release_date', '')
                    
                    self.stats['tmdb_hits'] += 1
                    logger.debug(f"🎬 TMDB HIT: {title}")
                    
                    return {
                        'poster_url': poster_url,
                        'source': PosterSource.TMDB.value,
                        'rating': str(rating),
                        'year': release_date[:4] if release_date else year,
                        'title': movie.get('title', title),
                        'found': True,
                        'tmdb_id': movie.get('id')
                    }
            
            return None
            
        except asyncio.TimeoutError:
            logger.debug(f"⏰ TMDB timeout for {title}")
        except Exception as e:
            logger.debug(f"⚠️ TMDB error for {title}: {e}")
        
        return None

    # ============================================================================
    # 3️⃣ OMDB - OPEN MOVIE DATABASE
    # ============================================================================
    
    async def _fetch_from_omdb(self, title: str, year: str = "") -> Optional[Dict]:
        """Fetch poster from OMDB"""
        try:
            session = await self._get_session()
            clean_title = self._clean_title_for_search(title)
            
            if not clean_title:
                return None
            
            params = {
                'apikey': self.omdb_api_key,
                't': clean_title,
                'plot': 'short'
            }
            
            if year:
                params['y'] = year
            
            async with session.get('http://www.omdbapi.com/', params=params) as response:
                if response.status != 200:
                    return None
                
                data = await response.json()
                
                if data.get('Response') != 'True':
                    return None
                
                poster_url = data.get('Poster')
                
                if not poster_url or poster_url == 'N/A':
                    return None
                
                self.stats['omdb_hits'] += 1
                logger.debug(f"🎬 OMDB HIT: {title}")
                
                return {
                    'poster_url': poster_url,
                    'source': PosterSource.OMDB.value,
                    'rating': data.get('imdbRating', '0.0'),
                    'year': data.get('Year', year),
                    'title': data.get('Title', title),
                    'found': True,
                    'imdb_id': data.get('imdbID')
                }
                
        except asyncio.TimeoutError:
            logger.debug(f"⏰ OMDB timeout for {title}")
        except Exception as e:
            logger.debug(f"⚠️ OMDB error for {title}: {e}")
        
        return None

    # ============================================================================
    # 4️⃣ LETTERBOXD
    # ============================================================================
    
    async def _fetch_from_letterboxd(self, title: str, year: str = "") -> Optional[Dict]:
        """Fetch poster from Letterboxd"""
        try:
            session = await self._get_session()
            clean_title = self._clean_title_for_search(title)
            
            if not clean_title:
                return None
            
            # Create slug
            slug = re.sub(r'[^a-z0-9]+', '-', clean_title.lower())
            slug = re.sub(r'-+', '-', slug).strip('-')
            
            # Try with year if available
            years_to_try = [year] if year else []
            if not year:
                # Try common years if no year provided
                years_to_try = ['', '2024', '2023', '2022', '2021', '2020', '2019', '2018']
            
            for y in years_to_try:
                url = f"https://letterboxd.com/film/{slug}/"
                if y:
                    url = f"https://letterboxd.com/film/{slug}-{y}/"
                
                async with session.get(url) as response:
                    if response.status != 200:
                        continue
                    
                    html = await response.text()
                    
                    # Find poster URL
                    poster_match = re.search(
                        r'<meta\s+property="og:image"\s+content="([^"]+)"',
                        html,
                        re.IGNORECASE
                    )
                    
                    if poster_match:
                        poster_url = poster_match.group(1)
                        # Clean URL - remove size parameters
                        poster_url = re.sub(r'-\d+x\d+\.jpg$', '.jpg', poster_url)
                        
                        # Get rating
                        rating_match = re.search(
                            r'<span\s+class="rating"[^>]*>([\d.]+)</span>',
                            html,
                            re.IGNORECASE
                        )
                        rating = rating_match.group(1) if rating_match else '0.0'
                        
                        self.stats['letterboxd_hits'] += 1
                        logger.debug(f"🎬 Letterboxd HIT: {title}")
                        
                        return {
                            'poster_url': poster_url,
                            'source': PosterSource.LETTERBOXD.value,
                            'rating': rating,
                            'year': y or year,
                            'title': title,
                            'found': True
                        }
            
            return None
            
        except asyncio.TimeoutError:
            logger.debug(f"⏰ Letterboxd timeout for {title}")
        except Exception as e:
            logger.debug(f"⚠️ Letterboxd error for {title}: {e}")
        
        return None

    # ============================================================================
    # 5️⃣ IMDB
    # ============================================================================
    
    async def _fetch_from_imdb(self, title: str, year: str = "") -> Optional[Dict]:
        """Fetch poster from IMDB"""
        try:
            session = await self._get_session()
            clean_title = self._clean_title_for_search(title)
            
            if not clean_title:
                return None
            
            # Search for movie
            first_char = clean_title[0].lower()
            search_term = clean_title.replace(' ', '_')
            url = f"https://v2.sg.media-imdb.com/suggestion/{first_char}/{urllib.parse.quote(search_term)}.json"
            
            async with session.get(url) as response:
                if response.status != 200:
                    return None
                
                data = await response.json()
                results = data.get('d', [])
                
                if not results:
                    return None
                
                # Find matching result (prefer exact match or year match)
                best_match = None
                for item in results:
                    title_lower = item.get('l', '').lower()
                    clean_lower = clean_title.lower()
                    
                    # Check if title contains our search or vice versa
                    if clean_lower in title_lower or title_lower in clean_lower:
                        best_match = item
                        break
                
                if not best_match:
                    best_match = results[0]
                
                # Get poster
                img = best_match.get('i')
                if isinstance(img, dict):
                    poster_url = img.get('imageUrl')
                elif isinstance(img, list):
                    poster_url = img[0] if img else None
                else:
                    poster_url = img if isinstance(img, str) else None
                
                if not poster_url or not poster_url.startswith('http'):
                    return None
                
                # Get year
                movie_year = str(best_match.get('yr', ''))
                if not movie_year and year:
                    movie_year = year
                
                self.stats['imdb_hits'] += 1
                logger.debug(f"🎬 IMDB HIT: {title}")
                
                return {
                    'poster_url': poster_url,
                    'source': PosterSource.IMDB.value,
                    'rating': '0.0',
                    'year': movie_year,
                    'title': best_match.get('l', title),
                    'found': True,
                    'imdb_id': best_match.get('id')
                }
                
        except asyncio.TimeoutError:
            logger.debug(f"⏰ IMDB timeout for {title}")
        except Exception as e:
            logger.debug(f"⚠️ IMDB error for {title}: {e}")
        
        return None

    # ============================================================================
    # 6️⃣ JUSTWATCH
    # ============================================================================
    
    async def _fetch_from_justwatch(self, title: str, year: str = "") -> Optional[Dict]:
        """Fetch poster from JustWatch"""
        try:
            session = await self._get_session()
            clean_title = self._clean_title_for_search(title)
            
            if not clean_title:
                return None
            
            # Create slug
            slug = re.sub(r'[^a-z0-9]+', '-', clean_title.lower())
            slug = re.sub(r'-+', '-', slug).strip('-')
            
            url = f"https://www.justwatch.com/in/movie/{slug}"
            
            async with session.get(url) as response:
                if response.status != 200:
                    # Try US domain
                    url = f"https://www.justwatch.com/us/movie/{slug}"
                    async with session.get(url) as response2:
                        if response2.status != 200:
                            return None
                        html = await response2.text()
                else:
                    html = await response.text()
                
                # Find poster URL
                poster_match = re.search(
                    r'<meta\s+property="og:image"\s+content="([^"]+)"',
                    html,
                    re.IGNORECASE
                )
                
                if poster_match:
                    poster_url = poster_match.group(1)
                    # Use higher quality version
                    poster_url = poster_url.replace('/poster/', '/poster/s')
                    
                    self.stats['justwatch_hits'] += 1
                    logger.debug(f"🎬 JustWatch HIT: {title}")
                    
                    return {
                        'poster_url': poster_url,
                        'source': PosterSource.JUSTWATCH.value,
                        'rating': '0.0',
                        'year': year,
                        'title': title,
                        'found': True
                    }
            
            return None
            
        except asyncio.TimeoutError:
            logger.debug(f"⏰ JustWatch timeout for {title}")
        except Exception as e:
            logger.debug(f"⚠️ JustWatch error for {title}: {e}")
        
        return None

    # ============================================================================
    # 7️⃣ IMPAWARDS
    # ============================================================================
    
    async def _fetch_from_impawards(self, title: str, year: str = "") -> Optional[Dict]:
        """Fetch poster from IMPAwards"""
        try:
            session = await self._get_session()
            clean_title = self._clean_title_for_search(title)
            
            if not clean_title:
                return None
            
            # Extract year from title if not provided
            if not year:
                year_match = re.search(r'\b(19|20)\d{2}\b', clean_title)
                if year_match:
                    year = year_match.group()
            
            if not year:
                return None
            
            # Create slug
            slug = re.sub(r'[^a-z0-9]+', '_', clean_title.lower())
            slug = re.sub(r'_+', '_', slug).strip('_')
            
            # Try multiple formats
            formats = [
                f"https://www.impawards.com/{year}/posters/{slug}.jpg",
                f"https://www.impawards.com/{year}/posters/{slug}_ver1.jpg",
                f"https://www.impawards.com/{year}/posters/{slug}_ver2.jpg",
            ]
            
            for url in formats:
                try:
                    async with session.head(url, timeout=3) as response:
                        if response.status == 200:
                            self.stats['impawards_hits'] += 1
                            logger.debug(f"🎬 IMPAwards HIT: {title}")
                            
                            return {
                                'poster_url': url,
                                'source': PosterSource.IMPAWARDS.value,
                                'rating': '0.0',
                                'year': year,
                                'title': title,
                                'found': True
                            }
                except:
                    continue
            
            return None
            
        except asyncio.TimeoutError:
            logger.debug(f"⏰ IMPAwards timeout for {title}")
        except Exception as e:
            logger.debug(f"⚠️ IMPAwards error for {title}: {e}")
        
        return None

    # ============================================================================
    # 8️⃣ TELEGRAM CHANNEL SEARCH - NEW!
    # ============================================================================
    
    async def _fetch_from_telegram(self, title: str, year: str = "") -> Optional[Dict]:
        """
        🆕 Fetch poster from Telegram MAIN_CHANNEL
        Searches the main channel for the movie and extracts thumbnail
        """
        try:
            # Get client (prefer user client for better search)
            client = self.user_client or self.bot_client
            
            if not client:
                logger.debug("⚠️ No Telegram client available for poster fetch")
                return None
            
            # Generate search variations
            variations = self._generate_search_variations(title, year)
            
            if not variations:
                return None
            
            logger.debug(f"🔍 Searching Telegram for {title} with {len(variations)} variations")
            
            # Search in main channel
            for search_term in variations[:10]:  # Limit variations
                if len(search_term) < 2:
                    continue
                
                try:
                    # Search for messages
                    async for msg in client.search_messages(
                        self.main_channel_id,
                        query=search_term,
                        limit=5
                    ):
                        if not msg:
                            continue
                        
                        # Check if message has media with thumbnail
                        thumbnail_file_id = None
                        
                        # Check video
                        if hasattr(msg, 'video') and msg.video:
                            if hasattr(msg.video, 'thumbnail') and msg.video.thumbnail:
                                thumbnail_file_id = msg.video.thumbnail.file_id
                            elif hasattr(msg.video, 'thumbs') and msg.video.thumbs:
                                thumbnail_file_id = msg.video.thumbs[0].file_id
                        
                        # Check document
                        if not thumbnail_file_id and hasattr(msg, 'document') and msg.document:
                            if hasattr(msg.document, 'thumbnail') and msg.document.thumbnail:
                                thumbnail_file_id = msg.document.thumbnail.file_id
                            elif hasattr(msg.document, 'thumbs') and msg.document.thumbs:
                                thumbnail_file_id = msg.document.thumbs[0].file_id
                        
                        # Check photo
                        if not thumbnail_file_id and hasattr(msg, 'photo') and msg.photo:
                            try:
                                photo = msg.photo
                                if hasattr(photo, 'file_id'):
                                    thumbnail_file_id = photo.file_id
                                elif isinstance(photo, list) and photo:
                                    thumbnail_file_id = photo[0].file_id
                            except:
                                pass
                        
                        if thumbnail_file_id:
                            try:
                                # Download thumbnail
                                thumb_data = await client.download_media(
                                    thumbnail_file_id,
                                    in_memory=True
                                )
                                
                                if thumb_data:
                                    # Convert to base64
                                    if isinstance(thumb_data, bytes):
                                        thumb_url = f"data:image/jpeg;base64,{base64.b64encode(thumb_data).decode()}"
                                    else:
                                        # BytesIO object
                                        thumb_data.seek(0)
                                        thumb_url = f"data:image/jpeg;base64,{base64.b64encode(thumb_data.read()).decode()}"
                                    
                                    # Validate size
                                    if thumb_url and len(thumb_url) < 500000:  # Max 500KB
                                        self.stats['telegram_hits'] += 1
                                        logger.info(f"📺 TELEGRAM poster found for: {title} (via '{search_term}')")
                                        
                                        return {
                                            'poster_url': thumb_url,
                                            'source': PosterSource.TELEGRAM.value,
                                            'rating': '0.0',
                                            'year': year,
                                            'title': title,
                                            'found': True,
                                            'message_id': msg.id,
                                            'channel_id': self.main_channel_id
                                        }
                            except Exception as e:
                                logger.debug(f"Download thumbnail error: {e}")
                                continue
                                
                except Exception as e:
                    logger.debug(f"Telegram search error for '{search_term}': {e}")
                    continue
            
            # If not found in main channel, try file channel
            if self.file_channel_id:
                logger.debug(f"🔍 Searching Telegram file channel for {title}")
                try:
                    for search_term in variations[:5]:
                        async for msg in client.search_messages(
                            self.file_channel_id,
                            query=search_term,
                            limit=3
                        ):
                            if not msg:
                                continue
                            
                            # Check for thumbnail in video
                            if hasattr(msg, 'video') and msg.video:
                                if hasattr(msg.video, 'thumbnail') and msg.video.thumbnail:
                                    thumbnail_file_id = msg.video.thumbnail.file_id
                                    try:
                                        thumb_data = await client.download_media(thumbnail_file_id, in_memory=True)
                                        if thumb_data:
                                            if isinstance(thumb_data, bytes):
                                                thumb_url = f"data:image/jpeg;base64,{base64.b64encode(thumb_data).decode()}"
                                            else:
                                                thumb_data.seek(0)
                                                thumb_url = f"data:image/jpeg;base64,{base64.b64encode(thumb_data.read()).decode()}"
                                            
                                            if thumb_url and len(thumb_url) < 500000:
                                                self.stats['telegram_hits'] += 1
                                                logger.info(f"📺 TELEGRAM file channel poster found for: {title}")
                                                return {
                                                    'poster_url': thumb_url,
                                                    'source': PosterSource.TELEGRAM.value,
                                                    'rating': '0.0',
                                                    'year': year,
                                                    'title': title,
                                                    'found': True,
                                                    'message_id': msg.id,
                                                    'channel_id': self.file_channel_id
                                                }
                                    except:
                                        pass
                except Exception as e:
                    logger.debug(f"File channel search error: {e}")
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Telegram poster fetch error: {e}")
            return None

    # ============================================================================
    # 9️⃣ CUSTOM / FALLBACK
    # ============================================================================
    
    async def _create_custom_poster(self, title: str, year: str = "") -> Dict:
        """Create custom fallback poster"""
        self.stats['custom_hits'] += 1
        logger.debug(f"🎨 Custom fallback for: {title}")
        
        return {
            'poster_url': CUSTOM_POSTER_URL,
            'source': PosterSource.CUSTOM.value,
            'rating': '0.0',
            'year': year,
            'title': title,
            'found': False
        }

    # ============================================================================
    # MAIN FETCH POSTER - COMPLETE PRIORITY ORDER
    # ============================================================================
    
    async def fetch_poster(self, title: str, year: str = "", quality: str = "") -> Dict[str, Any]:
        """
        🎬 MAIN POSTER FETCH METHOD
        Follows priority order:
        1️⃣ MongoDB Cache → 2️⃣ TMDB → 3️⃣ OMDB → 4️⃣ Letterboxd → 
        5️⃣ IMDB → 6️⃣ JustWatch → 7️⃣ IMPAwards → 8️⃣ TELEGRAM → 9️⃣ Custom
        """
        self.stats['total_requests'] += 1
        
        if not title:
            return {
                'poster_url': CUSTOM_POSTER_URL,
                'source': PosterSource.CUSTOM.value,
                'rating': '0.0',
                'year': year,
                'title': title,
                'found': False
            }
        
        # Clean title for search
        clean_title = self._clean_title_for_search(title)
        if not clean_title:
            clean_title = title
        
        # Extract year from title if not provided
        if not year:
            year_match = re.search(r'\b(19|20)\d{2}\b', clean_title)
            if year_match:
                year = year_match.group()
        
        # Generate cache key
        cache_key = self._get_cache_key(clean_title, year)
        
        # ========== 1️⃣ MONGODB CACHE ==========
        logger.debug(f"🔍 Checking MongoDB cache for: {title}")
        cached = await self._fetch_from_mongodb(clean_title, year)
        if cached and cached.get('found'):
            return cached
        
        # ========== MEMORY CACHE ==========
        memory_cached = await self._get_from_memory(cache_key)
        if memory_cached and memory_cached.get('found'):
            return memory_cached
        
        # ========== REDIS CACHE ==========
        redis_cached = await self._get_from_redis(cache_key)
        if redis_cached and redis_cached.get('found'):
            return redis_cached
        
        # ========== 2️⃣ TMDB ==========
        logger.debug(f"🔍 Searching TMDB for: {title}")
        result = await self._fetch_from_tmdb(clean_title, year)
        if result:
            await self._store_in_mongodb(cache_key, result)
            await self._store_in_redis(cache_key, result)
            await self._store_in_memory(cache_key, result)
            return result
        
        # ========== 3️⃣ OMDB ==========
        logger.debug(f"🔍 Searching OMDB for: {title}")
        result = await self._fetch_from_omdb(clean_title, year)
        if result:
            await self._store_in_mongodb(cache_key, result)
            await self._store_in_redis(cache_key, result)
            await self._store_in_memory(cache_key, result)
            return result
        
        # ========== 4️⃣ LETTERBOXD ==========
        logger.debug(f"🔍 Searching Letterboxd for: {title}")
        result = await self._fetch_from_letterboxd(clean_title, year)
        if result:
            await self._store_in_mongodb(cache_key, result)
            await self._store_in_redis(cache_key, result)
            await self._store_in_memory(cache_key, result)
            return result
        
        # ========== 5️⃣ IMDB ==========
        logger.debug(f"🔍 Searching IMDB for: {title}")
        result = await self._fetch_from_imdb(clean_title, year)
        if result:
            await self._store_in_mongodb(cache_key, result)
            await self._store_in_redis(cache_key, result)
            await self._store_in_memory(cache_key, result)
            return result
        
        # ========== 6️⃣ JUSTWATCH ==========
        logger.debug(f"🔍 Searching JustWatch for: {title}")
        result = await self._fetch_from_justwatch(clean_title, year)
        if result:
            await self._store_in_mongodb(cache_key, result)
            await self._store_in_redis(cache_key, result)
            await self._store_in_memory(cache_key, result)
            return result
        
        # ========== 7️⃣ IMPAWARDS ==========
        logger.debug(f"🔍 Searching IMPAwards for: {title}")
        result = await self._fetch_from_impawards(clean_title, year)
        if result:
            await self._store_in_mongodb(cache_key, result)
            await self._store_in_redis(cache_key, result)
            await self._store_in_memory(cache_key, result)
            return result
        
        # ========== 8️⃣ TELEGRAM CHANNEL SEARCH ==========
        logger.debug(f"🔍 Searching TELEGRAM for: {title}")
        result = await self._fetch_from_telegram(clean_title, year)
        if result:
            await self._store_in_mongodb(cache_key, result)
            await self._store_in_redis(cache_key, result)
            await self._store_in_memory(cache_key, result)
            return result
        
        # ========== 9️⃣ CUSTOM FALLBACK ==========
        logger.debug(f"⚠️ All sources failed for: {title}, using fallback")
        result = await self._create_custom_poster(clean_title, year)
        await self._store_in_mongodb(cache_key, result)
        await self._store_in_redis(cache_key, result)
        await self._store_in_memory(cache_key, result)
        
        self.stats['failed_requests'] += 1
        return result
    
    # ============================================================================
    # BATCH FETCH
    # ============================================================================
    
    async def fetch_batch_posters(self, titles: List[Tuple[str, str]]) -> Dict[str, Dict]:
        """
        Fetch posters for multiple titles in parallel
        titles: List of (title, year) tuples
        """
        tasks = []
        for title, year in titles:
            tasks.append(self.fetch_poster(title, year))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        poster_dict = {}
        for i, (title, _) in enumerate(titles):
            result = results[i]
            if isinstance(result, Exception):
                logger.error(f"❌ Error fetching poster for {title}: {result}")
                poster_dict[title] = {
                    'poster_url': CUSTOM_POSTER_URL,
                    'source': PosterSource.CUSTOM.value,
                    'title': title,
                    'found': False
                }
            else:
                poster_dict[title] = result
        
        return poster_dict
    
    # ============================================================================
    # STATISTICS
    # ============================================================================
    
    def get_stats(self) -> Dict[str, int]:
        """Get poster fetcher statistics"""
        total_hits = sum([
            self.stats['mongodb_hits'],
            self.stats['tmdb_hits'],
            self.stats['omdb_hits'],
            self.stats['letterboxd_hits'],
            self.stats['imdb_hits'],
            self.stats['justwatch_hits'],
            self.stats['impawards_hits'],
            self.stats['telegram_hits'],
            self.stats['custom_hits']
        ])
        
        return {
            **self.stats,
            'total_hits': total_hits,
            'cache_size': len(self.poster_cache),
            'success_rate': (
                (total_hits / self.stats['total_requests'] * 100) 
                if self.stats['total_requests'] > 0 else 0
            )
        }
    
    def reset_stats(self):
        """Reset statistics"""
        for key in self.stats:
            self.stats[key] = 0
    
    # ============================================================================
    # CLEANUP
    # ============================================================================
    
    async def close(self):
        """Close HTTP session and cleanup"""
        if self.http_session and not self.http_session.closed:
            await self.http_session.close()
            self.http_session = None
        
        self.poster_cache.clear()
        logger.info("🎬 PosterFetcher closed")

    async def clear_cache(self):
        """Clear all caches"""
        async with self.cache_lock:
            self.poster_cache.clear()
        
        if self.redis_client:
            try:
                # Delete all poster keys
                pattern = "poster:*"
                keys = await self.redis_client.keys(pattern)
                if keys:
                    await self.redis_client.delete(*keys)
            except Exception as e:
                logger.debug(f"Redis clear error: {e}")
        
        if self.posters_col:
            try:
                await self.posters_col.delete_many({})
            except Exception as e:
                logger.debug(f"MongoDB clear error: {e}")
        
        logger.info("🗑️ Poster cache cleared")

# ============================================================================
# FACTORY FUNCTION
# ============================================================================

async def create_poster_fetcher(config, mongo_client=None, redis_client=None):
    """Create and initialize poster fetcher"""
    fetcher = PosterFetcher(config, mongo_client, redis_client)
    await fetcher.initialize()
    return fetcher

# ============================================================================
# WRAPPER FUNCTION FOR BACKWARD COMPATIBILITY
# ============================================================================

async def get_poster_for_movie(
    title: str, 
    year: str = "", 
    quality: str = "",
    poster_fetcher: Optional[PosterFetcher] = None
) -> Dict[str, Any]:
    """Wrapper function for backward compatibility"""
    if not poster_fetcher:
        return {
            'poster_url': CUSTOM_POSTER_URL,
            'source': PosterSource.CUSTOM.value,
            'rating': '0.0',
            'year': year,
            'title': title,
            'found': False
        }
    
    return await poster_fetcher.fetch_poster(title, year, quality)

# ============================================================================
# TESTING
# ============================================================================

async def test_poster_fetcher():
    """Test the poster fetcher with all sources"""
    print("=" * 70)
    print("🎬 TESTING POSTER FETCHER v3.0 - ALL SOURCES")
    print("=" * 70)
    
    # Mock config
    class MockConfig:
        TMDB_API_KEY = "e547e17d4e91f3e62a571655cd1ccaff"
        OMDB_API_KEY = "8265bd1c"
        MAIN_CHANNEL_ID = -1001767371495
        FILE_CHANNEL_ID = -1001768249569
    
    # Create fetcher
    fetcher = PosterFetcher(MockConfig())
    await fetcher.initialize()
    
    # Test movies
    test_movies = [
        ("Inception", "2010"),
        ("The Dark Knight", "2008"),
        ("Interstellar", "2014"),
        ("Avatar", "2009"),
        ("Money Heist", "2017"),
        ("The Boys", "2019"),
        ("Fast & Furious 2", "2003"),
    ]
    
    print("\n🔍 Testing poster fetch for movies...")
    print("-" * 70)
    
    for title, year in test_movies:
        print(f"\n📽️ {title} ({year})")
        result = await fetcher.fetch_poster(title, year)
        
        if result and result.get('found'):
            source = result.get('source', 'unknown')
            print(f"   ✅ FOUND! Source: {source}")
            print(f"   📷 URL: {result.get('poster_url', '')[:60]}...")
            print(f"   ⭐ Rating: {result.get('rating', 'N/A')}")
        else:
            print(f"   ❌ NOT FOUND")
    
    # Print statistics
    print("\n" + "=" * 70)
    print("📊 STATISTICS:")
    stats = fetcher.get_stats()
    for key, value in sorted(stats.items()):
        if isinstance(value, (int, float)):
            if value > 0 or key in ['success_rate', 'cache_size']:
                print(f"   {key}: {value}")
    print("=" * 70)
    
    await fetcher.close()
    print("\n✅ Test complete!")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    asyncio.run(test_poster_fetcher())
