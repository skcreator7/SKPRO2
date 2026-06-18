# poster_fetching.py - Fixed version without Letterboxd

import asyncio
import aiohttp
import logging
import re
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from urllib.parse import quote

logger = logging.getLogger(__name__)

class PosterSource:
    """Poster source constants"""
    TMDB = "tmdb"
    OMDB = "omdb"
    TELEGRAM = "telegram"
    NONE = "none"

class PosterFetcher:
    """Fetch movie posters from various sources (NO LETTERBOXD)"""
    
    def __init__(self, config, redis_client=None):
        self.config = config
        self.redis_client = redis_client
        self.tmdb_api_key = getattr(config, 'TMDB_API_KEY', 'e547e17d4e91f3e62a571655cd1ccaff')
        self.omdb_api_key = getattr(config, 'OMDB_API_KEY', '8265bd1c')
        self.timeout = getattr(config, 'POSTER_FETCH_TIMEOUT', 5)
        self.cache_ttl = getattr(config, 'POSTER_CACHE_TTL', 86400)
        self.session = None
        self.stats = {
            'total_fetches': 0,
            'tmdb_hits': 0,
            'omdb_hits': 0,
            'telegram_hits': 0,
            'misses': 0,
            'cache_hits': 0
        }
        
        # TMDB image base URL
        self.tmdb_image_base = "https://image.tmdb.org/t/p/w500"
        self.tmdb_image_original = "https://image.tmdb.org/t/p/original"
        
        logger.info("🎬 PosterFetcher initialized (Letterboxd REMOVED)")

    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout + 2),
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            )
        return self.session

    async def fetch_poster(self, title: str, year: str = "") -> Dict[str, Any]:
        """
        Fetch poster for movie - TMDB only (Letterboxd removed)
        """
        self.stats['total_fetches'] += 1
        
        if not title:
            return self._empty_result(title, year)
        
        # Clean title - preserve numbers
        clean_title = self._clean_title(title)
        
        # Check cache first
        cache_key = f"poster:{clean_title}:{year}"
        if self.redis_client:
            try:
                cached = await self.redis_client.get(cache_key)
                if cached:
                    self.stats['cache_hits'] += 1
                    data = json.loads(cached)
                    logger.debug(f"📦 Cache hit: {clean_title}")
                    return data
            except Exception as e:
                logger.debug(f"Cache read error: {e}")
        
        result = None
        
        # Try TMDB first (primary source)
        result = await self._fetch_from_tmdb(clean_title, year)
        
        if result and result.get('poster_url'):
            self.stats['tmdb_hits'] += 1
            logger.info(f"✅ TMDB poster found: {clean_title}")
        else:
            # Try OMDB as fallback
            result = await self._fetch_from_omdb(clean_title, year)
            if result and result.get('poster_url'):
                self.stats['omdb_hits'] += 1
                logger.info(f"✅ OMDB poster found: {clean_title}")
            else:
                self.stats['misses'] += 1
                result = self._empty_result(clean_title, year)
                logger.debug(f"❌ No poster found: {clean_title}")
        
        # Cache result
        if self.redis_client and result:
            try:
                await self.redis_client.setex(
                    cache_key, 
                    self.cache_ttl,
                    json.dumps(result)
                )
            except Exception as e:
                logger.debug(f"Cache write error: {e}")
        
        return result

    async def fetch_poster_from_telegram(self, title: str, year: str = "") -> Dict[str, Any]:
        """
        Fetch poster from Telegram channel using title search
        """
        try:
            # Clean title - preserve numbers
            clean_title = self._clean_title(title)
            
            # Create search variations
            search_variations = self._generate_search_variations(clean_title, year)
            
            # Try to search in main channel
            if User and user_session_ready:
                for search_term in search_variations:
                    try:
                        async for msg in User.search_messages(
                            Config.MAIN_CHANNEL_ID,
                            query=search_term,
                            limit=3
                        ):
                            if msg and msg.text:
                                # Check if this message has media with thumbnail
                                if hasattr(msg, 'media') and msg.media:
                                    if hasattr(msg.media, 'thumbnail') and msg.media.thumbnail:
                                        thumb_data = await User.download_media(
                                            msg.media.thumbnail.file_id,
                                            in_memory=True
                                        )
                                        if thumb_data:
                                            if isinstance(thumb_data, bytes):
                                                thumb_url = f"data:image/jpeg;base64,{base64.b64encode(thumb_data).decode()}"
                                            else:
                                                thumb_url = f"data:image/jpeg;base64,{base64.b64encode(thumb_data.read()).decode()}"
                                            
                                            if thumb_url and len(thumb_url) < 100000:
                                                self.stats['telegram_hits'] += 1
                                                logger.info(f"✅ Telegram poster found: {clean_title} (via '{search_term}')")
                                                return {
                                                    'poster_url': thumb_url,
                                                    'source': PosterSource.TELEGRAM,
                                                    'title': clean_title,
                                                    'year': year,
                                                    'found': True,
                                                    'cached_at': datetime.now().isoformat()
                                                }
                    except Exception as e:
                        logger.debug(f"Telegram search error for '{search_term}': {e}")
                        continue
            
            return self._empty_result(clean_title, year)
            
        except Exception as e:
            logger.error(f"❌ Telegram poster fetch error: {e}")
            return self._empty_result(title, year)

    def _generate_search_variations(self, title: str, year: str = "") -> List[str]:
        """Generate search variations for better matching"""
        variations = []
        
        # Original title
        variations.append(title)
        
        # Title with year
        if year:
            variations.append(f"{title} {year}")
        
        # Remove common suffixes
        clean = re.sub(r'\s*(season|saison|part|episode|vol|volume)\s*\d+.*$', '', title, flags=re.IGNORECASE)
        if clean != title:
            variations.append(clean.strip())
            if year:
                variations.append(f"{clean.strip()} {year}")
        
        # For numbered movies (Don 2, Murder 2, Fast & Furious 2)
        numbered_match = re.search(r'^(.*?)\s+(\d+)$', title)
        if numbered_match:
            base = numbered_match.group(1).strip()
            num = numbered_match.group(2)
            variations.append(base)
            variations.append(f"{base} {num}")
            if year:
                variations.append(f"{base} {year}")
        
        # Remove "The" prefix
        if title.lower().startswith('the '):
            without_the = title[4:]
            variations.append(without_the)
            if year:
                variations.append(f"{without_the} {year}")
        
        # For series like "Money Heist" -> "Money Heist Season", "La Casa De Papel"
        if 'heist' in title.lower():
            variations.append("Money Heist")
            variations.append("La Casa De Papel")
            variations.append("La Casa De Papel Season")
            if year:
                variations.append(f"Money Heist {year}")
        
        # For "The Boys"
        if 'boys' in title.lower():
            variations.append("The Boys")
            variations.append("Boys")
            if year:
                variations.append(f"The Boys {year}")
        
        # Remove special characters
        clean_title = re.sub(r'[^\w\s]', ' ', title)
        clean_title = re.sub(r'\s+', ' ', clean_title).strip()
        if clean_title != title:
            variations.append(clean_title)
            if year:
                variations.append(f"{clean_title} {year}")
        
        # Deduplicate
        seen = set()
        unique_variations = []
        for v in variations:
            if v.lower() not in seen:
                seen.add(v.lower())
                unique_variations.append(v)
        
        return unique_variations[:10]  # Limit to 10 variations

    async def _fetch_from_tmdb(self, title: str, year: str = "") -> Dict[str, Any]:
        """Fetch poster from TMDB"""
        try:
            session = await self._get_session()
            
            # Search for movie
            search_query = quote(title)
            search_url = f"https://api.themoviedb.org/3/search/movie?api_key={self.tmdb_api_key}&query={search_query}&language=en-US&page=1"
            
            if year:
                search_url += f"&year={year}"
            
            async with session.get(search_url) as response:
                if response.status != 200:
                    logger.debug(f"TMDB search failed: {response.status}")
                    return self._empty_result(title, year)
                
                data = await response.json()
                results = data.get('results', [])
                
                if not results:
                    # Try without year if no results
                    if year:
                        return await self._fetch_from_tmdb(title, "")
                    return self._empty_result(title, year)
                
                # Get first result
                movie = results[0]
                poster_path = movie.get('poster_path')
                
                if not poster_path:
                    return self._empty_result(title, year)
                
                poster_url = f"{self.tmdb_image_base}{poster_path}"
                
                return {
                    'poster_url': poster_url,
                    'source': PosterSource.TMDB,
                    'title': movie.get('title', title),
                    'year': str(movie.get('release_date', year))[:4] if movie.get('release_date') else year,
                    'rating': str(movie.get('vote_average', 0)),
                    'id': str(movie.get('id', '')),
                    'found': True,
                    'cached_at': datetime.now().isoformat()
                }
                
        except asyncio.TimeoutError:
            logger.debug(f"TMDB timeout for {title}")
            return self._empty_result(title, year)
        except Exception as e:
            logger.debug(f"TMDB error for {title}: {e}")
            return self._empty_result(title, year)

    async def _fetch_from_omdb(self, title: str, year: str = "") -> Dict[str, Any]:
        """Fetch poster from OMDB"""
        try:
            session = await self._get_session()
            
            search_query = quote(title)
            url = f"http://www.omdbapi.com/?apikey={self.omdb_api_key}&t={search_query}&plot=short"
            
            if year:
                url += f"&y={year}"
            
            async with session.get(url) as response:
                if response.status != 200:
                    return self._empty_result(title, year)
                
                data = await response.json()
                
                if data.get('Response') != 'True':
                    return self._empty_result(title, year)
                
                poster_url = data.get('Poster', '')
                if not poster_url or poster_url == 'N/A':
                    return self._empty_result(title, year)
                
                return {
                    'poster_url': poster_url,
                    'source': PosterSource.OMDB,
                    'title': data.get('Title', title),
                    'year': data.get('Year', year),
                    'rating': data.get('imdbRating', '0.0'),
                    'id': data.get('imdbID', ''),
                    'found': True,
                    'cached_at': datetime.now().isoformat()
                }
                
        except asyncio.TimeoutError:
            logger.debug(f"OMDB timeout for {title}")
            return self._empty_result(title, year)
        except Exception as e:
            logger.debug(f"OMDB error for {title}: {e}")
            return self._empty_result(title, year)

    def _clean_title(self, title: str) -> str:
        """Clean movie title for searching - PRESERVE NUMBERS"""
        if not title:
            return ""
        
        # Remove special characters but keep numbers
        clean = re.sub(r'[^\w\s\-]', ' ', title)
        clean = re.sub(r'\s+', ' ', clean).strip()
        
        # Remove common tags but keep numbers
        clean = re.sub(r'\b(480p|720p|1080p|2160p|4k|hd|hevc|x264|x265|web-dl|webrip|bluray|hdtv)\b', '', clean, flags=re.IGNORECASE)
        clean = re.sub(r'\s+', ' ', clean).strip()
        
        return clean

    def _empty_result(self, title: str, year: str = "") -> Dict[str, Any]:
        """Return empty poster result"""
        return {
            'poster_url': '',
            'source': PosterSource.NONE,
            'title': title,
            'year': year,
            'rating': '0.0',
            'id': '',
            'found': False,
            'cached_at': datetime.now().isoformat()
        }

    async def close(self):
        """Close session"""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None

    def get_stats(self) -> Dict[str, int]:
        """Get fetcher statistics"""
        return self.stats.copy()
