"""
poster_fetching.py - Multi-source poster fetching with intelligent fallback
"""
import aiohttp
import asyncio
from typing import Dict, Any, Optional, List
import logging
from urllib.parse import quote
import re

logger = logging.getLogger(__name__)

class PosterFetcher:
    def __init__(self, cache_manager, config):
        self.cache = cache_manager
        self.config = config
        self.session = None
        
        # Source priorities and timeouts
        self.sources = {
            'letterboxd': {
                'priority': 1,
                'timeout': 3,
                'enabled': True
            },
            'omdb': {
                'priority': 2,
                'timeout': 2,
                'enabled': hasattr(config, 'OMDB_API_KEY') and config.OMDB_API_KEY
            },
            'imdb': {
                'priority': 3,
                'timeout': 4,
                'enabled': True
            },
            'impawards': {
                'priority': 4,
                'timeout': 3,
                'enabled': True
            },
            'justwatch': {
                'priority': 5,
                'timeout': 3,
                'enabled': True
            },
            'youtube': {
                'priority': 6,
                'timeout': 3,
                'enabled': True
            }
        }
        
        # Default fallback
        self.default_poster = "https://iili.io/fAeIwv9.th.png"
    
    async def init_session(self):
        """Initialize HTTP session"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=10)
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def fetch_poster(self, title: str, year: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch poster from multiple sources with intelligent fallback
        Returns: {
            'url': poster_url,
            'source': source_name,
            'quality': 'high'|'medium'|'low',
            'rating': '0.0-10.0',
            'cached': True|False,
            'response_time': ms
        }
        """
        start_time = asyncio.get_event_loop().time()
        
        # Check cache first
        for source_name in self.sources.keys():
            cached = await self.cache.get_poster(title, source_name)
            if cached:
                cached['cached'] = True
                cached['response_time'] = 0
                return cached
        
        # Prepare search query
        search_query = f"{title} {year}" if year else title
        
        # Fetch from sources in parallel with priority
        tasks = []
        for source_name, source_config in self.sources.items():
            if source_config['enabled']:
                task = self._fetch_from_source(source_name, title, year, search_query)
                tasks.append(task)
        
        # Get first successful result
        for task in asyncio.as_completed(tasks):
            try:
                result = await task
                if result and result.get('url'):
                    # Cache the result
                    await self.cache.cache_poster(title, result, result['source'])
                    
                    end_time = asyncio.get_event_loop().time()
                    result['response_time'] = int((end_time - start_time) * 1000)
                    result['cached'] = False
                    
                    return result
            except Exception as e:
                logger.debug(f"Source fetch failed: {e}")
                continue
        
        # Fallback to default
        return {
            'url': self.default_poster,
            'source': 'default',
            'quality': 'low',
            'rating': '0.0',
            'cached': False,
            'response_time': int((asyncio.get_event_loop().time() - start_time) * 1000)
        }
    
    async def _fetch_from_source(self, source: str, title: str, year: Optional[str], search_query: str) -> Optional[Dict[str, Any]]:
        """Fetch poster from specific source"""
        try:
            if source == 'letterboxd':
                return await self._fetch_letterboxd(title, year)
            elif source == 'omdb':
                return await self._fetch_omdb(title, year)
            elif source == 'imdb':
                return await self._fetch_imdb(title, year)
            elif source == 'impawards':
                return await self._fetch_impawards(title, year)
            elif source == 'justwatch':
                return await self._fetch_justwatch(title)
            elif source == 'youtube':
                return await self._fetch_youtube(title, year)
        except asyncio.TimeoutError:
            logger.debug(f"Timeout fetching from {source} for {title}")
        except Exception as e:
            logger.debug(f"Error fetching from {source}: {e}")
        
        return None
    
    async def fetch_from_source(self, source: str, title: str, year: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Fetch poster from specific source only"""
        try:
            if source == 'letterboxd':
                return await self._fetch_letterboxd(title, year)
            elif source == 'omdb':
                return await self._fetch_omdb(title, year)
            elif source == 'imdb':
                return await self._fetch_imdb(title, year)
            elif source == 'impawards':
                return await self._fetch_impawards(title, year)
            elif source == 'justwatch':
                return await self._fetch_justwatch(title)
            elif source == 'youtube':
                return await self._fetch_youtube(title, year)
            elif source == 'custom':
                # Custom poster generation
                year_match = re.search(r'\b(19|20)\d{2}\b', title)
                year_found = year_match.group() if year_match else year or ""
                
                return {
                    'url': f"{self.config.BACKEND_URL}/api/poster/custom?title={quote(title)}&year={year_found}",
                    'source': 'custom',
                    'quality': 'low',
                    'rating': '0.0'
                }
        except Exception as e:
            logger.debug(f"Source {source} fetch failed: {e}")
        
        return None
    
    async def _fetch_letterboxd(self, title: str, year: Optional[str]) -> Optional[Dict[str, Any]]:
        """Fetch from Letterboxd (Primary Source)"""
        try:
            await self.init_session()
            
            # Clean title for URL
            clean_title = re.sub(r'[^\w\s-]', '', title.lower())
            clean_title = clean_title.replace(' ', '-')
            
            # Try multiple URL formats
            urls_to_try = []
            
            if year:
                urls_to_try.append(f"https://letterboxd.com/film/{clean_title}-{year}/")
            
            urls_to_try.append(f"https://letterboxd.com/film/{clean_title}/")
            
            # Also try with common suffixes
            if year:
                urls_to_try.append(f"https://letterboxd.com/film/{clean_title}-{year}-1/")
                urls_to_try.append(f"https://letterboxd.com/film/{clean_title}-the-{year}/")
            
            for url in urls_to_try:
                try:
                    async with self.session.get(url, timeout=2) as response:
                        if response.status == 200:
                            html = await response.text()
                            
                            # Try multiple meta tag patterns
                            patterns = [
                                r'<meta property="og:image" content="([^"]+)"',
                                r'<meta name="twitter:image" content="([^"]+)"',
                                r'<img[^>]*data-src="([^"]+)"[^>]*class="image"',
                                r'<img[^>]*src="([^"]+)"[^>]*loading="lazy"[^>]*class="poster"'
                            ]
                            
                            for pattern in patterns:
                                match = re.search(pattern, html, re.IGNORECASE)
                                if match:
                                    image_url = match.group(1)
                                    
                                    # Convert to higher quality if possible
                                    if '_w_' in image_url:
                                        image_url = image_url.replace('_w_', '_c_')
                                    elif '-0-' in image_url:
                                        image_url = image_url.replace('-0-', '-500-')
                                    
                                    return {
                                        'url': image_url,
                                        'source': 'letterboxd',
                                        'quality': 'high',
                                        'rating': '8.5',
                                        'width': 500,
                                        'height': 750
                                    }
                except:
                    continue
            
            return None
            
        except Exception as e:
            logger.debug(f"Letterboxd fetch error: {e}")
            return None
    
    async def _fetch_omdb(self, title: str, year: Optional[str]) -> Optional[Dict[str, Any]]:
        """Fetch from OMDB API (Secondary Source)"""
        try:
            await self.init_session()
            
            if not hasattr(self.config, 'OMDB_API_KEY') or not self.config.OMDB_API_KEY:
                return None
            
            # Try multiple API keys
            for api_key in self.config.OMDB_KEYS:
                try:
                    params = {
                        'apikey': api_key,
                        't': title,
                        'type': 'movie'
                    }
                    if year:
                        params['y'] = year
                    
                    url = "http://www.omdbapi.com/"
                    async with self.session.get(url, params=params, timeout=2) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data.get('Poster') and data['Poster'] != 'N/A':
                                rating = data.get('imdbRating', '0.0')
                                if rating == 'N/A':
                                    rating = '0.0'
                                    
                                return {
                                    'url': data['Poster'],
                                    'source': 'omdb',
                                    'quality': 'medium',
                                    'rating': rating,
                                    'imdb_id': data.get('imdbID'),
                                    'title': data.get('Title')
                                }
                except:
                    continue
        
        except:
            pass
        
        return None
    
    async def _fetch_imdb(self, title: str, year: Optional[str]) -> Optional[Dict[str, Any]]:
        """Fast scraping from IMDb"""
        try:
            await self.init_session()
            
            search_term = quote(f"{title} {year}" if year else title)
            search_url = f"https://www.imdb.com/find?q={search_term}&s=tt&ttype=ft"
            
            async with self.session.get(search_url, timeout=4) as response:
                if response.status == 200:
                    html = await response.text()
                    
                    # Find first movie result
                    result_match = re.search(r'href="/title/(tt\d+)/[^"]*"[^>]*>([^<]+)</a>', html)
                    if result_match:
                        imdb_id = result_match.group(1)
                        
                        # Fetch poster from IMDB
                        poster_url = f"https://imdb-api.com/images/original/{imdb_id}.jpg"
                        
                        # Try to get actual poster
                        async with self.session.head(poster_url, timeout=2) as head_resp:
                            if head_resp.status == 200:
                                return {
                                    'url': poster_url,
                                    'source': 'imdb',
                                    'quality': 'high',
                                    'rating': '7.0',
                                    'imdb_id': imdb_id
                                }
                        
                        # Fallback to standard IMDB image
                        return {
                            'url': f"https://img.omdbapi.com/?i={imdb_id}&apikey={self.config.OMDB_KEYS[0] if self.config.OMDB_KEYS else ''}",
                            'source': 'imdb',
                            'quality': 'medium',
                            'rating': '7.0',
                            'imdb_id': imdb_id
                        }
        except:
            pass
        
        return None
    
    async def _fetch_impawards(self, title: str, year: Optional[str]) -> Optional[Dict[str, Any]]:
        """Fetch from IMPAwards (High Quality Posters)"""
        try:
            await self.init_session()
            
            if year:
                url = f"http://www.impawards.com/{year}/{title.replace(' ', '_')}.html"
            else:
                # Try recent years
                current_year = 2024
                for y in range(current_year, current_year - 5, -1):
                    url = f"http://www.impawards.com/{y}/{title.replace(' ', '_')}.html"
                    
                    async with self.session.head(url, timeout=2) as response:
                        if response.status == 200:
                            break
                else:
                    return None
            
            async with self.session.get(url, timeout=3) as response:
                if response.status == 200:
                    html = await response.text()
                    
                    # Look for poster image
                    pattern = r'<img src="([^"]+posters/[^"]+\.jpg)"[^>]*>'
                    match = re.search(pattern, html, re.IGNORECASE)
                    if match:
                        poster_url = f"http://www.impawards.com/{match.group(1)}"
                        return {
                            'url': poster_url,
                            'source': 'impawards',
                            'quality': 'high',
                            'rating': '8.0',
                            'width': 1000,
                            'height': 1500
                        }
        except:
            pass
        
        return None
    
    async def _fetch_justwatch(self, title: str) -> Optional[Dict[str, Any]]:
        """Fetch from JustWatch (Streaming Service Posters)"""
        try:
            await self.init_session()
            
            search_term = quote(title)
            url = f"https://apis.justwatch.com/content/titles/en_IN/popular?body=%7B%22query%22:%22{search_term}%22,%22page%22:1,%22page_size%22:10%7D"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            async with self.session.get(url, headers=headers, timeout=3) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('items'):
                        item = data['items'][0]
                        if item.get('poster'):
                            poster_url = item['poster'].replace('{profile}', 's592')
                            return {
                                'url': poster_url,
                                'source': 'justwatch',
                                'quality': 'medium',
                                'rating': '7.5',
                                'title': item.get('title')
                            }
        except:
            pass
        
        return None
    
    async def _fetch_youtube(self, title: str, year: Optional[str]) -> Optional[Dict[str, Any]]:
        """Fetch YouTube Thumbnails (Fallback)"""
        try:
            await self.init_session()
            
            search_term = quote(f"{title} {year} official trailer" if year else f"{title} official trailer")
            
            # Try to get from YouTube directly
            url = f"https://www.youtube.com/results?search_query={search_term}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            async with self.session.get(url, headers=headers, timeout=3) as response:
                if response.status == 200:
                    html = await response.text()
                    
                    # Extract first video ID
                    video_id_match = re.search(r'"videoId":"([^"]+)"', html)
                    if video_id_match:
                        video_id = video_id_match.group(1)
                        return {
                            'url': f'https://img.youtube.com/vi/{video_id}/hqdefault.jpg',
                            'source': 'youtube',
                            'quality': 'low',
                            'rating': '6.0'
                        }
        except:
            pass
        
        return None
    
    async def fetch_batch_posters(self, titles: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch posters for multiple titles efficiently"""
        results = {}
        
        # First check cache for all titles
        for title in titles:
            for source_name in self.sources.keys():
                cached = await self.cache.get_poster(title, source_name)
                if cached:
                    results[title] = cached
                    break
        
        # Fetch uncached titles
        uncached_titles = [title for title in titles if title not in results]
        
        if uncached_titles:
            tasks = []
            for title in uncached_titles:
                task = self.fetch_poster(title)
                tasks.append(task)
            
            fetched_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(fetched_results):
                if not isinstance(result, Exception) and result:
                    title = uncached_titles[i]
                    results[title] = result
        
        # Ensure all titles have at least default poster
        for title in titles:
            if title not in results:
                results[title] = {
                    'url': self.default_poster,
                    'source': 'default',
                    'quality': 'low',
                    'rating': '0.0',
                    'cached': False
                }
        
        return results
    
    def clear_cache(self):
        """Clear local cache"""
        if hasattr(self.cache, 'clear_pattern'):
            asyncio.create_task(self.cache.clear_pattern("poster:"))
    
    async def cleanup_expired_cache(self):
        """Cleanup expired cache entries"""
        if hasattr(self.cache, 'cleanup_expired'):
            await self.cache.cleanup_expired()
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
