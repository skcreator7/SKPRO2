"""
poster_fetcher.py - Multi-source poster fetching with intelligent fallback
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
    
    async def _fetch_letterboxd(self, title: str, year: Optional[str]) -> Optional[Dict[str, Any]]:
        """Fetch from Letterboxd (Primary Source)"""
        try:
            await self.init_session()
            
            # Clean title for URL
            clean_title = re.sub(r'[^\w\s-]', '', title.lower())
            clean_title = clean_title.replace(' ', '-')
            
            url = f"https://letterboxd.com/film/{clean_title}/"
            if year:
                url = f"https://letterboxd.com/film/{clean_title}-{year}/"
            
            async with self.session.get(url, timeout=3) as response:
                if response.status == 200:
                    html = await response.text()
                    
                    # Extract poster from meta tags
                    og_image_match = re.search(r'<meta property="og:image" content="([^"]+)"', html)
                    if og_image_match:
                        image_url = og_image_match.group(1)
                        # Convert to higher quality if possible
                        if '_w_' in image_url:
                            image_url = image_url.replace('_w_', '_c_')
                        
                        return {
                            'url': image_url,
                            'source': 'letterboxd',
                            'quality': 'high',
                            'width': 500,
                            'height': 750
                        }
        except:
            pass
        
        return None
    
    async def _fetch_omdb(self, title: str, year: Optional[str]) -> Optional[Dict[str, Any]]:
        """Fetch from OMDB API (Secondary Source)"""
        try:
            await self.init_session()
            
            if not hasattr(self.config, 'OMDB_API_KEY') or not self.config.OMDB_API_KEY:
                return None
            
            params = {
                'apikey': self.config.OMDB_API_KEY,
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
                        return {
                            'url': data['Poster'],
                            'source': 'omdb',
                            'quality': 'medium',
                            'imdb_id': data.get('imdbID'),
                            'title': data.get('Title')
                        }
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
            url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={search_term}&maxResults=1&type=video&key={getattr(self.config, 'YOUTUBE_API_KEY', '')}"
            
            async with self.session.get(url, timeout=3) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('items'):
                        item = data['items'][0]
                        thumbnail = item['snippet']['thumbnails'].get('high', 
                                    item['snippet']['thumbnails'].get('medium',
                                    item['snippet']['thumbnails'].get('default')))
                        
                        if thumbnail:
                            return {
                                'url': thumbnail['url'],
                                'source': 'youtube',
                                'quality': 'low',
                                'width': thumbnail.get('width', 480),
                                'height': thumbnail.get('height', 360)
                            }
        except:
            # Fallback to simple YouTube thumbnail
            try:
                # Try to get from YouTube directly
                search_term = quote(title)
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
                                'quality': 'low'
                            }
            except:
                pass
        
        return None
    
    async def fetch_multiple_posters(self, titles: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch posters for multiple titles efficiently"""
        results = {}
        
        # First check cache for all titles
        cached_results = await self.cache.get_multi_posters(titles)
        
        # Separate cached and uncached titles
        uncached_titles = []
        for title in titles:
            if title in cached_results and cached_results[title]:
                results[title] = cached_results[title]
                results[title]['cached'] = True
            else:
                uncached_titles.append(title)
        
        # Fetch uncached titles in parallel
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
                    'cached': False
                }
        
        return results
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
