# ============================================================================
# 🚀 ULTRA-FAST POSTER FETCHER - NO BLINKING FIX
# ============================================================================

import asyncio
import aiohttp
import hashlib
import urllib.parse
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import os

# ============================================================================
# ✅ CONFIGURATION
# ============================================================================

class PosterConfig:
    # API Keys - Environment variables से लें
    TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "e547e17d4e91f3e62a571655cd1ccaff")
    OMDB_API_KEY = os.environ.get("OMDB_API_KEY", "8265bd1c")
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
    GOOGLE_CX = os.environ.get("GOOGLE_CX", "")
    
    # Timeouts (Fast response के लिए)
    REQUEST_TIMEOUT = 2.0  # 2 seconds max per source
    TOTAL_TIMEOUT = 4.0   # 4 seconds total
    
    # Cache TTL
    CACHE_TTL = 24 * 60 * 60  # 24 hours
    
    # Fallback poster
    FALLBACK_POSTER = "https://iili.io/fAeIwv9.th.png"
    
    # Placeholder colors for different quality
    QUALITY_COLORS = {
        '2160p': '4a148c',  # Purple
        '1080p': '1565c0',  # Blue
        '720p': '0277bd',   # Light Blue
        '480p': '00838f',   # Teal
        '360p': '00695c',   # Dark Teal
        'default': '1a1a2e' # Dark Blue
    }

# ============================================================================
# ✅ POSTER SOURCES - PRIORITY ORDER
# ============================================================================

POSTER_SOURCES = [
    {
        'name': 'TMDB',
        'priority': 1,
        'enabled': bool(PosterConfig.TMDB_API_KEY),
        'function': '_fetch_tmdb_poster'
    },
    {
        'name': 'OMDB',
        'priority': 2,
        'enabled': bool(PosterConfig.OMDB_API_KEY),
        'function': '_fetch_omdb_poster'
    },
    {
        'name': 'IMDB_SCRAPE',
        'priority': 3,
        'enabled': True,
        'function': '_fetch_imdb_scrape'
    },
    {
        'name': 'IMPAWARDS',
        'priority': 4,
        'enabled': True,
        'function': '_fetch_impawards_poster'
    },
    {
        'name': 'JUSTWATCH',
        'priority': 5,
        'enabled': True,
        'function': '_fetch_justwatch_poster'
    },
    {
        'name': 'GOOGLE_IMAGES',
        'priority': 6,
        'enabled': bool(PosterConfig.GOOGLE_API_KEY and PosterConfig.GOOGLE_CX),
        'function': '_fetch_google_poster'
    },
    {
        'name': 'PLACEHOLDER',
        'priority': 7,
        'enabled': True,
        'function': '_create_placeholder_poster'
    }
]

# ============================================================================
# ✅ ULTRA-FAST POSTER FETCHER CLASS
# ============================================================================

class UltraFastPosterFetcher:
    def __init__(self, cache_manager=None):
        self.cache_manager = cache_manager
        self.http_session = None
        self.memory_cache = {}
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'successful_fetches': 0,
            'failed_fetches': 0
        }
    
    async def init_http_session(self):
        """Initialize HTTP session with optimized settings"""
        if self.http_session is None:
            connector = aiohttp.TCPConnector(
                limit=20,
                limit_per_host=5,
                ttl_dns_cache=300,
                force_close=False
            )
            
            timeout = aiohttp.ClientTimeout(
                total=PosterConfig.TOTAL_TIMEOUT,
                connect=1.5,
                sock_read=1.5
            )
            
            self.http_session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'application/json,text/html,image/*,*/*',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive'
                }
            )
    
    # ============================================================================
    # ✅ MAIN POSTER FETCH FUNCTION - NO BLINKING GUARANTEED
    # ============================================================================
    
    async def fetch_poster(self, title: str, year: str = "", quality: str = "") -> Dict[str, Any]:
        """
        Fetch poster with NO BLINKING - Always returns valid poster URL
        """
        # Generate cache key
        cache_key = f"poster:{hashlib.md5(f'{title}_{year}'.encode()).hexdigest()}"
        
        # Check memory cache first (FASTEST)
        now = time.time()
        if cache_key in self.memory_cache:
            data, timestamp = self.memory_cache[cache_key]
            if now - timestamp < PosterConfig.CACHE_TTL:
                self.stats['cache_hits'] += 1
                return self._ensure_poster_data(data, title, year, quality)
        
        # Check Redis cache if available
        if self.cache_manager and hasattr(self.cache_manager, 'get'):
            try:
                cached = await self.cache_manager.get(cache_key)
                if cached:
                    self.memory_cache[cache_key] = (cached, now)
                    self.stats['cache_hits'] += 1
                    return self._ensure_poster_data(cached, title, year, quality)
            except:
                pass
        
        self.stats['total_requests'] += 1
        
        # Step 1: Try FAST sources first (TMDB, OMDB)
        fast_sources = [s for s in POSTER_SOURCES if s['priority'] <= 3 and s['enabled']]
        poster_data = await self._try_sources_concurrently(fast_sources, title, year)
        
        # Step 2: If fast sources failed, try slower sources
        if not poster_data:
            slow_sources = [s for s in POSTER_SOURCES if s['priority'] > 3 and s['enabled']]
            poster_data = await self._try_sources_concurrently(slow_sources, title, year)
        
        # Step 3: If still no data, create placeholder
        if not poster_data:
            poster_data = await self._create_placeholder_poster(title, year, quality)
        
        # Ensure all required fields
        poster_data = self._ensure_poster_data(poster_data, title, year, quality)
        
        # Cache the result
        self.memory_cache[cache_key] = (poster_data, now)
        if self.cache_manager and hasattr(self.cache_manager, 'set'):
            try:
                await self.cache_manager.set(cache_key, poster_data, expire_seconds=PosterConfig.CACHE_TTL)
            except:
                pass
        
        self.stats['successful_fetches'] += 1
        return poster_data
    
    # ============================================================================
    # ✅ CONCURRENT SOURCE TRYING
    # ============================================================================
    
    async def _try_sources_concurrently(self, sources: List, title: str, year: str) -> Optional[Dict]:
        """Try multiple sources concurrently, return first successful"""
        if not sources:
            return None
        
        tasks = []
        for source in sources:
            func = getattr(self, source['function'], None)
            if func:
                task = asyncio.create_task(func(title, year))
                tasks.append((source['name'], task))
        
        if not tasks:
            return None
        
        # Wait for first successful result
        done, pending = await asyncio.wait(
            [task for _, task in tasks],
            return_when=asyncio.FIRST_COMPLETED,
            timeout=PosterConfig.REQUEST_TIMEOUT
        )
        
        # Cancel pending tasks
        for _, task in tasks:
            if not task.done():
                task.cancel()
        
        # Check completed tasks
        for task_name, task in tasks:
            if task in done:
                try:
                    result = await task
                    if result and result.get('url'):
                        result['source'] = task_name
                        return result
                except:
                    continue
        
        return None
    
    # ============================================================================
    # ✅ POSTER SOURCE FUNCTIONS
    # ============================================================================
    
    async def _fetch_tmdb_poster(self, title: str, year: str = "") -> Optional[Dict]:
        """Fetch from TMDB API"""
        try:
            await self.init_http_session()
            
            query = urllib.parse.quote(title)
            url = f"https://api.themoviedb.org/3/search/movie?api_key={PosterConfig.TMDB_API_KEY}&query={query}&language=en-US&page=1"
            
            if year:
                url += f"&year={year}"
            
            async with self.http_session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('results') and len(data['results']) > 0:
                        movie = data['results'][0]
                        
                        if movie.get('poster_path'):
                            poster_url = f"https://image.tmdb.org/t/p/w500{movie['poster_path']}"
                            
                            return {
                                'url': poster_url,
                                'source': 'tmdb',
                                'rating': str(movie.get('vote_average', '0.0')),
                                'year': str(movie.get('release_date', ''))[:4] if movie.get('release_date') else year,
                                'title': movie.get('title', title),
                                'backdrop_url': f"https://image.tmdb.org/t/p/original{movie['backdrop_path']}" if movie.get('backdrop_path') else None,
                                'quality': 'tmdb_hd'
                            }
        except:
            pass
        
        return None
    
    async def _fetch_omdb_poster(self, title: str, year: str = "") -> Optional[Dict]:
        """Fetch from OMDB API"""
        try:
            await self.init_http_session()
            
            query = urllib.parse.quote(title)
            url = f"http://www.omdbapi.com/?apikey={PosterConfig.OMDB_API_KEY}&t={query}"
            
            if year:
                url += f"&y={year}"
            
            url += "&plot=short"
            
            async with self.http_session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('Poster') and data['Poster'] != 'N/A':
                        return {
                            'url': data['Poster'],
                            'source': 'omdb',
                            'rating': data.get('imdbRating', '0.0'),
                            'year': data.get('Year', year),
                            'title': data.get('Title', title),
                            'plot': data.get('Plot', '')[:200] if data.get('Plot') else '',
                            'quality': 'omdb_original'
                        }
        except:
            pass
        
        return None
    
    async def _fetch_imdb_scrape(self, title: str, year: str = "") -> Optional[Dict]:
        """Fast IMDb scraping"""
        try:
            await self.init_http_session()
            
            # Clean title for search
            clean_title = re.sub(r'[^\w\s]', '', title.lower())
            search_query = urllib.parse.quote(f"{clean_title} {year} movie poster")
            
            # Try multiple image search endpoints
            endpoints = [
                f"https://www.imdb.com/find?q={search_query}&s=tt&ttype=ft",
                f"https://www.google.com/search?tbm=isch&q={search_query}+movie+poster",
                f"https://www.bing.com/images/search?q={search_query}+movie+poster"
            ]
            
            for endpoint in endpoints:
                try:
                    async with self.http_session.get(endpoint, allow_redirects=True) as response:
                        if response.status == 200:
                            html = await response.text()
                            
                            # Try to extract image URLs
                            import re
                            img_patterns = [
                                r'<img[^>]*src="([^"]*\.(?:jpg|jpeg|png|webp))[^"]*"[^>]*>',
                                r'src="(https://[^"]*\.(?:jpg|jpeg|png|webp))"',
                                r'data-src="(https://[^"]*\.(?:jpg|jpeg|png|webp))"'
                            ]
                            
                            for pattern in img_patterns:
                                matches = re.findall(pattern, html, re.IGNORECASE)
                                if matches:
                                    # Filter for likely poster images
                                    poster_keywords = ['poster', 'movie', 'film', 'cover']
                                    for img_url in matches:
                                        img_lower = img_url.lower()
                                        if any(kw in img_lower for kw in poster_keywords):
                                            if 'http' in img_url and 'google' not in img_url:
                                                return {
                                                    'url': img_url,
                                                    'source': 'imdb_scrape',
                                                    'rating': '0.0',
                                                    'year': year,
                                                    'title': title,
                                                    'quality': 'scraped'
                                                }
                except:
                    continue
        except:
            pass
        
        return None
    
    async def _fetch_impawards_poster(self, title: str, year: str = "") -> Optional[Dict]:
        """Fetch from IMPAwards (high quality posters)"""
        try:
            await self.init_http_session()
            
            # Clean title for IMPAwards URL
            clean_title = title.lower().replace(' ', '_').replace('-', '_')
            clean_title = re.sub(r'[^\w_]', '', clean_title)
            
            # Try with year
            if year:
                urls_to_try = [
                    f"https://www.impawards.com/{year}/posters/{clean_title}_ver{str(i)}.jpg"
                    for i in range(1, 4)
                ]
                
                urls_to_try.append(f"https://www.impawards.com/{year}/posters/{clean_title}.jpg")
            
            # Try without year
            urls_to_try.append(f"https://www.impawards.com/posters/{clean_title}.jpg")
            
            for url in urls_to_try:
                try:
                    async with self.http_session.head(url, allow_redirects=True) as response:
                        if response.status == 200:
                            # Check if it's actually an image
                            content_type = response.headers.get('Content-Type', '')
                            if 'image' in content_type:
                                return {
                                    'url': url,
                                    'source': 'impawards',
                                    'rating': '0.0',
                                    'year': year,
                                    'title': title,
                                    'quality': 'high_quality'
                                }
                except:
                    continue
        except:
            pass
        
        return None
    
    async def _fetch_justwatch_poster(self, title: str, year: str = "") -> Optional[Dict]:
        """Fetch from JustWatch streaming service"""
        try:
            await self.init_http_session()
            
            # JustWatch uses numeric IDs, we'll search
            query = urllib.parse.quote(title)
            search_url = f"https://apis.justwatch.com/content/titles/en_IN/popular?body={{\"query\":\"{query}\"}}"
            
            async with self.http_session.get(search_url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('items') and len(data['items']) > 0:
                        item = data['items'][0]
                        
                        # Try to get poster from offers
                        if item.get('offers'):
                            for offer in item['offers']:
                                if offer.get('urls'):
                                    for url_type, url in offer['urls'].items():
                                        if 'standard_web' in url_type:
                                            return {
                                                'url': url,
                                                'source': 'justwatch',
                                                'rating': '0.0',
                                                'year': str(item.get('original_release_year', year)),
                                                'title': item.get('title', title),
                                                'quality': 'streaming'
                                            }
        except:
            pass
        
        return None
    
    async def _fetch_google_poster(self, title: str, year: str = "") -> Optional[Dict]:
        """Fetch from Google Custom Search"""
        try:
            if not PosterConfig.GOOGLE_API_KEY or not PosterConfig.GOOGLE_CX:
                return None
            
            await self.init_http_session()
            
            query = urllib.parse.quote(f"{title} {year} movie poster official")
            url = f"https://www.googleapis.com/customsearch/v1?key={PosterConfig.GOOGLE_API_KEY}&cx={PosterConfig.GOOGLE_CX}&q={query}&searchType=image&num=3&safe=active"
            
            async with self.http_session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('items') and len(data['items']) > 0:
                        # Filter for movie posters
                        for item in data['items']:
                            img_url = item.get('link', '')
                            if img_url and any(ext in img_url.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                                return {
                                    'url': img_url,
                                    'source': 'google',
                                    'rating': '0.0',
                                    'year': year,
                                    'title': title,
                                    'quality': 'google_search'
                                }
        except:
            pass
        
        return None
    
    async def _create_placeholder_poster(self, title: str, year: str = "", quality: str = "") -> Dict:
        """Create placeholder poster as last resort - NO BLINKING"""
        # Clean title
        clean_title = title[:40]
        
        # Get color based on quality
        if quality:
            base_quality = quality.split()[0] if ' ' in quality else quality
            color = PosterConfig.QUALITY_COLORS.get(base_quality, PosterConfig.QUALITY_COLORS['default'])
        else:
            # Generate color from title hash
            title_hash = hashlib.md5(title.encode()).hexdigest()
            color = title_hash[:6]
        
        # URL encode
        encoded_title = urllib.parse.quote(clean_title)
        
        # Create placeholder URL
        if year:
            poster_url = f"https://via.placeholder.com/300x450/{color}/ffffff?text={encoded_title}&subtext={year}"
        else:
            poster_url = f"https://via.placeholder.com/300x450/{color}/ffffff?text={encoded_title}"
        
        # Add movie icon
        poster_url += "&logo=https://img.icons8.com/color/96/000000/movie.png"
        
        return {
            'url': poster_url,
            'source': 'placeholder',
            'rating': '0.0',
            'year': year,
            'title': title,
            'quality': 'placeholder'
        }
    
    # ============================================================================
    # ✅ UTILITY FUNCTIONS
    # ============================================================================
    
    def _ensure_poster_data(self, data: Dict, title: str, year: str, quality: str) -> Dict:
        """Ensure poster data has all required fields"""
        if not data:
            data = {}
        
        # Ensure URL is present
        if 'url' not in data or not data['url']:
            data['url'] = PosterConfig.FALLBACK_POSTER
        
        # Ensure required fields
        required_fields = {
            'source': data.get('source', 'fallback'),
            'rating': data.get('rating', '0.0'),
            'year': data.get('year', year),
            'title': data.get('title', title),
            'quality': data.get('quality', quality)
        }
        
        data.update(required_fields)
        return data
    
    async def fetch_multiple_posters(self, movies: List[Dict]) -> List[Dict]:
        """Fetch posters for multiple movies concurrently"""
        tasks = []
        for movie in movies:
            title = movie.get('title', '')
            year = movie.get('year', '')
            quality = movie.get('quality', '')
            
            task = asyncio.create_task(self.fetch_poster(title, year, quality))
            tasks.append((movie, task))
        
        results = []
        for movie, task in tasks:
            try:
                poster_data = await task
                movie['poster_data'] = poster_data
                movie['poster_url'] = poster_data['url']
                movie['poster_source'] = poster_data['source']
                movie['poster_rating'] = poster_data['rating']
                results.append(movie)
            except:
                # Fallback for failed fetches
                movie['poster_url'] = PosterConfig.FALLBACK_POSTER
                movie['poster_source'] = 'fallback'
                movie['poster_rating'] = '0.0'
                results.append(movie)
        
        return results
    
    def clear_cache(self):
        """Clear memory cache"""
        self.memory_cache.clear()
    
    def get_stats(self) -> Dict:
        """Get fetcher statistics"""
        return {
            **self.stats,
            'cache_size': len(self.memory_cache),
            'cache_hit_rate': self.stats['cache_hits'] / self.stats['total_requests'] if self.stats['total_requests'] > 0 else 0
        }
    
    async def close(self):
        """Cleanup resources"""
        if self.http_session is not None:
            await self.http_session.close()
            self.http_session = None

# ============================================================================
# ✅ SIMPLE USAGE EXAMPLE
# ============================================================================

async def example_usage():
    # Initialize fetcher
    fetcher = UltraFastPosterFetcher()
    await fetcher.init_http_session()
    
    try:
        # Fetch single poster
        poster = await fetcher.fetch_poster("Avengers Endgame", "2019", "1080p")
        print(f"Poster URL: {poster['url']}")
        print(f"Source: {poster['source']}")
        print(f"Rating: {poster['rating']}")
        
        # Fetch multiple posters
        movies = [
            {"title": "Inception", "year": "2010", "quality": "1080p"},
            {"title": "The Dark Knight", "year": "2008", "quality": "2160p"},
            {"title": "Interstellar", "year": "2014", "quality": "720p"}
        ]
        
        results = await fetcher.fetch_multiple_posters(movies)
        for movie in results:
            print(f"{movie['title']} - {movie['poster_url'][:50]}...")
        
        # Print stats
        print(f"\nStats: {fetcher.get_stats()}")
        
    finally:
        await fetcher.close()

# ============================================================================
# ✅ FAST INITIALIZATION
# ============================================================================

# Global instance for easy access
_poster_fetcher_instance = None

async def get_poster_fetcher(cache_manager=None) -> UltraFastPosterFetcher:
    """Get or create global poster fetcher instance"""
    global _poster_fetcher_instance
    
    if _poster_fetcher_instance is None:
        _poster_fetcher_instance = UltraFastPosterFetcher(cache_manager)
        await _poster_fetcher_instance.init_http_session()
    
    return _poster_fetcher_instance

async def fetch_poster_quick(title: str, year: str = "", quality: str = "") -> Dict:
    """Quick function to fetch poster"""
    fetcher = await get_poster_fetcher()
    return await fetcher.fetch_poster(title, year, quality)

# Run example
if __name__ == "__main__":
    asyncio.run(example_usage())
