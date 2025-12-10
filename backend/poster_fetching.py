import asyncio
import re
import json
import base64
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from io import BytesIO
import aiohttp
import urllib.parse
import logging

logger = logging.getLogger(__name__)

class PosterSource(Enum):
    LETTERBOXD = "letterboxd"
    IMDB = "imdb"
    JUSTWATCH = "justwatch"
    IMPAWARDS = "impawards"
    OMDB = "omdb"
    TMDB = "tmdb"
    CUSTOM = "custom"
    LOCAL = "local"

class PosterFetcher:
    def __init__(self, config, cache_manager=None):
        self.config = config
        self.cache_manager = cache_manager
        
        # API Keys
        self.omdb_keys = getattr(config, 'OMDB_KEYS', ["8265bd1c", "b9bd48a6", "3e7e1e9d"])
        self.tmdb_keys = getattr(config, 'TMDB_KEYS', ["e547e17d4e91f3e62a571655cd1ccaff", "8265bd1f"])
        
        # Statistics
        self.stats = {
            'letterboxd': 0, 'imdb': 0, 'justwatch': 0, 'impawards': 0,
            'omdb': 0, 'tmdb': 0, 'custom': 0, 'cache_hits': 0
        }
        
        # Cache with TTL (1 hour)
        self.poster_cache = {}
        self.cache_ttl = timedelta(hours=1)
        
    def clean_title(self, title: str) -> str:
        """Clean movie title for searching"""
        if not title:
            return ""
        
        # Remove year, quality, and technical terms
        cleaned = re.sub(
            r'\b(19|20)\d{2}\b|\b(480p|720p|1080p|2160p|4k|hd|fhd|uhd|hevc|x264|x265|h264|h265|'
            r'bluray|webrip|hdrip|web-dl|hdtv|hdrip|webdl|hindi|english|tamil|telugu|'
            r'malayalam|kannada|punjabi|bengali|marathi|gujarati|movie|film|series|'
            r'complete|full|part|episode|season|hdrc|dvdscr|pre-dvd|p-dvd|pdc|'
            r'rarbg|yts|amzn|netflix|hotstar|prime|disney|hc-esub|esub|subs)\b',
            '', 
            title.lower(), 
            flags=re.IGNORECASE
        )
        
        # Clean up special characters
        cleaned = re.sub(r'[\._\-]', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    async def fetch_from_letterboxd(self, title: str, session: aiohttp.ClientSession) -> Optional[Dict[str, Any]]:
        """Fetch poster from Letterboxd"""
        try:
            clean_title = re.sub(r'[^\w\s]', '', title).strip()
            slug = clean_title.lower().replace(' ', '-')
            slug = re.sub(r'-+', '-', slug)
            
            patterns = [
                f"https://letterboxd.com/film/{slug}/",
                f"https://letterboxd.com/film/{slug}-2024/",
                f"https://letterboxd.com/film/{slug}-2023/",
            ]
            
            for url in patterns:
                try:
                    async with session.get(url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'}) as r:
                        if r.status == 200:
                            html_content = await r.text()
                            
                            # Extract poster URL
                            poster_patterns = [
                                r'<meta property="og:image" content="([^"]+)"',
                                r'<img[^>]*class="[^"]*poster[^"]*"[^>]*src="([^"]+)"',
                            ]
                            
                            for pattern in poster_patterns:
                                poster_match = re.search(pattern, html_content)
                                if poster_match:
                                    poster_url = poster_match.group(1)
                                    if poster_url and poster_url.startswith('http'):
                                        # Enhance quality
                                        if 'cloudfront.net' in poster_url:
                                            poster_url = poster_url.replace('-0-500-0-750', '-0-1000-0-1500')
                                        elif 's.ltrbxd.com' in poster_url:
                                            poster_url = poster_url.replace('/width/500/', '/width/1000/')
                                        
                                        # Extract rating
                                        rating_match = re.search(
                                            r'<meta name="twitter:data2" content="([^"]+)"', 
                                            html_content
                                        )
                                        rating = rating_match.group(1) if rating_match else '0.0'
                                        
                                        # Extract year if available
                                        year_match = re.search(
                                            r'<meta property="og:title" content="[^"]+\((\d{4})\)"',
                                            html_content
                                        )
                                        year = year_match.group(1) if year_match else ''
                                        
                                        result = {
                                            'poster_url': poster_url,
                                            'source': PosterSource.LETTERBOXD.value,
                                            'rating': rating,
                                            'year': year,
                                            'title': clean_title
                                        }
                                        
                                        self.stats['letterboxd'] += 1
                                        return result
                except:
                    continue
            return None
            
        except Exception as e:
            logger.error(f"Letterboxd fetch error: {e}")
            return None
    
    async def fetch_from_imdb(self, title: str, session: aiohttp.ClientSession) -> Optional[Dict[str, Any]]:
        """Fetch poster from IMDb"""
        try:
            clean_title = re.sub(r'[^\w\s]', '', title).strip()
            search_url = f"https://v2.sg.media-imdb.com/suggestion/{clean_title[0].lower()}/" \
                        f"{urllib.parse.quote(clean_title.replace(' ', '_'))}.json"
            
            async with session.get(search_url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'}) as r:
                if r.status == 200:
                    data = await r.json()
                    if data.get('d'):
                        for item in data['d']:
                            if item.get('i'):
                                poster_url = item['i'][0] if isinstance(item['i'], list) else item['i']
                                if poster_url and poster_url.startswith('http'):
                                    # Enhance quality
                                    poster_url = poster_url.replace('._V1_UX128_', '._V1_UX512_')
                                    
                                    result = {
                                        'poster_url': poster_url,
                                        'source': PosterSource.IMDB.value,
                                        'rating': str(item.get('yr', '0.0')),
                                        'year': str(item.get('yr', '')),
                                        'title': item.get('l', clean_title),
                                        'imdb_id': item.get('id')
                                    }
                                    
                                    self.stats['imdb'] += 1
                                    return result
            return None
            
        except Exception as e:
            logger.error(f"IMDb fetch error: {e}")
            return None
    
    async def fetch_from_justwatch(self, title: str, session: aiohttp.ClientSession) -> Optional[Dict[str, Any]]:
        """Fetch poster from JustWatch"""
        try:
            clean_title = re.sub(r'[^\w\s]', '', title).strip()
            slug = clean_title.lower().replace(' ', '-')
            slug = re.sub(r'[^\w\-]', '', slug)
            
            domains = ['com', 'in', 'uk']
            
            for domain in domains:
                justwatch_url = f"https://www.justwatch.com/{domain}/movie/{slug}"
                try:
                    async with session.get(justwatch_url, timeout=5, 
                                         headers={'User-Agent': 'Mozilla/5.0'}) as r:
                        if r.status == 200:
                            html_content = await r.text()
                            poster_match = re.search(
                                r'<meta property="og:image" content="([^"]+)"', 
                                html_content
                            )
                            if poster_match:
                                poster_url = poster_match.group(1)
                                if poster_url and poster_url.startswith('http'):
                                    poster_url = poster_url.replace('http://', 'https://')
                                    
                                    # Extract year from title
                                    year_match = re.search(r'\b(19|20)\d{2}\b', title)
                                    year = year_match.group() if year_match else ''
                                    
                                    result = {
                                        'poster_url': poster_url,
                                        'source': PosterSource.JUSTWATCH.value,
                                        'rating': '0.0',
                                        'year': year,
                                        'title': clean_title
                                    }
                                    
                                    self.stats['justwatch'] += 1
                                    return result
                except:
                    continue
            return None
            
        except Exception as e:
            logger.error(f"JustWatch fetch error: {e}")
            return None
    
    async def fetch_from_impawards(self, title: str, session: aiohttp.ClientSession) -> Optional[Dict[str, Any]]:
        """Fetch poster from IMPAwards"""
        try:
            year_match = re.search(r'\b(19|20)\d{2}\b', title)
            if not year_match:
                return None
                
            year = year_match.group()
            clean_title = re.sub(r'\b(19|20)\d{2}\b', '', title).strip()
            clean_title = re.sub(r'[^\w\s]', '', clean_title).strip()
            slug = clean_title.lower().replace(' ', '_')
            
            formats = [
                f"https://www.impawards.com/{year}/posters/{slug}_xlg.jpg",
                f"https://www.impawards.com/{year}/posters/{slug}_ver7.jpg",
                f"https://www.impawards.com/{year}/posters/{slug}.jpg",
            ]
            
            for poster_url in formats:
                try:
                    async with session.head(poster_url, timeout=3) as r:
                        if r.status == 200:
                            result = {
                                'poster_url': poster_url,
                                'source': PosterSource.IMPAWARDS.value,
                                'rating': '0.0',
                                'year': year,
                                'title': clean_title
                            }
                            
                            self.stats['impawards'] += 1
                            return result
                except:
                    continue
            return None
            
        except Exception as e:
            logger.error(f"IMPAwards fetch error: {e}")
            return None
    
    async def fetch_from_omdb_tmdb(self, title: str, session: aiohttp.ClientSession) -> Optional[Dict[str, Any]]:
        """Fetch poster from OMDB/TMDB"""
        try:
            # Try OMDB first
            for api_key in self.omdb_keys:
                try:
                    url = f"http://www.omdbapi.com/?t={urllib.parse.quote(title)}&apikey={api_key}"
                    async with session.get(url, timeout=5) as r:
                        if r.status == 200:
                            data = await r.json()
                            if data.get('Response') == 'True' and data.get('Poster') and data.get('Poster') != 'N/A':
                                poster_url = data['Poster'].replace('http://', 'https://')
                                
                                result = {
                                    'poster_url': poster_url,
                                    'source': PosterSource.OMDB.value,
                                    'rating': data.get('imdbRating', '0.0'),
                                    'year': data.get('Year', ''),
                                    'title': data.get('Title', title),
                                    'plot': data.get('Plot', ''),
                                    'genre': data.get('Genre', ''),
                                    'director': data.get('Director', ''),
                                    'actors': data.get('Actors', ''),
                                    'imdb_id': data.get('imdbID', '')
                                }
                                
                                self.stats['omdb'] += 1
                                return result
                except:
                    continue
            
            # Try TMDB
            for api_key in self.tmdb_keys:
                try:
                    url = "https://api.themoviedb.org/3/search/movie"
                    params = {'api_key': api_key, 'query': title}
                    async with session.get(url, params=params, timeout=5) as r:
                        if r.status == 200:
                            data = await r.json()
                            if data.get('results') and len(data['results']) > 0:
                                result = data['results'][0]
                                poster_path = result.get('poster_path')
                                if poster_path:
                                    poster_url = f"https://image.tmdb.org/t/p/w780{poster_path}"
                                    
                                    poster_data = {
                                        'poster_url': poster_url,
                                        'source': PosterSource.TMDB.value,
                                        'rating': str(result.get('vote_average', 0.0)),
                                        'year': result.get('release_date', '')[:4] if result.get('release_date') else '',
                                        'title': result.get('title', title),
                                        'overview': result.get('overview', ''),
                                        'tmdb_id': result.get('id')
                                    }
                                    
                                    self.stats['tmdb'] += 1
                                    return poster_data
                except:
                    continue
            return None
            
        except Exception as e:
            logger.error(f"OMDB/TMDB fetch error: {e}")
            return None
    
    async def create_custom_poster(self, title: str, year: str = '') -> Dict[str, Any]:
        """Create fallback poster using provided image"""
        try:
            # Use the provided image URL as fallback
            result = {
                'poster_url': "https://iili.io/fAeIwv9.th.png",  # Your fallback image
                'source': PosterSource.CUSTOM.value,
                'rating': '0.0',
                'year': year,
                'title': title,
                'is_svg': False  # This is not an SVG anymore
            }
            
            self.stats['custom'] += 1
            return result
            
        except Exception as e:
            logger.error(f"Custom poster creation error: {e}")
            # Ultimate fallback in case of error
            return {
                'poster_url': f"{self.config.BACKEND_URL}/api/poster?title={urllib.parse.quote(title)}&year={year}",
                'source': PosterSource.CUSTOM.value,
                'rating': '0.0',
                'year': year,
                'title': title
            }
    
    async def fetch_poster(self, title: str, use_cache: bool = True) -> Dict[str, Any]:
        """Fetch poster for movie title from multiple sources"""
        cache_key = title.lower().strip()
        
        # Check cache first
        if use_cache and cache_key in self.poster_cache:
            data, timestamp = self.poster_cache[cache_key]
            if (datetime.now() - timestamp) < self.cache_ttl:
                self.stats['cache_hits'] += 1
                return data
            else:
                # Remove expired cache
                del self.poster_cache[cache_key]
        
        async with aiohttp.ClientSession() as session:
            # Try all sources concurrently
            sources = [
                self.fetch_from_letterboxd(title, session),
                self.fetch_from_imdb(title, session),
                self.fetch_from_justwatch(title, session),
                self.fetch_from_impawards(title, session),
                self.fetch_from_omdb_tmdb(title, session),
            ]
            
            results = await asyncio.gather(*sources, return_exceptions=True)
            
            # Find first successful result
            for result in results:
                if isinstance(result, dict) and result.get('poster_url'):
                    # Cache the result
                    self.poster_cache[cache_key] = (result, datetime.now())
                    return result
        
        # Fallback to custom poster
        year_match = re.search(r'\b(19|20)\d{2}\b', title)
        year = year_match.group() if year_match else ''
        
        custom_poster = await self.create_custom_poster(title, year)
        self.poster_cache[cache_key] = (custom_poster, datetime.now())
        
        return custom_poster
    
    async def fetch_poster_data(self, title: str) -> Optional[bytes]:
        """
        Fetch poster image data as bytes
        Returns: Image bytes or None if failed
        """
        try:
            poster_info = await self.fetch_poster(title)
            poster_url = poster_info.get('poster_url')
            
            if not poster_url:
                return None
            
            # Skip downloading if it's the fallback image or data URL
            if poster_url == "https://iili.io/fAeIwv9.th.png" or poster_info.get('is_svg') or poster_url.startswith('data:'):
                return None
            
            async with aiohttp.ClientSession() as session:
                async with session.get(poster_url, timeout=10) as response:
                    if response.status == 200:
                        return await response.read()
            
        except Exception as e:
            logger.error(f"Error fetching poster data for {title}: {e}")
        
        return None
    
    def get_stats(self) -> Dict[str, int]:
        """Get current statistics"""
        return self.stats.copy()
    
    def clear_cache(self):
        """Clear the poster cache"""
        self.poster_cache.clear()
    
    def get_cache_size(self) -> int:
        """Get current cache size"""
        return len(self.poster_cache)


# Example usage
async def main():
    class Config:
        OMDB_KEYS = ["8265bd1c", "b9bd48a6", "3e7e1e9d"]
        TMDB_KEYS = ["e547e17d4e91f3e62a571655cd1ccaff"]
        BACKEND_URL = "http://localhost:8000"
    
    config = Config()
    fetcher = PosterFetcher(config)
    
    # Example: Fetch poster for a movie
    movies = ["Inception 2010", "The Dark Knight 2008", "Interstellar 2014"]
    
    for movie in movies:
        poster = await fetcher.fetch_poster(movie)
        print(f"{movie}: {poster.get('source')} - {poster.get('poster_url')[:50]}...")
    
    # Test fallback for unknown movie
    unknown_movie = "Unknown Movie XYZ 2025"
    fallback_poster = await fetcher.fetch_poster(unknown_movie)
    print(f"\n{unknown_movie}: {fallback_poster.get('source')} - {fallback_poster.get('poster_url')}")
    
    # Print stats
    print("\nStats:", fetcher.get_stats())

if __name__ == "__main__":
    asyncio.run(main())
