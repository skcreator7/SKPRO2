# poster_fetching.py - Movie Poster Fetching System

import logging
import re
import asyncio
import aiohttp
import urllib.parse
from datetime import datetime
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class PosterManager:
    """Movie Poster Fetching & Caching System"""
    
    def __init__(self, config, movie_db):
        self.config = config
        self.movie_db = movie_db
        self.omdb_key_index = 0
        self.tmdb_key_index = 0
    
    def get_next_omdb_key(self):
        """Rotate through OMDB API keys"""
        key = self.config.OMDB_KEYS[self.omdb_key_index]
        self.omdb_key_index = (self.omdb_key_index + 1) % len(self.config.OMDB_KEYS)
        return key
    
    def get_next_tmdb_key(self):
        """Rotate through TMDB API keys"""
        key = self.config.TMDB_KEYS[self.tmdb_key_index]
        self.tmdb_key_index = (self.tmdb_key_index + 1) % len(self.config.TMDB_KEYS)
        return key
    
    async def get_poster_letterboxd(self, title: str, session: aiohttp.ClientSession) -> Optional[Dict]:
        """Fetch poster from Letterboxd"""
        try:
            clean_title = re.sub(r'[^\w\s]', '', title).strip()
            slug = clean_title.lower().replace(' ', '-')
            slug = re.sub(r'-+', '-', slug)
            
            patterns = [
                f"https://letterboxd.com/film/{slug}/",
                f"https://letterboxd.com/film/{slug}-2024/",
                f"https://letterboxd.com/film/{slug}-2023/",
                f"https://letterboxd.com/film/{slug}-2025/",
            ]
            
            for url in patterns:
                try:
                    async with session.get(url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'}) as r:
                        if r.status == 200:
                            html_content = await r.text()
                            
                            poster_patterns = [
                                r'<img[^>]*class="[^"]*poster[^"]*"[^>]*src="([^"]+)"',
                                r'<div[^>]*class="[^"]*poster[^"]*"[^>]*data-src="([^"]+)"',
                            ]
                            
                            for pattern in poster_patterns:
                                poster_match = re.search(pattern, html_content)
                                if poster_match:
                                    poster_url = poster_match.group(1)
                                    
                                    if poster_url and poster_url.startswith('http'):
                                        # Upgrade to higher quality
                                        if 'cloudfront.net' in poster_url:
                                            poster_url = poster_url.replace('-0-500-0-750', '-0-1000-0-1500')
                                        elif 's.ltrbxd.com' in poster_url:
                                            poster_url = poster_url.replace('/width/500/', '/width/1000/')
                                        
                                        # Try to get rating
                                        rating_match = re.search(r'<span class="average-rating"[^>]*>(\d+\.\d+)</span>', html_content)
                                        rating = rating_match.group(1) if rating_match else '0.0'
                                        
                                        self.movie_db['stats']['letterboxd'] += 1
                                        return {
                                            'poster_url': poster_url,
                                            'source': 'Letterboxd',
                                            'rating': rating
                                        }
                except asyncio.TimeoutError:
                    continue
                except:
                    continue
            
            return None
        
        except Exception as e:
            logger.error(f"Letterboxd poster error: {e}")
            return None
    
    async def get_poster_imdb(self, title: str, session: aiohttp.ClientSession) -> Optional[Dict]:
        """Fetch poster from IMDb"""
        try:
            search_url = f"https://www.imdb.com/find?q={urllib.parse.quote(title)}&s=tt&ttype=ft"
            
            async with session.get(search_url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'}) as r:
                if r.status == 200:
                    html = await r.text()
                    
                    # Find first movie result
                    movie_match = re.search(r'<a href="/title/(tt\d+)/', html)
                    if movie_match:
                        imdb_id = movie_match.group(1)
                        movie_url = f"https://www.imdb.com/title/{imdb_id}/"
                        
                        async with session.get(movie_url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'}) as r2:
                            if r2.status == 200:
                                movie_html = await r2.text()
                                
                                # Try to find poster
                                poster_patterns = [
                                    r'<img[^>]*class="[^"]*ipc-image[^"]*"[^>]*src="([^"]+)"',
                                    r'<meta property="og:image" content="([^"]+)"',
                                ]
                                
                                for pattern in poster_patterns:
                                    poster_match = re.search(pattern, movie_html)
                                    if poster_match:
                                        poster_url = poster_match.group(1)
                                        
                                        if poster_url and poster_url.startswith('http'):
                                            # Upgrade quality
                                            if 'media-amazon.com' in poster_url:
                                                poster_url = re.sub(r'_V1_.*\.jpg', '_V1_QL100_UX1000_.jpg', poster_url)
                                            
                                            # Try to get rating
                                            rating_match = re.search(r'"ratingValue":"(\d+\.\d+)"', movie_html)
                                            rating = rating_match.group(1) if rating_match else '0.0'
                                            
                                            self.movie_db['stats']['imdb'] += 1
                                            return {
                                                'poster_url': poster_url,
                                                'source': 'IMDb',
                                                'rating': rating
                                            }
            
            return None
        
        except Exception as e:
            logger.error(f"IMDb poster error: {e}")
            return None
    
    async def get_poster_justwatch(self, title: str, session: aiohttp.ClientSession) -> Optional[Dict]:
        """Fetch poster from JustWatch"""
        try:
            search_url = f"https://www.justwatch.com/in/search?q={urllib.parse.quote(title)}"
            
            async with session.get(search_url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'}) as r:
                if r.status == 200:
                    html = await r.text()
                    
                    # Find poster image
                    poster_match = re.search(r'"poster":"([^"]+)"', html)
                    if poster_match:
                        poster_path = poster_match.group(1)
                        
                        if poster_path:
                            # JustWatch uses relative paths
                            if not poster_path.startswith('http'):
                                poster_url = f"https://images.justwatch.com{poster_path}"
                            else:
                                poster_url = poster_path
                            
                            # Upgrade quality
                            poster_url = poster_url.replace('/s166/', '/s592/')
                            
                            self.movie_db['stats']['justwatch'] += 1
                            return {
                                'poster_url': poster_url,
                                'source': 'JustWatch',
                                'rating': '0.0'
                            }
            
            return None
        
        except Exception as e:
            logger.error(f"JustWatch poster error: {e}")
            return None
    
    async def get_poster_impawards(self, title: str, session: aiohttp.ClientSession) -> Optional[Dict]:
        """Fetch poster from IMPAwards"""
        try:
            clean_title = re.sub(r'[^\w\s]', '', title).strip().lower()
            search_title = clean_title.replace(' ', '_')
            
            # Try multiple year variations
            years = ['2024', '2023', '2025', '2022']
            
            for year in years:
                url = f"https://www.impawards.com/{year}/{search_title}.html"
                
                try:
                    async with session.get(url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'}) as r:
                        if r.status == 200:
                            html = await r.text()
                            
                            # Find poster image
                            poster_match = re.search(r'<img[^>]*src="([^"]*posters[^"]+\.jpg)"', html)
                            if poster_match:
                                poster_path = poster_match.group(1)
                                
                                if not poster_path.startswith('http'):
                                    poster_url = f"https://www.impawards.com{poster_path}"
                                else:
                                    poster_url = poster_path
                                
                                self.movie_db['stats']['impawards'] += 1
                                return {
                                    'poster_url': poster_url,
                                    'source': 'IMPAwards',
                                    'rating': '0.0'
                                }
                except:
                    continue
            
            return None
        
        except Exception as e:
            logger.error(f"IMPAwards poster error: {e}")
            return None
    
    async def get_poster_omdb_tmdb(self, title: str, session: aiohttp.ClientSession) -> Optional[Dict]:
        """Fetch poster from OMDB or TMDB API"""
        try:
            # Try OMDB first
            omdb_key = self.get_next_omdb_key()
            omdb_url = f"http://www.omdbapi.com/?apikey={omdb_key}&t={urllib.parse.quote(title)}"
            
            try:
                async with session.get(omdb_url, timeout=5) as r:
                    if r.status == 200:
                        data = await r.json()
                        
                        if data.get('Response') == 'True' and data.get('Poster') != 'N/A':
                            poster_url = data['Poster']
                            
                            # Upgrade quality
                            if 'media-amazon.com' in poster_url:
                                poster_url = re.sub(r'_V1_.*\.jpg', '_V1_QL100_UX1000_.jpg', poster_url)
                            
                            self.movie_db['stats']['omdb'] += 1
                            return {
                                'poster_url': poster_url,
                                'source': 'OMDB',
                                'rating': data.get('imdbRating', '0.0')
                            }
            except:
                pass
            
            # Try TMDB as fallback
            tmdb_key = self.get_next_tmdb_key()
            tmdb_search_url = f"https://api.themoviedb.org/3/search/movie?api_key={tmdb_key}&query={urllib.parse.quote(title)}"
            
            try:
                async with session.get(tmdb_search_url, timeout=5) as r:
                    if r.status == 200:
                        data = await r.json()
                        
                        if data.get('results') and len(data['results']) > 0:
                            result = data['results'][0]
                            poster_path = result.get('poster_path')
                            
                            if poster_path:
                                poster_url = f"https://image.tmdb.org/t/p/w780{poster_path}"
                                
                                self.movie_db['stats']['tmdb'] += 1
                                return {
                                    'poster_url': poster_url,
                                    'source': 'TMDB',
                                    'rating': str(result.get('vote_average', 0.0))
                                }
            except:
                pass
            
            return None
        
        except Exception as e:
            logger.error(f"OMDB/TMDB poster error: {e}")
            return None
    
    async def get_poster_guaranteed(self, title: str, session: aiohttp.ClientSession) -> Dict:
        """Get poster with multiple fallback sources (guaranteed to return something)"""
        try:
            # Check cache first
            cache_key = title.lower().strip()
            
            if cache_key in self.movie_db['poster_cache']:
                cached_data, cached_time = self.movie_db['poster_cache'][cache_key]
                
                # Cache valid for 1 hour
                if (datetime.now() - cached_time).seconds < 3600:
                    self.movie_db['stats']['cache_hits'] += 1
                    return cached_data
            
            # Try all sources concurrently
            sources = [
                self.get_poster_letterboxd,
                self.get_poster_imdb,
                self.get_poster_justwatch,
                self.get_poster_impawards,
                self.get_poster_omdb_tmdb,
            ]
            
            tasks = [source(title, session) for source in sources]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Return first successful result
            for result in results:
                if isinstance(result, dict) and result:
                    # Cache the result
                    self.movie_db['poster_cache'][cache_key] = (result, datetime.now())
                    return result
            
            # Fallback to custom backend API
            year_match = re.search(r'\b(19|20)\d{2}\b', title)
            year = year_match.group() if year_match else ""
            
            fallback_result = {
                'poster_url': f"{self.config.BACKEND_URL}/api/poster?title={urllib.parse.quote(title)}&year={year}",
                'source': 'CUSTOM',
                'rating': '0.0'
            }
            
            self.movie_db['stats']['custom'] += 1
            self.movie_db['poster_cache'][cache_key] = (fallback_result, datetime.now())
            
            return fallback_result
        
        except Exception as e:
            logger.error(f"Poster fetching error: {e}")
            
            # Ultimate fallback
            return {
                'poster_url': f"{self.config.BACKEND_URL}/api/poster?title={urllib.parse.quote(title)}",
                'source': 'CUSTOM',
                'rating': '0.0'
            }


# Initialize global poster manager (will be set in app.py)
poster_manager = None

def init_poster_manager(config, movie_db):
    """Initialize global poster manager"""
    global poster_manager
    poster_manager = PosterManager(config, movie_db)
    logger.info("✅ Poster Manager initialized")
    return poster_manager
