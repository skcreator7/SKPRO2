"""
poster_fetching.py - Movie poster and metadata fetching from various sources
"""
import asyncio
import re
import json
import base64
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
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
        
        # Cache
        self.poster_cache = {}  # title -> (data, timestamp)
        
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
        """Create custom SVG poster"""
        try:
            d = title[:20] + "..." if len(title) > 20 else title
            
            color_schemes = [
                {'bg1': '#667eea', 'bg2': '#764ba2', 'text': '#ffffff'},
                {'bg1': '#f093fb', 'bg2': '#f5576c', 'text': '#ffffff'},
                {'bg1': '#4facfe', 'bg2': '#00f2fe', 'text': '#ffffff'},
                {'bg1': '#43e97b', 'bg2': '#38f9d7', 'text': '#ffffff'},
                {'bg1': '#fa709a', 'bg2': '#fee140', 'text': '#ffffff'},
            ]
            
            scheme = color_schemes[hash(title) % len(color_schemes)]
            text_color = scheme['text']
            bg1_color = scheme['bg1']
            bg2_color = scheme['bg2']
            
            year_text = f'<text x="150" y="305" text-anchor="middle" fill="{text_color}" font-size="14" font-family="Arial">{year}</text>' if year else ''
            
            svg = f'''<svg width="300" height="450" xmlns="http://www.w3.org/2000/svg">
                <defs>
                    <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" style="stop-color:{bg1_color};stop-opacity:1"/>
                        <stop offset="100%" style="stop-color:{bg2_color};stop-opacity:1"/>
                    </linearGradient>
                </defs>
                <rect width="100%" height="100%" fill="url(#bg)"/>
                <rect x="10" y="10" width="280" height="430" fill="none" stroke="{text_color}" stroke-width="2" stroke-opacity="0.3" rx="10"/>
                <circle cx="150" cy="180" r="60" fill="rgba(255,255,255,0.1)"/>
                <text x="150" y="185" text-anchor="middle" fill="{text_color}" font-size="60" font-family="Arial">🎬</text>
                <text x="150" y="280" text-anchor="middle" fill="{text_color}" font-size="16" font-weight="bold" font-family="Arial">{d}</text>
                {year_text}
                <rect x="50" y="380" width="200" height="40" rx="20" fill="rgba(0,0,0,0.3)"/>
                <text x="150" y="405" text-anchor="middle" fill="{text_color}" font-size="16" font-weight="bold" font-family="Arial">SK4FiLM</text>
            </svg>'''
            
            # Convert to base64 data URL
            svg_bytes = svg.encode('utf-8')
            base64_svg = base64.b64encode(svg_bytes).decode('utf-8')
            data_url = f"data:image/svg+xml;base64,{base64_svg}"
            
            result = {
                'poster_url': data_url,
                'source': PosterSource.CUSTOM.value,
                'rating': '0.0',
                'year': year,
                'title': title,
                'is_svg': True
            }
            
            self.stats['custom'] += 1
            return result
            
        except Exception as e:
            logger.error(f"Custom poster creation error: {e}")
            # Fallback
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
            if (datetime.now() - timestamp).seconds < 3600:
                self.stats['cache_hits'] += 1
                return data
        
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
                if isinstance(result, dict) and result:
                    # Cache the result
                    self.poster_cache[cache_key] = (result, datetime.now())
                    return result
        
        # Fallback to custom poster
        year_match = re.search(r'\b(19|20)\d{2}\b', title)
        year = year_match.group() if year_match else ""
        
        custom_poster = await self.create_custom_poster(title, year)
        self.poster_cache[cache_key] = (custom_poster, datetime.now())
        
        return custom_poster
    
    async def fetch_batch_posters(self, titles: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch posters for multiple titles efficiently"""
        results = {}
        
        # First, check cache for all titles
        pending_titles = []
        
        for title in titles:
            cache_key = title.lower().strip()
            if cache_key in self.poster_cache:
                data, timestamp = self.poster_cache[cache_key]
                if (datetime.now() - timestamp).seconds < 3600:
                    self.stats['cache_hits'] += 1
                    results[title] = data
                else:
                    pending_titles.append(title)
            else:
                pending_titles.append(title)
        
        if not pending_titles:
            return results
        
        # Fetch remaining posters
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_poster(title, use_cache=False) for title in pending_titles]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for title, result in zip(pending_titles, batch_results):
                if isinstance(result, dict) and result:
                    results[title] = result
        
        return results
    
    def clear_cache(self):
        """Clear poster cache"""
        self.poster_cache.clear()
        logger.info("🧹 Poster cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get fetcher statistics"""
        return {
            'sources': self.stats.copy(),
            'cache_size': len(self.poster_cache),
            'cache_hits': self.stats['cache_hits']
        }
    
    async def cleanup_expired_cache(self):
        """Cleanup expired cache entries"""
        expired_keys = []
        now = datetime.now()
        
        for key, (data, timestamp) in self.poster_cache.items():
            if (now - timestamp).seconds > 3600:  # 1 hour expiry
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.poster_cache[key]
        
        if expired_keys:
            logger.info(f"🧹 Cleaned {len(expired_keys)} expired poster cache entries")
