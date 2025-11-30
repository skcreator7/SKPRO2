import re
import aiohttp
import urllib.parse
from datetime import datetime
import logging
import asyncio

logger = logging.getLogger(__name__)

class PosterFetcher:
    def __init__(self, config, stats_dict):
        self.config = config
        self.stats = stats_dict
        self.poster_cache = {}
    
    async def get_poster_letterboxd(self, title, session):
        """Get poster from Letterboxd"""
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
                            poster_patterns = [
                                r'<meta property="og:image" content="([^"]+)"',
                                r'<img[^>]*class="[^"]*poster[^"]*"[^>]*src="([^"]+)"',
                            ]
                            
                            for pattern in poster_patterns:
                                poster_match = re.search(pattern, html_content)
                                if poster_match:
                                    poster_url = poster_match.group(1)
                                    if poster_url and poster_url.startswith('http'):
                                        if 'cloudfront.net' in poster_url:
                                            poster_url = poster_url.replace('-0-500-0-750', '-0-1000-0-1500')
                                        elif 's.ltrbxd.com' in poster_url:
                                            poster_url = poster_url.replace('/width/500/', '/width/1000/')
                                        
                                        rating_match = re.search(r'<meta name="twitter:data2" content="([^"]+)"', html_content)
                                        rating = rating_match.group(1) if rating_match else '0.0'
                                        
                                        res = {'poster_url': poster_url, 'source': 'Letterboxd', 'rating': rating}
                                        self.stats['letterboxd'] += 1
                                        return res
                except:
                    continue
            return None
        except Exception as e:
            logger.error(f"Letterboxd poster error: {e}")
            return None

    async def get_poster_imdb(self, title, session):
        """Get poster from IMDb"""
        try:
            clean_title = re.sub(r'[^\w\s]', '', title).strip()
            search_url = f"https://v2.sg.media-imdb.com/suggestion/{clean_title[0].lower()}/{urllib.parse.quote(clean_title.replace(' ', '_'))}.json"
            
            async with session.get(search_url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'}) as r:
                if r.status == 200:
                    data = await r.json()
                    if data.get('d'):
                        for item in data['d']:
                            if item.get('i'):
                                poster_url = item['i'][0] if isinstance(item['i'], list) else item['i']
                                if poster_url and poster_url.startswith('http'):
                                    poster_url = poster_url.replace('._V1_UX128_', '._V1_UX512_')
                                    rating = str(item.get('yr', '0.0'))
                                    res = {'poster_url': poster_url, 'source': 'IMDb', 'rating': rating}
                                    self.stats['imdb'] += 1
                                    return res
            return None
        except Exception as e:
            logger.error(f"IMDb poster error: {e}")
            return None

    async def get_poster_justwatch(self, title, session):
        """Get poster from JustWatch"""
        try:
            clean_title = re.sub(r'[^\w\s]', '', title).strip()
            slug = clean_title.lower().replace(' ', '-')
            slug = re.sub(r'[^\w\-]', '', slug)
            
            domains = ['com', 'in', 'uk']
            
            for domain in domains:
                justwatch_url = f"https://www.justwatch.com/{domain}/movie/{slug}"
                try:
                    async with session.get(justwatch_url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'}) as r:
                        if r.status == 200:
                            html_content = await r.text()
                            poster_match = re.search(r'<meta property="og:image" content="([^"]+)"', html_content)
                            if poster_match:
                                poster_url = poster_match.group(1)
                                if poster_url and poster_url.startswith('http'):
                                    poster_url = poster_url.replace('http://', 'https://')
                                    res = {'poster_url': poster_url, 'source': 'JustWatch', 'rating': '0.0'}
                                    self.stats['justwatch'] += 1
                                    return res
                except:
                    continue
            return None
        except Exception as e:
            logger.error(f"JustWatch poster error: {e}")
            return None

    async def get_poster_impawards(self, title, session):
        """Get poster from IMPAwards"""
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
                            res = {'poster_url': poster_url, 'source': 'IMPAwards', 'rating': '0.0'}
                            self.stats['impawards'] += 1
                            return res
                except:
                    continue
            return None
        except Exception as e:
            logger.error(f"IMPAwards poster error: {e}")
            return None

    async def get_poster_omdb_tmdb(self, title, session):
        """Get poster from OMDB or TMDB"""
        try:
            # Try OMDB first
            for api_key in self.config.OMDB_KEYS:
                try:
                    url = f"http://www.omdbapi.com/?t={urllib.parse.quote(title)}&apikey={api_key}"
                    async with session.get(url, timeout=5) as r:
                        if r.status == 200:
                            data = await r.json()
                            if data.get('Response') == 'True' and data.get('Poster') and data.get('Poster') != 'N/A':
                                poster_url = data['Poster'].replace('http://', 'https://')
                                res = {'poster_url': poster_url, 'source': 'OMDB', 'rating': data.get('imdbRating', '0.0')}
                                self.stats['omdb'] += 1
                                return res
                except:
                    continue
            
            # Try TMDB
            for api_key in self.config.TMDB_KEYS:
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
                                    res = {'poster_url': poster_url, 'source': 'TMDB', 'rating': str(result.get('vote_average', 0.0))}
                                    self.stats['tmdb'] += 1
                                    return res
                except:
                    continue
            return None
        except Exception as e:
            logger.error(f"OMDB/TMDB poster error: {e}")
            return None

    async def get_poster_guaranteed(self, title, session):
        """Get poster from multiple sources with fallback"""
        cache_key = title.lower().strip()
        
        # Check cache first
        if cache_key in self.poster_cache:
            cached_data, timestamp = self.poster_cache[cache_key]
            if (datetime.now() - timestamp).seconds < 3600:
                self.stats['cache_hits'] += 1
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
                self.poster_cache[cache_key] = (result, datetime.now())
                return result
        
        # Fallback to custom poster
        year_match = re.search(r'\b(19|20)\d{2}\b', title)
        year = year_match.group() if year_match else ""
        
        res = {
            'poster_url': f"{self.config.BACKEND_URL}/api/poster?title={urllib.parse.quote(title)}&year={year}", 
            'source': 'CUSTOM', 
            'rating': '0.0'
        }
        self.poster_cache[cache_key] = (res, datetime.now())
        self.stats['custom'] += 1
        return res

    def clear_cache(self):
        """Clear poster cache"""
        self.poster_cache.clear()
        logger.info("🧹 Poster cache cleared")
