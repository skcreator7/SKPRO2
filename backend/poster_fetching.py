# poster_fetching.py - Complete Fixed Version with All Sources + Telegram

import asyncio
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import aiohttp
import urllib.parse
import json
import logging
import base64

logger = logging.getLogger(__name__)

TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w780"
CUSTOM_POSTER_URL = "https://iili.io/fAeIwv9.th.png"
CACHE_TTL = 3600  # 1 hour


class PosterSource(Enum):
    TMDB = "tmdb"
    OMDB = "omdb"
    LETTERBOXD = "letterboxd"
    IMDB = "imdb"
    JUSTWATCH = "justwatch"
    IMPAWARDS = "impawards"
    TELEGRAM = "telegram"
    CUSTOM = "custom"


class PosterFetcher:
    def __init__(self, config, redis=None):
        self.config = config
        self.redis = redis

        self.tmdb_keys = getattr(config, "TMDB_KEYS", [getattr(config, "TMDB_API_KEY", "e547e17d4e91f3e62a571655cd1ccaff")])
        self.omdb_keys = getattr(config, "OMDB_KEYS", [getattr(config, "OMDB_API_KEY", "8265bd1c")])
        
        # Store config for telegram search
        self.main_channel_id = getattr(config, "MAIN_CHANNEL_ID", -1001767371495)

        self.poster_cache: Dict[str, tuple] = {}
        self.http_session: Optional[aiohttp.ClientSession] = None
        self.lock = asyncio.Lock()

        self.stats = {
            "tmdb": 0,
            "imdb": 0,
            "letterboxd": 0,
            "justwatch": 0,
            "impawards": 0,
            "omdb": 0,
            "telegram": 0,
            "custom": 0,
            "cache_hits": 0,
        }
        
        # Store telegram client reference
        self.user_client = None
        self.bot_client = None

    # -------------------------------------------------
    async def get_http_session(self):
        async with self.lock:
            if not self.http_session or self.http_session.closed:
                self.http_session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=10),
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
                )
                logger.info("✅ HTTP session created")
            return self.http_session

    # -------------------------------------------------
    # REDIS CACHE
    # -------------------------------------------------
    async def redis_get(self, key: str):
        if not self.redis:
            return None
        try:
            data = await self.redis.get(key)
            if data:
                self.stats["cache_hits"] += 1
                return json.loads(data)
        except Exception:
            pass
        return None

    async def redis_set(self, key: str, value: Dict[str, Any]):
        if not self.redis:
            return
        try:
            await self.redis.setex(key, CACHE_TTL, json.dumps(value))
        except Exception:
            pass

    # -------------------------------------------------
    # NORMALIZER - PRESERVE NUMBERS
    # -------------------------------------------------
    def normalize_poster(self, poster: Dict[str, Any], title: str) -> Dict[str, Any]:
        return {
            "poster_url": poster.get("poster_url", CUSTOM_POSTER_URL),
            "source": poster.get("source", PosterSource.CUSTOM.value),
            "title": poster.get("title", title),
            "year": poster.get("year", ""),
            "rating": poster.get("rating", "0.0"),
            "found": True,
        }

    # -------------------------------------------------
    def _clean_title_for_search(self, title: str) -> str:
        """Clean title for search - PRESERVE NUMBERS"""
        if not title:
            return ""
        
        # Remove special characters but keep numbers
        clean = re.sub(r'[^\w\s\-]', ' ', title)
        clean = re.sub(r'\s+', ' ', clean).strip()
        
        # Remove common quality tags but keep numbers
        clean = re.sub(r'\b(480p|720p|1080p|2160p|4k|hd|hevc|x264|x265|web-dl|webrip|bluray|hdtv)\b', '', clean, flags=re.IGNORECASE)
        clean = re.sub(r'\s+', ' ', clean).strip()
        
        return clean

    # -------------------------------------------------
    def _generate_search_variations(self, title: str, year: str = "") -> List[str]:
        """Generate search variations for better matching - PRESERVE NUMBERS"""
        variations = []
        clean_title = self._clean_title_for_search(title)
        
        # Original title
        variations.append(clean_title)
        
        # Title with year
        if year:
            variations.append(f"{clean_title} {year}")
        
        # Remove common suffixes (season, part, episode, etc.)
        clean = re.sub(r'\s*(season|saison|part|episode|vol|volume|chapter)\s*\d+.*$', '', clean_title, flags=re.IGNORECASE)
        if clean.strip() and clean.strip() != clean_title:
            variations.append(clean.strip())
            if year:
                variations.append(f"{clean.strip()} {year}")
        
        # For numbered movies (Don 2, Murder 2, Fast & Furious 2)
        numbered_match = re.search(r'^(.*?)\s+(\d+)$', clean_title)
        if numbered_match:
            base = numbered_match.group(1).strip()
            num = numbered_match.group(2)
            variations.append(base)
            variations.append(f"{base} {num}")
            if year:
                variations.append(f"{base} {year}")
        
        # Remove "The" prefix
        if clean_title.lower().startswith('the '):
            without_the = clean_title[4:]
            variations.append(without_the)
            if year:
                variations.append(f"{without_the} {year}")
        
        # For series like "Money Heist" -> "La Casa De Papel"
        if 'heist' in clean_title.lower():
            variations.append("Money Heist")
            variations.append("La Casa De Papel")
            if year:
                variations.append(f"Money Heist {year}")
                variations.append(f"La Casa De Papel {year}")
        
        # For "The Boys"
        if 'boys' in clean_title.lower():
            variations.append("The Boys")
            variations.append("Boys")
            if year:
                variations.append(f"The Boys {year}")
        
        # For "Fast & Furious"
        if 'fast' in clean_title.lower() and 'furious' in clean_title.lower():
            variations.append("Fast and Furious")
            variations.append("Furious")
            if year:
                variations.append(f"Fast and Furious {year}")
        
        # Remove special characters
        clean2 = re.sub(r'[^\w\s]', ' ', clean_title)
        clean2 = re.sub(r'\s+', ' ', clean2).strip()
        if clean2 != clean_title:
            variations.append(clean2)
            if year:
                variations.append(f"{clean2} {year}")
        
        # Deduplicate
        seen = set()
        unique_variations = []
        for v in variations:
            v_lower = v.lower().strip()
            if v_lower and v_lower not in seen:
                seen.add(v_lower)
                unique_variations.append(v)
        
        return unique_variations[:15]  # Limit to 15 variations

    # -------------------------------------------------
    # TMDB (TOP PRIORITY)
    # -------------------------------------------------
    async def fetch_from_tmdb(self, title: str, year: str = ""):
        session = await self.get_http_session()
        clean_title = self._clean_title_for_search(title)
        
        for key in self.tmdb_keys:
            try:
                params = {"api_key": key, "query": clean_title}
                if year:
                    params["year"] = year
                    
                async with session.get(
                    "https://api.themoviedb.org/3/search/movie",
                    params=params,
                ) as r:
                    if r.status != 200:
                        continue
                    data = await r.json()
                    if not data.get("results"):
                        continue

                    m = data["results"][0]
                    if not m.get("poster_path"):
                        continue

                    self.stats["tmdb"] += 1
                    return {
                        "poster_url": f"{TMDB_IMAGE_BASE}{m['poster_path']}",
                        "source": PosterSource.TMDB.value,
                        "rating": str(m.get("vote_average", "0.0")),
                        "year": (m.get("release_date") or "")[:4],
                        "title": m.get("title", title),
                    }
            except Exception as e:
                logger.debug(f"TMDB error: {e}")
        return None

    # -------------------------------------------------
    # OMDB
    # -------------------------------------------------
    async def fetch_from_omdb(self, title: str, year: str = ""):
        session = await self.get_http_session()
        clean_title = self._clean_title_for_search(title)
        
        for key in self.omdb_keys:
            try:
                url = f"https://www.omdbapi.com/?t={urllib.parse.quote(clean_title)}&apikey={key}"
                if year:
                    url += f"&y={year}"
                    
                async with session.get(url) as r:
                    data = await r.json()
                    poster = data.get("Poster")
                    if poster and poster.startswith("http") and poster != "N/A":
                        self.stats["omdb"] += 1
                        return {
                            "poster_url": poster,
                            "source": PosterSource.OMDB.value,
                            "rating": data.get("imdbRating", "0.0"),
                            "year": data.get("Year", ""),
                            "title": data.get("Title", title),
                        }
            except Exception:
                pass
        return None

    # -------------------------------------------------
    # IMDB
    # -------------------------------------------------
    async def fetch_from_imdb(self, title: str):
        session = await self.get_http_session()
        clean = self._clean_title_for_search(title)
        if not clean:
            return None
        
        try:
            first_char = clean[0].lower()
            url = f"https://v2.sg.media-imdb.com/suggestion/{first_char}/{urllib.parse.quote(clean.replace(' ', '_'))}.json"
            async with session.get(url) as r:
                data = await r.json()
                for item in data.get("d", []):
                    img = item.get("i")
                    poster = (
                        img.get("imageUrl")
                        if isinstance(img, dict)
                        else img[0]
                        if isinstance(img, list)
                        else img
                        if isinstance(img, str)
                        else ""
                    )
                    if poster and poster.startswith("http"):
                        self.stats["imdb"] += 1
                        return {
                            "poster_url": poster,
                            "source": PosterSource.IMDB.value,
                            "year": str(item.get("yr", "")),
                            "title": item.get("l", title),
                            "rating": "0.0",
                        }
        except Exception:
            pass
        return None

    # -------------------------------------------------
    # LETTERBOXD
    # -------------------------------------------------
    async def fetch_from_letterboxd(self, title: str):
        session = await self.get_http_session()
        clean = self._clean_title_for_search(title)
        slug = re.sub(r'[^\w\s]', '', clean).lower().replace(' ', '-')
        try:
            async with session.get(f"https://letterboxd.com/film/{slug}/") as r:
                html = await r.text()
                m = re.search(r'property="og:image" content="([^"]+)"', html)
                if m:
                    self.stats["letterboxd"] += 1
                    return {
                        "poster_url": m.group(1),
                        "source": PosterSource.LETTERBOXD.value,
                        "title": title,
                        "rating": "0.0",
                        "year": "",
                    }
        except Exception:
            pass
        return None

    # -------------------------------------------------
    # JUSTWATCH
    # -------------------------------------------------
    async def fetch_from_justwatch(self, title: str):
        session = await self.get_http_session()
        clean = self._clean_title_for_search(title)
        slug = re.sub(r'[^\w\s]', '', clean).lower().replace(' ', '-')
        try:
            async with session.get(f"https://www.justwatch.com/in/movie/{slug}") as r:
                html = await r.text()
                m = re.search(r'property="og:image" content="([^"]+)"', html)
                if m:
                    self.stats["justwatch"] += 1
                    return {
                        "poster_url": m.group(1),
                        "source": PosterSource.JUSTWATCH.value,
                        "title": title,
                        "rating": "0.0",
                        "year": "",
                    }
        except Exception:
            pass
        return None

    # -------------------------------------------------
    # IMPAWARDS
    # -------------------------------------------------
    async def fetch_from_impawards(self, title: str):
        session = await self.get_http_session()
        clean = self._clean_title_for_search(title)
        year_match = re.search(r'\b(19|20)\d{2}\b', clean)
        if not year_match:
            return None
        year = year_match.group()
        clean_title = re.sub(r'\b(19|20)\d{2}\b', '', clean).strip().replace(' ', '_')
        url = f"https://www.impawards.com/{year}/posters/{clean_title}.jpg"
        try:
            async with session.head(url, timeout=3) as r:
                if r.status == 200:
                    self.stats["impawards"] += 1
                    return {
                        "poster_url": url,
                        "source": PosterSource.IMPAWARDS.value,
                        "title": title,
                        "year": year,
                        "rating": "0.0",
                    }
        except Exception:
            pass
        return None

    # -------------------------------------------------
    # TELEGRAM CHANNEL SEARCH (NEW - SECOND LAST PRIORITY)
    # -------------------------------------------------
    async def fetch_from_telegram(self, title: str, year: str = ""):
        """Fetch poster from Telegram channel using title search"""
        try:
            # Get telegram client - try user first, then bot
            client = self.user_client or self.bot_client
            if not client:
                logger.debug("⚠️ No Telegram client available for poster fetch")
                return None
            
            # Generate search variations
            variations = self._generate_search_variations(title, year)
            
            # Also add the raw title variations
            raw_variations = [
                title,
                f"{title} poster",
                f"{title} movie",
                f"{title} {year}" if year else "",
                f"{title} {year} poster" if year else "",
            ]
            all_variations = list(dict.fromkeys(variations + raw_variations))
            
            # Limit variations
            all_variations = all_variations[:12]
            
            # Search in main channel
            for search_term in all_variations:
                if not search_term or len(search_term) < 2:
                    continue
                    
                try:
                    async for msg in client.search_messages(
                        self.main_channel_id,
                        query=search_term,
                        limit=3
                    ):
                        if not msg or not msg.text:
                            continue
                        
                        # Check if message has media with thumbnail
                        if hasattr(msg, 'media') and msg.media:
                            # Try to get thumbnail
                            thumbnail_file_id = None
                            
                            # Check for video thumbnail
                            if hasattr(msg, 'video') and msg.video:
                                if hasattr(msg.video, 'thumbnail') and msg.video.thumbnail:
                                    thumbnail_file_id = msg.video.thumbnail.file_id
                                elif hasattr(msg.video, 'thumbs') and msg.video.thumbs:
                                    thumbnail_file_id = msg.video.thumbs[0].file_id
                            
                            # Check for document thumbnail
                            if not thumbnail_file_id and hasattr(msg, 'document') and msg.document:
                                if hasattr(msg.document, 'thumbnail') and msg.document.thumbnail:
                                    thumbnail_file_id = msg.document.thumbnail.file_id
                                elif hasattr(msg.document, 'thumbs') and msg.document.thumbs:
                                    thumbnail_file_id = msg.document.thumbs[0].file_id
                            
                            # Check for photo
                            if not thumbnail_file_id and hasattr(msg, 'photo') and msg.photo:
                                try:
                                    # Get smallest photo
                                    photo = msg.photo
                                    if hasattr(photo, 'file_id'):
                                        thumbnail_file_id = photo.file_id
                                    elif isinstance(photo, list) and photo:
                                        thumbnail_file_id = photo[0].file_id
                                except:
                                    pass
                            
                            # Also check media.thumbnail
                            if not thumbnail_file_id and hasattr(msg.media, 'thumbnail'):
                                if hasattr(msg.media.thumbnail, 'file_id'):
                                    thumbnail_file_id = msg.media.thumbnail.file_id
                            
                            if thumbnail_file_id:
                                try:
                                    # Download thumbnail
                                    thumb_data = await client.download_media(
                                        thumbnail_file_id,
                                        in_memory=True
                                    )
                                    if thumb_data:
                                        if isinstance(thumb_data, bytes):
                                            thumb_url = f"data:image/jpeg;base64,{base64.b64encode(thumb_data).decode()}"
                                        else:
                                            # BytesIO object
                                            thumb_data.seek(0)
                                            thumb_url = f"data:image/jpeg;base64,{base64.b64encode(thumb_data.read()).decode()}"
                                        
                                        if thumb_url and len(thumb_url) < 200000:
                                            self.stats["telegram"] += 1
                                            logger.info(f"📺 TELEGRAM poster found for: {title} (via '{search_term}')")
                                            return {
                                                "poster_url": thumb_url,
                                                "source": PosterSource.TELEGRAM.value,
                                                "title": title,
                                                "year": year,
                                                "rating": "0.0",
                                            }
                                except Exception as e:
                                    logger.debug(f"Download thumbnail error: {e}")
                                    continue
                                    
                except Exception as e:
                    logger.debug(f"Telegram search error for '{search_term}': {e}")
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Telegram poster fetch error: {e}")
            return None

    # -------------------------------------------------
    # CUSTOM POSTER (LAST RESORT - FALLBACK)
    # -------------------------------------------------
    async def create_custom_poster(self, title: str):
        self.stats["custom"] += 1
        return {
            "poster_url": CUSTOM_POSTER_URL,
            "source": PosterSource.CUSTOM.value,
            "title": title,
            "year": "",
            "rating": "0.0",
        }

    # -------------------------------------------------
    # SET TELEGRAM CLIENTS
    # -------------------------------------------------
    def set_telegram_clients(self, user_client=None, bot_client=None):
        """Set Telegram clients for poster fetching"""
        self.user_client = user_client
        self.bot_client = bot_client
        logger.info("✅ Telegram clients set for poster fetcher")

    # -------------------------------------------------
    # MAIN FETCH POSTER FUNCTION
    # -------------------------------------------------
    async def fetch_poster(self, title: str, year: str = "") -> Dict[str, Any]:
        """Fetch poster with priority order: TMDB > OMDB > Letterboxd > IMDB > JustWatch > IMPAwards > TELEGRAM > CUSTOM"""
        key = f"poster:{title.lower().strip()}:{year}"

        # Check Redis cache
        cached = await self.redis_get(key)
        if cached:
            return cached

        # Check memory cache
        if key in self.poster_cache:
            data, ts = self.poster_cache[key]
            if (datetime.now() - ts).seconds < CACHE_TTL:
                self.stats["cache_hits"] += 1
                return data

        # Clean title for search
        clean_title = self._clean_title_for_search(title)
        
        # Extract year from title if not provided
        if not year:
            year_match = re.search(r'\b(19|20)\d{2}\b', clean_title)
            if year_match:
                year = year_match.group()

        # Try all sources in priority order
        sources = [
            self.fetch_from_tmdb(clean_title, year),
            self.fetch_from_omdb(clean_title, year),
            self.fetch_from_letterboxd(clean_title),
            self.fetch_from_imdb(clean_title),
            self.fetch_from_justwatch(clean_title),
            self.fetch_from_impawards(clean_title),
            self.fetch_from_telegram(clean_title, year),  # TELEGRAM - SECOND LAST
        ]

        results = await asyncio.gather(*sources, return_exceptions=True)

        for r in results:
            if isinstance(r, dict) and r.get("poster_url", "").startswith("http"):
                normalized = self.normalize_poster(r, title)
                self.poster_cache[key] = (normalized, datetime.now())
                await self.redis_set(key, normalized)
                return normalized

        # Last resort: Custom fallback
        custom = await self.create_custom_poster(title)
        normalized = self.normalize_poster(custom, title)
        self.poster_cache[key] = (normalized, datetime.now())
        await self.redis_set(key, normalized)
        return normalized

    # -------------------------------------------------
    # FETCH BATCH POSTERS
    # -------------------------------------------------
    async def fetch_batch_posters(self, titles: List[str]) -> Dict[str, Dict]:
        """Fetch posters for multiple titles in parallel"""
        posters = await asyncio.gather(*(self.fetch_poster(t) for t in titles))
        return dict(zip(titles, posters))

    # -------------------------------------------------
    # GET STATS
    # -------------------------------------------------
    def get_stats(self) -> Dict[str, int]:
        """Get fetcher statistics"""
        return self.stats.copy()

    # -------------------------------------------------
    # CLOSE SESSION
    # -------------------------------------------------
    async def close(self):
        """Close HTTP session"""
        if self.http_session and not self.http_session.closed:
            await self.http_session.close()
            self.http_session = None
