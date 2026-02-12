import asyncio
import re
from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum
import aiohttp
import urllib.parse
import json
import logging

logger = logging.getLogger(__name__)

TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w780"
CUSTOM_POSTER_URL = ""  # Empty = No fallback
CACHE_TTL = 3600  # 1 hour


class PosterSource(Enum):
    TMDB = "tmdb"
    OMDB = "omdb"
    LETTERBOXD = "letterboxd"
    IMDB = "imdb"
    JUSTWATCH = "justwatch"
    IMPAWARDS = "impawards"
    CUSTOM = "custom"
    NONE = "none"


class PosterFetcher:
    def __init__(self, config, redis=None):
        self.config = config
        self.redis = redis

        self.tmdb_keys = [getattr(config, "TMDB_API_KEY", "")]
        self.omdb_keys = [getattr(config, "OMDB_API_KEY", "")]

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
            "custom": 0,
            "cache_hits": 0,
        }

    async def get_http_session(self):
        async with self.lock:
            if not self.http_session or self.http_session.closed:
                self.http_session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=10),
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
                )
                logger.info("✅ HTTP session created")
            return self.http_session

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

    def normalize_poster(self, poster: Dict[str, Any], title: str) -> Dict[str, Any]:
        return {
            "poster_url": poster.get("poster_url", ""),
            "source": poster.get("source", PosterSource.NONE.value),
            "title": poster.get("title", title),
            "year": poster.get("year", ""),
            "rating": poster.get("rating", "0.0"),
        }

    async def fetch_from_tmdb(self, title: str):
        session = await self.get_http_session()
        for key in self.tmdb_keys:
            if not key:
                continue
            try:
                async with session.get(
                    "https://api.themoviedb.org/3/search/movie",
                    params={"api_key": key, "query": title},
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

    async def fetch_from_omdb(self, title: str):
        session = await self.get_http_session()
        for key in self.omdb_keys:
            if not key:
                continue
            try:
                async with session.get(
                    f"https://www.omdbapi.com/?t={urllib.parse.quote(title)}&apikey={key}"
                ) as r:
                    if r.status != 200:
                        continue
                    data = await r.json()
                    if data.get("Response") != "True":
                        continue
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

    async def fetch_from_imdb(self, title: str):
        session = await self.get_http_session()
        try:
            clean = re.sub(r"[^\w\s]", "", title).strip()
            if not clean:
                return None
            url = f"https://v2.sg.media-imdb.com/suggestion/{clean[0].lower()}/{urllib.parse.quote(clean.replace(' ', '_'))}.json"
            async with session.get(url) as r:
                if r.status != 200:
                    return None
                data = await r.json()
                for item in data.get("d", []):
                    img = item.get("i")
                    poster = None
                    if isinstance(img, dict):
                        poster = img.get("imageUrl")
                    elif isinstance(img, list) and img:
                        poster = img[0]
                    elif isinstance(img, str):
                        poster = img
                    
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

    async def fetch_from_letterboxd(self, title: str):
        session = await self.get_http_session()
        slug = re.sub(r"[^\w\s]", "", title).lower().replace(" ", "-")
        try:
            async with session.get(f"https://letterboxd.com/film/{slug}/") as r:
                if r.status != 200:
                    return None
                html = await r.text()
                m = re.search(r'property="og:image" content="([^"]+)"', html)
                if m:
                    self.stats["letterboxd"] += 1
                    return {
                        "poster_url": m.group(1),
                        "source": PosterSource.LETTERBOXD.value,
                        "title": title,
                        "rating": "0.0",
                        "year": ""
                    }
        except Exception:
            pass
        return None

    async def fetch_from_justwatch(self, title: str):
        session = await self.get_http_session()
        slug = re.sub(r"[^\w\s]", "", title).lower().replace(" ", "-")
        try:
            async with session.get(f"https://www.justwatch.com/in/movie/{slug}") as r:
                if r.status != 200:
                    return None
                html = await r.text()
                m = re.search(r'property="og:image" content="([^"]+)"', html)
                if m:
                    self.stats["justwatch"] += 1
                    return {
                        "poster_url": m.group(1),
                        "source": PosterSource.JUSTWATCH.value,
                        "title": title,
                        "rating": "0.0",
                        "year": ""
                    }
        except Exception:
            pass
        return None

    async def fetch_from_impawards(self, title: str):
        session = await self.get_http_session()
        year = re.search(r"\b(19|20)\d{2}\b", title)
        if not year:
            return None
        clean = re.sub(r"\b(19|20)\d{2}\b", "", title).strip().replace(" ", "_")
        url = f"https://www.impawards.com/{year.group()}/posters/{clean}.jpg"
        try:
            async with session.head(url) as r:
                if r.status == 200:
                    self.stats["impawards"] += 1
                    return {
                        "poster_url": url,
                        "source": PosterSource.IMPAWARDS.value,
                        "title": title,
                        "year": year.group(),
                        "rating": "0.0",
                    }
        except Exception:
            pass
        return None

    async def fetch_poster(self, title: str) -> Dict[str, Any]:
        """Get poster for movie - RETURNS POSTER ONLY, NO FALLBACK"""
        key = f"poster:{title.lower().strip()}"

        cached = await self.redis_get(key)
        if cached:
            return cached

        if key in self.poster_cache:
            data, ts = self.poster_cache[key]
            if (datetime.now() - ts).seconds < CACHE_TTL:
                self.stats["cache_hits"] += 1
                return data

        sources = [
            self.fetch_from_tmdb(title),
            self.fetch_from_omdb(title),
            self.fetch_from_letterboxd(title),
            self.fetch_from_imdb(title),
            self.fetch_from_justwatch(title),
            self.fetch_from_impawards(title),
        ]

        results = await asyncio.gather(*sources, return_exceptions=True)

        for r in results:
            if isinstance(r, dict) and r.get("poster_url", "").startswith("http"):
                normalized = self.normalize_poster(r, title)
                self.poster_cache[key] = (normalized, datetime.now())
                await self.redis_set(key, normalized)
                return normalized

        # Return empty poster (NO FALLBACK)
        empty_poster = {
            "poster_url": "",
            "source": PosterSource.NONE.value,
            "title": title,
            "year": "",
            "rating": "0.0",
        }
        self.poster_cache[key] = (empty_poster, datetime.now())
        await self.redis_set(key, empty_poster)
        return empty_poster

    async def fetch_batch_posters(self, titles: List[str]) -> Dict[str, Dict]:
        posters = await asyncio.gather(*(self.fetch_poster(t) for t in titles))
        return dict(zip(titles, posters))
    
    async def close(self):
        """Close HTTP session"""
        if self.http_session and not self.http_session.closed:
            await self.http_session.close()
            logger.info("✅ HTTP session closed")
    
    def get_stats(self):
        """Get poster fetcher statistics"""
        return self.stats
