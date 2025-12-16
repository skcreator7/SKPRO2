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
CUSTOM_POSTER_URL = "https://iili.io/fAeIwv9.th.png"
CACHE_TTL = 3600  # 1 hour


class PosterSource(Enum):
    TMDB = "tmdb"
    OMDB = "omdb"
    LETTERBOXD = "letterboxd"
    IMDB = "imdb"
    JUSTWATCH = "justwatch"
    IMPAWARDS = "impawards"
    CUSTOM = "custom"


class PosterFetcher:
    def __init__(self, config, redis=None):
        self.config = config
        self.redis = redis  # redis.asyncio.Redis or None

        self.tmdb_keys = getattr(config, "TMDB_KEYS", [])
        self.omdb_keys = getattr(config, "OMDB_KEYS", [])

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

    # -------------------------------------------------
    async def get_http_session(self):
        async with self.lock:
            if not self.http_session or self.http_session.closed:
                self.http_session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=10),
                    headers={"User-Agent": "Mozilla/5.0"},
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
    # TMDB (TOP PRIORITY)
    # -------------------------------------------------
    async def fetch_from_tmdb(self, title: str):
        session = await self.get_http_session()
        for key in self.tmdb_keys:
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
                logger.error(f"TMDB error: {e}")
        return None

    # -------------------------------------------------
    async def fetch_from_omdb(self, title: str):
        session = await self.get_http_session()
        for key in self.omdb_keys:
            try:
                async with session.get(
                    f"https://www.omdbapi.com/?t={urllib.parse.quote(title)}&apikey={key}"
                ) as r:
                    data = await r.json()
                    poster = data.get("Poster")
                    if poster and poster.startswith("http"):
                        self.stats["omdb"] += 1
                        return {
                            "poster_url": poster,
                            "source": PosterSource.OMDB.value,
                            "year": data.get("Year", ""),
                            "title": data.get("Title", title),
                        }
            except Exception:
                pass
        return None

    # -------------------------------------------------
    async def fetch_from_imdb(self, title: str):
        session = await self.get_http_session()
        try:
            clean = re.sub(r"[^\w\s]", "", title).strip()
            url = f"https://v2.sg.media-imdb.com/suggestion/{clean[0].lower()}/{urllib.parse.quote(clean.replace(' ', '_'))}.json"
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
                    if poster.startswith("http"):
                        self.stats["imdb"] += 1
                        return {
                            "poster_url": poster,
                            "source": PosterSource.IMDB.value,
                            "year": str(item.get("yr", "")),
                            "title": item.get("l", title),
                        }
        except Exception:
            pass
        return None

    # -------------------------------------------------
    async def fetch_from_letterboxd(self, title: str):
        session = await self.get_http_session()
        slug = re.sub(r"[^\w\s]", "", title).lower().replace(" ", "-")
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
                    }
        except Exception:
            pass
        return None

    # -------------------------------------------------
    async def fetch_from_justwatch(self, title: str):
        session = await self.get_http_session()
        slug = re.sub(r"[^\w\s]", "", title).lower().replace(" ", "-")
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
                    }
        except Exception:
            pass
        return None

    # -------------------------------------------------
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
                    }
        except Exception:
            pass
        return None

    # -------------------------------------------------
    async def create_custom_poster(self, title: str):
        self.stats["custom"] += 1
        return {
            "poster_url": CUSTOM_POSTER_URL,
            "source": PosterSource.CUSTOM.value,
            "title": title,
        }

    # -------------------------------------------------
    async def fetch_poster(self, title: str) -> Dict[str, Any]:
        key = f"poster:{title.lower().strip()}"

        # 🔥 Redis cache
        cached = await self.redis_get(key)
        if cached:
            return cached

        # Local memory cache
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
                self.poster_cache[key] = (r, datetime.now())
                await self.redis_set(key, r)
                return r

        custom = await self.create_custom_poster(title)
        self.poster_cache[key] = (custom, datetime.now())
        await self.redis_set(key, custom)
        return custom

    # -------------------------------------------------
    async def fetch_batch_posters(self, titles: List[str]) -> Dict[str, Dict]:
        posters = await asyncio.gather(*(self.fetch_poster(t) for t in titles))
        return dict(zip(titles, posters))

