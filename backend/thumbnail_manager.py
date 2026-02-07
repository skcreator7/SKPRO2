# ============================================================================
# thumbnail_manager.py - CLEAN WORKING VERSION (NO ERRORS)
# ============================================================================

import asyncio
import base64
import logging
import time
import re
from datetime import datetime
from typing import Dict, Any, Optional, List
import aiohttp

logger = logging.getLogger(__name__)

class ThumbnailManager:
    """CLEAN Thumbnail management system - NO ERRORS"""
    
    def __init__(self, mongo_client, config, bot_handler=None):
        self.mongo_client = mongo_client
        self.db_name = "sk4film"
        self.db = mongo_client[self.db_name]
        self.thumbnails_col = self.db.thumbnails
        self.files_col = self.db.files
        self.config = config
        self.bot_handler = bot_handler
        
        # Simple cache
        self.thumbnail_cache = {}
        self.cache_ttl = 3600
        
        # API Keys
        self.tmdb_api_key = getattr(config, 'TMDB_API_KEY', '')
        self.omdb_api_key = getattr(config, 'OMDB_API_KEY', '')
        
        # Stats
        self.stats = {
            'total_requests': 0,
            'successful': 0,
            'failed': 0
        }
    
    async def initialize(self):
        """Initialize with error handling"""
        logger.info("🖼️ Initializing CLEAN Thumbnail Manager...")
        
        try:
            await self.db.command('ping')
            logger.info(f"✅ Database connection verified")
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False
        
        # Create indexes with error handling
        await self._safe_create_indexes()
        
        logger.info("✅ Thumbnail Manager initialized (CLEAN VERSION)")
        return True
    
    async def _safe_create_indexes(self):
        """Safely create indexes without conflicts"""
        try:
            # Drop existing problematic indexes first
            try:
                await self.thumbnails_col.drop_index("thumbnails_ttl_index")
                logger.info("✅ Dropped old TTL index")
            except Exception:
                pass
            
            try:
                await self.thumbnails_col.drop_index("thumbnails_title_unique")
                logger.info("✅ Dropped old title index")
            except Exception:
                pass
            
            try:
                await self.thumbnails_col.drop_index("thumbnails_message_index")
                logger.info("✅ Dropped old message index")
            except Exception:
                pass
            
            # Create new indexes
            # 1. TTL index (30 days = 2592000 seconds)
            await self.thumbnails_col.create_index(
                [("last_accessed", 1)],
                expireAfterSeconds=2592000,  # 30 days
                name="ttl_index_new",
                background=True
            )
            logger.info("✅ Created TTL index (30 days)")
            
            # 2. Normalized title index
            await self.thumbnails_col.create_index(
                [("normalized_title", 1)],
                unique=True,
                name="title_index_new",
                background=True
            )
            logger.info("✅ Created title unique index")
            
            # 3. Message index (non-unique, sparse)
            await self.thumbnails_col.create_index(
                [("channel_id", 1), ("message_id", 1)],
                sparse=True,
                name="message_index_new",
                background=True
            )
            logger.info("✅ Created message index")
            
        except Exception as e:
            logger.error(f"⚠️ Index creation error (non-critical): {e}")
    
    async def get_thumbnail_for_movie(self, title: str, channel_id: int = None, message_id: int = None) -> Dict[str, Any]:
        """
        SIMPLE and CLEAN thumbnail fetching
        """
        self.stats['total_requests'] += 1
        
        try:
            # Normalize title
            normalized_title = self.normalize_title(title)
            
            # 1. Check database first
            db_result = await self._get_from_database_simple(normalized_title)
            if db_result:
                self.stats['successful'] += 1
                return db_result
            
            # 2. Try Telegram extraction (if possible)
            telegram_thumb = None
            if channel_id and message_id and self.bot_handler:
                telegram_thumb = await self._simple_telegram_extract(channel_id, message_id)
            
            if telegram_thumb:
                await self._save_to_database_simple(
                    normalized_title=normalized_title,
                    thumbnail_url=telegram_thumb,
                    source='telegram',
                    channel_id=channel_id,
                    message_id=message_id
                )
                self.stats['successful'] += 1
                return {
                    'thumbnail_url': telegram_thumb,
                    'source': 'telegram',
                    'has_thumbnail': True,
                    'extracted': True
                }
            
            # 3. Try TMDB API
            tmdb_thumb = await self._fetch_tmdb_simple(title)
            if tmdb_thumb:
                await self._save_to_database_simple(
                    normalized_title=normalized_title,
                    thumbnail_url=tmdb_thumb,
                    source='tmdb',
                    channel_id=channel_id,
                    message_id=message_id
                )
                self.stats['successful'] += 1
                return {
                    'thumbnail_url': tmdb_thumb,
                    'source': 'tmdb',
                    'has_thumbnail': True,
                    'extracted': False
                }
            
            # 4. Try Letterboxd as fallback
            letterboxd_thumb = await self._fetch_letterboxd_simple(title)
            if letterboxd_thumb:
                await self._save_to_database_simple(
                    normalized_title=normalized_title,
                    thumbnail_url=letterboxd_thumb,
                    source='letterboxd',
                    channel_id=channel_id,
                    message_id=message_id
                )
                self.stats['successful'] += 1
                return {
                    'thumbnail_url': letterboxd_thumb,
                    'source': 'letterboxd',
                    'has_thumbnail': True,
                    'extracted': False
                }
            
            # 5. No thumbnail found
            await self._save_empty_to_database(
                normalized_title=normalized_title,
                channel_id=channel_id,
                message_id=message_id
            )
            
            self.stats['failed'] += 1
            return {
                'thumbnail_url': '',
                'source': 'none',
                'has_thumbnail': False,
                'extracted': False
            }
            
        except Exception as e:
            logger.error(f"❌ Thumbnail error for {title}: {e}")
            self.stats['failed'] += 1
            return {
                'thumbnail_url': '',
                'source': 'error',
                'has_thumbnail': False,
                'extracted': False
            }
    
    async def _get_from_database_simple(self, normalized_title: str) -> Optional[Dict]:
        """Simple database fetch"""
        try:
            doc = await self.thumbnails_col.find_one(
                {"normalized_title": normalized_title}
            )
            
            if doc:
                # Update last accessed
                await self.thumbnails_col.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"last_accessed": datetime.now()}}
                )
                
                has_thumb = bool(doc.get('thumbnail_url'))
                return {
                    'thumbnail_url': doc.get('thumbnail_url', ''),
                    'source': doc.get('source', 'database'),
                    'has_thumbnail': has_thumb,
                    'extracted': doc.get('extracted', False)
                }
            
            return None
        except Exception as e:
            logger.debug(f"Database fetch error: {e}")
            return None
    
    async def _save_to_database_simple(self, normalized_title: str, thumbnail_url: str, 
                                     source: str, channel_id: int = None, message_id: int = None):
        """Simple database save - NO CONFLICTS"""
        try:
            doc = {
                'normalized_title': normalized_title,
                'thumbnail_url': thumbnail_url,
                'source': source,
                'last_accessed': datetime.now(),
                'updated_at': datetime.now()
            }
            
            if channel_id is not None:
                doc['channel_id'] = channel_id
            if message_id is not None:
                doc['message_id'] = message_id
            
            # Simple upsert - no $inc operations
            await self.thumbnails_col.update_one(
                {'normalized_title': normalized_title},
                {'$set': doc, '$setOnInsert': {'created_at': datetime.now()}},
                upsert=True
            )
            
        except Exception as e:
            logger.debug(f"Save error (non-critical): {e}")
    
    async def _save_empty_to_database(self, normalized_title: str, channel_id: int = None, message_id: int = None):
        """Save empty thumbnail to avoid repeated searches"""
        try:
            doc = {
                'normalized_title': normalized_title,
                'thumbnail_url': '',
                'source': 'none',
                'last_accessed': datetime.now(),
                'updated_at': datetime.now()
            }
            
            if channel_id is not None:
                doc['channel_id'] = channel_id
            if message_id is not None:
                doc['message_id'] = message_id
            
            await self.thumbnails_col.update_one(
                {'normalized_title': normalized_title},
                {'$set': doc, '$setOnInsert': {'created_at': datetime.now()}},
                upsert=True
            )
            
        except Exception as e:
            logger.debug(f"Empty save error: {e}")
    
    async def _simple_telegram_extract(self, channel_id: int, message_id: int) -> Optional[str]:
        """Simple Telegram extraction"""
        try:
            if not self.bot_handler:
                return None
            
            # Try to get message
            message = await self.bot_handler.get_message(channel_id, message_id)
            if not message:
                return None
            
            # Check for video thumbnail
            if hasattr(message, 'video') and message.video:
                if hasattr(message.video, 'thumbs') and message.video.thumbs:
                    for thumb in message.video.thumbs:
                        if hasattr(thumb, 'bytes'):
                            return f"data:image/jpeg;base64,{base64.b64encode(thumb.bytes).decode()}"
            
            # Check for document thumbnail
            if hasattr(message, 'document') and message.document:
                if hasattr(message.document, 'thumbs') and message.document.thumbs:
                    for thumb in message.document.thumbs:
                        if hasattr(thumb, 'bytes'):
                            return f"data:image/jpeg;base64,{base64.b64encode(thumb.bytes).decode()}"
            
            return None
            
        except Exception as e:
            logger.debug(f"Telegram extraction error: {e}")
            return None
    
    async def _fetch_tmdb_simple(self, title: str) -> Optional[str]:
        """Simple TMDB fetch"""
        if not self.tmdb_api_key:
            return None
        
        try:
            clean_title = self.clean_title_for_api(title)
            if not clean_title:
                return None
            
            async with aiohttp.ClientSession() as session:
                # Movie search
                url = "https://api.themoviedb.org/3/search/movie"
                params = {
                    'api_key': self.tmdb_api_key,
                    'query': clean_title,
                    'language': 'en-US',
                    'page': 1
                }
                
                async with session.get(url, params=params, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('results') and len(data['results']) > 0:
                            poster = data['results'][0].get('poster_path')
                            if poster:
                                return f"https://image.tmdb.org/t/p/w500{poster}"
            
            return None
        except Exception:
            return None
    
    async def _fetch_letterboxd_simple(self, title: str) -> Optional[str]:
        """Simple Letterboxd fetch"""
        try:
            clean_title = self.clean_title_for_api(title)
            if not clean_title:
                return None
            
            # Create slug
            slug = re.sub(r'[^\w\s-]', '', clean_title).strip().lower()
            slug = re.sub(r'[-\s]+', '-', slug)
            
            async with aiohttp.ClientSession() as session:
                url = f"https://letterboxd.com/film/{slug}/"
                headers = {'User-Agent': 'Mozilla/5.0'}
                
                async with session.get(url, headers=headers, timeout=5) as response:
                    if response.status == 200:
                        html = await response.text()
                        # Look for og:image
                        match = re.search(r'property="og:image" content="([^"]+)"', html)
                        if match:
                            return match.group(1)
            
            return None
        except Exception:
            return None
    
    def normalize_title(self, title: str) -> str:
        """Simple title normalization"""
        if not title:
            return ""
        
        # Basic cleaning
        title = title.lower().strip()
        
        # Remove @mentions and hashtags
        title = re.sub(r'^[@#]\w+\s*', '', title)
        
        # Remove year patterns
        title = re.sub(r'\s*[\(\[]?\d{4}[\)\]]?', '', title)
        
        # Remove quality indicators
        patterns = [
            r'\b\d{3,4}p\b',
            r'\b(?:hd|fhd|uhd|4k|1080p|720p)\b',
            r'\b(?:hevc|x265|x264|h264|h265)\b',
            r'\b(?:webrip|web-dl|bluray|dvdrip|hdtv)\b',
            r'\b(?:hindi|english|tamil|telugu)\b',
            r'\[.*?\]',
            r'\(.*?\)',
        ]
        
        for pattern in patterns:
            title = re.sub(pattern, '', title, flags=re.IGNORECASE)
        
        # Clean up
        title = re.sub(r'\s+', ' ', title).strip()
        
        return title
    
    def clean_title_for_api(self, title: str) -> str:
        """Clean title for API"""
        title = self.normalize_title(title)
        
        # Remove numbers at end
        title = re.sub(r'\s+\d+$', '', title)
        
        # Take first few words
        words = title.split()
        if len(words) > 4:
            title = ' '.join(words[:3])
        
        return title.strip()
    
    async def get_thumbnails_batch(self, movies: List[Dict]) -> List[Dict]:
        """Simple batch processing"""
        try:
            results = []
            
            for movie in movies:
                title = movie.get('title', '')
                channel_id = movie.get('channel_id')
                message_id = movie.get('message_id') or movie.get('real_message_id')
                
                thumbnail = await self.get_thumbnail_for_movie(title, channel_id, message_id)
                
                movie_with_thumb = movie.copy()
                movie_with_thumb.update(thumbnail)
                results.append(movie_with_thumb)
            
            # Count success
            successful = sum(1 for r in results if r.get('has_thumbnail'))
            total = len(results)
            
            if total > 0:
                success_rate = (successful / total) * 100
                logger.info(f"📊 Batch: {successful}/{total} ({success_rate:.1f}%)")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Batch error: {e}")
            return [{
                **movie,
                'thumbnail_url': '',
                'source': 'error',
                'has_thumbnail': False,
                'extracted': False
            } for movie in movies]
    
    async def extract_thumbnails_for_existing_files(self):
        """Extract thumbnails for existing files"""
        if not self.files_col:
            return
        
        logger.info("🔄 Extracting thumbnails for existing files...")
        
        try:
            # Get files
            cursor = self.files_col.find(
                {'is_video_file': True},
                {'title': 1, 'channel_id': 1, 'real_message_id': 1, '_id': 1}
            ).limit(100)  # Limit to 100 for testing
            
            files = []
            async for doc in cursor:
                files.append({
                    'title': doc.get('title', ''),
                    'channel_id': doc.get('channel_id'),
                    'message_id': doc.get('real_message_id'),
                    'db_id': doc.get('_id')
                })
            
            logger.info(f"📊 Processing {len(files)} files...")
            
            # Process in small batches
            batch_size = 5
            for i in range(0, len(files), batch_size):
                batch = files[i:i + batch_size]
                
                # Convert to movie format
                movies = [{'title': f['title'], 'channel_id': f['channel_id'], 'message_id': f['message_id']} 
                         for f in batch]
                
                # Get thumbnails
                results = await self.get_thumbnails_batch(movies)
                
                # Update files
                for j, result in enumerate(results):
                    if j < len(batch):
                        await self.files_col.update_one(
                            {'_id': batch[j]['db_id']},
                            {'$set': {
                                'thumbnail_url': result.get('thumbnail_url', ''),
                                'thumbnail_source': result.get('source', 'none')
                            }}
                        )
                
                # Small delay
                if i + batch_size < len(files):
                    await asyncio.sleep(0.5)
            
            logger.info("✅ Thumbnail extraction complete")
            
        except Exception as e:
            logger.error(f"❌ Extraction error: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Simple statistics"""
        try:
            total = self.stats['total_requests']
            successful = self.stats['successful']
            failed = self.stats['failed']
            
            success_rate = (successful / total * 100) if total > 0 else 0
            
            return {
                'total_requests': total,
                'successful': successful,
                'failed': failed,
                'success_rate': f"{success_rate:.1f}%",
                'target_rate': "99%",
                'status': '✅ RUNNING (CLEAN VERSION)'
            }
        except Exception as e:
            logger.error(f"Stats error: {e}")
            return {'status': 'ERROR'}
    
    async def shutdown(self):
        """Clean shutdown"""
        self.thumbnail_cache.clear()
        logger.info("✅ Thumbnail Manager shutdown")
