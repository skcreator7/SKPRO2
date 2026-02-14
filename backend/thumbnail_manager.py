# ============================================================================
# 🚀 thumbnail_manager.py - OPTIMIZED THUMBNAIL MANAGEMENT
# ============================================================================

import os
import asyncio
import logging
import base64
import json
import time
import hashlib
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import aiofiles
import aiohttp
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

logger = logging.getLogger(__name__)

# ============================================================================
# ✅ FALLBACK THUMBNAIL URL
# ============================================================================
FALLBACK_THUMBNAIL_URL = "https://iili.io/fAeIwv9.th.png"

# ============================================================================
# ✅ THUMBNAIL MANAGER - OPTIMIZED FOR SPEED & STORAGE
# ============================================================================

class ThumbnailManager:
    """
    🚀 OPTIMIZED Thumbnail Manager
    - Stores ONLY successfully extracted thumbnails
    - Fast Redis caching
    - MongoDB for persistence
    - Background extraction
    - Rate limiting for Telegram
    """
    
    def __init__(
        self,
        download_path: str = "downloads/thumbnails",
        mongodb=None,
        bot_client=None,
        user_client=None,
        file_channel_id: int = None,
        redis_client=None
    ):
        self.download_path = Path(download_path)
        self.download_path.mkdir(parents=True, exist_ok=True)
        
        # MongoDB
        self.mongodb = mongodb
        self.thumbnails_col = None
        self.stats_col = None
        
        # Telegram clients
        self.bot_client = bot_client
        self.user_client = user_client
        self.file_channel_id = file_channel_id
        
        # Redis cache
        self.redis = redis_client
        
        # Stats
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'db_hits': 0,
            'extraction_attempts': 0,
            'extraction_success': 0,
            'extraction_failed': 0,
            'errors': 0,
            'start_time': datetime.now().isoformat()
        }
        
        # Background tasks
        self.cleanup_task = None
        self.is_running = False
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.5  # 500ms between requests
        
    async def initialize(self):
        """Initialize Thumbnail Manager"""
        try:
            if self.mongodb is not None:
                # Get database
                if hasattr(self.mongodb, 'sk4film'):
                    db = self.mongodb.sk4film
                else:
                    db = self.mongodb
                
                # Create collections
                self.thumbnails_col = db.thumbnails
                self.stats_col = db.thumbnail_stats
                
                # Create indexes
                await self._create_indexes()
                
                logger.info("✅ Using MongoDB database from client")
                logger.info("✅ Thumbnail collections initialized")
            else:
                logger.warning("⚠️ MongoDB not provided - using file-based storage only")
            
            # Start cleanup task
            self.is_running = True
            self.cleanup_task = asyncio.create_task(self._cleanup_old_thumbnails())
            
            logger.info("=" * 60)
            logger.info("🚀 THUMBNAIL MANAGER: OPTIMIZED")
            logger.info("=" * 60)
            logger.info("    • Store ONLY extracted: ✅ ENABLED")
            logger.info("    • Redis Caching: ✅ ENABLED" if self.redis else "    • Redis Caching: ❌ DISABLED")
            logger.info("    • Background Extraction: ✅ ENABLED")
            logger.info("    • Rate Limiting: ✅ ENABLED")
            logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ ThumbnailManager initialization error: {e}")
            return False
    
    async def _create_indexes(self):
        """Create MongoDB indexes"""
        if self.thumbnails_col is None:
            return
        
        try:
            # Drop existing indexes if they cause issues
            try:
                await self.thumbnails_col.drop_index("normalized_title_1")
            except:
                pass
            
            # Unique index for normalized_title
            await self.thumbnails_col.create_index(
                "normalized_title",
                unique=True,
                name="normalized_title_unique",
                background=True
            )
            
            # Index for expiry
            await self.thumbnails_col.create_index(
                "expires_at",
                expireAfterSeconds=0,
                name="ttl_index",
                background=True
            )
            
            # Index for extraction status
            await self.thumbnails_col.create_index(
                [("extraction_status", 1), ("extraction_attempts", 1)],
                name="extraction_status",
                background=True
            )
            
            # Index for file channel
            await self.thumbnails_col.create_index(
                [("channel_id", 1), ("message_id", 1)],
                name="channel_message",
                background=True,
                unique=True
            )
            
            # Compound index for queries
            await self.thumbnails_col.create_index(
                [("normalized_title", 1), ("has_thumbnail", 1)],
                name="title_thumbnail",
                background=True
            )
            
            logger.info("✅ Thumbnail indexes created successfully")
            
        except Exception as e:
            logger.warning(f"⚠️ Thumbnail index creation error: {e}")
    
    async def get_thumbnail_for_movie(
        self,
        title: str,
        channel_id: int = None,
        message_id: int = None,
        quality: str = None,
        force_extract: bool = False
    ) -> Dict[str, Any]:
        """
        🚀 OPTIMIZED: Get thumbnail for a movie
        - Checks Redis first (fastest)
        - Then MongoDB
        - Finally extracts if needed
        
        Returns: {
            'thumbnail_url': 'data:image... or None',
            'source': 'cache' | 'database' | 'extracted' | 'fallback',
            'extracted': bool,
            'quality': str,
            'message_id': int
        }
        """
        self.stats['total_requests'] += 1
        await self._rate_limit()
        
        normalized = self._normalize_title(title)
        
        # Result template
        result = {
            'thumbnail_url': None,
            'source': 'none',
            'extracted': False,
            'quality': quality,
            'message_id': message_id
        }
        
        try:
            # ============ STEP 1: Check Redis Cache (Fastest) ============
            if self.redis and not force_extract:
                cache_key = f"thumb:{normalized}"
                cached = await self.redis.get(cache_key)
                if cached:
                    try:
                        cached_data = json.loads(cached)
                        if cached_data.get('thumbnail_url'):
                            self.stats['cache_hits'] += 1
                            logger.debug(f"✅ Redis cache HIT for: {title[:30]}")
                            
                            # Update result
                            result.update({
                                'thumbnail_url': cached_data['thumbnail_url'],
                                'source': 'cache',
                                'extracted': cached_data.get('extracted', False)
                            })
                            
                            # If we have message_id but cached data doesn't, still return
                            return result
                    except:
                        pass
            
            # ============ STEP 2: Check MongoDB ============
            if self.thumbnails_col is not None and not force_extract:
                doc = await self.thumbnails_col.find_one(
                    {"normalized_title": normalized}
                )
                
                if doc and doc.get('thumbnail_url'):
                    self.stats['db_hits'] += 1
                    
                    # Update cache
                    if self.redis:
                        cache_data = {
                            'thumbnail_url': doc['thumbnail_url'],
                            'source': 'database',
                            'extracted': doc.get('extracted', True),
                            'quality': doc.get('quality'),
                            'message_id': doc.get('message_id')
                        }
                        await self.redis.setex(
                            f"thumb:{normalized}",
                            86400,  # 24 hours
                            json.dumps(cache_data)
                        )
                    
                    logger.debug(f"✅ MongoDB HIT for: {title[:30]}")
                    
                    result.update({
                        'thumbnail_url': doc['thumbnail_url'],
                        'source': 'database',
                        'extracted': doc.get('extracted', True),
                        'quality': doc.get('quality'),
                        'message_id': doc.get('message_id')
                    })
                    return result
            
            # ============ STEP 3: Extract if message_id provided ============
            if channel_id and message_id and (force_extract or not result['thumbnail_url']):
                self.stats['extraction_attempts'] += 1
                
                # Try to extract
                thumbnail_url = await self._extract_thumbnail(
                    channel_id, 
                    message_id,
                    title
                )
                
                if thumbnail_url:
                    self.stats['extraction_success'] += 1
                    
                    # Save to database
                    await self._save_thumbnail(
                        title=title,
                        normalized=normalized,
                        thumbnail_url=thumbnail_url,
                        source='extracted',
                        channel_id=channel_id,
                        message_id=message_id,
                        quality=quality,
                        extracted=True
                    )
                    
                    logger.info(f"✅ Extracted thumbnail: {title[:30]}")
                    
                    result.update({
                        'thumbnail_url': thumbnail_url,
                        'source': 'extracted',
                        'extracted': True,
                        'quality': quality,
                        'message_id': message_id
                    })
                    return result
                else:
                    self.stats['extraction_failed'] += 1
                    
                    # Save failed attempt (optional)
                    await self._save_thumbnail(
                        title=title,
                        normalized=normalized,
                        thumbnail_url=None,
                        source='extraction_failed',
                        channel_id=channel_id,
                        message_id=message_id,
                        quality=quality,
                        extracted=False
                    )
                    
                    logger.debug(f"❌ No thumbnail for: {title[:30]}")
            
            # No thumbnail found
            return result
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"❌ get_thumbnail_for_movie error for {title}: {e}")
            return result
    
    async def get_thumbnails_batch(
        self,
        movies: List[Dict]
    ) -> List[Dict[str, Any]]:
        """
        🚀 OPTIMIZED: Get thumbnails for multiple movies in batch
        - One MongoDB query for all
        - Returns results in same order
        """
        results = []
        
        if not movies:
            return results
        
        try:
            # Extract normalized titles
            normalized_titles = []
            for movie in movies:
                title = movie.get('title', '')
                if title:
                    normalized_titles.append(self._normalize_title(title))
            
            # Batch query MongoDB
            if self.thumbnails_col is not None and normalized_titles:
                cursor = self.thumbnails_col.find(
                    {
                        'normalized_title': {'$in': normalized_titles},
                        'has_thumbnail': True
                    },
                    {
                        'normalized_title': 1,
                        'thumbnail_url': 1,
                        'source': 1,
                        'extracted': 1,
                        'quality': 1,
                        'message_id': 1
                    }
                )
                
                thumbnails_map = {}
                async for doc in cursor:
                    thumbnails_map[doc['normalized_title']] = doc
            
            # Process each movie
            for movie in movies:
                try:
                    title = movie.get('title', '')
                    normalized = self._normalize_title(title)
                    
                    # Check if we have thumbnail in map
                    if normalized in thumbnails_map:
                        thumb_data = thumbnails_map[normalized]
                        results.append({
                            'thumbnail_url': thumb_data['thumbnail_url'],
                            'source': thumb_data.get('source', 'database'),
                            'extracted': thumb_data.get('extracted', True),
                            'quality': thumb_data.get('quality'),
                            'message_id': thumb_data.get('message_id')
                        })
                    else:
                        # No thumbnail found
                        results.append({
                            'thumbnail_url': None,
                            'source': 'none',
                            'extracted': False,
                            'quality': movie.get('quality'),
                            'message_id': movie.get('message_id')
                        })
                        
                except Exception as e:
                    logger.error(f"Batch item error: {e}")
                    results.append({
                        'thumbnail_url': None,
                        'source': 'error',
                        'extracted': False
                    })
            
        except Exception as e:
            logger.error(f"Batch thumbnail error: {e}")
            # Fill with None for all
            results = [{
                'thumbnail_url': None,
                'source': 'error',
                'extracted': False
            } for _ in movies]
        
        return results
    
    async def _extract_thumbnail(
        self, 
        channel_id: int, 
        message_id: int,
        title: str = ""
    ) -> Optional[str]:
        """Extract thumbnail using available clients"""
        
        # Try bot client first (priority)
        if self.bot_client:
            try:
                thumbnail_url = await self._extract_with_bot(channel_id, message_id)
                if thumbnail_url:
                    logger.debug(f"✅ Bot extracted: {title[:30]}")
                    return thumbnail_url
            except Exception as e:
                logger.debug(f"Bot extraction failed for {title[:30]}: {e}")
        
        # Try user client as fallback
        if self.user_client:
            try:
                thumbnail_url = await self._extract_with_user(channel_id, message_id)
                if thumbnail_url:
                    logger.debug(f"✅ User extracted: {title[:30]}")
                    return thumbnail_url
            except Exception as e:
                logger.debug(f"User extraction failed for {title[:30]}: {e}")
        
        return None
    
    async def _extract_with_bot(self, channel_id: int, message_id: int) -> Optional[str]:
        """Extract thumbnail using bot client"""
        try:
            if not self.bot_client:
                return None
            
            # Get message
            message = await self.bot_client.get_messages(channel_id, message_id)
            if not message:
                return None
            
            thumbnail_data = None
            
            # Video thumbnail
            if message.video and hasattr(message.video, 'thumbnail') and message.video.thumbnail:
                thumb_file_id = message.video.thumbnail.file_id
                thumbnail_data = await self._download_file(self.bot_client, thumb_file_id)
            
            # Document thumbnail
            elif message.document and hasattr(message.document, 'thumbnail') and message.document.thumbnail:
                thumb_file_id = message.document.thumbnail.file_id
                thumbnail_data = await self._download_file(self.bot_client, thumb_file_id)
            
            if thumbnail_data:
                # Convert to base64
                base64_data = base64.b64encode(thumbnail_data).decode('utf-8')
                return f"data:image/jpeg;base64,{base64_data}"
            
            return None
            
        except Exception as e:
            logger.error(f"Bot extraction error: {e}")
            return None
    
    async def _extract_with_user(self, channel_id: int, message_id: int) -> Optional[str]:
        """Extract thumbnail using user client"""
        try:
            if not self.user_client:
                return None
            
            # Get message
            message = await self.user_client.get_messages(channel_id, message_id)
            if not message:
                return None
            
            thumbnail_data = None
            
            # Video thumbnail
            if message.video and hasattr(message.video, 'thumbnail') and message.video.thumbnail:
                thumb_file_id = message.video.thumbnail.file_id
                thumbnail_data = await self._download_file(self.user_client, thumb_file_id)
            
            # Document thumbnail
            elif message.document and hasattr(message.document, 'thumbnail') and message.document.thumbnail:
                thumb_file_id = message.document.thumbnail.file_id
                thumbnail_data = await self._download_file(self.user_client, thumb_file_id)
            
            if thumbnail_data:
                # Convert to base64
                base64_data = base64.b64encode(thumbnail_data).decode('utf-8')
                return f"data:image/jpeg;base64,{base64_data}"
            
            return None
            
        except Exception as e:
            logger.error(f"User extraction error: {e}")
            return None
    
    async def _download_file(self, client, file_id: str) -> Optional[bytes]:
        """Download file from Telegram"""
        try:
            download_path = await client.download_media(file_id, in_memory=True)
            if not download_path:
                return None
            
            if isinstance(download_path, bytes):
                return download_path
            else:
                # If it's a file path, read it
                async with aiofiles.open(download_path, 'rb') as f:
                    return await f.read()
                    
        except Exception as e:
            logger.error(f"Download error: {e}")
            return None
    
    async def _save_thumbnail(
        self,
        title: str,
        normalized: str,
        thumbnail_url: Optional[str],
        source: str,
        channel_id: int = None,
        message_id: int = None,
        quality: str = None,
        extracted: bool = False
    ):
        """Save thumbnail info to database"""
        if self.thumbnails_col is None:
            return
        
        try:
            # Check if document exists
            existing = await self.thumbnails_col.find_one(
                {'normalized_title': normalized}
            )
            
            now = datetime.now()
            
            if existing:
                # Update existing
                update_data = {
                    'last_accessed': now,
                    'access_count': existing.get('access_count', 0) + 1
                }
                
                # Only update thumbnail if we have a new one
                if thumbnail_url and not existing.get('thumbnail_url'):
                    update_data['thumbnail_url'] = thumbnail_url
                    update_data['source'] = source
                    update_data['extracted'] = extracted
                    update_data['extracted_at'] = now
                
                await self.thumbnails_col.update_one(
                    {'_id': existing['_id']},
                    {'$set': update_data}
                )
                
            else:
                # Create new document
                doc = {
                    'title': title[:100],
                    'normalized_title': normalized,
                    'thumbnail_url': thumbnail_url,
                    'source': source,
                    'extracted': extracted,
                    'has_thumbnail': bool(thumbnail_url),
                    'channel_id': channel_id,
                    'message_id': message_id,
                    'quality': quality,
                    'extraction_attempts': 1,
                    'extraction_status': 'success' if thumbnail_url else 'failed',
                    'created_at': now,
                    'last_accessed': now,
                    'access_count': 1,
                    'expires_at': now + timedelta(days=30)  # 30 days TTL
                }
                
                await self.thumbnails_col.insert_one(doc)
            
            # Update cache if we have thumbnail
            if self.redis and thumbnail_url:
                cache_data = {
                    'thumbnail_url': thumbnail_url,
                    'source': source,
                    'extracted': extracted,
                    'quality': quality,
                    'message_id': message_id
                }
                await self.redis.setex(
                    f"thumb:{normalized}",
                    86400,  # 24 hours
                    json.dumps(cache_data)
                )
            
            logger.debug(f"💾 Saved thumbnail for: {title[:30]} - {source}")
            
        except Exception as e:
            logger.error(f"Save thumbnail error: {e}")
    
    async def _cleanup_old_thumbnails(self):
        """Background task to clean up old thumbnails"""
        logger.info("🧹 Starting thumbnail cleanup task...")
        
        while self.is_running:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                if self.thumbnails_col is None:
                    continue
                
                # Delete expired thumbnails
                result = await self.thumbnails_col.delete_many({
                    'expires_at': {'$lt': datetime.now()}
                })
                
                if result.deleted_count > 0:
                    logger.info(f"🧹 Cleaned up {result.deleted_count} expired thumbnails")
                
                # Clean up failed extraction attempts older than 7 days
                old_failed = await self.thumbnails_col.delete_many({
                    'extracted': False,
                    'created_at': {'$lt': datetime.now() - timedelta(days=7)}
                })
                
                if old_failed.deleted_count > 0:
                    logger.info(f"🧹 Cleaned up {old_failed.deleted_count} old failed attempts")
                
                # Also clean up physical files if any
                await self._cleanup_physical_files()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
    
    async def _cleanup_physical_files(self):
        """Clean up physical thumbnail files"""
        try:
            # Get list of files in download directory
            files = list(self.download_path.glob("*.jpg")) + list(self.download_path.glob("*.png"))
            
            # Check each file's age
            for file_path in files:
                if file_path.stat().st_mtime < (time.time() - 30 * 86400):  # 30 days
                    file_path.unlink()
                    logger.debug(f"🧹 Deleted old file: {file_path.name}")
                    
        except Exception as e:
            logger.error(f"Physical cleanup error: {e}")
    
    async def _rate_limit(self):
        """Rate limiting for Telegram requests"""
        now = time.time()
        time_since_last = now - self.last_request_time
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
    
    def _normalize_title(self, title: str) -> str:
        """Normalize title for consistent lookup"""
        if not title:
            return ""
        
        # Convert to lowercase
        normalized = title.lower().strip()
        
        # Remove special characters
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        # Remove extra spaces
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove year at the end
        normalized = re.sub(r'\s+\d{4}$', '', normalized)
        
        # Limit length
        return normalized[:100]
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get thumbnail manager statistics"""
        stats = self.stats.copy()
        
        # Calculate uptime
        start = datetime.fromisoformat(stats['start_time'])
        stats['uptime'] = str(datetime.now() - start)
        
        if self.thumbnails_col is not None:
            try:
                # Count total thumbnails
                stats['total_thumbnails'] = await self.thumbnails_col.count_documents({})
                
                # Count extracted vs failed
                stats['extracted_count'] = await self.thumbnails_col.count_documents({
                    'extracted': True
                })
                
                stats['failed_count'] = await self.thumbnails_col.count_documents({
                    'extracted': False
                })
                
                # Source distribution
                pipeline = [
                    {'$group': {'_id': '$source', 'count': {'$sum': 1}}}
                ]
                
                source_stats = {}
                async for doc in self.thumbnails_col.aggregate(pipeline):
                    source_stats[doc['_id']] = doc['count']
                
                stats['source_distribution'] = source_stats
                
                # Pending extractions
                stats['pending_extractions'] = await self.thumbnails_col.count_documents({
                    'extracted': False,
                    'extraction_attempts': {'$lt': 3}
                })
                
            except Exception as e:
                logger.error(f"Error getting stats: {e}")
        
        # Calculate success rate
        if stats['extraction_attempts'] > 0:
            stats['success_rate'] = f"{(stats['extraction_success'] / stats['extraction_attempts'] * 100):.1f}%"
        else:
            stats['success_rate'] = "0%"
        
        # Cache hit rate
        if stats['total_requests'] > 0:
            stats['cache_hit_rate'] = f"{(stats['cache_hits'] / stats['total_requests'] * 100):.1f}%"
        else:
            stats['cache_hit_rate'] = "0%"
        
        return stats
    
    async def preload_thumbnails_for_movies(self, movies: List[Dict]):
        """Preload thumbnails for a list of movies (background)"""
        logger.info(f"🔄 Preloading thumbnails for {len(movies)} movies...")
        
        for movie in movies:
            try:
                await self.get_thumbnail_for_movie(
                    title=movie.get('title', ''),
                    channel_id=movie.get('channel_id'),
                    message_id=movie.get('message_id')
                )
                await asyncio.sleep(0.5)  # Rate limiting
            except Exception as e:
                logger.error(f"Preload error: {e}")
        
        logger.info("✅ Thumbnail preloading complete")
    
    async def retry_failed_extractions(self, max_attempts: int = 3):
        """Retry failed thumbnail extractions"""
        logger.info("🔄 Retrying failed thumbnail extractions...")
        
        if self.thumbnails_col is None:
            return
        
        # Find failed extractions with less than max_attempts
        cursor = self.thumbnails_col.find({
            'extracted': False,
            'extraction_attempts': {'$lt': max_attempts},
            'channel_id': {'$ne': None},
            'message_id': {'$ne': None}
        })
        
        retry_count = 0
        success_count = 0
        
        async for doc in cursor:
            try:
                # Increment attempt count
                await self.thumbnails_col.update_one(
                    {'_id': doc['_id']},
                    {'$inc': {'extraction_attempts': 1}}
                )
                
                # Try extraction again
                result = await self.get_thumbnail_for_movie(
                    title=doc.get('title', ''),
                    channel_id=doc.get('channel_id'),
                    message_id=doc.get('message_id'),
                    quality=doc.get('quality'),
                    force_extract=True
                )
                
                if result.get('thumbnail_url'):
                    success_count += 1
                    logger.info(f"✅ Retry successful: {doc.get('title', '')[:30]}")
                
                retry_count += 1
                
                # Rate limiting
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Retry error: {e}")
        
        logger.info(f"✅ Retry complete: {success_count}/{retry_count} successful")
    
    async def shutdown(self):
        """Shutdown thumbnail manager"""
        logger.info("🛑 Shutting down Thumbnail Manager...")
        self.is_running = False
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Save final stats
        if self.stats_col is not None:
            try:
                self.stats['end_time'] = datetime.now().isoformat()
                await self.stats_col.insert_one({
                    'timestamp': datetime.now(),
                    'stats': self.stats
                })
            except:
                pass
        
        logger.info("✅ Thumbnail Manager shutdown complete")


# ============================================================================
# ✅ FALLBACK THUMBNAIL MANAGER (if imports fail)
# ============================================================================

class FallbackThumbnailManager:
    """Fallback when ThumbnailManager is not available"""
    
    def __init__(self, *args, **kwargs):
        self.stats = {
            'mode': 'fallback',
            'initialized': True
        }
        logger.warning("⚠️ Using FallbackThumbnailManager")
    
    async def initialize(self):
        logger.info("✅ FallbackThumbnailManager initialized")
        return True
    
    async def get_thumbnail_for_movie(self, title, *args, **kwargs):
        return {
            'thumbnail_url': FALLBACK_THUMBNAIL_URL,
            'source': 'fallback',
            'extracted': False,
            'quality': kwargs.get('quality'),
            'message_id': kwargs.get('message_id')
        }
    
    async def get_thumbnails_batch(self, movies):
        return [{
            'thumbnail_url': FALLBACK_THUMBNAIL_URL,
            'source': 'fallback',
            'extracted': False
        } for _ in movies]
    
    async def get_stats(self):
        return self.stats
    
    async def preload_thumbnails_for_movies(self, movies):
        logger.info(f"⚠️ Fallback: Would preload {len(movies)} movies")
    
    async def retry_failed_extractions(self, max_attempts=3):
        logger.info("⚠️ Fallback: Would retry failed extractions")
    
    async def shutdown(self):
        logger.info("✅ FallbackThumbnailManager shutdown")

# ============================================================================
# ✅ EXPORTS
# ============================================================================

__all__ = ['ThumbnailManager', 'FallbackThumbnailManager', 'FALLBACK_THUMBNAIL_URL']
