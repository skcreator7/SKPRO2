# ============================================================================
# 🚀 thumbnail_manager.py - COMPLETE THUMBNAIL EXTRACTION FOR ALL FILES
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
# ✅ THUMBNAIL MANAGER - EXTRACT ALL FILES THUMBNAILS
# ============================================================================

class ThumbnailManager:
    """
    🚀 COMPLETE THUMBNAIL EXTRACTION
    - Extracts thumbnails for ALL files in Telegram channel
    - Stores in MongoDB for fast retrieval
    - Redis caching for speed
    - Background extraction with rate limiting
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
            'start_time': datetime.now().isoformat(),
            'total_files_scanned': 0,
            'files_with_thumbnails': 0,
            'files_without_thumbnails': 0,
            'unique_movies': 0
        }
        
        # Background tasks
        self.cleanup_task = None
        self.is_running = False
        self.extraction_task = None
        
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
            logger.info("🚀 THUMBNAIL MANAGER: COMPLETE EXTRACTION")
            logger.info("=" * 60)
            logger.info("    • Extract ALL files: ✅ ENABLED")
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
            
            # Index for quality
            await self.thumbnails_col.create_index(
                [("normalized_title", 1), ("quality", 1)],
                name="title_quality",
                background=True
            )
            
            logger.info("✅ Thumbnail indexes created successfully")
            
        except Exception as e:
            logger.warning(f"⚠️ Thumbnail index creation error: {e}")
    
    async def extract_all_thumbnails_from_channel(self, force_re_extract: bool = False):
        """
        🚀 Extract thumbnails for ALL files in the file channel
        - Scans entire channel
        - Groups by movie title
        - Extracts thumbnails for each quality
        - Stores in MongoDB
        """
        if not self.user_client and not self.bot_client:
            logger.error("❌ No Telegram client available for thumbnail extraction")
            return False
        
        logger.info("=" * 60)
        logger.info("🚀 STARTING COMPLETE THUMBNAIL EXTRACTION")
        logger.info("=" * 60)
        logger.info(f"📁 Channel ID: {self.file_channel_id}")
        logger.info(f"🔄 Force re-extract: {force_re_extract}")
        logger.info("=" * 60)
        
        # Use user client if available (better for large channels)
        client = self.user_client if self.user_client else self.bot_client
        
        try:
            # Get channel info
            chat = await client.get_chat(self.file_channel_id)
            logger.info(f"📢 Channel: {chat.title}")
            
            # Scan all messages
            all_messages = []
            offset_id = 0
            batch_size = 200
            empty_batch_count = 0
            max_empty_batches = 3
            
            logger.info("📥 Scanning file channel for all messages...")
            
            while self.is_running:
                try:
                    messages = []
                    async for msg in client.get_chat_history(
                        self.file_channel_id,
                        limit=batch_size,
                        offset_id=offset_id
                    ):
                        messages.append(msg)
                        if len(messages) >= batch_size:
                            break
                    
                    if not messages:
                        empty_batch_count += 1
                        if empty_batch_count >= max_empty_batches:
                            logger.info("✅ No more messages to scan")
                            break
                        await asyncio.sleep(1)
                        continue
                    
                    empty_batch_count = 0
                    all_messages.extend(messages)
                    offset_id = messages[-1].id
                    
                    logger.info(f"📥 Scanned {len(all_messages)} messages so far...")
                    
                    if len(messages) < batch_size:
                        logger.info("✅ Reached end of channel")
                        break
                    
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"❌ Error scanning messages: {e}")
                    await asyncio.sleep(2)
                    continue
            
            logger.info(f"✅ Total messages scanned: {len(all_messages)}")
            self.stats['total_files_scanned'] = len(all_messages)
            
            # Group by movie and quality
            movies_dict = {}
            video_count = 0
            files_with_thumbnails = 0
            files_without_thumbnails = 0
            
            for msg in all_messages:
                if not msg or (not msg.document and not msg.video):
                    continue
                
                # Get file info
                file_name = None
                if msg.document:
                    file_name = msg.document.file_name
                elif msg.video:
                    file_name = msg.video.file_name or "video.mp4"
                
                if not file_name or not self._is_video_file(file_name):
                    continue
                
                video_count += 1
                
                # Check if message has thumbnail
                has_thumb = self._has_telegram_thumbnail(msg)
                
                if has_thumb:
                    files_with_thumbnails += 1
                else:
                    files_without_thumbnails += 1
                
                # Extract movie info
                clean_title = self._extract_clean_title(file_name)
                normalized = self._normalize_title(clean_title)
                quality = self._detect_quality_enhanced(file_name)
                year = self._extract_year(file_name)
                
                # Create file entry
                file_entry = {
                    'file_name': file_name,
                    'file_size': msg.document.file_size if msg.document else msg.video.file_size,
                    'file_size_formatted': self._format_size(msg.document.file_size if msg.document else msg.video.file_size),
                    'message_id': msg.id,
                    'file_id': msg.document.file_id if msg.document else msg.video.file_id,
                    'quality': quality,
                    'has_thumbnail_in_telegram': has_thumb,
                    'thumbnail_extracted': False,
                    'thumbnail_url': None,
                    'date': msg.date,
                    'channel_id': self.file_channel_id
                }
                
                # Group by movie
                if normalized not in movies_dict:
                    movies_dict[normalized] = {
                        'title': clean_title,
                        'original_title': clean_title,
                        'normalized_title': normalized,
                        'year': year,
                        'qualities': {},
                        'available_qualities': [],
                        'qualities_with_thumbnails': [],
                        'total_files': 0,
                        'files_with_thumbnails': 0
                    }
                
                movies_dict[normalized]['qualities'][quality] = file_entry
                movies_dict[normalized]['available_qualities'].append(quality)
                movies_dict[normalized]['total_files'] += 1
                
                if has_thumb:
                    movies_dict[normalized]['files_with_thumbnails'] += 1
                    movies_dict[normalized]['qualities_with_thumbnails'].append(quality)
            
            self.stats['files_with_thumbnails'] = files_with_thumbnails
            self.stats['files_without_thumbnails'] = files_without_thumbnails
            self.stats['unique_movies'] = len(movies_dict)
            
            logger.info("=" * 60)
            logger.info("📊 SCANNING COMPLETE")
            logger.info(f"   • Total video files: {video_count}")
            logger.info(f"   • Files WITH thumbnails: {files_with_thumbnails}")
            logger.info(f"   • Files WITHOUT thumbnails: {files_without_thumbnails}")
            logger.info(f"   • Unique movies: {len(movies_dict)}")
            logger.info("=" * 60)
            
            # Start background extraction for all files with thumbnails
            if files_with_thumbnails > 0:
                logger.info("🔄 Starting background thumbnail extraction...")
                self.extraction_task = asyncio.create_task(
                    self._extract_all_thumbnails_background(movies_dict, force_re_extract)
                )
                return True
            else:
                logger.warning("⚠️ No files with thumbnails found")
                return False
                
        except Exception as e:
            logger.error(f"❌ Complete extraction error: {e}")
            return False
    
    async def _extract_all_thumbnails_background(self, movies_dict: Dict, force_re_extract: bool):
        """Background mein saare thumbnails extract karo"""
        logger.info(f"🔄 Background extraction started for {len(movies_dict)} movies...")
        
        client = self.user_client if self.user_client else self.bot_client
        total_successful = 0
        total_failed = 0
        total_skipped = 0
        
        # Process each movie
        for normalized, movie in movies_dict.items():
            if not self.is_running:
                break
            
            # Process each quality
            for quality, file_data in movie['qualities'].items():
                if not file_data.get('has_thumbnail_in_telegram'):
                    continue
                
                # Check if already extracted (unless force re-extract)
                if not force_re_extract:
                    existing = await self.thumbnails_col.find_one({
                        'normalized_title': normalized,
                        'quality': quality,
                        'has_thumbnail': True
                    })
                    if existing:
                        total_skipped += 1
                        continue
                
                try:
                    # Extract thumbnail
                    thumbnail_url = await self._extract_single_thumbnail(
                        client,
                        self.file_channel_id,
                        file_data['message_id']
                    )
                    
                    if thumbnail_url:
                        # Save to MongoDB
                        await self._save_extracted_thumbnail(
                            normalized=normalized,
                            title=movie['title'],
                            quality=quality,
                            thumbnail_url=thumbnail_url,
                            channel_id=self.file_channel_id,
                            message_id=file_data['message_id'],
                            file_data=file_data
                        )
                        
                        total_successful += 1
                        logger.info(f"✅ Extracted: {movie['title']} - {quality}")
                        
                        # Update cache
                        if self.redis:
                            cache_key = f"thumb:{normalized}:{quality}"
                            await self.redis.setex(
                                cache_key,
                                86400,  # 24 hours
                                thumbnail_url
                            )
                    else:
                        total_failed += 1
                        logger.warning(f"⚠️ Failed: {movie['title']} - {quality}")
                    
                except Exception as e:
                    logger.error(f"❌ Error extracting {movie['title']} - {quality}: {e}")
                    total_failed += 1
                
                # Rate limiting
                await asyncio.sleep(self.min_request_interval)
            
            # Small delay between movies
            await asyncio.sleep(1)
        
        logger.info("=" * 60)
        logger.info("✅ BACKGROUND EXTRACTION COMPLETE")
        logger.info(f"   • Successful: {total_successful}")
        logger.info(f"   • Failed: {total_failed}")
        logger.info(f"   • Skipped (already exist): {total_skipped}")
        logger.info("=" * 60)
        
        self.stats['extraction_success'] += total_successful
        self.stats['extraction_failed'] += total_failed
        
        return {
            'successful': total_successful,
            'failed': total_failed,
            'skipped': total_skipped
        }
    
    async def _extract_single_thumbnail(self, client, channel_id: int, message_id: int) -> Optional[str]:
        """Extract single thumbnail from message"""
        try:
            message = await client.get_messages(channel_id, message_id)
            if not message:
                return None
            
            thumbnail_data = None
            
            # Video thumbnail
            if message.video and hasattr(message.video, 'thumbnail') and message.video.thumbnail:
                thumb_file_id = message.video.thumbnail.file_id
                thumbnail_data = await self._download_file(client, thumb_file_id)
            
            # Document thumbnail
            elif message.document and hasattr(message.document, 'thumbnail') and message.document.thumbnail:
                thumb_file_id = message.document.thumbnail.file_id
                thumbnail_data = await self._download_file(client, thumb_file_id)
            
            if thumbnail_data:
                # Convert to base64
                base64_data = base64.b64encode(thumbnail_data).decode('utf-8')
                return f"data:image/jpeg;base64,{base64_data}"
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Single extraction error: {e}")
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
                async with aiofiles.open(download_path, 'rb') as f:
                    return await f.read()
                    
        except Exception as e:
            logger.error(f"❌ Download error: {e}")
            return None
    
    async def _save_extracted_thumbnail(
        self,
        normalized: str,
        title: str,
        quality: str,
        thumbnail_url: str,
        channel_id: int,
        message_id: int,
        file_data: Dict
    ):
        """Save extracted thumbnail to MongoDB"""
        if self.thumbnails_col is None:
            return
        
        try:
            now = datetime.now()
            
            # Check if exists
            existing = await self.thumbnails_col.find_one({
                'normalized_title': normalized,
                'quality': quality
            })
            
            doc = {
                'normalized_title': normalized,
                'title': title[:100],
                'quality': quality,
                'thumbnail_url': thumbnail_url,
                'has_thumbnail': True,
                'source': 'extracted',
                'channel_id': channel_id,
                'message_id': message_id,
                'file_name': file_data.get('file_name'),
                'file_size': file_data.get('file_size'),
                'extracted_at': now,
                'last_accessed': now,
                'access_count': 1,
                'expires_at': now + timedelta(days=30)
            }
            
            if existing:
                # Update
                await self.thumbnails_col.update_one(
                    {'_id': existing['_id']},
                    {'$set': {
                        'thumbnail_url': thumbnail_url,
                        'has_thumbnail': True,
                        'extracted_at': now,
                        'last_accessed': now,
                        '$inc': {'access_count': 1}
                    }}
                )
            else:
                # Insert
                await self.thumbnails_col.insert_one(doc)
            
            logger.debug(f"💾 Saved thumbnail: {title[:30]} - {quality}")
            
        except Exception as e:
            logger.error(f"❌ Save thumbnail error: {e}")
    
    async def get_thumbnail_for_quality(
        self,
        title: str,
        quality: str
    ) -> Optional[str]:
        """Get thumbnail for specific movie quality"""
        normalized = self._normalize_title(title)
        
        # Check Redis first
        if self.redis:
            cache_key = f"thumb:{normalized}:{quality}"
            cached = await self.redis.get(cache_key)
            if cached:
                self.stats['cache_hits'] += 1
                return cached.decode() if isinstance(cached, bytes) else cached
        
        # Check MongoDB
        if self.thumbnails_col:
            doc = await self.thumbnails_col.find_one({
                'normalized_title': normalized,
                'quality': quality,
                'has_thumbnail': True
            })
            
            if doc and doc.get('thumbnail_url'):
                self.stats['db_hits'] += 1
                
                # Update cache
                if self.redis:
                    await self.redis.setex(
                        f"thumb:{normalized}:{quality}",
                        86400,
                        doc['thumbnail_url']
                    )
                
                return doc['thumbnail_url']
        
        return None
    
    async def get_all_thumbnails_for_movie(self, title: str) -> Dict[str, str]:
        """Get all quality thumbnails for a movie"""
        normalized = self._normalize_title(title)
        result = {}
        
        if self.thumbnails_col:
            cursor = self.thumbnails_col.find({
                'normalized_title': normalized,
                'has_thumbnail': True
            })
            
            async for doc in cursor:
                result[doc['quality']] = doc['thumbnail_url']
        
        return result
    
    async def get_thumbnail_for_movie(
        self,
        title: str,
        channel_id: int = None,
        message_id: int = None,
        quality: str = None,
        force_extract: bool = False
    ) -> Dict[str, Any]:
        """
        Get best thumbnail for movie
        If quality specified, try that first, then fallback to any
        """
        normalized = self._normalize_title(title)
        
        result = {
            'thumbnail_url': None,
            'source': 'none',
            'extracted': False,
            'quality': quality
        }
        
        try:
            # If quality specified, try that first
            if quality:
                thumb_url = await self.get_thumbnail_for_quality(title, quality)
                if thumb_url:
                    result.update({
                        'thumbnail_url': thumb_url,
                        'source': 'extracted',
                        'extracted': True,
                        'quality': quality
                    })
                    return result
            
            # Try any quality
            if self.thumbnails_col:
                doc = await self.thumbnails_col.find_one({
                    'normalized_title': normalized,
                    'has_thumbnail': True
                })
                
                if doc and doc.get('thumbnail_url'):
                    result.update({
                        'thumbnail_url': doc['thumbnail_url'],
                        'source': 'extracted',
                        'extracted': True,
                        'quality': doc.get('quality')
                    })
                    return result
            
            # Fallback
            result['thumbnail_url'] = FALLBACK_THUMBNAIL_URL
            result['source'] = 'fallback'
            return result
            
        except Exception as e:
            logger.error(f"❌ get_thumbnail_for_movie error: {e}")
            result['thumbnail_url'] = FALLBACK_THUMBNAIL_URL
            result['source'] = 'fallback'
            return result
    
    async def get_thumbnails_batch(self, movies: List[Dict]) -> List[Dict]:
        """Get thumbnails for multiple movies in batch"""
        results = []
        
        if not movies or not self.thumbnails_col:
            return [{'thumbnail_url': FALLBACK_THUMBNAIL_URL, 'source': 'fallback'} for _ in movies]
        
        try:
            # Get all normalized titles
            normalized_titles = []
            title_map = {}
            
            for i, movie in enumerate(movies):
                title = movie.get('title', '')
                if title:
                    norm = self._normalize_title(title)
                    normalized_titles.append(norm)
                    title_map[norm] = i
            
            # Batch query
            cursor = self.thumbnails_col.find({
                'normalized_title': {'$in': normalized_titles},
                'has_thumbnail': True
            })
            
            thumb_map = {}
            async for doc in cursor:
                norm = doc['normalized_title']
                if norm not in thumb_map:
                    thumb_map[norm] = {
                        'thumbnail_url': doc['thumbnail_url'],
                        'source': 'extracted',
                        'extracted': True,
                        'quality': doc.get('quality')
                    }
            
            # Build results
            for i, movie in enumerate(movies):
                norm = self._normalize_title(movie.get('title', ''))
                if norm in thumb_map:
                    results.append(thumb_map[norm])
                else:
                    results.append({
                        'thumbnail_url': FALLBACK_THUMBNAIL_URL,
                        'source': 'fallback',
                        'extracted': False
                    })
            
        except Exception as e:
            logger.error(f"Batch thumbnail error: {e}")
            results = [{
                'thumbnail_url': FALLBACK_THUMBNAIL_URL,
                'source': 'fallback',
                'extracted': False
            } for _ in movies]
        
        return results
    
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
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
    
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
        
        normalized = title.lower().strip()
        normalized = re.sub(r'[^\w\s]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = re.sub(r'\s+\d{4}$', '', normalized)
        return normalized[:100]
    
    def _extract_clean_title(self, filename: str) -> str:
        """Extract clean movie title from filename"""
        if not filename:
            return "Unknown"
        
        name = os.path.splitext(filename)[0]
        name = re.sub(r'[._\-]', ' ', name)
        name = re.sub(r'\b(480p|720p|1080p|2160p|4k|hd|hevc|x264|x265|web-dl|webrip|bluray|hdtv|hdr|dts|ac3|aac|ddp|5\.1|7\.1|2\.0|esub|sub|multi|dual|audio|hindi|english|tamil|telugu|malayalam|kannada|ben|eng|hin|tam|tel|mal|kan)\b.*$', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\s+(19|20)\d{2}\s*$', '', name)
        name = re.sub(r'\s*\([^)]*\)', '', name)
        name = re.sub(r'\s*\[[^\]]*\]', '', name)
        name = re.sub(r'\s+', ' ', name)
        return name.strip() or "Unknown"
    
    def _extract_year(self, filename: str) -> str:
        """Extract year from filename"""
        if not filename:
            return ""
        match = re.search(r'\b(19|20)\d{2}\b', filename)
        return match.group() if match else ""
    
    def _detect_quality_enhanced(self, filename: str) -> str:
        """Detect quality from filename"""
        if not filename:
            return "480p"
        
        filename_lower = filename.lower()
        
        quality_patterns = [
            (r'\b2160p\b|\b4k\b|\buhd\b', '2160p'),
            (r'\b1080p\b|\bfullhd\b|\bfhd\b', '1080p'),
            (r'\b720p\b|\bhd\b', '720p'),
            (r'\b480p\b', '480p'),
            (r'\b360p\b', '360p'),
        ]
        
        hevc_patterns = [r'\bhevc\b', r'\bx265\b', r'\bh\.?265\b']
        is_hevc = any(re.search(p, filename_lower) for p in hevc_patterns)
        
        for pattern, quality in quality_patterns:
            if re.search(pattern, filename_lower):
                if is_hevc and quality in ['720p', '1080p', '2160p']:
                    return f"{quality} HEVC"
                return quality
        
        return "480p"
    
    def _is_video_file(self, filename: str) -> bool:
        """Check if file is video"""
        if not filename:
            return False
        video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v']
        return any(filename.lower().endswith(ext) for ext in video_extensions)
    
    def _has_telegram_thumbnail(self, message) -> bool:
        """Check if message has thumbnail in Telegram"""
        try:
            if message.video and hasattr(message.video, 'thumbnail') and message.video.thumbnail:
                return True
            elif message.document and hasattr(message.document, 'thumbnail') and message.document.thumbnail:
                return True
            return False
        except:
            return False
    
    def _format_size(self, size: int) -> str:
        """Format file size"""
        if not size:
            return "Unknown"
        if size < 1024:
            return f"{size} B"
        elif size < 1024*1024:
            return f"{size/1024:.1f} KB"
        elif size < 1024*1024*1024:
            return f"{size/1024/1024:.1f} MB"
        else:
            return f"{size/1024/1024/1024:.2f} GB"
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get thumbnail manager statistics"""
        stats = self.stats.copy()
        
        start = datetime.fromisoformat(stats['start_time'])
        stats['uptime'] = str(datetime.now() - start)
        
        if self.thumbnails_col is not None:
            try:
                stats['total_thumbnails'] = await self.thumbnails_col.count_documents({})
                stats['extracted_count'] = await self.thumbnails_col.count_documents({'has_thumbnail': True})
                stats['failed_count'] = await self.thumbnails_col.count_documents({'has_thumbnail': False})
                
                # Quality distribution
                pipeline = [{'$group': {'_id': '$quality', 'count': {'$sum': 1}}}]
                quality_stats = {}
                async for doc in self.thumbnails_col.aggregate(pipeline):
                    quality_stats[doc['_id']] = doc['count']
                stats['quality_distribution'] = quality_stats
                
            except Exception as e:
                logger.error(f"Error getting stats: {e}")
        
        # Success rate
        if stats['extraction_attempts'] > 0:
            stats['success_rate'] = f"{(stats['extraction_success'] / stats['extraction_attempts'] * 100):.1f}%"
        else:
            stats['success_rate'] = "0%"
        
        return stats
    
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
        
        if self.extraction_task and not self.extraction_task.done():
            self.extraction_task.cancel()
            try:
                await self.extraction_task
            except asyncio.CancelledError:
                pass
        
        logger.info("✅ Thumbnail Manager shutdown complete")


# ============================================================================
# ✅ FALLBACK THUMBNAIL MANAGER
# ============================================================================

class FallbackThumbnailManager:
    """Fallback when ThumbnailManager is not available"""
    
    def __init__(self, *args, **kwargs):
        self.stats = {'mode': 'fallback', 'initialized': True}
        logger.warning("⚠️ Using FallbackThumbnailManager")
    
    async def initialize(self):
        logger.info("✅ FallbackThumbnailManager initialized")
        return True
    
    async def extract_all_thumbnails_from_channel(self, force_re_extract=False):
        logger.info("⚠️ Fallback: Would extract all thumbnails")
        return False
    
    async def get_thumbnail_for_movie(self, title, *args, **kwargs):
        return {
            'thumbnail_url': FALLBACK_THUMBNAIL_URL,
            'source': 'fallback',
            'extracted': False,
            'quality': kwargs.get('quality')
        }
    
    async def get_thumbnails_batch(self, movies):
        return [{
            'thumbnail_url': FALLBACK_THUMBNAIL_URL,
            'source': 'fallback',
            'extracted': False
        } for _ in movies]
    
    async def get_thumbnail_for_quality(self, title, quality):
        return FALLBACK_THUMBNAIL_URL
    
    async def get_all_thumbnails_for_movie(self, title):
        return {}
    
    async def get_stats(self):
        return self.stats
    
    async def shutdown(self):
        logger.info("✅ FallbackThumbnailManager shutdown")

# ============================================================================
# ✅ EXPORTS
# ============================================================================

__all__ = ['ThumbnailManager', 'FallbackThumbnailManager', 'FALLBACK_THUMBNAIL_URL']
