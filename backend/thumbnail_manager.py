# ============================================================================
# 🚀 thumbnail_manager.py - COMPLETE THUMBNAIL EXTRACTION FROM METADATA
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
# ✅ THUMBNAIL MANAGER - EXTRACT FROM MESSAGE METADATA
# ============================================================================

class ThumbnailManager:
    """
    🚀 COMPLETE THUMBNAIL EXTRACTION
    - Extracts thumbnails from message metadata (video.thumbs, document.thumbs)
    - Stores in MongoDB with quality-specific keys
    - Redis caching for fast access
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
        self.files_col = None  # Reference to files collection
        
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
            'unique_movies': 0,
            'thumbnails_extracted': 0
        }
        
        # Background tasks
        self.cleanup_task = None
        self.is_running = False
        self.extraction_task = None
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests
        
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
                self.files_col = db.files  # Reference to files collection
                
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
            logger.info("    • Extract from metadata: ✅ ENABLED")
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
            
            # Unique index for normalized_title + quality
            await self.thumbnails_col.create_index(
                [("normalized_title", 1), ("quality", 1)],
                unique=True,
                name="title_quality_unique",
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
                background=True
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
    
    async def scan_and_extract_all(self, force_re_extract: bool = False):
        """
        🚀 SCAN ENTIRE CHANNEL AND EXTRACT ALL THUMBNAILS
        - Scans every message in file channel
        - Checks for thumbnails in metadata (video.thumbs, document.thumbs)
        - Extracts and saves to MongoDB
        """
        if self.user_client is None and self.bot_client is None:
            logger.error("❌ No Telegram client available for thumbnail extraction")
            return False
        
        logger.info("=" * 60)
        logger.info("🚀 SCANNING CHANNEL FOR THUMBNAILS")
        logger.info("=" * 60)
        logger.info(f"📁 Channel ID: {self.file_channel_id}")
        logger.info(f"🔄 Force re-extract: {force_re_extract}")
        logger.info("=" * 60)
        
        # Use user client if available
        client = self.user_client if self.user_client is not None else self.bot_client
        
        try:
            # Get channel info
            chat = await client.get_chat(self.file_channel_id)
            logger.info(f"📢 Channel: {chat.title}")
            
            # Scan all messages
            all_messages = []
            offset_id = 0
            batch_size = 100  # Smaller batch size for rate limiting
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
                        await asyncio.sleep(2)
                        continue
                    
                    empty_batch_count = 0
                    all_messages.extend(messages)
                    offset_id = messages[-1].id
                    
                    logger.info(f"📥 Scanned {len(all_messages)} messages so far...")
                    self.stats['total_files_scanned'] = len(all_messages)
                    
                    if len(messages) < batch_size:
                        logger.info("✅ Reached end of channel")
                        break
                    
                    await asyncio.sleep(1)  # Rate limiting
                    
                except Exception as e:
                    logger.error(f"❌ Error scanning messages: {e}")
                    await asyncio.sleep(5)
                    continue
            
            logger.info(f"✅ Total messages scanned: {len(all_messages)}")
            
            # Process each message for thumbnails
            video_count = 0
            doc_count = 0
            with_thumb = 0
            without_thumb = 0
            extracted = 0
            failed = 0
            skipped = 0
            
            # Group by movie for better organization
            movies_dict = {}
            
            for msg in all_messages:
                if msg is None:
                    continue
                
                # ============ VIDEO FILES ============
                if hasattr(msg, 'video') and msg.video is not None:
                    video_count += 1
                    file_name = getattr(msg.video, 'file_name', f"video_{msg.id}.mp4")
                    
                    if not self._is_video_file(file_name):
                        continue
                    
                    # Check for thumbnail in metadata
                    has_thumb = False
                    if hasattr(msg.video, 'thumbs') and msg.video.thumbs:
                        has_thumb = len(msg.video.thumbs) > 0
                    
                    if has_thumb:
                        with_thumb += 1
                        
                        # Extract movie info
                        clean_title = self._extract_clean_title(file_name)
                        normalized = self._normalize_title(clean_title)
                        quality = self._detect_quality_enhanced(file_name)
                        year = self._extract_year(file_name)
                        
                        # Add to movies dict
                        if normalized not in movies_dict:
                            movies_dict[normalized] = {
                                'title': clean_title,
                                'normalized_title': normalized,
                                'year': year,
                                'qualities': {}
                            }
                        
                        movies_dict[normalized]['qualities'][quality] = {
                            'channel_id': self.file_channel_id,
                            'message_id': msg.id,
                            'file_name': file_name,
                            'file_size': getattr(msg.video, 'file_size', 0),
                            'has_thumbnail': True,
                            'thumb_file_id': msg.video.thumbs[0].file_id if msg.video.thumbs else None
                        }
                    else:
                        without_thumb += 1
                
                # ============ DOCUMENT FILES ============
                elif hasattr(msg, 'document') and msg.document is not None:
                    doc_count += 1
                    file_name = getattr(msg.document, 'file_name', f"doc_{msg.id}.bin")
                    
                    if not self._is_video_file(file_name):
                        continue
                    
                    # Check for thumbnail in metadata
                    has_thumb = False
                    if hasattr(msg.document, 'thumbs') and msg.document.thumbs:
                        has_thumb = len(msg.document.thumbs) > 0
                    
                    if has_thumb:
                        with_thumb += 1
                        
                        # Extract movie info
                        clean_title = self._extract_clean_title(file_name)
                        normalized = self._normalize_title(clean_title)
                        quality = self._detect_quality_enhanced(file_name)
                        year = self._extract_year(file_name)
                        
                        # Add to movies dict
                        if normalized not in movies_dict:
                            movies_dict[normalized] = {
                                'title': clean_title,
                                'normalized_title': normalized,
                                'year': year,
                                'qualities': {}
                            }
                        
                        movies_dict[normalized]['qualities'][quality] = {
                            'channel_id': self.file_channel_id,
                            'message_id': msg.id,
                            'file_name': file_name,
                            'file_size': getattr(msg.document, 'file_size', 0),
                            'has_thumbnail': True,
                            'thumb_file_id': msg.document.thumbs[0].file_id if msg.document.thumbs else None
                        }
                    else:
                        without_thumb += 1
            
            self.stats['files_with_thumbnails'] = with_thumb
            self.stats['files_without_thumbnails'] = without_thumb
            self.stats['unique_movies'] = len(movies_dict)
            
            logger.info("=" * 60)
            logger.info("📊 SCANNING COMPLETE")
            logger.info(f"   • Total video files: {video_count}")
            logger.info(f"   • Total document files: {doc_count}")
            logger.info(f"   • Files WITH thumbnails: {with_thumb}")
            logger.info(f"   • Files WITHOUT thumbnails: {without_thumb}")
            logger.info(f"   • Unique movies: {len(movies_dict)}")
            logger.info("=" * 60)
            
            # Start extraction for files with thumbnails
            if with_thumb > 0:
                logger.info(f"🔄 Starting thumbnail extraction for {with_thumb} files...")
                self.extraction_task = asyncio.create_task(
                    self._extract_all_thumbnails(movies_dict, client, force_re_extract)
                )
                return True
            else:
                logger.warning("⚠️ No files with thumbnails found in channel")
                logger.info("💡 Telegram channel mein kisi file ke paas thumbnail nahi hai!")
                return False
                
        except Exception as e:
            logger.error(f"❌ Scan and extract error: {e}")
            return False
    
    async def _extract_all_thumbnails(self, movies_dict: Dict, client, force_re_extract: bool):
        """Extract thumbnails for all movies with thumbnails"""
        logger.info(f"🔄 Starting thumbnail extraction for {len(movies_dict)} movies...")
        
        total_successful = 0
        total_failed = 0
        total_skipped = 0
        
        for normalized, movie in movies_dict.items():
            if not self.is_running:
                break
            
            for quality, file_data in movie['qualities'].items():
                # Check if already extracted
                if not force_re_extract and self.thumbnails_col is not None:
                    existing = await self.thumbnails_col.find_one({
                        'normalized_title': normalized,
                        'quality': quality,
                        'has_thumbnail': True
                    })
                    if existing is not None:
                        total_skipped += 1
                        continue
                
                try:
                    # Extract thumbnail using file_id
                    thumb_file_id = file_data.get('thumb_file_id')
                    if not thumb_file_id:
                        continue
                    
                    # Download thumbnail
                    thumb_data = await self._download_file(client, thumb_file_id)
                    
                    if thumb_data:
                        # Convert to base64
                        if isinstance(thumb_data, bytes):
                            base64_data = base64.b64encode(thumb_data).decode('utf-8')
                            thumb_url = f"data:image/jpeg;base64,{base64_data}"
                        else:
                            async with aiofiles.open(thumb_data, 'rb') as f:
                                file_bytes = await f.read()
                                base64_data = base64.b64encode(file_bytes).decode('utf-8')
                                thumb_url = f"data:image/jpeg;base64,{base64_data}"
                        
                        # Save to database
                        await self._save_thumbnail(
                            title=movie['title'],
                            normalized=normalized,
                            thumbnail_url=thumb_url,
                            source='extracted',
                            channel_id=self.file_channel_id,
                            message_id=file_data['message_id'],
                            quality=quality,
                            extracted=True
                        )
                        
                        # Update files collection if available
                        if self.files_col is not None:
                            await self.files_col.update_one(
                                {
                                    'normalized_title': normalized,
                                    'channel_id': self.file_channel_id
                                },
                                {
                                    '$set': {
                                        f'qualities.{quality}.thumbnail_url': thumb_url,
                                        f'qualities.{quality}.thumbnail_extracted': True,
                                        f'qualities.{quality}.thumbnail_extracted_at': datetime.now()
                                    }
                                }
                            )
                        
                        total_successful += 1
                        self.stats['extraction_success'] += 1
                        self.stats['thumbnails_extracted'] += 1
                        
                        logger.info(f"✅ Extracted: {movie['title'][:30]} - {quality}")
                    else:
                        total_failed += 1
                        self.stats['extraction_failed'] += 1
                        logger.warning(f"⚠️ Failed: {movie['title'][:30]} - {quality}")
                    
                except Exception as e:
                    logger.error(f"❌ Error extracting {movie['title']} - {quality}: {e}")
                    total_failed += 1
                    self.stats['extraction_failed'] += 1
                
                # Rate limiting
                await asyncio.sleep(self.min_request_interval)
            
            # Small delay between movies
            await asyncio.sleep(2)
        
        logger.info("=" * 60)
        logger.info("✅ EXTRACTION COMPLETE")
        logger.info(f"   • Successful: {total_successful}")
        logger.info(f"   • Failed: {total_failed}")
        logger.info(f"   • Skipped (already exist): {total_skipped}")
        logger.info("=" * 60)
        
        return {
            'successful': total_successful,
            'failed': total_failed,
            'skipped': total_skipped
        }
    
    async def _extract_single_thumbnail(self, client, channel_id: int, message_id: int) -> Optional[str]:
        """Extract single thumbnail from message metadata"""
        try:
            message = await client.get_messages(channel_id, message_id)
            if message is None:
                return None
            
            thumb_file_id = None
            
            # Video thumbnail
            if hasattr(message, 'video') and message.video is not None:
                if hasattr(message.video, 'thumbs') and message.video.thumbs:
                    thumb_file_id = message.video.thumbs[0].file_id
            
            # Document thumbnail
            elif hasattr(message, 'document') and message.document is not None:
                if hasattr(message.document, 'thumbs') and message.document.thumbs:
                    thumb_file_id = message.document.thumbs[0].file_id
            
            if not thumb_file_id:
                return None
            
            # Download thumbnail
            thumb_data = await self._download_file(client, thumb_file_id)
            
            if thumb_data:
                if isinstance(thumb_data, bytes):
                    base64_data = base64.b64encode(thumb_data).decode('utf-8')
                    return f"data:image/jpeg;base64,{base64_data}"
                else:
                    async with aiofiles.open(thumb_data, 'rb') as f:
                        file_bytes = await f.read()
                        base64_data = base64.b64encode(file_bytes).decode('utf-8')
                        return f"data:image/jpeg;base64,{base64_data}"
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Single extraction error: {e}")
            return None
    
    async def _download_file(self, client, file_id: str) -> Optional[bytes]:
        """Download file from Telegram"""
        try:
            download_path = await client.download_media(file_id, in_memory=True)
            if download_path is None:
                return None
            
            if isinstance(download_path, bytes):
                return download_path
            else:
                async with aiofiles.open(download_path, 'rb') as f:
                    return await f.read()
                    
        except Exception as e:
            logger.error(f"❌ Download error: {e}")
            return None
    
    async def _save_thumbnail(
        self,
        title: str,
        normalized: str,
        thumbnail_url: str,
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
            now = datetime.now()
            
            # Check if exists
            existing = await self.thumbnails_col.find_one({
                'normalized_title': normalized,
                'quality': quality
            })
            
            doc = {
                'title': title[:100],
                'normalized_title': normalized,
                'quality': quality,
                'thumbnail_url': thumbnail_url,
                'source': source,
                'extracted': extracted,
                'has_thumbnail': True,
                'channel_id': channel_id,
                'message_id': message_id,
                'extracted_at': now,
                'last_accessed': now,
                'access_count': 1,
                'expires_at': now + timedelta(days=30)
            }
            
            if existing is not None:
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
                logger.debug(f"🔄 Updated thumbnail: {title[:30]} - {quality}")
            else:
                # Insert
                await self.thumbnails_col.insert_one(doc)
                logger.debug(f"💾 Saved thumbnail: {title[:30]} - {quality}")
            
            # Update cache
            if self.redis is not None and thumbnail_url:
                cache_key = f"thumb:{normalized}:{quality}"
                await self.redis.setex(
                    cache_key,
                    86400,  # 24 hours
                    thumbnail_url
                )
            
        except Exception as e:
            logger.error(f"❌ Save thumbnail error: {e}")
    
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
        Priority:
        1. Specific quality from database
        2. Any quality from database
        3. Extract from Telegram (if message_id provided)
        4. Fallback
        """
        self.stats['total_requests'] += 1
        await self._rate_limit()
        
        normalized = self._normalize_title(title)
        
        result = {
            'thumbnail_url': FALLBACK_THUMBNAIL_URL,
            'source': 'fallback',
            'extracted': False,
            'quality': quality
        }
        
        try:
            # ============ STEP 1: Check Database for specific quality ============
            if quality is not None and self.thumbnails_col is not None:
                doc = await self.thumbnails_col.find_one({
                    'normalized_title': normalized,
                    'quality': quality,
                    'has_thumbnail': True
                })
                
                if doc is not None and doc.get('thumbnail_url'):
                    self.stats['db_hits'] += 1
                    
                    # Update cache
                    if self.redis is not None:
                        cache_key = f"thumb:{normalized}:{quality}"
                        await self.redis.setex(cache_key, 86400, doc['thumbnail_url'])
                    
                    result.update({
                        'thumbnail_url': doc['thumbnail_url'],
                        'source': doc.get('source', 'database'),
                        'extracted': True,
                        'quality': quality
                    })
                    return result
            
            # ============ STEP 2: Check Database for any quality ============
            if self.thumbnails_col is not None:
                doc = await self.thumbnails_col.find_one({
                    'normalized_title': normalized,
                    'has_thumbnail': True
                })
                
                if doc is not None and doc.get('thumbnail_url'):
                    self.stats['db_hits'] += 1
                    
                    result.update({
                        'thumbnail_url': doc['thumbnail_url'],
                        'source': doc.get('source', 'database'),
                        'extracted': True,
                        'quality': doc.get('quality')
                    })
                    return result
            
            # ============ STEP 3: Extract from Telegram if message provided ============
            if channel_id and message_id and (force_extract or not result['thumbnail_url']):
                self.stats['extraction_attempts'] += 1
                
                client = self.user_client if self.user_client is not None else self.bot_client
                if client:
                    thumbnail_url = await self._extract_single_thumbnail(
                        client, channel_id, message_id
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
                        
                        result.update({
                            'thumbnail_url': thumbnail_url,
                            'source': 'extracted',
                            'extracted': True,
                            'quality': quality
                        })
                        return result
                    else:
                        self.stats['extraction_failed'] += 1
            
            return result
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"❌ get_thumbnail_for_movie error: {e}")
            return result
    
    async def get_thumbnails_batch(self, movies: List[Dict]) -> List[Dict]:
        """Get thumbnails for multiple movies in batch"""
        results = []
        
        if not movies:
            return results
        
        if self.thumbnails_col is None:
            return [{
                'thumbnail_url': FALLBACK_THUMBNAIL_URL,
                'source': 'fallback',
                'extracted': False
            } for _ in movies]
        
        try:
            # Get all normalized titles
            normalized_titles = []
            for movie in movies:
                title = movie.get('title', '')
                if title:
                    normalized_titles.append(self._normalize_title(title))
            
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
                        'source': doc.get('source', 'database'),
                        'extracted': True,
                        'quality': doc.get('quality')
                    }
            
            # Build results
            for movie in movies:
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
    
    async def mark_for_extraction(self, normalized_title: str, quality: str, channel_id: int, message_id: int):
        """Mark a file for background thumbnail extraction"""
        if self.files_col is None:
            return
        
        try:
            await self.files_col.update_one(
                {
                    'normalized_title': normalized_title,
                    'channel_id': channel_id
                },
                {
                    '$set': {
                        f'qualities.{quality}.needs_extraction': True,
                        f'qualities.{quality}.extraction_attempts': 0
                    }
                }
            )
            logger.debug(f"📝 Marked for extraction: {normalized_title} - {quality}")
        except Exception as e:
            logger.error(f"❌ Mark for extraction error: {e}")
    
    async def process_pending_extractions(self, limit: int = 100):
        """Process all pending thumbnail extractions"""
        if self.files_col is None or self.thumbnails_col is None:
            logger.warning("⚠️ Files or thumbnails collection not available")
            return
        
        client = self.user_client if self.user_client is not None else self.bot_client
        if client is None:
            logger.warning("⚠️ No Telegram client available")
            return
        
        logger.info("🔄 Processing pending thumbnail extractions...")
        
        # Find files that need extraction
        cursor = self.files_col.find({
            'channel_id': self.file_channel_id,
            'has_any_thumbnail': True
        }).limit(limit)
        
        processed = 0
        successful = 0
        failed = 0
        
        async for movie in cursor:
            for quality, file_data in movie.get('qualities', {}).items():
                if file_data.get('has_thumbnail_in_telegram') and not file_data.get('thumbnail_extracted'):
                    try:
                        thumb_url = await self._extract_single_thumbnail(
                            client,
                            self.file_channel_id,
                            file_data['message_id']
                        )
                        
                        if thumb_url:
                            await self._save_thumbnail(
                                title=movie['title'],
                                normalized=movie['normalized_title'],
                                thumbnail_url=thumb_url,
                                source='extracted',
                                channel_id=self.file_channel_id,
                                message_id=file_data['message_id'],
                                quality=quality,
                                extracted=True
                            )
                            
                            await self.files_col.update_one(
                                {'_id': movie['_id']},
                                {
                                    '$set': {
                                        f'qualities.{quality}.thumbnail_url': thumb_url,
                                        f'qualities.{quality}.thumbnail_extracted': True,
                                        f'qualities.{quality}.thumbnail_extracted_at': datetime.now()
                                    }
                                }
                            )
                            
                            successful += 1
                            logger.info(f"✅ Extracted pending: {movie['title'][:30]} - {quality}")
                        else:
                            failed += 1
                        
                        processed += 1
                        await asyncio.sleep(self.min_request_interval)
                        
                    except Exception as e:
                        logger.error(f"❌ Pending extraction error: {e}")
                        failed += 1
        
        logger.info(f"✅ Pending extraction complete: {successful} successful, {failed} failed")
        return {'processed': processed, 'successful': successful, 'failed': failed}
    
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
        
        if self.cleanup_task is not None:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self.extraction_task is not None and not self.extraction_task.done():
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
    
    async def scan_and_extract_all(self, force_re_extract=False):
        logger.info("⚠️ Fallback: Would scan and extract all thumbnails")
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
    
    async def mark_for_extraction(self, normalized_title, quality, channel_id, message_id):
        logger.debug(f"Fallback: Would mark {normalized_title} - {quality} for extraction")
    
    async def process_pending_extractions(self, limit=100):
        logger.info("⚠️ Fallback: Would process pending extractions")
        return {'processed': 0, 'successful': 0, 'failed': 0}
    
    async def get_stats(self):
        return self.stats
    
    async def shutdown(self):
        logger.info("✅ FallbackThumbnailManager shutdown")

# ============================================================================
# ✅ EXPORTS
# ============================================================================

__all__ = ['ThumbnailManager', 'FallbackThumbnailManager', 'FALLBACK_THUMBNAIL_URL']
