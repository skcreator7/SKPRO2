# ============================================================================
# 🖼️ THUMBNAIL MANAGER v9.1 - WITH FALLBACK HANDLING
# ============================================================================

import os
import logging
import asyncio
import base64
import hashlib
import time
import math
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from pyrogram import Client
from pyrogram.types import Message

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# ✅ THUMBNAIL MANAGER - WITH FALLBACK HANDLING
# ============================================================================

class ThumbnailManager:
    """
    🖼️ THUMBNAIL MANAGER v9.1
    ==========================
    🔥 Features:
        • NO FFmpeg - Zero CPU usage
        • Telegram metadata only - ~50KB per file
        • Fallback handling when no thumbnails exist
        • Base64 storage - Instant display
        • MongoDB indexes - Fast search
        • Rate limiting - Telegram safe
        • Auto cleanup - 30 days TTL
    """
    
    def __init__(
        self,
        download_path: str = "downloads/thumbnails",
        mongodb: Optional[AsyncIOMotorClient] = None,
        bot_client: Optional[Client] = None,
        user_client: Optional[Client] = None,
        file_channel_id: Union[int, str] = None,
        batch_size: int = 5,
        extract_delay: float = 1.0,
        ttl_days: int = 30,
        max_retries: int = 3
    ):
        self.download_path = download_path
        self.mongodb = mongodb
        self.bot_client = bot_client
        self.user_client = user_client
        self.file_channel_id = file_channel_id
        self.batch_size = batch_size
        self.extract_delay = extract_delay
        self.ttl_days = ttl_days
        self.max_retries = max_retries
        
        # Collections
        self.db = None
        self.thumbnails_col: Optional[AsyncIOMotorCollection] = None
        self.stats_col: Optional[AsyncIOMotorCollection] = None
        
        # State
        self.initialized = False
        self.is_extracting = False
        self.extraction_task = None
        self.cleanup_task = None
        self.monitor_task = None
        
        # Statistics
        self.stats = {
            'total_extracted': 0,
            'total_failed': 0,
            'total_skipped': 0,
            'total_no_thumbnail': 0,  # Files without thumbnails
            'total_size_kb': 0,
            'avg_size_kb': 0,
            'start_time': None,
            'last_extraction': None,
            'extraction_rate': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Rate limiting
        self.extraction_lock = asyncio.Lock()
        self.rate_limiter = asyncio.Semaphore(batch_size)
        
        # Cache for recently accessed thumbnails
        self.thumbnail_cache = {}
        self.cache_ttl = 300  # 5 minutes
        self.cache_maxsize = 1000
        
        # Fallback thumbnail URL (can be configured)
        self.fallback_thumbnail_url = "https://iili.io/fAeIwv9.th.png"
        
        logger.info("🖼️ Thumbnail Manager initialized")
    
    # ========================================================================
    # ✅ INITIALIZATION & SETUP
    # ========================================================================
    
    async def initialize(self) -> bool:
        """Initialize database collections and indexes"""
        try:
            if not self.mongodb:
                logger.error("❌ MongoDB client not provided")
                return False
            
            # Get database
            self.db = self.mongodb.sk4film
            
            # Create/Get collections
            self.thumbnails_col = self.db.thumbnails
            self.stats_col = self.db.thumbnail_stats
            
            # Drop existing indexes to avoid conflicts
            try:
                await self.thumbnails_col.drop_indexes()
                logger.info("✅ Dropped existing indexes")
            except:
                pass
            
            # Create clean indexes
            await self._create_indexes()
            
            # Load existing stats
            await self._load_stats()
            
            self.initialized = True
            logger.info("✅ Thumbnail Manager initialized successfully")
            logger.info(f"   • Batch size: {self.batch_size}")
            logger.info(f"   • Extract delay: {self.extract_delay}s")
            logger.info(f"   • TTL: {self.ttl_days} days")
            logger.info(f"   • Max retries: {self.max_retries}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Thumbnail Manager initialization failed: {e}")
            return False
    
    async def _create_indexes(self):
        """Create optimized MongoDB indexes for fast search"""
        try:
            # Unique compound index for title+quality
            await self.thumbnails_col.create_index(
                [("normalized_title", 1), ("quality", 1)],
                unique=True,
                name="title_quality_unique",
                background=True
            )
            
            # Index for fast title search (non-unique)
            await self.thumbnails_col.create_index(
                [("normalized_title", 1)],
                name="title_search_idx",
                background=True
            )
            
            # Index for message ID lookups
            await self.thumbnails_col.create_index(
                [("message_id", 1)],
                name="message_idx",
                background=True
            )
            
            # Index for channel queries
            await self.thumbnails_col.create_index(
                [("channel_id", 1), ("message_id", -1)],
                name="channel_message_idx",
                background=True
            )
            
            # TTL index for auto cleanup
            await self.thumbnails_col.create_index(
                [("expires_at", 1)],
                name="ttl_cleanup_idx",
                expireAfterSeconds=0,
                background=True
            )
            
            # Index for files without thumbnails
            await self.thumbnails_col.create_index(
                [("has_thumbnail", 1)],
                name="has_thumbnail_idx",
                background=True
            )
            
            logger.info("✅ All thumbnail indexes created successfully")
            
        except Exception as e:
            logger.warning(f"⚠️ Index creation error: {e}")
    
    async def _load_stats(self):
        """Load existing statistics from database"""
        try:
            if self.stats_col:
                stats_doc = await self.stats_col.find_one({"_id": "thumbnail_stats"})
                if stats_doc:
                    self.stats.update(stats_doc.get('stats', {}))
                    logger.info(f"📊 Loaded existing stats: {self.stats['total_extracted']} thumbnails")
        except Exception as e:
            logger.error(f"❌ Error loading stats: {e}")
    
    # ========================================================================
    # ✅ CORE EXTRACTION METHODS
    # ========================================================================
    
    async def extract_thumbnail_metadata(
        self,
        channel_id: Union[int, str],
        message_id: int,
        retry_count: int = 0
    ) -> Optional[str]:
        """
        Extract thumbnail from Telegram metadata
        
        Returns:
            Base64 encoded thumbnail or None if no thumbnail exists
        """
        # Get available client
        client = self.bot_client or self.user_client
        if not client:
            logger.error("❌ No Telegram client available")
            return None
        
        try:
            # Fetch message
            message = await client.get_messages(channel_id, message_id)
            if not message:
                logger.warning(f"⚠️ Message {message_id} not found")
                return None
            
            thumbnail_data = None
            
            # 🎬 EXTRACT FROM VIDEO METADATA
            if message.video:
                if hasattr(message.video, 'thumbnail') and message.video.thumbnail:
                    thumbnail_file_id = message.video.thumbnail.file_id
                    logger.debug(f"🎬 Found video thumbnail for message {message_id}")
                    
                    # Download thumbnail
                    download_path = await client.download_media(
                        thumbnail_file_id,
                        in_memory=True
                    )
                    
                    if download_path:
                        if isinstance(download_path, bytes):
                            thumbnail_data = download_path
                        else:
                            with open(download_path, 'rb') as f:
                                thumbnail_data = f.read()
            
            # 📄 EXTRACT FROM DOCUMENT METADATA
            elif message.document:
                file_name = message.document.file_name or ""
                if self._is_video_file(file_name):
                    if hasattr(message.document, 'thumbnail') and message.document.thumbnail:
                        thumbnail_file_id = message.document.thumbnail.file_id
                        logger.debug(f"📄 Found document thumbnail for {file_name}")
                        
                        download_path = await client.download_media(
                            thumbnail_file_id,
                            in_memory=True
                        )
                        
                        if download_path:
                            if isinstance(download_path, bytes):
                                thumbnail_data = download_path
                            else:
                                with open(download_path, 'rb') as f:
                                    thumbnail_data = f.read()
            
            if thumbnail_data:
                # Convert to base64
                size_kb = len(thumbnail_data) / 1024
                logger.debug(f"✅ Thumbnail extracted: {size_kb:.1f}KB")
                
                base64_data = base64.b64encode(thumbnail_data).decode('utf-8')
                return f"data:image/jpeg;base64,{base64_data}"
            
            # No thumbnail found in metadata
            logger.debug(f"ℹ️ No thumbnail metadata for message {message_id}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Thumbnail extraction error: {e}")
            
            # Retry logic
            if retry_count < self.max_retries:
                logger.info(f"🔄 Retrying extraction (attempt {retry_count + 1}/{self.max_retries})")
                await asyncio.sleep(2 ** retry_count)
                return await self.extract_thumbnail_metadata(
                    channel_id, 
                    message_id, 
                    retry_count + 1
                )
            
            return None
    
    async def extract_and_store(
        self,
        channel_id: Union[int, str],
        message_id: int,
        file_name: str,
        title: Optional[str] = None,
        quality: Optional[str] = None,
        year: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract thumbnail and store in MongoDB
        
        Returns:
            Dict with extraction result and metadata
        """
        result = {
            'success': False,
            'thumbnail_url': None,
            'size_kb': 0,
            'error': None,
            'cached': False,
            'has_thumbnail': False
        }
        
        try:
            # Generate normalized title
            if not title:
                title = self._extract_clean_title(file_name)
            
            normalized = self._normalize_title(title)
            
            if not quality:
                quality = self._detect_quality_enhanced(file_name)
            
            if not year:
                year = self._extract_year(file_name)
            
            # Check if already exists in database
            existing = await self.thumbnails_col.find_one({
                'normalized_title': normalized,
                'quality': quality
            })
            
            if existing:
                logger.debug(f"📦 Using cached record for: {title} - {quality}")
                result['success'] = True
                result['thumbnail_url'] = existing.get('thumbnail_url')
                result['size_kb'] = existing.get('size_kb', 0)
                result['has_thumbnail'] = existing.get('has_thumbnail', False)
                result['cached'] = True
                self.stats['cache_hits'] += 1
                return result
            
            self.stats['cache_misses'] += 1
            
            # Extract thumbnail
            thumbnail_url = await self.extract_thumbnail_metadata(channel_id, message_id)
            
            # Prepare document
            thumbnail_doc = {
                'normalized_title': normalized,
                'title': title,
                'quality': quality,
                'year': year,
                'message_id': message_id,
                'channel_id': channel_id,
                'file_name': file_name,
                'extracted_at': datetime.now(),
                'expires_at': datetime.now() + timedelta(days=self.ttl_days),
                'access_count': 0,
                'last_accessed': None,
                'has_thumbnail': False  # Default to False
            }
            
            if thumbnail_url:
                # Has thumbnail
                size_kb = len(thumbnail_url) / 1024
                
                thumbnail_doc.update({
                    'thumbnail_url': thumbnail_url,
                    'thumbnail_source': 'telegram_metadata',
                    'size_kb': size_kb,
                    'extraction_status': 'success',
                    'has_thumbnail': True
                })
                
                # Update statistics
                self.stats['total_extracted'] += 1
                self.stats['total_size_kb'] += size_kb
                self.stats['avg_size_kb'] = (
                    self.stats['total_size_kb'] / self.stats['total_extracted']
                )
                
                result['success'] = True
                result['thumbnail_url'] = thumbnail_url
                result['size_kb'] = size_kb
                result['has_thumbnail'] = True
                
                logger.info(f"✅ Stored thumbnail: {title} - {quality} ({size_kb:.1f}KB)")
                
            else:
                # No thumbnail - store as "no_thumbnail" to avoid re-processing
                thumbnail_doc.update({
                    'thumbnail_url': None,
                    'thumbnail_source': None,
                    'size_kb': 0,
                    'extraction_status': 'no_thumbnail',
                    'has_thumbnail': False
                })
                
                self.stats['total_no_thumbnail'] += 1
                result['error'] = 'No thumbnail metadata'
                result['has_thumbnail'] = False
                
                logger.debug(f"ℹ️ No thumbnail for: {title} - {quality}")
            
            # Store in MongoDB
            await self.thumbnails_col.update_one(
                {
                    'normalized_title': normalized,
                    'quality': quality
                },
                {'$set': thumbnail_doc},
                upsert=True
            )
            
            # Update cache if we have a thumbnail
            if thumbnail_url:
                self._update_cache(normalized, quality, thumbnail_doc)
            
            # Save stats periodically
            if self.stats['total_extracted'] % 10 == 0:
                await self._save_stats()
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Extract and store error: {e}")
            result['error'] = str(e)
            self.stats['total_failed'] += 1
            return result
    
    # ========================================================================
    # ✅ BATCH PROCESSING
    # ========================================================================
    
    async def process_messages_batch(
        self,
        messages: List[Message],
        channel_id: Union[int, str]
    ) -> Dict[str, Any]:
        """
        Process a batch of messages for thumbnail extraction
        
        Returns:
            Dict with batch processing statistics
        """
        batch_stats = {
            'total': len(messages),
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'no_thumbnail': 0,
            'total_size_kb': 0,
            'errors': []
        }
        
        async with self.extraction_lock:
            for i, message in enumerate(messages):
                try:
                    # Rate limiting
                    async with self.rate_limiter:
                        # Get file info
                        file_name = None
                        if message.document:
                            file_name = message.document.file_name
                        elif message.video:
                            file_name = message.video.file_name or "video.mp4"
                        
                        if not file_name or not self._is_video_file(file_name):
                            batch_stats['skipped'] += 1
                            continue
                        
                        # Extract and store
                        result = await self.extract_and_store(
                            channel_id=channel_id,
                            message_id=message.id,
                            file_name=file_name
                        )
                        
                        batch_stats['processed'] += 1
                        
                        if result['success'] and result.get('has_thumbnail'):
                            batch_stats['successful'] += 1
                            batch_stats['total_size_kb'] += result.get('size_kb', 0)
                        elif not result.get('has_thumbnail'):
                            batch_stats['no_thumbnail'] += 1
                        else:
                            batch_stats['failed'] += 1
                            if result.get('error'):
                                batch_stats['errors'].append({
                                    'message_id': message.id,
                                    'error': result['error']
                                })
                    
                    # Delay between extractions
                    if i < len(messages) - 1:
                        await asyncio.sleep(self.extract_delay)
                        
                except Exception as e:
                    logger.error(f"❌ Batch processing error for message {message.id}: {e}")
                    batch_stats['failed'] += 1
                    batch_stats['errors'].append({
                        'message_id': message.id,
                        'error': str(e)
                    })
            
            return batch_stats
    
    # ========================================================================
    # ✅ RETRIEVAL METHODS
    # ========================================================================
    
    async def get_thumbnail(
        self,
        title: str,
        quality: Optional[str] = None,
        fallback_to_any: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Get thumbnail for a movie
        
        Args:
            title: Movie title
            quality: Specific quality (optional)
            fallback_to_any: If True, return any quality if specific not found
            
        Returns:
            Dict with thumbnail data or None
        """
        try:
            normalized = self._normalize_title(title)
            
            # Check cache first
            cache_key = f"{normalized}:{quality}" if quality else normalized
            if cache_key in self.thumbnail_cache:
                cache_entry = self.thumbnail_cache[cache_key]
                if time.time() - cache_entry['timestamp'] < self.cache_ttl:
                    self.stats['cache_hits'] += 1
                    return cache_entry['doc']
            
            self.stats['cache_misses'] += 1
            
            # Build query
            query = {'normalized_title': normalized}
            if quality:
                query['quality'] = quality
            
            # Try exact match first
            if quality:
                doc = await self.thumbnails_col.find_one(query)
                if doc and doc.get('has_thumbnail'):
                    doc['_id'] = str(doc['_id'])
                    self._update_cache(cache_key, quality, doc)
                    
                    # Update access count
                    await self.thumbnails_col.update_one(
                        {'_id': doc['_id']},
                        {
                            '$inc': {'access_count': 1},
                            '$set': {'last_accessed': datetime.now()}
                        }
                    )
                    
                    return doc
            
            # Try any quality with thumbnail
            if fallback_to_any:
                cursor = self.thumbnails_col.find(
                    {
                        'normalized_title': normalized,
                        'has_thumbnail': True
                    }
                ).sort([
                    ('quality_priority', 1),
                    ('extracted_at', -1)
                ]).limit(1)
                
                docs = await cursor.to_list(length=1)
                if docs:
                    doc = docs[0]
                    doc['_id'] = str(doc['_id'])
                    self._update_cache(normalized, None, doc)
                    
                    # Update access count
                    await self.thumbnails_col.update_one(
                        {'_id': doc['_id']},
                        {
                            '$inc': {'access_count': 1},
                            '$set': {'last_accessed': datetime.now()}
                        }
                    )
                    
                    return doc
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Get thumbnail error: {e}")
            return None
    
    async def get_thumbnails_batch(
        self,
        titles: List[str],
        qualities: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get thumbnails for multiple movies in batch
        """
        try:
            normalized_titles = [self._normalize_title(t) for t in titles]
            
            query = {
                'normalized_title': {'$in': normalized_titles},
                'has_thumbnail': True
            }
            
            cursor = self.thumbnails_col.find(query).hint([('normalized_title', 1)])
            
            result = {}
            async for doc in cursor:
                doc['_id'] = str(doc['_id'])
                result[doc['normalized_title']] = doc
                
                # Update cache
                cache_key = f"{doc['normalized_title']}:{doc.get('quality', '')}"
                self._update_cache(cache_key, doc.get('quality'), doc)
            
            logger.debug(f"📦 Batch retrieved {len(result)} thumbnails")
            return result
            
        except Exception as e:
            logger.error(f"❌ Batch get error: {e}")
            return {}
    
    async def has_thumbnail(self, title: str, quality: Optional[str] = None) -> bool:
        """Check if a movie has a thumbnail"""
        try:
            normalized = self._normalize_title(title)
            
            query = {'normalized_title': normalized}
            if quality:
                query['quality'] = quality
            query['has_thumbnail'] = True
            
            count = await self.thumbnails_col.count_documents(query)
            return count > 0
            
        except Exception as e:
            logger.error(f"❌ Has thumbnail error: {e}")
            return False
    
    # ========================================================================
    # ✅ MAINTENANCE METHODS
    # ========================================================================
    
    async def cleanup_expired(self) -> int:
        """Manually trigger cleanup of expired thumbnails"""
        try:
            result = await self.thumbnails_col.delete_many({
                'expires_at': {'$lt': datetime.now()}
            })
            
            if result.deleted_count > 0:
                logger.info(f"🧹 Cleaned up {result.deleted_count} expired thumbnails")
            
            return result.deleted_count
            
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")
            return 0
    
    async def retry_failed(self, max_retries: int = 3) -> int:
        """Retry failed thumbnail extractions"""
        try:
            # Find failed extractions
            cursor = self.thumbnails_col.find({
                'extraction_status': 'failed',
                'retry_count': {'$lt': max_retries}
            }).limit(100)
            
            retried = 0
            successful = 0
            
            async for doc in cursor:
                retried += 1
                
                # Retry extraction
                result = await self.extract_thumbnail_metadata(
                    doc['channel_id'],
                    doc['message_id']
                )
                
                if result:
                    # Update with success
                    await self.thumbnails_col.update_one(
                        {'_id': doc['_id']},
                        {
                            '$set': {
                                'thumbnail_url': result,
                                'extraction_status': 'success',
                                'extracted_at': datetime.now(),
                                'size_kb': len(result) / 1024,
                                'has_thumbnail': True,
                                'expires_at': datetime.now() + timedelta(days=self.ttl_days)
                            },
                            '$inc': {'retry_count': 1}
                        }
                    )
                    successful += 1
                    logger.info(f"✅ Retry successful for {doc['title']}")
                else:
                    # Update retry count
                    await self.thumbnails_col.update_one(
                        {'_id': doc['_id']},
                        {
                            '$inc': {'retry_count': 1},
                            '$set': {'extraction_status': 'no_thumbnail'}
                        }
                    )
                
                await asyncio.sleep(self.extract_delay)
            
            if retried > 0:
                logger.info(f"🔄 Retried {retried} failed extractions, {successful} successful")
            
            return successful
            
        except Exception as e:
            logger.error(f"❌ Retry failed error: {e}")
            return 0
    
    async def mark_no_thumbnail(self, title: str, quality: str) -> bool:
        """Mark a file as having no thumbnail (to avoid re-processing)"""
        try:
            normalized = self._normalize_title(title)
            
            result = await self.thumbnails_col.update_one(
                {
                    'normalized_title': normalized,
                    'quality': quality
                },
                {
                    '$set': {
                        'has_thumbnail': False,
                        'extraction_status': 'no_thumbnail',
                        'checked_at': datetime.now()
                    }
                },
                upsert=True
            )
            
            return result.modified_count > 0 or result.upserted_id is not None
            
        except Exception as e:
            logger.error(f"❌ Mark no thumbnail error: {e}")
            return False
    
    # ========================================================================
    # ✅ STATISTICS METHODS
    # ========================================================================
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        try:
            if not self.thumbnails_col:
                return self.stats
            
            # Get counts
            total = await self.thumbnails_col.count_documents({})
            with_thumbnail = await self.thumbnails_col.count_documents({
                'has_thumbnail': True
            })
            without_thumbnail = await self.thumbnails_col.count_documents({
                'has_thumbnail': False
            })
            
            # Get total size
            pipeline = [
                {'$match': {'has_thumbnail': True}},
                {'$group': {
                    '_id': None,
                    'total_size_kb': {'$sum': '$size_kb'},
                    'avg_size_kb': {'$avg': '$size_kb'},
                    'max_size_kb': {'$max': '$size_kb'},
                    'min_size_kb': {'$min': '$size_kb'}
                }}
            ]
            
            size_stats = await self.thumbnails_col.aggregate(pipeline).to_list(1)
            
            # Get quality distribution
            quality_pipeline = [
                {'$match': {'has_thumbnail': True}},
                {'$group': {
                    '_id': '$quality',
                    'count': {'$sum': 1}
                }},
                {'$sort': {'count': -1}}
            ]
            
            quality_stats = await self.thumbnails_col.aggregate(quality_pipeline).to_list(100)
            
            # Get recent extractions
            recent = await self.thumbnails_col.find(
                {'has_thumbnail': True}
            ).sort('extracted_at', -1).limit(10).to_list(10)
            
            for r in recent:
                r['_id'] = str(r['_id'])
            
            # Calculate extraction rate
            if self.stats['start_time']:
                duration = (datetime.now() - self.stats['start_time']).total_seconds() / 3600
                if duration > 0:
                    self.stats['extraction_rate'] = self.stats['total_extracted'] / duration
            
            stats = {
                **self.stats,
                'total_documents': total,
                'with_thumbnail': with_thumbnail,
                'without_thumbnail': without_thumbnail,
                'size_stats': size_stats[0] if size_stats else {},
                'quality_distribution': quality_stats,
                'recent_extractions': recent,
                'cache_size': len(self.thumbnail_cache),
                'cache_hit_rate': (
                    (self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses']) * 100)
                    if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0 else 0
                ),
                'ttl_days': self.ttl_days,
                'batch_size': self.batch_size,
                'extract_delay': self.extract_delay
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Get stats error: {e}")
            return self.stats
    
    # ========================================================================
    # ✅ BACKGROUND TASKS
    # ========================================================================
    
    async def start_background_tasks(self):
        """Start background monitoring and cleanup tasks"""
        self.is_extracting = True
        self.stats['start_time'] = datetime.now()
        
        # Start monitor task
        self.monitor_task = asyncio.create_task(self._monitor_new_files())
        
        # Start cleanup task
        self.cleanup_task = asyncio.create_task(self._auto_cleanup_loop())
        
        logger.info("✅ Background tasks started")
    
    async def stop_background_tasks(self):
        """Stop all background tasks"""
        self.is_extracting = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except:
                pass
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except:
                pass
        
        # Save final stats
        await self._save_stats()
        
        logger.info("🛑 Background tasks stopped")
    
    async def _monitor_new_files(self):
        """Monitor for new files and extract automatically"""
        last_check = datetime.now()
        
        while self.is_extracting:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                if not self.is_extracting or not self.thumbnails_col:
                    continue
                
                # Find latest message ID in database
                latest = await self.thumbnails_col.find_one(
                    {'channel_id': self.file_channel_id},
                    sort=[('message_id', -1)]
                )
                last_message_id = latest.get('message_id', 0) if latest else 0
                
                # Fetch new messages
                client = self.user_client or self.bot_client
                if not client:
                    continue
                
                new_messages = []
                async for msg in client.get_chat_history(
                    self.file_channel_id,
                    limit=50
                ):
                    if msg.id > last_message_id:
                        new_messages.append(msg)
                    else:
                        break
                
                if new_messages:
                    # Filter video files
                    video_files = []
                    for msg in new_messages:
                        if not msg or (not msg.document and not msg.video):
                            continue
                        
                        file_name = None
                        if msg.document:
                            file_name = msg.document.file_name
                        elif msg.video:
                            file_name = msg.video.file_name or "video.mp4"
                        
                        if file_name and self._is_video_file(file_name):
                            video_files.append(msg)
                    
                    if video_files:
                        logger.info(f"🆕 Found {len(video_files)} new video files")
                        
                        # Process in batches
                        for i in range(0, len(video_files), self.batch_size):
                            batch = video_files[i:i + self.batch_size]
                            await self.process_messages_batch(batch, self.file_channel_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Monitor error: {e}")
                await asyncio.sleep(120)
    
    async def _auto_cleanup_loop(self):
        """Auto cleanup loop for expired thumbnails"""
        while self.is_extracting:
            try:
                await asyncio.sleep(24 * 60 * 60)  # Run daily
                
                if not self.is_extracting:
                    break
                
                logger.info("🧹 Running auto cleanup for expired thumbnails...")
                await self.cleanup_expired()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Auto cleanup error: {e}")
    
    # ========================================================================
    # ✅ UTILITY METHODS
    # ========================================================================
    
    def _update_cache(self, key: str, quality: Optional[str], value: Any):
        """Update thumbnail cache"""
        cache_key = f"{key}:{quality}" if quality else key
        self.thumbnail_cache[cache_key] = {
            'doc': value,
            'timestamp': time.time()
        }
        
        # Maintain cache size
        if len(self.thumbnail_cache) > self.cache_maxsize:
            # Remove oldest entry
            oldest_key = min(
                self.thumbnail_cache.keys(),
                key=lambda k: self.thumbnail_cache[k]['timestamp']
            )
            del self.thumbnail_cache[oldest_key]
    
    def _normalize_title(self, title: str) -> str:
        """Normalize movie title for consistent matching"""
        if not title:
            return ""
        
        title = title.lower().strip()
        # Remove year in parentheses
        title = re.sub(r'\s*\(\d{4}\)', '', title)
        # Remove standalone year
        title = re.sub(r'\s+\d{4}$', '', title)
        # Remove special characters
        title = re.sub(r'[^\w\s]', ' ', title)
        # Remove extra spaces
        title = re.sub(r'\s+', ' ', title)
        
        return title.strip()
    
    def _extract_clean_title(self, filename: str) -> str:
        """Extract clean movie title from filename"""
        if not filename:
            return "Unknown"
        
        # Remove extension
        name = os.path.splitext(filename)[0]
        
        # Replace separators with spaces
        name = re.sub(r'[._\-]', ' ', name)
        
        # Remove quality tags
        name = re.sub(r'\b(480p|720p|1080p|2160p|4k|hd|hevc|x264|x265|web-dl|webrip|bluray|hdtv|hdr|dts|ac3|aac|ddp|5\.1|7\.1|2\.0|esub|sub|multi|dual|audio|hindi|english|tamil|telugu|malayalam|kannada|ben|eng|hin|tam|tel|mal|kan)\b.*$', '', name, flags=re.IGNORECASE)
        
        # Remove year
        name = re.sub(r'\s+(19|20)\d{2}\s*$', '', name)
        
        # Remove parentheses
        name = re.sub(r'\s*\([^)]*\)', '', name)
        name = re.sub(r'\s*\[[^\]]*\]', '', name)
        
        # Clean spaces
        name = re.sub(r'\s+', ' ', name)
        name = name.strip()
        
        return name if name else "Unknown"
    
    def _detect_quality_enhanced(self, filename: str) -> str:
        """Detect quality from filename"""
        if not filename:
            return "480p"
        
        filename_lower = filename.lower()
        
        QUALITY_PATTERNS = [
            (r'\b2160p\b|\b4k\b|\buhd\b', '2160p'),
            (r'\b1080p\b|\bfullhd\b|\bfhd\b', '1080p'),
            (r'\b720p\b|\bhd\b', '720p'),
            (r'\b480p\b', '480p'),
            (r'\b360p\b', '360p'),
        ]
        
        HEVC_PATTERNS = [
            r'\bhevc\b',
            r'\bx265\b',
            r'\bh\.?265\b',
        ]
        
        is_hevc = any(re.search(pattern, filename_lower) for pattern in HEVC_PATTERNS)
        
        for pattern, quality in QUALITY_PATTERNS:
            if re.search(pattern, filename_lower):
                if is_hevc and quality in ['720p', '1080p', '2160p']:
                    return f"{quality} HEVC"
                return quality
        
        return "480p"
    
    def _extract_year(self, filename: str) -> str:
        """Extract year from filename"""
        if not filename:
            return ""
        
        year_match = re.search(r'\b(19|20)\d{2}\b', filename)
        return year_match.group() if year_match else ""
    
    def _is_video_file(self, filename: str) -> bool:
        """Check if file is a video"""
        if not filename:
            return False
        
        video_extensions = [
            '.mp4', '.mkv', '.avi', '.mov', '.wmv', 
            '.flv', '.webm', '.m4v', '.mpg', '.mpeg',
            '.3gp', '.mkv', '.ts', '.m2ts'
        ]
        return any(filename.lower().endswith(ext) for ext in video_extensions)
    
    async def _save_stats(self):
        """Save statistics to database"""
        try:
            if self.stats_col:
                await self.stats_col.update_one(
                    {"_id": "thumbnail_stats"},
                    {"$set": {
                        "stats": self.stats,
                        "updated_at": datetime.now()
                    }},
                    upsert=True
                )
        except Exception as e:
            logger.error(f"❌ Error saving stats: {e}")
    
    # ========================================================================
    # ✅ SHUTDOWN
    # ========================================================================
    
    async def shutdown(self):
        """Gracefully shutdown the thumbnail manager"""
        logger.info("🖼️ Shutting down Thumbnail Manager...")
        
        await self.stop_background_tasks()
        await self._save_stats()
        
        # Clear cache
        self.thumbnail_cache.clear()
        
        logger.info(f"✅ Thumbnail Manager shutdown complete. "
                   f"Total with thumbnails: {self.stats['total_extracted']}, "
                   f"Without thumbnails: {self.stats['total_no_thumbnail']}")
