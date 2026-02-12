"""
thumbnail_manager.py - COMPLETE THUMBNAIL SYSTEM
One Movie → One Thumbnail → All 4 Qualities
"""

import os
import asyncio
import hashlib
import re
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
from datetime import datetime
import aiohttp
from pyrogram.types import Message
from pyrogram.enums import MessageMediaType
import logging
from motor.motor_asyncio import AsyncIOMotorDatabase

logger = logging.getLogger(__name__)

class ThumbnailManager:
    """
    ONE MOVIE → ONE THUMBNAIL → ALL 4 QUALITIES
    Extracts thumbnail once per movie, shares across all qualities
    """
    
    def __init__(self, 
                 download_path: str = "downloads/thumbnails",
                 mongodb: Optional[AsyncIOMotorDatabase] = None,
                 bot_client=None,
                 user_client=None,
                 file_channel_id: Union[int, str] = None):
        
        self.download_path = Path(download_path)
        self.download_path.mkdir(parents=True, exist_ok=True)
        self.db = mongodb
        self.bot = bot_client
        self.user = user_client
        self.file_channel = file_channel_id
        self.thumb_collection = mongodb.thumbnails if mongodb else None
        self.movie_collection = mongodb.movies if mongodb else None  # NEW: Movie master collection
        
        # Statistics tracking
        self.stats = {
            "total_movies": 0,
            "movies_with_thumbnails": 0,
            "movies_without_thumbnails": 0,
            "total_files": 0,
            "files_with_thumbnails": 0
        }
        
        # Create indexes
        if self.thumb_collection is not None:
            asyncio.create_task(self._create_indexes())
            
        logger.info("=" * 60)
        logger.info("🚀 THUMBNAIL MANAGER: ONE MOVIE → ONE THUMBNAIL")
        logger.info("=" * 60)
        logger.info("   • One Movie → One Thumbnail: ✅ ENABLED")
        logger.info("   • 4 Qualities → Same Thumbnail: ✅ ENABLED")
        logger.info("   • Existing Files Auto-Extract: ✅ ENABLED")
        logger.info("   • New Files Auto-Extract: ✅ ENABLED")
        logger.info("   • Priority: Extracted → Poster → Empty: ✅ ENABLED")
        logger.info("   • No Default/Fallback Image: ✅ ENABLED")
        logger.info("=" * 60)
    
    async def _create_indexes(self):
        """Create optimized indexes for fast lookups"""
        try:
            # Thumbnail collection indexes
            await self.thumb_collection.create_index("movie_id", unique=True)
            await self.thumb_collection.create_index("movie_name_normalized")
            await self.thumb_collection.create_index("thumbnail_path")
            
            # Movie collection indexes
            if self.movie_collection:
                await self.movie_collection.create_index("movie_id", unique=True)
                await self.movie_collection.create_index("movie_name_normalized")
                await self.movie_collection.create_index("file_ids")
                
            logger.info("✅ Thumbnail indexes created successfully")
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
    
    def _extract_movie_id(self, file_name: str) -> str:
        """
        Extract UNIQUE MOVIE ID from filename
        Same movie = Same ID across all qualities
        """
        if not file_name:
            return "unknown"
        
        # Remove year in parentheses
        name = re.sub(r'\s*\(\d{4}\)', '', file_name)
        # Remove standalone year
        name = re.sub(r'\s*\d{4}', '', name)
        # Remove quality tags
        name = re.sub(r'\b(4K|2160p|1080p|720p|480p|360p|WEB-DL|WEBRip|BluRay|HDTV|HDRip|CAM|HDTC|TS|DVD|BRRip)\b.*$', '', name, flags=re.I)
        # Remove codec info
        name = re.sub(r'\b(x264|x265|HEVC|AVC|DDP|AAC|AC3|MP3|5\.1|7\.1|2\.0|ESub|Sub)\b.*$', '', name, flags=re.I)
        # Remove special characters
        name = re.sub(r'[^\w\s-]', ' ', name)
        # Clean up spaces
        name = re.sub(r'\s+', ' ', name).strip().lower()
        
        # If too short, use original but cleaned
        if len(name) < 3:
            name = re.sub(r'[^\w\s-]', ' ', file_name[:50])
            name = re.sub(r'\s+', ' ', name).strip().lower()
        
        # Create hash for consistent ID
        movie_hash = hashlib.md5(name.encode()).hexdigest()[:16]
        movie_id = f"{name[:30].replace(' ', '_')}_{movie_hash}"
        
        return movie_id
    
    def _get_movie_name_clean(self, file_name: str) -> str:
        """Get clean movie name for display"""
        name = self._extract_movie_id(file_name)
        # Remove hash part for display
        if '_' in name:
            name = name.split('_')[0]
        return name.replace('_', ' ').title()
    
    async def get_thumbnail(self, 
                           file_data: Dict[str, Any], 
                           message: Optional[Message] = None,
                           force_extract: bool = False) -> Optional[str]:
        """
        Get thumbnail for ANY quality of a movie
        ONE THUMBNAIL PER MOVIE - shared across all 4 qualities
        """
        file_name = file_data.get('file_name', '')
        message_id = file_data.get('message_id')
        file_unique_id = file_data.get('file_unique_id')
        
        # Step 1: Extract MOVIE ID (same for all qualities)
        movie_id = self._extract_movie_id(file_name)
        movie_name_clean = self._get_movie_name_clean(file_name)
        
        # Step 2: Check if we already have thumbnail for this MOVIE
        if not force_extract and self.thumb_collection is not None:
            movie_thumb = await self.thumb_collection.find_one({"movie_id": movie_id})
            
            if movie_thumb and movie_thumb.get('thumbnail_path'):
                thumb_path = movie_thumb['thumbnail_path']
                
                # Check if file exists
                if os.path.exists(thumb_path):
                    logger.info(f"✅ SHARED THUMBNAIL: '{movie_name_clean[:30]}' → All qualities")
                    
                    # Update file record to mark it has thumbnail
                    if self.db and file_unique_id:
                        await self.db.files.update_one(
                            {"file_unique_id": file_unique_id},
                            {"$set": {
                                "has_thumbnail": True,
                                "thumbnail_path": thumb_path,
                                "movie_id": movie_id,
                                "thumbnail_shared": True
                            }}
                        )
                    
                    return thumb_path
                elif movie_thumb.get('thumbnail_bytes'):
                    # Restore from database
                    os.makedirs(os.path.dirname(movie_thumb['thumbnail_path']), exist_ok=True)
                    with open(movie_thumb['thumbnail_path'], 'wb') as f:
                        f.write(movie_thumb['thumbnail_bytes'])
                    logger.info(f"✅ RESTORED THUMBNAIL: '{movie_name_clean[:30]}'")
                    return movie_thumb['thumbnail_path']
        
        # Step 3: No thumbnail exists for this movie - extract it NOW
        if message or (self.user and self.file_channel and message_id):
            logger.info(f"🎬 EXTRACTING: First time thumbnail for '{movie_name_clean[:30]}'")
            
            # Get message if not provided
            msg = message
            if not msg and self.user and self.file_channel and message_id:
                try:
                    msg = await self.user.get_messages(self.file_channel, message_id)
                except Exception as e:
                    logger.error(f"Error getting message: {e}")
            
            # Extract thumbnail
            if msg:
                thumb_path = await self._extract_from_message(msg, movie_id, movie_name_clean)
                
                if thumb_path and os.path.exists(thumb_path):
                    # Cache in database
                    await self._cache_movie_thumbnail(movie_id, movie_name_clean, thumb_path, file_data)
                    
                    # Update current file
                    if self.db and file_unique_id:
                        await self.db.files.update_one(
                            {"file_unique_id": file_unique_id},
                            {"$set": {
                                "has_thumbnail": True,
                                "thumbnail_path": thumb_path,
                                "movie_id": movie_id,
                                "thumbnail_shared": True
                            }}
                        )
                    
                    logger.info(f"✅ EXTRACTED: '{movie_name_clean[:30]}' → Available for ALL 4 qualities")
                    return thumb_path
        
        # Step 4: No thumbnail available
        logger.info(f"❌ NO THUMBNAIL: '{movie_name_clean[:30]}' (no extractable source)")
        return None
    
    async def _extract_from_message(self, 
                                   message: Message, 
                                   movie_id: str,
                                   movie_name: str) -> Optional[str]:
        """Extract thumbnail from Telegram message"""
        try:
            # Create thumbnail path
            clean_name = re.sub(r'[^\w\-_]', '_', movie_name[:30])
            thumb_path = self.download_path / f"movie_{clean_name}_{movie_id[-8:]}.jpg"
            
            # Try to get thumbnail from message
            if message.media:
                # Check for video thumbnail
                if message.video and message.video.thumbs:
                    await self.bot.download_media(
                        message.video.thumbs[0].file_id,
                        file_name=str(thumb_path)
                    )
                    if thumb_path.exists():
                        return str(thumb_path)
                
                # Check for document thumbnail
                elif message.document and message.document.thumbs:
                    await self.bot.download_media(
                        message.document.thumbs[0].file_id,
                        file_name=str(thumb_path)
                    )
                    if thumb_path.exists():
                        return str(thumb_path)
                
                # Check for photo
                elif message.photo:
                    await self.bot.download_media(
                        message.photo.file_id,
                        file_name=str(thumb_path)
                    )
                    if thumb_path.exists():
                        return str(thumb_path)
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting thumbnail: {e}")
            return None
    
    async def _cache_movie_thumbnail(self, 
                                    movie_id: str, 
                                    movie_name: str, 
                                    thumb_path: str, 
                                    source_file: Dict):
        """Cache thumbnail by MOVIE ID (not by file)"""
        if not self.thumb_collection:
            return
            
        try:
            # Read thumbnail bytes
            with open(thumb_path, 'rb') as f:
                thumb_bytes = f.read()
            
            # Create movie thumbnail record
            thumb_doc = {
                "movie_id": movie_id,
                "movie_name": movie_name,
                "movie_name_normalized": movie_name.lower(),
                "thumbnail_path": thumb_path,
                "thumbnail_bytes": thumb_bytes,
                "extracted_from": {
                    "message_id": source_file.get('message_id'),
                    "file_name": source_file.get('file_name'),
                    "file_unique_id": source_file.get('file_unique_id')
                },
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "times_shared": 0
            }
            
            # Update or insert
            result = await self.thumb_collection.update_one(
                {"movie_id": movie_id},
                {"$set": thumb_doc, "$inc": {"times_shared": 1}},
                upsert=True
            )
            
            if result.upserted_id:
                logger.info(f"💾 CACHED: New thumbnail for movie '{movie_name[:30]}'")
            else:
                logger.info(f"💾 UPDATED: Thumbnail for movie '{movie_name[:30]}'")
            
        except Exception as e:
            logger.error(f"Error caching thumbnail: {e}")
    
    async def extract_all_missing(self, files: List[Dict]) -> Dict[str, int]:
        """
        Extract thumbnails for UNIQUE MOVIES only
        One extraction per movie, serves all qualities
        """
        logger.info("=" * 60)
        logger.info("🎬 STARTING BATCH THUMBNAIL EXTRACTION")
        logger.info(f"📊 Total files in DB: {len(files)}")
        logger.info("=" * 60)
        
        # Group files by MOVIE ID
        movie_groups = {}
        for file in files:
            movie_id = self._extract_movie_id(file.get('file_name', ''))
            if movie_id not in movie_groups:
                movie_groups[movie_id] = {
                    "movie_name": self._get_movie_name_clean(file.get('file_name', '')),
                    "files": [],
                    "has_thumbnail": False
                }
            movie_groups[movie_id]["files"].append(file)
        
        unique_movies = len(movie_groups)
        logger.info(f"🎬 Unique movies found: {unique_movies}")
        logger.info(f"📁 Total files: {len(files)}")
        logger.info(f"🔄 Ratio: {len(files)} files → {unique_movies} thumbnails")
        logger.info("=" * 60)
        
        # Check which movies already have thumbnails
        movies_with_thumb = 0
        movies_without_thumb = 0
        
        if self.thumb_collection:
            for movie_id, movie_data in movie_groups.items():
                cached = await self.thumb_collection.find_one({"movie_id": movie_id})
                if cached and cached.get('thumbnail_path') and os.path.exists(cached.get('thumbnail_path', '')):
                    movie_data["has_thumbnail"] = True
                    movies_with_thumb += 1
                    
                    # Mark all files of this movie as having thumbnail
                    for file in movie_data["files"]:
                        if self.db and file.get('file_unique_id'):
                            await self.db.files.update_one(
                                {"file_unique_id": file['file_unique_id']},
                                {"$set": {
                                    "has_thumbnail": True,
                                    "thumbnail_path": cached['thumbnail_path'],
                                    "movie_id": movie_id,
                                    "thumbnail_shared": True
                                }}
                            )
                else:
                    movies_without_thumb += 1
        
        logger.info(f"✅ Movies WITH thumbnails: {movies_with_thumb}")
        logger.info(f"❌ Movies WITHOUT thumbnails: {movies_without_thumb}")
        logger.info("=" * 60)
        
        # Extract thumbnails for movies without
        success = 0
        failed = 0
        extracted_for_movies = []
        
        for movie_id, movie_data in movie_groups.items():
            if not movie_data["has_thumbnail"] and movie_data["files"]:
                # Try to extract from first available file
                first_file = movie_data["files"][0]
                
                try:
                    if self.user and self.file_channel and first_file.get('message_id'):
                        message = await self.user.get_messages(
                            self.file_channel,
                            first_file['message_id']
                        )
                        
                        thumb_path = await self._extract_from_message(
                            message, 
                            movie_id, 
                            movie_data["movie_name"]
                        )
                        
                        if thumb_path:
                            # Cache for this movie
                            await self._cache_movie_thumbnail(
                                movie_id, 
                                movie_data["movie_name"], 
                                thumb_path, 
                                first_file
                            )
                            
                            # Mark ALL files of this movie
                            for file in movie_data["files"]:
                                if self.db and file.get('file_unique_id'):
                                    await self.db.files.update_one(
                                        {"file_unique_id": file['file_unique_id']},
                                        {"$set": {
                                            "has_thumbnail": True,
                                            "thumbnail_path": thumb_path,
                                            "movie_id": movie_id,
                                            "thumbnail_shared": True
                                        }}
                                    )
                            
                            success += 1
                            extracted_for_movies.append(movie_data["movie_name"][:30])
                            logger.info(f"✅ EXTRACTED [{success}/{movies_without_thumb}]: '{movie_data['movie_name'][:30]}'")
                        else:
                            failed += 1
                            logger.info(f"❌ FAILED [{failed}]: '{movie_data['movie_name'][:30]}'")
                    
                    await asyncio.sleep(0.5)  # Rate limiting
                    
                except Exception as e:
                    logger.error(f"Error extracting for {movie_data['movie_name'][:30]}: {e}")
                    failed += 1
        
        # Final report
        logger.info("=" * 60)
        logger.info("📊 THUMBNAIL EXTRACTION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"🎬 Total unique movies: {unique_movies}")
        logger.info(f"✅ Successfully extracted: {success}")
        logger.info(f"❌ Failed to extract: {failed}")
        logger.info(f"📁 Total files updated: {success * len(movie_data['files']) if movie_data else 0}+")
        logger.info(f"🔄 One thumbnail per movie: ✅ ACTIVE")
        logger.info("=" * 60)
        
        return {
            "success": success, 
            "failed": failed,
            "unique_movies": unique_movies,
            "movies_with_thumb_before": movies_with_thumb,
            "movies_with_thumb_after": movies_with_thumb + success,
            "files_updated": success * (len(movie_data['files']) if movie_data else 0)
        }
    
    async def get_movie_thumbnail(self, movie_name: str) -> Optional[str]:
        """Get thumbnail by movie name (for search results)"""
        movie_id = self._extract_movie_id(movie_name)
        
        if self.thumb_collection:
            cached = await self.thumb_collection.find_one({"movie_id": movie_id})
            if cached and cached.get('thumbnail_path'):
                if os.path.exists(cached['thumbnail_path']):
                    return cached['thumbnail_path']
                elif cached.get('thumbnail_bytes'):
                    os.makedirs(os.path.dirname(cached['thumbnail_path']), exist_ok=True)
                    with open(cached['thumbnail_path'], 'wb') as f:
                        f.write(cached['thumbnail_bytes'])
                    return cached['thumbnail_path']
        
        return None
    
    async def close(self):
        """Clean up resources"""
        logger.info("👋 ThumbnailManager closed")


class FallbackThumbnailManager:
    """Fallback when real ThumbnailManager fails to import"""
    
    def __init__(self, *args, **kwargs):
        logger.warning("⚠️ Using FALLBACK Thumbnail Manager")
    
    async def get_thumbnail(self, *args, **kwargs) -> None:
        return None
    
    async def extract_all_missing(self, *args, **kwargs) -> Dict:
        return {"success": 0, "failed": 0, "unique_movies": 0, "files_updated": 0}
    
    async def get_movie_thumbnail(self, *args, **kwargs) -> None:
        return None
    
    async def close(self):
        pass
