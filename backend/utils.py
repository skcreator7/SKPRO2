"""
Shared utility functions for SK4FiLM
Prevents circular imports by centralizing common functions
"""
import re
import logging
import asyncio
from typing import Any, Callable, Optional, AsyncGenerator
from datetime import datetime
from pyrogram.errors import FloodWait, RPCError

logger = logging.getLogger(__name__)

# ==================== TEXT PROCESSING ====================

def normalize_title(title: str) -> str:
    """
    Normalize title for comparison and search
    Removes special chars, converts to lowercase
    """
    if not title:
        return ""
    
    # Remove quality indicators
    title = re.sub(r'\b(480p|720p|1080p|2160p|4k|hd|uhd|bluray|webrip|hdts|cam|dvdrip)\b', '', title, flags=re.IGNORECASE)
    
    # Remove year
    title = re.sub(r'\b(19|20)\d{2}\b', '', title)
    
    # Remove special characters
    title = re.sub(r'[^\w\s]', ' ', title)
    
    # Remove extra spaces
    title = ' '.join(title.split())
    
    return title.lower().strip()


def extract_title_from_file(filename: str) -> str:
    """
    Extract clean title from filename
    Handles various file naming patterns
    """
    if not filename:
        return ""
    
    # Remove file extension
    name = re.sub(r'\.[a-z0-9]+$', '', filename, flags=re.IGNORECASE)
    
    # Remove quality, codec, etc
    name = re.sub(r'\b(x264|x265|hevc|aac|ac3|dts|480p|720p|1080p|2160p|4k)\b', '', name, flags=re.IGNORECASE)
    
    # Replace dots and underscores with spaces
    name = name.replace('.', ' ').replace('_', ' ').replace('-', ' ')
    
    # Remove season/episode info
    name = re.sub(r'\b(s\d+e\d+|season\s*\d+|episode\s*\d+)\b', '', name, flags=re.IGNORECASE)
    
    # Remove extra spaces
    name = ' '.join(name.split())
    
    return name.strip()


def extract_title_smart(text: str) -> str:
    """
    Smart title extraction from any text
    Tries multiple patterns
    """
    if not text:
        return ""
    
    # Try to find title before year
    match = re.search(r'^(.+?)\s+(19|20)\d{2}', text)
    if match:
        return match.group(1).strip()
    
    # Try to find title before quality
    match = re.search(r'^(.+?)\s+(480p|720p|1080p|2160p|4k)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Fallback to full extraction
    return extract_title_from_file(text)


# ==================== FILE PROCESSING ====================

def format_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if not size_bytes or size_bytes < 0:
        return "0 B"
    
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    size = float(size_bytes)
    unit_index = 0
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    return f"{size:.2f} {units[unit_index]}"


def detect_quality(filename: str) -> str:
    """
    Detect video quality from filename
    Returns: 4K, 1080p, 720p, 480p, or HD
    """
    if not filename:
        return "HD"
    
    filename_lower = filename.lower()
    
    if '2160p' in filename_lower or '4k' in filename_lower or 'uhd' in filename_lower:
        return '4K'
    elif '1080p' in filename_lower or 'fhd' in filename_lower:
        return '1080p'
    elif '720p' in filename_lower or 'hd' in filename_lower:
        return '720p'
    elif '480p' in filename_lower or 'sd' in filename_lower:
        return '480p'
    
    return 'HD'


def is_video_file(filename: str) -> bool:
    """Check if file is a video based on extension"""
    if not filename:
        return False
    
    video_extensions = [
        '.mp4', '.mkv', '.avi', '.mov', '.wmv',
        '.flv', '.webm', '.m4v', '.mpg', '.mpeg',
        '.3gp', '.ts', '.vob'
    ]
    
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in video_extensions)


# ==================== TELEGRAM OPERATIONS ====================

async def safe_telegram_operation(operation: Callable, *args, max_retries: int = 3, **kwargs) -> Any:
    """
    Execute Telegram operation with retry logic
    Handles FloodWait and other errors gracefully
    """
    for attempt in range(max_retries):
        try:
            return await operation(*args, **kwargs)
        
        except FloodWait as e:
            wait_time = e.value + 1
            if attempt < max_retries - 1:
                logger.warning(f"FloodWait: waiting {wait_time} seconds... (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"FloodWait: max retries reached")
                raise
        
        except RPCError as e:
            if attempt < max_retries - 1:
                logger.error(f"RPCError: {e}, retrying... (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(2)
            else:
                logger.error(f"RPCError: max retries reached - {e}")
                raise
        
        except Exception as e:
            if attempt < max_retries - 1:
                logger.error(f"Unexpected error: {e}, retrying... (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(2)
            else:
                logger.error(f"Unexpected error: max retries reached - {e}")
                raise
    
    return None


async def safe_telegram_generator(operation: Callable, *args, **kwargs) -> AsyncGenerator:
    """
    Async generator wrapper for Telegram operations
    Handles FloodWait during iteration
    """
    try:
        async for item in operation(*args, **kwargs):
            yield item
    
    except FloodWait as e:
        wait_time = e.value + 1
        logger.warning(f"FloodWait during iteration: waiting {wait_time} seconds...")
        await asyncio.sleep(wait_time)
        
        # Retry after wait
        async for item in operation(*args, **kwargs):
            yield item
    
    except Exception as e:
        logger.error(f"Error in generator: {e}")
        raise


async def auto_delete_file(message, delay: int):
    """
    Auto-delete message after specified delay
    Used for temporary file sharing
    """
    try:
        await asyncio.sleep(delay)
        await message.delete()
        logger.info(f"Auto-deleted message {message.id} after {delay}s")
    
    except Exception as e:
        logger.error(f"Failed to auto-delete message {message.id}: {e}")


# ==================== DATABASE HELPERS ====================

async def index_single_file(message):
    """
    Index a single file message
    Note: Actual DB operation should be in database.py
    This is just a placeholder
    """
    try:
        if not (message.document or message.video):
            return False
        
        # Extract file info
        file_obj = message.document or message.video
        filename = getattr(file_obj, 'file_name', '') or f"video_{message.id}.mp4"
        
        logger.info(f"Indexing file: {filename}")
        
        # TODO: Call database operation here
        # from database import store_file_metadata
        # await store_file_metadata(message)
        
        return True
    
    except Exception as e:
        logger.error(f"Failed to index file: {e}")
        return False


# ==================== VALIDATION ====================

def validate_user_id(user_id: Any) -> bool:
    """Validate Telegram user ID"""
    try:
        uid = int(user_id)
        return uid > 0
    except (ValueError, TypeError):
        return False


def validate_channel_id(channel_id: Any) -> bool:
    """Validate Telegram channel ID"""
    try:
        cid = int(channel_id)
        return cid != 0
    except (ValueError, TypeError):
        return False
