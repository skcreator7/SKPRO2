"""
utils.py - Optimized utility functions
"""
import re
import html
import logging
from datetime import datetime
from typing import Optional
from functools import lru_cache

logger = logging.getLogger(__name__)

@lru_cache(maxsize=10000)
def normalize_title(title: str) -> str:
    """Cached title normalization"""
    if not title:
        return ""
    normalized = title.lower().strip()
    
    # Remove common patterns
    patterns = r'\b(480p|720p|1080p|2160p|4k|hd|bluray|webrip|hindi|english|movie|film|\d{4})\b'
    normalized = re.sub(patterns, '', normalized, flags=re.IGNORECASE)
    
    normalized = re.sub(r'[\._\-]', ' ', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized

def extract_title_smart(text: str) -> Optional[str]:
    """Fast title extraction"""
    if not text or len(text) < 10:
        return None
    
    try:
        first_line = text.split('\n')[0].strip()
        if len(first_line) < 3:
            return None
        
        # Simple pattern matching
        patterns = [
            r'^([^\-\n\(]{3,50}?)\s*[\-\:]',
            r'^([A-Za-z\s]{3,50}?)\s*\(\d{4}\)',
            r'🎬\s*([^\n]{3,50}?)\s*(?:\(\d{4}\)|$)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, first_line, re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                if 3 <= len(title) <= 60:
                    return title
        
        return first_line[:60].strip()
    except:
        return None

@lru_cache(maxsize=10000)
def is_video_file(file_name: Optional[str]) -> bool:
    """Cached video file check"""
    if not file_name:
        return False
    video_extensions = ('.mp4', '.mkv', '.avi', '.mov')
    return file_name.lower().endswith(video_extensions)

def format_post(text: str) -> str:
    """Fast HTML formatting"""
    if not text:
        return ""
    text = html.escape(text)
    text = re.sub(r'(https?://[^\s]+)', r'<a href="\1" target="_blank">\1</a>', text)
    return text.replace('\n', '<br>')

@lru_cache(maxsize=10000)
def is_new(date_str: str) -> bool:
    """Cached new check"""
    try:
        if isinstance(date_str, str):
            date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        else:
            date = date_str
        
        hours = (datetime.now() - date.replace(tzinfo=None)).total_seconds() / 3600
        return hours <= 48
    except:
        return False

def detect_quality(filename: str) -> str:
    """Fast quality detection"""
    if not filename:
        return "480p"
    
    fl = filename.lower()
    if '2160p' in fl or '4k' in fl:
        return "2160p"
    elif '1080p' in fl:
        return "1080p"
    elif '720p' in fl:
        return "720p"
    return "480p"

def format_size(size: int) -> str:
    """Fast size formatting"""
    if size < 1024*1024:
        return f"{size/1024:.1f} KB"
    elif size < 1024*1024*1024:
        return f"{size/(1024*1024):.1f} MB"
    else:
        return f"{size/(1024*1024*1024):.2f} GB"
