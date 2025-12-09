"""
utils.py - Shared utility functions for SK4FiLM
"""
import re
import html
import logging
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def normalize_title(title: str) -> str:
    """Normalize title for searching"""
    if not title:
        return ""
    normalized = title.lower().strip()
    
    normalized = re.sub(
        r'\b(19|20)\d{2}\b|\b(480p|720p|1080p|2160p|4k|hd|fhd|uhd|hevc|x264|x265|h264|h265|bluray|webrip|hdrip|web-dl|hdtv|hdrip|webdl|hindi|english|tamil|telugu|malayalam|kannada|punjabi|bengali|marathi|gujarati|movie|film|series|complete|full|part|episode|season|hdrc|dvdscr|pre-dvd|p-dvd|pdc|rarbg|yts|amzn|netflix|hotstar|prime|disney|hc-esub|esub|subs)\b',
        '', 
        normalized, 
        flags=re.IGNORECASE
    )
    
    normalized = re.sub(r'[\._\-]', ' ', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized

def extract_title_smart(text: str) -> Optional[str]:
    """Smart title extraction"""
    if not text or len(text) < 10:
        return None
    
    try:
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if not lines:
            return None
        
        first_line = lines[0]
        
        patterns = [
            (r'^([A-Za-z\s]{3,50}?)\s*(?:\(?\d{4}\)?|\b(?:480p|720p|1080p|2160p|4k|hd|fhd|uhd)\b)', 1),
            (r'🎬\s*([^\n\-\(]{3,60}?)\s*(?:\(\d{4}\)|$)', 1),
            (r'^([^\-\n]{3,60}?)\s*\-', 1),
            (r'^([^\(\n]{3,60}?)\s*\(\d{4}\)', 1),
            (r'^([A-Za-z\s]{3,50}?)\s*(?:\d{4}|Hindi|Movie|Film|HDTC|WebDL|X264|AAC|ESub)', 1),
        ]
        
        for pattern, group in patterns:
            match = re.search(pattern, first_line, re.IGNORECASE)
            if match:
                title = match.group(group).strip()
                title = re.sub(r'\s+', ' ', title)
                if 3 <= len(title) <= 60:
                    return title
        
        # FALLBACK
        if len(first_line) >= 3:
            clean_title = re.sub(
                r'\b(480p|720p|1080p|2160p|4k|hd|fhd|uhd|hevc|x264|x265|h264|h265|bluray|webrip|hdrip|web-dl|hdtv|hdrip|webdl|hindi|english|tamil|telugu|malayalam|kannada|punjabi|bengali|marathi|gujarati|movie|film|series|complete|full|part|episode|season|\d{4}|hdrc|dvdscr|pre-dvd|p-dvd|pdc|rarbg|yts|amzn|netflix|hotstar|prime|disney|hc-esub|esub|subs)\b',
                '', 
                first_line, 
                flags=re.IGNORECASE
            )
            
            clean_title = re.sub(r'[\._\-]', ' ', clean_title)
            clean_title = re.sub(r'\s+', ' ', clean_title).strip()
            
            clean_title = re.sub(r'\s+\(\d{4}\)$', '', clean_title)
            clean_title = re.sub(r'\s+\d{4}$', '', clean_title)
            
            if 3 <= len(clean_title) <= 60:
                return clean_title
                
    except Exception as e:
        logger.error(f"Title extraction error: {e}")
    
    return None

def extract_title_from_file(file_name: str, caption: Optional[str] = None) -> Optional[str]:
    """Extract title from file name and caption"""
    try:
        if caption:
            t = extract_title_smart(caption)
            if t:
                return t
        
        if file_name:
            name = file_name.rsplit('.', 1)[0]
            name = re.sub(r'[\._\-]', ' ', name)
            
            name = re.sub(
                r'\b(720p|1080p|480p|2160p|4k|HDRip|WEBRip|WEB-DL|BluRay|BRRip|DVDRip|HDTV|HDTC|X264|X265|HEVC|H264|H265|AAC|AC3|DD5\.1|DDP5\.1|HC|ESub|Subs|Hindi|English|Dual|Multi|Complete|Full|Movie|Film|\d{4})\b',
                '', 
                name, 
                flags=re.IGNORECASE
            )
            
            name = re.sub(r'\s+', ' ', name).strip()
            name = re.sub(r'\s+\(\d{4}\)$', '', name)
            name = re.sub(r'\s+\d{4}$', '', name)
            
            if 4 <= len(name) <= 50:
                return name
    except Exception as e:
        logger.error(f"File title extraction error: {e}")
    return None

def format_size(size: int) -> str:
    """Format file size"""
    if not size:
        return "Unknown"
    if size < 1024*1024:
        return f"{size/1024:.1f} KB"
    elif size < 1024*1024*1024:
        return f"{size/(1024*1024):.1f} MB"
    else:
        return f"{size/(1024*1024*1024):.2f} GB"

def detect_quality(filename: str) -> str:
    """Detect video quality from filename"""
    if not filename:
        return "480p"
    fl = filename.lower()
    is_hevc = 'hevc' in fl or 'x265' in fl
    if '2160p' in fl or '4k' in fl:
        return "2160p HEVC" if is_hevc else "2160p"
    elif '1080p' in fl:
        return "1080p HEVC" if is_hevc else "1080p"
    elif '720p' in fl:
        return "720p HEVC" if is_hevc else "720p"
    elif '480p' in fl:
        return "480p HEVC" if is_hevc else "480p"
    return "480p"

def is_video_file(file_name: Optional[str]) -> bool:
    """Check if file is a video"""
    if not file_name:
        return False
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.mpg', '.mpeg']
    file_name_lower = file_name.lower()
    return any(file_name_lower.endswith(ext) for ext in video_extensions)

def format_post(text: str) -> str:
    """Format post text for HTML display"""
    if not text:
        return ""
    text = html.escape(text)
    text = re.sub(r'(https?://[^\s]+)', r'<a href="\1" target="_blank" style="color:#00ccff">\1</a>', text)
    return text.replace('\n', '<br>')

def is_new(date: Any) -> bool:
    """Check if date is within last 48 hours"""
    try:
        if isinstance(date, str):
            date = datetime.fromisoformat(date.replace('Z', '+00:00'))
        hours = (datetime.now() - date.replace(tzinfo=None)).total_seconds() / 3600
        return hours <= 48
    except:
        return False
