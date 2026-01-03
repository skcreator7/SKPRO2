"""
utils.py - Utility functions for SK4FiLM
"""
import re
import os
from datetime import datetime
from typing import Optional, List, Dict, Any

def normalize_title(title: str) -> str:
    """Normalize title for comparison"""
    if not title:
        return ""
    
    # Convert to lowercase
    title = title.lower().strip()
    
    # Remove common suffixes and patterns
    patterns_to_remove = [
        r'\s*\([^)]*\)$',  # Remove parentheses at end
        r'\s*\[[^\]]*\]$',  # Remove brackets at end
        r'\s*\d{4}$',  # Remove year at end
        r'\b(480p|720p|1080p|2160p|4k|uhd|hd|hevc|x265|x264|h264|h265)\b',
        r'\b(web|webrip|webdl|bluray|brrip|dvdrip|hdtv)\b',
        r'\b(hindi|english|tamil|telugu|malayalam|bengali)\b',
        r'\b(dual|multi)\b',
        r'\b(ac3|aac|dd5\.1|dts)\b',
        r'[._]',
        r'\s+'
    ]
    
    for pattern in patterns_to_remove:
        title = re.sub(pattern, ' ', title, flags=re.IGNORECASE)
    
    # Clean up multiple spaces
    title = re.sub(r'\s+', ' ', title)
    
    return title.strip()

def extract_title_smart(text: str) -> str:
    """Extract title from text smartly"""
    if not text:
        return ""
    
    # Split into lines
    lines = text.split('\n')
    
    # Find the first line that looks like a title
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
        
        # Skip lines that are too short or are URLs
        if len(line) < 10 or line.startswith('http'):
            continue
        
        # Skip common indicators
        if any(indicator in line.lower() for indicator in ['download', 'click here', 'join', 'telegram', '@']):
            continue
        
        # This looks like a title
        # Clean it up
        title = line
        
        # Remove emojis and special characters at beginning
        title = re.sub(r'^[🎬📥📹⭐🌟✅⚡🔗🔒🛡️🗑️⏰📁📦🔍🎯💎💰📅👑👥📊🛠️🧹👋🌐🤖👤🔌📡🔧📝📄📥🔍🎬]+', '', title)
        
        # Remove extra spaces
        title = title.strip()
        
        if title and len(title) > 5:
            return title[:200]
    
    # Fallback: return first 100 characters
    return text[:100].strip()

def extract_title_from_file(filename: Optional[str], caption: Optional[str] = None) -> str:
    """Extract title from filename or caption"""
    # Try filename first
    if filename:
        # Remove extension
        name = os.path.splitext(filename)[0]
        
        # Clean the name
        name = normalize_title(name)
        
        if name and len(name) > 3:
            return name
    
    # Try caption
    if caption:
        title = extract_title_smart(caption)
        if title:
            return title
    
    return "Unknown File"

def format_size(size: int) -> str:
    """Format file size in human readable format"""
    if not size:
        return "Unknown"
    
    if size < 1024:
        return f"{size} B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    elif size < 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    else:
        return f"{size / (1024 * 1024 * 1024):.2f} GB"

def detect_quality(filename: Optional[str]) -> str:
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
    
    for pattern, quality in quality_patterns:
        if re.search(pattern, filename_lower):
            return quality
    
    return "480p"

def is_video_file(filename: Optional[str]) -> bool:
    """Check if file is a video file"""
    if not filename:
        return False
    
    video_extensions = [
        '.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v',
        '.mpg', '.mpeg', '.3gp', '.ogg', '.ogv', '.qt', '.rm', '.rmvb',
        '.vob', '.asf', '.amv', '.m4p', '.m4v', '.f4v', '.f4p', '.f4a', '.f4b'
    ]
    
    return any(filename.lower().endswith(ext) for ext in video_extensions)

def format_post(text: str, max_length: Optional[int] = None) -> str:
    """Format post text"""
    if not text:
        return ""
    
    # Clean up text
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Remove extra blank lines
    
    # Truncate if needed
    if max_length and len(text) > max_length:
        text = text[:max_length] + "..."
    
    return text.strip()

def is_new(date) -> bool:
    """Check if date is recent (within 7 days)"""
    if not date:
        return False
    
    try:
        if isinstance(date, str):
            # Try to parse date string
            date = datetime.fromisoformat(date.replace('Z', '+00:00'))
        
        # Check if within last 7 days
        time_diff = datetime.now() - date
        return time_diff.days < 7
    
    except Exception:
        return False

def extract_year_from_title(title: str) -> str:
    """Extract year from title"""
    if not title:
        return ""
    
    year_match = re.search(r'\b(19|20)\d{2}\b', title)
    return year_match.group() if year_match else ""

def clean_filename(filename: str) -> str:
    """Clean filename by removing quality indicators"""
    if not filename:
        return ""
    
    # Remove extension
    name = os.path.splitext(filename)[0]
    
    # Remove common patterns
    patterns_to_remove = [
        r'\b(480p|720p|1080p|2160p|4k|uhd|hd|hevc|x265|x264|h264|h265)\b',
        r'\b(web|webrip|webdl|bluray|brrip|dvdrip|hdtv)\b',
        r'\b(hindi|english|tamil|telugu|malayalam|bengali)\b',
        r'\b(dual|multi)\b',
        r'\b(ac3|aac|dd5\.1|dts)\b',
        r'[._]',
        r'\[.*?\]',
        r'\(.*?\)'
    ]
    
    for pattern in patterns_to_remove:
        name = re.sub(pattern, ' ', name, flags=re.IGNORECASE)
    
    # Clean up
    name = re.sub(r'\s+', ' ', name)
    name = name.strip()
    
    return name

def get_file_extension(filename: str) -> str:
    """Get file extension"""
    if not filename:
        return ""
    
    return os.path.splitext(filename)[1].lower()

def is_valid_quality(quality: str) -> bool:
    """Check if quality string is valid"""
    valid_qualities = ['360p', '480p', '720p', '1080p', '2160p', '4k', 'uhd']
    return quality.lower() in valid_qualities or 'hevc' in quality.lower()
