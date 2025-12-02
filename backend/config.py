"""
SK4FiLM Configuration
Centralized configuration to avoid circular imports
"""
import os
from typing import List

class Config:
    """Application configuration"""
    
    # Telegram Bot Configuration
    API_ID = int(os.getenv("API_ID", "0"))
    API_HASH = os.getenv("API_HASH", "")
    BOT_TOKEN = os.getenv("BOT_TOKEN", "")
    BOT_USERNAME = os.getenv("BOT_USERNAME", "sk4film_bot")
    USER_SESSION_STRING = os.getenv("USER_SESSION_STRING", "")
    
    # Admin Configuration
    ADMIN_IDS_STR = os.getenv("ADMIN_IDS", "")
    ADMIN_IDS: List[int] = [int(x.strip()) for x in ADMIN_IDS_STR.split(",") if x.strip()]
    
    # Channel Configuration
    FILE_CHANNEL_ID = int(os.getenv("FILE_CHANNEL_ID", "0"))
    TEXT_CHANNEL_IDS_STR = os.getenv("TEXT_CHANNEL_IDS", "")
    TEXT_CHANNEL_IDS: List[int] = [int(x.strip()) for x in TEXT_CHANNEL_IDS_STR.split(",") if x.strip()]
    
    # Database Configuration
    MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    DATABASE_NAME = os.getenv("DATABASE_NAME", "sk4film")
    
    # Redis Configuration
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
    REDIS_DB = int(os.getenv("REDIS_DB", "0"))
    
    # Website Configuration
    WEBSITE_URL = os.getenv("WEBSITE_URL", "https://sk4film.com")
    PORT = int(os.getenv("PORT", "8000"))
    
    # Feature Flags
    VERIFICATION_REQUIRED = os.getenv("VERIFICATION_REQUIRED", "true").lower() == "true"
    AUTO_DELETE_TIME = int(os.getenv("AUTO_DELETE_TIME", "1800"))  # 30 minutes in seconds
    
    # Payment Configuration
    UPI_ID_BASIC = os.getenv("UPI_ID_BASIC", "")
    UPI_ID_PREMIUM = os.getenv("UPI_ID_PREMIUM", "")
    
    # Cache Configuration
    CACHE_EXPIRE_TIME = int(os.getenv("CACHE_EXPIRE_TIME", "3600"))  # 1 hour
    
    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "sk4film.log")
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        required = {
            "API_ID": cls.API_ID,
            "API_HASH": cls.API_HASH,
            "BOT_TOKEN": cls.BOT_TOKEN,
            "MONGODB_URI": cls.MONGODB_URI
        }
        
        missing = [k for k, v in required.items() if not v or (isinstance(v, int) and v == 0)]
        if missing:
            raise ValueError(f"Missing required config: {', '.join(missing)}")
        
        return True
