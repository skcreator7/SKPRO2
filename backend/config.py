"""
config.py - Configuration settings
"""
import os

class Config:
    # Telegram API
    API_ID = int(os.environ.get("API_ID", "0"))
    API_HASH = os.environ.get("API_HASH", "")
    USER_SESSION_STRING = os.environ.get("USER_SESSION_STRING", "")
    BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
    
    # Database
    MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
    REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
    
    # Channels
    MAIN_CHANNEL_ID = -1001891090100
    TEXT_CHANNEL_IDS = [-1001891090100, -1002024811395]
    FILE_CHANNEL_ID = -1001768249569
    
    # URLs
    MAIN_CHANNEL_LINK = "https://t.me/sk4film"
    UPDATES_CHANNEL_LINK = "https://t.me/sk4film_Request"
    CHANNEL_USERNAME = "sk4film"
    WEBSITE_URL = os.environ.get("WEBSITE_URL", "https://sk4film.vercel.app")
    BACKEND_URL = os.environ.get("BACKEND_URL", "https://sk4film.koyeb.app")
    
    # Bot
    BOT_USERNAME = os.environ.get("BOT_USERNAME", "sk4filmbot")
    ADMIN_IDS = [int(x) for x in os.environ.get("ADMIN_IDS", "123456789").split(",")]
    AUTO_DELETE_TIME = int(os.environ.get("AUTO_DELETE_TIME", "300"))
    
    # Shortener & Verification
    SHORTLINK_API = os.environ.get("SHORTLINK_API", "")
    VERIFICATION_REQUIRED = os.environ.get("VERIFICATION_REQUIRED", "false").lower() == "true"
    VERIFICATION_DURATION = 6 * 60 * 60  # 6 hours
    
    # Server
    WEB_SERVER_PORT = int(os.environ.get("PORT", 8000))
    
    # API Keys
    OMDB_KEYS = os.environ.get("OMDB_KEYS", "8265bd1c,b9bd48a6,3e7e1e9d").split(",")
    TMDB_KEYS = os.environ.get("TMDB_KEYS", "e547e17d4e91f3e62a571655cd1ccaff,8265bd1f").split(",")
    
    # UPI IDs for Premium
    UPI_ID_BASIC = os.environ.get("UPI_ID_BASIC", "sk4filmbot@ybl")
    UPI_ID_PREMIUM = os.environ.get("UPI_ID_PREMIUM", "sk4filmbot@ybl")
    UPI_ID_GOLD = os.environ.get("UPI_ID_GOLD", "sk4filmbot@ybl")
    UPI_ID_DIAMOND = os.environ.get("UPI_ID_DIAMOND", "sk4filmbot@ybl")
    
    # Admin
    ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "admin123")
