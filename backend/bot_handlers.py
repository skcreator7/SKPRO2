import asyncio
import time
import logging
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class BotInstance:
    """Bot instance with all systems"""

    def __init__(self, bot, config):
        self.bot = bot
        self.config = config
        self.user_client = None
        self.user_session_ready = False
        self.bot_session_ready = False
        self.bot_started = False

        # Database collections
        self.files_col = None
        self.verification_col = None

        # Systems
        self.verification_system = None
        self.premium_system = None

        # Rate limiting
        self.rate_limits = defaultdict(lambda: defaultdict(list))
        self.processing_requests = {}

        # Auto-delete
        self.auto_delete_tasks = {}
        self.file_messages_to_delete = {}

        # Premium tiers
        from enum import Enum
        class PremiumTier(Enum):
            FREE = "free"
            BASIC = "basic"
            PREMIUM = "premium"
            GOLD = "gold"
            DIAMOND = "diamond"

        self.PremiumTier = PremiumTier

        logger.info("🎮 BotInstance created")

    async def initialize(self):
        """Initialize all bot systems"""
        try:
            logger.info("🔧 Initializing bot systems...")

            # Initialize verification system if needed
            if self.verification_col is not None:
                logger.info("✅ Verification system ready")

            # Initialize premium system if needed
            # Can add premium initialization here

            self.bot_started = True
            logger.info("✅ Bot systems initialized")
            return True

        except Exception as e:
            logger.error(f"❌ Bot initialization error: {e}")
            return False

    async def check_rate_limit(self, user_id, limit=5, window=60, request_type="general"):
        """Check if user is within rate limit"""
        try:
            now = time.time()
            key = f"{user_id}_{request_type}"

            # Clean old entries
            self.rate_limits[key] = [t for t in self.rate_limits[key] if now - t < window]

            # Check limit
            if len(self.rate_limits[key]) >= limit:
                return False

            # Add current request
            self.rate_limits[key].append(now)
            return True

        except Exception as e:
            logger.error(f"Rate limit check error: {e}")
            return True  # Allow on error

    async def is_request_duplicate(self, user_id, request_data, request_type="general"):
        """Check if request is duplicate"""
        key = f"{user_id}_{request_type}_{request_data}"

        if key in self.processing_requests:
            return True

        self.processing_requests[key] = time.time()
        return False

    async def clear_processing_request(self, user_id, request_data, request_type="general"):
        """Clear processing request"""
        key = f"{user_id}_{request_type}_{request_data}"
        if key in self.processing_requests:
            del self.processing_requests[key]

    async def schedule_file_deletion(self, user_id, message_id, file_name, minutes):
        """Schedule file deletion"""
        try:
            await asyncio.sleep(minutes * 60)

            try:
                await self.bot.delete_messages(user_id, message_id)
                logger.info(f"🗑️ Auto-deleted: {file_name} for user {user_id}")
            except Exception as e:
                logger.error(f"Delete error: {e}")

            # Clean up tracking
            task_id = f"{user_id}_{message_id}"
            if task_id in self.file_messages_to_delete:
                del self.file_messages_to_delete[task_id]
            if task_id in self.auto_delete_tasks:
                del self.auto_delete_tasks[task_id]

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Schedule deletion error: {e}")


def format_size(size_bytes):
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0B"

    import math
    size_names = ("B", "KB", "MB", "GB", "TB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"
