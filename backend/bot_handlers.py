import asyncio
import logging
import re
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

# Try to import Pyrogram with better error handling
try:
    # Try to install Pyrogram if not available
    import subprocess
    import sys
    
    # First check if pyrogram is installed
    try:
        import pyrogram
        logger.info("✅ Pyrogram already installed")
    except ImportError:
        logger.info("Installing Pyrogram...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyrogram", "tgcrypto", "--quiet"])
        logger.info("✅ Pyrogram installed successfully")
    
    # Now import
    from pyrogram import Client, filters
    from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton, Message, CallbackQuery
    from pyrogram.errors import FloodWait, BadRequest, MessageDeleteForbidden
    
    PYROGRAM_AVAILABLE = True
    logger.info("✅ Pyrogram imported successfully")
    
except Exception as e:
    logger.error(f"❌ Pyrogram import failed: {e}")
    PYROGRAM_AVAILABLE = False
    
    # Create dummy classes for fallback
    class Client:
        def __init__(self, *args, **kwargs):
            self.name = kwargs.get('name', 'bot')
            self.api_id = kwargs.get('api_id')
            self.api_hash = kwargs.get('api_hash')
            self.bot_token = kwargs.get('bot_token')
            self.session_string = kwargs.get('session_string')
            
        async def start(self):
            logger.info(f"Started client: {self.name}")
            return self
            
        async def stop(self):
            logger.info(f"Stopped client: {self.name}")
            
        async def get_messages(self, chat_id, message_id):
            class DummyMessage:
                def __init__(self):
                    self.id = message_id
                    self.chat = type('obj', (object,), {'id': chat_id})
                    self.document = None
                    self.video = None
            return DummyMessage()
            
        async def send_message(self, chat_id, text, **kwargs):
            logger.info(f"Send message to {chat_id}: {text[:50]}...")
            return type('obj', (object,), {'id': 123, 'chat': type('obj', (object,), {'id': chat_id})})()
            
        async def send_document(self, chat_id, document, **kwargs):
            logger.info(f"Send document to {chat_id}")
            return type('obj', (object,), {'id': 124})()
            
        async def send_video(self, chat_id, video, **kwargs):
            logger.info(f"Send video to {chat_id}")
            return type('obj', (object,), {'id': 125})()
            
        async def delete_messages(self, chat_id, message_ids):
            logger.info(f"Delete messages {message_ids} from {chat_id}")
            
        async def get_me(self):
            return type('obj', (object,), {
                'id': 123456789,
                'username': 'test_bot',
                'first_name': 'Test Bot'
            })()
    
    class filters:
        @staticmethod
        def command(cmd):
            return lambda func: func
        
        @staticmethod
        def private():
            return lambda func: func
        
        @staticmethod
        def regex(pattern):
            return lambda func: func
        
        @staticmethod
        def user(users):
            return lambda func: func
        
        @staticmethod
        def photo():
            return lambda func: func
        
        @staticmethod
        def document():
            return lambda func: func
        
        text = staticmethod(lambda: lambda func: func)
    
    class InlineKeyboardMarkup:
        def __init__(self, buttons):
            self.buttons = buttons
    
    class InlineKeyboardButton:
        def __init__(self, text, url=None, callback_data=None):
            self.text = text
            self.url = url
            self.callback_data = callback_data
    
    class Message:
        def __init__(self):
            self.id = 0
            self.from_user = type('obj', (object,), {
                'id': 0,
                'first_name': 'User',
                'last_name': None,
                'username': None
            })()
            self.chat = type('obj', (object,), {'id': 0})()
            self.text = ''
            self.photo = None
            self.document = None
            self.command = []
    
    class CallbackQuery:
        def __init__(self):
            self.id = 'test_callback'
            self.from_user = type('obj', (object,), {
                'id': 0,
                'first_name': 'User'
            })()
            self.message = Message()
            self.data = ''
            
        async def answer(self, text=None, show_alert=False):
            logger.info(f"Callback answer: {text}")

class SK4FiLMBot:
    def __init__(self, config, db_manager=None):
        self.config = config
        self.db_manager = db_manager
        self.bot = None
        self.user_client = None
        self.bot_started = False
        self.user_session_ready = False
        
        # Track auto-delete tasks
        self.auto_delete_tasks = {}
        self.file_messages_to_delete = {}
        
        # Rate limiting
        self.user_request_times = defaultdict(list)
        self.processing_requests = {}
        self.verification_processing = {}
        
        logger.info("🔄 Initializing SK4FiLMBot systems...")
        
        # Initialize all systems with error handling
        self.verification_system = None
        self.premium_system = None
        self.PremiumTier = None
        self.poster_fetcher = None
        self.cache_manager = None
        
        try:
            from verification import VerificationSystem
            self.verification_system = VerificationSystem(config, db_manager)
            logger.info("✅ Verification System initialized")
        except Exception as e:
            logger.error(f"❌ Verification System init error: {e}")
        
        try:
            from premium import PremiumSystem, PremiumTier
            self.premium_system = PremiumSystem(config, db_manager)
            self.PremiumTier = PremiumTier
            logger.info("✅ Premium System initialized")
        except Exception as e:
            logger.error(f"❌ Premium System init error: {e}")
        
        try:
            from poster_fetching import PosterFetcher
            self.poster_fetcher = PosterFetcher(config)
            logger.info("✅ Poster Fetcher initialized")
        except Exception as e:
            logger.error(f"❌ Poster Fetcher init error: {e}")
        
        try:
            from cache import CacheManager
            self.cache_manager = CacheManager(config)
            asyncio.create_task(self.cache_manager.init_redis())
            logger.info("✅ Cache Manager initialized")
        except Exception as e:
            logger.error(f"❌ Cache Manager init error: {e}")
    
    async def initialize(self):
        """Initialize bot"""
        try:
            logger.info("🚀 Initializing SK4FiLM Bot...")
            
            if not PYROGRAM_AVAILABLE:
                logger.error("❌ Pyrogram not available. Running in limited mode.")
                # Even if Pyrogram is not available, we can still set up dummy handlers
                self.bot = Client(
                    "bot",
                    api_id=self.config.API_ID,
                    api_hash=self.config.API_HASH,
                    bot_token=self.config.BOT_TOKEN
                )
                self.bot_started = True
                await self.setup_handlers()
                logger.info("✅ Bot running in limited mode (no real Telegram connection)")
                return True
            
            # Initialize bot with real Pyrogram
            self.bot = Client(
                "bot",
                api_id=self.config.API_ID,
                api_hash=self.config.API_HASH,
                bot_token=self.config.BOT_TOKEN,
                workers=20,
                in_memory=True
            )
            
            # Initialize user client if available
            if hasattr(self.config, 'USER_SESSION_STRING') and self.config.USER_SESSION_STRING:
                try:
                    self.user_client = Client(
                        "user",
                        api_id=self.config.API_ID,
                        api_hash=self.config.API_HASH,
                        session_string=self.config.USER_SESSION_STRING,
                        in_memory=True
                    )
                    await self.user_client.start()
                    self.user_session_ready = True
                    logger.info("✅ User session started")
                except Exception as e:
                    logger.error(f"❌ User session failed: {e}")
                    self.user_session_ready = False
            
            # Start bot
            await self.bot.start()
            self.bot_started = True
            
            # Get bot info
            me = await self.bot.get_me()
            logger.info(f"✅ Bot started successfully: @{me.username}")
            
            # Setup handlers
            await self.setup_handlers()
            
            # Start cleanup tasks
            if self.verification_system:
                try:
                    asyncio.create_task(self.verification_system.start_cleanup_task())
                    logger.info("✅ Verification cleanup started")
                except:
                    pass
            
            if self.premium_system:
                try:
                    asyncio.create_task(self.premium_system.start_cleanup_task())
                    logger.info("✅ Premium cleanup started")
                except:
                    pass
            
            # Start auto-delete monitor
            asyncio.create_task(self._monitor_auto_delete())
            
            logger.info("🎉 Bot initialization complete!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Bot initialization failed: {e}")
            traceback.print_exc()
            
            # Even if initialization fails, set up dummy bot
            self.bot = Client(
                "bot_dummy",
                api_id=0,
                api_hash="dummy",
                bot_token="dummy"
            )
            self.bot_started = True
            await self.setup_handlers()
            logger.info("✅ Bot running in fallback mode")
            return True
    
    async def setup_handlers(self):
        """Setup all handlers"""
        try:
            # Import and setup handlers
            from bot_commands import setup_bot_handlers
            await setup_bot_handlers(self.bot, self, PYROGRAM_AVAILABLE)
            logger.info("✅ Bot handlers setup complete")
        except Exception as e:
            logger.error(f"❌ Handler setup failed: {e}")
            traceback.print_exc()
    
    async def shutdown(self):
        """Shutdown bot"""
        try:
            # Cancel all auto-delete tasks
            for task_id, task in self.auto_delete_tasks.items():
                try:
                    task.cancel()
                except:
                    pass
            
            if self.bot and self.bot_started:
                await self.bot.stop()
                logger.info("✅ Bot stopped")
            
            if self.user_client and self.user_session_ready:
                await self.user_client.stop()
                logger.info("✅ User client stopped")
                
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    # ✅ AUTO-DELETE SYSTEM
    async def schedule_file_deletion(self, user_id: int, message_id: int, file_name: str, delete_after_minutes: int):
        """Schedule file deletion after specified minutes"""
        try:
            task_id = f"{user_id}_{message_id}"
            
            # Wait for the specified time
            await asyncio.sleep(delete_after_minutes * 60)
            
            logger.info(f"⏰ Auto-delete time reached for message {message_id} (user {user_id})")
            
            # Try to delete the file message
            try:
                await self.bot.delete_messages(user_id, message_id)
                logger.info(f"✅ Auto-deleted message {message_id} for user {user_id}")
                
                # Send deletion notification
                await self.send_deletion_notification(user_id, file_name, delete_after_minutes)
                
            except Exception as e:
                logger.error(f"Error in auto-delete: {e}")
                await self.send_deletion_notification(user_id, file_name, delete_after_minutes, deleted=False)
            
            # Remove from tracking
            self.auto_delete_tasks.pop(task_id, None)
            self.file_messages_to_delete.pop(task_id, None)
            
        except asyncio.CancelledError:
            logger.info(f"Auto-delete task cancelled for message {message_id}")
        except Exception as e:
            logger.error(f"Error in auto-delete task: {e}")
    
    async def send_deletion_notification(self, user_id: int, file_name: str, delete_after_minutes: int, deleted: bool = True):
        """Send notification about file deletion"""
        try:
            website_url = getattr(self.config, 'WEBSITE_URL', 'https://sk4film.com')
            
            if deleted:
                text = (
                    f"🗑️ **File Auto-Deleted**\n\n"
                    f"`{file_name}`\n\n"
                    f"⏰ **Deleted after:** {delete_after_minutes} minutes\n"
                    f"✅ **Security measure completed**\n\n"
                    f"🔁 **Need the file again?**\n"
                    f"Visit website and download again\n"
                    f"🎬 @SK4FiLM"
                )
            else:
                text = (
                    f"⏰ **File Auto-Delete Time Reached**\n\n"
                    f"`{file_name}`\n\n"
                    f"⏰ **Delete time:** {delete_after_minutes} minutes\n"
                    f"⚠️ **File not deleted (permissions)**\n\n"
                    f"🔁 **Download again from:** {website_url}\n"
                    f"🎬 @SK4FiLM"
                )
            
            if PYROGRAM_AVAILABLE:
                buttons = InlineKeyboardMarkup([
                    [InlineKeyboardButton("🌐 VISIT WEBSITE", url=website_url)],
                    [InlineKeyboardButton("🔄 GET ANOTHER FILE", callback_data="back_to_start")]
                ])
                
                await self.bot.send_message(user_id, text, reply_markup=buttons)
                logger.info(f"✅ Deletion notification sent to user {user_id}")
            else:
                logger.info(f"[DUMMY] Would send deletion notification to user {user_id}: {text}")
            
        except Exception as e:
            logger.error(f"Failed to send deletion notification: {e}")
    
    async def _monitor_auto_delete(self):
        """Monitor auto-delete tasks"""
        while True:
            try:
                await asyncio.sleep(60)
                if self.auto_delete_tasks:
                    logger.info(f"📊 Auto-delete monitoring: {len(self.auto_delete_tasks)} active tasks")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-delete monitor error: {e}")
    
    # ✅ RATE LIMITING METHODS
    async def check_rate_limit(self, user_id, limit=3, window=60, request_type="file"):
        """Check if user is within rate limits"""
        now = time.time()
        key = f"{user_id}_{request_type}"
        
        # Clean old requests
        self.user_request_times[key] = [
            t for t in self.user_request_times.get(key, []) 
            if now - t < window
        ]
        
        # Check if limit exceeded
        if len(self.user_request_times[key]) >= limit:
            logger.warning(f"⚠️ Rate limit exceeded for user {user_id} ({request_type})")
            return False
        
        # Add current request
        self.user_request_times[key].append(now)
        return True
    
    async def is_request_duplicate(self, user_id, request_data, request_type="file"):
        """Check if this is a duplicate request"""
        request_hash = f"{user_id}_{request_type}_{hash(request_data)}"
        
        if request_type == "verification":
            processing_dict = self.verification_processing
        else:
            processing_dict = self.processing_requests
        
        if request_hash in processing_dict:
            if time.time() - processing_dict[request_hash] < 30:
                return True
        
        processing_dict[request_hash] = time.time()
        return False
    
    async def clear_processing_request(self, user_id, request_data, request_type="file"):
        """Clear from processing requests"""
        request_hash = f"{user_id}_{request_type}_{hash(request_data)}"
        
        if request_type == "verification":
            self.verification_processing.pop(request_hash, None)
        else:
            self.processing_requests.pop(request_hash, None)
