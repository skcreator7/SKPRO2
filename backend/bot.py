import asyncio
import os
import logging
from datetime import datetime, timedelta
from pyrogram import Client, filters, errors
from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton, Message, CallbackQuery
from motor.motor_asyncio import AsyncIOMotorClient
import hashlib
import time
import aiohttp

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Config:
    # Telegram API
    API_ID = int(os.environ.get("API_ID", "0"))
    API_HASH = os.environ.get("API_HASH", "")
    BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
    
    # Channels
    TEXT_CHANNEL_IDS = [
        int(x) for x in os.environ.get("TEXT_CHANNEL_IDS", "-1001891090100,-1002024811395").split(",")
    ]
    
    FILE_CHANNEL_480P = int(os.environ.get("FILE_CHANNEL_480P", "-1001768249569"))
    FILE_CHANNEL_720P = int(os.environ.get("FILE_CHANNEL_720P", "-1001768249569"))
    FILE_CHANNEL_1080P = int(os.environ.get("FILE_CHANNEL_1080P", "-1001768249569"))
    
    FILE_CHANNELS = {
        "480p": FILE_CHANNEL_480P,
        "720p": FILE_CHANNEL_720P,
        "1080p": FILE_CHANNEL_1080P
    }
    
    # MongoDB
    MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
    DATABASE_NAME = os.environ.get("DATABASE_NAME", "sk4film_bot")
    
    # Force Subscribe
    FORCE_SUB_CHANNEL = int(os.environ.get("FORCE_SUB_CHANNEL", "-1002555323872"))
    
    # Link Shortener
    SHORTENER_API = os.environ.get("SHORTENER_API", "")
    SHORTENER_DOMAIN = os.environ.get("SHORTENER_DOMAIN", "https://gplinks.in/api")
    
    # Website
    WEBSITE_URL = os.environ.get("WEBSITE_URL", "https://sk4film.vercel.app")
    
    # Admin
    ADMIN_IDS = [int(x) for x in os.environ.get("ADMIN_IDS", "5928972764").split(",")]
    
    # Auto Delete Time (seconds)
    AUTO_DELETE_TIME = int(os.environ.get("AUTO_DELETE_TIME", "300"))  # 5 minutes

# Initialize Bot
bot = Client(
    "sk4film_bot",
    api_id=Config.API_ID,
    api_hash=Config.API_HASH,
    bot_token=Config.BOT_TOKEN
)

# MongoDB Setup
mongo_client = AsyncIOMotorClient(Config.MONGODB_URI)
db = mongo_client[Config.DATABASE_NAME]
files_collection = db.files
users_collection = db.users
stats_collection = db.stats

# Cache
indexed_files = {}
pending_deletes = {}

async def init_database():
    """Initialize database indexes"""
    try:
        await files_collection.create_index([("file_id", 1)], unique=True)
        await files_collection.create_index([("unique_id", 1)], unique=True)
        await files_collection.create_index([("title", "text")])
        await files_collection.create_index([("indexed_at", -1)])
        await users_collection.create_index([("user_id", 1)], unique=True)
        logger.info("✅ Database indexes created successfully")
    except Exception as e:
        logger.error(f"Database init error: {e}")

async def check_force_subscribe(user_id: int) -> bool:
    """Check if user has subscribed to channel"""
    try:
        member = await bot.get_chat_member(Config.FORCE_SUB_CHANNEL, user_id)
        return member.status in ["member", "administrator", "creator"]
    except Exception as e:
        logger.warning(f"Force subscribe check error: {e}")
        return False

async def generate_short_link(long_url: str) -> str:
    """Generate short link using shortener API"""
    if not Config.SHORTENER_API:
        return long_url
    
    try:
        async with aiohttp.ClientSession() as session:
            params = {
                "api": Config.SHORTENER_API,
                "url": long_url
            }
            
            async with session.get(Config.SHORTENER_DOMAIN, params=params, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    short_url = data.get("shortenedUrl", long_url)
                    logger.info(f"✅ Short URL generated: {short_url}")
                    return short_url
        
        return long_url
    except Exception as e:
        logger.error(f"Shortener error: {e}")
        return long_url

async def save_user(user_id: int, username: str = None, first_name: str = None):
    """Save/update user in database"""
    try:
        await users_collection.update_one(
            {"user_id": user_id},
            {
                "$set": {
                    "user_id": user_id,
                    "username": username,
                    "first_name": first_name,
                    "last_active": datetime.now()
                },
                "$setOnInsert": {
                    "joined_at": datetime.now(),
                    "files_requested": 0
                }
            },
            upsert=True
        )
    except Exception as e:
        logger.error(f"Save user error: {e}")

async def index_file(message: Message, quality: str = None):
    """Auto-index file from channel"""
    try:
        # Extract file info
        if message.document:
            file_id = message.document.file_id
            file_name = message.document.file_name
            file_size = message.document.file_size
            mime_type = message.document.mime_type
        elif message.video:
            file_id = message.video.file_id
            file_name = message.video.file_name or f"video_{message.video.file_unique_id}.mp4"
            file_size = message.video.file_size
            mime_type = message.video.mime_type
        else:
            return
        
        # Extract title
        if message.caption:
            title = message.caption.split('\n')[0].strip()[:200]
        else:
            title = file_name.rsplit('.', 1)[0]
            title = title.replace('_', ' ').replace('-', ' ')[:200]
        
        # Clean title
        import re
        title = re.sub(r'\d{3,4}p', '', title, flags=re.IGNORECASE)
        title = re.sub(r'(HDRip|WEB-DL|BluRay|x264|x265)', '', title, flags=re.IGNORECASE)
        title = ' '.join(title.split())
        
        # Generate unique ID
        unique_id = hashlib.md5(f"{file_id}{time.time()}".encode()).hexdigest()
        
        # Determine quality from channel or message
        if not quality:
            # Try to detect quality from filename or caption
            text = (message.caption or file_name).lower()
            if '480p' in text:
                quality = '480p'
            elif '720p' in text:
                quality = '720p'
            elif '1080p' in text:
                quality = '1080p'
            else:
                quality = 'Unknown'
        
        # Save to MongoDB
        file_data = {
            "file_id": file_id,
            "unique_id": unique_id,
            "title": title,
            "file_name": file_name,
            "file_size": file_size,
            "mime_type": mime_type,
            "quality": quality,
            "caption": message.caption,
            "message_id": message.id,
            "chat_id": message.chat.id,
            "indexed_at": datetime.now(),
            "downloads": 0
        }
        
        result = await files_collection.update_one(
            {"file_id": file_id},
            {"$set": file_data},
            upsert=True
        )
        
        # Cache in memory
        indexed_files[unique_id] = file_data
        
        logger.info(f"✅ Indexed [{quality}]: {title}")
        return unique_id
        
    except Exception as e:
        logger.error(f"Index file error: {e}")
        return None

async def get_file_by_unique_id(unique_id: str):
    """Get file from database by unique ID"""
    try:
        # Check cache first
        if unique_id in indexed_files:
            return indexed_files[unique_id]
        
        # Get from database
        file_data = await files_collection.find_one({"unique_id": unique_id})
        
        if file_data:
            indexed_files[unique_id] = file_data
            return file_data
        
        return None
    except Exception as e:
        logger.error(f"Get file error: {e}")
        return None

async def auto_delete_message(chat_id: int, message_id: int, delay: int = Config.AUTO_DELETE_TIME):
    """Auto delete message after delay"""
    try:
        await asyncio.sleep(delay)
        await bot.delete_messages(chat_id, message_id)
        logger.info(f"🗑️ Auto-deleted message {message_id} after {delay}s")
    except Exception as e:
        logger.error(f"Auto delete error: {e}")

# START COMMAND
@bot.on_message(filters.command("start") & filters.private)
async def start_command(client: Client, message: Message):
    user_id = message.from_user.id
    username = message.from_user.username
    first_name = message.from_user.first_name
    
    # Save user
    await save_user(user_id, username, first_name)
    
    # Check if it's a file request
    if len(message.command) > 1:
        request_data = message.command[1]
        
        # Parse: unique_id or unique_id_quality
        parts = request_data.split('_')
        unique_id = parts[0]
        selected_quality = parts[1] if len(parts) > 1 else None
        
        # Check force subscribe
        is_subscribed = await check_force_subscribe(user_id)
        
        if not is_subscribed:
            try:
                channel = await bot.get_chat(Config.FORCE_SUB_CHANNEL)
                channel_link = f"https://t.me/{channel.username}" if channel.username else f"https://t.me/c/{str(Config.FORCE_SUB_CHANNEL)[4:]}"
            except:
                channel_link = "https://t.me/sk4film"
            
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🔔 Join Channel", url=channel_link)],
                [InlineKeyboardButton("✅ I Joined", callback_data=f"check_sub:{request_data}")]
            ])
            
            await message.reply_text(
                "**🔒 Access Restricted**\n\n"
                "Please join our Telegram channel to download files.\n\n"
                "**Steps:**\n"
                "1. Click 'Join Channel' button\n"
                "2. Join the channel\n"
                "3. Click 'I Joined' button\n\n"
                "👇 **Join now:**",
                reply_markup=keyboard
            )
            return
        
        # Get file from database
        file_data = await get_file_by_unique_id(unique_id)
        
        if not file_data:
            await message.reply_text(
                "❌ **File Not Found**\n\n"
                "This file may have been:\n"
                "• Not indexed yet\n"
                "• Removed from database\n"
                "• Expired\n\n"
                f"Please search again on: {Config.WEBSITE_URL}"
            )
            return
        
        # Generate verification link if shortener is enabled
        if Config.SHORTENER_API:
            verify_url = f"{Config.WEBSITE_URL}/verify?id={unique_id}&user={user_id}"
            if selected_quality:
                verify_url += f"&quality={selected_quality}"
            
            short_url = await generate_short_link(verify_url)
            
            keyboard = InlineKeyboardMarkup([
                [InlineKeyboardButton("🔓 Verify & Download", url=short_url)]
            ])
            
            await message.reply_text(
                f"**✅ Subscription Verified**\n\n"
                f"**📁 File:** {file_data['title']}\n"
                f"**📏 Size:** {file_data['file_size'] / (1024*1024):.2f} MB\n"
                f"**🎬 Quality:** {selected_quality or file_data.get('quality', 'Unknown')}\n\n"
                f"Click the button below to verify and get your file:\n\n"
                f"⚠️ **Note:** File will auto-delete after {Config.AUTO_DELETE_TIME//60} minutes",
                reply_markup=keyboard
            )
        else:
            # Direct send (no verification)
            await send_file_to_user(message, file_data)
    
    else:
        # Normal start message - Send website link
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🌐 Visit Website", url=Config.WEBSITE_URL)],
            [InlineKeyboardButton("📢 Join Channel", url=f"https://t.me/{Config.FORCE_SUB_CHANNEL}")]
        ])
        
        await message.reply_text(
            f"**👋 Welcome to SK4FiLM Bot!**\n\n"
            f"🎬 Search and download movies with multiple quality options\n\n"
            f"**✨ Features:**\n"
            f"✅ Multiple qualities (480p, 720p, 1080p)\n"
            f"✅ Auto file indexing\n"
            f"✅ Fast search\n"
            f"✅ Direct download\n"
            f"✅ Auto delete ({Config.AUTO_DELETE_TIME//60} min)\n"
            f"✅ Force subscribe\n"
            f"✅ Link verification\n\n"
            f"**🌐 Visit our website:**\n"
            f"{Config.WEBSITE_URL}\n\n"
            f"**📝 How to use:**\n"
            f"1. Search movies on website\n"
            f"2. Click Download button\n"
            f"3. Select quality (480p/720p/1080p)\n"
            f"4. Get file via bot\n\n"
            f"💡 **Tip:** Send any message to get website link",
            reply_markup=keyboard
        )

async def send_file_to_user(message: Message, file_data: dict):
    """Send file to user with auto-delete"""
    try:
        # Send file
        sent_msg = await bot.send_document(
            chat_id=message.chat.id,
            document=file_data['file_id'],
            caption=(
                f"**📁 {file_data['title']}**\n\n"
                f"**📏 Size:** {file_data['file_size'] / (1024*1024):.2f} MB\n"
                f"**🎬 Quality:** {file_data.get('quality', 'Unknown')}\n\n"
                f"⚠️ **Auto-delete in {Config.AUTO_DELETE_TIME//60} minutes**\n\n"
                f"**Powered by:** SK4FiLM"
            ),
            reply_to_message_id=message.id
        )
        
        # Update download count
        await files_collection.update_one(
            {"unique_id": file_data['unique_id']},
            {"$inc": {"downloads": 1}}
        )
        
        # Update user stats
        await users_collection.update_one(
            {"user_id": message.from_user.id},
            {"$inc": {"files_requested": 1}}
        )
        
        # Schedule auto-delete
        asyncio.create_task(auto_delete_message(message.chat.id, sent_msg.id, Config.AUTO_DELETE_TIME))
        
        logger.info(f"✅ File sent to {message.from_user.id}: {file_data['title']}")
        
    except Exception as e:
        logger.error(f"Send file error: {e}")
        await message.reply_text(
            "❌ **Error sending file**\n\n"
            "File may have been deleted from channel.\n"
            "Please contact admin."
        )

# CHECK SUBSCRIPTION CALLBACK
@bot.on_callback_query(filters.regex(r"^check_sub:"))
async def check_subscription(client: Client, callback: CallbackQuery):
    user_id = callback.from_user.id
    request_data = callback.data.split(":", 1)[1]
    
    is_subscribed = await check_force_subscribe(user_id)
    
    if not is_subscribed:
        await callback.answer("❌ You haven't joined the channel yet!", show_alert=True)
        return
    
    # Parse request
    parts = request_data.split('_')
    unique_id = parts[0]
    selected_quality = parts[1] if len(parts) > 1 else None
    
    # Get file
    file_data = await get_file_by_unique_id(unique_id)
    
    if not file_data:
        await callback.message.edit_text("❌ File not found")
        return
    
    # Generate verification or send directly
    if Config.SHORTENER_API:
        verify_url = f"{Config.WEBSITE_URL}/verify?id={unique_id}&user={user_id}"
        if selected_quality:
            verify_url += f"&quality={selected_quality}"
        
        short_url = await generate_short_link(verify_url)
        
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔓 Verify & Download", url=short_url)]
        ])
        
        await callback.message.edit_text(
            f"**✅ Subscription Verified**\n\n"
            f"**📁 File:** {file_data['title']}\n"
            f"**📏 Size:** {file_data['file_size'] / (1024*1024):.2f} MB\n"
            f"**🎬 Quality:** {selected_quality or file_data.get('quality', 'Unknown')}\n\n"
            f"Click button to verify and download:\n\n"
            f"⚠️ Auto-delete after {Config.AUTO_DELETE_TIME//60} minutes",
            reply_markup=keyboard
        )
    else:
        await send_file_to_user(callback.message, file_data)
    
    await callback.answer("✅ Subscription verified!", show_alert=False)

# HANDLE ALL MESSAGES - Send website link
@bot.on_message(filters.private & ~filters.command(["start", "help", "stats", "index"]))
async def handle_message(client: Client, message: Message):
    user_id = message.from_user.id
    username = message.from_user.username
    first_name = message.from_user.first_name
    
    await save_user(user_id, username, first_name)
    
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("🌐 Visit SK4FiLM Website", url=Config.WEBSITE_URL)],
        [InlineKeyboardButton("🔍 Search Movies", url=Config.WEBSITE_URL)]
    ])
    
    await message.reply_text(
        "**🎬 SK4FiLM - Movie Search & Download**\n\n"
        f"Visit our website to search movies:\n"
        f"👉 {Config.WEBSITE_URL}\n\n"
        "**✨ Features:**\n"
        "✅ Multiple qualities available\n"
        "✅ Latest movies & web series\n"
        "✅ Fast search\n"
        "✅ Direct Telegram download\n"
        "✅ Auto-delete for privacy\n\n"
        "**📝 How it works:**\n"
        "1. Search on website\n"
        "2. Select quality\n"
        "3. Download via bot\n\n"
        "Click button below to start:",
        reply_markup=keyboard
    )

# ADMIN: Manual index command
@bot.on_message(filters.command("index") & filters.user(Config.ADMIN_IDS))
async def manual_index(client: Client, message: Message):
    if not message.reply_to_message:
        await message.reply_text("❌ Reply to a media file to index it")
        return
    
    replied_msg = message.reply_to_message
    
    if not (replied_msg.document or replied_msg.video):
        await message.reply_text("❌ Reply to a document or video file")
        return
    
    # Get quality from command
    quality = None
    if len(message.command) > 1:
        quality = message.command[1].lower()
        if quality not in ['480p', '720p', '1080p']:
            await message.reply_text("❌ Invalid quality. Use: 480p, 720p, or 1080p")
            return
    
    status_msg = await message.reply_text("⏳ Indexing file...")
    
    unique_id = await index_file(replied_msg, quality)
    
    if unique_id:
        await status_msg.edit_text(
            f"✅ **File indexed successfully!**\n\n"
            f"**Unique ID:** `{unique_id}`\n"
            f"**Quality:** {quality or 'Auto-detected'}"
        )
    else:
        await status_msg.edit_text("❌ Failed to index file")

# AUTO-INDEX FILES FROM CHANNELS
@bot.on_message(filters.channel & (filters.document | filters.video))
async def auto_index_handler(client: Client, message: Message):
    channel_id = message.chat.id
    
    # Determine quality based on channel
    quality = None
    for qual, ch_id in Config.FILE_CHANNELS.items():
        if channel_id == ch_id:
            quality = qual
            break
    
    logger.info(f"📥 New file in channel [{quality or 'Unknown'}]: {message.chat.title}")
    
    unique_id = await index_file(message, quality)
    
    if unique_id:
        logger.info(f"✅ Auto-indexed: {unique_id}")

# STATS COMMAND
@bot.on_message(filters.command("stats") & filters.user(Config.ADMIN_IDS))
async def stats_command(client: Client, message: Message):
    try:
        total_files = await files_collection.count_documents({})
        total_users = await users_collection.count_documents({})
        
        recent_files = await files_collection.count_documents({
            "indexed_at": {"$gte": datetime.now() - timedelta(days=1)}
        })
        
        total_downloads = await files_collection.aggregate([
            {"$group": {"_id": None, "total": {"$sum": "$downloads"}}}
        ]).to_list(1)
        
        downloads = total_downloads[0]['total'] if total_downloads else 0
        
        # Quality breakdown
        quality_stats = {}
        for quality in ['480p', '720p', '1080p', 'Unknown']:
            count = await files_collection.count_documents({"quality": quality})
            quality_stats[quality] = count
        
        await message.reply_text(
            f"**📊 SK4FiLM Bot Statistics**\n\n"
            f"**📁 Files:**\n"
            f"• Total: {total_files}\n"
            f"• Today: {recent_files}\n"
            f"• 480p: {quality_stats.get('480p', 0)}\n"
            f"• 720p: {quality_stats.get('720p', 0)}\n"
            f"• 1080p: {quality_stats.get('1080p', 0)}\n\n"
            f"**👥 Users:**\n"
            f"• Total: {total_users}\n\n"
            f"**📥 Downloads:**\n"
            f"• Total: {downloads}\n\n"
            f"**⚙️ Settings:**\n"
            f"• Auto Delete: {Config.AUTO_DELETE_TIME}s\n"
            f"• Force Subscribe: {'✅' if Config.FORCE_SUB_CHANNEL else '❌'}\n"
            f"• Link Shortener: {'✅' if Config.SHORTENER_API else '❌'}"
        )
    except Exception as e:
        await message.reply_text(f"❌ Error: {e}")

# RUN BOT
async def main():
    try:
        logger.info("🚀 Starting SK4FiLM Bot...")
        
        await init_database()
        
        # Load indexed files into memory (limit to recent 1000)
        files = await files_collection.find().sort("indexed_at", -1).limit(1000).to_list(1000)
        for file_data in files:
            indexed_files[file_data['unique_id']] = file_data
        
        logger.info(f"✅ Loaded {len(indexed_files)} files into memory")
        
        await bot.start()
        me = await bot.get_me()
        logger.info(f"✅ Bot started: @{me.username}")
        logger.info(f"📁 Monitoring {len(Config.FILE_CHANNELS)} file channels")
        logger.info(f"📝 Monitoring {len(Config.TEXT_CHANNEL_IDS)} text channels")
        
        await asyncio.Event().wait()
        
    except Exception as e:
        logger.error(f"Bot error: {e}")
    finally:
        await bot.stop()
        logger.info("Bot stopped")

if __name__ == "__main__":
    asyncio.run(main())
