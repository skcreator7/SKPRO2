import asyncio
import logging
from datetime import datetime
from pyrogram import filters
from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from pyrogram.errors import FloodWait

logger = logging.getLogger(__name__)

# Import utility functions from main
from main import format_size, extract_title_smart, normalize_title, extract_title_from_file, safe_telegram_operation, safe_telegram_generator, index_files_background, is_video_file, detect_quality

async def setup_bot_handlers(bot, User, verification_system, files_col, redis_cache, poster_fetcher, movie_db, flood_protection, Config):
    """Setup all bot handlers in a separate file"""
    
    @bot.on_message(filters.command("start") & filters.private)
    async def start_handler(client, message):
        uid = message.from_user.id
        user_name = message.from_user.first_name or "User"
        
        if len(message.command) > 1:
            fid = message.command[1]
            
            if Config.VERIFICATION_REQUIRED:
                is_verified, status = await verification_system.check_verification(uid)
                
                if not is_verified:
                    verification_url = await verification_system.generate_verification_url(uid)
                    
                    keyboard = InlineKeyboardMarkup([
                        [InlineKeyboardButton("🔗 VERIFY NOW", url=verification_url)],
                        [InlineKeyboardButton("🔄 CHECK VERIFICATION", callback_data=f"check_verify_{uid}")],
                        [InlineKeyboardButton("📢 JOIN CHANNEL", url=Config.MAIN_CHANNEL_LINK)]
                    ])
                    
                    await message.reply_text(
                        f"👋 **Hello {user_name}!**\n\n"
                        "🔒 **Verification Required**\n"
                        "To download files, you need to complete URL verification.\n\n"
                        "🚀 **Quick Steps:**\n"
                        "1. Click **VERIFY NOW** below\n"
                        "2. Complete the verification process\n"
                        "3. Come back and click **CHECK VERIFICATION**\n"
                        "4. Start downloading!\n\n"
                        "⏰ **Verification valid for 6 hours**",
                        reply_markup=keyboard,
                        disable_web_page_preview=True
                    )
                    return
            
            try:
                parts = fid.split('_')
                if len(parts) >= 2:
                    channel_id = int(parts[0])
                    message_id = int(parts[1])
                    quality = parts[2] if len(parts) > 2 else "HD"
                    
                    pm = await message.reply_text(f"⏳ **Preparing your file...**\n\n📦 Quality: {quality}")
                    
                    file_message = await safe_telegram_operation(
                        bot.get_messages,
                        channel_id, 
                        message_id
                    )
                    
                    if not file_message or (not file_message.document and not file_message.video):
                        await pm.edit_text("❌ **File not found**\n\nThe file may have been deleted.")
                        return
                    
                    if file_message.document:
                        sent = await safe_telegram_operation(
                            bot.send_document,
                            uid, 
                            file_message.document.file_id, 
                            caption=f"♻ **Please forward this file/video to your saved messages**\n\n"
                                   f"📹 Quality: {quality}\n"
                                   f"📦 Size: {format_size(file_message.document.file_size)}\n\n"
                                   f"⚠️ Will auto-delete in {Config.AUTO_DELETE_TIME//60} minutes\n\n"
                                   f"@SK4FiLM 🍿"
                        )
                    else:
                        sent = await safe_telegram_operation(
                            bot.send_video,
                            uid, 
                            file_message.video.file_id, 
                            caption=f"♻ **Please forward this file/video to your saved messages**\n\n"
                                   f"📹 Quality: {quality}\n" 
                                   f"📦 Size: {format_size(file_message.video.file_size)}\n\n"
                                   f"⚠️ Will auto-delete in {Config.AUTO_DELETE_TIME//60} minutes\n\n"
                                   f"@SK4FiLM 🍿"
                        )
                    
                    await pm.delete()
                    
                    if Config.AUTO_DELETE_TIME > 0:
                        async def auto_delete():
                            await asyncio.sleep(Config.AUTO_DELETE_TIME)
                            try:
                                await sent.delete()
                            except:
                                pass
                        asyncio.create_task(auto_delete())
                        
                else:
                    await message.reply_text("❌ **Invalid file link**\n\nPlease get a fresh link from the website.")
                    
            except Exception as e:
                try:
                    await message.reply_text(f"❌ **Download Failed**\n\nError: `{str(e)}`")
                except:
                    pass
            return
        
        welcome_text = (
            f"🎬 **Welcome to SK4FiLM, {user_name}!**\n\n"
            "🌐 **Use our website to browse and download movies:**\n"
            f"{Config.WEBSITE_URL}\n\n"
        )
        
        if Config.VERIFICATION_REQUIRED:
            welcome_text += "🔒 **URL Verification Required**\n• Complete one-time verification\n• Valid for 6 hours\n\n"
        
        welcome_text += (
            "✨ **Enhanced Features:**\n"
            "• 🎥 Latest movies from MULTIPLE channels\n" 
            "• 📺 Multiple quality options\n"
            "• ⚡ Fast multi-channel search\n"
            "• 🖼️ Video thumbnails\n"
            "• 🔍 Redis-cached search\n"
            "• 🔄 Concurrent channel processing\n\n"
            "👇 **Get started below:**"
        )
        
        buttons = []
        if Config.VERIFICATION_REQUIRED:
            verification_url = await verification_system.generate_verification_url(uid)
            buttons.append([InlineKeyboardButton("🔗 GET VERIFIED", url=verification_url)])
        
        buttons.extend([
            [InlineKeyboardButton("🌐 VISIT WEBSITE", url=Config.WEBSITE_URL)],
            [
                InlineKeyboardButton("📢 Mᴀɪɴ Cʜᴀɴɴᴇʟ", url=Config.MAIN_CHANNEL_LINK),
                InlineKeyboardButton("🔎 Mᴏᴠɪᴇꜱ Gʀᴏᴜᴘ", url=Config.UPDATES_CHANNEL_LINK)
            ]
        ])
        
        keyboard = InlineKeyboardMarkup(buttons)
        await message.reply_text(welcome_text, reply_markup=keyboard, disable_web_page_preview=True)
    
    # Setup verification system bot handlers
    verification_system.setup_handlers(bot)
    
    @bot.on_message(filters.text & filters.private & ~filters.command(['start', 'stats', 'index', 'verify', 'clear_cache']))
    async def text_handler(client, message):
        user_name = message.from_user.first_name or "User"
        await message.reply_text(
            f"👋 **Hi {user_name}!**\n\n"
            "🔍 **Please Use Our Website To Search For Movies:**\n\n"
            f"{Config.WEBSITE_URL}\n\n"
            "This bot only handles file downloads via website links.",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🌐 OPEN WEBSITE", url=Config.WEBSITE_URL)],
                [
                    InlineKeyboardButton("📢 Mᴀɪɴ Cʜᴀɴɴᴇʟ", url=Config.MAIN_CHANNEL_LINK),
                    InlineKeyboardButton("🔎 Mᴏᴠɪᴇꜱ Gʀᴏᴜᴘ", url=Config.UPDATES_CHANNEL_LINK)
                ]
            ]),
            disable_web_page_preview=True
        )
    
    @bot.on_message(filters.command("channel") & filters.private)
    async def channel_command(client, message):
        await message.reply_text(
            "📢 **SK4FiLM Channels**\n\n"
            "Join our channels for the latest movies and updates:\n\n"
            "🎬 **Main Channel:**\n"
            "• Latest movie releases\n"
            "• High quality files\n"
            "• Daily updates\n\n"
            "🔎 **Movies Group:**\n"
            "• Movie discussions\n"
            "• Requests & updates\n"
            "• Community interaction\n\n"
            "👇 **Click below to join:**",
            reply_markup=InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("🎬 MAIN CHANNEL", url=Config.MAIN_CHANNEL_LINK),
                    InlineKeyboardButton("🔎 Mᴏᴠɪᴇꜱ Gʀᴏᴜᴘ", url=Config.UPDATES_CHANNEL_LINK)
                ],
                [InlineKeyboardButton("🌐 WEBSITE", url=Config.WEBSITE_URL)]
            ]),
            disable_web_page_preview=True
        )
    
    @bot.on_message(filters.command("index") & filters.user(Config.ADMIN_IDS))
    async def index_handler(client, message):
        msg = await message.reply_text("🔄 **Starting ENHANCED background indexing (NEW FILES ONLY)...**")
        asyncio.create_task(index_files_background())
        await msg.edit_text("✅ **Enhanced indexing started in background!**\n\nOnly new files will be indexed with batch thumbnail processing. Check /stats for progress.")
    
    @bot.on_message(filters.command("clear_cache") & filters.user(Config.ADMIN_IDS))
    async def clear_cache_handler(client, message):
        msg = await message.reply_text("🧹 **Clearing all caches...**")
        
        # Clear Redis cache
        redis_cleared = await redis_cache.clear_search_cache()
        
        # Clear memory cache
        movie_db['search_cache'].clear()
        movie_db['poster_cache'].clear()
        movie_db['title_cache'].clear()
        
        # Clear poster fetcher cache
        poster_fetcher.clear_cache()
        
        movie_db['stats']['redis_hits'] = 0
        movie_db['stats']['redis_misses'] = 0
        movie_db['stats']['multi_channel_searches'] = 0
        
        await msg.edit_text(
            f"✅ **All caches cleared!**\n\n"
            f"• Redis cache: {'✅ Cleared' if redis_cleared else '❌ Failed'}\n"
            f"• Memory cache: ✅ Cleared\n"
            f"• Search cache: ✅ Cleared\n"
            f"• Poster cache: ✅ Cleared\n"
            f"• Poster fetcher cache: ✅ Cleared\n"
            f"• Multi-channel stats: ✅ Reset\n\n"
            f"Next search will be fresh from database."
        )
    
    @bot.on_message(filters.command("stats") & filters.user(Config.ADMIN_IDS))
    async def stats_handler(client, message):
        tf = await files_col.count_documents({}) if files_col is not None else 0
        video_files = await files_col.count_documents({'is_video_file': True}) if files_col is not None else 0
        video_thumbnails = await files_col.count_documents({'is_video_file': True, 'thumbnail': {'$ne': None}}) if files_col is not None else 0
        total_thumbnails = await files_col.count_documents({'thumbnail': {'$ne': None}}) if files_col is not None else 0
        
        thumbnail_coverage = f"{(video_thumbnails/video_files*100):.1f}%" if video_files > 0 else "0%"
        
        # Get last indexed file info
        last_indexed = await files_col.find_one({}, sort=[('message_id', -1)])
        last_msg_id = last_indexed['message_id'] if last_indexed else 'None'
        
        stats_text = (
            f"📊 **SK4FiLM MULTI-CHANNEL STATISTICS**\n\n"
            f"📁 **Total Files:** {tf}\n"
            f"🎥 **Video Files:** {video_files}\n"
            f"🖼️ **Video Thumbnails:** {video_thumbnails}\n"
            f"📸 **Total Thumbnails:** {total_thumbnails}\n"
            f"📈 **Coverage:** {thumbnail_coverage}\n"
            f"📨 **Last Message ID:** {last_msg_id}\n\n"
            f"🔴 **Live Posts:** Active\n"
            f"🤖 **Bot Status:** Online\n"
            f"👤 **User Session:** {'Ready' if User else 'Flood Wait'}\n"
            f"🔧 **Indexing Mode:** MULTI-CHANNEL ENHANCED\n"
            f"🔍 **Redis Cache:** {'✅ Enabled' if redis_cache.enabled else '❌ Disabled'}\n"
            f"📡 **Channels Active:** {len(Config.TEXT_CHANNEL_IDS)} text + 1 file\n\n"
            f"**🎨 Poster Sources:**\n"
            f"• Letterboxd: {movie_db['stats']['letterboxd']}\n"
            f"• IMDb: {movie_db['stats']['imdb']}\n"
            f"• JustWatch: {movie_db['stats']['justwatch']}\n"
            f"• IMPAwards: {movie_db['stats']['impawards']}\n"
            f"• OMDB: {movie_db['stats']['omdb']}\n"
            f"• TMDB: {movie_db['stats']['tmdb']}\n" 
            f"• Custom: {movie_db['stats']['custom']}\n"
            f"• Cache Hits: {movie_db['stats']['cache_hits']}\n"
            f"• Video Thumbnails: {movie_db['stats']['video_thumbnails']}\n\n"
            f"**🔍 Search Statistics:**\n"
            f"• Redis Hits: {movie_db['stats']['redis_hits']}\n"
            f"• Redis Misses: {movie_db['stats']['redis_misses']}\n"
            f"• Multi-channel Searches: {movie_db['stats']['multi_channel_searches']}\n"
            f"• Hit Rate: {(movie_db['stats']['redis_hits']/(movie_db['stats']['redis_hits'] + movie_db['stats']['redis_misses'])*100):.1f}%\n\n"
            f"**⚡ Enhanced Features:**\n"
            f"• ✅ Multi-channel search & posts\n"
            f"• ✅ Concurrent channel processing\n"
            f"• ✅ Enhanced file indexing (NEW ONLY)\n"
            f"• ✅ Batch thumbnail processing\n"
            f"• ✅ Redis search caching\n"
            f"• ✅ Enhanced flood protection\n\n"
            f"**🔗 Verification:** {'ENABLED (6 hours)' if Config.VERIFICATION_REQUIRED else 'DISABLED'}"
        )
        await message.reply_text(stats_text)

    # Additional handlers can be added here
    
    logger.info("✅ Bot handlers setup completed!")
