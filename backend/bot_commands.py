import asyncio
import logging
import re
from datetime import datetime, timedelta

try:
    from pyrogram import Client, filters
    from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton, Message
    from pyrogram.errors import FloodWait, BadRequest
except ImportError:
    pass

from bot_handlers import format_size

logger = logging.getLogger(__name__)


async def setup_bot_handlers(bot, bot_instance):
    """Setup basic bot command handlers"""

    config = bot_instance.config

    @bot.on_message(filters.command("start") & filters.private)
    async def handle_start_command(client, message):
        """Handle /start command"""
        user_name = message.from_user.first_name or "User"

        welcome_text = (
            f"🎬 **Welcome to SK4FiLM, {user_name}!**\n\n"
            f"🌐 **Website:** {config.WEBSITE_URL}\n\n"
            "**How to use:**\n"
            "1. Visit website above\n"
            "2. Search for movies\n"
            "3. Click download button\n"
            "4. File will appear here\n\n"
            "🎬 **Happy watching!**"
        )

        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🌐 OPEN WEBSITE", url=config.WEBSITE_URL)],
            [InlineKeyboardButton("📢 JOIN CHANNEL", url=config.MAIN_CHANNEL_LINK)]
        ])

        await message.reply_text(welcome_text, reply_markup=keyboard, disable_web_page_preview=True)
        logger.info(f"👤 /start command from user {message.from_user.id}")

    @bot.on_message(filters.command("help") & filters.private)
    async def help_command(client, message):
        """Show help message"""
        help_text = (
            "🆘 **SK4FiLM Bot Help**\n\n"
            "**Commands:**\n"
            "• /start - Start the bot\n"
            "• /help - Show this help\n\n"
            "**How to Download:**\n"
            "1. Visit our website\n"
            "2. Search for movies\n"
            "3. Click download button\n"
            "4. File appears here\n\n"
            f"🌐 Website: {config.WEBSITE_URL}\n"
            "📢 Channel: @SK4FiLM"
        )

        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🌐 OPEN WEBSITE", url=config.WEBSITE_URL)]
        ])

        await message.reply_text(help_text, reply_markup=keyboard, disable_web_page_preview=True)

    # Admin stats command
    @bot.on_message(filters.command("stats") & filters.user(config.ADMIN_IDS))
    async def stats_command(client, message):
        """Show bot statistics"""
        try:
            stats_text = (
                f"📊 **SK4FiLM Bot Statistics**\n\n"
                f"🔄 **Status:**\n"
                f"• Bot: {'✅ Online' if bot_instance.bot_started else '❌ Offline'}\n"
                f"• User Client: {'✅ Connected' if bot_instance.user_session_ready else '❌ Disconnected'}\n"
                f"\n⏰ **Server Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )

            await message.reply_text(stats_text, disable_web_page_preview=True)
        except Exception as e:
            logger.error(f"Stats command error: {e}")
            await message.reply_text(f"❌ Error: {str(e)[:100]}")

    logger.info("✅ Bot command handlers registered")
