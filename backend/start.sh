#!/bin/bash
set -e

echo "🚀 Starting SK4FiLM Backend..."

# Start bot
echo "🤖 Starting Bot..."
python -u bot.py 2>&1 | sed 's/^/[BOT] /' &
BOT_PID=$!

# Wait
sleep 5

# Start web
echo "🌐 Starting Web Server on port ${PORT:-8000}..."
exec python -u main.py 2>&1 | sed 's/^/[WEB] /'
