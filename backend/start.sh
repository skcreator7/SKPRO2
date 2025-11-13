#!/bin/bash

echo "🚀 Starting SK4FiLM Backend System..."

# Start bot in background
echo "🤖 Starting Telegram Bot..."
python bot.py &
BOT_PID=$!
echo "Bot PID: $BOT_PID"

# Wait for bot initialization
sleep 5

# Start web server (foreground - keeps container alive)
echo "🌐 Starting API Server on port $PORT..."
exec python main.py
