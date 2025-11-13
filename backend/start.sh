#!/bin/bash

echo "🚀 Starting SK4FiLM Complete System..."

# Start bot in background
echo "🤖 Starting Telegram Bot..."
python bot.py &
BOT_PID=$!

# Wait a bit for bot to initialize
sleep 5

# Start web server
echo "🌐 Starting Web Server..."
python main.py &
WEB_PID=$!

# Wait for both processes
wait $BOT_PID
wait $WEB_PID
