#!/bin/bash

# Trade-2 Telegram Bot Launcher
echo "🎯 Trade-2 Telegram Bot Launcher"
echo "=================================="

# Check if virtual environment exists
if [ ! -d "telegram_bot_env" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv telegram_bot_env
fi

# Activate virtual environment
echo "⚡ Activating virtual environment..."
source telegram_bot_env/bin/activate

# Check if dependencies are installed
if [ ! -f "telegram_bot_env/.deps_installed" ]; then
    echo "📦 Installing dependencies..."
    pip install --upgrade pip
    pip install -r telegram_bot_requirements.txt
    touch telegram_bot_env/.deps_installed
    echo "✅ Dependencies installed"
fi

# Check environment variables
if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
    echo ""
    echo "⚠️  TELEGRAM_BOT_TOKEN not set!"
    echo "Please set your bot token:"
    echo "export TELEGRAM_BOT_TOKEN='your_token_here'"
    echo ""
    read -p "Enter bot token now (or press Enter to skip): " token
    if [ ! -z "$token" ]; then
        export TELEGRAM_BOT_TOKEN="$token"
    else
        echo "❌ Cannot start bot without token"
        exit 1
    fi
fi

# Check required files
echo "🔍 Checking required files..."

if [ ! -f "best_model.pth" ]; then
    echo "❌ best_model.pth not found!"
    exit 1
fi

if [ ! -f "compatible_trade2_predictor.py" ]; then
    echo "❌ compatible_trade2_predictor.py not found!"
    exit 1
fi

echo "✅ All files found"

# Create logs directory
mkdir -p logs

# Choose bot version
echo ""
echo "🤖 Choose bot version:"
echo "1. Basic Bot (trade2_telegram_bot.py)"
echo "2. Enhanced Bot with Alerts (enhanced_trade2_telegram_bot.py)"
echo ""
read -p "Enter choice (1 or 2, default: 1): " choice

case $choice in
    2)
        BOT_FILE="enhanced_trade2_telegram_bot.py"
        echo "🚀 Starting Enhanced Trade-2 Telegram Bot..."
        ;;
    *)
        BOT_FILE="trade2_telegram_bot.py"
        echo "🚀 Starting Basic Trade-2 Telegram Bot..."
        ;;
esac

# Start the bot
if [ -f "$BOT_FILE" ]; then
    echo "Bot starting... Press Ctrl+C to stop"
    echo ""
    python3 "$BOT_FILE"
else
    echo "❌ Bot file $BOT_FILE not found!"
    exit 1
fi
