#!/bin/bash

# 🚀 Quick Start Auto-Scanning Telegram Bot
echo "🚀 STARTING AUTO-SCANNING FOREX ALERT BOT"
echo "=========================================="

# Check if setup is complete
if [ ! -f "bot_config.json" ]; then
    echo "❌ Bot not configured yet!"
    echo "Please run: ./setup_autoscan_bot.sh first"
    exit 1
fi

# Check model files
MODEL_STATUS=""
if [ -f "best_model.pth" ] && [ -f "best_model_scaler.pkl" ]; then
    MODEL_STATUS="✅ Real AI model loaded"
else
    MODEL_STATUS="⚠️  Using dummy predictions (train model first)"
fi

echo "📊 Status Check:"
echo "   Config: ✅ bot_config.json found"
echo "   Model: $MODEL_STATUS"
echo "   Logs: logs/ directory ready"
echo ""

# Get bot info from config
BOT_TOKEN=$(python3 -c "import json; print(json.load(open('bot_config.json'))['telegram_token'])" 2>/dev/null)
if [ -z "$BOT_TOKEN" ]; then
    echo "❌ Invalid bot configuration"
    exit 1
fi

echo "🎯 Bot Configuration:"
echo "   • Monitors: EURUSD, GBPUSD, USDJPY"
echo "   • Scan Interval: Every 15 minutes"
echo "   • Alert Threshold: 70% confidence (customizable)"
echo "   • Smart Cooldown: Prevents spam"
echo ""

read -p "Start the auto-scanning bot? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "🚀 Starting Auto-Scanning Forex Alert Bot..."
    echo "📱 Find your bot on Telegram and send /start"
    echo "⚙️  Configure settings with /settings"
    echo "🔄 Bot will auto-scan every 15 minutes"
    echo ""
    echo "To stop: Press Ctrl+C"
    echo "To run in background: Use 'nohup python3 autoscan_telegram_bot.py &'"
    echo ""
    echo "============================================"
    
    # Start the bot
    python3 autoscan_telegram_bot.py
else
    echo "Bot start cancelled."
    echo ""
    echo "💡 To start manually: python3 autoscan_telegram_bot.py"
    echo "💡 To run in background: nohup python3 autoscan_telegram_bot.py &"
fi
