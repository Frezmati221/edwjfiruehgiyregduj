#!/bin/bash

# 🚀 Auto-Scanning Telegram Bot Setup Script
echo "🚀 Setting up Auto-Scanning Forex Alert Bot"
echo "==========================================="

# Check if bot_config.json exists
if [ ! -f "bot_config.json" ]; then
    echo "📝 Creating bot configuration file..."
    
    echo "Please enter your Telegram Bot Token:"
    echo "(Get it from @BotFather on Telegram)"
    read -p "Bot Token: " BOT_TOKEN
    
    cat > bot_config.json << EOF
{
    "telegram_token": "$BOT_TOKEN"
}
EOF
    
    echo "✅ Configuration saved to bot_config.json"
else
    echo "✅ Configuration file already exists"
fi

# Check if model files exist
if [ ! -f "best_model.pth" ]; then
    echo "⚠️  WARNING: best_model.pth not found"
    echo "   The bot will use dummy predictions for testing"
    echo "   Make sure to train your model first with trade-2.py"
fi

if [ ! -f "best_model_scaler.pkl" ]; then
    echo "⚠️  WARNING: best_model_scaler.pkl not found"
    echo "   Make sure to train your model first with trade-2.py"
fi

# Create logs directory
mkdir -p logs

# Check Python dependencies
echo "🔧 Checking Python dependencies..."

# Check if required packages are installed
python3 -c "import telegram, yfinance, pandas, sqlite3, apscheduler" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installing required packages..."
    pip3 install python-telegram-bot yfinance pandas apscheduler
else
    echo "✅ All dependencies are installed"
fi

# Make bot executable
chmod +x autoscan_telegram_bot.py

echo ""
echo "✅ Setup Complete!"
echo ""
echo "🚀 To start the bot:"
echo "   python3 autoscan_telegram_bot.py"
echo ""
echo "🔧 To run in background:"
echo "   nohup python3 autoscan_telegram_bot.py > logs/bot.log 2>&1 &"
echo ""
echo "📋 Bot Features:"
echo "   • Auto-scans EURUSD, GBPUSD, USDJPY every 15 minutes"
echo "   • Sends alerts for high-confidence LONG/SHORT signals"
echo "   • Customizable confidence thresholds and quiet hours"
echo "   • No spam - smart cooldown between notifications"
echo ""
echo "💡 Next Steps:"
echo "   1. Start the bot: python3 autoscan_telegram_bot.py"
echo "   2. Find your bot on Telegram and send /start"
echo "   3. Configure your settings with /settings"
echo "   4. Wait for auto-alerts or use /scan for manual check"
