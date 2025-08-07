#!/bin/bash

# Trade-2 Telegram Bot Setup Script
echo "🎯 Setting up Trade-2 Telegram Bot..."

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Warning: No virtual environment detected"
    echo "It's recommended to use a virtual environment"
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Setting up virtual environment..."
        python3 -m venv telegram_bot_env
        source telegram_bot_env/bin/activate
        echo "✅ Virtual environment activated"
    fi
fi

# Install required packages
echo "📦 Installing required packages..."
pip install --upgrade pip

# Install telegram bot dependencies
pip install python-telegram-bot[all]
pip install yfinance
pip install pandas
pip install numpy
pip install torch
pip install scikit-learn
pip install asyncio
pip install aiohttp

echo "✅ Dependencies installed"

# Check if model files exist
echo "🔍 Checking model files..."
if [ ! -f "best_model.pth" ]; then
    echo "❌ best_model.pth not found!"
    echo "Please ensure your trained model is in the current directory"
    exit 1
fi

if [ ! -f "best_model_scaler.pkl" ]; then
    echo "⚠️  best_model_scaler.pkl not found"
    echo "The bot will work but might have reduced accuracy"
fi

if [ ! -f "compatible_trade2_predictor.py" ]; then
    echo "❌ compatible_trade2_predictor.py not found!"
    echo "Please ensure the predictor module is available"
    exit 1
fi

echo "✅ Model files check complete"

# Set up configuration
echo "⚙️ Setting up configuration..."

# Check if bot token is set
if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
    echo ""
    echo "🤖 TELEGRAM BOT TOKEN SETUP"
    echo "================================================"
    echo "1. Go to @BotFather on Telegram"
    echo "2. Send /newbot"
    echo "3. Choose a name: Trade-2 Forex Bot"
    echo "4. Choose a username: your_trade2_bot"
    echo "5. Copy the token provided"
    echo ""
    read -p "Enter your bot token: " BOT_TOKEN
    
    if [ ! -z "$BOT_TOKEN" ]; then
        echo "export TELEGRAM_BOT_TOKEN='$BOT_TOKEN'" >> ~/.bashrc
        export TELEGRAM_BOT_TOKEN="$BOT_TOKEN"
        echo "✅ Bot token saved to ~/.bashrc"
    else
        echo "❌ No token provided. You can set it later with:"
        echo "export TELEGRAM_BOT_TOKEN='your_token_here'"
    fi
else
    echo "✅ Bot token already configured"
fi

# Create log directory
mkdir -p logs
echo "✅ Log directory created"

# Set permissions
chmod +x trade2_telegram_bot.py
echo "✅ Permissions set"

# Test the setup
echo ""
echo "🧪 Testing bot setup..."
python3 -c "
import sys
try:
    from compatible_trade2_predictor import CompatibleSupervisedForexPredictor
    print('✅ Predictor module imported successfully')
except ImportError as e:
    print(f'❌ Predictor import failed: {e}')
    sys.exit(1)

try:
    import telegram
    print('✅ Telegram library imported successfully')
except ImportError as e:
    print(f'❌ Telegram import failed: {e}')
    sys.exit(1)

try:
    import yfinance as yf
    print('✅ Yahoo Finance library imported successfully')
except ImportError as e:
    print(f'❌ YFinance import failed: {e}')
    sys.exit(1)

try:
    import torch
    print('✅ PyTorch imported successfully')
except ImportError as e:
    print(f'❌ PyTorch import failed: {e}')
    sys.exit(1)

print('🎉 All dependencies are working!')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 SETUP COMPLETE!"
    echo "================================================"
    echo ""
    echo "📋 Next Steps:"
    echo "1. Make sure TELEGRAM_BOT_TOKEN is set:"
    echo "   export TELEGRAM_BOT_TOKEN='your_token'"
    echo ""
    echo "2. Start the bot:"
    echo "   python3 trade2_telegram_bot.py"
    echo ""
    echo "3. Find your bot on Telegram and send /start"
    echo ""
    echo "📊 Bot Features:"
    echo "• Real-time forex predictions"
    echo "• Multiple confidence levels"
    echo "• Interactive buttons"
    echo "• Detailed technical analysis"
    echo "• 8+ major forex pairs"
    echo "• User preferences"
    echo ""
    echo "🛡️  Security Notes:"
    echo "• Keep your bot token private"
    echo "• The bot provides analysis only"
    echo "• Never share personal trading info"
    echo ""
    
    if [ ! -z "$TELEGRAM_BOT_TOKEN" ]; then
        read -p "🚀 Start the bot now? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Starting Trade-2 Telegram Bot..."
            python3 trade2_telegram_bot.py
        fi
    fi
else
    echo "❌ Setup failed. Please check the errors above."
    exit 1
fi
