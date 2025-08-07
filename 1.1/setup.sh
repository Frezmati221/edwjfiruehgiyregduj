#!/bin/bash

# ETERNITY AI FOREX TRADING SYSTEM SETUP
# Enhanced 75-epoch models with live trading capability

echo "🌟 ETERNITY AI FOREX SETUP"
echo "=========================="

# Check if we're in the eternity directory
if [ ! -f "enhanced_loss_learning_trainer.py" ]; then
    echo "❌ Please run this script from the eternity directory"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "eternity_env" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv eternity_env
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source eternity_env/bin/activate

# Install requirements
echo "📥 Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "🎉 ETERNITY SETUP COMPLETE!"
echo ""
echo "📋 NEXT STEPS:"
echo "1. Activate environment:    source eternity_env/bin/activate"
echo "2. Train enhanced models:   python enhanced_loss_learning_trainer.py"
echo "3. Test models:            python quick_enhanced_validator.py"
echo "4. Start live trading:     python live_trading_launcher.py"
echo ""
echo "📖 Read QUICK_START_LIVE_TRADING.md for detailed instructions"
echo ""
