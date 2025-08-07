#!/bin/bash
# Production Setup Script for Forex AI Trading System
# This script sets up the production-ready trading environment

echo "🚀 Setting up Production-Grade Forex AI Trading System"
echo "====================================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    echo "Please install Python 3.8+ and try again."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "forex_env" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv forex_env
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source forex_env/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install core dependencies
echo "📚 Installing core dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "📈 Installing trading and analysis libraries..."
pip install pandas numpy yfinance talib-binary scikit-learn

echo "📊 Installing visualization libraries..."
pip install matplotlib seaborn plotly

echo "🛠️ Installing utility libraries..."
pip install tqdm jupyter ipykernel

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p models
mkdir -p logs
mkdir -p validation_reports
mkdir -p data_cache

# Set permissions
chmod +x train.py
chmod +x production_validator.py

echo ""
echo "🎉 SETUP COMPLETE!"
echo "==================="
echo "📋 Next steps:"
echo "   1. Activate environment: source forex_env/bin/activate"
echo "   2. Run validation: python production_validator.py"
echo "   3. Start training: python train.py --epochs 100 --walk-forward"
echo ""
echo "⚠️  IMPORTANT REMINDERS:"
echo "   • Paper trade for 3-6 months before going live"
echo "   • Run validation tests before each deployment"
echo "   • Never risk more than you can afford to lose"
echo "   • This is for educational purposes only"
echo ""
echo "📖 See README_PRODUCTION.md for complete documentation"
