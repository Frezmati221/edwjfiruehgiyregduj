#!/bin/bash
# Vast.ai Setup Script for Forex AI Training
# Run this script when your Vast.ai instance starts

echo "🚀 Setting up Forex AI Training Environment on Vast.ai..."

# Update system
apt update && apt upgrade -y

# Install Python dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install yfinance pandas numpy talib-binary scikit-learn tqdm matplotlib seaborn

# Clone your repository (replace with your GitHub repo)
git clone https://github.com/yourusername/forex-ai.git /workspace/forex-ai
cd /workspace/forex-ai

# Set up data directory
mkdir -p models logs

# Download required data (optional - can be done during training)
echo "📊 Environment ready for training!"
echo "💡 Run: python train.py --epochs 500 --pairs USDJPY --period 2y"

# Keep container running
tail -f /dev/null
