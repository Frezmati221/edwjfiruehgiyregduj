# Trade-2 Telegram Bot

## 🎯 Real-time Forex Predictions via Telegram

This Telegram bot provides AI-powered forex predictions using your trained Trade-2 supervised learning model. Get instant market analysis, technical indicators, and trading signals directly in Telegram!

## ✨ Features

### 🤖 AI Predictions
- **Real-time Analysis**: Live forex predictions using your trained LSTM model
- **8 Major Pairs**: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, EURJPY, GBPJPY
- **Confidence Levels**: Adjustable from 30% to 80% to match your risk tolerance
- **Smart Signals**: BUY/SELL/HOLD recommendations with stop-loss and take-profit levels

### 📊 Technical Analysis
- **Multi-timeframe**: 1-hour candles with 3-month lookback
- **Risk Management**: Automatic SL/TP calculation with risk-reward ratios
- **Market Overview**: Current price, 24h change, volatility metrics
- **Key Levels**: Support and resistance identification

### 🎛️ Interactive Interface
- **Quick Buttons**: One-click predictions for major pairs
- **Settings Menu**: Customize confidence thresholds and preferences
- **Real-time Updates**: Refresh predictions with latest market data
- **User Preferences**: Save favorite pairs and notification settings

## 🚀 Quick Setup

### 1. Prerequisites
```bash
# Your trained model files
best_model.pth                    # Required
best_model_scaler.pkl            # Optional but recommended
compatible_trade2_predictor.py   # Required predictor module
```

### 2. Get Telegram Bot Token
1. Message [@BotFather](https://t.me/BotFather) on Telegram
2. Send `/newbot`
3. Name: `Trade-2 Forex Bot`
4. Username: `your_trade2_bot` (must be unique)
5. Copy the provided token

### 3. Install & Run
```bash
# Run the setup script
./setup_telegram_bot.sh

# Or manual setup:
pip install python-telegram-bot[all] yfinance pandas torch

# Set your bot token
export TELEGRAM_BOT_TOKEN='your_token_here'

# Start the bot
python3 trade2_telegram_bot.py
```

### 4. Use Your Bot
1. Find your bot on Telegram
2. Send `/start`
3. Click buttons for instant predictions!

## 📋 Commands

### Basic Commands
- `/start` - Welcome message and main menu
- `/predict [PAIR]` - Get prediction (e.g., `/predict EURUSD`)
- `/analysis [PAIR]` - Detailed technical analysis
- `/pairs` - List all available forex pairs
- `/settings` - Configure bot preferences
- `/status` - Model and system information
- `/help` - Show available commands

### Quick Examples
```
/predict EURUSD     # Get EURUSD prediction
/analysis GBPUSD    # Detailed GBP/USD analysis
/settings           # Open settings menu
```

## 🎛️ Settings & Configuration

### Confidence Levels
- **30-40%**: More signals, lower accuracy (aggressive)
- **50%**: Balanced approach (recommended)
- **60-70%**: Fewer signals, higher accuracy (conservative)
- **80%+**: Very selective, rare but strong signals

### Bot Configuration
Edit `bot_config.json` to customize:
```json
{
  "trading": {
    "default_confidence": 0.5,
    "max_predictions_per_user": 50
  },
  "market_data": {
    "history_period": "1mo",
    "interval": "1h"
  }
}
```

## 📊 Sample Output

### Quick Prediction
```
🎯 EURUSD Analysis

💹 Current Market:
📈 Price: 1.08945
📊 Change: +0.00125 (+0.11%)
🕐 Updated: 14:32:15 UTC

🤖 AI Prediction:
🟢📈 Action: LONG
🎯 Confidence: 67%
🛡️ Stop Loss: 1.08750 (19.5 pips)
🎯 Take Profit: 1.09280 (33.5 pips)
📊 Risk:Reward: 1:1.72

💡 Interpretation:
🟠 Medium confidence - Consider carefully
```

### Detailed Analysis
```
📊 EURUSD Detailed Analysis

📈 Market Overview:
💹 Current: 1.08945
🔺 24h High: 1.09125
🔻 24h Low: 1.08720
📊 Volatility: 0.85%

🤖 Multi-Confidence Analysis:
30% Threshold: 🟢📈 LONG (72%)
50% Threshold: 🟢📈 LONG (67%)
70% Threshold: 🔄 HOLD (45%)

📏 Key Levels:
🔺 Resistance: 1.09250
🔻 Support: 1.08650
```

## 🔧 Advanced Features

### Environment Variables
```bash
export TELEGRAM_BOT_TOKEN='your_bot_token'
export LOG_LEVEL='INFO'                    # DEBUG, INFO, WARNING, ERROR
export MAX_USERS='100'                     # Limit concurrent users
export CACHE_MINUTES='5'                   # Prediction cache duration
```

### Custom Forex Pairs
Add more pairs in `bot_config.json`:
```json
"forex_pairs": {
  "NZDUSD": "NZDUSD=X",
  "EURCAD": "EURCAD=X",
  "GBPCAD": "GBPCAD=X"
}
```

### Logging
Logs are saved to:
- `logs/telegram_bot.log` - Bot activity
- Console output - Real-time status

## 🛡️ Security & Best Practices

### Bot Token Security
- Never share your bot token publicly
- Store in environment variables, not code
- Regenerate if compromised via @BotFather

### Trading Disclaimer
- **Educational Purpose**: Bot provides analysis only
- **Not Financial Advice**: Always do your own research
- **Risk Management**: Never risk more than you can afford to lose
- **Paper Trading**: Test strategies before live trading

### Privacy
- Bot doesn't store personal data
- Predictions are not logged to external services
- User settings stored locally only

## 📈 Performance Tips

### Optimize Predictions
- Lower confidence = More frequent signals
- Higher confidence = More selective signals
- Test different levels with your trading style

### Resource Usage
- Predictions cached for 5 minutes
- Rate limited to prevent abuse
- Automatic cleanup of old data

## 🐛 Troubleshooting

### Common Issues

**Bot not responding:**
```bash
# Check if bot is running
ps aux | grep telegram_bot

# Check logs
tail -f logs/telegram_bot.log
```

**Import errors:**
```bash
# Reinstall dependencies
pip install --upgrade python-telegram-bot yfinance torch

# Check Python path
python3 -c "import sys; print(sys.path)"
```

**Model loading failed:**
```bash
# Verify model files exist
ls -la best_model.pth compatible_trade2_predictor.py

# Test model loading
python3 -c "from compatible_trade2_predictor import CompatibleSupervisedForexPredictor; p=CompatibleSupervisedForexPredictor(); p.load_model('best_model.pth')"
```

**No market data:**
```bash
# Test Yahoo Finance connection
python3 -c "import yfinance as yf; print(yf.Ticker('EURUSD=X').history(period='1d'))"
```

### Error Codes
- `❌ Model loading failed` - Check model file path and compatibility
- `❌ No data available` - Yahoo Finance connection issue
- `❌ Prediction failed` - Model input format error
- `❌ Unknown pair` - Pair not in configured list

## 🔄 Updates & Maintenance

### Keep Dependencies Updated
```bash
pip install --upgrade python-telegram-bot yfinance torch pandas
```

### Monitor Performance
```bash
# Check bot logs
tail -f logs/telegram_bot.log

# Monitor system resources
htop
```

### Backup Important Files
```bash
# Backup model and config
tar -czf trade2_bot_backup.tar.gz best_model.pth bot_config.json compatible_trade2_predictor.py
```

## 🤝 Support

### Getting Help
1. Check this README first
2. Review log files for errors
3. Test individual components (model, data, telegram)
4. Verify all dependencies are installed

### Feature Requests
The bot is designed to be extensible. Common enhancements:
- Additional forex pairs
- Different timeframes (15m, 4h, daily)
- Economic calendar integration
- Portfolio tracking
- Trade alerts and notifications

---

**⚠️ Disclaimer**: This bot provides AI-generated analysis for educational purposes only. It is not financial advice. Trading forex carries significant risk of loss. Always do your own research and never risk more than you can afford to lose.
