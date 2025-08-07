# 🤖 Telegram Trading Bot

Control your enhanced forex trading system remotely via Telegram! Get real-time notifications, monitor positions, and manage your trading directly from your phone.

## ✨ Features

- 🎮 **Remote Control**: Start/stop trading from anywhere
- 📊 **Live Monitoring**: Real-time status, balance, and positions
- 🔔 **Smart Notifications**: Instant alerts for trades and system events  
- 🔒 **Secure Access**: Authorized users only
- 💡 **Interactive Interface**: Easy-to-use buttons and commands
- 📱 **Mobile Friendly**: Perfect for monitoring on the go

## 🚀 Quick Start

### 1. One-Command Setup (Easiest)

```bash
bash quick_bot_setup.sh
```

This interactive script will guide you through everything!

### 2. Manual Setup

1. **Create Telegram Bot**
   - Message [@BotFather](https://t.me/botfather)
   - Send `/newbot` and follow instructions
   - Save your bot token

2. **Get Your User ID**
   - Message [@userinfobot](https://t.me/userinfobot)
   - Save your user ID number

3. **Start Bot**
   ```bash
   ./start_bot.sh --token YOUR_BOT_TOKEN --users YOUR_USER_ID
   ```

## 📱 Using the Bot

### Main Commands

```
/start     - Welcome and main menu
/help      - Show all commands
/status    - Trading system status
/balance   - Account balance and P&L
/positions - Open positions
/history   - Recent trades
/config    - Current settings
```

### Trading Controls

```
/start_trading - Start the system
/stop_trading  - Stop the system
```

### Quick Actions

The bot provides interactive buttons for instant access to all features.

## 🔔 Notifications

Get notified about:
- 🚀 System start/stop
- 📈 New positions opened
- 💰 Positions closed (with P&L)
- 🕐 Hourly status updates
- ⚠️ Errors and warnings

## 🛡️ Security

- Only authorized users can access the bot
- All actions are logged
- Unauthorized attempts are blocked
- Bot token is never exposed

## 🖥️ Server Deployment

### Option 1: Screen Session (Simple)
```bash
screen -S trading_bot
./start_bot.sh --token YOUR_TOKEN --users YOUR_ID
# Press Ctrl+A, then D to detach
# Reconnect later: screen -r trading_bot
```

### Option 2: Systemd Service (Advanced)
See `TELEGRAM_BOT_SETUP.md` for detailed instructions.

## 📊 Example Usage

```
You: /start
Bot: 🤖 Welcome to Enhanced Trading Bot!
     [Interactive buttons appear]

You: Click "📊 Status"
Bot: 📊 Trading System Status
     🔴 Status: Stopped
     🧪 Mode: Demo
     💰 Balance: $10,000.00

You: /start_trading
Bot: 🚀 Trading system started!

[Later...]
Bot: 🔔 New Position Opened
     💱 EURUSD=X
     🔼 BUY | Size: 200.00
     🎯 Confidence: 72.3%

[Even later...]
Bot: ✅ Position Closed
     💰 P&L: +$15.50
     📊 Take Profit Hit
```

## 🔧 Configuration

### Environment Variables (.env)
```bash
BOT_TOKEN=your_bot_token_here
ALLOWED_USERS=123456789,987654321
DEMO_MODE=true
```

### Trading Config (trading_config.json)
```json
{
  "initial_balance": 10000,
  "demo_mode": true,
  "position_size_pct": 0.02,
  "max_positions": 3,
  "pairs": ["EURUSD=X", "GBPUSD=X", "USDJPY=X"]
}
```

## 📋 Requirements

- Python 3.8+
- Trained enhanced models
- Internet connection
- Telegram account

## 🐛 Troubleshooting

**Common Issues:**

1. **"Bot token not found"**
   - Check your token format
   - Ensure no extra spaces

2. **"Models not found"**
   - Run: `python enhanced_loss_learning_trainer.py`

3. **"Unauthorized user"**
   - Add your user ID to the allowed list

4. **Connection problems**
   - Check internet connection
   - Verify token is correct

## 📝 Logs

Monitor bot activity:
```bash
tail -f telegram_bot.log    # Bot logs
tail -f live_trading.log    # Trading logs
```

## ⚠️ Important Notes

- **Always test with demo mode first**
- **Never share your bot token**
- **Keep your user ID private**
- **Monitor logs regularly**
- **Have a stop-loss strategy**

## 🆘 Support

1. Check the logs for error messages
2. Verify your configuration files
3. Test with demo mode first
4. Read the full setup guide: `TELEGRAM_BOT_SETUP.md`

---

**Ready to trade from anywhere? Set up your Telegram bot and take control of your trading system! 📱💹**
