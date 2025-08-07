# Telegram Trading Bot Setup Guide

## Overview

This Telegram bot provides a complete interface to control your enhanced forex trading system remotely. You can start/stop trading, monitor positions, check balance, and receive real-time notifications directly through Telegram.

## Features

- 🤖 **Remote Control**: Start/stop trading system via Telegram
- 📊 **Real-time Monitoring**: Check status, balance, positions, and trade history
- 🔔 **Smart Notifications**: Get notified about new trades, closed positions, and system status
- 🔒 **Secure Access**: Only authorized users can control the system
- 💰 **Demo & Live Modes**: Support for both demo and live trading
- 📈 **Multi-pair Trading**: EURUSD, GBPUSD, USDJPY support
- ⚙️ **Configurable**: Customizable trading parameters and notifications

## Prerequisites

1. **Trained Models**: Ensure you have trained enhanced models
2. **Telegram Bot**: Create a Telegram bot and get the token
3. **Python Environment**: Virtual environment with dependencies

## Quick Setup

### 1. Create Telegram Bot

1. Message [@BotFather](https://t.me/botfather) on Telegram
2. Send `/newbot`
3. Choose a name for your bot (e.g., "My Trading Bot")
4. Choose a username (must end with 'bot', e.g., "my_trading_bot")
5. Copy the bot token (format: `123456789:ABCdefGHIjklMNOpqrSTUvwxyz`)

### 2. Get Your User ID

1. Message [@userinfobot](https://t.me/userinfobot) on Telegram
2. Send any message
3. Copy your user ID (a number like `123456789`)

### 3. Install Dependencies

```bash
# Activate virtual environment
source forex_env/bin/activate

# Install Telegram bot dependency
pip install python-telegram-bot>=20.7
```

### 4. Configure Bot

Edit `bot_config.json`:
```json
{
  "bot_token": "YOUR_BOT_TOKEN_HERE",
  "allowed_users": [
    123456789
  ]
}
```

### 5. Start Bot

```bash
# Method 1: Using startup script (recommended)
./start_bot.sh --token YOUR_BOT_TOKEN --users YOUR_USER_ID

# Method 2: Direct command
python telegram_trading_bot.py --token YOUR_BOT_TOKEN --users YOUR_USER_ID

# Method 3: Environment variables
export BOT_TOKEN="YOUR_BOT_TOKEN"
export ALLOWED_USERS="123456789,987654321"
./start_bot.sh
```

## Bot Commands

### Main Commands
- `/start` - Welcome message and main menu
- `/help` - Show help information

### Trading Control
- `/start_trading` - Start the trading system
- `/stop_trading` - Stop the trading system
- `/status` - Current system status

### Monitoring
- `/balance` - Account balance and P&L
- `/positions` - View open positions
- `/history` - Recent trade history
- `/config` - View current configuration

## Interactive Features

The bot provides interactive buttons for quick access:

```
┌─────────────────────────────────┐
│ 📊 Status    | 💰 Balance       │
│ 📈 Positions | 📋 History       │
│ ▶️ Start      | ⏹️ Stop          │
│ ⚙️ Config     | 🔔 Notifications │
└─────────────────────────────────┘
```

## Notifications

The bot sends automatic notifications for:

- 🚀 **System Events**: Start/stop trading
- 🔔 **New Positions**: When positions are opened
- ✅ **Closed Positions**: When positions are closed (with P&L)
- 🕐 **Periodic Updates**: Hourly status updates
- ❌ **Errors**: System errors and warnings

## Security Features

- 🔒 **User Authorization**: Only specified users can access the bot
- 📝 **Audit Logging**: All commands and actions are logged
- 🚫 **Unauthorized Access**: Unauthorized attempts are blocked and logged
- 🔐 **Token Security**: Bot token is never exposed in logs

## Advanced Configuration

### Trading Parameters

Modify `trading_config.json`:
```json
{
  "initial_balance": 10000,
  "demo_mode": true,
  "position_size_pct": 0.02,
  "max_daily_risk": 0.06,
  "max_positions": 3,
  "update_interval": 60,
  "pairs": ["EURUSD=X", "GBPUSD=X", "USDJPY=X"]
}
```

### Bot Settings

Modify `bot_config.json`:
```json
{
  "notifications": {
    "enabled": true,
    "periodic_updates": true,
    "update_interval_hours": 1,
    "trade_notifications": true
  },
  "security": {
    "require_authorization": true,
    "log_unauthorized_attempts": true
  }
}
```

## Running on Server

### Using Screen (Recommended)

```bash
# Start a new screen session
screen -S trading_bot

# Run the bot
./start_bot.sh --token YOUR_BOT_TOKEN --users YOUR_USER_ID

# Detach from screen: Ctrl+A, then D
# Reattach later: screen -r trading_bot
```

### Using Systemd Service

Create `/etc/systemd/system/trading-bot.service`:
```ini
[Unit]
Description=Telegram Trading Bot
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/path/to/eternity
Environment=BOT_TOKEN=YOUR_BOT_TOKEN
Environment=ALLOWED_USERS=YOUR_USER_ID
ExecStart=/path/to/eternity/start_bot.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl daemon-reload
sudo systemctl enable trading-bot
sudo systemctl start trading-bot
sudo systemctl status trading-bot
```

## Monitoring Logs

```bash
# Bot logs
tail -f telegram_bot.log

# Trading logs
tail -f live_trading.log

# All logs together
tail -f *.log
```

## Troubleshooting

### Common Issues

1. **"Bot token not found"**
   - Ensure you've provided the bot token correctly
   - Check token format: `123456789:ABCdefGHIjklMNOpqrSTUvwxyz`

2. **"Models not found"**
   - Run training first: `python enhanced_loss_learning_trainer.py`
   - Check `enhanced_models/` directory exists

3. **"Unauthorized user"**
   - Add your user ID to allowed_users list
   - Get your ID from @userinfobot

4. **Connection errors**
   - Check internet connection
   - Verify bot token is correct
   - Ensure Telegram isn't blocked by firewall

### Debug Mode

Run with debug logging:
```bash
python telegram_trading_bot.py --token YOUR_TOKEN --users YOUR_ID --debug
```

## Example Usage Session

```
User: /start
Bot: 🤖 Enhanced Trading Bot
     Welcome! Use buttons below or commands.

User: /status
Bot: 📊 Trading System Status
     🔴 Status: Stopped
     🧪 Mode: Demo
     💰 Balance: $10,000.00

User: /start_trading
Bot: 🚀 Trading system started successfully!
     Mode: 🧪 Demo
     Balance: $10,000.00
     Pairs: EURUSD=X, GBPUSD=X, USDJPY=X

[After some time...]
Bot: 🔔 New Position Opened
     💱 Pair: EURUSD=X
     🔼 Direction: BUY
     💰 Size: 200.00
     🎯 Confidence: 72.3%

[Later...]
Bot: ✅ Position Closed
     💱 Pair: EURUSD=X
     🔼 Direction: BUY
     💰 P&L: $15.50
     📊 Reason: Take Profit
```

## Support

- Check logs for error messages
- Verify all configuration files are correct
- Ensure virtual environment is activated
- Test with demo mode first before live trading

---

**⚠️ Important**: Always test with demo mode before using live trading. Never share your bot token or run untrusted code with live trading enabled.
