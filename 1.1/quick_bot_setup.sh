#!/bin/bash

# Quick Telegram Bot Setup Script

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BLUE}${BOLD}🤖 Telegram Trading Bot Quick Setup${NC}"
echo "==========================================="

# Check if we're in the right directory
if [ ! -f "enhanced_live_trader.py" ]; then
    echo -e "${RED}❌ Please run this script from the eternity directory${NC}"
    exit 1
fi

# Function to read user input
read_input() {
    echo -e "${YELLOW}$1${NC}"
    read -r response
    echo "$response"
}

# Check if already configured
if [ -f ".env" ]; then
    echo -e "${GREEN}✅ Configuration file already exists${NC}"
    echo "Do you want to reconfigure? (y/N)"
    read -r reconfigure
    if [[ ! "$reconfigure" =~ ^[Yy]$ ]]; then
        echo "Using existing configuration..."
        source .env
        ./start_bot.sh
        exit 0
    fi
fi

echo ""
echo -e "${BOLD}📝 Let's configure your Telegram bot!${NC}"
echo ""

# Get bot token
echo -e "${YELLOW}Step 1: Bot Token${NC}"
echo "If you don't have a bot token yet:"
echo "1. Message @BotFather on Telegram"
echo "2. Send /newbot"
echo "3. Follow the instructions"
echo "4. Copy the token"
echo ""

while true; do
    BOT_TOKEN=$(read_input "Enter your bot token:")
    if [[ "$BOT_TOKEN" =~ ^[0-9]+:[a-zA-Z0-9_-]+$ ]]; then
        break
    else
        echo -e "${RED}❌ Invalid token format. Should be like: 123456789:ABCdefGHI${NC}"
    fi
done

echo ""
echo -e "${YELLOW}Step 2: User Authorization${NC}"
echo "If you don't know your user ID:"
echo "1. Message @userinfobot on Telegram"
echo "2. Send any message"
echo "3. Copy your user ID"
echo ""

while true; do
    USER_ID=$(read_input "Enter your Telegram user ID:")
    if [[ "$USER_ID" =~ ^[0-9]+$ ]]; then
        break
    else
        echo -e "${RED}❌ User ID should be a number${NC}"
    fi
done

# Optional: Add more users
echo ""
echo "Do you want to add more authorized users? (y/N)"
read -r add_more
ALLOWED_USERS="$USER_ID"

if [[ "$add_more" =~ ^[Yy]$ ]]; then
    while true; do
        echo "Enter another user ID (or press Enter to finish):"
        read -r additional_user
        if [ -z "$additional_user" ]; then
            break
        elif [[ "$additional_user" =~ ^[0-9]+$ ]]; then
            ALLOWED_USERS="$ALLOWED_USERS,$additional_user"
            echo "Added user: $additional_user"
        else
            echo -e "${RED}❌ User ID should be a number${NC}"
        fi
    done
fi

echo ""
echo -e "${YELLOW}Step 3: Trading Configuration${NC}"

# Demo mode
echo "Do you want to start with demo mode? (Y/n)"
read -r demo_response
if [[ "$demo_response" =~ ^[Nn]$ ]]; then
    DEMO_MODE="false"
    echo -e "${RED}⚠️  LIVE TRADING MODE SELECTED - BE CAREFUL!${NC}"
else
    DEMO_MODE="true"
    echo -e "${GREEN}✅ Demo mode selected (safe for testing)${NC}"
fi

# Balance
if [ "$DEMO_MODE" = "true" ]; then
    INITIAL_BALANCE=$(read_input "Enter initial demo balance (default: 10000):")
    INITIAL_BALANCE=${INITIAL_BALANCE:-10000}
else
    INITIAL_BALANCE=$(read_input "Enter your actual trading balance:")
fi

echo ""
echo -e "${YELLOW}Step 4: Creating configuration...${NC}"

# Create .env file
cat > .env << EOF
# Telegram Trading Bot Configuration
BOT_TOKEN=$BOT_TOKEN
ALLOWED_USERS=$ALLOWED_USERS

# Trading Configuration
DEMO_MODE=$DEMO_MODE
INITIAL_BALANCE=$INITIAL_BALANCE
NOTIFICATIONS_ENABLED=true
PERIODIC_UPDATES=true
UPDATE_INTERVAL_HOURS=1
EOF

echo -e "${GREEN}✅ Configuration saved to .env${NC}"

# Update trading config if needed
if [ "$DEMO_MODE" = "false" ] || [ "$INITIAL_BALANCE" != "10000" ]; then
    echo "Updating trading_config.json..."
    
    # Create backup
    cp trading_config.json trading_config.json.backup
    
    # Update config
    python3 -c "
import json
with open('trading_config.json', 'r') as f:
    config = json.load(f)
config['demo_mode'] = $DEMO_MODE
config['initial_balance'] = $INITIAL_BALANCE
with open('trading_config.json', 'w') as f:
    json.dump(config, f, indent=2)
"
fi

echo ""
echo -e "${YELLOW}Step 5: Checking dependencies...${NC}"

# Check virtual environment
if [ ! -d "forex_env" ]; then
    echo -e "${RED}❌ Virtual environment not found!${NC}"
    echo "Please run: bash setup.sh"
    exit 1
fi

# Activate and check dependencies
source forex_env/bin/activate

echo "Installing Telegram bot dependencies..."
pip install python-telegram-bot>=20.7 > /dev/null 2>&1

echo ""
echo -e "${YELLOW}Step 6: Validating setup...${NC}"

# Check models
if [ ! -d "enhanced_models" ] || [ ! -f "enhanced_models/EURUSD_X_enhanced_loss_learning.pkl" ]; then
    echo -e "${YELLOW}⚠️  Enhanced models not found. Training models...${NC}"
    echo "This may take several minutes..."
    python enhanced_loss_learning_trainer.py
fi

echo ""
echo -e "${GREEN}${BOLD}🎉 Setup Complete!${NC}"
echo ""
echo -e "${BOLD}📱 Next steps:${NC}"
echo "1. Open Telegram"
echo "2. Search for your bot"
echo "3. Send /start to begin"
echo ""
echo -e "${BOLD}🚀 Starting your bot now...${NC}"
echo "Press Ctrl+C to stop the bot"
echo ""

# Start the bot
./start_bot.sh
