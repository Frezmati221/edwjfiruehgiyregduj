#!/bin/bash

# Telegram Trading Bot Startup Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🤖 Telegram Trading Bot Launcher${NC}"
echo "=================================="

# Check if virtual environment exists
if [ ! -d "forex_env" ]; then
    echo -e "${RED}❌ Virtual environment not found!${NC}"
    echo "Please run setup.sh first to create the environment."
    exit 1
fi

# Activate virtual environment
echo -e "${YELLOW}📦 Activating virtual environment...${NC}"
source forex_env/bin/activate

# Check if Telegram bot dependencies are installed
echo -e "${YELLOW}📋 Checking dependencies...${NC}"
python -c "import telegram" 2>/dev/null || {
    echo -e "${YELLOW}📦 Installing Telegram bot dependencies...${NC}"
    pip install python-telegram-bot>=20.7
}

# Check if enhanced models exist
if [ ! -d "enhanced_models" ] || [ ! -f "enhanced_models/EURUSD_X_enhanced_loss_learning.pkl" ]; then
    echo -e "${RED}❌ Enhanced models not found!${NC}"
    echo "Please train the models first:"
    echo "  python enhanced_loss_learning_trainer.py"
    exit 1
fi

# Check for bot token
if [ ! -f "bot_config.json" ]; then
    echo -e "${YELLOW}⚙️ Creating default bot configuration...${NC}"
    cp bot_config.json.example bot_config.json 2>/dev/null || echo "Please create bot_config.json with your bot token"
fi

# Function to get bot token
get_bot_token() {
    if [ -n "$BOT_TOKEN" ]; then
        echo "$BOT_TOKEN"
    elif [ -f ".env" ]; then
        grep "BOT_TOKEN=" .env | cut -d'=' -f2
    else
        echo ""
    fi
}

# Function to get allowed users
get_allowed_users() {
    if [ -n "$ALLOWED_USERS" ]; then
        echo "$ALLOWED_USERS"
    elif [ -f ".env" ]; then
        grep "ALLOWED_USERS=" .env | cut -d'=' -f2 | tr ',' ' '
    else
        echo ""
    fi
}

# Get configuration
BOT_TOKEN_VALUE=$(get_bot_token)
ALLOWED_USERS_VALUE=$(get_allowed_users)

# Check for bot token
if [ -z "$BOT_TOKEN_VALUE" ] && [ "$1" != "--token" ]; then
    echo -e "${RED}❌ Bot token not found!${NC}"
    echo ""
    echo "Please provide your Telegram bot token in one of these ways:"
    echo ""
    echo "1. Command line argument:"
    echo "   ./start_bot.sh --token YOUR_BOT_TOKEN"
    echo ""
    echo "2. Environment variable:"
    echo "   export BOT_TOKEN=YOUR_BOT_TOKEN"
    echo "   ./start_bot.sh"
    echo ""
    echo "3. Create .env file:"
    echo "   echo 'BOT_TOKEN=YOUR_BOT_TOKEN' > .env"
    echo "   echo 'ALLOWED_USERS=123456789,987654321' >> .env"
    echo ""
    echo "To get a bot token:"
    echo "1. Message @BotFather on Telegram"
    echo "2. Send /newbot"
    echo "3. Follow the instructions"
    echo "4. Copy the token provided"
    exit 1
fi

# Parse command line arguments
PYTHON_ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --token)
            BOT_TOKEN_VALUE="$2"
            PYTHON_ARGS="$PYTHON_ARGS --token $2"
            shift 2
            ;;
        --users)
            shift
            USER_LIST=""
            while [[ $# -gt 0 && $1 != --* ]]; do
                USER_LIST="$USER_LIST $1"
                shift
            done
            PYTHON_ARGS="$PYTHON_ARGS --users$USER_LIST"
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --token TOKEN    Telegram bot token"
            echo "  --users USERS    Space-separated list of authorized user IDs"
            echo "  --help           Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --token 123456:ABC --users 123456789 987654321"
            echo "  $0 --token 123456:ABC"
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Add users from environment if not provided in command line
if [ -n "$ALLOWED_USERS_VALUE" ] && [[ "$PYTHON_ARGS" != *"--users"* ]]; then
    PYTHON_ARGS="$PYTHON_ARGS --users $ALLOWED_USERS_VALUE"
fi

# Validate bot token format
if [[ ! "$BOT_TOKEN_VALUE" =~ ^[0-9]+:[a-zA-Z0-9_-]+$ ]]; then
    echo -e "${RED}❌ Invalid bot token format!${NC}"
    echo "Bot token should look like: 123456789:ABCdefGHIjklMNOpqrSTUvwxyz"
    exit 1
fi

# Final check for required files
echo -e "${YELLOW}🔍 Checking required files...${NC}"
required_files=(
    "enhanced_live_trader.py"
    "telegram_trading_bot.py"
    "trading_config.json"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo -e "${RED}❌ Required file missing: $file${NC}"
        exit 1
    fi
done

echo -e "${GREEN}✅ All checks passed!${NC}"
echo ""
echo -e "${BLUE}🚀 Starting Telegram Trading Bot...${NC}"
echo "Bot Token: ${BOT_TOKEN_VALUE:0:10}..."
echo "Arguments: $PYTHON_ARGS"
echo ""
echo -e "${YELLOW}📱 To interact with your bot:${NC}"
echo "1. Open Telegram"
echo "2. Search for your bot"
echo "3. Send /start"
echo ""
echo -e "${YELLOW}⏹️  To stop the bot: Press Ctrl+C${NC}"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Start the bot with proper error handling
source forex_env/bin/activate
python telegram_trading_bot.py $PYTHON_ARGS 2>&1 | tee logs/bot_$(date +%Y%m%d_%H%M%S).log
