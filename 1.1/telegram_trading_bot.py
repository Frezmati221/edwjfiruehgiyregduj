"""
TELEGRAM TRADING BOT INTERFACE
Control your enhanced trading system via Telegram
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import threading
import time

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler, 
    ContextTypes
)

from telegram_signal_bot import TelegramSignalBot

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('telegram_bot.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class TelegramTradingBot:
    def __init__(self, config_path: str = "bot_config.json"):
        """Initialize the Telegram trading bot"""
        self.config = self.load_config(config_path)
        self.trader = None
        self.application = None
        self._status_messages = {}
        
        # Bot settings
        self.bot_token = self.config.get('bot_token')
        self.authorized_users = self.config.get('authorized_users', [])
        
        if not self.bot_token:
            raise ValueError("Bot token not found in config!")
        
        logger.info(f"Bot initialized. Authorized users: {self.authorized_users}")
    
    def load_config(self, config_path: str) -> Dict:
        """Load bot configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Config file {config_path} not found!")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {config_path}: {e}")
            return {}
    
    def is_authorized(self, user_id: int) -> bool:
        """Check if user is authorized"""
        return user_id in self.authorized_users
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start command handler"""
        user_id = update.effective_user.id
        
        if not self.is_authorized(user_id):
            await update.message.reply_text("❌ Unauthorized access.")
            return
        
        welcome_message = """
🤖 **Enhanced Trading Bot - Signal Mode**

Welcome! This bot provides trading signals only.

**Available Commands:**
/start - Show this menu
/status - Current status
/help - Help and info

Use the buttons below for quick access:
        """
        
        keyboard = [
            [
                InlineKeyboardButton("▶️ Start", callback_data='start_scanner'),
                InlineKeyboardButton("⏹️ Stop", callback_data='stop_scanner')
            ],
            [
                InlineKeyboardButton("📊 Status", callback_data='status'),
                InlineKeyboardButton("🔔 Toggle Signals", callback_data='toggle_signals')
            ],
            [
                InlineKeyboardButton("🔍 Analyze Now", callback_data='analyze_now'),
                InlineKeyboardButton("⚙️ Config", callback_data='config')
            ],
            [
                InlineKeyboardButton("❓ Help", callback_data='help')
            ]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(welcome_message, reply_markup=reply_markup, parse_mode='Markdown')
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Status command handler"""
        user_id = update.effective_user.id
        
        if not self.is_authorized(user_id):
            await update.message.reply_text("❌ Unauthorized access.")
            return
        
        status_text = self.get_status_text()
        
        keyboard = [
            [
                InlineKeyboardButton("🔄 Refresh", callback_data='status'),
                InlineKeyboardButton("🔙 Back", callback_data='start')
            ]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(status_text, reply_markup=reply_markup, parse_mode='Markdown')
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Help command handler"""
        user_id = update.effective_user.id
        
        if not self.is_authorized(user_id):
            await update.message.reply_text("❌ Unauthorized access.")
            return
        
        help_text = """
📚 **Signal Bot Help**

**Commands:**
• `/start` - Main menu and bot controls
• `/status` - View current scanner status
• `/help` - Show this help message

**Signal Mode:**
• Bot scans EURUSD, GBPUSD, USDJPY
• Sends signals when confidence > 70%
• Includes SL/TP levels for manual trading
• No auto-trading (signals only)

**Features:**
• **🔍 Analyze Now** - Immediate market scan for current opportunities
• **▶️ Start/Stop** - Control automatic signal scanning
• **🔔 Toggle Signals** - Enable/disable notifications

**Thresholds:**
• Confidence: 70% minimum
• Signal strength: 8% minimum
• Cooldown: 30 minutes between signals (except manual analysis)

For technical support, check the logs.
        """
        
        keyboard = [
            [InlineKeyboardButton("🔙 Back to Menu", callback_data='start')]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(help_text, reply_markup=reply_markup, parse_mode='Markdown')
    
    async def send_response(self, update: Update, text: str, reply_markup=None):
        """Send response handling both messages and callback queries"""
        try:
            # Escape special characters for Markdown
            safe_text = text.replace('_', '\\_').replace('*', '\\*').replace('[', '\\[').replace(']', '\\]').replace('`', '\\`')
            
            if update.callback_query:
                if reply_markup:
                    await update.callback_query.edit_message_text(safe_text, reply_markup=reply_markup, parse_mode='Markdown')
                else:
                    await update.callback_query.edit_message_text(safe_text, parse_mode='Markdown')
            else:
                await update.message.reply_text(safe_text, reply_markup=reply_markup, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Error sending response: {e}")
            # Fallback - try sending as plain text
            try:
                if update.callback_query:
                    if reply_markup:
                        await update.callback_query.edit_message_text(text, reply_markup=reply_markup)
                    else:
                        await update.callback_query.edit_message_text(text)
                else:
                    await update.message.reply_text(text, reply_markup=reply_markup)
            except Exception as e2:
                logger.error(f"Fallback send also failed: {e2}")
                # Final fallback - new message
                try:
                    chat_id = update.effective_chat.id
                    await self.application.bot.send_message(chat_id=chat_id, text=text, reply_markup=reply_markup)
                except Exception as e3:
                    logger.error(f"Final fallback also failed: {e3}")

    async def start_scanner_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start the trading scanner"""
        user_id = update.effective_user.id
        
        if not self.is_authorized(user_id):
            await self.send_response(update, "❌ Unauthorized access.")
            return
        
        try:
            if not self.trader:
                await self.send_response(update, "🔄 Initializing signal scanner...")
                
                # Initialize trader with Telegram callback
                self.trader = TelegramSignalBot(
                    initial_balance=10000,
                    demo_mode=True,
                    telegram_bot=self
                )
                await asyncio.sleep(1)
                
                await self.send_response(update, "✅ Signal scanner initialized!")
            
            if not self.trader.is_running:
                success = await self.trader.start_trading()
                if success:
                    await self.send_response(update, "🚀 **Signal scanner started!**\n\nWatching for trading opportunities...")
                else:
                    await self.send_response(update, "❌ Failed to start signal scanner. Check logs.")
            else:
                await self.send_response(update, "ℹ️ Signal scanner is already running.")
                
        except Exception as e:
            logger.error(f"Error starting scanner: {e}")
            await self.send_response(update, f"❌ Error: {str(e)}")
    
    async def stop_scanner_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Stop the trading scanner"""
        user_id = update.effective_user.id
        
        if not self.is_authorized(user_id):
            await self.send_response(update, "❌ Unauthorized access.")
            return
        
        try:
            if self.trader and self.trader.is_running:
                await self.trader.stop_trading()
                await self.send_response(update, "⏹️ **Signal scanner stopped.**")
            else:
                await self.send_response(update, "ℹ️ Signal scanner is not running.")
                
        except Exception as e:
            logger.error(f"Error stopping scanner: {e}")
            await self.send_response(update, f"❌ Error: {str(e)}")
    
    async def toggle_signals_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Toggle signal notifications"""
        user_id = update.effective_user.id
        
        if not self.is_authorized(user_id):
            await self.send_response(update, "❌ Unauthorized access.")
            return
        
        try:
            if not self.trader:
                await self.send_response(update, "❌ Signal scanner not initialized.")
                return
            
            # Toggle signals
            if hasattr(self.trader, 'signal_enabled'):
                self.trader.signal_enabled = not self.trader.signal_enabled
                status = "enabled" if self.trader.signal_enabled else "disabled"
                emoji = "🔔" if self.trader.signal_enabled else "🔕"
                await self.send_response(update, f"{emoji} **Signals {status}.**")
            else:
                await self.send_response(update, "❌ Signal control not available.")
                
        except Exception as e:
            logger.error(f"Error toggling signals: {e}")
            await self.send_response(update, f"❌ Error: {str(e)}")
    
    async def config_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show current configuration"""
        user_id = update.effective_user.id
        
        if not self.is_authorized(user_id):
            await self.send_response(update, "❌ Unauthorized access.")
            return
        
        try:
            config_text = """
⚙️ **Current Configuration**

**Signal Thresholds:**
• Confidence: 70% minimum
• Signal strength: 8% minimum  
• Cooldown: 30 minutes

**Monitored Pairs:**
• EUR/USD
• GBP/USD
• USD/JPY

**Mode:** Signals Only
**Account:** Demo Mode
            """
            
            keyboard = [
                [InlineKeyboardButton("🔙 Back to Menu", callback_data='start')]
            ]
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            await self.send_response(update, config_text, reply_markup)
            
        except Exception as e:
            logger.error(f"Error showing config: {e}")
            await self.send_response(update, f"❌ Error: {str(e)}")
    
    async def analyze_now_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Perform immediate analysis of all pairs"""
        user_id = update.effective_user.id
        
        if not self.is_authorized(user_id):
            await self.send_response(update, "❌ Unauthorized access.")
            return
        
        try:
            if not self.trader:
                await self.send_response(update, "❌ Signal scanner not initialized. Please start the scanner first.")
                return
            
            await self.send_response(update, "🔍 **Analyzing markets now...**\n\nScanning all pairs for immediate opportunities...")
            
            # Perform immediate analysis
            signals_found = await self.trader.analyze_all_pairs_now()
            
            if signals_found and len(signals_found) > 0:
                summary = f"✅ **Analysis Complete!**\n\n🎯 Found {len(signals_found)} trading opportunities.\n\nSignals have been sent!"
            else:
                summary = "✅ **Analysis Complete!**\n\n📊 No strong signals found at this time.\n\nMarket conditions may not meet our strict criteria (70% confidence minimum)."
            
            await self.send_response(update, summary)
            
        except Exception as e:
            logger.error(f"Error in immediate analysis: {e}")
            await self.send_response(update, f"❌ Analysis failed: {str(e)}")
    
    async def button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle button callbacks"""
        query = update.callback_query
        await query.answer()
        
        user_id = query.from_user.id
        if not self.is_authorized(user_id):
            await query.edit_message_text("❌ Unauthorized access.")
            return
        
        data = query.data
        
        if data == 'start':
            await self.start_command(update, context)
        elif data == 'start_scanner':
            await self.start_scanner_command(update, context)
        elif data == 'stop_scanner':
            await self.stop_scanner_command(update, context)
        elif data == 'status':
            await self.status_command(update, context)
        elif data == 'config':
            await self.config_command(update, context)
        elif data == 'toggle_signals':
            await self.toggle_signals_command(update, context)
        elif data == 'analyze_now':
            await self.analyze_now_command(update, context)
        elif data == 'help':
            await self.help_command(update, context)
    
    def get_status_text(self) -> str:
        """Get current system status text"""
        if not self.trader:
            return "❌ Signal scanner not initialized."
        
        status_emoji = "🟢" if self.trader.is_running else "🔴"
        mode_emoji = "🧪" if self.trader.demo_mode else "💸"
        
        # Signal status
        signal_status = ""
        if hasattr(self.trader, 'signal_enabled'):
            signal_emoji = "🔔" if self.trader.signal_enabled else "🔕"
            signal_status = f"{signal_emoji} Signals: {'On' if self.trader.signal_enabled else 'Off'}\n"
        
        status_text = f"""
📊 **Signal Scanner Status**

{status_emoji} Status: {'Running' if self.trader.is_running else 'Stopped'}
{mode_emoji} Account: {'Demo' if self.trader.demo_mode else 'Live'}
📡 Mode: Signals Only
{signal_status}
⏱️ Last Update: {datetime.now().strftime('%H:%M:%S')}
        """
        
        return status_text
    
    async def send_signal_notification(self, signal_data: Dict):
        """Send signal notification to authorized users"""
        try:
            pair = signal_data.get('pair', 'Unknown')
            action = signal_data.get('action', 'Unknown')
            confidence = signal_data.get('confidence', 0)
            entry_price = signal_data.get('entry_price', 0)
            stop_loss = signal_data.get('stop_loss', 0)
            take_profit = signal_data.get('take_profit', 0)
            timestamp = signal_data.get('timestamp', datetime.now())
            
            # Format signal message
            signal_text = f"""
🎯 **TRADING SIGNAL**

📈 **Pair:** {pair}
📊 **Action:** {action.upper()}
💯 **Confidence:** {confidence:.1f}%

📍 **Entry:** {entry_price:.5f}
🛑 **Stop Loss:** {stop_loss:.5f}
🎯 **Take Profit:** {take_profit:.5f}

⏰ **Time:** {timestamp.strftime('%H:%M:%S')}

*Signal-only mode - Execute manually*
            """
            
            # Send to all authorized users
            for user_id in self.authorized_users:
                try:
                    await self.application.bot.send_message(
                        chat_id=user_id,
                        text=signal_text,
                        parse_mode='Markdown'
                    )
                    logger.info(f"Signal sent to user {user_id}")
                except Exception as e:
                    logger.error(f"Failed to send signal to user {user_id}: {e}")
            
        except Exception as e:
            logger.error(f"Error sending signal notification: {e}")
    
    def run(self):
        """Run the Telegram bot"""
        try:
            # Create application
            self.application = Application.builder().token(self.bot_token).build()
            
            # Add handlers
            self.application.add_handler(CommandHandler("start", self.start_command))
            self.application.add_handler(CommandHandler("status", self.status_command))
            self.application.add_handler(CommandHandler("help", self.help_command))
            self.application.add_handler(CallbackQueryHandler(self.button_handler))
            
            logger.info("🤖 Telegram bot started!")
            logger.info("💡 Send /start to begin")
            
            # Run the bot with proper initialization
            self.application.run_polling(
                drop_pending_updates=True,
                allowed_updates=Update.ALL_TYPES
            )
            
        except Exception as e:
            logger.error(f"Critical error in bot: {e}")
            raise

if __name__ == "__main__":
    bot = TelegramTradingBot()
    bot.run()
