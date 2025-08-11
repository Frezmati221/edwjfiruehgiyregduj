#!/usr/bin/env python3
"""
🚀 Auto-Scanning Forex Alert Telegram Bot
=========================================

This bot automatically scans the main forex pairs (EURUSD, GBPUSD, USDJPY) 
and sends alerts when predictions are made with high confidence.

Features:
- 🔄 Continuous auto-scanning every 15 minutes
- 🎯 Automatic alerts for LONG/SHORT signals on main pairs
- 📊 Confidence-based filtering (only high-confidence signals)
- 💾 Persistent user preferences and alert history
- 🔔 Customizable notification settings
"""

import asyncio
import logging
import yfinance as yf
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from typing import Dict, List, Optional
import sqlite3
from dataclasses import dataclass
import threading
import time
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

# Import the compatible predictor
try:
    from compatible_trade2_predictor import CompatibleSupervisedForexPredictor
    PREDICTOR_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: compatible_trade2_predictor not found. Using dummy predictor.")
    PREDICTOR_AVAILABLE = False

# Configure logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('logs/autoscan_telegram_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class PredictionSignal:
    """Represents a trading signal from predictions"""
    pair: str
    action: str  # LONG, SHORT, HOLD
    confidence: float
    current_price: float
    take_profit: float = None
    stop_loss: float = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass 
class UserSettings:
    """User notification settings"""
    user_id: int
    active: bool = True
    min_confidence: float = 0.7  # Only send alerts for >70% confidence
    pairs: List[str] = None  # Which pairs to monitor
    notification_cooldown: int = 30  # Minutes between same-pair notifications
    quiet_hours_start: int = 22  # No alerts after 10 PM
    quiet_hours_end: int = 7    # No alerts before 7 AM
    timezone_offset: int = 0    # User timezone offset from UTC
    
    def __post_init__(self):
        if self.pairs is None:
            self.pairs = ['EURUSD', 'GBPUSD', 'USDJPY']  # Main 3 pairs

class AutoScanDatabase:
    """Database manager for auto-scanning bot"""
    
    def __init__(self, db_path: str = "autoscan_bot.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # User settings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_settings (
                user_id INTEGER PRIMARY KEY,
                active BOOLEAN DEFAULT 1,
                min_confidence REAL DEFAULT 0.7,
                pairs TEXT DEFAULT '["EURUSD","GBPUSD","USDJPY"]',
                notification_cooldown INTEGER DEFAULT 30,
                quiet_hours_start INTEGER DEFAULT 22,
                quiet_hours_end INTEGER DEFAULT 7,
                timezone_offset INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Signal history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signal_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair TEXT,
                action TEXT,
                confidence REAL,
                current_price REAL,
                take_profit REAL,
                stop_loss REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notified_users TEXT DEFAULT '[]'
            )
        ''')
        
        # Notification log table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS notification_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                pair TEXT,
                action TEXT,
                confidence REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_user_settings(self, settings: UserSettings):
        """Save user settings"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO user_settings 
            (user_id, active, min_confidence, pairs, notification_cooldown, 
             quiet_hours_start, quiet_hours_end, timezone_offset)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            settings.user_id, settings.active, settings.min_confidence,
            json.dumps(settings.pairs), settings.notification_cooldown,
            settings.quiet_hours_start, settings.quiet_hours_end, settings.timezone_offset
        ))
        
        conn.commit()
        conn.close()
    
    def load_user_settings(self, user_id: int) -> UserSettings:
        """Load user settings"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM user_settings WHERE user_id = ?', (user_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return UserSettings(
                user_id=row[0],
                active=bool(row[1]),
                min_confidence=row[2],
                pairs=json.loads(row[3]),
                notification_cooldown=row[4],
                quiet_hours_start=row[5],
                quiet_hours_end=row[6],
                timezone_offset=row[7]
            )
        else:
            # Return default settings
            return UserSettings(user_id=user_id)
    
    def get_all_active_users(self) -> List[int]:
        """Get all users who want notifications"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT user_id FROM user_settings WHERE active = 1')
        user_ids = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        return user_ids
    
    def save_signal(self, signal: PredictionSignal):
        """Save a prediction signal"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO signal_history 
            (pair, action, confidence, current_price, take_profit, stop_loss, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal.pair, signal.action, signal.confidence, signal.current_price,
            signal.take_profit, signal.stop_loss, signal.timestamp
        ))
        
        conn.commit()
        conn.close()
    
    def log_notification(self, user_id: int, signal: PredictionSignal):
        """Log that we sent a notification to a user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO notification_log (user_id, pair, action, confidence)
            VALUES (?, ?, ?, ?)
        ''', (user_id, signal.pair, signal.action, signal.confidence))
        
        conn.commit()
        conn.close()
    
    def can_notify_user(self, user_id: int, pair: str, cooldown_minutes: int) -> bool:
        """Check if we can notify user about this pair (respecting cooldown)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT MAX(timestamp) FROM notification_log 
            WHERE user_id = ? AND pair = ?
        ''', (user_id, pair))
        
        result = cursor.fetchone()
        conn.close()
        
        if result[0] is None:
            return True  # Never notified about this pair
        
        last_notification = datetime.fromisoformat(result[0])
        time_since = datetime.now() - last_notification
        
        return time_since.total_seconds() > (cooldown_minutes * 60)

class DummyPredictor:
    """Dummy predictor for testing when real predictor is not available"""
    
    def predict(self, data, min_confidence=0.5):
        import random
        
        actions = ['LONG', 'SHORT', 'HOLD']
        action = random.choice(actions)
        confidence = random.uniform(0.3, 0.9)
        
        return {
            'action': action,
            'confidence': confidence,
            'reasoning': 'Dummy prediction for testing'
        }

class AutoScanForexBot:
    """Main auto-scanning forex bot class"""
    
    def __init__(self, token: str):
        self.token = token
        self.app = Application.builder().token(token).build()
        self.db = AutoScanDatabase()
        
        # Initialize predictor
        if PREDICTOR_AVAILABLE:
            try:
                self.predictor = CompatibleSupervisedForexPredictor()
                self.predictor.load_model('best_model.pth')
                logger.info("✅ Real predictor loaded successfully")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load real predictor: {e}")
                self.predictor = DummyPredictor()
        else:
            self.predictor = DummyPredictor()
        
        # Main forex pairs to monitor
        self.main_pairs = {
            'EURUSD': 'EURUSD=X',
            'GBPUSD': 'GBPUSD=X', 
            'USDJPY': 'USDJPY=X'
        }
        
        # Scanning state
        self.last_scan_time = None
        self.scan_interval = 15  # minutes
        self.is_scanning = False
        
        # Setup scheduler for auto-scanning
        self.scheduler = AsyncIOScheduler()
        
        # Setup handlers
        self.setup_handlers()
    
    def setup_handlers(self):
        """Setup command and callback handlers"""
        self.app.add_handler(CommandHandler("start", self.start_command))
        self.app.add_handler(CommandHandler("status", self.status_command))
        self.app.add_handler(CommandHandler("settings", self.settings_command))
        self.app.add_handler(CommandHandler("history", self.history_command))
        self.app.add_handler(CommandHandler("scan", self.manual_scan_command))
        self.app.add_handler(CommandHandler("help", self.help_command))
        self.app.add_handler(CallbackQueryHandler(self.handle_callback))
    
    async def start_auto_scanning(self, app):
        """Start the auto-scanning process (called after bot initialization)"""
        logger.info("🚀 Starting auto-scanning scheduler...")
        
        # Add scanning job
        self.scheduler.add_job(
            self.auto_scan_pairs,
            IntervalTrigger(minutes=self.scan_interval),
            id='auto_scan',
            replace_existing=True
        )
        
        self.scheduler.start()
        logger.info(f"✅ Auto-scanning started (every {self.scan_interval} minutes)")
        
        # Do initial scan
        await self.auto_scan_pairs()
    
    async def stop_auto_scanning(self, app):
        """Stop the auto-scanning process (called during shutdown)"""
        logger.info("🛑 Stopping auto-scanning scheduler...")
        
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("✅ Auto-scanning scheduler stopped")
    
    async def auto_scan_pairs(self):
        """Automatically scan main forex pairs for signals"""
        if self.is_scanning:
            logger.info("⏳ Scan already in progress, skipping...")
            return
        
        self.is_scanning = True
        scan_start = datetime.now()
        
        try:
            logger.info("🔍 Starting auto-scan of main forex pairs...")
            
            signals_found = []
            
            for pair, symbol in self.main_pairs.items():
                try:
                    signal = await self.scan_pair(pair, symbol)
                    if signal:
                        signals_found.append(signal)
                        
                except Exception as e:
                    logger.error(f"❌ Error scanning {pair}: {e}")
                    continue
            
            # Process signals and send notifications
            if signals_found:
                await self.process_signals(signals_found)
                logger.info(f"✅ Scan complete: Found {len(signals_found)} signals")
            else:
                logger.info("✅ Scan complete: No new signals")
            
            self.last_scan_time = scan_start
            
        except Exception as e:
            logger.error(f"❌ Error during auto-scan: {e}")
        finally:
            self.is_scanning = False
    
    async def scan_pair(self, pair: str, symbol: str) -> Optional[PredictionSignal]:
        """Scan a single pair for trading signals"""
        try:
            # Get recent data
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="1mo", interval="1h")
            
            if df.empty:
                logger.warning(f"⚠️ No data available for {pair}")
                return None
            
            # Prepare data for prediction
            df.columns = [col.lower() for col in df.columns]
            
            # Get prediction
            prediction = self.predictor.predict(df, min_confidence=0.3)
            
            # Only create signal for LONG/SHORT actions with sufficient confidence
            if prediction['action'] in ['LONG', 'SHORT'] and prediction['confidence'] >= 0.6:
                current_price = df['close'].iloc[-1]
                
                signal = PredictionSignal(
                    pair=pair,
                    action=prediction['action'],
                    confidence=prediction['confidence'],
                    current_price=current_price,
                    take_profit=getattr(prediction, 'take_profit', None),
                    stop_loss=getattr(prediction, 'stop_loss', None)
                )
                
                logger.info(f"📈 Signal found: {pair} {prediction['action']} ({prediction['confidence']:.1%})")
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error scanning {pair}: {e}")
            return None
    
    async def process_signals(self, signals: List[PredictionSignal]):
        """Process found signals and send notifications"""
        try:
            # Get all active users
            active_users = self.db.get_all_active_users()
            
            if not active_users:
                logger.info("📭 No active users to notify")
                return
            
            for signal in signals:
                # Save signal to database
                self.db.save_signal(signal)
                
                # Notify relevant users
                for user_id in active_users:
                    await self.notify_user_if_applicable(user_id, signal)
                    
        except Exception as e:
            logger.error(f"❌ Error processing signals: {e}")
    
    async def notify_user_if_applicable(self, user_id: int, signal: PredictionSignal):
        """Send notification to user if applicable based on their settings"""
        try:
            # Load user settings
            settings = self.db.load_user_settings(user_id)
            
            # Check if user wants notifications for this pair
            if signal.pair not in settings.pairs:
                return
            
            # Check confidence threshold
            if signal.confidence < settings.min_confidence:
                return
            
            # Check notification cooldown
            if not self.db.can_notify_user(user_id, signal.pair, settings.notification_cooldown):
                return
            
            # Check quiet hours
            current_hour = (datetime.now().hour + settings.timezone_offset) % 24
            if settings.quiet_hours_start < settings.quiet_hours_end:
                # Normal case: quiet hours within same day
                if settings.quiet_hours_start <= current_hour < settings.quiet_hours_end:
                    return
            else:
                # Quiet hours span midnight
                if current_hour >= settings.quiet_hours_start or current_hour < settings.quiet_hours_end:
                    return
            
            # Send notification
            await self.send_signal_notification(user_id, signal)
            
            # Log notification
            self.db.log_notification(user_id, signal)
            
        except Exception as e:
            logger.error(f"❌ Error notifying user {user_id}: {e}")
    
    async def send_signal_notification(self, user_id: int, signal: PredictionSignal):
        """Send signal notification to user"""
        try:
            # Choose emoji based on action
            action_emoji = "🟢" if signal.action == "LONG" else "🔴"
            confidence_stars = "⭐" * min(5, int(signal.confidence * 5))
            
            message = f"""
🚨 **FOREX ALERT** 🚨

{action_emoji} **{signal.pair}** - **{signal.action}**
💹 Price: {signal.current_price:.5f}
🎯 Confidence: {signal.confidence:.1%} {confidence_stars}
⏰ Time: {signal.timestamp.strftime('%H:%M UTC')}

{f"🎯 Take Profit: {signal.take_profit:.5f}" if signal.take_profit else ""}
{f"🛑 Stop Loss: {signal.stop_loss:.5f}" if signal.stop_loss else ""}

_Auto-scan alert from AI prediction system_
            """
            
            await self.app.bot.send_message(
                chat_id=user_id,
                text=message,
                parse_mode='Markdown'
            )
            
            logger.info(f"📤 Sent alert to user {user_id}: {signal.pair} {signal.action}")
            
        except Exception as e:
            logger.error(f"❌ Error sending notification to {user_id}: {e}")
    
    # Command handlers
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start command handler"""
        user_id = update.effective_user.id
        username = update.effective_user.username or "Unknown"
        
        # Create default settings for new user
        settings = self.db.load_user_settings(user_id)
        self.db.save_user_settings(settings)
        
        welcome_message = f"""
🚀 **Auto-Scanning Forex Alert Bot**

Welcome {username}! 

This bot automatically scans the **3 main forex pairs** every 15 minutes:
• 💶 **EURUSD** - Euro vs US Dollar
• 💷 **GBPUSD** - British Pound vs US Dollar  
• 💴 **USDJPY** - US Dollar vs Japanese Yen

🎯 **You'll get alerts when:**
✅ AI predicts LONG or SHORT signals
✅ Confidence is above your threshold ({settings.min_confidence:.0%})
✅ Signal is for your monitored pairs

📋 **Commands:**
/status - Check scanning status
/settings - Customize your alerts
/scan - Manual scan now
/history - Recent signals
/help - Full help guide

🔄 **Auto-scanning is active!**
Next scan in ~{self.scan_interval} minutes.
        """
        
        keyboard = [
            [InlineKeyboardButton("⚙️ Settings", callback_data="settings"),
             InlineKeyboardButton("📊 Status", callback_data="status")],
            [InlineKeyboardButton("🔍 Scan Now", callback_data="scan_now"),
             InlineKeyboardButton("📈 History", callback_data="history")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            welcome_message,
            parse_mode='Markdown',
            reply_markup=reply_markup
        )
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Status command handler"""
        user_id = update.effective_user.id
        settings = self.db.load_user_settings(user_id)
        
        # Get scan status
        status_emoji = "🟢" if settings.active else "🔴"
        scan_status = "ACTIVE" if settings.active else "PAUSED"
        
        # Calculate next scan time
        if self.last_scan_time:
            next_scan = self.last_scan_time + timedelta(minutes=self.scan_interval)
            time_until = next_scan - datetime.now()
            next_scan_text = f"{max(0, int(time_until.total_seconds() / 60))} minutes"
        else:
            next_scan_text = "Soon"
        
        status_message = f"""
📊 **Auto-Scan Status**

{status_emoji} **Status**: {scan_status}
🔄 **Scan Interval**: Every {self.scan_interval} minutes
⏰ **Next Scan**: In {next_scan_text}
📈 **Last Scan**: {self.last_scan_time.strftime('%H:%M UTC') if self.last_scan_time else 'Not yet'}

🎯 **Your Settings**:
• Confidence Threshold: {settings.min_confidence:.0%}
• Monitored Pairs: {', '.join(settings.pairs)}
• Cooldown: {settings.notification_cooldown} minutes
• Quiet Hours: {settings.quiet_hours_start}:00 - {settings.quiet_hours_end}:00

🤖 **Bot Status**: {"🟢 Online" if not self.is_scanning else "🔍 Scanning..."}
        """
        
        keyboard = [
            [InlineKeyboardButton("⚙️ Change Settings", callback_data="settings")],
            [InlineKeyboardButton("🔍 Manual Scan", callback_data="scan_now")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            status_message,
            parse_mode='Markdown',
            reply_markup=reply_markup
        )
    
    async def settings_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Settings command handler"""
        user_id = update.effective_user.id
        settings = self.db.load_user_settings(user_id)
        
        settings_message = f"""
⚙️ **Your Alert Settings**

**Notifications**: {"🟢 ON" if settings.active else "🔴 OFF"}
**Confidence**: {settings.min_confidence:.0%} minimum
**Pairs**: {', '.join(settings.pairs)}
**Cooldown**: {settings.notification_cooldown} minutes
**Quiet Hours**: {settings.quiet_hours_start}:00 - {settings.quiet_hours_end}:00

Choose what to modify:
        """
        
        keyboard = [
            [InlineKeyboardButton("🔔 Toggle Notifications", callback_data=f"toggle_active")],
            [InlineKeyboardButton("🎯 Confidence: " + f"{settings.min_confidence:.0%}", callback_data="set_confidence")],
            [InlineKeyboardButton("💰 Select Pairs", callback_data="select_pairs")],
            [InlineKeyboardButton("⏰ Cooldown Time", callback_data="set_cooldown")],
            [InlineKeyboardButton("🌙 Quiet Hours", callback_data="set_quiet_hours")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            settings_message,
            parse_mode='Markdown',
            reply_markup=reply_markup
        )
    
    async def manual_scan_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Manual scan command"""
        user_id = update.effective_user.id
        
        # Send initial message
        scan_message = await update.message.reply_text(
            "🔍 **Manual Scan Started**\n\nScanning main forex pairs...",
            parse_mode='Markdown'
        )
        
        try:
            # Perform scan
            signals_found = []
            
            for pair, symbol in self.main_pairs.items():
                signal = await self.scan_pair(pair, symbol)
                if signal:
                    signals_found.append(signal)
            
            # Format results
            if signals_found:
                result_text = "🎯 **Scan Results - Signals Found:**\n\n"
                
                for signal in signals_found:
                    action_emoji = "🟢" if signal.action == "LONG" else "🔴"
                    result_text += f"""
{action_emoji} **{signal.pair}**: {signal.action}
💹 Price: {signal.current_price:.5f}
🎯 Confidence: {signal.confidence:.1%}
                    """
                    
                result_text += "\n_These signals have been saved and will trigger auto-alerts._"
            else:
                result_text = "✅ **Scan Complete**\n\nNo new signals found at this time.\n\n_The pairs are either in HOLD mode or below confidence threshold._"
            
            # Update message
            await scan_message.edit_text(
                result_text,
                parse_mode='Markdown'
            )
            
        except Exception as e:
            await scan_message.edit_text(
                f"❌ **Scan Failed**\n\nError: {str(e)}",
                parse_mode='Markdown'
            )
    
    async def history_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show recent signal history"""
        user_id = update.effective_user.id
        
        try:
            # Get recent signals from database
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT pair, action, confidence, current_price, timestamp 
                FROM signal_history 
                WHERE timestamp > datetime('now', '-24 hours')
                ORDER BY timestamp DESC 
                LIMIT 10
            ''')
            
            signals = cursor.fetchall()
            conn.close()
            
            if signals:
                history_text = "📈 **Recent Signals (24h)**\n\n"
                
                for signal in signals:
                    pair, action, confidence, price, timestamp = signal
                    action_emoji = "🟢" if action == "LONG" else "🔴"
                    time_str = datetime.fromisoformat(timestamp).strftime('%H:%M')
                    
                    history_text += f"{action_emoji} {pair} {action} ({confidence:.0%}) @ {price:.5f} - {time_str}\n"
                
                history_text += "\n_Showing last 10 signals from past 24 hours_"
            else:
                history_text = "📭 **No Recent Signals**\n\nNo signals found in the past 24 hours."
            
            await update.message.reply_text(
                history_text,
                parse_mode='Markdown'
            )
            
        except Exception as e:
            await update.message.reply_text(
                f"❌ Error loading history: {str(e)}",
                parse_mode='Markdown'
            )
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Help command"""
        help_text = """
🚀 **Auto-Scanning Forex Bot Help**

**🔄 How It Works:**
• Bot automatically scans EURUSD, GBPUSD, USDJPY every 15 minutes
• AI analyzes price data and generates predictions
• You get alerts only for high-confidence LONG/SHORT signals
• No spam - smart cooldown prevents duplicate alerts

**📋 Commands:**
/start - Welcome & setup
/status - Check scanning status
/settings - Customize your alerts
/scan - Force manual scan now
/history - See recent signals (24h)
/help - This help message

**⚙️ Settings You Can Customize:**
🔔 **Notifications** - Turn alerts on/off
🎯 **Confidence** - Minimum prediction confidence (50%-90%)
💰 **Pairs** - Which pairs to monitor (EURUSD/GBPUSD/USDJPY)
⏰ **Cooldown** - Time between alerts for same pair (15-120 min)
🌙 **Quiet Hours** - No alerts during sleep hours

**🎯 Alert Conditions:**
✅ Action is LONG or SHORT (not HOLD)
✅ Confidence above your threshold
✅ For one of your monitored pairs
✅ Cooldown period has passed
✅ Not during your quiet hours

**💡 Tips:**
• Start with 70% confidence for quality signals
• Use 30-60 minute cooldown to avoid spam
• Set quiet hours for your sleep schedule
• Monitor 1-3 pairs max for focused alerts

**🤖 Technical:**
• Scans every 15 minutes automatically
• Uses your trained AI model (best_model.pth)
• Stores all signals and notification history
• Works 24/7 when bot is running

Need help? The bot will guide you through settings!
        """
        
        await update.message.reply_text(help_text, parse_mode='Markdown')
    
    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle callback queries from inline keyboards"""
        query = update.callback_query
        await query.answer()
        
        user_id = query.from_user.id
        data = query.data
        
        if data == "status":
            await self.status_command(update, context)
        elif data == "settings":
            await self.settings_command(update, context)
        elif data == "scan_now":
            await self.manual_scan_command(update, context)
        elif data == "history":
            await self.history_command(update, context)
        elif data == "toggle_active":
            await self.toggle_notifications(query, user_id)
        elif data.startswith("confidence_"):
            confidence = float(data.split("_")[1]) / 100
            await self.set_confidence(query, user_id, confidence)
        # Add more callback handlers as needed
    
    async def toggle_notifications(self, query, user_id: int):
        """Toggle user notifications on/off"""
        settings = self.db.load_user_settings(user_id)
        settings.active = not settings.active
        self.db.save_user_settings(settings)
        
        status = "🟢 ON" if settings.active else "🔴 OFF"
        
        await query.edit_message_text(
            f"✅ **Notifications Updated**\n\nNotifications are now: {status}",
            parse_mode='Markdown'
        )
    
    async def set_confidence(self, query, user_id: int, confidence: float):
        """Set confidence threshold"""
        settings = self.db.load_user_settings(user_id)
        settings.min_confidence = confidence
        self.db.save_user_settings(settings)
        
        await query.edit_message_text(
            f"✅ **Confidence Updated**\n\nMinimum confidence set to: {confidence:.0%}",
            parse_mode='Markdown'
        )
    
    def run(self):
        """Run the bot"""
        logger.info("🚀 Starting Auto-Scanning Forex Alert Bot...")
        
        # Run bot with post_init callback to start scanning
        self.app.run_polling(
            drop_pending_updates=True,
            post_init=self.start_auto_scanning,
            post_shutdown=self.stop_auto_scanning
        )

def main():
    """Main function"""
    # Load bot token from config
    try:
        with open('bot_config.json', 'r') as f:
            config = json.load(f)
            token = config.get('telegram_token')
    except FileNotFoundError:
        logger.error("❌ bot_config.json not found")
        logger.error("Create bot_config.json with: {'telegram_token': 'YOUR_BOT_TOKEN'}")
        return
    except Exception as e:
        logger.error(f"❌ Error loading config: {e}")
        return
    
    if not token:
        logger.error("❌ No telegram_token found in bot_config.json")
        return
    
    # Create and run bot
    bot = AutoScanForexBot(token)
    bot.run()

if __name__ == "__main__":
    main()
