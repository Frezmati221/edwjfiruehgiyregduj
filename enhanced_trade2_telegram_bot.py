#!/usr/bin/env python3
"""
Enhanced Trade-2 Telegram Bot with Alerts and Advanced Features
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
import schedule
import threading
import time

from compatible_trade2_predictor import CompatibleSupervisedForexPredictor

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('logs/enhanced_telegram_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Alert:
    user_id: int
    pair: str
    condition: str  # 'above', 'below', 'confidence_above'
    value: float
    active: bool = True
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class UserPreferences:
    user_id: int
    confidence: float = 0.5
    notifications: bool = True
    favorite_pairs: List[str] = None
    alert_sound: bool = True
    daily_summary: bool = False
    timezone: str = "UTC"
    
    def __post_init__(self):
        if self.favorite_pairs is None:
            self.favorite_pairs = ['EURUSD', 'GBPUSD']

class DatabaseManager:
    def __init__(self, db_path: str = "bot_data.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # User preferences table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id INTEGER PRIMARY KEY,
                confidence REAL DEFAULT 0.5,
                notifications BOOLEAN DEFAULT 1,
                favorite_pairs TEXT DEFAULT '["EURUSD","GBPUSD"]',
                alert_sound BOOLEAN DEFAULT 1,
                daily_summary BOOLEAN DEFAULT 0,
                timezone TEXT DEFAULT 'UTC'
            )
        ''')
        
        # Alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                pair TEXT,
                condition TEXT,
                value REAL,
                active BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Usage stats table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS usage_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                command TEXT,
                pair TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_user_preferences(self, prefs: UserPreferences):
        """Save user preferences"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO user_preferences 
            (user_id, confidence, notifications, favorite_pairs, alert_sound, daily_summary, timezone)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            prefs.user_id, prefs.confidence, prefs.notifications,
            json.dumps(prefs.favorite_pairs), prefs.alert_sound,
            prefs.daily_summary, prefs.timezone
        ))
        
        conn.commit()
        conn.close()
    
    def load_user_preferences(self, user_id: int) -> UserPreferences:
        """Load user preferences"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM user_preferences WHERE user_id = ?', (user_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return UserPreferences(
                user_id=row[0],
                confidence=row[1],
                notifications=bool(row[2]),
                favorite_pairs=json.loads(row[3]),
                alert_sound=bool(row[4]),
                daily_summary=bool(row[5]),
                timezone=row[6]
            )
        else:
            # Return default preferences
            return UserPreferences(user_id=user_id)
    
    def save_alert(self, alert: Alert):
        """Save alert"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO alerts (user_id, pair, condition, value, active, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (alert.user_id, alert.pair, alert.condition, alert.value, alert.active, alert.created_at))
        
        conn.commit()
        conn.close()
    
    def get_user_alerts(self, user_id: int) -> List[Alert]:
        """Get active alerts for user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT user_id, pair, condition, value, active, created_at 
            FROM alerts WHERE user_id = ? AND active = 1
        ''', (user_id,))
        
        alerts = []
        for row in cursor.fetchall():
            alerts.append(Alert(
                user_id=row[0],
                pair=row[1],
                condition=row[2],
                value=row[3],
                active=bool(row[4]),
                created_at=datetime.fromisoformat(row[5])
            ))
        
        conn.close()
        return alerts
    
    def log_usage(self, user_id: int, command: str, pair: str = None):
        """Log command usage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO usage_stats (user_id, command, pair)
            VALUES (?, ?, ?)
        ''', (user_id, command, pair))
        
        conn.commit()
        conn.close()

class EnhancedTrade2TelegramBot:
    def __init__(self, bot_token: str, model_path: str, config_path: str = "bot_config.json"):
        self.bot_token = bot_token
        self.model_path = model_path
        self.config_path = config_path
        self.config = self.load_config()
        
        # Initialize components
        self.predictor = None
        self.app = None
        self.db = DatabaseManager()
        
        # Cache for predictions
        self.prediction_cache = {}
        self.cache_duration = timedelta(minutes=self.config.get('trading', {}).get('cache_duration_minutes', 5))
        
        # Forex pairs from config
        self.forex_pairs = self.config.get('market_data', {}).get('forex_pairs', {})
        
        # Load model
        self.load_model()
        
        # Start background tasks
        self.start_background_tasks()
    
    def load_config(self) -> dict:
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_path} not found, using defaults")
            return {}
    
    def load_model(self):
        """Load the trained model"""
        try:
            self.predictor = CompatibleSupervisedForexPredictor()
            self.predictor.load_model(self.model_path)
            logger.info(f"✅ Enhanced model loaded: {self.model_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def start_background_tasks(self):
        """Start background monitoring tasks"""
        def run_scheduler():
            # Schedule alert checking every minute
            schedule.every(1).minutes.do(self.check_alerts)
            
            # Schedule daily summaries at 8 AM UTC
            schedule.every().day.at("08:00").do(self.send_daily_summaries)
            
            # Schedule cache cleanup every hour
            schedule.every().hour.do(self.cleanup_cache)
            
            while True:
                schedule.run_pending()
                time.sleep(60)
        
        # Run scheduler in background thread
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        logger.info("🕐 Background tasks started")
    
    async def check_alerts(self):
        """Check all active alerts and send notifications"""
        try:
            # Get all users with alerts
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT DISTINCT user_id FROM alerts WHERE active = 1')
            user_ids = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            for user_id in user_ids:
                alerts = self.db.get_user_alerts(user_id)
                for alert in alerts:
                    await self.check_single_alert(alert)
                    
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
    
    async def check_single_alert(self, alert: Alert):
        """Check a single alert and notify if triggered"""
        try:
            # Get current market data
            symbol = self.forex_pairs.get(alert.pair)
            if not symbol:
                return
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval="1h")
            
            if data.empty:
                return
            
            current_price = data['Close'].iloc[-1]
            
            # Check alert condition
            triggered = False
            if alert.condition == 'above' and current_price > alert.value:
                triggered = True
            elif alert.condition == 'below' and current_price < alert.value:
                triggered = True
            elif alert.condition == 'confidence_above':
                # Get prediction and check confidence
                data.columns = [col.lower() for col in data.columns]
                prediction = self.predictor.predict(data, min_confidence=0.3)
                if prediction['confidence'] > alert.value:
                    triggered = True
            
            if triggered:
                await self.send_alert_notification(alert, current_price)
                # Deactivate alert after triggering
                self.deactivate_alert(alert)
                
        except Exception as e:
            logger.error(f"Error checking alert for {alert.pair}: {e}")
    
    async def send_alert_notification(self, alert: Alert, current_price: float):
        """Send alert notification to user"""
        try:
            if alert.condition == 'above':
                message = f"🚨 **ALERT TRIGGERED**\n\n{alert.pair} is now **ABOVE** {alert.value:.5f}\nCurrent: {current_price:.5f}"
            elif alert.condition == 'below':
                message = f"🚨 **ALERT TRIGGERED**\n\n{alert.pair} is now **BELOW** {alert.value:.5f}\nCurrent: {current_price:.5f}"
            elif alert.condition == 'confidence_above':
                message = f"🚨 **CONFIDENCE ALERT**\n\n{alert.pair} prediction confidence is above {alert.value:.0%}"
            
            await self.app.bot.send_message(
                chat_id=alert.user_id,
                text=message,
                parse_mode='Markdown'
            )
            
        except Exception as e:
            logger.error(f"Error sending alert notification: {e}")
    
    def deactivate_alert(self, alert: Alert):
        """Deactivate an alert"""
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE alerts SET active = 0 
            WHERE user_id = ? AND pair = ? AND condition = ? AND value = ?
        ''', (alert.user_id, alert.pair, alert.condition, alert.value))
        conn.commit()
        conn.close()
    
    async def send_daily_summaries(self):
        """Send daily summaries to users who opted in"""
        try:
            # Get users who want daily summaries
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT user_id FROM user_preferences WHERE daily_summary = 1')
            user_ids = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            for user_id in user_ids:
                await self.send_user_daily_summary(user_id)
                
        except Exception as e:
            logger.error(f"Error sending daily summaries: {e}")
    
    async def send_user_daily_summary(self, user_id: int):
        """Send daily summary to a user"""
        try:
            prefs = self.db.load_user_preferences(user_id)
            
            # Generate summary for favorite pairs
            summary_text = "📊 **Daily Market Summary**\n\n"
            
            for pair in prefs.favorite_pairs[:3]:  # Limit to 3 pairs
                try:
                    prediction_data = await self.get_prediction(pair, prefs.confidence)
                    pred = prediction_data['prediction']
                    
                    action_emoji = "🟢" if pred['action'] == 'LONG' else "🔴" if pred['action'] == 'SHORT' else "🔄"
                    
                    summary_text += f"""
{action_emoji} **{pair}**: {pred['action']} ({pred['confidence']:.0%})
Price: {prediction_data['current_price']:.5f} ({prediction_data['price_change_pct']:+.2f}%)
                    """
                except:
                    continue
            
            summary_text += "\n🎯 Have a great trading day!"
            
            await self.app.bot.send_message(
                chat_id=user_id,
                text=summary_text,
                parse_mode='Markdown'
            )
            
        except Exception as e:
            logger.error(f"Error sending daily summary to {user_id}: {e}")
    
    def cleanup_cache(self):
        """Clean up old cached predictions"""
        current_time = datetime.now()
        expired_keys = [
            key for key, (data, timestamp) in self.prediction_cache.items()
            if current_time - timestamp > self.cache_duration
        ]
        
        for key in expired_keys:
            del self.prediction_cache[key]
        
        logger.info(f"🧹 Cache cleanup: removed {len(expired_keys)} expired entries")
    
    async def get_prediction(self, pair: str, confidence: float = 0.5) -> dict:
        """Get prediction with caching"""
        cache_key = f"{pair}_{confidence}"
        current_time = datetime.now()
        
        # Check cache first
        if cache_key in self.prediction_cache:
            data, timestamp = self.prediction_cache[cache_key]
            if current_time - timestamp < self.cache_duration:
                return data
        
        # Generate new prediction
        try:
            symbol = self.forex_pairs[pair]
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="1mo", interval="1h")
            
            if df.empty:
                raise ValueError(f"No data available for {pair}")
            
            df.columns = [col.lower() for col in df.columns]
            prediction = self.predictor.predict(df, min_confidence=confidence)
            
            current_price = df['close'].iloc[-1]
            price_change = df['close'].iloc[-1] - df['close'].iloc[-2]
            price_change_pct = (price_change / df['close'].iloc[-2]) * 100
            
            result = {
                'pair': pair,
                'current_price': current_price,
                'price_change': price_change,
                'price_change_pct': price_change_pct,
                'prediction': prediction,
                'timestamp': current_time,
                'data_points': len(df)
            }
            
            # Cache the result
            self.prediction_cache[cache_key] = (result, current_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting prediction for {pair}: {e}")
            raise
    
    # Enhanced command handlers
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Enhanced start command with user registration"""
        user_id = update.effective_user.id
        username = update.effective_user.username or "Unknown"
        
        # Log usage
        self.db.log_usage(user_id, 'start')
        
        # Load or create user preferences
        prefs = self.db.load_user_preferences(user_id)
        self.db.save_user_preferences(prefs)  # Ensure user is in database
        
        welcome_message = f"""
🎯 **Enhanced Trade-2 Forex Bot** 

Welcome {username}! Your AI-powered trading assistant is ready.

**🚀 New Features:**
🔔 **Smart Alerts** - Get notified when conditions are met
📊 **Daily Summaries** - Morning market overview
⭐ **Favorites** - Quick access to your preferred pairs
🎯 **Custom Confidence** - Personalized prediction thresholds

**📋 Commands:**
/predict [PAIR] - Get AI prediction
/alerts - Manage price & confidence alerts  
/favorites - Your favorite pairs
/analysis [PAIR] - Deep technical analysis
/summary - Today's market overview
/settings - Customize your experience
        """
        
        # Enhanced keyboard with new features
        keyboard = [
            [
                InlineKeyboardButton("💶 EURUSD", callback_data="predict_EURUSD"),
                InlineKeyboardButton("💷 GBPUSD", callback_data="predict_GBPUSD")
            ],
            [
                InlineKeyboardButton("⭐ Favorites", callback_data="favorites"),
                InlineKeyboardButton("🔔 Alerts", callback_data="alerts_menu")
            ],
            [
                InlineKeyboardButton("📊 Summary", callback_data="daily_summary"),
                InlineKeyboardButton("⚙️ Settings", callback_data="settings")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            welcome_message,
            parse_mode='Markdown',
            reply_markup=reply_markup
        )
    
    async def alerts_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Alerts management command"""
        user_id = update.effective_user.id
        self.db.log_usage(user_id, 'alerts')
        
        if context.args:
            # Parse alert creation: /alerts EURUSD above 1.09000
            if len(context.args) >= 3:
                pair = context.args[0].upper()
                condition = context.args[1].lower()
                try:
                    value = float(context.args[2])
                    
                    if pair in self.forex_pairs and condition in ['above', 'below']:
                        alert = Alert(user_id, pair, condition, value)
                        self.db.save_alert(alert)
                        
                        await update.message.reply_text(
                            f"✅ Alert created!\n\n🔔 {pair} {condition} {value:.5f}",
                            parse_mode='Markdown'
                        )
                    else:
                        await update.message.reply_text(
                            "❌ Invalid alert format\nUse: /alerts PAIR above/below PRICE"
                        )
                except ValueError:
                    await update.message.reply_text("❌ Invalid price value")
            else:
                await update.message.reply_text(
                    "❌ Usage: /alerts PAIR above/below PRICE\nExample: /alerts EURUSD above 1.09000"
                )
        else:
            # Show alerts menu
            await self.show_alerts_menu(update)
    
    async def show_alerts_menu(self, update: Update):
        """Show alerts management menu"""
        user_id = update.effective_user.id
        alerts = self.db.get_user_alerts(user_id)
        
        message = "🔔 **Alert Management**\n\n"
        
        if alerts:
            message += "**Active Alerts:**\n"
            for i, alert in enumerate(alerts[:5], 1):  # Show max 5
                message += f"{i}. {alert.pair} {alert.condition} {alert.value:.5f}\n"
            
            if len(alerts) > 5:
                message += f"... and {len(alerts) - 5} more\n"
        else:
            message += "No active alerts\n"
        
        message += "\n**Create New Alert:**\nUse: `/alerts PAIR above/below PRICE`\nExample: `/alerts EURUSD above 1.09000`"
        
        keyboard = [
            [
                InlineKeyboardButton("➕ Quick Alert", callback_data="quick_alert"),
                InlineKeyboardButton("🗑️ Clear All", callback_data="clear_alerts")
            ],
            [
                InlineKeyboardButton("📊 Confidence Alert", callback_data="confidence_alert"),
                InlineKeyboardButton("🔙 Back", callback_data="main_menu")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        if update.callback_query:
            await update.callback_query.edit_message_text(
                message, parse_mode='Markdown', reply_markup=reply_markup
            )
        else:
            await update.message.reply_text(
                message, parse_mode='Markdown', reply_markup=reply_markup
            )
    
    async def favorites_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Favorites management"""
        user_id = update.effective_user.id
        self.db.log_usage(user_id, 'favorites')
        
        prefs = self.db.load_user_preferences(user_id)
        
        message = "⭐ **Your Favorite Pairs**\n\n"
        
        keyboard = []
        for pair in prefs.favorite_pairs:
            keyboard.append([
                InlineKeyboardButton(f"📊 {pair}", callback_data=f"predict_{pair}"),
                InlineKeyboardButton("📈 Analysis", callback_data=f"analysis_{pair}")
            ])
        
        keyboard.extend([
            [
                InlineKeyboardButton("➕ Add Favorite", callback_data="add_favorite"),
                InlineKeyboardButton("➖ Remove", callback_data="remove_favorite")
            ],
            [
                InlineKeyboardButton("🔙 Back", callback_data="main_menu")
            ]
        ])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            message, parse_mode='Markdown', reply_markup=reply_markup
        )
    
    async def summary_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Daily summary command"""
        user_id = update.effective_user.id
        self.db.log_usage(user_id, 'summary')
        
        await self.send_user_daily_summary(user_id)
    
    # Enhanced callback handlers
    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Enhanced button callback handler"""
        query = update.callback_query
        await query.answer()
        
        data = query.data
        user_id = update.effective_user.id
        
        if data.startswith('predict_'):
            pair = data.replace('predict_', '')
            await self.handle_pair_prediction(query, pair, user_id)
            
        elif data.startswith('analysis_'):
            pair = data.replace('analysis_', '')
            await self.handle_detailed_analysis(query, pair, user_id)
            
        elif data == 'alerts_menu':
            await self.show_alerts_menu(query)
            
        elif data == 'favorites':
            await self.favorites_from_callback(query, user_id)
            
        elif data == 'daily_summary':
            await query.edit_message_text("📊 Generating your daily summary...")
            await self.send_user_daily_summary(user_id)
            
        elif data == 'settings':
            await self.enhanced_settings_menu(query, user_id)
            
        elif data == 'main_menu':
            await self.show_main_menu(query)
        
        # Settings callbacks
        elif data.startswith('set_conf_'):
            conf = float(data.replace('set_conf_', ''))
            await self.update_user_confidence(query, user_id, conf)
            
        elif data == 'toggle_notifications':
            await self.toggle_user_notifications(query, user_id)
            
        elif data == 'toggle_daily_summary':
            await self.toggle_daily_summary(query, user_id)
    
    async def favorites_from_callback(self, query, user_id: int):
        """Show favorites from callback"""
        prefs = self.db.load_user_preferences(user_id)
        
        message = "⭐ **Your Favorite Pairs**\n\nClick for instant prediction:"
        
        keyboard = []
        for pair in prefs.favorite_pairs:
            keyboard.append([
                InlineKeyboardButton(f"📊 {pair}", callback_data=f"predict_{pair}")
            ])
        
        keyboard.append([
            InlineKeyboardButton("⚙️ Manage", callback_data="manage_favorites"),
            InlineKeyboardButton("🔙 Back", callback_data="main_menu")
        ])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            message, parse_mode='Markdown', reply_markup=reply_markup
        )
    
    async def enhanced_settings_menu(self, query, user_id: int):
        """Enhanced settings menu"""
        prefs = self.db.load_user_preferences(user_id)
        
        message = f"""
⚙️ **Enhanced Settings**

**Current Configuration:**
🎯 Confidence: {prefs.confidence:.0%}
🔔 Notifications: {'On' if prefs.notifications else 'Off'}
📊 Daily Summary: {'On' if prefs.daily_summary else 'Off'}
⭐ Favorites: {len(prefs.favorite_pairs)} pairs
🌍 Timezone: {prefs.timezone}

**Customize your experience:**
        """
        
        keyboard = [
            [
                InlineKeyboardButton(f"🎯 Confidence ({prefs.confidence:.0%})", callback_data="conf_settings"),
                InlineKeyboardButton("🔔 Notifications", callback_data="toggle_notifications")
            ],
            [
                InlineKeyboardButton("📊 Daily Summary", callback_data="toggle_daily_summary"),
                InlineKeyboardButton("⭐ Favorites", callback_data="manage_favorites")
            ],
            [
                InlineKeyboardButton("📈 Usage Stats", callback_data="usage_stats"),
                InlineKeyboardButton("🌍 Timezone", callback_data="timezone_settings")
            ],
            [
                InlineKeyboardButton("🔙 Back to Main", callback_data="main_menu")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            message, parse_mode='Markdown', reply_markup=reply_markup
        )
    
    async def update_user_confidence(self, query, user_id: int, confidence: float):
        """Update user confidence setting"""
        prefs = self.db.load_user_preferences(user_id)
        prefs.confidence = confidence
        self.db.save_user_preferences(prefs)
        
        await query.edit_message_text(
            f"✅ Confidence updated to {confidence:.0%}\n\nThis affects all your predictions.",
            parse_mode='Markdown'
        )
    
    async def toggle_user_notifications(self, query, user_id: int):
        """Toggle user notifications"""
        prefs = self.db.load_user_preferences(user_id)
        prefs.notifications = not prefs.notifications
        self.db.save_user_preferences(prefs)
        
        status = "enabled" if prefs.notifications else "disabled"
        await query.edit_message_text(
            f"✅ Notifications {status}\n\nYou will {'receive' if prefs.notifications else 'not receive'} alert notifications.",
            parse_mode='Markdown'
        )
    
    async def toggle_daily_summary(self, query, user_id: int):
        """Toggle daily summary"""
        prefs = self.db.load_user_preferences(user_id)
        prefs.daily_summary = not prefs.daily_summary
        self.db.save_user_preferences(prefs)
        
        status = "enabled" if prefs.daily_summary else "disabled"
        await query.edit_message_text(
            f"✅ Daily summary {status}\n\nYou will {'receive' if prefs.daily_summary else 'not receive'} morning market summaries.",
            parse_mode='Markdown'
        )
    
    # Copy other methods from basic bot...
    async def handle_pair_prediction(self, query, pair: str, user_id: int):
        """Handle pair prediction from button"""
        prefs = self.db.load_user_preferences(user_id)
        
        await query.edit_message_text(f"🔄 Analyzing {pair}...")
        
        try:
            prediction_data = await self.get_prediction(pair, prefs.confidence)
            message = self.format_prediction_message(pair, prediction_data)
            
            keyboard = [
                [
                    InlineKeyboardButton("🔄 Refresh", callback_data=f"predict_{pair}"),
                    InlineKeyboardButton("📈 Analysis", callback_data=f"analysis_{pair}")
                ],
                [
                    InlineKeyboardButton("🔔 Set Alert", callback_data=f"alert_{pair}"),
                    InlineKeyboardButton("⭐ Add Favorite", callback_data=f"fav_add_{pair}")
                ],
                [
                    InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                message, parse_mode='Markdown', reply_markup=reply_markup
            )
            
        except Exception as e:
            await query.edit_message_text(f"❌ Error: {str(e)}")
    
    def format_prediction_message(self, pair: str, data: dict) -> str:
        """Enhanced prediction message formatting"""
        pred = data['prediction']
        
        price_emoji = "📈" if data['price_change'] > 0 else "📉" if data['price_change'] < 0 else "➡️"
        
        # Enhanced message with more details
        message = f"""
🎯 **{pair} AI Analysis**

💹 **Market Status:**
{price_emoji} Price: `{data['current_price']:.5f}`
📊 Change: `{data['price_change']:+.5f}` ({data['price_change_pct']:+.2f}%)
🕐 Updated: {data['timestamp'].strftime('%H:%M:%S UTC')}
📈 Data Points: {data['data_points']}

🤖 **AI Prediction:**
        """
        
        action = pred['action'].upper()
        confidence = pred['confidence']
        
        if action == 'HOLD':
            message += f"""
🔄 **Action: HOLD**
📊 Confidence: {confidence:.1%}
💭 No clear trading signal detected
        """
        else:
            action_emoji = "🟢 📈" if action == 'LONG' else "🔴 📉"
            message += f"""
{action_emoji} **Action: {action}**
🎯 Confidence: {confidence:.1%}
            """
            
            if 'stop_loss' in pred and 'take_profit' in pred:
                sl = pred['stop_loss']
                tp = pred['take_profit']
                rr = pred.get('risk_reward_ratio', 0)
                risk_pips = pred.get('risk_pips', 0)
                reward_pips = pred.get('reward_pips', 0)
                
                message += f"""
🛡️ Stop Loss: `{sl:.5f}` ({risk_pips:.1f} pips)
🎯 Take Profit: `{tp:.5f}` ({reward_pips:.1f} pips)
📊 Risk:Reward: `1:{rr:.2f}`
                """
        
        # Enhanced confidence interpretation
        if confidence < 0.3:
            conf_text = "🔴 Very Low - Avoid trading"
        elif confidence < 0.5:
            conf_text = "🟡 Low - Monitor only"
        elif confidence < 0.7:
            conf_text = "🟠 Medium - Consider carefully"
        elif confidence < 0.8:
            conf_text = "🟢 High - Strong signal"
        else:
            conf_text = "💚 Very High - Excellent signal"
        
        message += f"""

💡 **Signal Strength:**
{conf_text}

⚠️ *AI analysis for educational purposes only*
        """
        
        return message
    
    async def show_main_menu(self, query):
        """Enhanced main menu"""
        keyboard = [
            [
                InlineKeyboardButton("💶 EURUSD", callback_data="predict_EURUSD"),
                InlineKeyboardButton("💷 GBPUSD", callback_data="predict_GBPUSD")
            ],
            [
                InlineKeyboardButton("⭐ Favorites", callback_data="favorites"),
                InlineKeyboardButton("🔔 Alerts", callback_data="alerts_menu")
            ],
            [
                InlineKeyboardButton("📊 Summary", callback_data="daily_summary"),
                InlineKeyboardButton("⚙️ Settings", callback_data="settings")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            "🎯 **Enhanced Trade-2 Bot**\n\nChoose an option:",
            parse_mode='Markdown',
            reply_markup=reply_markup
        )
    
    def run(self):
        """Start the enhanced bot"""
        # Create application
        self.app = Application.builder().token(self.bot_token).build()
        
        # Add handlers
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("alerts", self.alerts_command))
        self.app.add_handler(CommandHandler("favorites", self.favorites_command))
        self.app.add_handler(CommandHandler("summary", self.summary_command))
        self.app.add_handler(CallbackQueryHandler(self.button_callback))
        
        # Start bot
        logger.info("🚀 Starting Enhanced Trade-2 Telegram Bot...")
        self.app.run_polling()

def main():
    """Main function for enhanced bot"""
    BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    MODEL_PATH = "best_model.pth"
    CONFIG_PATH = "bot_config.json"
    
    if not BOT_TOKEN:
        print("❌ Please set TELEGRAM_BOT_TOKEN environment variable")
        return
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        return
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Create and run enhanced bot
    bot = EnhancedTrade2TelegramBot(BOT_TOKEN, MODEL_PATH, CONFIG_PATH)
    bot.run()

if __name__ == "__main__":
    main()
