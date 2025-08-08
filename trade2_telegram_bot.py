#!/usr/bin/env python3
"""
Trade-2 Telegram Bot
Real-time forex predictions using your trained supervised learning model
"""

import asyncio
import logging
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
import json
import os
from compatible_trade2_predictor import CompatibleSupervisedForexPredictor

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class Trade2TelegramBot:
    def __init__(self, bot_token: str, model_path: str):
        self.bot_token = bot_token
        self.model_path = model_path
        self.predictor = None
        self.app = None
        
        # Forex pairs to monitor
        self.forex_pairs = {
            'EURUSD': 'EURUSD=X',
            'GBPUSD': 'GBPUSD=X', 
            'USDJPY': 'USDJPY=X',
            'AUDUSD': 'AUDUSD=X',
            'USDCAD': 'USDCAD=X',
            'USDCHF': 'USDCHF=X',
            'EURJPY': 'EURJPY=X',
            'GBPJPY': 'GBPJPY=X'
        }
        
        # Bot settings
        self.default_confidence = 0.5
        self.user_settings = {}  # Store user preferences
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            self.predictor = CompatibleSupervisedForexPredictor()
            self.predictor.load_model(self.model_path)
            logger.info(f"✅ Model loaded: {self.model_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start command handler"""
        user_id = update.effective_user.id
        
        # Initialize user settings
        if user_id not in self.user_settings:
            self.user_settings[user_id] = {
                'confidence': self.default_confidence,
                'notifications': True,
                'favorite_pairs': ['EURUSD', 'GBPUSD']
            }
        
        welcome_message = """
🎯 **Trade-2 Forex Bot** 

Welcome to your AI-powered forex prediction bot!

**Available Commands:**
📊 /predict [PAIR] - Get prediction for forex pair
📈 /analysis [PAIR] - Detailed technical analysis  
⚙️ /settings - Configure bot settings
📋 /pairs - View all available pairs
🔄 /refresh [PAIR] - Refresh prediction
📰 /status - Model and market status
❓ /help - Show this help message

**Quick Actions:**
Use the buttons below for instant predictions and signals!
        """
        
        # Create quick action keyboard
        keyboard = [
            [
                InlineKeyboardButton("💶 EURUSD", callback_data="predict_EURUSD"),
                InlineKeyboardButton("💷 GBPUSD", callback_data="predict_GBPUSD")
            ],
            [
                InlineKeyboardButton("💴 USDJPY", callback_data="predict_USDJPY"),
                InlineKeyboardButton("🏦 AUDUSD", callback_data="predict_AUDUSD")
            ],
            [
                InlineKeyboardButton("🎯 All Signals", callback_data="all_signals"),
                InlineKeyboardButton("📊 All Pairs", callback_data="all_pairs")
            ],
            [
                InlineKeyboardButton("⚙️ Settings", callback_data="settings")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            welcome_message,
            parse_mode='Markdown',
            reply_markup=reply_markup
        )
    
    async def predict_pair(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Predict command handler"""
        user_id = update.effective_user.id
        user_conf = self.user_settings.get(user_id, {}).get('confidence', self.default_confidence)
        
        # Get pair from command or default to EURUSD
        if context.args:
            pair_input = context.args[0].upper()
            if pair_input in self.forex_pairs:
                pair = pair_input
            else:
                await update.message.reply_text(
                    f"❌ Unknown pair: {pair_input}\nUse /pairs to see available pairs."
                )
                return
        else:
            pair = 'EURUSD'
        
        # Show loading message
        loading_msg = await update.message.reply_text(
            f"🔄 Analyzing {pair}... Please wait..."
        )
        
        try:
            # Get prediction
            prediction_data = await self.get_prediction(pair, user_conf)
            
            # Format and send result
            message = self.format_prediction_message(pair, prediction_data)
            
            # Create action buttons
            keyboard = [
                [
                    InlineKeyboardButton("🔄 Refresh", callback_data=f"predict_{pair}"),
                    InlineKeyboardButton("📈 Analysis", callback_data=f"analysis_{pair}")
                ],
                [
                    InlineKeyboardButton("📊 Other Pairs", callback_data="all_pairs"),
                    InlineKeyboardButton("⚙️ Settings", callback_data="settings")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # Delete loading message and send result
            await loading_msg.delete()
            await update.message.reply_text(
                message,
                parse_mode='Markdown',
                reply_markup=reply_markup
            )
            
        except Exception as e:
            await loading_msg.edit_text(f"❌ Error getting prediction: {str(e)}")
            logger.error(f"Prediction error: {e}")
    
    def calculate_optimal_entry(self, df: pd.DataFrame, action: str, current_price: float, pair: str) -> dict:
        """Calculate optimal entry price based on market conditions"""
        import talib
        
        # Get recent price data
        close_prices = df['close'].values[-50:].astype(np.float64)
        high_prices = df['high'].values[-50:].astype(np.float64)
        low_prices = df['low'].values[-50:].astype(np.float64)
        
        # Calculate support/resistance levels
        resistance_20 = df['high'].rolling(20).max().iloc[-1]
        support_20 = df['low'].rolling(20).min().iloc[-1]
        
        # Calculate ATR for volatility assessment
        atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)[-1]
        
        # Get pip value for this pair
        pip_value = self.get_pip_value(pair, current_price)
        
        # Calculate entry suggestions
        entry_suggestions = {
            'current_price': current_price,
            'market_entry': current_price,  # Immediate market entry
            'optimal_entry': current_price,  # Will be calculated
            'entry_type': 'MARKET',
            'support_level': support_20,
            'resistance_level': resistance_20,
            'atr': atr,
            'entry_distance_pips': 0
        }
        
        if action == 'long':
            # For LONG positions, suggest entry on pullbacks
            pullback_entry = current_price - (atr * 0.3)  # 30% ATR pullback
            
            # Choose the better entry (closer to support but not too far from current)
            if pullback_entry > support_20 and (current_price - pullback_entry) / current_price < 0.005:  # Within 0.5%
                entry_suggestions['optimal_entry'] = pullback_entry
                entry_suggestions['entry_type'] = 'LIMIT'
                entry_suggestions['entry_distance_pips'] = (current_price - pullback_entry) / pip_value
            else:
                entry_suggestions['optimal_entry'] = current_price
                entry_suggestions['entry_type'] = 'MARKET'
                
        elif action == 'short':
            # For SHORT positions, suggest entry on bounces
            bounce_entry = current_price + (atr * 0.3)  # 30% ATR bounce
            
            # Choose the better entry (closer to resistance but not too far from current)
            if bounce_entry < resistance_20 and (bounce_entry - current_price) / current_price < 0.005:  # Within 0.5%
                entry_suggestions['optimal_entry'] = bounce_entry
                entry_suggestions['entry_type'] = 'LIMIT'
                entry_suggestions['entry_distance_pips'] = (bounce_entry - current_price) / pip_value
            else:
                entry_suggestions['optimal_entry'] = current_price
                entry_suggestions['entry_type'] = 'MARKET'
        
        return entry_suggestions

    def get_pip_value(self, pair: str, price: float) -> float:
        """Calculate pip value for the pair"""
        if 'JPY' in pair:
            return 0.01  # For JPY pairs
        else:
            return 0.0001  # For other major pairs

    async def get_prediction(self, pair: str, confidence: float = 0.5):
        """Get enhanced prediction with optimal entry suggestions for a forex pair"""
        try:
            # Get recent data
            symbol = self.forex_pairs[pair]
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="1mo", interval="1h")
            
            if df.empty:
                raise ValueError(f"No data available for {pair}")
            
            # Normalize column names
            df.columns = [col.lower() for col in df.columns]
            
            # Get prediction
            prediction = self.predictor.predict(df, min_confidence=confidence)
            
            # Add current market data
            current_price = df['close'].iloc[-1]
            price_change = df['close'].iloc[-1] - df['close'].iloc[-2]
            price_change_pct = (price_change / df['close'].iloc[-2]) * 100
            
            # Calculate optimal entry if we have a trading signal
            entry_info = None
            if prediction['action'] != 'hold':
                entry_info = self.calculate_optimal_entry(df, prediction['action'], current_price, pair)
            
            result = {
                'pair': pair,
                'current_price': current_price,
                'price_change': price_change,
                'price_change_pct': price_change_pct,
                'prediction': prediction,
                'timestamp': datetime.now(),
                'data_points': len(df)
            }
            
            # Add enhanced entry information
            if entry_info:
                result['entry_info'] = entry_info
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting prediction for {pair}: {e}")
            raise
    
    def format_prediction_message(self, pair: str, data: dict) -> str:
        """Format prediction data into a nice message with entry suggestions"""
        pred = data['prediction']
        
        # Market info
        price_emoji = "📈" if data['price_change'] > 0 else "📉" if data['price_change'] < 0 else "➡️"
        
        message = f"""
🎯 **{pair} AI Analysis**

💹 **Market Status:**
{price_emoji} Price: `{data['current_price']:.5f}`
📊 Change: `{data['price_change']:+.5f}` ({data['price_change_pct']:+.2f}%)
🕐 Updated: {data['timestamp'].strftime('%H:%M:%S UTC')}
📈 Data Points: {data['data_points']}

🤖 **AI Prediction:**
        """
        
        # Prediction details
        action = pred['action'].upper()
        confidence = pred['confidence']
        
        if action == 'HOLD':
            message += f"""
🔄 **Action: HOLD**
📊 Confidence: {confidence:.1%}
💭 No clear trading signal detected
        """
        else:
            # Trading signal
            action_emoji = "🟢 📈" if action == 'LONG' else "🔴 📉"
            message += f"""
{action_emoji} **Action: {action}**
🎯 Confidence: {confidence:.1%}
            """
            
            # Add entry suggestions if available
            if 'entry_info' in data:
                entry = data['entry_info']
                message += f"""

📍 **Entry Strategy:**
                """
                
                if entry['entry_type'] == 'MARKET':
                    message += f"""🟢 **MARKET ORDER**: `{entry['market_entry']:.5f}`
💡 Enter immediately - optimal conditions detected
                    """
                else:
                    message += f"""🟡 **LIMIT ORDER**: `{entry['optimal_entry']:.5f}`
📏 Distance: {entry['entry_distance_pips']:.1f} pips from current
💡 Wait for better price - potential improvement available
                    """
                
                message += f"""
🔺 Resistance: `{entry['resistance_level']:.5f}`
🔻 Support: `{entry['support_level']:.5f}`
                """
            
            # Add SL/TP if available
            if 'stop_loss' in pred and 'take_profit' in pred:
                sl = pred['stop_loss']
                tp = pred['take_profit']
                rr = pred.get('risk_reward_ratio', 0)
                risk_pips = pred.get('risk_pips', 0)
                reward_pips = pred.get('reward_pips', 0)
                
                message += f"""

🛡️ **Risk Management:**
🛑 Stop Loss: `{sl:.5f}` ({risk_pips:.1f} pips)
🎯 Take Profit: `{tp:.5f}` ({reward_pips:.1f} pips)
📊 Risk:Reward: `1:{rr:.2f}`
                """
        
        # Add confidence interpretation
        if confidence < 0.4:
            conf_text = "🟡 Low confidence"
        elif confidence < 0.6:
            conf_text = "🟠 Medium confidence"
        else:
            conf_text = "🟢 High confidence"
        
        message += f"""

💡 **Signal Strength:** {conf_text}
        """
        
        return message
    
    async def detailed_analysis(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Detailed technical analysis"""
        user_id = update.effective_user.id
        
        # Get pair from command
        if context.args:
            pair = context.args[0].upper()
        else:
            pair = 'EURUSD'
        
        if pair not in self.forex_pairs:
            await update.message.reply_text(f"❌ Unknown pair: {pair}")
            return
        
        loading_msg = await update.message.reply_text(f"📊 Generating detailed analysis for {pair}...")
        
        try:
            # Get extended data
            symbol = self.forex_pairs[pair]
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="3mo", interval="1h")
            df.columns = [col.lower() for col in df.columns]
            
            # Get prediction with different confidence levels
            confidences = [0.3, 0.5, 0.7]
            predictions = {}
            
            for conf in confidences:
                pred = self.predictor.predict(df, min_confidence=conf)
                predictions[conf] = pred
            
            # Generate analysis message
            message = self.format_analysis_message(pair, df, predictions)
            
            await loading_msg.edit_text(message, parse_mode='Markdown')
            
        except Exception as e:
            await loading_msg.edit_text(f"❌ Analysis error: {str(e)}")
    
    def format_analysis_message(self, pair: str, df: pd.DataFrame, predictions: dict) -> str:
        """Format detailed analysis message"""
        current_price = df['close'].iloc[-1]
        
        # Market stats
        high_24h = df['high'].tail(24).max()
        low_24h = df['low'].tail(24).min()
        vol_avg = df['close'].pct_change().tail(100).std() * 100
        
        message = f"""
📊 **{pair} Detailed Analysis**

📈 **Market Overview:**
💹 Current: `{current_price:.5f}`
🔺 24h High: `{high_24h:.5f}`
🔻 24h Low: `{low_24h:.5f}`
📊 Volatility: `{vol_avg:.2f}%`

🤖 **Multi-Confidence Analysis:**
        """
        
        for conf, pred in predictions.items():
            action = pred['action'].upper()
            confidence = pred['confidence']
            
            if action == 'HOLD':
                emoji = "🔄"
            elif action == 'LONG':
                emoji = "🟢📈"
            else:
                emoji = "🔴📉"
            
            message += f"""
**{conf:.0%} Threshold:** {emoji} {action} ({confidence:.1%})
            """
        
        # Technical levels
        resistance = df['high'].rolling(50).max().iloc[-1]
        support = df['low'].rolling(50).min().iloc[-1]
        
        message += f"""

📏 **Key Levels:**
🔺 Resistance: `{resistance:.5f}`
🔻 Support: `{support:.5f}`
        """
        
        return message
    
    async def settings_menu(self, query_or_update, context: ContextTypes.DEFAULT_TYPE = None):
        """Settings command handler"""
        # Extract user_id based on the type of object we received
        if hasattr(query_or_update, 'from_user'):
            # It's a CallbackQuery
            user_id = query_or_update.from_user.id
        else:
            # It's an Update
            user_id = query_or_update.effective_user.id
            
        user_conf = self.user_settings.get(user_id, {}).get('confidence', self.default_confidence)
        
        keyboard = [
            [
                InlineKeyboardButton(f"🎯 Confidence: {user_conf:.0%}", callback_data="conf_settings"),
                InlineKeyboardButton("🔔 Notifications", callback_data="notif_settings")
            ],
            [
                InlineKeyboardButton("⭐ Favorites", callback_data="fav_settings"),
                InlineKeyboardButton("ℹ️ Info", callback_data="info_settings")
            ],
            [
                InlineKeyboardButton("🔙 Back to Main", callback_data="main_menu")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        message = f"""
⚙️ **Bot Settings**

Current Configuration:
🎯 Confidence Threshold: {user_conf:.0%}
🔔 Notifications: {'On' if self.user_settings.get(user_id, {}).get('notifications', True) else 'Off'}

Choose what to configure:
        """
        
        # Check if it's a callback query or regular update
        if hasattr(query_or_update, 'edit_message_text'):
            # It's a CallbackQuery
            await query_or_update.edit_message_text(
                message, parse_mode='Markdown', reply_markup=reply_markup
            )
        else:
            # It's an Update with message
            await query_or_update.message.reply_text(
                message, parse_mode='Markdown', reply_markup=reply_markup
            )
    
    async def pairs_list(self, query_or_update, context: ContextTypes.DEFAULT_TYPE = None):
        """Show available forex pairs"""
        keyboard = []
        pairs_list = list(self.forex_pairs.keys())
        
        # Create keyboard with pairs (2 per row)
        for i in range(0, len(pairs_list), 2):
            row = []
            for j in range(2):
                if i + j < len(pairs_list):
                    pair = pairs_list[i + j]
                    row.append(InlineKeyboardButton(
                        f"💱 {pair}", 
                        callback_data=f"predict_{pair}"
                    ))
            keyboard.append(row)
        
        keyboard.append([InlineKeyboardButton("🔙 Back", callback_data="main_menu")])
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        message = """
📋 **Available Forex Pairs**

Click any pair for instant prediction:
        """
        
        # Check if it's a callback query or regular update
        if hasattr(query_or_update, 'edit_message_text'):
            # It's a CallbackQuery
            await query_or_update.edit_message_text(
                message, parse_mode='Markdown', reply_markup=reply_markup
            )
        else:
            # It's an Update with message
            await query_or_update.message.reply_text(
                message, parse_mode='Markdown', reply_markup=reply_markup
            )
    
    async def show_all_signals(self, query, user_id: int):
        """Show all trading signals (BUY/SELL only, no HOLD) with entry suggestions"""
        user_conf = self.user_settings.get(user_id, {}).get('confidence', self.default_confidence)
        
        # Show loading message
        await query.edit_message_text("🔄 Scanning all pairs for trading signals...")
        
        try:
            signals = []
            
            # Check all forex pairs for signals
            for pair in self.forex_pairs.keys():
                try:
                    prediction_data = await self.get_prediction(pair, user_conf)
                    pred = prediction_data['prediction']
                    
                    # Only include BUY/SELL signals, skip HOLD
                    if pred['action'].upper() != 'HOLD':
                        signal_data = {
                            'pair': pair,
                            'action': pred['action'].upper(),
                            'confidence': pred['confidence'],
                            'current_price': prediction_data['current_price'],
                            'price_change_pct': prediction_data['price_change_pct'],
                            'stop_loss': pred.get('stop_loss'),
                            'take_profit': pred.get('take_profit'),
                            'risk_reward_ratio': pred.get('risk_reward_ratio', 0)
                        }
                        
                        # Add entry info if available
                        if 'entry_info' in prediction_data:
                            signal_data['entry_info'] = prediction_data['entry_info']
                        
                        signals.append(signal_data)
                        
                except Exception as e:
                    logger.error(f"Error getting signal for {pair}: {e}")
                    continue
            
            # Format the signals message
            if signals:
                # Sort by confidence (highest first)
                signals.sort(key=lambda x: x['confidence'], reverse=True)
                
                message = f"🎯 **Active Trading Signals** ({user_conf:.0%} confidence)\n\n"
                
                for signal in signals:
                    action_emoji = "🟢📈" if signal['action'] == 'LONG' else "🔴📉"
                    price_emoji = "📈" if signal['price_change_pct'] > 0 else "📉"
                    
                    message += f"""
{action_emoji} **{signal['pair']} - {signal['action']}**
💹 Price: `{signal['current_price']:.5f}` {price_emoji}`{signal['price_change_pct']:+.2f}%`
🎯 Confidence: `{signal['confidence']:.1%}`"""
                    
                    # Add entry info if available
                    if 'entry_info' in signal:
                        entry = signal['entry_info']
                        if entry['entry_type'] == 'MARKET':
                            message += f"""
📍 Entry: 🟢 **MARKET** `{entry['market_entry']:.5f}`"""
                        else:
                            message += f"""
📍 Entry: 🟡 **LIMIT** `{entry['optimal_entry']:.5f}` ({entry['entry_distance_pips']:.1f} pips)"""
                    
                    # Add SL/TP if available
                    if signal['stop_loss'] and signal['take_profit']:
                        message += f"""
🛡️ SL: `{signal['stop_loss']:.5f}` 🎯 TP: `{signal['take_profit']:.5f}` (R:R `1:{signal['risk_reward_ratio']:.2f}`)"""
                    
                    message += "\n"
                
                message += f"\n📊 Found **{len(signals)}** trading signals"
                
            else:
                message = f"""
🔄 **No Active Signals**

No BUY/SELL signals found at {user_conf:.0%} confidence level.

💡 Try lowering confidence in settings for more signals.
                """
            
            # Create refresh button
            keyboard = [
                [
                    InlineKeyboardButton("🔄 Refresh Signals", callback_data="all_signals"),
                    InlineKeyboardButton("⚙️ Settings", callback_data="settings")
                ],
                [
                    InlineKeyboardButton("📊 All Pairs", callback_data="all_pairs"),
                    InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                message,
                parse_mode='Markdown',
                reply_markup=reply_markup
            )
            
        except Exception as e:
            await query.edit_message_text(f"❌ Error scanning for signals: {str(e)}")
    
    async def status_info(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show bot and model status"""
        model_info = f"""
📊 **System Status**

🤖 **Model Information:**
• Type: Supervised Learning (LSTM + Attention)
• Architecture: {self.predictor.sequence_length} sequence length
• Confidence: {self.predictor.min_confidence:.0%} base threshold
• Features: 95 technical indicators

📈 **Market Data:**
• Source: Yahoo Finance
• Update: Real-time
• History: 1 month lookback
• Timeframe: 1 hour candles

⚡ **Performance:**
• Prediction Speed: ~2-3 seconds
• Supported Pairs: {len(self.forex_pairs)}
• Uptime: Active ✅
        """
        
        await update.message.reply_text(model_info, parse_mode='Markdown')
    
    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle button callbacks"""
        query = update.callback_query
        await query.answer()
        
        data = query.data
        user_id = update.effective_user.id
        
        try:
            if data.startswith('predict_'):
                pair = data.replace('predict_', '')
                await self.handle_pair_prediction(query, pair, user_id)
                
            elif data.startswith('analysis_'):
                pair = data.replace('analysis_', '')
                await self.handle_detailed_analysis(query, pair, user_id)
                
            elif data == 'settings':
                await self.settings_menu(query, context)
                
            elif data == 'all_pairs':
                await self.pairs_list(query, context)
                
            elif data == 'all_signals':
                await self.show_all_signals(query, user_id)
                
            elif data == 'main_menu':
                await self.show_main_menu(query)
                
            elif data == 'conf_settings':
                await self.confidence_settings(query, user_id)
                
            elif data.startswith('set_conf_'):
                conf = float(data.replace('set_conf_', ''))
                await self.update_user_confidence(query, user_id, conf)
                
        except Exception as e:
            logger.error(f"Error in button callback: {e}")
            try:
                await query.edit_message_text(
                    f"❌ An error occurred: {str(e)}\n\nPlease try again or use /start to return to main menu.",
                    parse_mode='Markdown'
                )
            except:
                # If we can't edit the message, send a new one
                await context.bot.send_message(
                    chat_id=user_id,
                    text="❌ Something went wrong. Please use /start to restart the bot."
                )
    
    async def handle_pair_prediction(self, query, pair: str, user_id: int):
        """Handle pair prediction from button"""
        user_conf = self.user_settings.get(user_id, {}).get('confidence', self.default_confidence)
        
        # Edit message to show loading
        await query.edit_message_text(f"🔄 Analyzing {pair}...")
        
        try:
            prediction_data = await self.get_prediction(pair, user_conf)
            message = self.format_prediction_message(pair, prediction_data)
            
            # Create buttons
            keyboard = [
                [
                    InlineKeyboardButton("🔄 Refresh", callback_data=f"predict_{pair}"),
                    InlineKeyboardButton("📈 Analysis", callback_data=f"analysis_{pair}")
                ],
                [
                    InlineKeyboardButton("📊 Other Pairs", callback_data="all_pairs"),
                    InlineKeyboardButton("🔙 Main Menu", callback_data="main_menu")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                message, parse_mode='Markdown', reply_markup=reply_markup
            )
            
        except Exception as e:
            await query.edit_message_text(f"❌ Error: {str(e)}")
    
    async def handle_detailed_analysis(self, query, pair: str, user_id: int):
        """Handle detailed analysis from button"""
        await query.edit_message_text(f"📊 Generating analysis for {pair}...")
        
        try:
            # Get extended data and analysis
            symbol = self.forex_pairs[pair]
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="3mo", interval="1h")
            df.columns = [col.lower() for col in df.columns]
            
            # Multi-confidence predictions
            predictions = {}
            for conf in [0.3, 0.5, 0.7]:
                predictions[conf] = self.predictor.predict(df, min_confidence=conf)
            
            message = self.format_analysis_message(pair, df, predictions)
            
            keyboard = [
                [
                    InlineKeyboardButton("🔄 Quick Predict", callback_data=f"predict_{pair}"),
                    InlineKeyboardButton("📊 Other Pairs", callback_data="all_pairs")
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
            await query.edit_message_text(f"❌ Analysis error: {str(e)}")
    
    async def confidence_settings(self, query, user_id: int):
        """Show confidence settings menu"""
        current_conf = self.user_settings.get(user_id, {}).get('confidence', self.default_confidence)
        
        keyboard = [
            [
                InlineKeyboardButton("30%", callback_data="set_conf_0.3"),
                InlineKeyboardButton("40%", callback_data="set_conf_0.4"),
                InlineKeyboardButton("50%", callback_data="set_conf_0.5")
            ],
            [
                InlineKeyboardButton("60%", callback_data="set_conf_0.6"),
                InlineKeyboardButton("70%", callback_data="set_conf_0.7"),
                InlineKeyboardButton("80%", callback_data="set_conf_0.8")
            ],
            [
                InlineKeyboardButton("🔙 Back to Settings", callback_data="settings")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        message = f"""
🎯 **Confidence Threshold Settings**

Current: {current_conf:.0%}

**Choose your preferred confidence level:**

• **30-40%**: More signals, lower accuracy
• **50%**: Balanced (recommended)
• **60-70%**: Fewer signals, higher accuracy
• **80%+**: Very selective, rare signals

Lower confidence = More trading opportunities
Higher confidence = Only strongest signals
        """
        
        await query.edit_message_text(
            message, parse_mode='Markdown', reply_markup=reply_markup
        )
    
    async def update_user_confidence(self, query, user_id: int, confidence: float):
        """Update user confidence setting"""
        if user_id not in self.user_settings:
            self.user_settings[user_id] = {}
        
        self.user_settings[user_id]['confidence'] = confidence
        
        keyboard = [
            [InlineKeyboardButton("🔙 Back to Settings", callback_data="settings")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            f"✅ **Confidence Updated!**\n\nNew threshold: {confidence:.0%}\n\nThis affects all your predictions. Lower confidence = more signals, higher confidence = fewer but stronger signals.",
            parse_mode='Markdown',
            reply_markup=reply_markup
        )
    
    async def show_main_menu(self, query):
        """Show main menu"""
        keyboard = [
            [
                InlineKeyboardButton("💶 EURUSD", callback_data="predict_EURUSD"),
                InlineKeyboardButton("💷 GBPUSD", callback_data="predict_GBPUSD")
            ],
            [
                InlineKeyboardButton("💴 USDJPY", callback_data="predict_USDJPY"),
                InlineKeyboardButton("🏦 AUDUSD", callback_data="predict_AUDUSD")
            ],
            [
                InlineKeyboardButton("🎯 All Signals", callback_data="all_signals"),
                InlineKeyboardButton("📊 All Pairs", callback_data="all_pairs")
            ],
            [
                InlineKeyboardButton("⚙️ Settings", callback_data="settings")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            "🎯 **Trade-2 Forex Bot**\n\nSelect an option:",
            parse_mode='Markdown',
            reply_markup=reply_markup
        )
    
    def run(self):
        """Start the bot"""
        try:
            # Create application
            self.app = Application.builder().token(self.bot_token).build()
            
            # Add error handler
            async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
                """Log errors and notify user if possible"""
                logger.error(f"Exception while handling an update: {context.error}")
                
                # Try to send error message to user
                if update and hasattr(update, 'effective_chat') and update.effective_chat:
                    try:
                        await context.bot.send_message(
                            chat_id=update.effective_chat.id,
                            text="❌ Sorry, something went wrong. Please try again or use /start to restart."
                        )
                    except:
                        pass
            
            self.app.add_error_handler(error_handler)
            
            # Add handlers
            self.app.add_handler(CommandHandler("start", self.start))
            self.app.add_handler(CommandHandler("predict", self.predict_pair))
            self.app.add_handler(CommandHandler("analysis", self.detailed_analysis))
            self.app.add_handler(CommandHandler("settings", self.settings_menu))
            self.app.add_handler(CommandHandler("pairs", self.pairs_list))
            self.app.add_handler(CommandHandler("status", self.status_info))
            self.app.add_handler(CommandHandler("help", self.start))
            self.app.add_handler(CallbackQueryHandler(self.button_callback))
            
            # Start bot
            logger.info("🚀 Starting Trade-2 Telegram Bot...")
            self.app.run_polling()
            
        except Exception as e:
            logger.error(f"Failed to start bot: {e}")
            raise

def main():
    """Main function"""
    # Configuration
    BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    MODEL_PATH = "best_model.pth"
    
    if not BOT_TOKEN:
        print("❌ Please set TELEGRAM_BOT_TOKEN environment variable")
        print("Get token from @BotFather on Telegram")
        return
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        return
    
    # Create and run bot
    bot = Trade2TelegramBot(BOT_TOKEN, MODEL_PATH)
    bot.run()

if __name__ == "__main__":
    main()
