"""
TELEGRAM TRADING SIGNAL BOT
Automatically scans pairs and sends trading signals with SL/TP recommendations
"""

import sys
import os
import asyncio
import threading
import time
from datetime import datetime
from enhanced_live_trader import EnhancedLiveTrader

class TelegramSignalBot(EnhancedLiveTrader):
    """Signal-only trader - scans and sends recommendations without auto-trading"""
    
    def __init__(self, initial_balance: float = 10000, demo_mode: bool = True, telegram_bot=None):
        super().__init__(initial_balance, demo_mode)
        self.telegram_bot = telegram_bot
        self.signal_enabled = True
        self.auto_trading_enabled = False  # Permanently disabled - signals only
        self.last_signal_time = {}  # Track when we last sent signal for each pair
        self.signal_cooldown = 1800  # 30 minutes between signals for same pair
        
        # Signal thresholds (more conservative for manual trading)
        self.signal_confidence_threshold = 0.70  # 70% confidence minimum
        self.signal_strength_threshold = 0.08   # 8% signal strength minimum
    
    def set_telegram_bot(self, telegram_bot):
        """Set the Telegram bot instance for notifications"""
        self.telegram_bot = telegram_bot
    
    def enable_signals(self, enabled: bool = True):
        """Enable or disable signal notifications"""
        self.signal_enabled = enabled
        
    def enable_auto_trading(self, enabled: bool = False):
        """Auto-trading is permanently disabled for signal-only mode"""
        self.auto_trading_enabled = False  # Always disabled
        print("ℹ️ Auto-trading is disabled in signal-only mode")
    
    def send_notification_sync(self, message: str):
        """Send notification synchronously from trading thread"""
        if self.telegram_bot:
            try:
                # Use the application's job queue to send notification
                if hasattr(self.telegram_bot, 'application') and self.telegram_bot.application:
                    job_queue = self.telegram_bot.application.job_queue
                    if job_queue:
                        job_queue.run_once(
                            lambda context: self.telegram_bot.send_notification(message),
                            when=0
                        )
            except Exception as e:
                print(f"Failed to send notification: {e}")
    
    def calculate_sl_tp_levels(self, pair: str, side: str, entry_price: float, confidence: float):
        """Calculate stop loss and take profit levels based on pair and confidence"""
        
        # Get pair-specific parameters
        pair_params = self.get_pair_params(pair)
        pip_value = pair_params['pip_value']
        
        # Base SL/TP in pips (adjust based on confidence)
        if confidence >= 0.80:
            sl_pips = 15  # Tighter SL for high confidence
            tp_pips = 30  # 1:2 risk/reward
        elif confidence >= 0.70:
            sl_pips = 20
            tp_pips = 35
        else:
            sl_pips = 25
            tp_pips = 40
        
        # Convert pips to price levels
        if side == 'buy':
            stop_loss = entry_price - (sl_pips * pip_value)
            take_profit = entry_price + (tp_pips * pip_value)
        else:  # sell
            stop_loss = entry_price + (sl_pips * pip_value)
            take_profit = entry_price - (tp_pips * pip_value)
            
        return stop_loss, take_profit, sl_pips, tp_pips
    
    def should_send_signal(self, prediction: dict) -> bool:
        """Check if we should send a trading signal"""
        if not self.signal_enabled:
            return False
            
        pair = prediction['pair']
        confidence = prediction['confidence']
        signal_strength = abs(prediction.get('signal', 0))
        
        # Check confidence and signal strength thresholds
        if confidence < self.signal_confidence_threshold:
            return False
            
        if signal_strength < self.signal_strength_threshold:
            return False
        
        # Check cooldown period
        current_time = time.time()
        if pair in self.last_signal_time:
            time_since_last = current_time - self.last_signal_time[pair]
            if time_since_last < self.signal_cooldown:
                return False
        
        return True
    
    def send_trading_signal(self, prediction: dict, current_price: float):
        """Send trading signal notification"""
        pair = prediction['pair']
        side = prediction['side']
        confidence = prediction['confidence']
        signal_strength = abs(prediction.get('signal', 0))
        
        # Calculate SL/TP levels
        stop_loss, take_profit, sl_pips, tp_pips = self.calculate_sl_tp_levels(
            pair, side, current_price, confidence
        )
        
        # Calculate risk/reward ratio
        risk_reward = tp_pips / sl_pips
        
        # Determine signal quality
        if confidence >= 0.80 and signal_strength >= 0.10:
            signal_quality = "🔥 STRONG"
        elif confidence >= 0.75:
            signal_quality = "⚡ GOOD"
        else:
            signal_quality = "📊 MODERATE"
        
        direction_emoji = "🔼" if side == 'buy' else "🔽"
        
        message = f"""
🎯 **TRADING SIGNAL** {signal_quality}

💱 **Pair**: {pair}
{direction_emoji} **Direction**: {side.upper()}
💰 **Entry**: {current_price:.5f}

🛡️ **Stop Loss**: {stop_loss:.5f} ({sl_pips} pips)
🎯 **Take Profit**: {take_profit:.5f} ({tp_pips} pips)
📊 **Risk/Reward**: 1:{risk_reward:.1f}

🎯 **Confidence**: {confidence:.1%}
📈 **Signal Strength**: {signal_strength:.1%}

⏰ **Time**: {datetime.now().strftime('%H:%M:%S')}

💡 **Suggested Position Size**: {self.position_size_pct:.1%} of balance
        """
        
        # Send the signal
        self.send_notification_sync(message)
        
        # Update last signal time
        self.last_signal_time[pair] = time.time()
        
        # Log the signal
        print(f"📡 Signal sent: {pair} {side.upper()} @ {current_price:.5f} (Confidence: {confidence:.1%})")
    
    def scan_for_signals(self):
        """Override the signal scanning to send notifications instead of trading"""
        try:
            signals_found = 0
            
            for pair in self.pairs:
                try:
                    # Get live data
                    data = self.get_live_data(pair)
                    if data is None or len(data) < 50:
                        continue
                    
                    # Create features
                    features = self.create_features(data)
                    if features is None or len(features) == 0:
                        continue
                    
                    # Make prediction
                    prediction = self.make_prediction(pair, features)
                    if not prediction:
                        continue
                    
                    # Check if we should send a signal
                    if self.should_send_signal(prediction):
                        current_price = data['Close'].iloc[-1]
                        self.send_trading_signal(prediction, current_price)
                        signals_found += 1
                    
                    # If auto-trading is enabled, also execute the trade
                    if self.auto_trading_enabled and self.should_trade(prediction):
                        self.open_position(prediction)
                        
                except Exception as e:
                    print(f"Error scanning {pair}: {e}")
                    continue
            
            if signals_found > 0:
                print(f"📡 Sent {signals_found} trading signal(s)")
                
        except Exception as e:
            print(f"Error in signal scanning: {e}")
    
    async def analyze_all_pairs_now(self):
        """Perform immediate analysis of all pairs and return signals found"""
        try:
            signals_found = []
            
            print("🔍 Starting immediate market analysis...")
            
            for pair in self.pairs:
                try:
                    print(f"Analyzing {pair}...")
                    
                    # Get live data
                    data = self.get_live_data(pair)
                    if data is None or len(data) < 50:
                        print(f"❌ {pair}: Insufficient data")
                        continue
                    
                    # Create features
                    features = self.create_features(data)
                    if features is None or len(features) == 0:
                        print(f"❌ {pair}: Feature creation failed")
                        continue
                    
                    # Make prediction
                    prediction = self.make_prediction(pair, features)
                    if not prediction:
                        print(f"❌ {pair}: Prediction failed")
                        continue
                    
                    confidence = prediction['confidence']
                    signal_strength = abs(prediction.get('signal', 0))
                    
                    print(f"📊 {pair}: Confidence {confidence:.1%}, Signal {signal_strength:.1%}")
                    
                    # Check if this meets our signal criteria (ignore cooldown for immediate analysis)
                    if (confidence >= self.signal_confidence_threshold and 
                        signal_strength >= self.signal_strength_threshold and
                        self.signal_enabled):
                        
                        current_price = data['Close'].iloc[-1]
                        self.send_trading_signal(prediction, current_price)
                        signals_found.append({
                            'pair': pair,
                            'confidence': confidence,
                            'signal_strength': signal_strength,
                            'price': current_price
                        })
                        print(f"✅ {pair}: Signal sent!")
                    else:
                        print(f"⚪ {pair}: No signal (below thresholds)")
                        
                except Exception as e:
                    print(f"❌ Error analyzing {pair}: {e}")
                    continue
            
            print(f"🔍 Analysis complete. Found {len(signals_found)} signals.")
            return signals_found
            
        except Exception as e:
            print(f"Error in immediate analysis: {e}")
            return []
    
    def open_position(self, prediction: dict) -> bool:
        """Override to only trade if auto-trading is enabled"""
        if not self.auto_trading_enabled:
            return False  # Don't auto-trade, signals only
            
        result = super().open_position(prediction)
        
        if result and self.telegram_bot:
            pair = prediction['pair']
            side = prediction['side']
            confidence = prediction['confidence']
            size = self.calculate_position_size()
            
            message = f"""
✅ **AUTO-TRADE EXECUTED**

💱 Pair: {pair}
{'🔼' if side == 'buy' else '🔽'} Direction: {side.upper()}
💰 Size: {size:,.2f}
🎯 Confidence: {confidence:.1%}
💵 Balance: ${self.balance:,.2f}

⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

*Auto-trading is enabled*
            """
            
            self.send_notification_sync(message)
        
        return result
    
    def close_position(self, pair: str, exit_reason: str, current_price: float) -> bool:
        """Override to send notifications for closed positions"""
        if not self.auto_trading_enabled:
            return False  # No positions to close if not auto-trading
            
        position = self.positions.get(pair)
        if not position:
            return False
        
        # Calculate P&L before closing
        if position['side'] == 'buy':
            pnl = (current_price - position['entry_price']) * position['size']
        else:
            pnl = (position['entry_price'] - current_price) * position['size']
        
        result = super().close_position(pair, exit_reason, current_price)
        
        if result and self.telegram_bot:
            pnl_emoji = "✅" if pnl > 0 else "❌"
            direction_emoji = "🔼" if position['side'] == 'buy' else "🔽"
            
            message = f"""
{pnl_emoji} **AUTO-POSITION CLOSED**

💱 Pair: {pair}
{direction_emoji} Direction: {position['side'].upper()}
💰 P&L: ${pnl:.2f}
📊 Reason: {exit_reason}
💵 New Balance: ${self.balance:,.2f}

⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            self.send_notification_sync(message)
        
        return result
    
    def start_signal_scanning(self):
        """Start the signal scanning system"""
        if self.telegram_bot:
            message = f"""
🚀 **Signal Scanner Started**

📡 Mode: Signals Only
🎯 Confidence Threshold: {self.signal_confidence_threshold:.0%}
📊 Signal Strength: {self.signal_strength_threshold:.1%}
⏱️ Scan Interval: {self.update_interval}s
🔄 Signal Cooldown: {self.signal_cooldown//60}min

💱 Monitoring Pairs:
{chr(10).join(f"• {pair}" for pair in self.pairs)}

⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            self.send_notification_sync(message)
        
        # Start the scanning loop
        super().start_trading()
    
    def stop_signal_scanning(self):
        """Stop the signal scanning system"""
        if self.telegram_bot:
            message = f"""
⏹️ **Signal Scanner Stopped**

📊 Signals Sent: {len(self.last_signal_time)}
⏰ Stopped: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            self.send_notification_sync(message)
        
        super().stop_trading()
    
    def trading_loop(self):
        """Override to use signal scanning instead of auto-trading"""
        print(f"🔄 Starting signal scanning loop...")
        print(f"📡 Mode: Signals Only")
        
        while self.is_running:
            try:
                loop_start = time.time()
                
                # Scan for new signals (always active)
                self.scan_for_signals()
                
                # Save state
                self.save_state()
                
                # Calculate sleep time
                loop_duration = time.time() - loop_start
                sleep_time = max(0, self.update_interval - loop_duration)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                print(f"❌ Error in signal scanning loop: {e}")
                time.sleep(10)  # Wait before retrying

def main():
    """Main function for testing the signal bot"""
    import json
    
    # Load configuration
    config_file = 'trading_config.json'
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Create signal bot
    signal_bot = TelegramSignalBot(
        initial_balance=config.get('initial_balance', 10000),
        demo_mode=config.get('demo_mode', True)
    )
    
    # Apply configuration
    signal_bot.position_size_pct = config.get('position_size_pct', 0.02)
    signal_bot.max_daily_risk = config.get('max_daily_risk', 0.06)
    signal_bot.max_positions = config.get('max_positions', 3)
    signal_bot.update_interval = config.get('update_interval', 60)
    signal_bot.pairs = config.get('pairs', ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X'])
    
    try:
        signal_bot.load_enhanced_models()
        signal_bot.start_signal_scanning()
    except KeyboardInterrupt:
        print("\n⏹️ Stopping signal scanner...")
        signal_bot.stop_signal_scanning()
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
