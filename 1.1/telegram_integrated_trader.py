"""
ENHANCED LIVE TRADER WITH TELEGRAM NOTIFICATIONS
Extended version with Telegram bot integration
"""

import sys
import os
import asyncio
import threading
from datetime import datetime
from enhanced_live_trader import EnhancedLiveTrader

class TelegramIntegratedTrader(EnhancedLiveTrader):
    """Enhanced trader with Telegram notifications"""
    
    def __init__(self, initial_balance: float = 10000, demo_mode: bool = True, telegram_bot=None):
        super().__init__(initial_balance, demo_mode)
        self.telegram_bot = telegram_bot
        self.notification_enabled = True
    
    def set_telegram_bot(self, telegram_bot):
        """Set the Telegram bot instance for notifications"""
        self.telegram_bot = telegram_bot
    
    def send_notification_sync(self, message: str):
        """Send notification synchronously from trading thread"""
        if self.telegram_bot and self.notification_enabled:
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
    
    async def send_notification(self, message: str):
        """Send notification via Telegram bot"""
        if self.telegram_bot and self.notification_enabled:
            try:
                await self.telegram_bot.send_notification(message)
            except Exception as e:
                self.logger.error(f"Failed to send Telegram notification: {e}")
    
    def open_position(self, prediction: dict) -> bool:
        """Override to send Telegram notification when opening position"""
        result = super().open_position(prediction)
        
        if result and self.telegram_bot:
            pair = prediction['pair']
            side = prediction['side']
            confidence = prediction['confidence']
            size = self.calculate_position_size()
            
            message = f"""
🔔 **New Position Opened**

💱 Pair: {pair}
{'🔼' if side == 'buy' else '🔽'} Direction: {side.upper()}
💰 Size: {size:,.2f}
🎯 Confidence: {confidence:.1%}
💵 Balance: ${self.balance:,.2f}

⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            # Send notification using sync method
            self.send_notification_sync(message)
        
        return result
    
    def close_position(self, pair: str, exit_reason: str, current_price: float) -> bool:
        """Override to send Telegram notification when closing position"""
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
{pnl_emoji} **Position Closed**

💱 Pair: {pair}
{direction_emoji} Direction: {position['side'].upper()}
💰 P&L: ${pnl:.2f}
📊 Reason: {exit_reason}
💵 New Balance: ${self.balance:,.2f}

⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            # Send notification using sync method
            self.send_notification_sync(message)
        
        return result
    
    def start_trading(self):
        """Override to send start notification"""
        if self.telegram_bot:
            message = f"""
🚀 **Trading System Started**

💰 Initial Balance: ${self.balance:,.2f}
{'🧪' if self.demo_mode else '💸'} Mode: {'Demo' if self.demo_mode else 'Live'}
📊 Max Positions: {self.max_positions}
⚠️ Risk per Trade: {self.position_size_pct:.1%}

💱 Trading Pairs:
{chr(10).join(f"• {pair}" for pair in self.pairs)}

⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            # Send notification using sync method
            self.send_notification_sync(message)
        
        super().start_trading()
    
    def stop_trading(self):
        """Override to send stop notification"""
        if self.telegram_bot and self.is_running:
            total_pnl = self.balance - self.initial_balance
            pnl_emoji = "📈" if total_pnl > 0 else "📉"
            
            message = f"""
⏹️ **Trading System Stopped**

{pnl_emoji} Total P&L: ${total_pnl:.2f}
💰 Final Balance: ${self.balance:,.2f}
📊 Total Trades: {len(self.trade_history)}
📈 Open Positions: {len(self.positions)}

⏰ Stopped: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            # Send notification using sync method
            self.send_notification_sync(message)
        
        super().stop_trading()

def main():
    """Main function for testing the integrated trader"""
    import json
    
    # Load configuration
    config_file = 'trading_config.json'
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Create integrated trader
    trader = TelegramIntegratedTrader(
        initial_balance=config.get('initial_balance', 10000),
        demo_mode=config.get('demo_mode', True)
    )
    
    # Apply configuration
    trader.position_size_pct = config.get('position_size_pct', 0.02)
    trader.max_daily_risk = config.get('max_daily_risk', 0.06)
    trader.max_positions = config.get('max_positions', 3)
    trader.update_interval = config.get('update_interval', 60)
    trader.pairs = config.get('pairs', ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X'])
    
    try:
        trader.load_enhanced_models()
        trader.start_trading()
    except KeyboardInterrupt:
        print("\n⏹️ Stopping trading...")
        trader.stop_trading()
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
