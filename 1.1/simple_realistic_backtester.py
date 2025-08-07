"""
REALISTIC BACKTESTING SYSTEM - SIMPLIFIED VERSION
Simulates real trading with full balance usage and rule-based signals for testing
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
import warnings
warnings.filterwarnings('ignore')

class SimpleRealisticBacktester:
    def __init__(self, initial_balance: float = 1500.0):
        """Initialize realistic backtester with proper balance management"""
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.available_balance = initial_balance
        
        # Trading state
        self.open_position = None
        self.trades_history = []
        self.equity_curve = []
        self.daily_balances = []
        
        # Risk management
        self.max_risk_per_trade = 0.02  # 2% risk per trade
        self.confidence_threshold = 0.70
        self.signal_strength_threshold = 0.08
        
        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.max_drawdown = 0
        self.max_equity = initial_balance
        
        # Trading pairs
        self.pairs = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X']
        
        print(f"🎯 Realistic Backtester initialized")
        print(f"💰 Starting balance: ${initial_balance:,.2f}")
        print(f"⚡ Risk per trade: {self.max_risk_per_trade:.1%}")
        print(f"📊 Using rule-based signals for testing")
    
    def get_historical_data(self, pair: str, period: str = "1y") -> pd.DataFrame:
        """Get historical data for backtesting"""
        try:
            ticker = yf.Ticker(pair)
            data = ticker.history(period=period, interval="1h")
            
            if data.empty:
                return None
            
            # Clean data
            data = data.dropna()
            
            print(f"📊 Downloaded {len(data)} hours of data for {pair}")
            return data
            
        except Exception as e:
            print(f"❌ Error getting data for {pair}: {e}")
            return None
    
    def create_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create technical features for signal generation"""
        try:
            df = data.copy()
            
            # Price features
            df['returns'] = df['Close'].pct_change()
            
            # Moving averages
            df['sma_5'] = df['Close'].rolling(5).mean()
            df['sma_10'] = df['Close'].rolling(10).mean()
            df['sma_20'] = df['Close'].rolling(20).mean()
            df['ema_12'] = df['Close'].ewm(span=12).mean()
            df['ema_26'] = df['Close'].ewm(span=26).mean()
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['bb_middle'] = df['Close'].rolling(20).mean()
            df['bb_std'] = df['Close'].rolling(20).std()
            df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
            df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
            df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Volatility
            df['volatility'] = df['returns'].rolling(20).std()
            
            return df
            
        except Exception as e:
            print(f"❌ Error creating features: {e}")
            return None
    
    def generate_signal(self, data: pd.DataFrame, idx: int) -> Optional[Dict]:
        """Generate trading signal based on technical analysis"""
        try:
            if idx < 50:  # Need enough history
                return None
            
            current = data.iloc[idx]
            prev = data.iloc[idx-1]
            
            # Initialize signal
            signal = {
                'pair': None,
                'direction': None,
                'confidence': 0.0,
                'signal_strength': 0.0,
                'price': current['Close']
            }
            
            # Signal conditions
            signals = []
            
            # 1. MACD Crossover
            if (current['macd'] > current['macd_signal'] and 
                prev['macd'] <= prev['macd_signal']):
                signals.append(('buy', 0.75, 0.12))
            elif (current['macd'] < current['macd_signal'] and 
                  prev['macd'] >= prev['macd_signal']):
                signals.append(('sell', 0.75, 0.12))
            
            # 2. RSI Oversold/Overbought
            if current['rsi'] < 30 and prev['rsi'] >= 30:
                signals.append(('buy', 0.80, 0.10))
            elif current['rsi'] > 70 and prev['rsi'] <= 70:
                signals.append(('sell', 0.80, 0.10))
            
            # 3. Bollinger Band Bounce
            if current['bb_position'] < 0.1 and current['Close'] > prev['Close']:
                signals.append(('buy', 0.72, 0.09))
            elif current['bb_position'] > 0.9 and current['Close'] < prev['Close']:
                signals.append(('sell', 0.72, 0.09))
            
            # 4. Moving Average Cross
            if (current['sma_5'] > current['sma_20'] and 
                prev['sma_5'] <= prev['sma_20']):
                signals.append(('buy', 0.71, 0.08))
            elif (current['sma_5'] < current['sma_20'] and 
                  prev['sma_5'] >= prev['sma_20']):
                signals.append(('sell', 0.71, 0.08))
            
            # 5. Momentum + Trend
            if (current['Close'] > current['sma_10'] and 
                current['macd_histogram'] > 0 and 
                current['rsi'] > 50 and current['rsi'] < 70):
                signals.append(('buy', 0.73, 0.085))
            elif (current['Close'] < current['sma_10'] and 
                  current['macd_histogram'] < 0 and 
                  current['rsi'] < 50 and current['rsi'] > 30):
                signals.append(('sell', 0.73, 0.085))
            
            # Select best signal
            if signals:
                # Choose signal with highest confidence
                best_signal = max(signals, key=lambda x: x[1])
                signal['direction'] = best_signal[0]
                signal['confidence'] = best_signal[1]
                signal['signal_strength'] = best_signal[2]
                
                return signal
            
            return None
            
        except Exception as e:
            return None
    
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """Calculate position size based on risk management"""
        # Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_loss)
        
        # Calculate maximum risk amount
        max_risk_amount = self.available_balance * self.max_risk_per_trade
        
        # Calculate position size
        if risk_per_unit > 0:
            position_size = max_risk_amount / risk_per_unit
            # Use full available balance for position value
            max_position_value = self.available_balance
            position_size = min(position_size, max_position_value / entry_price)
        else:
            position_size = self.available_balance / entry_price
        
        return position_size
    
    def calculate_sl_tp_levels(self, pair: str, direction: str, entry_price: float, confidence: float) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels"""
        # Get pip value
        if 'JPY' in pair:
            pip_value = 0.01
        else:
            pip_value = 0.0001
        
        # Adjust SL/TP based on confidence
        if confidence >= 0.80:
            sl_pips = 15
            tp_pips = 30
        elif confidence >= 0.75:
            sl_pips = 18
            tp_pips = 35
        elif confidence >= 0.70:
            sl_pips = 20
            tp_pips = 40
        else:
            sl_pips = 25
            tp_pips = 45
        
        if direction == 'buy':
            stop_loss = entry_price - (sl_pips * pip_value)
            take_profit = entry_price + (tp_pips * pip_value)
        else:  # sell
            stop_loss = entry_price + (sl_pips * pip_value)
            take_profit = entry_price - (tp_pips * pip_value)
        
        return stop_loss, take_profit
    
    def open_trade(self, pair: str, direction: str, entry_price: float, confidence: float, timestamp: datetime) -> bool:
        """Open a new trade"""
        if self.open_position is not None:
            return False  # Already have open position
        
        if self.available_balance <= 0:
            return False  # No available balance
        
        # Calculate SL/TP
        stop_loss, take_profit = self.calculate_sl_tp_levels(pair, direction, entry_price, confidence)
        
        # Calculate position size
        position_size = self.calculate_position_size(entry_price, stop_loss)
        
        if position_size <= 0:
            return False
        
        # Lock the balance (realistic - all balance is tied up)
        position_value = position_size * entry_price
        self.available_balance = 0  # All balance is now tied up in this trade
        
        # Create position
        self.open_position = {
            'pair': pair,
            'direction': direction,
            'entry_price': entry_price,
            'position_size': position_size,
            'position_value': position_value,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'confidence': confidence,
            'entry_time': timestamp,
            'entry_balance': self.current_balance
        }
        
        print(f"📈 OPENED {direction.upper()} {pair} @ {entry_price:.5f}")
        print(f"   💰 Position Value: ${position_value:,.2f}")
        print(f"   🛑 Stop Loss: {stop_loss:.5f}")
        print(f"   🎯 Take Profit: {take_profit:.5f}")
        print(f"   💯 Confidence: {confidence:.1%}")
        
        return True
    
    def close_trade(self, exit_price: float, exit_reason: str, timestamp: datetime) -> bool:
        """Close the open trade"""
        if self.open_position is None:
            return False
        
        # Calculate P&L
        position = self.open_position
        position_size = position['position_size']
        entry_price = position['entry_price']
        direction = position['direction']
        
        if direction == 'buy':
            pnl = (exit_price - entry_price) * position_size
        else:  # sell
            pnl = (entry_price - exit_price) * position_size
        
        # Update balance
        new_balance = self.current_balance + pnl
        self.current_balance = new_balance
        self.available_balance = new_balance  # Balance is now available again
        
        # Update max equity and drawdown
        if new_balance > self.max_equity:
            self.max_equity = new_balance
        
        current_drawdown = (self.max_equity - new_balance) / self.max_equity
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
        
        # Record trade
        trade_record = {
            'pair': position['pair'],
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position_size': position_size,
            'entry_time': position['entry_time'],
            'exit_time': timestamp,
            'exit_reason': exit_reason,
            'pnl': pnl,
            'pnl_pct': (pnl / position['position_value']) * 100,
            'confidence': position['confidence'],
            'entry_balance': position['entry_balance'],
            'exit_balance': new_balance,
            'duration_hours': (timestamp - position['entry_time']).total_seconds() / 3600
        }
        
        self.trades_history.append(trade_record)
        self.total_trades += 1
        
        if pnl > 0:
            self.winning_trades += 1
            print(f"✅ CLOSED {direction.upper()} {position['pair']} @ {exit_price:.5f} ({exit_reason})")
            print(f"   💰 P&L: +${pnl:,.2f} ({trade_record['pnl_pct']:+.2f}%)")
        else:
            self.losing_trades += 1
            print(f"❌ CLOSED {direction.upper()} {position['pair']} @ {exit_price:.5f} ({exit_reason})")
            print(f"   💰 P&L: ${pnl:,.2f} ({trade_record['pnl_pct']:+.2f}%)")
        
        print(f"   💼 New Balance: ${new_balance:,.2f}")
        
        # Clear position
        self.open_position = None
        
        return True
    
    def check_exit_conditions(self, current_data: pd.Series) -> Tuple[bool, str]:
        """Check if current position should be closed"""
        if self.open_position is None:
            return False, ""
        
        position = self.open_position
        current_price = current_data['Close']
        direction = position['direction']
        stop_loss = position['stop_loss']
        take_profit = position['take_profit']
        
        if direction == 'buy':
            if current_price <= stop_loss:
                return True, "Stop Loss"
            elif current_price >= take_profit:
                return True, "Take Profit"
        else:  # sell
            if current_price >= stop_loss:
                return True, "Stop Loss"
            elif current_price <= take_profit:
                return True, "Take Profit"
        
        return False, ""
    
    def run_backtest(self, start_date: str = "2024-01-01", end_date: str = "2024-12-31"):
        """Run the realistic backtest"""
        print(f"\n🚀 Starting Realistic Backtest")
        print(f"📅 Period: {start_date} to {end_date}")
        print("=" * 50)
        
        # Get data for all pairs
        all_data = {}
        for pair in self.pairs:
            data = self.get_historical_data(pair, period="1y")
            if data is not None:
                # Filter by date range
                data = data.loc[start_date:end_date]
                # Add technical features
                data = self.create_technical_features(data)
                if data is not None:
                    all_data[pair] = data
                    print(f"📊 {pair}: {len(data)} data points with features")
        
        if not all_data:
            print("❌ No data available for backtesting")
            return
        
        # Create unified timeline
        all_timestamps = set()
        for data in all_data.values():
            all_timestamps.update(data.index)
        
        timeline = sorted(all_timestamps)
        print(f"⏰ Timeline: {len(timeline)} total timestamps")
        
        # Run simulation
        signals_checked = 0
        for i, timestamp in enumerate(timeline):
            # Record equity curve
            self.equity_curve.append({
                'timestamp': timestamp,
                'balance': self.current_balance,
                'available_balance': self.available_balance,
                'in_position': self.open_position is not None
            })
            
            # Check exit conditions first
            if self.open_position is not None:
                pair = self.open_position['pair']
                if pair in all_data and timestamp in all_data[pair].index:
                    current_data = all_data[pair].loc[timestamp]
                    should_exit, exit_reason = self.check_exit_conditions(current_data)
                    
                    if should_exit:
                        self.close_trade(current_data['Close'], exit_reason, timestamp)
            
            # Look for new entry signals (only if no open position)
            if self.open_position is None and self.available_balance > 0:
                for pair in self.pairs:
                    if pair not in all_data or timestamp not in all_data[pair].index:
                        continue
                    
                    # Get index for signal generation
                    data_idx = all_data[pair].index.get_loc(timestamp)
                    if data_idx < 50:  # Need enough history
                        continue
                    
                    # Generate signal
                    signal = self.generate_signal(all_data[pair], data_idx)
                    signals_checked += 1
                    
                    if signal is None:
                        continue
                    
                    # Check if signal meets criteria
                    if (signal['confidence'] >= self.confidence_threshold and 
                        signal['signal_strength'] >= self.signal_strength_threshold):
                        
                        current_price = all_data[pair].loc[timestamp, 'Close']
                        
                        # Open trade
                        if self.open_trade(pair, signal['direction'], current_price, 
                                         signal['confidence'], timestamp):
                            break  # Only one position at a time
            
            # Progress update
            if i % 500 == 0:
                progress = (i / len(timeline)) * 100
                print(f"📈 Progress: {progress:.1f}% - Balance: ${self.current_balance:,.2f} - Signals checked: {signals_checked}")
        
        # Close any remaining position
        if self.open_position is not None:
            pair = self.open_position['pair']
            if pair in all_data:
                last_price = all_data[pair]['Close'].iloc[-1]
                self.close_trade(last_price, "End of backtest", timeline[-1])
        
        print(f"\n✅ Backtest completed! Checked {signals_checked} potential signals")
        self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive backtest report"""
        print("\n" + "=" * 60)
        print("📊 REALISTIC BACKTEST RESULTS")
        print("=" * 60)
        
        # Basic metrics
        total_return = self.current_balance - self.initial_balance
        total_return_pct = (total_return / self.initial_balance) * 100
        
        print(f"💰 Initial Balance: ${self.initial_balance:,.2f}")
        print(f"💰 Final Balance: ${self.current_balance:,.2f}")
        print(f"📈 Total Return: ${total_return:,.2f} ({total_return_pct:+.2f}%)")
        print(f"📉 Maximum Drawdown: {self.max_drawdown:.2%}")
        
        # Trade statistics
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        print(f"\n📊 Trading Statistics:")
        print(f"   Total Trades: {self.total_trades}")
        print(f"   Winning Trades: {self.winning_trades}")
        print(f"   Losing Trades: {self.losing_trades}")
        print(f"   Win Rate: {win_rate:.1f}%")
        
        if self.trades_history:
            # P&L analysis
            pnls = [trade['pnl'] for trade in self.trades_history]
            avg_win = np.mean([pnl for pnl in pnls if pnl > 0]) if any(pnl > 0 for pnl in pnls) else 0
            avg_loss = np.mean([pnl for pnl in pnls if pnl < 0]) if any(pnl < 0 for pnl in pnls) else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            
            print(f"\n💹 P&L Analysis:")
            print(f"   Average Win: ${avg_win:,.2f}")
            print(f"   Average Loss: ${avg_loss:,.2f}")
            print(f"   Profit Factor: {profit_factor:.2f}")
            print(f"   Best Trade: ${max(pnls):,.2f}")
            print(f"   Worst Trade: ${min(pnls):,.2f}")
            
            # Trade duration
            durations = [trade['duration_hours'] for trade in self.trades_history]
            avg_duration = np.mean(durations)
            
            print(f"\n⏱️ Duration Analysis:")
            print(f"   Average Duration: {avg_duration:.1f} hours")
            print(f"   Shortest Trade: {min(durations):.1f} hours")
            print(f"   Longest Trade: {max(durations):.1f} hours")
            
            # Show sample trades
            print(f"\n📋 Sample Trades:")
            for i, trade in enumerate(self.trades_history[:5]):
                print(f"   Trade {i+1}: {trade['direction'].upper()} {trade['pair']} | "
                      f"P&L: ${trade['pnl']:,.2f} | Duration: {trade['duration_hours']:.1f}h | "
                      f"Reason: {trade['exit_reason']}")
        
        # Save results
        self.save_results()
        
        print(f"\n🎯 Realistic Trading Simulation Complete!")
        print(f"📊 Key insight: Balance only updates when positions close")
        print(f"💰 Risk management: Full balance committed per trade")
    
    def save_results(self):
        """Save backtest results to files"""
        # Save trades history
        if self.trades_history:
            trades_df = pd.DataFrame(self.trades_history)
            trades_df.to_csv('realistic_backtest_trades.csv', index=False)
            print(f"\n💾 Trades saved to: realistic_backtest_trades.csv")
        
        # Save equity curve
        if self.equity_curve:
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df.to_csv('realistic_backtest_equity.csv', index=False)
            print(f"💾 Equity curve saved to: realistic_backtest_equity.csv")


if __name__ == "__main__":
    # Run realistic backtest
    backtester = SimpleRealisticBacktester(initial_balance=1500.0)
    backtester.run_backtest(start_date="2024-06-01", end_date="2024-12-31")
