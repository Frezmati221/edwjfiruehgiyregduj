"""
REALISTIC BACKTESTING SYSTEM
Simulates real trading with full balance usage and proper position management
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class RealisticBacktester:
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
        
        # Load models and scalers
        self.models = {}
        self.scalers = {}
        self.pairs = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X']
        
        print(f"🎯 Realistic Backtester initialized")
        print(f"💰 Starting balance: ${initial_balance:,.2f}")
        print(f"⚡ Risk per trade: {self.max_risk_per_trade:.1%}")
    
    def load_models_and_scalers(self):
        """Load trained models and scalers"""
        for pair in self.pairs:
            try:
                # Load model (filename uses = format)
                model_path = f"enhanced_models/{pair}_best_model.h5"
                self.models[pair] = load_model(model_path)
                
                # Load scaler (filename uses _ format) 
                scaler_path = f"enhanced_models/{pair.replace('=', '_')}_enhanced_loss_learning.pkl"
                with open(scaler_path, 'rb') as f:
                    self.scalers[pair] = pickle.load(f)
                
                print(f"✅ Loaded model and scaler for {pair}")
                
            except Exception as e:
                print(f"❌ Failed to load {pair}: {e}")
    
    def get_historical_data(self, pair: str, period: str = "2y") -> pd.DataFrame:
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
    
    def create_features(self, data: pd.DataFrame) -> np.ndarray:
        """Create technical features for prediction"""
        try:
            df = data.copy()
            
            # Price features
            df['returns'] = df['Close'].pct_change()
            df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
            
            # Moving averages
            df['sma_5'] = df['Close'].rolling(5).mean()
            df['sma_10'] = df['Close'].rolling(10).mean()
            df['sma_20'] = df['Close'].rolling(20).mean()
            df['ema_5'] = df['Close'].ewm(span=5).mean()
            df['ema_10'] = df['Close'].ewm(span=10).mean()
            
            # Volatility
            df['volatility'] = df['returns'].rolling(20).std()
            df['atr'] = self.calculate_atr(df, 14)
            
            # RSI
            df['rsi'] = self.calculate_rsi(df['Close'], 14)
            
            # MACD
            macd_line, macd_signal = self.calculate_macd(df['Close'])
            df['macd'] = macd_line
            df['macd_signal'] = macd_signal
            df['macd_histogram'] = macd_line - macd_signal
            
            # Bollinger Bands
            df['bb_upper'], df['bb_lower'] = self.calculate_bollinger_bands(df['Close'], 20, 2)
            df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Price position relative to MAs
            df['price_vs_sma20'] = (df['Close'] - df['sma_20']) / df['sma_20']
            df['price_vs_ema10'] = (df['Close'] - df['ema_10']) / df['ema_10']
            
            # Volume features (if available)
            if 'Volume' in df.columns and not df['Volume'].isna().all():
                df['volume_sma'] = df['Volume'].rolling(20).mean()
                df['volume_ratio'] = df['Volume'] / df['volume_sma']
            else:
                df['volume_ratio'] = 1.0
            
            # Time features
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            
            # Select features
            feature_columns = [
                'returns', 'log_returns', 'volatility', 'atr', 'rsi',
                'macd', 'macd_signal', 'macd_histogram', 'bb_position',
                'price_vs_sma20', 'price_vs_ema10', 'volume_ratio',
                'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'
            ]
            
            features = df[feature_columns].dropna()
            return features.values
            
        except Exception as e:
            print(f"❌ Error creating features: {e}")
            return None
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        return macd_line, macd_signal
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close_prev = np.abs(df['High'] - df['Close'].shift())
        low_close_prev = np.abs(df['Low'] - df['Close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        return true_range.rolling(period).mean()
    
    def make_prediction(self, pair: str, features: np.ndarray) -> Optional[Dict]:
        """Make prediction using trained model"""
        try:
            if pair not in self.models or pair not in self.scalers:
                return None
            
            # Use last row of features
            if len(features) < 1:
                return None
            
            feature_row = features[-1:].reshape(1, -1)
            
            # Scale features
            scaled_features = self.scalers[pair].transform(feature_row)
            
            # Reshape for LSTM (assuming sequence length of 1 for simplicity)
            lstm_input = scaled_features.reshape(1, 1, -1)
            
            # Make prediction
            prediction = self.models[pair].predict(lstm_input, verbose=0)[0][0]
            
            # Calculate confidence (distance from 0.5)
            confidence = abs(prediction - 0.5) * 2
            
            # Determine signal
            signal_strength = abs(prediction - 0.5)
            direction = 'buy' if prediction > 0.5 else 'sell'
            
            return {
                'pair': pair,
                'prediction': prediction,
                'confidence': confidence,
                'signal_strength': signal_strength,
                'direction': direction
            }
            
        except Exception as e:
            print(f"❌ Error making prediction for {pair}: {e}")
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
        elif confidence >= 0.70:
            sl_pips = 20
            tp_pips = 35
        else:
            sl_pips = 25
            tp_pips = 40
        
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
        
        # Lock the balance
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
    
    def run_backtest(self, start_date: str = "2023-01-01", end_date: str = "2024-12-31"):
        """Run the realistic backtest"""
        print(f"\n🚀 Starting Realistic Backtest")
        print(f"📅 Period: {start_date} to {end_date}")
        print("=" * 50)
        
        # Load models
        self.load_models_and_scalers()
        
        # Get data for all pairs
        all_data = {}
        for pair in self.pairs:
            data = self.get_historical_data(pair, period="2y")
            if data is not None:
                # Filter by date range
                data = data.loc[start_date:end_date]
                all_data[pair] = data
                print(f"📊 {pair}: {len(data)} data points")
        
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
                    
                    # Get enough data for features
                    end_idx = all_data[pair].index.get_loc(timestamp)
                    if end_idx < 50:  # Need enough history
                        continue
                    
                    start_idx = max(0, end_idx - 100)
                    data_slice = all_data[pair].iloc[start_idx:end_idx + 1]
                    
                    # Create features
                    features = self.create_features(data_slice)
                    if features is None or len(features) < 1:
                        continue
                    
                    # Make prediction
                    prediction = self.make_prediction(pair, features)
                    if prediction is None:
                        continue
                    
                    # Check if signal meets criteria
                    if (prediction['confidence'] >= self.confidence_threshold and 
                        prediction['signal_strength'] >= self.signal_strength_threshold):
                        
                        current_price = all_data[pair].loc[timestamp, 'Close']
                        
                        # Open trade
                        if self.open_trade(pair, prediction['direction'], current_price, 
                                         prediction['confidence'], timestamp):
                            break  # Only one position at a time
            
            # Progress update
            if i % 1000 == 0:
                progress = (i / len(timeline)) * 100
                print(f"📈 Progress: {progress:.1f}% - Balance: ${self.current_balance:,.2f}")
        
        # Close any remaining position
        if self.open_position is not None:
            pair = self.open_position['pair']
            if pair in all_data:
                last_price = all_data[pair]['Close'].iloc[-1]
                self.close_trade(last_price, "End of backtest", timeline[-1])
        
        print("\n✅ Backtest completed!")
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
            
            # Pair analysis
            pair_stats = {}
            for trade in self.trades_history:
                pair = trade['pair']
                if pair not in pair_stats:
                    pair_stats[pair] = {'trades': 0, 'wins': 0, 'total_pnl': 0}
                
                pair_stats[pair]['trades'] += 1
                pair_stats[pair]['total_pnl'] += trade['pnl']
                if trade['pnl'] > 0:
                    pair_stats[pair]['wins'] += 1
            
            print(f"\n🌍 Pair Performance:")
            for pair, stats in pair_stats.items():
                pair_win_rate = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
                print(f"   {pair}: {stats['trades']} trades, {pair_win_rate:.1f}% win rate, ${stats['total_pnl']:,.2f} P&L")
        
        # Save results
        self.save_results()
        
        # Generate plots
        self.plot_results()
    
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
        
        # Save summary
        summary = {
            'initial_balance': self.initial_balance,
            'final_balance': self.current_balance,
            'total_return': self.current_balance - self.initial_balance,
            'total_return_pct': ((self.current_balance - self.initial_balance) / self.initial_balance) * 100,
            'max_drawdown': self.max_drawdown,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        }
        
        with open('realistic_backtest_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 Summary saved to: realistic_backtest_summary.json")
    
    def plot_results(self):
        """Generate performance plots"""
        if not self.equity_curve:
            return
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Realistic Backtest Results', fontsize=16, fontweight='bold')
        
        # Equity curve
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
        
        ax1.plot(equity_df['timestamp'], equity_df['balance'], linewidth=2, color='blue')
        ax1.set_title('Account Balance Over Time', fontweight='bold')
        ax1.set_ylabel('Balance ($)')
        ax1.grid(True, alpha=0.3)
        ax1.ticklabel_format(style='plain', axis='y')
        
        # Drawdown curve
        equity_df['max_balance'] = equity_df['balance'].expanding().max()
        equity_df['drawdown'] = (equity_df['max_balance'] - equity_df['balance']) / equity_df['max_balance'] * 100
        
        ax2.fill_between(equity_df['timestamp'], equity_df['drawdown'], 0, color='red', alpha=0.3)
        ax2.plot(equity_df['timestamp'], equity_df['drawdown'], color='red', linewidth=1)
        ax2.set_title('Drawdown (%)', fontweight='bold')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        if self.trades_history:
            # P&L distribution
            pnls = [trade['pnl'] for trade in self.trades_history]
            ax3.hist(pnls, bins=20, alpha=0.7, color='green', edgecolor='black')
            ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7)
            ax3.set_title('P&L Distribution', fontweight='bold')
            ax3.set_xlabel('P&L ($)')
            ax3.set_ylabel('Frequency')
            ax3.grid(True, alpha=0.3)
            
            # Monthly returns
            trades_df = pd.DataFrame(self.trades_history)
            trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
            trades_df['year_month'] = trades_df['exit_time'].dt.to_period('M')
            monthly_pnl = trades_df.groupby('year_month')['pnl'].sum()
            
            colors = ['green' if x >= 0 else 'red' for x in monthly_pnl.values]
            ax4.bar(range(len(monthly_pnl)), monthly_pnl.values, color=colors, alpha=0.7)
            ax4.set_title('Monthly P&L', fontweight='bold')
            ax4.set_xlabel('Month')
            ax4.set_ylabel('P&L ($)')
            ax4.grid(True, alpha=0.3)
            
            # Set x-axis labels for monthly chart
            if len(monthly_pnl) <= 12:
                ax4.set_xticks(range(len(monthly_pnl)))
                ax4.set_xticklabels([str(x) for x in monthly_pnl.index], rotation=45)
        
        plt.tight_layout()
        plt.savefig('realistic_backtest_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Charts saved to: realistic_backtest_results.png")


if __name__ == "__main__":
    # Run realistic backtest
    backtester = RealisticBacktester(initial_balance=1500.0)
    backtester.run_backtest(start_date="2024-01-01", end_date="2024-12-31")
