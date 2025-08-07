"""
QUICK ML BACKTESTER - Simplified version for faster testing
Uses your enhanced ML models with optimized prediction for speed
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import pickle
import ta
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

class QuickMLBacktester:
    def __init__(self, initial_balance: float = 1500.0):
        """Initialize quick ML backtester"""
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        
        # Trading state
        self.trades_history = []
        self.models = {}
        self.scalers = {}
        
        # Settings
        self.confidence_threshold = 0.75
        self.signal_threshold = 0.6
        
        print(f"⚡ Quick ML Backtester initialized")
        print(f"💰 Starting balance: ${initial_balance:,.2f}")
        
        # Load models
        self.load_models()
    
    def load_models(self):
        """Load ML models quickly"""
        print(f"\n🤖 Loading ML Models...")
        
        for pair in ['EURUSD_X', 'GBPUSD_X', 'USDJPY_X']:
            try:
                with open(f"enhanced_models/{pair}_enhanced_loss_learning.pkl", 'rb') as f:
                    model_data = pickle.load(f)
                
                self.models[pair] = model_data['model']
                self.scalers[pair] = model_data['scaler']
                print(f"✅ {pair}")
                
            except Exception as e:
                print(f"❌ {pair}: {e}")
        
        print(f"🎯 {len(self.models)} models loaded")
    
    def get_data(self, pair: str) -> pd.DataFrame:
        """Get and prepare data"""
        ticker = yf.Ticker(pair)
        data = ticker.history(period="6mo", interval="1h")  # Shorter period for speed
        
        # Add features
        df = data.copy()
        df['returns'] = df['Close'].pct_change()
        df['sma_20'] = df['Close'].rolling(20).mean()
        df['sma_50'] = df['Close'].rolling(50).mean()
        df['ema_12'] = df['Close'].ewm(span=12).mean()
        
        # MACD
        ema_26 = df['Close'].ewm(span=26).mean()
        df['macd'] = df['ema_12'] - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        bb_middle = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['bb_upper'] = bb_middle + (bb_std * 2)
        df['bb_lower'] = bb_middle - (bb_std * 2)
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ADX (simplified)
        df['adx'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)
        df['volatility'] = df['returns'].rolling(20).std()
        
        # Boolean features
        df['trend_up'] = (df['Close'] > df['sma_20']).astype(float)
        df['trend_down'] = (df['Close'] < df['sma_20']).astype(float)
        df['strong_trend'] = (df['adx'] > 25).astype(float)
        df['oversold'] = (df['rsi'] < 30).astype(float)
        df['overbought'] = (df['rsi'] > 70).astype(float)
        df['macd_bullish'] = (df['macd'] > df['macd_signal']).astype(float)
        df['macd_bearish'] = (df['macd'] < df['macd_signal']).astype(float)
        
        return df.dropna()
    
    def predict_signal(self, data: pd.DataFrame, idx: int, pair: str) -> dict:
        """Quick ML prediction"""
        model_pair = pair.replace('=', '_')
        
        if model_pair not in self.models or idx < 30:
            return None
        
        # Get sequence
        sequence = data.iloc[idx-30:idx]
        
        # Feature columns
        features = ['Close', 'returns', 'volatility', 'sma_20', 'sma_50', 'ema_12',
                   'macd', 'macd_signal', 'macd_histogram', 'rsi', 'bb_position', 'adx',
                   'trend_up', 'trend_down', 'strong_trend', 'oversold', 'overbought',
                   'macd_bullish', 'macd_bearish']
        
        available_features = [f for f in features if f in sequence.columns]
        
        if len(available_features) < 15:
            return None
        
        # Prepare input
        X = sequence[available_features].values
        X = np.nan_to_num(X)
        
        # Scale
        scaler = self.scalers[model_pair]
        X_scaled = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(1, 30, -1)
        
        # Predict using call method (faster)
        try:
            outputs = self.models[model_pair](X_scaled, training=False)
            signal_prob = float(outputs[0][0][0])
            confidence = float(outputs[1][0][0])
            reward = float(outputs[2][0][0])
            
            current = data.iloc[idx]
            
            # Determine direction
            if signal_prob > self.signal_threshold and confidence > self.confidence_threshold:
                if current['macd_bullish'] and reward > 0:
                    direction = 'buy'
                elif current['macd_bearish'] and reward > 0:
                    direction = 'sell'
                else:
                    return None
                
                return {
                    'pair': pair,
                    'direction': direction,
                    'confidence': confidence,
                    'signal_prob': signal_prob,
                    'reward': reward,
                    'price': current['Close']
                }
            
        except Exception as e:
            return None
        
        return None
    
    def simulate_trade(self, signal: dict, data: pd.DataFrame, start_idx: int) -> dict:
        """Simulate trade outcome"""
        direction = signal['direction']
        entry_price = signal['price']
        pair = signal['pair']
        
        # Calculate SL/TP
        if 'JPY' in pair:
            pip_value = 0.01
        else:
            pip_value = 0.0001
        
        confidence = signal['confidence']
        if confidence >= 0.85:
            sl_pips, tp_pips = 15, 30
        elif confidence >= 0.80:
            sl_pips, tp_pips = 18, 35
        else:
            sl_pips, tp_pips = 20, 40
        
        if direction == 'buy':
            sl_price = entry_price - (sl_pips * pip_value)
            tp_price = entry_price + (tp_pips * pip_value)
        else:
            sl_price = entry_price + (sl_pips * pip_value)
            tp_price = entry_price - (tp_pips * pip_value)
        
        # Check outcome over next 20 bars
        for i in range(1, min(21, len(data) - start_idx)):
            bar = data.iloc[start_idx + i]
            
            if direction == 'buy':
                if bar['Low'] <= sl_price:
                    pnl = (sl_price - entry_price) / entry_price
                    return {'result': 'loss', 'pnl_pct': pnl, 'bars': i, 'exit_price': sl_price}
                elif bar['High'] >= tp_price:
                    pnl = (tp_price - entry_price) / entry_price
                    return {'result': 'win', 'pnl_pct': pnl, 'bars': i, 'exit_price': tp_price}
            else:
                if bar['High'] >= sl_price:
                    pnl = (entry_price - sl_price) / entry_price
                    return {'result': 'loss', 'pnl_pct': pnl, 'bars': i, 'exit_price': sl_price}
                elif bar['Low'] <= tp_price:
                    pnl = (entry_price - tp_price) / entry_price
                    return {'result': 'win', 'pnl_pct': pnl, 'bars': i, 'exit_price': tp_price}
        
        # Timeout
        final_price = data.iloc[start_idx + min(20, len(data) - start_idx - 1)]['Close']
        if direction == 'buy':
            pnl = (final_price - entry_price) / entry_price
        else:
            pnl = (entry_price - final_price) / entry_price
        
        return {'result': 'timeout', 'pnl_pct': pnl, 'bars': 20, 'exit_price': final_price}
    
    def run_quick_backtest(self):
        """Run quick ML backtest"""
        print(f"\n🚀 Running Quick ML Backtest")
        print("=" * 40)
        
        all_data = {}
        for pair in ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X']:
            data = self.get_data(pair)
            if data is not None:
                all_data[pair] = data
                print(f"📊 {pair}: {len(data)} bars")
        
        # Run backtest
        total_trades = 0
        winning_trades = 0
        total_pnl = 0
        
        balance = self.initial_balance
        
        for pair, data in all_data.items():
            print(f"\n🔍 Testing {pair}...")
            
            pair_trades = 0
            pair_wins = 0
            pair_pnl = 0
            
            # Sample every 10th bar for speed
            for i in range(50, len(data) - 25, 10):
                signal = self.predict_signal(data, i, pair)
                
                if signal is None:
                    continue
                
                # Simulate trade
                outcome = self.simulate_trade(signal, data, i)
                
                # Calculate P&L in dollars
                position_value = balance * 0.95  # Use 95% of balance
                dollar_pnl = position_value * outcome['pnl_pct']
                
                balance += dollar_pnl
                
                # Record trade
                trade = {
                    'pair': pair,
                    'direction': signal['direction'],
                    'confidence': signal['confidence'],
                    'signal_prob': signal['signal_prob'],
                    'result': outcome['result'],
                    'pnl_pct': outcome['pnl_pct'],
                    'pnl_dollars': dollar_pnl,
                    'balance': balance
                }
                
                self.trades_history.append(trade)
                
                total_trades += 1
                pair_trades += 1
                total_pnl += dollar_pnl
                pair_pnl += dollar_pnl
                
                if outcome['result'] == 'win':
                    winning_trades += 1
                    pair_wins += 1
                
                # Progress
                if pair_trades % 5 == 0:
                    print(f"   {pair_trades} trades, {pair_wins} wins, ${pair_pnl:,.0f} P&L")
            
            pair_win_rate = (pair_wins / pair_trades * 100) if pair_trades > 0 else 0
            print(f"   {pair} Summary: {pair_trades} trades, {pair_win_rate:.1f}% win rate, ${pair_pnl:,.0f} P&L")
        
        # Final results
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_return = (balance - self.initial_balance) / self.initial_balance * 100
        
        print(f"\n🎯 QUICK ML BACKTEST RESULTS")
        print("=" * 40)
        print(f"💰 Initial Balance: ${self.initial_balance:,.0f}")
        print(f"💰 Final Balance: ${balance:,.0f}")
        print(f"📈 Total Return: {total_return:+.1f}%")
        print(f"📊 Total Trades: {total_trades}")
        print(f"✅ Win Rate: {win_rate:.1f}%")
        print(f"💵 Total P&L: ${total_pnl:,.0f}")
        
        if self.trades_history:
            wins = [t for t in self.trades_history if t['result'] == 'win']
            losses = [t for t in self.trades_history if t['result'] == 'loss']
            
            if wins and losses:
                avg_win = np.mean([t['pnl_dollars'] for t in wins])
                avg_loss = np.mean([t['pnl_dollars'] for t in losses])
                profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
                
                print(f"📈 Average Win: ${avg_win:,.0f}")
                print(f"📉 Average Loss: ${avg_loss:,.0f}")
                print(f"⚖️ Profit Factor: {profit_factor:.2f}")
            
            # Show model performance
            avg_confidence = np.mean([t['confidence'] for t in self.trades_history])
            avg_signal_prob = np.mean([t['signal_prob'] for t in self.trades_history])
            
            print(f"\n🤖 ML Performance:")
            print(f"   Average Confidence: {avg_confidence:.1%}")
            print(f"   Average Signal Prob: {avg_signal_prob:.1%}")
            
            # Performance comparison
            if total_return > 0:
                print(f"\n🏆 ML MODELS PERFORMED WELL!")
                print(f"   Positive return: {total_return:+.1f}%")
                print(f"   Much better than rule-based: -3.81%")
            else:
                print(f"\n📊 Performance Analysis:")
                print(f"   Return: {total_return:+.1f}%")
                print(f"   Compared to rule-based: -3.81%")
        
        return balance


if __name__ == "__main__":
    backtester = QuickMLBacktester(initial_balance=1500.0)
    final_balance = backtester.run_quick_backtest()
