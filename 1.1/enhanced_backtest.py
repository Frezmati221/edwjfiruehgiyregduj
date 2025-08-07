"""
ENHANCED MODEL BACKTESTING SYSTEM
Compare 75-epoch enhanced models vs 30-epoch baseline models
"""

import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import ta
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class EnhancedBacktester:
    """Comprehensive backtesting for enhanced vs baseline models"""
    
    def __init__(self):
        self.enhanced_models = {}
        self.baseline_models = {}
        self.results = {}
        
        print("🔬 ENHANCED MODEL BACKTESTING SYSTEM")
        print("Comparing 75-epoch Enhanced vs 30-epoch Baseline models")
        print("=" * 60)
    
    def load_models(self):
        """Load both enhanced and baseline models"""
        
        pairs = ['EURUSD_X', 'GBPUSD_X', 'USDJPY_X']
        
        print("📂 Loading models...")
        
        # Load enhanced models (75 epochs)
        for pair in pairs:
            try:
                with open(f'enhanced_models/{pair}_enhanced_loss_learning.pkl', 'rb') as f:
                    self.enhanced_models[pair] = pickle.load(f)
                print(f"✅ Enhanced {pair} loaded")
            except Exception as e:
                print(f"❌ Failed to load enhanced {pair}: {e}")
        
        # Load baseline models (30 epochs)
        for pair in pairs:
            baseline_files = [
                f'advanced_models/{pair}_focused_loss_learning.pkl',
                f'models/{pair}_loss_learning.pkl',
                f'trained_models/{pair}_model.pkl'
            ]
            
            loaded = False
            for filepath in baseline_files:
                try:
                    with open(filepath, 'rb') as f:
                        self.baseline_models[pair] = pickle.load(f)
                    print(f"✅ Baseline {pair} loaded from {filepath}")
                    loaded = True
                    break
                except:
                    continue
            
            if not loaded:
                print(f"❌ Failed to load baseline {pair}")
        
        print(f"📊 Enhanced models: {len(self.enhanced_models)}")
        print(f"📊 Baseline models: {len(self.baseline_models)}")
    
    def download_test_data(self, pair, period='3mo'):
        """Download fresh test data"""
        
        try:
            # Convert pair format for yfinance
            yahoo_pair = pair.replace('_', '=')
            
            ticker = yf.Ticker(yahoo_pair)
            data = ticker.history(period=period, interval='1h')
            
            if len(data) < 100:
                print(f"❌ Insufficient test data for {pair}")
                return None
            
            print(f"✅ Downloaded {len(data)} hours of test data for {pair}")
            return data
            
        except Exception as e:
            print(f"❌ Failed to download {pair}: {e}")
            return None
    
    def create_features(self, data):
        """Create identical features as training"""
        
        df = data.copy()
        
        # Basic price features
        df['returns'] = df['Close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        
        # Trend indicators
        df['sma_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['ema_12'] = ta.trend.ema_indicator(df['Close'], window=12)
        
        # MACD system
        macd = ta.trend.MACD(df['Close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
        # Momentum
        df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['Close'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Trend strength
        df['adx'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)
        
        # Market structure (boolean features)
        df['trend_up'] = (df['Close'] > df['sma_20']).astype(float)
        df['trend_down'] = (df['Close'] < df['sma_20']).astype(float)
        df['strong_trend'] = (df['adx'] > 25).astype(float)
        df['weak_trend'] = (df['adx'] < 20).astype(float)
        
        # Momentum conditions
        df['oversold'] = (df['rsi'] < 30).astype(float)
        df['overbought'] = (df['rsi'] > 70).astype(float)
        df['neutral_rsi'] = ((df['rsi'] >= 40) & (df['rsi'] <= 60)).astype(float)
        
        # MACD signals
        df['macd_bullish'] = (df['macd'] > df['macd_signal']).astype(float)
        df['macd_bearish'] = (df['macd'] < df['macd_signal']).astype(float)
        
        return df.dropna()
    
    def make_predictions(self, model_data, features_df, pair):
        """Make predictions using model"""
        
        try:
            model = model_data['model']
            scaler = model_data['scaler']
            
            # Feature columns matching training
            feature_cols = ['Close', 'returns', 'volatility', 'sma_20', 'sma_50', 'ema_12',
                           'macd', 'macd_signal', 'macd_histogram', 'rsi', 'bb_position', 'adx',
                           'trend_up', 'trend_down', 'strong_trend', 'oversold', 'overbought',
                           'macd_bullish', 'macd_bearish']
            
            available_cols = [col for col in feature_cols if col in features_df.columns]
            
            if len(available_cols) < 15:
                print(f"❌ Insufficient features for {pair}")
                return None
            
            predictions = []
            sequence_length = 30
            
            for i in range(sequence_length, len(features_df)):
                # Get sequence
                sequence = features_df.iloc[i-sequence_length:i][available_cols].values
                
                # Scale features
                sequence_scaled = scaler.transform(sequence.reshape(-1, sequence.shape[-1])).reshape(1, sequence.shape[0], sequence.shape[1])
                
                # Predict
                pred = model.predict(sequence_scaled, verbose=0)
                
                predictions.append({
                    'timestamp': features_df.index[i],
                    'signal_prob': float(pred[0][0][0]),
                    'confidence': float(pred[1][0][0]),
                    'reward_estimate': float(pred[2][0][0]),
                    'close': features_df.iloc[i]['Close']
                })
            
            return pd.DataFrame(predictions)
            
        except Exception as e:
            print(f"❌ Prediction failed for {pair}: {e}")
            return None
    
    def simulate_trading(self, predictions, pair, model_type):
        """Simulate realistic trading with the predictions"""
        
        print(f"\n💹 Simulating {model_type} trading for {pair}...")
        
        if predictions is None or len(predictions) == 0:
            return None
        
        # Debug predictions
        print(f"📊 Predictions summary:")
        print(f"   Signal prob range: {predictions['signal_prob'].min():.3f} - {predictions['signal_prob'].max():.3f}")
        print(f"   Confidence range: {predictions['confidence'].min():.3f} - {predictions['confidence'].max():.3f}")
        print(f"   Reward estimate range: {predictions['reward_estimate'].min():.3f} - {predictions['reward_estimate'].max():.3f}")
        
        # Trading parameters
        initial_balance = 10000
        balance = initial_balance
        position_size_pct = 0.02  # 2% risk per trade
        max_daily_risk = 0.06     # 6% max daily risk
        
        # Pair-specific parameters
        if 'JPY' in pair:
            pip_value = 0.01
            spread = 1.5 * pip_value  # 1.5 pips spread
            tp_pips = 40 * pip_value
            sl_pips = 20 * pip_value
        else:
            pip_value = 0.0001
            spread = 1.5 * pip_value  # 1.5 pips spread
            tp_pips = 40 * pip_value
            sl_pips = 20 * pip_value
        
        # Enhanced thresholds for better models
        if model_type == 'Enhanced':
            signal_threshold = 0.065  # Lower for enhanced
            confidence_threshold = 0.650  # Lower for enhanced
        else:
            signal_threshold = 0.070   # Baseline thresholds
            confidence_threshold = 0.660
        
        trades = []
        daily_trades = 0
        daily_risk = 0
        last_trade_day = None
        open_position = None
        
        for i, row in predictions.iterrows():
            current_time = row['timestamp']
            current_day = current_time.date()
            
            # Reset daily counters
            if last_trade_day != current_day:
                daily_trades = 0
                daily_risk = 0
                last_trade_day = current_day
            
            # Check for signal
            should_trade = (
                row['signal_prob'] > signal_threshold and
                row['confidence'] > confidence_threshold and
                row['reward_estimate'] > 0 and
                daily_trades < 3 and  # Max 3 trades per day
                daily_risk < max_daily_risk and
                open_position is None  # No overlapping positions
            )
            
            if should_trade:
                # Determine direction based on reward estimate and confidence
                direction = 'LONG' if row['reward_estimate'] > 0 else 'SHORT'
                
                # Position sizing
                risk_amount = balance * position_size_pct
                
                if direction == 'LONG':
                    entry_price = row['close'] + spread/2  # Include spread
                    tp_price = entry_price + tp_pips
                    sl_price = entry_price - sl_pips
                else:
                    entry_price = row['close'] - spread/2  # Include spread
                    tp_price = entry_price - tp_pips
                    sl_price = entry_price + sl_pips
                
                open_position = {
                    'direction': direction,
                    'entry_time': current_time,
                    'entry_price': entry_price,
                    'tp_price': tp_price,
                    'sl_price': sl_price,
                    'risk_amount': risk_amount,
                    'signal_prob': row['signal_prob'],
                    'confidence': row['confidence'],
                    'reward_estimate': row['reward_estimate']
                }
                
                daily_trades += 1
                daily_risk += position_size_pct
            
            # Check for position closure
            if open_position and i < len(predictions) - 1:
                # Look ahead for TP/SL hit (simplified simulation)
                next_rows = predictions.iloc[i+1:min(i+21, len(predictions))]  # Next 20 hours max
                
                for j, next_row in next_rows.iterrows():
                    hit_tp = False
                    hit_sl = False
                    
                    if open_position['direction'] == 'LONG':
                        if next_row['close'] >= open_position['tp_price']:
                            hit_tp = True
                        elif next_row['close'] <= open_position['sl_price']:
                            hit_sl = True
                    else:
                        if next_row['close'] <= open_position['tp_price']:
                            hit_tp = True
                        elif next_row['close'] >= open_position['sl_price']:
                            hit_sl = True
                    
                    if hit_tp or hit_sl:
                        # Calculate P&L
                        if hit_tp:
                            pnl = open_position['risk_amount'] * 2  # 2:1 RR
                            result = 'WIN'
                            exit_price = open_position['tp_price']
                        else:
                            pnl = -open_position['risk_amount']
                            result = 'LOSS'
                            exit_price = open_position['sl_price']
                        
                        balance += pnl
                        
                        trades.append({
                            'entry_time': open_position['entry_time'],
                            'exit_time': next_row['timestamp'],
                            'direction': open_position['direction'],
                            'entry_price': open_position['entry_price'],
                            'exit_price': exit_price,
                            'pnl': pnl,
                            'result': result,
                            'signal_prob': open_position['signal_prob'],
                            'confidence': open_position['confidence'],
                            'reward_estimate': open_position['reward_estimate'],
                            'balance': balance
                        })
                        
                        open_position = None
                        break
        
        if not trades:
            print(f"❌ No trades generated for {pair} {model_type}")
            return None
        
        # Calculate comprehensive metrics
        trades_df = pd.DataFrame(trades)
        
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['result'] == 'WIN'])
        losing_trades = len(trades_df[trades_df['result'] == 'LOSS'])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = balance - initial_balance
        total_return = (total_pnl / initial_balance) * 100
        
        avg_win = trades_df[trades_df['result'] == 'WIN']['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = abs(trades_df[trades_df['result'] == 'LOSS']['pnl'].mean()) if losing_trades > 0 else 0
        
        profit_factor = (avg_win * winning_trades) / (avg_loss * losing_trades) if losing_trades > 0 else float('inf')
        
        # Risk metrics
        returns = trades_df['pnl'] / initial_balance
        sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
        
        max_balance = trades_df['balance'].cummax()
        drawdown = (max_balance - trades_df['balance']) / max_balance
        max_drawdown = drawdown.max() * 100
        
        results = {
            'pair': pair,
            'model_type': model_type,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate * 100,
            'total_return': total_return,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_signal_prob': trades_df['signal_prob'].mean(),
            'avg_confidence': trades_df['confidence'].mean(),
            'avg_reward_estimate': trades_df['reward_estimate'].mean(),
            'final_balance': balance,
            'trades_df': trades_df
        }
        
        print(f"📊 {model_type} Results:")
        print(f"   Trades: {total_trades}")
        print(f"   Win Rate: {win_rate:.1%}")
        print(f"   Return: {total_return:.1f}%")
        print(f"   Profit Factor: {profit_factor:.2f}")
        print(f"   Max Drawdown: {max_drawdown:.1f}%")
        
        return results
    
    def run_comparison_backtest(self):
        """Run comprehensive comparison backtest"""
        
        print("\n🚀 STARTING ENHANCED VS BASELINE COMPARISON")
        print("=" * 60)
        
        self.load_models()
        
        if not self.enhanced_models or not self.baseline_models:
            print("❌ Missing models for comparison")
            return
        
        results = []
        
        for pair in ['EURUSD_X', 'GBPUSD_X', 'USDJPY_X']:
            
            if pair not in self.enhanced_models or pair not in self.baseline_models:
                print(f"⚠️ Skipping {pair} - missing models")
                continue
            
            print(f"\n🎯 TESTING {pair}")
            print("-" * 40)
            
            # Download test data
            data = self.download_test_data(pair, period='2mo')
            if data is None:
                continue
            
            # Create features
            features = self.create_features(data)
            if len(features) < 100:
                print(f"❌ Insufficient feature data for {pair}")
                continue
            
            # Test Enhanced Model (75 epochs)
            enhanced_predictions = self.make_predictions(
                self.enhanced_models[pair], features, pair
            )
            enhanced_results = self.simulate_trading(
                enhanced_predictions, pair, 'Enhanced'
            )
            
            # Test Baseline Model (30 epochs)
            baseline_predictions = self.make_predictions(
                self.baseline_models[pair], features, pair
            )
            baseline_results = self.simulate_trading(
                baseline_predictions, pair, 'Baseline'
            )
            
            if enhanced_results and baseline_results:
                results.extend([enhanced_results, baseline_results])
        
        if results:
            self.analyze_comparison_results(results)
        else:
            print("❌ No results to analyze")
    
    def analyze_comparison_results(self, results):
        """Analyze and compare the results"""
        
        print(f"\n📊 ENHANCED VS BASELINE ANALYSIS")
        print("=" * 60)
        
        # Create results DataFrame
        df = pd.DataFrame([{
            'Pair': r['pair'],
            'Model': r['model_type'],
            'Trades': r['total_trades'],
            'Win Rate': f"{r['win_rate']:.1f}%",
            'Return': f"{r['total_return']:.1f}%",
            'Profit Factor': f"{r['profit_factor']:.2f}",
            'Sharpe': f"{r['sharpe_ratio']:.2f}",
            'Max DD': f"{r['max_drawdown']:.1f}%",
            'Avg Signal': f"{r['avg_signal_prob']:.3f}",
            'Avg Confidence': f"{r['avg_confidence']:.3f}"
        } for r in results])
        
        print("\n📋 DETAILED RESULTS:")
        print(df.to_string(index=False))
        
        # Compare by pair
        print(f"\n🔍 PAIR-BY-PAIR COMPARISON:")
        
        for pair in ['EURUSD_X', 'GBPUSD_X', 'USDJPY_X']:
            pair_results = [r for r in results if r['pair'] == pair]
            if len(pair_results) == 2:
                enhanced = next(r for r in pair_results if r['model_type'] == 'Enhanced')
                baseline = next(r for r in pair_results if r['model_type'] == 'Baseline')
                
                print(f"\n{pair}:")
                print(f"   Enhanced: {enhanced['total_return']:.1f}% return, {enhanced['win_rate']:.1f}% win rate")
                print(f"   Baseline: {baseline['total_return']:.1f}% return, {baseline['win_rate']:.1f}% win rate")
                
                return_improvement = enhanced['total_return'] - baseline['total_return']
                winrate_improvement = enhanced['win_rate'] - baseline['win_rate']
                
                print(f"   📈 Return improvement: {return_improvement:+.1f}%")
                print(f"   🎯 Win rate improvement: {winrate_improvement:+.1f}%")
        
        # Overall comparison
        enhanced_results = [r for r in results if r['model_type'] == 'Enhanced']
        baseline_results = [r for r in results if r['model_type'] == 'Baseline']
        
        if enhanced_results and baseline_results:
            enhanced_avg_return = np.mean([r['total_return'] for r in enhanced_results])
            baseline_avg_return = np.mean([r['total_return'] for r in baseline_results])
            
            enhanced_avg_winrate = np.mean([r['win_rate'] for r in enhanced_results])
            baseline_avg_winrate = np.mean([r['win_rate'] for r in baseline_results])
            
            print(f"\n🏆 OVERALL COMPARISON:")
            print(f"   Enhanced Models (75 epochs):")
            print(f"     Average Return: {enhanced_avg_return:.1f}%")
            print(f"     Average Win Rate: {enhanced_avg_winrate:.1f}%")
            print(f"   Baseline Models (30 epochs):")
            print(f"     Average Return: {baseline_avg_return:.1f}%")
            print(f"     Average Win Rate: {baseline_avg_winrate:.1f}%")
            
            overall_return_improvement = enhanced_avg_return - baseline_avg_return
            overall_winrate_improvement = enhanced_avg_winrate - baseline_avg_winrate
            
            print(f"\n💡 EPOCH TRAINING IMPACT:")
            print(f"   📈 Average return improvement: {overall_return_improvement:+.1f}%")
            print(f"   🎯 Average win rate improvement: {overall_winrate_improvement:+.1f}%")
            
            if overall_return_improvement > 0:
                print(f"   ✅ Extended training (75 vs 30 epochs) IMPROVED performance")
            else:
                print(f"   ⚠️ Extended training did not improve performance")
            
            # Training efficiency analysis
            print(f"\n📚 TRAINING ANALYSIS:")
            print(f"   • 75-epoch models show {'better' if overall_return_improvement > 0 else 'similar/worse'} trading performance")
            print(f"   • Training time: ~2.5x longer for 75 epochs")
            print(f"   • Model complexity: Higher capacity architecture")
            print(f"   • Recommendation: {'Use enhanced models' if overall_return_improvement > 2 else 'Baseline models sufficient'}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        with open(f'enhanced_backtest_results_{timestamp}.json', 'w') as f:
            import json
            # Convert results for JSON serialization
            json_results = []
            for r in results:
                json_r = r.copy()
                json_r.pop('trades_df', None)  # Remove DataFrame
                json_results.append(json_r)
            
            json.dump(json_results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to enhanced_backtest_results_{timestamp}.json")

if __name__ == "__main__":
    backtester = EnhancedBacktester()
    backtester.run_comparison_backtest()
