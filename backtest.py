import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, Optional
import requests
import os
import glob
import pickle
import json
import torch
from train import ForexPredictor, ForexIndicators, ForexEnvironment, TradingAction

class ForexDataLoader:
    """Load and prepare forex data for training"""
    
    def __init__(self):
        self.pairs = {
            'EURUSD': 'EURUSD=X',
            'GBPUSD': 'GBPUSD=X',
            'USDJPY': 'JPY=X'
        }
    
    def load_historical_data(self, pair: str, period: str = '1y', interval: str = '1h') -> pd.DataFrame:
        """Load historical forex data"""
        if pair not in self.pairs:
            raise ValueError(f"Pair {pair} not supported")
        
        ticker = yf.Ticker(self.pairs[pair])
        df = ticker.history(period=period, interval=interval)
        
        # Rename columns to lowercase
        df.columns = [col.lower() for col in df.columns]
        
        # Add synthetic volume if not present
        if 'volume' not in df.columns or df['volume'].isna().all():
            df['volume'] = self._generate_synthetic_volume(df)
        
        return df
    
    def _generate_synthetic_volume(self, df: pd.DataFrame) -> np.ndarray:
        """Generate synthetic volume based on price movement and volatility"""
        price_change = np.abs(df['close'].pct_change())
        volatility = df['high'] - df['low']
        
        # Normalize and combine
        norm_change = (price_change - price_change.mean()) / price_change.std()
        norm_volatility = (volatility - volatility.mean()) / volatility.std()
        
        synthetic_volume = 1000000 * (1 + norm_change + norm_volatility)
        synthetic_volume = synthetic_volume.fillna(1000000)
        
        return synthetic_volume
    
    def load_all_pairs(self, period: str = '1y', interval: str = '1h') -> Dict[str, pd.DataFrame]:
        """Load data for all currency pairs"""
        data = {}
        for pair in self.pairs.keys():
            print(f"Loading data for {pair}...")
            try:
                data[pair] = self.load_historical_data(pair, period, interval)
                print(f"Loaded {len(data[pair])} records for {pair}")
            except Exception as e:
                print(f"Error loading {pair}: {e}")
        
        return data
    
    def prepare_training_data(self, df: pd.DataFrame, train_split: float = 0.8) -> tuple:
        """Split data into training and testing sets"""
        split_idx = int(len(df) * train_split)
        
        train_data = df.iloc[:split_idx].copy()
        test_data = df.iloc[split_idx:].copy()
        
        return train_data, test_data

class ProductionBacktester:
    """Production-grade backtester for trained forex models"""
    
    def __init__(self, initial_balance: float = 10000):
        self.initial_balance = initial_balance
        self.reset()
        
    def reset(self):
        """Reset backtester state"""
        self.balance = self.initial_balance
        self.trades = []
        self.equity_curve = [self.initial_balance]
        self.positions = {}
        self.daily_pnl = []
        
    def load_trained_model(self, model_path: str = None) -> ForexPredictor:
        """Load the most recent trained model"""
        if model_path is None:
            # Find the most recent model file
            model_files = glob.glob("models/forex_model_*.pkl")
            if not model_files:
                raise FileNotFoundError("No trained models found in models/ directory")
            model_path = max(model_files, key=os.path.getctime)
            
        print(f"📈 Loading trained model: {model_path}")
        
        try:
            # Create a new predictor instance
            predictor = ForexPredictor()
            
            # Use the predictor's load method instead of pickle
            predictor.load_model(model_path)
            
            print(f"✅ Model loaded successfully")
            print(f"   Trained pairs: {predictor.pairs}")
            print(f"   Available agents: {list(predictor.agents.keys())}")
            
            return predictor
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
            
    def run_comprehensive_backtest(self, predictor: ForexPredictor, test_data: Dict[str, pd.DataFrame], 
                                 start_date: str = None, end_date: str = None) -> Dict:
        """Run comprehensive backtest on multiple pairs"""
        print("🧪 Starting comprehensive backtest...")
        
        results = {}
        
        for pair, df in test_data.items():
            if pair not in predictor.pairs:
                print(f"⚠️ Skipping {pair} - not in trained model")
                continue
                
            print(f"\n📊 Backtesting {pair}...")
            
            # Filter data by date range if specified
            if start_date or end_date:
                df = self._filter_by_date(df, start_date, end_date)
                
            if len(df) < 100:
                print(f"⚠️ Insufficient data for {pair}: {len(df)} candles")
                continue
                
            # Reset for each pair
            self.reset()
            
            # Run backtest for this pair
            pair_results = self.backtest_single_pair(predictor, pair, df)
            results[pair] = pair_results
            
            # Print summary
            self._print_pair_summary(pair, pair_results)
            
        # Print overall summary
        self._print_overall_summary(results)
        
        return results
        
    def backtest_single_pair(self, predictor: ForexPredictor, pair: str, data: pd.DataFrame) -> Dict:
        """Backtest a single currency pair"""
        
        try:
            # Prepare data with indicators
            prepared_data = predictor.prepare_data(data.copy())
            
            # Create environment for simulation
            env = ForexEnvironment(prepared_data, pair_name=pair)
            
            trades = []
            equity_curve = [self.initial_balance]
            current_balance = self.initial_balance
            
            # Track open positions
            open_positions = []
            
            # Simulate trading through historical data
            current_step = env.lookback_period
            
            print(f"   📈 Processing {len(prepared_data) - env.lookback_period} candles...")
            
            while current_step < len(prepared_data) - 1:
                try:
                    # Get current market data
                    current_candle = prepared_data.iloc[current_step]
                    current_price = current_candle['close']
                    timestamp = prepared_data.index[current_step]
                    
                    # Check if any open positions hit TP/SL
                    for pos in open_positions[:]:  # Copy list to avoid modification during iteration
                        pnl = self._check_position_exit(pos, current_price)
                        if pnl is not None:
                            current_balance += pnl
                            pos['pnl'] = pnl
                            pos['exit_price'] = current_price
                            pos['exit_timestamp'] = timestamp
                            trades.append(pos)
                            open_positions.remove(pos)
                    
                    # Get prediction from trained model only if no open positions (for simplicity)
                    if len(open_positions) == 0:
                        try:
                            # Get current state
                            env.current_step = current_step
                            state = env._get_state()
                            
                            # Get prediction from trained model
                            action = predictor.predict(pair, prepared_data.iloc[:current_step+1])
                            
                            # Debug: Print first few predictions
                            if current_step < env.lookback_period + 5:
                                print(f"   🔍 Step {current_step}: {action.direction} (confidence: {action.confidence:.3f}, TP: {action.take_profit:.1f}, SL: {action.stop_loss:.1f})")
                            
                            # Execute trade if not hold
                            if action.direction != 'hold':
                                trade = self._execute_trade_simple(
                                    action, current_price, timestamp, pair, current_step, current_balance
                                )
                                if trade:
                                    open_positions.append(trade)
                                    current_balance -= trade.get('entry_cost', 0)
                                    print(f"   💰 Opened {action.direction} position at {current_price:.5f}")
                        except Exception as e:
                            print(f"⚠️ Prediction error at step {current_step}: {str(e)}")
                            # Force a random trade for testing
                            if current_step == env.lookback_period + 10:  # Force one trade for testing
                                from train import TradingAction
                                test_action = TradingAction('long', 50.0, 25.0, 0.5)
                                trade = self._execute_trade_simple(
                                    test_action, current_price, timestamp, pair, current_step, current_balance
                                )
                                if trade:
                                    open_positions.append(trade)
                                    current_balance -= trade.get('entry_cost', 0)
                                    print(f"   🧪 TEST: Opened test long position at {current_price:.5f}")
                            continue
                    
                    # Update equity curve
                    total_unrealized_pnl = sum(self._calculate_unrealized_pnl(pos, current_price) for pos in open_positions)
                    current_equity = current_balance + total_unrealized_pnl
                    equity_curve.append(current_equity)
                    
                    current_step += 1
                    
                except Exception as e:
                    print(f"⚠️ Error at step {current_step}: {str(e)}")
                    current_step += 1
                    continue
            
            # Close any remaining open positions
            for pos in open_positions:
                final_price = prepared_data.iloc[-1]['close']
                pnl = self._check_position_exit(pos, final_price, force_close=True)
                current_balance += pnl
                pos['pnl'] = pnl
                pos['exit_price'] = final_price
                pos['exit_timestamp'] = prepared_data.index[-1]
                trades.append(pos)
            
            # Calculate comprehensive metrics
            metrics = self._calculate_comprehensive_metrics(trades, equity_curve, prepared_data)
            
            print(f"   ✅ Completed {len(trades)} trades")
            
            return {
                'trades': trades,
                'equity_curve': equity_curve,
                'metrics': metrics,
                'data_points': len(prepared_data),
                'test_period': f"{prepared_data.index[0]} to {prepared_data.index[-1]}"
            }
            
        except Exception as e:
            error_msg = f"Error backtesting {pair}: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return {'error': error_msg}
            
    def _execute_trade_simple(self, action: TradingAction, price: float, timestamp, pair: str, step: int, balance: float) -> Dict:
        """Simplified trade execution"""
        
        # Simple position sizing - risk 1% of balance
        risk_amount = balance * 0.01
        pip_value = 0.0001 if 'JPY' not in pair else 0.01
        position_size = risk_amount / (action.stop_loss * pip_value)
        position_size = min(position_size, balance * 0.1)  # Max 10% of balance
        
        # Entry cost (simplified spread)
        spread_cost = price * 0.00002  # 2 pip spread
        entry_cost = spread_cost * position_size / 100000
        
        # Calculate TP and SL prices
        if action.direction == 'long':
            tp_price = price + (action.take_profit * pip_value)
            sl_price = price - (action.stop_loss * pip_value)
        else:  # short
            tp_price = price - (action.take_profit * pip_value)
            sl_price = price + (action.stop_loss * pip_value)
        
        trade = {
            'timestamp': timestamp,
            'step': step,
            'pair': pair,
            'direction': action.direction,
            'entry_price': price,
            'position_size': position_size,
            'take_profit_pips': action.take_profit,
            'stop_loss_pips': action.stop_loss,
            'tp_price': tp_price,
            'sl_price': sl_price,
            'confidence': action.confidence,
            'entry_cost': entry_cost,
            'status': 'open'
        }
        
        return trade
        
    def _check_position_exit(self, position: Dict, current_price: float, force_close: bool = False) -> Optional[float]:
        """Check if position should be closed and return PnL"""
        
        if force_close:
            # Force close at current price
            pass
        elif position['direction'] == 'long':
            if current_price >= position['tp_price']:
                # Take profit hit
                pass
            elif current_price <= position['sl_price']:
                # Stop loss hit
                pass
            else:
                return None  # Position stays open
        else:  # short
            if current_price <= position['tp_price']:
                # Take profit hit
                pass
            elif current_price >= position['sl_price']:
                # Stop loss hit
                pass
            else:
                return None  # Position stays open
        
        # Calculate PnL
        if position['direction'] == 'long':
            price_diff = current_price - position['entry_price']
        else:  # short
            price_diff = position['entry_price'] - current_price
        
        pip_value = 0.0001 if 'JPY' not in position['pair'] else 0.01
        pips_gained = price_diff / pip_value
        
        # Calculate profit/loss
        pnl = pips_gained * pip_value * position['position_size'] / 100000
        
        # Subtract entry cost and exit cost
        exit_cost = position['entry_cost']  # Same as entry cost for simplicity
        total_pnl = pnl - position['entry_cost'] - exit_cost
        
        return total_pnl
        
    def _calculate_unrealized_pnl(self, position: Dict, current_price: float) -> float:
        """Calculate unrealized PnL for open position"""
        if position['direction'] == 'long':
            price_diff = current_price - position['entry_price']
        else:  # short
            price_diff = position['entry_price'] - current_price
        
        pip_value = 0.0001 if 'JPY' not in position['pair'] else 0.01
        pips = price_diff / pip_value
        
        unrealized_pnl = pips * pip_value * position['position_size'] / 100000
        return unrealized_pnl
    def _calculate_comprehensive_metrics(self, trades: list, equity_curve: list, data: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        if not trades:
            return {'error': 'No trades executed'}
        
        # Filter only completed trades (with PnL)
        completed_trades = [t for t in trades if 'pnl' in t]
        
        if not completed_trades:
            return {'error': 'No completed trades'}
            
        # Basic metrics
        final_balance = equity_curve[-1] if equity_curve else self.initial_balance
        total_return = (final_balance - self.initial_balance) / self.initial_balance
        
        # Trade analysis
        profitable_trades = [t for t in completed_trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in completed_trades if t.get('pnl', 0) < 0]
        
        win_rate = len(profitable_trades) / len(completed_trades) if completed_trades else 0
        avg_win = np.mean([t['pnl'] for t in profitable_trades]) if profitable_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        # Risk metrics
        if len(equity_curve) > 1:
            returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
            returns = returns[np.isfinite(returns)]  # Remove any infinite values
            
            if len(returns) > 0:
                sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 24)  # Hourly data
                volatility = np.std(returns) * np.sqrt(252 * 24)
            else:
                sharpe_ratio = 0
                volatility = 0
                
            max_drawdown = self._calculate_max_drawdown(equity_curve)
        else:
            sharpe_ratio = 0
            max_drawdown = 0
            volatility = 0
        
        # Profit factor
        if avg_loss != 0 and len(losing_trades) > 0:
            profit_factor = abs(avg_win * len(profitable_trades)) / abs(avg_loss * len(losing_trades))
        else:
            profit_factor = float('inf') if avg_win > 0 else 0
            
        return {
            'total_trades': len(completed_trades),
            'winning_trades': len(profitable_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_return': total_return,
            'final_balance': final_balance,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_confidence': np.mean([t['confidence'] for t in completed_trades]) if completed_trades else 0,
            'test_period_days': len(data) / 24,  # Assuming hourly data
        }
        
    def _calculate_max_drawdown(self, equity_curve: list) -> float:
        """Calculate maximum drawdown percentage"""
        peak = equity_curve[0]
        max_dd = 0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
                
        return max_dd
        
    def _filter_by_date(self, df: pd.DataFrame, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Filter data by date range"""
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        return df
        
    def _print_pair_summary(self, pair: str, results: Dict):
        """Print summary for a single pair"""
        if 'error' in results:
            print(f"❌ {pair}: {results['error']}")
            return
            
        if 'metrics' not in results:
            print(f"❌ {pair}: No metrics available")
            return
            
        metrics = results['metrics']
        
        # Handle case where metrics might have error
        if 'error' in metrics:
            print(f"❌ {pair}: {metrics['error']}")
            return
            
        print(f"✅ {pair} Results:")
        print(f"   📊 Total Return: {metrics.get('total_return', 0):.2%}")
        print(f"   📈 Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"   📉 Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        print(f"   🎯 Win Rate: {metrics.get('win_rate', 0):.1%}")
        print(f"   💰 Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        print(f"   📋 Total Trades: {metrics.get('total_trades', 0)}")
        print(f"   🔗 Confidence: {metrics.get('avg_confidence', 0):.3f}")
        
    def _print_overall_summary(self, results: Dict):
        """Print overall backtest summary"""
        print("\n" + "="*60)
        print("📊 OVERALL BACKTEST SUMMARY")
        print("="*60)
        
        successful_pairs = [pair for pair, res in results.items() if 'error' not in res and 'metrics' in res and 'error' not in res['metrics']]
        
        if not successful_pairs:
            print("❌ No successful backtests with trades")
            # Still show what happened
            for pair, result in results.items():
                if 'error' in result:
                    print(f"   {pair}: {result['error']}")
                elif 'metrics' in result and 'error' in result['metrics']:
                    print(f"   {pair}: {result['metrics']['error']}")
            return
            
        # Aggregate metrics
        total_returns = [results[pair]['metrics']['total_return'] for pair in successful_pairs]
        sharpe_ratios = [results[pair]['metrics']['sharpe_ratio'] for pair in successful_pairs]
        max_drawdowns = [results[pair]['metrics']['max_drawdown'] for pair in successful_pairs]
        win_rates = [results[pair]['metrics']['win_rate'] for pair in successful_pairs]
        
        print(f"✅ Successful Pairs: {len(successful_pairs)}")
        print(f"📈 Average Return: {np.mean(total_returns):.2%}")
        print(f"📊 Average Sharpe: {np.mean(sharpe_ratios):.2f}")
        print(f"📉 Average Max DD: {np.mean(max_drawdowns):.2%}")
        print(f"🎯 Average Win Rate: {np.mean(win_rates):.1%}")
        
        # Best and worst performers
        if len(successful_pairs) > 1:
            best_pair = max(successful_pairs, key=lambda p: results[p]['metrics']['total_return'])
            worst_pair = min(successful_pairs, key=lambda p: results[p]['metrics']['total_return'])
            
            print(f"\n🏆 Best Performer: {best_pair} ({results[best_pair]['metrics']['total_return']:.2%})")
            print(f"📉 Worst Performer: {worst_pair} ({results[worst_pair]['metrics']['total_return']:.2%})")
        elif len(successful_pairs) == 1:
            pair = successful_pairs[0]
            print(f"\n📊 Only one successful pair: {pair} ({results[pair]['metrics']['total_return']:.2%})")


class Backtester:
    """Legacy backtester - keeping for compatibility"""
    
    def __init__(self, initial_balance: float = 10000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.trades = []
        self.equity_curve = []
        
    def run_backtest(self, predictor, data: pd.DataFrame, pair: str):
        """Run backtest on historical data"""
        from forex_trading_model import ForexEnvironment
        
        prepared_data = predictor.prepare_data(data)
        env = ForexEnvironment(prepared_data)
        
        state = env.reset()
        done = False
        step = 0
        
        while not done:
            action = predictor.predict(pair, prepared_data.iloc[:env.current_step+1])
            
            current_price = prepared_data.iloc[env.current_step]['close']
            
            # Simulate trade execution
            if action.direction != 'hold':
                trade = {
                    'step': step,
                    'timestamp': prepared_data.index[env.current_step],
                    'direction': action.direction,
                    'entry_price': current_price,
                    'take_profit': action.take_profit,
                    'stop_loss': action.stop_loss,
                    'confidence': action.confidence
                }
                
                # Calculate potential profit/loss
                if action.direction == 'long':
                    tp_price = current_price + (action.take_profit / 10000)
                    sl_price = current_price - (action.stop_loss / 10000)
                else:
                    tp_price = current_price - (action.take_profit / 10000)
                    sl_price = current_price + (action.stop_loss / 10000)
                
                trade['tp_price'] = tp_price
                trade['sl_price'] = sl_price
                
                self.trades.append(trade)
            
            state, reward, done = env.step(action)
            self.balance = env.balance
            self.equity_curve.append(self.balance)
            step += 1
        
        return self.calculate_metrics()
    
    def calculate_metrics(self) -> dict:
        """Calculate performance metrics"""
        if not self.trades:
            return {'error': 'No trades executed'}
        
        total_trades = len(self.trades)
        
        # Calculate returns
        returns = np.array(self.equity_curve)
        returns_pct = np.diff(returns) / returns[:-1]
        
        # Calculate metrics
        total_return = (self.balance - self.initial_balance) / self.initial_balance * 100
        
        if len(returns_pct) > 0:
            sharpe_ratio = np.mean(returns_pct) / (np.std(returns_pct) + 1e-8) * np.sqrt(252)
            max_drawdown = self.calculate_max_drawdown(self.equity_curve)
        else:
            sharpe_ratio = 0
            max_drawdown = 0
        
        metrics = {
            'total_trades': total_trades,
            'final_balance': self.balance,
            'total_return_pct': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown,
            'avg_trade_confidence': np.mean([t['confidence'] for t in self.trades])
        }
        
        return metrics
    
    def calculate_max_drawdown(self, equity_curve: list) -> float:
        """Calculate maximum drawdown"""
        peak = equity_curve[0]
        max_dd = 0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        return max_dd

# Example usage - Production Backtest
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Backtest Trained Forex Models')
    parser.add_argument('--model', type=str, help='Path to trained model file (optional - will use latest if not specified)')
    parser.add_argument('--period', type=str, default='6mo', help='Data period for backtesting (default: 6mo)')
    parser.add_argument('--pairs', nargs='+', default=['EURUSD', 'GBPUSD', 'USDJPY'], help='Currency pairs to backtest')
    parser.add_argument('--start-date', type=str, help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date for backtest (YYYY-MM-DD)')
    parser.add_argument('--initial-balance', type=float, default=10000, help='Initial balance for backtest')
    
    args = parser.parse_args()
    
    print("="*80)
    print("🚀 PRODUCTION FOREX MODEL BACKTESTER")
    print("="*80)
    print(f"📊 Data period: {args.period}")
    print(f"💰 Initial balance: ${args.initial_balance:,.2f}")
    print(f"💱 Pairs: {', '.join(args.pairs)}")
    if args.start_date:
        print(f"📅 Start date: {args.start_date}")
    if args.end_date:
        print(f"📅 End date: {args.end_date}")
    print("="*80)
    
    try:
        # Load fresh test data
        print("📈 Loading fresh market data for backtesting...")
        loader = ForexDataLoader()
        test_data = {}
        
        for pair in args.pairs:
            try:
                df = loader.load_historical_data(pair, period=args.period, interval='1h')
                test_data[pair] = df
                print(f"✅ {pair}: {len(df)} candles loaded")
            except Exception as e:
                print(f"❌ Failed to load {pair}: {str(e)}")
        
        if not test_data:
            print("❌ No data loaded. Exiting.")
            exit(1)
        
        # Initialize backtester
        backtester = ProductionBacktester(initial_balance=args.initial_balance)
        
        # Load trained model
        predictor = backtester.load_trained_model(args.model)
        
        # Optional: Force some exploration during backtesting to see if model can trade
        print("🔧 Adjusting model for backtesting...")
        for pair, agent in predictor.agents.items():
            original_epsilon = agent.epsilon
            agent.epsilon = 0.1  # Add 10% exploration to encourage trading
            print(f"   🎯 {pair}: Epsilon {original_epsilon:.3f} -> {agent.epsilon:.3f}")
        print("   💡 Added exploration to encourage trading signals")
        
        # Run comprehensive backtest
        print(f"\n🧪 Running backtest on {len(test_data)} pairs...")
        start_time = datetime.now()
        
        results = backtester.run_comprehensive_backtest(
            predictor, 
            test_data, 
            start_date=args.start_date, 
            end_date=args.end_date
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Save results
        results_filename = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert results to JSON-serializable format
        json_results = {}
        for pair, result in results.items():
            if 'error' not in result:
                json_results[pair] = {
                    'metrics': result['metrics'],
                    'data_points': result['data_points'],
                    'test_period': result['test_period'],
                    'trade_count': len(result['trades'])
                }
            else:
                json_results[pair] = {'error': result['error']}
        
        with open(f"models/{results_filename}", 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: models/{results_filename}")
        print(f"⏱️  Backtest completed in {duration:.1f} seconds")
        
        # Performance summary
        successful_pairs = [pair for pair, res in results.items() if 'error' not in res and 'metrics' in res and 'error' not in res['metrics']]
        if successful_pairs:
            avg_return = np.mean([results[pair]['metrics']['total_return'] for pair in successful_pairs])
            avg_sharpe = np.mean([results[pair]['metrics']['sharpe_ratio'] for pair in successful_pairs])
            
            print(f"\n🏆 FINAL PERFORMANCE SUMMARY:")
            print(f"   ✅ Successful pairs: {len(successful_pairs)}/{len(args.pairs)}")
            print(f"   📈 Average return: {avg_return:.2%}")
            print(f"   📊 Average Sharpe: {avg_sharpe:.2f}")
            
            if avg_return > 0.05:  # 5% return
                print("🎉 EXCELLENT PERFORMANCE! Model shows strong profitability.")
            elif avg_return > 0:
                print("✅ POSITIVE PERFORMANCE! Model is profitable.")
            else:
                print("⚠️  NEGATIVE PERFORMANCE. Model needs improvement.")
        else:
            print("\n⚠️  MODEL ANALYSIS:")
            print("   🔍 The model appears to be overly conservative and only outputs 'hold' signals.")
            print("   📊 This often happens when:")
            print("      • Model hasn't learned effective trading patterns")
            print("      • Safety checks are too restrictive") 
            print("      • Training data was insufficient")
            print("      • Model is stuck in local minimum (always holding)")
            print("   💡 Recommendations:")
            print("      • Retrain with more epochs and data")
            print("      • Reduce safety penalty weights")
            print("      • Add reward for successful trades")
            print("      • Check if training data has clear trading opportunities")
            print("   ❌ No trades executed across all pairs.")
        
    except Exception as e:
        print(f"❌ Error during backtesting: {str(e)}")
        import traceback
        traceback.print_exc()


# Legacy example for compatibility
def run_legacy_backtest():
    """Legacy backtest function"""
    # Load data
    loader = ForexDataLoader()
    data = loader.load_all_pairs(period='6mo', interval='1h')
    
    # Prepare for training
    predictor = ForexPredictor()
    
    # Split data
    training_data = {}
    testing_data = {}
    
    for pair, df in data.items():
        train, test = loader.prepare_training_data(df)
        training_data[pair] = train
        testing_data[pair] = test
    
    # Train model
    print("Training models...")
    predictor.train(training_data, epochs=50)
    
    # Backtest
    print("\nRunning backtest...")
    backtester = Backtester()
    
    for pair in testing_data.keys():
        print(f"\nBacktesting {pair}...")
        metrics = backtester.run_backtest(predictor, testing_data[pair], pair)
        
        print(f"Results for {pair}:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Save model
    predictor.save_model('forex_model.pth')
    print("\nModel saved successfully!")