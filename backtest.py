import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, Optional
import requests

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

class Backtester:
    """Backtest trading strategies"""
    
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

# Example usage
if __name__ == "__main__":
    # Load data
    loader = ForexDataLoader()
    data = loader.load_all_pairs(period='6mo', interval='1h')
    
    # Prepare for training
    from forex_trading_model import ForexPredictor
    
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