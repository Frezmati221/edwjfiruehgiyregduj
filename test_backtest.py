#!/usr/bin/env python3
"""
Quick Backtester for Trained Forex AI Model
Tests the newly trained DQN model with historical data
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import pickle
import argparse
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn

# Import from our training script
from train import ForexIndicators, load_forex_data, ForexTradingAgent, ForexEnvironment

class ForexBacktester:
    def __init__(self, model_path: str, initial_balance: float = 10000.0):
        """Initialize backtester with trained model"""
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.positions = {}  # {pair: {'amount': float, 'entry_price': float, 'type': 'buy'/'sell'}}
        self.trades_history = []
        self.model_path = model_path
        self.agents = {}  # Will hold the DQN agents
        self.indicators = ForexIndicators()
        
        # Trading parameters
        self.max_position_size = 0.1  # 10% of balance per trade
        self.spread = 0.0002  # 2 pips spread
        
        print(f"🔄 Loading trained model from: {model_path}")
        self.load_model()
        
    def load_model(self):
        """Load the trained DQN model"""
        try:
            # Load PyTorch checkpoint
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            print(f"✅ Model loaded successfully!")
            print(f"📊 Model contains agents for: {list(checkpoint['agents'].keys())}")
            
            # Extract agents
            self.agents = {}
            for pair, agent_data in checkpoint['agents'].items():
                # Determine state size from saved weights
                state_size = agent_data['q_network']['feature_extractor.0.weight'].shape[1]
                
                # Create agent and load weights
                agent = ForexTradingAgent(state_size)
                agent.q_network.load_state_dict(agent_data['q_network'])
                agent.target_network.load_state_dict(agent_data['target_network'])
                agent.epsilon = agent_data['epsilon']
                
                # Set to evaluation mode
                agent.q_network.eval()
                agent.target_network.eval()
                
                self.agents[pair] = agent
                print(f"✅ {pair}: state_size={state_size}, epsilon={agent.epsilon:.3f}")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def get_model_prediction(self, state: np.ndarray, pair: str) -> int:
        """Get prediction from trained model"""
        try:
            # Convert pair name to match agent naming (remove =X)
            agent_pair = pair.replace('=X', '')
            
            if agent_pair not in self.agents:
                return 0  # Hold if no agent for this pair
            
            agent = self.agents[agent_pair]
            
            # Convert to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
            
            # Get Q-values from model
            with torch.no_grad():
                q_values = agent.q_network(state_tensor)
            
            # Return action with highest Q-value
            return q_values.argmax().item()
            
        except Exception as e:
            print(f"⚠️ Prediction error for {pair}: {e}")
            return 0  # Hold action
    
    def calculate_position_size(self, pair: str, price: float) -> float:
        """Calculate position size based on available balance"""
        max_amount = self.current_balance * self.max_position_size
        
        if 'JPY' in pair:
            # For JPY pairs, 1 lot = 100,000 JPY
            position_size = max_amount / price
        else:
            # For other pairs, 1 lot = 100,000 base currency
            position_size = max_amount / price
            
        return min(position_size, max_amount / price)
    
    def execute_trade(self, pair: str, action: int, price: float, timestamp: datetime):
        """Execute trade based on model prediction"""
        # Actions: 0=hold, 1=buy, 2=sell
        
        if action == 0:  # Hold
            return
            
        # Close existing position if any
        if pair in self.positions:
            self.close_position(pair, price, timestamp)
            
        # Open new position
        if action == 1:  # Buy
            self.open_position(pair, 'buy', price, timestamp)
        elif action == 2:  # Sell
            self.open_position(pair, 'sell', price, timestamp)
    
    def open_position(self, pair: str, direction: str, price: float, timestamp: datetime):
        """Open a new position"""
        # Apply spread
        entry_price = price + self.spread if direction == 'buy' else price - self.spread
        
        position_size = self.calculate_position_size(pair, entry_price)
        
        if position_size < 1000:  # Minimum position size
            return
            
        self.positions[pair] = {
            'amount': position_size,
            'entry_price': entry_price,
            'type': direction,
            'timestamp': timestamp
        }
        
        print(f"📈 {direction.upper()} {pair}: {position_size:.0f} units @ {entry_price:.4f}")
    
    def close_position(self, pair: str, price: float, timestamp: datetime):
        """Close existing position"""
        if pair not in self.positions:
            return
            
        position = self.positions[pair]
        
        # Apply spread
        exit_price = price - self.spread if position['type'] == 'buy' else price + self.spread
        
        # Calculate P&L
        if position['type'] == 'buy':
            pnl = (exit_price - position['entry_price']) * position['amount']
        else:
            pnl = (position['entry_price'] - exit_price) * position['amount']
        
        # Update balance
        self.current_balance += pnl
        
        # Record trade
        trade = {
            'pair': pair,
            'type': position['type'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'amount': position['amount'],
            'pnl': pnl,
            'entry_time': position['timestamp'],
            'exit_time': timestamp,
            'duration': timestamp - position['timestamp']
        }
        
        self.trades_history.append(trade)
        
        print(f"📉 CLOSE {pair}: P&L = ${pnl:.2f} | Balance = ${self.current_balance:.2f}")
        
        # Remove position
        del self.positions[pair]
    
    def run_backtest(self, start_date: str, end_date: str, pairs: List[str] = None):
        """Run backtest on historical data"""
        if pairs is None:
            pairs = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X']
            
        print(f"\n🚀 Starting backtest from {start_date} to {end_date}")
        print(f"💰 Initial balance: ${self.initial_balance:,.2f}")
        print(f"📊 Testing pairs: {', '.join(pairs)}")
        print("=" * 60)
        
        # Load data for all pairs
        all_data = {}
        for pair in pairs:
            print(f"📥 Loading data for {pair}...")
            ticker = yf.Ticker(pair)
            df = ticker.history(start=start_date, end=end_date, interval='1h')
            
            if df.empty:
                print(f"⚠️ No data for {pair}")
                continue
                
            # Calculate indicators
            # First normalize column names to lowercase
            df.columns = [col.lower() for col in df.columns]
            df = self.indicators.calculate_all_indicators(df)
            all_data[pair] = df
            print(f"✅ {pair}: {len(df)} data points")
        
        if not all_data:
            print("❌ No data loaded for any pairs")
            return
        
        # Get common time range
        common_index = None
        for pair, df in all_data.items():
            if common_index is None:
                common_index = df.index
            else:
                common_index = common_index.intersection(df.index)
        
        print(f"📅 Common timeframe: {len(common_index)} hours")
        
        # Run simulation
        action_counts = {pair: {'hold': 0, 'buy': 0, 'sell': 0} for pair in pairs if pair in all_data}
        
        for i, timestamp in enumerate(common_index[100:]):  # Skip first 100 for indicators
            if i % 500 == 0:
                print(f"⏱️ Progress: {i}/{len(common_index)-100} | Balance: ${self.current_balance:.2f}")
                # Print action counts so far
                for pair, counts in action_counts.items():
                    total = sum(counts.values())
                    if total > 0:
                        print(f"   {pair}: Hold={counts['hold']}, Buy={counts['buy']}, Sell={counts['sell']}")
            
            for pair in pairs:
                if pair not in all_data:
                    continue
                    
                df = all_data[pair]
                
                if timestamp not in df.index:
                    continue
                
                current_row = df.loc[timestamp]
                
                # Prepare state (same as in training)
                try:
                    # Get recent data for state
                    recent_data = df.loc[:timestamp].tail(50)
                    if len(recent_data) < 50:
                        continue
                        
                    # Create temporary environment to get proper state
                    temp_env = ForexEnvironment(recent_data)
                    temp_env.current_step = len(recent_data) - 1
                    state = temp_env._get_state()
                    
                    # Get model prediction
                    action = self.get_model_prediction(state, pair)
                    
                    # Count actions
                    action_names = ['hold', 'buy', 'sell']
                    if pair in action_counts and action < len(action_names):
                        action_counts[pair][action_names[action]] += 1
                    
                    # Execute trade
                    self.execute_trade(pair, action, current_row['close'], timestamp)
                    
                except Exception as e:
                    if i % 1000 == 0:  # Only print occasional errors
                        print(f"⚠️ Error processing {pair} at {timestamp}: {e}")
                    continue
        
        # Close all remaining positions
        final_timestamp = common_index[-1]
        for pair in list(self.positions.keys()):
            if pair in all_data:
                final_price = all_data[pair].loc[final_timestamp, 'close']
                self.close_position(pair, final_price, final_timestamp)
        
        self.print_results()
    
    def print_results(self):
        """Print backtest results"""
        print("\n" + "=" * 60)
        print("📊 BACKTEST RESULTS")
        print("=" * 60)
        
        total_return = self.current_balance - self.initial_balance
        return_pct = (total_return / self.initial_balance) * 100
        
        print(f"💰 Initial Balance:    ${self.initial_balance:,.2f}")
        print(f"💰 Final Balance:      ${self.current_balance:,.2f}")
        print(f"📈 Total Return:       ${total_return:,.2f}")
        print(f"📊 Return %:           {return_pct:.2f}%")
        
        if self.trades_history:
            trades_df = pd.DataFrame(self.trades_history)
            
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] < 0]
            
            print(f"\n📋 TRADE STATISTICS")
            print(f"Total Trades:          {len(trades_df)}")
            print(f"Winning Trades:        {len(winning_trades)} ({len(winning_trades)/len(trades_df)*100:.1f}%)")
            print(f"Losing Trades:         {len(losing_trades)} ({len(losing_trades)/len(trades_df)*100:.1f}%)")
            
            if len(winning_trades) > 0:
                print(f"Average Win:           ${winning_trades['pnl'].mean():.2f}")
                print(f"Largest Win:           ${winning_trades['pnl'].max():.2f}")
            
            if len(losing_trades) > 0:
                print(f"Average Loss:          ${losing_trades['pnl'].mean():.2f}")
                print(f"Largest Loss:          ${losing_trades['pnl'].min():.2f}")
            
            print(f"Average Trade:         ${trades_df['pnl'].mean():.2f}")
            
            # By pair
            print(f"\n📈 PERFORMANCE BY PAIR")
            for pair in trades_df['pair'].unique():
                pair_trades = trades_df[trades_df['pair'] == pair]
                pair_pnl = pair_trades['pnl'].sum()
                print(f"{pair}:           ${pair_pnl:.2f} ({len(pair_trades)} trades)")
        
        else:
            print("\n⚠️ No trades executed during backtest period")

def main():
    parser = argparse.ArgumentParser(description='Backtest trained Forex AI model')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model file')
    parser.add_argument('--start', type=str, default='2024-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2024-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--balance', type=float, default=10000.0, help='Initial balance')
    parser.add_argument('--pairs', type=str, nargs='+', default=['EURUSD=X', 'GBPUSD=X', 'USDJPY=X'], 
                        help='Currency pairs to test')
    
    args = parser.parse_args()
    
    # Initialize backtester
    backtester = ForexBacktester(
        model_path=args.model,
        initial_balance=args.balance
    )
    
    # Run backtest
    backtester.run_backtest(
        start_date=args.start,
        end_date=args.end,
        pairs=args.pairs
    )

if __name__ == "__main__":
    main()
