#!/usr/bin/env python3
"""
Realistic Backtester for Trade-2.py Supervised Learning Model
Simulates real-time trading with proper balance management and realistic conditions
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import torch
import argparse
from typing import Dict, List
import warnings
import matplotlib.pyplot as plt
import json
import talib
warnings.filterwarnings('ignore')

# Import compatible predictor
from compatible_trade2_predictor import CompatibleSupervisedForexPredictor

class RealisticTrade2Backtester:
    def __init__(self, model_path: str, initial_balance: float = 1000.0, use_optimal_entry: bool = True):
        """Initialize realistic backtester for trade-2.py model"""
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.equity_curve = []
        self.trades_history = []
        self.model_path = model_path
        self.predictor = None
        self.use_optimal_entry = use_optimal_entry
        
        # Trading parameters - more conservative for supervised learning model
        self.leverage = 30  # 1:100 leverage
        self.spread_pips = 2  # 2 pip spread
        self.min_trade_interval = 4  # 4 hours between trades (more conservative)
        self.max_daily_trades = 3  # Max 5 trades per day
        
        # Risk management - aligned with model's training
        self.risk_per_trade = 0.02  # 2% risk per trade
        self.min_confidence = 0.5  # Start with 60% confidence
        self.min_risk_reward = 2  # Minimum 1.5:1 risk-reward ratio
        
        # Market hours (UTC)
        self.market_hours = {
            'start': 22,  # Sunday 22:00 UTC (Monday Sydney open)
            'end': 22     # Friday 22:00 UTC (Friday NY close)
        }
        
        # Current position tracking
        self.current_position = None
        self.last_trade_time = None
        self.daily_trade_count = 0
        self.current_day = None
        
        # Model-specific parameters
        self.sequence_length = 60  # Same as model training
        
        print(f"💰 Realistic Trade-2 Backtester initialized")
        print(f"💵 Starting balance: ${initial_balance:,.2f}")
        print(f"🎯 Risk per trade: {self.risk_per_trade:.1%}")
        print(f"🤖 Min confidence: {self.min_confidence:.1%}")
        print(f"📊 Min risk-reward: {self.min_risk_reward}:1")
        print(f"📍 Optimal entry: {'Enabled' if use_optimal_entry else 'Disabled'}")
        
        self.load_model()
    
    def load_model(self):
        """Load the trained supervised learning model"""
        try:
            self.predictor = CompatibleSupervisedForexPredictor()
            self.predictor.load_model(self.model_path)
            
            print(f"✅ Model loaded from: {self.model_path}")
            print(f"🔧 Model confidence threshold: {self.predictor.min_confidence:.1%}")
            print(f"📏 Sequence length: {self.predictor.sequence_length}")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def calculate_optimal_entry(self, df: pd.DataFrame, action: str, current_price: float, pair: str) -> dict:
        """Calculate optimal entry price based on market conditions"""
        # Get recent price data for technical analysis
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
        """Calculate pip value for position sizing"""
        # Handle different asset classes
        pair_upper = pair.upper()
        
        if 'JPY' in pair_upper:
            return 0.01  # For JPY pairs, 1 pip = 0.01
        elif any(metal in pair_upper for metal in ['XAU', 'GOLD', 'GC=F', 'GLD', 'IAU']):
            return 0.01  # For Gold, 1 pip = $0.01 (per ounce)
        elif any(oil in pair_upper for oil in ['XTI', 'CL=F', 'USO']):
            return 0.01  # For Oil, 1 pip = $0.01 (per barrel)
        elif pair_upper.endswith('=F'):  # Futures contracts
            return 0.01  # Most futures use 0.01 as minimum tick
        else:
            return 0.0001  # For forex pairs, 1 pip = 0.0001
    
    def get_suggested_symbols(self, failed_symbol: str) -> List[str]:
        """Get suggested alternative symbols for common assets"""
        symbol_upper = failed_symbol.upper()
        suggestions = []
        
        if any(gold in symbol_upper for gold in ['XAUUSD', 'GOLD', 'XAU']):
            suggestions = [
                'GC=F (Gold Futures)',
                'GLD (SPDR Gold Trust ETF)', 
                'IAU (iShares Gold Trust ETF)'
            ]
        elif any(oil in symbol_upper for oil in ['XTIUSD', 'OIL', 'XTI']):
            suggestions = [
                'CL=F (Crude Oil Futures)',
                'USO (United States Oil Fund)',
                'XLE (Energy Select Sector SPDR Fund)'
            ]
        elif any(silver in symbol_upper for silver in ['XAGUSD', 'SILVER', 'XAG']):
            suggestions = [
                'SI=F (Silver Futures)',
                'SLV (iShares Silver Trust ETF)'
            ]
        elif 'BTC' in symbol_upper or 'BITCOIN' in symbol_upper:
            suggestions = [
                'BTC-USD (Bitcoin)',
                'GBTC (Grayscale Bitcoin Trust)'
            ]
        elif 'ETH' in symbol_upper or 'ETHEREUM' in symbol_upper:
            suggestions = [
                'ETH-USD (Ethereum)',
                'ETHE (Grayscale Ethereum Trust)'
            ]
        
        return suggestions
    
    def calculate_spread(self, pair: str, price: float) -> float:
        """Calculate spread in price units based on asset type"""
        pip_value = self.get_pip_value(pair, price)
        pair_upper = pair.upper()
        
        # Adjust spread based on asset type
        if any(metal in pair_upper for metal in ['XAU', 'GOLD', 'GC=F', 'GLD', 'IAU']):
            spread_pips = 5  # Gold typically has wider spreads
        elif any(oil in pair_upper for oil in ['XTI', 'CL=F', 'USO']):
            spread_pips = 3  # Oil moderate spreads
        elif pair_upper.endswith('=F'):  # Other futures
            spread_pips = 3  # Futures moderate spreads
        elif any(crypto in pair_upper for crypto in ['BTC', 'ETH']):
            spread_pips = 10  # Crypto wider spreads
        else:
            spread_pips = self.spread_pips  # Default forex spread (2 pips)
            
        return spread_pips * pip_value
    
    def is_market_open(self, timestamp: datetime) -> bool:
        """Check if forex market is open"""
        weekday = timestamp.weekday()
        hour = timestamp.hour
        
        # Market is closed on weekends (Saturday & Sunday until 22:00)
        if weekday == 5:  # Saturday
            return False
        if weekday == 6 and hour < 22:  # Sunday before 22:00
            return False
        if weekday == 4 and hour >= 22:  # Friday after 22:00
            return False
            
        return True
    
    def can_trade(self, timestamp: datetime) -> bool:
        """Check if we can make a new trade"""
        if not self.is_market_open(timestamp):
            return False
            
        # Reset daily trade count at start of new day
        current_day = timestamp.date()
        if self.current_day != current_day:
            self.current_day = current_day
            self.daily_trade_count = 0
            
        # Check daily trade limit
        if self.daily_trade_count >= self.max_daily_trades:
            return False
            
        # Check minimum time between trades
        if self.last_trade_time:
            time_diff = timestamp - self.last_trade_time
            if time_diff < timedelta(hours=self.min_trade_interval):
                return False
                
        return True
    
    def get_model_prediction(self, df: pd.DataFrame, pair: str):
        """Get prediction from trained supervised learning model with optional entry suggestions"""
        try:
            # Clean pair name for prediction (remove =X suffix)
            clean_pair = pair.replace('=X', '')
            
            # Use the predictor's predict method with current confidence threshold
            prediction = self.predictor.predict(df, min_confidence=self.min_confidence)
            
            # Add optimal entry suggestions if enabled and we have a trading signal
            if self.use_optimal_entry and prediction['action'] != 'hold':
                current_price = df['close'].iloc[-1]
                entry_info = self.calculate_optimal_entry(df, prediction['action'], current_price, pair)
                prediction['entry_info'] = entry_info
            
            return prediction
            
        except Exception as e:
            print(f"⚠️ Prediction error for {pair}: {str(e)}")
            return {'action': 'hold', 'confidence': 0.0}
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, pair: str) -> float:
        """Calculate position size based on 2% risk management"""
        # Calculate risk distance
        risk_distance = abs(entry_price - stop_loss)
        
        # Calculate risk amount (2% of current balance)
        risk_amount = self.current_balance * self.risk_per_trade
        
        # Calculate pip value
        pip_value = self.get_pip_value(pair, entry_price)
        
        # Calculate position size
        # Position size = Risk Amount / (Risk Distance in currency units)
        position_size = risk_amount / risk_distance
        
        return position_size
    
    def open_position(self, pair: str, prediction: dict, price: float, timestamp: datetime):
        """Open a new position based on model prediction using realistic entry logic"""
        if self.current_position:
            return  # Already have a position
            
        action = prediction['action']
        confidence = prediction['confidence']
        
        # Check if prediction includes SL/TP levels
        if 'stop_loss' not in prediction or 'take_profit' not in prediction:
            print(f"⚠️ No SL/TP levels in prediction for {pair}")
            return
            
        stop_loss = prediction['stop_loss']
        take_profit = prediction['take_profit']
        
        # Verify risk-reward ratio
        if 'risk_reward_ratio' in prediction:
            if prediction['risk_reward_ratio'] < self.min_risk_reward:
                print(f"⚠️ Risk-reward ratio too low: {prediction['risk_reward_ratio']:.2f}")
                return
        
        # Calculate spread
        spread = self.calculate_spread(pair, price)
        
        # Determine entry strategy
        entry_type = 'MARKET'
        entry_price = price  # Default to market price
        
        if self.use_optimal_entry and 'entry_info' in prediction:
            entry_info = prediction['entry_info']
            suggested_entry_type = entry_info['entry_type']
            optimal_entry_price = entry_info['optimal_entry']
            
            if suggested_entry_type == 'LIMIT':
                # Use optimal entry price for better execution
                entry_type = 'LIMIT'
                print(f"📍 LIMIT order executed: {optimal_entry_price:.5f} (improved from market {price:.5f})")
                print(f"📏 Entry improvement: {entry_info.get('entry_distance_pips', 0):.1f} pips")
            else:
                print(f"� Using MARKET entry: {price:.5f}")
                entry_type = 'MARKET'
        else:
            print(f"📍 Using MARKET entry: {price:.5f}")
        
        # Entry price with spread applied to actual entry price
        # Use optimal entry price if available, otherwise current market price
        if self.use_optimal_entry and 'entry_info' in prediction and prediction['entry_info']['entry_type'] == 'LIMIT':
            actual_entry_price = prediction['entry_info']['optimal_entry']
        else:
            actual_entry_price = price
            
        if action == 'long':
            final_entry_price = actual_entry_price + spread
        else:
            final_entry_price = actual_entry_price - spread
            
        # Calculate position size using 2% risk management
        position_size = self.calculate_position_size(final_entry_price, stop_loss, pair)
        
        # Verify we have enough balance (margin requirement)
        required_margin = (position_size * final_entry_price) / self.leverage
        if required_margin > self.current_balance * 0.9:  # Keep 10% buffer
            print(f"⚠️ Insufficient margin for {pair} trade")
            return
        
        self.current_position = {
            'pair': pair,
            'direction': action,
            'entry_price': final_entry_price,
            'position_size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'timestamp': timestamp,
            'confidence': confidence,
            'risk_reward': prediction.get('risk_reward_ratio', 0),
            'entry_type': entry_type,
            'suggested_entry': entry_info.get('optimal_entry', price) if self.use_optimal_entry and 'entry_info' in prediction else price
        }
        
        self.daily_trade_count += 1
        self.last_trade_time = timestamp
        
        # Calculate risk and reward in pips
        pip_value = self.get_pip_value(pair, final_entry_price)
        risk_pips = abs(final_entry_price - stop_loss) / pip_value
        reward_pips = abs(take_profit - final_entry_price) / pip_value
        
        print(f"🟢 {action.upper()} {pair}: Size={position_size:.0f} @ {final_entry_price:.5f} ({entry_type})")
        print(f"   📊 SL={stop_loss:.5f} ({risk_pips:.1f} pips) | TP={take_profit:.5f} ({reward_pips:.1f} pips)")
        print(f"   🎯 Confidence: {confidence:.1%} | RR: 1:{prediction.get('risk_reward_ratio', 0):.2f}")
        
        # Show additional info for LIMIT orders
        if self.use_optimal_entry and 'entry_info' in prediction and entry_type == 'LIMIT':
            entry_info = prediction['entry_info']
            print(f"   🔺 Resistance: {entry_info['resistance_level']:.5f}")
            print(f"   🔻 Support: {entry_info['support_level']:.5f}")
            print(f"   � ATR: {entry_info['atr']:.5f}")
    
    def check_exit_conditions(self, current_price: float, timestamp: datetime) -> bool:
        """Check if position should be closed"""
        if not self.current_position:
            return False
            
        pos = self.current_position
        
        # Check stop loss and take profit
        if pos['direction'] == 'long':
            if current_price <= pos['stop_loss']:
                self.close_position(current_price, timestamp, 'Stop Loss')
                return True
            elif current_price >= pos['take_profit']:
                self.close_position(current_price, timestamp, 'Take Profit')
                return True
        else:  # short
            if current_price >= pos['stop_loss']:
                self.close_position(current_price, timestamp, 'Stop Loss')
                return True
            elif current_price <= pos['take_profit']:
                self.close_position(current_price, timestamp, 'Take Profit')
                return True
                
        return False
    
    def close_position(self, price: float, timestamp: datetime, reason: str = 'Manual'):
        """Close current position"""
        if not self.current_position:
            return
            
        pos = self.current_position
        spread = self.calculate_spread(pos['pair'], price)
        
        # Exit price with spread
        if pos['direction'] == 'long':
            exit_price = price - spread
        else:
            exit_price = price + spread
            
        # Calculate P&L
        if pos['direction'] == 'long':
            price_diff = exit_price - pos['entry_price']
        else:
            price_diff = pos['entry_price'] - exit_price
            
        # P&L in account currency
        pnl = price_diff * pos['position_size']
        
        # Update balance
        old_balance = self.current_balance
        self.current_balance += pnl
        
        # Prevent balance going negative
        if self.current_balance < 0:
            self.current_balance = 0
            
        # Calculate pips
        pip_value = self.get_pip_value(pos['pair'], pos['entry_price'])
        pips_gained = price_diff / pip_value
        
        # Record trade
        trade = {
            'pair': pos['pair'],
            'direction': pos['direction'],
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'position_size': pos['position_size'],
            'pnl': pnl,
            'pips': pips_gained,
            'entry_time': pos['timestamp'],
            'exit_time': timestamp,
            'duration': timestamp - pos['timestamp'],
            'reason': reason,
            'confidence': pos['confidence'],
            'risk_reward': pos['risk_reward'],
            'balance_before': old_balance,
            'balance_after': self.current_balance,
            'entry_type': pos.get('entry_type', 'MARKET'),
            'suggested_entry': pos.get('suggested_entry', pos['entry_price']),
            'actual_entry': pos['entry_price']
        }
        
        self.trades_history.append(trade)
        
        pnl_pct = (pnl / old_balance) * 100
        color = "🟢" if pnl > 0 else "🔴"
        print(f"{color} CLOSE {pos['pair']} ({reason}): {pips_gained:+.1f} pips | P&L=${pnl:+.2f} ({pnl_pct:+.1f}%) | Balance=${self.current_balance:.2f}")
        
        self.current_position = None
    
    def run_backtest(self, start_date: str, end_date: str, pair: str = 'EURUSD=X'):
        """Run realistic backtest"""
        print(f"\n🚀 Starting realistic backtest for Trade-2.py model")
        print(f"📅 Period: {start_date} to {end_date}")
        print(f"💱 Pair: {pair}")
        print("=" * 60)
        
        # Load data
        print(f"📥 Loading {pair} data...")
        ticker = yf.Ticker(pair)
        
        try:
            df = ticker.history(start=start_date, end=end_date, interval='1h')
        except Exception as e:
            print(f"❌ Error loading data for {pair}: {e}")
            suggestions = self.get_suggested_symbols(pair)
            if suggestions:
                print(f"💡 Suggested alternatives for {pair}:")
                for suggestion in suggestions:
                    print(f"   - {suggestion}")
            return
        
        if df.empty:
            print(f"❌ No data for {pair}")
            suggestions = self.get_suggested_symbols(pair)
            if suggestions:
                print(f"💡 Suggested alternatives for {pair}:")
                for suggestion in suggestions:
                    print(f"   - {suggestion}")
                print(f"\n🔄 Try running with one of these symbols:")
                for i, suggestion in enumerate(suggestions[:3]):
                    symbol = suggestion.split(' ')[0]
                    print(f"   python realistic_trade2_backtest.py --model best_model.pth --pair {symbol} --start {start_date} --end {end_date}")
            return
            
        # Normalize column names
        df.columns = [col.lower() for col in df.columns]
        
        print(f"✅ Loaded {len(df)} hourly candles")
        
        # Track statistics
        signals_generated = 0
        trades_attempted = 0
        confident_signals = 0
        
        # Run simulation
        for i, (timestamp, row) in enumerate(df.iterrows()):
            # Need enough history for model prediction
            if i < self.sequence_length + 50:
                continue
                
            # Update equity curve
            if self.current_position:
                # Mark-to-market current position
                current_price = row['close']
                pos = self.current_position
                spread = self.calculate_spread(pos['pair'], current_price)
                
                if pos['direction'] == 'long':
                    mark_price = current_price - spread
                    unrealized_pnl = (mark_price - pos['entry_price']) * pos['position_size']
                else:
                    mark_price = current_price + spread
                    unrealized_pnl = (pos['entry_price'] - mark_price) * pos['position_size']
                    
                current_equity = self.current_balance + unrealized_pnl
            else:
                current_equity = self.current_balance
                
            self.equity_curve.append({
                'timestamp': timestamp,
                'balance': self.current_balance,
                'equity': current_equity
            })
            
            # Check exit conditions first
            if self.current_position:
                if self.check_exit_conditions(row['close'], timestamp):
                    continue  # Position was closed
                    
            # Generate trading signals if no position and can trade
            if not self.current_position and self.can_trade(timestamp):
                try:
                    # Get recent data for prediction
                    recent_data = df.iloc[:i+1]  # All data up to current point
                    
                    if len(recent_data) < self.sequence_length:
                        continue
                        
                    # Get model prediction
                    prediction = self.get_model_prediction(recent_data, pair)
                    signals_generated += 1
                    
                    # Check if model is confident enough to trade
                    if prediction['action'] != 'hold' and prediction['confidence'] >= self.min_confidence:
                        confident_signals += 1
                        self.open_position(pair, prediction, row['close'], timestamp)
                        trades_attempted += 1
                        
                except Exception as e:
                    print(f"⚠️ Error at step {i}: {str(e)}")
                    continue
            
            # Progress update
            if i % 168 == 0:  # Every week
                print(f"📊 Week {i//168}: Balance=${self.current_balance:.2f} | Trades={len(self.trades_history)} | Signals={signals_generated}")
        
        # Close any remaining position
        if self.current_position:
            final_price = df.iloc[-1]['close']
            self.close_position(final_price, df.index[-1], 'End of Period')
        
        print(f"\n📈 SIGNAL SUMMARY:")
        print(f"   Total signals: {signals_generated}")
        print(f"   Confident signals: {confident_signals} ({confident_signals/max(1,signals_generated)*100:.1f}%)")
        print(f"   Trades attempted: {trades_attempted}")
        
        self.print_results()
        self.save_results()
    
    def print_results(self):
        """Print detailed backtest results"""
        print("\n" + "=" * 60)
        print("📊 REALISTIC TRADE-2 BACKTEST RESULTS")
        print("=" * 60)
        
        final_balance = self.current_balance
        total_return = final_balance - self.initial_balance
        return_pct = (total_return / self.initial_balance) * 100
        
        print(f"💰 Initial Balance:    ${self.initial_balance:,.2f}")
        print(f"💰 Final Balance:      ${final_balance:,.2f}")
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
                print(f"Average Win:           ${winning_trades['pnl'].mean():.2f} ({winning_trades['pips'].mean():.1f} pips)")
                print(f"Largest Win:           ${winning_trades['pnl'].max():.2f} ({winning_trades['pips'].max():.1f} pips)")
            
            if len(losing_trades) > 0:
                print(f"Average Loss:          ${losing_trades['pnl'].mean():.2f} ({losing_trades['pips'].mean():.1f} pips)")
                print(f"Largest Loss:          ${losing_trades['pnl'].min():.2f} ({losing_trades['pips'].min():.1f} pips)")
            
            print(f"Average Trade:         ${trades_df['pnl'].mean():.2f} ({trades_df['pips'].mean():.1f} pips)")
            print(f"Average Confidence:    {trades_df['confidence'].mean():.1%}")
            print(f"Average Risk-Reward:   1:{trades_df['risk_reward'].mean():.2f}")
            
            # Entry optimization statistics (only if enabled)
            if self.use_optimal_entry:
                market_orders = trades_df[trades_df['entry_type'] == 'MARKET']
                limit_orders = trades_df[trades_df['entry_type'] == 'LIMIT']
                
                print(f"\n📍 ENTRY ANALYSIS (Optimal Entry Enabled)")
                print(f"Market Orders:         {len(market_orders)} ({len(market_orders)/len(trades_df)*100:.1f}%)")
                print(f"LIMIT Orders:          {len(limit_orders)} ({len(limit_orders)/len(trades_df)*100:.1f}%)")
                print(f"� Note: LIMIT orders used optimal entry prices")
                
                if len(market_orders) > 0:
                    market_avg_pnl = market_orders['pnl'].mean()
                    market_win_rate = len(market_orders[market_orders['pnl'] > 0]) / len(market_orders) * 100
                    print(f"Market Orders Avg P&L: ${market_avg_pnl:.2f}")
                    print(f"Market Orders Win Rate: {market_win_rate:.1f}%")
                
                if len(limit_orders) > 0:
                    limit_avg_pnl = limit_orders['pnl'].mean()
                    limit_win_rate = len(limit_orders[limit_orders['pnl'] > 0]) / len(limit_orders) * 100
                    print(f"LIMIT Orders Avg P&L: ${limit_avg_pnl:.2f}")
                    print(f"LIMIT Orders Win Rate: {limit_win_rate:.1f}%")
                    
                    # Calculate average improvement
                    limit_trades = trades_df[trades_df['entry_type'] == 'LIMIT']
                    if len(limit_trades) > 0:
                        avg_improvement = (limit_trades['suggested_entry'] - limit_trades['actual_entry']).abs().mean()
                        pip_value = self.get_pip_value('EURUSD=X', 1.1)  # Rough estimate
                        avg_improvement_pips = avg_improvement / pip_value
                        print(f"Avg Entry Improvement: {avg_improvement_pips:.1f} pips per LIMIT order")
            
            # Calculate max drawdown
            equity_df = pd.DataFrame(self.equity_curve)
            if not equity_df.empty:
                equity_df['running_max'] = equity_df['equity'].expanding().max()
                equity_df['drawdown'] = equity_df['equity'] - equity_df['running_max']
                max_drawdown = equity_df['drawdown'].min()
                max_drawdown_pct = (max_drawdown / equity_df['running_max'].max()) * 100
                
                print(f"\n📉 RISK METRICS")
                print(f"Max Drawdown:          ${max_drawdown:.2f} ({max_drawdown_pct:.2f}%)")
            
            # Trading frequency
            if len(trades_df) > 0:
                duration = trades_df['exit_time'].max() - trades_df['entry_time'].min()
                trades_per_day = len(trades_df) / max(1, duration.days)
                print(f"Trades per Day:        {trades_per_day:.2f}")
            
            print(f"\n🕐 RECENT TRADES (Last 10):")
            for _, trade in trades_df.tail(10).iterrows():
                pnl_sign = "+" if trade['pnl'] > 0 else ""
                confidence_str = f"{trade['confidence']:.0%}"
                
                if self.use_optimal_entry:
                    entry_type_str = trade['entry_type']
                    if entry_type_str == 'MARKET':
                        entry_emoji = "🟢"
                        entry_display = "MARKET"
                    else:  # LIMIT
                        entry_emoji = "�"
                        entry_display = "LIMIT"
                    print(f"   {trade['direction'].upper()} {trade['pair']} | {pnl_sign}${trade['pnl']:.2f} ({trade['pips']:+.1f} pips) | {confidence_str} | {entry_emoji}{entry_display} | {trade['reason']}")
                else:
                    print(f"   {trade['direction'].upper()} {trade['pair']} | {pnl_sign}${trade['pnl']:.2f} ({trade['pips']:+.1f} pips) | {confidence_str} | {trade['reason']}")
        
        else:
            print("\n⚠️ No trades executed during backtest period")
            print("   Consider:")
            print("   - Lowering confidence threshold")
            print("   - Checking model compatibility with data")
            print("   - Verifying SL/TP calculation")
    
    def save_results(self):
        """Save backtest results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save summary
        summary = {
            'initial_balance': self.initial_balance,
            'final_balance': self.current_balance,
            'total_return': self.current_balance - self.initial_balance,
            'return_pct': ((self.current_balance - self.initial_balance) / self.initial_balance) * 100,
            'total_trades': len(self.trades_history),
            'model_path': self.model_path,
            'min_confidence': self.min_confidence,
            'timestamp': timestamp
        }
        
        with open(f'results/trade2_backtest_summary_{timestamp}.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save trades
        if self.trades_history:
            trades_df = pd.DataFrame(self.trades_history)
            trades_df.to_csv(f'results/trade2_backtest_trades_{timestamp}.csv', index=False)
            print(f"💾 Trades saved to: results/trade2_backtest_trades_{timestamp}.csv")
        
        # Save equity curve
        if self.equity_curve:
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df.to_csv(f'results/trade2_backtest_equity_{timestamp}.csv', index=False)
            print(f"💾 Equity curve saved to: results/trade2_backtest_equity_{timestamp}.csv")
        
        print(f"💾 Summary saved to: results/trade2_backtest_summary_{timestamp}.json")

def main():
    parser = argparse.ArgumentParser(description='Realistic Backtester for Trade-2.py Model')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model file (.pth)')
    parser.add_argument('--start', type=str, default='2024-11-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2024-12-01', help='End date (YYYY-MM-DD)')
    parser.add_argument('--balance', type=float, default=1000.0, help='Initial balance')
    parser.add_argument('--pair', type=str, default='EURUSD=X', help='Currency pair to trade')
    parser.add_argument('--confidence', type=float, default=0.6, help='Minimum confidence threshold (0.0-1.0)')
    parser.add_argument('--optimal-entry', action='store_true', default=True, help='Use optimal entry suggestions (default: enabled)')
    parser.add_argument('--no-optimal-entry', action='store_true', help='Disable optimal entry suggestions')
    
    args = parser.parse_args()
    
    # Handle optimal entry flag
    use_optimal_entry = args.optimal_entry and not args.no_optimal_entry
    
    # Initialize backtester
    backtester = RealisticTrade2Backtester(
        model_path=args.model,
        initial_balance=args.balance,
        use_optimal_entry=use_optimal_entry
    )
    
    # Set confidence threshold if provided
    if args.confidence:
        backtester.min_confidence = args.confidence
        print(f"🎯 Using confidence threshold: {args.confidence:.1%}")
    
    # Run backtest
    backtester.run_backtest(
        start_date=args.start,
        end_date=args.end,
        pair=args.pair
    )

if __name__ == "__main__":
    main()
