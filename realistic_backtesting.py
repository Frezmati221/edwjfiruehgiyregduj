import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple
import json
import os
from trade_2 import SupervisedForexPredictor, load_forex_data
import yfinance as yf

warnings.filterwarnings('ignore')

class RealisticForexBacktester:
    """Comprehensive backtesting system with realistic trading conditions"""
    
    def __init__(self, initial_balance=10000, leverage=1, spread_pips=None):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.leverage = leverage
        self.spread_pips = spread_pips or {
            'EURUSD': 1.5, 'GBPUSD': 2.0, 'USDJPY': 1.8, 
            'AUDUSD': 2.2, 'USDCAD': 2.5, 'USDCHF': 2.8
        }
        
        # Trading costs
        self.commission_per_lot = 7.0  # $7 per standard lot
        self.swap_rates = {  # Daily swap rates (in pips)
            'EURUSD': {'long': -0.5, 'short': 0.2},
            'GBPUSD': {'long': -0.8, 'short': 0.4},
            'USDJPY': {'long': -1.2, 'short': 0.8},
            'AUDUSD': {'long': -0.6, 'short': 0.3},
            'USDCAD': {'long': -0.4, 'short': 0.1},
            'USDCHF': {'long': -0.3, 'short': 0.0}
        }
        
        # Risk management
        self.max_risk_per_trade = 0.02  # 2% of balance
        self.max_drawdown_limit = 0.20  # 20% max drawdown
        self.max_daily_trades = 10
        self.min_confidence_threshold = 0.7
        
        # Tracking variables
        self.trades = []
        self.equity_curve = [initial_balance]
        self.drawdown_series = []
        self.daily_pnl = []
        self.current_positions = {}
        
    def get_pip_value(self, pair, lot_size=1.0):
        """Calculate pip value in USD for given pair and lot size"""
        if 'JPY' in pair:
            return lot_size * 1000  # $1000 per pip for 1 standard lot
        else:
            return lot_size * 10    # $10 per pip for 1 standard lot
    
    def calculate_position_size(self, pair, entry_price, stop_loss_pips):
        """Calculate optimal position size based on risk management"""
        risk_amount = self.current_balance * self.max_risk_per_trade
        pip_value = self.get_pip_value(pair, lot_size=1.0)
        
        # Position size = Risk Amount / (Stop Loss in Pips * Pip Value)
        max_position_size = risk_amount / (stop_loss_pips * pip_value)
        
        # Apply leverage limit
        max_leverage_position = (self.current_balance * self.leverage) / (entry_price * 100000)
        
        return min(max_position_size, max_leverage_position, 10.0)  # Max 10 lots
    
    def calculate_spread_cost(self, pair, position_size):
        """Calculate spread cost for opening position"""
        spread = self.spread_pips.get(pair, 2.0)
        pip_value = self.get_pip_value(pair, position_size)
        return spread * pip_value / 1000  # Convert pips to dollars
    
    def calculate_commission(self, position_size):
        """Calculate commission cost"""
        return position_size * self.commission_per_lot
    
    def calculate_swap(self, pair, direction, position_size, days_held):
        """Calculate overnight financing costs"""
        if pair not in self.swap_rates:
            return 0
        
        swap_rate = self.swap_rates[pair][direction]
        pip_value = self.get_pip_value(pair, position_size)
        return (swap_rate * pip_value * days_held) / 1000
    
    def open_position(self, pair, direction, entry_price, confidence, stop_loss_pips=50, take_profit_pips=100, timestamp=None):
        """Open a new trading position"""
        
        # Check if we can trade (daily limit, etc.)
        today_trades = sum(1 for t in self.trades if t['timestamp'].date() == timestamp.date())
        if today_trades >= self.max_daily_trades:
            return False, "Daily trade limit reached"
        
        # Check confidence threshold
        if confidence < self.min_confidence_threshold:
            return False, f"Confidence {confidence:.3f} below threshold {self.min_confidence_threshold}"
        
        # Calculate position size
        position_size = self.calculate_position_size(pair, entry_price, stop_loss_pips)
        
        if position_size < 0.01:  # Minimum position size
            return False, "Position size too small"
        
        # Calculate entry costs
        spread_cost = self.calculate_spread_cost(pair, position_size)
        commission = self.calculate_commission(position_size)
        total_entry_cost = spread_cost + commission
        
        # Check if we have enough balance
        required_margin = (entry_price * position_size * 100000) / self.leverage
        if required_margin + total_entry_cost > self.current_balance * 0.8:  # Use max 80% of balance
            return False, "Insufficient margin"
        
        # Create position
        position = {
            'id': len(self.trades) + 1,
            'pair': pair,
            'direction': direction,
            'entry_price': entry_price,
            'position_size': position_size,
            'stop_loss': entry_price - (stop_loss_pips * self.get_pip_value(pair, 1) / 100000) if direction == 'long' else entry_price + (stop_loss_pips * self.get_pip_value(pair, 1) / 100000),
            'take_profit': entry_price + (take_profit_pips * self.get_pip_value(pair, 1) / 100000) if direction == 'long' else entry_price - (take_profit_pips * self.get_pip_value(pair, 1) / 100000),
            'timestamp': timestamp,
            'confidence': confidence,
            'entry_cost': total_entry_cost,
            'status': 'open'
        }
        
        self.current_positions[position['id']] = position
        self.current_balance -= total_entry_cost
        
        return True, f"Position opened: {position['id']}"
    
    def close_position(self, position_id, exit_price, exit_reason, timestamp=None):
        """Close an existing position"""
        
        if position_id not in self.current_positions:
            return False, "Position not found"
        
        position = self.current_positions[position_id]
        
        # Calculate P&L
        if position['direction'] == 'long':
            price_change = exit_price - position['entry_price']
        else:
            price_change = position['entry_price'] - exit_price
        
        # Convert to pips
        pip_value_unit = self.get_pip_value(position['pair'], 1) / 100000
        pips_gained = price_change / pip_value_unit
        
        # Calculate gross P&L
        gross_pnl = pips_gained * self.get_pip_value(position['pair'], position['position_size']) / 1000
        
        # Calculate exit costs
        exit_spread_cost = self.calculate_spread_cost(position['pair'], position['position_size']) / 2  # Half spread on exit
        exit_commission = self.calculate_commission(position['position_size'])
        
        # Calculate swap costs
        days_held = max(1, (timestamp - position['timestamp']).days)
        swap_cost = self.calculate_swap(position['pair'], position['direction'], position['position_size'], days_held)
        
        total_exit_cost = exit_spread_cost + exit_commission + abs(swap_cost)
        net_pnl = gross_pnl - position['entry_cost'] - total_exit_cost
        
        # Update balance
        self.current_balance += net_pnl
        
        # Record trade
        trade_record = {
            'id': position['id'],
            'pair': position['pair'],
            'direction': position['direction'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'position_size': position['position_size'],
            'entry_timestamp': position['timestamp'],
            'exit_timestamp': timestamp,
            'duration_hours': (timestamp - position['timestamp']).total_seconds() / 3600,
            'confidence': position['confidence'],
            'pips_gained': pips_gained,
            'gross_pnl': gross_pnl,
            'entry_cost': position['entry_cost'],
            'exit_cost': total_exit_cost,
            'swap_cost': swap_cost,
            'net_pnl': net_pnl,
            'exit_reason': exit_reason,
            'win': net_pnl > 0
        }
        
        self.trades.append(trade_record)
        del self.current_positions[position_id]
        
        # Update equity curve
        self.equity_curve.append(self.current_balance)
        
        # Calculate drawdown
        peak = max(self.equity_curve)
        drawdown = (peak - self.current_balance) / peak if peak > 0 else 0
        self.drawdown_series.append(drawdown)
        
        return True, f"Position closed: {net_pnl:.2f} USD"
    
    def check_stop_loss_take_profit(self, current_prices, timestamp):
        """Check if any positions should be closed due to SL/TP"""
        
        positions_to_close = []
        
        for pos_id, position in self.current_positions.items():
            pair = position['pair']
            if pair not in current_prices:
                continue
                
            current_price = current_prices[pair]
            
            # Check stop loss
            if position['direction'] == 'long' and current_price <= position['stop_loss']:
                positions_to_close.append((pos_id, current_price, 'Stop Loss'))
            elif position['direction'] == 'short' and current_price >= position['stop_loss']:
                positions_to_close.append((pos_id, current_price, 'Stop Loss'))
            
            # Check take profit
            elif position['direction'] == 'long' and current_price >= position['take_profit']:
                positions_to_close.append((pos_id, current_price, 'Take Profit'))
            elif position['direction'] == 'short' and current_price <= position['take_profit']:
                positions_to_close.append((pos_id, current_price, 'Take Profit'))
        
        # Close positions
        for pos_id, exit_price, reason in positions_to_close:
            self.close_position(pos_id, exit_price, reason, timestamp)
    
    def run_backtest(self, predictor, data_dict, start_date=None, end_date=None):
        """Run comprehensive backtest"""
        
        print("🔍 Starting realistic backtest...")
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Leverage: {self.leverage}:1")
        print(f"Max Risk per Trade: {self.max_risk_per_trade:.1%}")
        print("-" * 60)
        
        # Prepare data
        all_data = {}
        for pair, df in data_dict.items():
            df = df.copy()
            df['pair'] = pair
            all_data[pair] = df
        
        # Combine all data with timestamps
        combined_data = []
        for pair, df in all_data.items():
            for idx, row in df.iterrows():
                combined_data.append({
                    'timestamp': idx,
                    'pair': pair,
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row.get('volume', 0)
                })
        
        # Sort by timestamp
        combined_data = sorted(combined_data, key=lambda x: x['timestamp'])
        
        # Filter by date range if specified
        if start_date:
            combined_data = [d for d in combined_data if d['timestamp'] >= start_date]
        if end_date:
            combined_data = [d for d in combined_data if d['timestamp'] <= end_date]
        
        print(f"Backtesting period: {combined_data[0]['timestamp']} to {combined_data[-1]['timestamp']}")
        print(f"Total data points: {len(combined_data)}")
        
        # Group by timestamp for simultaneous processing
        timestamp_groups = {}
        for data_point in combined_data:
            ts = data_point['timestamp']
            if ts not in timestamp_groups:
                timestamp_groups[ts] = {}
            timestamp_groups[ts][data_point['pair']] = data_point
        
        processed_timestamps = 0
        
        for timestamp in sorted(timestamp_groups.keys()):
            current_prices = {}
            for pair, data_point in timestamp_groups[timestamp].items():
                current_prices[pair] = data_point['close']
            
            # Check existing positions for SL/TP
            self.check_stop_loss_take_profit(current_prices, timestamp)
            
            # Generate new signals for each pair
            for pair, data_point in timestamp_groups[timestamp].items():
                if pair not in data_dict:
                    continue
                
                # Get historical data up to current point
                pair_df = data_dict[pair]
                current_idx = pair_df.index.get_loc(timestamp)
                
                # Need at least sequence_length + 1 points for prediction
                if current_idx < predictor.sequence_length + 1:
                    continue
                
                # Get data up to current point (not including current candle)
                historical_data = pair_df.iloc[:current_idx]
                
                try:
                    # Get prediction with dynamic SL/TP
                    prediction = predictor.predict(historical_data, min_confidence=self.min_confidence_threshold, risk_reward_ratio=2.0)
                    
                    if prediction['action'] in ['long', 'short'] and 'stop_loss' in prediction:
                        # Use calculated SL/TP from the prediction
                        current_price = data_point['close']
                        stop_loss_price = prediction['stop_loss']
                        take_profit_price = prediction['take_profit']
                        
                        # Convert to pips for position sizing
                        pip_value = self.get_pip_value(pair, 1.0) / 100000
                        if prediction['action'] == 'long':
                            stop_loss_pips = (current_price - stop_loss_price) / pip_value
                            take_profit_pips = (take_profit_price - current_price) / pip_value
                        else:
                            stop_loss_pips = (stop_loss_price - current_price) / pip_value
                            take_profit_pips = (current_price - take_profit_price) / pip_value
                        
                        # Open new position with calculated SL/TP
                        success, message = self.open_position(
                            pair=pair,
                            direction=prediction['action'],
                            entry_price=current_price,
                            confidence=prediction['confidence'],
                            stop_loss_pips=max(10, min(100, stop_loss_pips)),  # Limit SL to 10-100 pips
                            take_profit_pips=max(20, min(200, take_profit_pips)), # Limit TP to 20-200 pips
                            timestamp=timestamp
                        )
                        
                        if success:
                            print(f"✅ {pair} {prediction['action'].upper()}: Entry={current_price:.5f}, "
                                  f"SL={stop_loss_price:.5f} ({stop_loss_pips:.1f}p), "
                                  f"TP={take_profit_price:.5f} ({take_profit_pips:.1f}p), "
                                  f"RR=1:{prediction['risk_reward_ratio']:.1f}, "
                                  f"Conf={prediction['confidence']:.1%}")
                        elif 'confidence' not in message and 'limit' not in message:
                            print(f"Failed to open {pair} {prediction['action']}: {message}")
                
                except Exception as e:
                    print(f"Error processing {pair} at {timestamp}: {str(e)}")
                    continue
            
            processed_timestamps += 1
            if processed_timestamps % 1000 == 0:
                print(f"Processed {processed_timestamps} timestamps... Current balance: ${self.current_balance:,.2f}")
                
                # Check drawdown limit
                if len(self.drawdown_series) > 0 and self.drawdown_series[-1] > self.max_drawdown_limit:
                    print(f"⚠️ Maximum drawdown limit ({self.max_drawdown_limit:.1%}) exceeded!")
                    break
        
        # Close all remaining positions at the end
        final_prices = current_prices
        for pos_id in list(self.current_positions.keys()):
            position = self.current_positions[pos_id]
            if position['pair'] in final_prices:
                self.close_position(pos_id, final_prices[position['pair']], 'End of Backtest', timestamp)
        
        print("\n✅ Backtest completed!")
        return self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive backtest report"""
        
        if not self.trades:
            return {"error": "No trades executed during backtest"}
        
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = sum(1 for t in self.trades if t['win'])
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # P&L metrics
        total_pnl = sum(t['net_pnl'] for t in self.trades)
        gross_profit = sum(t['net_pnl'] for t in self.trades if t['net_pnl'] > 0)
        gross_loss = sum(t['net_pnl'] for t in self.trades if t['net_pnl'] < 0)
        
        # Returns
        total_return = (self.current_balance - self.initial_balance) / self.initial_balance
        
        # Risk metrics
        returns = np.diff(self.equity_curve) / np.array(self.equity_curve[:-1])
        returns = returns[~np.isnan(returns)]
        
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            max_drawdown = max(self.drawdown_series) if self.drawdown_series else 0
        else:
            sharpe_ratio = 0
            max_drawdown = 0
        
        # Trading costs analysis
        total_spread_costs = sum(t['entry_cost'] + t['exit_cost'] for t in self.trades)
        total_swap_costs = sum(abs(t['swap_cost']) for t in self.trades)
        
        # Average trade metrics
        avg_win = gross_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = abs(gross_loss) / losing_trades if losing_trades > 0 else 0
        profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')
        
        # Confidence analysis
        avg_confidence = np.mean([t['confidence'] for t in self.trades])
        high_conf_trades = [t for t in self.trades if t['confidence'] > 0.8]
        high_conf_win_rate = sum(1 for t in high_conf_trades if t['win']) / len(high_conf_trades) if high_conf_trades else 0
        
        report = {
            'summary': {
                'initial_balance': self.initial_balance,
                'final_balance': self.current_balance,
                'total_return': total_return,
                'total_pnl': total_pnl,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio
            },
            'trade_analysis': {
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'largest_win': max([t['net_pnl'] for t in self.trades]),
                'largest_loss': min([t['net_pnl'] for t in self.trades]),
                'avg_trade_duration': np.mean([t['duration_hours'] for t in self.trades])
            },
            'cost_analysis': {
                'total_spread_costs': total_spread_costs,
                'total_swap_costs': total_swap_costs,
                'cost_as_percent_of_pnl': abs(total_spread_costs + total_swap_costs) / abs(total_pnl) if total_pnl != 0 else 0
            },
            'confidence_analysis': {
                'avg_confidence': avg_confidence,
                'high_confidence_trades': len(high_conf_trades),
                'high_confidence_win_rate': high_conf_win_rate
            },
            'monthly_returns': self.calculate_monthly_returns(),
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'drawdown_series': self.drawdown_series
        }
        
        return report
    
    def calculate_monthly_returns(self):
        """Calculate monthly returns breakdown"""
        if not self.trades:
            return {}
        
        monthly_pnl = {}
        for trade in self.trades:
            month_key = trade['exit_timestamp'].strftime('%Y-%m')
            if month_key not in monthly_pnl:
                monthly_pnl[month_key] = 0
            monthly_pnl[month_key] += trade['net_pnl']
        
        return monthly_pnl
    
    def plot_results(self, report, save_path='backtest_results.png'):
        """Create comprehensive visualization of backtest results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Forex AI Backtest Results', fontsize=16, fontweight='bold')
        
        # Equity curve
        axes[0, 0].plot(self.equity_curve, linewidth=2)
        axes[0, 0].set_title('Equity Curve')
        axes[0, 0].set_ylabel('Balance ($)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=self.initial_balance, color='r', linestyle='--', alpha=0.5, label='Initial Balance')
        axes[0, 0].legend()
        
        # Drawdown
        axes[0, 1].fill_between(range(len(self.drawdown_series)), 0, [-d*100 for d in self.drawdown_series], 
                               color='red', alpha=0.3)
        axes[0, 1].set_title('Drawdown (%)')
        axes[0, 1].set_ylabel('Drawdown (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # P&L distribution
        pnls = [t['net_pnl'] for t in self.trades]
        axes[1, 0].hist(pnls, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        axes[1, 0].set_title('P&L Distribution')
        axes[1, 0].set_xlabel('P&L ($)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Monthly returns
        monthly_returns = report['monthly_returns']
        if monthly_returns:
            months = list(monthly_returns.keys())
            returns = list(monthly_returns.values())
            colors = ['green' if r > 0 else 'red' for r in returns]
            
            axes[1, 1].bar(range(len(months)), returns, color=colors, alpha=0.7)
            axes[1, 1].set_title('Monthly P&L')
            axes[1, 1].set_ylabel('P&L ($)')
            axes[1, 1].set_xticks(range(len(months)))
            axes[1, 1].set_xticklabels(months, rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return save_path


def print_detailed_report(report):
    """Print comprehensive backtest report"""
    
    print("\n" + "="*80)
    print("📊 REALISTIC FOREX BACKTEST RESULTS")
    print("="*80)
    
    summary = report['summary']
    trade_analysis = report['trade_analysis']
    cost_analysis = report['cost_analysis']
    confidence_analysis = report['confidence_analysis']
    
    print(f"\n💰 PERFORMANCE SUMMARY:")
    print(f"  Initial Balance:     ${summary['initial_balance']:>10,.2f}")
    print(f"  Final Balance:       ${summary['final_balance']:>10,.2f}")
    print(f"  Total Return:        {summary['total_return']:>10.2%}")
    print(f"  Total P&L:           ${summary['total_pnl']:>10,.2f}")
    print(f"  Max Drawdown:        {summary['max_drawdown']:>10.2%}")
    print(f"  Sharpe Ratio:        {summary['sharpe_ratio']:>10.2f}")
    
    print(f"\n📈 TRADE ANALYSIS:")
    print(f"  Total Trades:        {summary['total_trades']:>10}")
    print(f"  Winning Trades:      {trade_analysis['winning_trades']:>10}")
    print(f"  Losing Trades:       {trade_analysis['losing_trades']:>10}")
    print(f"  Win Rate:            {summary['win_rate']:>10.2%}")
    print(f"  Profit Factor:       {summary['profit_factor']:>10.2f}")
    print(f"  Average Win:         ${trade_analysis['avg_win']:>10.2f}")
    print(f"  Average Loss:        ${trade_analysis['avg_loss']:>10.2f}")
    print(f"  Largest Win:         ${trade_analysis['largest_win']:>10.2f}")
    print(f"  Largest Loss:        ${trade_analysis['largest_loss']:>10.2f}")
    print(f"  Avg Trade Duration:  {trade_analysis['avg_trade_duration']:>10.1f} hours")
    
    print(f"\n💸 COST ANALYSIS:")
    print(f"  Total Spread Costs:  ${cost_analysis['total_spread_costs']:>10.2f}")
    print(f"  Total Swap Costs:    ${cost_analysis['total_swap_costs']:>10.2f}")
    print(f"  Costs as % of P&L:   {cost_analysis['cost_as_percent_of_pnl']:>10.2%}")
    
    print(f"\n🎯 CONFIDENCE ANALYSIS:")
    print(f"  Average Confidence:  {confidence_analysis['avg_confidence']:>10.2%}")
    print(f"  High-Conf Trades:    {confidence_analysis['high_confidence_trades']:>10}")
    print(f"  High-Conf Win Rate:  {confidence_analysis['high_confidence_win_rate']:>10.2%}")
    
    # Monthly breakdown
    print(f"\n📅 MONTHLY BREAKDOWN:")
    monthly_returns = report['monthly_returns']
    for month, pnl in monthly_returns.items():
        status = "📈" if pnl > 0 else "📉"
        print(f"  {month}: {status} ${pnl:>8.2f}")
    
    print("\n" + "="*80)


def run_comprehensive_backtest():
    """Run complete backtesting pipeline"""
    
    print("🚀 Loading trained model and data...")
    
    # Load forex data
    data = load_forex_data(period="2y", interval="1h")
    
    if not data:
        print("❌ No data available for backtesting")
        return
    
    # Load trained model
    predictor = SupervisedForexPredictor()
    
    try:
        predictor.load_model('best_model.pth')
        print("✅ Model loaded successfully")
    except FileNotFoundError:
        print("❌ Model file 'best_model.pth' not found. Please train the model first.")
        print("   Run: python trade-2.py")
        return
    
    # Initialize backtester
    backtester = RealisticForexBacktester(
        initial_balance=10000,
        leverage=30,  # Typical forex leverage
        spread_pips={'EURUSD': 1.5, 'GBPUSD': 2.0, 'USDJPY': 1.8, 'AUDUSD': 2.2, 'USDCAD': 2.5, 'USDCHF': 2.8}
    )
    
    # Run backtest on last 6 months of data
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=180)
    
    print(f"\n🔍 Running backtest from {start_date.date()} to {end_date.date()}")
    
    # Filter data for backtest period
    backtest_data = {}
    for pair, df in data.items():
        mask = (df.index >= start_date) & (df.index <= end_date)
        backtest_data[pair] = df[mask]
        print(f"  {pair}: {len(backtest_data[pair])} data points")
    
    # Run backtest
    report = backtester.run_backtest(predictor, backtest_data, start_date, end_date)
    
    if 'error' in report:
        print(f"❌ Backtest failed: {report['error']}")
        return
    
    # Print results
    print_detailed_report(report)
    
    # Create visualizations
    plot_path = backtester.plot_results(report, 'realistic_backtest_results.png')
    print(f"\n📊 Results chart saved as: {plot_path}")
    
    # Save detailed results
    results_file = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Convert numpy types for JSON serialization
    json_report = convert_numpy_types(report)
    
    with open(results_file, 'w') as f:
        json.dump(json_report, f, indent=2, default=str)
    
    print(f"📁 Detailed results saved as: {results_file}")
    
    # Performance rating
    win_rate = report['summary']['win_rate']
    total_return = report['summary']['total_return']
    max_drawdown = report['summary']['max_drawdown']
    sharpe_ratio = report['summary']['sharpe_ratio']
    
    print(f"\n🏆 PERFORMANCE RATING:")
    
    # Calculate overall score
    score = 0
    if win_rate > 0.6: score += 25
    elif win_rate > 0.5: score += 15
    elif win_rate > 0.4: score += 5
    
    if total_return > 0.2: score += 25
    elif total_return > 0.1: score += 15
    elif total_return > 0: score += 5
    
    if max_drawdown < 0.1: score += 25
    elif max_drawdown < 0.2: score += 15
    elif max_drawdown < 0.3: score += 5
    
    if sharpe_ratio > 2: score += 25
    elif sharpe_ratio > 1: score += 15
    elif sharpe_ratio > 0.5: score += 5
    
    if score >= 80:
        rating = "🌟 EXCELLENT - Ready for live trading"
    elif score >= 60:
        rating = "✅ GOOD - Consider live trading with small size"
    elif score >= 40:
        rating = "⚠️ AVERAGE - Needs improvement"
    else:
        rating = "❌ POOR - Requires significant optimization"
    
    print(f"  Score: {score}/100")
    print(f"  Rating: {rating}")


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    else:
        return obj


if __name__ == "__main__":
    run_comprehensive_backtest()
