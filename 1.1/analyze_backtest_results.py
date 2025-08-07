import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# Load the backtest results
try:
    trades_df = pd.read_csv('realistic_backtest_trades.csv')
    equity_df = pd.read_csv('realistic_backtest_equity.csv')
    
    print("📊 Analyzing Realistic Backtest Results")
    print("=" * 50)
    
    # Convert timestamps
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'], utc=True)
    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'], utc=True)
    equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'], utc=True)
    
    # Create a comprehensive analysis plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Realistic Backtest Analysis - $1,500 Initial Balance', fontsize=16, fontweight='bold')
    
    # 1. Equity Curve
    axes[0, 0].plot(equity_df['timestamp'], equity_df['balance'], linewidth=2, color='navy')
    axes[0, 0].axhline(y=1500, color='green', linestyle='--', alpha=0.7, label='Initial Balance')
    axes[0, 0].set_title('Balance Evolution', fontweight='bold')
    axes[0, 0].set_ylabel('Balance ($)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # 2. P&L Distribution
    axes[0, 1].hist(trades_df['pnl'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    axes[0, 1].set_title('P&L Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('P&L ($)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Wins vs Losses by Pair
    trades_df['result'] = trades_df['pnl'].apply(lambda x: 'Win' if x > 0 else 'Loss')
    pair_performance = trades_df.groupby(['pair', 'result']).size().unstack(fill_value=0)
    pair_performance.plot(kind='bar', ax=axes[0, 2], color=['red', 'green'])
    axes[0, 2].set_title('Wins vs Losses by Pair', fontweight='bold')
    axes[0, 2].set_ylabel('Number of Trades')
    axes[0, 2].legend(['Loss', 'Win'])
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # 4. Monthly P&L
    trades_df['month'] = trades_df['entry_time'].dt.to_period('M')
    monthly_pnl = trades_df.groupby('month')['pnl'].sum()
    monthly_pnl.plot(kind='bar', ax=axes[1, 0], color=['red' if x < 0 else 'green' for x in monthly_pnl])
    axes[1, 0].set_title('Monthly P&L', fontweight='bold')
    axes[1, 0].set_ylabel('P&L ($)')
    axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 5. Trade Duration Analysis
    trades_df['duration_hours'] = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / 3600
    axes[1, 1].scatter(trades_df['duration_hours'], trades_df['pnl'], alpha=0.6, 
                      c=['green' if x > 0 else 'red' for x in trades_df['pnl']])
    axes[1, 1].set_title('P&L vs Trade Duration', fontweight='bold')
    axes[1, 1].set_xlabel('Duration (hours)')
    axes[1, 1].set_ylabel('P&L ($)')
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Cumulative Returns
    trades_df_sorted = trades_df.sort_values('exit_time')
    cumulative_pnl = trades_df_sorted['pnl'].cumsum()
    axes[1, 2].plot(range(len(cumulative_pnl)), cumulative_pnl, linewidth=2, color='purple')
    axes[1, 2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 2].set_title('Cumulative P&L by Trade', fontweight='bold')
    axes[1, 2].set_xlabel('Trade Number')
    axes[1, 2].set_ylabel('Cumulative P&L ($)')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('realistic_backtest_analysis.png', dpi=300, bbox_inches='tight')
    print("📈 Analysis chart saved as 'realistic_backtest_analysis.png'")
    
    # Print detailed statistics
    print("\n📊 DETAILED ANALYSIS")
    print("=" * 50)
    
    # Overall statistics
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    losing_trades = len(trades_df[trades_df['pnl'] < 0])
    win_rate = (winning_trades / total_trades) * 100
    
    print(f"🎯 Trading Performance:")
    print(f"   Total Trades: {total_trades}")
    print(f"   Winning Trades: {winning_trades}")
    print(f"   Losing Trades: {losing_trades}")
    print(f"   Win Rate: {win_rate:.1f}%")
    
    # P&L statistics
    total_pnl = trades_df['pnl'].sum()
    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean()
    avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean()
    best_trade = trades_df['pnl'].max()
    worst_trade = trades_df['pnl'].min()
    
    print(f"\n💰 P&L Analysis:")
    print(f"   Total P&L: ${total_pnl:.2f}")
    print(f"   Average Win: ${avg_win:.2f}")
    print(f"   Average Loss: ${avg_loss:.2f}")
    print(f"   Best Trade: ${best_trade:.2f}")
    print(f"   Worst Trade: ${worst_trade:.2f}")
    print(f"   Profit Factor: {abs(avg_win * winning_trades / (avg_loss * losing_trades)):.2f}")
    
    # Pair performance
    print(f"\n📈 Performance by Pair:")
    pair_stats = trades_df.groupby('pair').agg({
        'pnl': ['count', 'sum', 'mean']
    }).round(2)
    
    for pair in trades_df['pair'].unique():
        pair_trades = trades_df[trades_df['pair'] == pair]
        pair_pnl = pair_trades['pnl'].sum()
        pair_wins = len(pair_trades[pair_trades['pnl'] > 0])
        pair_total = len(pair_trades)
        pair_win_rate = (pair_wins / pair_total) * 100 if pair_total > 0 else 0
        print(f"   {pair}: {pair_total} trades, ${pair_pnl:.2f} P&L, {pair_win_rate:.1f}% win rate")
    
    # Risk analysis
    print(f"\n⚠️ Risk Analysis:")
    max_balance = equity_df['balance'].max()
    min_balance = equity_df['balance'].min()
    max_drawdown = ((max_balance - min_balance) / max_balance) * 100
    
    print(f"   Initial Balance: $1,500.00")
    print(f"   Final Balance: ${equity_df['balance'].iloc[-1]:.2f}")
    print(f"   Peak Balance: ${max_balance:.2f}")
    print(f"   Valley Balance: ${min_balance:.2f}")
    print(f"   Maximum Drawdown: {max_drawdown:.2f}%")
    
    # Duration analysis
    avg_duration = trades_df['duration_hours'].mean()
    min_duration = trades_df['duration_hours'].min()
    max_duration = trades_df['duration_hours'].max()
    
    print(f"\n⏱️ Duration Analysis:")
    print(f"   Average Duration: {avg_duration:.1f} hours")
    print(f"   Shortest Trade: {min_duration:.1f} hours")
    print(f"   Longest Trade: {max_duration:.1f} hours")
    
    plt.show()
    
except Exception as e:
    print(f"❌ Error loading backtest results: {e}")
    print("Make sure the backtest has been run and CSV files exist.")
