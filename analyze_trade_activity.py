#!/usr/bin/env python3
"""
More Active Trade-2 Backtester
Optimized for more frequent trading while maintaining good risk management
"""

import subprocess
import sys
import os
from datetime import datetime, timedelta

def run_active_backtest():
    """Run backtests with different activity levels"""
    
    print("🎯 TRADE-2 ACTIVE BACKTESTING ANALYSIS")
    print("=" * 80)
    
    model_path = "/home/timakha/main/t/eternity/best_model.pth"
    
    # Test different confidence levels
    confidence_levels = [0.55, 0.60, 0.65, 0.70]
    
    print(f"Testing different confidence thresholds...")
    print(f"Period: 2025-07-01 to 2025-08-07 (37 days)")
    print(f"Balance: $1,000")
    print("-" * 60)
    
    results = []
    
    for conf in confidence_levels:
        print(f"\n🧪 Testing confidence threshold: {conf:.0%}")
        
        cmd = [
            "python", "realistic_trade2_backtest.py",
            "--model", model_path,
            "--start", "2025-07-01",
            "--end", "2025-08-07", 
            "--balance", "1000",
            "--pair", "EURUSD=X",
            "--confidence", str(conf)
        ]
        
        try:
            result = subprocess.run(cmd, cwd="/home/timakha/main/t/eternity",
                                  capture_output=True, text=True, timeout=120)
            
            # Extract key metrics
            output = result.stdout
            
            # Parse trades and return
            trades = 0
            final_return = 0.0
            confident_signals = 0
            total_signals = 0
            
            lines = output.split('\n')
            for line in lines:
                if "Total Trades:" in line:
                    trades = int(line.split(":")[1].strip())
                elif "Return %:" in line:
                    final_return = float(line.split(":")[1].strip().replace('%', ''))
                elif "Confident signals:" in line:
                    parts = line.split(":")
                    if len(parts) > 1:
                        confident_part = parts[1].strip()
                        if "(" in confident_part:
                            confident_signals = int(confident_part.split("(")[0].strip())
                elif "Total signals:" in line:
                    total_signals = int(line.split(":")[1].strip())
            
            results.append({
                'confidence': conf,
                'trades': trades,
                'return_pct': final_return,
                'confident_signals': confident_signals,
                'total_signals': total_signals,
                'activity_rate': confident_signals / max(1, total_signals) * 100
            })
            
            print(f"   Trades: {trades}")
            print(f"   Return: {final_return:.2f}%")
            print(f"   Activity: {confident_signals}/{total_signals} ({confident_signals/max(1, total_signals)*100:.1f}%)")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                'confidence': conf,
                'trades': 0,
                'return_pct': 0.0,
                'confident_signals': 0,
                'total_signals': 0,
                'activity_rate': 0
            })
    
    # Display summary
    print(f"\n📊 CONFIDENCE THRESHOLD ANALYSIS")
    print("-" * 60)
    print(f"{'Confidence':<12} {'Trades':<8} {'Return%':<10} {'Activity%':<12} {'Signals':<10}")
    print("-" * 60)
    
    for r in results:
        print(f"{r['confidence']:.0%}          {r['trades']:<8} {r['return_pct']:<10.2f} "
              f"{r['activity_rate']:<12.1f} {r['confident_signals']}/{r['total_signals']}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    best_trades = max(results, key=lambda x: x['trades'])
    best_return = max(results, key=lambda x: x['return_pct'])
    best_activity = max(results, key=lambda x: x['activity_rate'])
    
    print(f"   🔥 Most Active: {best_trades['confidence']:.0%} confidence ({best_trades['trades']} trades)")
    print(f"   💰 Best Return: {best_return['confidence']:.0%} confidence ({best_return['return_pct']:.2f}%)")
    print(f"   📈 Most Signals: {best_activity['confidence']:.0%} confidence ({best_activity['activity_rate']:.1f}% activity)")
    
    if best_trades['trades'] < 5:
        print(f"\n⚠️  All configurations show low activity. Consider:")
        print(f"   • Reducing minimum risk-reward ratio from 2:1 to 1.5:1")
        print(f"   • Reducing time between trades from 4h to 2h")
        print(f"   • Using confidence threshold of 50-55%")
        print(f"   • Checking if SL/TP calculation is too restrictive")

def suggest_optimizations():
    """Suggest specific optimizations for more trading activity"""
    
    print(f"\n🔧 OPTIMIZATION SUGGESTIONS FOR MORE ACTIVITY:")
    print("=" * 60)
    
    print(f"""
1. 🎯 CONFIDENCE THRESHOLD:
   Current: 65-70% (very conservative)
   Suggested: 55-60% (more balanced)
   
2. 📊 RISK-REWARD RATIO:
   Current: 2:1 minimum (restrictive)
   Suggested: 1.5:1 minimum (more flexible)
   
3. ⏰ TIME BETWEEN TRADES:
   Current: 4 hours (conservative)
   Suggested: 2 hours (more active)
   
4. 📈 DAILY TRADE LIMIT:
   Current: 5 trades per day
   Suggested: 8 trades per day
   
5. 🛡️ STOP LOSS CALCULATION:
   May be too tight, rejecting good trades
   Consider more flexible ATR-based calculations
   
6. 💡 MODEL BEHAVIOR:
   The model was trained on specific patterns
   Lower confidence = more pattern variety
   Higher confidence = only strongest patterns
   
🚀 RECOMMENDED TEST COMMAND:
python realistic_trade2_backtest.py --model best_model.pth \\
    --start 2025-07-01 --end 2025-08-07 \\
    --confidence 0.55 --balance 1000 --pair EURUSD=X
""")

if __name__ == "__main__":
    choice = input("Choose analysis:\n1. Test different confidence levels\n2. Show optimization suggestions\n3. Both\nEnter (1-3): ")
    
    if choice == "1":
        run_active_backtest()
    elif choice == "2":
        suggest_optimizations()
    elif choice == "3":
        run_active_backtest()
        suggest_optimizations()
    else:
        print("Invalid choice. Showing suggestions:")
        suggest_optimizations()
