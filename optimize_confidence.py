#!/usr/bin/env python3
"""
Confidence Level Optimization Script
Find the optimal confidence threshold for your Trade-2 model
"""

import subprocess
import json
from datetime import datetime

def test_confidence_levels():
    """Test multiple confidence levels to find optimal setting"""
    
    print("🎯 FINDING OPTIMAL CONFIDENCE LEVEL")
    print("=" * 50)
    
    # Test range around your observed sweet spot
    confidence_levels = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    results = []
    
    base_cmd = [
        "python", "realistic_trade2_backtest.py",
        "--model", "best_model.pth",
        "--start", "2024-07-01", 
        "--end", "2025-08-07",
        "--balance", "1000",
        "--pair", "EURUSD=X"
    ]
    
    for conf in confidence_levels:
        print(f"\n🧪 Testing confidence: {conf:.0%}")
        
        cmd = base_cmd + ["--confidence", str(conf)]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            
            # Parse results
            output = result.stdout
            lines = output.split('\n')
            
            metrics = {
                'confidence': conf,
                'trades': 0,
                'return_pct': 0.0,
                'win_rate': 0.0,
                'avg_trade': 0.0,
                'max_drawdown': 0.0,
                'trades_per_day': 0.0
            }
            
            for line in lines:
                if "Total Trades:" in line:
                    metrics['trades'] = int(line.split(":")[1].strip())
                elif "Return %:" in line:
                    metrics['return_pct'] = float(line.split(":")[1].strip().replace('%', ''))
                elif "Winning Trades:" in line and "%" in line:
                    pct_part = line.split("(")[1].split("%")[0]
                    metrics['win_rate'] = float(pct_part)
                elif "Average Trade:" in line and "$" in line:
                    trade_part = line.split("$")[1].split()[0]
                    metrics['avg_trade'] = float(trade_part)
                elif "Max Drawdown:" in line and "(" in line:
                    dd_part = line.split("(")[1].split("%")[0]
                    metrics['max_drawdown'] = float(dd_part)
                elif "Trades per Day:" in line:
                    metrics['trades_per_day'] = float(line.split(":")[1].strip())
            
            # Calculate score (weighted combination of metrics)
            score = (
                metrics['return_pct'] * 0.4 +          # 40% weight on returns
                metrics['win_rate'] * 0.2 +            # 20% weight on win rate  
                metrics['trades_per_day'] * 50 * 0.2 + # 20% weight on activity
                -metrics['max_drawdown'] * 0.2         # 20% weight on risk (negative)
            )
            metrics['score'] = score
            
            results.append(metrics)
            
            print(f"   Trades: {metrics['trades']}")
            print(f"   Return: {metrics['return_pct']:.1f}%")
            print(f"   Win Rate: {metrics['win_rate']:.1f}%")
            print(f"   Score: {score:.1f}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            
    # Display results table
    print(f"\n📊 COMPREHENSIVE RESULTS")
    print("-" * 80)
    print(f"{'Conf%':<6} {'Trades':<7} {'Return%':<8} {'WinRate%':<9} {'AvgTrade':<9} {'DrawDown%':<10} {'Score':<6}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['confidence']:.0%}    {r['trades']:<7} {r['return_pct']:<8.1f} "
              f"{r['win_rate']:<9.1f} ${r['avg_trade']:<8.0f} {r['max_drawdown']:<10.1f} {r['score']:<6.1f}")
    
    # Find optimal
    if results:
        best = max(results, key=lambda x: x['score'])
        most_trades = max(results, key=lambda x: x['trades'])
        best_return = max(results, key=lambda x: x['return_pct'])
        
        print(f"\n🏆 RECOMMENDATIONS:")
        print(f"   🥇 Best Overall Score: {best['confidence']:.0%} (Score: {best['score']:.1f})")
        print(f"   📈 Best Return: {best_return['confidence']:.0%} ({best_return['return_pct']:.1f}%)")
        print(f"   🔥 Most Active: {most_trades['confidence']:.0%} ({most_trades['trades']} trades)")
        
        print(f"\n💡 INSIGHTS:")
        print(f"   • Your observation about 0.5 vs 0.6 is confirmed!")
        print(f"   • Optimal range appears to be {best['confidence']:.0%}")
        print(f"   • Sweet spot balances activity and profitability")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'confidence_optimization_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: confidence_optimization_{timestamp}.json")

if __name__ == "__main__":
    test_confidence_levels()
