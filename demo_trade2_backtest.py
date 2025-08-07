#!/usr/bin/env python3
"""
Demo script for Trade-2 Realistic Backtesting
"""

import subprocess
import sys
import os
from datetime import datetime, timedelta

def run_realistic_backtest_demo():
    """Run demonstration of realistic backtesting"""
    
    print("=" * 80)
    print("🎯 TRADE-2 REALISTIC BACKTESTING DEMONSTRATION")
    print("=" * 80)
    
    # Check if model files exist
    model_path = "/home/timakha/main/t/eternity/best_model.pth"
    scaler_path = "/home/timakha/main/t/eternity/best_model_scaler.pkl"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("   Please train a model first using trade-2.py")
        return False
        
    if not os.path.exists(scaler_path):
        print(f"❌ Scaler file not found: {scaler_path}")
        print("   Please train a model first using trade-2.py")
        return False
    
    print(f"✅ Model file found: {model_path}")
    print(f"✅ Scaler file found: {scaler_path}")
    
    # Set backtest parameters
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
    
    print(f"\n📅 Backtest period: {start_date} to {end_date}")
    print(f"💰 Initial balance: $1,000")
    print(f"🎯 Confidence threshold: 65%")
    print(f"💱 Trading pair: EURUSD=X")
    
    # Prepare command
    cmd = [
        "python", "realistic_trade2_backtest.py",
        "--model", model_path,
        "--start", start_date,
        "--end", end_date,
        "--balance", "1000",
        "--pair", "EURUSD=X",
        "--confidence", "0.65"
    ]
    
    print(f"\n🚀 Running backtest command:")
    print(f"   {' '.join(cmd)}")
    print("\n" + "=" * 60)
    
    try:
        # Run the backtest
        result = subprocess.run(cmd, cwd="/home/timakha/main/t/eternity", 
                              capture_output=True, text=True, timeout=300)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("\n✅ Backtest completed successfully!")
            print("\n📊 Check the generated files:")
            print("   - trade2_backtest_summary_*.json")
            print("   - trade2_backtest_trades_*.csv")
            print("   - trade2_backtest_equity_*.csv")
        else:
            print(f"\n❌ Backtest failed with return code: {result.returncode}")
            
    except subprocess.TimeoutExpired:
        print("\n⏰ Backtest timed out after 5 minutes")
    except Exception as e:
        print(f"\n❌ Error running backtest: {e}")
    
    return True

def show_usage_examples():
    """Show various usage examples"""
    print("\n" + "=" * 60)
    print("📖 USAGE EXAMPLES")
    print("=" * 60)
    
    print("\n1. Basic backtest (last 2 months):")
    print("   python realistic_trade2_backtest.py --model best_model.pth")
    
    print("\n2. Custom date range:")
    print("   python realistic_trade2_backtest.py --model best_model.pth \\")
    print("       --start 2024-10-01 --end 2024-12-01")
    
    print("\n3. Different currency pair:")
    print("   python realistic_trade2_backtest.py --model best_model.pth \\")
    print("       --pair GBPUSD=X")
    
    print("\n4. Higher confidence threshold (more conservative):")
    print("   python realistic_trade2_backtest.py --model best_model.pth \\")
    print("       --confidence 0.75")
    
    print("\n5. Larger account balance:")
    print("   python realistic_trade2_backtest.py --model best_model.pth \\")
    print("       --balance 10000")
    
    print("\n📊 Key Features:")
    print("   ✓ Realistic spread and slippage simulation")
    print("   ✓ Proper risk management (2% per trade)")
    print("   ✓ Dynamic Stop Loss and Take Profit levels")
    print("   ✓ Market hours validation")
    print("   ✓ Position sizing based on account balance")
    print("   ✓ Comprehensive trade statistics")
    print("   ✓ Equity curve tracking")
    print("   ✓ Results saved to CSV and JSON files")

if __name__ == "__main__":
    print("Trade-2 Realistic Backtesting System")
    print("=" * 40)
    
    choice = input("\nChoose an option:\n1. Run demo backtest\n2. Show usage examples\n3. Both\nEnter choice (1-3): ")
    
    if choice == "1":
        run_realistic_backtest_demo()
    elif choice == "2":
        show_usage_examples()
    elif choice == "3":
        show_usage_examples()
        print("\n")
        run_realistic_backtest_demo()
    else:
        print("Invalid choice. Showing usage examples:")
        show_usage_examples()
