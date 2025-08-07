#!/usr/bin/env python3
"""
Quick test script to demonstrate the enhanced SL/TP system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trade_2 import SupervisedForexPredictor, load_forex_data
import pandas as pd
import numpy as np

def test_sl_tp_system():
    """Test the SL/TP calculation without training"""
    
    print("🧪 TESTING STOP LOSS & TAKE PROFIT SYSTEM")
    print("="*60)
    
    # Load some sample data
    print("Loading sample forex data...")
    data = load_forex_data(period="5d", interval="1h")  # Just 5 days for quick test
    
    if not data:
        print("❌ No data available")
        return
    
    # Create a predictor instance (we won't train it, just use for SL/TP calculation)
    predictor = SupervisedForexPredictor()
    
    print("\n🎯 SL/TP CALCULATION EXAMPLES:")
    print("-" * 60)
    
    for pair, df in data.items():
        if len(df) < 100:  # Skip if not enough data
            continue
            
        current_price = df['close'].iloc[-1]
        print(f"\n📊 {pair}:")
        print(f"  Current Price: {current_price:.5f}")
        
        # Test both long and short scenarios
        for action in ['long', 'short']:
            try:
                sl, tp = predictor.calculate_optimal_sl_tp(df, action, risk_reward_ratio=2.0)
                
                if sl is not None and tp is not None:
                    pip_value = predictor.get_pip_value(current_price)
                    
                    if action == 'long':
                        risk_pips = (current_price - sl) / pip_value
                        reward_pips = (tp - current_price) / pip_value
                        risk_percent = (current_price - sl) / current_price * 100
                        reward_percent = (tp - current_price) / current_price * 100
                    else:
                        risk_pips = (sl - current_price) / pip_value
                        reward_pips = (current_price - tp) / pip_value
                        risk_percent = (sl - current_price) / current_price * 100
                        reward_percent = (current_price - tp) / current_price * 100
                    
                    actual_rr = reward_pips / risk_pips if risk_pips > 0 else 0
                    
                    print(f"  {action.upper()}:")
                    print(f"    Stop Loss: {sl:.5f} ({risk_pips:.1f} pips, {risk_percent:.2f}%)")
                    print(f"    Take Profit: {tp:.5f} ({reward_pips:.1f} pips, {reward_percent:.2f}%)")
                    print(f"    Risk/Reward: 1:{actual_rr:.2f}")
                    
                    # Calculate potential outcomes
                    if action == 'long':
                        print(f"    If TP hit: +{reward_pips:.1f} pips profit")
                        print(f"    If SL hit: -{risk_pips:.1f} pips loss")
                    else:
                        print(f"    If TP hit: +{reward_pips:.1f} pips profit")
                        print(f"    If SL hit: -{risk_pips:.1f} pips loss")
                else:
                    print(f"  {action.upper()}: No valid SL/TP found (market conditions unfavorable)")
                    
            except Exception as e:
                print(f"  {action.upper()}: Error calculating SL/TP - {str(e)}")
        
        print("-" * 40)
        break  # Just test first pair for demonstration

def test_different_risk_rewards():
    """Test different risk-reward ratios"""
    
    print("\n🎯 TESTING DIFFERENT RISK-REWARD RATIOS")
    print("="*60)
    
    # Load data
    data = load_forex_data(period="5d", interval="1h")
    if not data:
        return
    
    # Get first pair
    pair, df = next(iter(data.items()))
    predictor = SupervisedForexPredictor()
    current_price = df['close'].iloc[-1]
    
    print(f"Testing on {pair} at price {current_price:.5f}")
    print("-" * 40)
    
    # Test different risk-reward ratios
    risk_rewards = [1.5, 2.0, 2.5, 3.0]
    
    for rr in risk_rewards:
        print(f"\nRisk-Reward Ratio 1:{rr}")
        
        for action in ['long', 'short']:
            try:
                sl, tp = predictor.calculate_optimal_sl_tp(df, action, risk_reward_ratio=rr)
                
                if sl is not None and tp is not None:
                    pip_value = predictor.get_pip_value(current_price)
                    
                    if action == 'long':
                        risk_pips = (current_price - sl) / pip_value
                        reward_pips = (tp - current_price) / pip_value
                    else:
                        risk_pips = (sl - current_price) / pip_value
                        reward_pips = (current_price - tp) / pip_value
                    
                    actual_rr = reward_pips / risk_pips if risk_pips > 0 else 0
                    
                    print(f"  {action.upper()}: SL={sl:.5f}, TP={tp:.5f}, "
                          f"Risk={risk_pips:.1f}p, Reward={reward_pips:.1f}p, "
                          f"Actual RR=1:{actual_rr:.2f}")
                else:
                    print(f"  {action.upper()}: No valid SL/TP for RR 1:{rr}")
                    
            except Exception as e:
                print(f"  {action.upper()}: Error - {str(e)}")

def show_market_analysis():
    """Show current market analysis with SL/TP"""
    
    print("\n📈 CURRENT MARKET ANALYSIS WITH SL/TP")
    print("="*60)
    
    # Load fresh data
    data = load_forex_data(period="2d", interval="1h")
    if not data:
        return
    
    predictor = SupervisedForexPredictor()
    
    for pair, df in data.items():
        if len(df) < 50:
            continue
            
        current_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2]
        change = (current_price - prev_price) / prev_price * 100
        
        # Get ATR for volatility assessment
        try:
            import talib
            high_prices = df['high'].values[-20:].astype(np.float64)
            low_prices = df['low'].values[-20:].astype(np.float64)
            close_prices = df['close'].values[-20:].astype(np.float64)
            atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)[-1]
            
            print(f"\n{pair}:")
            print(f"  Price: {current_price:.5f} ({change:+.2f}%)")
            print(f"  ATR: {atr:.5f} (volatility indicator)")
            
            # Calculate SL/TP for both directions
            for action in ['long', 'short']:
                sl, tp = predictor.calculate_optimal_sl_tp(df, action, risk_reward_ratio=2.0)
                
                if sl and tp:
                    pip_value = predictor.get_pip_value(current_price)
                    
                    if action == 'long':
                        risk_pips = (current_price - sl) / pip_value
                        reward_pips = (tp - current_price) / pip_value
                    else:
                        risk_pips = (sl - current_price) / pip_value
                        reward_pips = (current_price - tp) / pip_value
                    
                    print(f"  {action.upper()}: SL={sl:.5f} ({risk_pips:.1f}p), TP={tp:.5f} ({reward_pips:.1f}p)")
                
        except Exception as e:
            print(f"{pair}: Error in analysis - {str(e)}")

if __name__ == "__main__":
    try:
        # Run all tests
        test_sl_tp_system()
        test_different_risk_rewards()
        show_market_analysis()
        
        print("\n✅ SL/TP SYSTEM TEST COMPLETE!")
        print("\nKey Features:")
        print("• Dynamic SL/TP based on ATR, support/resistance, and volatility")
        print("• Configurable risk-reward ratios (1.5:1, 2:1, 2.5:1, 3:1)")
        print("• Market condition analysis for optimal entry/exit points")
        print("• Automatic risk management with maximum loss limits")
        print("• Integration with confidence-based trading decisions")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
