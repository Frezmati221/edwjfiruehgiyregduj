#!/usr/bin/env python3
"""
Quick Real-Time Test for Trade-2 Model
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
from compatible_trade2_predictor import CompatibleSupervisedForexPredictor

def quick_realtime_test():
    """Quick test of real-time predictions"""
    
    print("🤖 QUICK REAL-TIME TRADE-2 TEST")
    print("=" * 50)
    
    # Load model
    print("📥 Loading model...")
    try:
        predictor = CompatibleSupervisedForexPredictor()
        predictor.load_model("best_model.pth")
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return
    
    # Test pairs
    pairs = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X']
    confidence_threshold = 0.5
    
    print(f"\n🔍 Scanning {len(pairs)} pairs with {confidence_threshold:.1%} confidence...")
    print("-" * 50)
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"🕐 Current time: {current_time}")
    
    for pair in pairs:
        print(f"\n💱 {pair}")
        try:
            # Get recent data
            ticker = yf.Ticker(pair)
            df = ticker.history(period="1mo", interval="1h")
            
            if df.empty or len(df) < 100:
                print("   ❌ Insufficient data")
                continue
            
            # Normalize columns
            df.columns = [col.lower() for col in df.columns]
            
            # Get current price
            current_price = df['close'].iloc[-1]
            daily_change = ((current_price - df['open'].iloc[-24]) / df['open'].iloc[-24]) * 100
            
            print(f"   💰 Current Price: {current_price:.5f}")
            print(f"   📈 24h Change: {daily_change:+.2f}%")
            
            # Generate prediction
            prediction = predictor.predict(df, min_confidence=confidence_threshold)
            
            print(f"   🎯 Action: {prediction['action'].upper()}")
            print(f"   📊 Confidence: {prediction['confidence']:.1%}")
            
            if prediction['action'] != 'hold' and 'stop_loss' in prediction:
                print(f"   🛡️ Stop Loss: {prediction['stop_loss']:.5f} ({prediction['risk_pips']:.1f} pips)")
                print(f"   🎯 Take Profit: {prediction['take_profit']:.5f} ({prediction['reward_pips']:.1f} pips)")
                print(f"   ⚖️ Risk/Reward: 1:{prediction['risk_reward_ratio']:.2f}")
                
                if prediction['action'] == 'long':
                    signal_strength = "🟢 BUY SIGNAL"
                else:
                    signal_strength = "🔴 SELL SIGNAL"
                print(f"   {signal_strength}")
            else:
                print(f"   ⚪ HOLD (confidence below threshold or no SL/TP)")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n✅ Real-time test completed!")
    print(f"💡 To run continuous monitoring:")
    print(f"   python realtime_trade2_predictor.py --continuous --interval 15")

if __name__ == "__main__":
    quick_realtime_test()
