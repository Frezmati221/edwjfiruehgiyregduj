"""
QUICK SIGNAL CHECKER
Get immediate trading signals from your enhanced ML models
"""

import sys
import os
sys.path.append('/home/timakha/main/t/eternity')

from enhanced_live_trader import EnhancedLiveTrader
import json
from datetime import datetime

def get_signals_now():
    """Get trading signals right now"""
    
    print("🔍 QUICK SIGNAL CHECK")
    print("=" * 40)
    print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Create trader instance
        trader = EnhancedLiveTrader(initial_balance=1500, demo_mode=True)
        
        print(f"✅ Enhanced models loaded: {len(trader.enhanced_models)}")
        
        # Check each pair
        signals_found = []
        
        for pair in trader.pairs:
            print(f"\n📊 Analyzing {pair}...")
            
            # Get live data
            data = trader.get_live_data(pair)
            if data is None:
                print(f"   ❌ No data available")
                continue
            
            # Make prediction
            prediction = trader.make_prediction(pair, data)
            if prediction is None:
                print(f"   ❌ No prediction available")
                continue
            
            # Check if signal meets criteria
            signal_prob = prediction['signal_prob']
            confidence = prediction['confidence'] 
            reward_estimate = prediction['reward_estimate']
            
            print(f"   🤖 ML Analysis:")
            print(f"      Signal Probability: {signal_prob:.1%}")
            print(f"      Confidence: {confidence:.1%}")
            print(f"      Reward Estimate: {reward_estimate:.3f}")
            print(f"      Price: {prediction['price']:.5f}")
            print(f"      RSI: {prediction['rsi']:.1f}")
            print(f"      MACD Bullish: {prediction['macd_bullish']}")
            print(f"      Trend Up: {prediction['trend_up']}")
            
            # FORCE SIGNAL GENERATION - bypassing strict conditions
            print(f"   🎯 GENERATING SIGNAL (conditions bypassed)")
            
            # Determine direction based on multiple factors
            # Positive factors for BUY: MACD bullish, trend up, higher confidence, better reward
            # Negative factors for SELL: opposite conditions
            
            buy_score = 0
            sell_score = 0
            
            # MACD factor
            if prediction['macd_bullish']:
                buy_score += 1
            else:
                sell_score += 1
            
            # Trend factor
            if prediction['trend_up']:
                buy_score += 1
            else:
                sell_score += 1
            
            # RSI factor (oversold = buy, overbought = sell)
            if prediction['rsi'] < 40:
                buy_score += 1
            elif prediction['rsi'] > 60:
                sell_score += 1
            
            # Signal probability factor (higher = more bullish)
            if signal_prob > 5.0:
                buy_score += 1
            else:
                sell_score += 1
            
            # Reward estimate factor
            if reward_estimate > -0.100:
                buy_score += 1
            else:
                sell_score += 1
            
            # Determine direction
            if buy_score > sell_score:
                direction = 'buy'
                reasoning = f"BUY signals: {buy_score}, SELL signals: {sell_score}"
            else:
                direction = 'sell'
                reasoning = f"SELL signals: {sell_score}, BUY signals: {buy_score}"
            
            signal = {
                'pair': pair,
                'direction': direction,
                'price': prediction['price'],
                'confidence': confidence,
                'signal_prob': signal_prob,
                'reward_estimate': reward_estimate,
                'reasoning': reasoning,
                'timestamp': datetime.now()
            }
            
            signals_found.append(signal)
            
            print(f"      Direction: {direction.upper()}")
            print(f"      Reasoning: {reasoning}")
            print(f"      Entry Price: {prediction['price']:.5f}")
            print(f"      ML Confidence: {confidence:.1%}")
            print(f"      Signal Strength: {signal_prob:.1%}")
        
        # Summary
        print(f"\n🎯 SIGNAL SUMMARY")
        print("=" * 40)
        
        if signals_found:
            print(f"🚀 Generated {len(signals_found)} FORCED trading signals:")
            
            for i, signal in enumerate(signals_found, 1):
                print(f"\n   Signal {i}: {signal['direction'].upper()} {signal['pair']}")
                print(f"   Entry: {signal['price']:.5f}")
                print(f"   ML Confidence: {signal['confidence']:.1%}")
                print(f"   Signal Strength: {signal['signal_prob']:.1%}")
                print(f"   Reward Estimate: {signal['reward_estimate']:+.3f}")
                print(f"   Reasoning: {signal['reasoning']}")
                
                # Calculate SL/TP with wider spreads for forced signals
                if 'JPY' in signal['pair']:
                    pip_value = 0.01
                else:
                    pip_value = 0.0001
                
                # Wider stops for forced signals (more conservative)
                sl_pips, tp_pips = 25, 35
                
                if signal['direction'] == 'buy':
                    sl = signal['price'] - (sl_pips * pip_value)
                    tp = signal['price'] + (tp_pips * pip_value)
                else:
                    sl = signal['price'] + (sl_pips * pip_value)
                    tp = signal['price'] - (tp_pips * pip_value)
                
                print(f"   Stop Loss: {sl:.5f}")
                print(f"   Take Profit: {tp:.5f}")
                print(f"   Risk/Reward: 1:{tp_pips/sl_pips:.1f}")
                
                # Risk warning for forced signals
                print(f"   ⚠️  WARNING: This is a FORCED signal (bypassed normal conditions)")
        else:
            print("❌ No trading signals could be generated (technical error)")
        
        print(f"\n💡 FORCED SIGNALS MODE:")
        print(f"   • These signals bypass normal safety conditions")
        print(f"   • Use smaller position sizes and wider stops")
        print(f"   • Consider this as ML 'suggestions' not strong signals")
        print(f"   • Monitor closely and exit early if price moves against you")
        
        print(f"\n💡 FORCED SIGNALS MODE:")
        print(f"   • These signals bypass normal safety conditions")
        print(f"   • Use smaller position sizes and wider stops")
        print(f"   • Consider this as ML 'suggestions' not strong signals")
        print(f"   • Monitor closely and exit early if price moves against you")
        
        return signals_found
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("   • Make sure you're in the right directory")
        print("   • Check if enhanced models exist")
        print("   • Verify internet connection for market data")
        return []

if __name__ == "__main__":
    signals = get_signals_now()
    print(f"\n🏁 Check complete. Found {len(signals)} signals.")
