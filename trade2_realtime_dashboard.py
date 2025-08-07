#!/usr/bin/env python3
"""
Trade-2 Real-Time Signal Dashboard
Enhanced real-time monitoring with signal alerts and history tracking
"""

import os
import json
import time
from datetime import datetime, timedelta
from quick_realtime_test import quick_realtime_test
import yfinance as yf
from compatible_trade2_predictor import CompatibleSupervisedForexPredictor

class Trade2Dashboard:
    def __init__(self, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.predictor = None
        self.signal_history = []
        self.pairs = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X', 'USDCHF=X']
        
        # Load model
        self.load_model()
        
        # Create results directory
        os.makedirs('results', exist_ok=True)
    
    def load_model(self):
        """Load the Trade-2 model"""
        try:
            self.predictor = CompatibleSupervisedForexPredictor()
            self.predictor.load_model("best_model.pth")
            print(f"✅ Trade-2 model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def get_live_signal(self, pair):
        """Get live signal for a currency pair"""
        try:
            # Get recent data
            ticker = yf.Ticker(pair)
            df = ticker.history(period="2mo", interval="1h")
            
            if df.empty or len(df) < 100:
                return {'pair': pair, 'status': 'no_data'}
            
            # Normalize columns
            df.columns = [col.lower() for col in df.columns]
            
            # Get current market info
            current_price = df['close'].iloc[-1]
            daily_change = ((current_price - df['open'].iloc[-24]) / df['open'].iloc[-24]) * 100
            
            # Generate prediction
            prediction = self.predictor.predict(df, min_confidence=self.confidence_threshold)
            
            signal = {
                'pair': pair,
                'timestamp': datetime.now().isoformat(),
                'current_price': current_price,
                'daily_change_pct': daily_change,
                'action': prediction['action'],
                'confidence': prediction['confidence'],
                'status': 'active' if prediction['action'] != 'hold' else 'hold'
            }
            
            # Add trade details if active
            if prediction['action'] != 'hold' and 'stop_loss' in prediction:
                signal.update({
                    'entry_price': prediction['entry_price'],
                    'stop_loss': prediction['stop_loss'], 
                    'take_profit': prediction['take_profit'],
                    'risk_reward': prediction['risk_reward_ratio'],
                    'risk_pips': prediction['risk_pips'],
                    'reward_pips': prediction['reward_pips']
                })
            
            return signal
            
        except Exception as e:
            return {'pair': pair, 'status': 'error', 'error': str(e)}
    
    def scan_markets(self):
        """Scan all markets for signals"""
        timestamp = datetime.now()
        print(f"\n🔍 MARKET SCAN - {timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print("=" * 70)
        
        signals = []
        active_count = 0
        
        for pair in self.pairs:
            signal = self.get_live_signal(pair)
            signals.append(signal)
            
            pair_name = pair.replace('=X', '')
            
            if signal['status'] == 'active':
                active_count += 1
                action_emoji = "🟢" if signal['action'] == 'long' else "🔴"
                print(f"{action_emoji} {pair_name}: {signal['action'].upper()} @ {signal['confidence']:.1%} confidence")
                print(f"   💰 Price: {signal['current_price']:.5f} ({signal['daily_change_pct']:+.2f}%)")
                print(f"   🎯 TP: {signal['reward_pips']:.1f} pips | 🛡️ SL: {signal['risk_pips']:.1f} pips")
                
            elif signal['status'] == 'hold':
                print(f"⚪ {pair_name}: HOLD @ {signal['confidence']:.1%} confidence")
                
            else:
                print(f"❌ {pair_name}: {signal['status']}")
        
        # Summary
        print(f"\n📊 SUMMARY: {active_count}/{len(self.pairs)} pairs with active signals")
        
        # Save to history
        self.signal_history.append({
            'timestamp': timestamp.isoformat(),
            'signals': signals,
            'active_count': active_count
        })
        
        return signals, active_count
    
    def save_current_signals(self, signals):
        """Save current signals to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/live_signals_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'confidence_threshold': self.confidence_threshold,
                'signals': signals
            }, f, indent=2)
        
        return filename
    
    def display_active_signals_detailed(self, signals):
        """Display detailed view of active signals"""
        active_signals = [s for s in signals if s.get('status') == 'active']
        
        if not active_signals:
            print(f"\n⚪ No active trading signals found")
            print(f"💡 Consider lowering confidence threshold from {self.confidence_threshold:.1%}")
            return
        
        print(f"\n🚨 ACTIVE TRADING OPPORTUNITIES ({len(active_signals)})")
        print("=" * 80)
        
        for i, signal in enumerate(active_signals, 1):
            pair_name = signal['pair'].replace('=X', '')
            action_color = "🟢 LONG" if signal['action'] == 'long' else "🔴 SHORT"
            
            print(f"\n{i}. {action_color} {pair_name}")
            print(f"   🎯 Confidence: {signal['confidence']:.1%}")
            print(f"   💰 Entry: {signal['entry_price']:.5f}")
            print(f"   🛡️ Stop Loss: {signal['stop_loss']:.5f} ({signal['risk_pips']:.1f} pips)")
            print(f"   🎯 Take Profit: {signal['take_profit']:.5f} ({signal['reward_pips']:.1f} pips)")
            print(f"   ⚖️ Risk/Reward: 1:{signal['risk_reward']:.2f}")
            print(f"   📈 Daily Move: {signal['daily_change_pct']:+.2f}%")
            
            # Calculate potential profit/loss percentages
            entry = signal['entry_price']
            sl = signal['stop_loss']
            tp = signal['take_profit']
            
            if signal['action'] == 'long':
                risk_pct = abs(entry - sl) / entry * 100
                reward_pct = abs(tp - entry) / entry * 100
            else:
                risk_pct = abs(sl - entry) / entry * 100  
                reward_pct = abs(entry - tp) / entry * 100
            
            print(f"   📊 Risk: {risk_pct:.2f}% | Reward: {reward_pct:.2f}%")
    
    def run_continuous_monitoring(self, interval_minutes=15):
        """Run continuous monitoring"""
        print(f"🤖 TRADE-2 CONTINUOUS MONITORING STARTED")
        print(f"⏰ Scan interval: {interval_minutes} minutes")
        print(f"🎯 Confidence threshold: {self.confidence_threshold:.1%}")
        print(f"💱 Monitoring {len(self.pairs)} currency pairs")
        print(f"Press Ctrl+C to stop\n")
        
        scan_count = 0
        
        try:
            while True:
                scan_count += 1
                print(f"\n{'='*20} SCAN #{scan_count} {'='*20}")
                
                signals, active_count = self.scan_markets()
                self.display_active_signals_detailed(signals)
                
                # Save signals
                filename = self.save_current_signals(signals)
                print(f"\n💾 Signals saved to: {filename}")
                
                # Alert if active signals found
                if active_count > 0:
                    print(f"\n🔔 ALERT: {active_count} active trading signals detected!")
                
                print(f"\n⏰ Next scan in {interval_minutes} minutes...")
                print(f"📊 Total scans completed: {scan_count}")
                
                time.sleep(interval_minutes * 60)
                
        except KeyboardInterrupt:
            print(f"\n\n🛑 Monitoring stopped")
            print(f"📈 Total scans completed: {scan_count}")
            print(f"💾 Signal history contains {len(self.signal_history)} scans")

def main():
    """Main function with command line options"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Trade-2 Real-Time Signal Dashboard')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold (default: 0.5)')
    parser.add_argument('--continuous', action='store_true', help='Run continuous monitoring')
    parser.add_argument('--interval', type=int, default=15, help='Scan interval in minutes (default: 15)')
    
    args = parser.parse_args()
    
    # Create dashboard
    dashboard = Trade2Dashboard(confidence_threshold=args.confidence)
    
    if args.continuous:
        dashboard.run_continuous_monitoring(args.interval)
    else:
        # Single scan
        signals, active_count = dashboard.scan_markets()
        dashboard.display_active_signals_detailed(signals)
        filename = dashboard.save_current_signals(signals)
        print(f"\n💾 Results saved to: {filename}")

if __name__ == "__main__":
    main()
