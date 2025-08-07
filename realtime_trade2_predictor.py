#!/usr/bin/env python3
"""
Real-Time Trade-2 Predictions
Live forex signal generator using the trained supervised learning model
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import os
from compatible_trade2_predictor import CompatibleSupervisedForexPredictor
import warnings
warnings.filterwarnings('ignore')

class RealTimeTrade2Predictor:
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        """Initialize real-time predictor"""
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.predictor = None
        self.currency_pairs = [
            'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 
            'AUDUSD=X', 'USDCAD=X', 'USDCHF=X'
        ]
        self.last_signals = {}
        self.signal_history = []
        
        print(f"🤖 Real-Time Trade-2 Predictor Initialized")
        print(f"🎯 Confidence threshold: {confidence_threshold:.1%}")
        print(f"💱 Monitoring pairs: {len(self.currency_pairs)}")
        
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            self.predictor = CompatibleSupervisedForexPredictor()
            self.predictor.load_model(self.model_path)
            print(f"✅ Model loaded successfully")
            print(f"🔧 Model base confidence: {self.predictor.min_confidence:.1%}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def get_live_data(self, pair: str, period: str = "2mo") -> pd.DataFrame:
        """Get recent market data for prediction"""
        try:
            ticker = yf.Ticker(pair)
            # Get recent hourly data
            df = ticker.history(period=period, interval="1h")
            
            if df.empty:
                print(f"⚠️ No data received for {pair}")
                return None
            
            # Normalize column names
            df.columns = [col.lower() for col in df.columns]
            
            return df
            
        except Exception as e:
            print(f"❌ Error getting data for {pair}: {e}")
            return None
    
    def get_current_market_info(self, pair: str) -> dict:
        """Get current market information"""
        try:
            ticker = yf.Ticker(pair)
            info = ticker.info
            
            # Get recent price data
            recent = ticker.history(period="1d", interval="1m")
            if not recent.empty:
                current_price = recent['Close'].iloc[-1]
                daily_change = ((current_price - recent['Open'].iloc[0]) / recent['Open'].iloc[0]) * 100
                
                return {
                    'current_price': current_price,
                    'daily_change_pct': daily_change,
                    'volume': recent['Volume'].iloc[-1] if 'Volume' in recent.columns else 0,
                    'timestamp': recent.index[-1]
                }
        except:
            pass
        
        return {'current_price': 0, 'daily_change_pct': 0, 'volume': 0, 'timestamp': datetime.now()}
    
    def generate_signal(self, pair: str) -> dict:
        """Generate trading signal for a currency pair"""
        try:
            # Get market data
            df = self.get_live_data(pair)
            if df is None or len(df) < 100:
                return {'pair': pair, 'status': 'insufficient_data'}
            
            # Get current market info
            market_info = self.get_current_market_info(pair)
            
            # Generate prediction
            prediction = self.predictor.predict(df, min_confidence=self.confidence_threshold)
            
            # Enhance signal with market context
            signal = {
                'pair': pair,
                'timestamp': datetime.now().isoformat(),
                'action': prediction['action'],
                'confidence': prediction['confidence'],
                'current_price': market_info['current_price'],
                'daily_change_pct': market_info['daily_change_pct'],
                'status': 'active' if prediction['action'] != 'hold' else 'hold'
            }
            
            # Add SL/TP if available
            if 'stop_loss' in prediction and prediction['stop_loss'] is not None:
                signal.update({
                    'entry_price': prediction['entry_price'],
                    'stop_loss': prediction['stop_loss'],
                    'take_profit': prediction['take_profit'],
                    'risk_reward_ratio': prediction['risk_reward_ratio'],
                    'risk_pips': prediction['risk_pips'],
                    'reward_pips': prediction['reward_pips'],
                    'sl_distance_pct': prediction['sl_distance_percent'],
                    'tp_distance_pct': prediction['tp_distance_percent']
                })
            
            return signal
            
        except Exception as e:
            print(f"⚠️ Error generating signal for {pair}: {e}")
            return {'pair': pair, 'status': 'error', 'error': str(e)}
    
    def scan_all_pairs(self) -> list:
        """Scan all currency pairs for signals"""
        print(f"\n🔍 Scanning {len(self.currency_pairs)} currency pairs...")
        signals = []
        
        for pair in self.currency_pairs:
            print(f"   📊 Analyzing {pair}...", end=' ')
            signal = self.generate_signal(pair)
            signals.append(signal)
            
            if signal['status'] == 'active':
                print(f"🟢 {signal['action'].upper()} @ {signal['confidence']:.1%}")
            elif signal['status'] == 'hold':
                print(f"⚪ HOLD @ {signal['confidence']:.1%}")
            else:
                print(f"❌ {signal['status']}")
        
        return signals
    
    def display_active_signals(self, signals: list):
        """Display active trading signals"""
        active_signals = [s for s in signals if s.get('status') == 'active']
        
        if not active_signals:
            print(f"\n⚪ No active signals above {self.confidence_threshold:.1%} confidence")
            return
        
        print(f"\n🚨 ACTIVE TRADING SIGNALS ({len(active_signals)})")
        print("=" * 80)
        
        for signal in active_signals:
            print(f"\n💱 {signal['pair']}")
            print(f"   🎯 Action: {signal['action'].upper()}")
            print(f"   📊 Confidence: {signal['confidence']:.1%}")
            print(f"   💰 Current Price: {signal['current_price']:.5f}")
            print(f"   📈 Daily Change: {signal['daily_change_pct']:+.2f}%")
            
            if 'stop_loss' in signal:
                print(f"   🛡️ Stop Loss: {signal['stop_loss']:.5f} ({signal['risk_pips']:.1f} pips)")
                print(f"   🎯 Take Profit: {signal['take_profit']:.5f} ({signal['reward_pips']:.1f} pips)")
                print(f"   ⚖️ Risk/Reward: 1:{signal['risk_reward_ratio']:.2f}")
            
            print(f"   🕐 Generated: {signal['timestamp']}")
    
    def display_summary_table(self, signals: list):
        """Display summary table of all pairs"""
        print(f"\n📋 MARKET OVERVIEW")
        print("-" * 70)
        print(f"{'Pair':<10} {'Price':<10} {'Change%':<8} {'Action':<6} {'Conf%':<6} {'Status':<12}")
        print("-" * 70)
        
        for signal in signals:
            pair = signal['pair'].replace('=X', '')
            price = f"{signal.get('current_price', 0):.5f}" if signal.get('current_price') else "N/A"
            change = f"{signal.get('daily_change_pct', 0):+.2f}" if signal.get('daily_change_pct') else "N/A"
            action = signal.get('action', 'N/A').upper()[:5]
            conf = f"{signal.get('confidence', 0):.0%}" if signal.get('confidence') else "N/A"
            status = signal.get('status', 'unknown')[:11]
            
            print(f"{pair:<10} {price:<10} {change:<8} {action:<6} {conf:<6} {status:<12}")
    
    def save_signals(self, signals: list):
        """Save signals to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        filename = f"results/realtime_signals_{timestamp}.json"
        
        # Add to signal history
        self.signal_history.extend(signals)
        
        # Save current signals
        with open(filename, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'confidence_threshold': self.confidence_threshold,
                'signals': signals,
                'summary': {
                    'total_pairs': len(signals),
                    'active_signals': len([s for s in signals if s.get('status') == 'active']),
                    'hold_signals': len([s for s in signals if s.get('status') == 'hold']),
                    'errors': len([s for s in signals if s.get('status') == 'error'])
                }
            }, f, indent=2)
        
        print(f"\n💾 Signals saved to: {filename}")
    
    def run_single_scan(self):
        """Run a single scan of all pairs"""
        print(f"🤖 REAL-TIME FOREX SIGNALS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        signals = self.scan_all_pairs()
        self.display_active_signals(signals)
        self.display_summary_table(signals)
        self.save_signals(signals)
        
        return signals
    
    def run_continuous(self, interval_minutes: int = 15):
        """Run continuous monitoring"""
        print(f"🔄 Starting continuous monitoring (every {interval_minutes} minutes)")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                self.run_single_scan()
                
                print(f"\n⏰ Next scan in {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)
                
        except KeyboardInterrupt:
            print(f"\n\n🛑 Monitoring stopped by user")
            print(f"📊 Total scans performed: {len(self.signal_history) // len(self.currency_pairs)}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-Time Trade-2 Forex Predictor')
    parser.add_argument('--model', type=str, default='best_model.pth', help='Path to model file')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold (0.0-1.0)')
    parser.add_argument('--continuous', action='store_true', help='Run continuous monitoring')
    parser.add_argument('--interval', type=int, default=15, help='Scan interval in minutes (for continuous mode)')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = RealTimeTrade2Predictor(
        model_path=args.model,
        confidence_threshold=args.confidence
    )
    
    if args.continuous:
        predictor.run_continuous(args.interval)
    else:
        predictor.run_single_scan()

if __name__ == "__main__":
    main()
