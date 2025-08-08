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
    
    def calculate_optimal_entry(self, df: pd.DataFrame, action: str, current_price: float) -> dict:
        """Calculate optimal entry price based on market conditions"""
        import talib
        
        # Get recent price data
        close_prices = df['close'].values[-50:].astype(np.float64)
        high_prices = df['high'].values[-50:].astype(np.float64)
        low_prices = df['low'].values[-50:].astype(np.float64)
        
        # Calculate support/resistance levels
        resistance_20 = df['high'].rolling(20).max().iloc[-1]
        support_20 = df['low'].rolling(20).min().iloc[-1]
        
        # Calculate ATR for volatility assessment
        atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)[-1]
        
        # Calculate entry suggestions
        entry_suggestions = {
            'current_price': current_price,
            'market_entry': current_price,  # Immediate market entry
            'optimal_entry': current_price,  # Will be calculated
            'entry_type': 'MARKET',
            'support_level': support_20,
            'resistance_level': resistance_20,
            'atr': atr
        }
        
        if action == 'long':
            # For LONG positions, suggest entry on pullbacks
            pullback_entry = current_price - (atr * 0.3)  # 30% ATR pullback
            resistance_entry = min(current_price, resistance_20 - (atr * 0.1))
            
            # Choose the better entry (closer to support but not too far from current)
            if pullback_entry > support_20 and (current_price - pullback_entry) / current_price < 0.005:  # Within 0.5%
                entry_suggestions['optimal_entry'] = pullback_entry
                entry_suggestions['entry_type'] = 'LIMIT'
            else:
                entry_suggestions['optimal_entry'] = current_price
                entry_suggestions['entry_type'] = 'MARKET'
                
        elif action == 'short':
            # For SHORT positions, suggest entry on bounces
            bounce_entry = current_price + (atr * 0.3)  # 30% ATR bounce
            support_entry = max(current_price, support_20 + (atr * 0.1))
            
            # Choose the better entry (closer to resistance but not too far from current)
            if bounce_entry < resistance_20 and (bounce_entry - current_price) / current_price < 0.005:  # Within 0.5%
                entry_suggestions['optimal_entry'] = bounce_entry
                entry_suggestions['entry_type'] = 'LIMIT'
            else:
                entry_suggestions['optimal_entry'] = current_price
                entry_suggestions['entry_type'] = 'MARKET'
        
        return entry_suggestions

    def generate_signal(self, pair: str) -> dict:
        """Generate enhanced trading signal with optimal entry suggestions"""
        try:
            # Get market data
            df = self.get_live_data(pair)
            if df is None or len(df) < 100:
                return {'pair': pair, 'status': 'insufficient_data'}
            
            # Get current market info
            market_info = self.get_current_market_info(pair)
            current_price = market_info['current_price']
            
            # Generate prediction
            prediction = self.predictor.predict(df, min_confidence=self.confidence_threshold)
            
            # Calculate optimal entry if we have a trading signal
            entry_info = None
            if prediction['action'] != 'hold':
                entry_info = self.calculate_optimal_entry(df, prediction['action'], current_price)
            
            # Enhance signal with market context
            signal = {
                'pair': pair,
                'timestamp': datetime.now().isoformat(),
                'action': prediction['action'],
                'confidence': prediction['confidence'],
                'current_price': current_price,
                'daily_change_pct': market_info['daily_change_pct'],
                'status': 'active' if prediction['action'] != 'hold' else 'hold'
            }
            
            # Add enhanced entry information
            if entry_info and prediction['action'] != 'hold':
                signal.update({
                    'entry_type': entry_info['entry_type'],
                    'market_entry': entry_info['market_entry'],
                    'optimal_entry': entry_info['optimal_entry'],
                    'support_level': entry_info['support_level'],
                    'resistance_level': entry_info['resistance_level'],
                    'entry_distance_pips': abs(entry_info['optimal_entry'] - current_price) / self.get_pip_value(pair, current_price),
                    'atr_value': entry_info['atr']
                })
            
            # Add SL/TP if available (recalculate based on optimal entry)
            if 'stop_loss' in prediction and prediction['stop_loss'] is not None:
                # Use optimal entry for SL/TP calculations if available
                entry_price = entry_info['optimal_entry'] if entry_info else prediction['entry_price']
                
                signal.update({
                    'entry_price': entry_price,
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
    
    def get_pip_value(self, pair: str, price: float) -> float:
        """Calculate pip value for the pair"""
        if 'JPY' in pair:
            return 0.01  # For JPY pairs
        else:
            return 0.0001  # For other major pairs

    def display_active_signals(self, signals: list):
        """Display enhanced active trading signals with entry suggestions"""
        active_signals = [s for s in signals if s.get('status') == 'active']
        
        if not active_signals:
            print(f"\n⚪ No active signals above {self.confidence_threshold:.1%} confidence")
            return
        
        print(f"\n🚨 ACTIVE TRADING SIGNALS ({len(active_signals)})")
        print("=" * 90)
        
        for signal in active_signals:
            print(f"\n💱 {signal['pair']} - {signal['action'].upper()}")
            print(f"   🎯 Confidence: {signal['confidence']:.1%}")
            print(f"   💰 Current Price: {signal['current_price']:.5f}")
            print(f"   📈 Daily Change: {signal['daily_change_pct']:+.2f}%")
            
            # Entry suggestions
            if 'entry_type' in signal:
                print(f"\n   📍 ENTRY SUGGESTIONS:")
                if signal['entry_type'] == 'MARKET':
                    print(f"   🟢 MARKET ORDER: {signal['market_entry']:.5f} (Enter immediately)")
                else:
                    print(f"   🟡 LIMIT ORDER: {signal['optimal_entry']:.5f} (Wait for better price)")
                    print(f"   📏 Distance: {signal['entry_distance_pips']:.1f} pips from current")
                    
                print(f"   🔺 Resistance: {signal['resistance_level']:.5f}")
                print(f"   🔻 Support: {signal['support_level']:.5f}")
            
            # Risk management
            if 'stop_loss' in signal:
                print(f"\n   🛡️ RISK MANAGEMENT:")
                print(f"   🛑 Stop Loss: {signal['stop_loss']:.5f} ({signal['risk_pips']:.1f} pips)")
                print(f"   🎯 Take Profit: {signal['take_profit']:.5f} ({signal['reward_pips']:.1f} pips)")
                print(f"   ⚖️ Risk/Reward: 1:{signal['risk_reward_ratio']:.2f}")
            
            print(f"   🕐 Generated: {signal['timestamp']}")
            print("-" * 70)
    
    def display_summary_table(self, signals: list):
        """Display enhanced summary table with entry information"""
        print(f"\n📋 MARKET OVERVIEW")
        print("-" * 85)
        print(f"{'Pair':<10} {'Price':<10} {'Change%':<8} {'Action':<6} {'Conf%':<6} {'Entry':<12} {'Status':<12}")
        print("-" * 85)
        
        for signal in signals:
            pair = signal['pair'].replace('=X', '')
            price = f"{signal.get('current_price', 0):.5f}" if signal.get('current_price') else "N/A"
            change = f"{signal.get('daily_change_pct', 0):+.2f}" if signal.get('daily_change_pct') else "N/A"
            action = signal.get('action', 'N/A').upper()[:5]
            conf = f"{signal.get('confidence', 0):.0%}" if signal.get('confidence') else "N/A"
            
            # Entry type information
            entry_info = "N/A"
            if 'entry_type' in signal:
                if signal['entry_type'] == 'MARKET':
                    entry_info = "MARKET NOW"
                else:
                    entry_info = f"LIMIT @{signal['optimal_entry']:.5f}"[:11]
            
            status = signal.get('status', 'unknown')[:11]
            
            print(f"{pair:<10} {price:<10} {change:<8} {action:<6} {conf:<6} {entry_info:<12} {status:<12}")
        
        print("-" * 85)
    
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
    
    def analyze_single_pair(self, pair: str) -> dict:
        """Detailed analysis of a single currency pair with entry timing"""
        print(f"\n🔍 DETAILED ANALYSIS: {pair}")
        print("=" * 60)
        
        signal = self.generate_signal(pair)
        
        if signal['status'] != 'active':
            print(f"⚪ No active signal for {pair}")
            print(f"   Status: {signal['status']}")
            if 'confidence' in signal:
                print(f"   Confidence: {signal['confidence']:.1%} (below {self.confidence_threshold:.1%} threshold)")
            return signal
        
        # Display detailed information
        print(f"💱 Pair: {pair}")
        print(f"🎯 Signal: {signal['action'].upper()}")
        print(f"📊 Confidence: {signal['confidence']:.1%}")
        print(f"💰 Current Price: {signal['current_price']:.5f}")
        print(f"📈 Daily Change: {signal['daily_change_pct']:+.2f}%")
        
        # Entry strategy
        print(f"\n📍 ENTRY STRATEGY:")
        if signal['entry_type'] == 'MARKET':
            print(f"🟢 IMMEDIATE ENTRY RECOMMENDED")
            print(f"   Market Order: {signal['market_entry']:.5f}")
            print(f"   Reason: Optimal conditions for immediate entry")
        else:
            print(f"🟡 WAIT FOR BETTER PRICE")
            print(f"   Limit Order: {signal['optimal_entry']:.5f}")
            print(f"   Distance: {signal['entry_distance_pips']:.1f} pips from current price")
            print(f"   Potential Improvement: {abs(signal['optimal_entry'] - signal['current_price']):.5f}")
        
        # Market context
        print(f"\n📊 MARKET CONTEXT:")
        print(f"   🔺 Resistance: {signal['resistance_level']:.5f}")
        print(f"   🔻 Support: {signal['support_level']:.5f}")
        print(f"   📏 ATR (14): {signal['atr_value']:.5f}")
        
        # Risk management
        if 'stop_loss' in signal:
            print(f"\n🛡️ RISK MANAGEMENT:")
            print(f"   🛑 Stop Loss: {signal['stop_loss']:.5f}")
            print(f"   🎯 Take Profit: {signal['take_profit']:.5f}")
            print(f"   📏 Risk: {signal['risk_pips']:.1f} pips")
            print(f"   📈 Reward: {signal['reward_pips']:.1f} pips")
            print(f"   ⚖️ Risk/Reward Ratio: 1:{signal['risk_reward_ratio']:.2f}")
        
        # Entry recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if signal['entry_type'] == 'MARKET':
            print(f"   • Enter immediately with market order")
            print(f"   • Set stop loss at {signal['stop_loss']:.5f}")
            print(f"   • Set take profit at {signal['take_profit']:.5f}")
        else:
            print(f"   • Place limit order at {signal['optimal_entry']:.5f}")
            print(f"   • If not filled within 2-4 hours, consider market entry")
            print(f"   • Set stop loss at {signal['stop_loss']:.5f}")
            print(f"   • Set take profit at {signal['take_profit']:.5f}")
        
        print(f"   • Monitor position closely")
        print(f"   • Risk only 1-2% of account balance")
        
        return signal

    def run_single_scan(self, target_pair: str = None):
        """Run a single scan of all pairs or specific pair"""
        print(f"🤖 REAL-TIME FOREX SIGNALS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 90)
        
        if target_pair:
            # Analyze single pair in detail
            signal = self.analyze_single_pair(target_pair)
            self.save_signals([signal])
            return [signal]
        else:
            # Scan all pairs
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
    
    parser = argparse.ArgumentParser(description='Enhanced Real-Time Trade-2 Forex Predictor')
    parser.add_argument('--model', type=str, default='best_model.pth', help='Path to model file')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold (0.0-1.0)')
    parser.add_argument('--pair', type=str, help='Analyze specific pair (e.g., EURUSD=X)')
    parser.add_argument('--continuous', action='store_true', help='Run continuous monitoring')
    parser.add_argument('--interval', type=int, default=15, help='Scan interval in minutes (for continuous mode)')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = RealTimeTrade2Predictor(
        model_path=args.model,
        confidence_threshold=args.confidence
    )
    
    if args.continuous:
        if args.pair:
            print(f"🔄 Starting continuous monitoring for {args.pair} (every {args.interval} minutes)")
            print("Press Ctrl+C to stop")
            
            try:
                while True:
                    predictor.run_single_scan(args.pair)
                    print(f"\n⏰ Next scan in {args.interval} minutes...")
                    time.sleep(args.interval * 60)
            except KeyboardInterrupt:
                print(f"\n\n🛑 Monitoring stopped by user")
        else:
            predictor.run_continuous(args.interval)
    else:
        predictor.run_single_scan(args.pair)

if __name__ == "__main__":
    main()
