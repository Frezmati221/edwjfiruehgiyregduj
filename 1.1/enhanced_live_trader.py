"""
ENHANCED REAL-TIME TRADING SYSTEM
Live trading with enhanced 75-epoch models
"""

import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import ta
from datetime import datetime, timedelta
import time
import threading
import json
import logging
from typing import Dict, Optional, List
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('live_trading.log'),
        logging.StreamHandler()
    ]
)

class EnhancedLiveTrader:
    """Real-time trading system with enhanced models"""
    
    def __init__(self, initial_balance: float = 10000, demo_mode: bool = True):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.demo_mode = demo_mode
        self.enhanced_models = {}
        self.positions = {}
        self.trade_history = []
        self.is_running = False
        self.last_data_update = {}
        
        # Trading parameters
        self.position_size_pct = 0.02  # 2% risk per trade
        self.max_daily_risk = 0.06     # 6% max daily risk
        self.max_positions = 3         # Max concurrent positions
        self.update_interval = 60      # Update every 60 seconds
        
        # Trading pairs
        self.pairs = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X']
        
        # Daily tracking
        self.daily_trades = 0
        self.daily_risk = 0
        self.last_trade_day = None
        
        print("🚀 ENHANCED LIVE TRADING SYSTEM")
        print(f"Mode: {'DEMO' if demo_mode else 'LIVE'}")
        print(f"Initial Balance: ${initial_balance:,.2f}")
        print("=" * 60)
        
        self.load_enhanced_models()
    
    def load_enhanced_models(self):
        """Load enhanced models for trading"""
        
        logging.info("Loading enhanced models...")
        
        for pair in self.pairs:
            pair_key = pair.replace('=', '_')
            try:
                with open(f'enhanced_models/{pair_key}_enhanced_loss_learning.pkl', 'rb') as f:
                    self.enhanced_models[pair] = pickle.load(f)
                logging.info(f"✅ Enhanced model loaded: {pair}")
            except Exception as e:
                logging.error(f"❌ Failed to load model {pair}: {e}")
        
        if not self.enhanced_models:
            raise Exception("No enhanced models loaded!")
        
        logging.info(f"📊 Enhanced models ready: {len(self.enhanced_models)}")
    
    def get_live_data(self, pair: str, period: str = '5d') -> Optional[pd.DataFrame]:
        """Get live market data"""
        
        try:
            ticker = yf.Ticker(pair)
            data = ticker.history(period=period, interval='1h')
            
            if len(data) < 50:
                logging.warning(f"Insufficient data for {pair}")
                return None
            
            # Update last data timestamp
            self.last_data_update[pair] = datetime.now()
            
            return data
            
        except Exception as e:
            logging.error(f"Failed to get data for {pair}: {e}")
            return None
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create technical features for prediction"""
        
        df = data.copy()
        
        # Basic price features
        df['returns'] = df['Close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        
        # Trend indicators
        df['sma_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['ema_12'] = ta.trend.ema_indicator(df['Close'], window=12)
        
        # MACD system
        macd = ta.trend.MACD(df['Close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
        # Momentum
        df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['Close'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Trend strength
        df['adx'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)
        
        # Market structure
        df['trend_up'] = (df['Close'] > df['sma_20']).astype(float)
        df['trend_down'] = (df['Close'] < df['sma_20']).astype(float)
        df['strong_trend'] = (df['adx'] > 25).astype(float)
        df['weak_trend'] = (df['adx'] < 20).astype(float)
        
        # Momentum conditions
        df['oversold'] = (df['rsi'] < 30).astype(float)
        df['overbought'] = (df['rsi'] > 70).astype(float)
        df['neutral_rsi'] = ((df['rsi'] >= 40) & (df['rsi'] <= 60)).astype(float)
        
        # MACD signals
        df['macd_bullish'] = (df['macd'] > df['macd_signal']).astype(float)
        df['macd_bearish'] = (df['macd'] < df['macd_signal']).astype(float)
        
        return df.dropna()
    
    def make_prediction(self, pair: str, data: pd.DataFrame) -> Optional[Dict]:
        """Make prediction using enhanced model"""
        
        if pair not in self.enhanced_models:
            return None
        
        try:
            model_data = self.enhanced_models[pair]
            model = model_data['model']
            scaler = model_data['scaler']
            
            features_df = self.create_features(data)
            
            if len(features_df) < 30:
                return None
            
            # Feature columns
            feature_cols = ['Close', 'returns', 'volatility', 'sma_20', 'sma_50', 'ema_12',
                           'macd', 'macd_signal', 'macd_histogram', 'rsi', 'bb_position', 'adx',
                           'trend_up', 'trend_down', 'strong_trend', 'oversold', 'overbought',
                           'macd_bullish', 'macd_bearish']
            
            available_cols = [col for col in feature_cols if col in features_df.columns]
            
            if len(available_cols) < 15:
                return None
            
            # Get latest sequence
            sequence = features_df.iloc[-30:][available_cols].values
            sequence_scaled = scaler.transform(sequence.reshape(-1, sequence.shape[-1])).reshape(1, sequence.shape[0], sequence.shape[1])
            
            # Make prediction
            pred = model.predict(sequence_scaled, verbose=0)
            
            current_price = features_df.iloc[-1]['Close']
            current_time = features_df.index[-1]
            
            prediction = {
                'timestamp': current_time,
                'pair': pair,
                'price': current_price,
                'signal_prob': float(pred[0][0][0]),
                'confidence': float(pred[1][0][0]),
                'reward_estimate': float(pred[2][0][0]),
                'rsi': features_df.iloc[-1]['rsi'],
                'adx': features_df.iloc[-1]['adx'],
                'macd_bullish': bool(features_df.iloc[-1]['macd_bullish']),
                'trend_up': bool(features_df.iloc[-1]['trend_up'])
            }
            
            return prediction
            
        except Exception as e:
            logging.error(f"Prediction failed for {pair}: {e}")
            return None
    
    def should_trade(self, prediction: Dict) -> bool:
        """Determine if we should trade based on prediction"""
        
        # Enhanced model thresholds (optimized from backtesting)
        signal_threshold = 0.070
        confidence_threshold = 0.640
        reward_threshold = -0.090  # Accept some negative rewards (conservative model)
        
        # Basic signal check
        signal_check = (
            prediction['signal_prob'] > signal_threshold and
            prediction['confidence'] > confidence_threshold and
            prediction['reward_estimate'] > reward_threshold
        )
        
        if not signal_check:
            return False
        
        # Risk management checks
        current_day = datetime.now().date()
        if self.last_trade_day != current_day:
            self.daily_trades = 0
            self.daily_risk = 0
            self.last_trade_day = current_day
        
        # Daily limits
        if self.daily_trades >= 10:  # Max 10 trades per day
            logging.info("Daily trade limit reached")
            return False
        
        if self.daily_risk >= self.max_daily_risk:
            logging.info("Daily risk limit reached")
            return False
        
        # Position limits
        if len(self.positions) >= self.max_positions:
            logging.info("Maximum positions reached")
            return False
        
        # Check if we already have a position in this pair
        if prediction['pair'] in self.positions:
            logging.info(f"Already have position in {prediction['pair']}")
            return False
        
        return True
    
    def calculate_position_size(self) -> float:
        """Calculate position size based on current balance"""
        return self.balance * self.position_size_pct
    
    def get_pair_params(self, pair: str) -> Dict:
        """Get pair-specific trading parameters"""
        
        if 'JPY' in pair:
            return {
                'pip_value': 0.01,
                'spread': 1.5 * 0.01,
                'tp_pips': 40 * 0.01,
                'sl_pips': 20 * 0.01
            }
        else:
            return {
                'pip_value': 0.0001,
                'spread': 1.5 * 0.0001,
                'tp_pips': 40 * 0.0001,
                'sl_pips': 20 * 0.0001
            }
    
    def open_position(self, prediction: Dict) -> bool:
        """Open a new trading position"""
        
        try:
            pair = prediction['pair']
            current_price = prediction['price']
            
            # Determine direction
            direction = 'LONG' if prediction['reward_estimate'] > -0.080 else 'SHORT'
            
            # Get pair parameters
            params = self.get_pair_params(pair)
            
            # Calculate position size
            risk_amount = self.calculate_position_size()
            
            # Calculate entry price with spread
            if direction == 'LONG':
                entry_price = current_price + params['spread'] / 2
                tp_price = entry_price + params['tp_pips']
                sl_price = entry_price - params['sl_pips']
            else:
                entry_price = current_price - params['spread'] / 2
                tp_price = entry_price - params['tp_pips']
                sl_price = entry_price + params['sl_pips']
            
            # Create position
            position = {
                'pair': pair,
                'direction': direction,
                'entry_time': datetime.now(),
                'entry_price': entry_price,
                'tp_price': tp_price,
                'sl_price': sl_price,
                'risk_amount': risk_amount,
                'signal_prob': prediction['signal_prob'],
                'confidence': prediction['confidence'],
                'reward_estimate': prediction['reward_estimate'],
                'status': 'OPEN'
            }
            
            # Store position
            self.positions[pair] = position
            
            # Update daily tracking
            self.daily_trades += 1
            self.daily_risk += self.position_size_pct
            
            # Log trade
            logging.info(f"🔵 OPENED {direction} {pair}")
            logging.info(f"   Entry: {entry_price:.5f}")
            logging.info(f"   TP: {tp_price:.5f} | SL: {sl_price:.5f}")
            logging.info(f"   Risk: ${risk_amount:.2f}")
            logging.info(f"   Signal: {prediction['signal_prob']:.3f} | Confidence: {prediction['confidence']:.3f}")
            
            if self.demo_mode:
                print(f"📈 DEMO TRADE: {direction} {pair} @ {entry_price:.5f}")
                print(f"   Risk: ${risk_amount:.2f} | TP: {tp_price:.5f} | SL: {sl_price:.5f}")
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to open position: {e}")
            return False
    
    def check_position_exit(self, pair: str, current_price: float) -> Optional[str]:
        """Check if position should be closed"""
        
        if pair not in self.positions:
            return None
        
        position = self.positions[pair]
        
        if position['direction'] == 'LONG':
            if current_price >= position['tp_price']:
                return 'TP'
            elif current_price <= position['sl_price']:
                return 'SL'
        else:  # SHORT
            if current_price <= position['tp_price']:
                return 'TP'
            elif current_price >= position['sl_price']:
                return 'SL'
        
        # Time-based exit (24 hours max)
        if datetime.now() - position['entry_time'] > timedelta(hours=24):
            return 'TIMEOUT'
        
        return None
    
    def close_position(self, pair: str, exit_reason: str, current_price: float) -> bool:
        """Close an existing position"""
        
        if pair not in self.positions:
            return False
        
        try:
            position = self.positions[pair]
            
            # Calculate P&L
            if exit_reason == 'TP':
                pnl = position['risk_amount'] * 2  # 2:1 RR
                result = 'WIN'
                exit_price = position['tp_price']
            elif exit_reason == 'SL':
                pnl = -position['risk_amount']
                result = 'LOSS'
                exit_price = position['sl_price']
            else:  # TIMEOUT
                # Calculate actual P&L based on current price
                if position['direction'] == 'LONG':
                    price_diff = current_price - position['entry_price']
                else:
                    price_diff = position['entry_price'] - current_price
                
                params = self.get_pair_params(pair)
                pnl_ratio = price_diff / params['sl_pips']
                pnl = position['risk_amount'] * pnl_ratio
                result = 'WIN' if pnl > 0 else 'LOSS'
                exit_price = current_price
            
            # Update balance
            self.balance += pnl
            
            # Create trade record
            trade_record = {
                'entry_time': position['entry_time'],
                'exit_time': datetime.now(),
                'pair': pair,
                'direction': position['direction'],
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'exit_reason': exit_reason,
                'pnl': pnl,
                'result': result,
                'balance': self.balance,
                'signal_prob': position['signal_prob'],
                'confidence': position['confidence'],
                'reward_estimate': position['reward_estimate']
            }
            
            self.trade_history.append(trade_record)
            
            # Remove position
            del self.positions[pair]
            
            # Log trade
            logging.info(f"🔴 CLOSED {position['direction']} {pair} - {exit_reason}")
            logging.info(f"   Exit: {exit_price:.5f}")
            logging.info(f"   P&L: ${pnl:+.2f} | Balance: ${self.balance:.2f}")
            logging.info(f"   Result: {result}")
            
            if self.demo_mode:
                print(f"📉 DEMO CLOSE: {position['direction']} {pair} - {exit_reason}")
                print(f"   P&L: ${pnl:+.2f} | Balance: ${self.balance:.2f}")
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to close position: {e}")
            return False
    
    def update_positions(self):
        """Update all open positions"""
        
        positions_to_close = []
        
        for pair in list(self.positions.keys()):
            # Get current price
            data = self.get_live_data(pair, period='1d')
            if data is None or len(data) == 0:
                continue
            
            current_price = data['Close'].iloc[-1]
            
            # Check for exit
            exit_reason = self.check_position_exit(pair, current_price)
            if exit_reason:
                positions_to_close.append((pair, exit_reason, current_price))
        
        # Close positions that need to be closed
        for pair, exit_reason, current_price in positions_to_close:
            self.close_position(pair, exit_reason, current_price)
    
    def scan_for_signals(self):
        """Scan all pairs for trading signals"""
        
        for pair in self.pairs:
            try:
                # Get live data
                data = self.get_live_data(pair)
                if data is None:
                    continue
                
                # Make prediction
                prediction = self.make_prediction(pair, data)
                if prediction is None:
                    continue
                
                # Check if we should trade
                if self.should_trade(prediction):
                    self.open_position(prediction)
                
            except Exception as e:
                logging.error(f"Error scanning {pair}: {e}")
    
    def print_status(self):
        """Print current trading status"""
        
        print(f"\n📊 LIVE TRADING STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        print(f"💰 Balance: ${self.balance:,.2f} ({((self.balance - self.initial_balance) / self.initial_balance * 100):+.1f}%)")
        print(f"📈 Open Positions: {len(self.positions)}")
        print(f"📊 Daily Trades: {self.daily_trades}")
        print(f"⚠️ Daily Risk: {self.daily_risk:.1%}")
        
        if self.positions:
            print(f"\n🔵 OPEN POSITIONS:")
            for pair, pos in self.positions.items():
                duration = datetime.now() - pos['entry_time']
                print(f"   {pos['direction']} {pair}: {duration} | Entry: {pos['entry_price']:.5f}")
        
        if self.trade_history:
            recent_trades = self.trade_history[-5:]
            wins = sum(1 for t in recent_trades if t['result'] == 'WIN')
            print(f"\n📈 Recent Performance (last 5 trades): {wins}/{len(recent_trades)} wins")
    
    def save_state(self):
        """Save current trading state"""
        
        state = {
            'timestamp': datetime.now().isoformat(),
            'balance': self.balance,
            'initial_balance': self.initial_balance,
            'positions': self.positions,
            'trade_history': self.trade_history,
            'daily_trades': self.daily_trades,
            'daily_risk': self.daily_risk
        }
        
        with open('live_trading_state.json', 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def trading_loop(self):
        """Main trading loop"""
        
        logging.info("🚀 Starting live trading loop...")
        
        while self.is_running:
            try:
                # Update existing positions
                self.update_positions()
                
                # Scan for new signals
                self.scan_for_signals()
                
                # Print status every 10 minutes
                if datetime.now().minute % 10 == 0:
                    self.print_status()
                
                # Save state
                self.save_state()
                
                # Wait before next update
                time.sleep(self.update_interval)
                
            except KeyboardInterrupt:
                logging.info("Received stop signal")
                break
            except Exception as e:
                logging.error(f"Error in trading loop: {e}")
                time.sleep(self.update_interval)
        
        logging.info("Trading loop stopped")
    
    def start_trading(self):
        """Start the live trading system"""
        
        if self.is_running:
            print("Trading system is already running!")
            return
        
        self.is_running = True
        
        print(f"\n🚀 STARTING ENHANCED LIVE TRADING")
        print(f"Mode: {'DEMO' if self.demo_mode else 'LIVE'}")
        print(f"Pairs: {', '.join(self.pairs)}")
        print(f"Update Interval: {self.update_interval}s")
        print(f"Position Size: {self.position_size_pct:.1%}")
        print("Press Ctrl+C to stop")
        print("=" * 60)
        
        # Start trading in a separate thread
        trading_thread = threading.Thread(target=self.trading_loop)
        trading_thread.daemon = True
        trading_thread.start()
        
        try:
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping trading system...")
            self.stop_trading()
    
    def stop_trading(self):
        """Stop the trading system"""
        
        self.is_running = False
        
        # Close all open positions
        if self.positions:
            print("Closing all open positions...")
            for pair in list(self.positions.keys()):
                data = self.get_live_data(pair, period='1d')
                if data is not None:
                    current_price = data['Close'].iloc[-1]
                    self.close_position(pair, 'MANUAL_CLOSE', current_price)
        
        # Final status
        self.print_status()
        self.save_state()
        
        # Summary
        if self.trade_history:
            total_trades = len(self.trade_history)
            wins = sum(1 for t in self.trade_history if t['result'] == 'WIN')
            win_rate = wins / total_trades * 100
            total_pnl = sum(t['pnl'] for t in self.trade_history)
            
            print(f"\n📊 TRADING SESSION SUMMARY:")
            print(f"   Total Trades: {total_trades}")
            print(f"   Win Rate: {win_rate:.1f}%")
            print(f"   Total P&L: ${total_pnl:+.2f}")
            print(f"   Final Balance: ${self.balance:.2f}")
            print(f"   Return: {((self.balance - self.initial_balance) / self.initial_balance * 100):+.1f}%")
        
        logging.info("Trading system stopped")

if __name__ == "__main__":
    # Create live trader instance
    trader = EnhancedLiveTrader(
        initial_balance=10000,
        demo_mode=True  # Set to False for real trading
    )
    
    # Start trading
    trader.start_trading()
