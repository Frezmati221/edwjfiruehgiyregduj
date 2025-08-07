import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler  # For mixed precision
from collections import deque
import random
from typing import Dict, List, Tuple, Optional
import talib
from dataclasses import dataclass
import warnings
import argparse
import yfinance as yf
from datetime import datetime, timedelta
import os
from tqdm import tqdm
import time
warnings.filterwarnings('ignore')

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Using device: {device}")

# Enable optimizations for maximum GPU usage
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
    torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms for speed
    torch.cuda.empty_cache()  # Clear GPU cache
    
    # Set memory management for better allocation
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("🚀 GPU optimizations enabled")
    print("💾 Memory fragmentation prevention enabled")
else:
    print("⚠️  CUDA not available, using CPU")

@dataclass
class TradingAction:
    """Trading action with position details"""
    direction: str  # 'long', 'short', or 'hold'
    take_profit: float
    stop_loss: float
    confidence: float

class ForexIndicators:
    """Calculate advanced forex indicators"""
    
    @staticmethod
    def vmc_cipher(high, low, close, volume, period=14):
        """VMC Cipher B indicator simulation"""
        mfi = talib.MFI(high, low, close, volume, timeperiod=period)
        rsi = talib.RSI(close, timeperiod=period)
        
        # Weighted combination
        vmc = (mfi * 0.7 + rsi * 0.3)
        return vmc
    
    @staticmethod
    def support_resistance_levels(high, low, close, window=20):
        """Calculate dynamic support and resistance levels"""
        pivot = (high + low + close) / 3
        resistance1 = 2 * pivot - low
        support1 = 2 * pivot - high
        resistance2 = pivot + (high - low)
        support2 = pivot - (high - low)
        
        return {
            'pivot': pivot,
            'resistance1': resistance1,
            'resistance2': resistance2,
            'support1': support1,
            'support2': support2
        }
    
    def calculate_all_indicators(self, df):
        """Calculate all indicators and prepare feature matrix"""
        # Technical indicators
        df['sma_10'] = talib.SMA(df['close'], timeperiod=10)
        df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
        df['ema_12'] = talib.EMA(df['close'], timeperiod=12)
        df['ema_26'] = talib.EMA(df['close'], timeperiod=26)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(df['close'])
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist
        
        # RSI
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        
        # Stochastic
        slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'])
        df['stoch_k'] = slowk
        df['stoch_d'] = slowd
        
        # Average True Range
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Williams %R
        df['williams_r'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Custom VMC Cipher
        df['vmc_cipher'] = self.vmc_cipher(df['high'], df['low'], df['close'], df['volume'])
        
        # Support/Resistance levels
        sr_levels = self.support_resistance_levels(df['high'], df['low'], df['close'])
        for key, value in sr_levels.items():
            df[key] = value
        
        # Price action features
        df['price_change'] = df['close'].pct_change()
        df['volatility'] = df['price_change'].rolling(window=10).std()
        df['high_low_ratio'] = df['high'] / df['low']
        
        # Volume indicators
        df['volume_sma'] = talib.SMA(df['volume'], timeperiod=20)
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Fill any NaN values
        df = df.bfill().ffill()
        
        return df

def load_forex_data(period: str = "1y", interval: str = "1h") -> Dict[str, pd.DataFrame]:
    """
    Load forex data from Yahoo Finance (free source)
    
    Args:
        period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
    
    Returns:
        Dictionary with pair names as keys and DataFrames as values
    """
    # Major forex pairs available on Yahoo Finance
    pairs = {
        'EURUSD': 'EURUSD=X',
        'GBPUSD': 'GBPUSD=X', 
        'USDJPY': 'USDJPY=X',
        'USDCHF': 'USDCHF=X',
        'AUDUSD': 'AUDUSD=X',
        'USDCAD': 'USDCAD=X'
    }
    
    data = {}
    print(f"Loading forex data for period: {period}, interval: {interval}")
    
    for pair_name, yahoo_symbol in pairs.items():
        try:
            print(f"Downloading {pair_name}...")
            ticker = yf.Ticker(yahoo_symbol)
            df = ticker.history(period=period, interval=interval)
            
            if not df.empty:
                # Ensure we have the required columns
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                df.columns = ['open', 'high', 'low', 'close', 'volume']
                
                # Fill any missing volume data with average volume
                if df['volume'].isna().all():
                    df['volume'] = 1000000  # Default volume for forex
                else:
                    df['volume'] = df['volume'].fillna(df['volume'].mean())
                
                # Drop any remaining NaN values
                df = df.dropna()
                
                if len(df) > 100:  # Ensure we have enough data
                    data[pair_name] = df
                    print(f"✓ {pair_name}: {len(df)} data points loaded")
                else:
                    print(f"✗ {pair_name}: Insufficient data ({len(df)} points)")
            else:
                print(f"✗ {pair_name}: No data available")
                
        except Exception as e:
            print(f"✗ Error loading {pair_name}: {str(e)}")
    
    if not data:
        print("Warning: No forex data could be loaded. Check your internet connection.")
        # Create sample data for testing
        print("Creating sample data for testing...")
        sample_data = create_sample_data()
        return sample_data
    
    return data

def create_sample_data() -> Dict[str, pd.DataFrame]:
    """Create sample forex data for testing when real data is unavailable"""
    pairs = ['EURUSD', 'GBPUSD', 'USDJPY']
    data = {}
    
    for pair in pairs:
        # Generate realistic forex price movements
        np.random.seed(42)
        n_points = 1000
        
        # Starting prices
        if pair == 'USDJPY':
            base_price = 110.0
        else:
            base_price = 1.20
        
        # Generate price series with random walk
        returns = np.random.normal(0, 0.001, n_points)
        prices = [base_price]
        
        for i in range(1, n_points):
            new_price = prices[-1] * (1 + returns[i])
            prices.append(new_price)
        
        # Create OHLC data
        df_data = []
        for i in range(len(prices) - 1):
            open_price = prices[i]
            close_price = prices[i + 1]
            high_price = max(open_price, close_price) * (1 + np.random.uniform(0, 0.002))
            low_price = min(open_price, close_price) * (1 - np.random.uniform(0, 0.002))
            volume = np.random.uniform(500000, 2000000)
            
            df_data.append({
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
        
        df = pd.DataFrame(df_data)
        data[pair] = df
        print(f"✓ {pair}: {len(df)} sample data points created")
    
    return data
    
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        df = df.copy()
        
        # Price-based indicators
        df['sma_5'] = talib.SMA(df['close'], timeperiod=5)
        df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
        df['sma_50'] = talib.SMA(df['close'], timeperiod=50)
        df['ema_12'] = talib.EMA(df['close'], timeperiod=12)
        df['ema_26'] = talib.EMA(df['close'], timeperiod=26)
        
        # Momentum indicators
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
        df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'], df['low'], df['close'])
        
        # Volatility indicators
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'])
        
        # Volume indicators
        if 'volume' in df.columns:
            df['obv'] = talib.OBV(df['close'], df['volume'])
            df['vmc_cipher'] = ForexIndicators.vmc_cipher(
                df['high'], df['low'], df['close'], df['volume']
            )
        
        # Support/Resistance
        sr_levels = ForexIndicators.support_resistance_levels(
            df['high'], df['low'], df['close']
        )
        for key, value in sr_levels.items():
            df[f'sr_{key}'] = value
        
        # Pattern recognition
        df['doji'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
        df['hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
        df['engulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
        
        # Price action features
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        df['upper_shadow'] = df['high'] - np.maximum(df['close'], df['open'])
        df['lower_shadow'] = np.minimum(df['close'], df['open']) - df['low']
        
        return df

class ForexEnvironment:
    """Trading environment for reinforcement learning"""
    
    def __init__(self, data: pd.DataFrame, lookback_period: int = 30):
        self.data = data
        self.lookback_period = lookback_period
        self.current_step = lookback_period
        self.max_steps = len(data) - 1
        self.position = None
        self.entry_price = 0
        self.balance = 10000
        self.initial_balance = 10000
        self.trade_history = []
        
    def get_pip_value(self, price: float) -> float:
        """Get pip value for CFD contract based on instrument type"""
        # Extract pair name from data if available, otherwise detect from price range
        # For intraday CFD trading
        
        if hasattr(self.data, 'name') and self.data.name:
            pair = self.data.name
        else:
            # Detect pair type from price range
            if price > 100:  # Likely JPY pair (e.g., USDJPY ~150)
                pair = 'JPY'
            elif 0.5 < price < 2.0:  # Major pairs (e.g., EURUSD ~1.05)
                pair = 'MAJOR'
            else:
                pair = 'OTHER'
        
        # CFD pip values for standard retail contracts
        if 'JPY' in str(pair) or price > 100:
            return 0.01  # JPY pairs: 1 pip = 0.01
        else:
            return 0.0001  # Major pairs: 1 pip = 0.0001
        
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = self.lookback_period
        self.position = None
        self.entry_price = 0
        self.balance = self.initial_balance
        self.trade_history = []
        return self._get_state()
    
    def _get_state(self):
        """Get current state including lookback window"""
        lookback_data = self.data.iloc[
            self.current_step - self.lookback_period:self.current_step
        ]
        
        # Normalize features
        features = []
        for col in lookback_data.columns:
            if col not in ['open', 'high', 'low', 'close', 'volume']:
                values = lookback_data[col].values
                if not np.isnan(values).all():
                    normalized = (values - np.nanmean(values)) / (np.nanstd(values) + 1e-8)
                    features.append(normalized)
        
        return np.array(features).flatten()
    
    def step(self, action: TradingAction):
        """Execute trading action and return reward"""
        current_price = self.data.iloc[self.current_step]['close']
        reward = 0
        
        # Close existing position if any
        if self.position:
            if self.position == 'long':
                profit = current_price - self.entry_price
            else:  # short
                profit = self.entry_price - current_price
            
            # Calculate pip-based profit for consistent rewards across pairs
            pip_value = self.get_pip_value(current_price)
            pips_gained = profit / pip_value
            
            # REALISTIC TRAINING: Proper risk management (2% risk per trade)
            # Calculate position size based on 2% risk and 20-pip stop loss
            risk_amount = self.balance * 0.02  # Risk 2% of balance per trade
            pip_value = self.get_pip_value(current_price)
            
            # Position size = Risk Amount / (Stop Loss in Pips * Pip Value)
            # For USDJPY: $20 risk / (20 pips * $0.01) = 100 units
            position_size = risk_amount / (20 * pip_value)  # 20-pip stop loss
            actual_profit = profit * position_size
            
            # Apply realistic stop loss/take profit with improved reward structure
            if pips_gained <= -20:  # Stop loss hit
                actual_profit = -risk_amount  # Lose exactly 2% of balance
                reward = -50  # Further reduced penalty to prevent spiraling
            elif pips_gained >= 40:  # Take profit hit (2:1 ratio)
                actual_profit = risk_amount * 2  # Gain 4% of balance (2:1 ratio)
                reward = 100  # Balanced reward for correct direction prediction
            else:
                # Regular profit/loss - more conservative rewards to prevent extremes
                reward = max(-25, min(25, pips_gained * 1))  # Much more conservative clamping
            
            self.balance += actual_profit
            self.position = None
        
        # Open new position
        if action.direction != 'hold':
            # Prevent trading if balance too low (realistic risk management)
            if self.balance < 100:  # Minimum $100 to trade
                reward -= 5  # Further reduced penalty to prevent spiral
            else:
                self.position = action.direction
                self.entry_price = current_price
                self.trade_history.append({
                    'step': self.current_step,
                    'action': action.direction,
                    'price': current_price,
                    'tp': action.take_profit,
                    'sl': action.stop_loss
                })
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # More conservative penalty for negative balance (prevent extreme spiraling)
        if self.balance <= 0:
            reward -= 50  # Much more conservative penalty
            done = True
        
        return self._get_state(), reward, done

class DQNNetwork(nn.Module):
    """Deep Q-Network for trading decisions"""
    
    def __init__(self, input_size: int, hidden_size: int = 1536):  # Reduced from 2048 for memory
        super(DQNNetwork, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size * 12),  # Reduced from 16x
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size * 12, hidden_size * 6),  # Reduced from 8x
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size * 6, hidden_size * 3),  # Reduced from 4x
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size * 3, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU()
        )
        
        # Separate heads for different outputs
        self.direction_head = nn.Linear(hidden_size // 2, 3)  # long, short, hold
        self.tp_head = nn.Linear(hidden_size // 2, 10)  # TP levels
        self.sl_head = nn.Linear(hidden_size // 2, 10)  # SL levels
        
    def forward(self, x):
        features = self.feature_extractor(x)
        
        direction = self.direction_head(features)
        tp_levels = self.tp_head(features)
        sl_levels = self.sl_head(features)
        
        return direction, tp_levels, sl_levels

class ForexTradingAgent:
    """Main trading agent with DQN"""
    
    def __init__(self, state_size: int, learning_rate: float = 0.001):
        self.state_size = state_size
        self.epsilon = 1.0
        self.epsilon_min = 0.05  # Higher minimum to maintain exploration
        self.epsilon_decay = 0.998  # Slower decay for more stable learning
        self.learning_rate = learning_rate
        self.gamma = 0.95
        
                # Balanced batch size based on GPU memory (optimized for stability)
        if device.type == 'cuda':
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_memory_gb >= 15:  # High-end GPU (RTX 5080, etc.)
                self.batch_size = 1024  # Balanced for stability and performance
            elif gpu_memory_gb >= 12:  # Mid-high GPU (RTX 4070, etc.)
                self.batch_size = 512   # Balanced for good performance
            elif gpu_memory_gb >= 8:   # Mid-range GPU
                self.batch_size = 256
            else:  # Lower-end GPU
                self.batch_size = 128
        else:
            self.batch_size = 64
        
        # Optimize gradient accumulation for balanced learning
        self.gradient_accumulation_steps = 2  # Reduced for better flow
        
        # Optimized memory buffer for performance and stability
        self.memory = deque(maxlen=30000)  # Reduced to prevent memory buildup
        
        # Neural networks - move to GPU
        self.q_network = DQNNetwork(state_size).to(device)
        self.target_network = DQNNetwork(state_size).to(device)
        
        # Only use DataParallel for multiple high-end GPUs (disabled for better performance)
        # DataParallel often slows down training for most forex trading scenarios
        if False:  # Disabled multi-GPU for optimal performance
            print(f"🔥 Using {torch.cuda.device_count()} GPUs with DataParallel")
            self.q_network = nn.DataParallel(self.q_network)
            self.target_network = nn.DataParallel(self.target_network)
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Mixed precision training
        self.scaler = GradScaler() if device.type == 'cuda' else None
        
        self.update_target_network()
        
        print(f"🧠 Neural networks initialized on {device}")
        print(f"📊 Batch size: {self.batch_size} (effective: {self.batch_size * self.gradient_accumulation_steps})")
        print(f"🔄 Gradient accumulation: {self.gradient_accumulation_steps} steps")
        print(f"💾 Memory buffer: {self.memory.maxlen:,} experiences")
        if self.scaler:
            print("⚡ Mixed precision training enabled")
        print("🎯 BALANCED TRAINING: ACCURACY + STABILITY")
        
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray) -> TradingAction:
        """Choose action using epsilon-greedy policy with timeout protection"""
        if random.random() <= self.epsilon:
            # Random action
            direction = random.choice(['long', 'short', 'hold'])
            tp = random.uniform(20, 100)  # pips
            sl = random.uniform(10, 50)   # pips
        else:
            # Predict using network with timeout protection
            try:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device, non_blocking=True)
                with torch.no_grad():
                    # Add timeout for inference
                    start_time = time.time()
                    direction_q, tp_q, sl_q = self.q_network(state_tensor)
                    inference_time = time.time() - start_time
                    
                    if inference_time > 2:  # If inference takes more than 2 seconds
                        print(f"⚠️  Slow inference: {inference_time:.1f}s")
                
                direction_idx = torch.argmax(direction_q).item()
                direction = ['long', 'short', 'hold'][direction_idx]
                
                tp_idx = torch.argmax(tp_q).item()
                sl_idx = torch.argmax(sl_q).item()
                
                tp = 20 + tp_idx * 10  # 20-120 pips
                sl = 10 + sl_idx * 5   # 10-60 pips
                
                # Clear tensor less frequently for performance
                del state_tensor
                # Only clear cache very occasionally to avoid performance hit
                if device.type == 'cuda' and random.random() < 0.01:  # 1% chance instead of 10%
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"⚠️  Neural network prediction failed: {str(e)} - using random action")
                direction = random.choice(['long', 'short', 'hold'])
                tp = random.uniform(20, 100)
                sl = random.uniform(10, 50)
        
        confidence = 1.0 - self.epsilon
        return TradingAction(direction, tp, sl, confidence)
    
    def replay(self):
        """Train the network on batch of experiences with focus on prediction accuracy"""
        if len(self.memory) < self.batch_size:
            return
        
        try:
            # Initialize accumulation variables for better gradient quality
            total_loss = 0
            self.optimizer.zero_grad()
            
            # Gradient accumulation loop for better learning quality
            for accumulation_step in range(self.gradient_accumulation_steps):
                # Sample different batches for each accumulation step
                batch_indices = random.sample(range(len(self.memory)), self.batch_size)
                batch = [self.memory[i] for i in batch_indices]
                
                # Efficient batch preprocessing
                states_list = []
                next_states_list = []
                rewards_list = []
                dones_list = []
                actions_list = []
                
                for e in batch:
                    states_list.append(e[0])
                    next_states_list.append(e[3])
                    rewards_list.append(e[2])
                    dones_list.append(e[4])
                    # Inline action encoding for speed
                    if e[1].direction == 'long':
                        actions_list.append(0)
                    elif e[1].direction == 'short':
                        actions_list.append(1)
                    else:
                        actions_list.append(2)
                
                # Create tensors
                states = torch.FloatTensor(states_list).to(device, non_blocking=True)
                next_states = torch.FloatTensor(next_states_list).to(device, non_blocking=True)
                rewards = torch.FloatTensor(rewards_list).to(device, non_blocking=True)
                dones = torch.FloatTensor(dones_list).to(device, non_blocking=True)
                actions = torch.LongTensor(actions_list).unsqueeze(1).to(device, non_blocking=True)
                
                if self.scaler:  # Mixed precision training
                    with autocast():
                        current_q_values = self.q_network(states)
                        with torch.no_grad():
                            next_q_values = self.target_network(next_states)
                        
                        # Calculate target Q-values
                        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values[0].max(1)[0]
                        current_q = current_q_values[0].gather(1, actions).squeeze()
                        loss = nn.MSELoss()(current_q, target_q_values.detach())
                        
                        # Scale loss for gradient accumulation
                        loss = loss / self.gradient_accumulation_steps
                    
                    # Accumulate gradients
                    self.scaler.scale(loss).backward()
                    total_loss += loss.item()
                else:
                    current_q_values = self.q_network(states)
                    with torch.no_grad():
                        next_q_values = self.target_network(next_states)
                    
                    # Calculate target Q-values
                    target_q_values = rewards + (1 - dones) * self.gamma * next_q_values[0].max(1)[0]
                    current_q = current_q_values[0].gather(1, actions).squeeze()
                    loss = nn.MSELoss()(current_q, target_q_values.detach())
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps
                    
                    # Accumulate gradients
                    loss.backward()
                    total_loss += loss.item()
                
                # Clear intermediate tensors
                del states, next_states, rewards, dones, actions, states_list, next_states_list
                del rewards_list, dones_list, actions_list
            
            # Apply accumulated gradients
            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
        except torch.cuda.OutOfMemoryError:
            # Handle OOM gracefully
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            print("⚠️  GPU memory warning - clearing cache and continuing...")
            self.optimizer.zero_grad()
        except Exception as e:
            print(f"⚠️  Training error: {str(e)} - skipping batch...")
            self.optimizer.zero_grad()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class ForexPredictor:
    """Main class for forex prediction"""
    
    def __init__(self, pairs: List[str] = ['EURUSD', 'GBPUSD', 'USDJPY']):
        self.pairs = pairs
        self.agents = {}
        self.indicators = ForexIndicators()
        
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data with all indicators"""
        return self.indicators.calculate_all_indicators(df)
    
    def train(self, data: Dict[str, pd.DataFrame], epochs: int = 100):
        """Train the model for each currency pair"""
        for pair_idx, pair in enumerate(self.pairs):
            if pair not in data:
                continue
            
            print(f"\n📈 Training model for {pair} ({pair_idx + 1}/{len(self.pairs)})...")
            prepared_data = self.prepare_data(data[pair])
            
            # Initialize environment and agent
            env = ForexEnvironment(prepared_data)
            state = env.reset()
            agent = ForexTradingAgent(state_size=len(state))
            
            # Training progress bar for this pair
            pbar = tqdm(range(epochs), desc=f"🔄 {pair}", 
                       bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} epochs [⏱️{elapsed}<⏳{remaining}, 💰{postfix}]')
            
            best_reward = float('-inf')
            rewards_history = []
            
            for epoch in pbar:
                start_time = time.time()
                state = env.reset()
                total_reward = 0
                done = False
                steps = 0
                last_step_time = start_time
                
                while not done:
                    step_start_time = time.time()
                    
                    try:
                        action = agent.act(state)
                        next_state, reward, done = env.step(action)
                        agent.remember(state, action, reward, next_state, done)
                        state = next_state
                        total_reward += reward
                        steps += 1
                    except Exception as e:
                        print(f"⚠️  Step {steps} failed: {str(e)} - skipping...")
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        break
                    
                    # Optimize training for better stability and performance
                    # More conservative training approach to prevent negative spirals
                    min_training_memory = agent.batch_size * 4  # Larger buffer for more stable learning
                    
                    if len(agent.memory) > min_training_memory:
                        # More conservative training frequency for stability
                        training_frequency = 25  # Train every 25 steps for more stable learning
                        
                        if steps % training_frequency == 0:
                            training_start = time.time()
                            max_training_time = 5  # Shorter timeout to prevent slowdowns
                            
                            try:
                                # Only train if recent performance isn't terrible
                                if total_reward > -1000:  # Don't train when performance is very poor
                                    agent.replay()
                                else:
                                    print(f"    🚫 Skipping training due to poor performance (reward: {total_reward:.1f})")
                                    
                                training_time = time.time() - training_start
                                
                                if training_time > max_training_time:
                                    print(f"⚠️  Training took {training_time:.1f}s - clearing cache...")
                                    if device.type == 'cuda':
                                        torch.cuda.empty_cache()
                                        torch.cuda.synchronize()
                                    
                            except Exception as e:
                                print(f"⚠️  Training failed: {str(e)} - clearing cache and continuing")
                                if device.type == 'cuda':
                                    torch.cuda.empty_cache()
                        
                        # More frequent cache management to prevent memory buildup
                        if steps % 1000 == 0 and device.type == 'cuda':  # More frequent clearing
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                    
                    # More aggressive step timeout to catch issues early
                    step_time = time.time() - step_start_time
                    if step_time > 3:  # Reduced from 5 to 3 seconds
                        print(f"⚠️  Step {steps} took {step_time:.1f}s - clearing cache...")
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        # Don't break - just warn and continue
                    
                    # Progress monitoring with performance tracking
                    if steps % 200 == 0:  # Every 200 steps
                        elapsed = time.time() - last_step_time
                        print(f"    📊 Step {steps}, Reward: {total_reward:.1f}, Last 200 steps: {elapsed:.1f}s, ε: {agent.epsilon:.3f}")
                        last_step_time = time.time()
                        
                        # Check for performance degradation
                        if elapsed > 10:  # If 200 steps take more than 10 seconds
                            print(f"    ⚠️  Performance degradation detected - clearing memory and resetting")
                            if device.type == 'cuda':
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()
                            # Increase exploration to break out of bad patterns
                            agent.epsilon = min(0.5, agent.epsilon + 0.1)
                        
                        # Regular cache management
                        if device.type == 'cuda' and steps % 2000 == 0:  # More frequent cache clearing
                            torch.cuda.empty_cache()
                
                # Update target network periodically
                if epoch % 10 == 0:
                    agent.update_target_network()
                
                # Improved early stopping with complete reset (prevent spiraling)
                if total_reward < -1500:  # Much more aggressive early stopping
                    print(f"\n⚠️  Early stopping due to poor performance (reward: {total_reward:.1f})")
                    print("    🔄 Complete environment and agent reset...")
                    # Complete reset to prevent cascading failures
                    env.balance = env.initial_balance
                    env.position = None
                    env.current_step = env.lookback_period
                    # Reset agent exploration to add more randomness
                    agent.epsilon = min(0.4, agent.epsilon + 0.2)
                    total_reward = 0  # Complete reset instead of partial cap
                    state = env.reset()  # Get fresh state
                    print(f"    ✅ Complete reset done, epsilon increased to {agent.epsilon:.3f}")
                    # Clear memory issues
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    break  # Exit episode early
                
                # Track performance
                rewards_history.append(total_reward)
                if total_reward > best_reward:
                    best_reward = total_reward
                
                # Calculate average reward over last 10 epochs
                recent_avg = np.mean(rewards_history[-10:]) if len(rewards_history) >= 10 else np.mean(rewards_history)
                
                # Update progress bar with detailed info
                epoch_time = time.time() - start_time
                
                # Add GPU memory info if using CUDA
                gpu_info = ""
                if device.type == 'cuda':
                    torch.cuda.synchronize()  # Ensure all operations complete
                    gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                    gpu_max_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
                    gpu_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
                    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    gpu_utilization = (gpu_reserved / total_memory) * 100
                    gpu_info = f", GPU: {gpu_memory:.1f}/{gpu_reserved:.1f}/{total_memory:.1f}GB ({gpu_utilization:.1f}%)"
                
                postfix = f"Reward: {total_reward:.2f}, Avg: {recent_avg:.2f}, Best: {best_reward:.2f}, ε: {agent.epsilon:.3f}, Steps: {steps}, Time: {epoch_time:.1f}s{gpu_info}"
                pbar.set_postfix_str(postfix)
                
                # Print detailed progress every 20 epochs
                if epoch % 20 == 0 and epoch > 0:
                    print(f"\n    📊 Epoch {epoch:3d} Summary:")
                    print(f"       💰 Reward: {total_reward:8.2f} | Avg(10): {recent_avg:8.2f} | Best: {best_reward:8.2f}")
                    print(f"       🎯 Epsilon: {agent.epsilon:.3f} | Steps: {steps:4d} | Time: {epoch_time:.1f}s")
                    print(f"       📈 Memory: {len(agent.memory):5d} | Loss Rate: {(1-agent.epsilon)*100:.1f}%")
            
            pbar.close()
            print(f"✅ {pair} training completed! Best reward: {best_reward:.2f}")
            self.agents[pair] = agent
    
    def predict(self, pair: str, current_data: pd.DataFrame) -> TradingAction:
        """Predict trading action for given pair"""
        if pair not in self.agents:
            raise ValueError(f"No trained model for {pair}")
        
        prepared_data = self.prepare_data(current_data)
        env = ForexEnvironment(prepared_data)
        state = env._get_state()
        
        # Set epsilon to 0 for prediction (no exploration)
        self.agents[pair].epsilon = 0
        action = self.agents[pair].act(state)
        
        return action
    
    def save_model(self, filepath: str):
        """Save trained models"""
        checkpoint = {
            'agents': {}
        }
        for pair, agent in self.agents.items():
            checkpoint['agents'][pair] = {
                'q_network': agent.q_network.state_dict(),
                'target_network': agent.target_network.state_dict(),
                'epsilon': agent.epsilon
            }
        torch.save(checkpoint, filepath)
    
    def prepare_state(self, df):
        """Prepare state vector from dataframe for prediction"""
        if len(df) < 20:
            # Pad with last available values if not enough data
            last_row = df.iloc[-1]
            while len(df) < 20:
                df = pd.concat([df, last_row.to_frame().T], ignore_index=True)
        
        # Use last 20 rows and prepare state like the environment does
        recent_data = df.tail(20)
        
        # Prepare data with indicators
        prepared_data = self.prepare_data(recent_data)
        
        # Create a temporary environment to get the state
        temp_env = ForexEnvironment(prepared_data)
        state = temp_env._get_state()
        
        return state
    
    def load_model(self, filepath: str):
        """Load trained models"""
        checkpoint = torch.load(filepath, map_location=device)
        for pair, agent_data in checkpoint['agents'].items():
            state_size = agent_data['q_network']['feature_extractor.0.weight'].shape[1]
            agent = ForexTradingAgent(state_size)
            agent.q_network.load_state_dict(agent_data['q_network'])
            agent.target_network.load_state_dict(agent_data['target_network'])
            agent.epsilon = agent_data['epsilon']
            # Ensure models are on the correct device
            agent.q_network.to(device)
            agent.target_network.to(device)
            self.agents[pair] = agent

# Example usage
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Forex AI Trading Model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs (default: 100)')
    parser.add_argument('--period', type=str, default='1y', help='Data period (default: 1y)')
    parser.add_argument('--interval', type=str, default='1h', help='Data interval (default: 1h)')
    parser.add_argument('--save-dir', type=str, default='models', help='Directory to save models (default: models)')
    parser.add_argument('--pairs', type=str, nargs='+', default=['EURUSD', 'GBPUSD', 'USDJPY'], 
                       help='Currency pairs to train (default: EURUSD GBPUSD USDJPY)')
    parser.add_argument('--no-gpu', action='store_true', help='Force CPU usage even if GPU is available')
    
    args = parser.parse_args()
    
    # Override device if --no-gpu is specified
    if args.no_gpu:
        device = torch.device('cpu')
        print("🔄 Forced CPU usage (--no-gpu flag)")
    
    print("="*60)
    print("🚀 Forex AI Trading Model Training")
    print("="*60)
    print(f"🖥️  Device: {device}")
    print(f"📊 Data period: {args.period}")
    print(f"⏰ Data interval: {args.interval}")
    print(f"🔄 Training epochs: {args.epochs}")
    print(f"💰 Currency pairs: {', '.join(args.pairs)}")
    print(f"💾 Save directory: {args.save_dir}")
    print("="*60)
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    try:
        # Load forex data
        print("\n📈 Loading forex data...")
        all_data = load_forex_data(period=args.period, interval=args.interval)
        
        if not all_data:
            print("❌ No data available for training. Exiting.")
            exit(1)
        
        # Filter data to only include requested pairs
        data = {pair: df for pair, df in all_data.items() if pair in args.pairs}
        
        if not data:
            print(f"❌ None of the requested pairs {args.pairs} are available.")
            print(f"Available pairs: {list(all_data.keys())}")
            exit(1)
        
        print(f"\n✅ Data loaded successfully for {len(data)} pairs")
        for pair, df in data.items():
            print(f"   • {pair}: {len(df)} data points")
        
        # Initialize predictor with the actual pairs that have data
        predictor = ForexPredictor(pairs=list(data.keys()))
        print(f"\n🎯 Initialized predictor for pairs: {predictor.pairs}")
        
        # Train the models
        print(f"\n🎯 Starting training for {args.epochs} epochs...")
        predictor.train(data, epochs=args.epochs)
        
        # Save the trained models
        model_filename = f"forex_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        model_path = os.path.join(args.save_dir, model_filename)
        
        print(f"\n💾 Saving trained model to: {model_path}")
        predictor.save_model(model_path)
        
        print("\n" + "="*60)
        print("✅ Training completed successfully!")
        print(f"📁 Model saved as: {model_path}")
        print("="*60)
        
        # Test the trained model with a sample prediction
        print("\n🧪 Testing trained model...")
        for pair, df in data.items():
            try:
                # Use the last row for prediction
                test_state = predictor.prepare_state(df.iloc[-20:])
                action = predictor.predict(pair, test_state)
                print(f"   • {pair}: {action.direction} (confidence: {action.confidence:.3f})")
            except Exception as e:
                print(f"   • {pair}: Error in prediction - {str(e)}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\nForex AI Trading Model training session ended.")