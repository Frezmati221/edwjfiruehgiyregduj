import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler  # Mixed precision for RTX 5080
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
import json
import logging
from sklearn.model_selection import TimeSeriesSplit
warnings.filterwarnings('ignore')

# GPU setup for MAXIMUM performance
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Enable MAXIMUM GPU optimizations for RTX 5080
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # Speed optimization
    torch.backends.cudnn.deterministic = False  # Allow fastest algorithms
    torch.cuda.empty_cache()
    
    # Set CUDA memory settings for maximum performance
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:2048,expandable_segments:True'
    
    print(f"🚀 GPU: {torch.cuda.get_device_name(0)} - MAXIMUM POWER MODE!")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("⚡ CUDA optimizations: MAXIMUM")
else:
    print("💻 Using CPU")

@dataclass
class TradingAction:
    """Trading action with position details"""
    direction: str  # 'long', 'short', or 'hold'
    take_profit: float
    stop_loss: float
    confidence: float

@dataclass
class MarketConditions:
    """Current market conditions for regime detection"""
    regime: str  # 'trending_up', 'trending_down', 'ranging', 'high_volatility'
    volatility_percentile: float
    trend_strength: float
    support_level: float
    resistance_level: float

class RiskManager:
    """Production-grade risk management system"""
    
    def __init__(self, 
                 max_drawdown: float = 0.20,
                 max_daily_loss: float = 0.06,
                 max_open_positions: int = 3,
                 max_correlation: float = 0.7):
        self.max_drawdown = max_drawdown
        self.max_daily_loss = max_daily_loss
        self.max_open_positions = max_open_positions
        self.max_correlation = max_correlation
        self.daily_losses = []
        self.peak_balance = 10000
        self.daily_start_balance = 10000
        self.open_positions = []
        
        # Position correlation matrix
        self.correlation_pairs = {
            'EURUSD': ['GBPUSD', 'AUDUSD'],
            'GBPUSD': ['EURUSD', 'AUDUSD'],
            'USDJPY': ['USDCHF', 'USDCAD'],
            'USDCHF': ['USDJPY', 'USDCAD'],
            'AUDUSD': ['EURUSD', 'GBPUSD'],
            'USDCAD': ['USDJPY', 'USDCHF']
        }
        
    def can_trade(self, current_balance: float, pair: str, direction: str) -> Tuple[bool, str]:
        """Check if trading is allowed based on risk limits"""
        # Check maximum drawdown
        if self.peak_balance > 0:
            current_drawdown = (self.peak_balance - current_balance) / self.peak_balance
            if current_drawdown >= self.max_drawdown:
                return False, f"Max drawdown exceeded: {current_drawdown:.2%}"
        
        # Check daily loss limit
        daily_pnl = current_balance - self.daily_start_balance
        if daily_pnl <= -self.max_daily_loss * self.peak_balance:
            return False, f"Daily loss limit exceeded: {daily_pnl:.2f}"
        
        # Check maximum open positions
        if len(self.open_positions) >= self.max_open_positions:
            return False, f"Max open positions reached: {len(self.open_positions)}"
        
        # Check correlation limits
        if self._check_correlation_risk(pair, direction):
            return False, f"Correlation risk too high for {pair} {direction}"
        
        # Update peak balance
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
        
        return True, "OK"
    
    def _check_correlation_risk(self, pair: str, direction: str) -> bool:
        """Check if new position would create excessive correlation risk"""
        if pair not in self.correlation_pairs:
            return False
        
        correlated_pairs = self.correlation_pairs[pair]
        same_direction_count = 0
        
        for pos in self.open_positions:
            if pos['pair'] in correlated_pairs and pos['direction'] == direction:
                same_direction_count += 1
        
        # Limit correlated positions in same direction
        return same_direction_count >= 2
    
    def add_position(self, pair: str, direction: str, size: float, entry_price: float):
        """Add new position to tracking"""
        self.open_positions.append({
            'pair': pair,
            'direction': direction,
            'size': size,
            'entry_price': entry_price,
            'timestamp': datetime.now()
        })
    
    def remove_position(self, pair: str):
        """Remove position from tracking"""
        self.open_positions = [pos for pos in self.open_positions if pos['pair'] != pair]
    
    def reset_daily(self, current_balance: float):
        """Reset daily tracking"""
        self.daily_start_balance = current_balance
        self.daily_losses = []

class TradingCostCalculator:
    """Calculate realistic trading costs including spread, slippage, and commission"""
    
    def __init__(self):
        # Typical spreads in pips for major pairs during London/NY session
        self.spreads = {
            'EURUSD': 0.8,
            'GBPUSD': 1.2,
            'USDJPY': 0.9,
            'USDCHF': 1.5,
            'AUDUSD': 1.0,
            'USDCAD': 1.3,
            'NZDUSD': 1.8,
            'EURJPY': 1.4,
            'GBPJPY': 2.1,
            'CHFJPY': 2.5
        }
        
        # Commission per lot (round turn)
        self.commission_per_lot = 7.0  # USD
        
        # Slippage factors
        self.base_slippage = 0.3  # pips
        self.volatility_multiplier = 1.5
        
    def calculate_entry_cost(self, pair: str, price: float, volatility: float, 
                           position_size: float, market_conditions: MarketConditions) -> float:
        """Calculate total cost to enter position"""
        pip_value = self._get_pip_value(pair, price)
        
        # Base spread
        base_spread = self.spreads.get(pair, 1.5)
        
        # Adjust spread for market conditions
        if market_conditions.regime == 'high_volatility':
            spread = base_spread * 2.0  # Spreads widen during high volatility
        elif market_conditions.volatility_percentile > 0.8:
            spread = base_spread * 1.5
        else:
            spread = base_spread
        
        # Calculate slippage
        slippage = self.base_slippage + (volatility * self.volatility_multiplier)
        
        # Total pip cost
        total_pip_cost = spread + slippage
        
        # Convert to dollar cost
        pip_cost = total_pip_cost * pip_value * (position_size / 100000)  # per standard lot
        
        # Add commission
        lots = position_size / 100000
        commission = self.commission_per_lot * lots
        
        return pip_cost + commission
    
    def _get_pip_value(self, pair: str, price: float) -> float:
        """Get pip value for currency pair"""
        if 'JPY' in pair:
            return 0.01
        else:
            return 0.0001

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
        df['sma_50'] = talib.SMA(df['close'], timeperiod=50)  # Added for regime detection
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
        
        # Market regime indicators
        df['volatility_percentile'] = df['volatility'].rolling(window=100).rank(pct=True)
        df['trend_strength'] = abs(df['sma_20'] - df['sma_50']) / df['atr']
        
        # Fill any NaN values
        df = df.bfill().ffill()
        
        return df
    
    def detect_market_regime(self, df: pd.DataFrame) -> MarketConditions:
        """Detect current market regime for adaptive strategy"""
        if len(df) < 50:
            return MarketConditions('ranging', 0.5, 0.0, df['low'].min(), df['high'].max())
        
        current_price = df['close'].iloc[-1]
        atr_current = df['atr'].iloc[-1]
        atr_avg = df['atr'].tail(20).mean()
        volatility_percentile = df['volatility_percentile'].iloc[-1]
        
        sma_20 = df['sma_20'].iloc[-1]
        sma_50 = df['sma_50'].iloc[-1] if 'sma_50' in df else sma_20
        
        # Calculate trend strength
        trend_strength = abs(sma_20 - sma_50) / atr_current if atr_current > 0 else 0
        
        # Support and resistance levels
        recent_lows = df['low'].tail(50).min()
        recent_highs = df['high'].tail(50).max()
        
        # Determine regime
        if atr_current > atr_avg * 1.8:
            regime = 'high_volatility'
        elif trend_strength > 2.0:
            if current_price > sma_20 > sma_50:
                regime = 'trending_up'
            elif current_price < sma_20 < sma_50:
                regime = 'trending_down'
            else:
                regime = 'ranging'
        else:
            regime = 'ranging'
        
        return MarketConditions(
            regime=regime,
            volatility_percentile=volatility_percentile,
            trend_strength=trend_strength,
            support_level=recent_lows,
            resistance_level=recent_highs
        )

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

class ForexEnvironment:
    """Production-grade trading environment with realistic costs and risk management"""
    
    def __init__(self, data: pd.DataFrame, pair_name: str = 'EURUSD', lookback_period: int = 30):
        self.data = data
        self.pair_name = pair_name
        self.lookback_period = lookback_period
        self.current_step = lookback_period
        self.max_steps = len(data) - 1
        self.position = None
        self.entry_price = 0
        self.balance = 10000
        self.initial_balance = 10000
        self.trade_history = []
        
        # Production components
        self.risk_manager = RiskManager()
        self.cost_calculator = TradingCostCalculator()
        self.indicators = ForexIndicators()
        
        # Performance tracking
        self.equity_curve = [self.initial_balance]
        self.drawdown_series = []
        self.trade_count = 0
        self.winning_trades = 0
        
        # Market conditions
        self.market_conditions = None
        
    def get_pip_value(self, price: float) -> float:
        """Get pip value for CFD contract based on instrument type"""
        if 'JPY' in self.pair_name:
            return 0.01  # JPY pairs: 1 pip = 0.01
        else:
            return 0.0001  # Major pairs: 1 pip = 0.0001
        
    def calculate_position_size(self, price: float, stop_loss_pips: float, risk_percent: float = 0.02) -> float:
        """Calculate position size based on risk management"""
        risk_amount = self.balance * risk_percent
        pip_value = self.get_pip_value(price)
        
        # Position size = Risk Amount / (Stop Loss in Pips * Pip Value per Unit)
        # For standard lot (100,000 units), pip value is already calculated
        position_size = risk_amount / (stop_loss_pips * pip_value * 100000)
        
        # Convert to units (1 standard lot = 100,000 units)
        position_units = position_size * 100000
        
        # Apply maximum position limits
        max_position = self.balance * 10  # 10:1 leverage maximum
        return min(position_units, max_position)
        
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = self.lookback_period
        self.position = None
        self.entry_price = 0
        self.balance = self.initial_balance
        self.trade_history = []
        self.equity_curve = [self.initial_balance]
        self.drawdown_series = []
        self.trade_count = 0
        self.winning_trades = 0
        self.risk_manager = RiskManager()
        return self._get_state()
    
    def _get_state(self):
        """Get current state including lookback window and market conditions"""
        lookback_data = self.data.iloc[
            self.current_step - self.lookback_period:self.current_step
        ]
        
        # Update market conditions
        self.market_conditions = self.indicators.detect_market_regime(lookback_data)
        
        # Normalize features
        features = []
        for col in lookback_data.columns:
            if col not in ['open', 'high', 'low', 'close', 'volume']:
                values = lookback_data[col].values
                if not np.isnan(values).all():
                    normalized = (values - np.nanmean(values)) / (np.nanstd(values) + 1e-8)
                    features.append(normalized)
        
        # Add market regime features
        regime_features = self._encode_market_regime()
        features.append(regime_features)
        
        return np.array(features).flatten()
    
    def _encode_market_regime(self) -> np.ndarray:
        """Encode market regime as features"""
        if self.market_conditions is None:
            return np.zeros(6)
        
        regime_encoding = {
            'trending_up': [1, 0, 0, 0],
            'trending_down': [0, 1, 0, 0],
            'ranging': [0, 0, 1, 0],
            'high_volatility': [0, 0, 0, 1]
        }
        
        regime_vector = regime_encoding.get(self.market_conditions.regime, [0, 0, 0, 0])
        additional_features = [
            self.market_conditions.volatility_percentile,
            self.market_conditions.trend_strength
        ]
        
        return np.array(regime_vector + additional_features)
    
    def step(self, action: TradingAction):
        """Execute trading action with realistic costs and risk management"""
        current_price = self.data.iloc[self.current_step]['close']
        current_volatility = self.data.iloc[self.current_step]['volatility']
        reward = 0
        
        # Check if risk manager allows trading
        can_trade, risk_message = self.risk_manager.can_trade(
            self.balance, self.pair_name, action.direction
        )
        
        # Close existing position if any
        if self.position:
            if self.position == 'long':
                price_change = current_price - self.entry_price
            else:  # short
                price_change = self.entry_price - current_price
            
            # Calculate pips gained/lost
            pip_value = self.get_pip_value(current_price)
            pips_gained = price_change / pip_value
            
            # Calculate position size for this trade
            position_size = self.calculate_position_size(
                self.entry_price, 
                action.stop_loss,
                risk_percent=0.01 if self.market_conditions.regime == 'high_volatility' else 0.02
            )
            
            # Apply realistic stop loss and take profit with slippage
            exit_cost = self.cost_calculator.calculate_entry_cost(
                self.pair_name, current_price, current_volatility, 
                position_size, self.market_conditions
            )
            
            # Check for stop loss or take profit hits
            if pips_gained <= -action.stop_loss:  # Stop loss hit
                # Add slippage to stop loss
                slippage_pips = max(1, current_volatility * 100)  # More slippage in volatile markets
                actual_loss_pips = action.stop_loss + slippage_pips
                actual_profit = -(actual_loss_pips * pip_value * position_size / 100000) - exit_cost
                reward = -500  # Heavy penalty for stop loss
                
            elif pips_gained >= action.take_profit:  # Take profit hit
                actual_profit = (action.take_profit * pip_value * position_size / 100000) - exit_cost
                reward = 200  # Reward for take profit
                self.winning_trades += 1
                
            else:
                # Regular exit with costs
                actual_profit = (pips_gained * pip_value * position_size / 100000) - exit_cost
                reward = pips_gained * 1  # Reduced reward for realism
                if actual_profit > 0:
                    self.winning_trades += 1
            
            self.balance += actual_profit
            self.trade_count += 1
            
            # Remove position from risk manager
            self.risk_manager.remove_position(self.pair_name)
            self.position = None
            
            # Record trade
            self.trade_history.append({
                'step': self.current_step,
                'action': 'close',
                'price': current_price,
                'pnl': actual_profit,
                'pips': pips_gained,
                'balance': self.balance
            })
        
        # Open new position if allowed
        if action.direction != 'hold' and can_trade:
            # Calculate entry costs
            position_size = self.calculate_position_size(
                current_price, 
                action.stop_loss,
                risk_percent=0.01 if self.market_conditions.regime == 'high_volatility' else 0.02
            )
            
            entry_cost = self.cost_calculator.calculate_entry_cost(
                self.pair_name, current_price, current_volatility, 
                position_size, self.market_conditions
            )
            
            # Deduct entry costs immediately
            self.balance -= entry_cost
            
            # Open position
            self.position = action.direction
            self.entry_price = current_price
            
            # Add to risk manager
            self.risk_manager.add_position(
                self.pair_name, action.direction, position_size, current_price
            )
            
            self.trade_history.append({
                'step': self.current_step,
                'action': action.direction,
                'price': current_price,
                'tp': action.take_profit,
                'sl': action.stop_loss,
                'cost': entry_cost,
                'size': position_size
            })
            
        elif action.direction != 'hold' and not can_trade:
            # Penalty for trying to trade when not allowed
            reward -= 100
        
        # Update performance tracking
        self.equity_curve.append(self.balance)
        
        # Calculate drawdown
        peak = max(self.equity_curve)
        drawdown = (peak - self.balance) / peak if peak > 0 else 0
        self.drawdown_series.append(drawdown)
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # Additional penalties for poor risk management
        if self.balance <= 0:
            reward -= 2000  # Severe penalty for blowing account
            done = True
        elif drawdown > 0.15:  # 15% drawdown warning
            reward -= 50
        
        # Reward for maintaining good win rate
        if self.trade_count > 10:
            win_rate = self.winning_trades / self.trade_count
            if win_rate > 0.6:
                reward += 10
            elif win_rate < 0.4:
                reward -= 10
        
        return self._get_state(), reward, done
    
    def get_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        if self.trade_count == 0:
            return {'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 'win_rate': 0}
        
        # Calculate returns
        returns = np.diff(self.equity_curve) / np.array(self.equity_curve[:-1])
        
        total_return = (self.balance - self.initial_balance) / self.initial_balance
        
        # Sharpe ratio (assuming 252 trading days per year)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        max_drawdown = max(self.drawdown_series) if self.drawdown_series else 0
        win_rate = self.winning_trades / self.trade_count if self.trade_count > 0 else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': self.trade_count,
            'final_balance': self.balance
        }

class DQNNetwork(nn.Module):
    """Simple and fast Deep Q-Network for trading decisions"""
    
    def __init__(self, input_size: int, hidden_size: int = 512):  # Bigger network for RTX 5080
        super(DQNNetwork, self).__init__()
        
        # BIGGER network to utilize RTX 5080 power
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size * 4),  # 2048 neurons
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 4, hidden_size * 2),  # 1024 neurons
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 2, hidden_size),  # 512 neurons
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),  # 256 neurons
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
    """Simple and fast trading agent with DQN"""
    
    def __init__(self, state_size: int, learning_rate: float = 0.001):
        self.state_size = state_size
        # MASSIVE memory for RTX 5080 - USE ALL THE VRAM!
        if device.type == 'cuda':
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_memory_gb >= 16:  # RTX 5080
                self.memory = deque(maxlen=100000)  # 10x larger memory buffer
            elif gpu_memory_gb >= 12:
                self.memory = deque(maxlen=50000)
            else:
                self.memory = deque(maxlen=20000)
        else:
            self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.gamma = 0.95
        
        # AGGRESSIVE batch size for RTX 5080 - USE ALL THE POWER!
        if device.type == 'cuda':
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_memory_gb >= 16:  # RTX 5080 has 16GB
                self.batch_size = 2048  # MASSIVE batch for speed
            elif gpu_memory_gb >= 12:
                self.batch_size = 1024
            else:
                self.batch_size = 512
        else:
            self.batch_size = 32
        
        # Neural networks - move to GPU for speed
        self.q_network = DQNNetwork(state_size).to(device)
        self.target_network = DQNNetwork(state_size).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # Mixed precision for RTX 5080 SPEED!
        self.scaler = GradScaler() if device.type == 'cuda' else None
        
        self.update_target_network()
        
        print(f"🧠 MASSIVE neural network initialized on {device}")
        print(f"📊 AGGRESSIVE batch size: {self.batch_size} (RTX 5080 POWER!)")
        print(f"💾 HUGE memory buffer: {len(self.memory):,}")
        if self.scaler:
            print("⚡ Mixed precision training: ENABLED")
        print("🚀 MAXIMUM GPU UTILIZATION MODE!")
        
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray) -> TradingAction:
        """Choose action using epsilon-greedy policy"""
        if random.random() <= self.epsilon:
            # Random action
            direction = random.choice(['long', 'short', 'hold'])
            tp = random.uniform(20, 100)  # pips
            sl = random.uniform(10, 50)   # pips
        else:
            # Predict using network - OPTIMIZED FOR RTX 5080
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device, non_blocking=True)
            with torch.no_grad():
                direction_q, tp_q, sl_q = self.q_network(state_tensor)
            
            direction_idx = torch.argmax(direction_q).item()
            direction = ['long', 'short', 'hold'][direction_idx]
            
            tp_idx = torch.argmax(tp_q).item()
            sl_idx = torch.argmax(sl_q).item()
            
            tp = 20 + tp_idx * 10  # 20-120 pips
            sl = 10 + sl_idx * 5   # 10-60 pips
        
        confidence = 1.0 - self.epsilon
        return TradingAction(direction, tp, sl, confidence)
    
    def replay(self):
        """Train the network on batch of experiences - GPU optimized"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        
        # GPU-optimized tensor creation with pin_memory for RTX 5080
        states = torch.FloatTensor([e[0] for e in batch]).to(device, non_blocking=True)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(device, non_blocking=True)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(device, non_blocking=True)
        dones = torch.FloatTensor([e[4] for e in batch]).to(device, non_blocking=True)
        
        # Simple action encoding for speed
        actions = []
        for e in batch:
            if e[1].direction == 'long':
                actions.append(0)
            elif e[1].direction == 'short':
                actions.append(1)
            else:
                actions.append(2)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device, non_blocking=True)
        
        current_q_values = self.q_network(states)
        next_q_values = self.target_network(next_states)
        
        # Calculate target Q-values
        target_q_values = rewards + (1 - dones) * self.gamma * torch.max(next_q_values[0], dim=1)[0]
        
        # Calculate loss
        current_q = current_q_values[0].gather(1, actions).squeeze()
        
        # MIXED PRECISION TRAINING for RTX 5080 speed
        if self.scaler:
            with autocast():
                loss = nn.MSELoss()(current_q, target_q_values.detach())
            
            # Backpropagation with mixed precision
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss = nn.MSELoss()(current_q, target_q_values.detach())
            
            # Regular backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class ForexPredictor:
    """Production-grade forex prediction system with walk-forward optimization"""
    
    def __init__(self, pairs: List[str] = ['EURUSD', 'GBPUSD', 'USDJPY']):
        self.pairs = pairs
        self.agents = {}
        self.indicators = ForexIndicators()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.validation_results = {}
        
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data with all indicators"""
        return self.indicators.calculate_all_indicators(df)
    
    def walk_forward_optimization(self, data: Dict[str, pd.DataFrame], 
                                 train_periods: int = 1000,  # Training window
                                 test_periods: int = 200,     # Test window
                                 step_size: int = 100,        # Step size
                                 epochs_per_window: int = 50) -> Dict:
        """Implement walk-forward optimization for robust validation"""
        
        self.logger.info("Starting walk-forward optimization...")
        results = {}
        
        for pair in self.pairs:
            if pair not in data:
                continue
                
            self.logger.info(f"Walk-forward optimization for {pair}")
            df = data[pair]
            pair_results = []
            
            # Ensure we have enough data
            if len(df) < train_periods + test_periods:
                self.logger.warning(f"Insufficient data for {pair}: {len(df)} < {train_periods + test_periods}")
                continue
            
            # Walk-forward windows
            for i in range(0, len(df) - train_periods - test_periods, step_size):
                train_start = i
                train_end = i + train_periods
                test_start = train_end
                test_end = test_start + test_periods
                
                self.logger.info(f"Training window: {train_start}-{train_end}, Test: {test_start}-{test_end}")
                
                # Prepare training data
                train_data = df.iloc[train_start:train_end].copy()
                prepared_train_data = self.prepare_data(train_data)
                
                # Train model on this window
                env = ForexEnvironment(prepared_train_data, pair_name=pair)
                state = env.reset()
                agent = ForexTradingAgent(state_size=len(state))
                
                # Training with progress tracking
                train_rewards = []
                for epoch in range(epochs_per_window):
                    state = env.reset()
                    total_reward = 0
                    done = False
                    
                    while not done:
                        action = agent.act(state)
                        next_state, reward, done = env.step(action)
                        agent.remember(state, action, reward, next_state, done)
                        state = next_state
                        total_reward += reward
                        
                        if len(agent.memory) > agent.batch_size:
                            agent.replay()
                    
                    train_rewards.append(total_reward)
                    if epoch % 10 == 0:
                        agent.update_target_network()
                
                # Test on out-of-sample data
                test_data = df.iloc[test_start:test_end].copy()
                prepared_test_data = self.prepare_data(test_data)
                
                test_env = ForexEnvironment(prepared_test_data, pair_name=pair)
                agent.epsilon = 0  # No exploration during testing
                
                state = test_env.reset()
                test_reward = 0
                done = False
                
                while not done:
                    action = agent.act(state)
                    next_state, reward, done = test_env.step(action)
                    state = next_state
                    test_reward += reward
                
                # Get performance metrics
                test_metrics = test_env.get_performance_metrics()
                
                pair_results.append({
                    'train_window': (train_start, train_end),
                    'test_window': (test_start, test_end),
                    'train_reward_avg': np.mean(train_rewards[-10:]),
                    'test_reward': test_reward,
                    'test_metrics': test_metrics
                })
                
                self.logger.info(f"Window completed - Train: {np.mean(train_rewards[-10:]):.2f}, Test: {test_reward:.2f}")
            
            results[pair] = pair_results
            
            # Calculate overall performance for this pair
            if pair_results:
                avg_test_return = np.mean([r['test_metrics']['total_return'] for r in pair_results])
                avg_sharpe = np.mean([r['test_metrics']['sharpe_ratio'] for r in pair_results])
                avg_drawdown = np.mean([r['test_metrics']['max_drawdown'] for r in pair_results])
                
                self.logger.info(f"{pair} Summary - Return: {avg_test_return:.2%}, "
                               f"Sharpe: {avg_sharpe:.2f}, Drawdown: {avg_drawdown:.2%}")
        
        return results
    
    def train(self, data: Dict[str, pd.DataFrame], epochs: int = 100, 
             use_walk_forward: bool = True):
        """Train the model with optional walk-forward validation"""
        
        if use_walk_forward:
            self.logger.info("Using walk-forward optimization for training...")
            self.validation_results = self.walk_forward_optimization(data, epochs_per_window=epochs//2)
        
        # Final training on full dataset for each pair
        for pair_idx, pair in enumerate(self.pairs):
            if pair not in data:
                continue
            
            self.logger.info(f"Final training for {pair} ({pair_idx + 1}/{len(self.pairs)})...")
            prepared_data = self.prepare_data(data[pair])
            
            # Initialize environment and agent
            env = ForexEnvironment(prepared_data, pair_name=pair)
            state = env.reset()
            agent = ForexTradingAgent(state_size=len(state))
            
            # Training progress bar
            pbar = tqdm(range(epochs), desc=f"🔄 {pair}", 
                       bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} epochs [⏱️{elapsed}<⏳{remaining}, 💰{postfix}]')
            
            best_sharpe = float('-inf')
            best_metrics = None
            rewards_history = []
            
            for epoch in pbar:
                start_time = time.time()
                state = env.reset()
                total_reward = 0
                done = False
                steps = 0
                
                while not done:
                    action = agent.act(state)
                    next_state, reward, done = env.step(action)
                    agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    total_reward += reward
                    steps += 1
                    
                    # Training frequency based on GPU power
                    min_memory = agent.batch_size
                    if len(agent.memory) > min_memory:
                        agent.replay()
                        
                        # Additional training for powerful GPUs
                        if device.type == 'cuda' and steps % 5 == 0:
                            for _ in range(2):
                                if len(agent.memory) > agent.batch_size:
                                    agent.replay()
                
                # Update target network
                if epoch % 10 == 0:
                    agent.update_target_network()
                
                # Get performance metrics
                metrics = env.get_performance_metrics()
                rewards_history.append(total_reward)
                
                # Track best model by Sharpe ratio
                if metrics['sharpe_ratio'] > best_sharpe:
                    best_sharpe = metrics['sharpe_ratio']
                    best_metrics = metrics
                
                # Update progress bar
                epoch_time = time.time() - start_time
                recent_avg = np.mean(rewards_history[-10:]) if len(rewards_history) >= 10 else np.mean(rewards_history)
                
                postfix = (f"Reward: {total_reward:.0f}, Avg: {recent_avg:.0f}, "
                          f"Sharpe: {metrics['sharpe_ratio']:.2f}, "
                          f"WinRate: {metrics['win_rate']:.1%}, "
                          f"DD: {metrics['max_drawdown']:.1%}")
                pbar.set_postfix_str(postfix)
                
                # Detailed progress every 20 epochs
                if epoch % 20 == 0 and epoch > 0:
                    self.logger.info(f"\n    {pair} Epoch {epoch} - "
                                   f"Return: {metrics['total_return']:.2%}, "
                                   f"Sharpe: {metrics['sharpe_ratio']:.2f}, "
                                   f"Trades: {metrics['total_trades']}")
            
            pbar.close()
            self.logger.info(f"✅ {pair} training completed! Best Sharpe: {best_sharpe:.2f}")
            self.agents[pair] = agent
            
            # Log final performance
            if best_metrics:
                self.logger.info(f"   📊 Final metrics - Return: {best_metrics['total_return']:.2%}, "
                               f"Sharpe: {best_metrics['sharpe_ratio']:.2f}, "
                               f"Win Rate: {best_metrics['win_rate']:.1%}, "
                               f"Max DD: {best_metrics['max_drawdown']:.1%}")
    
    def predict(self, pair: str, current_data: pd.DataFrame) -> TradingAction:
        """Predict trading action for given pair"""
        if pair not in self.agents:
            raise ValueError(f"No trained model for {pair}")
        
        prepared_data = self.prepare_data(current_data)
        env = ForexEnvironment(prepared_data, pair_name=pair)
        state = env._get_state()
        
        # Set epsilon to 0 for prediction (no exploration)
        self.agents[pair].epsilon = 0
        action = self.agents[pair].act(state)
        
        return action
    
    def save_model(self, filepath: str):
        """Save trained models with metadata"""
        checkpoint = {
            'agents': {},
            'validation_results': self.validation_results,
            'timestamp': datetime.now().isoformat(),
            'pairs': self.pairs
        }
        
        for pair, agent in self.agents.items():
            checkpoint['agents'][pair] = {
                'q_network': agent.q_network.state_dict(),
                'target_network': agent.target_network.state_dict(),
                'epsilon': agent.epsilon
            }
        
        torch.save(checkpoint, filepath)
        
        # Also save validation results as JSON for analysis
        results_path = filepath.replace('.pkl', '_validation.json')
        with open(results_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_results = self._convert_for_json(self.validation_results)
            json.dump(json_results, f, indent=2)
        
        self.logger.info(f"Model saved to {filepath}")
        self.logger.info(f"Validation results saved to {results_path}")
    
    def _convert_for_json(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj
    
    def prepare_state(self, df):
        """Prepare state vector from dataframe for prediction"""
        if len(df) < 30:
            # Pad with last available values if not enough data
            last_row = df.iloc[-1]
            while len(df) < 30:
                df = pd.concat([df, last_row.to_frame().T], ignore_index=True)
        
        # Use last 30 rows and prepare state like the environment does
        recent_data = df.tail(30)
        
        # Prepare data with indicators
        prepared_data = self.prepare_data(recent_data)
        
        # Create a temporary environment to get the state
        temp_env = ForexEnvironment(prepared_data)
        state = temp_env._get_state()
        
        return state
    
    def load_model(self, filepath: str):
        """Load trained models with metadata"""
        checkpoint = torch.load(filepath, map_location=device)
        
        # Load validation results if available
        if 'validation_results' in checkpoint:
            self.validation_results = checkpoint['validation_results']
        
        for pair, agent_data in checkpoint['agents'].items():
            state_size = agent_data['q_network']['feature_extractor.0.weight'].shape[1]
            agent = ForexTradingAgent(state_size)
            agent.q_network.load_state_dict(agent_data['q_network'])
            agent.target_network.load_state_dict(agent_data['target_network'])
            agent.epsilon = agent_data['epsilon']
            
            # Ensure models are on correct device
            agent.q_network.to(device)
            agent.target_network.to(device)
            self.agents[pair] = agent
        
        self.logger.info(f"Model loaded from {filepath}")
        if 'timestamp' in checkpoint:
            self.logger.info(f"Model trained on: {checkpoint['timestamp']}")
    
    def get_validation_summary(self) -> Dict:
        """Get summary of walk-forward validation results"""
        if not self.validation_results:
            return {}
        
        summary = {}
        for pair, results in self.validation_results.items():
            if not results:
                continue
                
            returns = [r['test_metrics']['total_return'] for r in results]
            sharpes = [r['test_metrics']['sharpe_ratio'] for r in results]
            drawdowns = [r['test_metrics']['max_drawdown'] for r in results]
            win_rates = [r['test_metrics']['win_rate'] for r in results]
            
            summary[pair] = {
                'avg_return': np.mean(returns),
                'std_return': np.std(returns),
                'avg_sharpe': np.mean(sharpes),
                'avg_drawdown': np.mean(drawdowns),
                'avg_win_rate': np.mean(win_rates),
                'consistency': len([r for r in returns if r > 0]) / len(returns),
                'num_tests': len(results)
            }
        
        return summary

# Example usage
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Production-Grade Forex AI Trading Model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs (default: 100)')
    parser.add_argument('--period', type=str, default='2y', help='Data period (default: 2y for more data)')
    parser.add_argument('--interval', type=str, default='1h', help='Data interval (default: 1h)')
    parser.add_argument('--save-dir', type=str, default='models', help='Directory to save models (default: models)')
    parser.add_argument('--pairs', type=str, nargs='+', default=['EURUSD', 'GBPUSD', 'USDJPY'], 
                       help='Currency pairs to train (default: EURUSD GBPUSD USDJPY)')
    parser.add_argument('--walk-forward', action='store_true', default=True,
                       help='Use walk-forward optimization (default: True)')
    parser.add_argument('--validation-only', action='store_true', default=False,
                       help='Run only walk-forward validation without final training')
    
    args = parser.parse_args()
    
    print("="*80)
    print("🚀 PRODUCTION-GRADE FOREX AI TRAINING SYSTEM")
    print("="*80)
    print(f"📊 Data period: {args.period}")
    print(f"⏰ Data interval: {args.interval}")
    print(f"🔄 Training epochs: {args.epochs}")
    print(f"💰 Currency pairs: {', '.join(args.pairs)}")
    print(f"💾 Save directory: {args.save_dir}")
    print(f"🔬 Walk-forward validation: {'ENABLED' if args.walk_forward else 'DISABLED'}")
    if device.type == 'cuda':
        print(f"🚀 GPU acceleration: ENABLED ({torch.cuda.get_device_name(0)})")
    print("="*80)
    print("⚠️  PRODUCTION FEATURES ENABLED:")
    print("   • Realistic trading costs (spread, slippage, commission)")
    print("   • Advanced risk management system")
    print("   • Market regime detection")
    print("   • Walk-forward optimization")
    print("   • Comprehensive performance metrics")
    print("   • Position correlation limits")
    print("="*80)
    
    # Setup logging
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    try:
        # Load forex data
        logger.info("📈 Loading forex data...")
        all_data = load_forex_data(period=args.period, interval=args.interval)
        
        if not all_data:
            logger.error("❌ No data available for training. Exiting.")
            exit(1)
        
        # Filter data to only include requested pairs
        data = {pair: df for pair, df in all_data.items() if pair in args.pairs}
        
        if not data:
            logger.error(f"❌ None of the requested pairs {args.pairs} are available.")
            logger.info(f"Available pairs: {list(all_data.keys())}")
            exit(1)
        
        logger.info(f"✅ Data loaded successfully for {len(data)} pairs")
        for pair, df in data.items():
            logger.info(f"   • {pair}: {len(df)} data points ({df.index[0]} to {df.index[-1]})")
        
        # Initialize predictor
        predictor = ForexPredictor(pairs=list(data.keys()))
        logger.info(f"🎯 Initialized production-grade predictor for pairs: {predictor.pairs}")
        
        # Training
        logger.info(f"🎯 Starting {'validation-only' if args.validation_only else 'full training'} process...")
        start_total = time.time()
        
        if args.validation_only:
            # Run only walk-forward validation
            validation_results = predictor.walk_forward_optimization(data, epochs_per_window=args.epochs//2)
            
            # Print validation summary
            summary = predictor.get_validation_summary()
            logger.info("\n" + "="*60)
            logger.info("📊 WALK-FORWARD VALIDATION SUMMARY")
            logger.info("="*60)
            
            for pair, metrics in summary.items():
                logger.info(f"{pair}:")
                logger.info(f"  📈 Avg Return: {metrics['avg_return']:.2%} ± {metrics['std_return']:.2%}")
                logger.info(f"  📊 Avg Sharpe: {metrics['avg_sharpe']:.2f}")
                logger.info(f"  📉 Avg Drawdown: {metrics['avg_drawdown']:.2%}")
                logger.info(f"  🎯 Avg Win Rate: {metrics['avg_win_rate']:.1%}")
                logger.info(f"  ✅ Consistency: {metrics['consistency']:.1%} ({metrics['num_tests']} tests)")
                logger.info("")
        else:
            # Full training with optional validation
            predictor.train(data, epochs=args.epochs, use_walk_forward=args.walk_forward)
            
            total_time = time.time() - start_total
            
            # Save the trained models
            model_filename = f"production_forex_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            model_path = os.path.join(args.save_dir, model_filename)
            
            logger.info(f"💾 Saving production model to: {model_path}")
            predictor.save_model(model_path)
            
            # Print training summary
            logger.info("\n" + "="*80)
            logger.info("✅ PRODUCTION TRAINING COMPLETED SUCCESSFULLY!")
            logger.info("="*80)
            logger.info(f"📁 Model saved as: {model_path}")
            logger.info(f"⏱️  Total training time: {total_time/60:.1f} minutes")
            
            if args.walk_forward:
                summary = predictor.get_validation_summary()
                logger.info("\n📊 Walk-Forward Validation Results:")
                for pair, metrics in summary.items():
                    logger.info(f"   {pair}: Return {metrics['avg_return']:.2%}, "
                              f"Sharpe {metrics['avg_sharpe']:.2f}, "
                              f"Consistency {metrics['consistency']:.1%}")
            
            logger.info("="*80)
            
            # Test the trained model
            logger.info("🧪 Testing production model...")
            for pair, df in data.items():
                try:
                    test_state = predictor.prepare_state(df.iloc[-30:])
                    action = predictor.predict(pair, test_state)
                    logger.info(f"   • {pair}: {action.direction} "
                              f"(confidence: {action.confidence:.3f}, "
                              f"TP: {action.take_profit:.1f}, SL: {action.stop_loss:.1f})")
                except Exception as e:
                    logger.error(f"   • {pair}: Error in prediction - {str(e)}")
        
        logger.info("\n⚠️  IMPORTANT PRODUCTION REMINDERS:")
        logger.info("   1. 📝 Paper trade for 3-6 months before going live")
        logger.info("   2. 🔍 Monitor performance daily with strict risk limits")
        logger.info("   3. 🛑 Implement circuit breakers for extreme market conditions")
        logger.info("   4. 📊 Use tick data for final production testing")
        logger.info("   5. ⚖️  Ensure regulatory compliance in your jurisdiction")
        logger.info("   6. 💰 NEVER risk more than you can afford to lose")
        
    except KeyboardInterrupt:
        logger.warning("⚠️  Training interrupted by user")
    except Exception as e:
        logger.error(f"❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
    
    logger.info("Production-grade Forex AI training session ended.")
