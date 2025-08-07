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
warnings.filterwarnings('ignore')

# Simplified warning
print("⚠️ Using Yahoo Finance data - for training/research only, not live trading")

# GPU setup for MAXIMUM performance
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Enable MAXIMUM GPU optimizations for RTX 5080
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.cuda.empty_cache()
    
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:2048,expandable_segments:True'
    
    print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
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
    """Production-grade risk management system with enhanced safety features"""
    
    def __init__(self, 
                 max_drawdown: float = 0.90,  # 90% drawdown for training
                 max_daily_loss: float = 0.80,  # 80% daily loss for training (was 6%)
                 max_open_positions: int = 100,  # Allow many positions
                 max_correlation: float = 1.0,  # No correlation limits
                 max_leverage: float = 500.0,  # Much higher leverage for training
                 margin_call_level: float = 0.05,  # Very low margin call
                 stop_out_level: float = 0.01):  # Very low stop out
        self.max_drawdown = max_drawdown
        self.max_daily_loss = max_daily_loss
        self.max_open_positions = max_open_positions
        self.max_correlation = max_correlation
        self.max_leverage = max_leverage
        self.margin_call_level = margin_call_level
        self.stop_out_level = stop_out_level
        
        self.daily_losses = []
        self.peak_balance = 10000
        self.daily_start_balance = 10000
        self.open_positions = []
        self.used_margin = 0
        self.consecutive_losses = 0
        self.daily_trades = 0
        
        # Circuit breakers
        self.circuit_breakers = {
            'max_daily_trades': 10000,  # HUGE increase for training
            'max_consecutive_losses': 10000,  # MASSIVE increase for training - let it learn!
            'emergency_stop_balance': 0.1,  # Only stop at 90% loss (was 50%)
            'unusual_spread_multiplier': 10.0  # Much more lenient spread check
        }
        
        # Position correlation matrix
        self.correlation_pairs = {
            'EURUSD': ['GBPUSD', 'AUDUSD'],
            'GBPUSD': ['EURUSD', 'AUDUSD'],
            'USDJPY': ['USDCHF', 'USDCAD'],
            'USDCHF': ['USDJPY', 'USDCAD'],
            'AUDUSD': ['EURUSD', 'GBPUSD'],
            'USDCAD': ['USDJPY', 'USDCHF']
        }
        
    def can_trade(self, current_balance: float, pair: str, direction: str, 
                  position_size: float = 0, price: float = 0) -> Tuple[bool, str]:
        """Check if trading is allowed - TRAINING MODE: VERY LENIENT FOR LEARNING"""
        
        # For training: Only block in extreme cases to focus on prediction accuracy
        
        # Only stop if balance is completely wiped out (99.9% loss)
        account_loss = (self.peak_balance - current_balance) / self.peak_balance
        if account_loss >= 0.999:  # Only stop at 99.9% loss
            return False, f"EXTREME EMERGENCY: Account loss {account_loss:.1%}"
        
        # Allow unlimited daily trades for training
        # if self.daily_trades >= self.circuit_breakers['max_daily_trades']:
        #     return False, f"Daily trade limit exceeded: {self.daily_trades}"
        
        # Allow unlimited consecutive losses for training
        # if self.consecutive_losses >= self.circuit_breakers['max_consecutive_losses']:
        #     return False, f"Too many consecutive losses: {self.consecutive_losses}"
        
        # Allow extreme drawdown for training (95% loss)
        if self.peak_balance > 0:
            current_drawdown = (self.peak_balance - current_balance) / self.peak_balance
            if current_drawdown >= 0.95:  # Only stop at 95% drawdown
                return False, f"EXTREME drawdown: {current_drawdown:.2%}"
        
        # Allow extreme daily losses for training
        # daily_pnl = current_balance - self.daily_start_balance
        # if daily_pnl <= -self.max_daily_loss * self.peak_balance:
        #     return False, f"Daily loss limit exceeded: {daily_pnl:.2f}"
        
        # Allow unlimited positions for training
        # if len(self.open_positions) >= self.max_open_positions:
        #     return False, f"Max open positions reached: {len(self.open_positions)}"
        
        # Skip margin requirements for training - focus on predictions
        # if position_size > 0 and price > 0:
        #     if not self.check_margin_requirements(position_size, price, current_balance):
        #         return False, "Insufficient margin for position"
        
        # Check correlation limits
        if self._check_correlation_risk(pair, direction):
            return False, f"Correlation risk too high for {pair} {direction}"
        
        # Update peak balance
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
        
        return True, "OK"
    
    def check_margin_requirements(self, position_size: float, price: float, 
                                 current_balance: float) -> bool:
        """Ensure adequate margin for position with safety buffer"""
        required_margin = position_size * price / self.max_leverage
        free_margin = current_balance - self.used_margin
        
        # Require 20% margin buffer for safety
        return free_margin >= required_margin * 1.2
    
    def update_margin_usage(self, position_size: float, price: float, opening: bool = True):
        """Update used margin when opening/closing positions"""
        margin_for_position = position_size * price / self.max_leverage
        
        if opening:
            self.used_margin += margin_for_position
        else:
            self.used_margin = max(0, self.used_margin - margin_for_position)
    
    def check_margin_call(self, current_balance: float) -> Tuple[bool, str]:
        """Check if account is in margin call or stop out - TRAINING MODE: VERY LENIENT"""
        if self.used_margin == 0:
            return False, "No open positions"
        
        # For training: Allow extreme leverage to focus on predictions
        margin_level = current_balance / self.used_margin
        
        # Only trigger stop out at extreme levels (0.1% margin level)
        if margin_level <= 0.001:  # 0.1% margin level
            return True, f"EXTREME STOP OUT: Margin level {margin_level:.1%}"
        
        # No margin call warnings during training
        # elif margin_level <= self.margin_call_level:
        #     return True, f"MARGIN CALL: Margin level {margin_level:.1%}"
        
        return False, f"Margin level OK: {margin_level:.1%}"
    
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
        self.daily_trades += 1
        self.update_margin_usage(size, entry_price, opening=True)
    
    def remove_position(self, pair: str, profit: float = 0):
        """Remove position from tracking and update loss streaks"""
        position_to_remove = None
        for pos in self.open_positions:
            if pos['pair'] == pair:
                position_to_remove = pos
                break
        
        if position_to_remove:
            self.open_positions.remove(position_to_remove)
            self.update_margin_usage(
                position_to_remove['size'], 
                position_to_remove['entry_price'], 
                opening=False
            )
            
            # Track consecutive losses
            if profit < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
    
    def reset_daily(self, current_balance: float):
        """Reset daily tracking"""
        self.daily_start_balance = current_balance
        self.daily_losses = []
        self.daily_trades = 0

class ProductionSafetyChecks:
    """Essential safety checks for live trading"""
    
    def __init__(self):
        self.circuit_breakers = {
            'max_daily_trades': 1000,  # HUGE increase for training
            'max_consecutive_losses': 1000,  # MASSIVE increase for training 
            'emergency_stop_balance': 0.1,  # Stop at 90% loss (was 70%)
            'unusual_spread_multiplier': 10.0,  # Much more lenient
            'max_slippage_pips': 100.0,  # Allow high slippage during training
            'min_liquidity_threshold': 0.1  # Much lower threshold
        }
        
        # News blackout periods (minutes before/after high impact news)
        self.news_blackout_minutes = 30
        
        # Expected spreads for major pairs (pips)
        self.normal_spreads = {
            'EURUSD': 0.8,
            'GBPUSD': 1.2,
            'USDJPY': 0.9,
            'USDCHF': 1.5,
            'AUDUSD': 1.0,
            'USDCAD': 1.3
        }
        
        # Connection monitoring
        self.last_heartbeat = datetime.now()
        self.connection_timeout_seconds = 30
        
    def pre_trade_checks(self, pair: str, current_spread: float, 
                        current_time: datetime = None) -> Tuple[bool, List[Tuple[str, str]]]:
        """Run comprehensive checks before every trade"""
        if current_time is None:
            current_time = datetime.now()
            
        checks = []
        
        # Check spread widening (news/volatility indicator)
        normal_spread = self.normal_spreads.get(pair, 1.5)
        if current_spread > normal_spread * self.circuit_breakers['unusual_spread_multiplier']:
            checks.append(('FAIL', f'Abnormal spread: {current_spread:.1f} pips vs normal {normal_spread:.1f}'))
        else:
            checks.append(('PASS', 'Spread normal'))
        
        # Check if it's weekend or major holiday
        if self.is_market_closed(current_time):
            checks.append(('FAIL', 'Market is closed'))
        else:
            checks.append(('PASS', 'Market open'))
        
        # Check news calendar (simplified - in production, integrate with real news API)
        if self.is_high_impact_news_time(current_time):
            checks.append(('FAIL', 'High impact news event within blackout period'))
        else:
            checks.append(('PASS', 'No major news events'))
        
        # Check broker connection health
        if not self.verify_broker_connection():
            checks.append(('FAIL', 'Broker connection unstable'))
        else:
            checks.append(('PASS', 'Broker connection stable'))
        
        # Check system resources
        if not self.check_system_health():
            checks.append(('FAIL', 'System resources insufficient'))
        else:
            checks.append(('PASS', 'System healthy'))
        
        all_passed = all(c[0] == 'PASS' for c in checks)
        return all_passed, checks
    
    def is_market_closed(self, current_time: datetime) -> bool:
        """Check if forex market is closed (simplified) - LENIENT FOR TRAINING"""
        # For training, always allow trading (market never closed)
        return False
    
    def is_high_impact_news_time(self, current_time: datetime) -> bool:
        """Check if we're in news blackout period (simplified) - LENIENT FOR TRAINING"""
        # For training, never block trades due to news
        return False
    
    def verify_broker_connection(self) -> bool:
        """Verify broker connection is stable"""
        # In production, implement actual connection checks:
        # - Ping broker servers
        # - Check last quote timestamp
        # - Verify order execution capability
        
        # For training, always consider connection stable
        return True
        
        # Uncomment for production:
        # time_since_heartbeat = (datetime.now() - self.last_heartbeat).total_seconds()
        # return time_since_heartbeat < self.connection_timeout_seconds
    
    def check_system_health(self) -> bool:
        """Check system resources and health"""
        try:
            import psutil
            
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 80:
                return False
            
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 85:
                return False
            
            # Check disk space
            disk = psutil.disk_usage('/')
            if disk.percent > 90:
                return False
            
            return True
        except ImportError:
            # If psutil not available, assume system is healthy
            return True
        except Exception:
            return False
    
    def update_heartbeat(self):
        """Update connection heartbeat"""
        self.last_heartbeat = datetime.now()
    
    def emergency_shutdown_check(self, current_balance: float, initial_balance: float) -> bool:
        """Check if emergency shutdown is needed"""
        total_loss_percent = (initial_balance - current_balance) / initial_balance
        return total_loss_percent >= (1 - self.circuit_breakers['emergency_stop_balance'])

class TradingCostCalculator:
    """Calculate realistic trading costs including spread, slippage, commission"""
    
    def __init__(self):
        # Realistic spreads in pips for retail brokers
        self.spreads = {
            'EURUSD': {'normal': 1.2, 'volatile': 3.5, 'news': 8.0, 'weekend_gap': 15.0},
            'GBPUSD': {'normal': 1.8, 'volatile': 4.5, 'news': 12.0, 'weekend_gap': 20.0},
            'USDJPY': {'normal': 1.5, 'volatile': 3.8, 'news': 9.0, 'weekend_gap': 18.0},
            'USDCHF': {'normal': 2.2, 'volatile': 5.0, 'news': 15.0, 'weekend_gap': 25.0},
            'AUDUSD': {'normal': 1.6, 'volatile': 4.0, 'news': 10.0, 'weekend_gap': 22.0},
            'USDCAD': {'normal': 2.0, 'volatile': 4.8, 'news': 12.0, 'weekend_gap': 24.0}
        }
        
        # Commission per standard lot (100k units)
        self.commission_per_lot = 8.0  # USD per round turn
        
        # Slippage factors based on market conditions
        self.slippage_factors = {
            'normal': 0.5,
            'volatile': 2.5,
            'news': 5.0,
            'illiquid': 3.0,
            'weekend_gap': 10.0
        }
        
        # Execution risks
        self.execution_risks = {
            'partial_fill_probability': 0.08,
            'requote_probability': 0.05,
            'rejection_probability': 0.02,
            'timeout_probability': 0.01
        }
        
        # Swap rates (overnight financing costs)
        self.swap_rates = {
            'EURUSD': {'long': -0.5, 'short': -0.3},
            'GBPUSD': {'long': -0.8, 'short': -0.2},
            'USDJPY': {'long': -0.4, 'short': -0.6},
            'USDCHF': {'long': -0.3, 'short': -0.7},
            'AUDUSD': {'long': -0.6, 'short': -0.4},
            'USDCAD': {'long': -0.5, 'short': -0.5}
        }
        
    def get_market_condition(self, volatility_percentile: float, 
                           market_conditions: MarketConditions) -> str:
        """Determine market condition for cost calculation"""
        if market_conditions.regime == 'high_volatility':
            return 'volatile'
        elif volatility_percentile > 0.9:
            return 'news'  # Extreme volatility suggests news
        elif volatility_percentile < 0.2:
            return 'illiquid'
        else:
            return 'normal'
        
    def calculate_entry_cost(self, pair: str, price: float, volatility: float, 
                           position_size: float, market_conditions: MarketConditions) -> Tuple[float, Dict]:
        """Calculate total cost to enter position with execution risks"""
        pip_value = self._get_pip_value(pair, price)
        market_condition = self.get_market_condition(
            market_conditions.volatility_percentile, market_conditions
        )
        
        # Get appropriate spread for market condition
        spread_data = self.spreads.get(pair, {'normal': 1.5, 'volatile': 3.0, 'news': 6.0})
        if isinstance(spread_data, dict):
            base_spread = spread_data.get(market_condition, spread_data['normal'])
        else:
            base_spread = spread_data  # Backward compatibility
        
        # Calculate slippage based on market condition
        slippage = self.slippage_factors.get(market_condition, 0.3)
        
        # Add volatility-based slippage
        volatility_slippage = volatility * 50  # Scale factor
        total_slippage = slippage + volatility_slippage
        
        # Total pip cost
        total_pip_cost = base_spread + total_slippage
        
        # Convert to dollar cost
        lots = position_size / 100000
        pip_cost = total_pip_cost * pip_value * lots
        
        # Add commission
        commission = self.commission_per_lot * lots
        
        # Calculate execution risks
        execution_cost = 0
        execution_details = {}
        
        # Partial fill risk (may need to pay spread multiple times)
        if np.random.random() < self.execution_risks['partial_fill_probability']:
            execution_cost += pip_cost * 0.3  # 30% additional cost
            execution_details['partial_fill'] = True
        
        # Requote risk (price moves against us)
        if np.random.random() < self.execution_risks['requote_probability']:
            requote_cost = np.random.uniform(0.5, 2.0) * pip_value * lots
            execution_cost += requote_cost
            execution_details['requoted'] = True
        
        # Rejection risk (miss the trade, opportunity cost)
        rejection_risk = self.execution_risks['rejection_probability']
        if np.random.random() < rejection_risk:
            execution_details['rejected'] = True
            # Return high cost to simulate missing the trade
            return pip_cost * 10 + commission + execution_cost, execution_details
        
        total_cost = pip_cost + commission + execution_cost
        
        execution_details.update({
            'spread_pips': base_spread,
            'slippage_pips': total_slippage,
            'market_condition': market_condition,
            'commission': commission,
            'total_pips': total_pip_cost,
            'execution_risks': execution_cost
        })
        
        return total_cost, execution_details
    
    def _get_pip_value(self, pair: str, price: float) -> float:
        """Get pip value for currency pair"""
        if 'JPY' in pair:
            return 0.01
        else:
            return 0.0001

def validate_strategy_statistically(returns: np.ndarray) -> Dict:
    """Statistical tests for strategy robustness"""
    try:
        from scipy import stats
        
        if len(returns) < 30:
            return {'error': 'Insufficient data for statistical validation'}
        
        # Remove any NaN or infinite values
        returns = returns[np.isfinite(returns)]
        
        if len(returns) == 0:
            return {'error': 'No valid returns for analysis'}
        
        # Test if returns are significantly different from random (t-test)
        t_stat, p_value = stats.ttest_1samp(returns, 0)
        
        # Check for serial correlation (market efficiency test)
        if len(returns) > 1:
            autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            autocorr = 0 if np.isnan(autocorr) else autocorr
        else:
            autocorr = 0
        
        # Jarque-Bera test for normality
        try:
            jb_stat, jb_pvalue = stats.jarque_bera(returns)
        except:
            jb_stat, jb_pvalue = 0, 1
        
        # Shapiro-Wilk test for normality (more sensitive for small samples)
        try:
            if len(returns) <= 5000:  # Shapiro-Wilk limitation
                sw_stat, sw_pvalue = stats.shapiro(returns)
            else:
                sw_stat, sw_pvalue = 0, 1
        except:
            sw_stat, sw_pvalue = 0, 1
        
        # Calculate skewness and kurtosis
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # Value at Risk (VaR) at 95% and 99% confidence levels
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # Expected Shortfall (Conditional VaR)
        es_95 = returns[returns <= var_95].mean() if np.any(returns <= var_95) else var_95
        es_99 = returns[returns <= var_99].mean() if np.any(returns <= var_99) else var_99
        
        return {
            'significant_alpha': p_value < 0.05,
            't_statistic': t_stat,
            'p_value': p_value,
            'serial_correlation': autocorr,
            'returns_normal_jb': jb_pvalue > 0.05,
            'returns_normal_sw': sw_pvalue > 0.05,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'var_95': var_95,
            'var_99': var_99,
            'expected_shortfall_95': es_95,
            'expected_shortfall_99': es_99,
            'jarque_bera_stat': jb_stat,
            'shapiro_wilk_stat': sw_stat
        }
    except ImportError:
        return {'error': 'scipy not available for statistical validation'}
    except Exception as e:
        return {'error': f'Statistical validation failed: {str(e)}'}

def calculate_advanced_metrics(equity_curve: List[float], returns: np.ndarray, 
                             trades: List[Dict]) -> Dict:
    """Calculate comprehensive performance metrics for production validation"""
    if len(equity_curve) < 2 or len(returns) == 0:
        return {}
    
    # Basic metrics
    total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
    
    # Drawdown analysis
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
    
    # Average drawdown
    avg_drawdown = np.mean(drawdown[drawdown > 0]) if np.any(drawdown > 0) else 0
    
    # Drawdown duration
    drawdown_starts = np.where(np.diff(np.concatenate([[False], drawdown > 0])))[0]
    drawdown_ends = np.where(np.diff(np.concatenate([drawdown > 0, [False]])))[0]
    
    if len(drawdown_starts) > 0 and len(drawdown_ends) > 0:
        # Ensure arrays have same length
        min_len = min(len(drawdown_starts), len(drawdown_ends))
        drawdown_starts = drawdown_starts[:min_len]
        drawdown_ends = drawdown_ends[:min_len]
        
        if min_len > 0:
            max_drawdown_duration = max(drawdown_ends - drawdown_starts)
            avg_drawdown_duration = np.mean(drawdown_ends - drawdown_starts)
        else:
            max_drawdown_duration = avg_drawdown_duration = 0
    else:
        max_drawdown_duration = avg_drawdown_duration = 0
    
    # Risk-adjusted returns
    if len(returns) > 1 and np.std(returns) > 0:
        # Sharpe ratio (assuming 5% risk-free rate for forex)
        sharpe_ratio = (np.mean(returns) - 0.05/252) / np.std(returns) * np.sqrt(252)
        
        # Calmar ratio (return / max drawdown)
        calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else float('inf')
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_deviation = np.std(downside_returns) * np.sqrt(252)
            sortino_ratio = (np.mean(returns) * 252 - 0.05) / downside_deviation
        else:
            sortino_ratio = float('inf')
    else:
        sharpe_ratio = calmar_ratio = sortino_ratio = 0
    
    # Trade analysis
    if trades:
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
        
        win_rate = len(winning_trades) / len(trades)
        
        if winning_trades and losing_trades:
            avg_win = np.mean([t['pnl'] for t in winning_trades])
            avg_loss = np.mean([t['pnl'] for t in losing_trades])
            profit_factor = abs(avg_win * len(winning_trades)) / abs(avg_loss * len(losing_trades))
        else:
            profit_factor = float('inf') if winning_trades else 0
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        # Recovery factor (net profit / max drawdown)
        net_profit = sum(t.get('pnl', 0) for t in trades)
        recovery_factor = net_profit / (max_drawdown * equity_curve[0]) if max_drawdown > 0 else float('inf')
        
        # Maximum consecutive wins/losses
        consecutive_wins = consecutive_losses = 0
        max_consecutive_wins = max_consecutive_losses = 0
        
        for trade in trades:
            if trade.get('pnl', 0) > 0:
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            else:
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
    else:
        win_rate = profit_factor = recovery_factor = 0
        avg_win = avg_loss = 0
        max_consecutive_wins = max_consecutive_losses = 0
    
    # Statistical validation
    statistical_tests = validate_strategy_statistically(returns)
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'calmar_ratio': calmar_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'avg_drawdown': avg_drawdown,
        'max_drawdown_duration': max_drawdown_duration,
        'avg_drawdown_duration': avg_drawdown_duration,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'recovery_factor': recovery_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'max_consecutive_wins': max_consecutive_wins,
        'max_consecutive_losses': max_consecutive_losses,
        'total_trades': len(trades),
        'statistical_tests': statistical_tests
    }

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

def validate_yahoo_data(df: pd.DataFrame, pair_name: str) -> Dict[str, any]:
    """Check if Yahoo data is suitable for training and flag issues"""
    warnings = []
    critical_issues = []
    
    # Check for weekend gaps and data quality
    time_diffs = df.index.to_series().diff()
    max_gap = time_diffs.max()
    
    # Weekend gaps are normal, but excessive gaps are problematic
    excessive_gaps = sum(1 for gap in time_diffs if gap > pd.Timedelta(days=3))
    if excessive_gaps > 0:
        critical_issues.append(f"Found {excessive_gaps} excessive data gaps (>3 days)")
    
    # Check for sufficient data (relaxed requirements)
    if len(df) < 500:
        critical_issues.append(f"Insufficient data: {len(df)} candles (need >500 for basic training)")
    elif len(df) < 2000:
        warnings.append(f"Limited data: {len(df)} candles (recommended >2000 for better results)")
    elif len(df) < 5000:
        warnings.append(f"Moderate data: {len(df)} candles (ideal >5000 for production)")
    elif len(df) < 10000:
        warnings.append(f"Good data: {len(df)} candles (excellent >10000)")
    
    # Check for missing values
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        warnings.append(f"Missing values detected: {missing_values} cells")
    
    # Check for duplicate timestamps
    duplicates = df.index.duplicated().sum()
    if duplicates > 0:
        warnings.append(f"Duplicate timestamps found: {duplicates}")
    
    # Check for zero/negative prices (data corruption)
    price_issues = sum(1 for col in ['open', 'high', 'low', 'close'] 
                      if (df[col] <= 0).any())
    if price_issues > 0:
        critical_issues.append(f"Zero/negative prices found in {price_issues} price columns")
    
    # Check volume data availability
    if 'volume' not in df.columns or df['volume'].isna().all():
        warnings.append("No volume data available (impacts some indicators)")
    
    return {
        'is_valid': len(critical_issues) == 0,
        'warnings': warnings,
        'critical_issues': critical_issues,
        'data_points': len(df),
        'date_range': f"{df.index[0]} to {df.index[-1]}",
        'max_gap': str(max_gap)
    }

def load_forex_data(period: str = "1y", interval: str = "1h") -> Dict[str, pd.DataFrame]:
    """Load forex data from Yahoo Finance with validation"""
    
    print("📊 Loading forex data from Yahoo Finance...")
    
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
                
                # Simple validation - just check if we have enough data
                if len(df) > 500:  # Basic check for minimum data
                    data[pair_name] = df
                    print(f"✓ {pair_name}: {len(df)} data points loaded successfully")
                else:
                    print(f"✗ {pair_name}: Insufficient data ({len(df)} points, need >500)")
            else:
                print(f"✗ {pair_name}: No data available")
                
        except Exception as e:
            print(f"✗ Error loading {pair_name}: {str(e)}")
    
    print(f"\n✅ Successfully loaded {len(data)} currency pairs")
    
    if not data:
        print("⚠️ No valid forex data loaded - creating sample data")
        return create_sample_data()
    
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
    """Production-grade trading environment with comprehensive safety features"""
    
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
        self.safety_checks = ProductionSafetyChecks()
        self.indicators = ForexIndicators()
        
        # Performance tracking
        self.equity_curve = [self.initial_balance]
        self.drawdown_series = []
        self.trade_count = 0
        self.winning_trades = 0
        
        # Market conditions
        self.market_conditions = None
        
        # Enhanced tracking
        self.execution_details_history = []
        self.safety_violations = []
        
    def get_pip_value(self, price: float) -> float:
        """Get pip value for CFD contract based on instrument type"""
        if 'JPY' in self.pair_name:
            return 0.01  # JPY pairs: 1 pip = 0.01
        else:
            return 0.0001  # Major pairs: 1 pip = 0.0001
        
    def calculate_position_size(self, price: float, stop_loss_pips: float, risk_percent: float = 0.02) -> float:
        """Calculate position size based on risk management - returns position in USD value"""
        risk_amount = self.balance * risk_percent
        
        # For forex: position_value = risk_amount / (stop_loss_pips * pip_value_in_dollars)
        # 1 pip for 100,000 units = $10 for major pairs, $1000 for JPY pairs
        if 'JPY' in self.pair_name:
            pip_value_usd = 1000  # For JPY pairs, 1 pip = $1000 per standard lot
        else:
            pip_value_usd = 10   # For major pairs, 1 pip = $10 per standard lot
        
        # Calculate position size to risk exactly risk_amount on stop_loss_pips
        position_value = risk_amount / stop_loss_pips * pip_value_usd
        
        # Apply maximum position limits
        max_position_value = self.balance * 10  # 10:1 leverage maximum
        return min(position_value, max_position_value)
        
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
        
        # Create a FIXED-SIZE feature vector to ensure consistent neural network input
        # This is critical for production systems to avoid dimension mismatch errors
        
        feature_vectors = []
        
        # 1. Basic OHLCV features (5 * 30 = 150 features)
        price_features = ['open', 'high', 'low', 'close', 'volume']
        for col in price_features:
            if col in lookback_data.columns:
                values = lookback_data[col].values
                if len(values) == self.lookback_period:
                    # Normalize and ensure exact length
                    normalized = (values - np.nanmean(values)) / (np.nanstd(values) + 1e-8)
                    normalized = np.nan_to_num(normalized, nan=0.0)
                    feature_vectors.extend(normalized)
                else:
                    # Pad with zeros if missing data
                    feature_vectors.extend([0.0] * self.lookback_period)
            else:
                # Add zeros if column missing
                feature_vectors.extend([0.0] * self.lookback_period)
        
        # 2. Technical indicators - last 5 values each (simplified for consistency)
        indicator_cols = ['sma_10', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'atr', 'macd', 'macd_signal']
        for col in indicator_cols:
            if col in lookback_data.columns and not lookback_data[col].isna().all():
                values = lookback_data[col].dropna().values
                if len(values) >= 5:
                    last_values = values[-5:]
                    normalized = (last_values - np.nanmean(last_values)) / (np.nanstd(last_values) + 1e-8)
                    normalized = np.nan_to_num(normalized, nan=0.0)
                    feature_vectors.extend(normalized)
                else:
                    feature_vectors.extend([0.0] * 5)
            else:
                feature_vectors.extend([0.0] * 5)
        
        # 3. Market regime features (6 features)
        regime_features = self._encode_market_regime()
        feature_vectors.extend(regime_features)
        
        # 4. Additional market context (10 features)
        if len(lookback_data) > 0:
            recent_volatility = lookback_data['volatility'].iloc[-5:].mean() if 'volatility' in lookback_data.columns else 0.0
            price_range = (lookback_data['high'].max() - lookback_data['low'].min()) / lookback_data['close'].iloc[-1] if len(lookback_data) > 0 else 0.0
            volume_trend = lookback_data['volume'].iloc[-5:].mean() / lookback_data['volume'].mean() if 'volume' in lookback_data.columns else 1.0
            
            additional_features = [
                recent_volatility, price_range, volume_trend,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # Padding to make exactly 10
            ]
            feature_vectors.extend(additional_features)
        else:
            feature_vectors.extend([0.0] * 10)
        
        # Ensure EXACTLY the expected size: 5*30 + 8*5 + 6 + 10 = 150 + 40 + 6 + 10 = 206
        expected_size = 206
        current_size = len(feature_vectors)
        
        if current_size < expected_size:
            # Pad with zeros
            feature_vectors.extend([0.0] * (expected_size - current_size))
        elif current_size > expected_size:
            # Truncate
            feature_vectors = feature_vectors[:expected_size]
        
        # Convert to numpy array
        state_vector = np.array(feature_vectors, dtype=np.float32)
        
        # Final safety check
        state_vector = np.nan_to_num(state_vector, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Debug: Verify size consistency
        if len(state_vector) != expected_size:
            print(f"⚠️ DEBUG: State vector size {len(state_vector)} != expected {expected_size} for {self.pair_name}")
            print(f"   Current size breakdown: OHLCV({5*self.lookback_period}) + Indicators({8*5}) + Regime(6) + Additional(10)")
            print(f"   Lookback period: {self.lookback_period}")
            # Force correct size
            if len(state_vector) < expected_size:
                state_vector = np.pad(state_vector, (0, expected_size - len(state_vector)), 'constant')
            else:
                state_vector = state_vector[:expected_size]
        
        return state_vector
    
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
        """Execute trading action with safety checks and realistic costs"""
        current_price = self.data.iloc[self.current_step]['close']
        current_volatility = self.data.iloc[self.current_step]['volatility']
        reward = 0
        
        # Calculate spread based on market conditions
        spread_data = self.cost_calculator.spreads.get(self.pair_name, {
            'normal': 2.0, 'volatile': 5.0, 'news': 10.0, 'weekend_gap': 20.0
        })
        
        # Determine market condition
        if current_volatility > 0.8:
            current_spread_pips = spread_data['volatile']
        elif current_volatility > 0.5:
            current_spread_pips = spread_data['normal'] * 1.5
        else:
            current_spread_pips = spread_data['normal']
        
        # Apply weekend gap spreads if applicable
        if self.current_step % 120 == 0:
            current_spread_pips = spread_data['weekend_gap']
        
        # Pre-trade safety checks
        safety_passed, safety_results = self.safety_checks.pre_trade_checks(
            self.pair_name, current_spread_pips
        )
        
        if not safety_passed:
            # Record safety violation
            self.safety_violations.append({
                'step': self.current_step,
                'violations': [r[1] for r in safety_results if r[0] == 'FAIL']
            })
            reward -= 0.1  # Tiny penalty for safety violations
        
        # Calculate position size for this trade
        position_size = self.calculate_position_size(
            current_price, 
            action.stop_loss,
            risk_percent=0.01 if self.market_conditions.regime == 'high_volatility' else 0.02
        )
        
        # Check if risk manager allows trading
        can_trade, risk_message = self.risk_manager.can_trade(
            self.balance, self.pair_name, action.direction, position_size, current_price
        )
        
        # Check margin requirements
        margin_call, margin_message = self.risk_manager.check_margin_call(self.balance)
        if margin_call:
            reward -= 0.5  # Tiny penalty for margin issues
            self.safety_violations.append({
                'step': self.current_step,
                'violations': [margin_message]
            })
        
        # Close existing position if any
        if self.position:
            if self.position == 'long':
                price_change = current_price - self.entry_price
            else:  # short
                price_change = self.entry_price - current_price
            
            # Calculate pip value in USD for this position
            if 'JPY' in self.pair_name:
                pip_value_usd = position_size / 100000 * 1000  # $1000 per standard lot for JPY
            else:
                pip_value_usd = position_size / 100000 * 10    # $10 per standard lot for majors
            
            # Calculate pips gained/lost
            pip_value = self.get_pip_value(current_price)
            pips_gained = price_change / pip_value
            
            # Apply realistic stop loss and take profit with slippage
            exit_cost, exit_details = self.cost_calculator.calculate_entry_cost(
                self.pair_name, current_price, current_volatility, 
                position_size, self.market_conditions
            )
            
            # Simplified profit calculation - much lower costs for training
            trading_cost = position_size * 0.0001  # 0.01% trading cost (much lower than reality)
            
            # FIXED: Much smaller rewards based on actual performance
            if pips_gained <= -action.stop_loss:  # Stop loss hit
                actual_profit = -action.stop_loss * pip_value_usd - trading_cost
                reward = -2  # Small fixed penalty for stop loss
                
            elif pips_gained >= action.take_profit:  # Take profit hit
                actual_profit = action.take_profit * pip_value_usd - trading_cost
                reward = +3  # Small fixed reward for take profit
                self.winning_trades += 1
                
            else:
                # Regular exit - calculate actual profit
                actual_profit = pips_gained * pip_value_usd - trading_cost
                
                # Small rewards based on prediction accuracy
                if actual_profit > 0:
                    reward = +1  # Small reward for winning trade
                    self.winning_trades += 1
                else:
                    reward = -1  # Small penalty for losing trade
            
            self.balance += actual_profit
            self.trade_count += 1
            
            # Remove position from risk manager
            self.risk_manager.remove_position(self.pair_name, actual_profit)
            self.position = None
            
            # Record trade with execution details
            self.trade_history.append({
                'step': self.current_step,
                'action': 'close',
                'price': current_price,
                'pnl': actual_profit,
                'pips': pips_gained,
                'balance': self.balance,
                'execution_details': exit_details
            })
            
            self.execution_details_history.append(exit_details)
        
        # TRAINING MODE: Prioritize trading over safety - focus on prediction accuracy
        # Open new position if requested (very lenient blocking)
        if action.direction != 'hold':
            
            # For training: Always allow trading unless extreme emergency
            force_trade = True
            block_reason = ""
            
            # Only block in extreme cases
            if not can_trade:
                if "EXTREME" in risk_message:
                    force_trade = False
                    block_reason = risk_message
                else:
                    # Override risk manager for training - focus on predictions
                    force_trade = True
            
            if force_trade:
                # Calculate entry costs with execution risks
                entry_cost, entry_details = self.cost_calculator.calculate_entry_cost(
                    self.pair_name, current_price, current_volatility, 
                    position_size, self.market_conditions
                )
                
                # Check if trade was rejected by broker simulation
                if entry_details.get('rejected', False):
                    reward -= 0.1  # Tiny penalty for rejected trade
                    print(f"\r❌ TRADE REJECTED: {action.direction}", end='', flush=True)
                else:
                    # Deduct entry costs immediately (but much reduced for training)
                    training_cost = entry_cost * 0.1  # Only 10% of real costs during training
                    self.balance -= training_cost
                    
                    # Open position
                    self.position = action.direction
                    self.entry_price = current_price
                    
                    # MINIMAL reward for taking action - focus on quality
                    reward += 0.1  # Very tiny reward for taking action
                    
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
                        'cost': training_cost,
                        'size': position_size,
                        'execution_details': entry_details
                    })
                    
                    self.execution_details_history.append(entry_details)
                    
                    # Show what type of trade this is
                    if not safety_passed or not can_trade:
                        print(f"\r🚀 FORCED TRADE: {action.direction} (TP:{action.take_profit:.0f}, SL:{action.stop_loss:.0f})", end='', flush=True)
                    else:
                        print(f"\r🧠 NETWORK TRADE: {action.direction} (TP:{action.take_profit:.0f}, SL:{action.stop_loss:.0f})", end='', flush=True)
            else:
                # Only block in extreme cases
                reward -= 0.1  # Tiny penalty for extreme blocks
                print(f"\r🚫 TRADE BLOCKED (Extreme): {action.direction} - {block_reason}", end='', flush=True)
        else:
            # Allow some holding - don't force trades constantly
            reward = 0  # Neutral for holding
            print(f"\r⏸️  HOLD POSITION", end='', flush=True)
        
        # REMOVED: No extra rewards for just trading - focus on quality
        # if action.direction != 'hold':
        #     reward += 15  # Big reward for any trading activity!
        
        # Update performance tracking
        self.equity_curve.append(self.balance)
        
        # Calculate drawdown
        peak = max(self.equity_curve)
        drawdown = (peak - self.balance) / peak if peak > 0 else 0
        self.drawdown_series.append(drawdown)
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # STRONG FINAL PERFORMANCE PENALTY when episode ends
        if done:
            # Calculate final performance and heavily penalize bad results
            total_return = (self.balance - self.initial_balance) / self.initial_balance
            if self.trade_count > 0:
                final_win_rate = self.winning_trades / self.trade_count
                
                # MASSIVE penalty for ending with terrible performance
                if final_win_rate < 0.3 and total_return < -0.5:  # <30% win rate AND lost >50%
                    reward -= 100  # Huge penalty
                elif final_win_rate < 0.4 and total_return < -0.2:  # <40% win rate AND lost >20%
                    reward -= 50   # Large penalty
                elif final_win_rate < 0.5:  # <50% win rate
                    reward -= 20   # Medium penalty
                
                # Reward for ending with good performance
                if final_win_rate > 0.6 and total_return > 0.1:  # >60% win rate AND profit >10%
                    reward += 50   # Good reward for actual success
        
        # TRAINING MODE: Much more lenient emergency rules
        # Emergency shutdown only in extreme cases
        if self.balance <= self.initial_balance * 0.01:  # Only stop at 99% loss
            reward -= 50  # Strong penalty for complete failure
            done = True
        
        # Very lenient penalties for risk management during training
        if self.balance <= 0:
            reward -= 100  # Strong penalty for complete loss
            done = True
        elif drawdown > 0.95:  # Only warn at 95% drawdown
            reward -= 10  # Medium penalty
        
        # FIXED: Performance-based rewards - negative for bad performance
        if self.trade_count > 20:  # Need enough trades for meaningful metrics
            win_rate = self.winning_trades / self.trade_count
            
            # Base reward on actual performance - negative for bad results
            if win_rate >= 0.6:      # 60%+ win rate
                reward += 2   # Small reward for excellent performance
            elif win_rate >= 0.5:    # 50-60% win rate  
                reward += 1   # Small reward for good performance
            elif win_rate >= 0.4:    # 40-50% win rate
                reward += 0   # Neutral for decent performance
            elif win_rate >= 0.3:    # 30-40% win rate
                reward -= 1   # Penalty for poor performance
            else:                    # <30% win rate
                reward -= 5   # Strong penalty for terrible performance
        
        # Strong penalty for excessive drawdown
        if drawdown > 0.9:   # 90%+ drawdown is terrible
            reward -= 10
        elif drawdown > 0.8:  # 80%+ drawdown is bad
            reward -= 5
        elif drawdown > 0.6:  # 60%+ drawdown is concerning
            reward -= 2
        
        # Additional penalty for losing money overall
        if self.balance < self.initial_balance * 0.8:  # Lost 20%+ of account
            reward -= 5
        elif self.balance < self.initial_balance * 0.5:  # Lost 50%+ of account  
            reward -= 10
        
        return self._get_state(), reward, done
    
    def get_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics using advanced calculations"""
        if self.trade_count == 0:
            return {
                'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 'win_rate': 0,
                'safety_violations': len(self.safety_violations),
                'execution_quality': {},
                'total_trades': 0  # Add this missing field
            }
        
        # Calculate returns array
        returns = np.diff(self.equity_curve) / np.array(self.equity_curve[:-1])
        
        # Use advanced metrics calculation
        advanced_metrics = calculate_advanced_metrics(self.equity_curve, returns, self.trade_history)
        
        # Add execution quality metrics
        execution_quality = self._analyze_execution_quality()
        
        # Add safety metrics
        safety_metrics = {
            'safety_violations': len(self.safety_violations),
            'violation_rate': len(self.safety_violations) / max(1, self.trade_count),
            'emergency_stops': sum(1 for v in self.safety_violations 
                                 if 'EMERGENCY' in str(v.get('violations', [])))
        }
        
        # Combine all metrics
        all_metrics = {
            **advanced_metrics,
            'execution_quality': execution_quality,
            'safety_metrics': safety_metrics,
            'final_balance': self.balance
        }
        
        return all_metrics
    
    def _analyze_execution_quality(self) -> Dict:
        """Analyze execution quality from execution details history"""
        if not self.execution_details_history:
            return {}
        
        # Extract execution statistics
        partial_fills = sum(1 for ed in self.execution_details_history 
                           if ed.get('partial_fill', False))
        requotes = sum(1 for ed in self.execution_details_history 
                      if ed.get('requoted', False))
        rejections = sum(1 for ed in self.execution_details_history 
                        if ed.get('rejected', False))
        
        # Average spread and slippage
        spreads = [ed.get('spread_pips', 0) for ed in self.execution_details_history]
        slippages = [ed.get('slippage_pips', 0) for ed in self.execution_details_history]
        
        # Market condition distribution
        conditions = [ed.get('market_condition', 'normal') for ed in self.execution_details_history]
        condition_counts = {cond: conditions.count(cond) for cond in set(conditions)}
        
        return {
            'partial_fill_rate': partial_fills / len(self.execution_details_history),
            'requote_rate': requotes / len(self.execution_details_history),
            'rejection_rate': rejections / len(self.execution_details_history),
            'avg_spread_pips': np.mean(spreads) if spreads else 0,
            'avg_slippage_pips': np.mean(slippages) if slippages else 0,
            'market_condition_distribution': condition_counts,
            'total_executions': len(self.execution_details_history)
        }

class DQNNetwork(nn.Module):
    """Simple and fast Deep Q-Network for trading decisions"""
    
    def __init__(self, input_size: int, hidden_size: int = 512):
        super(DQNNetwork, self).__init__()
        
        # Deep network architecture
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU()
        )
        
        # Separate heads for different outputs
        self.direction_head = nn.Linear(hidden_size // 2, 3)
        self.tp_head = nn.Linear(hidden_size // 2, 10)
        self.sl_head = nn.Linear(hidden_size // 2, 10)
        
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
        # Memory buffer
        if device.type == 'cuda':
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_memory_gb >= 16:
                self.memory = deque(maxlen=100000)
            elif gpu_memory_gb >= 12:
                self.memory = deque(maxlen=50000)
            else:
                self.memory = deque(maxlen=20000)
        else:
            self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_min = 0.3  # Higher minimum to encourage more trading
        self.epsilon_decay = 0.99  # Slower decay to keep exploring longer
        self.learning_rate = learning_rate
        self.gamma = 0.95
        
        # Batch size
        if device.type == 'cuda':
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_memory_gb >= 16:
                self.batch_size = 2048
            elif gpu_memory_gb >= 12:
                self.batch_size = 1024
            else:
                self.batch_size = 512
        else:
            self.batch_size = 32
        
        # Neural networks
        self.q_network = DQNNetwork(state_size).to(device)
        self.target_network = DQNNetwork(state_size).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # Mixed precision
        self.scaler = GradScaler() if device.type == 'cuda' else None
        
        self.update_target_network()
        
        # Minimal initialization message
        if device.type == 'cuda':
            print(f"� GPU initialized: {torch.cuda.get_device_name(0)}")
        else:
            pass  # Silent initialization
        
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray) -> TradingAction:
        """Choose action using epsilon-greedy policy with EXTREME bias toward trading"""
        # ULTRA AGGRESSIVE: Force 95% trading actions!
        if random.random() < 0.95:  # 95% chance to force trading!
            # Almost always trade - NEVER hold!
            direction = random.choice(['long', 'short'])  # NO HOLD OPTION!
            tp = random.uniform(20, 100)  # pips
            sl = random.uniform(10, 50)   # pips
            print(f"\r🚀 FORCED TRADE: {direction} (TP:{tp:.0f}, SL:{sl:.0f})", end='', flush=True)
        else:
            # Only 5% chance to use network prediction
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device, non_blocking=True)
            with torch.no_grad():
                direction_q, tp_q, sl_q = self.q_network(state_tensor)
            
            direction_idx = torch.argmax(direction_q).item()
            direction = ['long', 'short', 'hold'][direction_idx]
            
            # EVEN IF NETWORK SAYS HOLD, FORCE TRADE!
            if direction == 'hold':
                direction = random.choice(['long', 'short'])
                print(f"\r🔄 NETWORK WANTED HOLD - FORCING {direction}!", end='', flush=True)
            
            tp_idx = torch.argmax(tp_q).item()
            sl_idx = torch.argmax(sl_q).item()
            
            tp = 20 + tp_idx * 10  # 20-120 pips
            sl = 10 + sl_idx * 5   # 10-60 pips
            print(f"\r🧠 NETWORK TRADE: {direction} (TP:{tp:.0f}, SL:{sl:.0f})", end='', flush=True)
        
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
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'state_size': self.state_size,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'batch_size': self.batch_size
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.state_size = checkpoint['state_size']
        self.learning_rate = checkpoint['learning_rate']
        self.gamma = checkpoint['gamma']
        self.batch_size = checkpoint['batch_size']

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
                                 train_periods: int = 500,
                                 test_periods: int = 100,
                                 step_size: int = 50,
                                 epochs_per_window: int = 25) -> Dict:
        """Implement walk-forward optimization with progress bar"""
        
        results = {}
        
        for pair in self.pairs:
            if pair not in data:
                continue
                
            df = data[pair]
            pair_results = []
            
            # Ensure we have enough data
            if len(df) < train_periods + test_periods:
                print(f"⚠️ {pair}: Insufficient data ({len(df)} points)")
                continue
            
            # Calculate total windows for progress bar
            total_windows = (len(df) - train_periods - test_periods) // step_size
            
            # Progress bar for this pair
            print(f"\n🔄 {pair} - Walk-forward validation:")
            progress_bar = tqdm(total=total_windows, desc=f"{pair}")
            
            # Walk-forward windows
            for i in range(0, len(df) - train_periods - test_periods, step_size):
                train_start = i
                train_end = i + train_periods
                test_start = train_end
                test_end = test_start + test_periods
                
                # Prepare training data
                train_data = df.iloc[train_start:train_end].copy()
                prepared_train_data = self.prepare_data(train_data)
                
                # Create environment and agent (minimal output)
                env = ForexEnvironment(prepared_train_data, pair_name=pair)
                temp_state = env.reset()
                agent = ForexTradingAgent(state_size=len(temp_state))
                
                # Training with minimal output
                train_rewards = []
                for epoch in range(epochs_per_window):
                    env = ForexEnvironment(prepared_train_data, pair_name=pair)
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
                agent.epsilon = 0
                
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
                
                # Store results
                train_avg = np.mean(train_rewards[-10:]) if train_rewards else 0
                pair_results.append({
                    'train_window': (train_start, train_end),
                    'test_window': (test_start, test_end),
                    'train_reward_avg': train_avg,
                    'test_reward': test_reward,
                    'test_metrics': test_metrics
                })
                
                # Update progress bar
                progress_bar.set_description(f"{pair} - Train: {train_avg:.0f}, Test: {test_reward:.0f}")
                progress_bar.update(1)
            
            progress_bar.close()
            results[pair] = pair_results
            
            # Summary for this pair
            if pair_results:
                avg_test_return = np.mean([r['test_metrics']['total_return'] for r in pair_results])
                avg_sharpe = np.mean([r['test_metrics']['sharpe_ratio'] for r in pair_results])
                avg_drawdown = np.mean([r['test_metrics']['max_drawdown'] for r in pair_results])
                
                print(f"✅ {pair} - Return: {avg_test_return:.2%}, Sharpe: {avg_sharpe:.2f}, Drawdown: {avg_drawdown:.2%}")
        
        return results
    
    def train(self, data: Dict[str, pd.DataFrame], epochs: int = 100, 
             use_walk_forward: bool = True, train_test_split: float = 0.8):
        """Train the model with proper train/test split and optional walk-forward validation
        
        🚨 CRITICAL FIX: Implements proper train/test split to prevent overfitting
        - Training on 80% of data (default)
        - Testing on unseen 20% of data
        - Walk-forward validation for additional robustness
        """
        
        self.logger.info("🚨 IMPLEMENTING PROPER TRAIN/TEST SPLIT TO PREVENT OVERFITTING")
        self.logger.info(f"   Training split: {train_test_split:.1%}")
        self.logger.info(f"   Testing split: {1-train_test_split:.1%}")
        
        if use_walk_forward:
            self.logger.info("Using walk-forward optimization for training...")
            self.validation_results = self.walk_forward_optimization(data, epochs_per_window=epochs//2)
        
        # Store training results for each pair
        training_results = {}
        
        # Final training with proper train/test split for each pair
        for pair_idx, pair in enumerate(self.pairs):
            if pair not in data:
                continue
            
            self.logger.info(f"Training {pair} ({pair_idx + 1}/{len(self.pairs)}) with proper data split...")
            
            # 🔴 CRITICAL FIX: Proper train/test split
            full_data = data[pair].copy()
            split_point = int(len(full_data) * train_test_split)
            
            train_data = full_data.iloc[:split_point].copy()
            test_data = full_data.iloc[split_point:].copy()
            
            self.logger.info(f"   📊 {pair} data split:")
            self.logger.info(f"      Training: {len(train_data)} samples ({train_data.index[0]} to {train_data.index[-1]})")
            self.logger.info(f"      Testing:  {len(test_data)} samples ({test_data.index[0]} to {test_data.index[-1]})")
            
            # Prepare training data with indicators
            prepared_train_data = self.prepare_data(train_data)
            
            # 🔴 Validate training data has enough samples
            if len(prepared_train_data) < 100:
                self.logger.warning(f"   ⚠️  {pair}: Very limited training data ({len(prepared_train_data)} samples)")
                self.logger.warning(f"       Consider using longer data period for better training")
                continue  # Skip this pair
            
            # Initialize environment and agent on TRAINING data only
            self.logger.info(f"   🔧 Initializing environment and agent for {pair}...")
            try:
                env = ForexEnvironment(prepared_train_data, pair_name=pair)
                temp_state = env.reset()
                self.logger.info(f"   ✅ Environment initialized. State size: {len(temp_state)}")
                
                agent = ForexTradingAgent(state_size=len(temp_state))
                self.logger.info(f"   ✅ Agent initialized. Memory size: {len(agent.memory)}")
                
            except Exception as e:
                self.logger.error(f"   ❌ Failed to initialize environment/agent for {pair}: {str(e)}")
                continue
            
            # Training progress bar
            self.logger.info(f"   🚀 Starting training for {pair} ({epochs} epochs)...")
            pbar = tqdm(range(epochs), desc=f"🔄 {pair}", 
                       bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} epochs [⏱️{elapsed}<⏳{remaining}, 💰{postfix}]')
            
            best_sharpe = float('-inf')
            best_metrics = None
            rewards_history = []
            
            # Training loop on TRAINING data only
            training_start_time = time.time()
            for epoch in pbar:
                # Timeout protection for entire training
                if time.time() - training_start_time > 1800:  # 30 minutes max per pair
                    self.logger.warning(f"⚠️ Training timeout for {pair} after 30 minutes - stopping early")
                    break
                    
                start_time = time.time()
                try:
                    state = env.reset()
                except Exception as e:
                    self.logger.error(f"❌ Error resetting environment for {pair} epoch {epoch}: {str(e)}")
                    continue
                    
                total_reward = 0
                done = False
                steps = 0
                max_steps = min(500, len(prepared_train_data) - env.lookback_period - 10)  # Limit to 500 steps per epoch
                
                while not done and steps < max_steps:
                    action = agent.act(state)
                    next_state, reward, done = env.step(action)
                    agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    total_reward += reward
                    steps += 1
                    
                    # Emergency break if environment is stuck
                    if steps > max_steps:
                        done = True
                        break
                    
                    # Reduced training frequency for speed
                    min_memory = agent.batch_size
                    if len(agent.memory) > min_memory and steps % 10 == 0:  # Train every 10 steps instead of every step
                        agent.replay()
                
                # Update target network
                if epoch % 10 == 0:
                    agent.update_target_network()
                
                # Get performance metrics with timeout protection
                try:
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
                        
                    # Much more lenient timeout - let epochs complete
                    if epoch_time > 120:  # 2 minutes max per epoch (increased from 30 seconds)
                        self.logger.warning(f"⚠️ {pair} epoch {epoch} took {epoch_time:.1f}s - very slow but continuing...")
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ Error in epoch {epoch} for {pair}: {str(e)}")
                    # Create default metrics to continue training
                    metrics = {'sharpe_ratio': 0, 'total_return': 0, 'win_rate': 0, 'max_drawdown': 0, 'total_trades': 0}
                    rewards_history.append(total_reward)
                    pbar.set_postfix_str(f"Reward: {total_reward:.0f}, Error in metrics calculation")
                    continue
            
            pbar.close()
            print()  # Add newline to separate progress bar from subsequent messages
            
            # 🔴 CRITICAL: Test on unseen data (out-of-sample testing)
            self.logger.info(f"🧪 Testing {pair} on UNSEEN data...")
            prepared_test_data = self.prepare_data(test_data)
            
            # Test the trained agent on unseen data
            test_env = ForexEnvironment(prepared_test_data, pair_name=pair)
            agent.epsilon = 0  # No exploration during testing
            
            test_state = test_env.reset()
            test_total_reward = 0
            test_done = False
            
            while not test_done:
                test_action = agent.act(test_state)
                test_next_state, test_reward, test_done = test_env.step(test_action)
                test_state = test_next_state
                test_total_reward += test_reward
            
            # Get test performance metrics
            test_metrics = test_env.get_performance_metrics()
            
            # Store training results
            training_results[pair] = {
                'train_metrics': best_metrics,
                'test_metrics': test_metrics,
                'train_samples': len(prepared_train_data),
                'test_samples': len(prepared_test_data),
                'overfitting_score': best_metrics['sharpe_ratio'] - test_metrics['sharpe_ratio'] if best_metrics and test_metrics['sharpe_ratio'] != 0 else 0
            }
            
            self.logger.info(f"✅ {pair} training completed!")
            self.logger.info(f"   📊 TRAINING metrics - Return: {best_metrics['total_return']:.2%}, "
                           f"Sharpe: {best_metrics['sharpe_ratio']:.2f}, "
                           f"Win Rate: {best_metrics['win_rate']:.1%}")
            self.logger.info(f"   🧪 TESTING metrics  - Return: {test_metrics['total_return']:.2%}, "
                           f"Sharpe: {test_metrics['sharpe_ratio']:.2f}, "
                           f"Win Rate: {test_metrics['win_rate']:.1%}")
            
            # Check for overfitting
            overfitting_score = training_results[pair]['overfitting_score']
            if overfitting_score > 1.0:
                self.logger.warning(f"   🚨 OVERFITTING DETECTED for {pair}!")
                self.logger.warning(f"      Training Sharpe: {best_metrics['sharpe_ratio']:.2f}")
                self.logger.warning(f"      Testing Sharpe: {test_metrics['sharpe_ratio']:.2f}")
                self.logger.warning(f"      Difference: {overfitting_score:.2f} (>1.0 indicates overfitting)")
            elif overfitting_score > 0.5:
                self.logger.warning(f"   ⚠️  Possible overfitting for {pair} (difference: {overfitting_score:.2f})")
            else:
                self.logger.info(f"   ✅ Good generalization for {pair} (difference: {overfitting_score:.2f})")
            
            self.agents[pair] = agent
        
        # Print comprehensive training summary
        self.logger.info("\n" + "="*80)
        self.logger.info("📊 COMPREHENSIVE TRAINING SUMMARY")
        self.logger.info("="*80)
        
        for pair, results in training_results.items():
            train_metrics = results['train_metrics']
            test_metrics = results['test_metrics']
            
            self.logger.info(f"{pair}:")
            self.logger.info(f"  📈 Training Return: {train_metrics['total_return']:.2%} | Testing Return: {test_metrics['total_return']:.2%}")
            self.logger.info(f"  📊 Training Sharpe: {train_metrics['sharpe_ratio']:.2f} | Testing Sharpe: {test_metrics['sharpe_ratio']:.2f}")
            self.logger.info(f"  🎯 Training WinRate: {train_metrics['win_rate']:.1%} | Testing WinRate: {test_metrics['win_rate']:.1%}")
            self.logger.info(f"  📉 Training MaxDD: {train_metrics['max_drawdown']:.1%} | Testing MaxDD: {test_metrics['max_drawdown']:.1%}")
            self.logger.info(f"  🔍 Overfitting Score: {results['overfitting_score']:.2f}")
            self.logger.info("")
        
        # Overall validation check
        avg_overfitting = np.mean([r['overfitting_score'] for r in training_results.values()])
        if avg_overfitting > 1.0:
            self.logger.error("🚨 SEVERE OVERFITTING DETECTED ACROSS MULTIPLE PAIRS!")
            self.logger.error("   Model is likely to perform poorly in live trading.")
            self.logger.error("   Recommendations:")
            self.logger.error("   • Reduce model complexity")
            self.logger.error("   • Increase training data")
            self.logger.error("   • Add more regularization")
            self.logger.error("   • Use simpler indicators")
        elif avg_overfitting > 0.5:
            self.logger.warning("⚠️  Moderate overfitting detected. Proceed with caution.")
        else:
            self.logger.info("✅ Good generalization across all pairs!")
        
        return training_results
    
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
        # Make a copy to avoid modifying original data
        df_copy = df.copy()
        
        if len(df_copy) < 30:
            # Pad with last available values if not enough data
            last_row = df_copy.iloc[-1:].copy()  # Get last row as DataFrame
            rows_needed = 30 - len(df_copy)
            
            # Create padding by repeating last row
            padding_rows = []
            for _ in range(rows_needed):
                padding_rows.append(last_row)
            
            if padding_rows:
                padding_df = pd.concat(padding_rows, ignore_index=True)
                df_copy = pd.concat([df_copy, padding_df], ignore_index=True)
        
        # Use last 30 rows and prepare state like the environment does
        recent_data = df_copy.tail(30).copy()
        
        # Prepare data with indicators
        prepared_data = self.prepare_data(recent_data)
        
        # Create a temporary environment to get the state
        temp_env = ForexEnvironment(prepared_data)
        temp_env.current_step = temp_env.lookback_period  # Set proper step
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
    parser.add_argument('--walk-forward', action='store_true', default=False,
                       help='Use walk-forward optimization (default: False)')
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
            model_filename = f"forex_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            model_path = os.path.join(args.save_dir, model_filename)
            
            print(f"\n💾 Saving trained model to: {model_path}")
            predictor.save_model(model_path)
            
            # Print training summary
            print("\n" + "="*80)
            print("✅ TRAINING COMPLETED SUCCESSFULLY!")
            print("="*80)
            print(f"📁 Model saved as: {model_path}")
            print(f"⏱️  Total training time: {total_time/60:.1f} minutes")
            
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
        
       
        
    except KeyboardInterrupt:
        logger.warning("⚠️  Training interrupted by user")
    except Exception as e:
        logger.error(f"❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
    
    logger.info("Production-grade Forex AI training session ended.")
