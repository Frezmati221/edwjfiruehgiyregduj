import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import talib
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import logging
from tqdm import tqdm
import os
import pickle
from typing import Dict, List, Tuple, Optional
import json

warnings.filterwarnings('ignore')

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
else:
    print("💻 Using CPU")

class GoldPatternDataset(Dataset):
    """Dataset for supervised learning of gold patterns"""
    
    def __init__(self, features, labels, sequence_length=60):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class GoldPatternRecognitionNetwork(nn.Module):
    """Advanced pattern recognition network optimized for Gold trading"""
    
    def __init__(self, input_dim, hidden_dim=256, num_layers=4, dropout=0.3):
        super().__init__()
        
        # Larger network for Gold's complex patterns
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # Enhanced attention for Gold's trend patterns
        self.attention = nn.MultiheadAttention(
            hidden_dim * 2,  # bidirectional
            num_heads=8,  # More heads for complex patterns
            dropout=dropout
        )
        
        # Deeper classifier for Gold's volatility patterns
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 4, 3)  # 3 classes: long, short, hold
        )
        
        # Confidence estimator tuned for Gold
        self.confidence = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use the last output with attention
        if len(attn_out.shape) == 3:
            features = attn_out[:, -1, :]
        else:
            features = attn_out
        
        # Classification
        logits = self.classifier(features)
        confidence = self.confidence(features)
        
        return logits, confidence

class SupervisedGoldPredictor:
    """Supervised learning system optimized for Gold trading"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.sequence_length = 72  # Longer sequences for Gold trends
        self.min_confidence = 0.6  # Higher confidence for Gold volatility
        
    def create_gold_features(self, df):
        """Create comprehensive technical features optimized for Gold trading"""
        
        features_df = pd.DataFrame(index=df.index)
        
        # Price features - more important for Gold
        features_df['returns'] = df['close'].pct_change()
        features_df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        features_df['high_low_ratio'] = df['high'] / df['low']
        features_df['close_open_ratio'] = df['close'] / df['open']
        features_df['price_momentum'] = df['close'] / df['close'].shift(5)
        
        # Gold-specific price levels
        features_df['price_level'] = df['close'] // 50  # Round to nearest $50
        features_df['distance_to_round_level'] = df['close'] % 50
        features_df['above_2000'] = (df['close'] > 2000).astype(int)
        features_df['above_1900'] = (df['close'] > 1900).astype(int)
        features_df['above_1800'] = (df['close'] > 1800).astype(int)
        
        # Enhanced volatility features for Gold
        for period in [5, 10, 20, 30]:
            features_df[f'volatility_{period}'] = features_df['returns'].rolling(period).std()
            features_df[f'volatility_rank_{period}'] = features_df[f'volatility_{period}'].rolling(period*2).rank() / (period*2)
        
        # Volatility ratios
        features_df['vol_ratio_5_20'] = features_df['volatility_5'] / features_df['volatility_20']
        features_df['vol_ratio_10_30'] = features_df['volatility_10'] / features_df['volatility_30']
        
        # Multiple timeframe SMAs - critical for Gold trends
        for period in [5, 10, 20, 50, 100, 200]:
            features_df[f'sma_{period}'] = df['close'].rolling(period).mean()
            features_df[f'sma_{period}_slope'] = features_df[f'sma_{period}'].diff()
            features_df[f'price_to_sma_{period}'] = df['close'] / features_df[f'sma_{period}']
            features_df[f'sma_{period}_above'] = (df['close'] > features_df[f'sma_{period}']).astype(int)
        
        # SMA relationships - Gold trend strength
        features_df['sma_alignment_bull'] = (
            (features_df['sma_5'] > features_df['sma_10']) &
            (features_df['sma_10'] > features_df['sma_20']) &
            (features_df['sma_20'] > features_df['sma_50'])
        ).astype(int)
        
        features_df['sma_alignment_bear'] = (
            (features_df['sma_5'] < features_df['sma_10']) &
            (features_df['sma_10'] < features_df['sma_20']) &
            (features_df['sma_20'] < features_df['sma_50'])
        ).astype(int)
        
        # EMAs with Gold-specific periods
        for period in [9, 12, 21, 26, 50]:
            features_df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            features_df[f'price_to_ema_{period}'] = df['close'] / features_df[f'ema_{period}']
            features_df[f'ema_{period}_slope'] = features_df[f'ema_{period}'].diff()
        
        # Convert price arrays to float64 for TA-Lib compatibility
        close_prices = df['close'].values.astype(np.float64)
        high_prices = df['high'].values.astype(np.float64)
        low_prices = df['low'].values.astype(np.float64)
        open_prices = df['open'].values.astype(np.float64)
        
        # MACD variations - important for Gold
        macd, signal, hist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
        features_df['macd'] = macd
        features_df['macd_signal'] = signal
        features_df['macd_hist'] = hist
        features_df['macd_hist_slope'] = features_df['macd_hist'].diff()
        features_df['macd_cross'] = (macd > signal).astype(int)
        
        # Additional MACD timeframes
        macd_fast, signal_fast, hist_fast = talib.MACD(close_prices, fastperiod=8, slowperiod=17, signalperiod=6)
        features_df['macd_fast'] = macd_fast
        features_df['macd_fast_hist'] = hist_fast
        
        # RSI with multiple periods - Gold momentum
        for period in [7, 14, 21, 30]:
            features_df[f'rsi_{period}'] = talib.RSI(close_prices, timeperiod=period)
            features_df[f'rsi_{period}_slope'] = features_df[f'rsi_{period}'].diff()
            features_df[f'rsi_{period}_overbought'] = (features_df[f'rsi_{period}'] > 70).astype(int)
            features_df[f'rsi_{period}_oversold'] = (features_df[f'rsi_{period}'] < 30).astype(int)
        
        # RSI divergence detection
        features_df['rsi_price_div'] = (
            (features_df['rsi_14'].diff() > 0) & (df['close'].diff() < 0)
        ).astype(int) - (
            (features_df['rsi_14'].diff() < 0) & (df['close'].diff() > 0)
        ).astype(int)
        
        # Stochastic oscillator
        slowk, slowd = talib.STOCH(high_prices, low_prices, close_prices)
        features_df['stoch_k'] = slowk
        features_df['stoch_d'] = slowd
        features_df['stoch_cross'] = slowk - slowd
        features_df['stoch_overbought'] = ((slowk > 80) & (slowd > 80)).astype(int)
        features_df['stoch_oversold'] = ((slowk < 20) & (slowd < 20)).astype(int)
        
        # Bollinger Bands - critical for Gold range trading
        for period in [10, 20, 30]:
            upper, middle, lower = talib.BBANDS(close_prices, timeperiod=period, nbdevup=2, nbdevdn=2)
            features_df[f'bb_upper_{period}'] = upper
            features_df[f'bb_lower_{period}'] = lower
            features_df[f'bb_middle_{period}'] = middle
            features_df[f'bb_width_{period}'] = (upper - lower) / middle
            features_df[f'bb_position_{period}'] = (df['close'] - lower) / (upper - lower)
            features_df[f'bb_squeeze_{period}'] = (features_df[f'bb_width_{period}'] < features_df[f'bb_width_{period}'].rolling(20).quantile(0.2)).astype(int)
        
        # ATR and volatility - crucial for Gold
        for period in [14, 20, 30]:
            features_df[f'atr_{period}'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=period)
            features_df[f'atr_{period}_pct'] = features_df[f'atr_{period}'] / df['close']
        
        features_df['atr_ratio'] = features_df['atr_14'] / features_df['atr_30']
        features_df['atr_expansion'] = (features_df['atr_14'] > features_df['atr_14'].rolling(10).mean()).astype(int)
        
        # Momentum indicators
        for period in [10, 20, 30]:
            features_df[f'momentum_{period}'] = talib.MOM(close_prices, timeperiod=period)
            features_df[f'roc_{period}'] = talib.ROC(close_prices, timeperiod=period)
        
        # Williams %R
        features_df['williams_r'] = talib.WILLR(high_prices, low_prices, close_prices)
        features_df['williams_oversold'] = (features_df['williams_r'] < -80).astype(int)
        features_df['williams_overbought'] = (features_df['williams_r'] > -20).astype(int)
        
        # CCI
        features_df['cci'] = talib.CCI(high_prices, low_prices, close_prices)
        features_df['cci_overbought'] = (features_df['cci'] > 100).astype(int)
        features_df['cci_oversold'] = (features_df['cci'] < -100).astype(int)
        
        # ADX for trend strength - very important for Gold
        features_df['adx'] = talib.ADX(high_prices, low_prices, close_prices)
        features_df['plus_di'] = talib.PLUS_DI(high_prices, low_prices, close_prices)
        features_df['minus_di'] = talib.MINUS_DI(high_prices, low_prices, close_prices)
        features_df['adx_strong_trend'] = (features_df['adx'] > 25).astype(int)
        features_df['di_diff'] = features_df['plus_di'] - features_df['minus_di']
        
        # Support/Resistance levels with Gold-specific periods
        for period in [20, 50, 100, 200]:
            features_df[f'resistance_{period}'] = df['high'].rolling(period).max()
            features_df[f'support_{period}'] = df['low'].rolling(period).min()
            features_df[f'price_to_resistance_{period}'] = df['close'] / features_df[f'resistance_{period}']
            features_df[f'price_to_support_{period}'] = df['close'] / features_df[f'support_{period}']
            features_df[f'near_resistance_{period}'] = (features_df[f'price_to_resistance_{period}'] > 0.995).astype(int)
            features_df[f'near_support_{period}'] = (features_df[f'price_to_support_{period}'] < 1.005).astype(int)
        
        # Volume features if available
        if 'volume' in df.columns and not df['volume'].isna().all():
            volume_data = df['volume'].values.astype(np.float64)
            for period in [10, 20, 50]:
                features_df[f'volume_sma_{period}'] = df['volume'].rolling(period).mean()
                features_df[f'volume_ratio_{period}'] = df['volume'] / features_df[f'volume_sma_{period}']
            
            features_df['volume_spike'] = (df['volume'] > df['volume'].rolling(20).mean() * 2).astype(int)
            features_df['obv'] = talib.OBV(close_prices, volume_data)
            features_df['obv_slope'] = features_df['obv'].diff()
            features_df['ad'] = talib.AD(high_prices, low_prices, close_prices, volume_data)
        
        # Gold-specific pattern recognition
        features_df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        features_df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        features_df['inside_bar'] = ((df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))).astype(int)
        features_df['outside_bar'] = ((df['high'] > df['high'].shift(1)) & (df['low'] < df['low'].shift(1))).astype(int)
        
        # Enhanced candlestick analysis for Gold
        features_df['body_size'] = abs(df['close'] - df['open'])
        features_df['upper_shadow'] = df['high'] - np.maximum(df['close'], df['open'])
        features_df['lower_shadow'] = np.minimum(df['close'], df['open']) - df['low']
        features_df['body_to_range'] = features_df['body_size'] / (df['high'] - df['low'] + 1e-8)
        features_df['shadow_ratio'] = (features_df['upper_shadow'] + features_df['lower_shadow']) / features_df['body_size']
        
        # Doji detection
        features_df['doji'] = (features_df['body_size'] < (df['high'] - df['low']) * 0.1).astype(int)
        features_df['hammer'] = (
            (features_df['lower_shadow'] > features_df['body_size'] * 2) &
            (features_df['upper_shadow'] < features_df['body_size'] * 0.5)
        ).astype(int)
        
        # Time features optimized for Gold trading sessions
        if hasattr(df.index, 'hour'):
            features_df['hour'] = df.index.hour
            features_df['day_of_week'] = df.index.dayofweek
            
            # Gold-specific trading sessions
            features_df['asian_session'] = ((df.index.hour >= 23) | (df.index.hour < 8)).astype(int)
            features_df['london_session'] = ((df.index.hour >= 8) & (df.index.hour < 16)).astype(int)
            features_df['ny_session'] = ((df.index.hour >= 13) & (df.index.hour < 22)).astype(int)
            features_df['overlap_london_ny'] = ((df.index.hour >= 13) & (df.index.hour < 16)).astype(int)
            
            # Market day features
            features_df['monday'] = (df.index.dayofweek == 0).astype(int)
            features_df['friday'] = (df.index.dayofweek == 4).astype(int)
            features_df['weekend_gap'] = features_df['monday'] * abs(df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        # Gold-specific economic indicators (proxy)
        features_df['price_acceleration'] = features_df['returns'].diff()
        features_df['trend_consistency'] = features_df['returns'].rolling(10).apply(lambda x: (x > 0).sum() if (x > 0).sum() > (x < 0).sum() else -(x < 0).sum())
        
        # Multi-timeframe analysis
        features_df['short_trend'] = (features_df['sma_5'] > features_df['sma_20']).astype(int)
        features_df['medium_trend'] = (features_df['sma_20'] > features_df['sma_50']).astype(int)
        features_df['long_trend'] = (features_df['sma_50'] > features_df['sma_200']).astype(int)
        features_df['trend_alignment'] = features_df['short_trend'] + features_df['medium_trend'] + features_df['long_trend']
        
        # Fill NaN values
        features_df = features_df.fillna(method='ffill').fillna(0)
        
        return features_df
    
    def create_labels(self, df, lookahead=12, min_profit_dollars=15):
        """Create labels based on Gold-specific price movements (in dollars)"""
        
        labels = []
        long_count = 0
        short_count = 0
        hold_count = 0
        
        print(f"Creating Gold labels with ${min_profit_dollars} minimum profit requirement...")
        
        for i in range(len(df) - lookahead):
            current_price = df['close'].iloc[i]
            
            # Look at future price movements
            future_highs = df['high'].iloc[i+1:i+lookahead+1].values
            future_lows = df['low'].iloc[i+1:i+lookahead+1].values
            exit_price = df['close'].iloc[i + lookahead]
            
            # Calculate maximum favorable excursion in dollars
            max_profit_long = future_highs.max() - current_price
            max_profit_short = current_price - future_lows.min()
            
            # Calculate actual exit profit in dollars
            exit_profit_long = exit_price - current_price
            exit_profit_short = current_price - exit_price
            
            # Calculate risk (potential loss to support/resistance)
            recent_support = df['low'].iloc[max(0, i-20):i+1].min()
            recent_resistance = df['high'].iloc[max(0, i-20):i+1].max()
            
            risk_long = current_price - recent_support
            risk_short = recent_resistance - current_price
            
            # Gold-specific labeling with risk-reward consideration
            if (max_profit_long >= min_profit_dollars and 
                exit_profit_long > min_profit_dollars * 0.3 and  # At least 30% of target
                max_profit_long > max_profit_short and
                risk_long < max_profit_long):  # Positive risk-reward
                labels.append(0)  # Long
                long_count += 1
                
            elif (max_profit_short >= min_profit_dollars and 
                  exit_profit_short > min_profit_dollars * 0.3 and
                  max_profit_short > max_profit_long and
                  risk_short < max_profit_short):  # Positive risk-reward
                labels.append(1)  # Short
                short_count += 1
            else:
                labels.append(2)  # Hold
                hold_count += 1
        
        # Pad the end with hold signals
        labels.extend([2] * lookahead)
        hold_count += lookahead
        
        print(f"Label distribution: Long={long_count}, Short={short_count}, Hold={hold_count}")
        print(f"Signal rate: {(long_count + short_count) / len(labels) * 100:.1f}%")
        
        return np.array(labels)
    
    def prepare_sequences(self, features, labels):
        """Prepare sequences for LSTM training"""
        
        X, y = [], []
        
        for i in range(self.sequence_length, len(features)):
            X.append(features[i-self.sequence_length:i])
            y.append(labels[i])
        
        return np.array(X), np.array(y)
    
    def train(self, data_dict, epochs=150, batch_size=32, learning_rate=0.0001):
        """Train the model on historical Gold data"""
        
        all_features = []
        all_labels = []
        
        print("📊 Preparing Gold training data...")
        
        for symbol, df in data_dict.items():
            print(f"Processing {symbol}...")
            
            # Create Gold-specific features
            features_df = self.create_gold_features(df)
            
            # Create labels with Gold-specific requirements
            labels = self.create_labels(df, lookahead=12, min_profit_dollars=15)
            
            # Convert to numpy
            features = features_df.values
            
            # Create sequences
            X, y = self.prepare_sequences(features, labels)
            
            all_features.append(X)
            all_labels.append(y)
        
        # Combine all data
        X = np.vstack(all_features)
        y = np.hstack(all_labels)
        
        print(f"Combined data: {len(X)} samples")
        print(f"Initial class distribution: Long={np.sum(y==0)}, Short={np.sum(y==1)}, Hold={np.sum(y==2)}")
        
        # Check if we have any data at all
        if len(X) == 0:
            raise ValueError("No training data generated! Check data quality and labeling parameters.")
        
        # Balance classes for Gold trading
        long_mask = y == 0
        short_mask = y == 1
        hold_mask = y == 2
        
        n_long = np.sum(long_mask)
        n_short = np.sum(short_mask)
        n_hold = np.sum(hold_mask)
        
        # If no trading signals, try with lower requirements
        if n_long + n_short == 0:
            print("⚠️ No trading signals found! Reducing profit requirements...")
            all_features = []
            all_labels = []
            
            for symbol, df in data_dict.items():
                print(f"Re-processing {symbol} with lower requirements...")
                features_df = self.create_gold_features(df)
                labels = self.create_labels(df, lookahead=12, min_profit_dollars=10)
                features = features_df.values
                X_pair, y_pair = self.prepare_sequences(features, labels)
                all_features.append(X_pair)
                all_labels.append(y_pair)
            
            X = np.vstack(all_features)
            y = np.hstack(all_labels)
        
        # Balance classes - keep all signals but limit holds
        if n_hold > 0:
            max_hold_samples = max(n_long + n_short, len(X) // 2)  # Allow more holds for Gold
            if n_hold > max_hold_samples:
                hold_indices = np.where(hold_mask)[0]
                selected_hold = np.random.choice(hold_indices, max_hold_samples, replace=False)
                
                keep_mask = (y != 2) | np.isin(np.arange(len(y)), selected_hold)
                X = X[keep_mask]
                y = y[keep_mask]
        
        print(f"Final training samples: {len(X)}")
        print(f"Final class distribution: Long={np.sum(y==0)}, Short={np.sum(y==1)}, Hold={np.sum(y==2)}")
        
        if len(X) == 0:
            raise ValueError("No training samples generated!")
        
        # Normalize features
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_normalized = self.scaler.fit_transform(X_reshaped)
        X = X_normalized.reshape(X.shape)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Create datasets
        train_dataset = GoldPatternDataset(X_train, y_train)
        val_dataset = GoldPatternDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # Initialize model
        input_dim = X.shape[-1]
        self.model = GoldPatternRecognitionNetwork(input_dim).to(device)
        
        # Loss and optimizer
        class_weights = self.calculate_class_weights(y_train)
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
        confidence_criterion = nn.BCELoss()
        
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)
        
        # Training loop
        best_val_acc = 0
        patience_counter = 0
        
        print(f"\n🚀 Starting Gold model training with {len(train_dataset)} training samples...")
        print(f"   Validation samples: {len(val_dataset)}")
        print(f"   Confidence threshold: {self.min_confidence:.1%}")
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_confident_correct = 0
            train_confident_total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for features, labels in pbar:
                features = features.to(device)
                labels = labels.long().to(device)
                
                optimizer.zero_grad()
                logits, confidence = self.model(features)
                
                classification_loss = criterion(logits, labels)
                
                _, predicted = torch.max(logits, 1)
                correct_mask = (predicted == labels).float()
                
                # Target higher confidence for Gold
                target_confidence = correct_mask * 0.85 + (1 - correct_mask) * 0.25
                confidence_loss = confidence_criterion(confidence.squeeze(), target_confidence)
                
                total_loss = classification_loss + 0.15 * confidence_loss
                total_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += total_loss.item()
                train_correct += (predicted == labels).sum().item()
                
                high_conf_mask = confidence.squeeze() > self.min_confidence
                if high_conf_mask.any():
                    train_confident_correct += ((predicted == labels) & high_conf_mask).sum().item()
                    train_confident_total += high_conf_mask.sum().item()
                
                pbar.set_postfix({
                    'loss': total_loss.item(), 
                    'conf_avg': f"{confidence.mean().item():.3f}"
                })
            
            # Validation
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_confident_correct = 0
            val_confident_total = 0
            avg_confidence = 0
            
            with torch.no_grad():
                for features, labels in val_loader:
                    features = features.to(device)
                    labels = labels.long().to(device)
                    
                    logits, confidence = self.model(features)
                    
                    classification_loss = criterion(logits, labels)
                    _, predicted = torch.max(logits, 1)
                    correct_mask = (predicted == labels).float()
                    target_confidence = correct_mask * 0.85 + (1 - correct_mask) * 0.25
                    confidence_loss = confidence_criterion(confidence.squeeze(), target_confidence)
                    
                    total_loss = classification_loss + 0.15 * confidence_loss
                    val_loss += total_loss.item()
                    
                    val_correct += (predicted == labels).sum().item()
                    avg_confidence += confidence.mean().item()
                    
                    high_conf_mask = confidence.squeeze() > self.min_confidence
                    if high_conf_mask.any():
                        val_confident_correct += ((predicted == labels) & high_conf_mask).sum().item()
                        val_confident_total += high_conf_mask.sum().item()
            
            # Calculate metrics
            train_acc = train_correct / len(train_dataset)
            val_acc = val_correct / len(val_dataset)
            avg_confidence = avg_confidence / len(val_loader)
            
            train_conf_acc = train_confident_correct / train_confident_total if train_confident_total > 0 else 0
            val_conf_acc = val_confident_correct / val_confident_total if val_confident_total > 0 else 0
            
            print(f"\nEpoch {epoch+1}:")
            print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2%}, High-Conf Acc: {train_conf_acc:.2%}")
            print(f"  Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.2%}, High-Conf Acc: {val_conf_acc:.2%}")
            print(f"  High-Conf Trades: {val_confident_total}/{len(val_dataset)} ({val_confident_total/len(val_dataset):.1%})")
            print(f"  Avg Confidence: {avg_confidence:.1%}")
            
            scheduler.step(val_loss)
            
            # Gradually increase confidence threshold for Gold
            if epoch > 20 and avg_confidence > 0.65 and self.min_confidence < 0.75:
                self.min_confidence = min(0.75, self.min_confidence + 0.01)
                print(f"  📈 Increased confidence threshold to {self.min_confidence:.1%}")
            
            val_gap = train_acc - val_acc
            combined_score = val_acc + val_conf_acc * 0.4 + avg_confidence * 0.3 - val_gap * 0.3
            
            if combined_score > best_val_acc:
                best_val_acc = combined_score
                patience_counter = 0
                try:
                    self.save_model('best_gold_model.pth')
                    print(f"  💾 Best Gold model saved (Score: {combined_score:.3f})")
                except Exception as e:
                    print(f"  ⚠️ Failed to save model: {e}")
            else:
                patience_counter += 1
                if patience_counter >= 15:
                    print("⏹️ Early stopping triggered")
                    break
        
        print(f"\n✅ Gold model training complete!")
        
        # Load best model
        try:
            self.load_model('best_gold_model.pth')
            print("✅ Best Gold model loaded successfully")
        except Exception as e:
            print(f"⚠️ Failed to load best model: {e}")
    
    def calculate_class_weights(self, labels):
        """Calculate class weights for imbalanced dataset"""
        unique, counts = np.unique(labels, return_counts=True)
        weights = len(labels) / (len(unique) * counts)
        return weights / weights.min()
    
    def calculate_optimal_sl_tp(self, df, action, risk_reward_ratio=2.5):
        """Calculate optimal Stop Loss and Take Profit levels for Gold"""
        
        if len(df) < 50:
            return None, None
        
        current_price = df['close'].iloc[-1]
        
        # Calculate ATR for Gold volatility
        high_prices = df['high'].values[-50:].astype(np.float64)
        low_prices = df['low'].values[-50:].astype(np.float64)
        close_prices = df['close'].values[-50:].astype(np.float64)
        
        atr_14 = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)[-1]
        atr_20 = talib.ATR(high_prices, low_prices, close_prices, timeperiod=20)[-1]
        avg_atr = (atr_14 + atr_20) / 2
        
        # Gold-specific support/resistance
        support_20 = df['low'].rolling(20).min().iloc[-1]
        resistance_20 = df['high'].rolling(20).max().iloc[-1]
        support_50 = df['low'].rolling(50).min().iloc[-1]
        resistance_50 = df['high'].rolling(50).max().iloc[-1]
        
        # Calculate volatility for Gold
        volatility = df['close'].pct_change().rolling(20).std().iloc[-1]
        vol_multiplier = max(2.0, min(4.0, volatility * 200))  # Scale for Gold
        
        if action == 'long':
            # SL options for Gold longs
            sl_atr = current_price - (avg_atr * 2.0)  # Wider stops for Gold
            sl_support = min(support_20, support_50)
            sl_vol = current_price - (avg_atr * vol_multiplier)
            
            # Choose conservative SL
            stop_loss = max(sl_atr, sl_support, sl_vol)
            
            # Ensure reasonable SL for Gold (max 2% loss)
            min_sl = current_price * 0.98
            max_sl = current_price - (avg_atr * 1.0)
            stop_loss = max(min_sl, min(stop_loss, max_sl))
            
            # Calculate TP
            risk_distance = current_price - stop_loss
            take_profit = current_price + (risk_distance * risk_reward_ratio)
            
            # Adjust for resistance
            if take_profit > resistance_20:
                take_profit = resistance_20 * 0.998
                actual_rr = (take_profit - current_price) / risk_distance
                if actual_rr < 1.5:  # Min RR for Gold
                    return None, None
            
        else:  # short
            # SL options for Gold shorts
            sl_atr = current_price + (avg_atr * 2.0)
            sl_resistance = max(resistance_20, resistance_50)
            sl_vol = current_price + (avg_atr * vol_multiplier)
            
            # Choose conservative SL
            stop_loss = min(sl_atr, sl_resistance, sl_vol)
            
            # Ensure reasonable SL for Gold
            max_sl = current_price * 1.02
            min_sl = current_price + (avg_atr * 1.0)
            stop_loss = min(max_sl, max(stop_loss, min_sl))
            
            # Calculate TP
            risk_distance = stop_loss - current_price
            take_profit = current_price - (risk_distance * risk_reward_ratio)
            
            # Adjust for support
            if take_profit < support_20:
                take_profit = support_20 * 1.002
                actual_rr = (current_price - take_profit) / risk_distance
                if actual_rr < 1.5:
                    return None, None
        
        return stop_loss, take_profit

    def predict(self, df, min_confidence=None, risk_reward_ratio=2.5):
        """Make prediction on Gold data with optimal SL/TP levels"""
        
        if min_confidence is None:
            min_confidence = self.min_confidence
        
        # Create Gold-specific features
        features_df = self.create_gold_features(df)
        
        # Get last sequence
        features = features_df.values[-self.sequence_length:]
        
        # Normalize
        features = self.scaler.transform(features.reshape(-1, features.shape[-1])).reshape(features.shape)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            logits, confidence = self.model(features_tensor)
            
        # Get prediction
        _, predicted = torch.max(logits, 1)
        action = ['long', 'short', 'hold'][predicted.item()]
        conf = confidence.item()
        
        # Calculate SL/TP for Gold trading
        stop_loss = None
        take_profit = None
        risk_reward = None
        
        if action != 'hold' and conf >= min_confidence:
            stop_loss, take_profit = self.calculate_optimal_sl_tp(df, action, risk_reward_ratio)
            
            if stop_loss is not None and take_profit is not None:
                current_price = df['close'].iloc[-1]
                if action == 'long':
                    risk_distance = current_price - stop_loss
                    reward_distance = take_profit - current_price
                else:
                    risk_distance = stop_loss - current_price
                    reward_distance = current_price - take_profit
                
                risk_reward = reward_distance / risk_distance if risk_distance > 0 else 0
            else:
                action = 'hold'
        
        if conf < min_confidence:
            action = 'hold'
            stop_loss = None
            take_profit = None
            risk_reward = None
        
        result = {
            'action': action,
            'confidence': conf,
            'probabilities': torch.softmax(logits, dim=1).cpu().numpy()[0]
        }
        
        # Add SL/TP information for Gold
        if stop_loss is not None and take_profit is not None:
            current_price = df['close'].iloc[-1]
            
            result.update({
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward_ratio': risk_reward,
                'risk_dollars': abs(current_price - stop_loss),
                'reward_dollars': abs(take_profit - current_price),
                'sl_distance_percent': abs(current_price - stop_loss) / current_price * 100,
                'tp_distance_percent': abs(take_profit - current_price) / current_price * 100
            })
        
        return result
    
    def save_model(self, filepath):
        """Save Gold model and scaler"""
        import pickle
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'sequence_length': self.sequence_length,
            'min_confidence': self.min_confidence,
            'input_dim': self.model.lstm.input_size
        }, filepath)
        
        scaler_filepath = filepath.replace('.pth', '_scaler.pkl')
        with open(scaler_filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"Gold model saved to {filepath}")
        print(f"Scaler saved to {scaler_filepath}")
        
    def load_model(self, filepath):
        """Load Gold model and scaler"""
        import pickle
        
        try:
            checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        except:
            checkpoint = torch.load(filepath, map_location=device)
        
        input_dim = checkpoint['input_dim']
        self.model = GoldPatternRecognitionNetwork(input_dim).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'scaler' in checkpoint:
            self.scaler = checkpoint['scaler']
        else:
            scaler_filepath = filepath.replace('.pth', '_scaler.pkl')
            try:
                with open(scaler_filepath, 'rb') as f:
                    self.scaler = pickle.load(f)
                print(f"Scaler loaded from {scaler_filepath}")
            except FileNotFoundError:
                print(f"Warning: Scaler file {scaler_filepath} not found. Creating new scaler.")
                self.scaler = StandardScaler()
        
        self.sequence_length = checkpoint['sequence_length']
        self.min_confidence = checkpoint['min_confidence']
        
        print(f"Gold model loaded from {filepath}")

def load_gold_data(period="2y", interval="1h"):
    """Load Gold data from various sources"""
    
    gold_symbols = {
        'Gold_Futures': 'GC=F',
        'Gold_ETF_GLD': 'GLD',
        'Gold_ETF_IAU': 'IAU'
    }
    
    data = {}
    
    for name, symbol in gold_symbols.items():
        print(f"Loading {name} ({symbol})...")
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if not df.empty:
                df.columns = [col.lower() for col in df.columns]
                if len(df) > 500:
                    data[name] = df
                    print(f"✓ {name}: {len(df)} candles loaded")
                    
                    # Gold data quality check
                    price_range = df['close'].max() - df['close'].min()
                    avg_price = df['close'].mean()
                    volatility = df['close'].pct_change().std()
                    print(f"  Price range: ${price_range:.2f} (avg: ${avg_price:.2f})")
                    print(f"  Daily volatility: {volatility:.4f}")
                else:
                    print(f"⚠️ {name}: Insufficient data ({len(df)} candles)")
            else:
                print(f"❌ {name}: No data received")
        except Exception as e:
            print(f"❌ {name}: Error loading data - {str(e)}")
    
    return data

def backtest_gold_predictions(predictor, df, min_confidence=0.65):
    """Backtest Gold predictions with realistic execution"""
    
    results = []
    
    for i in range(predictor.sequence_length, len(df) - 15):
        # Get prediction
        current_data = df.iloc[:i+1]
        pred = predictor.predict(current_data, min_confidence=min_confidence)
        
        if pred['action'] != 'hold':
            entry_price = df['close'].iloc[i]
            
            if 'stop_loss' in pred and 'take_profit' in pred:
                stop_loss = pred['stop_loss']
                take_profit = pred['take_profit']
                
                # Check each future candle for SL/TP hits
                final_profit = 0
                for j in range(1, min(16, len(df) - i)):
                    high_price = df['high'].iloc[i + j]
                    low_price = df['low'].iloc[i + j]
                    
                    if pred['action'] == 'long':
                        if high_price >= take_profit:
                            final_profit = take_profit - entry_price
                            break
                        elif low_price <= stop_loss:
                            final_profit = stop_loss - entry_price
                            break
                    else:  # short
                        if low_price <= take_profit:
                            final_profit = entry_price - take_profit
                            break
                        elif high_price >= stop_loss:
                            final_profit = entry_price - stop_loss
                            break
                else:
                    # Exit at end of period
                    exit_price = df['close'].iloc[i + 15]
                    if pred['action'] == 'long':
                        final_profit = exit_price - entry_price
                    else:
                        final_profit = entry_price - exit_price
                
                # Calculate max profit for comparison
                future_prices = df['close'].iloc[i+1:i+16].values
                if pred['action'] == 'long':
                    max_profit = future_prices.max() - entry_price
                else:
                    max_profit = entry_price - future_prices.min()
                
                results.append({
                    'action': pred['action'],
                    'confidence': pred['confidence'],
                    'max_profit': max_profit,
                    'final_profit': final_profit,
                    'win': final_profit > 0,
                    'entry_price': entry_price
                })
    
    if results:
        wins = sum(1 for r in results if r['win'])
        win_rate = wins / len(results)
        avg_confidence = np.mean([r['confidence'] for r in results])
        avg_profit = np.mean([r['final_profit'] for r in results])
        total_profit = sum(r['final_profit'] for r in results)
        
        print(f"\n📊 Gold Backtest Results (min confidence: {min_confidence:.1%}):")
        print(f"  Total Trades: {len(results)}")
        print(f"  Win Rate: {win_rate:.1%}")
        print(f"  Average Confidence: {avg_confidence:.1%}")
        print(f"  Average Max Profit: ${np.mean([r['max_profit'] for r in results]):.2f}")
        print(f"  Average Final Profit: ${avg_profit:.2f}")
        print(f"  Total Profit: ${total_profit:.2f}")
        print(f"  Expected Value: ${avg_profit:.2f} per trade")
    
    return results

def demonstrate_gold_sl_tp_system(predictor, data_dict):
    """Demonstrate the Gold SL/TP calculation system"""
    
    print("\n🥇 GOLD STOP LOSS & TAKE PROFIT DEMONSTRATION")
    print("="*70)
    
    for name, df in data_dict.items():
        print(f"\n📊 {name} Analysis:")
        
        prediction = predictor.predict(df, min_confidence=predictor.min_confidence, risk_reward_ratio=2.5)
        
        if prediction['action'] != 'hold':
            current_price = df['close'].iloc[-1]
            
            print(f"  Action: {prediction['action'].upper()}")
            print(f"  Confidence: {prediction['confidence']:.1%}")
            print(f"  Entry Price: ${current_price:.2f}")
            
            if 'stop_loss' in prediction:
                print(f"  Stop Loss: ${prediction['stop_loss']:.2f}")
                print(f"  Take Profit: ${prediction['take_profit']:.2f}")
                print(f"  Risk/Reward: 1:{prediction['risk_reward_ratio']:.2f}")
                print(f"  Risk: ${prediction['risk_dollars']:.2f} ({prediction['sl_distance_percent']:.2f}%)")
                print(f"  Reward: ${prediction['reward_dollars']:.2f} ({prediction['tp_distance_percent']:.2f}%)")
                
                print(f"  Potential Profit: +${prediction['reward_dollars']:.2f}")
                print(f"  Potential Loss: -${prediction['risk_dollars']:.2f}")
        else:
            print(f"  Action: HOLD (Confidence: {prediction['confidence']:.1%})")
        
        print("-" * 50)

if __name__ == "__main__":
    print("="*80)
    print("🥇 SUPERVISED LEARNING GOLD PREDICTOR - HIGH WIN RATE SYSTEM")
    print("="*80)
    
    # Load Gold data
    print("\n📊 Loading Gold data...")
    data = load_gold_data(period="2y", interval="1h")
    
    if not data:
        print("❌ No Gold data loaded")
        print("💡 Suggested alternatives:")
        print("   - GC=F (Gold Futures)")
        print("   - GLD (SPDR Gold Trust ETF)")
        print("   - IAU (iShares Gold Trust ETF)")
        exit(1)
    
    # Initialize Gold predictor
    predictor = SupervisedGoldPredictor()
    
    # Train model
    print("\n🚀 Training Gold pattern recognition model...")
    predictor.train(data, epochs=150, batch_size=32, learning_rate=0.0001)
    
    # Demonstrate SL/TP system
    demonstrate_gold_sl_tp_system(predictor, data)
    
    # Test on each Gold instrument
    print("\n🧪 Testing high-confidence Gold predictions...")
    for name, df in data.items():
        print(f"\n{name}:")
        # Use last 20% of data for testing
        test_start = int(len(df) * 0.8)
        test_df = df.iloc[test_start:]
        
        # Test with Gold-specific confidence thresholds
        for min_conf in [0.60, 0.70, 0.80]:
            results = backtest_gold_predictions(predictor, test_df, min_confidence=min_conf)
    
    print("\n✅ Gold model training complete!")
    print("   🥇 Specialized for Gold market characteristics")
    print("   📈 Optimized for Gold volatility and trends")
    print("   🎯 Higher confidence thresholds for Gold precision")
    print("   💰 Dollar-based profit targets and risk management")
