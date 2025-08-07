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

class ForexPatternDataset(Dataset):
    """Dataset for supervised learning of forex patterns"""
    
    def __init__(self, features, labels, sequence_length=60):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class PatternRecognitionNetwork(nn.Module):
    """Advanced pattern recognition network for high win-rate predictions"""
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.4):
        super().__init__()
        
        # Smaller, more regularized LSTM
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # Attention mechanism for important patterns
        self.attention = nn.MultiheadAttention(
            hidden_dim * 2,  # bidirectional
            num_heads=4,  # Reduced heads
            dropout=dropout
        )
        
        # Simplified classifier with more dropout
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout + 0.1),  # Extra dropout
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, 3)  # 3 classes: long, short, hold
        )
        
        # Confidence estimator with regularization
        self.confidence = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
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

class SupervisedForexPredictor:
    """Supervised learning system for high win-rate forex trading"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.sequence_length = 60  # Look at 60 candles of history
        self.min_confidence = 0.5  # Start with lower threshold, increase gradually
        
    def create_advanced_features(self, df):
        """Create comprehensive technical features for pattern recognition"""
        
        features_df = pd.DataFrame(index=df.index)
        
        # Price features
        features_df['returns'] = df['close'].pct_change()
        features_df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        features_df['high_low_ratio'] = df['high'] / df['low']
        features_df['close_open_ratio'] = df['close'] / df['open']
        
        # Volatility features
        features_df['volatility_10'] = features_df['returns'].rolling(10).std()
        features_df['volatility_20'] = features_df['returns'].rolling(20).std()
        features_df['volatility_ratio'] = features_df['volatility_10'] / features_df['volatility_20']
        
        # Multiple timeframe SMAs
        for period in [5, 10, 20, 50, 100, 200]:
            features_df[f'sma_{period}'] = df['close'].rolling(period).mean()
            features_df[f'sma_{period}_slope'] = features_df[f'sma_{period}'].diff()
            features_df[f'price_to_sma_{period}'] = df['close'] / features_df[f'sma_{period}']
        
        # EMAs
        for period in [12, 26, 50]:
            features_df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            features_df[f'price_to_ema_{period}'] = df['close'] / features_df[f'ema_{period}']
        
        # Convert price arrays to float64 for TA-Lib compatibility
        close_prices = df['close'].values.astype(np.float64)
        high_prices = df['high'].values.astype(np.float64)
        low_prices = df['low'].values.astype(np.float64)
        open_prices = df['open'].values.astype(np.float64)
        
        # MACD variations
        macd, signal, hist = talib.MACD(close_prices)
        features_df['macd'] = macd
        features_df['macd_signal'] = signal
        features_df['macd_hist'] = hist
        features_df['macd_hist_slope'] = features_df['macd_hist'].diff()
        
        # RSI with multiple periods
        for period in [7, 14, 21]:
            features_df[f'rsi_{period}'] = talib.RSI(close_prices, timeperiod=period)
            features_df[f'rsi_{period}_slope'] = features_df[f'rsi_{period}'].diff()
        
        # Stochastic oscillator
        slowk, slowd = talib.STOCH(high_prices, low_prices, close_prices)
        features_df['stoch_k'] = slowk
        features_df['stoch_d'] = slowd
        features_df['stoch_cross'] = slowk - slowd
        
        # Bollinger Bands
        for period in [10, 20, 30]:
            upper, middle, lower = talib.BBANDS(close_prices, timeperiod=period)
            features_df[f'bb_upper_{period}'] = upper
            features_df[f'bb_lower_{period}'] = lower
            features_df[f'bb_width_{period}'] = upper - lower
            features_df[f'bb_position_{period}'] = (df['close'] - lower) / (upper - lower)
        
        # ATR and volatility
        features_df['atr_14'] = talib.ATR(high_prices, low_prices, close_prices)
        features_df['atr_20'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=20)
        features_df['atr_ratio'] = features_df['atr_14'] / features_df['atr_20']
        
        # Momentum indicators
        features_df['momentum_10'] = talib.MOM(close_prices, timeperiod=10)
        features_df['momentum_20'] = talib.MOM(close_prices, timeperiod=20)
        features_df['roc_10'] = talib.ROC(close_prices, timeperiod=10)
        
        # Williams %R
        features_df['williams_r'] = talib.WILLR(high_prices, low_prices, close_prices)
        
        # CCI
        features_df['cci'] = talib.CCI(high_prices, low_prices, close_prices)
        
        # ADX for trend strength
        features_df['adx'] = talib.ADX(high_prices, low_prices, close_prices)
        features_df['plus_di'] = talib.PLUS_DI(high_prices, low_prices, close_prices)
        features_df['minus_di'] = talib.MINUS_DI(high_prices, low_prices, close_prices)
        
        # Support/Resistance levels
        for period in [20, 50, 100]:
            features_df[f'resistance_{period}'] = df['high'].rolling(period).max()
            features_df[f'support_{period}'] = df['low'].rolling(period).min()
            features_df[f'price_to_resistance_{period}'] = df['close'] / features_df[f'resistance_{period}']
            features_df[f'price_to_support_{period}'] = df['close'] / features_df[f'support_{period}']
        
        # Volume features if available
        if 'volume' in df.columns and not df['volume'].isna().all():
            volume_data = df['volume'].values.astype(np.float64)
            features_df['volume_sma_20'] = df['volume'].rolling(20).mean()
            features_df['volume_ratio'] = df['volume'] / features_df['volume_sma_20']
            features_df['obv'] = talib.OBV(close_prices, volume_data)
            features_df['ad'] = talib.AD(high_prices, low_prices, close_prices, volume_data)
        
        # Pattern recognition helpers
        features_df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        features_df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        features_df['inside_bar'] = ((df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))).astype(int)
        
        # Candlestick patterns
        features_df['body_size'] = abs(df['close'] - df['open'])
        features_df['upper_shadow'] = df['high'] - np.maximum(df['close'], df['open'])
        features_df['lower_shadow'] = np.minimum(df['close'], df['open']) - df['low']
        features_df['body_to_range'] = features_df['body_size'] / (df['high'] - df['low'])
        
        # Time features (for session analysis)
        if hasattr(df.index, 'hour'):
            features_df['hour'] = df.index.hour
            features_df['day_of_week'] = df.index.dayofweek
            features_df['is_london'] = ((df.index.hour >= 8) & (df.index.hour < 16)).astype(int)
            features_df['is_newyork'] = ((df.index.hour >= 13) & (df.index.hour < 21)).astype(int)
            features_df['is_tokyo'] = ((df.index.hour >= 0) & (df.index.hour < 8)).astype(int)
        
        # Fill NaN values
        features_df = features_df.fillna(method='ffill').fillna(0)
        
        return features_df
    
    def create_labels(self, df, lookahead=10, min_profit_pips=8):
        """Create labels based on realistic future price movements"""
        
        labels = []
        long_count = 0
        short_count = 0
        hold_count = 0
        
        print(f"Creating labels with {min_profit_pips} pip minimum profit requirement...")
        
        for i in range(len(df) - lookahead):
            current_price = df['close'].iloc[i]
            
            # Look at future price movements
            future_highs = df['high'].iloc[i+1:i+lookahead+1].values
            future_lows = df['low'].iloc[i+1:i+lookahead+1].values
            exit_price = df['close'].iloc[i + lookahead]
            
            # Calculate maximum favorable excursion
            max_profit_long = (future_highs.max() - current_price) / self.get_pip_value(current_price)
            max_profit_short = (current_price - future_lows.min()) / self.get_pip_value(current_price)
            
            # Calculate actual exit profit
            exit_profit_long = (exit_price - current_price) / self.get_pip_value(current_price)
            exit_profit_short = (current_price - exit_price) / self.get_pip_value(current_price)
            
            # More permissive labeling - focus on directional accuracy
            if (max_profit_long >= min_profit_pips and 
                exit_profit_long > 0 and 
                max_profit_long > max_profit_short):
                labels.append(0)  # Long
                long_count += 1
                
            elif (max_profit_short >= min_profit_pips and 
                  exit_profit_short > 0 and 
                  max_profit_short > max_profit_long):
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
    
    def get_pip_value(self, price):
        """Get pip value based on price (simplified)"""
        if price > 50:  # Likely JPY pair
            return 0.01
        else:
            return 0.0001
    
    def prepare_sequences(self, features, labels):
        """Prepare sequences for LSTM training"""
        
        X, y = [], []
        
        for i in range(self.sequence_length, len(features)):
            X.append(features[i-self.sequence_length:i])
            y.append(labels[i])
        
        return np.array(X), np.array(y)
    
    def train(self, data_dict, epochs=100, batch_size=32, learning_rate=0.0001):
        """Train the model on historical data"""
        
        all_features = []
        all_labels = []
        
        print("📊 Preparing training data...")
        
        for pair, df in data_dict.items():
            print(f"Processing {pair}...")
            
            # Create features
            features_df = self.create_advanced_features(df)
            
            # Create labels (more realistic requirements)
            labels = self.create_labels(df, lookahead=10, min_profit_pips=8)
            
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
        
        # Keep all data but balance classes better
        long_mask = y == 0
        short_mask = y == 1
        hold_mask = y == 2
        
        # Count each class
        n_long = np.sum(long_mask)
        n_short = np.sum(short_mask)
        n_hold = np.sum(hold_mask)
        
        # Only proceed if we have some trading signals
        if n_long + n_short == 0:
            print("⚠️ No trading signals found! Reducing profit requirements...")
            # Try with even lower requirements
            all_features = []
            all_labels = []
            
            for pair, df in data_dict.items():
                print(f"Re-processing {pair} with lower requirements...")
                features_df = self.create_advanced_features(df)
                labels = self.create_labels(df, lookahead=10, min_profit_pips=5)
                features = features_df.values
                X_pair, y_pair = self.prepare_sequences(features, labels)
                all_features.append(X_pair)
                all_labels.append(y_pair)
            
            X = np.vstack(all_features)
            y = np.hstack(all_labels)
            
            if len(X) == 0:
                raise ValueError("Still no training data! Check your data sources.")
        
        # Balance classes - keep all signals but limit holds
        if n_hold > 0:
            max_hold_samples = max(n_long + n_short, len(X) // 3)
            if n_hold > max_hold_samples:
                hold_indices = np.where(hold_mask)[0]
                selected_hold = np.random.choice(hold_indices, max_hold_samples, replace=False)
                
                # Keep all trading signals + selected holds
                keep_mask = (y != 2) | np.isin(np.arange(len(y)), selected_hold)
                X = X[keep_mask]
                y = y[keep_mask]
        
        print(f"Final training samples: {len(X)}")
        print(f"Final class distribution: Long={np.sum(y==0)}, Short={np.sum(y==1)}, Hold={np.sum(y==2)}")
        
        # Check if we have enough training data
        if len(X) < 100:
            print("⚠️ Warning: Very few training samples. Consider:")
            print("   - Reducing min_profit_pips further")
            print("   - Increasing data period")
            print("   - Checking data quality")
            print(f"   - Current data size: {len(X)} samples")
            
        if len(X) == 0:
            raise ValueError("No training samples generated!")
        
        # Check if we have any trading signals
        trading_signals = np.sum(y != 2)
        if trading_signals == 0:
            print("⚠️ Warning: No trading signals found! Model will only learn to hold.")
        else:
            print(f"✓ Found {trading_signals} trading signals ({trading_signals/len(X)*100:.1f}% of data)")
        
        # Normalize features
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_normalized = self.scaler.fit_transform(X_reshaped)
        X = X_normalized.reshape(X.shape)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Create datasets
        train_dataset = ForexPatternDataset(X_train, y_train)
        val_dataset = ForexPatternDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # Initialize model
        input_dim = X.shape[-1]
        self.model = PatternRecognitionNetwork(input_dim).to(device)
        
        # Loss and optimizer
        # Use weighted loss to handle class imbalance
        class_weights = self.calculate_class_weights(y_train)
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
        
        # Confidence loss to encourage higher confidence predictions
        confidence_criterion = nn.BCELoss()
        
        # Use stronger regularization
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training loop
        best_val_acc = 0
        patience_counter = 0
        
        print(f"\n🚀 Starting training with {len(train_dataset)} training samples...")
        print(f"   Validation samples: {len(val_dataset)}")
        print(f"   Confidence threshold: {self.min_confidence:.1%}")
        print(f"   Target: Gradually increase confidence and accuracy")
        
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
                
                # Classification loss
                classification_loss = criterion(logits, labels)
                
                # Confidence loss - encourage higher confidence for correct predictions
                _, predicted = torch.max(logits, 1)
                correct_mask = (predicted == labels).float()
                
                # Target confidence should be high for correct predictions, moderate for incorrect
                target_confidence = correct_mask * 0.8 + (1 - correct_mask) * 0.3
                confidence_loss = confidence_criterion(confidence.squeeze(), target_confidence)
                
                # Combined loss
                total_loss = classification_loss + 0.1 * confidence_loss
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()
                
                train_loss += total_loss.item()
                
                # Calculate accuracy
                train_correct += (predicted == labels).sum().item()
                
                # Calculate high-confidence accuracy
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
                    
                    # Classification loss
                    classification_loss = criterion(logits, labels)
                    
                    # Confidence loss
                    _, predicted = torch.max(logits, 1)
                    correct_mask = (predicted == labels).float()
                    target_confidence = correct_mask * 0.8 + (1 - correct_mask) * 0.3
                    confidence_loss = confidence_criterion(confidence.squeeze(), target_confidence)
                    
                    total_loss = classification_loss + 0.1 * confidence_loss
                    val_loss += total_loss.item()
                    
                    val_correct += (predicted == labels).sum().item()
                    avg_confidence += confidence.mean().item()
                    
                    # High-confidence accuracy
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
            
            # Gradually increase confidence threshold (more conservative)
            if epoch > 15 and avg_confidence > 0.55 and self.min_confidence < 0.65:
                self.min_confidence = min(0.65, self.min_confidence + 0.015)
                print(f"  📈 Increased confidence threshold to {self.min_confidence:.1%}")
            
            # More strict early stopping based on generalization gap
            val_gap = train_acc - val_acc  # Gap between train and validation
            combined_score = val_acc + val_conf_acc * 0.3 + avg_confidence * 0.2 - val_gap * 0.5
            
            if combined_score > best_val_acc:
                best_val_acc = combined_score
                patience_counter = 0
                # Save best model
                try:
                    self.save_model('best_model.pth')
                    print(f"  💾 Best model saved (Score: {combined_score:.3f}, Gap: {val_gap:.2%})")
                except Exception as e:
                    print(f"  ⚠️ Failed to save model: {e}")
            else:
                patience_counter += 1
                if patience_counter >= 10:  # Earlier stopping
                    print("⏹️ Early stopping triggered")
                    break
        
        print(f"\n✅ Training complete! Best validation high-confidence accuracy: {best_val_acc:.2%}")
        
        # Load best model
        try:
            self.load_model('best_model.pth')
            print("✅ Best model loaded successfully")
        except Exception as e:
            print(f"⚠️ Failed to load best model: {e}")
            print("   Using current model state")
    
    def calculate_class_weights(self, labels):
        """Calculate class weights for imbalanced dataset"""
        unique, counts = np.unique(labels, return_counts=True)
        weights = len(labels) / (len(unique) * counts)
        return weights / weights.min()
    
    def calculate_optimal_sl_tp(self, df, action, risk_reward_ratio=2.0):
        """Calculate optimal Stop Loss and Take Profit levels"""
        
        if len(df) < 50:
            return None, None
        
        current_price = df['close'].iloc[-1]
        
        # Calculate ATR for dynamic SL/TP
        high_prices = df['high'].values[-50:].astype(np.float64)
        low_prices = df['low'].values[-50:].astype(np.float64)
        close_prices = df['close'].values[-50:].astype(np.float64)
        
        atr_14 = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)[-1]
        atr_20 = talib.ATR(high_prices, low_prices, close_prices, timeperiod=20)[-1]
        
        # Use average ATR for more stable SL/TP
        avg_atr = (atr_14 + atr_20) / 2
        
        # Calculate support/resistance levels
        support_20 = df['low'].rolling(20).min().iloc[-1]
        resistance_20 = df['high'].rolling(20).max().iloc[-1]
        support_50 = df['low'].rolling(50).min().iloc[-1]
        resistance_50 = df['high'].rolling(50).max().iloc[-1]
        
        # Calculate volatility-based SL distance
        volatility = df['close'].pct_change().rolling(20).std().iloc[-1]
        vol_multiplier = max(1.5, min(3.0, volatility * 100))  # Scale volatility
        
        if action == 'long':
            # For long positions
            # SL options: ATR-based, support levels, or volatility-based
            sl_atr = current_price - (avg_atr * 1.5)
            sl_support = min(support_20, support_50)
            sl_vol = current_price - (current_price * volatility * vol_multiplier)
            
            # Choose the most conservative (highest) SL
            stop_loss = max(sl_atr, sl_support, sl_vol)
            
            # Ensure SL is reasonable (not too close or too far)
            min_sl = current_price - (current_price * 0.03)  # Max 3% loss
            max_sl = current_price - (avg_atr * 0.8)  # Min ATR*0.8 distance
            stop_loss = max(min_sl, min(stop_loss, max_sl))
            
            # Calculate TP based on risk-reward ratio
            risk_distance = current_price - stop_loss
            take_profit = current_price + (risk_distance * risk_reward_ratio)
            
            # Adjust TP if it hits resistance
            if take_profit > resistance_20:
                # Use resistance as TP and recalculate risk-reward
                take_profit = resistance_20 * 0.995  # Slightly below resistance
                actual_rr = (take_profit - current_price) / risk_distance
                if actual_rr < 1.2:  # If RR becomes too low, skip trade
                    return None, None
            
        else:  # short
            # For short positions
            # SL options: ATR-based, resistance levels, or volatility-based
            sl_atr = current_price + (avg_atr * 1.5)
            sl_resistance = max(resistance_20, resistance_50)
            sl_vol = current_price + (current_price * volatility * vol_multiplier)
            
            # Choose the most conservative (lowest) SL
            stop_loss = min(sl_atr, sl_resistance, sl_vol)
            
            # Ensure SL is reasonable
            max_sl = current_price + (current_price * 0.03)  # Max 3% loss
            min_sl = current_price + (avg_atr * 0.8)  # Min ATR*0.8 distance
            stop_loss = min(max_sl, max(stop_loss, min_sl))
            
            # Calculate TP based on risk-reward ratio
            risk_distance = stop_loss - current_price
            take_profit = current_price - (risk_distance * risk_reward_ratio)
            
            # Adjust TP if it hits support
            if take_profit < support_20:
                # Use support as TP and recalculate risk-reward
                take_profit = support_20 * 1.005  # Slightly above support
                actual_rr = (current_price - take_profit) / risk_distance
                if actual_rr < 1.2:  # If RR becomes too low, skip trade
                    return None, None
        
        return stop_loss, take_profit

    def predict(self, df, min_confidence=None, risk_reward_ratio=2.0):
        """Make prediction on new data with optimal SL/TP levels"""
        
        if min_confidence is None:
            min_confidence = self.min_confidence
        
        # Create features
        features_df = self.create_advanced_features(df)
        
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
        
        # Calculate SL/TP for trading actions
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
                
                # Convert to pips for display
                pip_value = self.get_pip_value(current_price)
                risk_pips = risk_distance / pip_value
                reward_pips = reward_distance / pip_value
            else:
                # If we can't calculate good SL/TP, don't trade
                action = 'hold'
        
        # Only trade if confidence is high
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
        
        # Add SL/TP information if available
        if stop_loss is not None and take_profit is not None:
            current_price = df['close'].iloc[-1]
            pip_value = self.get_pip_value(current_price)
            
            result.update({
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward_ratio': risk_reward,
                'risk_pips': abs(current_price - stop_loss) / pip_value,
                'reward_pips': abs(take_profit - current_price) / pip_value,
                'sl_distance_percent': abs(current_price - stop_loss) / current_price * 100,
                'tp_distance_percent': abs(take_profit - current_price) / current_price * 100
            })
        
        return result
    
    def save_model(self, filepath):
        """Save model and scaler"""
        import pickle
        
        # Save model state dict separately
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'sequence_length': self.sequence_length,
            'min_confidence': self.min_confidence,
            'input_dim': self.model.lstm.input_size
        }, filepath)
        
        # Save scaler separately using pickle
        scaler_filepath = filepath.replace('.pth', '_scaler.pkl')
        with open(scaler_filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"Model saved to {filepath}")
        print(f"Scaler saved to {scaler_filepath}")
        
    def load_model(self, filepath):
        """Load model and scaler"""
        import pickle
        
        # Load with weights_only=False for compatibility
        try:
            checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        except:
            # Fallback for older PyTorch versions or if weights_only fails
            checkpoint = torch.load(filepath, map_location=device)
        
        # Recreate model architecture
        input_dim = checkpoint['input_dim']
        self.model = PatternRecognitionNetwork(input_dim).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Try to load scaler from checkpoint first (old format)
        if 'scaler' in checkpoint:
            self.scaler = checkpoint['scaler']
        else:
            # Load scaler from separate file (new format)
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
        
        print(f"Model loaded from {filepath}")

def load_forex_data(period="2y", interval="1h"):
    """Load forex data from Yahoo Finance"""
    
    pairs = {
        'EURUSD': 'EURUSD=X',
        'GBPUSD': 'GBPUSD=X',
        'USDJPY': 'USDJPY=X',
        'AUDUSD': 'AUDUSD=X',
        'USDCAD': 'USDCAD=X',
        'USDCHF': 'USDCHF=X'
    }
    
    data = {}
    
    for pair, symbol in pairs.items():
        print(f"Loading {pair}...")
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if not df.empty:
                df.columns = [col.lower() for col in df.columns]
                if len(df) > 500:
                    data[pair] = df
                    print(f"✓ {pair}: {len(df)} candles loaded")
                    
                    # Quick data quality check
                    price_range = df['close'].max() - df['close'].min()
                    avg_price = df['close'].mean()
                    volatility = df['close'].pct_change().std()
                    print(f"  Price range: {price_range:.5f} (avg: {avg_price:.5f})")
                    print(f"  Volatility: {volatility:.4f}")
                else:
                    print(f"⚠️ {pair}: Insufficient data ({len(df)} candles)")
            else:
                print(f"❌ {pair}: No data received")
        except Exception as e:
            print(f"❌ {pair}: Error loading data - {str(e)}")
    
    return data

def backtest_high_confidence(predictor, df, min_confidence=0.6):
    """Backtest only high-confidence predictions with realistic SL/TP"""
    
    results = []
    
    for i in range(predictor.sequence_length, len(df) - 10):
        # Get prediction
        current_data = df.iloc[:i+1]
        pred = predictor.predict(current_data, min_confidence=min_confidence)
        
        if pred['action'] != 'hold':
            # Simulate realistic trade with SL/TP
            entry_price = df['close'].iloc[i]
            
            # Use SL/TP if available, otherwise simple exit
            if 'stop_loss' in pred and 'take_profit' in pred:
                stop_loss = pred['stop_loss']
                take_profit = pred['take_profit']
                
                # Check each future candle for SL/TP hits
                final_profit = 0
                for j in range(1, min(11, len(df) - i)):
                    high_price = df['high'].iloc[i + j]
                    low_price = df['low'].iloc[i + j]
                    
                    if pred['action'] == 'long':
                        # Check TP first (more realistic)
                        if high_price >= take_profit:
                            final_profit = (take_profit - entry_price) / predictor.get_pip_value(entry_price)
                            break
                        # Then check SL
                        elif low_price <= stop_loss:
                            final_profit = (stop_loss - entry_price) / predictor.get_pip_value(entry_price)
                            break
                    else:  # short
                        # Check TP first
                        if low_price <= take_profit:
                            final_profit = (entry_price - take_profit) / predictor.get_pip_value(entry_price)
                            break
                        # Then check SL
                        elif high_price >= stop_loss:
                            final_profit = (entry_price - stop_loss) / predictor.get_pip_value(entry_price)
                            break
                else:
                    # Exit at end of period if no SL/TP hit
                    exit_price = df['close'].iloc[i + 10]
                    if pred['action'] == 'long':
                        final_profit = (exit_price - entry_price) / predictor.get_pip_value(entry_price)
                    else:
                        final_profit = (entry_price - exit_price) / predictor.get_pip_value(entry_price)
            else:
                # Fallback to simple exit
                exit_price = df['close'].iloc[i + 10]
                if pred['action'] == 'long':
                    final_profit = (exit_price - entry_price) / predictor.get_pip_value(entry_price)
                else:
                    final_profit = (entry_price - exit_price) / predictor.get_pip_value(entry_price)
            
            # Calculate max profit for comparison
            future_prices = df['close'].iloc[i+1:i+11].values
            if pred['action'] == 'long':
                max_profit = (future_prices.max() - entry_price) / predictor.get_pip_value(entry_price)
            else:
                max_profit = (entry_price - future_prices.min()) / predictor.get_pip_value(entry_price)
            
            results.append({
                'action': pred['action'],
                'confidence': pred['confidence'],
                'max_profit': max_profit,
                'final_profit': final_profit,
                'win': final_profit > 0
            })
    
    if results:
        wins = sum(1 for r in results if r['win'])
        win_rate = wins / len(results)
        avg_confidence = np.mean([r['confidence'] for r in results])
        avg_profit = np.mean([r['final_profit'] for r in results])
        
        print(f"\n📊 Backtest Results (min confidence: {min_confidence:.1%}):")
        print(f"  Total Trades: {len(results)}")
        print(f"  Win Rate: {win_rate:.1%}")
        print(f"  Average Confidence: {avg_confidence:.1%}")
        print(f"  Average Max Profit: {np.mean([r['max_profit'] for r in results]):.1f} pips")
        print(f"  Average Final Profit: {avg_profit:.1f} pips")
        print(f"  Expected Value: {avg_profit:.2f} pips per trade")
    
    return results

def demonstrate_sl_tp_system(predictor, data_dict):
    """Demonstrate the SL/TP calculation system"""
    
    print("\n🎯 STOP LOSS & TAKE PROFIT DEMONSTRATION")
    print("="*60)
    
    for pair, df in data_dict.items():
        print(f"\n📊 {pair} Analysis:")
        
        # Get prediction with SL/TP (use current model's confidence threshold)
        prediction = predictor.predict(df, min_confidence=predictor.min_confidence, risk_reward_ratio=2.0)
        
        if prediction['action'] != 'hold':
            current_price = df['close'].iloc[-1]
            
            print(f"  Action: {prediction['action'].upper()}")
            print(f"  Confidence: {prediction['confidence']:.1%}")
            print(f"  Entry Price: {current_price:.5f}")
            
            if 'stop_loss' in prediction:
                print(f"  Stop Loss: {prediction['stop_loss']:.5f}")
                print(f"  Take Profit: {prediction['take_profit']:.5f}")
                print(f"  Risk/Reward: 1:{prediction['risk_reward_ratio']:.2f}")
                print(f"  Risk: {prediction['risk_pips']:.1f} pips ({prediction['sl_distance_percent']:.2f}%)")
                print(f"  Reward: {prediction['reward_pips']:.1f} pips ({prediction['tp_distance_percent']:.2f}%)")
                
                # Calculate potential profit/loss
                if prediction['action'] == 'long':
                    profit_if_tp = prediction['reward_pips']
                    loss_if_sl = -prediction['risk_pips']
                else:
                    profit_if_tp = prediction['reward_pips']
                    loss_if_sl = -prediction['risk_pips']
                
                print(f"  Potential Profit: +{profit_if_tp:.1f} pips")
                print(f"  Potential Loss: {loss_if_sl:.1f} pips")
        else:
            print(f"  Action: HOLD (Confidence: {prediction['confidence']:.1%})")
        
        print("-" * 40)

if __name__ == "__main__":
    print("="*80)
    print("🎯 SUPERVISED LEARNING FOREX PREDICTOR - HIGH WIN RATE SYSTEM")
    print("="*80)
    
    # Load data
    print("\n📊 Loading forex data...")
    data = load_forex_data(period="2y", interval="1h")
    
    if not data:
        print("❌ No data loaded")
        exit(1)
    
    # Initialize predictor
    predictor = SupervisedForexPredictor()
    
    # Train model
    print("\n🚀 Training pattern recognition model...")
    predictor.train(data, epochs=100, batch_size=64, learning_rate=0.0001)
    
    # Demonstrate SL/TP system
    demonstrate_sl_tp_system(predictor, data)
    
    # Test on each pair
    print("\n🧪 Testing high-confidence predictions...")
    for pair, df in data.items():
        print(f"\n{pair}:")
        # Use last 20% of data for testing
        test_start = int(len(df) * 0.8)
        test_df = df.iloc[test_start:]
        
        # Test with more realistic confidence thresholds
        for min_conf in [0.55, 0.65, 0.75]:
            results = backtest_high_confidence(predictor, test_df, min_confidence=min_conf)
    
    print("\n✅ Training complete! Model focuses on HIGH-CONFIDENCE patterns only.")
    print("   The system will only trade when pattern confidence exceeds threshold.")
    print("   This approach prioritizes WIN RATE over trade frequency.")
    print("   🎯 Now includes dynamic SL/TP calculation with 2:1 risk-reward ratio!")
