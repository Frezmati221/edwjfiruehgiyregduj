#!/usr/bin/env python3
"""
Compatible Pattern Recognition Network for loading saved models
This version matches the architecture of the saved best_model.pth
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler

class CompatiblePatternRecognitionNetwork(nn.Module):
    """Compatible pattern recognition network that matches saved model architecture"""
    
    def __init__(self, input_dim=95, hidden_dim=256, num_layers=4, dropout=0.4):
        super().__init__()
        
        # LSTM with architecture matching saved model
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # Attention mechanism (4 heads, embed_dim = hidden_dim * 2 for bidirectional)
        self.attention = nn.MultiheadAttention(
            hidden_dim * 2,  # 512 for bidirectional 256
            num_heads=4,
            dropout=dropout
        )
        
        # Classifier with exact original architecture (matching saved indices)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # 0: 512 -> 256
            nn.BatchNorm1d(hidden_dim),             # 1: 256
            nn.ReLU(),                              # 2
            nn.Dropout(dropout + 0.1),             # 3
            
            nn.Linear(hidden_dim, hidden_dim // 2), # 4: 256 -> 128  
            nn.BatchNorm1d(hidden_dim // 2),        # 5: 128
            nn.ReLU(),                              # 6
            nn.Dropout(dropout),                    # 7
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),  # 8: 128 -> 64
            nn.BatchNorm1d(hidden_dim // 4),              # 9: 64
            nn.ReLU(),                                     # 10
            nn.Linear(hidden_dim // 4, 3)                 # 11: 64 -> 3 classes
        )
        
        # Confidence estimator with exact original architecture
        self.confidence = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),  # 0: 512 -> 64
            nn.ReLU(),                      # 1
            nn.Linear(64, 1),               # 2: 64 -> 1  
            nn.Sigmoid()                    # 3
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

class CompatibleSupervisedForexPredictor:
    """Compatible version of SupervisedForexPredictor for loading saved models"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.sequence_length = 60
        self.min_confidence = 0.7
        
    def load_model(self, filepath):
        """Load the compatible model"""
        import pickle
        
        # Load checkpoint
        try:
            checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        except:
            checkpoint = torch.load(filepath, map_location='cpu')
        
        # Get architecture parameters from checkpoint
        input_dim = checkpoint['input_dim']
        self.sequence_length = checkpoint['sequence_length']
        self.min_confidence = checkpoint['min_confidence']
        
        print(f"Loading model with architecture:")
        print(f"  Input dim: {input_dim}")
        print(f"  Sequence length: {self.sequence_length}")
        print(f"  Min confidence: {self.min_confidence}")
        
        # Create model with correct architecture
        self.model = CompatiblePatternRecognitionNetwork(input_dim=input_dim)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load scaler
        scaler_filepath = filepath.replace('.pth', '_scaler.pkl')
        try:
            with open(scaler_filepath, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"Scaler loaded from {scaler_filepath}")
        except FileNotFoundError:
            print(f"Warning: Scaler file {scaler_filepath} not found")
            self.scaler = StandardScaler()
        
        print(f"✅ Compatible model loaded successfully")
    
    def create_advanced_features(self, df):
        """Create comprehensive technical features for pattern recognition"""
        
        import talib
        import pandas as pd
        import numpy as np
        
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
        features_df = features_df.ffill().fillna(0)
        
        return features_df
    
    def get_pip_value(self, price):
        """Get pip value based on price"""
        if price > 50:  # Likely JPY pair
            return 0.01
        else:
            return 0.0001
    
    def calculate_optimal_sl_tp(self, df, action, risk_reward_ratio=2.0):
        """Calculate optimal Stop Loss and Take Profit levels"""
        
        import talib
        import numpy as np
        
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
        """Make prediction on new data"""
        if min_confidence is None:
            min_confidence = self.min_confidence
        
        # Create features using built-in method (no more device prints)
        features_df = self.create_advanced_features(df)
        
        # Get last sequence
        features = features_df.values[-self.sequence_length:]
        
        # Normalize
        features = self.scaler.transform(features.reshape(-1, features.shape[-1])).reshape(features.shape)
        
        # Convert to tensor (get device once)
        device = next(self.model.parameters()).device
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

if __name__ == "__main__":
    # Test the compatible predictor
    print("🧪 Testing Compatible Predictor...")
    
    predictor = CompatibleSupervisedForexPredictor()
    try:
        predictor.load_model("best_model.pth")
        print("✅ Compatible model loaded successfully!")
        
        # Test with some dummy data
        import pandas as pd
        import yfinance as yf
        
        print("\n📊 Testing prediction...")
        ticker = yf.Ticker("EURUSD=X")
        df = ticker.history(period="1mo", interval="1h")
        df.columns = [col.lower() for col in df.columns]
        
        if len(df) > 60:
            prediction = predictor.predict(df, min_confidence=0.6)
            print(f"✅ Prediction: {prediction['action']} (confidence: {prediction['confidence']:.1%})")
        else:
            print("⚠️ Not enough data for prediction test")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
