"""
ENHANCED LOSS-LEARNING TRAINER
Extended training with more epochs and better monitoring
"""

import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import ta
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class EnhancedLossLearningTrainer:
    """Enhanced trainer with extended epochs and better monitoring"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.training_histories = {}
        
        print("🚀 ENHANCED LOSS-LEARNING TRAINER INITIALIZED")
        print("Features: Extended epochs, better callbacks, validation monitoring")
    
    def download_data(self, pair, period='1y'):
        """Download training data"""
        print(f"\n📈 Downloading {pair} data...")
        
        try:
            ticker = yf.Ticker(pair)
            data = ticker.history(period=period, interval='1h')
            
            if len(data) < 1000:
                print(f"❌ Insufficient data for {pair}")
                return None
            
            print(f"✅ Downloaded {len(data)} hours of data")
            return data
            
        except Exception as e:
            print(f"❌ Failed to download {pair}: {e}")
            return None
    
    def create_features(self, data):
        """Create comprehensive technical features"""
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
        
        # Market structure (boolean features)
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
        
        return df
    
    def simulate_trade_outcome(self, data, entry_idx, direction, pair_name=None):
        """Simulate trade outcome with TP/SL"""
        
        if entry_idx >= len(data) - 20:
            return None
        
        entry_price = data.iloc[entry_idx]['Close']
        
        # Set TP/SL levels based on pair type
        if pair_name and 'JPY' in pair_name:
            tp_pips = 40 * 0.01
            sl_pips = 20 * 0.01
        else:
            tp_pips = 40 * 0.0001
            sl_pips = 20 * 0.0001
        
        if direction == 'LONG':
            tp_price = entry_price + tp_pips
            sl_price = entry_price - sl_pips
        else:
            tp_price = entry_price - tp_pips
            sl_price = entry_price + sl_pips
        
        # Check next 20 bars for TP/SL
        for i in range(1, min(21, len(data) - entry_idx)):
            bar = data.iloc[entry_idx + i]
            
            if direction == 'LONG':
                if bar['Low'] <= sl_price:
                    return {'result': 'loss', 'bars_held': i, 'exit_price': sl_price}
                elif bar['High'] >= tp_price:
                    return {'result': 'win', 'bars_held': i, 'exit_price': tp_price}
            else:
                if bar['High'] >= sl_price:
                    return {'result': 'loss', 'bars_held': i, 'exit_price': sl_price}
                elif bar['Low'] <= tp_price:
                    return {'result': 'win', 'bars_held': i, 'exit_price': tp_price}
        
        # Timeout
        final_price = data.iloc[entry_idx + min(20, len(data) - entry_idx - 1)]['Close']
        return {'result': 'timeout', 'bars_held': 20, 'exit_price': final_price}
    
    def should_signal(self, outcome, market_conditions, direction):
        """Determine if AI should signal based on outcome and conditions"""
        
        # Only signal if trade would be profitable
        if outcome['result'] != 'win':
            return False
        
        # Quality filters
        if direction == 'LONG':
            good_conditions = (
                not market_conditions['overbought'] and
                market_conditions['macd_bullish'] and
                market_conditions['bb_position'] < 0.8
            )
        else:
            good_conditions = (
                not market_conditions['oversold'] and
                market_conditions['macd_bearish'] and
                market_conditions['bb_position'] > 0.2
            )
        
        trend_ok = market_conditions['strong_trend'] or market_conditions['neutral_rsi']
        return good_conditions and trend_ok
    
    def calculate_confidence(self, outcome, market_conditions, direction):
        """Calculate dynamic confidence"""
        
        confidence = 0.5
        
        if outcome['result'] == 'win':
            confidence += 0.3
            if outcome['bars_held'] <= 5:
                confidence += 0.1
        
        # Market condition adjustments
        if direction == 'LONG':
            if market_conditions['trend_up'] and market_conditions['strong_trend']:
                confidence += 0.15
            if market_conditions['macd_bullish']:
                confidence += 0.1
        else:
            if market_conditions['trend_down'] and market_conditions['strong_trend']:
                confidence += 0.15
            if market_conditions['macd_bearish']:
                confidence += 0.1
        
        return min(confidence, 1.0)
    
    def calculate_reward(self, outcome, direction, entry_price):
        """Calculate reward based on outcome"""
        
        if outcome['result'] == 'win':
            return 2.0  # 2:1 reward ratio
        elif outcome['result'] == 'loss':
            return -1.0
        else:
            # Timeout - small penalty for indecision
            return -0.1
    
    def generate_training_data(self, pair, num_samples=15000):
        """Generate enhanced training dataset"""
        
        print(f"\n🔄 Generating training data for {pair}...")
        
        data = self.download_data(pair, period='2y')  # More data for better training
        if data is None:
            return None, None, None, None
        
        features_df = self.create_features(data)
        features_df = features_df.dropna()
        
        if len(features_df) < 500:
            print(f"❌ Insufficient feature data for {pair}")
            return None, None, None, None
        
        print(f"📊 Creating {num_samples} training examples...")
        
        sequences = []
        signals = []
        confidences = []
        rewards = []
        
        sequence_length = 30
        long_count = 0
        short_count = 0
        
        for i in range(sequence_length, len(features_df) - 25):
            
            if len(sequences) >= num_samples:
                break
            
            # Get sequence and current bar
            sequence = features_df.iloc[i-sequence_length:i]
            current_bar = features_df.iloc[i]
            
            # Try both directions
            for direction in ['LONG', 'SHORT']:
                
                # Balance directions
                if direction == 'LONG' and long_count >= num_samples // 2:
                    continue
                if direction == 'SHORT' and short_count >= num_samples // 2:
                    continue
                
                # Simulate trade outcome
                outcome = self.simulate_trade_outcome(features_df, i, direction, pair)
                if outcome is None:
                    continue
                
                # Market conditions
                market_conditions = {
                    'trend_up': bool(current_bar['trend_up']),
                    'trend_down': bool(current_bar['trend_down']),
                    'strong_trend': bool(current_bar['strong_trend']),
                    'weak_trend': bool(current_bar['weak_trend']),
                    'oversold': bool(current_bar['oversold']),
                    'overbought': bool(current_bar['overbought']),
                    'neutral_rsi': bool(current_bar['neutral_rsi']),
                    'macd_bullish': bool(current_bar['macd_bullish']),
                    'macd_bearish': bool(current_bar['macd_bearish']),
                    'bb_position': float(current_bar['bb_position'])
                }
                
                # Generate labels
                should_signal = self.should_signal(outcome, market_conditions, direction)
                confidence = self.calculate_confidence(outcome, market_conditions, direction)
                reward = self.calculate_reward(outcome, direction, current_bar['Close'])
                
                # Feature columns
                feature_cols = ['Close', 'returns', 'volatility', 'sma_20', 'sma_50', 'ema_12',
                               'macd', 'macd_signal', 'macd_histogram', 'rsi', 'bb_position', 'adx',
                               'trend_up', 'trend_down', 'strong_trend', 'oversold', 'overbought',
                               'macd_bullish', 'macd_bearish']
                
                available_cols = [col for col in feature_cols if col in sequence.columns]
                
                if len(available_cols) >= 15:  # Ensure sufficient features
                    sequences.append(sequence[available_cols].values)
                    signals.append(1.0 if should_signal else 0.0)
                    confidences.append(confidence)
                    rewards.append(reward)
                    
                    if direction == 'LONG':
                        long_count += 1
                    else:
                        short_count += 1
        
        if len(sequences) < 100:
            print(f"❌ Insufficient training examples for {pair}")
            return None, None, None, None
        
        print(f"✅ Generated {len(sequences)} examples")
        print(f"   LONG: {long_count}, SHORT: {short_count}")
        print(f"   Signal rate: {(sum(signals) / len(signals)):.1%}")
        
        return (np.array(sequences), np.array(signals), 
                np.array(confidences), np.array(rewards))
    
    def create_enhanced_model(self, input_shape):
        """Create enhanced neural network with better architecture"""
        
        inputs = Input(shape=input_shape)
        
        # Enhanced LSTM layers with more capacity
        x = LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(inputs)
        x = BatchNormalization()(x)
        x = LSTM(96, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(x)
        x = BatchNormalization()(x)
        x = LSTM(64, dropout=0.3, recurrent_dropout=0.3)(x)
        x = BatchNormalization()(x)
        
        # Shared dense layers
        shared = Dense(128, activation='relu')(x)
        shared = Dropout(0.4)(shared)
        shared = Dense(64, activation='relu')(shared)
        shared = Dropout(0.3)(shared)
        
        # Output branches
        signal_output = Dense(32, activation='relu', name='signal_dense')(shared)
        signal_output = Dropout(0.2)(signal_output)
        signal_output = Dense(1, activation='sigmoid', name='signal_prob')(signal_output)
        
        confidence_output = Dense(32, activation='relu', name='confidence_dense')(shared)
        confidence_output = Dropout(0.2)(confidence_output)
        confidence_output = Dense(1, activation='sigmoid', name='confidence')(confidence_output)
        
        reward_output = Dense(32, activation='relu', name='reward_dense')(shared)
        reward_output = Dropout(0.2)(reward_output)
        reward_output = Dense(1, activation='tanh', name='reward_estimate')(reward_output)
        
        model = Model(inputs=inputs, outputs=[signal_output, confidence_output, reward_output])
        
        # Enhanced compilation with better metrics
        model.compile(
            optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
            loss={
                'signal_prob': 'binary_crossentropy',
                'confidence': 'mse',
                'reward_estimate': 'mse'
            },
            loss_weights={
                'signal_prob': 2.0,    # Increased weight for signal accuracy
                'confidence': 1.0,
                'reward_estimate': 1.5  # Increased weight for reward prediction
            },
            metrics={
                'signal_prob': ['accuracy'],
                'confidence': ['mae'],
                'reward_estimate': ['mae']
            }
        )
        
        return model
    
    def train_enhanced_model(self, pair):
        """Train model with extended epochs and enhanced monitoring"""
        
        print(f"\n🎯 ENHANCED TRAINING: {pair}")
        print("=" * 50)
        
        # Generate training data
        X, y_signal, y_confidence, y_reward = self.generate_training_data(pair)
        
        if X is None:
            print(f"❌ Failed to generate training data for {pair}")
            return False
        
        # Create and scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        
        # Create enhanced model
        model = self.create_enhanced_model((X.shape[1], X.shape[2]))
        
        print(f"🏗️ Model Architecture:")
        print(f"   Input shape: {X.shape}")
        print(f"   LSTM layers: 128→96→64 units")
        print(f"   Dense layers: 128→64→32 units")
        print(f"   Total parameters: {model.count_params():,}")
        
        # Create enhanced_models directory
        import os
        os.makedirs('enhanced_models', exist_ok=True)
        
        # Enhanced callbacks with extended patience
        callbacks = [
            EarlyStopping(
                monitor='val_loss', 
                patience=15,  # Increased patience
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=7,  # Increased patience
                min_lr=0.00001,
                verbose=1
            ),
            ModelCheckpoint(
                f'enhanced_models/{pair}_best_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        print(f"\n🚀 Training with EXTENDED EPOCHS...")
        print(f"   Epochs: 75 (vs previous 30)")
        print(f"   Early stopping patience: 15 (vs previous 8)")
        print(f"   Learning rate patience: 7 (vs previous 4)")
        
        # Train with extended epochs
        history = model.fit(
            X_scaled,
            [y_signal, y_confidence, y_reward],
            epochs=75,  # Increased from 30
            batch_size=32,
            validation_split=0.25,  # Increased validation split
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )
        
        # Store results with converted pair name
        pair_key = pair.replace('=', '_')
        self.models[pair_key] = model
        self.scalers[pair_key] = scaler
        self.training_histories[pair_key] = history
        
        # Enhanced results analysis
        final_loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]
        signal_accuracy = history.history['signal_prob_accuracy'][-1]
        val_signal_accuracy = history.history['val_signal_prob_accuracy'][-1]
        
        print(f"\n📊 ENHANCED TRAINING RESULTS:")
        print(f"   Final Training Loss: {final_loss:.4f}")
        print(f"   Final Validation Loss: {val_loss:.4f}")
        print(f"   Signal Accuracy: {signal_accuracy:.1%}")
        print(f"   Val Signal Accuracy: {val_signal_accuracy:.1%}")
        print(f"   Epochs Completed: {len(history.history['loss'])}")
        print(f"   Improvement vs Previous: TBD after testing")
        
        # Check for overfitting
        overfitting_gap = abs(signal_accuracy - val_signal_accuracy)
        if overfitting_gap > 0.05:
            print(f"   ⚠️ Potential overfitting detected (gap: {overfitting_gap:.1%})")
        else:
            print(f"   ✅ Good generalization (gap: {overfitting_gap:.1%})")
        
        return True
    
    def save_enhanced_models(self):
        """Save enhanced models"""
        
        print(f"\n💾 Saving enhanced models...")
        
        import os
        os.makedirs('enhanced_models', exist_ok=True)
        
        for pair in self.models:
            try:
                model_data = {
                    'model': self.models[pair],
                    'scaler': self.scalers[pair],
                    'training_history': self.training_histories[pair].history if pair in self.training_histories else None
                }
                
                filepath = f"enhanced_models/{pair}_enhanced_loss_learning.pkl"
                with open(filepath, 'wb') as f:
                    pickle.dump(model_data, f)
                
                print(f"✅ Saved {pair}")
                
            except Exception as e:
                print(f"❌ Failed to save {pair}: {e}")
    
    def train_all_enhanced_models(self):
        """Train all pairs with enhanced settings"""
        
        print("🚀 ENHANCED LOSS-LEARNING TRAINING SESSION")
        print("=" * 60)
        print("🔧 ENHANCEMENTS:")
        print("   • Extended epochs: 30 → 75")
        print("   • Increased model capacity: 64→96→128 LSTM units")
        print("   • Better callbacks with longer patience")
        print("   • Enhanced architecture with BatchNorm")
        print("   • Improved loss weighting")
        print("   • More training data (2 years vs 1 year)")
        print("=" * 60)
        
        pairs = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X']
        successful_pairs = []
        
        for pair in pairs:
            # Convert pair format for training
            training_pair = pair.replace('=', '_')
            success = self.train_enhanced_model(pair)  # Use original format for download
            if success:
                successful_pairs.append(pair)
        
        if successful_pairs:
            self.save_enhanced_models()
            
            print(f"\n🏆 ENHANCED TRAINING COMPLETE!")
            print(f"✅ Successfully trained: {', '.join(successful_pairs)}")
            print(f"📈 Expected improvements:")
            print(f"   • Higher accuracy (target: 95%+ vs current 90%+)")
            print(f"   • Better confidence calibration")
            print(f"   • Improved signal quality")
            print(f"   • More robust pattern recognition")
            print(f"\n💡 Next step: Run enhanced backtest to compare results!")
        
        else:
            print(f"❌ No models trained successfully")

if __name__ == "__main__":
    trainer = EnhancedLossLearningTrainer()
    trainer.train_all_enhanced_models()
