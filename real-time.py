from flask import Flask, jsonify, request
import pandas as pd
from datetime import datetime
from forex_trading_model import ForexPredictor, TradingAction
from forex_data_loader import ForexDataLoader
import threading
import time

app = Flask(__name__)

class RealTimePredictor:
    """Real-time forex prediction service"""
    
    def __init__(self):
        self.predictor = ForexPredictor()
        self.loader = ForexDataLoader()
        self.predictions = {}
        self.is_running = False
        
        # Load pre-trained model
        try:
            self.predictor.load_model('forex_model.pth')
            print("Model loaded successfully")
        except:
            print("No pre-trained model found. Please train first.")
    
    def start_predictions(self):
        """Start real-time prediction loop"""
        self.is_running = True
        threading.Thread(target=self._prediction_loop, daemon=True).start()
    
    def _prediction_loop(self):
        """Continuous prediction loop"""
        while self.is_running:
            for pair in ['EURUSD', 'GBPUSD', 'USDJPY']:
                try:
                    # Get latest data
                    data = self.loader.load_historical_data(pair, period='5d', interval='5m')
                    
                    # Make prediction
                    action = self.predictor.predict(pair, data)
                    
                    self.predictions[pair] = {
                        'timestamp': datetime.now().isoformat(),
                        'direction': action.direction,
                        'take_profit': action.take_profit,
                        'stop_loss': action.stop_loss,
                        'confidence': action.confidence,
                        'current_price': float(data['close'].iloc[-1])
                    }
                    
                except Exception as e:
                    print(f"Error predicting {pair}: {e}")
            
            time.sleep(60)  # Update every minute
    
    def get_prediction(self, pair: str):
        """Get latest prediction for a pair"""
        return self.predictions.get(pair, None)

# Initialize predictor
rt_predictor = RealTimePredictor()
rt_predictor.start_predictions()

@app.route('/predict/<pair>', methods=['GET'])
def predict(pair):
    """API endpoint for predictions"""
    pair = pair.upper()
    prediction = rt_predictor.get_prediction(pair)
    
    if prediction:
        return jsonify({
            'status': 'success',
            'pair': pair,
            'prediction': prediction
        })
    else:
        return jsonify({
            'status': 'error',
            'message': f'No prediction available for {pair}'
        }), 404

@app.route('/predict/all', methods=['GET'])
def predict_all():
    """Get predictions for all pairs"""
    return jsonify({
        'status': 'success',
        'predictions': rt_predictor.predictions
    })

@app.route('/train', methods=['POST'])
def train():
    """Trigger model training"""
    try:
        # Load fresh data
        loader = ForexDataLoader()
        data = loader.load_all_pairs(period='1y', interval='1h')
        
        # Train model
        rt_predictor.predictor.train(data, epochs=100)
        rt_predictor.predictor.save_model('forex_model.pth')
        
        return jsonify({
            'status': 'success',
            'message': 'Model trained successfully'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/backtest/<pair>', methods=['GET'])
def backtest(pair):
    """Run backtest for a specific pair"""
    from forex_data_loader import Backtester
    
    try:
        pair = pair.upper()
        loader = ForexDataLoader()
        data = loader.load_historical_data(pair, period='3mo', interval='1h')
        
        backtester = Backtester()
        metrics = backtester.run_backtest(rt_predictor.predictor, data, pair)
        
        return jsonify({
            'status': 'success',
            'pair': pair,
            'metrics': metrics
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    print("Starting Forex Prediction API...")
    print("Available endpoints:")
    print("  GET  /predict/<pair>     - Get prediction for specific pair")
    print("  GET  /predict/all        - Get all predictions")
    print("  POST /train              - Train the model")
    print("  GET  /backtest/<pair>    - Run backtest for pair")
    
    app.run(host='0.0.0.0', port=5000, debug=False)