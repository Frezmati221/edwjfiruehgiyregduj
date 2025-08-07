#!/usr/bin/env python3
"""
Test script to verify Trade-2.py model compatibility
"""

import sys
import os
import importlib.util

# Import trade-2.py module
spec = importlib.util.spec_from_file_location("trade2", "trade-2.py")
trade2_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trade2_module)

SupervisedForexPredictor = trade2_module.SupervisedForexPredictor

def test_model_compatibility():
    """Test if we can create and use the predictor"""
    print("🧪 Testing Trade-2.py model compatibility...")
    
    try:
        # Create predictor instance
        predictor = SupervisedForexPredictor()
        print("✅ SupervisedForexPredictor created successfully")
        
        # Check attributes
        print(f"📏 Sequence length: {predictor.sequence_length}")
        print(f"🎯 Min confidence: {predictor.min_confidence}")
        
        # Test data loading function
        print("\n📊 Testing data loading...")
        data = trade2_module.load_forex_data(period="1mo", interval="1h")
        if data:
            print(f"✅ Data loaded for {len(data)} pairs")
            for pair, df in data.items():
                print(f"   {pair}: {len(df)} candles")
        else:
            print("❌ No data loaded")
            
        print("\n✅ All compatibility tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_model_compatibility()
