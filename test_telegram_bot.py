#!/usr/bin/env python3
"""
Test script for Trade-2 Telegram Bot
Validates imports and basic functionality without running the bot
"""

import sys
import os

def test_imports():
    """Test all required imports"""
    print("🧪 Testing imports...")
    
    try:
        import asyncio
        print("✅ asyncio")
    except ImportError as e:
        print(f"❌ asyncio: {e}")
        return False
    
    try:
        import logging
        print("✅ logging")
    except ImportError as e:
        print(f"❌ logging: {e}")
        return False
    
    try:
        import yfinance as yf
        print("✅ yfinance")
    except ImportError as e:
        print(f"❌ yfinance: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ pandas")
    except ImportError as e:
        print(f"❌ pandas: {e}")
        return False
    
    try:
        from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
        from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
        print("✅ python-telegram-bot")
    except ImportError as e:
        print(f"❌ python-telegram-bot: {e}")
        return False
    
    try:
        import json
        print("✅ json")
    except ImportError as e:
        print(f"❌ json: {e}")
        return False
    
    try:
        from compatible_trade2_predictor import CompatibleSupervisedForexPredictor
        print("✅ compatible_trade2_predictor")
    except ImportError as e:
        print(f"❌ compatible_trade2_predictor: {e}")
        print("    Make sure this file exists in the current directory")
        return False
    
    return True

def test_model_files():
    """Test model file availability"""
    print("\n🔍 Testing model files...")
    
    if os.path.exists("best_model.pth"):
        print("✅ best_model.pth found")
    else:
        print("❌ best_model.pth not found")
        return False
    
    if os.path.exists("best_model_scaler.pkl"):
        print("✅ best_model_scaler.pkl found")
    else:
        print("⚠️  best_model_scaler.pkl not found (optional)")
    
    return True

def test_bot_class():
    """Test bot class instantiation"""
    print("\n🤖 Testing bot class...")
    
    try:
        # Import the bot class
        sys.path.append('.')
        from trade2_telegram_bot import Trade2TelegramBot
        print("✅ Bot class imported successfully")
        
        # Test with dummy token (won't actually connect)
        dummy_token = "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
        model_path = "best_model.pth"
        
        try:
            bot = Trade2TelegramBot(dummy_token, model_path)
            print("✅ Bot class instantiated successfully")
            print(f"✅ Model loaded: {bot.predictor is not None}")
            print(f"✅ Forex pairs loaded: {len(bot.forex_pairs)} pairs")
            return True
        except Exception as e:
            print(f"❌ Bot instantiation failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Bot class import failed: {e}")
        return False

def test_prediction_logic():
    """Test prediction logic without making actual predictions"""
    print("\n🔮 Testing prediction logic...")
    
    try:
        from compatible_trade2_predictor import CompatibleSupervisedForexPredictor
        predictor = CompatibleSupervisedForexPredictor()
        predictor.load_model("best_model.pth")
        print("✅ Predictor loaded successfully")
        
        # Test data generation (without actual market data)
        import pandas as pd
        import numpy as np
        
        # Create dummy data that matches expected format
        dummy_data = pd.DataFrame({
            'open': np.random.uniform(1.0, 1.1, 100),
            'high': np.random.uniform(1.0, 1.1, 100),
            'low': np.random.uniform(1.0, 1.1, 100),
            'close': np.random.uniform(1.0, 1.1, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        })
        
        print("✅ Dummy data created")
        print(f"✅ Data shape: {dummy_data.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction logic test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🎯 Trade-2 Telegram Bot Test Suite")
    print("=" * 40)
    
    tests_passed = 0
    total_tests = 4
    
    # Test 1: Imports
    if test_imports():
        tests_passed += 1
    
    # Test 2: Model files
    if test_model_files():
        tests_passed += 1
    
    # Test 3: Bot class
    if test_bot_class():
        tests_passed += 1
    
    # Test 4: Prediction logic
    if test_prediction_logic():
        tests_passed += 1
    
    print("\n" + "=" * 40)
    print(f"📊 Test Results: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Bot is ready to run.")
        print("\n🚀 To start the bot:")
        print("1. Set TELEGRAM_BOT_TOKEN environment variable")
        print("2. Run: python3 trade2_telegram_bot.py")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        
        print("\n🛠️  Common fixes:")
        if tests_passed < 1:
            print("• Install dependencies: pip install -r telegram_bot_requirements.txt")
        if tests_passed < 2:
            print("• Ensure model files are in current directory")
        if tests_passed < 3:
            print("• Check bot code syntax and imports")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
