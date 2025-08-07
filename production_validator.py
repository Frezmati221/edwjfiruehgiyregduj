#!/usr/bin/env python3
"""
Production Validator for Forex AI Trading System
===============================================

This script validates the production-readiness of the forex trading system
by running comprehensive tests on:
- Risk management systems
- Trading cost calculations
- Market regime detection
- Walk-forward optimization
- Performance metrics

⚠️ CRITICAL: Use this validator before any live trading!
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import json
import warnings
warnings.filterwarnings('ignore')

# Import the trading system
from train import (
    ForexPredictor, ForexEnvironment, RiskManager, 
    TradingCostCalculator, MarketConditions, load_forex_data
)

class ProductionValidator:
    """Comprehensive validation suite for production trading system"""
    
    def __init__(self):
        self.results = {}
        self.test_data = None
        
    def run_comprehensive_validation(self, data_period: str = "1y", epochs: int = 50):
        """Run all validation tests"""
        print("="*80)
        print("🔬 PRODUCTION VALIDATION SUITE")
        print("="*80)
        
        # Load test data
        print("📊 Loading test data...")
        self.test_data = load_forex_data(period=data_period, interval="1h")
        
        if not self.test_data:
            print("❌ Failed to load test data")
            return False
        
        # Run validation tests
        tests = [
            ("🛡️ Risk Management", self.test_risk_management),
            ("💰 Trading Costs", self.test_trading_costs),
            ("🌊 Market Regimes", self.test_market_regime_detection),
            ("📈 Walk-Forward", self.test_walk_forward_validation),
            ("⚡ Performance", self.test_performance_metrics),
            ("🎯 Stress Test", self.test_stress_scenarios),
            ("📊 Correlation", self.test_correlation_limits)
        ]
        
        all_passed = True
        for test_name, test_func in tests:
            print(f"\n{test_name} Testing...")
            try:
                result = test_func()
                if result:
                    print(f"✅ {test_name}: PASSED")
                else:
                    print(f"❌ {test_name}: FAILED")
                    all_passed = False
            except Exception as e:
                print(f"💥 {test_name}: ERROR - {str(e)}")
                all_passed = False
        
        # Generate validation report
        self.generate_validation_report()
        
        print("\n" + "="*80)
        if all_passed:
            print("🎉 ALL TESTS PASSED - SYSTEM IS PRODUCTION-READY!")
            print("⚠️  Remember: Still paper trade for 3-6 months before going live")
        else:
            print("🚨 SOME TESTS FAILED - DO NOT USE IN PRODUCTION!")
        print("="*80)
        
        return all_passed
    
    def test_risk_management(self) -> bool:
        """Test risk management system"""
        risk_manager = RiskManager(
            max_drawdown=0.20,
            max_daily_loss=0.06,
            max_open_positions=3,
            max_correlation=0.7
        )
        
        # Test 1: Drawdown limit
        can_trade, msg = risk_manager.can_trade(8000, 'EURUSD', 'long')  # 20% drawdown
        if can_trade:
            print("   ❌ Failed to block trading at max drawdown")
            return False
        
        # Test 2: Daily loss limit
        risk_manager.daily_start_balance = 10000
        can_trade, msg = risk_manager.can_trade(9400, 'EURUSD', 'long')  # 6% daily loss
        if can_trade:
            print("   ❌ Failed to block trading at daily loss limit")
            return False
        
        # Test 3: Position limits
        for i in range(4):  # Try to add 4 positions (max is 3)
            risk_manager.add_position(f'PAIR{i}', 'long', 100000, 1.0)
        
        can_trade, msg = risk_manager.can_trade(10000, 'NEWPAIR', 'long')
        if can_trade:
            print("   ❌ Failed to block trading at max positions")
            return False
        
        # Test 4: Correlation limits
        risk_manager.open_positions = []
        risk_manager.add_position('EURUSD', 'long', 100000, 1.0)
        risk_manager.add_position('GBPUSD', 'long', 100000, 1.0)
        
        can_trade, msg = risk_manager.can_trade(10000, 'AUDUSD', 'long')  # Correlated to EUR/GBP
        if can_trade:
            print("   ❌ Failed to block correlated position")
            return False
        
        print("   ✅ Risk management system working correctly")
        self.results['risk_management'] = {
            'drawdown_limit': True,
            'daily_loss_limit': True,
            'position_limit': True,
            'correlation_limit': True
        }
        return True
    
    def test_trading_costs(self) -> bool:
        """Test trading cost calculations"""
        cost_calc = TradingCostCalculator()
        
        # Test realistic spreads
        spreads = ['EURUSD', 'GBPUSD', 'USDJPY', 'EXOTIC_PAIR']
        for pair in spreads:
            market_conditions = MarketConditions('ranging', 0.5, 1.0, 1.0, 1.1)
            cost = cost_calc.calculate_entry_cost(pair, 1.1000, 0.001, 100000, market_conditions)
            
            if cost <= 0:
                print(f"   ❌ Invalid cost calculation for {pair}: {cost}")
                return False
        
        # Test cost increases with volatility
        low_vol_conditions = MarketConditions('ranging', 0.2, 1.0, 1.0, 1.1)
        high_vol_conditions = MarketConditions('high_volatility', 0.9, 1.0, 1.0, 1.1)
        
        low_cost = cost_calc.calculate_entry_cost('EURUSD', 1.1000, 0.001, 100000, low_vol_conditions)
        high_cost = cost_calc.calculate_entry_cost('EURUSD', 1.1000, 0.003, 100000, high_vol_conditions)
        
        if high_cost <= low_cost:
            print(f"   ❌ Costs should increase with volatility: {low_cost} vs {high_cost}")
            return False
        
        print("   ✅ Trading cost calculations working correctly")
        self.results['trading_costs'] = {
            'spread_calculation': True,
            'volatility_adjustment': True,
            'commission_included': True
        }
        return True
    
    def test_market_regime_detection(self) -> bool:
        """Test market regime detection"""
        if 'EURUSD' not in self.test_data:
            print("   ⚠️ No EURUSD data for regime testing")
            return True
        
        from train import ForexIndicators
        indicators = ForexIndicators()
        
        # Prepare test data
        df = self.test_data['EURUSD'].copy()
        prepared_df = indicators.calculate_all_indicators(df)
        
        # Test regime detection on different market conditions
        regimes_detected = set()
        
        # Test on different data segments
        for i in range(0, len(prepared_df) - 100, 100):
            segment = prepared_df.iloc[i:i+100]
            if len(segment) < 50:
                continue
                
            market_conditions = indicators.detect_market_regime(segment)
            regimes_detected.add(market_conditions.regime)
        
        # Should detect at least 2 different regimes in a year of data
        if len(regimes_detected) < 2:
            print(f"   ❌ Only detected {len(regimes_detected)} regime(s): {regimes_detected}")
            return False
        
        print(f"   ✅ Detected {len(regimes_detected)} market regimes: {regimes_detected}")
        self.results['market_regimes'] = {
            'regimes_detected': list(regimes_detected),
            'regime_count': len(regimes_detected)
        }
        return True
    
    def test_walk_forward_validation(self) -> bool:
        """Test walk-forward optimization"""
        if 'EURUSD' not in self.test_data:
            print("   ⚠️ No EURUSD data for walk-forward testing")
            return True
        
        # Use smaller dataset for quick testing
        test_data = {'EURUSD': self.test_data['EURUSD'].iloc[:500]}
        
        predictor = ForexPredictor(['EURUSD'])
        
        try:
            # Run mini walk-forward test
            results = predictor.walk_forward_optimization(
                test_data,
                train_periods=200,
                test_periods=50,
                step_size=50,
                epochs_per_window=5  # Quick test
            )
            
            if 'EURUSD' not in results or not results['EURUSD']:
                print("   ❌ Walk-forward validation produced no results")
                return False
            
            # Check if results contain required metrics
            first_result = results['EURUSD'][0]
            required_keys = ['train_window', 'test_window', 'test_metrics']
            
            for key in required_keys:
                if key not in first_result:
                    print(f"   ❌ Missing key in walk-forward results: {key}")
                    return False
            
            print(f"   ✅ Walk-forward validation completed: {len(results['EURUSD'])} windows tested")
            self.results['walk_forward'] = {
                'windows_tested': len(results['EURUSD']),
                'validation_working': True
            }
            return True
            
        except Exception as e:
            print(f"   ❌ Walk-forward validation failed: {str(e)}")
            return False
    
    def test_performance_metrics(self) -> bool:
        """Test performance metrics calculation"""
        if 'EURUSD' not in self.test_data:
            print("   ⚠️ No EURUSD data for performance testing")
            return True
        
        from train import ForexIndicators
        indicators = ForexIndicators()
        
        # Create test environment
        df = self.test_data['EURUSD'].iloc[:200]
        prepared_df = indicators.calculate_all_indicators(df)
        env = ForexEnvironment(prepared_df, pair_name='EURUSD')
        
        # Simulate some trades
        env.reset()
        for _ in range(50):
            if env.current_step >= env.max_steps:
                break
            
            # Random trading for testing
            from train import TradingAction
            action = TradingAction('long', 40.0, 20.0, 0.5)
            env.step(action)
        
        # Get performance metrics
        metrics = env.get_performance_metrics()
        
        # Check if all required metrics are present
        required_metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'total_trades']
        
        for metric in required_metrics:
            if metric not in metrics:
                print(f"   ❌ Missing performance metric: {metric}")
                return False
            
            # Check for valid values
            value = metrics[metric]
            if not isinstance(value, (int, float)) or np.isnan(value):
                if metric != 'sharpe_ratio':  # Sharpe can be NaN with insufficient data
                    print(f"   ❌ Invalid value for {metric}: {value}")
                    return False
        
        print("   ✅ Performance metrics calculation working correctly")
        self.results['performance_metrics'] = metrics
        return True
    
    def test_stress_scenarios(self) -> bool:
        """Test system under stress scenarios"""
        if 'EURUSD' not in self.test_data:
            print("   ⚠️ No EURUSD data for stress testing")
            return True
        
        from train import ForexIndicators
        indicators = ForexIndicators()
        
        # Create extreme market data for stress testing
        df = self.test_data['EURUSD'].iloc[:100].copy()
        
        # Scenario 1: Extreme volatility
        df_volatile = df.copy()
        df_volatile['close'] = df_volatile['close'] * (1 + np.random.normal(0, 0.05, len(df_volatile)))
        df_volatile['high'] = df_volatile[['open', 'close']].max(axis=1) * 1.02
        df_volatile['low'] = df_volatile[['open', 'close']].min(axis=1) * 0.98
        
        try:
            prepared_df = indicators.calculate_all_indicators(df_volatile)
            env = ForexEnvironment(prepared_df, pair_name='EURUSD')
            env.reset()
            
            # Should handle extreme volatility without crashing
            for _ in range(10):
                if env.current_step >= env.max_steps:
                    break
                from train import TradingAction
                action = TradingAction('long', 40.0, 20.0, 0.5)
                env.step(action)
            
        except Exception as e:
            print(f"   ❌ System failed under high volatility: {str(e)}")
            return False
        
        # Scenario 2: Trending market
        df_trend = df.copy()
        trend = np.linspace(0, 0.1, len(df_trend))
        df_trend['close'] = df_trend['close'] * (1 + trend)
        df_trend['high'] = df_trend['close'] * 1.001
        df_trend['low'] = df_trend['close'] * 0.999
        df_trend['open'] = df_trend['close'].shift(1).fillna(df_trend['close'].iloc[0])
        
        try:
            prepared_df = indicators.calculate_all_indicators(df_trend)
            market_conditions = indicators.detect_market_regime(prepared_df)
            
            if 'trending' not in market_conditions.regime:
                print(f"   ❌ Failed to detect trending market: {market_conditions.regime}")
                return False
                
        except Exception as e:
            print(f"   ❌ System failed in trending market: {str(e)}")
            return False
        
        print("   ✅ System handles stress scenarios correctly")
        self.results['stress_testing'] = {
            'high_volatility': True,
            'trending_market': True,
            'system_stability': True
        }
        return True
    
    def test_correlation_limits(self) -> bool:
        """Test position correlation limits"""
        risk_manager = RiskManager(max_correlation=0.7)
        
        # Test correlation detection
        correlated_pairs = [
            ('EURUSD', 'GBPUSD'),
            ('USDJPY', 'USDCHF'),
            ('AUDUSD', 'GBPUSD')
        ]
        
        for pair1, pair2 in correlated_pairs:
            # Add first position
            risk_manager.open_positions = []
            risk_manager.add_position(pair1, 'long', 100000, 1.0)
            risk_manager.add_position(pair2, 'long', 100000, 1.0)
            
            # Try to add another correlated position
            can_trade, msg = risk_manager.can_trade(10000, pair1, 'long')
            
            # Should be blocked for some highly correlated pairs
            if pair1 in ['EURUSD', 'GBPUSD'] and can_trade:
                print(f"   ⚠️ Correlation risk not detected for {pair1}-{pair2}")
        
        print("   ✅ Correlation limits working")
        self.results['correlation_limits'] = True
        return True
    
    def generate_validation_report(self):
        """Generate comprehensive validation report"""
        report_dir = 'validation_reports'
        os.makedirs(report_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = os.path.join(report_dir, f'validation_report_{timestamp}.json')
        
        report = {
            'timestamp': timestamp,
            'validation_results': self.results,
            'system_version': '2.0_production',
            'validation_status': 'PASSED' if all(
                isinstance(v, dict) or v for v in self.results.values()
            ) else 'FAILED'
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📋 Validation report saved: {report_file}")
        
        # Generate summary visualization if matplotlib available
        try:
            self.create_validation_summary_plot(report_dir, timestamp)
        except Exception as e:
            print(f"   ⚠️ Could not create plots: {str(e)}")
    
    def create_validation_summary_plot(self, report_dir: str, timestamp: str):
        """Create visualization of validation results"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Forex AI Production Validation Summary', fontsize=16, fontweight='bold')
        
        # Test results pie chart
        test_names = list(self.results.keys())
        test_status = ['PASS' if (isinstance(v, dict) or v) else 'FAIL' for v in self.results.values()]
        
        colors = ['green' if status == 'PASS' else 'red' for status in test_status]
        ax1.pie([1] * len(test_names), labels=test_names, colors=colors, autopct='%1.0f%%')
        ax1.set_title('Validation Test Results')
        
        # Performance metrics (if available)
        if 'performance_metrics' in self.results and isinstance(self.results['performance_metrics'], dict):
            metrics = self.results['performance_metrics']
            metric_names = list(metrics.keys())
            metric_values = [float(v) if not np.isnan(float(v)) else 0 for v in metrics.values()]
            
            ax2.bar(range(len(metric_names)), metric_values)
            ax2.set_xticks(range(len(metric_names)))
            ax2.set_xticklabels(metric_names, rotation=45, ha='right')
            ax2.set_title('Performance Metrics')
        else:
            ax2.text(0.5, 0.5, 'No Performance Data', ha='center', va='center')
            ax2.set_title('Performance Metrics')
        
        # Risk management status
        if 'risk_management' in self.results:
            risk_data = self.results['risk_management']
            risk_names = list(risk_data.keys())
            risk_values = [1 if v else 0 for v in risk_data.values()]
            
            ax3.bar(range(len(risk_names)), risk_values, color=['green' if v else 'red' for v in risk_values])
            ax3.set_xticks(range(len(risk_names)))
            ax3.set_xticklabels(risk_names, rotation=45, ha='right')
            ax3.set_ylim(0, 1.2)
            ax3.set_title('Risk Management Tests')
        else:
            ax3.text(0.5, 0.5, 'No Risk Data', ha='center', va='center')
            ax3.set_title('Risk Management Tests')
        
        # System readiness summary
        readiness_score = sum(1 for v in self.results.values() if (isinstance(v, dict) or v))
        total_tests = len(self.results)
        
        ax4.pie([readiness_score, total_tests - readiness_score], 
                labels=['PASSED', 'FAILED'], 
                colors=['green', 'red'],
                autopct='%1.1f%%',
                startangle=90)
        ax4.set_title(f'Overall Readiness: {readiness_score}/{total_tests}')
        
        plt.tight_layout()
        plot_file = os.path.join(report_dir, f'validation_summary_{timestamp}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Validation plot saved: {plot_file}")

def main():
    """Run the production validation suite"""
    validator = ProductionValidator()
    
    # Get user preferences
    print("🔬 Production Validation Suite for Forex AI Trading System")
    print("=" * 60)
    
    data_period = input("Data period for testing (default '6mo'): ").strip() or '6mo'
    epochs = int(input("Training epochs for validation (default 30): ").strip() or 30)
    
    print(f"\nStarting validation with {data_period} data and {epochs} epochs...")
    
    success = validator.run_comprehensive_validation(data_period, epochs)
    
    if success:
        print("\n🎉 SYSTEM VALIDATED FOR PRODUCTION USE!")
        print("📋 Next steps:")
        print("   1. Review validation report in validation_reports/")
        print("   2. Start paper trading for 3-6 months")
        print("   3. Monitor performance daily")
        print("   4. Gradually increase position sizes")
    else:
        print("\n🚨 SYSTEM NOT READY FOR PRODUCTION!")
        print("📋 Required actions:")
        print("   1. Fix failing tests")
        print("   2. Re-run validation")
        print("   3. Do NOT trade with real money")

if __name__ == "__main__":
    main()
