#!/usr/bin/env python3
"""
System Test Script

Quick test to verify all components are working correctly.
"""

def test_imports():
    """Test that all required modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        import torch
        print("✅ PyTorch imported successfully")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        from stable_baselines3 import PPO
        print("✅ Stable-Baselines3 imported successfully")
    except ImportError as e:
        print(f"❌ Stable-Baselines3 import failed: {e}")
        return False
    
    try:
        import yfinance as yf
        print("✅ yfinance imported successfully")
    except ImportError as e:
        print(f"❌ yfinance import failed: {e}")
        return False
    
    try:
        from tabular_bert import TabularBERT, TabularBERTPreTrainer
        print("✅ TabularBERT imported successfully")
    except ImportError as e:
        print(f"❌ TabularBERT import failed: {e}")
        return False
    
    try:
        from confidence_trading_env import ConfidenceBasedTradingEnv
        print("✅ Confidence trading environment imported successfully")
    except ImportError as e:
        print(f"❌ Confidence trading environment import failed: {e}")
        return False
    
    try:
        from enhanced_bert_policy import EnhancedBERTBasedActorCriticPolicy
        print("✅ Enhanced BERT policy imported successfully")
    except ImportError as e:
        print(f"❌ Enhanced BERT policy import failed: {e}")
        return False
    
    try:
        from confidence_aware_wrapper import create_confidence_aware_wrapper
        print("✅ Confidence wrapper imported successfully")
    except ImportError as e:
        print(f"❌ Confidence wrapper import failed: {e}")
        return False
    
    return True

def test_configuration():
    """Test that configuration is accessible."""
    print("\n⚙️ Testing configuration...")
    
    try:
        from AdamTrading2 import CONFIG
        print(f"✅ Configuration loaded successfully")
        print(f"   • Ticker: {CONFIG['ticker']}")
        print(f"   • Model type: {CONFIG['model_type']}")
        print(f"   • Confidence trading: {CONFIG.get('use_confidence_trading', False)}")
        print(f"   • Training timesteps: {CONFIG['ppo_total_timesteps']:,}")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_data_access():
    """Test that data can be downloaded."""
    print("\n📊 Testing data access...")
    
    try:
        import yfinance as yf
        # Try to download a small amount of data
        ticker = yf.Ticker("MSFT")
        data = ticker.history(period="5d")
        
        if not data.empty:
            print(f"✅ Data download successful")
            print(f"   • Downloaded {len(data)} days of data")
            print(f"   • Columns: {list(data.columns)}")
            return True
        else:
            print("❌ Data download returned empty dataset")
            return False
            
    except Exception as e:
        print(f"❌ Data access test failed: {e}")
        return False

def test_model_creation():
    """Test that models can be created."""
    print("\n🧠 Testing model creation...")
    
    try:
        import torch
        from tabular_bert import TabularBERT
        
        # Create a small TabularBERT model
        model = TabularBERT(
            n_features=15,
            d_model=64,
            nhead=4,
            num_layers=2,
            max_seq_length=60
        )
        
        # Test forward pass
        test_input = torch.randn(1, 60, 15)
        with torch.no_grad():
            output = model(test_input, return_embeddings=True)
        
        print(f"✅ TabularBERT model created successfully")
        print(f"   • Input shape: {test_input.shape}")
        print(f"   • Output shape: {output.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Model creation test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 TabularBERT Trading System - Component Test")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_configuration),
        ("Data Access Test", test_data_access),
        ("Model Creation Test", test_model_creation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("\n🚀 To run the full trading system:")
        print("   python AdamTrading2.py")
    else:
        print("⚠️ Some tests failed. Please check the error messages above.")
        print("   Make sure all dependencies are installed: pip install -r requirements.txt")
    
    return passed == total

if __name__ == "__main__":
    main() 