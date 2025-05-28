#!/usr/bin/env python3
"""
Quick Run Script for TabularBERT Trading System

This script provides easy access to different trading configurations.
"""

import sys
import os

def run_demo():
    """Run a quick demo with fast settings."""
    print("🚀 Running TabularBERT Trading Demo...")
    print("This will use fast settings for quick results (30-60 minutes)")
    
    # Modify config for demo
    import AdamTrading2
    
    # Save original config
    original_config = AdamTrading2.CONFIG.copy()
    
    # Demo settings (faster training)
    demo_config = {
        "ticker": "AAPL",
        "ppo_total_timesteps": 400_000,
        "embedding_dim": 64,  # Smaller embedding dimension for faster training
        "bert_pretrain_epochs": 20,
        "lookback_window": 60,
        "confidence_min_threshold": 0.3,
        "confidence_position_multiplier": 2.0,
    }
    
    # Update config
    AdamTrading2.CONFIG.update(demo_config)
    
    try:
        # Run the system
        AdamTrading2.main()
    finally:
        # Restore original config
        AdamTrading2.CONFIG.clear()
        AdamTrading2.CONFIG.update(original_config)

def run_standard():
    """Run with standard settings."""
    print("🚀 Running TabularBERT Trading System (Standard Settings)...")
    print("This will use standard settings (2-3 hours)")
    
    import AdamTrading2
    AdamTrading2.main()

def run_production():
    """Run with production settings for best performance."""
    print("🚀 Running TabularBERT Trading System (Production Settings)...")
    print("This will use production settings for best performance (4-6 hours)")
    
    import AdamTrading2
    
    # Save original config
    original_config = AdamTrading2.CONFIG.copy()
    
    # Production settings
    production_config = {
        "ppo_total_timesteps": 2_000_000,
        "embedding_dim": 256,  # Larger embedding dimension for better performance
        "bert_pretrain_epochs": 100,
        "lookback_window": 180,
        "ppo_batch_size": 256,
        "ppo_n_steps": 4096,
    }
    
    # Update config
    AdamTrading2.CONFIG.update(production_config)
    
    try:
        # Run the system
        AdamTrading2.main()
    finally:
        # Restore original config
        AdamTrading2.CONFIG.clear()
        AdamTrading2.CONFIG.update(original_config)

def run_test():
    """Run system tests."""
    print("🧪 Running System Tests...")
    
    try:
        import test_system
        test_system.main()
    except ImportError:
        print("❌ test_system.py not found")
        return False
    
    return True

def show_menu():
    """Show the main menu."""
    print("\n" + "="*60)
    print("🎯 TabularBERT Confidence-Based Trading System")
    print("="*60)
    print("Choose an option:")
    print()
    print("1. 🚀 Demo Mode (Fast, 30-60 min)")
    print("   - Quick results with AAPL")
    print("   - Smaller model for faster training")
    print("   - Good for testing and learning")
    print()
    print("2. 📊 Standard Mode (Balanced, 2-3 hours)")
    print("   - Default settings with MSFT")
    print("   - Balanced performance and training time")
    print("   - Recommended for most users")
    print()
    print("3. 🏆 Production Mode (Best Performance, 4-6 hours)")
    print("   - Larger model for best results")
    print("   - Longer training for optimal performance")
    print("   - For serious trading applications")
    print()
    print("4. 🧪 Test System")
    print("   - Verify all components are working")
    print("   - Check dependencies and configuration")
    print("   - Run before first use")
    print()
    print("5. ❌ Exit")
    print()

def main():
    """Main menu loop."""
    while True:
        show_menu()
        
        try:
            choice = input("Enter your choice (1-5): ").strip()
            
            if choice == "1":
                print("\n" + "🚀 Starting Demo Mode...")
                run_demo()
                break
                
            elif choice == "2":
                print("\n" + "📊 Starting Standard Mode...")
                run_standard()
                break
                
            elif choice == "3":
                print("\n" + "🏆 Starting Production Mode...")
                run_production()
                break
                
            elif choice == "4":
                print("\n" + "🧪 Running Tests...")
                if run_test():
                    print("\n✅ Tests completed. You can now run the trading system.")
                else:
                    print("\n❌ Tests failed. Please check the error messages.")
                input("\nPress Enter to continue...")
                
            elif choice == "5":
                print("\n👋 Goodbye!")
                break
                
            else:
                print("\n❌ Invalid choice. Please enter 1-5.")
                input("Press Enter to continue...")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            input("Press Enter to continue...")

if __name__ == "__main__":
    main() 