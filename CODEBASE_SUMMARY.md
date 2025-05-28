# TabularBERT Trading System - Codebase Summary

## Overview
A production-ready deep reinforcement learning system for stock trading that uses BERT-style pre-training on tabular financial data.

## Core Architecture

### 1. Pre-training Phase
- **TabularBERT Model** (`tabular_bert.py`): Transformer model adapted for tabular data
  - Uses masked feature prediction (15% masking) to learn patterns
  - Outputs rich embeddings for financial time series
  - Implements positional encoding for temporal awareness

### 2. Fine-tuning Phase  
- **Frozen Backbone Approach**: Pre-computed embeddings prevent catastrophic forgetting
- **PPO Agent Training**: Only task-specific heads are trained
- **Confidence-Based Position Sizing**: Dynamic position sizing based on model confidence

## File Structure

### Core Implementation
- `AdamTrading2.py` (1,687 lines): Main implementation file
  - Data processing and technical indicators
  - Environment definitions
  - Policy implementations
  - Training and evaluation logic
  
- `tabular_bert.py` (639 lines): TabularBERT model and pre-training
  - `TabularBERT`: Core transformer model
  - `TabularBERTPreTrainer`: Handles masked feature pre-training
  - `EmbeddingBasedExtractor`: Feature extractor for fine-tuning
  - `prepare_sequences_for_pretraining`: Data preparation utility

### Enhanced Features
- `enhanced_bert_policy.py`: Enhanced policy with confidence features
- `confidence_trading_env.py`: Environment with confidence-based position sizing
- `confidence_aware_wrapper.py`: Wrapper for confidence-aware predictions

### Utilities
- `run_trading.py`: Quick-start script with different configurations
- `test_system.py`: Component testing script
- `requirements.txt`: Python dependencies
- `CONFIGURATION.md`: Detailed configuration guide

## Key Features

### 1. Data Processing
- **Technical Indicators**: 50+ indicators including:
  - Moving averages (SMA, EMA)
  - Oscillators (RSI, Stochastic, MACD)
  - Volatility measures (ATR, Bollinger Bands)
  - Trend indicators (ADX, DI+/DI-)
  - Volume indicators (OBV)

### 2. Risk Management
- Position sizing based on volatility
- Drawdown penalties
- Minimum holding periods
- Transaction cost modeling
- Wrong-side penalty for adverse moves

### 3. Confidence Features
- Model confidence scores (0-1)
- Dynamic position sizing based on confidence
- Confidence tracking and metrics
- Attention-based confidence calculation

## Training Pipeline

1. **Data Preparation**
   - Download price data (cached)
   - Calculate technical indicators
   - Normalize features
   - Create sequences

2. **Pre-training**
   - Mask 15% of features randomly
   - Train TabularBERT to reconstruct
   - Extract embeddings

3. **Fine-tuning**
   - Use pre-computed embeddings
   - Train PPO agent
   - Only update task-specific heads

4. **Evaluation**
   - Backtesting on test data
   - Risk metrics (Sharpe, Sortino, Calmar)
   - Trading statistics
   - Confidence analysis

## Configuration

Key parameters in `CONFIG`:
- `embedding_dim`: 128 (TabularBERT embedding dimension)
- `lookback_window`: 120 days
- `bert_pretrain_epochs`: 50
- `ppo_total_timesteps`: 1.2M
- `use_confidence_trading`: True
- `max_position`: 0.4 (40% of portfolio)

## Performance Metrics

The system tracks:
- Portfolio value over time
- Risk-adjusted returns (Sharpe, Sortino, Calmar ratios)
- Maximum drawdown
- Win rate and profit factor
- Trade frequency
- Confidence distribution

## Production Readiness

✅ **Clean Architecture**: BERT-only implementation with clear separation
✅ **Error Handling**: Try-except blocks for optional features
✅ **Type Hints**: Comprehensive type annotations
✅ **Documentation**: Clear docstrings throughout
✅ **Configurability**: All parameters exposed in CONFIG
✅ **Reproducibility**: Seed handling and model saving

## Usage

```bash
# Quick demo
python run_trading.py

# Full system test
python test_system.py

# Production run
python AdamTrading2.py
```

## Open Source Ready

- MIT License included
- Comprehensive README
- Clean dependencies
- No hardcoded paths
- Proper .gitignore

The codebase is polished, well-documented, and ready for open-source release. 