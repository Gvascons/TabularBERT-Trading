# TabularBERT Trading System

A sophisticated deep reinforcement learning system for stock trading that adapts BERT-style pre-training to tabular financial data. The system uses masked feature modeling to learn rich representations of financial time series, then fine-tunes these representations for optimal trading decisions.

## 🌟 Key Features

- **TabularBERT Pre-training**: Novel adaptation of BERT's masked language modeling to tabular financial data
- **Frozen Backbone Fine-tuning**: Prevents catastrophic forgetting by using pre-computed embeddings
- **Confidence-Based Position Sizing**: Dynamic position sizing based on model confidence
- **Advanced Risk Management**: Sophisticated drawdown control and volatility-based position scaling
- **Production-Ready**: Clean, maintainable code with comprehensive error handling

## 📊 Architecture Overview

The system follows a three-phase approach:

1. **Pre-training Phase**: TabularBERT learns general patterns in financial data through masked feature prediction
2. **Embedding Extraction**: Pre-trained model generates rich embeddings for time series sequences
3. **Fine-tuning Phase**: PPO agent learns optimal trading policies using frozen embeddings

```
Financial Data → Technical Indicators → TabularBERT Pre-training
                                              ↓
                                     Embedding Extraction
                                              ↓
                                     PPO Fine-tuning → Trading Agent
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/tabularbert-trading.git
cd tabularbert-trading

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the System

```bash
# Run with interactive menu
python run_trading.py

# Or run directly
python AdamTrading2.py
```

Choose from three modes:
- **Demo Mode** (30-60 min): Quick results with reduced model size
- **Standard Mode** (2-3 hours): Balanced performance and training time
- **Production Mode** (4-6 hours): Maximum performance with larger models

## 📈 Performance

Example results from demo mode (AAPL, 2012-2023):
- **Portfolio Return**: 7.10% (vs 18.94% buy-and-hold)
- **Sharpe Ratio**: 0.35
- **Max Drawdown**: -13.57% (vs -31.31% buy-and-hold)
- **Win Rate**: 49.50%

The system prioritizes risk-adjusted returns and drawdown control over absolute returns.

## 🔧 Configuration

Key parameters in `CONFIG` dictionary (AdamTrading2.py):

```python
CONFIG = {
    "ticker": "MSFT",                    # Stock symbol
    "embedding_dim": 128,                # TabularBERT embedding dimension
    "bert_pretrain_epochs": 50,          # Pre-training epochs
    "ppo_total_timesteps": 1_200_000,    # PPO training steps
    "confidence_min_threshold": 0.3,     # Minimum confidence to trade
    "max_position": 0.4,                 # Maximum position size
    # ... more options
}
```

## 🏗️ Project Structure

```
tabularbert-trading/
├── AdamTrading2.py              # Main system implementation
├── tabular_bert.py              # TabularBERT model and pre-training
├── confidence_trading_env.py    # Confidence-based trading environment
├── enhanced_bert_policy.py      # Enhanced policy with confidence
├── confidence_aware_wrapper.py  # Model wrapper for confidence integration
├── run_trading.py              # User-friendly launcher
├── test_system.py              # System tests
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## 🧠 Technical Details

### TabularBERT Architecture

- **Input**: Financial time series with technical indicators
- **Embedding**: Each feature embedded to d-dimensional space
- **Masking**: Random 15% of features masked during pre-training
- **Objective**: Predict original values of masked features
- **Output**: Rich embeddings capturing temporal and cross-feature relationships

### Trading Environment

- **Observation Space**: Pre-computed embeddings + agent state
- **Action Space**: Discrete (Flat, Long, Short)
- **Reward**: Risk-adjusted returns with drawdown penalties
- **Position Sizing**: Volatility-scaled with confidence adjustment

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{tabularbert_trading,
  title = {TabularBERT Trading System},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/tabularbert-trading}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Do not use it for actual trading without understanding the risks involved. Past performance does not guarantee future results. Always consult with a qualified financial advisor before making investment decisions.

## 🙏 Acknowledgments

- Inspired by BERT (Devlin et al., 2018) and its applications to non-NLP domains
- Built with [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- Financial data from [yfinance](https://github.com/ranaroussi/yfinance) 