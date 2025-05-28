# Configuration Guide

This guide explains all configuration options available in the TabularBERT trading system.

## 📊 **Data Settings**

```python
# Asset and time period
"ticker": "MSFT",                    # Stock ticker symbol
"start_date": "2012-01-01",          # Start date for data
"end_date": "2023-12-31",            # End date for data
"train_split_ratio": 0.8,            # 80% for training, 20% for testing
```

**Popular Tickers to Try:**
- `"AAPL"` - Apple (tech stock)
- `"SPY"` - S&P 500 ETF (market index)
- `"QQQ"` - NASDAQ ETF (tech-heavy)
- `"TSLA"` - Tesla (high volatility)
- `"GOOGL"` - Google (large cap tech)

## 🎯 **Confidence Trading Settings**

```python
# Core confidence settings
"use_confidence_trading": True,       # Enable confidence-based position sizing
"use_enhanced_bert_policy": True,     # Use enhanced policy with confidence
"confidence_scaling": True,           # Enable confidence scaling

# Position sizing parameters
"confidence_min_threshold": 0.3,      # Minimum confidence to trade (0.0-1.0)
"confidence_position_multiplier": 2.0, # Maximum position multiplier
"confidence_weight": 0.1,             # Weight for confidence loss in training
```

### **Confidence Strategies**

**Conservative (Lower Risk)**
```python
"confidence_min_threshold": 0.4,      # Higher threshold = fewer trades
"confidence_position_multiplier": 1.8, # Lower multiplier = smaller positions
```

**Balanced (Default)**
```python
"confidence_min_threshold": 0.3,      # Moderate threshold
"confidence_position_multiplier": 2.0, # Moderate multiplier
```

**Aggressive (Higher Risk)**
```python
"confidence_min_threshold": 0.25,     # Lower threshold = more trades
"confidence_position_multiplier": 2.5, # Higher multiplier = larger positions
```

## 🧠 **Model Architecture**

```python
# Model type selection
"model_type": "bert_transformer",     # Options: "lstm", "transformer", "bert_transformer"

# TabularBERT/Transformer parameters
"transformer_d_model": 128,           # Model dimension (64, 128, 256, 512)
"transformer_nhead": 8,               # Number of attention heads (4, 8, 16)
"transformer_num_layers": 4,          # Number of transformer layers (2, 4, 6, 8)
"transformer_dim_feedforward": 512,   # Feedforward dimension
"transformer_dropout": 0.2,           # Dropout rate (0.1, 0.2, 0.3)
"lookback_window": 120,               # Days of history to consider (60, 120, 180)
"max_features": 15,                   # Maximum number of features to use
```

### **Model Size Recommendations**

**Small Model (Fast Training)**
```python
"transformer_d_model": 64,
"transformer_num_layers": 2,
"lookback_window": 60,
```

**Medium Model (Balanced)**
```python
"transformer_d_model": 128,
"transformer_num_layers": 4,
"lookback_window": 120,
```

**Large Model (Best Performance)**
```python
"transformer_d_model": 256,
"transformer_num_layers": 6,
"lookback_window": 180,
```

## 🏋️ **Training Settings**

```python
# PPO hyperparameters
"ppo_total_timesteps": 1_200_000,     # Total training steps (800k, 1.2M, 2M)
"ppo_n_steps": 2048,                  # Steps per rollout
"ppo_batch_size": 128,                # Batch size for updates
"ppo_n_epochs": 15,                   # Epochs per update
"ppo_learning_rate": 2e-5,            # Learning rate (1e-5, 2e-5, 5e-5)
"ppo_gamma": 0.997,                   # Discount factor
"ppo_gae_lambda": 0.97,               # GAE lambda
"ppo_clip_range": 0.2,                # PPO clip range
"ppo_ent_coef": 0.018,                # Entropy coefficient
"ppo_vf_coef": 0.5,                   # Value function coefficient
```

### **Training Speed vs Quality**

**Fast Training (Quick Results)**
```python
"ppo_total_timesteps": 400_000,
"ppo_n_steps": 1024,
"ppo_batch_size": 64,
```

**Standard Training (Balanced)**
```python
"ppo_total_timesteps": 1_200_000,
"ppo_n_steps": 2048,
"ppo_batch_size": 128,
```

**Thorough Training (Best Results)**
```python
"ppo_total_timesteps": 2_000_000,
"ppo_n_steps": 4096,
"ppo_batch_size": 256,
```

## 💰 **Trading Environment**

```python
# Portfolio settings
"initial_balance": 10_000,            # Starting capital
"transaction_cost": 0.0022,           # Transaction cost (0.22%)
"capital_cost": 0.00015,              # Cost of leverage
"risk_free_rate": 0.055,              # Risk-free rate (5.5% annually)
"episode_length": 252,                # Trading days per episode

# Position management
"allow_short": True,                  # Enable short selling
"max_position": 0.4,                  # Maximum position size (40% of portfolio)
"min_holding_period": 15,             # Minimum days to hold position
"volatility_scaling": True,           # Scale positions by volatility

# Risk management
"drawdown_penalty": 2.5,              # Penalty for drawdowns
"reward_trade_penalty": 0.006,        # Penalty for excessive trading
"wrong_side_penalty_factor": 0.15,    # Penalty for wrong-side moves
```

### **Risk Profiles**

**Conservative Trading**
```python
"max_position": 0.3,                  # Smaller positions
"drawdown_penalty": 3.0,              # Higher drawdown penalty
"min_holding_period": 20,             # Longer holding periods
```

**Moderate Trading**
```python
"max_position": 0.4,                  # Moderate positions
"drawdown_penalty": 2.5,              # Moderate penalty
"min_holding_period": 15,             # Standard holding
```

**Aggressive Trading**
```python
"max_position": 0.6,                  # Larger positions
"drawdown_penalty": 2.0,              # Lower penalty
"min_holding_period": 10,             # Shorter holding
```

## 🔧 **Pre-training Settings**

```python
# TabularBERT pre-training
"bert_pretrain_epochs": 50,           # Maximum pre-training epochs
"bert_pretrain_batch_size": 32,       # Pre-training batch size
"bert_pretrain_lr": 1e-4,             # Pre-training learning rate
```

## 📈 **Performance Tuning**

### **For Higher Returns**
```python
"confidence_min_threshold": 0.25,     # Trade more often
"confidence_position_multiplier": 2.5, # Larger positions when confident
"max_position": 0.5,                  # Allow larger positions
"ppo_ent_coef": 0.02,                 # More exploration
```

### **For Lower Risk**
```python
"confidence_min_threshold": 0.4,      # Trade less often
"confidence_position_multiplier": 1.8, # Smaller position multipliers
"max_position": 0.3,                  # Limit position size
"drawdown_penalty": 3.0,              # Higher drawdown penalty
```

### **For Faster Training**
```python
"ppo_total_timesteps": 600_000,       # Fewer training steps
"transformer_d_model": 64,            # Smaller model
"transformer_num_layers": 2,          # Fewer layers
"lookback_window": 60,                # Shorter memory
```

## 🎛️ **Advanced Settings**

```python
# LSTM settings (if using LSTM model)
"lstm_hidden_size": 128,
"n_lstm_layers": 2,
"lstm_dropout": 0.1,

# Output dimensions
"latent_dim_pi_out": 128,             # Policy network output dimension
"latent_dim_vf_out": 128,             # Value network output dimension

# Confidence training
"confidence_diversity_weight": 0.05,   # Weight for confidence diversity loss
```

## 🚀 **Quick Start Configurations**

### **Demo Mode (Fast Results)**
```python
CONFIG = {
    "ticker": "AAPL",
    "ppo_total_timesteps": 200_000,
    "transformer_d_model": 64,
    "transformer_num_layers": 2,
    "lookback_window": 60,
    "confidence_min_threshold": 0.3,
    "confidence_position_multiplier": 2.0,
}
```

### **Production Mode (Best Performance)**
```python
CONFIG = {
    "ticker": "MSFT",
    "ppo_total_timesteps": 2_000_000,
    "transformer_d_model": 256,
    "transformer_num_layers": 6,
    "lookback_window": 180,
    "confidence_min_threshold": 0.3,
    "confidence_position_multiplier": 2.0,
}
```

## 📊 **Expected Performance by Configuration**

| Configuration | Training Time | Expected Return | Max Drawdown | Sharpe Ratio |
|---------------|---------------|-----------------|--------------|--------------|
| Demo          | 30 min        | 4-8%           | 10-20%       | 0.2-0.4      |
| Standard      | 2-3 hours     | 6-12%          | 8-15%        | 0.3-0.6      |
| Production    | 4-6 hours     | 8-15%          | 6-12%        | 0.4-0.7      |

## 🔍 **Troubleshooting**

**If training is too slow:**
- Reduce `ppo_total_timesteps`
- Use smaller `transformer_d_model`
- Reduce `lookback_window`

**If performance is poor:**
- Increase `ppo_total_timesteps`
- Try different `confidence_min_threshold`
- Adjust `confidence_position_multiplier`

**If too risky:**
- Increase `confidence_min_threshold`
- Reduce `max_position`
- Increase `drawdown_penalty`

**If too conservative:**
- Reduce `confidence_min_threshold`
- Increase `confidence_position_multiplier`
- Reduce `min_holding_period` 