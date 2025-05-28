#!/usr/bin/env python3
"""
TabularBERT Trading System - BERT-Only Version

This is a streamlined version that only supports the TabularBERT approach.
All LSTM and plain transformer code has been removed for clarity and focus.

Key features:
• TabularBERT pre-training with masked feature modeling
• Embedding-based fine-tuning with frozen backbone
• Confidence-based position sizing
• Enhanced risk management
"""


# ───────────────────────────── Imports ──────────────────────────────
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import os  # Added for file path handling
import datetime  # Added for timestamps in saved models

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Callable, List, Optional, Type, Union # Added for type hints

# Import TabularBERT implementation
from tabular_bert import (
    TabularBERTPreTrainer, 
    EmbeddingBasedExtractor,
    prepare_sequences_for_pretraining
)

# Import confidence-based improvements (conditional to avoid circular imports)
try:
    from confidence_trading_env import ConfidenceBasedTradingEnv
    from enhanced_bert_policy import (
        EnhancedBERTBasedActorCriticPolicy, 
        create_enhanced_bert_policy_kwargs
    )
    from confidence_aware_wrapper import create_confidence_aware_wrapper
    CONFIDENCE_IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Confidence-based features not available: {e}")
    ConfidenceBasedTradingEnv = None
    EnhancedBERTBasedActorCriticPolicy = None
    create_enhanced_bert_policy_kwargs = None
    create_confidence_aware_wrapper = None
    CONFIDENCE_IMPORTS_AVAILABLE = False

class BERTBasedActorCriticPolicy(ActorCriticPolicy):
    """
    Actor-Critic policy that uses pre-trained TabularBERT embeddings.
    This implements the fine-tuning phase with frozen/low-lr backbone.
    """
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        # Embedding specific parameters
        embedding_dim: int = 128,
        latent_dim_pi_out: int = 128,
        latent_dim_vf_out: int = 128,
        dropout: float = 0.1,
        # Standard ActorCriticPolicy args
        activation_fn: Type[nn.Module] = nn.Tanh,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.embedding_dim_config = embedding_dim
        self.latent_dim_pi_config = latent_dim_pi_out
        self.latent_dim_vf_config = latent_dim_vf_out
        self.dropout_config = dropout

        # Validate observation space
        if not isinstance(observation_space, spaces.Dict):
            raise ValueError(
                "BERTBasedActorCriticPolicy requires a Dict observation space "
                "with 'embeddings_seq' and 'agent_state' keys."
            )
            
        # Get dimensions from observation space
        self.state_dim = observation_space['agent_state'].shape[0]
        
        # For self.extract_features - required by SB3 but not used directly
        self.features_dim = 0

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=[],  # We're replacing the mlp_extractor
            activation_fn=activation_fn,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        """
        Builds the embedding-based feature extractor.
        """
        self.mlp_extractor = EmbeddingBasedExtractor(
            embedding_dim=self.embedding_dim_config,
            state_dim=self.state_dim,
            latent_dim_pi=self.latent_dim_pi_config,
            latent_dim_vf=self.latent_dim_vf_config,
            dropout=self.dropout_config,
            output_confidence=True,  # Enable confidence output
        )
            
        self.latent_dim_pi = self.mlp_extractor.latent_dim_pi
        self.latent_dim_vf = self.mlp_extractor.latent_dim_vf

    def extract_features(self, obs):
        """
        For Dict observation spaces, we return the original dict.
        """
        if isinstance(obs, dict):
            return obs
            
        raise ValueError(
            "BERTBasedActorCriticPolicy requires dictionary observations "
            "with 'embeddings_seq' and 'agent_state' keys."
        )
        
    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in actor-critic for prediction.
        """
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        
        return actions, values, log_prob

    def predict_values(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute value estimates from observations.
        """
        features = self.extract_features(obs)
        _, latent_vf = self.mlp_extractor(features)
        return self.value_net(latent_vf)

    def evaluate_actions(
        self, 
        obs: Dict[str, torch.Tensor], 
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions according to the current policy, for PPO update.
        """
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        
        return values, log_prob, entropy

    def get_confidence_score(self) -> torch.Tensor:
        """
        Get the confidence score from the feature extractor.
        
        Returns:
            confidence: (batch_size, 1) confidence scores between 0 and 1
        """
        return self.mlp_extractor.get_confidence_score()

# ─────────────────────── 1. Configuration ───────────────────────────
CONFIG = {
    "ticker":             "MSFT",
    "start_date":         "2012-01-01",
    "end_date":           "2023-12-31",
    "train_split_ratio":  0.8,

    # Environment settings
    "initial_balance":    10_000,
    "lookback_window":    120,        # 120-day lookback window
    "transaction_cost":   0.0022,     # Adjusted from 0.0025, original 0.002
    "capital_cost":       0.00015,    # For leverage cost (Reduced from 0.0002)
    "risk_free_rate":     0.055,       # 5% p.a.
    "episode_length":     252,        # Full trading year
    "allow_short":        True,       # Enable short selling
    "max_position":       0.4,        # Maximum position size (Reduced from 0.6)
    "drawdown_penalty":   2.5,        # Penalty for drawdowns (Adjusted from 2.0, original 4.0)
    "volatility_scaling": True,       # Enable position sizing based on volatility
    "min_holding_period": 15,         # Minimum holding period (Increased from 12)
    "reward_trade_penalty": 0.006, # Adjusted from 0.005, prev 0.01
    "wrong_side_penalty_factor": 0.15, # Penalty for being on wrong side of sharp move (Reduced from 0.25)

    # Model selection
    "model_type":          "bert_transformer",  # Only BERT approach is supported
    
    # BERT-specific settings
    "embedding_dim":       128,       # Embedding dimension for TabularBERT
    "bert_dropout":        0.1,       # Dropout for BERT model
    "bert_pretrain_epochs": 50,       # Number of epochs for pre-training
    "bert_pretrain_batch_size": 32,   # Batch size for pre-training
    "bert_pretrain_lr":    1e-4,      # Learning rate for pre-training

    # PPO hyper‑parameters
    "ppo_total_timesteps": 1_200_000,   # Total timesteps for training (increased for confidence learning)
    "ppo_n_steps":         2048,
    "ppo_batch_size":      128,
    "ppo_n_epochs":        15,
    "ppo_gamma":           0.997,     # Discount factor
    "ppo_gae_lambda":      0.97,
    "ppo_clip_range":      0.2,
    "ppo_ent_coef":        0.018,     # Entropy coefficient (Increased from 0.01)
    "ppo_vf_coef":         0.5,
    "ppo_learning_rate":   2e-5,      # Learning rate

    # Output dimensions for policy networks
    "latent_dim_pi_out":   128,
    "latent_dim_vf_out":   128,
    
    # Feature selection
    "max_features":             15,    # Maximum number of features to include
    
    # Confidence-based trading settings
    "use_confidence_trading":   True,   # Enable confidence-based position sizing
    "confidence_min_threshold": 0.3,    # Minimum confidence to trade (Conservative: 0.4, Aggressive: 0.25)
    "confidence_position_multiplier": 2.0,  # Max position multiplier (Conservative: 1.8, Aggressive: 2.5)
    "confidence_scaling":       True,   # Enable confidence scaling
    "use_enhanced_bert_policy": True,   # Use enhanced BERT policy with confidence
    "confidence_weight":        0.1,    # Weight for confidence loss
}

CACHE_DIR = Path("./price_cache")
CACHE_DIR.mkdir(exist_ok=True)

# ─────────────── 2. Data acquisition & processing ───────────────────

def _cache_path(ticker: str, start: str, end: str) -> Path:
    return CACHE_DIR / f"{ticker}_{start}_{end}.csv"


def _download_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    print(f"Downloading {ticker} from yfinance …")
    # Explicitly set auto_adjust=False to ensure consistent column structure
    # and get 'Adj Close' separately. Existing code uses 'Close'.
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"No data for {ticker}.")
    df.reset_index(inplace=True)       # keep 'Date' as a column
    return df


def _validate_numeric(df: pd.DataFrame, cols=("Close",)) -> bool:
    for c in cols:
        if c not in df.columns:
            return False
        if not pd.api.types.is_numeric_dtype(df[c]):
            return False
    return True


def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    csv_path = _cache_path(ticker, start_date, end_date)

    if csv_path.exists():
        df = pd.read_csv(csv_path, parse_dates=["Date"])
        for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if _validate_numeric(df, ("Close",)):
            print(f"Loaded cached data '{csv_path.name}'")
            return df.reset_index(drop=True)
        print(f"Cached file '{csv_path.name}' corrupted – re‑downloading.")

    df = _download_prices(ticker, start_date, end_date)
    df.to_csv(csv_path, index=False)
    print(f"Cached to '{csv_path.name}'")
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators for stock data."""
    # Get a clean copy of the dataframe
    df = df.copy()
    
    # Extract raw price data to numpy arrays
    # This avoids issues with dataframes vs series
    dates = df.index.values
    
    # Convert all price data to numpy arrays
    close_values = np.array(df["Close"].values.flatten()) if hasattr(df["Close"], 'values') else np.array(df["Close"])
    high_values = np.array(df["High"].values.flatten()) if hasattr(df["High"], 'values') else np.array(df["High"])
    low_values = np.array(df["Low"].values.flatten()) if hasattr(df["Low"], 'values') else np.array(df["Low"])
    volume_values = np.array(df["Volume"].values.flatten()) if hasattr(df["Volume"], 'values') else np.array(df["Volume"])
    
    # Create a new dataframe with just Date and Close for building on
    new_df = pd.DataFrame({
        "Date": df["Date"] if "Date" in df.columns else pd.Series(dates),
        "Close": close_values,
        "High": high_values,
        "Low": low_values,
        "Volume": volume_values
    })
    
    # Moving Averages
    for period in [5, 10, 20, 30, 50, 100, 200]:
        col_name = f"SMA_{period}"
        new_df[col_name] = np.nan
        for i in range(period - 1, len(close_values)):
            new_df.loc[i, col_name] = np.mean(close_values[i-period+1:i+1])
    
    # Exponential Moving Averages
    for period in [9, 12, 21, 26, 55]:
        col_name = f"EMA_{period}"
        alpha = 2 / (period + 1)
        ema = np.zeros_like(close_values) + np.nan
        # Initialize with SMA
        ema[period-1] = np.mean(close_values[:period])
        # Calculate EMA
        for i in range(period, len(close_values)):
            ema[i] = close_values[i] * alpha + ema[i-1] * (1 - alpha)
        new_df[col_name] = ema
    
    # MACD
    new_df["MACD"] = new_df["EMA_12"] - new_df["EMA_26"]
    
    # MACD Signal
    macd_values = new_df["MACD"].values
    macd_signal = np.zeros_like(macd_values) + np.nan
    # Initialize with SMA
    period = 9
    valid_indices = ~np.isnan(macd_values)
    first_valid = np.where(valid_indices)[0][0] if np.any(valid_indices) else 0
    
    if first_valid + period < len(macd_values):
        macd_signal[first_valid+period-1] = np.mean(macd_values[first_valid:first_valid+period])
        # Calculate EMA
        alpha = 2 / (period + 1)
        for i in range(first_valid+period, len(macd_values)):
            if not np.isnan(macd_values[i]) and not np.isnan(macd_signal[i-1]):
                macd_signal[i] = macd_values[i] * alpha + macd_signal[i-1] * (1 - alpha)
    
    new_df["MACD_Signal"] = macd_signal
    new_df["MACD_Hist"] = new_df["MACD"] - new_df["MACD_Signal"]
    
    # RSI
    delta = np.zeros_like(close_values)
    delta[1:] = close_values[1:] - close_values[:-1]
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close_values) + np.nan
    avg_loss = np.zeros_like(close_values) + np.nan
    
    # Initialize with simple averages
    period = 14
    if len(gain) >= period:
        avg_gain[period-1] = np.mean(gain[:period])
        avg_loss[period-1] = np.mean(loss[:period])
        
        # Use EMA-like calculation after
        for i in range(period, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = avg_gain / np.where(avg_loss == 0, 0.00001, avg_loss)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    new_df["RSI"] = rsi
    
    # Bollinger Bands
    period = 20
    std_dev = np.zeros_like(close_values) + np.nan
    for i in range(period - 1, len(close_values)):
        std_dev[i] = np.std(close_values[i-period+1:i+1], ddof=0)
    
    new_df["BB_Middle"] = new_df["SMA_20"]
    new_df["BB_Std"] = std_dev
    new_df["BB_Upper"] = new_df["BB_Middle"] + 2 * new_df["BB_Std"]
    new_df["BB_Lower"] = new_df["BB_Middle"] - 2 * new_df["BB_Std"]
    new_df["BB_Width"] = (new_df["BB_Upper"] - new_df["BB_Lower"]) / new_df["BB_Middle"]
    
    # BB_Pct calculated directly
    bb_upper = new_df["BB_Upper"].values
    bb_lower = new_df["BB_Lower"].values
    bb_pct = np.zeros_like(close_values) + np.nan
    for i in range(len(close_values)):
        if not np.isnan(bb_upper[i]) and not np.isnan(bb_lower[i]):
            if bb_upper[i] - bb_lower[i] > 0:  # Avoid division by zero
                bb_pct[i] = (close_values[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i])
    
    new_df["BB_Pct"] = bb_pct
    
    # Returns
    returns = np.zeros_like(close_values) + np.nan
    returns[1:] = close_values[1:] / close_values[:-1] - 1
    new_df["Daily_Return"] = returns
    
    # N-day returns
    for period in [5, 10, 21, 60, 120]:
        ret_vals = np.zeros_like(close_values) + np.nan
        for i in range(period, len(close_values)):
            ret_vals[i] = close_values[i] / close_values[i-period] - 1
        new_df[f"Return_{period}d"] = ret_vals
    
    # Volatility indicators
    for period in [10, 21, 60]:
        vol_vals = np.zeros_like(close_values) + np.nan
        for i in range(period, len(returns)):
            if i >= period:
                vol_vals[i] = np.std(returns[i-period+1:i+1], ddof=0) * (252 ** 0.5)
        new_df[f"Hist_Vol_{period}"] = vol_vals
    
    # Average True Range (ATR)
    high_low = high_values - low_values
    high_close_prev = np.zeros_like(close_values) + np.nan
    high_close_prev[1:] = np.abs(high_values[1:] - close_values[:-1])
    low_close_prev = np.zeros_like(close_values) + np.nan
    low_close_prev[1:] = np.abs(low_values[1:] - close_values[:-1])
    
    true_range = np.zeros_like(close_values) + np.nan
    for i in range(1, len(close_values)):
        true_range[i] = max(high_low[i], high_close_prev[i], low_close_prev[i])
    
    new_df["TrueRange"] = true_range
    
    # ATR - 14-day average of true range
    atr = np.zeros_like(close_values) + np.nan
    period = 14
    if len(true_range) >= period:
        atr[period-1] = np.nanmean(true_range[:period])
        for i in range(period, len(true_range)):
            atr[i] = (atr[i-1] * 13 + true_range[i]) / 14
    
    new_df["ATR"] = atr
    new_df["ATR_Pct"] = atr / close_values
    
    # Stochastic Oscillator
    period = 14
    low_min = np.zeros_like(low_values) + np.nan
    high_max = np.zeros_like(high_values) + np.nan
    
    for i in range(period - 1, len(low_values)):
        low_min[i] = np.min(low_values[i-period+1:i+1])
        high_max[i] = np.max(high_values[i-period+1:i+1])
    
    stoch_k = np.zeros_like(close_values) + np.nan
    for i in range(period - 1, len(close_values)):
        if high_max[i] - low_min[i] > 0:  # Avoid division by zero
            stoch_k[i] = 100 * ((close_values[i] - low_min[i]) / (high_max[i] - low_min[i]))
    
    new_df["Stoch_K"] = stoch_k
    
    # Stoch_D (3-day SMA of Stoch_K)
    stoch_d = np.zeros_like(stoch_k) + np.nan
    period = 3
    for i in range(period - 1, len(stoch_k)):
        if not np.isnan(stoch_k[i-period+1:i+1]).any():
            stoch_d[i] = np.mean(stoch_k[i-period+1:i+1])
    
    new_df["Stoch_D"] = stoch_d
    
    # On-Balance Volume
    obv = np.zeros_like(volume_values)
    obv[0] = volume_values[0]
    for i in range(1, len(close_values)):
        if close_values[i] > close_values[i-1]:
            obv[i] = obv[i-1] + volume_values[i]
        elif close_values[i] < close_values[i-1]:
            obv[i] = obv[i-1] - volume_values[i]
        else:
            obv[i] = obv[i-1]
    
    new_df["OBV"] = obv
    
    # OBV SMA
    obv_sma = np.zeros_like(obv) + np.nan
    period = 20
    for i in range(period - 1, len(obv)):
        obv_sma[i] = np.mean(obv[i-period+1:i+1])
    
    new_df["OBV_SMA"] = obv_sma
    
    # Directional Movement Index
    plus_dm = np.zeros_like(high_values) + np.nan
    minus_dm = np.zeros_like(low_values) + np.nan
    
    for i in range(1, len(high_values)):
        high_diff = high_values[i] - high_values[i-1]
        low_diff = low_values[i-1] - low_values[i]
        
        if high_diff > 0 and high_diff > low_diff:
            plus_dm[i] = high_diff
        else:
            plus_dm[i] = 0
            
        if low_diff > 0 and low_diff > high_diff:
            minus_dm[i] = low_diff
        else:
            minus_dm[i] = 0
    
    # Smoothed DM and TR
    period = 14
    tr_sum = np.zeros_like(true_range) + np.nan
    plus_dm_sum = np.zeros_like(plus_dm) + np.nan
    minus_dm_sum = np.zeros_like(minus_dm) + np.nan
    
    if len(true_range) >= period:
        # Initialize with sum
        tr_sum[period-1] = np.nansum(true_range[:period])
        plus_dm_sum[period-1] = np.nansum(plus_dm[:period])
        minus_dm_sum[period-1] = np.nansum(minus_dm[:period])
        
        # Wilder's smoothing
        for i in range(period, len(true_range)):
            tr_sum[i] = tr_sum[i-1] - (tr_sum[i-1]/period) + true_range[i]
            plus_dm_sum[i] = plus_dm_sum[i-1] - (plus_dm_sum[i-1]/period) + plus_dm[i]
            minus_dm_sum[i] = minus_dm_sum[i-1] - (minus_dm_sum[i-1]/period) + minus_dm[i]
    
    # Calculate DI
    plus_di = np.zeros_like(plus_dm_sum) + np.nan
    minus_di = np.zeros_like(minus_dm_sum) + np.nan
    
    for i in range(len(tr_sum)):
        if not np.isnan(tr_sum[i]) and tr_sum[i] > 0:
            plus_di[i] = 100 * (plus_dm_sum[i] / tr_sum[i])
            minus_di[i] = 100 * (minus_dm_sum[i] / tr_sum[i])
    
    new_df["Plus_DI"] = plus_di
    new_df["Minus_DI"] = minus_di
    
    # DX and ADX
    dx = np.zeros_like(plus_di) + np.nan
    for i in range(len(plus_di)):
        if not np.isnan(plus_di[i]) and not np.isnan(minus_di[i]):
            if plus_di[i] + minus_di[i] > 0:  # Avoid division by zero
                dx[i] = 100 * np.abs((plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]))
    
    new_df["DX"] = dx
    
    # ADX (smoothed DX)
    adx = np.zeros_like(dx) + np.nan
    period = 14
    
    # Find first valid DX value
    valid_dx = ~np.isnan(dx)
    first_valid = np.where(valid_dx)[0][0] if np.any(valid_dx) else 0
    
    if first_valid + period < len(dx):
        # Initialize with mean
        adx[first_valid+period-1] = np.nanmean(dx[first_valid:first_valid+period])
        
        # Wilder's smoothing for subsequent values
        for i in range(first_valid+period, len(dx)):
            if not np.isnan(dx[i]) and not np.isnan(adx[i-1]):
                adx[i] = ((period-1) * adx[i-1] + dx[i]) / period
    
    new_df["ADX"] = adx
    
    # Strong trend detection
    strong_trend = np.zeros_like(adx)
    for i in range(1, len(adx)):
        if not np.isnan(adx[i]) and not np.isnan(adx[i-1]):
            if adx[i] > 25 and adx[i] > adx[i-1]:
                strong_trend[i] = 1
    
    new_df["Strong_Trend"] = strong_trend
    
    # Price relative to moving averages
    for ma in ["SMA_10", "SMA_30", "SMA_50", "SMA_200"]:
        ma_vals = new_df[ma].values
        price_to_ma = np.zeros_like(close_values) + np.nan
        for i in range(len(close_values)):
            if not np.isnan(ma_vals[i]) and ma_vals[i] > 0:
                price_to_ma[i] = close_values[i] / ma_vals[i] - 1
        
        new_df[f"Price_to_{ma}"] = price_to_ma
    
    # Moving average slopes
    for ma, period in [("SMA_10", 5), ("SMA_30", 5), ("SMA_50", 10), ("SMA_200", 20)]:
        ma_vals = new_df[ma].values
        ma_slope = np.zeros_like(ma_vals) + np.nan
        
        for i in range(period, len(ma_vals)):
            if not np.isnan(ma_vals[i]) and not np.isnan(ma_vals[i-period]) and ma_vals[i-period] > 0:
                ma_slope[i] = ma_vals[i] / ma_vals[i-period] - 1
        
        new_df[f"{ma}_Slope"] = ma_slope
    
    # Golden Cross / Death Cross
    golden_cross = np.zeros_like(close_values)
    death_cross = np.zeros_like(close_values)
    
    sma50_vals = new_df["SMA_50"].values
    sma200_vals = new_df["SMA_200"].values
    
    for i in range(1, len(sma50_vals)):
        if (not np.isnan(sma50_vals[i]) and not np.isnan(sma200_vals[i]) and 
            not np.isnan(sma50_vals[i-1]) and not np.isnan(sma200_vals[i-1])):
            
            if sma50_vals[i] > sma200_vals[i] and sma50_vals[i-1] <= sma200_vals[i-1]:
                golden_cross[i] = 1
                
            if sma50_vals[i] < sma200_vals[i] and sma50_vals[i-1] >= sma200_vals[i-1]:
                death_cross[i] = 1
    
    new_df["Golden_Cross"] = golden_cross
    new_df["Death_Cross"] = death_cross
    
    # Bull Market
    bull_market = np.zeros_like(close_values)
    for i in range(len(sma50_vals)):
        if not np.isnan(sma50_vals[i]) and not np.isnan(sma200_vals[i]):
            if sma50_vals[i] > sma200_vals[i]:
                bull_market[i] = 1
    
    new_df["Bull_Market"] = bull_market
    
    # Higher Highs and Lower Lows
    higher_high = np.zeros_like(high_values)
    lower_low = np.zeros_like(low_values)
    
    for i in range(2, len(high_values)):
        if high_values[i] > high_values[i-1] and high_values[i-1] > high_values[i-2]:
            higher_high[i] = 1
            
        if low_values[i] < low_values[i-1] and low_values[i-1] < low_values[i-2]:
            lower_low[i] = 1
    
    new_df["Higher_High"] = higher_high
    new_df["Lower_Low"] = lower_low
    
    # Volume indicators
    volume_sma20 = np.zeros_like(volume_values) + np.nan
    period = 20
    for i in range(period - 1, len(volume_values)):
        volume_sma20[i] = np.mean(volume_values[i-period+1:i+1])
    
    new_df["Volume_SMA20"] = volume_sma20
    
    # Volume ratio
    volume_ratio = np.zeros_like(volume_values) + np.nan
    for i in range(len(volume_values)):
        if not np.isnan(volume_sma20[i]) and volume_sma20[i] > 0:
            volume_ratio[i] = volume_values[i] / volume_sma20[i]
    
    new_df["Volume_Ratio"] = volume_ratio
    
    # Volume spike
    volume_spike = np.zeros_like(volume_values)
    for i in range(len(volume_values)):
        if not np.isnan(volume_sma20[i]) and volume_values[i] > 2 * volume_sma20[i]:
            volume_spike[i] = 1
    
    new_df["Volume_Spike"] = volume_spike
    
    # RSI features
    rsi_ma5 = np.zeros_like(rsi) + np.nan
    period = 5
    for i in range(period - 1, len(rsi)):
        rsi_slice = rsi[i-period+1:i+1]
        if not np.isnan(rsi_slice).any():
            rsi_ma5[i] = np.mean(rsi_slice)
    
    new_df["RSI_MA5"] = rsi_ma5
    
    # RSI trend
    rsi_trend = np.zeros_like(rsi)
    for i in range(len(rsi)):
        if not np.isnan(rsi[i]) and not np.isnan(rsi_ma5[i]):
            if rsi[i] > rsi_ma5[i]:
                rsi_trend[i] = 1
    
    new_df["RSI_Trend"] = rsi_trend
    
    # RSI extremes
    rsi_extreme_high = np.zeros_like(rsi)
    rsi_extreme_low = np.zeros_like(rsi)
    
    for i in range(len(rsi)):
        if not np.isnan(rsi[i]):
            if rsi[i] > 70:
                rsi_extreme_high[i] = 1
            if rsi[i] < 30:
                rsi_extreme_low[i] = 1
    
    new_df["RSI_Extreme_High"] = rsi_extreme_high
    new_df["RSI_Extreme_Low"] = rsi_extreme_low
    
    # Rate of Change (ROC)
    for period in [5, 10, 21]:
        roc = np.zeros_like(close_values) + np.nan
        for i in range(period, len(close_values)):
            if close_values[i-period] > 0:  # Avoid division by zero
                roc[i] = ((close_values[i] - close_values[i-period]) / close_values[i-period]) * 100
        
        new_df[f"ROC_{period}"] = roc
    
    # Make sure all column types are float
    for col in new_df.columns:
        if col != "Date" and col in new_df.columns:
            new_df[col] = new_df[col].astype(float)
    
    # Final comprehensive NaN filling
    # First, forward-fill to carry last valid observation forward
    new_df.ffill(inplace=True)
    # Then, back-fill to fill any remaining NaNs at the beginning
    new_df.bfill(inplace=True)
    # As a last resort, fill any remaining NaNs (e.g., if an entire column was NaN) with 0
    new_df.fillna(0, inplace=True)

    # Return the new dataframe
    return new_df


def split_data(df: pd.DataFrame, ratio: float):
    idx = int(len(df) * ratio)
    return df.iloc[:idx].reset_index(drop=True), df.iloc[idx:].reset_index(drop=True)

# ─────────────────────── 3. Custom Gym Env ──────────────────────────

class EmbeddingBasedStockTradingEnv(gym.Env):
    """
    Stock Trading Environment that works with pre-computed TabularBERT embeddings.
    This environment is used during the fine-tuning phase.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        embeddings: torch.Tensor,  # Pre-computed embeddings
        df: pd.DataFrame,  # Original dataframe for price info
        lookback_window: int,
        initial_balance: float,
        transaction_cost: float,
        capital_cost: float,
        risk_free_step: float,
        episode_length: int,
        random_start: bool = True,
        allow_short: bool = True,
        max_position: float = 1.0,
        drawdown_penalty: float = 1.0,
        volatility_scaling: bool = True,
        min_holding_period: int = 5,
        reward_trade_penalty: float = 0.0015,
        wrong_side_penalty_factor: float = 0.5,
    ):
        super().__init__()
        if df.empty:
            raise ValueError("Empty DataFrame given to environment.")
        if len(embeddings) == 0:
            raise ValueError("Empty embeddings given to environment.")

        self.embeddings = embeddings  # Shape: (num_sequences, seq_len, embedding_dim)
        self.df = df.reset_index(drop=True)
        self.lookback_window = lookback_window
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.capital_cost = capital_cost
        self.risk_free_step = risk_free_step
        self.episode_length = episode_length
        self.random_start = random_start
        self.allow_short = allow_short
        self.max_position = max_position
        self.drawdown_penalty = drawdown_penalty
        self.volatility_scaling = volatility_scaling
        self.min_holding_period = min_holding_period
        self.trade_penalty = reward_trade_penalty
        self.wrong_side_penalty_factor = wrong_side_penalty_factor

        # actions: 0 = Flat, 1 = Long, 2 = Short (if allow_short is True)
        self.action_space = spaces.Discrete(3 if allow_short else 2)

        # Observation space using embeddings
        embedding_dim = embeddings.shape[-1]
        self.observation_space = spaces.Dict({
            # Pre-computed embeddings sequence
            'embeddings_seq': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(lookback_window, embedding_dim), 
                dtype=np.float32
            ),
            # Agent state variables
            'agent_state': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(5,),  # position%, cash%, drawdown, steps_since_trade, position_days
                dtype=np.float32
            )
        })
        
        self.reset()

    def _get_observation(self):
        """Get observation using pre-computed embeddings."""
        # Get the corresponding embedding sequence
        embedding_seq = self.embeddings[self.current_embedding_idx].numpy()
        
        # Get current price for state calculation
        price = float(self.df.loc[self.current_step, "Close"])
        position_value = self.shares_held * price
        position_pct = position_value / self.net_worth if self.net_worth > 0 else 0
        cash_pct = self.balance / self.net_worth if self.net_worth > 0 else 0
        drawdown = (self.net_worth / self.max_net_worth) - 1.0 if self.max_net_worth > 0 else 0
        steps_since_trade_norm = min(self.steps_since_last_trade / 20.0, 1.0)
        position_days_norm = min(self.current_position_days / 20.0, 1.0)
        
        agent_state = np.array([
            position_pct, cash_pct, drawdown, steps_since_trade_norm, position_days_norm
        ], dtype=np.float32)
        
        return {
            'embeddings_seq': embedding_seq.astype(np.float32),
            'agent_state': agent_state
        }

    def _take_action(self, action: int):
        """Same action logic as original environment."""
        price = float(self.df.loc[self.current_step, "Close"])
        
        # Skip action if we haven't met the minimum holding period
        skip_due_to_holding_period = False
        if self.current_position_days < self.min_holding_period:
            if self.shares_held > 0 and action != 1:
                skip_due_to_holding_period = True
            elif self.shares_held < 0 and action != 2:
                skip_due_to_holding_period = True
        
        # Override holding period if significant drawdown
        current_drawdown = 0
        if self.max_net_worth > 0:
            current_drawdown = 1.0 - (self.net_worth / self.max_net_worth)
        
        if current_drawdown > 0.10 and skip_due_to_holding_period:
            skip_due_to_holding_period = False
        
        if skip_due_to_holding_period:
            if self.shares_held > 0:
                action = 1
            elif self.shares_held < 0:
                action = 2
            else:
                action = 0
        
        # Volatility-based position sizing
        position_scale = 1.0
        if self.volatility_scaling:
            position_scale = 0.20 / max(0.10, 0.20)  # Simplified for embedding env
        
        target_position_ratio = 0.0
        if action == 1:  # Long
            target_position_ratio = self.max_position * position_scale
        elif action == 2 and self.allow_short:  # Short
            target_position_ratio = -self.max_position * position_scale
        
        target_position_value = target_position_ratio * self.net_worth
        target_shares = target_position_value / price if price > 0 else 0
        shares_to_trade = target_shares - self.shares_held
        
        executed_trade = False
        if abs(shares_to_trade) * price < 0.01 * self.initial_balance:
            return executed_trade
            
        # Execute trade logic (simplified version)
        if shares_to_trade > 0:  # Buy
            cost = shares_to_trade * price * (1 + self.transaction_cost)
            if cost <= self.balance:
                self.balance -= cost
                self.shares_held += shares_to_trade
                self.trade_history.append(("BUY", self.current_step, price, shares_to_trade))
                executed_trade = True
                if self.current_position_days < 0:
                    self.current_position_days = 0
            else:
                affordable_shares = self.balance / (price * (1 + self.transaction_cost))
                if affordable_shares > 0.001:
                    cost = affordable_shares * price * (1 + self.transaction_cost)
                    self.balance -= cost
                    self.shares_held += affordable_shares
                    self.trade_history.append(("BUY", self.current_step, price, affordable_shares))
                    executed_trade = True
                    if self.current_position_days < 0:
                        self.current_position_days = 0
                
        elif shares_to_trade < 0:  # Sell
            shares_to_sell = -shares_to_trade
            if self.shares_held > 0:
                shares_to_sell = min(shares_to_sell, self.shares_held)
                
            trade_value = shares_to_sell * price
            fee = trade_value * self.transaction_cost
            self.balance += trade_value - fee
            self.shares_held -= shares_to_sell
            self.trade_history.append(("SELL", self.current_step, price, shares_to_sell))
            executed_trade = True
            
            if self.current_position_days > 0 and self.shares_held <= 0:
                self.current_position_days = 0 if self.shares_held == 0 else -1
                
        # Update net worth
        self.net_worth = self.balance + self.shares_held * price
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        
        return executed_trade

    def _calculate_reward(self, prev_worth):
        """Same reward calculation as original environment."""
        if prev_worth > 1e-9:
            base_return = np.log((self.net_worth + 1e-9) / (prev_worth + 1e-9))
        else:
            base_return = 0
            
        price = float(self.df.loc[self.current_step, "Close"])
        prev_price = float(self.df.loc[self.current_step - 1, "Close"]) if self.current_step > 0 else price
        market_return = np.log((price + 1e-9) / (prev_price + 1e-9))
        
        alpha = base_return - market_return
        
        drawdown_penalty = 0
        if self.max_net_worth > 0:
            drawdown = max(0, 1.0 - (self.net_worth / self.max_net_worth))
            if drawdown > 0.05:
                drawdown_penalty = self.drawdown_penalty * (drawdown - 0.05)
        
        capital_charge = 0
        if self.shares_held != 0:
            position_size = abs(self.shares_held * price / self.net_worth) if self.net_worth > 0 else 0
            capital_charge = self.capital_cost * position_size
            base_return -= capital_charge
        else:
            base_return = np.log(1 + self.risk_free_step)
        
        reward = base_return - drawdown_penalty
        
        # Wrong side penalty
        wrong_side_penalty = 0
        significant_move_threshold = 0.015
        if self.shares_held > 0 and market_return < -significant_move_threshold:
            wrong_side_penalty = self.wrong_side_penalty_factor * abs(market_return)
        elif self.shares_held < 0 and market_return > significant_move_threshold:
            wrong_side_penalty = self.wrong_side_penalty_factor * market_return

        reward -= wrong_side_penalty
        
        self.reward_components = {
            'base_return': base_return,
            'market_return': market_return,
            'drawdown_penalty': drawdown_penalty,
            'capital_charge': capital_charge,
            'wrong_side_penalty': wrong_side_penalty,
            'total_reward': reward
        }
        
        return reward

    def step(self, action):
        prev_worth = self.net_worth

        action = int(action if not isinstance(action, np.ndarray) else action.item())
        action = max(0, min(action, self.action_space.n - 1))
        
        trade_executed = self._take_action(action)

        # Move to next step
        self.current_step += 1
        self.current_embedding_idx += 1  # Move to next embedding
        self.steps_in_episode += 1
        
        self.steps_since_last_trade += 1
        
        if self.shares_held > 0:
            self.current_position_days += 1
        elif self.shares_held < 0:
            self.current_position_days -= 1
        else:
            self.current_position_days = 0
        
        if trade_executed:
            self.steps_since_last_trade = 0

        # Check termination conditions
        truncated = (self.steps_in_episode >= self.episode_length) or \
                   (self.current_embedding_idx >= len(self.embeddings)) or \
                   (self.current_step >= len(self.df) - 1)
        terminated = False
        
        reward = self._calculate_reward(prev_worth)
        
        if trade_executed:
            trade_penalty_factor = max(1.0, 3.0 - (self.steps_since_last_trade / 10.0))
            reward -= self.trade_penalty * trade_penalty_factor

        info = {
            "step": self.current_step,
            "embedding_idx": self.current_embedding_idx,
            "balance": self.balance,
            "shares_held": self.shares_held,
            "net_worth": self.net_worth,
            "max_net_worth": self.max_net_worth,
            "action": action,
            "trade_executed": trade_executed,
            "steps_since_last_trade": self.steps_since_last_trade,
            "current_position_days": self.current_position_days,
            "reward_components": self.reward_components,
        }
        
        return self._get_observation(), reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        
        # Choose random starting point for embeddings
        max_start_idx = len(self.embeddings) - self.episode_length
        if self.random_start and max_start_idx > 0:
            self.current_embedding_idx = int(self.np_random.integers(0, max_start_idx))
        else:
            self.current_embedding_idx = 0
            
        # Set corresponding step in dataframe and episode start for evaluation compatibility
        self.current_step = self.lookback_window + self.current_embedding_idx
        self.episode_start = self.current_step  # For evaluation function compatibility

        self.balance = float(self.initial_balance)
        self.shares_held = 0.0
        self.net_worth = float(self.initial_balance)
        self.max_net_worth = float(self.initial_balance)
        self.trade_history = []
        self.steps_in_episode = 0
        self.reward_components = {}
        self.steps_since_last_trade = 0
        self.current_position_days = 0

        obs = self._get_observation()
        info = {}
        return obs, info

# ─────────────────────── 4. Train PPO ───────────────────────────────

def make_vec_env_embeddings(embeddings, df, cfg):
    """Create vectorized environment for embedding-based training."""
    def _init():
        # Use confidence-based environment if enabled
        if cfg.get("use_confidence_trading", False) and CONFIDENCE_IMPORTS_AVAILABLE:
            env = ConfidenceBasedTradingEnv(
                embeddings,
                df,
                cfg["lookback_window"],
                cfg["initial_balance"],
                cfg["transaction_cost"],
                cfg["capital_cost"],
                cfg["risk_free_rate"] / 252,
                cfg["episode_length"],
                random_start=True,
                allow_short=cfg["allow_short"],
                max_position=cfg["max_position"],
                drawdown_penalty=cfg["drawdown_penalty"],
                volatility_scaling=cfg["volatility_scaling"],
                min_holding_period=cfg["min_holding_period"],
                reward_trade_penalty=cfg["reward_trade_penalty"],
                wrong_side_penalty_factor=cfg.get("wrong_side_penalty_factor", 0.5),
                # Confidence-specific parameters
                confidence_min_threshold=cfg.get("confidence_min_threshold", 0.3),
                confidence_position_multiplier=cfg.get("confidence_position_multiplier", 2.0),
                confidence_scaling=cfg.get("confidence_scaling", True),
            )
        else:
            env = EmbeddingBasedStockTradingEnv(
                embeddings,
                df,
                cfg["lookback_window"],
                cfg["initial_balance"],
                cfg["transaction_cost"],
                cfg["capital_cost"],
                cfg["risk_free_rate"] / 252,
                cfg["episode_length"],
                random_start=True,
                allow_short=cfg["allow_short"],
                max_position=cfg["max_position"],
                drawdown_penalty=cfg["drawdown_penalty"],
                volatility_scaling=cfg["volatility_scaling"],
                min_holding_period=cfg["min_holding_period"],
                reward_trade_penalty=cfg["reward_trade_penalty"],
                wrong_side_penalty_factor=cfg.get("wrong_side_penalty_factor", 0.5),
            )
        return env
    
    env_type = "confidence-based" if cfg.get("use_confidence_trading", False) else "embedding-based"
    print(f"Using DummyVecEnv for {env_type} training")
    return DummyVecEnv([_init for _ in range(1)])

def train_ppo_agent(train_df, cfg, feats, train_embeddings):
    """
    Train PPO agent with TabularBERT embeddings.
    
    Args:
        train_df: Training dataframe
        cfg: Configuration dictionary
        feats: Features list (not used in BERT approach)
        train_embeddings: Pre-computed embeddings from TabularBERT
    """
    
    # Only BERT approach is supported
    if train_embeddings is None:
        raise ValueError("BERT approach requires pre-computed embeddings from TabularBERT pre-training")
    
    print("Using BERT-based approach with pre-computed embeddings")
    vec_env = make_vec_env_embeddings(train_embeddings, train_df, cfg)
    
    # Use enhanced BERT policy if confidence trading is enabled
    if (cfg.get("use_enhanced_bert_policy", False) and 
        cfg.get("use_confidence_trading", False) and 
        CONFIDENCE_IMPORTS_AVAILABLE):
        print("🎯 Using Enhanced BERT Policy with confidence features...")
        policy_class = EnhancedBERTBasedActorCriticPolicy
        policy_kwargs = create_enhanced_bert_policy_kwargs(cfg)
    else:
        policy_class = BERTBasedActorCriticPolicy
        policy_kwargs = {
            "embedding_dim":       cfg.get("embedding_dim", 128),  # Use embedding_dim from config
            "latent_dim_pi_out":   cfg["latent_dim_pi_out"],
            "latent_dim_vf_out":   cfg["latent_dim_vf_out"],
            "dropout":             cfg.get("bert_dropout", 0.1),  # Use bert_dropout from config
        }

    model = PPO(
        policy_class,
        vec_env,
        n_steps       = cfg["ppo_n_steps"],
        batch_size    = cfg["ppo_batch_size"],
        n_epochs      = cfg["ppo_n_epochs"],
        gamma         = cfg["ppo_gamma"],
        gae_lambda    = cfg["ppo_gae_lambda"],
        clip_range    = cfg["ppo_clip_range"],
        ent_coef      = cfg["ppo_ent_coef"],
        vf_coef       = cfg["ppo_vf_coef"],
        learning_rate = cfg["ppo_learning_rate"],
        verbose       = 1,
        tensorboard_log="./ppo_stock_tensorboard/",
        policy_kwargs = policy_kwargs
    )
    
    # For BERT approach, only task-specific heads are trained (backbone is pre-computed embeddings)
    print("Training with BERT approach - only task-specific heads will be trained")
    print("Pre-trained embeddings are used as frozen features")
    
    model.learn(total_timesteps=cfg["ppo_total_timesteps"])
    
    # Save the model with appropriate name
    model_type_name = cfg["model_type"].lower().replace("_", "-")
    bert_tag = "_bert" if cfg["model_type"].lower() == "bert_transformer" else ""
    filename = f"ppo_{cfg['ticker']}_{model_type_name}{bert_tag}_model"
    model.save(f"{filename}.zip")
    
    vec_env.close()
    return model

# ─────────────────────── 5. Evaluation / Plot ───────────────────────

def _sharpe_ratio(returns, freq=252):
    if returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(freq)


def _max_drawdown(values):
    vals = np.array(values)
    cummax = np.maximum.accumulate(vals)
    drawdowns = (vals - cummax) / cummax
    return drawdowns.min()


def evaluate_agent(model, env, cfg):
    # Create confidence-aware wrapper if available and needed
    if (CONFIDENCE_IMPORTS_AVAILABLE and 
        hasattr(env, 'confidence_scaling') and 
        hasattr(model.policy, 'get_confidence_score')):
        print("🎯 Using confidence-aware model wrapper for evaluation...")
        model = create_confidence_aware_wrapper(model, env)
    
    # Reset environment and get initial observation
    obs, _ = env.reset()
    
    done = False
    rewards, net_vals = [], [env.initial_balance]
    action_counts = {0: 0, 1: 0, 2: 0}  # Flat, Long, Short
    position_history = []  # Track position for visualization
    dd_history = []  # Track drawdown history
    reward_components_history = []  # Track reward components

    # Add initial position (zero at start)
    price = float(env.df.loc[env.current_step, "Close"])
    position_value = env.shares_held * price  # Will be 0 at start
    position_pct = position_value / env.initial_balance if env.initial_balance > 0 else 0
    position_history.append(position_pct)
    dd_history.append(0.0)  # No drawdown at start
    
    # Get device from model
    device = getattr(model, "device", "cpu")

    while not done:
        # Handle prediction correctly for dictionary observations
        # This bypasses SB3's default feature extraction which tries to flatten dict observations
        try:
            # Convert numpy arrays to tensors for model input
            obs_tensors = {
                k: torch.as_tensor(v, device=device).float().unsqueeze(0)
                for k, v in obs.items()
            }
            
            # Use policy directly to avoid SB3's default processing pipeline
            with torch.no_grad():
                actions, _, _ = model.policy.forward(obs_tensors)
                action_int = int(actions.cpu().numpy().flatten()[0])
        except Exception as e:
            print(f"Warning: Direct policy prediction failed: {e}")
            print("Falling back to SB3's predict method - this may fail if observations are dictionaries")
            # Fallback to standard predict (will likely fail with dict observations)
            action, _ = model.predict(obs, deterministic=True)
            action_int = int(action if not isinstance(action, np.ndarray) else action.item())

        # Take step in environment
        obs, r, terminated, truncated, info = env.step(action_int)
        rewards.append(r)
        net_vals.append(info["net_worth"])
        
        # Calculate position as percentage of net worth
        position_value = env.shares_held * float(env.df.loc[env.current_step, "Close"])
        position_pct = position_value / info["net_worth"] if info["net_worth"] > 0 else 0
        position_history.append(position_pct)
        
        # Calculate drawdown
        dd_history.append(1.0 - (info["net_worth"] / info["max_net_worth"]))
        
        # Store reward components if available
        if "reward_components" in info:
            reward_components_history.append(info["reward_components"])
            
        action_counts[action_int] += 1
        done = terminated or truncated

    # Buy & Hold baseline (includes one‑way transaction cost)
    lookback = env.lookback_window
    start_ix = env.episode_start
    first_price = env.df.loc[start_ix, "Close"]
    shares_bh = env.initial_balance / (first_price * (1 + env.transaction_cost))
    bh_series = shares_bh * env.df["Close"].iloc[start_ix : start_ix + len(net_vals)].reset_index(drop=True)

    # Ensure all arrays have exactly the same length
    min_len = min(len(net_vals), len(bh_series), len(position_history), len(dd_history))
    net_vals = net_vals[:min_len]
    bh_series = bh_series.iloc[:min_len]
    position_history = position_history[:min_len]
    dd_history = dd_history[:min_len]
    steps = np.arange(min_len)

    # Extract and group trade markers
    trades = []
    for trade in env.trade_history:
        # Check if trade format has changed
        if len(trade) == 4:  # New format with shares
            trade_type, step, price, shares = trade
            adjusted_step = step - start_ix
            if 0 <= adjusted_step < min_len:
                trades.append((trade_type, adjusted_step, price, shares, net_vals[adjusted_step]))
        else:  # Old format without shares for backward compatibility
            trade_type, step, price = trade
            adjusted_step = step - start_ix
            if 0 <= adjusted_step < min_len:
                trades.append((trade_type, adjusted_step, price, 0, net_vals[adjusted_step]))

    # Create plots with more information
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Portfolio Value Plot
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(steps, net_vals, label="PPO Agent", linewidth=2, zorder=2)
    ax1.plot(steps, bh_series, label="Buy & Hold", linestyle="--", linewidth=1.5, zorder=1)
    
    # Mark trades with direction indicators
    buy_steps = [t[1] for t in trades if t[0] == "BUY"]
    sell_steps = [t[1] for t in trades if t[0] == "SELL"]
    buy_vals = [t[4] for t in trades if t[0] == "BUY"]
    sell_vals = [t[4] for t in trades if t[0] == "SELL"]
    
    if buy_steps:
        ax1.scatter(buy_steps, buy_vals, marker="^", s=100, color="green", label="Buy", zorder=3)
    if sell_steps:
        ax1.scatter(sell_steps, sell_vals, marker="v", s=100, color="red", label="Sell", zorder=3)
    
    ax1.set_title(f"{cfg['ticker']} – Enhanced PPO Trading Strategy Performance", fontsize=14)
    ax1.set_ylabel("Portfolio Value ($)", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")
    
    # 2. Position Size as % of Portfolio
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    position_arr = np.array(position_history)
    
    # Color positions by direction (positive=long, negative=short)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    ax2.fill_between(steps, position_arr, 0, 
                     where=(position_arr >= 0), color='green', alpha=0.3, label='Long')
    ax2.fill_between(steps, position_arr, 0, 
                     where=(position_arr <= 0), color='red', alpha=0.3, label='Short')
    ax2.plot(steps, position_arr, color='blue', linewidth=1)
    
    ax2.set_ylabel("Position Size\n(% of Portfolio)", fontsize=12)
    ax2.set_ylim(-1.1, 1.1)  # Allow for short positions
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")
    
    # 3. Drawdown Plot
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    
    ax3.fill_between(steps, dd_history, 0, color='purple', alpha=0.3)
    ax3.plot(steps, dd_history, color='purple', linewidth=1)
    ax3.set_ylabel("Drawdown (%)", fontsize=12)
    ax3.set_xlabel("Steps in test period", fontsize=12)
    ax3.set_ylim(0, max(max(dd_history) + 0.05, 0.01))
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.25)
    plt.show()
    
    # Risk metrics
    agent_returns = np.diff(net_vals) / np.array(net_vals[:-1])
    bh_returns = np.diff(bh_series) / np.array(bh_series[:-1])

    sharpe_agent = _sharpe_ratio(agent_returns)
    sharpe_bh = _sharpe_ratio(bh_returns)
    mdd_agent = _max_drawdown(net_vals)
    mdd_bh = _max_drawdown(bh_series)
    
    # Calculate more advanced metrics
    # Sortino ratio (downside risk only)
    downside_returns = agent_returns.copy()
    downside_returns[downside_returns > 0] = 0
    sortino_ratio = (np.mean(agent_returns) / np.std(downside_returns)) * np.sqrt(252) if np.std(downside_returns) > 0 else 0
    
    # Calmar ratio (return / max drawdown)
    total_return = (net_vals[-1] / net_vals[0]) - 1
    calmar_ratio = (total_return / abs(mdd_agent)) if abs(mdd_agent) > 0 else 0
    
    # Calculate win rate and average return per trade
    if env.trade_history:
        # Extract the net worth before and after each trade
        trade_returns = []
        for i in range(1, len(env.trade_history)):
            prev_trade = env.trade_history[i-1]
            curr_trade = env.trade_history[i]
            
            # Handle both old and new trade formats
            prev_step = prev_trade[1] if len(prev_trade) >= 2 else 0
            curr_step = curr_trade[1] if len(curr_trade) >= 2 else 0
            
            if prev_step < curr_step and prev_step >= start_ix and curr_step < start_ix + min_len:
                prev_worth = net_vals[prev_step - start_ix]
                curr_worth = net_vals[curr_step - start_ix]
                trade_return = (curr_worth / prev_worth) - 1
                trade_returns.append(trade_return)
        
        win_rate = np.mean([r > 0 for r in trade_returns]) if trade_returns else 0
        avg_win = np.mean([r for r in trade_returns if r > 0]) if any(r > 0 for r in trade_returns) else 0
        avg_loss = np.mean([r for r in trade_returns if r <= 0]) if any(r <= 0 for r in trade_returns) else 0
        profit_factor = -np.sum([r for r in trade_returns if r > 0]) / np.sum([r for r in trade_returns if r <= 0]) if np.sum([r for r in trade_returns if r <= 0]) != 0 else float('inf')
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0

    # Stats printout
    final_agent = net_vals[-1]
    final_bh = float(bh_series.iloc[-1])

    print("\n═════════ Strategy Evaluation ═════════")
    print(f"Initial balance    : ${env.initial_balance:,.2f}")
    print(f"PPO final value    : ${final_agent:,.2f}  "
          f"({(final_agent/env.initial_balance - 1)*100:.2f} %)")
    print(f"Buy & Hold value   : ${final_bh:,.2f}  "
          f"({(final_bh/env.initial_balance - 1)*100:.2f} %)")
    print(f"Total reward       : {np.sum(rewards):.4f}")
    
    print("\n─────── Risk Metrics ───────")
    print(f"Sharpe Ratio       : {sharpe_agent:.2f} (PPO) / {sharpe_bh:.2f} (B&H)")
    print(f"Sortino Ratio      : {sortino_ratio:.2f}")
    print(f"Calmar Ratio       : {calmar_ratio:.2f}")
    print(f"Max Drawdown       : {mdd_agent:.2%} (PPO) / {mdd_bh:.2%} (B&H)")
    
    print("\n─────── Trading Statistics ───────")
    print(f"Actions            : Flat={action_counts[0]}, Long={action_counts[1]}, Short={action_counts.get(2, 0)}")
    print(f"Number of Trades   : {len(env.trade_history)}")
    print(f"Win Rate           : {win_rate:.2%}")
    print(f"Avg Win/Loss       : {avg_win:.2%} / {avg_loss:.2%}")
    print(f"Profit Factor      : {profit_factor:.2f}")
    
    # Add confidence metrics if available
    if hasattr(env, 'confidence_history') and env.confidence_history:
        avg_confidence = np.mean(env.confidence_history)
        confidence_std = np.std(env.confidence_history)
        high_confidence_trades = sum(1 for c in env.confidence_history if c > 0.7)
        print(f"\n─────── Confidence Metrics ───────")
        print(f"Avg Confidence     : {avg_confidence:.3f} ± {confidence_std:.3f}")
        print(f"High Conf Trades   : {high_confidence_trades}/{len(env.confidence_history)} ({high_confidence_trades/len(env.confidence_history)*100:.1f}%)")
        print(f"Confidence Range   : {min(env.confidence_history):.3f} - {max(env.confidence_history):.3f}")
    
    # Return metrics for potential automation/comparison
    metrics = {
        'final_value': final_agent,
        'total_return': (final_agent/env.initial_balance - 1),
        'sharpe': sharpe_agent,
        'sortino': sortino_ratio, 
        'max_drawdown': mdd_agent,
        'calmar': calmar_ratio,
        'win_rate': win_rate,
        'trades': len(env.trade_history)
    }
    
    return metrics

# ─────────────────────── 6. Main driver ─────────────────────────────

def main():
    print("── AdamTrading PPO demo – improved with optional pre-training ──")
    print(f"Using {CONFIG['model_type'].upper()} model for sequence modeling")
    
    try:
        # Process data and prepare environments
        raw = fetch_stock_data(CONFIG["ticker"], CONFIG["start_date"], CONFIG["end_date"])
        proc = add_indicators(raw)

        # Split before normalisation to avoid data leakage
        train_df, test_df = split_data(proc, CONFIG["train_split_ratio"])

        # Select which features to normalize
        norm_cols = [
            # Price and Volume
            "Close", "Volume", 
            
            # Moving Averages (reduced set)
            "SMA_10", "SMA_50", "SMA_200",
            "EMA_9", "EMA_21",
            
            # MACD
            "MACD", "MACD_Signal",
            
            # Oscillators
            "RSI", "Stoch_K",
            
            # Volatility
            "ATR_Pct", "Hist_Vol_21", 
            
            # Returns
            "Return_5d", "Return_21d",
            
            # Price Relatives
            "Price_to_SMA50", "Price_to_SMA200",
            
            # Momentum
            "ROC_21",
            
            # Trend
            "ADX", "Plus_DI", "Minus_DI",
        ]
        
        # Categorical/Binary features don't need normalization
        binary_feats = [
            "Bull_Market", "Golden_Cross", "Death_Cross", 
            "RSI_Extreme_High", "RSI_Extreme_Low"
        ]
        
        norm_params = {}

        # Fit scaling on training slice only
        for col in norm_cols:
            if col in train_df.columns:
                mean = train_df[col].mean()
                std = train_df[col].std(ddof=0)
                # Avoid division by zero if std is 0
                norm_params[col] = {"mean": mean, "std": std if std > 1e-9 else 1.0}

        # Apply normalizations to both datasets
        for df in (train_df, test_df):
            for col in norm_cols:
                if col in df.columns and col in norm_params:
                    params = norm_params[col]
                    df[f"{col}_Norm"] = (df[col] - params["mean"]) / params["std"]
            
            # Add binary features directly (no normalization needed)
            for col in binary_feats:
                if col in df.columns:
                    df[f"{col}_Feat"] = df[col].astype(float)

        # Feature selection - core set of features, reduced to avoid overfitting
        core_features = [
            # Most important features
            "Close_Norm", "Volume_Norm", 
            "SMA_50_Norm", "SMA_200_Norm",
            "RSI_Norm", "ATR_Pct_Norm",
            "Return_21d_Norm", 
            "Price_to_SMA200_Norm", 
            "ADX_Norm", 
            "Bull_Market_Feat"
        ]
        
        # Additional features to consider
        additional_features = [
            "MACD_Norm",
            "Stoch_K_Norm",
            "Hist_Vol_21_Norm",
            "Return_5d_Norm",
            "Price_to_SMA50_Norm",
            "ROC_21_Norm",
            "Plus_DI_Norm", "Minus_DI_Norm",
            "Golden_Cross_Feat", "Death_Cross_Feat",
            "RSI_Extreme_High_Feat", "RSI_Extreme_Low_Feat"
        ]
        
        # Filter to only include features that actually exist in the dataframe
        core_features = [f for f in core_features if f in train_df.columns]
        additional_features = [f for f in additional_features if f in train_df.columns]
        
        # Apply feature dimensionality reduction
        max_features = CONFIG["max_features"]
        available_features = core_features
        
        # Add additional features up to max_features
        remaining_slots = max_features - len(available_features)
        if remaining_slots > 0 and additional_features:
            # Use top N from additional features
            available_features.extend(additional_features[:remaining_slots])
        
        # Ensure we don't exceed max_features
        available_features = available_features[:max_features]
        
        print(f"Using {len(available_features)} features for the model:")
        print(", ".join(available_features))

        # Prepare dataframes for environment
        train_env_df = train_df[["Date", "Close"] + available_features] 
        test_env_df = test_df[["Date", "Close"] + available_features]
        
        # Handle different model approaches
        pretrained_path = None
        train_embeddings = None
        test_embeddings = None
        
        if CONFIG["model_type"].lower() == "bert_transformer":
            print("\n🚀 Starting BERT-like approach for tabular financial data!")
            print("Phase 1: Pre-training TabularBERT with masked feature modeling...")
            
            # Pre-train TabularBERT
            trainer = TabularBERTPreTrainer(
                n_features=len(available_features),
                d_model=CONFIG.get("embedding_dim", 128),  # Use embedding_dim from config
                nhead=8,  # Default value for TabularBERT
                num_layers=4,  # Default value for TabularBERT
                dim_feedforward=512,  # Default value for TabularBERT
                dropout=CONFIG.get("bert_dropout", 0.1),  # Use bert_dropout from config
                max_seq_length=CONFIG["lookback_window"],
                lr=CONFIG.get("bert_pretrain_lr", 1e-4),
                batch_size=CONFIG.get("bert_pretrain_batch_size", 32),
            )
            
            # Prepare sequences for pre-training
            train_sequences = prepare_sequences_for_pretraining(
                train_env_df, available_features, CONFIG["lookback_window"]
            )
            
            # Train the model
            trainer.train(
                train_sequences,
                num_epochs=CONFIG.get("bert_pretrain_epochs", 50),
                validation_split=0.1,
                early_stopping_patience=5
            )
            
            # Save the pre-trained model
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            pretrained_path = f"./pretrained_models/tabular_bert_{CONFIG['ticker']}_{timestamp}.pt"
            Path("./pretrained_models").mkdir(exist_ok=True)
            trainer.save_model(pretrained_path)
            
            print(f"\nPhase 2: Extracting embeddings from pre-trained TabularBERT...")
            
            # Extract embeddings for training data
            train_embeddings = trainer.extract_embeddings(train_sequences)
            print(f"Train embeddings shape: {train_embeddings.shape}")
            
            # Extract embeddings for test data
            test_sequences = prepare_sequences_for_pretraining(
                test_env_df, available_features, CONFIG["lookback_window"]
            )
            test_embeddings = trainer.extract_embeddings(test_sequences)
            print(f"Test embeddings shape: {test_embeddings.shape}")
            
            print(f"\nPhase 3: Fine-tuning with frozen TabularBERT backbone...")
            
        
        # Train model with the appropriate approach
        model = train_ppo_agent(
            train_env_df, 
            CONFIG, 
            available_features, 
            train_embeddings
        )

        # Create evaluation environment based on model type
        if CONFIG["model_type"].lower() == "bert_transformer" and test_embeddings is not None:
            print("\nUsing embedding-based environment for evaluation...")
            
            # Use confidence-based environment if enabled
            if CONFIG.get("use_confidence_trading", False) and CONFIDENCE_IMPORTS_AVAILABLE:
                print("🎯 Using confidence-based trading environment...")
                test_env = ConfidenceBasedTradingEnv(
                    test_embeddings,
                    test_env_df,
                    CONFIG["lookback_window"],
                    CONFIG["initial_balance"],
                    CONFIG["transaction_cost"],
                    CONFIG["capital_cost"],
                    CONFIG["risk_free_rate"] / 252,
                    CONFIG["episode_length"] * 2,  # Longer episode for testing
                    random_start=False,
                    allow_short=CONFIG["allow_short"],
                    max_position=CONFIG["max_position"],
                    drawdown_penalty=CONFIG["drawdown_penalty"],
                    volatility_scaling=CONFIG["volatility_scaling"],
                    min_holding_period=CONFIG["min_holding_period"],
                    reward_trade_penalty=CONFIG["reward_trade_penalty"],
                    wrong_side_penalty_factor=CONFIG.get("wrong_side_penalty_factor", 0.5),
                    # Confidence-specific parameters
                    confidence_min_threshold=CONFIG.get("confidence_min_threshold", 0.3),
                    confidence_position_multiplier=CONFIG.get("confidence_position_multiplier", 2.0),
                    confidence_scaling=CONFIG.get("confidence_scaling", True),
                )
            else:
                test_env = EmbeddingBasedStockTradingEnv(
                    test_embeddings,
                    test_env_df,
                    CONFIG["lookback_window"],
                    CONFIG["initial_balance"],
                    CONFIG["transaction_cost"],
                    CONFIG["capital_cost"],
                    CONFIG["risk_free_rate"] / 252,
                    CONFIG["episode_length"] * 2,  # Longer episode for testing
                    random_start=False,
                    allow_short=CONFIG["allow_short"],
                    max_position=CONFIG["max_position"],
                    drawdown_penalty=CONFIG["drawdown_penalty"],
                    volatility_scaling=CONFIG["volatility_scaling"],
                    min_holding_period=CONFIG["min_holding_period"],
                    reward_trade_penalty=CONFIG["reward_trade_penalty"],
                    wrong_side_penalty_factor=CONFIG.get("wrong_side_penalty_factor", 0.5),
                )
        
        evaluate_agent(model, test_env, CONFIG)
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
