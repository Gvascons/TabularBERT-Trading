#!/usr/bin/env python3
"""
Confidence-Based Trading Environment

This environment extends the EmbeddingBasedStockTradingEnv to use confidence scores
from TabularBERT for dynamic position sizing, potentially improving returns while
maintaining risk management.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional

# We'll define the base class inline to avoid circular imports
# This is a simplified version of EmbeddingBasedStockTradingEnv

class BaseEmbeddingTradingEnv(gym.Env):
    """Base class for embedding-based trading environments."""
    
    def __init__(
        self,
        embeddings: torch.Tensor,
        df: pd.DataFrame,
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
        
        # Store parameters
        self.embeddings = embeddings
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
        self.reward_trade_penalty = reward_trade_penalty
        self.wrong_side_penalty_factor = wrong_side_penalty_factor
        
        # Action space: 0=flat, 1=long, 2=short (if allowed)
        n_actions = 3 if allow_short else 2
        self.action_space = spaces.Discrete(n_actions)
        
        # Observation space: embeddings sequence + agent state (matching EmbeddingBasedStockTradingEnv)
        embedding_dim = embeddings.shape[-1]
        self.observation_space = spaces.Dict({
            'embeddings_seq': spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(lookback_window, embedding_dim), dtype=np.float32
            ),
            'agent_state': spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(5,), dtype=np.float32
            )
        })
        
        # Initialize state
        self.reset()
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize episode
        if self.random_start:
            max_start = len(self.df) - self.episode_length - self.lookback_window
            self.episode_start = self.np_random.integers(self.lookback_window, max_start)
        else:
            self.episode_start = self.lookback_window
        
        self.current_step = self.episode_start
        
        # Initialize trading state
        self.cash = self.initial_balance
        self.shares_held = 0.0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.trade_history = []
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Get current observation."""
        # Get embedding sequence for current step (matching EmbeddingBasedStockTradingEnv format)
        embedding_idx = min(self.current_step - self.lookback_window, len(self.embeddings) - 1)
        embedding_seq = self.embeddings[embedding_idx].numpy().astype(np.float32)
        
        # Get agent state
        current_price = float(self.df.loc[self.current_step, "Close"])
        position_value = self.shares_held * current_price
        position_pct = position_value / self.net_worth if self.net_worth > 0 else 0
        
        agent_state = np.array([
            self.cash / self.initial_balance,  # Normalized cash
            position_pct,  # Position as % of net worth
            self.net_worth / self.initial_balance,  # Normalized net worth
            (self.net_worth - self.max_net_worth) / self.initial_balance,  # Drawdown
            len(self.trade_history) / 100.0,  # Normalized trade count
        ], dtype=np.float32)
        
        return {
            'embeddings_seq': embedding_seq,
            'agent_state': agent_state
        }
    
    def _calculate_reward(self, prev_worth):
        """Calculate reward (simplified version)."""
        # Basic return-based reward
        return_pct = (self.net_worth - prev_worth) / prev_worth if prev_worth > 0 else 0
        return float(return_pct)
    
    def _take_action(self, action: int):
        """Take trading action (simplified version)."""
        current_price = float(self.df.loc[self.current_step, "Close"])
        
        if action == 0:  # Flat
            target_shares = 0
        elif action == 1:  # Long
            max_shares = (self.net_worth * self.max_position) / current_price
            target_shares = max_shares
        elif action == 2 and self.allow_short:  # Short
            max_shares = (self.net_worth * self.max_position) / current_price
            target_shares = -max_shares
        else:
            target_shares = self.shares_held
        
        # Execute trade
        shares_to_trade = target_shares - self.shares_held
        if abs(shares_to_trade) > 1e-6:
            trade_value = abs(shares_to_trade) * current_price
            transaction_cost = trade_value * self.transaction_cost
            
            if shares_to_trade > 0:  # Buying
                total_cost = trade_value + transaction_cost
                if total_cost <= self.cash:
                    self.cash -= total_cost
                    self.shares_held += shares_to_trade
                    self.trade_history.append(("BUY", self.current_step, current_price, shares_to_trade))
            else:  # Selling
                self.cash += trade_value - transaction_cost
                self.shares_held += shares_to_trade
                self.trade_history.append(("SELL", self.current_step, current_price, abs(shares_to_trade)))
        
        # Update net worth
        self.net_worth = self.cash + self.shares_held * current_price
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth
    
    def step(self, action):
        """Take a step in the environment."""
        prev_worth = self.net_worth
        
        # Take action
        self._take_action(action)
        
        # Move to next step
        self.current_step += 1
        
        # Calculate reward
        reward = self._calculate_reward(prev_worth)
        
        # Check if done
        terminated = (
            self.current_step >= len(self.df) - 1 or
            self.current_step >= self.episode_start + self.episode_length or
            self.net_worth <= self.initial_balance * 0.1
        )
        
        obs = self._get_observation()
        info = {
            "net_worth": self.net_worth,
            "max_net_worth": self.max_net_worth,
        }
        
        return obs, reward, terminated, False, info

class ConfidenceBasedTradingEnv(BaseEmbeddingTradingEnv):
    """
    Enhanced trading environment that uses confidence scores for position sizing.
    
    Key improvements:
    1. Dynamic position sizing based on model confidence
    2. Risk-adjusted position limits
    3. Confidence-based reward scaling
    """
    
    def __init__(
        self,
        embeddings: torch.Tensor,
        df: pd.DataFrame,
        lookback_window: int,
        initial_balance: float,
        transaction_cost: float,
        capital_cost: float,
        risk_free_step: float,
        episode_length: int,
        random_start: bool = True,
        allow_short: bool = True,
        max_position: float = 1.0,
        confidence_scaling: bool = True,  # NEW: Enable confidence-based scaling
        confidence_min_threshold: float = 0.3,  # NEW: Minimum confidence to trade
        confidence_position_multiplier: float = 2.0,  # NEW: Max position multiplier
        **kwargs
    ):
        # Filter out confidence-specific parameters before passing to parent
        base_kwargs = {k: v for k, v in kwargs.items() 
                      if k not in ['confidence_scaling', 'confidence_min_threshold', 'confidence_position_multiplier']}
        
        super().__init__(
            embeddings=embeddings,
            df=df,
            lookback_window=lookback_window,
            initial_balance=initial_balance,
            transaction_cost=transaction_cost,
            capital_cost=capital_cost,
            risk_free_step=risk_free_step,
            episode_length=episode_length,
            random_start=random_start,
            allow_short=allow_short,
            max_position=max_position,
            **base_kwargs
        )
        
        # Confidence-based parameters
        self.confidence_scaling = confidence_scaling
        self.min_confidence_threshold = confidence_min_threshold
        self.confidence_position_multiplier = confidence_position_multiplier
        
        # Track confidence history
        self.confidence_history = []
        self.position_size_history = []
    
    def reset(self, *, seed=None, options=None):
        """
        Enhanced reset that ensures proper initialization of confidence tracking.
        """
        obs, info = super().reset(seed=seed, options=options)
        
        # Reset confidence tracking
        self.confidence_history = []
        self.position_size_history = []
        
        return obs, info
        
    def _calculate_confidence_adjusted_position(self, base_action: int, confidence: float) -> Tuple[int, float]:
        """
        Calculate position size based on confidence score.
        
        Args:
            base_action: Original action (0=flat, 1=long, 2=short)
            confidence: Confidence score between 0 and 1
            
        Returns:
            adjusted_action: Action (potentially modified)
            position_multiplier: Position size multiplier
        """
        if not self.confidence_scaling:
            return base_action, 1.0
        
        # Don't trade if confidence is too low
        if confidence < self.min_confidence_threshold:
            return 0, 0.0  # Force flat position
        
        # Scale position size based on confidence
        # Linear scaling: confidence 0.5 -> 1.0x, confidence 1.0 -> max_multiplier
        if confidence >= 0.5:
            position_multiplier = 1.0 + (confidence - 0.5) * 2.0 * (self.confidence_position_multiplier - 1.0)
        else:
            # Below 0.5 confidence, reduce position size
            position_multiplier = confidence * 2.0
        
        # Ensure we don't exceed maximum position
        position_multiplier = min(position_multiplier, self.confidence_position_multiplier)
        
        return base_action, position_multiplier
    
    def _take_action(self, action: int, confidence: Optional[float] = None):
        """
        Enhanced action taking with confidence-based position sizing.
        
        Args:
            action: Trading action (0=flat, 1=long, 2=short)
            confidence: Optional confidence score for position sizing
        """
        if confidence is None:
            confidence = 0.5  # Neutral confidence if not provided
        
        # Store confidence for analysis
        self.confidence_history.append(confidence)
        
        # Calculate confidence-adjusted position
        adjusted_action, position_multiplier = self._calculate_confidence_adjusted_position(action, confidence)
        
        # Store position multiplier for analysis
        self.position_size_history.append(position_multiplier)
        
        # Get current price and net worth
        current_price = float(self.df.loc[self.current_step, "Close"])
        prev_worth = self.net_worth
        
        # Calculate target position based on confidence
        if adjusted_action == 0:  # Flat
            target_shares = 0
        elif adjusted_action == 1:  # Long
            # Calculate position size based on confidence
            max_shares = (self.net_worth * self.max_position) / current_price
            target_shares = max_shares * position_multiplier
        elif adjusted_action == 2 and self.allow_short:  # Short
            max_shares = (self.net_worth * self.max_position) / current_price
            target_shares = -max_shares * position_multiplier
        else:
            target_shares = self.shares_held  # No change
        
        # Calculate shares to trade
        shares_to_trade = target_shares - self.shares_held
        
        # Apply transaction costs and execute trade
        if abs(shares_to_trade) > 1e-6:  # Only trade if significant change
            trade_value = abs(shares_to_trade) * current_price
            transaction_cost = trade_value * self.transaction_cost
            
            # Ensure we have cash attribute (fallback initialization)
            if not hasattr(self, 'cash'):
                self.cash = self.initial_balance - self.shares_held * current_price
            
            # Check if we have enough cash for the trade
            if shares_to_trade > 0:  # Buying
                total_cost = trade_value + transaction_cost
                if total_cost <= self.cash:
                    self.cash -= total_cost
                    self.shares_held += shares_to_trade
                    self.trade_history.append(("BUY", self.current_step, current_price, shares_to_trade))
            else:  # Selling
                self.cash += trade_value - transaction_cost
                self.shares_held += shares_to_trade  # shares_to_trade is negative
                self.trade_history.append(("SELL", self.current_step, current_price, abs(shares_to_trade)))
        
        # Update net worth
        self.net_worth = self.cash + self.shares_held * current_price
        
        # Update max net worth for drawdown calculation
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth
    
    def step(self, action):
        """
        Enhanced step function that can accept confidence scores.
        """
        # For compatibility, we'll extract confidence from the policy if available
        # This will be set by the enhanced policy during prediction
        confidence = getattr(self, '_last_confidence', 0.5)
        
        # Store previous values
        prev_worth = self.net_worth
        prev_step = self.current_step
        
        # Take action with confidence
        self._take_action(action, confidence)
        
        # Move to next step
        self.current_step += 1
        
        # Calculate reward (potentially confidence-adjusted)
        reward = self._calculate_confidence_adjusted_reward(prev_worth, confidence)
        
        # Check if episode is done
        terminated = (
            self.current_step >= len(self.df) - 1 or
            self.current_step >= self.episode_start + self.episode_length or
            self.net_worth <= self.initial_balance * 0.1  # Stop if we lose 90%
        )
        
        truncated = False
        
        # Get new observation
        obs = self._get_observation()
        
        # Create info dictionary with confidence metrics
        info = {
            "net_worth": self.net_worth,
            "max_net_worth": self.max_net_worth,
            "confidence": confidence,
            "position_multiplier": self.position_size_history[-1] if self.position_size_history else 1.0,
            "avg_confidence": np.mean(self.confidence_history) if self.confidence_history else 0.5,
            "confidence_std": np.std(self.confidence_history) if len(self.confidence_history) > 1 else 0.0,
        }
        
        return obs, reward, terminated, truncated, info
    
    def _calculate_confidence_adjusted_reward(self, prev_worth: float, confidence: float) -> float:
        """
        Calculate reward with confidence adjustments.
        
        Args:
            prev_worth: Previous net worth
            confidence: Current confidence score
            
        Returns:
            adjusted_reward: Confidence-adjusted reward
        """
        # Calculate base reward using parent method
        base_reward = self._calculate_reward(prev_worth)
        
        if not self.confidence_scaling:
            return base_reward
        
        # Confidence-based reward adjustments
        confidence_bonus = 0.0
        
        # Bonus for high-confidence correct predictions
        if base_reward > 0 and confidence > 0.7:
            confidence_bonus = base_reward * 0.1 * (confidence - 0.7) / 0.3
        
        # Penalty for high-confidence wrong predictions
        elif base_reward < 0 and confidence > 0.7:
            confidence_penalty = abs(base_reward) * 0.1 * (confidence - 0.7) / 0.3
            confidence_bonus = -confidence_penalty
        
        return base_reward + confidence_bonus
    
    def set_confidence(self, confidence: float):
        """
        Set confidence score for the next action.
        This method is called by the enhanced policy.
        """
        self._last_confidence = float(confidence)
    
    def get_confidence_metrics(self) -> Dict[str, float]:
        """
        Get confidence-related metrics for analysis.
        
        Returns:
            metrics: Dictionary of confidence metrics
        """
        if not self.confidence_history:
            return {}
        
        return {
            "avg_confidence": np.mean(self.confidence_history),
            "confidence_std": np.std(self.confidence_history),
            "min_confidence": np.min(self.confidence_history),
            "max_confidence": np.max(self.confidence_history),
            "avg_position_multiplier": np.mean(self.position_size_history) if self.position_size_history else 1.0,
            "high_confidence_trades": np.sum(np.array(self.confidence_history) > 0.7),
            "low_confidence_trades": np.sum(np.array(self.confidence_history) < 0.3),
            "total_trades": len(self.confidence_history)
        }

if __name__ == "__main__":
    print("Confidence-Based Trading Environment")
    print("This environment uses TabularBERT confidence scores for dynamic position sizing")
    print("Expected benefits:")
    print("• Higher returns through confident position sizing")
    print("• Better risk management through confidence thresholds")
    print("• Adaptive trading based on model certainty") 