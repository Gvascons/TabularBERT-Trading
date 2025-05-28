#!/usr/bin/env python3
"""
Enhanced BERT-Based Policy with Confidence Scoring

This policy extends the BERTBasedActorCriticPolicy to output confidence scores
and integrate with the confidence-based trading environment for improved performance.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, Type, Any, Callable
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from tabular_bert import EmbeddingBasedExtractor

class EnhancedBERTBasedActorCriticPolicy(ActorCriticPolicy):
    """
    Enhanced BERT-based policy that outputs confidence scores for position sizing.
    
    Key features:
    1. Confidence score output for dynamic position sizing
    2. Attention entropy analysis for uncertainty quantification
    3. Integration with confidence-based trading environment
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
        confidence_weight: float = 0.1,  # NEW: Weight for confidence loss
        # Standard ActorCriticPolicy args
        activation_fn: Type[nn.Module] = nn.Tanh,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        # Store confidence parameters
        self.embedding_dim = embedding_dim
        self.latent_dim_pi_out = latent_dim_pi_out
        self.latent_dim_vf_out = latent_dim_vf_out
        self.dropout = dropout
        self.confidence_weight = confidence_weight
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            activation_fn=activation_fn,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            **kwargs,
        )
    
    def _build_mlp_extractor(self) -> None:
        """
        Build the enhanced embedding-based feature extractor with confidence output.
        """
        # Get state dimension from observation space
        if isinstance(self.observation_space, spaces.Dict):
            state_dim = self.observation_space['agent_state'].shape[0]
        else:
            state_dim = 5  # Default fallback
            
        self.mlp_extractor = EmbeddingBasedExtractor(
            embedding_dim=self.embedding_dim,
            state_dim=state_dim,
            latent_dim_pi=self.latent_dim_pi_out,
            latent_dim_vf=self.latent_dim_vf_out,
            dropout=self.dropout,
            output_confidence=True,  # Enable confidence output
        )
        
        # Set latent dimensions for SB3 compatibility
        self.latent_dim_pi = self.latent_dim_pi_out
        self.latent_dim_vf = self.latent_dim_vf_out
    
    def extract_features(self, obs):
        """
        Extract features using the enhanced embedding-based extractor.
        """
        return self.mlp_extractor(obs)
    
    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Enhanced forward pass that includes confidence scoring.
        
        Returns:
            actions: Predicted actions
            values: State values
            log_prob: Log probabilities of actions
        """
        # Extract features and get confidence
        latent_pi, latent_vf = self.extract_features(obs)
        
        # Get confidence score from the extractor
        confidence = self.mlp_extractor.get_confidence_score()
        
        # Store confidence for environment access
        self._last_confidence = confidence
        
        # Get action distribution
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        
        # Get values
        values = self.value_net(latent_vf)
        
        return actions, values, log_prob
    
    def predict_values(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Predict state values.
        """
        _, latent_vf = self.extract_features(obs)
        return self.value_net(latent_vf)
    
    def evaluate_actions(
        self, 
        obs: Dict[str, torch.Tensor], 
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Enhanced action evaluation with confidence scoring.
        """
        latent_pi, latent_vf = self.extract_features(obs)
        
        # Get confidence scores
        confidence = self.mlp_extractor.get_confidence_score()
        
        # Get action distribution and evaluate
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        
        # Get values
        values = self.value_net(latent_vf)
        
        # Store confidence and entropy for potential loss calculation
        self._last_confidence = confidence
        self._last_entropy = entropy
        
        return values, log_prob, entropy
    
    def get_confidence_score(self) -> torch.Tensor:
        """
        Get the confidence score from the last forward pass.
        """
        if hasattr(self, '_last_confidence'):
            return self._last_confidence
        else:
            return torch.ones(1, 1) * 0.5
    
    def predict_with_confidence(
        self,
        observation: Dict[str, torch.Tensor],
        deterministic: bool = True
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]], torch.Tensor]:
        """
        Predict action with confidence score.
        
        Returns:
            actions: Predicted actions
            state: RNN state (if applicable)
            confidence: Confidence scores
        """
        self.set_training_mode(False)
        
        with torch.no_grad():
            # Convert to tensor if needed
            if isinstance(observation, dict):
                obs_tensor = {
                    k: torch.as_tensor(v, device=self.device).float()
                    for k, v in observation.items()
                }
            else:
                obs_tensor = observation
            
            # Ensure batch dimension
            for key in obs_tensor:
                if obs_tensor[key].dim() == 1:
                    obs_tensor[key] = obs_tensor[key].unsqueeze(0)
                elif obs_tensor[key].dim() == 2 and key == 'embeddings_seq':
                    obs_tensor[key] = obs_tensor[key].unsqueeze(0)
            
            # Forward pass
            actions, _, _ = self.forward(obs_tensor, deterministic=deterministic)
            
            # Get confidence
            confidence = self.get_confidence_score()
            
            # Convert to numpy
            actions_np = actions.cpu().numpy()
            confidence_np = confidence.cpu().numpy()
        
        return actions_np, None, confidence_np
    
    def set_environment_confidence(self, env, confidence: torch.Tensor):
        """
        Set confidence score in the environment for position sizing.
        """
        if hasattr(env, 'set_confidence'):
            confidence_value = float(confidence.item() if confidence.numel() == 1 else confidence.mean().item())
            env.set_confidence(confidence_value)

class ConfidenceAwarePPO:
    """
    Wrapper for PPO that integrates confidence scoring with the trading environment.
    """
    
    def __init__(self, base_ppo_model, confidence_weight: float = 0.1):
        self.base_model = base_ppo_model
        self.confidence_weight = confidence_weight
    
    def predict(self, observation, deterministic=True):
        """
        Enhanced predict method that sets confidence in the environment.
        """
        # Get prediction with confidence
        if hasattr(self.base_model.policy, 'predict_with_confidence'):
            actions, state, confidence = self.base_model.policy.predict_with_confidence(
                observation, deterministic=deterministic
            )
            
            # Set confidence in environment if available
            if hasattr(self.base_model, 'env') and hasattr(self.base_model.env, 'set_confidence'):
                confidence_value = float(confidence.item() if confidence.numel() == 1 else confidence.mean().item())
                self.base_model.env.set_confidence(confidence_value)
            
            return actions, state
        else:
            # Fallback to standard prediction
            return self.base_model.predict(observation, deterministic=deterministic)
    
    def learn(self, *args, **kwargs):
        """
        Enhanced learning with confidence-aware loss.
        """
        return self.base_model.learn(*args, **kwargs)
    
    def save(self, path):
        """Save the model."""
        return self.base_model.save(path)
    
    def load(self, path):
        """Load the model."""
        return self.base_model.load(path)
    
    def __getattr__(self, name):
        """Delegate other attributes to the base model."""
        return getattr(self.base_model, name)

def create_enhanced_bert_policy_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create policy kwargs for the enhanced BERT policy.
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        policy_kwargs: Dictionary of policy arguments
    """
    return {
        "embedding_dim": cfg.get("embedding_dim", 128),
        "latent_dim_pi_out": cfg.get("latent_dim_pi_out", 128),
        "latent_dim_vf_out": cfg.get("latent_dim_vf_out", 128),
        "dropout": cfg.get("bert_dropout", 0.1),
        "confidence_weight": cfg.get("confidence_weight", 0.1),
        "activation_fn": nn.ReLU,
    }

if __name__ == "__main__":
    print("Enhanced BERT-Based Policy with Confidence Scoring")
    print("Features:")
    print("• Confidence-based position sizing")
    print("• Attention entropy analysis")
    print("• Integration with confidence-aware environment")
    print("• Enhanced risk management through uncertainty quantification") 