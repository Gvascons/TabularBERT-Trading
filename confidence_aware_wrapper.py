#!/usr/bin/env python3
"""
Confidence-Aware Model Wrapper

This wrapper ensures that confidence scores from the enhanced BERT policy
are properly communicated to the confidence-based trading environment.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, Any

class ConfidenceAwareModelWrapper:
    """
    Wrapper that ensures confidence scores are properly passed from policy to environment.
    """
    
    def __init__(self, base_model, environment=None):
        self.base_model = base_model
        self.environment = environment
        self.last_confidence = 0.5  # Default confidence
    
    def predict(self, observation, deterministic=True):
        """
        Enhanced predict method that extracts and sets confidence scores.
        """
        # Check if we have an enhanced policy with confidence capabilities
        if hasattr(self.base_model.policy, 'predict_with_confidence'):
            try:
                # Use the enhanced prediction method
                actions, state, confidence = self.base_model.policy.predict_with_confidence(
                    observation, deterministic=deterministic
                )
                
                # Extract confidence value
                if isinstance(confidence, torch.Tensor):
                    confidence_value = float(confidence.item() if confidence.numel() == 1 else confidence.mean().item())
                elif isinstance(confidence, np.ndarray):
                    confidence_value = float(confidence.item() if confidence.size == 1 else confidence.mean())
                else:
                    confidence_value = float(confidence)
                
                # Store confidence for tracking
                self.last_confidence = confidence_value
                
                # Set confidence in environment if available
                if self.environment is not None and hasattr(self.environment, 'set_confidence'):
                    self.environment.set_confidence(confidence_value)
                
                return actions, state
                
            except Exception as e:
                print(f"Warning: Enhanced prediction failed: {e}")
                # Fallback to standard prediction
                return self._standard_predict(observation, deterministic)
        else:
            # Use standard prediction
            return self._standard_predict(observation, deterministic)
    
    def _standard_predict(self, observation, deterministic=True):
        """
        Standard prediction fallback.
        """
        # Try to extract confidence from the policy if possible
        try:
            # Convert observation to proper format for policy
            if isinstance(observation, dict):
                obs_tensors = {
                    k: torch.as_tensor(v, device=self.base_model.device).float()
                    for k, v in observation.items()
                }
                
                # Ensure batch dimension
                for key in obs_tensors:
                    if obs_tensors[key].dim() == 1:
                        obs_tensors[key] = obs_tensors[key].unsqueeze(0)
                    elif obs_tensors[key].dim() == 2 and key == 'embeddings_seq':
                        obs_tensors[key] = obs_tensors[key].unsqueeze(0)
                
                # Forward pass through policy
                with torch.no_grad():
                    actions, _, _ = self.base_model.policy.forward(obs_tensors, deterministic=deterministic)
                    
                    # Try to get confidence if available
                    if hasattr(self.base_model.policy, 'get_confidence_score'):
                        confidence = self.base_model.policy.get_confidence_score()
                        confidence_value = float(confidence.item() if confidence.numel() == 1 else confidence.mean().item())
                        self.last_confidence = confidence_value
                        
                        # Set in environment
                        if self.environment is not None and hasattr(self.environment, 'set_confidence'):
                            self.environment.set_confidence(confidence_value)
                    
                    # Convert actions to numpy
                    actions_np = actions.cpu().numpy()
                    return actions_np, None
                    
            else:
                # Fallback to SB3's standard predict
                return self.base_model.predict(observation, deterministic=deterministic)
                
        except Exception as e:
            print(f"Warning: Policy forward pass failed: {e}")
            # Final fallback
            return self.base_model.predict(observation, deterministic=deterministic)
    
    def set_environment(self, environment):
        """
        Set the environment for confidence communication.
        """
        self.environment = environment
    
    def get_last_confidence(self):
        """
        Get the last confidence score.
        """
        return self.last_confidence
    
    def __getattr__(self, name):
        """
        Delegate other attributes to the base model.
        """
        return getattr(self.base_model, name)

def create_confidence_aware_wrapper(model, environment=None):
    """
    Create a confidence-aware wrapper for a model.
    
    Args:
        model: The base PPO model
        environment: Optional environment to set confidence in
        
    Returns:
        wrapper: ConfidenceAwareModelWrapper instance
    """
    wrapper = ConfidenceAwareModelWrapper(model, environment)
    return wrapper

if __name__ == "__main__":
    print("Confidence-Aware Model Wrapper")
    print("This wrapper ensures proper confidence communication between policy and environment") 