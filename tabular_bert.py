"""
TabularBERT: BERT-like implementation for tabular financial data

This module implements a BERT-style transformer model adapted for tabular time series data.
Instead of masking tokens, we mask entire features at random timesteps and train the model
to reconstruct them, learning rich representations of financial patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional, List
import datetime
import matplotlib.pyplot as plt
from pathlib import Path

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer models."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TabularBERT(nn.Module):
    """
    BERT-like model adapted for tabular financial time series data.
    """
    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_length: int = 120,
        activation: str = "gelu",
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Feature embedding layer (projects each feature to d_model dimensions)
        self.feature_embedding = nn.Linear(n_features, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True  # Pre-LN architecture for better stability
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # Output projection for masked feature prediction
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_features)
        )
        
        # Special tokens
        self.mask_token = nn.Parameter(torch.randn(1, 1, n_features))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_embeddings(self, features_seq, mask=None):
        """
        Create embeddings from feature sequences.
        
        Args:
            features_seq: (batch_size, seq_len, n_features)
            mask: Optional boolean mask for masked positions
            
        Returns:
            embeddings: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = features_seq.shape
        
        # Apply mask if provided (for pre-training)
        if mask is not None:
            # Expand mask token to match batch size
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            # Apply mask
            features_seq = torch.where(mask.unsqueeze(-1), mask_tokens, features_seq)
        
        # Project features to d_model dimensions
        embeddings = self.feature_embedding(features_seq)
        
        # Add positional encoding
        embeddings = self.pos_encoder(embeddings)
        
        return embeddings
    
    def encode(self, embeddings):
        """
        Encode embeddings using transformer.
        
        Args:
            embeddings: (batch_size, seq_len, d_model)
            
        Returns:
            encoded: (batch_size, seq_len, d_model)
        """
        # No attention mask needed - we want the model to see all positions
        encoded = self.transformer_encoder(embeddings)
        return encoded
    
    def forward(self, features_seq, mask=None, return_embeddings=False):
        """
        Forward pass for TabularBERT.
        
        Args:
            features_seq: (batch_size, seq_len, n_features)
            mask: Optional boolean mask indicating positions to mask
            return_embeddings: If True, return embeddings instead of predictions
            
        Returns:
            If return_embeddings=True: embeddings (batch_size, seq_len, d_model)
            Otherwise: predictions (batch_size, seq_len, n_features)
        """
        # Create embeddings (with masking if provided)
        embeddings = self.create_embeddings(features_seq, mask)
        
        # Encode with transformer
        encoded = self.encode(embeddings)
        
        if return_embeddings:
            return encoded
        
        # Project back to feature space for reconstruction
        predictions = self.output_projection(encoded)
        
        return predictions

class TabularBERTPreTrainer:
    """
    Handles pre-training of TabularBERT using masked feature prediction.
    """
    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_length: int = 120,
        mask_prob: float = 0.15,
        lr: float = 1e-4,
        batch_size: int = 64,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.n_features = n_features
        self.mask_prob = mask_prob
        self.batch_size = batch_size
        self.device = device
        
        # Initialize model
        self.model = TabularBERT(
            n_features=n_features,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_length=max_seq_length
        ).to(device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        # Loss function - MSE for continuous features
        self.criterion = nn.MSELoss(reduction='none')
    
    def create_mask(self, features_seq):
        """
        Create random mask for features.
        
        Args:
            features_seq: (batch_size, seq_len, n_features)
            
        Returns:
            mask: (batch_size, seq_len) boolean mask
        """
        batch_size, seq_len, _ = features_seq.shape
        
        # Create random mask
        mask = torch.rand(batch_size, seq_len) < self.mask_prob
        
        return mask.to(self.device)
    
    def train_epoch(self, features_array):
        """
        Train for one epoch.
        
        Args:
            features_array: (num_samples, seq_len, n_features) tensor
            
        Returns:
            average_loss: Average loss for the epoch
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Shuffle data
        indices = torch.randperm(len(features_array))
        
        for start_idx in range(0, len(features_array), self.batch_size):
            end_idx = min(start_idx + self.batch_size, len(features_array))
            batch_indices = indices[start_idx:end_idx]
            batch_features = features_array[batch_indices].to(self.device)
            
            # Create mask
            mask = self.create_mask(batch_features)
            
            # Forward pass
            predictions = self.model(batch_features, mask=mask)
            
            # Calculate loss only on masked positions
            loss_per_element = self.criterion(predictions, batch_features)
            
            # Average over features dimension
            loss_per_position = loss_per_element.mean(dim=-1)
            
            # Apply mask and average
            masked_loss = (loss_per_position * mask).sum() / mask.sum()
            
            # Backward pass
            self.optimizer.zero_grad()
            masked_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += masked_loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def evaluate(self, features_array):
        """
        Evaluate on validation data.
        
        Args:
            features_array: (num_samples, seq_len, n_features) tensor
            
        Returns:
            average_loss: Average validation loss
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for start_idx in range(0, len(features_array), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(features_array))
                batch_features = features_array[start_idx:end_idx].to(self.device)
                
                # Create mask
                mask = self.create_mask(batch_features)
                
                # Forward pass
                predictions = self.model(batch_features, mask=mask)
                
                # Calculate loss
                loss_per_element = self.criterion(predictions, batch_features)
                loss_per_position = loss_per_element.mean(dim=-1)
                masked_loss = (loss_per_position * mask).sum() / mask.sum()
                
                total_loss += masked_loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, features_array, num_epochs, validation_split=0.1, early_stopping_patience=5):
        """
        Train the TabularBERT model.
        
        Args:
            features_array: (num_samples, seq_len, n_features) tensor
            num_epochs: Number of training epochs
            validation_split: Fraction of data to use for validation
            early_stopping_patience: Patience for early stopping
            
        Returns:
            loss_history: List of training losses
            val_loss_history: List of validation losses
        """
        # Split data
        n_samples = len(features_array)
        n_val = int(n_samples * validation_split)
        indices = torch.randperm(n_samples)
        
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        train_features = features_array[train_indices]
        val_features = features_array[val_indices] if n_val > 0 else None
        
        # Training loop
        loss_history = []
        val_loss_history = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_features)
            loss_history.append(train_loss)
            
            # Validate
            if val_features is not None:
                val_loss = self.evaluate(val_features)
                val_loss_history.append(val_loss)
                
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                
                # Update learning rate
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}")
        
        return loss_history, val_loss_history
    
    def extract_embeddings(self, features_array, batch_size=None):
        """
        Extract embeddings from the pre-trained model.
        
        Args:
            features_array: (num_samples, seq_len, n_features) tensor
            batch_size: Batch size for extraction (default: self.batch_size)
            
        Returns:
            embeddings: (num_samples, seq_len, d_model) tensor
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        self.model.eval()
        all_embeddings = []
        
        with torch.no_grad():
            for start_idx in range(0, len(features_array), batch_size):
                end_idx = min(start_idx + batch_size, len(features_array))
                batch_features = features_array[start_idx:end_idx].to(self.device)
                
                # Extract embeddings (no masking during extraction)
                embeddings = self.model(batch_features, return_embeddings=True)
                all_embeddings.append(embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0)
    
    def save_model(self, save_path):
        """
        Save the pre-trained model.
        
        Args:
            save_path: Path to save the model
        """
        state_dict = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        
        # Save metadata
        metadata = {
            "n_features": self.n_features,
            "d_model": self.model.d_model,
            "nhead": 8,  # Default from model
            "num_layers": 4,  # Default from model
            "dim_feedforward": 512,  # Default from model
            "dropout": 0.1,  # Default from model
            "max_seq_length": self.model.max_seq_length,
            "mask_prob": self.mask_prob,
            "date_created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        torch.save(
            {
                "state_dict": state_dict,
                "metadata": metadata
            },
            save_path
        )
        print(f"TabularBERT model saved to {save_path}")

class EmbeddingBasedExtractor(nn.Module):
    """
    Feature extractor that uses pre-computed embeddings from TabularBERT.
    This is used during the fine-tuning phase.
    
    IMPORTANT: This implements the "frozen backbone" approach by design:
    - The TabularBERT backbone is effectively frozen because we use pre-computed embeddings
    - Only the task-specific heads (policy_head, value_head) are trained
    - This prevents catastrophic forgetting of pre-trained knowledge
    """
    def __init__(
        self,
        embedding_dim: int,
        state_dim: int = 5,
        latent_dim_pi: int = 128,
        latent_dim_vf: int = 128,
        dropout: float = 0.1,
        output_confidence: bool = True,  # New: output confidence scores
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.latent_dim_pi = latent_dim_pi
        self.latent_dim_vf = latent_dim_vf
        self.output_confidence = output_confidence
        
        # Project agent state to same dimension as embeddings
        self.state_projection = nn.Linear(state_dim, embedding_dim)
        
        # Attention mechanism to aggregate sequence embeddings
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        # Combination weights
        self.combination_weights = nn.Parameter(torch.ones(2) / 2)
        
        # Task-specific heads (these will be trained during fine-tuning)
        self.policy_head = nn.Sequential(
            nn.Linear(embedding_dim, latent_dim_pi),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(embedding_dim, latent_dim_vf),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # NEW: Confidence head for position sizing
        if self.output_confidence:
            self.confidence_head = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim // 2, 1),
                nn.Sigmoid()  # Output between 0 and 1
            )
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process pre-computed embeddings with agent state for trading decisions.
        
        Args:
            observations: Dictionary containing:
                - embeddings_seq: (batch_size, seq_len, embedding_dim) - Pre-computed embeddings
                - agent_state: (batch_size, 5) - Agent-specific state variables
                
        Returns:
            latent_pi: Policy network latent features
            latent_vf: Value network latent features
        """
        embeddings_seq = observations['embeddings_seq']  # (batch_size, seq_len, embedding_dim)
        agent_state = observations['agent_state']        # (batch_size, 5)
        
        batch_size, seq_len, _ = embeddings_seq.shape
        
        # Use attention to aggregate sequence information
        attended_embeddings, attention_weights = self.attention(
            embeddings_seq, embeddings_seq, embeddings_seq
        )  # (batch_size, seq_len, embedding_dim)
        
        # Store attention weights for confidence calculation
        self.last_attention_weights = attention_weights
        
        # Get the most recent embedding (last in sequence)
        recent_embedding = attended_embeddings[:, -1, :]  # (batch_size, embedding_dim)
        
        # Global representation (mean of attended embeddings)
        global_embedding = attended_embeddings.mean(dim=1)  # (batch_size, embedding_dim)
        
        # Agent state representation
        agent_embedding = self.state_projection(agent_state)  # (batch_size, embedding_dim)
        
        # Combine representations with learnable weights
        weights = F.softmax(self.combination_weights, dim=0)
        # Only use recent and agent state for now (simpler combination)
        combined_embedding = weights[0] * recent_embedding + weights[1] * agent_embedding
        
        # Apply layer normalization
        combined_embedding = self.layer_norm(combined_embedding)
        
        # Generate task-specific outputs
        latent_pi = self.policy_head(combined_embedding)
        latent_vf = self.value_head(combined_embedding)
        
        # NEW: Calculate confidence if enabled
        if self.output_confidence:
            confidence = self.confidence_head(combined_embedding)
            # Store confidence for use in position sizing
            self.last_confidence = confidence
        
        return latent_pi, latent_vf
    
    def get_confidence_score(self) -> torch.Tensor:
        """
        Get the confidence score from the last forward pass.
        
        Returns:
            confidence: (batch_size, 1) confidence scores between 0 and 1
        """
        if hasattr(self, 'last_confidence'):
            return self.last_confidence
        else:
            # Fallback to neutral confidence if not available
            return torch.ones(1, 1) * 0.5
    
    def get_attention_entropy(self) -> torch.Tensor:
        """
        Calculate attention entropy as an additional confidence measure.
        Higher entropy = less focused attention = lower confidence
        
        Returns:
            entropy: (batch_size,) attention entropy scores
        """
        if hasattr(self, 'last_attention_weights'):
            # Calculate entropy of attention weights
            attn_weights = self.last_attention_weights.mean(dim=1)  # Average over heads
            # Add small epsilon to avoid log(0)
            epsilon = 1e-8
            entropy = -torch.sum(attn_weights * torch.log(attn_weights + epsilon), dim=-1)
            # Normalize entropy to [0, 1] range (lower entropy = higher confidence)
            max_entropy = torch.log(torch.tensor(attn_weights.shape[-1], dtype=torch.float))
            normalized_entropy = entropy / max_entropy
            confidence_from_entropy = 1.0 - normalized_entropy
            return confidence_from_entropy
        else:
            return torch.ones(1) * 0.5
    
    def forward_actor(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        latent_pi, _ = self.forward(observations)
        return latent_pi
    
    def forward_critic(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        _, latent_vf = self.forward(observations)
        return latent_vf

def prepare_sequences_for_pretraining(df, features, lookback_window):
    """
    Prepare sequences for pretraining from a dataframe.
    
    Args:
        df: DataFrame with feature columns
        features: List of feature column names
        lookback_window: Length of sequences to extract
        
    Returns:
        sequences: Tensor of shape (num_sequences, lookback_window, len(features))
    """
    if len(df) < lookback_window:
        raise ValueError(f"DataFrame has fewer rows ({len(df)}) than lookback_window ({lookback_window})")
    
    # Extract feature data
    feature_data = df[features].values
    
    # Create sequences
    sequences = []
    for i in range(len(df) - lookback_window + 1):
        seq = feature_data[i:i+lookback_window]
        sequences.append(seq)
    
    # Convert to tensor
    return torch.FloatTensor(np.array(sequences))

if __name__ == "__main__":
    # Example usage
    print("TabularBERT implementation for financial time series")
    print("This module implements BERT-like pre-training for tabular data")
    print("Use this for:")
    print("1. Pre-training on unlabeled financial data")
    print("2. Extracting rich embeddings")
    print("3. Fine-tuning for downstream tasks") 