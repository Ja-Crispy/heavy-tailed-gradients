"""
Nano Transformer for Experiment 1.2.

Purpose: Verify heavy-tail phenomenon with real gradient flow.

Architecture:
- 4 layers
- d_model ∈ [128, 256]
- 2 attention heads
- FFN with d_ff = 4 * d_model
- RMSNorm (LLaMA-style)
- No positional encoding (simplified for Phase 1)
"""

from typing import Optional, Dict, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    RMS Normalization (LLaMA-style).

    RMSNorm(x) = x / RMS(x) * scale

    Where RMS(x) = sqrt(mean(x²) + eps)
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        Initialize RMSNorm.

        Args:
            d_model: Model dimension
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization.

        Args:
            x: Input tensor (..., d_model)

        Returns:
            Normalized tensor
        """
        # Compute RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        # Normalize and scale
        return x / rms * self.scale


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.

    For Experiment 1.2: 2 heads, no causal masking (simplified).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        """
        Initialize multi-head attention.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # Output projection
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)  # (batch, seq_len, d_model)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head: (batch, n_heads, seq_len, d_head)
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # Compute attention scores
        # (batch, n_heads, seq_len, seq_len)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        # (batch, n_heads, seq_len, d_head)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back: (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Output projection
        output = self.o_proj(attn_output)

        return output


class FeedForward(nn.Module):
    """
    Feed-forward network (FFN) with gating.

    Standard transformer FFN: FFN(x) = W_down @ GELU(W_up @ x)
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        """
        Initialize FFN.

        Args:
            d_model: Model dimension
            d_ff: Hidden dimension (typically 4 * d_model)
            dropout: Dropout probability
        """
        super().__init__()

        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (..., d_model)

        Returns:
            Output tensor (..., d_model)
        """
        # up projection + GELU
        hidden = F.gelu(self.up(x))

        # down projection
        output = self.down(hidden)
        output = self.dropout(output)

        return output


class TransformerBlock(nn.Module):
    """
    Single transformer block with pre-norm architecture.

    Block(x) = x + FFN(RMSNorm(x + Attention(RMSNorm(x))))
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        """
        Initialize transformer block.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: FFN hidden dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.norm1 = RMSNorm(d_model)
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)

        self.norm2 = RMSNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        # Pre-norm + attention + residual
        x = x + self.attention(self.norm1(x), mask)

        # Pre-norm + FFN + residual
        x = x + self.ffn(self.norm2(x))

        return x


class NanoTransformer(nn.Module):
    """
    Nano Transformer for Experiment 1.2.

    Architecture:
    - 4 transformer layers
    - d_model ∈ [128, 256]
    - 2 attention heads
    - d_ff = 4 * d_model
    - RMSNorm
    - No positional encoding (simplified)
    - Next-token prediction task
    """

    def __init__(self, vocab_size: int, d_model: int, n_layers: int = 4,
                 n_heads: int = 2, d_ff_multiplier: int = 4,
                 dropout: float = 0.0, use_positional_encoding: bool = False):
        """
        Initialize NanoTransformer.

        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            n_layers: Number of transformer layers (default: 4)
            n_heads: Number of attention heads (default: 2)
            d_ff_multiplier: FFN hidden dim multiplier (default: 4)
            dropout: Dropout probability
            use_positional_encoding: Whether to use positional encoding (default: False for Exp 1.2)
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_model * d_ff_multiplier
        self.use_positional_encoding = use_positional_encoding

        # Token embeddings
        self.embeddings = nn.Embedding(vocab_size, d_model)

        # Positional encoding (optional, not used in Exp 1.2)
        self.pos_encoding = None
        if use_positional_encoding:
            # Learnable positional embeddings
            max_seq_len = 1024
            self.pos_encoding = nn.Embedding(max_seq_len, d_model)

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, self.d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.norm = RMSNorm(d_model)

        # Output projection to vocabulary
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values."""
        # Embeddings
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.02)

        if self.pos_encoding is not None:
            nn.init.normal_(self.pos_encoding.weight, mean=0.0, std=0.02)

        # Output projection (tied with embeddings for parameter efficiency - optional)
        # For Exp 1.2, we'll keep them separate for cleaner α measurements
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs (batch, seq_len)
            mask: Optional attention mask

        Returns:
            Logits (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        x = self.embeddings(input_ids)  # (batch, seq_len, d_model)

        # Add positional encoding if used
        if self.pos_encoding is not None:
            positions = torch.arange(seq_len, device=input_ids.device)
            x = x + self.pos_encoding(positions)

        # Apply transformer blocks
        for layer in self.layers:
            x = layer(x, mask)

        # Final norm
        x = self.norm(x)

        # Project to vocabulary
        logits = self.output_proj(x)  # (batch, seq_len, vocab_size)

        return logits

    def get_parameter_groups(self) -> Dict[str, list]:
        """
        Get parameter groups for layer-wise α measurement.

        Returns:
            Dictionary mapping group names to list of (name, param) tuples
        """
        groups = {
            'embeddings': [('embeddings', self.embeddings.weight)],
            'output': [('output_proj', self.output_proj.weight)],
            'rmsnorm': [],
            'attention': [],
            'ffn': [],
        }

        for layer_idx, layer in enumerate(self.layers):
            # RMSNorm parameters (scale vectors)
            groups['rmsnorm'].append((f'layer{layer_idx}.norm1.scale', layer.norm1.scale))
            groups['rmsnorm'].append((f'layer{layer_idx}.norm2.scale', layer.norm2.scale))

            # Attention parameters
            groups['attention'].append((f'layer{layer_idx}.attn.q_proj', layer.attention.q_proj.weight))
            groups['attention'].append((f'layer{layer_idx}.attn.k_proj', layer.attention.k_proj.weight))
            groups['attention'].append((f'layer{layer_idx}.attn.v_proj', layer.attention.v_proj.weight))
            groups['attention'].append((f'layer{layer_idx}.attn.o_proj', layer.attention.o_proj.weight))

            # FFN parameters
            groups['ffn'].append((f'layer{layer_idx}.ffn.up', layer.ffn.up.weight))
            groups['ffn'].append((f'layer{layer_idx}.ffn.down', layer.ffn.down.weight))

        # Final norm
        groups['rmsnorm'].append(('final_norm.scale', self.norm.scale))

        return groups


def create_nano_transformer(vocab_size: int, d_model: int, **kwargs) -> NanoTransformer:
    """
    Factory function to create NanoTransformer.

    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        **kwargs: Additional arguments for NanoTransformer

    Returns:
        NanoTransformer model
    """
    return NanoTransformer(vocab_size, d_model, **kwargs)


if __name__ == '__main__':
    # Test the model
    print("Testing NanoTransformer...")

    # Create model
    vocab_size = 1000
    d_model = 128
    model = NanoTransformer(vocab_size=vocab_size, d_model=d_model, n_layers=4, n_heads=2)
    print(f"✓ Created NanoTransformer with d_model={d_model}, n_layers=4, n_heads=2")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {n_params:,}")

    # Test forward pass
    batch_size = 8
    seq_len = 32
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(input_ids)
    print(f"✓ Forward pass: {input_ids.shape} -> {logits.shape}")

    # Test backward pass
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        input_ids.view(-1)
    )
    loss.backward()
    print(f"✓ Backward pass completed, loss={loss.item():.4f}")

    # Check gradients
    has_grads = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"  Parameters with gradients: {has_grads}/{len(list(model.parameters()))}")

    # Test parameter grouping
    groups = model.get_parameter_groups()
    print(f"✓ Parameter groups:")
    for group_name, params in groups.items():
        print(f"  {group_name}: {len(params)} parameters")

    print("\n✓ All tests passed!")
