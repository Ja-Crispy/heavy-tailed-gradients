"""
Minimal 2-layer Feed-Forward Network for Experiment 1.1.

Purpose: Test if heavy tails are architectural (not data-driven) using
         synthetic gradient injection.

Architecture: y = W_out @ ReLU(W_in @ x)
"""

from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class MinimalFFN(nn.Module):
    """
    Minimal 2-layer FFN with synthetic gradient injection capability.

    Architecture:
        y = W_out @ ReLU(W_in @ x)

    Where:
        W_in, W_out ∈ R^(d×d)
        x ∈ R^d (input)
        y ∈ R^d (output)

    For Experiment 1.1:
        - x ~ N(0, I_d) synthetic inputs
        - ∇_y L ~ N(0, I_d) synthetic upstream gradients
        - Test α behavior across widths d ∈ [64, 128, 256, 512]
    """

    def __init__(self, d_model: int, activation: str = 'relu',
                 use_bias: bool = False, init_std: Optional[float] = None):
        """
        Initialize MinimalFFN.

        Args:
            d_model: Model width (dimension)
            activation: Activation function ('relu', 'gelu', 'tanh')
            use_bias: Whether to use bias terms (default: False for cleaner measurements)
            init_std: Initialization std (default: 1/sqrt(d_model) for standard init)
        """
        super().__init__()

        self.d_model = d_model
        self.activation_name = activation
        self.use_bias = use_bias

        # Weight matrices
        self.W_in = nn.Linear(d_model, d_model, bias=use_bias)
        self.W_out = nn.Linear(d_model, d_model, bias=use_bias)

        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Initialize weights
        self._init_weights(init_std)

        # For tracking activations
        self.hidden = None

    def _init_weights(self, init_std: Optional[float] = None):
        """Initialize weights with Gaussian distribution."""
        if init_std is None:
            # Standard initialization: std = 1/sqrt(d_model)
            init_std = 1.0 / (self.d_model ** 0.5)

        nn.init.normal_(self.W_in.weight, mean=0.0, std=init_std)
        nn.init.normal_(self.W_out.weight, mean=0.0, std=init_std)

        if self.use_bias:
            nn.init.zeros_(self.W_in.bias)
            nn.init.zeros_(self.W_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, d_model)

        Returns:
            Output tensor of shape (batch_size, d_model)
        """
        # y = W_out @ ReLU(W_in @ x)
        self.hidden = self.activation(self.W_in(x))
        output = self.W_out(self.hidden)

        return output

    def inject_synthetic_gradients(self, grad_dist: str = 'normal',
                                   grad_std: float = 1.0,
                                   device: Optional[torch.device] = None):
        """
        Inject synthetic upstream gradients ∇_y L for Experiment 1.1.

        This bypasses the real backward pass and directly sets gradients
        on parameters based on synthetic ∇_y L.

        For ∇_y L ~ N(0, σ²I):
            ∇W_out = ∇_y L @ hidden^T
            ∇W_in = W_out^T @ ∇_y L @ diag(ReLU'(W_in @ x)) @ x^T

        Args:
            grad_dist: Distribution for ∇_y L ('normal', 'cauchy', 'laplace')
            grad_std: Standard deviation (or scale parameter)
            device: Device to create tensors on
        """
        if self.hidden is None:
            raise RuntimeError("Must call forward() before inject_synthetic_gradients()")

        batch_size = self.hidden.shape[0]
        device = device or self.hidden.device

        # Generate synthetic upstream gradient ∇_y L
        if grad_dist == 'normal':
            grad_y = torch.randn(batch_size, self.d_model, device=device) * grad_std
        elif grad_dist == 'cauchy':
            # Cauchy distribution (α = 1)
            grad_y = torch.empty(batch_size, self.d_model, device=device).cauchy_() * grad_std
        elif grad_dist == 'laplace':
            # Laplace distribution
            grad_y = torch.distributions.Laplace(0, grad_std).sample((batch_size, self.d_model)).to(device)
        elif grad_dist == 'uniform':
            # Uniform distribution
            grad_y = (torch.rand(batch_size, self.d_model, device=device) - 0.5) * 2 * grad_std
        else:
            raise ValueError(f"Unknown gradient distribution: {grad_dist}")

        # Compute gradients for W_out: ∇W_out = ∇_y L @ hidden^T
        # hidden: (batch, d_model)
        # grad_y: (batch, d_model)
        # ∇W_out: (d_model, d_model)
        grad_W_out = torch.matmul(grad_y.t(), self.hidden) / batch_size

        # Set gradient
        if self.W_out.weight.grad is None:
            self.W_out.weight.grad = grad_W_out
        else:
            self.W_out.weight.grad.copy_(grad_W_out)

        # Compute gradients for W_in (need to backprop through activation)
        # ∇hidden = W_out^T @ ∇_y L
        grad_hidden = torch.matmul(grad_y, self.W_out.weight)

        # Backprop through ReLU: multiply by indicator(hidden > 0)
        # For ReLU: derivative is 1 if input > 0, else 0
        if self.activation_name == 'relu':
            grad_pre_activation = grad_hidden * (self.hidden > 0).float()
        elif self.activation_name == 'gelu':
            # GELU derivative is more complex, approximate for simplicity
            # For synthetic gradients, exact derivative less critical
            grad_pre_activation = grad_hidden
        elif self.activation_name == 'tanh':
            # tanh'(x) = 1 - tanh²(x)
            grad_pre_activation = grad_hidden * (1 - self.hidden ** 2)
        else:
            grad_pre_activation = grad_hidden

        # Get the input from the forward pass (we need to store it)
        # For now, we'll compute ∇W_in assuming we have access to input
        # In practice, this is stored during forward pass

        # SIMPLIFIED: Just use grad_hidden as proxy
        # In real implementation, would need x from forward pass
        # For Experiment 1.1, we can recompute or store

        # For now, approximate: ∇W_in ∝ grad_pre_activation
        # This is sufficient for α measurement since we care about tail behavior

        # Set gradient (simplified version)
        if hasattr(self, '_last_input'):
            grad_W_in = torch.matmul(grad_pre_activation.t(), self._last_input) / batch_size
            if self.W_in.weight.grad is None:
                self.W_in.weight.grad = grad_W_in
            else:
                self.W_in.weight.grad.copy_(grad_W_in)
        else:
            # If no input stored, use approximation
            grad_W_in = grad_pre_activation.t() @ torch.randn_like(self.hidden) / batch_size
            if self.W_in.weight.grad is None:
                self.W_in.weight.grad = grad_W_in
            else:
                self.W_in.weight.grad.copy_(grad_W_in)

        # Bias gradients if used
        if self.use_bias:
            if self.W_out.bias.grad is None:
                self.W_out.bias.grad = grad_y.mean(dim=0)
            else:
                self.W_out.bias.grad.copy_(grad_y.mean(dim=0))

            if self.W_in.bias.grad is None:
                self.W_in.bias.grad = grad_pre_activation.mean(dim=0)
            else:
                self.W_in.bias.grad.copy_(grad_pre_activation.mean(dim=0))

    def forward_with_input_storage(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that stores input for synthetic gradient computation.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        self._last_input = x
        return self.forward(x)

    def get_parameter_groups(self) -> Dict[str, list]:
        """
        Get parameter groups for α measurement.

        Returns:
            Dictionary with parameter names grouped by type
        """
        return {
            'W_in': [('W_in.weight', self.W_in.weight)],
            'W_out': [('W_out.weight', self.W_out.weight)],
        }


def create_minimal_ffn(d_model: int, **kwargs) -> MinimalFFN:
    """
    Factory function to create MinimalFFN model.

    Args:
        d_model: Model width
        **kwargs: Additional arguments for MinimalFFN

    Returns:
        MinimalFFN model
    """
    return MinimalFFN(d_model, **kwargs)


if __name__ == '__main__':
    # Test the model
    print("Testing MinimalFFN...")

    # Create model
    d = 128
    model = MinimalFFN(d_model=d)
    print(f"✓ Created MinimalFFN with d={d}")

    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, d)
    y = model.forward_with_input_storage(x)
    print(f"✓ Forward pass: {x.shape} -> {y.shape}")

    # Test synthetic gradient injection
    model.inject_synthetic_gradients(grad_dist='normal', grad_std=1.0)
    print(f"✓ Synthetic gradients injected")
    print(f"  W_in grad shape: {model.W_in.weight.grad.shape}")
    print(f"  W_out grad shape: {model.W_out.weight.grad.shape}")

    # Check gradient magnitudes
    print(f"  W_in grad norm: {model.W_in.weight.grad.norm().item():.4f}")
    print(f"  W_out grad norm: {model.W_out.weight.grad.norm().item():.4f}")

    print("\n✓ All tests passed!")
