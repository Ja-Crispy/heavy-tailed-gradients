"""
Metrics for analyzing neural network training dynamics.

Includes:
- Sublayer gain computation
- Spectral analysis (singular values, power laws)
- Weight and gradient norms
- Correlation structure analysis
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


def compute_sublayer_gain(input_tensor: torch.Tensor,
                          output_tensor: torch.Tensor,
                          eps: float = 1e-8) -> float:
    """
    Compute sublayer gain: ||output||_rms / ||input||_rms

    This should remain approximately constant across widths according to
    the sublayer gain invariance principle (Kosson et al., 2023).

    Args:
        input_tensor: Layer input tensor (batch, ...)
        output_tensor: Layer output tensor (batch, ...)
        eps: Small constant for numerical stability

    Returns:
        Sublayer gain ratio
    """
    # Compute RMS norms
    input_rms = torch.sqrt(torch.mean(input_tensor ** 2) + eps)
    output_rms = torch.sqrt(torch.mean(output_tensor ** 2) + eps)

    gain = output_rms / input_rms

    return gain.item()


def compute_spectral_analysis(weight_matrix: torch.Tensor,
                              n_singular_values: int = 10) -> Dict[str, Union[float, np.ndarray]]:
    """
    Perform spectral analysis on weight matrix.

    Computes:
    - Top-k singular values
    - Spectral norm (largest singular value)
    - Power law exponent (σ_i ∝ i^(-β))
    - Expected scaling: σ_top ∝ √(η/λ) · d^0.75 (from paper)

    Args:
        weight_matrix: Weight matrix tensor
        n_singular_values: Number of top singular values to return

    Returns:
        Dictionary with spectral statistics
    """
    # Handle batched weights or single matrix
    if weight_matrix.dim() > 2:
        weight_matrix = weight_matrix.reshape(-1, weight_matrix.shape[-1])

    # Ensure 2D matrix
    if weight_matrix.dim() != 2:
        raise ValueError(f"Expected 2D matrix, got shape {weight_matrix.shape}")

    # Compute SVD
    with torch.no_grad():
        try:
            # Use torch.linalg.svd (more stable than torch.svd)
            U, S, Vh = torch.linalg.svd(weight_matrix.float(), full_matrices=False)
        except RuntimeError as e:
            # If SVD fails, return NaNs
            return {
                'singular_values': np.full(n_singular_values, np.nan),
                'spectral_norm': np.nan,
                'top_singular_value': np.nan,
                'power_law_exponent': np.nan,
                'error': str(e)
            }

    # Top-k singular values
    n_sv = min(n_singular_values, len(S))
    top_singular_values = S[:n_sv].cpu().numpy()

    # Spectral norm (largest singular value)
    spectral_norm = S[0].item()

    # Fit power law: log(σ_i) = log(C) - β * log(i)
    # Use top 50% of singular values to avoid noise at tail
    n_fit = max(10, len(S) // 2)
    n_fit = min(n_fit, len(S))

    if n_fit >= 3:
        indices = np.arange(1, n_fit + 1)
        log_indices = np.log(indices)
        log_sv = np.log(S[:n_fit].cpu().numpy() + 1e-10)

        # Linear regression: log(σ) = a + b * log(i)
        # β = -b (power law exponent)
        coeffs = np.polyfit(log_indices, log_sv, deg=1)
        power_law_exponent = -coeffs[0]  # Negative of slope
    else:
        power_law_exponent = np.nan

    return {
        'singular_values': top_singular_values,
        'spectral_norm': spectral_norm,
        'top_singular_value': spectral_norm,
        'power_law_exponent': float(power_law_exponent),
        'n_singular_values': len(S),
    }


def compute_weight_norms(model: nn.Module) -> Dict[str, Dict[str, float]]:
    """
    Compute various norms for all weight matrices in model.

    Args:
        model: PyTorch model

    Returns:
        Dictionary of {param_name: {norm_type: value}}
    """
    norms = {}

    for name, param in model.named_parameters():
        if param.dim() < 2:
            # Vector parameter (bias, norm scales, etc.)
            norms[name] = {
                'l1': torch.norm(param, p=1).item(),
                'l2': torch.norm(param, p=2).item(),
                'linf': torch.norm(param, p=float('inf')).item(),
                'rms': torch.sqrt(torch.mean(param ** 2)).item(),
            }
        else:
            # Matrix parameter
            # Frobenius norm (L2 for matrices)
            frobenius = torch.norm(param, p='fro').item()

            # Spectral norm (largest singular value)
            try:
                spectral = torch.linalg.matrix_norm(param.float(), ord=2).item()
            except:
                spectral = np.nan

            # RMS norm
            rms = torch.sqrt(torch.mean(param ** 2)).item()

            norms[name] = {
                'frobenius': frobenius,
                'spectral': spectral,
                'rms': rms,
                'l1': torch.norm(param, p=1).item(),
            }

    return norms


def compute_gradient_stats(model: nn.Module) -> Dict[str, Dict[str, float]]:
    """
    Compute statistics for gradients.

    Args:
        model: PyTorch model with gradients

    Returns:
        Dictionary of {param_name: {stat: value}}
    """
    stats = {}

    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        grad = param.grad

        param_stats = {
            'l1_norm': torch.norm(grad, p=1).item(),
            'l2_norm': torch.norm(grad, p=2).item(),
            'linf_norm': torch.norm(grad, p=float('inf')).item(),
            'mean': grad.mean().item(),
            'std': grad.std().item(),
            'abs_mean': grad.abs().mean().item(),
        }

        # Higher moments (for tail behavior analysis)
        grad_flat = grad.flatten()
        if len(grad_flat) > 1:
            # Skewness and kurtosis (using scipy definitions)
            grad_np = grad_flat.cpu().numpy()
            mean = np.mean(grad_np)
            std = np.std(grad_np)

            if std > 0:
                normalized = (grad_np - mean) / std
                param_stats['skewness'] = float(np.mean(normalized ** 3))
                param_stats['kurtosis'] = float(np.mean(normalized ** 4))
            else:
                param_stats['skewness'] = 0.0
                param_stats['kurtosis'] = 0.0

        stats[name] = param_stats

    # Global gradient norm
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.norm(2).item() ** 2
    total_norm = total_norm ** 0.5

    stats['__global__'] = {'l2_norm': total_norm}

    return stats


def compute_weight_decay_scaling_check(model: nn.Module,
                                       learning_rate: float,
                                       weight_decay: float,
                                       width: int) -> Dict[str, float]:
    """
    Check if steady-state relationship ||W|| ∝ √(η/λ) holds.

    According to the paper (Kosson et al., 2023):
    - Weight matrix singular values: σ_i(W) ∝ √(η/λ)
    - With width scaling: σ_top ∝ √(η/λ) · d^0.75

    Args:
        model: PyTorch model
        learning_rate: Current learning rate η
        weight_decay: Weight decay λ
        width: Model width d

    Returns:
        Dictionary with scaling analysis
    """
    if weight_decay == 0:
        # Can't compute ratio with zero weight decay
        return {'error': 'weight_decay is zero'}

    # Expected relationship: ||W|| ∝ √(η/λ)
    expected_scale = np.sqrt(learning_rate / weight_decay)

    # With width: expected ∝ expected_scale * d^0.75
    expected_with_width = expected_scale * (width ** 0.75)

    # Measure actual top singular values
    top_singular_values = []

    for name, param in model.named_parameters():
        if param.dim() >= 2:  # Matrix parameters
            spectral = compute_spectral_analysis(param, n_singular_values=1)
            top_sv = spectral['top_singular_value']
            if not np.isnan(top_sv):
                top_singular_values.append(top_sv)

    if not top_singular_values:
        return {'error': 'no matrix parameters found'}

    # Average top singular value
    avg_top_sv = np.mean(top_singular_values)

    # Ratio: actual / expected
    ratio_basic = avg_top_sv / expected_scale
    ratio_with_width = avg_top_sv / expected_with_width

    return {
        'expected_scale': expected_scale,
        'expected_with_width': expected_with_width,
        'measured_avg_top_sv': avg_top_sv,
        'ratio_basic': ratio_basic,
        'ratio_with_width': ratio_with_width,
        'top_singular_values': top_singular_values,
    }


def compute_gradient_correlation(model: nn.Module,
                                 grad_history: Optional[List[Dict[str, torch.Tensor]]] = None) -> Dict[str, float]:
    """
    Compute correlation structure of gradients.

    Useful for understanding Muon's effect: orthogonalization should
    reduce correlation and potentially affect tail behavior.

    Args:
        model: PyTorch model
        grad_history: Optional history of gradients for temporal correlation

    Returns:
        Dictionary with correlation metrics
    """
    # Flatten all gradients into single vector
    grad_list = []
    for param in model.parameters():
        if param.grad is not None:
            grad_list.append(param.grad.flatten())

    if not grad_list:
        return {'error': 'no gradients available'}

    grad_vector = torch.cat(grad_list)

    # Compute autocorrelation if history available
    if grad_history and len(grad_history) > 1:
        # Correlation with previous step
        prev_grads = []
        for param_name, prev_grad in grad_history[-1].items():
            prev_grads.append(prev_grad.flatten())

        prev_vector = torch.cat(prev_grads)

        # Pearson correlation
        correlation = torch.corrcoef(torch.stack([grad_vector, prev_vector]))[0, 1]

        return {
            'temporal_correlation': correlation.item(),
            'grad_norm_current': grad_vector.norm().item(),
            'grad_norm_previous': prev_vector.norm().item(),
        }
    else:
        return {
            'grad_norm': grad_vector.norm().item(),
        }


class MetricsTracker:
    """
    Tracks multiple metrics over training for analysis.

    Usage:
        tracker = MetricsTracker()
        for step in training:
            metrics = tracker.compute_all(model, inputs, outputs)
            logger.log(metrics, step)
    """

    def __init__(self):
        self.grad_history = []
        self.max_history_length = 100

    def compute_all(self, model: nn.Module,
                   inputs: Optional[torch.Tensor] = None,
                   outputs: Optional[torch.Tensor] = None,
                   learning_rate: Optional[float] = None,
                   weight_decay: Optional[float] = None,
                   width: Optional[int] = None) -> Dict[str, Union[float, Dict]]:
        """
        Compute all available metrics.

        Args:
            model: PyTorch model
            inputs: Optional input tensor for sublayer gain
            outputs: Optional output tensor for sublayer gain
            learning_rate: Current LR for scaling checks
            weight_decay: Current WD for scaling checks
            width: Model width for scaling checks

        Returns:
            Dictionary of all computed metrics
        """
        metrics = {}

        # Weight norms
        metrics['weight_norms'] = compute_weight_norms(model)

        # Gradient statistics
        metrics['gradient_stats'] = compute_gradient_stats(model)

        # Sublayer gain (if inputs/outputs provided)
        if inputs is not None and outputs is not None:
            try:
                metrics['sublayer_gain'] = compute_sublayer_gain(inputs, outputs)
            except:
                pass

        # Scaling check (if hyperparams provided)
        if learning_rate is not None and weight_decay is not None and width is not None:
            try:
                metrics['scaling_check'] = compute_weight_decay_scaling_check(
                    model, learning_rate, weight_decay, width
                )
            except:
                pass

        # Gradient correlation
        try:
            metrics['gradient_correlation'] = compute_gradient_correlation(model, self.grad_history)
        except:
            pass

        # Store gradient history for correlation analysis
        current_grads = {name: param.grad.detach().clone()
                        for name, param in model.named_parameters()
                        if param.grad is not None}
        self.grad_history.append(current_grads)

        # Trim history
        if len(self.grad_history) > self.max_history_length:
            self.grad_history.pop(0)

        return metrics


def flatten_metrics(metrics: Dict, prefix: str = '') -> Dict[str, float]:
    """
    Flatten nested metrics dictionary for logging.

    Converts {'weight_norms': {'layer1': {'l2': 0.5}}}
    to {'weight_norms/layer1/l2': 0.5}

    Args:
        metrics: Nested metrics dictionary
        prefix: Prefix for keys

    Returns:
        Flattened dictionary
    """
    flattened = {}

    for key, value in metrics.items():
        new_key = f"{prefix}/{key}" if prefix else key

        if isinstance(value, dict):
            flattened.update(flatten_metrics(value, new_key))
        elif isinstance(value, (int, float, np.number)):
            flattened[new_key] = float(value)
        elif isinstance(value, (list, tuple, np.ndarray)):
            # For arrays, log statistics
            if isinstance(value, (list, tuple)):
                value = np.array(value)
            flattened[f"{new_key}_mean"] = float(np.mean(value))
            flattened[f"{new_key}_std"] = float(np.std(value))
        # Skip non-numeric types

    return flattened
