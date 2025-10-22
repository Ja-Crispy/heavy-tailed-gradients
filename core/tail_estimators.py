"""
Tail index (α) estimation for heavy-tailed distributions.

Implements three methods:
1. Hill Estimator - Classic, simple, sensitive to k
2. Pickands Estimator - More robust to k choice
3. Maximum Likelihood - Full α-stable distribution fit

Reference: Technical Deep Dives section of research docs
"""

import warnings
from collections import deque, defaultdict
from typing import Union, Optional, Dict, List, Tuple

import numpy as np
import torch

# Use scipy's levy_stable for α-stable ML estimation
try:
    from scipy.stats import levy_stable
    LEVY_AVAILABLE = True
except ImportError:
    LEVY_AVAILABLE = False
    warnings.warn("scipy.stats.levy_stable not available for ML estimation. Upgrade scipy to >=1.10.0")


def preprocess_gradients(gradients: Union[torch.Tensor, np.ndarray],
                         remove_zeros: bool = True,
                         remove_nans: bool = True) -> np.ndarray:
    """
    Preprocess gradients for tail index estimation.

    Args:
        gradients: Gradient tensor or array
        remove_zeros: Remove zero values
        remove_nans: Remove NaN/Inf values

    Returns:
        Preprocessed absolute values as numpy array, sorted descending
    """
    # Convert to numpy if torch tensor
    if isinstance(gradients, torch.Tensor):
        gradients = gradients.detach().cpu().numpy()

    # Flatten and take absolute values
    g_flat = np.abs(gradients.flatten())

    # Remove NaNs and Infs
    if remove_nans:
        g_flat = g_flat[np.isfinite(g_flat)]

    # Remove zeros
    if remove_zeros:
        g_flat = g_flat[g_flat > 0]

    # Sort in descending order
    g_sorted = np.sort(g_flat)[::-1]

    return g_sorted


def estimate_alpha_hill(gradients: Union[torch.Tensor, np.ndarray],
                       k_ratio: float = 0.1,
                       k: Optional[int] = None) -> float:
    """
    Hill estimator for tail index α.

    Formula: α_Hill = 1 / (1/k * Σ_{i=1}^k log(X_i / X_{k+1}))

    Where X_i are order statistics (sorted magnitudes in descending order).

    Args:
        gradients: Gradient values (tensor or array)
        k_ratio: Fraction of top order statistics to use (if k not specified)
        k: Explicit number of order statistics (overrides k_ratio)

    Returns:
        Estimated tail index α

    Reference:
        Hill, B. M. (1975). "A simple general approach to inference about the tail
        of a distribution." Annals of Statistics, 3(5), 1163-1174.
    """
    g_sorted = preprocess_gradients(gradients)

    if len(g_sorted) < 10:
        warnings.warn(f"Too few samples ({len(g_sorted)}) for reliable Hill estimation")
        return np.nan

    # Determine k (number of order statistics to use)
    if k is None:
        k = max(int(len(g_sorted) * k_ratio), 5)
    k = min(k, len(g_sorted) - 1)

    # Hill estimator formula
    # α = 1 / (1/k * Σ log(X_i / X_{k+1}))
    log_ratios = np.log(g_sorted[:k] / g_sorted[k])

    # Handle edge cases
    log_ratios = log_ratios[np.isfinite(log_ratios)]
    if len(log_ratios) == 0:
        warnings.warn("All log ratios are non-finite in Hill estimator")
        return np.nan

    mean_log_ratio = np.mean(log_ratios)

    if mean_log_ratio <= 0:
        warnings.warn(f"Non-positive mean log ratio in Hill estimator: {mean_log_ratio}")
        return np.nan

    alpha = 1.0 / mean_log_ratio

    # Sanity check: α should be in (0, 2] for heavy-tailed or Gaussian
    if alpha < 0.1 or alpha > 10:
        warnings.warn(f"Hill estimator gave extreme value: α={alpha:.3f}")

    return float(alpha)


def estimate_alpha_pickands(gradients: Union[torch.Tensor, np.ndarray],
                           k_ratio: float = 0.1,
                           k: Optional[int] = None) -> float:
    """
    Pickands estimator for tail index α.

    Formula: α_Pickands = 1 / (log2 * log((X_k - X_2k) / (X_2k - X_4k)))

    More robust to k choice than Hill, but higher variance.

    Args:
        gradients: Gradient values (tensor or array)
        k_ratio: Fraction to use for k calculation
        k: Explicit k (overrides k_ratio)

    Returns:
        Estimated tail index α

    Reference:
        Pickands, J. (1975). "Statistical inference using extreme order statistics."
        Annals of Statistics, 3(1), 119-131.
    """
    g_sorted = preprocess_gradients(gradients)

    if len(g_sorted) < 20:
        warnings.warn(f"Too few samples ({len(g_sorted)}) for Pickands estimation")
        return np.nan

    # Determine k
    if k is None:
        k = max(int(len(g_sorted) * k_ratio), 5)

    # Need X_k, X_2k, X_4k
    k2 = 2 * k
    k4 = 4 * k

    if k4 >= len(g_sorted):
        # Not enough samples, reduce k
        k4 = len(g_sorted) - 1
        k2 = k4 // 2
        k = k2 // 2

    if k < 2:
        return np.nan

    # Pickands formula
    numerator = g_sorted[k-1] - g_sorted[k2-1]  # k is 1-indexed in formula, 0-indexed in array
    denominator = g_sorted[k2-1] - g_sorted[k4-1]

    if numerator <= 0 or denominator <= 0:
        warnings.warn("Non-positive differences in Pickands estimator")
        return np.nan

    ratio = numerator / denominator

    if ratio <= 0:
        return np.nan

    # α = 1 / (log2 * log(ratio))
    log_ratio = np.log(ratio)

    if log_ratio <= 0:
        warnings.warn(f"Non-positive log ratio in Pickands: {log_ratio}")
        return np.nan

    alpha = 1.0 / (np.log(2.0) * log_ratio)

    # Sanity check
    if alpha < 0.1 or alpha > 10:
        warnings.warn(f"Pickands estimator gave extreme value: α={alpha:.3f}")

    return float(alpha)


def estimate_alpha_ml(gradients: Union[torch.Tensor, np.ndarray],
                     method: str = 'scipy') -> Dict[str, float]:
    """
    Maximum Likelihood estimation for α-stable distribution.

    Fits full α-stable distribution to get parameters (α, β, γ, δ).
    Most informative but also most computationally expensive.

    Args:
        gradients: Gradient values
        method: 'scipy' for scipy.stats.levy_stable

    Returns:
        Dictionary with parameters: {'alpha', 'beta', 'gamma', 'delta'}
        beta: Skewness parameter (-1 to 1)
        gamma: Scale parameter (> 0)
        delta: Location parameter

    Reference:
        Nolan, J. P. (2020). "Univariate Stable Distributions: Models for
        Heavy Tailed Data." Springer.
    """
    if not LEVY_AVAILABLE:
        warnings.warn("scipy.stats.levy_stable not available for ML estimation")
        return {'alpha': np.nan, 'beta': np.nan, 'gamma': np.nan, 'delta': np.nan}

    g_flat = preprocess_gradients(gradients, remove_zeros=True)

    if len(g_flat) < 100:
        warnings.warn(f"Too few samples ({len(g_flat)}) for reliable ML estimation")
        return {'alpha': np.nan, 'beta': np.nan, 'gamma': np.nan, 'delta': np.nan}

    try:
        # We need signed values for proper fitting
        if isinstance(gradients, torch.Tensor):
            g_signed = gradients.detach().cpu().numpy().flatten()
        else:
            g_signed = gradients.flatten()

        # Remove non-finite values
        g_signed = g_signed[np.isfinite(g_signed)]

        # Fit α-stable distribution using scipy
        # levy_stable.fit returns (alpha, beta, loc, scale)
        alpha, beta, delta, gamma = levy_stable.fit(g_signed)

        # Sanity checks
        if not (0 < alpha <= 2):
            warnings.warn(f"ML estimation gave α={alpha:.3f} outside (0,2]")

        if not (-1 <= beta <= 1):
            warnings.warn(f"ML estimation gave β={beta:.3f} outside [-1,1]")

        return {
            'alpha': float(alpha),
            'beta': float(beta),
            'gamma': float(gamma),
            'delta': float(delta),
        }

    except Exception as e:
        warnings.warn(f"ML estimation failed: {e}")
        return {'alpha': np.nan, 'beta': np.nan, 'gamma': np.nan, 'delta': np.nan}


class AlphaTracker:
    """
    Tracks tail index α over time with sliding window.

    Maintains history of α measurements for each parameter and computes
    summary statistics.

    Usage:
        tracker = AlphaTracker(window_size=100)
        for step in training:
            tracker.update(model, prefix='step_{step}')
            summary = tracker.get_summary()
    """

    def __init__(self, window_size: int = 100, estimators: Optional[List[str]] = None,
                 k_ratios: Optional[List[float]] = None):
        """
        Initialize AlphaTracker.

        Args:
            window_size: Number of measurements to keep in sliding window
            estimators: List of estimators to use ['hill', 'pickands', 'ml']
            k_ratios: List of k_ratios to try for robustness
        """
        self.window_size = window_size
        self.estimators = estimators or ['hill', 'pickands']
        self.k_ratios = k_ratios or [0.1]

        # Storage: {param_name: {estimator: {k_ratio: deque}}}
        self.alphas = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: deque(maxlen=window_size))))

        # Also store global estimates (aggregated across parameters)
        self.global_alphas = defaultdict(lambda: defaultdict(lambda: deque(maxlen=window_size)))

    def update(self, model: torch.nn.Module, prefix: str = ''):
        """
        Update α estimates for all parameters in model.

        Args:
            model: PyTorch model
            prefix: Prefix for parameter names (e.g., 'step_100')
        """
        all_gradients = []

        for name, param in model.named_parameters():
            if param.grad is None:
                continue

            full_name = f"{prefix}/{name}" if prefix else name

            # Estimate α with each method
            for estimator in self.estimators:
                if estimator == 'hill':
                    for k_ratio in self.k_ratios:
                        alpha = estimate_alpha_hill(param.grad, k_ratio=k_ratio)
                        if not np.isnan(alpha):
                            self.alphas[full_name][estimator][k_ratio].append(alpha)

                elif estimator == 'pickands':
                    for k_ratio in self.k_ratios:
                        alpha = estimate_alpha_pickands(param.grad, k_ratio=k_ratio)
                        if not np.isnan(alpha):
                            self.alphas[full_name][estimator][k_ratio].append(alpha)

                elif estimator == 'ml':
                    result = estimate_alpha_ml(param.grad)
                    alpha = result.get('alpha', np.nan)
                    if not np.isnan(alpha):
                        # ML doesn't use k_ratio, use a placeholder
                        self.alphas[full_name][estimator]['ml'].append(alpha)

            # Collect for global estimate
            all_gradients.append(param.grad.detach().flatten())

        # Compute global α (across all parameters)
        if all_gradients:
            global_grads = torch.cat(all_gradients)

            for estimator in self.estimators:
                if estimator == 'hill':
                    for k_ratio in self.k_ratios:
                        alpha = estimate_alpha_hill(global_grads, k_ratio=k_ratio)
                        if not np.isnan(alpha):
                            self.global_alphas[estimator][k_ratio].append(alpha)

                elif estimator == 'pickands':
                    for k_ratio in self.k_ratios:
                        alpha = estimate_alpha_pickands(global_grads, k_ratio=k_ratio)
                        if not np.isnan(alpha):
                            self.global_alphas[estimator][k_ratio].append(alpha)

                elif estimator == 'ml':
                    result = estimate_alpha_ml(global_grads)
                    alpha = result.get('alpha', np.nan)
                    if not np.isnan(alpha):
                        self.global_alphas[estimator]['ml'].append(alpha)

    def get_summary(self, param_name: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for α estimates.

        Args:
            param_name: Specific parameter name, or None for global

        Returns:
            Dictionary of {estimator/k_ratio: {mean, std, min, max, current}}
        """
        summary = {}

        if param_name is None:
            # Global summary
            alphas_dict = self.global_alphas
        else:
            alphas_dict = self.alphas.get(param_name, {})

        for estimator, k_dict in alphas_dict.items():
            for k_ratio, values in k_dict.items():
                if len(values) == 0:
                    continue

                key = f"{estimator}_k{k_ratio}" if k_ratio != 'ml' else estimator

                summary[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'current': float(values[-1]),
                    'count': len(values),
                }

        return summary

    def get_all_param_summaries(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Get summaries for all tracked parameters."""
        return {param: self.get_summary(param) for param in self.alphas.keys()}

    def get_layer_grouped_summary(self, param_patterns: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
        """
        Get α summary grouped by layer types (e.g., attention, ffn, vector).

        Args:
            param_patterns: Dict mapping group names to parameter name patterns
                           e.g., {'attention': ['q_proj', 'k_proj'], 'ffn': ['gate', 'up']}

        Returns:
            Dictionary of {group_name: {estimator: stats}}
        """
        grouped = defaultdict(lambda: defaultdict(list))

        for param_name, estimator_dict in self.alphas.items():
            # Determine which group this parameter belongs to
            group = 'other'
            for group_name, patterns in param_patterns.items():
                if any(pattern in param_name for pattern in patterns):
                    group = group_name
                    break

            # Collect α values for this group
            for estimator, k_dict in estimator_dict.items():
                for k_ratio, values in k_dict.items():
                    if len(values) > 0:
                        key = f"{estimator}_k{k_ratio}" if k_ratio != 'ml' else estimator
                        grouped[group][key].extend(values)

        # Compute summary statistics for each group
        summary = {}
        for group, estimator_dict in grouped.items():
            summary[group] = {}
            for estimator_key, values in estimator_dict.items():
                if len(values) > 0:
                    summary[group][estimator_key] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                    }

        return summary


def estimate_alpha_ensemble(gradients: Union[torch.Tensor, np.ndarray],
                            estimators: Optional[List[str]] = None,
                            k_ratios: Optional[List[float]] = None) -> Dict[str, float]:
    """
    Ensemble estimate using multiple methods and k-ratios.

    Args:
        gradients: Gradient values
        estimators: List of estimators ['hill', 'pickands', 'ml']
        k_ratios: List of k_ratios to try

    Returns:
        Dictionary with individual estimates and ensemble statistics
    """
    estimators = estimators or ['hill', 'pickands']
    k_ratios = k_ratios or [0.05, 0.1, 0.2]

    results = {}
    all_estimates = []

    # Hill estimates
    if 'hill' in estimators:
        for k_ratio in k_ratios:
            alpha = estimate_alpha_hill(gradients, k_ratio=k_ratio)
            key = f'hill_k{k_ratio}'
            results[key] = alpha
            if not np.isnan(alpha):
                all_estimates.append(alpha)

    # Pickands estimates
    if 'pickands' in estimators:
        for k_ratio in k_ratios:
            alpha = estimate_alpha_pickands(gradients, k_ratio=k_ratio)
            key = f'pickands_k{k_ratio}'
            results[key] = alpha
            if not np.isnan(alpha):
                all_estimates.append(alpha)

    # ML estimate
    if 'ml' in estimators:
        ml_result = estimate_alpha_ml(gradients)
        results['ml'] = ml_result['alpha']
        if not np.isnan(ml_result['alpha']):
            all_estimates.append(ml_result['alpha'])
            results['ml_beta'] = ml_result['beta']
            results['ml_gamma'] = ml_result['gamma']
            results['ml_delta'] = ml_result['delta']

    # Ensemble statistics
    if all_estimates:
        results['ensemble_mean'] = float(np.mean(all_estimates))
        results['ensemble_std'] = float(np.std(all_estimates))
        results['ensemble_median'] = float(np.median(all_estimates))
        results['ensemble_min'] = float(np.min(all_estimates))
        results['ensemble_max'] = float(np.max(all_estimates))
    else:
        results['ensemble_mean'] = np.nan

    return results
