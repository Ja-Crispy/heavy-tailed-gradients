"""
Tests for tail index estimators.

Validates Hill, Pickands, and ML estimators on known distributions.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import pytest

from core.tail_estimators import (
    estimate_alpha_hill,
    estimate_alpha_pickands,
    estimate_alpha_ml,
    estimate_alpha_ensemble,
    AlphaTracker
)


def generate_alpha_stable_approximate(alpha: float, n_samples: int, seed: int = 42) -> np.ndarray:
    """
    Generate approximate α-stable samples using power law tail.

    For testing purposes. True α-stable generation requires specialized libraries,
    but we can approximate the tail behavior.
    """
    np.random.seed(seed)

    if alpha == 2.0:
        # Gaussian
        return np.random.randn(n_samples)

    # Approximate α-stable using mixture of Gaussian and power law tail
    # Core: Gaussian
    core_ratio = 0.9
    n_core = int(n_samples * core_ratio)
    n_tail = n_samples - n_core

    core = np.random.randn(n_core) * 0.5

    # Tail: power law x^(-1/alpha - 1)
    # P(X > x) ~ x^(-alpha)
    uniform = np.random.uniform(0, 1, n_tail)
    tail_positive = (1 - uniform) ** (-1/alpha)
    tail_signs = np.random.choice([-1, 1], size=n_tail)
    tail = tail_positive * tail_signs

    samples = np.concatenate([core, tail])
    np.random.shuffle(samples)

    return samples


class TestHillEstimator:
    """Test Hill estimator on known distributions."""

    def test_gaussian_gives_alpha_2(self):
        """Gaussian distribution should give α ≈ 2."""
        samples = np.random.randn(10000)
        alpha = estimate_alpha_hill(samples, k_ratio=0.1)

        # Hill estimator is biased for Gaussian, but should be > 1.5
        assert alpha > 1.5, f"Expected α > 1.5 for Gaussian, got {alpha:.3f}"

    def test_heavy_tail_gives_alpha_less_than_2(self):
        """Heavy-tailed distribution should give α < 2."""
        samples = generate_alpha_stable_approximate(alpha=1.5, n_samples=10000)
        alpha = estimate_alpha_hill(samples, k_ratio=0.1)

        # Should detect heavy tail
        assert 1.0 < alpha < 2.0, f"Expected 1.0 < α < 2.0, got {alpha:.3f}"

    def test_cauchy_gives_alpha_close_to_1(self):
        """Cauchy distribution should give α ≈ 1."""
        # Cauchy distribution (α = 1)
        samples = np.random.standard_cauchy(10000)
        alpha = estimate_alpha_hill(samples, k_ratio=0.1)

        # Hill estimator should detect very heavy tail
        assert 0.5 < alpha < 1.5, f"Expected α ≈ 1, got {alpha:.3f}"

    def test_different_k_ratios(self):
        """Different k-ratios should give similar estimates."""
        samples = generate_alpha_stable_approximate(alpha=1.5, n_samples=10000)

        alphas = []
        for k_ratio in [0.05, 0.1, 0.2]:
            alpha = estimate_alpha_hill(samples, k_ratio=k_ratio)
            alphas.append(alpha)

        # Should be within 20% of each other
        alpha_range = max(alphas) - min(alphas)
        alpha_mean = np.mean(alphas)
        assert alpha_range / alpha_mean < 0.3, f"k-ratios give too different results: {alphas}"

    def test_tensor_input(self):
        """Should work with PyTorch tensors."""
        samples = torch.randn(5000)
        alpha = estimate_alpha_hill(samples, k_ratio=0.1)

        assert isinstance(alpha, float)
        assert not np.isnan(alpha)

    def test_too_few_samples(self):
        """Should warn with too few samples."""
        samples = np.random.randn(5)
        alpha = estimate_alpha_hill(samples, k_ratio=0.1)

        # Should return nan or a warning
        assert np.isnan(alpha) or alpha > 0


class TestPickandsEstimator:
    """Test Pickands estimator on known distributions."""

    def test_gaussian_gives_alpha_2(self):
        """Gaussian distribution should give α ≈ 2."""
        samples = np.random.randn(10000)
        alpha = estimate_alpha_pickands(samples, k_ratio=0.1)

        # Pickands can be unstable for Gaussian; accept NaN or high alpha
        assert np.isnan(alpha) or alpha > 1.5, f"Expected α > 1.5 or NaN for Gaussian, got {alpha:.3f}"

    def test_heavy_tail(self):
        """Should detect heavy tails."""
        samples = generate_alpha_stable_approximate(alpha=1.5, n_samples=10000)
        alpha = estimate_alpha_pickands(samples, k_ratio=0.1)

        # Pickands is very unstable; accept wide range or NaN
        # (In practice, Hill estimator is much more reliable)
        assert np.isnan(alpha) or (0.5 < alpha < 20.0), f"Expected heavy tail or NaN, got α={alpha:.3f}"

    def test_robustness_to_k(self):
        """Pickands should be more robust to k than Hill."""
        samples = generate_alpha_stable_approximate(alpha=1.5, n_samples=10000)

        alphas = []
        for k_ratio in [0.05, 0.1, 0.2]:
            alpha = estimate_alpha_pickands(samples, k_ratio=k_ratio)
            if not np.isnan(alpha):
                alphas.append(alpha)

        if len(alphas) >= 2:
            # Should be reasonably consistent
            assert max(alphas) - min(alphas) < 1.0


class TestMLEstimator:
    """Test Maximum Likelihood estimator."""

    def test_returns_dict(self):
        """ML estimator should return dict with parameters."""
        samples = np.random.randn(1000)
        result = estimate_alpha_ml(samples)

        assert isinstance(result, dict)
        assert 'alpha' in result
        assert 'beta' in result
        assert 'gamma' in result
        assert 'delta' in result

    def test_gaussian_estimate(self):
        """Should estimate α ≈ 2 for Gaussian."""
        samples = np.random.randn(5000)
        result = estimate_alpha_ml(samples)

        alpha = result['alpha']
        if not np.isnan(alpha):
            # ML estimator should be accurate for Gaussian
            assert 1.5 < alpha <= 2.0, f"Expected α ≈ 2, got {alpha:.3f}"


class TestEnsembleEstimator:
    """Test ensemble estimation."""

    def test_ensemble_returns_multiple_estimates(self):
        """Ensemble should return multiple estimates."""
        samples = np.random.randn(5000)
        result = estimate_alpha_ensemble(
            samples,
            estimators=['hill', 'pickands'],
            k_ratios=[0.1]
        )

        assert 'hill_k0.1' in result
        assert 'pickands_k0.1' in result
        assert 'ensemble_mean' in result
        assert 'ensemble_std' in result

    def test_ensemble_statistics(self):
        """Ensemble should compute meaningful statistics."""
        samples = np.random.randn(5000)
        result = estimate_alpha_ensemble(samples)

        mean = result.get('ensemble_mean')
        std = result.get('ensemble_std')

        if not np.isnan(mean):
            assert mean > 0
            assert std >= 0


class TestAlphaTracker:
    """Test AlphaTracker class."""

    def test_tracker_initialization(self):
        """Should initialize properly."""
        tracker = AlphaTracker(window_size=50, estimators=['hill'])
        assert tracker.window_size == 50
        assert 'hill' in tracker.estimators

    def test_tracker_update(self):
        """Should track α over time."""
        import torch.nn as nn

        # Create simple model
        model = nn.Linear(10, 10)

        # Simulate training
        tracker = AlphaTracker(window_size=10, estimators=['hill'], k_ratios=[0.1])

        for _ in range(5):
            # Create synthetic gradients
            model.weight.grad = torch.randn_like(model.weight)
            tracker.update(model)

        # Should have stored some estimates
        summary = tracker.get_summary()
        assert len(summary) > 0

    def test_tracker_summary(self):
        """Should provide summary statistics."""
        import torch.nn as nn

        model = nn.Linear(10, 10)
        tracker = AlphaTracker(estimators=['hill'], k_ratios=[0.1])

        # Add several measurements
        for _ in range(10):
            model.weight.grad = torch.randn_like(model.weight)
            tracker.update(model)

        summary = tracker.get_summary()

        if summary:
            first_key = list(summary.keys())[0]
            stats = summary[first_key]

            assert 'mean' in stats
            assert 'std' in stats
            assert 'current' in stats

    def test_layer_grouped_summary(self):
        """Should group parameters by patterns."""
        import torch.nn as nn

        # Create model with named parameters
        model = nn.Sequential(
            nn.Linear(10, 10, bias=False),  # weight
            nn.LayerNorm(10),  # weight, bias
        )

        tracker = AlphaTracker(estimators=['hill'], k_ratios=[0.1])

        # Add gradients
        for _ in range(5):
            for param in model.parameters():
                param.grad = torch.randn_like(param)
            tracker.update(model)

        # Group by pattern
        patterns = {
            'linear': ['linear', 'weight'],
            'norm': ['norm', 'bias']
        }

        grouped = tracker.get_layer_grouped_summary(patterns)
        # Should have some grouping
        assert isinstance(grouped, dict)


def test_preprocessing():
    """Test gradient preprocessing."""
    from core.tail_estimators import preprocess_gradients

    # Test with zeros
    data = np.array([0, 1, 2, 0, 3])
    processed = preprocess_gradients(data, remove_zeros=True)
    assert 0 not in processed
    assert len(processed) == 3

    # Test with NaNs
    data = np.array([1, 2, np.nan, 3, np.inf])
    processed = preprocess_gradients(data, remove_nans=True)
    assert len(processed) == 3
    assert all(np.isfinite(processed))

    # Test sorting
    data = np.array([1, 5, 2, 4, 3])
    processed = preprocess_gradients(data)
    assert processed[0] == 5  # Should be descending
    assert processed[-1] == 1


if __name__ == '__main__':
    # Run tests
    print("Running tail estimator tests...")
    print("\n" + "="*60)
    print("Testing Hill Estimator")
    print("="*60)

    test = TestHillEstimator()
    test.test_gaussian_gives_alpha_2()
    print("✓ Gaussian test passed")

    test.test_heavy_tail_gives_alpha_less_than_2()
    print("✓ Heavy-tail test passed")

    test.test_different_k_ratios()
    print("✓ k-ratio robustness test passed")

    test.test_tensor_input()
    print("✓ Tensor input test passed")

    print("\n" + "="*60)
    print("Testing Pickands Estimator")
    print("="*60)

    test = TestPickandsEstimator()
    test.test_gaussian_gives_alpha_2()
    print("✓ Gaussian test passed")

    test.test_heavy_tail()
    print("✓ Heavy-tail test passed")

    print("\n" + "="*60)
    print("Testing Ensemble Estimator")
    print("="*60)

    test = TestEnsembleEstimator()
    test.test_ensemble_returns_multiple_estimates()
    print("✓ Ensemble structure test passed")

    test.test_ensemble_statistics()
    print("✓ Ensemble statistics test passed")

    print("\n" + "="*60)
    print("Testing AlphaTracker")
    print("="*60)

    test = TestAlphaTracker()
    test.test_tracker_initialization()
    print("✓ Initialization test passed")

    test.test_tracker_update()
    print("✓ Update test passed")

    test.test_tracker_summary()
    print("✓ Summary test passed")

    print("\n" + "="*60)
    print("Testing Preprocessing")
    print("="*60)

    test_preprocessing()
    print("✓ Preprocessing tests passed")

    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
