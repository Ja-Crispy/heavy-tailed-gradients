"""
Smoke tests for Phase 1 experiments.

Runs minimal versions of both experiments to verify everything works end-to-end.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import tempfile
import shutil
import yaml
import torch

from experiments.measure_alpha import run_experiment_1_1, run_experiment_1_2


def create_minimal_config_1_1(output_dir: str) -> dict:
    """Create minimal config for Experiment 1.1 smoke test."""
    return {
        'experiment': {
            'name': 'smoke_test_1_1',
            'seed': 42,
            'output_dir': output_dir,
        },
        'model': {
            'type': 'minimal_ffn',
            'widths': [64],  # Just one small width
            'activation': 'relu',
            'use_bias': False,
        },
        'training': {
            'steps': 50,  # Very few steps
            'batch_size': 32,
            'optimizer': 'adamw',
            'lr': 0.001,
            'weight_decay': 0.0,
            'gradient_clips': [None],  # No clipping for speed
        },
        'measurement': {
            'alpha_interval': 25,  # Measure twice
            'estimators': ['hill'],  # Just Hill for speed
            'k_ratios': [0.1],
            'window_size': 10,
            'per_parameter': True,
            'compute_aggregates': True,
        },
        'synthetic': {
            'input_dist': 'normal',
            'input_mean': 0.0,
            'input_std': 1.0,
            'grad_dist': 'normal',
            'grad_mean': 0.0,
            'grad_std': 1.0,
        },
        'logging': {
            'use_wandb': False,  # Disable for smoke test
            'use_files': True,
            'save_plots': False,  # Skip plots for speed
        },
    }


def create_minimal_config_1_2(output_dir: str) -> dict:
    """Create minimal config for Experiment 1.2 smoke test."""
    return {
        'experiment': {
            'name': 'smoke_test_1_2',
            'seed': 42,
            'output_dir': output_dir,
        },
        'model': {
            'type': 'nano_transformer',
            'd_models': [128],  # Just one small size
            'n_layers': 2,  # Fewer layers
            'n_heads': 2,
            'd_ff_multiplier': 4,
            'dropout': 0.0,
            'use_positional_encoding': False,
        },
        'data': {
            'type': 'synthetic_sequences',
            'vocab_size': 100,  # Small vocab
            'seq_length': 32,  # Short sequences
            'num_sequences': 500,
            'split': [0.9, 0.1],
            'task': 'next_token_prediction',
        },
        'training': {
            'steps': 50,  # Very few steps
            'batch_size': 8,
            'optimizer': 'adamw',
            'lr': 0.0003,
            'weight_decay': 0.01,
            'gradient_clip': None,
        },
        'measurement': {
            'alpha_intervals': [25, 50],
            'continuous_interval': 25,
            'estimators': ['hill'],  # Just Hill for speed
            'k_ratios': [0.1],
            'layer_wise': True,
            'param_groups': ['attention', 'ffn', 'vector'],
            'track_by_depth': True,
            'window_size': 10,
        },
        'logging': {
            'use_wandb': False,
            'use_files': True,
            'save_plots': False,
        },
    }


def run_smoke_test_1_1():
    """Run Experiment 1.1 smoke test."""
    print("\n" + "="*60)
    print("SMOKE TEST: Experiment 1.1 (Synthetic Gradients)")
    print("="*60)

    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix='smoke_test_1_1_')

    try:
        # Create config
        config = create_minimal_config_1_1(temp_dir)

        # Run experiment
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        run_experiment_1_1(config, device)

        # Check outputs
        output_dir = Path(temp_dir)
        assert output_dir.exists(), "Output directory not created"

        # Check for log files
        log_files = list(output_dir.glob('**/*.json*'))
        assert len(log_files) > 0, "No log files created"

        print("\n✅ Experiment 1.1 smoke test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Experiment 1.1 smoke test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def run_smoke_test_1_2():
    """Run Experiment 1.2 smoke test."""
    print("\n" + "="*60)
    print("SMOKE TEST: Experiment 1.2 (Real Gradients)")
    print("="*60)

    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix='smoke_test_1_2_')

    try:
        # Create config
        config = create_minimal_config_1_2(temp_dir)

        # Run experiment
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        run_experiment_1_2(config, device)

        # Check outputs
        output_dir = Path(temp_dir)
        assert output_dir.exists(), "Output directory not created"

        # Check for log files
        log_files = list(output_dir.glob('**/*.json*'))
        assert len(log_files) > 0, "No log files created"

        print("\n✅ Experiment 1.2 smoke test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Experiment 1.2 smoke test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def test_model_instantiation():
    """Test that models can be instantiated."""
    print("\n" + "="*60)
    print("MODEL INSTANTIATION TESTS")
    print("="*60)

    try:
        from models.minimal_ffn import MinimalFFN
        from models.nano_transformer import NanoTransformer

        # Test MinimalFFN
        model = MinimalFFN(d_model=64)
        x = torch.randn(8, 64)
        y = model(x)
        assert y.shape == (8, 64), f"MinimalFFN output shape wrong: {y.shape}"
        print("✓ MinimalFFN instantiation and forward pass")

        # Test NanoTransformer
        model = NanoTransformer(vocab_size=100, d_model=128, n_layers=2, n_heads=2)
        input_ids = torch.randint(0, 100, (4, 16))
        logits = model(input_ids)
        assert logits.shape == (4, 16, 100), f"NanoTransformer output shape wrong: {logits.shape}"
        print("✓ NanoTransformer instantiation and forward pass")

        return True

    except Exception as e:
        print(f"❌ Model instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_synthetic_data():
    """Test synthetic data generation."""
    print("\n" + "="*60)
    print("SYNTHETIC DATA TESTS")
    print("="*60)

    try:
        from experiments.synthetic_data import (
            create_synthetic_dataloader,
            create_token_dataloader
        )

        # Test input data
        loader = create_synthetic_dataloader(
            d_model=64, num_samples=100, batch_size=16, seed=42
        )
        batch = next(iter(loader))
        assert batch.shape == (16, 64), f"Input batch shape wrong: {batch.shape}"
        print("✓ Synthetic input data generation")

        # Test token data
        loader = create_token_dataloader(
            vocab_size=100, seq_length=32, num_sequences=50,
            batch_size=8, seed=42
        )
        inp, tgt = next(iter(loader))
        assert inp.shape == (8, 31), f"Token input shape wrong: {inp.shape}"
        assert tgt.shape == (8, 31), f"Token target shape wrong: {tgt.shape}"
        print("✓ Synthetic token data generation")

        return True

    except Exception as e:
        print(f"❌ Synthetic data tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_smoke_tests():
    """Run all smoke tests."""
    print("\n" + "🧪" * 30)
    print("RUNNING ALL SMOKE TESTS")
    print("🧪" * 30)

    results = {}

    # Test 1: Model instantiation
    results['model_instantiation'] = test_model_instantiation()

    # Test 2: Synthetic data
    results['synthetic_data'] = test_synthetic_data()

    # Test 3: Experiment 1.1 (takes longer)
    results['experiment_1_1'] = run_smoke_test_1_1()

    # Test 4: Experiment 1.2 (takes longer)
    results['experiment_1_2'] = run_smoke_test_1_2()

    # Summary
    print("\n" + "="*60)
    print("SMOKE TEST SUMMARY")
    print("="*60)

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:30s} {status}")

    all_passed = all(results.values())

    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL SMOKE TESTS PASSED!")
        print("="*60)
        print("\nThe codebase is ready for full Phase 1 experiments.")
        print("\nTo run full experiments:")
        print("  python experiments/measure_alpha.py --config config/experiment_1_1.yaml")
        print("  python experiments/measure_alpha.py --config config/experiment_1_2.yaml")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("="*60)
        print("\nPlease review errors above and fix issues before running full experiments.")

    return all_passed


if __name__ == '__main__':
    import sys

    success = run_all_smoke_tests()
    sys.exit(0 if success else 1)
