"""
Smoke test for Phase 2: Batch Scaling Validation

Quick test to verify batch_scaling.py and phase_2_analysis.py work correctly.
"""

import sys
from pathlib import Path
import tempfile
import shutil
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from experiments.batch_scaling import train_single_config, run_batch_sweep
from analysis.phase_2_analysis import load_results, find_optimal_lr_per_batch, fit_power_law


def test_single_config():
    """Test training a single (batch_size, lr) configuration."""
    print("\n" + "="*60)
    print("TEST 1: Single Config Training")
    print("="*60)

    # Minimal config for quick test
    config = {
        'experiment': {
            'name': 'smoke_test',
            'seed': 42
        },
        'model': {
            'vocab_size': 65,
            'd_model': 32,  # Small for speed
            'n_layers': 2,
            'n_heads': 2,
            'seq_length': 64,  # Short sequences
            'dropout': 0.0
        },
        'training': {
            'steps': 50,  # Very short
            'optimizer': 'adamw',
            'betas': [0.9, 0.999],
            'weight_decay': 0.01,
            'warmup_steps': 10,
            'grad_clip': 1.0,
            'eval_interval': 25,
            'convergence_window': 10,
            'convergence_threshold': 0.01
        },
        'dataset': {
            'num_sequences': 1000,
            'pattern': 'random'
        }
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Train
    print("\nTraining with B=32, LR=0.001...")
    results, _ = train_single_config(
        batch_size=32,
        lr=0.001,
        config=config,
        device=device,
        logger=None,  # No wandb for smoke test
        global_step=0,
        verbose=False
    )

    # Verify results
    assert 'batch_size' in results
    assert 'lr' in results
    assert 'final_train_loss' in results
    assert 'final_val_loss' in results
    assert results['batch_size'] == 32
    assert results['lr'] == 0.001
    assert results['num_steps'] == 50

    print(f"✓ Training completed")
    print(f"  Final train loss: {results['final_train_loss']:.4f}")
    print(f"  Final val loss: {results['final_val_loss']:.4f}")
    print(f"  Converged: {results['converged']}")

    return True


def test_batch_sweep():
    """Test full batch sweep (minimal version)."""
    print("\n" + "="*60)
    print("TEST 2: Batch Sweep (3 batches × 3 LRs = 9 experiments)")
    print("="*60)

    # Create temporary output directory
    temp_dir = tempfile.mkdtemp()
    print(f"Temp output dir: {temp_dir}")

    try:
        # Minimal config for quick test
        config = {
            'experiment': {
                'name': 'phase_2_smoke_test',
                'seed': 42,
                'output_dir': temp_dir
            },
            'model': {
                'vocab_size': 65,
                'd_model': 32,
                'n_layers': 2,
                'n_heads': 2,
                'seq_length': 64,
                'dropout': 0.0
            },
            'training': {
                'steps': 50,
                'optimizer': 'adamw',
                'betas': [0.9, 0.999],
                'weight_decay': 0.01,
                'warmup_steps': 10,
                'grad_clip': 1.0,
                'eval_interval': 25,
                'convergence_window': 10,
                'convergence_threshold': 0.01
            },
            'batch_sweep': {
                'batch_sizes': [16, 32, 64],  # Only 3 batch sizes
                'lr_candidates': [0.0001, 0.001, 0.01]  # Only 3 LRs
            },
            'dataset': {
                'num_sequences': 1000,
                'pattern': 'random'
            },
            'logging': {
                'use_wandb': False,
                'save_results_csv': True,
                'results_file': 'results.csv'
            }
        }

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Run sweep
        print("\nRunning batch sweep...")
        results = run_batch_sweep(config, device)

        # Verify results
        assert len(results) == 9  # 3 batches × 3 LRs
        print(f"✓ Completed {len(results)} experiments")

        # Verify CSV was created
        results_file = Path(temp_dir) / 'logs' / 'results.csv'
        assert results_file.exists(), f"Results CSV not found: {results_file}"
        print(f"✓ Results CSV created: {results_file}")

        # Test analysis on these results
        print("\nTesting analysis...")
        loaded_results = load_results(str(results_file))
        assert len(loaded_results) == 9
        print(f"✓ Loaded {len(loaded_results)} results")

        optimal_configs = find_optimal_lr_per_batch(loaded_results)
        assert len(optimal_configs) == 3  # 3 batch sizes
        print(f"✓ Found optimal configs for {len(optimal_configs)} batch sizes")

        # Test power law fit
        import numpy as np
        batch_sizes = np.array(sorted(optimal_configs.keys()))
        lr_opts = np.array([optimal_configs[b]['lr'] for b in batch_sizes])

        β, C, r_squared, β_std = fit_power_law(batch_sizes, lr_opts)
        print(f"✓ Power law fit: β={β:.3f}, C={C:.6f}, R²={r_squared:.3f}")

        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"✓ Cleaned up temp directory")

        return True

    except Exception as e:
        # Cleanup on error
        if Path(temp_dir).exists():
            shutil.rmtree(temp_dir)
        raise e


def main():
    print("\n" + "="*60)
    print("PHASE 2 SMOKE TESTS")
    print("="*60)

    try:
        # Test 1: Single config
        test_single_config()

        # Test 2: Batch sweep
        test_batch_sweep()

        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60)
        print("\nPhase 2 implementation is working correctly!")
        print("You can now run the full experiment:")
        print("  python experiments/batch_scaling.py --config config/phase_2_batch_sweep.yaml")

        return 0

    except Exception as e:
        print("\n" + "="*60)
        print("✗ TEST FAILED")
        print("="*60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
