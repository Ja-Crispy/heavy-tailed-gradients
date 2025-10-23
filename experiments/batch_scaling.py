"""
Phase 2: Batch Scaling Validation Experiment

Tests the hypothesis that optimal learning rate scales as:
- LR ∝ Batch^(2/3) (Laplace, α=3) - our Phase 1 finding
- vs LR ∝ Batch^(1/2) (Gaussian, standard assumption)

Runs grid search over (batch_size, learning_rate) to find optimal LR for each batch size.

Usage:
    python experiments/batch_scaling.py --config config/phase_2_batch_sweep.yaml
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import yaml
import csv
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.logger import FlexibleLogger, get_system_info
from core.metrics import MetricsTracker, flatten_metrics
from models.nano_transformer import NanoTransformer
from experiments.synthetic_data import (
    create_token_dataloader,
    infinite_dataloader,
    set_seed
)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_optimizer(model: nn.Module, lr: float, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """Create optimizer with specified learning rate."""
    opt_config = config['training']
    opt_type = opt_config.get('optimizer', 'adamw').lower()

    if opt_type == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=opt_config.get('weight_decay', 0.0),
            betas=opt_config.get('betas', [0.9, 0.999]),
            eps=opt_config.get('eps', 1e-8)
        )
    elif opt_type == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=opt_config.get('weight_decay', 0.0),
            momentum=opt_config.get('momentum', 0.0)
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")

    return optimizer


def get_lr_schedule(optimizer: torch.optim.Optimizer, warmup_steps: int, step: int) -> float:
    """Linear warmup LR schedule."""
    if step < warmup_steps:
        lr_mult = (step + 1) / warmup_steps
    else:
        lr_mult = 1.0

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * lr_mult

    return optimizer.param_groups[0]['lr']


def train_single_config(
    batch_size: int,
    lr: float,
    config: Dict[str, Any],
    device: torch.device,
    logger: Optional[FlexibleLogger] = None,
    global_step: int = 0,
    verbose: bool = True
) -> Tuple[Dict[str, Any], int]:
    """
    Train a single (batch_size, lr) configuration.

    Args:
        batch_size: Batch size for this config
        lr: Learning rate for this config
        config: Full experiment config
        device: Device to train on
        logger: Optional logger for wandb/file logging
        global_step: Global step counter across all configs (for wandb)
        verbose: Whether to show progress bar

    Returns:
        results: Dict with final_loss, convergence info, training curve
        global_step: Updated global step counter
    """
    model_config = config['model']
    train_config = config['training']

    # Create model
    model = NanoTransformer(
        vocab_size=model_config['vocab_size'],
        d_model=model_config['d_model'],
        n_layers=model_config['n_layers'],
        n_heads=model_config['n_heads'],
        dropout=model_config.get('dropout', 0.0)
    ).to(device)

    # Create optimizer
    optimizer = create_optimizer(model, lr, config)

    # Create dataloader with specified batch size
    # Note: Using synthetic token sequences (same as Phase 1)
    dataset_config = config.get('dataset', {})
    vocab_size = model_config['vocab_size']
    seq_length = model_config['seq_length']

    train_loader = create_token_dataloader(
        vocab_size=vocab_size,
        seq_length=seq_length,
        num_sequences=dataset_config.get('num_sequences', 10000),
        batch_size=batch_size,
        pattern=dataset_config.get('pattern', 'random'),
        seed=config['experiment'].get('seed', 42),
        shuffle=True
    )
    train_iter = infinite_dataloader(train_loader)

    # Create validation dataloader (smaller)
    val_loader = create_token_dataloader(
        vocab_size=vocab_size,
        seq_length=seq_length,
        num_sequences=dataset_config.get('num_sequences', 10000) // 10,  # 10% for val
        batch_size=batch_size,
        pattern=dataset_config.get('pattern', 'random'),
        seed=config['experiment'].get('seed', 42) + 1,  # Different seed for val
        shuffle=False
    )
    val_iter = infinite_dataloader(val_loader)

    # Training loop
    steps = train_config['steps']
    warmup_steps = train_config.get('warmup_steps', 100)
    eval_interval = train_config.get('eval_interval', 100)
    grad_clip = train_config.get('grad_clip', 1.0)

    train_losses = []
    val_losses = []

    pbar = tqdm(range(steps), desc=f"B={batch_size}, LR={lr:.6f}", disable=not verbose)

    for step in pbar:
        model.train()

        # Get batch (returns tuple: (input_ids, target_ids))
        batch = next(train_iter)
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        logits = model(inputs)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # Optimizer step
        optimizer.step()

        # LR schedule (warmup only)
        current_lr = optimizer.param_groups[0]['lr']
        if step < warmup_steps:
            current_lr = get_lr_schedule(optimizer, warmup_steps, step)

        # Log training loss
        train_losses.append(loss.item())

        # Validation
        if (step + 1) % eval_interval == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                val_batch = next(val_iter)
                val_inputs, val_targets = val_batch
                val_inputs = val_inputs.to(device)
                val_targets = val_targets.to(device)

                val_logits = model(val_inputs)
                val_loss = F.cross_entropy(
                    val_logits.view(-1, val_logits.size(-1)),
                    val_targets.view(-1)
                )
                val_losses.append(val_loss.item())

            # Log to wandb if logger is available
            if logger is not None:
                logger.log({
                    f'B{batch_size}_LR{lr:.6f}/train_loss': loss.item(),
                    f'B{batch_size}_LR{lr:.6f}/val_loss': val_losses[-1],
                    f'B{batch_size}_LR{lr:.6f}/learning_rate': current_lr,
                    # Also log to general metrics for easy comparison
                    'train_loss': loss.item(),
                    'val_loss': val_losses[-1],
                    'batch_size': batch_size,
                    'lr': lr
                }, step=global_step + step)

        # Update progress bar
        if len(val_losses) > 0:
            pbar.set_postfix({'train_loss': f"{loss.item():.4f}", 'val_loss': f"{val_losses[-1]:.4f}"})

    # Compute final metrics
    convergence_window = train_config.get('convergence_window', 100)
    final_train_losses = train_losses[-convergence_window:]
    final_val_losses = val_losses[-max(1, convergence_window // eval_interval):]

    final_train_loss = np.mean(final_train_losses)
    final_val_loss = np.mean(final_val_losses)
    final_train_std = np.std(final_train_losses)

    # Check convergence
    convergence_threshold = train_config.get('convergence_threshold', 0.01)
    converged = final_train_std < convergence_threshold

    results = {
        'batch_size': batch_size,
        'lr': lr,
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
        'train_loss_std': final_train_std,
        'converged': converged,
        'num_steps': steps,
        'train_losses': train_losses,
        'val_losses': val_losses
    }

    # Update global step counter
    global_step += steps

    return results, global_step


def run_batch_sweep(config: Dict[str, Any], device: torch.device):
    """
    Run full batch size sweep experiment.

    For each batch size, test all LR candidates and find optimal.
    """
    print("\n" + "="*80)
    print("PHASE 2: BATCH SCALING VALIDATION")
    print("="*80)
    print(f"Testing hypothesis: LR ∝ Batch^(2/3) (Laplace, α=3)")
    print(f"vs standard:        LR ∝ Batch^(1/2) (Gaussian)")
    print("="*80)

    exp_config = config['experiment']
    sweep_config = config['batch_sweep']
    log_config = config.get('logging', {})

    # Set seed
    set_seed(exp_config.get('seed', 42))

    # Create output directory
    output_dir = Path(exp_config.get('output_dir', 'outputs/phase_2'))
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / 'logs'
    logs_dir.mkdir(exist_ok=True)

    # Initialize logger (wandb + files)
    logger = None
    if log_config.get('use_wandb', True):
        logger = FlexibleLogger(
            logging_config_path='config/logging_config.yaml',
            experiment_config=config
        )
        print(f"\n✓ WandB logging enabled: {log_config.get('wandb_project', 'laplace-scaling-phase2')}")
    else:
        print(f"\n✗ WandB logging disabled")

    # Get sweep parameters
    batch_sizes = sweep_config['batch_sizes']
    lr_candidates = sweep_config['lr_candidates']

    print(f"\nBatch sizes: {batch_sizes}")
    print(f"LR candidates: {lr_candidates}")
    print(f"Total experiments: {len(batch_sizes)} × {len(lr_candidates)} = {len(batch_sizes) * len(lr_candidates)}")
    print(f"Output directory: {output_dir}")

    # Initialize results CSV
    results_file = logs_dir / config['logging'].get('results_file', 'results.csv')
    fieldnames = [
        'batch_size', 'lr', 'final_train_loss', 'final_val_loss',
        'train_loss_std', 'converged', 'num_steps', 'timestamp'
    ]

    csv_file = open(results_file, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

    # Run experiments
    all_results = []
    total_experiments = len(batch_sizes) * len(lr_candidates)
    experiment_num = 0
    global_step = 0  # Global step counter for wandb (monotonically increasing)

    start_time = time.time()

    for batch_size in batch_sizes:
        print(f"\n{'='*80}")
        print(f"BATCH SIZE: {batch_size}")
        print(f"{'='*80}")

        for lr in lr_candidates:
            experiment_num += 1
            print(f"\n[{experiment_num}/{total_experiments}] Training: B={batch_size}, LR={lr:.6f}")

            # Log config to wandb (use global_step for monotonicity)
            if logger is not None:
                logger.log({
                    'config/batch_size': batch_size,
                    'config/lr': lr,
                    'config/experiment_num': experiment_num
                }, step=global_step)

            # Train
            results, global_step = train_single_config(
                batch_size=batch_size,
                lr=lr,
                config=config,
                device=device,
                logger=logger,
                global_step=global_step,
                verbose=True
            )

            # Add timestamp
            results['timestamp'] = time.time() - start_time

            # Save to CSV (write immediately for recovery)
            csv_row = {k: results[k] for k in fieldnames}
            csv_writer.writerow(csv_row)
            csv_file.flush()  # Flush to disk immediately

            # Store full results
            all_results.append(results)

            # Log summary to wandb (use global_step - already updated after training)
            if logger is not None:
                logger.log({
                    f'summary/B{batch_size}_LR{lr:.6f}_final_val_loss': results['final_val_loss'],
                    f'summary/B{batch_size}_LR{lr:.6f}_final_train_loss': results['final_train_loss'],
                    f'summary/B{batch_size}_LR{lr:.6f}_converged': 1.0 if results['converged'] else 0.0,
                    # Also log to general summary for easy viewing
                    'summary/final_val_loss': results['final_val_loss'],
                    'summary/final_train_loss': results['final_train_loss'],
                    'summary/batch_size': batch_size,
                    'summary/lr': lr
                }, step=global_step)

            # Print summary
            print(f"  Final val loss: {results['final_val_loss']:.4f}")
            print(f"  Converged: {results['converged']}")

    csv_file.close()

    # Finalize logger
    if logger is not None:
        logger.finish()
        print(f"\n✓ WandB logging finalized")

    # Print summary
    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"BATCH SWEEP COMPLETE")
    print(f"{'='*80}")
    print(f"Total experiments: {total_experiments}")
    print(f"Total time: {elapsed/3600:.2f} hours")
    print(f"Results saved to: {results_file}")
    print(f"\nNext step: Run analysis script to measure scaling exponent β")
    print(f"  python analysis/phase_2_analysis.py --results {results_file}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Batch Scaling Validation")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto/cpu/cuda)')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Print system info
    system_info = get_system_info()
    print(f"\nSystem Info:")
    print(f"  Python: {system_info['python_version']}")
    print(f"  PyTorch: {system_info['pytorch_version']}")
    print(f"  CUDA available: {system_info['cuda_available']}")
    if system_info['cuda_available']:
        print(f"  CUDA version: {system_info['cuda_version']}")
        print(f"  GPU: {system_info.get('gpu_name', 'Unknown')}")

    # Run experiment
    results = run_batch_sweep(config, device)

    print("\n✓ Experiment complete!")


if __name__ == '__main__':
    main()
