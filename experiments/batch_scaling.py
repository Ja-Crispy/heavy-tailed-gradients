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
from typing import Dict, Any, List, Tuple, Optional, Set
import yaml
import csv
import time
import json
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
    infinite_dataloader as synthetic_infinite_dataloader,
    set_seed
)
from experiments.wikitext_data import (
    create_wikitext_dataloader,
    infinite_dataloader as wikitext_infinite_dataloader
)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


# ============================================================================
# Checkpointing Functions
# ============================================================================

def load_checkpoint(checkpoint_file: Path) -> Set[Tuple[int, float]]:
    """
    Load checkpoint of completed configurations.

    Args:
        checkpoint_file: Path to checkpoint JSON file

    Returns:
        Set of (batch_size, lr) tuples that have been completed
    """
    if not checkpoint_file.exists():
        return set()

    try:
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
            # Convert list of lists to set of tuples
            return {(item[0], item[1]) for item in data.get('completed', [])}
    except (json.JSONDecodeError, KeyError, ValueError):
        print(f"⚠ Warning: Checkpoint file corrupted, starting fresh")
        return set()


def save_checkpoint(checkpoint_file: Path, batch_size: int, lr: float):
    """
    Save completed configuration to checkpoint.

    Args:
        checkpoint_file: Path to checkpoint JSON file
        batch_size: Batch size that completed
        lr: Learning rate that completed
    """
    # Load existing checkpoint
    completed = load_checkpoint(checkpoint_file)

    # Add new completion
    completed.add((batch_size, lr))

    # Convert to list of lists for JSON serialization
    data = {
        'completed': list(list(item) for item in completed),
        'last_updated': time.time()
    }

    # Write atomically (write to temp, then rename)
    temp_file = checkpoint_file.with_suffix('.tmp')
    with open(temp_file, 'w') as f:
        json.dump(data, f, indent=2)

    # Atomic rename
    temp_file.replace(checkpoint_file)


def get_checkpoint_status(
    checkpoint_file: Path,
    all_configs: List[Tuple[int, float]]
) -> Dict[str, Any]:
    """
    Get human-readable checkpoint status.

    Args:
        checkpoint_file: Path to checkpoint file
        all_configs: List of all (batch_size, lr) configurations

    Returns:
        Dictionary with status information
    """
    completed = load_checkpoint(checkpoint_file)
    total = len(all_configs)
    num_completed = len(completed)
    num_remaining = total - num_completed
    percent = (num_completed / total * 100) if total > 0 else 0

    return {
        'total': total,
        'num_completed': num_completed,
        'num_remaining': num_remaining,
        'percent_complete': percent,
        'completed_configs': completed
    }


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
    verbose: bool = True,
    gradient_clip: Optional[float] = None
) -> Tuple[Dict[str, Any], int]:
    """
    Train a single (batch_size, lr, gradient_clip) configuration.

    Args:
        batch_size: Batch size for this config
        lr: Learning rate for this config
        config: Full experiment config
        device: Device to train on
        logger: Optional logger for wandb/file logging
        global_step: Global step counter across all configs (for wandb)
        verbose: Whether to show progress bar
        gradient_clip: Optional gradient clipping threshold (overrides config if provided)

    Returns:
        results: Dict with final_loss, convergence info, training curve, gradient stats
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

    # Apply torch.compile() if enabled (PyTorch 2.0+, significant speedup)
    use_compile = train_config.get('use_compile', False)
    if use_compile:
        try:
            model = torch.compile(model)
            if verbose:
                print("  ✓ torch.compile() enabled (expect 20-30% speedup)")
        except Exception as e:
            if verbose:
                print(f"  ⚠ torch.compile() failed: {e}, using eager mode")

    # Create optimizer
    optimizer = create_optimizer(model, lr, config)

    # Create dataloader with specified batch size
    dataset_config = config.get('dataset', {})
    dataset_name = dataset_config.get('name', 'synthetic')  # 'synthetic' or 'wikitext2'
    vocab_size = model_config['vocab_size']
    seq_length = model_config['seq_length']

    if dataset_name == 'wikitext2':
        # WikiText-2 with character-level tokenization (HuggingFace)
        train_loader = create_wikitext_dataloader(
            split='train',
            seq_length=seq_length,
            batch_size=batch_size,
            vocab_size=vocab_size,
            max_sequences=dataset_config.get('max_sequences', None),
            shuffle=True,
            num_workers=0  # Keep 0 for compatibility
        )
        train_iter = wikitext_infinite_dataloader(train_loader)

        # Validation dataloader
        val_loader = create_wikitext_dataloader(
            split='val',
            seq_length=seq_length,
            batch_size=batch_size,
            vocab_size=vocab_size,
            max_sequences=dataset_config.get('max_sequences', None),
            shuffle=False,
            num_workers=0
        )
        val_iter = wikitext_infinite_dataloader(val_loader)

    else:
        # Synthetic token sequences (Phase 2 default)
        train_loader = create_token_dataloader(
            vocab_size=vocab_size,
            seq_length=seq_length,
            num_sequences=dataset_config.get('num_sequences', 10000),
            batch_size=batch_size,
            pattern=dataset_config.get('pattern', 'random'),
            seed=config['experiment'].get('seed', 42),
            shuffle=True
        )
        train_iter = synthetic_infinite_dataloader(train_loader)

        # Validation dataloader (smaller)
        val_loader = create_token_dataloader(
            vocab_size=vocab_size,
            seq_length=seq_length,
            num_sequences=dataset_config.get('num_sequences', 10000) // 10,  # 10% for val
            batch_size=batch_size,
            pattern=dataset_config.get('pattern', 'random'),
            seed=config['experiment'].get('seed', 42) + 1,  # Different seed for val
            shuffle=False
        )
        val_iter = synthetic_infinite_dataloader(val_loader)

    # Training loop
    steps = train_config['steps']
    warmup_steps = train_config.get('warmup_steps', 100)
    eval_interval = train_config.get('eval_interval', 100)

    # Use provided gradient_clip parameter, or fall back to config
    # Note: gradient_clip can be:
    #   - None (not provided) → use config default
    #   - 0 or negative → disable clipping (set to None)
    #   - positive number → use as clip threshold
    if gradient_clip is None:
        # Not explicitly set in config list, use training config default
        grad_clip = train_config.get('grad_clip', 1.0)
    elif gradient_clip <= 0:
        # Explicitly disabled (use 0 or negative to mean "no clipping")
        grad_clip = None
    else:
        # Explicitly set to a positive threshold
        grad_clip = gradient_clip

    train_losses = []
    val_losses = []

    # Track gradient statistics (for Phase 3b)
    grad_norms_before = []
    grad_norms_after = []
    clip_events = []  # 1 if clipped, 0 otherwise

    # Format gradient clip for display
    clip_str = f"clip={grad_clip:.2f}" if grad_clip is not None else "clip=None"
    pbar = tqdm(range(steps), desc=f"B={batch_size}, LR={lr:.6f}, {clip_str}", disable=not verbose)

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

        # Track gradient norm before clipping
        total_norm_before = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm_before += param_norm.item() ** 2
        total_norm_before = total_norm_before ** 0.5

        # Gradient clipping (conditional)
        if grad_clip is not None:
            total_norm_after = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            total_norm_after = total_norm_after.item()
            was_clipped = 1.0 if total_norm_before > grad_clip else 0.0
        else:
            # No clipping applied
            total_norm_after = total_norm_before
            was_clipped = 0.0

        # Track clipping statistics
        grad_norms_before.append(total_norm_before)
        grad_norms_after.append(total_norm_after)
        clip_events.append(was_clipped)

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
                log_dict = {
                    f'B{batch_size}_LR{lr:.6f}/train_loss': loss.item(),
                    f'B{batch_size}_LR{lr:.6f}/val_loss': val_losses[-1],
                    f'B{batch_size}_LR{lr:.6f}/learning_rate': current_lr,
                    f'B{batch_size}_LR{lr:.6f}/grad_norm_before': total_norm_before,
                    f'B{batch_size}_LR{lr:.6f}/grad_norm_after': total_norm_after,
                    f'B{batch_size}_LR{lr:.6f}/clip_frequency': was_clipped,
                    # Also log to general metrics for easy comparison
                    'train_loss': loss.item(),
                    'val_loss': val_losses[-1],
                    'batch_size': batch_size,
                    'lr': lr,
                    'grad_norm_before': total_norm_before,
                    'grad_norm_after': total_norm_after,
                    'clip_frequency': was_clipped
                }

                # Add gradient clip value if it's being varied (Phase 3b)
                if gradient_clip is not None:
                    log_dict['gradient_clip'] = gradient_clip

                logger.log(log_dict, step=global_step + step)

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

    # Compute gradient statistics
    avg_grad_norm_before = np.mean(grad_norms_before)
    avg_grad_norm_after = np.mean(grad_norms_after)
    avg_clip_frequency = np.mean(clip_events)  # Fraction of steps where clipping occurred

    # Check convergence
    convergence_threshold = train_config.get('convergence_threshold', 0.01)
    converged = final_train_std < convergence_threshold

    results = {
        'batch_size': batch_size,
        'lr': lr,
        'gradient_clip': grad_clip if grad_clip is not None else float('nan'),  # Store None as NaN for CSV compatibility
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
        'train_loss_std': final_train_std,
        'converged': converged,
        'num_steps': steps,
        'avg_grad_norm_before': avg_grad_norm_before,
        'avg_grad_norm_after': avg_grad_norm_after,
        'avg_clip_frequency': avg_clip_frequency,
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

    # Get sweep parameters - support explicit configs or cartesian product
    if 'configs' in sweep_config:
        # Explicit config list (Phase 3b - gradient clipping sweep)
        explicit_configs = sweep_config['configs']
        all_configs = [
            (c['batch_size'], c['lr'], c.get('gradient_clip', None))
            for c in explicit_configs
        ]
        total_experiments = len(all_configs)

        print(f"\nExplicit config list: {total_experiments} configurations")
        print(f"Output directory: {output_dir}")

        # Show summary of what's being tested
        unique_batches = sorted(set(c[0] for c in all_configs))
        unique_lrs = sorted(set(c[1] for c in all_configs))
        unique_clips = sorted(set(c[2] for c in all_configs if c[2] is not None))
        print(f"  Batch sizes: {unique_batches}")
        print(f"  Learning rates: {unique_lrs}")
        if unique_clips:
            print(f"  Gradient clips: {unique_clips} + [None]")

    else:
        # Cartesian product (Phase 2/3a - standard batch/LR sweep)
        batch_sizes = sweep_config['batch_sizes']
        lr_candidates = sweep_config['lr_candidates']
        gradient_clip_default = sweep_config.get('gradient_clip', None)

        all_configs = [
            (b, lr, gradient_clip_default)
            for b in batch_sizes
            for lr in lr_candidates
        ]
        total_experiments = len(all_configs)

        print(f"\nBatch sizes: {batch_sizes}")
        print(f"LR candidates: {lr_candidates}")
        print(f"Total experiments: {len(batch_sizes)} × {len(lr_candidates)} = {total_experiments}")
        print(f"Output directory: {output_dir}")

    # Load checkpoint and filter completed configs
    checkpoint_file = output_dir / 'checkpoint.json'
    completed_configs = load_checkpoint(checkpoint_file)

    if completed_configs:
        status = get_checkpoint_status(checkpoint_file, all_configs)
        print(f"\n📊 Checkpoint Status:")
        print(f"  Completed: {status['num_completed']}/{status['total']} ({status['percent_complete']:.1f}%)")
        print(f"  Remaining: {status['num_remaining']}")
        print(f"  Status: Resuming from checkpoint")

        # Filter out completed configs (check first 2 elements for backward compatibility)
        remaining_configs = [
            (b, lr, clip) for b, lr, clip in all_configs
            if (b, lr) not in completed_configs
        ]
    else:
        print(f"\n📊 Checkpoint Status:")
        print(f"  No previous checkpoint found")
        print(f"  Status: Starting fresh")
        remaining_configs = all_configs

    print(f"  Configs to run: {len(remaining_configs)}")

    # Initialize results CSV with gradient statistics
    results_file = logs_dir / config['logging'].get('results_file', 'results.csv')
    fieldnames = [
        'batch_size', 'lr', 'gradient_clip',
        'final_train_loss', 'final_val_loss', 'train_loss_std',
        'avg_grad_norm_before', 'avg_grad_norm_after', 'avg_clip_frequency',
        'converged', 'num_steps', 'timestamp'
    ]

    # Open CSV in append mode if resuming, write mode if starting fresh
    csv_mode = 'a' if completed_configs else 'w'
    csv_file = open(results_file, csv_mode, newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not completed_configs:
        csv_writer.writeheader()

    # Run experiments
    all_results = []
    experiment_num = len(completed_configs)  # Start from number of completed
    global_step = 0  # Global step counter for wandb (monotonically increasing)

    start_time = time.time()

    for batch_size, lr, gradient_clip in remaining_configs:
        experiment_num += 1

        # Format display message
        clip_str = f", clip={gradient_clip:.2f}" if gradient_clip is not None else ", clip=None"
        print(f"\n[{experiment_num}/{total_experiments}] Training: B={batch_size}, LR={lr:.6f}{clip_str}")

        # Check batch size boundary (for printing header)
        if experiment_num == 1 or (batch_size != remaining_configs[max(0, experiment_num-2)][0]):
            print(f"\n{'='*80}")
            print(f"BATCH SIZE: {batch_size}")
            print(f"{'='*80}")

        # Log config to wandb (use global_step for monotonicity)
        if logger is not None:
            log_dict = {
                'config/batch_size': batch_size,
                'config/lr': lr,
                'config/experiment_num': experiment_num
            }
            if gradient_clip is not None:
                log_dict['config/gradient_clip'] = gradient_clip
            logger.log(log_dict, step=global_step)

        # Train (MUST be outside the batch size header check!)
        results, global_step = train_single_config(
            batch_size=batch_size,
            lr=lr,
            config=config,
            device=device,
            logger=logger,
            global_step=global_step,
            verbose=True,
            gradient_clip=gradient_clip  # Pass gradient_clip parameter
        )

        # Add timestamp
        results['timestamp'] = time.time() - start_time

        # Save to CSV (write immediately for recovery)
        csv_row = {k: results[k] for k in fieldnames}
        csv_writer.writerow(csv_row)
        csv_file.flush()  # Flush to disk immediately

        # Save checkpoint (for resume capability)
        save_checkpoint(checkpoint_file, batch_size, lr)

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
