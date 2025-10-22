"""
Main experiment runner for Phase 1.

Runs:
- Experiment 1.1: Synthetic gradient model (MinimalFFN)
- Experiment 1.2: Real gradient flow (NanoTransformer)

Usage:
    python experiments/measure_alpha.py --config config/experiment_1_1.yaml
    python experiments/measure_alpha.py --config config/experiment_1_2.yaml
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.logger import FlexibleLogger, get_system_info
from core.tail_estimators import AlphaTracker, estimate_alpha_ensemble
from core.metrics import MetricsTracker, flatten_metrics
from models.minimal_ffn import MinimalFFN
from models.nano_transformer import NanoTransformer
from experiments.synthetic_data import (
    create_synthetic_dataloader,
    create_token_dataloader,
    infinite_dataloader,
    set_seed
)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """Create optimizer from config."""
    opt_config = config['training']
    opt_type = opt_config.get('optimizer', 'adamw').lower()

    if opt_type == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=opt_config['lr'],
            weight_decay=opt_config.get('weight_decay', 0.0),
            betas=opt_config.get('betas', [0.9, 0.999]),
            eps=opt_config.get('eps', 1e-8)
        )
    elif opt_type == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=opt_config['lr'],
            weight_decay=opt_config.get('weight_decay', 0.0),
            momentum=opt_config.get('momentum', 0.0)
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")

    return optimizer


def run_experiment_1_1(config: Dict[str, Any], device: torch.device):
    """
    Run Experiment 1.1: Synthetic Gradient Model.

    Tests if heavy tails are architectural (not data-driven).
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1.1: Synthetic Gradient Model")
    print("="*60)

    exp_config = config['experiment']
    model_config = config['model']
    train_config = config['training']
    meas_config = config['measurement']
    synth_config = config['synthetic']

    # Set seed
    set_seed(exp_config.get('seed', 42))

    # Create output directory
    output_dir = Path(exp_config.get('output_dir', 'outputs/exp_1_1'))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize logger
    logger = FlexibleLogger(
        logging_config_path='config/logging_config.yaml',
        experiment_config=config
    )

    # Log system info
    sys_info = get_system_info()
    print(f"\n📋 System Info:")
    print(f"  Python: {sys_info['python_version'].split()[0]}")
    print(f"  PyTorch: {sys_info['pytorch_version']}")
    print(f"  CUDA: {sys_info.get('cuda_available', False)}")
    if 'git_commit' in sys_info:
        print(f"  Git: {sys_info['git_commit'][:8]}")

    # Get widths to test
    widths = model_config['widths']
    print(f"\n🔬 Testing widths: {widths}")

    # Get gradient clipping values to test
    gradient_clips = train_config.get('gradient_clips', [None])

    # Global step counter for wandb (must be monotonically increasing)
    global_step = 0

    # Run for each width and clipping setting
    for width in widths:
        for clip_value in gradient_clips:
            global_step = run_single_experiment_1_1(
                width=width,
                clip_value=clip_value,
                config=config,
                device=device,
                logger=logger,
                global_step=global_step
            )

    # Finalize logging
    logger.finish()
    print(f"\n✅ Experiment 1.1 complete! Results saved to {output_dir}")


def run_single_experiment_1_1(width: int, clip_value: Optional[float],
                              config: Dict[str, Any],
                              device: torch.device,
                              logger: FlexibleLogger,
                              global_step: int = 0):
    """
    Run single configuration of Experiment 1.1.

    Returns:
        Updated global_step counter for wandb logging
    """
    train_config = config['training']
    meas_config = config['measurement']
    synth_config = config['synthetic']

    clip_str = f"_clip{clip_value}" if clip_value is not None else "_noclip"
    print(f"\n{'─'*60}")
    print(f"Width: {width}, Gradient Clip: {clip_value if clip_value else 'None'}")
    print(f"{'─'*60}")

    # Create model
    model = MinimalFFN(
        d_model=width,
        activation=config['model'].get('activation', 'relu'),
        use_bias=config['model'].get('use_bias', False)
    ).to(device)

    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create synthetic data
    # For Exp 1.1, we generate inputs and synthetic gradients
    num_samples = train_config['steps'] * train_config['batch_size']
    dataloader = create_synthetic_dataloader(
        d_model=width,
        num_samples=num_samples,
        batch_size=train_config['batch_size'],
        distribution=synth_config.get('input_dist', 'normal'),
        scale=synth_config.get('input_std', 1.0),
        seed=config['experiment'].get('seed', 42),
        shuffle=True
    )

    data_iter = infinite_dataloader(dataloader)

    # Create optimizer
    optimizer = create_optimizer(model, config)

    # Initialize alpha tracker
    alpha_tracker = AlphaTracker(
        window_size=meas_config.get('window_size', 100),
        estimators=meas_config.get('estimators', ['hill', 'pickands']),
        k_ratios=meas_config.get('k_ratios', [0.1])
    )

    # Training loop
    pbar = tqdm(range(train_config['steps']), desc=f"d={width}{clip_str}")

    for step in pbar:
        # Get batch
        batch = next(data_iter).to(device)

        # Forward pass
        output = model.forward_with_input_storage(batch)

        # Inject synthetic gradients (instead of real backward)
        model.inject_synthetic_gradients(
            grad_dist=synth_config.get('grad_dist', 'normal'),
            grad_std=synth_config.get('grad_std', 1.0),
            device=device
        )

        # Apply gradient clipping if specified
        if clip_value is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        # Measure alpha BEFORE clearing gradients!
        if step % meas_config.get('alpha_interval', 100) == 0 or step == train_config['steps'] - 1:
            # Update alpha tracker (needs gradients to be present!)
            alpha_tracker.update(model, prefix=f'd{width}{clip_str}')

            # Get summary statistics
            summary = alpha_tracker.get_summary()

            # Prepare metrics for logging
            metrics = {
                'local_step': step,  # Step within this width/clip combo
                'width': width,
                'gradient_clip': clip_value if clip_value is not None else 0.0,
            }

            # Add alpha estimates
            for est_name, stats in summary.items():
                metrics[f'alpha/{est_name}'] = stats.get('mean', float('nan'))
                metrics[f'alpha/{est_name}_std'] = stats.get('std', float('nan'))
                metrics[f'alpha/{est_name}_current'] = stats.get('current', float('nan'))

            # Add per-parameter alphas
            param_summaries = alpha_tracker.get_all_param_summaries()
            for param_name in ['W_in.weight', 'W_out.weight']:
                if param_name in param_summaries:
                    for est_name, stats in param_summaries[param_name].items():
                        metrics[f'alpha_param/{param_name}/{est_name}'] = stats.get('mean', float('nan'))

            # Compute gradient norms
            for name, param in model.named_parameters():
                if param.grad is not None:
                    metrics[f'grad_norm/{name}'] = param.grad.norm().item()

            # Log with global step (monotonically increasing across all runs)
            logger.log(metrics, global_step)
            global_step += 1  # Increment for next log

            # Update progress bar
            if summary:
                first_alpha = list(summary.values())[0]['mean']
                pbar.set_postfix({'α': f"{first_alpha:.3f}"})

        # Optimizer step (after measuring alpha!)
        optimizer.step()
        optimizer.zero_grad()

        # Periodic plot saving
        if step > 0 and step % 1000 == 0:
            logger.save_plots()

    return global_step


def run_experiment_1_2(config: Dict[str, Any], device: torch.device):
    """
    Run Experiment 1.2: Minimal Real Gradient Flow.

    Verifies phenomenon with real gradient flow in transformer.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1.2: Minimal Real Gradient Flow")
    print("="*60)

    exp_config = config['experiment']
    model_config = config['model']
    data_config = config['data']
    train_config = config['training']
    meas_config = config['measurement']

    # Set seed
    set_seed(exp_config.get('seed', 42))

    # Create output directory
    output_dir = Path(exp_config.get('output_dir', 'outputs/exp_1_2'))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize logger
    logger = FlexibleLogger(
        logging_config_path='config/logging_config.yaml',
        experiment_config=config
    )

    # Log system info
    sys_info = get_system_info()
    print(f"\n📋 System Info:")
    print(f"  Python: {sys_info['python_version'].split()[0]}")
    print(f"  PyTorch: {sys_info['pytorch_version']}")
    print(f"  CUDA: {sys_info.get('cuda_available', False)}")

    # Get model sizes to test
    d_models = model_config['d_models']
    print(f"\n🔬 Testing d_model: {d_models}")

    # Global step counter for wandb (must be monotonically increasing)
    global_step = 0

    # Run for each model size
    for d_model in d_models:
        global_step = run_single_experiment_1_2(
            d_model=d_model,
            config=config,
            device=device,
            logger=logger,
            global_step=global_step
        )

    # Finalize logging
    logger.finish()
    print(f"\n✅ Experiment 1.2 complete! Results saved to {output_dir}")


def run_single_experiment_1_2(d_model: int, config: Dict[str, Any],
                              device: torch.device, logger: FlexibleLogger,
                              global_step: int = 0):
    """
    Run single configuration of Experiment 1.2.

    Returns:
        Updated global_step counter for wandb logging
    """
    model_config = config['model']
    data_config = config['data']
    train_config = config['training']
    meas_config = config['measurement']

    print(f"\n{'─'*60}")
    print(f"Model size: d_model={d_model}")
    print(f"{'─'*60}")

    # Create model
    model = NanoTransformer(
        vocab_size=data_config.get('vocab_size', 1000),
        d_model=d_model,
        n_layers=model_config.get('n_layers', 4),
        n_heads=model_config.get('n_heads', 2),
        d_ff_multiplier=model_config.get('d_ff_multiplier', 4),
        dropout=model_config.get('dropout', 0.0),
        use_positional_encoding=model_config.get('use_positional_encoding', False)
    ).to(device)

    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create data
    num_sequences = data_config.get('num_sequences', 100000)
    train_split = data_config.get('split', [0.9, 0.1])[0]
    num_train = int(num_sequences * train_split)

    train_loader = create_token_dataloader(
        vocab_size=data_config.get('vocab_size', 1000),
        seq_length=data_config.get('seq_length', 128),
        num_sequences=num_train,
        batch_size=train_config['batch_size'],
        pattern=data_config.get('pattern', 'random'),
        seed=config['experiment'].get('seed', 42),
        shuffle=True
    )

    data_iter = infinite_dataloader(train_loader)

    # Create optimizer
    optimizer = create_optimizer(model, config)

    # Initialize trackers
    alpha_tracker = AlphaTracker(
        window_size=meas_config.get('window_size', 100),
        estimators=meas_config.get('estimators', ['hill', 'pickands']),
        k_ratios=meas_config.get('k_ratios', [0.1])
    )

    metrics_tracker = MetricsTracker()

    # Get parameter groups for layer-wise analysis
    param_groups_config = meas_config.get('param_groups', [])
    if isinstance(param_groups_config, list):
        # Convert list to dict for pattern matching
        param_patterns = {
            'attention': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'attn'],
            'ffn': ['ffn', 'up', 'down', 'gate'],
            'vector': ['embed', 'norm', 'scale']
        }
    else:
        # Already a dict
        param_patterns = param_groups_config

    # Training loop
    pbar = tqdm(range(train_config['steps']), desc=f"d_model={d_model}")

    for step in pbar:
        # Get batch
        input_ids, target_ids = next(data_iter)
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        # Forward pass
        logits = model(input_ids)

        # Compute loss (cross-entropy for next-token prediction)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1)
        )

        # Backward pass
        loss.backward()

        # Apply gradient clipping if specified
        clip_value = train_config.get('gradient_clip', None)
        if clip_value is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        # Measure at specified intervals (BEFORE clearing gradients!)
        alpha_intervals = meas_config.get('alpha_intervals', [100, 1000, 10000])
        continuous_interval = meas_config.get('continuous_interval', 100)

        should_measure = (
            step in alpha_intervals or
            step % continuous_interval == 0 or
            step == train_config['steps'] - 1
        )

        if should_measure:
            # Update alpha tracker (needs gradients!)
            alpha_tracker.update(model, prefix=f'd{d_model}')

            # Get overall summary
            summary = alpha_tracker.get_summary()

            # Get layer-grouped summary
            grouped_summary = alpha_tracker.get_layer_grouped_summary(param_patterns)

            # Prepare metrics
            metrics = {
                'local_step': step,  # Step within this d_model run
                'd_model': d_model,
                'loss': loss.item(),
            }

            # Add alpha estimates (global)
            for est_name, stats in summary.items():
                metrics[f'alpha/{est_name}'] = stats.get('mean', float('nan'))
                metrics[f'alpha/{est_name}_std'] = stats.get('std', float('nan'))

            # Add layer-grouped alphas
            for group_name, group_stats in grouped_summary.items():
                for est_name, stats in group_stats.items():
                    metrics[f'alpha_group/{group_name}/{est_name}'] = stats.get('mean', float('nan'))

            # Compute gradient norms by group
            grad_norms_by_group = {group: 0.0 for group in param_patterns.keys()}
            grad_norms_by_group['other'] = 0.0

            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()

                    # Assign to group
                    assigned = False
                    for group_name, patterns in param_patterns.items():
                        if any(pat in name for pat in patterns):
                            grad_norms_by_group[group_name] += grad_norm ** 2
                            assigned = True
                            break

                    if not assigned:
                        grad_norms_by_group['other'] += grad_norm ** 2

            # Take square root for L2 norm
            for group in grad_norms_by_group:
                grad_norms_by_group[group] = grad_norms_by_group[group] ** 0.5
                metrics[f'grad_norm_group/{group}'] = grad_norms_by_group[group]

            # Log with global step (monotonically increasing across all runs)
            logger.log(metrics, global_step)
            global_step += 1  # Increment for next log

            # Update progress bar
            if summary:
                first_alpha = list(summary.values())[0]['mean']
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'α': f"{first_alpha:.3f}"})
            else:
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Optimizer step (after measuring alpha!)
        optimizer.step()
        optimizer.zero_grad()

        # Periodic plot saving
        if step > 0 and step % 1000 == 0:
            logger.save_plots()

    return global_step


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run Phase 1 experiments')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to experiment config file')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu). Auto-detect if not specified.')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n🚀 Starting experiments on device: {device}")

    # Determine which experiment to run
    exp_name = config['experiment']['name']

    if 'exp_1_1' in exp_name or 'synthetic' in exp_name.lower():
        run_experiment_1_1(config, device)
    elif 'exp_1_2' in exp_name or 'real' in exp_name.lower():
        run_experiment_1_2(config, device)
    else:
        print(f"⚠️  Cannot determine experiment type from name: {exp_name}")
        print("Please ensure config name contains 'exp_1_1' or 'exp_1_2'")
        sys.exit(1)


if __name__ == '__main__':
    main()
