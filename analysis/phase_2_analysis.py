"""
Phase 2 Analysis: Batch Scaling Validation

Analyzes results from batch_scaling.py to measure scaling exponent β.

Tests hypothesis:
- LR_opt ∝ Batch^β
- Laplace (α=3): β ≈ 2/3 = 0.67
- Gaussian:     β ≈ 1/2 = 0.50

Usage:
    python analysis/phase_2_analysis.py --results outputs/phase_2/logs/results.csv
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import csv
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
from scipy.optimize import curve_fit

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_results(results_file: str) -> List[Dict]:
    """Load results CSV file."""
    results = []
    with open(results_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            row['batch_size'] = int(row['batch_size'])
            row['lr'] = float(row['lr'])
            row['final_train_loss'] = float(row['final_train_loss'])
            row['final_val_loss'] = float(row['final_val_loss'])
            row['train_loss_std'] = float(row['train_loss_std'])
            row['converged'] = row['converged'] == 'True'
            row['num_steps'] = int(row['num_steps'])
            row['timestamp'] = float(row['timestamp'])
            results.append(row)
    return results


def find_optimal_lr_per_batch(results: List[Dict]) -> Dict[int, Dict]:
    """
    Find optimal LR for each batch size.

    Returns:
        optimal_configs: {batch_size: {lr, final_loss, ...}}
    """
    # Group by batch size
    batch_groups = {}
    for result in results:
        batch_size = result['batch_size']
        if batch_size not in batch_groups:
            batch_groups[batch_size] = []
        batch_groups[batch_size].append(result)

    # Find minimum loss for each batch size
    optimal_configs = {}
    for batch_size, group in batch_groups.items():
        # Find config with minimum validation loss
        best_config = min(group, key=lambda x: x['final_val_loss'])
        optimal_configs[batch_size] = best_config

    return optimal_configs


def fit_power_law(batch_sizes: np.ndarray, lr_opts: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Fit power law: LR_opt = C · Batch^β

    Using log-log linear regression:
        log(LR_opt) = log(C) + β·log(Batch)

    Returns:
        β: scaling exponent
        C: multiplicative constant
        r_squared: goodness of fit
        β_std: standard error of β
    """
    # Log-log transform
    log_batch = np.log(batch_sizes)
    log_lr = np.log(lr_opts)

    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_batch, log_lr)

    β = slope
    C = np.exp(intercept)
    r_squared = r_value ** 2
    β_std = std_err

    return β, C, r_squared, β_std


def compute_transfer_error(
    optimal_configs: Dict[int, Dict],
    baseline_batch: int,
    baseline_lr: float,
    laplace_beta: float = 2/3,
    gaussian_beta: float = 1/2
) -> Dict[str, List[float]]:
    """
    Compute transfer error for Laplace vs Gaussian scaling rules.

    Transfer error = |Loss_transferred - Loss_optimal|

    Returns:
        errors: {
            'batch_sizes': [8, 16, ...],
            'laplace_errors': [...],
            'gaussian_errors': [...]
        }
    """
    batch_sizes = sorted(optimal_configs.keys())

    laplace_errors = []
    gaussian_errors = []

    for batch_size in batch_sizes:
        if batch_size == baseline_batch:
            # No transfer error at baseline
            laplace_errors.append(0.0)
            gaussian_errors.append(0.0)
            continue

        # Compute transferred LR using scaling rules
        ratio = batch_size / baseline_batch
        laplace_lr = baseline_lr * (ratio ** laplace_beta)
        gaussian_lr = baseline_lr * (ratio ** gaussian_beta)

        # Get optimal loss at this batch size
        optimal_loss = optimal_configs[batch_size]['final_val_loss']

        # We don't have the actual loss for transferred LRs, so we estimate
        # by finding the closest LR in our grid and using its loss
        # This is an approximation - ideally we'd retrain at the transferred LRs

        # For now, use optimal loss as reference (transfer error = 0 for optimal)
        # and compute what the theoretical LRs would be
        optimal_lr = optimal_configs[batch_size]['lr']

        # Compute relative error in LR (proxy for transfer error)
        laplace_lr_error = abs(laplace_lr - optimal_lr) / optimal_lr
        gaussian_lr_error = abs(gaussian_lr - optimal_lr) / optimal_lr

        laplace_errors.append(laplace_lr_error)
        gaussian_errors.append(gaussian_lr_error)

    return {
        'batch_sizes': batch_sizes,
        'laplace_errors': laplace_errors,
        'gaussian_errors': gaussian_errors
    }


def plot_lr_vs_batch(
    optimal_configs: Dict[int, Dict],
    β: float,
    C: float,
    r_squared: float,
    output_file: str,
    laplace_beta: float = 2/3,
    gaussian_beta: float = 1/2
):
    """
    Plot LR_opt vs Batch on log-log scale with fitted power law.
    """
    batch_sizes = np.array(sorted(optimal_configs.keys()))
    lr_opts = np.array([optimal_configs[b]['lr'] for b in batch_sizes])

    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot data points
    ax.scatter(batch_sizes, lr_opts, s=100, alpha=0.7, label='Measured optimal LR', zorder=3)

    # Plot fitted power law
    batch_range = np.linspace(batch_sizes.min(), batch_sizes.max(), 100)
    lr_fit = C * (batch_range ** β)
    ax.plot(batch_range, lr_fit, 'r--', linewidth=2,
            label=f'Fitted: LR ∝ B^{{{β:.3f}}} (R²={r_squared:.3f})', zorder=2)

    # Plot theoretical predictions
    # Use first data point as reference
    ref_batch = batch_sizes[0]
    ref_lr = lr_opts[0]

    laplace_lr = ref_lr * (batch_range / ref_batch) ** laplace_beta
    gaussian_lr = ref_lr * (batch_range / ref_batch) ** gaussian_beta

    ax.plot(batch_range, laplace_lr, 'g:', linewidth=2,
            label=f'Laplace theory: B^{{{laplace_beta:.3f}}} (α=3)', zorder=1)
    ax.plot(batch_range, gaussian_lr, 'b:', linewidth=2,
            label=f'Gaussian theory: B^{{{gaussian_beta:.3f}}} (α=∞)', zorder=1)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Batch Size', fontsize=14)
    ax.set_ylabel('Optimal Learning Rate', fontsize=14)
    ax.set_title('Phase 2: Optimal LR vs Batch Size Scaling', fontsize=16)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")


def plot_loss_heatmap(results: List[Dict], output_file: str):
    """
    Plot loss heatmap across (Batch, LR) grid.
    """
    # Get unique batch sizes and LRs
    batch_sizes = sorted(set(r['batch_size'] for r in results))
    lrs = sorted(set(r['lr'] for r in results))

    # Create loss matrix
    loss_matrix = np.zeros((len(batch_sizes), len(lrs)))
    for i, batch_size in enumerate(batch_sizes):
        for j, lr in enumerate(lrs):
            # Find result for this (batch, lr)
            matching = [r for r in results if r['batch_size'] == batch_size and r['lr'] == lr]
            if matching:
                loss_matrix[i, j] = matching[0]['final_val_loss']
            else:
                loss_matrix[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.imshow(loss_matrix, aspect='auto', cmap='viridis', interpolation='nearest')

    # Set ticks
    ax.set_xticks(np.arange(len(lrs)))
    ax.set_yticks(np.arange(len(batch_sizes)))
    ax.set_xticklabels([f"{lr:.6f}" for lr in lrs], rotation=45, ha='right')
    ax.set_yticklabels([str(b) for b in batch_sizes])

    ax.set_xlabel('Learning Rate', fontsize=14)
    ax.set_ylabel('Batch Size', fontsize=14)
    ax.set_title('Phase 2: Validation Loss Heatmap', fontsize=16)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Final Validation Loss', fontsize=12)

    # Mark optimal points
    optimal_configs = find_optimal_lr_per_batch(results)
    for i, batch_size in enumerate(batch_sizes):
        optimal_lr = optimal_configs[batch_size]['lr']
        j = lrs.index(optimal_lr)
        ax.scatter(j, i, color='red', s=200, marker='*', edgecolors='white', linewidths=2)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")


def plot_transfer_comparison(
    transfer_errors: Dict[str, List],
    output_file: str
):
    """
    Plot transfer error comparison: Laplace vs Gaussian scaling rules.
    """
    batch_sizes = transfer_errors['batch_sizes']
    laplace_errors = transfer_errors['laplace_errors']
    gaussian_errors = transfer_errors['gaussian_errors']

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(batch_sizes))
    width = 0.35

    ax.bar(x - width/2, laplace_errors, width, label='Laplace rule (B^{2/3})', alpha=0.8)
    ax.bar(x + width/2, gaussian_errors, width, label='Gaussian rule (B^{1/2})', alpha=0.8)

    ax.set_xlabel('Batch Size', fontsize=14)
    ax.set_ylabel('Relative LR Error', fontsize=14)
    ax.set_title('Phase 2: Transfer Error Comparison', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in batch_sizes])
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")


def plot_scaling_exponent(
    β: float,
    β_std: float,
    laplace_beta: float,
    gaussian_beta: float,
    output_file: str
):
    """
    Plot measured scaling exponent with confidence interval.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Theoretical values
    theories = ['Laplace\n(α=3)', 'Measured', 'Gaussian\n(α=∞)']
    betas = [laplace_beta, β, gaussian_beta]
    colors = ['green', 'red', 'blue']

    x_pos = np.arange(len(theories))
    ax.bar(x_pos, betas, color=colors, alpha=0.7, width=0.6)

    # Error bar for measured value
    ax.errorbar(1, β, yerr=2*β_std, fmt='none', color='black', capsize=10, linewidth=2)

    ax.set_ylabel('Scaling Exponent β', fontsize=14)
    ax.set_title('Phase 2: Measured Scaling Exponent\nLR ∝ Batch^β', fontsize=16)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(theories, fontsize=12)
    ax.axhline(y=laplace_beta, color='green', linestyle='--', alpha=0.5, label='Laplace (2/3)')
    ax.axhline(y=gaussian_beta, color='blue', linestyle='--', alpha=0.5, label='Gaussian (1/2)')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Add text with values
    for i, (theory, beta) in enumerate(zip(theories, betas)):
        if i == 1:  # Measured value
            ax.text(i, beta + 0.02, f'{beta:.3f} ± {2*β_std:.3f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        else:
            ax.text(i, beta + 0.02, f'{beta:.3f}',
                   ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")


def analyze_results(results_file: str, output_dir: str = None):
    """
    Main analysis function.
    """
    print("\n" + "="*80)
    print("PHASE 2 ANALYSIS: BATCH SCALING VALIDATION")
    print("="*80)

    # Load results
    print(f"\nLoading results from: {results_file}")
    results = load_results(results_file)
    print(f"  Loaded {len(results)} experiments")

    # Find optimal LR for each batch size
    print("\nFinding optimal LR for each batch size...")
    optimal_configs = find_optimal_lr_per_batch(results)

    print("\nOptimal configurations:")
    print(f"{'Batch Size':<15} {'Optimal LR':<15} {'Final Val Loss':<20} {'Converged'}")
    print("-" * 70)
    for batch_size in sorted(optimal_configs.keys()):
        config = optimal_configs[batch_size]
        print(f"{batch_size:<15} {config['lr']:<15.6f} {config['final_val_loss']:<20.4f} {config['converged']}")

    # Fit power law
    print("\nFitting power law: LR_opt = C · Batch^β")
    batch_sizes = np.array(sorted(optimal_configs.keys()))
    lr_opts = np.array([optimal_configs[b]['lr'] for b in batch_sizes])

    β, C, r_squared, β_std = fit_power_law(batch_sizes, lr_opts)

    print(f"\nResults:")
    print(f"  β (measured): {β:.4f} ± {2*β_std:.4f} (95% CI)")
    print(f"  C (constant): {C:.6f}")
    print(f"  R²:           {r_squared:.4f}")

    # Compare to theoretical predictions
    laplace_beta = 2/3
    gaussian_beta = 1/2

    print(f"\nTheoretical predictions:")
    print(f"  Laplace (α=3):  β = 2/3 ≈ {laplace_beta:.4f}")
    print(f"  Gaussian (α=∞): β = 1/2 = {gaussian_beta:.4f}")

    print(f"\nInterpretation:")
    if 0.60 <= β <= 0.74:
        print(f"  ✓ LAPLACE CONFIRMED: β ≈ 2/3")
        print(f"    Validates Phase 1 finding (α ≈ 3)")
        print(f"    Optimal scaling: LR ∝ Batch^(2/3)")
    elif 0.45 <= β <= 0.55:
        print(f"  × GAUSSIAN BEHAVIOR: β ≈ 1/2")
        print(f"    Contradicts Phase 1 finding")
        print(f"    Standard scaling: LR ∝ Batch^(1/2)")
    elif β < 0.45:
        print(f"  ? HEAVIER TAILS than expected")
        print(f"    Investigate discrepancy")
    else:
        print(f"  ? LIGHTER TAILS than expected")
        print(f"    Investigate discrepancy")

    # Compute transfer error
    baseline_batch = 256
    baseline_lr = 0.001
    print(f"\nComputing transfer error (baseline: B={baseline_batch}, LR={baseline_lr})...")
    transfer_errors = compute_transfer_error(
        optimal_configs, baseline_batch, baseline_lr, laplace_beta, gaussian_beta
    )

    mean_laplace_error = np.mean(transfer_errors['laplace_errors'])
    mean_gaussian_error = np.mean(transfer_errors['gaussian_errors'])

    print(f"  Mean transfer error (Laplace rule):  {mean_laplace_error:.4f}")
    print(f"  Mean transfer error (Gaussian rule): {mean_gaussian_error:.4f}")
    if mean_laplace_error < mean_gaussian_error:
        improvement = (1 - mean_laplace_error / mean_gaussian_error) * 100
        print(f"  → Laplace rule {improvement:.1f}% better")
    else:
        print(f"  → Gaussian rule is better")

    # Create output directory
    if output_dir is None:
        output_dir = Path(results_file).parent.parent / 'plots'
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    print(f"\nGenerating plots...")
    plot_lr_vs_batch(
        optimal_configs, β, C, r_squared,
        output_dir / 'lr_vs_batch.png',
        laplace_beta, gaussian_beta
    )

    plot_loss_heatmap(
        results,
        output_dir / 'loss_heatmap.png'
    )

    plot_transfer_comparison(
        transfer_errors,
        output_dir / 'transfer_comparison.png'
    )

    plot_scaling_exponent(
        β, β_std, laplace_beta, gaussian_beta,
        output_dir / 'scaling_exponent.png'
    )

    # Save summary JSON
    summary = {
        'β_measured': float(β),
        'β_std': float(β_std),
        'β_95ci': float(2 * β_std),
        'C': float(C),
        'r_squared': float(r_squared),
        'laplace_beta': laplace_beta,
        'gaussian_beta': gaussian_beta,
        'interpretation': 'laplace' if 0.60 <= β <= 0.74 else 'gaussian' if 0.45 <= β <= 0.55 else 'unknown',
        'optimal_configs': {
            int(b): {
                'lr': float(optimal_configs[b]['lr']),
                'final_val_loss': float(optimal_configs[b]['final_val_loss']),
                'converged': bool(optimal_configs[b]['converged'])
            }
            for b in optimal_configs
        }
    }

    summary_file = output_dir.parent / 'scaling_fit.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved: {summary_file}")

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Results summary saved to: {summary_file}")
    print(f"Plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Analyze batch scaling results")
    parser.add_argument('--results', type=str, required=True, help='Path to results CSV file')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for plots')

    args = parser.parse_args()

    analyze_results(args.results, args.output_dir)


if __name__ == '__main__':
    main()
