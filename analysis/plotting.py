"""
Plotting and visualization utilities for Phase 1 experiments.

Generates publication-quality figures for:
- α evolution over training
- α vs width scaling
- Layer-wise α heatmaps
- Estimator comparisons
- QQ-plots for distribution validation
- Gradient tail plots (log-log)
"""

from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9


def load_metrics_from_json(log_file: str) -> Dict[str, List]:
    """
    Load metrics from JSONL log file.

    Args:
        log_file: Path to metrics.jsonl file

    Returns:
        Dictionary of {metric_name: [values]}
    """
    metrics = {}

    with open(log_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            for key, value in entry.items():
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(value)

    return metrics


def plot_alpha_evolution(metrics: Dict[str, List], output_path: str,
                        estimators: Optional[List[str]] = None,
                        title: str = "Tail Index Evolution During Training"):
    """
    Plot α evolution over training steps.

    Args:
        metrics: Dictionary of metrics
        output_path: Where to save plot
        estimators: Which estimators to plot (None = all)
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    steps = metrics.get('step', [])

    # Get all alpha keys
    alpha_keys = [k for k in metrics.keys() if k.startswith('alpha/') and not k.endswith('_std')]

    # Filter by estimators if specified
    if estimators:
        alpha_keys = [k for k in alpha_keys if any(est in k for est in estimators)]

    # Plot each estimator
    for key in alpha_keys:
        values = metrics[key]
        label = key.replace('alpha/', '').replace('_k', ' k=')

        # Check if std available
        std_key = f"{key}_std"
        if std_key in metrics:
            std_values = metrics[std_key]
            ax.plot(steps, values, label=label, alpha=0.8)
            ax.fill_between(steps,
                           np.array(values) - np.array(std_values),
                           np.array(values) + np.array(std_values),
                           alpha=0.2)
        else:
            ax.plot(steps, values, label=label, alpha=0.8)

    # Reference lines
    ax.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='α=2 (Gaussian)')
    ax.axhline(y=1.5, color='orange', linestyle='--', alpha=0.5, label='α=1.5 (Heavy-tail)')

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Tail Index (α)')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Set reasonable y-limits
    ax.set_ylim(0.5, 2.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_alpha_vs_width(metrics_by_width: Dict[int, Dict[str, List]],
                       output_path: str,
                       estimator: str = 'hill_k0.1',
                       title: str = "Tail Index vs Model Width"):
    """
    Plot α vs model width (Experiment 1.1).

    Args:
        metrics_by_width: Dict mapping width -> metrics
        output_path: Where to save
        estimator: Which estimator to use
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    widths = sorted(metrics_by_width.keys())
    alpha_means = []
    alpha_stds = []

    for width in widths:
        metrics = metrics_by_width[width]
        alpha_key = f'alpha/{estimator}'

        if alpha_key in metrics:
            # Use final 20% of training for steady-state estimate
            n_steps = len(metrics[alpha_key])
            steady_state_start = int(0.8 * n_steps)
            steady_state_alphas = metrics[alpha_key][steady_state_start:]

            # Remove NaNs
            steady_state_alphas = [a for a in steady_state_alphas if not np.isnan(a)]

            if steady_state_alphas:
                alpha_means.append(np.mean(steady_state_alphas))
                alpha_stds.append(np.std(steady_state_alphas))
            else:
                alpha_means.append(np.nan)
                alpha_stds.append(np.nan)
        else:
            alpha_means.append(np.nan)
            alpha_stds.append(np.nan)

    # Plot
    ax.errorbar(widths, alpha_means, yerr=alpha_stds,
               fmt='o-', capsize=5, capthick=2, markersize=8,
               label='Measured α (steady-state)')

    # Reference lines
    ax.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='α=2 (Gaussian)')
    ax.axhline(y=1.5, color='orange', linestyle='--', alpha=0.5, label='α=1.5')

    ax.set_xlabel('Model Width (d)')
    ax.set_ylabel('Tail Index (α)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_ylim(0.5, 2.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_layerwise_alpha(metrics: Dict[str, List], output_path: str,
                        step: Optional[int] = None,
                        title: str = "Layer-wise Tail Index Distribution"):
    """
    Plot heatmap of α by layer and parameter group (Experiment 1.2).

    Args:
        metrics: Metrics dictionary
        output_path: Where to save
        step: Which step to plot (None = final)
        title: Plot title
    """
    # Get layer-grouped alpha keys
    group_keys = [k for k in metrics.keys() if k.startswith('alpha_group/')]

    if not group_keys:
        print("No layer-grouped alpha metrics found")
        return

    # Extract group names and estimators
    groups = sorted(list(set([k.split('/')[1] for k in group_keys])))
    estimators = sorted(list(set([k.split('/')[-1] for k in group_keys])))

    # Create matrix
    if step is None:
        step_idx = -1  # Use final step
    else:
        steps = metrics.get('step', [])
        step_idx = steps.index(step) if step in steps else -1

    data = np.zeros((len(groups), len(estimators)))

    for i, group in enumerate(groups):
        for j, estimator in enumerate(estimators):
            key = f'alpha_group/{group}/{estimator}'
            if key in metrics and len(metrics[key]) > abs(step_idx):
                data[i, j] = metrics[key][step_idx]
            else:
                data[i, j] = np.nan

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(data, cmap='RdYlGn_r', aspect='auto', vmin=1.0, vmax=2.0)

    # Set ticks
    ax.set_xticks(np.arange(len(estimators)))
    ax.set_yticks(np.arange(len(groups)))
    ax.set_xticklabels(estimators, rotation=45, ha='right')
    ax.set_yticklabels(groups)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Tail Index (α)', rotation=270, labelpad=20)

    # Add text annotations
    for i in range(len(groups)):
        for j in range(len(estimators)):
            if not np.isnan(data[i, j]):
                text = ax.text(j, i, f'{data[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=9)

    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_estimator_comparison(metrics: Dict[str, List], output_path: str,
                              title: str = "Tail Index Estimates by Method"):
    """
    Boxplot comparing different estimators.

    Args:
        metrics: Metrics dictionary
        output_path: Where to save
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get alpha keys (steady-state only)
    alpha_keys = [k for k in metrics.keys()
                 if k.startswith('alpha/') and not k.endswith('_std')]

    # Collect steady-state values
    estimator_values = {}

    for key in alpha_keys:
        values = metrics[key]

        # Use final 20% for steady-state
        n_steps = len(values)
        steady_start = int(0.8 * n_steps)
        steady_values = [v for v in values[steady_start:] if not np.isnan(v)]

        if steady_values:
            label = key.replace('alpha/', '')
            estimator_values[label] = steady_values

    if not estimator_values:
        print("No alpha estimates found")
        return

    # Create boxplot
    labels = list(estimator_values.keys())
    data = [estimator_values[label] for label in labels]

    bp = ax.boxplot(data, labels=labels, patch_artist=True)

    # Color boxes
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)

    # Reference lines
    ax.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='α=2 (Gaussian)')
    ax.axhline(y=1.5, color='orange', linestyle='--', alpha=0.5, label='α=1.5')

    ax.set_ylabel('Tail Index (α)')
    ax.set_title(title)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0.5, 2.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_qq_plot(gradients: np.ndarray, alpha: float, output_path: str,
                title: str = "Q-Q Plot: Empirical vs α-Stable"):
    """
    Generate Q-Q plot to validate α-stable distribution assumption.

    Args:
        gradients: Gradient values
        alpha: Fitted alpha value
        output_path: Where to save
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Flatten and sort empirical data
    data = np.abs(gradients.flatten())
    data = data[np.isfinite(data)]
    data = np.sort(data)

    # Compute empirical quantiles
    n = len(data)
    empirical_quantiles = data

    # For α-stable, use power law approximation for tail
    # Theoretical quantiles: x_q such that P(X > x_q) = q
    # For α-stable: P(X > x) ~ x^(-α)
    probabilities = np.arange(1, n + 1) / (n + 1)
    theoretical_quantiles = (1 - probabilities) ** (-1/alpha)

    # Normalize for comparison
    theoretical_quantiles = theoretical_quantiles * np.median(empirical_quantiles) / np.median(theoretical_quantiles)

    # Plot
    ax.scatter(theoretical_quantiles, empirical_quantiles, alpha=0.5, s=10)

    # Reference line (perfect fit)
    min_val = min(theoretical_quantiles.min(), empirical_quantiles.min())
    max_val = max(theoretical_quantiles.max(), empirical_quantiles.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect fit')

    ax.set_xlabel(f'Theoretical Quantiles (α={alpha:.2f})')
    ax.set_ylabel('Empirical Quantiles')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Use log scale for better visualization
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_gradient_tail(gradients: np.ndarray, output_path: str,
                      title: str = "Gradient Tail Behavior (Log-Log Plot)"):
    """
    Plot gradient magnitudes in log-log scale to visualize tail behavior.

    Args:
        gradients: Gradient values
        output_path: Where to save
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Flatten and sort
    data = np.abs(gradients.flatten())
    data = data[np.isfinite(data) & (data > 0)]
    data = np.sort(data)[::-1]  # Descending order

    # Rank
    ranks = np.arange(1, len(data) + 1)

    # Plot
    ax.loglog(ranks, data, 'b.', alpha=0.5, markersize=2)

    # Fit power law to tail (top 10%)
    n_tail = int(0.1 * len(data))
    if n_tail > 10:
        log_ranks = np.log(ranks[:n_tail])
        log_data = np.log(data[:n_tail])

        # Linear fit in log-log space
        slope, intercept = np.polyfit(log_ranks, log_data, 1)
        alpha_estimated = -slope

        # Plot fitted line
        fitted = np.exp(intercept + slope * log_ranks)
        ax.loglog(ranks[:n_tail], fitted, 'r-', linewidth=2,
                 label=f'Power law fit: α ≈ {alpha_estimated:.2f}')

    ax.set_xlabel('Rank')
    ax.set_ylabel('Gradient Magnitude')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_clipping_effect(metrics_by_clip: Dict[Optional[float], Dict[str, List]],
                        output_path: str,
                        estimator: str = 'hill_k0.1',
                        title: str = "Effect of Gradient Clipping on Tail Index"):
    """
    Plot how gradient clipping affects α (Experiment 1.1).

    Args:
        metrics_by_clip: Dict mapping clip_value -> metrics
        output_path: Where to save
        estimator: Which estimator to use
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for clip_value, metrics in metrics_by_clip.items():
        steps = metrics.get('step', [])
        alpha_key = f'alpha/{estimator}'

        if alpha_key in metrics:
            values = metrics[alpha_key]
            label = f'Clip={clip_value}' if clip_value is not None else 'No clipping'
            ax.plot(steps, values, label=label, alpha=0.8)

    # Reference lines
    ax.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='α=2 (Gaussian)')

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Tail Index (α)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 2.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_experiment_1_1_plots(log_dir: str, output_dir: str):
    """
    Generate all plots for Experiment 1.1.

    Args:
        log_dir: Directory containing metrics.jsonl files
        output_dir: Where to save plots
    """
    log_dir = Path(log_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating Experiment 1.1 plots...")

    # Load all metrics
    metrics_files = list(log_dir.glob('**/*.jsonl')) + list(log_dir.glob('**/*.json'))

    if not metrics_files:
        print(f"No metrics files found in {log_dir}")
        return

    # For simplicity, use first file
    metrics = load_metrics_from_json(str(metrics_files[0]))

    # 1. Alpha evolution
    plot_alpha_evolution(
        metrics,
        str(output_dir / 'alpha_evolution.png'),
        title="Tail Index Evolution (Experiment 1.1)"
    )
    print("  ✓ alpha_evolution.png")

    # 2. Estimator comparison
    plot_estimator_comparison(
        metrics,
        str(output_dir / 'estimator_comparison.png')
    )
    print("  ✓ estimator_comparison.png")

    print(f"Plots saved to {output_dir}")


def generate_experiment_1_2_plots(log_dir: str, output_dir: str):
    """
    Generate all plots for Experiment 1.2.

    Args:
        log_dir: Directory containing metrics.jsonl
        output_dir: Where to save plots
    """
    log_dir = Path(log_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating Experiment 1.2 plots...")

    # Load metrics
    metrics_files = list(log_dir.glob('**/*.jsonl')) + list(log_dir.glob('**/*.json'))

    if not metrics_files:
        print(f"No metrics files found in {log_dir}")
        return

    metrics = load_metrics_from_json(str(metrics_files[0]))

    # 1. Alpha evolution
    plot_alpha_evolution(
        metrics,
        str(output_dir / 'alpha_evolution.png'),
        title="Tail Index Evolution (Experiment 1.2)"
    )
    print("  ✓ alpha_evolution.png")

    # 2. Layer-wise heatmap
    plot_layerwise_alpha(
        metrics,
        str(output_dir / 'alpha_layerwise.png')
    )
    print("  ✓ alpha_layerwise.png")

    # 3. Estimator comparison
    plot_estimator_comparison(
        metrics,
        str(output_dir / 'estimator_comparison.png')
    )
    print("  ✓ estimator_comparison.png")

    print(f"Plots saved to {output_dir}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate plots from experiment logs')
    parser.add_argument('--log_dir', type=str, required=True,
                       help='Directory containing log files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Where to save plots')
    parser.add_argument('--experiment', type=str, choices=['1.1', '1.2'],
                       required=True, help='Which experiment')

    args = parser.parse_args()

    if args.experiment == '1.1':
        generate_experiment_1_1_plots(args.log_dir, args.output_dir)
    else:
        generate_experiment_1_2_plots(args.log_dir, args.output_dir)
