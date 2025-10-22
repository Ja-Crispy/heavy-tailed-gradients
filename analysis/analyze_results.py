"""
Analyze Phase 1 experiment results.

Loads metrics from file logs and generates analysis plots + summary statistics.
"""

import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_style("whitegrid")


def load_metrics(log_file: Path) -> pd.DataFrame:
    """Load metrics from JSONL file."""
    metrics = []
    with open(log_file, 'r') as f:
        for line in f:
            if line.strip():
                metrics.append(json.loads(line))

    return pd.DataFrame(metrics)


def analyze_experiment_1_1(log_dir: Path, output_dir: Path):
    """
    Analyze Experiment 1.1: Synthetic Gradients.

    Tests hypotheses:
    - H1: α < 2 (heavy-tailed gradients)
    - H2: α varies with width
    - H3: Gradient clipping increases α (makes tails lighter)
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1.1 ANALYSIS: Synthetic Gradients")
    print("="*60)

    # Load data
    metrics_file = log_dir / "metrics.jsonl"
    if not metrics_file.exists():
        print(f"❌ Metrics file not found: {metrics_file}")
        return

    df = load_metrics(metrics_file)
    print(f"✓ Loaded {len(df)} measurements")

    # Check what columns we have
    print(f"\n📊 Available metrics: {list(df.columns)[:20]}...")

    # Look for alpha columns
    alpha_cols = [col for col in df.columns if 'alpha' in col.lower()]
    print(f"\n🔍 Alpha-related columns ({len(alpha_cols)}):")
    for col in alpha_cols[:10]:
        print(f"  - {col}")

    if not alpha_cols:
        print("\n⚠️  WARNING: No alpha measurements found in logs!")
        print("This might indicate:")
        print("  1. Alpha tracker didn't record values")
        print("  2. Too few gradient samples for estimation")
        print("  3. All estimates were NaN")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Analysis 1: Alpha vs Width (H2)
    print("\n📈 Analysis 1: Alpha vs Width")
    if 'width' in df.columns and any('hill' in col for col in alpha_cols):
        fig, ax = plt.subplots(figsize=(10, 6))

        # Get Hill estimates (most reliable)
        hill_col = [col for col in alpha_cols if 'hill' in col and 'std' not in col][0]

        for clip in df['gradient_clip'].unique():
            data = df[df['gradient_clip'] == clip]
            clip_label = f"clip={clip}" if clip > 0 else "no clip"

            ax.scatter(data['width'], data[hill_col], label=clip_label, alpha=0.6, s=20)

        ax.axhline(y=2.0, color='red', linestyle='--', label='α=2 (Gaussian)', alpha=0.5)
        ax.set_xlabel('Model Width')
        ax.set_ylabel('Tail Index (α)')
        ax.set_title('Tail Index vs Model Width')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.savefig(output_dir / 'alpha_vs_width.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: alpha_vs_width.png")
        plt.close()

    # Analysis 2: Alpha Evolution Over Training
    print("\n📈 Analysis 2: Alpha Evolution")
    if 'local_step' in df.columns:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        widths = sorted(df['width'].unique())[:4]

        for idx, width in enumerate(widths):
            data = df[df['width'] == width]

            for clip in data['gradient_clip'].unique():
                clip_data = data[data['gradient_clip'] == clip]
                clip_label = f"clip={clip}" if clip > 0 else "no clip"

                for col in alpha_cols:
                    if 'hill' in col and 'std' not in col and col in clip_data.columns:
                        axes[idx].plot(clip_data['local_step'], clip_data[col],
                                      label=clip_label, alpha=0.7)

            axes[idx].axhline(y=2.0, color='red', linestyle='--', alpha=0.3)
            axes[idx].set_title(f'Width {width}')
            axes[idx].set_xlabel('Training Step')
            axes[idx].set_ylabel('α (Hill)')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'alpha_evolution.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: alpha_evolution.png")
        plt.close()

    # Summary Statistics
    print("\n📊 Summary Statistics:")
    print("\nMean α by width and clipping:")

    if 'width' in df.columns and alpha_cols:
        hill_col = [col for col in alpha_cols if 'hill' in col and 'std' not in col][0]
        summary = df.groupby(['width', 'gradient_clip'])[hill_col].agg(['mean', 'std', 'min', 'max'])
        print(summary)

        # Save to CSV
        summary.to_csv(output_dir / 'summary_statistics.csv')
        print(f"\n✓ Saved: summary_statistics.csv")

    # Hypothesis Testing
    print("\n🔬 HYPOTHESIS TESTING:")
    print("\nH1: α < 2 (heavy-tailed gradients)?")
    if alpha_cols:
        hill_col = [col for col in alpha_cols if 'hill' in col and 'std' not in col][0]
        mean_alpha = df[hill_col].mean()
        median_alpha = df[hill_col].median()

        print(f"  Mean α: {mean_alpha:.3f}")
        print(f"  Median α: {median_alpha:.3f}")

        if mean_alpha < 2.0:
            print(f"  ✓ SUPPORTED: Mean α = {mean_alpha:.3f} < 2.0")
        else:
            print(f"  ✗ NOT SUPPORTED: Mean α = {mean_alpha:.3f} ≥ 2.0")

    print(f"\n✓ Analysis complete! Results saved to {output_dir}")


def analyze_experiment_1_2(log_dir: Path, output_dir: Path):
    """
    Analyze Experiment 1.2: Real Gradients.

    Tests hypotheses:
    - H1: α < 2 with real gradients
    - H2: Layer-specific behavior (attention vs FFN vs vector)
    - H3: Early vs steady-state dynamics
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1.2 ANALYSIS: Real Gradients")
    print("="*60)

    # Load data
    metrics_file = log_dir / "metrics.jsonl"
    if not metrics_file.exists():
        print(f"❌ Metrics file not found: {metrics_file}")
        return

    df = load_metrics(metrics_file)
    print(f"✓ Loaded {len(df)} measurements")

    # Check what columns we have
    print(f"\n📊 Available metrics: {list(df.columns)[:20]}...")

    # Look for alpha columns
    alpha_cols = [col for col in df.columns if 'alpha' in col.lower()]
    print(f"\n🔍 Alpha-related columns ({len(alpha_cols)}):")
    for col in alpha_cols[:15]:
        print(f"  - {col}")

    if not alpha_cols:
        print("\n⚠️  WARNING: No alpha measurements found in logs!")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Analysis 1: Layer-wise Alpha
    print("\n📈 Analysis 1: Layer-wise Alpha Comparison")
    group_cols = [col for col in alpha_cols if 'alpha_group' in col and 'hill' in col]

    if group_cols:
        fig, ax = plt.subplots(figsize=(10, 6))

        for col in group_cols:
            group_name = col.split('/')[1]  # Extract group name
            ax.plot(df['local_step'], df[col], label=group_name, alpha=0.7, linewidth=2)

        ax.axhline(y=2.0, color='red', linestyle='--', label='α=2 (Gaussian)', alpha=0.5)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Tail Index (α)')
        ax.set_title('Layer-wise Tail Index Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.savefig(output_dir / 'alpha_layerwise.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: alpha_layerwise.png")
        plt.close()

    # Analysis 2: Loss curve
    print("\n📈 Analysis 2: Training Loss")
    if 'loss' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['local_step'], df['loss'], alpha=0.7, linewidth=2)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        plt.savefig(output_dir / 'training_loss.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: training_loss.png")
        plt.close()

    # Summary Statistics
    print("\n📊 Summary Statistics:")

    if group_cols:
        print("\nMean α by parameter group:")
        group_summary = {}

        for col in group_cols:
            group_name = col.split('/')[1]
            mean_val = df[col].mean()
            std_val = df[col].std()
            group_summary[group_name] = {'mean': mean_val, 'std': std_val}
            print(f"  {group_name:15s}: α = {mean_val:.3f} ± {std_val:.3f}")

        # Save summary
        summary_df = pd.DataFrame(group_summary).T
        summary_df.to_csv(output_dir / 'layer_summary.csv')
        print(f"\n✓ Saved: layer_summary.csv")

    print(f"\n✓ Analysis complete! Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Analyze Phase 1 experiment results')
    parser.add_argument('--experiment', type=str, choices=['1.1', '1.2', 'both'], default='both',
                       help='Which experiment to analyze')
    parser.add_argument('--log-dir-1-1', type=str, default='outputs/exp_1_1',
                       help='Log directory for Experiment 1.1')
    parser.add_argument('--log-dir-1-2', type=str, default='outputs/exp_1_2',
                       help='Log directory for Experiment 1.2')
    parser.add_argument('--output-dir', type=str, default='outputs/analysis',
                       help='Output directory for analysis results')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.experiment in ['1.1', 'both']:
        analyze_experiment_1_1(
            log_dir=Path(args.log_dir_1_1),
            output_dir=output_dir / 'exp_1_1'
        )

    if args.experiment in ['1.2', 'both']:
        analyze_experiment_1_2(
            log_dir=Path(args.log_dir_1_2),
            output_dir=output_dir / 'exp_1_2'
        )

    print("\n" + "="*60)
    print("🎉 ALL ANALYSES COMPLETE!")
    print("="*60)
    print(f"\nResults saved to: {output_dir.absolute()}")
    print("\nNext steps:")
    print("  1. Review plots and summary statistics")
    print("  2. Check hypothesis test results")
    print("  3. Compare with wandb dashboard for interactive exploration")


if __name__ == '__main__':
    main()
