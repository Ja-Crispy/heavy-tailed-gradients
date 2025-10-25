"""
Phase 3b Analysis: Gradient Clipping Mechanism Test

Analyzes the relationship between gradient clipping thresholds and:
1. Final performance (loss)
2. Clipping frequency (how often gradients exceed threshold)
3. Gradient norm reduction (effective LR scaling)
4. Batch size interactions

Generates:
- Loss vs clip value (shows optimal clip threshold)
- Clip frequency vs clip value (shows when clipping activates)
- Gradient norm reduction (effective LR scaling from clipping)
- Summary table

Usage:
    python analysis/phase_3b_clip_analysis.py \
        --results outputs/phase_3b/logs/results.csv \
        --output outputs/phase_3b/plots
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List


def load_results(csv_path: Path) -> pd.DataFrame:
    """Load Phase 3b results CSV."""
    df = pd.read_csv(csv_path)

    # Convert 'null' strings to None for gradient_clip column
    df['gradient_clip'] = df['gradient_clip'].replace('null', None)
    df['gradient_clip'] = df['gradient_clip'].replace('None', None)
    df['gradient_clip'] = pd.to_numeric(df['gradient_clip'], errors='coerce')

    return df


def plot_loss_vs_clip(df: pd.DataFrame, output_dir: Path):
    """
    Plot final loss vs gradient clip value for each batch size.

    Shows how sensitive each batch size is to clipping threshold.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    unique_batches = sorted(df['batch_size'].unique())

    for batch in unique_batches:
        batch_df = df[df['batch_size'] == batch].copy()

        # Separate null (no clipping) from numeric clip values
        no_clip = batch_df[batch_df['gradient_clip'].isna()]
        with_clip = batch_df[batch_df['gradient_clip'].notna()].sort_values('gradient_clip')

        # Plot with clipping
        if not with_clip.empty:
            ax.plot(
                with_clip['gradient_clip'],
                with_clip['final_val_loss'],
                marker='o',
                label=f'B={batch}',
                linewidth=2,
                markersize=8
            )

        # Plot no-clipping as a separate point (at far right)
        if not no_clip.empty:
            # Plot at 10× max clip value for visibility
            max_clip = with_clip['gradient_clip'].max() if not with_clip.empty else 1.0
            ax.plot(
                [max_clip * 10],
                [no_clip['final_val_loss'].values[0]],
                marker='*',
                markersize=15,
                linestyle='',
                color=ax.get_lines()[-1].get_color() if ax.get_lines() else None,
                label=f'B={batch} (no clip)' if ax.get_lines() else None
            )

    ax.set_xscale('log')
    ax.set_xlabel('Gradient Clip Threshold (log scale)', fontsize=12)
    ax.set_ylabel('Final Validation Loss', fontsize=12)
    ax.set_title('Loss vs Gradient Clipping Threshold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / 'loss_vs_clip.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def plot_clip_frequency(df: pd.DataFrame, output_dir: Path):
    """
    Plot clipping frequency vs clip threshold for each batch size.

    Shows how often gradients exceed the threshold (require clipping).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    unique_batches = sorted(df['batch_size'].unique())

    for batch in unique_batches:
        batch_df = df[df['batch_size'] == batch].copy()

        # Only plot configs with clipping enabled
        with_clip = batch_df[batch_df['gradient_clip'].notna()].sort_values('gradient_clip')

        if not with_clip.empty:
            # Convert fraction to percentage
            clip_freq_pct = with_clip['avg_clip_frequency'] * 100

            ax.plot(
                with_clip['gradient_clip'],
                clip_freq_pct,
                marker='s',
                label=f'B={batch}',
                linewidth=2,
                markersize=8
            )

    ax.set_xscale('log')
    ax.set_xlabel('Gradient Clip Threshold (log scale)', fontsize=12)
    ax.set_ylabel('Clipping Frequency (%)', fontsize=12)
    ax.set_title('How Often Gradients Exceed Clip Threshold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / 'clip_frequency.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def plot_gradient_norm_reduction(df: pd.DataFrame, output_dir: Path):
    """
    Plot effective LR scaling from gradient clipping.

    Computes: norm_reduction = grad_norm_before / grad_norm_after
    Values > 1 indicate clipping is reducing effective LR.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    unique_batches = sorted(df['batch_size'].unique())

    for batch in unique_batches:
        batch_df = df[df['batch_size'] == batch].copy()

        # Only plot configs with clipping enabled
        with_clip = batch_df[batch_df['gradient_clip'].notna()].sort_values('gradient_clip')

        if not with_clip.empty:
            # Compute norm reduction ratio
            norm_reduction = with_clip['avg_grad_norm_before'] / with_clip['avg_grad_norm_after']

            ax.plot(
                with_clip['gradient_clip'],
                norm_reduction,
                marker='d',
                label=f'B={batch}',
                linewidth=2,
                markersize=8
            )

    # Add reference line at 1.0 (no reduction)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='No reduction')

    ax.set_xscale('log')
    ax.set_xlabel('Gradient Clip Threshold (log scale)', fontsize=12)
    ax.set_ylabel('Gradient Norm Reduction (before/after)', fontsize=12)
    ax.set_title('Effective LR Scaling from Gradient Clipping', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / 'gradient_norm_reduction.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def plot_grad_norms_comparison(df: pd.DataFrame, output_dir: Path):
    """
    Plot average gradient norms (before clipping) vs clip threshold.

    Shows whether batch size affects gradient scale.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    unique_batches = sorted(df['batch_size'].unique())

    for batch in unique_batches:
        batch_df = df[df['batch_size'] == batch].copy()

        # Sort by clip threshold
        sorted_df = batch_df.sort_values('gradient_clip') if 'gradient_clip' in batch_df else batch_df

        ax.plot(
            range(len(sorted_df)),
            sorted_df['avg_grad_norm_before'],
            marker='o',
            label=f'B={batch}',
            linewidth=2,
            markersize=8
        )

    ax.set_xlabel('Config Index', fontsize=12)
    ax.set_ylabel('Average Gradient Norm (before clip)', fontsize=12)
    ax.set_title('Gradient Norms Across Configurations', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / 'grad_norms_before.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def generate_summary_table(df: pd.DataFrame):
    """
    Print summary table of Phase 3b results.

    Shows key metrics for each configuration.
    """
    print("\n" + "="*100)
    print("PHASE 3B: GRADIENT CLIPPING MECHANISM TEST - SUMMARY")
    print("="*100)
    print(f"{'Batch':<8}{'LR':<10}{'Clip':<12}{'Val Loss':<12}{'Clip Freq %':<14}{'Norm Before':<14}{'Norm After':<14}")
    print("-"*100)

    for _, row in df.iterrows():
        batch = row['batch_size']
        lr = row['lr']

        # Format clip value
        clip = row['gradient_clip']
        if pd.isna(clip):
            clip_str = 'None'
        else:
            clip_str = f"{float(clip):.2f}"

        loss = row['final_val_loss']
        freq = row['avg_clip_frequency'] * 100  # Convert to percentage
        norm_before = row['avg_grad_norm_before']
        norm_after = row['avg_grad_norm_after']

        print(f"{batch:<8}{lr:<10.4f}{clip_str:<12}{loss:<12.4f}{freq:<14.2f}{norm_before:<14.4f}{norm_after:<14.4f}")

    print("="*100)


def analyze_clip_mechanism(df: pd.DataFrame):
    """
    Analyze and report on the gradient clipping mechanism hypothesis.

    Tests:
    1. Do small batches clip more frequently than large batches?
    2. Does clipping reduce performance?
    3. Is there an optimal clip threshold?
    """
    print("\n" + "="*100)
    print("MECHANISM ANALYSIS")
    print("="*100)

    unique_batches = sorted(df['batch_size'].unique())

    # Hypothesis 1: Small batches clip more frequently
    print("\nHypothesis 1: Small batches have higher clipping frequency")
    print("-" * 60)

    for batch in unique_batches:
        batch_df = df[df['batch_size'] == batch]
        with_clip = batch_df[batch_df['gradient_clip'].notna()]

        if not with_clip.empty:
            avg_freq = with_clip['avg_clip_frequency'].mean() * 100
            max_freq = with_clip['avg_clip_frequency'].max() * 100
            min_freq = with_clip['avg_clip_frequency'].min() * 100

            print(f"  B={batch}: avg={avg_freq:.1f}%, max={max_freq:.1f}%, min={min_freq:.1f}%")

    # Hypothesis 2: Optimal clip threshold exists
    print("\nHypothesis 2: Optimal clip threshold for each batch")
    print("-" * 60)

    for batch in unique_batches:
        batch_df = df[df['batch_size'] == batch].copy()

        # Find config with best loss
        best_idx = batch_df['final_val_loss'].idxmin()
        best_row = batch_df.loc[best_idx]

        clip_val = best_row['gradient_clip']
        clip_str = f"{clip_val:.2f}" if pd.notna(clip_val) else "None"

        print(f"  B={batch}: Best loss={best_row['final_val_loss']:.4f} at clip={clip_str}")

    # Hypothesis 3: Gradient norm scaling with batch size
    print("\nHypothesis 3: Gradient norms scale with batch size")
    print("-" * 60)

    # Compare average gradient norms across batches (at clip=1.0)
    for batch in unique_batches:
        batch_df = df[df['batch_size'] == batch]

        # Find clip=1.0 config (Phase 2.5 default)
        default_clip = batch_df[batch_df['gradient_clip'] == 1.0]

        if not default_clip.empty:
            norm_before = default_clip['avg_grad_norm_before'].values[0]
            norm_after = default_clip['avg_grad_norm_after'].values[0]
            reduction = norm_before / norm_after if norm_after > 0 else 1.0

            print(f"  B={batch} (clip=1.0): norm_before={norm_before:.4f}, norm_after={norm_after:.4f}, reduction={reduction:.2f}×")

    print("="*100)


def main():
    parser = argparse.ArgumentParser(description="Analyze Phase 3b gradient clipping experiment")
    parser.add_argument('--results', type=str, required=True, help="Path to results.csv")
    parser.add_argument('--output', type=str, required=True, help="Output directory for plots")
    args = parser.parse_args()

    results_path = Path(args.results)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*100)
    print("PHASE 3B ANALYSIS: GRADIENT CLIPPING MECHANISM TEST")
    print("="*100)

    print(f"\nLoading results from: {results_path}")
    df = load_results(results_path)
    print(f"  Loaded {len(df)} experiments")

    # Generate plots
    print("\nGenerating plots...")
    plot_loss_vs_clip(df, output_dir)
    plot_clip_frequency(df, output_dir)
    plot_gradient_norm_reduction(df, output_dir)
    plot_grad_norms_comparison(df, output_dir)

    # Generate summary table
    generate_summary_table(df)

    # Analyze mechanism
    analyze_clip_mechanism(df)

    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)
    print(f"Plots saved to: {output_dir}")
    print("="*100)


if __name__ == '__main__':
    main()
