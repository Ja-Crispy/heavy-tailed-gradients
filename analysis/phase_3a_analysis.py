"""
Phase 3a Analysis: Extended LR Range and Phase Transition Discovery

Analyzes how optimal LR changes with extended LR range and reveals:
1. Phase transition from super-linear (β=1.17) to zero scaling (β=0)
2. 100% gradient clipping frequency at high LR
3. Optimal LR plateau at 0.05 for both batch sizes
4. Performance improvements despite loss of batch scaling

Generates:
- Loss vs LR curves for each batch size
- Gradient norms vs LR
- Clipping frequency vs LR (reveals saturation)
- Combined Phase 2.5 + 3a view (full phase transition)
- Summary tables

Usage:
    python analysis/phase_3a_analysis.py \
        --phase3a_results outputs/phase_3a/logs/results.csv \
        --phase2_5_results outputs/phase_2_5/logs/results.csv \
        --output outputs/phase_3a/plots_analysis
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional


def load_results(csv_path: Path) -> pd.DataFrame:
    """Load results CSV."""
    df = pd.read_csv(csv_path)
    return df


def plot_loss_vs_lr(df_3a: pd.DataFrame, df_2_5: Optional[pd.DataFrame], output_dir: Path):
    """
    Plot loss vs LR for each batch size, combining Phase 2.5 and 3a data.

    Shows how optimal LR evolves and where it plateaus.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    unique_batches = sorted(df_3a['batch_size'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_batches)))

    for idx, batch in enumerate(unique_batches):
        color = colors[idx]

        # Phase 3a data
        batch_df_3a = df_3a[df_3a['batch_size'] == batch].sort_values('lr')

        # Plot train loss
        ax1.plot(batch_df_3a['lr'], batch_df_3a['final_train_loss'],
                marker='o', label=f'B={batch} (Phase 3a)',
                color=color, linewidth=2, markersize=8)

        # Plot val loss
        ax2.plot(batch_df_3a['lr'], batch_df_3a['final_val_loss'],
                marker='o', label=f'B={batch} (Phase 3a)',
                color=color, linewidth=2, markersize=8)

        # Add Phase 2.5 data if available
        if df_2_5 is not None and batch in df_2_5['batch_size'].values:
            batch_df_2_5 = df_2_5[df_2_5['batch_size'] == batch].sort_values('lr')

            ax1.plot(batch_df_2_5['lr'], batch_df_2_5['final_train_loss'],
                    marker='s', linestyle='--', alpha=0.6,
                    color=color, linewidth=1.5, markersize=6,
                    label=f'B={batch} (Phase 2.5)')

            ax2.plot(batch_df_2_5['lr'], batch_df_2_5['final_val_loss'],
                    marker='s', linestyle='--', alpha=0.6,
                    color=color, linewidth=1.5, markersize=6,
                    label=f'B={batch} (Phase 2.5)')

    # Mark optimal LR for each batch
    for batch in unique_batches:
        batch_df = df_3a[df_3a['batch_size'] == batch]
        opt_idx = batch_df['final_val_loss'].idxmin()
        opt_row = batch_df.loc[opt_idx]

        ax2.axvline(x=opt_row['lr'], color='red', linestyle=':', alpha=0.3)
        ax2.text(opt_row['lr'], ax2.get_ylim()[1] * 0.98,
                f"B={batch} opt", rotation=90, va='top', fontsize=9)

    ax1.set_xscale('log')
    ax1.set_xlabel('Learning Rate (log scale)', fontsize=12)
    ax1.set_ylabel('Final Training Loss', fontsize=12)
    ax1.set_title('Training Loss vs LR', fontsize=14)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xscale('log')
    ax2.set_xlabel('Learning Rate (log scale)', fontsize=12)
    ax2.set_ylabel('Final Validation Loss', fontsize=12)
    ax2.set_title('Validation Loss vs LR (with optimal markers)', fontsize=14)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / 'loss_vs_lr.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def plot_gradient_norms_vs_lr(df: pd.DataFrame, output_dir: Path):
    """
    Plot gradient norms (before clipping) vs LR.

    Shows how gradients scale with LR.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    unique_batches = sorted(df['batch_size'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_batches)))

    for idx, batch in enumerate(unique_batches):
        batch_df = df[df['batch_size'] == batch].sort_values('lr')

        ax.plot(batch_df['lr'], batch_df['avg_grad_norm_before'],
                marker='o', label=f'B={batch}',
                color=colors[idx], linewidth=2, markersize=8)

    # Add horizontal line at clip threshold
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2,
               label='Clip threshold (1.0)', alpha=0.7)

    ax.set_xscale('log')
    ax.set_xlabel('Learning Rate (log scale)', fontsize=12)
    ax.set_ylabel('Average Gradient Norm (before clipping)', fontsize=12)
    ax.set_title('Gradient Norms vs LR\n(shows when clipping activates)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / 'grad_norms_vs_lr.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def plot_clip_frequency_vs_lr(df: pd.DataFrame, output_dir: Path):
    """
    Plot clipping frequency vs LR.

    Shows phase transition to 100% clipping saturation.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    unique_batches = sorted(df['batch_size'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_batches)))

    for idx, batch in enumerate(unique_batches):
        batch_df = df[df['batch_size'] == batch].sort_values('lr')

        # Convert to percentage
        clip_freq_pct = batch_df['avg_clip_frequency'] * 100

        ax.plot(batch_df['lr'], clip_freq_pct,
                marker='o', label=f'B={batch}',
                color=colors[idx], linewidth=2, markersize=8)

    # Add 100% saturation line
    ax.axhline(y=100, color='red', linestyle='--', linewidth=2,
               label='Saturation (100%)', alpha=0.7)

    # Shade saturation region
    ax.fill_between([df['lr'].min(), df['lr'].max()], 90, 100,
                     color='red', alpha=0.1, label='Saturation zone')

    ax.set_xscale('log')
    ax.set_xlabel('Learning Rate (log scale)', fontsize=12)
    ax.set_ylabel('Clipping Frequency (%)', fontsize=12)
    ax.set_title('Clipping Frequency vs LR\n(Phase transition to saturation)', fontsize=14)
    ax.set_ylim([0, 105])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / 'clip_frequency_vs_lr.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def plot_phase_transition(df_3a: pd.DataFrame, df_2_5: Optional[pd.DataFrame], output_dir: Path):
    """
    Combined visualization showing the phase transition from Phase 2.5 to 3a.

    Shows how β evolves across LR ranges.
    """
    if df_2_5 is None:
        print("  Skipping phase transition plot (no Phase 2.5 data)")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Find optimal LR for each batch in each phase
    optimal_lrs = []

    for phase_name, df in [('Phase 2.5', df_2_5), ('Phase 3a', df_3a)]:
        for batch in sorted(df['batch_size'].unique()):
            batch_df = df[df['batch_size'] == batch]
            if len(batch_df) > 0:
                opt_idx = batch_df['final_val_loss'].idxmin()
                opt_row = batch_df.loc[opt_idx]

                optimal_lrs.append({
                    'phase': phase_name,
                    'batch': batch,
                    'lr': opt_row['lr'],
                    'loss': opt_row['final_val_loss'],
                    'clip_freq': opt_row.get('avg_clip_frequency', 0) * 100
                })

    opt_df = pd.DataFrame(optimal_lrs)

    # Plot 1: Optimal LR vs Batch (shows β evolution)
    for phase in ['Phase 2.5', 'Phase 3a']:
        phase_data = opt_df[opt_df['phase'] == phase]
        marker = 'o' if phase == 'Phase 2.5' else 's'
        ax1.plot(phase_data['batch'], phase_data['lr'],
                marker=marker, label=phase, linewidth=2, markersize=10)

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Batch Size (log scale)', fontsize=12)
    ax1.set_ylabel('Optimal LR (log scale)', fontsize=12)
    ax1.set_title('Optimal LR vs Batch Size\n(Phase 2.5: β=1.17, Phase 3a: β=0)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Loss vs Batch
    for phase in ['Phase 2.5', 'Phase 3a']:
        phase_data = opt_df[opt_df['phase'] == phase]
        marker = 'o' if phase == 'Phase 2.5' else 's'
        ax2.plot(phase_data['batch'], phase_data['loss'],
                marker=marker, label=phase, linewidth=2, markersize=10)

    ax2.set_xscale('log')
    ax2.set_xlabel('Batch Size (log scale)', fontsize=12)
    ax2.set_ylabel('Final Validation Loss', fontsize=12)
    ax2.set_title('Best Loss vs Batch Size\n(Phase 3a achieves 23% better loss)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Clip frequency at optimal LR
    for phase in ['Phase 2.5', 'Phase 3a']:
        phase_data = opt_df[opt_df['phase'] == phase]
        marker = 'o' if phase == 'Phase 2.5' else 's'
        ax3.plot(phase_data['batch'], phase_data['clip_freq'],
                marker=marker, label=phase, linewidth=2, markersize=10)

    ax3.axhline(y=100, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax3.fill_between([opt_df['batch'].min(), opt_df['batch'].max()], 90, 100,
                      color='red', alpha=0.1, label='Saturation zone')

    ax3.set_xscale('log')
    ax3.set_xlabel('Batch Size (log scale)', fontsize=12)
    ax3.set_ylabel('Clipping Frequency at Optimal LR (%)', fontsize=12)
    ax3.set_title('Clipping Frequency Evolution\n(Phase 3a: 100% saturation)', fontsize=14)
    ax3.set_ylim([0, 105])
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: LR range tested
    phase_2_5_lrs = sorted(df_2_5['lr'].unique())
    phase_3a_lrs = sorted(df_3a['lr'].unique())

    ax4.scatter([1]*len(phase_2_5_lrs), phase_2_5_lrs, s=100, label='Phase 2.5 LR range', alpha=0.7)
    ax4.scatter([2]*len(phase_3a_lrs), phase_3a_lrs, s=100, label='Phase 3a LR range', alpha=0.7)

    # Connect max LRs
    ax4.plot([1, 2], [max(phase_2_5_lrs), max(phase_3a_lrs)],
             color='green', linestyle='--', linewidth=2, label='Max LR increase (5×)')

    ax4.set_yscale('log')
    ax4.set_xticks([1, 2])
    ax4.set_xticklabels(['Phase 2.5', 'Phase 3a'])
    ax4.set_ylabel('Learning Rate (log scale)', fontsize=12)
    ax4.set_title('LR Range Extension\n(Phase 3a tests 5× higher max LR)', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = output_dir / 'phase_transition_summary.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def generate_summary_table(df: pd.DataFrame):
    """Print summary table of Phase 3a results."""
    print("\n" + "="*100)
    print("PHASE 3A: EXTENDED LR RANGE - SUMMARY")
    print("="*100)
    print(f"{'Batch':<8}{'LR':<10}{'Val Loss':<12}{'Clip Freq %':<14}{'Grad Norm':<14}{'Converged':<12}")
    print("-"*100)

    for _, row in df.sort_values(['batch_size', 'lr']).iterrows():
        batch = row['batch_size']
        lr = row['lr']
        loss = row['final_val_loss']
        freq = row['avg_clip_frequency'] * 100
        norm = row['avg_grad_norm_before']
        conv = "✓" if row['converged'] else "✗"

        print(f"{batch:<8}{lr:<10.4f}{loss:<12.4f}{freq:<14.1f}{norm:<14.2f}{conv:<12}")

    print("="*100)

    # Find optimal for each batch
    print("\nOptimal Configurations:")
    print("-"*60)
    for batch in sorted(df['batch_size'].unique()):
        batch_df = df[df['batch_size'] == batch]
        opt_idx = batch_df['final_val_loss'].idxmin()
        opt_row = batch_df.loc[opt_idx]

        print(f"B={batch}: LR={opt_row['lr']:.4f}, Loss={opt_row['final_val_loss']:.4f}, "
              f"ClipFreq={opt_row['avg_clip_frequency']*100:.1f}%")

    print("="*100)


def analyze_phase_transition(df_3a: pd.DataFrame, df_2_5: Optional[pd.DataFrame]):
    """Analyze and report on the phase transition discovery."""
    print("\n" + "="*100)
    print("PHASE TRANSITION ANALYSIS: β=1.17 → β=0")
    print("="*100)

    if df_2_5 is not None:
        print("\n1. Optimal LR Evolution:")
        print("-"*60)

        for batch in sorted(set(df_2_5['batch_size'].unique()) & set(df_3a['batch_size'].unique())):
            # Phase 2.5 optimal
            batch_2_5 = df_2_5[df_2_5['batch_size'] == batch]
            opt_2_5_idx = batch_2_5['final_val_loss'].idxmin()
            opt_2_5 = batch_2_5.loc[opt_2_5_idx]

            # Phase 3a optimal
            batch_3a = df_3a[df_3a['batch_size'] == batch]
            opt_3a_idx = batch_3a['final_val_loss'].idxmin()
            opt_3a = batch_3a.loc[opt_3a_idx]

            lr_increase = opt_3a['lr'] / opt_2_5['lr']
            loss_improvement = (opt_2_5['final_val_loss'] - opt_3a['final_val_loss']) / opt_2_5['final_val_loss'] * 100

            print(f"B={batch}:")
            print(f"  Phase 2.5: LR={opt_2_5['lr']:.4f}, Loss={opt_2_5['final_val_loss']:.4f}")
            print(f"  Phase 3a:  LR={opt_3a['lr']:.4f}, Loss={opt_3a['final_val_loss']:.4f}")
            print(f"  Change: LR {lr_increase:.1f}× higher, Loss {loss_improvement:.1f}% better")

    print("\n2. Clipping Frequency Saturation:")
    print("-"*60)

    # Check if all configs have 100% clipping
    all_saturated = (df_3a['avg_clip_frequency'] >= 0.99).all()

    if all_saturated:
        print("  ✓ ALL configs show 100% clipping frequency")
        print("  → Gradient clipping is active every single step")
        print("  → Effective LR decoupled from nominal LR")
        print("  → This explains β=0 (no batch scaling)")
    else:
        below_100 = df_3a[df_3a['avg_clip_frequency'] < 0.99]
        print(f"  {len(below_100)} configs below 100% clipping:")
        for _, row in below_100.iterrows():
            print(f"    B={row['batch_size']}, LR={row['lr']:.4f}: {row['avg_clip_frequency']*100:.1f}%")

    print("\n3. Batch Scaling Behavior:")
    print("-"*60)

    # Check if optimal LRs are the same
    optimals = {}
    for batch in sorted(df_3a['batch_size'].unique()):
        batch_df = df_3a[df_3a['batch_size'] == batch]
        opt_idx = batch_df['final_val_loss'].idxmin()
        optimals[batch] = batch_df.loc[opt_idx, 'lr']

    if len(set(optimals.values())) == 1:
        print(f"  ✓ Both batches chose SAME optimal LR: {list(optimals.values())[0]:.4f}")
        print("  → β = 0 (no batch size dependence)")
        print("  → Gradient clipping ceiling dominates")
    else:
        print("  Different optimal LRs:")
        for batch, lr in optimals.items():
            print(f"    B={batch}: LR={lr:.4f}")

    print("="*100)


def main():
    parser = argparse.ArgumentParser(description="Analyze Phase 3a extended LR range experiment")
    parser.add_argument('--phase3a_results', type=str, required=True,
                        help="Path to Phase 3a results.csv")
    parser.add_argument('--phase2_5_results', type=str, default=None,
                        help="Path to Phase 2.5 results.csv (optional, for comparison)")
    parser.add_argument('--output', type=str, required=True,
                        help="Output directory for plots")
    args = parser.parse_args()

    phase3a_path = Path(args.phase3a_results)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*100)
    print("PHASE 3A ANALYSIS: EXTENDED LR RANGE & PHASE TRANSITION")
    print("="*100)

    print(f"\nLoading Phase 3a results from: {phase3a_path}")
    df_3a = load_results(phase3a_path)
    print(f"  Loaded {len(df_3a)} experiments")

    df_2_5 = None
    if args.phase2_5_results:
        phase2_5_path = Path(args.phase2_5_results)
        print(f"\nLoading Phase 2.5 results from: {phase2_5_path}")
        df_2_5 = load_results(phase2_5_path)
        print(f"  Loaded {len(df_2_5)} experiments")

    # Generate summary table
    generate_summary_table(df_3a)

    # Analyze phase transition
    analyze_phase_transition(df_3a, df_2_5)

    # Generate plots
    print("\nGenerating plots...")
    plot_loss_vs_lr(df_3a, df_2_5, output_dir)
    plot_gradient_norms_vs_lr(df_3a, output_dir)
    plot_clip_frequency_vs_lr(df_3a, output_dir)

    if df_2_5 is not None:
        plot_phase_transition(df_3a, df_2_5, output_dir)

    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)
    print(f"Plots saved to: {output_dir}")
    print("="*100)


if __name__ == '__main__':
    main()
