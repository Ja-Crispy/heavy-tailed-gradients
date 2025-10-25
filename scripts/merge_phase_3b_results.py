"""
Merge Phase 3b results with missing configs results.

Usage:
    python scripts/merge_phase_3b_results.py
"""

import pandas as pd
from pathlib import Path

# Paths
phase_3b_original = Path("outputs/phase_3b/logs/results.csv")
phase_3b_missing = Path("outputs/phase_3b_missing/logs/results.csv")
phase_3b_combined = Path("outputs/phase_3b_combined/logs/results.csv")

# Also handle l40-output paths
l40_phase_3b_original = Path("../l40-output/phase_3b/logs/results.csv")
l40_phase_3b_missing = Path("../l40-output/phase_3b_missing/logs/results.csv")
l40_phase_3b_combined = Path("../l40-output/phase_3b_combined/logs/results.csv")

def merge_results(original_path, missing_path, output_path):
    """Merge original and missing results, removing duplicates."""

    print(f"\nMerging Phase 3b results:")
    print(f"  Original: {original_path}")
    print(f"  Missing:  {missing_path}")
    print(f"  Output:   {output_path}")

    # Load both CSVs
    df_original = pd.read_csv(original_path)
    df_missing = pd.read_csv(missing_path)

    print(f"\n  Original configs: {len(df_original)}")
    print(f"  Missing configs:  {len(df_missing)}")

    # Combine
    df_combined = pd.concat([df_original, df_missing], ignore_index=True)

    # Remove duplicates (keep first occurrence)
    # Duplicates defined by (batch_size, lr, gradient_clip)
    df_combined = df_combined.drop_duplicates(
        subset=['batch_size', 'lr', 'gradient_clip'],
        keep='first'
    )

    print(f"  Combined (after dedup): {len(df_combined)}")

    # Sort for readability
    df_combined = df_combined.sort_values(['batch_size', 'gradient_clip'])

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_csv(output_path, index=False)

    print(f"\n✓ Merged results saved to: {output_path}")

    # Show summary
    print("\nFinal config summary:")
    for batch in sorted(df_combined['batch_size'].unique()):
        batch_df = df_combined[df_combined['batch_size'] == batch]
        clips = batch_df['gradient_clip'].tolist()
        clips_str = [f"{c:.2f}" if pd.notna(c) else "null" for c in clips]
        print(f"  B={int(batch)}: clips=[{', '.join(clips_str)}]")

    return df_combined

# Try both local and l40-output paths
if phase_3b_missing.exists():
    print("Found local outputs, merging...")
    merge_results(phase_3b_original, phase_3b_missing, phase_3b_combined)

if l40_phase_3b_missing.exists():
    print("\nFound l40-output, merging...")
    merge_results(l40_phase_3b_original, l40_phase_3b_missing, l40_phase_3b_combined)

print("\n" + "="*80)
print("MERGE COMPLETE")
print("="*80)
print("\nNext step: Run analysis on combined results:")
print("  python analysis/phase_3b_clip_analysis.py \\")
print("      --results outputs/phase_3b_combined/logs/results.csv \\")
print("      --output outputs/phase_3b_combined/plots")
