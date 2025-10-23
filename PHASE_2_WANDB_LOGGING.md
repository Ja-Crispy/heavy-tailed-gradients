# Phase 2: WandB Logging Structure

## Overview

Phase 2 batch scaling experiments now log to WandB with proper differentiation for each (batch_size, lr) configuration.

## WandB Project

- **Project name**: `laplace-scaling-phase2` (configured in `config/phase_2_batch_sweep.yaml`)
- **Total runs**: 49 (7 batch sizes × 7 LR values)
- **All metrics logged to a single run** for easy comparison

## Metric Namespaces

### 1. Per-Config Metrics: `B{batch}_LR{lr}/`

Each (batch_size, lr) configuration has its own namespace:

```
B8_LR0.000030/train_loss
B8_LR0.000030/val_loss
B8_LR0.000030/learning_rate
B8_LR0.000030/step

B16_LR0.000100/train_loss
B16_LR0.000100/val_loss
...
```

**Usage**: Track training curves for specific configurations

**Example**: Compare learning curves across different LRs for B=64:
- Filter by: `B64_LR0.000100/*`, `B64_LR0.001000/*`, etc.

### 2. General Metrics (for comparison)

Also logged to flat namespace for easy cross-config comparison:

```
train_loss        # Latest train loss across all configs
val_loss          # Latest val loss
batch_size        # Current batch size
lr                # Current LR
step              # Current step
```

**Usage**: See all configs on the same plot

**Example**: Plot `val_loss` vs `step` and use `batch_size` or `lr` for grouping

### 3. Summary Metrics: `summary/`

Final metrics logged after each config completes:

```
summary/B8_LR0.000030_final_val_loss
summary/B8_LR0.000030_final_train_loss
summary/B8_LR0.000030_converged

summary/final_val_loss     # Latest final val loss (for easy viewing)
summary/final_train_loss
summary/batch_size
summary/lr
```

**Usage**: Create summary table of all 49 experiments

**Example**: Create scatter plot of `summary/final_val_loss` vs `summary/lr` grouped by `summary/batch_size`

### 4. Config Tags: `config/`

Configuration info logged at start of each experiment:

```
config/batch_size
config/lr
config/experiment_num    # 1-49
```

## How to Use WandB Dashboard

### View 1: Compare LRs for Single Batch Size

**Goal**: Find optimal LR for B=64

**Steps**:
1. Go to Charts → Create new chart
2. Add metrics: `B64_LR0.000100/val_loss`, `B64_LR0.001000/val_loss`, etc.
3. X-axis: Step
4. Result: See all 7 LR curves for B=64

### View 2: Compare Optimal Configs Across Batch Sizes

**Goal**: See if optimal LR scales with batch size

**Steps**:
1. Create chart with `summary/final_val_loss`
2. X-axis: `summary/batch_size`
3. Color by: `summary/lr`
4. Result: For each batch size, see which LR gives lowest loss

### View 3: Heatmap of Final Loss

**Goal**: Visualize loss landscape across (B, LR) grid

**Steps**:
1. Create table with columns: `summary/batch_size`, `summary/lr`, `summary/final_val_loss`
2. Export to CSV for analysis
3. Or use WandB's parallel coordinates plot

### View 4: Training Curves for All Configs

**Goal**: See all 49 training curves at once

**Steps**:
1. Create chart with metric: `val_loss`
2. X-axis: Step
3. Group by: `batch_size` or `lr`
4. Result: See all configs, grouped by batch size or LR

## Example Queries

### Find best LR for each batch size:

```python
# In WandB dashboard or API
import wandb

api = wandb.Api()
run = api.run("your-entity/laplace-scaling-phase2/run-id")

# Get summary metrics
history = run.scan_history(keys=["summary/batch_size", "summary/lr", "summary/final_val_loss"])

# Group by batch size, find min loss
import pandas as pd
df = pd.DataFrame(history)
best_per_batch = df.groupby('summary/batch_size').apply(
    lambda x: x.loc[x['summary/final_val_loss'].idxmin()]
)
print(best_per_batch[['summary/batch_size', 'summary/lr', 'summary/final_val_loss']])
```

### Plot optimal LR vs batch size:

```python
import matplotlib.pyplot as plt
import numpy as np

batch_sizes = best_per_batch['summary/batch_size'].values
optimal_lrs = best_per_batch['summary/lr'].values

plt.loglog(batch_sizes, optimal_lrs, 'o-')
plt.xlabel('Batch Size')
plt.ylabel('Optimal LR')
plt.title('Phase 2: LR Scaling')
plt.grid(True, alpha=0.3)

# Fit power law
from scipy.stats import linregress
slope, intercept, r, p, se = linregress(np.log(batch_sizes), np.log(optimal_lrs))
print(f"β = {slope:.3f} ± {se:.3f}")
print(f"R² = {r**2:.3f}")
```

## Tips

1. **Use metric prefixes** for filtering:
   - All B=64 metrics: `B64_*`
   - All summary metrics: `summary/*`
   - Specific config: `B256_LR0.001000/*`

2. **Export data** for custom analysis:
   - WandB dashboard → Export → CSV
   - Or use WandB API to download programmatically

3. **Create custom reports**:
   - WandB Reports feature for combining multiple charts
   - Document findings with markdown + charts

4. **Compare to Phase 1**:
   - Phase 1 project: `heavy-tail-scaling` (or as configured)
   - Phase 2 project: `laplace-scaling-phase2`
   - Can create cross-project comparisons

## Expected Usage Workflow

1. **During training** (~12 hours):
   - Monitor general `val_loss` to ensure all configs are learning
   - Check for divergence or numerical issues

2. **After training**:
   - View `summary/*` metrics to find optimal configs
   - Create scatter plot: `summary/final_val_loss` vs `summary/lr` grouped by batch size
   - Identify optimal LR for each batch size

3. **Analysis**:
   - Use analysis script: `python analysis/phase_2_analysis.py --results outputs/phase_2/logs/results.csv`
   - Fits power law and generates plots locally
   - Compare to WandB metrics as validation

4. **Interpretation**:
   - If β ≈ 0.67: Laplace scaling confirmed ✓
   - If β ≈ 0.50: Gaussian scaling dominates
   - Compare WandB plots to generated analysis plots

---

**Date**: 2025-10-23
**Status**: WandB logging implemented and tested
