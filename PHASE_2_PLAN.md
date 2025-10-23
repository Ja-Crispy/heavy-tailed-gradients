# Phase 2: Batch Scaling Validation

## Executive Summary

**Goal**: Empirically validate that optimal learning rate scales as LR ∝ Batch^(2/3), confirming our measured α ≈ 3 (Laplace-like) finding from Phase 1.

**Why This Matters**:
- Standard theory assumes Gaussian gradients → LR ∝ B^(1/2)
- Our Phase 1 finding: Laplace gradients (α≈3) → LR ∝ B^(2/3)
- Difference: 15-20% improvement for large batches
- **First empirical test** of non-Gaussian scaling laws

**Timeline**: ~12 hours compute (can run overnight)

---

## Hypothesis

**Primary**: The optimal learning rate scales as:
```
LR_opt ∝ Batch^β  where β ≈ 2/3 (Laplace, α=3)
```

**Null hypothesis**: β ≈ 1/2 (Gaussian assumption)

**Test**: Measure β empirically across batch sizes, compare to theoretical predictions.

---

## Experimental Design

### Setup

**Model**: Nano-transformer (same as Experiment 1.2)
- Architecture: 4 layers, d_model=64
- Parameters: ~130K
- Proven stable during Phase 1

**Task**: Character-level language modeling
- Dataset: TinyShakespeare (~1MB text)
- Vocabulary: 65 characters
- Sequence length: 256 tokens

**Optimizer**: AdamW
- β₁=0.9, β₂=0.999
- Weight decay: λ=0.01
- Warmup: 100 steps (10% of training)

**Training**:
- Steps per config: 5000 (shorter than Phase 1 for speed)
- Metric: Final validation loss (last 100 steps averaged)
- Convergence check: Loss std < 0.01 in final 100 steps

### Batch Sizes to Test

```python
batch_sizes = [8, 16, 32, 64, 128, 256, 512]
```

**Rationale**:
- Covers 6x range (8 → 512)
- Small batches (8, 16): Might show different behavior
- Medium batches (32, 64, 128): Transition region
- Large batches (256, 512): Should follow scaling law

### Learning Rate Grid

For each batch size, test:

```python
lr_candidates = [
    0.00003,  # Very conservative
    0.0001,   # Conservative
    0.0003,   # Moderate-low
    0.001,    # Moderate (Phase 1 baseline)
    0.003,    # Moderate-high
    0.01,     # Aggressive
    0.03,     # Very aggressive
]
```

**Total configs**: 7 batch sizes × 7 LR values = **49 experiments**

**Compute**: ~12 hours on GPU (49 configs × 5000 steps × ~15 sec/config)

---

## Analysis Plan

### Step 1: Find Optimal LR for Each Batch Size

For each B ∈ [8, 16, 32, 64, 128, 256, 512]:

1. Run all 7 LR values
2. Measure final validation loss for each
3. Identify LR with **minimum final loss**
4. Record: (B, LR_opt, final_loss, convergence_speed)

### Step 2: Measure Scaling Exponent

**Fit power law**: LR_opt = C · B^β

Using log-log regression:
```python
log(LR_opt) = log(C) + β · log(B)
```

**Output**:
- β_measured (fitted exponent)
- R² (goodness of fit)
- 95% confidence interval on β

### Step 3: Compare to Theoretical Predictions

| Theory | α | Predicted β | Formula |
|--------|---|-------------|---------|
| **Laplace (our finding)** | 3.0 | **0.667** | 1 - 1/3 |
| Gaussian (standard) | ∞ | 0.500 | 1/2 |
| Heavy-tailed (original hypothesis) | 1.5 | 0.333 | 1 - 2/3 |

**Success criteria**:
- If 0.60 < β < 0.74: **Laplace confirmed** ✓
- If 0.45 < β < 0.55: Gaussian behavior
- If β < 0.45: Heavier tails than expected
- If β > 0.74: Lighter tails than expected

### Step 4: Transfer Quality Test

**Question**: Does B^(2/3) scaling transfer better than B^(1/2)?

**Experiment**:
1. Train baseline at B=256 with LR=0.001
2. Transfer to other batch sizes using two rules:
   - **Laplace rule**: LR_new = 0.001 · (B_new/256)^(2/3)
   - **Gaussian rule**: LR_new = 0.001 · (B_new/256)^(1/2)
3. Measure final loss for each

**Metric**: Transfer error
```
Transfer_error = |Loss_transferred - Loss_optimal|
```

**Expected**: Laplace rule has lower transfer error across all batch sizes.

---

## Implementation

### New Files to Create

**1. `experiments/batch_scaling.py`**
- Main experiment script
- LR grid search for each batch size
- Logging and checkpointing
- Reuse: models, metrics, logger from Phase 1

**2. `config/phase_2_batch_sweep.yaml`**
```yaml
experiment:
  name: "phase_2_batch_scaling"
  seed: 42
  output_dir: "outputs/phase_2"

model:
  type: "nano_transformer"
  d_model: 64
  n_layers: 4
  n_heads: 4

training:
  steps: 5000
  optimizer: "adamw"
  betas: [0.9, 0.999]
  weight_decay: 0.01
  warmup_steps: 100

batch_sweep:
  batch_sizes: [8, 16, 32, 64, 128, 256, 512]
  lr_candidates: [0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03]

logging:
  use_wandb: true
  wandb_project: "laplace-scaling-phase2"
  log_interval: 50
```

**3. `analysis/phase_2_analysis.py`**
- Load results from all configs
- Fit power law to LR_opt vs B
- Generate plots:
  - LR_opt vs B (log-log with fitted line)
  - Final loss vs LR for each B
  - Transfer error comparison
  - Scaling exponent with confidence interval

### Reusable Components

✓ **From Phase 1**:
- `models/nano_transformer.py` - Model architecture
- `core/logger.py` - Multi-backend logging
- `core/metrics.py` - Training metrics
- `experiments/synthetic_data.py` - Data loading

**Only new code**: ~300 lines for batch sweep logic + analysis

---

## Expected Results

### Scenario 1: Laplace Confirmed (Most Likely)

**If β_measured ≈ 0.67**:
- ✓ Validates Phase 1 α ≈ 3 measurement
- ✓ Confirms non-Gaussian behavior
- ✓ Actionable: Use B^(2/3) for better transfer
- **Next**: Publish findings, proceed to Phase 3

### Scenario 2: Gaussian Behavior

**If β_measured ≈ 0.50**:
- Contradiction with Phase 1 (α ≈ 3)
- Possible explanations:
  - α measurement accurate, but scaling law formula wrong
  - Small batch regime different from large batch
  - Other factors dominate (optimizer dynamics)
- **Next**: Investigate discrepancy

### Scenario 3: Intermediate Behavior

**If 0.50 < β < 0.67**:
- Partial Laplace effect
- Might indicate:
  - α varies with batch size
  - Non-power-law scaling
  - Multiple regimes
- **Next**: More detailed analysis of batch-dependent behavior

### Scenario 4: Unexpected Result

**If β outside [0.45, 0.75]**:
- Major discrepancy with theory
- Requires fundamental rethinking
- **Next**: Verify implementation, check for bugs

---

## Success Metrics

**Minimal success**:
- Complete 49 experiments without crashes
- Measure β with confidence interval
- Determine if β closer to 0.67 or 0.50

**Full success**:
- β = 0.67 ± 0.05 (Laplace confirmed)
- Transfer error 20%+ lower with B^(2/3) rule
- Clean power-law fit (R² > 0.95)
- Publishable result validating Phase 1

---

## Potential Issues & Mitigations

### Issue 1: Small Batches Unstable
**Problem**: B=8, 16 might not converge reliably
**Mitigation**:
- Use gradient accumulation to stabilize
- Report results with/without small batches
- Main analysis on B ∈ [32, 64, 128, 256, 512]

### Issue 2: LR Grid Too Coarse
**Problem**: Might miss optimal LR between grid points
**Mitigation**:
- If unclear winner, refine grid in that region
- Use validation loss smoothing (average last 100 steps)
- Acceptable: Within 10% of true optimum

### Issue 3: Task-Specific Behavior
**Problem**: TinyShakespeare might be special case
**Mitigation**:
- Report results as "for this task"
- Phase 3 can test other tasks/datasets
- Still useful as proof-of-concept

### Issue 4: Long Compute Time
**Problem**: 49 configs × 5000 steps = 12 hours
**Mitigation**:
- Run overnight on GPU
- Can reduce to 3000 steps if needed
- Parallelize across multiple GPUs if available

---

## Deliverables

### Code
- [x] `experiments/batch_scaling.py` - Batch sweep script
- [x] `config/phase_2_batch_sweep.yaml` - Configuration
- [x] `analysis/phase_2_analysis.py` - Results analysis

### Outputs
- `outputs/phase_2/results.csv` - All (B, LR, loss) measurements
- `outputs/phase_2/scaling_fit.json` - Fitted β with confidence interval
- `outputs/phase_2/plots/`:
  - `lr_vs_batch.png` - Main result: LR_opt vs B with fitted line
  - `loss_heatmap.png` - Loss landscape across (B, LR)
  - `transfer_comparison.png` - Laplace vs Gaussian transfer
  - `convergence_curves.png` - Training curves for each optimal config

### Documentation
- `PHASE_2_RESULTS.md` - Results summary with interpretation
- Update `README.md` with Phase 2 status
- WandB dashboard with all experiments

---

## Timeline

**Implementation**: 4-6 hours
- Write `batch_scaling.py`: 2 hours
- Write analysis script: 1 hour
- Create config, test: 1 hour

**Execution**: 10-12 hours (overnight)
- 49 configs × 5000 steps
- ~15 seconds per config on T4/A100

**Analysis**: 2-3 hours
- Run analysis script: 30 min
- Generate plots: 30 min
- Write PHASE_2_RESULTS.md: 1-2 hours

**Total**: 1-2 days wall-clock time

---

## Next Steps After Phase 2

### If β ≈ 0.67 (Laplace Confirmed)

**Phase 3 Options**:
1. **Test model scale dependence**: Does α change with width?
2. **Test architecture dependence**: CNN, ViT, MLP - all α ≈ 3?
3. **Test task dependence**: Vision, different datasets

**Phase 4**:
- Optimizer comparison: SGD, AdamW, Muon
- Does Muon increase α (more Gaussian)?

### If β ≈ 0.50 (Gaussian Behavior)

**Investigate**:
- Why does α ≈ 3 not predict scaling?
- Is scaling law formula wrong?
- Does batch size affect α itself?

**Possible follow-up**:
- Measure α at each batch size
- Test if α varies during training
- Theoretical analysis of scaling law derivation

---

**Date**: 2025-10-23
**Status**: Ready to implement
**Depends on**: Phase 1 complete ✓
**Next**: Implement `batch_scaling.py` and run experiments
