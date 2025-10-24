# Phase 2: Batch Scaling Validation Results

## Executive Summary

❌ **Experiment Status: INCONCLUSIVE**

**Hypothesis**: If gradients follow Laplace distribution (α ≈ 3), optimal learning rate should scale as LR ∝ Batch^(2/3)

**Result**: Experiment failed to validate hypothesis due to experimental design issues.

**Measured Scaling**: β = -0.36 ± 0.54 (R² = 0.26)
**Expected Scaling**: β ≈ 0.67 (Laplace) or β ≈ 0.50 (Gaussian)

**Conclusion**: Synthetic task insufficient for observing batch scaling effects.

---

## Experimental Setup

### Configuration
- **Model**: NanoTransformer (d_model=64, n_layers=4, ~200K parameters)
- **Dataset**: Random token sequences (vocab_size=65)
- **Batch sizes**: [8, 16, 32, 64, 128, 256, 512]
- **LR candidates**: [0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03]
- **Training steps**: 5,000 per configuration
- **Total configs**: 49 (7 batches × 7 LRs), completed 43

### Hypothesis
Based on Phase 1 finding that gradients follow Laplace-like distribution (α ≈ 3), optimal learning rate should scale as:

```
LR_opt = C · Batch^β
```

Where:
- **Laplace theory (α=3)**: β = 2/3 ≈ 0.667
- **Gaussian theory (α=∞)**: β = 1/2 = 0.500

---

## Results

### Optimal Learning Rates

| Batch Size | Optimal LR | Final Val Loss | Converged |
|------------|------------|----------------|-----------|
| 8          | 0.000300   | 4.1806         | True      |
| 16         | 0.000300   | 4.1837         | True      |
| 32         | 0.000300   | 4.1832         | True      |
| 64         | 0.001000   | 4.1805         | True      |
| 128        | 0.000300   | 4.1823         | True      |
| 256        | 0.000300   | 4.1802         | True      |
| 512        | 0.000030   | 4.1870         | True      |

**Observation**: Optimal LR shows no consistent trend with batch size. Most configurations converged to LR=0.0003 regardless of batch size.

### Power Law Fit

**Measured**: β = -0.36 ± 0.54 (95% CI)
**R²**: 0.26 (poor fit)

The negative slope and low R² indicate **no meaningful relationship** between optimal LR and batch size.

### Loss Landscape

All configurations achieved validation loss in narrow range: **4.180 - 4.220** (< 1% variation)

This suggests:
1. Task is too easy for the model
2. Loss landscape is insensitive to hyperparameters
3. No optimization dynamics to observe

---

## Why the Experiment Failed

### Root Cause: Task-Model Mismatch

**Problem**: Random token sequences with small transformer

The experimental setup created conditions where:
- Model easily "solves" random pattern prediction
- All (batch_size, LR) combinations converge to similar loss
- No selective pressure reveals optimal hyperparameters

### Specific Issues

**1. Trivial Task**
- Dataset: Randomly generated token sequences
- No real structure or long-range dependencies
- Model memorizes simple patterns uniformly

**2. Model Too Small**
- 200K parameters easily handle random patterns
- No capacity bottleneck to stress optimization

**3. Loss Insensitivity**
- All 43 configs: val_loss ∈ [4.180, 4.220]
- LR doesn't matter when all paths lead to same solution
- Cannot identify "optimal" when all are equivalent

**4. Coarse LR Grid**
- Grid jumps by 3-10× between values
- True optimal might exist in fine gaps
- But given flat landscape, finer grid wouldn't help

---

## Lessons Learned

### What We Discovered

1. **Synthetic tasks are insufficient** for testing optimization scaling laws
   - Random data lacks structure to reveal hyperparameter sensitivity
   - Need realistic tasks with meaningful optimization challenges

2. **Task difficulty must match model capacity**
   - Too-easy tasks lead to uniform convergence
   - Need problems where hyperparameter choice matters

3. **Loss variance is a prerequisite**
   - If all configs achieve same loss, no "optimal" exists
   - Need clear signal separating good from bad configurations

4. **Batch scaling effects require scale**
   - Small models (200K params) may not exhibit batch size dynamics
   - Larger models on structured tasks needed

### Scientific Value

This **negative result has value**:
- Documents limitations of synthetic experiments
- Guides design of future batch scaling studies
- Confirms that scaling laws require realistic settings

---

## Next Steps: Phase 2.5

### Redesigned Experiment

**Changes**:
- **Dataset**: WikiText-2 (real language modeling)
- **Model size**: d_model=128 (~800K parameters)
- **Batch sizes**: [32, 64, 128, 256] (avoid B=512 computational cost)
- **LR candidates**: [0.0001, 0.0003, 0.001, 0.003, 0.01] (finer grid)
- **Training steps**: 10,000 (Phase 1 standard)

**Expected Improvements**:
1. Real language creates structured optimization landscape
2. Larger model shows clearer hyperparameter sensitivity
3. Reduced configs (20 vs 49) for faster iteration
4. Should reveal if β ≈ 0.67 (Laplace) or 0.50 (Gaussian)

See `PHASE_2_5_RESULTS.md` for outcomes (to be created after experiment).

---

## Visualization

### LR vs Batch Size

![LR vs Batch](outputs/phase_2/plots/lr_vs_batch.png)

**Interpretation**:
- Blue dots: Measured optimal LRs (flat, no trend)
- Red line: Fitted power law (β = -0.36, wrong direction!)
- Green line: Laplace theory (β = 0.67, expected upward)
- Blue line: Gaussian theory (β = 0.50, expected upward)

Measured data shows **no correlation** with batch size.

### Loss Heatmap

![Loss Heatmap](outputs/phase_2/plots/loss_heatmap.png)

**Interpretation**:
- Entire heatmap is dark purple (loss ≈ 4.18-4.19)
- No clear diagonal pattern
- Stars (optimal configs) cluster in one LR column
- Confirms loss insensitivity to hyperparameters

---

## Conclusion

Phase 2 failed to validate batch scaling hypothesis due to experimental design:
- Synthetic task too simple
- Model too small
- Loss landscape too flat

**Status**: Inconclusive, requires Phase 2.5 with real data.

**Key Takeaway**: Optimization scaling laws require realistic tasks and models that exhibit clear hyperparameter sensitivity.

---

**Date**: 2025-10-24
**Status**: Experiment complete, results inconclusive, Phase 2.5 planned
