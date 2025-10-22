# Phase 1 Implementation - Validation Checklist

## Pre-Experiment Checklist

Before running full Phase 1 experiments, verify:

### ✅ 1. Installation & Setup

- [ ] Python ≥ 3.9 installed
- [ ] Virtual environment created and activated
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] PyTorch with CUDA support (if using GPU)
- [ ] wandb configured (if using wandb logging)

**Validation Command:**
```bash
python -c "import torch; import numpy; import scipy; import yaml; print('✅ Core dependencies OK')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

### ✅ 2. Core Components

#### Tail Estimators
- [ ] Hill estimator implemented and tested
- [ ] Pickands estimator implemented and tested
- [ ] ML estimator implemented (levy package)
- [ ] AlphaTracker class functional
- [ ] Preprocessing handles edge cases (zeros, NaNs)

**Validation Command:**
```bash
python tests/test_tail_estimators.py
```

**Expected Output:** All tests pass, estimators give reasonable α values for known distributions

---

#### Models
- [ ] MinimalFFN forward pass works
- [ ] MinimalFFN synthetic gradient injection works
- [ ] NanoTransformer forward pass works
- [ ] NanoTransformer backward pass works
- [ ] Parameter grouping works

**Validation Commands:**
```bash
python models/minimal_ffn.py
python models/nano_transformer.py
```

**Expected Output:** Forward/backward tests pass, gradient shapes correct

---

#### Data Generation
- [ ] Synthetic input generation (normal, cauchy, laplace)
- [ ] Synthetic token generation (random, repeated, arithmetic)
- [ ] DataLoaders work correctly
- [ ] Infinite iterator works
- [ ] Seeds provide reproducibility

**Validation Command:**
```bash
python experiments/synthetic_data.py
```

**Expected Output:** All distribution types work, data shapes correct

---

#### Logging System
- [ ] WandB logger (if enabled)
- [ ] File logger (JSON/CSV)
- [ ] Plot logger
- [ ] Flexible backend system
- [ ] Metric flattening

**Validation:** Verify outputs created in `outputs/` after smoke tests

---

### ✅ 3. Configuration Files

- [ ] `config/experiment_1_1.yaml` - properly formatted
- [ ] `config/experiment_1_2.yaml` - properly formatted
- [ ] `config/logging_config.yaml` - properly formatted
- [ ] All required fields present
- [ ] Reasonable hyperparameter values

**Validation:** Check YAML syntax
```bash
python -c "import yaml; yaml.safe_load(open('config/experiment_1_1.yaml'))"
python -c "import yaml; yaml.safe_load(open('config/experiment_1_2.yaml'))"
```

---

### ✅ 4. Smoke Tests

Run minimal experiments to verify end-to-end functionality:

**Validation Command:**
```bash
python tests/smoke_test.py
```

**Expected Results:**
- [ ] Model instantiation tests pass
- [ ] Synthetic data tests pass
- [ ] Experiment 1.1 smoke test completes (50 steps)
- [ ] Experiment 1.2 smoke test completes (50 steps)
- [ ] Log files created in temp directories
- [ ] No crashes or errors

**Success Criteria:** All 4 smoke tests pass

---

### ✅ 5. Comprehensive Test Suite

Run all tests together:

**Validation Command:**
```bash
python run_tests.py
```

**Expected Results:**
- [ ] Tail estimator tests: PASSED
- [ ] Smoke tests: PASSED
- [ ] MinimalFFN tests: PASSED
- [ ] NanoTransformer tests: PASSED
- [ ] Synthetic data tests: PASSED

**Success Criteria:** 5/5 tests passed

---

## Experiment Execution Checklist

### Before Running Experiment 1.1

- [ ] Review `config/experiment_1_1.yaml`
- [ ] Ensure sufficient disk space (~500MB per width)
- [ ] Estimate runtime: ~30 min for all widths on GPU, ~2 hours on CPU
- [ ] Configure wandb or disable it
- [ ] Check output directory permissions

**Command:**
```bash
python experiments/measure_alpha.py --config config/experiment_1_1.yaml
```

---

### Before Running Experiment 1.2

- [ ] Review `config/experiment_1_2.yaml`
- [ ] Ensure sufficient disk space (~1GB per model size)
- [ ] Estimate runtime: ~1-2 hours on GPU, ~4-6 hours on CPU
- [ ] Configure wandb or disable it
- [ ] Check output directory permissions

**Command:**
```bash
python experiments/measure_alpha.py --config config/experiment_1_2.yaml
```

---

## Post-Experiment Validation

### ✅ 1. Verify Outputs

After running experiments, check:

#### Experiment 1.1 Outputs
```
outputs/exp_1_1_synthetic_gradients/
├── logs/
│   ├── metrics.json      # Should exist, ~100-500KB
│   └── metrics.csv       # Should exist
├── plots/
│   ├── alpha_evolution.png     # Should show α over time
│   ├── estimator_comparison.png
│   └── (other plots)
└── checkpoints/          # Optional
```

**Validation Checks:**
- [ ] Log files exist and are non-empty
- [ ] Metrics contain 'alpha/hill', 'alpha/pickands' keys
- [ ] At least 2 alpha measurements per width (based on alpha_interval)
- [ ] No excessive NaN values in alpha estimates
- [ ] Plots generated (if enabled)

---

#### Experiment 1.2 Outputs
```
outputs/exp_1_2_real_gradients/
├── logs/
│   ├── metrics.json      # Should exist, ~200-800KB
│   └── metrics.csv
├── plots/
│   ├── alpha_evolution.png
│   ├── alpha_layerwise.png      # Heatmap by layer
│   └── (other plots)
└── checkpoints/
```

**Validation Checks:**
- [ ] Log files exist
- [ ] Metrics contain 'alpha_group/attention', 'alpha_group/ffn', 'alpha_group/vector'
- [ ] Loss decreases over training
- [ ] Layer-wise alpha measurements present
- [ ] Plots generated (if enabled)

---

### ✅ 2. Scientific Validation

#### Experiment 1.1 - Key Findings to Check

**Hypothesis H1:** α < 2 (heavy-tailed)
- [ ] Hill estimates: 1.0 < α < 2.0?
- [ ] Pickands estimates: 1.0 < α < 2.0?
- [ ] Consistent across estimators?

**Hypothesis H2:** α varies with width
- [ ] Plot alpha vs width
- [ ] Any trend visible?
- [ ] Document in results

**Hypothesis H3:** Gradient clipping reduces α
- [ ] If tested: clipped gradients have higher α?
- [ ] Document findings

**Generate plots:**
```bash
python analysis/plotting.py --log_dir outputs/exp_1_1_synthetic_gradients/logs \
                            --output_dir outputs/exp_1_1_synthetic_gradients/plots \
                            --experiment 1.1
```

---

#### Experiment 1.2 - Key Findings to Check

**Hypothesis H1:** α < 2 with real gradients
- [ ] Global alpha estimates < 2.0?
- [ ] Consistent with Exp 1.1?

**Hypothesis H2:** Layer-specific behavior
- [ ] Attention layers: α = ?
- [ ] FFN layers: α = ?
- [ ] Vector params: α = ?
- [ ] Any systematic differences?

**Hypothesis H3:** Early vs steady-state
- [ ] Alpha at step 100 vs step 10000?
- [ ] Does α decrease/increase/stabilize?

**Generate plots:**
```bash
python analysis/plotting.py --log_dir outputs/exp_1_2_real_gradients/logs \
                            --output_dir outputs/exp_1_2_real_gradients/plots \
                            --experiment 1.2
```

---

### ✅ 3. Reproducibility Check

To verify reproducibility:

1. **Save experiment configs:**
   - Config files are already version-controlled
   - Log files contain full config

2. **Document system info:**
   - Check first lines of experiment output
   - Should show Python version, PyTorch version, CUDA, git commit

3. **Rerun with same seed:**
   ```bash
   # Should get identical results (within numerical precision)
   python experiments/measure_alpha.py --config config/experiment_1_1.yaml
   ```

4. **Test different seeds:**
   ```yaml
   # In config, change:
   experiment:
     seed: 43  # Different seed
   ```
   - Results should be similar but not identical
   - Alpha estimates should be within ~10% of original

---

## Troubleshooting Common Issues

### Issue: NaN in Alpha Estimates

**Symptoms:** `alpha/hill` or `alpha/pickands` shows NaN

**Causes:**
1. Gradients are all zero (model not learning)
2. Too few gradient samples (very small batch/width)
3. Numerical instability

**Fixes:**
- Check model is actually training (`loss` should change)
- Increase batch size
- Check gradient norms are reasonable (not too large/small)
- Try different k_ratio values

---

### Issue: Memory Error

**Symptoms:** `RuntimeError: CUDA out of memory`

**Fixes:**
- Reduce batch size in config
- Use smaller model widths
- Run on CPU (slower but works)
- Clear CUDA cache: `torch.cuda.empty_cache()`

---

### Issue: Slow Experiments

**Symptoms:** Taking much longer than expected

**Causes:**
- Running on CPU instead of GPU
- ML estimator is slow for large gradients
- Too many alpha measurements

**Fixes:**
- Verify GPU is being used: check console output
- Disable ML estimator, use only Hill/Pickands
- Increase `alpha_interval` (measure less frequently)
- Reduce number of steps for initial tests

---

### Issue: Plots Not Generated

**Symptoms:** No plots in outputs directory

**Checks:**
- Is `save_plots: true` in logging config?
- Check console for plotting errors
- Verify matplotlib installed

**Manual plot generation:**
```bash
python analysis/plotting.py --log_dir <log_dir> --output_dir <output_dir> --experiment 1.1
```

---

## Final Pre-Submission Checklist

Before considering Phase 1 complete:

### Scientific Requirements
- [ ] Both experiments run successfully
- [ ] Alpha estimates obtained for all conditions
- [ ] Results documented (even if negative)
- [ ] Plots generated and interpretable
- [ ] Key findings summarized

### Technical Requirements
- [ ] All tests pass
- [ ] Code documented
- [ ] Configs version-controlled
- [ ] Results saved and backed up
- [ ] Reproducibility verified

### Documentation
- [ ] README.md accurate
- [ ] IMPLEMENTATION_DECISIONS.md reviewed
- [ ] Results documented in notebook or report
- [ ] Any deviations from plan noted

---

## Success Criteria Summary

### Minimum Success (Phase 1 Valid)
- ✅ Both experiments run to completion
- ✅ Alpha measurements obtained (even if α ≈ 2)
- ✅ No major bugs or crashes
- ✅ Results reproducible

### Good Success
- ✅ α < 2 observed consistently
- ✅ Clear differences between estimators understood
- ✅ Layer-wise patterns identified
- ✅ Preliminary insights for Phase 2

### Excellent Success
- ✅ Strong evidence for heavy-tails (α ∈ [1.2, 1.8])
- ✅ Width/clipping effects quantified
- ✅ Layer-specific patterns clear
- ✅ Ready to proceed to scaling law derivation (Phase 2)

---

## Questions for Expert Review

When submitting for expert review, prepare answers to:

1. **Estimator Agreement:**
   - How well do Hill, Pickands, and ML estimates agree?
   - Which estimator is most reliable for your data?

2. **Tail Index Values:**
   - What α values did you observe?
   - Are they consistent with heavy-tail hypothesis?

3. **Layer-wise Patterns:**
   - Do attention and FFN layers have different tail behavior?
   - Why might this be?

4. **Gradient Clipping:**
   - Does clipping increase α (as expected)?
   - Implications for training?

5. **Next Steps:**
   - Are results sufficient to proceed to Phase 2?
   - What additional experiments would be valuable?

---

**Document Version:** 1.0
**Date:** 2025-10-22
**Status:** Phase 1 Implementation Complete - Ready for Validation
