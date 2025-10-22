# Heavy-Tailed Gradient Scaling Laws Research

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Research into optimal hyperparameter scaling laws considering heavy-tailed gradient distributions, building on μP and recent weight decay scaling discoveries.

## 🎯 Research Hypothesis

**Central Question**: Does gradient noise in neural networks follow α-stable distributions with α < 2 (heavy-tailed) rather than Gaussian, and if so, how does this change optimal hyperparameter scaling laws?

**Key Implications**:
- Standard theory: Learning rate ∝ Batch^(-0.5) (assumes Gaussian noise)
- Heavy-tail theory: Learning rate ∝ Batch^(1/α - 1) (if α < 2)
- For α = 1.5: LR ∝ Batch^(-0.33) (weaker than sqrt scaling!)

## 📋 Phase 1: Establish Heavy-Tail Phenomenon

This implementation covers **Phase 1** of the research plan:

### Experiment 1.1: Synthetic Gradient Model
- **Purpose**: Test if heavy tails are architectural (not data-driven)
- **Setup**: 2-layer FFN with synthetic gradient injection
- **Widths**: d ∈ [64, 128, 256, 512]
- **Measurements**: α for ∇W_in and ∇W_out, evolution during training

### Experiment 1.2: Minimal Real Gradient Flow
- **Purpose**: Verify phenomenon with real gradient flow
- **Setup**: 4-layer transformer on synthetic token sequences
- **Dimensions**: d_model ∈ [128, 256]
- **Measurements**: Layer-wise α for attention, FFN, and vector parameters

## 🛠️ Installation

### Requirements
- Python ≥ 3.9
- PyTorch ≥ 2.0
- CUDA (optional, for GPU acceleration)

### Setup

1. **Clone the repository**:
   ```bash
   cd "E:\Work\Projects\heavy tailed gradient scaling laws"
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   cd tail-scaling
   pip install -r requirements.txt
   ```

4. **(Optional) Configure Weights & Biases**:
   ```bash
   wandb login
   ```
   Or disable wandb in experiment configs:
   ```yaml
   logging:
     use_wandb: false
   ```

## 🚀 Quick Start

### Run Experiment 1.1 (Synthetic Gradients)

```bash
python experiments/measure_alpha.py --config config/experiment_1_1.yaml
```

This will:
- Train 2-layer FFNs with widths [64, 128, 256, 512]
- Inject synthetic gradients ∇_y L ~ N(0, I)
- Measure tail index α using Hill, Pickands, and ML estimators
- Test effect of gradient clipping on α
- Generate plots and save logs

**Expected Runtime**: ~30 minutes (all widths)

### Run Experiment 1.2 (Real Gradients)

```bash
python experiments/measure_alpha.py --config config/experiment_1_2.yaml
```

This will:
- Train 4-layer transformers (d_model = 128, 256)
- Measure layer-wise α for attention/FFN/vector parameters
- Track α evolution at steps [100, 1000, 10000]
- Compare early vs steady-state behavior

**Expected Runtime**: ~1-2 hours (both model sizes)

## 📂 Directory Structure

```
tail-scaling/
├── config/                      # Experiment configurations
│   ├── experiment_1_1.yaml     # Synthetic gradient experiment
│   ├── experiment_1_2.yaml     # Real gradient experiment
│   └── logging_config.yaml     # Logging settings
│
├── core/                        # Core research code
│   ├── tail_estimators.py      # α estimation (Hill, Pickands, ML)
│   ├── metrics.py              # Sublayer gain, spectral analysis
│   ├── logger.py               # Flexible logging (wandb + files)
│   └── scaling_rules.py        # (Future: Phase 2+)
│
├── models/                      # Neural network models
│   ├── minimal_ffn.py          # 2-layer FFN for Exp 1.1
│   └── nano_transformer.py     # 4-layer transformer for Exp 1.2
│
├── experiments/                 # Experiment runners
│   ├── measure_alpha.py        # Main experiment runner
│   └── synthetic_data.py       # Synthetic data generation
│
├── analysis/                    # Visualization & analysis
│   └── plotting.py             # Plot generation utilities
│
└── outputs/                     # Generated outputs (created at runtime)
    ├── logs/                   # JSON/CSV logs
    ├── plots/                  # Generated plots
    ├── checkpoints/            # Model checkpoints
    └── wandb_local/            # Local wandb sync
```

## 📊 Configuration

Experiments are configured via YAML files. Key parameters:

### Experiment 1.1 Config (`config/experiment_1_1.yaml`)

```yaml
model:
  widths: [64, 128, 256, 512]

training:
  steps: 10000
  batch_size: 256
  lr: 0.001
  gradient_clips: [null, 1.0, 0.1]  # Test clipping effect

measurement:
  alpha_interval: 100  # Measure α every 100 steps
  estimators: ['hill', 'pickands', 'ml']
  k_ratios: [0.05, 0.1, 0.2]  # Multiple k for robustness

synthetic:
  input_dist: "normal"    # x ~ N(0, I)
  grad_dist: "normal"     # ∇_y L ~ N(0, I)
```

### Experiment 1.2 Config (`config/experiment_1_2.yaml`)

```yaml
model:
  d_models: [128, 256]
  n_layers: 4
  n_heads: 2

measurement:
  alpha_intervals: [100, 1000, 10000]  # Checkpoints
  layer_wise: true
  param_groups: ['attention', 'ffn', 'vector']
```

## 📈 Outputs

After running experiments, outputs are organized as:

```
outputs/
├── exp_1_1_synthetic_gradients/
│   ├── logs/
│   │   ├── metrics.json        # Full metrics log
│   │   └── metrics.csv         # CSV format
│   ├── plots/
│   │   ├── alpha_evolution.png # α vs training step
│   │   ├── alpha_vs_width.png  # α vs model width
│   │   └── estimator_comparison.png
│   └── checkpoints/
│       └── model_step_10000.pt
│
└── exp_1_2_real_gradients/
    ├── logs/
    ├── plots/
    │   ├── alpha_layerwise.png  # Heatmap by layer
    │   ├── attention_vs_ffn.png
    │   └── early_vs_steady.png
    └── checkpoints/
```

### Wandb Dashboards

If wandb is enabled, view real-time metrics at:
```
https://wandb.ai/<your-entity>/heavy-tail-scaling
```

Metrics include:
- `alpha/hill`, `alpha/pickands`, `alpha/ml` (per parameter)
- `alpha/ensemble_mean`, `alpha/ensemble_std`
- `loss/train`, `grad_norm`, `weight_norm`
- `spectral/singular_values`, `sublayer_gain`

## 🔬 Understanding the Results

### Success Criteria for Experiment 1.1

✅ **If α < 2 consistently**: Heavy-tail phenomenon confirmed
✅ **If α varies with width d**: Reveals scaling relationship
✅ **If clipping reduces α**: Confirms tail taming hypothesis

**What to look for**:
- `alpha_evolution.png`: Is α stable during training?
- `alpha_vs_width.png`: Does α scale with d?
- Log files: Are Hill/Pickands/ML estimates consistent?

### Success Criteria for Experiment 1.2

✅ **If α < 2 with real gradients**: Phenomenon not synthetic-only
✅ **If attention ≠ FFN tail behavior**: Layer-specific patterns
✅ **If α decreases in steady-state**: Confirms optimizer effect

**What to look for**:
- `alpha_layerwise.png`: Which layers are most heavy-tailed?
- `early_vs_steady.png`: Does α evolve during training?
- Are vector params (embeddings, norms) different from matrices?

## 🧪 Advanced Usage

### Custom Gradient Distributions (Exp 1.1)

Test different upstream gradient distributions:

```yaml
synthetic:
  grad_dist: "cauchy"  # Heavy-tailed (α = 1)
  # Options: normal, cauchy, laplace, uniform
```

### Modify Tail Estimators

Change which estimators to use:

```yaml
measurement:
  estimators: ['hill', 'pickands']  # Faster, no ML
  k_ratios: [0.1]  # Single k-ratio
```

### Adjust Measurement Frequency

For faster experiments:

```yaml
measurement:
  alpha_interval: 500  # Measure less frequently
  continuous_interval: 500
```

## 🐛 Troubleshooting

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'levy'`

**Solution**:
```bash
pip install levy
# Or remove 'ml' from estimators in config
```

### Wandb Authentication

**Problem**: `wandb.errors.UsageError: api_key not configured`

**Solution**:
```bash
wandb login
# Or set use_wandb: false in config
```

### CUDA Out of Memory

**Problem**: `RuntimeError: CUDA out of memory`

**Solution**:
- Reduce batch size in config
- Use smaller model widths
- Run on CPU (slower but works)

### NaN in Alpha Estimates

**Problem**: `α = nan` in logs

**Possible Causes**:
- Too few gradient samples (increase batch size)
- Gradients are all zero (check model/data)
- Extreme gradient magnitudes (check for numerical instability)

**Debug**:
```python
# Add to experiment code
print(f"Grad norm: {model.W_in.weight.grad.norm()}")
print(f"Grad min/max: {model.W_in.weight.grad.min()}/{model.W_in.weight.grad.max()}")
```

## 📚 Theoretical Background

### α-Stable Distributions

α-stable distributions are characterized by:
- **α ∈ (0, 2]**: Stability parameter (tail index)
  - α = 2: Gaussian
  - α = 1: Cauchy
  - α < 2: Heavy-tailed (infinite variance)

- **Key Property**: Stable under addition
  - Sum of α-stable variables is α-stable
  - Generalized Central Limit Theorem

### Batch Size Scaling

**Standard (Gaussian) Theory**:
```
Gradient = True_gradient + Noise/√B
→ Optimal LR ∝ √B
```

**Heavy-Tailed Theory**:
```
Gradient = True_gradient + Noise/B^(1/α)
→ Optimal LR ∝ B^(1/α - 1)
```

**Example** (α = 1.5):
- Standard: LR ∝ B^0.5
- Reality: LR ∝ B^(-0.33)
- **Implication**: Large batches need LOWER learning rates!

### Steady-State Scaling

According to Kosson et al. (2023):
```
||W||_steady ∝ √(η/λ) · d^0.75
```

Where:
- η: Learning rate
- λ: Weight decay
- d: Model width

**Our extension**:
```
||W||_steady ∝ √(η/λ) · d^0.75 · B^(1/2α - 1/2)
```

## 📖 References

1. **Kosson et al. (2025)** - "Robust Layerwise Scaling Rules by Proper Weight Decay Tuning"
2. **Yang et al. (2022)** - "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer" (μP)
3. **Nolan, J. P. (2020)** - "Univariate Stable Distributions: Models for Heavy Tailed Data"
4. **Hill, B. M. (1975)** - "A Simple General Approach to Inference About the Tail of a Distribution"

## 🤝 Contributing

This is a research project. For questions or discussions about the implementation:

1. Check `IMPLEMENTATION_DECISIONS.md` for detailed design rationale
2. Review experiment configs and adjust parameters
3. Examine output logs and plots for insights

## 📝 Citation

If you use this code or findings in your research, please cite:

```bibtex
@software{heavy_tail_scaling_2025,
  title = {Heavy-Tailed Gradient Scaling Laws Research},
  year = {2025},
  note = {Phase 1: Establishing the Heavy-Tail Phenomenon}
}
```

## 📄 License

MIT License - see LICENSE file for details

## 🔮 Future Work (Phases 2-4)

**Phase 2**: Batch Size Scaling
- Test η_opt vs B relationship
- Validate B^(1/α - 1) scaling

**Phase 3**: Joint Width-Batch-Decay Scaling
- Full grid search over (d, B, η, λ)
- Unified scaling law derivation

**Phase 4**: The Muon Connection
- Compare tail behavior: SGD vs AdamW vs Muon
- Test "tail taming" hypothesis

---

**Status**: Phase 1 Implementation Complete ✅

For detailed implementation decisions and expert review notes, see `IMPLEMENTATION_DECISIONS.md`.
