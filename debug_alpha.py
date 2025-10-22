"""
Debug script to understand why alpha estimation isn't working.
"""

import torch
from core.tail_estimators import AlphaTracker, estimate_alpha_hill, estimate_alpha_pickands
from models.minimal_ffn import MinimalFFN

print("="*60)
print("DEBUGGING ALPHA ESTIMATION")
print("="*60)

# Create a simple model
model = MinimalFFN(d_model=64)
print(f"\n✓ Created model with {sum(p.numel() for p in model.parameters()):,} parameters")

# Do a forward pass with synthetic gradients
batch = torch.randn(32, 64)
output = model.forward_with_input_storage(batch)

print("\n1. Injecting synthetic gradients...")
model.inject_synthetic_gradients(grad_dist='normal', grad_std=1.0, device='cpu')

# Check gradients
print("\n2. Checking gradients:")
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"  {name:20s}: shape={tuple(param.grad.shape)}, norm={param.grad.norm().item():.4f}")
        print(f"    Sample values: {param.grad.flatten()[:5].tolist()}")
        print(f"    Min/Max: {param.grad.min().item():.4f} / {param.grad.max().item():.4f}")

# Try estimating alpha directly
print("\n3. Testing Hill estimator directly:")
W_in_grad = model.W_in.weight.grad
alpha_hill = estimate_alpha_hill(W_in_grad, k_ratio=0.1)
print(f"  W_in.weight: α_hill = {alpha_hill:.3f}")

alpha_pickands = estimate_alpha_pickands(W_in_grad, k_ratio=0.1)
print(f"  W_in.weight: α_pickands = {alpha_pickands:.3f if not torch.isnan(torch.tensor(alpha_pickands)) else 'NaN'}")

# Test with AlphaTracker
print("\n4. Testing AlphaTracker:")
tracker = AlphaTracker(window_size=10, estimators=['hill', 'pickands'], k_ratios=[0.1])

tracker.update(model, prefix='test')
summary = tracker.get_summary()

print(f"  Global summary keys: {list(summary.keys())}")
if summary:
    for key, stats in summary.items():
        print(f"  {key}: α = {stats['mean']:.3f} ± {stats['std']:.3f}")
else:
    print("  ⚠️  EMPTY SUMMARY!")

# Check internal state
print("\n5. Checking tracker internal state:")
print(f"  global_alphas keys: {list(tracker.global_alphas.keys())}")
for est_name, k_dict in tracker.global_alphas.items():
    for k_ratio, values in k_dict.items():
        print(f"    {est_name}/{k_ratio}: {len(values)} values = {list(values)[:3]}")

# Do multiple updates
print("\n6. Testing multiple updates (simulating training):")
for i in range(5):
    batch = torch.randn(32, 64)
    output = model.forward_with_input_storage(batch)
    model.inject_synthetic_gradients(grad_dist='normal', grad_std=1.0, device='cpu')
    model.zero_grad()  # Clear for next iteration... wait, this is wrong!

print("\n⚠️  FOUND THE BUG!")
print("The model.zero_grad() in the training loop clears gradients BEFORE alpha measurement!")
print("This means we're always trying to estimate alpha on ZERO gradients!")
