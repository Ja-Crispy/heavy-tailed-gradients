"""
Synthetic data generation utilities for experiments.

Supports:
- Experiment 1.1: Synthetic input generation (x ~ N(0, I))
- Experiment 1.2: Synthetic token sequences for next-token prediction
"""

from typing import Tuple, Optional, Iterator
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SyntheticInputDataset(Dataset):
    """
    Synthetic input dataset for Experiment 1.1.

    Generates random inputs from specified distribution:
    - x ~ N(0, σ²I) (default)
    - Also supports: uniform, cauchy, laplace
    """

    def __init__(self, d_model: int, num_samples: int,
                 distribution: str = 'normal',
                 scale: float = 1.0,
                 seed: Optional[int] = None):
        """
        Initialize synthetic input dataset.

        Args:
            d_model: Input dimension
            num_samples: Number of samples to generate
            distribution: Distribution type ('normal', 'uniform', 'cauchy', 'laplace')
            scale: Scale parameter (std for normal, half-width for uniform)
            seed: Random seed for reproducibility
        """
        self.d_model = d_model
        self.num_samples = num_samples
        self.distribution = distribution
        self.scale = scale

        # Set seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Pre-generate all data
        self.data = self._generate_data()

    def _generate_data(self) -> torch.Tensor:
        """Generate synthetic data according to specified distribution."""
        if self.distribution == 'normal':
            data = torch.randn(self.num_samples, self.d_model) * self.scale

        elif self.distribution == 'uniform':
            data = (torch.rand(self.num_samples, self.d_model) - 0.5) * 2 * self.scale

        elif self.distribution == 'cauchy':
            data = torch.empty(self.num_samples, self.d_model).cauchy_() * self.scale

        elif self.distribution == 'laplace':
            dist = torch.distributions.Laplace(0, self.scale)
            data = dist.sample((self.num_samples, self.d_model))

        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

        return data

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


class SyntheticTokenDataset(Dataset):
    """
    Synthetic token sequence dataset for Experiment 1.2.

    Generates random token sequences for next-token prediction:
    - Random token IDs from [0, vocab_size)
    - Simple pattern-based sequences (optional)
    """

    def __init__(self, vocab_size: int, seq_length: int, num_sequences: int,
                 pattern: str = 'random',
                 seed: Optional[int] = None):
        """
        Initialize synthetic token dataset.

        Args:
            vocab_size: Size of vocabulary
            seq_length: Sequence length
            num_sequences: Number of sequences to generate
            pattern: Generation pattern ('random', 'repeated', 'arithmetic')
            seed: Random seed
        """
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.num_sequences = num_sequences
        self.pattern = pattern

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Pre-generate sequences
        self.sequences = self._generate_sequences()

    def _generate_sequences(self) -> torch.Tensor:
        """Generate synthetic token sequences."""
        if self.pattern == 'random':
            # Purely random tokens
            sequences = torch.randint(
                0, self.vocab_size,
                (self.num_sequences, self.seq_length)
            )

        elif self.pattern == 'repeated':
            # Repeated patterns (for easier learning)
            sequences = []
            for _ in range(self.num_sequences):
                # Generate pattern of length 4, repeat it
                pattern = torch.randint(0, self.vocab_size, (4,))
                seq = pattern.repeat(self.seq_length // 4 + 1)[:self.seq_length]
                sequences.append(seq)
            sequences = torch.stack(sequences)

        elif self.pattern == 'arithmetic':
            # Arithmetic sequences (token_i = (token_0 + i) % vocab_size)
            sequences = []
            for _ in range(self.num_sequences):
                start = torch.randint(0, self.vocab_size, (1,)).item()
                seq = torch.tensor([(start + i) % self.vocab_size
                                   for i in range(self.seq_length)])
                sequences.append(seq)
            sequences = torch.stack(sequences)

        else:
            raise ValueError(f"Unknown pattern: {self.pattern}")

        return sequences

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get input and target for next-token prediction.

        Returns:
            input_ids: Tokens [0:seq_length-1]
            target_ids: Tokens [1:seq_length] (shifted by 1)
        """
        seq = self.sequences[idx]
        input_ids = seq[:-1]  # All but last
        target_ids = seq[1:]  # All but first (shifted by 1)

        return input_ids, target_ids


def create_synthetic_dataloader(d_model: int, num_samples: int, batch_size: int,
                                distribution: str = 'normal',
                                scale: float = 1.0,
                                seed: Optional[int] = None,
                                shuffle: bool = True,
                                num_workers: int = 0) -> DataLoader:
    """
    Create DataLoader for synthetic inputs (Experiment 1.1).

    Args:
        d_model: Input dimension
        num_samples: Total number of samples
        batch_size: Batch size
        distribution: Distribution type
        scale: Scale parameter
        seed: Random seed
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes

    Returns:
        DataLoader yielding batches of synthetic inputs
    """
    dataset = SyntheticInputDataset(
        d_model=d_model,
        num_samples=num_samples,
        distribution=distribution,
        scale=scale,
        seed=seed
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return loader


def create_token_dataloader(vocab_size: int, seq_length: int,
                            num_sequences: int, batch_size: int,
                            pattern: str = 'random',
                            seed: Optional[int] = None,
                            shuffle: bool = True,
                            num_workers: int = 0) -> DataLoader:
    """
    Create DataLoader for synthetic token sequences (Experiment 1.2).

    Args:
        vocab_size: Vocabulary size
        seq_length: Sequence length
        num_sequences: Total sequences
        batch_size: Batch size
        pattern: Generation pattern
        seed: Random seed
        shuffle: Whether to shuffle
        num_workers: Number of workers

    Returns:
        DataLoader yielding (input_ids, target_ids) batches
    """
    dataset = SyntheticTokenDataset(
        vocab_size=vocab_size,
        seq_length=seq_length,
        num_sequences=num_sequences,
        pattern=pattern,
        seed=seed
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return loader


def infinite_dataloader(dataloader: DataLoader) -> Iterator:
    """
    Create infinite iterator over dataloader.

    Useful for training with a fixed number of steps rather than epochs.

    Args:
        dataloader: DataLoader to iterate over

    Yields:
        Batches from dataloader indefinitely
    """
    while True:
        for batch in dataloader:
            yield batch


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make CUDA operations deterministic (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    print("Testing synthetic data generation...")

    # Test synthetic inputs (Exp 1.1)
    print("\n=== Experiment 1.1: Synthetic Inputs ===")
    loader = create_synthetic_dataloader(
        d_model=128,
        num_samples=1000,
        batch_size=32,
        distribution='normal',
        scale=1.0,
        seed=42
    )

    batch = next(iter(loader))
    print(f"✓ Synthetic input batch shape: {batch.shape}")
    print(f"  Mean: {batch.mean().item():.4f}, Std: {batch.std().item():.4f}")

    # Test different distributions
    for dist in ['uniform', 'cauchy', 'laplace']:
        loader = create_synthetic_dataloader(
            d_model=64, num_samples=100, batch_size=10,
            distribution=dist, seed=42
        )
        batch = next(iter(loader))
        print(f"✓ Distribution '{dist}': shape {batch.shape}, range [{batch.min():.2f}, {batch.max():.2f}]")

    # Test synthetic tokens (Exp 1.2)
    print("\n=== Experiment 1.2: Synthetic Tokens ===")
    loader = create_token_dataloader(
        vocab_size=1000,
        seq_length=128,
        num_sequences=500,
        batch_size=16,
        pattern='random',
        seed=42
    )

    input_ids, target_ids = next(iter(loader))
    print(f"✓ Token batch: input shape {input_ids.shape}, target shape {target_ids.shape}")
    print(f"  Input range: [{input_ids.min()}, {input_ids.max()}]")
    print(f"  Target range: [{target_ids.min()}, {target_ids.max()}]")

    # Verify shift
    print(f"  First input: {input_ids[0, :5].tolist()}")
    print(f"  First target: {target_ids[0, :5].tolist()}")
    print(f"  Shift correct: {torch.equal(input_ids[0, 1:], target_ids[0, :-1])}")

    # Test patterns
    for pattern in ['repeated', 'arithmetic']:
        loader = create_token_dataloader(
            vocab_size=100, seq_length=32, num_sequences=10,
            batch_size=2, pattern=pattern, seed=42
        )
        inp, tgt = next(iter(loader))
        print(f"✓ Pattern '{pattern}': {inp[0, :8].tolist()}")

    # Test infinite loader
    print("\n=== Infinite DataLoader ===")
    loader = create_synthetic_dataloader(d_model=64, num_samples=50, batch_size=10, seed=42)
    infinite_iter = infinite_dataloader(loader)

    for i in range(3):
        batch = next(infinite_iter)
        print(f"  Batch {i}: shape {batch.shape}")

    print("\n✓ All synthetic data tests passed!")
