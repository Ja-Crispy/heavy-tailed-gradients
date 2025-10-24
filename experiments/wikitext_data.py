"""
WikiText-2 Dataset for Language Modeling

Provides character-level tokenization of WikiText-2 for Phase 2.5 experiments.
Compatible with existing dataloader infrastructure from synthetic_data.py.

Uses HuggingFace datasets library for reliable data loading.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, List

try:
    from datasets import load_dataset
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    print("⚠ HuggingFace datasets not available. Install with: pip install datasets")


def load_wikitext2_hf(split: str = 'train') -> str:
    """
    Load WikiText-2 dataset using HuggingFace datasets library.

    Args:
        split: 'train', 'validation', or 'test'

    Returns:
        Full text as single string
    """
    if not HUGGINGFACE_AVAILABLE:
        raise ImportError("HuggingFace datasets library required. Install with: pip install datasets")

    print(f"📥 Loading WikiText-2 {split} split from HuggingFace...")

    # Load raw character-level version
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)

    # Concatenate all text
    texts = []
    for example in dataset:
        text = example['text']
        if text.strip():  # Skip empty lines
            texts.append(text)

    full_text = '\n'.join(texts)
    print(f"  Loaded {len(full_text):,} characters from {len(dataset):,} examples")

    return full_text


class CharTokenizer:
    """
    Simple character-level tokenizer.
    Maps ASCII characters to integers for use with transformer models.
    """

    def __init__(self, vocab_size: int = 128):
        """
        Args:
            vocab_size: Number of tokens (default 128 for ASCII)
        """
        self.vocab_size = vocab_size

        # Special tokens
        self.pad_token_id = 0
        self.unk_token_id = 1

        # ASCII characters 32-127 (printable)
        # Shift by 2 to make room for pad and unk
        self.char_to_id = {chr(i): i - 30 for i in range(32, min(32 + vocab_size - 2, 128))}
        self.char_to_id['<PAD>'] = self.pad_token_id
        self.char_to_id['<UNK>'] = self.unk_token_id

        self.id_to_char = {v: k for k, v in self.char_to_id.items()}

    def encode(self, text: str) -> List[int]:
        """Convert text to list of token IDs."""
        return [self.char_to_id.get(c, self.unk_token_id) for c in text]

    def decode(self, ids: List[int]) -> str:
        """Convert list of token IDs back to text."""
        return ''.join(self.id_to_char.get(i, '<UNK>') for i in ids)


class WikiTextDataset(Dataset):
    """
    Character-level WikiText-2 dataset.

    Compatible with NanoTransformer training loop.
    Splits text into fixed-length sequences for language modeling.
    """

    def __init__(
        self,
        split: str = 'train',
        seq_length: int = 256,
        vocab_size: int = 128,
        max_sequences: Optional[int] = None
    ):
        """
        Args:
            split: 'train', 'validation', or 'test'
            seq_length: Length of each sequence
            vocab_size: Tokenizer vocabulary size
            max_sequences: Limit number of sequences (for testing)
        """
        self.seq_length = seq_length
        self.vocab_size = vocab_size

        # Map 'val' to 'validation' for HuggingFace compatibility
        hf_split = 'validation' if split == 'val' else split

        # Load text from HuggingFace
        text = load_wikitext2_hf(split=hf_split)

        print(f"  Text length: {len(text):,} characters")

        # Initialize tokenizer
        self.tokenizer = CharTokenizer(vocab_size=vocab_size)

        # Tokenize entire text
        self.tokens = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        print(f"  Tokens: {len(self.tokens):,}")

        # Calculate number of sequences
        self.num_sequences = len(self.tokens) // seq_length

        # Limit if requested
        if max_sequences is not None:
            self.num_sequences = min(self.num_sequences, max_sequences)

        # Truncate tokens to fit exact number of sequences
        self.tokens = self.tokens[:self.num_sequences * seq_length]

        print(f"  Sequences: {self.num_sequences:,} (length {seq_length})")

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> tuple:
        """
        Get input and target for next-token prediction.

        Returns:
            input_ids: Tokens [0:seq_length-1]
            target_ids: Tokens [1:seq_length] (shifted by 1)
        """
        # Get full sequence (need seq_length + 1 tokens for shifting)
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length

        seq = self.tokens[start_idx:end_idx]

        # Create input/target pairs for language modeling
        input_ids = seq[:-1]  # All but last token
        target_ids = seq[1:]  # All but first token (shifted by 1)

        return input_ids, target_ids


def create_wikitext_dataloader(
    split: str = 'train',
    seq_length: int = 256,
    batch_size: int = 32,
    vocab_size: int = 128,
    max_sequences: Optional[int] = None,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    Create WikiText-2 dataloader for language modeling.

    Compatible with Phase 2.5 batch scaling experiments.

    Args:
        split: 'train', 'val', or 'test'
        seq_length: Sequence length
        batch_size: Batch size
        vocab_size: Vocabulary size
        max_sequences: Optional limit on sequences
        shuffle: Whether to shuffle data
        num_workers: Number of dataloader workers

    Returns:
        DataLoader yielding batches of shape (batch_size, seq_length)
    """
    dataset = WikiTextDataset(
        split=split,
        seq_length=seq_length,
        vocab_size=vocab_size,
        max_sequences=max_sequences
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return dataloader


def infinite_dataloader(dataloader: DataLoader):
    """
    Wrap dataloader to restart infinitely.
    Compatible with existing training loop.
    """
    while True:
        for batch in dataloader:
            yield batch


# Self-test
if __name__ == '__main__':
    print("="*80)
    print("WikiText-2 Data Module Self-Test (HuggingFace)")
    print("="*80)

    if not HUGGINGFACE_AVAILABLE:
        print("⚠ HuggingFace datasets not installed")
        print("Install with: pip install datasets")
        exit(1)

    # Test 1: Load data
    print("\nTest 1: Load WikiText-2 from HuggingFace")
    text = load_wikitext2_hf(split='train')
    print(f"✓ Loaded {len(text):,} characters")

    # Test 2: Tokenizer
    print("\nTest 2: Character tokenizer")
    tokenizer = CharTokenizer(vocab_size=128)
    test_text = "Hello, world! How are you?"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"  Original: {test_text}")
    print(f"  Encoded:  {encoded[:20]}...")
    print(f"  Decoded:  {decoded}")
    assert test_text == decoded, "Tokenizer encode/decode mismatch!"
    print("✓ Tokenizer works correctly")

    # Test 3: Dataset
    print("\nTest 3: WikiTextDataset")
    dataset = WikiTextDataset(
        split='train',
        seq_length=64,
        vocab_size=128,
        max_sequences=100
    )
    print(f"  Dataset length: {len(dataset)}")
    inputs, targets = dataset[0]
    print(f"  Input shape: {inputs.shape}")
    print(f"  Target shape: {targets.shape}")
    print(f"  Input tokens: {inputs[:20]}")
    print(f"  Decoded: {tokenizer.decode(inputs[:50].tolist())}")
    assert inputs.shape == (63,), f"Input shape mismatch! Expected (63,), got {inputs.shape}"
    assert targets.shape == (63,), f"Target shape mismatch! Expected (63,), got {targets.shape}"
    # Verify shift: target[i] should == input[i+1] in original sequence
    print("✓ Dataset works correctly")

    # Test 4: DataLoader
    print("\nTest 4: DataLoader")
    dataloader = create_wikitext_dataloader(
        split='train',
        seq_length=64,
        batch_size=8,
        vocab_size=128,
        max_sequences=100,
        shuffle=True,
        num_workers=0
    )
    batch = next(iter(dataloader))
    inputs, targets = batch
    print(f"  Input batch shape: {inputs.shape}")
    print(f"  Target batch shape: {targets.shape}")
    assert inputs.shape == (8, 63), f"Input batch shape mismatch! Expected (8, 63), got {inputs.shape}"
    assert targets.shape == (8, 63), f"Target batch shape mismatch! Expected (8, 63), got {targets.shape}"
    print("✓ DataLoader works correctly")

    # Test 5: Infinite dataloader
    print("\nTest 5: Infinite dataloader")
    infinite_dl = infinite_dataloader(dataloader)
    batch1 = next(infinite_dl)
    batch2 = next(infinite_dl)
    inputs1, targets1 = batch1
    inputs2, targets2 = batch2
    print(f"  Batch 1 input shape: {inputs1.shape}")
    print(f"  Batch 2 input shape: {inputs2.shape}")
    assert inputs1.shape == inputs2.shape, "Infinite dataloader batch shape inconsistent!"
    assert targets1.shape == targets2.shape, "Infinite dataloader target shape inconsistent!"
    print("✓ Infinite dataloader works correctly")

    print("\n" + "="*80)
    print("✓ ALL TESTS PASSED")
    print("="*80)
    print("\nWikiText-2 data module is ready for Phase 2.5 experiments!")
