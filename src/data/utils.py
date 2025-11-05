"""
Data utilities for tokenization and dataloader creation
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, List
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoTokenizer


def tokenize_dataset(
    dataset: Dataset,
    text_column: str,
    label_column: str,
    tokenizer,
    max_length: int = 256,
    padding: str = "max_length",
    truncation: bool = True
) -> Dataset:
    """
    Tokenize a HuggingFace dataset

    Args:
        dataset: HuggingFace Dataset
        text_column: Name of text column
        label_column: Name of label column
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        padding: Padding strategy
        truncation: Whether to truncate

    Returns:
        Tokenized dataset with 'input_ids', 'attention_mask', 'labels'
    """
    def tokenize_function(examples):
        # Tokenize text
        tokenized = tokenizer(
            examples[text_column],
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=None  # Return lists, not tensors
        )

        # Add labels
        tokenized['labels'] = examples[label_column]

        return tokenized

    # Tokenize entire dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )

    # Set format for PyTorch
    tokenized_dataset.set_format(
        type='torch',
        columns=['input_ids', 'attention_mask', 'labels']
    )

    return tokenized_dataset


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0
) -> DataLoader:
    """
    Create a PyTorch DataLoader from a HuggingFace dataset

    Args:
        dataset: Tokenized HuggingFace Dataset
        batch_size: Batch size
        shuffle: Whether to shuffle
        drop_last: Whether to drop last incomplete batch
        num_workers: Number of workers for data loading

    Returns:
        PyTorch DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=False  # Don't use pin_memory for TPU
    )


def save_split_indices(indices: np.ndarray, filepath: str) -> None:
    """Save dataset split indices to file"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(filepath, indices, fmt='%d')


def load_split_indices(filepath: str) -> np.ndarray:
    """Load dataset split indices from file"""
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Split file not found: {filepath}")
    return np.loadtxt(filepath, dtype=int)


class DataCollator:
    """
    Custom data collator for batching tokenized data
    Handles padding and tensor conversion
    """

    def __init__(self, tokenizer, padding: str = "longest", max_length: Optional[int] = None):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length

    def __call__(self, features: List[dict]) -> dict:
        """
        Collate a batch of features

        Args:
            features: List of dicts with 'input_ids', 'attention_mask', 'labels'

        Returns:
            Batched dict with tensors
        """
        # Separate labels from input features
        labels = [f['labels'] for f in features]

        # Pad input sequences
        batch = self.tokenizer.pad(
            [{'input_ids': f['input_ids'], 'attention_mask': f['attention_mask']}
             for f in features],
            padding=self.padding,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Add labels back
        batch['labels'] = torch.tensor(labels, dtype=torch.long)

        return batch


def get_tokenizer(model_name: str = "Qwen/Qwen2-0.5B", cache_dir: Optional[str] = None):
    """
    Load Qwen2 tokenizer

    Args:
        model_name: HuggingFace model name
        cache_dir: Cache directory for tokenizer

    Returns:
        HuggingFace tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True
    )

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def verify_dataset_integrity(
    head_init_dataset: Dataset,
    train_dataset: Dataset,
    val_dataset: Dataset,
    config
) -> None:
    """
    Verify dataset splits are valid

    Args:
        head_init_dataset: Head initialization split
        train_dataset: Training split
        val_dataset: Validation split
        config: DatasetConfig

    Raises:
        AssertionError if validation fails
    """
    print("\n  Dataset integrity checks:")

    # Check sizes
    assert len(head_init_dataset) > 0, "head_init dataset is empty"
    assert len(train_dataset) > 0, "train dataset is empty"
    assert len(val_dataset) > 0, "val dataset is empty"
    print(f"    ✓ All splits have samples")

    # Check required columns
    for split_name, split in [("head_init", head_init_dataset),
                               ("train", train_dataset),
                               ("val", val_dataset)]:
        if 'input_ids' in split.column_names:
            # Tokenized dataset
            required = ['input_ids', 'attention_mask', 'labels']
        else:
            # Raw dataset
            required = [config.text_column, config.label_column]

        for col in required:
            assert col in split.column_names, f"{split_name} missing column: {col}"

    print(f"    ✓ All required columns present")

    # Check label range
    for split_name, split in [("head_init", head_init_dataset),
                               ("train", train_dataset),
                               ("val", val_dataset)]:
        label_col = 'labels' if 'labels' in split.column_names else config.label_column
        labels = split[label_col]

        min_label = min(labels)
        max_label = max(labels)

        assert min_label >= 0, f"{split_name} has negative labels: {min_label}"
        assert max_label < config.num_labels, \
            f"{split_name} has label {max_label} >= num_labels {config.num_labels}"

    print(f"    ✓ All labels in valid range [0, {config.num_labels})")

    print("\n  ✓ Dataset integrity verified")


if __name__ == "__main__":
    # Test tokenization
    from .loader import load_and_split_dataset

    print("Testing tokenization...")

    head_init, train, val, config = load_and_split_dataset(
        "sst2",
        config_dir="../../config/datasets",
        output_dir="../../outputs/splits"
    )

    tokenizer = get_tokenizer()

    print(f"\nTokenizer info:")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"  EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")

    # Tokenize a single example
    example_text = head_init[0][config.text_column]
    print(f"\nExample text: {example_text}")

    tokens = tokenizer(example_text, max_length=256, padding='max_length', truncation=True)
    print(f"Tokenized length: {len(tokens['input_ids'])}")
    print(f"First 10 token IDs: {tokens['input_ids'][:10]}")

    # Tokenize full dataset
    tokenized = tokenize_dataset(head_init, config.text_column, config.label_column, tokenizer)
    print(f"\nTokenized dataset columns: {tokenized.column_names}")
    print(f"Sample shape: input_ids={tokenized[0]['input_ids'].shape}")

    verify_dataset_integrity(tokenized, tokenized, val, config)
    print("\n✓ All tests passed!")
