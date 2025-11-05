"""
Dataset Loading and Stratified Splitting

Loads datasets from HuggingFace and creates stratified splits:
- head_init: 500 samples for baseline head training
- train: 1000 samples for methods
- val: standard HF validation split
"""

import os
import yaml
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
from collections import Counter

from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split


@dataclass
class DatasetConfig:
    """Configuration for a single dataset"""
    dataset_name: str
    hf_path: str
    hf_name: Optional[str]
    num_labels: int
    text_column: str
    label_column: str
    label_names: Dict[int, str]

    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Load config from YAML file"""
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(**config)


def check_stratification(
    labels: List[int],
    split_labels: List[int],
    split_name: str,
    tolerance: float = 0.1
) -> None:
    """
    Verify stratification: each class should be within ±tolerance of dataset prior

    Args:
        labels: All labels in full dataset
        split_labels: Labels in the split
        split_name: Name for logging (e.g., "head_init", "train")
        tolerance: Allowed deviation (default: 10%)
    """
    full_dist = Counter(labels)
    split_dist = Counter(split_labels)

    total_full = len(labels)
    total_split = len(split_labels)

    print(f"\n  Stratification check for {split_name}:")
    print(f"    Total samples: {total_split}")

    for label in sorted(full_dist.keys()):
        full_ratio = full_dist[label] / total_full
        split_ratio = split_dist.get(label, 0) / total_split
        diff = abs(split_ratio - full_ratio)

        status = "✓" if diff <= tolerance else "✗"
        print(f"    Class {label}: {split_dist.get(label, 0):4d} samples "
              f"({split_ratio:.1%} vs {full_ratio:.1%} full) "
              f"diff={diff:.1%} {status}")

        if diff > tolerance:
            print(f"      WARNING: Class {label} deviation {diff:.1%} exceeds tolerance {tolerance:.1%}")


def stratified_split_indices(
    labels: np.ndarray,
    sizes: Tuple[int, int],
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create stratified splits with fixed sizes

    Args:
        labels: Array of label integers
        sizes: (head_init_size, train_size) e.g., (500, 1000)
        seed: Random seed

    Returns:
        (head_init_indices, train_indices, remaining_indices)
    """
    head_size, train_size = sizes

    # First split: head_init vs rest
    head_init_idx, rest_idx = train_test_split(
        np.arange(len(labels)),
        train_size=head_size,
        stratify=labels,
        random_state=seed
    )

    # Second split: train vs remaining
    rest_labels = labels[rest_idx]
    train_idx_local, remaining_idx_local = train_test_split(
        np.arange(len(rest_idx)),
        train_size=train_size,
        stratify=rest_labels,
        random_state=seed + 1
    )

    # Map local indices back to global
    train_idx = rest_idx[train_idx_local]
    remaining_idx = rest_idx[remaining_idx_local]

    return head_init_idx, train_idx, remaining_idx


def load_and_split_dataset(
    dataset_name: str,
    config_dir: str = "config/datasets",
    output_dir: str = "outputs/splits",
    seed: int = 42,
    head_init_samples: int = 500,
    train_samples: int = 1000,
    force_resplit: bool = False
) -> Tuple[Dataset, Dataset, Dataset, DatasetConfig]:
    """
    Load dataset and create stratified splits

    Args:
        dataset_name: Name of dataset (e.g., "sst2")
        config_dir: Directory containing dataset YAML configs
        output_dir: Directory to save split indices
        seed: Random seed for reproducibility
        head_init_samples: Number of samples for head initialization
        train_samples: Number of samples for training methods
        force_resplit: If True, ignore cached splits and recreate

    Returns:
        (head_init_dataset, train_dataset, val_dataset, config)
    """
    # Load dataset config
    config_path = Path(config_dir) / f"{dataset_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {config_path}")

    config = DatasetConfig.from_yaml(str(config_path))
    print(f"\n{'='*60}")
    print(f"Loading dataset: {config.dataset_name}")
    print(f"  HF path: {config.hf_path}")
    print(f"  Num labels: {config.num_labels}")
    print(f"  Text column: {config.text_column}")
    print(f"  Label column: {config.label_column}")
    print('='*60)

    # Load from HuggingFace
    if config.hf_name:
        dataset = load_dataset(config.hf_path, config.hf_name)
    else:
        dataset = load_dataset(config.hf_path)

    # Get train split (we'll create our own train/head_init from this)
    train_full = dataset['train']

    # Validation set is the standard HF split
    if 'validation' in dataset:
        val_dataset = dataset['validation']
    elif 'test' in dataset:
        print("  Note: Using 'test' split as validation")
        val_dataset = dataset['test']
    else:
        raise ValueError(f"No validation/test split found in {config.dataset_name}")

    # Check if we have cached splits
    split_dir = Path(output_dir) / dataset_name / f"seed{seed}"
    head_init_path = split_dir / "head500.txt"
    train_path = split_dir / "train1000.txt"

    if head_init_path.exists() and train_path.exists() and not force_resplit:
        print(f"\n  Loading cached splits from {split_dir}")
        head_init_indices = np.loadtxt(head_init_path, dtype=int)
        train_indices = np.loadtxt(train_path, dtype=int)

        # Verify indices are valid
        max_idx = len(train_full) - 1
        if head_init_indices.max() > max_idx or train_indices.max() > max_idx:
            print(f"  WARNING: Cached indices out of range, recreating splits...")
            force_resplit = True

    if not head_init_path.exists() or not train_path.exists() or force_resplit:
        print(f"\n  Creating new stratified splits...")

        # Get labels from train split
        labels = np.array(train_full[config.label_column])

        # Verify we have enough samples
        required = head_init_samples + train_samples
        if len(labels) < required:
            raise ValueError(
                f"Dataset {dataset_name} has only {len(labels)} train samples, "
                f"but need {required} ({head_init_samples} + {train_samples})"
            )

        # Create stratified splits
        head_init_indices, train_indices, _ = stratified_split_indices(
            labels,
            sizes=(head_init_samples, train_samples),
            seed=seed
        )

        # Check stratification
        check_stratification(labels, labels[head_init_indices], "head_init", tolerance=0.1)
        check_stratification(labels, labels[train_indices], "train", tolerance=0.1)

        # Check no overlap
        overlap = set(head_init_indices) & set(train_indices)
        assert len(overlap) == 0, f"Found {len(overlap)} overlapping indices!"
        print(f"\n  ✓ No overlap between head_init and train splits")

        # Save indices
        split_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(head_init_path, head_init_indices, fmt='%d')
        np.savetxt(train_path, train_indices, fmt='%d')
        print(f"\n  Saved split indices to {split_dir}")

    # Create dataset splits
    head_init_dataset = train_full.select(head_init_indices)
    train_dataset = train_full.select(train_indices)

    print(f"\n  Final split sizes:")
    print(f"    head_init: {len(head_init_dataset)} samples")
    print(f"    train:     {len(train_dataset)} samples")
    print(f"    val:       {len(val_dataset)} samples")

    return head_init_dataset, train_dataset, val_dataset, config


if __name__ == "__main__":
    # Test loading all datasets
    datasets = ["sst2", "agnews", "dbpedia", "yelp"]

    for dataset_name in datasets:
        try:
            head_init, train, val, config = load_and_split_dataset(
                dataset_name,
                config_dir="../../config/datasets",
                output_dir="../../outputs/splits"
            )
            print(f"\n✓ {dataset_name} loaded successfully")
            print(f"  Sample text: {head_init[0][config.text_column][:100]}...")
            print(f"  Sample label: {head_init[0][config.label_column]}")
        except Exception as e:
            print(f"\n✗ Failed to load {dataset_name}: {e}")
