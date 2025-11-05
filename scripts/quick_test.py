#!/usr/bin/env python3
"""
Quick Test Script

Runs a quick test of the entire pipeline on a small subset of data.
Useful for debugging and sanity checking before running full experiments.

Usage:
    python scripts/quick_test.py
    python scripts/quick_test.py --dataset agnews --samples 50
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.data.loader import load_and_split_dataset
from src.data.utils import tokenize_dataset, get_tokenizer
from src.training.baseline_trainer import run_baseline_arm
from src.training.influence_trainer import run_influence_arm
from src.training.lora_trainer import run_lora_arm
from src.training.fullft_trainer import run_fullft_arm


def quick_test(
    dataset_name: str = "sst2",
    train_samples: int = 50,
    val_samples: int = 50,
    batch_size: int = 16,
    max_epochs: int = 2
):
    """
    Run quick test on small data subset

    Args:
        dataset_name: Dataset to test
        train_samples: Number of training samples
        val_samples: Number of validation samples
        batch_size: Batch size
        max_epochs: Max epochs for training arms
    """
    print("\n" + "=" * 80)
    print("QUICK TEST")
    print("=" * 80)
    print(f"Dataset: {dataset_name}")
    print(f"Train samples: {train_samples}")
    print(f"Val samples: {val_samples}")
    print("=" * 80)

    # Check if baseline checkpoint exists
    checkpoint_path = f"outputs/checkpoints/{dataset_name}/baseline_start"

    if not Path(checkpoint_path).exists():
        print(f"\nERROR: Checkpoint not found: {checkpoint_path}")
        print("Please run scripts/0_create_baselines.py first")
        print(f"\nQuick command:")
        print(f"  python scripts/0_create_baselines.py --datasets {dataset_name}")
        return 1

    # Load data
    print("\nLoading data...")
    _, train, val, config = load_and_split_dataset(
        dataset_name=dataset_name,
        config_dir="config/datasets",
        output_dir="outputs/splits"
    )

    # Take small subsets
    train_small = train.select(range(min(train_samples, len(train))))
    val_small = val.select(range(min(val_samples, len(val))))

    print(f"  Train subset: {len(train_small)} samples")
    print(f"  Val subset: {len(val_small)} samples")

    # Tokenize
    print("\nTokenizing...")
    tokenizer = get_tokenizer()
    train_tokenized = tokenize_dataset(
        train_small,
        config.text_column,
        config.label_column,
        tokenizer
    )
    val_tokenized = tokenize_dataset(
        val_small,
        config.text_column,
        config.label_column,
        tokenizer
    )

    # Test all arms
    results = {}

    # Arm A: Baseline
    print("\n" + "=" * 80)
    print("Testing ARM A: Baseline")
    print("=" * 80)

    try:
        results['baseline'] = run_baseline_arm(
            checkpoint_path=checkpoint_path,
            val_dataset=val_tokenized,
            num_labels=config.num_labels,
            batch_size=batch_size,
            use_tpu=False,
            model_dtype="float32"
        )
        print("ARM A: PASSED")
    except Exception as e:
        print(f"ARM A: FAILED - {e}")
        import traceback
        traceback.print_exc()

    # Arm B: Influence
    print("\n" + "=" * 80)
    print("Testing ARM B: Influence")
    print("=" * 80)

    try:
        results['influence'] = run_influence_arm(
            checkpoint_path=checkpoint_path,
            train_dataset=train_tokenized,
            val_dataset=val_tokenized,
            num_labels=config.num_labels,
            batch_size=batch_size,
            mask_fraction=0.05,
            use_tpu=False,
            model_dtype="float32"
        )
        print("ARM B: PASSED")
    except Exception as e:
        print(f"ARM B: FAILED - {e}")
        import traceback
        traceback.print_exc()

    # Arm C: LoRA
    print("\n" + "=" * 80)
    print("Testing ARM C: LoRA")
    print("=" * 80)

    try:
        results['lora'] = run_lora_arm(
            checkpoint_path=checkpoint_path,
            train_dataset=train_tokenized,
            val_dataset=val_tokenized,
            num_labels=config.num_labels,
            batch_size=batch_size,
            learning_rate=3e-4,
            max_epochs=max_epochs,
            early_stop_patience=1,
            lora_rank=8,
            lora_alpha=16,
            use_tpu=False,
            model_dtype="float32"
        )
        print("ARM C: PASSED")
    except Exception as e:
        print(f"ARM C: FAILED - {e}")
        import traceback
        traceback.print_exc()

    # Arm D: Full FT
    print("\n" + "=" * 80)
    print("Testing ARM D: Full Fine-tuning")
    print("=" * 80)

    try:
        results['fullft'] = run_fullft_arm(
            checkpoint_path=checkpoint_path,
            train_dataset=train_tokenized,
            val_dataset=val_tokenized,
            num_labels=config.num_labels,
            batch_size=batch_size,
            learning_rate=1e-5,
            max_epochs=max_epochs,
            early_stop_patience=1,
            use_tpu=False,
            model_dtype="float32"
        )
        print("ARM D: PASSED")
    except Exception as e:
        print(f"ARM D: FAILED - {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("QUICK TEST SUMMARY")
    print("=" * 80)

    for arm_name in ['baseline', 'influence', 'lora', 'fullft']:
        if arm_name in results:
            acc = results[arm_name]['val_accuracy']
            status = "PASSED"
            print(f"  {arm_name:12s}: {status:8s} (accuracy: {acc:.2%})")
        else:
            print(f"  {arm_name:12s}: FAILED")

    num_passed = len(results)
    print(f"\n{num_passed}/4 arms passed")

    if num_passed == 4:
        print("\nAll arms working! Ready for full experiments.")
        return 0
    else:
        print(f"\n{4 - num_passed} arms failed. Please debug before running full experiments.")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Quick test of experiment pipeline"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="sst2",
        choices=["sst2", "agnews", "dbpedia", "yelp"],
        help="Dataset to test"
    )
    parser.add_argument(
        "--train_samples",
        type=int,
        default=50,
        help="Number of training samples"
    )
    parser.add_argument(
        "--val_samples",
        type=int,
        default=50,
        help="Number of validation samples"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size"
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=2,
        help="Max epochs for training arms"
    )

    args = parser.parse_args()

    exit_code = quick_test(
        dataset_name=args.dataset,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs
    )

    return exit_code


if __name__ == "__main__":
    exit(main())
