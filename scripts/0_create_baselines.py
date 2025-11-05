#!/usr/bin/env python3
"""
Script 0: Create Baseline Checkpoints

Creates baseline_start checkpoints for all 4 datasets:
- Loads Qwen2-0.5B with frozen backbone
- Trains classifier head for 1 epoch on 500-sample head_init split
- Saves checkpoint to outputs/checkpoints/{dataset}/baseline_start/

Usage:
    python scripts/0_create_baselines.py --datasets sst2 agnews dbpedia yelp
    python scripts/0_create_baselines.py --datasets sst2 --use_tpu
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.data.loader import load_and_split_dataset
from src.data.utils import tokenize_dataset, get_tokenizer
from src.models.qwen2_wrapper import load_qwen2_model, freeze_backbone, print_model_info
from src.models.head_trainer import create_baseline_checkpoint
from src.utils.tpu_utils import setup_tpu_environment, get_device


def create_baseline_for_dataset(
    dataset_name: str,
    config_dir: str = "config/datasets",
    splits_dir: str = "outputs/splits",
    output_dir: str = "outputs/checkpoints",
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    use_tpu: bool = False,
    model_dtype: str = "bfloat16"
):
    """
    Create baseline checkpoint for a single dataset

    Args:
        dataset_name: Name of dataset (e.g., "sst2")
        config_dir: Directory containing dataset configs
        splits_dir: Directory for dataset splits
        output_dir: Output directory for checkpoints
        batch_size: Training batch size
        learning_rate: Learning rate for head training
        weight_decay: Weight decay
        use_tpu: Whether to use TPU
        model_dtype: Model precision ("bfloat16" or "float32")

    Returns:
        Path to saved checkpoint
    """
    print("\n" + "=" * 70)
    print(f"Creating baseline for: {dataset_name.upper()}")
    print("=" * 70)

    # Load dataset
    head_init, _, _, config = load_and_split_dataset(
        dataset_name=dataset_name,
        config_dir=config_dir,
        output_dir=splits_dir,
        seed=42,
        head_init_samples=500,
        train_samples=1000,
        force_resplit=False
    )

    print(f"\nLoaded {dataset_name}:")
    print(f"  Head init samples: {len(head_init)}")
    print(f"  Num labels: {config.num_labels}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = get_tokenizer()

    # Tokenize head_init split
    print("\nTokenizing head_init split...")
    tokenized_head_init = tokenize_dataset(
        dataset=head_init,
        text_column=config.text_column,
        label_column=config.label_column,
        tokenizer=tokenizer,
        max_length=256
    )

    # Load model
    print("\nLoading model...")
    dtype = torch.bfloat16 if model_dtype == "bfloat16" else torch.float32

    device = get_device(use_tpu=use_tpu)

    model = load_qwen2_model(
        num_labels=config.num_labels,
        device=None,  # We'll move to device in trainer
        dtype=dtype
    )

    # Freeze backbone
    print("\nFreezing backbone...")
    freeze_backbone(model, freeze=True, verbose=True)

    # Print model info
    print_model_info(model)

    # Train head
    print("\nTraining classifier head...")
    trained_model, save_path = create_baseline_checkpoint(
        model=model,
        tokenized_dataset=tokenized_head_init,
        output_dir=output_dir,
        dataset_name=dataset_name,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        grad_clip=1.0,
        device=device,
        use_tpu=use_tpu
    )

    print(f"\n{'='*70}")
    print(f"Baseline created successfully for {dataset_name}")
    print(f"Saved to: {save_path}")
    print('='*70)

    return save_path


def main():
    parser = argparse.ArgumentParser(
        description="Create baseline_start checkpoints for all datasets"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["sst2", "agnews", "dbpedia", "yelp"],
        choices=["sst2", "agnews", "dbpedia", "yelp"],
        help="Datasets to create baselines for"
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        default="config/datasets",
        help="Directory containing dataset configs"
    )
    parser.add_argument(
        "--splits_dir",
        type=str,
        default="outputs/splits",
        help="Directory for dataset splits"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/checkpoints",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for head training"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay"
    )
    parser.add_argument(
        "--use_tpu",
        action="store_true",
        help="Use TPU for training"
    )
    parser.add_argument(
        "--model_dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float32"],
        help="Model precision"
    )

    args = parser.parse_args()

    # Setup TPU environment if needed
    if args.use_tpu:
        setup_tpu_environment()

    # Create baselines for all datasets
    print("\n" + "=" * 70)
    print("CREATING BASELINE CHECKPOINTS")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Use TPU: {args.use_tpu}")
    print("=" * 70)

    results = {}

    for dataset_name in args.datasets:
        try:
            save_path = create_baseline_for_dataset(
                dataset_name=dataset_name,
                config_dir=args.config_dir,
                splits_dir=args.splits_dir,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                use_tpu=args.use_tpu,
                model_dtype=args.model_dtype
            )
            results[dataset_name] = {"status": "success", "path": save_path}

        except Exception as e:
            print(f"\nERROR creating baseline for {dataset_name}:")
            print(f"  {str(e)}")
            import traceback
            traceback.print_exc()
            results[dataset_name] = {"status": "failed", "error": str(e)}

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for dataset_name, result in results.items():
        if result['status'] == 'success':
            print(f"  {dataset_name:10s}: SUCCESS - {result['path']}")
        else:
            print(f"  {dataset_name:10s}: FAILED - {result['error']}")

    # Check overall success
    num_success = sum(1 for r in results.values() if r['status'] == 'success')
    num_total = len(results)

    print(f"\nCompleted: {num_success}/{num_total} datasets")

    if num_success == num_total:
        print("\nAll baselines created successfully!")
        return 0
    else:
        print(f"\nWARNING: {num_total - num_success} datasets failed")
        return 1


if __name__ == "__main__":
    exit(main())
