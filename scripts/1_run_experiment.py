#!/usr/bin/env python3
"""
Script 1: Run Full Experiment

Runs all 4 arms for a single dataset:
  A. Pretrained Baseline (eval only)
  B. Influence-based Masking (compute influence, mask, eval)
  C. LoRA Fine-tuning (train with early stopping)
  D. Full Fine-tuning (train with early stopping)

Usage:
    python scripts/1_run_experiment.py --dataset sst2
    python scripts/1_run_experiment.py --dataset sst2 --use_tpu
    python scripts/1_run_experiment.py --dataset sst2 --arms baseline influence
"""

import argparse
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_and_split_dataset
from src.data.utils import tokenize_dataset, get_tokenizer
from src.training.baseline_trainer import run_baseline_arm
from src.training.influence_trainer import run_influence_arm
from src.training.lora_trainer import run_lora_arm
from src.training.fullft_trainer import run_fullft_arm
from src.utils.tpu_utils import setup_tpu_environment


def load_config(config_path: str = "config/experiment.yaml") -> dict:
    """Load experiment configuration"""
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_experiment(
    dataset_name: str,
    arms: list = None,
    config_dir: str = "config/datasets",
    splits_dir: str = "outputs/splits",
    checkpoints_dir: str = "outputs/checkpoints",
    results_dir: str = "outputs/results",
    use_tpu: bool = False,
    experiment_config: dict = None
):
    """
    Run experiment for a single dataset

    Args:
        dataset_name: Dataset to run (sst2, agnews, dbpedia, yelp)
        arms: List of arms to run (default: all)
        config_dir: Directory with dataset configs
        splits_dir: Directory with dataset splits
        checkpoints_dir: Directory with baseline checkpoints
        results_dir: Directory to save results
        use_tpu: Whether using TPU
        experiment_config: Experiment configuration dict
    """
    print("\n" + "=" * 80)
    print(f"RUNNING EXPERIMENT: {dataset_name.upper()}")
    print("=" * 80)

    if arms is None:
        arms = ['baseline', 'influence', 'lora', 'fullft']

    # Load experiment config if not provided
    if experiment_config is None:
        experiment_config = load_config()

    # Get hyperparameters
    model_dtype = experiment_config['model']['dtype']
    batch_size = experiment_config['data'].get('batch_size', experiment_config['data'].get('effective_batch_size', 128))

    influence_config = experiment_config['influence']
    lora_config = experiment_config['training']['lora']
    fullft_config = experiment_config['training']['fullft']

    # Load dataset
    print(f"\nLoading dataset: {dataset_name}")
    _, train, val, config = load_and_split_dataset(
        dataset_name=dataset_name,
        config_dir=config_dir,
        output_dir=splits_dir
    )

    print(f"  Train samples: {len(train)}")
    print(f"  Val samples: {len(val)}")
    print(f"  Num labels: {config.num_labels}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = get_tokenizer()

    # Tokenize datasets
    print("\nTokenizing datasets...")
    train_tokenized = tokenize_dataset(
        dataset=train,
        text_column=config.text_column,
        label_column=config.label_column,
        tokenizer=tokenizer
    )

    val_tokenized = tokenize_dataset(
        dataset=val,
        text_column=config.text_column,
        label_column=config.label_column,
        tokenizer=tokenizer
    )

    # Checkpoint path
    checkpoint_path = f"{checkpoints_dir}/{dataset_name}/baseline_start"

    if not Path(checkpoint_path).exists():
        print(f"\nERROR: Checkpoint not found: {checkpoint_path}")
        print("Please run scripts/0_create_baselines.py first")
        sys.exit(1)

    print(f"\nUsing checkpoint: {checkpoint_path}")

    # Results storage
    all_results = {
        'dataset': dataset_name,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'num_labels': config.num_labels,
            'train_samples': len(train),
            'val_samples': len(val),
            'batch_size': batch_size,
            'use_tpu': use_tpu,
            'model_dtype': model_dtype
        },
        'arms': {}
    }

    # Run each arm
    if 'baseline' in arms:
        print("\n" + "=" * 80)
        print("Running ARM A: Pretrained Baseline")
        print("=" * 80)

        results = run_baseline_arm(
            checkpoint_path=checkpoint_path,
            val_dataset=val_tokenized,
            num_labels=config.num_labels,
            batch_size=batch_size,
            use_tpu=use_tpu,
            model_dtype=model_dtype
        )

        all_results['arms']['baseline'] = results

    if 'influence' in arms:
        print("\n" + "=" * 80)
        print("Running ARM B: Influence-based Masking")
        print("=" * 80)

        results = run_influence_arm(
            checkpoint_path=checkpoint_path,
            train_dataset=train_tokenized,
            val_dataset=val_tokenized,
            num_labels=config.num_labels,
            batch_size=batch_size,
            mask_fraction=influence_config['mask_fraction'],
            use_tpu=use_tpu,
            model_dtype=model_dtype
        )

        all_results['arms']['influence'] = results

    if 'lora' in arms:
        print("\n" + "=" * 80)
        print("Running ARM C: LoRA Fine-tuning")
        print("=" * 80)

        results = run_lora_arm(
            checkpoint_path=checkpoint_path,
            train_dataset=train_tokenized,
            val_dataset=val_tokenized,
            num_labels=config.num_labels,
            batch_size=batch_size,
            learning_rate=lora_config['learning_rate'],
            weight_decay=lora_config['weight_decay'],
            max_epochs=lora_config['max_epochs'],
            early_stop_patience=lora_config['early_stop_patience'],
            early_stop_min_delta=lora_config['early_stop_min_delta'],
            lora_rank=lora_config['rank'],
            lora_alpha=lora_config['alpha'],
            use_tpu=use_tpu,
            model_dtype=model_dtype
        )

        all_results['arms']['lora'] = results

    if 'fullft' in arms:
        print("\n" + "=" * 80)
        print("Running ARM D: Full Fine-tuning")
        print("=" * 80)

        results = run_fullft_arm(
            checkpoint_path=checkpoint_path,
            train_dataset=train_tokenized,
            val_dataset=val_tokenized,
            num_labels=config.num_labels,
            batch_size=batch_size,
            learning_rate=fullft_config['learning_rate'],
            weight_decay=fullft_config['weight_decay'],
            max_epochs=fullft_config['max_epochs'],
            early_stop_patience=fullft_config['early_stop_patience'],
            early_stop_min_delta=fullft_config['early_stop_min_delta'],
            use_tpu=use_tpu,
            model_dtype=model_dtype
        )

        all_results['arms']['fullft'] = results

    # Save results
    results_path = Path(results_dir) / dataset_name
    results_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_path / f"results_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {results_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nDataset: {dataset_name}")
    print(f"Arms run: {', '.join(arms)}")
    print("\nValidation Accuracy:")

    for arm_name, arm_results in all_results['arms'].items():
        acc = arm_results['val_accuracy']
        f1 = arm_results['val_macro_f1']
        print(f"  {arm_name:12s}: {acc:.2%}  (F1: {f1:.4f})")

    print("=" * 80)

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run influence experiment for a dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["sst2", "agnews", "dbpedia", "yelp"],
        help="Dataset to run experiment on"
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        default=None,
        choices=["baseline", "influence", "lora", "fullft"],
        help="Which arms to run (default: all)"
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
        "--checkpoints_dir",
        type=str,
        default="outputs/checkpoints",
        help="Directory with baseline checkpoints"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="outputs/results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--use_tpu",
        action="store_true",
        help="Use TPU for training"
    )

    args = parser.parse_args()

    # Setup TPU environment if needed
    if args.use_tpu:
        setup_tpu_environment()

    # Load experiment config
    experiment_config = load_config()

    # Run experiment
    results = run_experiment(
        dataset_name=args.dataset,
        arms=args.arms,
        config_dir=args.config_dir,
        splits_dir=args.splits_dir,
        checkpoints_dir=args.checkpoints_dir,
        results_dir=args.results_dir,
        use_tpu=args.use_tpu,
        experiment_config=experiment_config
    )

    return 0


if __name__ == "__main__":
    exit(main())
