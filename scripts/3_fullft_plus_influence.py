#!/usr/bin/env python3
"""
Script 3: Full Fine-Tuning + Influence Masking Experiment

Workflow:
1. Load baseline checkpoint
2. Full fine-tune on 1000 training samples (track training time only)
3. Validate fine-tuned model
4. Compute influence scores and mask top k% parameters (track influence time only)
5. Validate masked model

Usage:
    python scripts/3_fullft_plus_influence.py --dataset sst2 --use_tpu
    python scripts/3_fullft_plus_influence.py --datasets sst2 agnews dbpedia yelp --use_tpu
"""

import argparse
import sys
import json
import yaml
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_and_split_dataset
from src.data.utils import tokenize_dataset, get_tokenizer, create_dataloader
from src.models.qwen2_wrapper import load_qwen2_model, freeze_backbone
from src.evaluation.evaluator import evaluate_model
from src.influence.compute_scores import compute_influence_scores
from src.influence.parameter_selector import select_parameters_to_mask, apply_masks_to_model
from src.utils.tpu_utils import setup_tpu_environment, get_device, mark_step, optimizer_step, print_once
import torch
import torch.nn as nn


def load_config(config_path: str = "config/experiment.yaml") -> dict:
    """Load experiment configuration"""
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_fullft_plus_influence(
    dataset_name: str,
    mask_fraction: float = 0.01,
    config_dir: str = "config/datasets",
    splits_dir: str = "outputs/splits",
    checkpoints_dir: str = "outputs/checkpoints",
    results_dir: str = "outputs/results",
    use_tpu: bool = False,
    experiment_config: dict = None
):
    """
    Run Full FT + Influence experiment for a single dataset

    Args:
        dataset_name: Dataset to run
        mask_fraction: Fraction of parameters to mask with influence
        config_dir: Directory with dataset configs
        splits_dir: Directory with dataset splits
        checkpoints_dir: Directory with baseline checkpoints
        results_dir: Directory to save results
        use_tpu: Whether using TPU
        experiment_config: Experiment configuration dict
    """
    print("\n" + "=" * 80)
    print(f"FULL FT + INFLUENCE EXPERIMENT: {dataset_name.upper()}")
    print("=" * 80)

    # Load experiment config
    if experiment_config is None:
        experiment_config = load_config()

    # Get hyperparameters
    model_dtype = experiment_config['model']['dtype']
    batch_size = experiment_config['data'].get('batch_size', experiment_config['data'].get('effective_batch_size', 128))
    fullft_config = experiment_config['training']['fullft']

    # Get device
    device = get_device(use_tpu=use_tpu)

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

    # ========================================================================
    # STEP 1: FULL FINE-TUNING
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: FULL FINE-TUNING")
    print("=" * 80)

    # Load model
    dtype = torch.bfloat16 if model_dtype == "bfloat16" else torch.float32
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_path,
        num_labels=config.num_labels,
        torch_dtype=dtype,
        trust_remote_code=True
    )

    # Unfreeze entire model
    freeze_backbone(model, freeze=False, verbose=True)
    model = model.to(device)

    # Create dataloaders
    train_loader = create_dataloader(train_tokenized, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = create_dataloader(val_tokenized, batch_size=batch_size, shuffle=False, drop_last=False)

    # Training setup
    learning_rate = fullft_config.get('learning_rate', fullft_config.get('lr', 1e-5))
    weight_decay = fullft_config['weight_decay']
    max_epochs = fullft_config['max_epochs']
    early_stop_patience = fullft_config.get('early_stop_patience', fullft_config.get('patience', 2))
    early_stop_min_delta = fullft_config.get('early_stop_min_delta', fullft_config.get('min_delta', 0.001))

    print_once(f"\nTraining configuration:")
    print_once(f"  Learning rate: {learning_rate}")
    print_once(f"  Weight decay: {weight_decay}")
    print_once(f"  Max epochs: {max_epochs}")
    print_once(f"  Batch size: {batch_size}")
    print_once(f"  Steps per epoch: {len(train_loader)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Track training time only
    training_start_time = time.time()

    best_val_accuracy = 0.0
    epochs_without_improvement = 0
    total_optimizer_steps = 0

    for epoch in range(max_epochs):
        print_once(f"\n{'='*60}")
        print_once(f"Epoch {epoch + 1}/{max_epochs}")
        print_once(f"{'='*60}")

        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            loss = criterion(outputs.logits, batch['labels'])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if use_tpu:
                optimizer_step(optimizer, barrier=True)
            else:
                optimizer.step()
                optimizer.zero_grad()

            if use_tpu:
                mark_step()

            train_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)
            train_correct += (preds == batch['labels']).sum().item()
            train_total += batch['labels'].size(0)
            total_optimizer_steps += 1

            if (step + 1) % max(1, len(train_loader) // 4) == 0:
                print_once(f"  Step {step + 1}/{len(train_loader)}: Loss = {train_loss / (step + 1):.4f}")

        # End training time tracking BEFORE validation
        training_time_minutes = (time.time() - training_start_time) / 60.0

        # Validation (not included in training time)
        print_once(f"\nValidating...")
        val_metrics = evaluate_model(model, val_loader, device, config.num_labels, use_tpu, desc=f"Epoch {epoch+1} Val")
        val_accuracy = val_metrics['accuracy']

        print_once(f"\nEpoch {epoch + 1}: Train Acc = {train_correct/train_total:.2%}, Val Acc = {val_accuracy:.2%}")

        # Early stopping
        if val_accuracy > best_val_accuracy + early_stop_min_delta:
            print_once(f"  Improvement: {best_val_accuracy:.2%} -> {val_accuracy:.2%}")
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print_once(f"  No improvement ({epochs_without_improvement}/{early_stop_patience})")
            if epochs_without_improvement >= early_stop_patience:
                print_once(f"\nEarly stopping triggered")
                break

        # Resume training time tracking for next epoch
        training_start_time = time.time()

    # Final validation after full FT
    print_once("\n" + "=" * 80)
    print_once("STEP 1 COMPLETE: Full Fine-Tuning")
    print_once("=" * 80)

    final_ft_metrics = evaluate_model(model, val_loader, device, config.num_labels, use_tpu, desc="Final FT Eval")

    print_once(f"\nFull FT Results:")
    print_once(f"  Training time: {training_time_minutes:.2f} minutes")
    print_once(f"  Validation Accuracy: {final_ft_metrics['accuracy']:.2%}")
    print_once(f"  Validation Macro F1: {final_ft_metrics['macro_f1']:.4f}")
    print_once(f"  Validation Loss: {final_ft_metrics['loss']:.4f}")

    # ========================================================================
    # STEP 2: INFLUENCE-BASED MASKING
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: INFLUENCE-BASED PARAMETER MASKING")
    print("=" * 80)

    influence_start_time = time.time()

    # Compute influence scores
    print_once(f"\nComputing influence scores (mask fraction = {mask_fraction:.1%})...")

    influence_scores = compute_influence_scores(
        model=model,
        train_dataset=train_tokenized,
        batch_size=len(train_tokenized),
        device=device,
        use_tpu=use_tpu
    )

    # Select parameters to mask
    masks = select_parameters_to_mask(
        influence_scores=influence_scores,
        mask_fraction=mask_fraction
    )

    # Apply masks
    num_masked = apply_masks_to_model(model=model, masks=masks, mask_value=0.0)

    influence_time_minutes = (time.time() - influence_start_time) / 60.0

    print_once(f"\nInfluence computation time: {influence_time_minutes:.2f} minutes")

    # Validate masked model
    print_once("\n" + "=" * 80)
    print_once("STEP 2 COMPLETE: Influence-Based Masking")
    print_once("=" * 80)

    final_masked_metrics = evaluate_model(model, val_loader, device, config.num_labels, use_tpu, desc="Masked Model Eval")

    print_once(f"\nMasked Model Results:")
    print_once(f"  Parameters masked: {num_masked:,} ({mask_fraction:.1%})")
    print_once(f"  Validation Accuracy: {final_masked_metrics['accuracy']:.2%}")
    print_once(f"  Validation Macro F1: {final_masked_metrics['macro_f1']:.4f}")
    print_once(f"  Validation Loss: {final_masked_metrics['loss']:.4f}")

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    results = {
        'dataset': dataset_name,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'num_labels': config.num_labels,
            'train_samples': len(train),
            'val_samples': len(val),
            'batch_size': batch_size,
            'mask_fraction': mask_fraction,
            'use_tpu': use_tpu,
            'model_dtype': model_dtype
        },
        'full_ft': {
            'training_time_minutes': training_time_minutes,
            'val_accuracy': final_ft_metrics['accuracy'],
            'val_macro_f1': final_ft_metrics['macro_f1'],
            'val_loss': final_ft_metrics['loss'],
            'epochs_run': epoch + 1,
            'optimizer_steps': total_optimizer_steps
        },
        'influence_masked': {
            'influence_time_minutes': influence_time_minutes,
            'val_accuracy': final_masked_metrics['accuracy'],
            'val_macro_f1': final_masked_metrics['macro_f1'],
            'val_loss': final_masked_metrics['loss'],
            'num_masked_params': num_masked,
            'mask_fraction': mask_fraction
        },
        'comparison': {
            'accuracy_change': final_masked_metrics['accuracy'] - final_ft_metrics['accuracy'],
            'accuracy_change_pct': ((final_masked_metrics['accuracy'] - final_ft_metrics['accuracy']) / final_ft_metrics['accuracy']) * 100
        }
    }

    # Save results
    results_path = Path(results_dir) / "fullft_influence" / dataset_name
    results_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_path / f"results_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print_once("\n" + "=" * 80)
    print_once("EXPERIMENT COMPLETE")
    print_once("=" * 80)
    print_once(f"\nResults saved to: {results_file}")

    # Print summary
    print_once("\n" + "=" * 80)
    print_once("SUMMARY")
    print_once("=" * 80)
    print_once(f"\nDataset: {dataset_name}")
    print_once(f"\nFull FT Training: {training_time_minutes:.2f} min → Accuracy: {final_ft_metrics['accuracy']:.2%}")
    print_once(f"Influence Masking: {influence_time_minutes:.2f} min → Accuracy: {final_masked_metrics['accuracy']:.2%}")
    print_once(f"\nAccuracy Change: {results['comparison']['accuracy_change']:+.2%} ({results['comparison']['accuracy_change_pct']:+.2f}%)")
    print_once("=" * 80)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run Full FT + Influence experiment")
    parser.add_argument("--dataset", type=str, help="Single dataset to run")
    parser.add_argument("--datasets", nargs="+", help="Multiple datasets to run")
    parser.add_argument("--mask_fraction", type=float, default=0.01, help="Fraction of parameters to mask (default: 0.01 = 1%)")
    parser.add_argument("--config_dir", type=str, default="config/datasets")
    parser.add_argument("--splits_dir", type=str, default="outputs/splits")
    parser.add_argument("--checkpoints_dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--results_dir", type=str, default="outputs/results")
    parser.add_argument("--use_tpu", action="store_true")

    args = parser.parse_args()

    # Setup TPU if needed
    if args.use_tpu:
        setup_tpu_environment()

    # Load config
    experiment_config = load_config()

    # Determine which datasets to run
    if args.dataset:
        datasets = [args.dataset]
    elif args.datasets:
        datasets = args.datasets
    else:
        datasets = ["sst2", "agnews", "dbpedia", "yelp"]

    # Run experiments
    all_results = {}

    for dataset_name in datasets:
        print(f"\n\n{'='*80}")
        print(f"DATASET: {dataset_name.upper()}")
        print(f"{'='*80}\n")

        try:
            results = run_fullft_plus_influence(
                dataset_name=dataset_name,
                mask_fraction=args.mask_fraction,
                config_dir=args.config_dir,
                splits_dir=args.splits_dir,
                checkpoints_dir=args.checkpoints_dir,
                results_dir=args.results_dir,
                use_tpu=args.use_tpu,
                experiment_config=experiment_config
            )
            all_results[dataset_name] = results

        except Exception as e:
            print(f"\nERROR running {dataset_name}: {e}")
            import traceback
            traceback.print_exc()

    # Print final summary
    if len(all_results) > 1:
        print("\n\n" + "=" * 80)
        print("ALL DATASETS SUMMARY")
        print("=" * 80)
        print(f"\n{'Dataset':<12s} {'FT Acc':>8s} {'Masked Acc':>10s} {'Change':>8s} {'FT Time':>10s} {'Inf Time':>10s}")
        print("-" * 70)

        for dataset_name, results in all_results.items():
            ft_acc = results['full_ft']['val_accuracy']
            masked_acc = results['influence_masked']['val_accuracy']
            change = results['comparison']['accuracy_change']
            ft_time = results['full_ft']['training_time_minutes']
            inf_time = results['influence_masked']['influence_time_minutes']

            print(f"{dataset_name:<12s} {ft_acc:>7.2%} {masked_acc:>9.2%} {change:>+7.2%} {ft_time:>9.2f}m {inf_time:>9.2f}m")

        print("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
