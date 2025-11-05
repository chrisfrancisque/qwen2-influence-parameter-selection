"""
Arm A: Pretrained Baseline Evaluation

Simply loads baseline_start checkpoint and evaluates on validation set.
No training is performed.
"""

import time
import torch
from pathlib import Path
from typing import Dict, Optional
from transformers import AutoModelForSequenceClassification

from ..data.utils import create_dataloader
from ..evaluation.evaluator import evaluate_model
from ..utils.tpu_utils import get_device, is_master_ordinal, print_once


def run_baseline_arm(
    checkpoint_path: str,
    val_dataset,
    num_labels: int,
    batch_size: int = 128,
    device: Optional[torch.device] = None,
    use_tpu: bool = False,
    model_dtype: str = "bfloat16"
) -> Dict[str, float]:
    """
    Arm A: Evaluate baseline_start checkpoint on validation set

    Args:
        checkpoint_path: Path to baseline_start checkpoint
        val_dataset: Tokenized validation dataset
        num_labels: Number of classification labels
        batch_size: Batch size for evaluation
        device: Device to run on
        use_tpu: Whether using TPU
        model_dtype: Model precision

    Returns:
        Dictionary with metrics and metadata
    """
    print_once("\n" + "=" * 60)
    print_once("ARM A: PRETRAINED BASELINE (No Training)")
    print_once("=" * 60)

    start_time = time.time()

    # Get device
    if device is None:
        device = get_device(use_tpu=use_tpu)

    # Load model from checkpoint
    print_once(f"\nLoading checkpoint: {checkpoint_path}")

    dtype = torch.bfloat16 if model_dtype == "bfloat16" else torch.float32

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_path,
        num_labels=num_labels,
        torch_dtype=dtype,
        trust_remote_code=True
    )

    model = model.to(device)
    model.eval()

    print_once(f"  Model loaded successfully")

    # Verify model is frozen (backbone should be frozen from baseline creation)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print_once(f"\nModel status:")
    print_once(f"  Total parameters: {total_params:,}")
    print_once(f"  Trainable parameters: {trainable_params:,}")

    # Create dataloader
    val_loader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    print_once(f"\nEvaluation setup:")
    print_once(f"  Validation samples: {len(val_dataset)}")
    print_once(f"  Batch size: {batch_size}")
    print_once(f"  Num batches: {len(val_loader)}")

    # Evaluate
    print_once("\nEvaluating on validation set...")

    metrics = evaluate_model(
        model=model,
        dataloader=val_loader,
        device=device,
        num_labels=num_labels,
        use_tpu=use_tpu,
        desc="Arm A Eval"
    )

    # Calculate runtime
    wall_time_minutes = (time.time() - start_time) / 60.0

    # Add metadata
    results = {
        'arm': 'baseline',
        'val_accuracy': metrics['accuracy'],
        'val_macro_f1': metrics['macro_f1'],
        'val_loss': metrics['loss'],
        'epochs_run': 0,
        'optimizer_steps': 0,
        'tokens_processed': 0,
        'wall_time_minutes': wall_time_minutes,
        'notes': 'Pretrained baseline (frozen backbone + 1-epoch head)'
    }

    # Print results
    print_once("\n" + "=" * 60)
    print_once("ARM A RESULTS")
    print_once("=" * 60)
    print_once(f"  Validation Accuracy: {results['val_accuracy']:.2%}")
    print_once(f"  Validation Macro F1: {results['val_macro_f1']:.4f}")
    print_once(f"  Validation Loss:     {results['val_loss']:.4f}")
    print_once(f"  Wall Time:           {results['wall_time_minutes']:.2f} minutes")
    print_once("=" * 60)

    return results


if __name__ == "__main__":
    # Test Arm A
    print("Testing Arm A (Baseline evaluation)...")

    from ..data.loader import load_and_split_dataset
    from ..data.utils import tokenize_dataset, get_tokenizer

    # This assumes baseline checkpoint already exists
    # You would need to run 0_create_baselines.py first

    dataset_name = "sst2"
    checkpoint_path = f"../../outputs/checkpoints/{dataset_name}/baseline_start"

    if not Path(checkpoint_path).exists():
        print(f"\nCheckpoint not found: {checkpoint_path}")
        print("Please run scripts/0_create_baselines.py first")
        exit(1)

    # Load data
    _, _, val, config = load_and_split_dataset(
        dataset_name,
        config_dir="../../config/datasets",
        output_dir="../../outputs/splits"
    )

    # Take small subset for testing
    val_small = val.select(range(100))

    # Tokenize
    tokenizer = get_tokenizer()
    val_tokenized = tokenize_dataset(
        val_small,
        config.text_column,
        config.label_column,
        tokenizer
    )

    # Run Arm A
    results = run_baseline_arm(
        checkpoint_path=checkpoint_path,
        val_dataset=val_tokenized,
        num_labels=config.num_labels,
        batch_size=32,
        use_tpu=False,
        model_dtype="float32"
    )

    print("\nTest completed!")
    print(f"Results: {results}")
