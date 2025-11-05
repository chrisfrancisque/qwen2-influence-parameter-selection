"""
Arm B: Influence-Based Parameter Masking

1. Loads baseline_start checkpoint
2. Computes influence scores on train split (1000 samples, single full-batch)
3. Selects bottom 5% parameters (most negative scores)
4. Masks selected parameters to zero
5. Evaluates masked model on validation (NO TRAINING)
"""

import time
import torch
from pathlib import Path
from typing import Dict, Optional
from transformers import AutoModelForSequenceClassification

from ..data.utils import create_dataloader
from ..evaluation.evaluator import evaluate_model
from ..influence.compute_scores import compute_influence_scores
from ..influence.parameter_selector import select_parameters_to_mask, apply_masks_to_model
from ..utils.tpu_utils import get_device, is_master_ordinal, print_once


def run_influence_arm(
    checkpoint_path: str,
    train_dataset,
    val_dataset,
    num_labels: int,
    batch_size: int = 128,
    mask_fraction: float = 0.05,
    device: Optional[torch.device] = None,
    use_tpu: bool = False,
    model_dtype: str = "bfloat16"
) -> Dict[str, float]:
    """
    Arm B: Compute influence scores, mask parameters, evaluate

    Args:
        checkpoint_path: Path to baseline_start checkpoint
        train_dataset: Tokenized training dataset (1000 samples)
        val_dataset: Tokenized validation dataset
        num_labels: Number of classification labels
        batch_size: Batch size for evaluation (not for influence computation)
        mask_fraction: Fraction of parameters to mask (default 0.05 = 5%)
        device: Device to run on
        use_tpu: Whether using TPU
        model_dtype: Model precision

    Returns:
        Dictionary with metrics and metadata
    """
    print_once("\n" + "=" * 60)
    print_once("ARM B: INFLUENCE-BASED PARAMETER MASKING")
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

    # Print model info
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print_once(f"\nModel status:")
    print_once(f"  Total parameters: {total_params:,}")
    print_once(f"  Trainable parameters: {trainable_params:,}")

    # Step 1: Compute influence scores
    print_once(f"\nStep 1: Computing influence scores on {len(train_dataset)} training samples")

    influence_scores = compute_influence_scores(
        model=model,
        train_dataset=train_dataset,
        batch_size=len(train_dataset),  # Single full-batch
        device=device,
        use_tpu=use_tpu
    )

    # Step 2: Select parameters to mask
    print_once(f"\nStep 2: Selecting parameters to mask (fraction={mask_fraction:.2%})")

    masks = select_parameters_to_mask(
        influence_scores=influence_scores,
        mask_fraction=mask_fraction
    )

    # Step 3: Apply masks to model
    print_once(f"\nStep 3: Applying masks to model")

    num_masked = apply_masks_to_model(
        model=model,
        masks=masks,
        mask_value=0.0
    )

    # Step 4: Evaluate masked model
    print_once(f"\nStep 4: Evaluating masked model on validation set")

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

    metrics = evaluate_model(
        model=model,
        dataloader=val_loader,
        device=device,
        num_labels=num_labels,
        use_tpu=use_tpu,
        desc="Arm B Eval"
    )

    # Calculate runtime
    wall_time_minutes = (time.time() - start_time) / 60.0

    # Add metadata
    results = {
        'arm': 'influence',
        'val_accuracy': metrics['accuracy'],
        'val_macro_f1': metrics['macro_f1'],
        'val_loss': metrics['loss'],
        'epochs_run': 0,
        'optimizer_steps': 0,
        'tokens_processed': 0,
        'num_masked_params': num_masked,
        'mask_fraction': mask_fraction,
        'total_params': total_params,
        'wall_time_minutes': wall_time_minutes,
        'notes': f'Influence-based masking ({mask_fraction:.1%} params masked)'
    }

    # Print results
    print_once("\n" + "=" * 60)
    print_once("ARM B RESULTS")
    print_once("=" * 60)
    print_once(f"  Parameters masked:   {results['num_masked_params']:,} ({mask_fraction:.2%})")
    print_once(f"  Validation Accuracy: {results['val_accuracy']:.2%}")
    print_once(f"  Validation Macro F1: {results['val_macro_f1']:.4f}")
    print_once(f"  Validation Loss:     {results['val_loss']:.4f}")
    print_once(f"  Wall Time:           {results['wall_time_minutes']:.2f} minutes")
    print_once("=" * 60)

    return results


if __name__ == "__main__":
    # Test Arm B
    print("Testing Arm B (Influence-based masking)...")

    from ..data.loader import load_and_split_dataset
    from ..data.utils import tokenize_dataset, get_tokenizer

    dataset_name = "sst2"
    checkpoint_path = f"../../outputs/checkpoints/{dataset_name}/baseline_start"

    if not Path(checkpoint_path).exists():
        print(f"\nCheckpoint not found: {checkpoint_path}")
        print("Please run scripts/0_create_baselines.py first")
        exit(1)

    # Load data
    _, train, val, config = load_and_split_dataset(
        dataset_name,
        config_dir="../../config/datasets",
        output_dir="../../outputs/splits"
    )

    # Take small subsets for testing
    train_small = train.select(range(100))
    val_small = val.select(range(100))

    # Tokenize
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

    # Run Arm B
    results = run_influence_arm(
        checkpoint_path=checkpoint_path,
        train_dataset=train_tokenized,
        val_dataset=val_tokenized,
        num_labels=config.num_labels,
        batch_size=32,
        mask_fraction=0.05,
        use_tpu=False,
        model_dtype="float32"
    )

    print("\nTest completed!")
    print(f"Results: {results}")
