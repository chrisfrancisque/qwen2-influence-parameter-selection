"""
Influence Score Computation

Computes sign-corrected influence scores for all model parameters:
    score_i = sign(θ_i) × ∂L/∂θ_i

Uses single full-batch gradient (all 1000 training samples) for snapshot.
"""

import torch
import torch.nn as nn
from typing import Dict, OrderedDict
from collections import OrderedDict as ODict

from ..data.utils import create_dataloader
from ..utils.tpu_utils import get_device, mark_step, is_master_ordinal, print_once


def compute_influence_scores(
    model,
    train_dataset,
    batch_size: int = 1000,
    device: torch.device = None,
    use_tpu: bool = False
) -> OrderedDict[str, torch.Tensor]:
    """
    Compute sign-corrected influence scores for all model parameters

    Uses single full-batch gradient computation:
        score_i = sign(θ_i) × ∂L/∂θ_i

    Args:
        model: Qwen2 model (will temporarily unfreeze backbone)
        train_dataset: Tokenized training dataset (1000 samples)
        batch_size: Batch size (default 1000 for single full-batch)
        device: Device to compute on
        use_tpu: Whether using TPU

    Returns:
        OrderedDict mapping parameter name -> influence score tensor
    """
    print_once("\n" + "=" * 60)
    print_once("Computing Influence Scores")
    print_once("=" * 60)

    # Get device
    if device is None:
        device = get_device(use_tpu=use_tpu)

    model = model.to(device)

    # Store original requires_grad states
    original_requires_grad = {}
    for name, param in model.named_parameters():
        original_requires_grad[name] = param.requires_grad

    # Unfreeze backbone for gradient computation
    # Keep classifier head frozen (we don't want to mask it)
    print_once("\nTemporarily unfreezing backbone for gradient computation...")
    for name, param in model.named_parameters():
        if 'score' in name or 'classifier' in name:
            # Keep head frozen
            param.requires_grad = False
        else:
            # Unfreeze backbone
            param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_once(f"  Trainable parameters for gradient: {trainable_params:,}")

    # Create dataloader
    # For single full-batch: batch_size should equal dataset size (1000)
    if len(train_dataset) != batch_size:
        print_once(f"\nWARNING: Dataset size ({len(train_dataset)}) != batch_size ({batch_size})")
        print_once(f"  Expected single full-batch with batch_size={len(train_dataset)}")

    dataloader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle for single batch
        drop_last=False
    )

    print_once(f"\nDataloader setup:")
    print_once(f"  Dataset size: {len(train_dataset)}")
    print_once(f"  Batch size: {batch_size}")
    print_once(f"  Num batches: {len(dataloader)}")

    if len(dataloader) > 1:
        print_once(f"\n  WARNING: Multiple batches detected ({len(dataloader)})")
        print_once(f"  Using first batch only for single-batch gradient")

    # Get first (and ideally only) batch
    batch = next(iter(dataloader))
    batch = {k: v.to(device) for k, v in batch.items()}

    actual_batch_size = batch['labels'].size(0)
    print_once(f"\nComputing gradient on batch of size: {actual_batch_size}")

    # Set model to train mode for gradient computation
    model.train()

    # Zero gradients
    model.zero_grad()

    # Forward pass
    outputs = model(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask']
    )
    logits = outputs.logits

    # Compute loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, batch['labels'])

    print_once(f"  Batch loss: {loss.item():.4f}")

    # Backward pass to compute gradients
    print_once("\nComputing gradients...")
    loss.backward()

    # Mark step for TPU
    if use_tpu:
        mark_step()

    # Compute influence scores: score_i = sign(θ_i) × ∂L/∂θ_i
    print_once("\nComputing influence scores: score_i = sign(θ_i) × grad_i")

    influence_scores = ODict()
    total_params_scored = 0

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            # Compute sign-corrected influence
            # score = sign(param) * grad
            score = torch.sign(param.data) * param.grad.data

            # Store on CPU to save device memory
            influence_scores[name] = score.cpu()
            total_params_scored += param.numel()

    print_once(f"\nInfluence scores computed:")
    print_once(f"  Parameters scored: {total_params_scored:,}")
    print_once(f"  Num parameter tensors: {len(influence_scores)}")

    # Restore original requires_grad states
    print_once("\nRestoring original parameter freeze states...")
    for name, param in model.named_parameters():
        param.requires_grad = original_requires_grad[name]

    # Clear gradients
    model.zero_grad()

    # Set back to eval mode
    model.eval()

    print_once("=" * 60)

    return influence_scores


def print_influence_statistics(
    influence_scores: OrderedDict[str, torch.Tensor],
    top_k: int = 10
):
    """
    Print statistics about computed influence scores

    Args:
        influence_scores: OrderedDict of parameter name -> score tensor
        top_k: Number of top/bottom parameters to display
    """
    if not is_master_ordinal():
        return

    print("\n" + "=" * 60)
    print("Influence Score Statistics")
    print("=" * 60)

    # Flatten all scores
    all_scores = torch.cat([s.flatten() for s in influence_scores.values()])

    print(f"\nGlobal statistics:")
    print(f"  Total parameters: {len(all_scores):,}")
    print(f"  Mean: {all_scores.mean().item():.6f}")
    print(f"  Std: {all_scores.std().item():.6f}")
    print(f"  Min: {all_scores.min().item():.6f}")
    print(f"  Max: {all_scores.max().item():.6f}")
    print(f"  Median: {all_scores.median().item():.6f}")

    # Percentiles
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\nPercentiles:")
    for p in percentiles:
        val = torch.quantile(all_scores, p / 100.0).item()
        print(f"  {p:2d}%: {val:>12.6f}")

    # Per-layer statistics
    print(f"\nPer-parameter statistics:")
    print(f"  {'Parameter Name':<60s} {'Mean':>12s} {'Std':>12s} {'Min':>12s} {'Max':>12s}")
    print("-" * 100)

    for name, scores in influence_scores.items():
        mean = scores.mean().item()
        std = scores.std().item()
        min_val = scores.min().item()
        max_val = scores.max().item()

        print(f"  {name:<60s} {mean:>12.6f} {std:>12.6f} {min_val:>12.6f} {max_val:>12.6f}")

    print("=" * 60)


if __name__ == "__main__":
    # Test influence computation
    print("Testing influence score computation...")

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.data.loader import load_and_split_dataset
    from src.data.utils import tokenize_dataset, get_tokenizer
    from src.models.qwen2_wrapper import load_qwen2_model, freeze_backbone

    # Load data
    _, train, _, config = load_and_split_dataset(
        "sst2",
        config_dir="../../config/datasets",
        output_dir="../../outputs/splits"
    )

    # Take small subset for testing (50 samples)
    train_small = train.select(range(50))

    # Tokenize
    tokenizer = get_tokenizer()
    train_tokenized = tokenize_dataset(
        train_small,
        config.text_column,
        config.label_column,
        tokenizer
    )

    # Load model
    print("\nLoading model...")
    device = torch.device('cpu')
    model = load_qwen2_model(
        num_labels=config.num_labels,
        device=device,
        dtype=torch.float32
    )

    # Freeze backbone (will be temporarily unfrozen during influence computation)
    freeze_backbone(model, freeze=True)

    # Compute influence scores
    print("\nComputing influence scores...")
    influence_scores = compute_influence_scores(
        model=model,
        train_dataset=train_tokenized,
        batch_size=50,  # Single batch of 50 samples
        device=device,
        use_tpu=False
    )

    # Print statistics
    print_influence_statistics(influence_scores, top_k=5)

    print("\nTest completed!")
