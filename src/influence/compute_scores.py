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

    # Create dataloader with smaller batch size for gradient accumulation
    # Use smaller batches to avoid OOM, but accumulate gradients across all samples
    micro_batch_size = min(batch_size, 128)  # Max 128 per micro-batch to avoid OOM

    dataloader = create_dataloader(
        train_dataset,
        batch_size=micro_batch_size,
        shuffle=False,  # No shuffling for reproducibility
        drop_last=False
    )

    num_batches = len(dataloader)
    total_samples = len(train_dataset)

    print_once(f"\nDataloader setup (gradient accumulation):")
    print_once(f"  Total samples: {total_samples}")
    print_once(f"  Micro-batch size: {micro_batch_size}")
    print_once(f"  Num micro-batches: {num_batches}")

    # Set model to train mode for gradient computation
    model.train()

    # Enable gradient checkpointing to save memory
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print_once("  Enabled gradient checkpointing")

    # Zero gradients
    model.zero_grad()

    # Accumulate gradients across all batches
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    print_once(f"\nComputing gradients with accumulation...")

    for batch_idx, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        logits = outputs.logits

        # Compute loss (normalized by total samples for proper averaging)
        loss = criterion(logits, batch['labels'])
        normalized_loss = loss * (len(batch['labels']) / total_samples)

        # Backward pass (accumulate gradients)
        normalized_loss.backward()

        total_loss += loss.item() * len(batch['labels'])

        # Periodic mark_step to avoid memory buildup
        if use_tpu and (batch_idx + 1) % 2 == 0:
            mark_step()

        if (batch_idx + 1) % max(1, num_batches // 4) == 0:
            print_once(f"  Processed {batch_idx + 1}/{num_batches} batches")

    # Final mark step for TPU
    if use_tpu:
        print_once("  Synchronizing TPU...")
        mark_step()

    avg_loss = total_loss / total_samples
    print_once(f"  Average loss across all samples: {avg_loss:.4f}")

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
