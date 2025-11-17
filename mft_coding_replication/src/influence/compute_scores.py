"""
Influence Score Computation for Causal Language Models

Computes sign-corrected influence scores for all model parameters:
    score_i = sign(θ_i) × ∂L/∂θ_i

Uses gradient accumulation across all SCI samples with causal LM loss.
"""

import torch
import torch.nn as nn
from typing import Dict, OrderedDict
from collections import OrderedDict as ODict
from torch.utils.data import DataLoader


def get_device(use_tpu: bool = False):
    """Get appropriate device."""
    if use_tpu:
        import torch_xla.core.xla_model as xm
        return xm.xla_device()
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def mark_step():
    """Mark XLA step for TPU."""
    try:
        import torch_xla.core.xla_model as xm
        xm.mark_step()
    except ImportError:
        pass


def is_master_ordinal():
    """Check if this is the master process."""
    try:
        import torch_xla.core.xla_model as xm
        return xm.is_master_ordinal()
    except ImportError:
        return True


def print_once(msg):
    """Print only from master process."""
    if is_master_ordinal():
        print(msg)


def create_dataloader(dataset, batch_size, shuffle=False, drop_last=False):
    """Create DataLoader with proper collation for causal LM."""
    def collate_fn(examples):
        # Pad sequences to max length in batch
        input_ids = [ex['input_ids'] for ex in examples]
        attention_mask = [ex['attention_mask'] for ex in examples]

        max_len = max(len(ids) for ids in input_ids)

        # Pad with 0 (usually pad_token_id)
        input_ids_padded = []
        attention_mask_padded = []

        for ids, mask in zip(input_ids, attention_mask):
            padding_len = max_len - len(ids)
            input_ids_padded.append(ids + [0] * padding_len)
            attention_mask_padded.append(mask + [0] * padding_len)

        return {
            'input_ids': torch.tensor(input_ids_padded, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask_padded, dtype=torch.long)
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=collate_fn
    )


def compute_influence_scores(
    model,
    sci_dataset,
    batch_size: int = 32,
    device: torch.device = None,
    use_tpu: bool = False
) -> OrderedDict[str, torch.Tensor]:
    """
    Compute sign-corrected influence scores for all model parameters.

    Uses gradient accumulation with causal LM loss:
        score_i = sign(θ_i) × ∂L/∂θ_i

    Args:
        model: Qwen2 causal LM model
        sci_dataset: Tokenized SCI dataset (~1k samples)
        batch_size: Micro-batch size for gradient accumulation
        device: Device to compute on
        use_tpu: Whether using TPU

    Returns:
        OrderedDict mapping parameter name -> influence score tensor
    """
    print_once("\n" + "=" * 60)
    print_once("Computing Sign-Corrected Influence Scores (Causal LM)")
    print_once("=" * 60)

    # Get device
    if device is None:
        device = get_device(use_tpu=use_tpu)

    model = model.to(device)

    # Store original requires_grad states
    original_requires_grad = {}
    for name, param in model.named_parameters():
        original_requires_grad[name] = param.requires_grad

    # Unfreeze all model parameters for gradient computation
    print_once("\nUnfreezing all parameters for gradient computation...")
    for param in model.parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_once(f"  Trainable parameters: {trainable_params:,}")

    # Create dataloader with small batch size for gradient accumulation
    dataloader = create_dataloader(
        sci_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffling for reproducibility
        drop_last=False
    )

    num_batches = len(dataloader)
    total_samples = len(sci_dataset)

    print_once(f"\nDataloader setup:")
    print_once(f"  Total SCI samples: {total_samples}")
    print_once(f"  Micro-batch size: {batch_size}")
    print_once(f"  Num micro-batches: {num_batches}")

    # Set model to train mode for gradient computation
    model.train()

    # Zero gradients
    model.zero_grad()

    # Accumulate gradients across all batches
    total_loss = 0.0

    print_once(f"\nComputing gradients with accumulation...")

    for batch_idx, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass with causal LM loss
        # For causal LM, we use input_ids as labels (shifted internally by the model)
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['input_ids']  # Causal LM: predict next token
        )

        # Loss is already computed by the model
        loss = outputs.loss

        # Normalize by total samples for proper averaging
        normalized_loss = loss * (batch['input_ids'].size(0) / total_samples)

        # Backward pass (accumulate gradients)
        normalized_loss.backward()

        total_loss += loss.item() * batch['input_ids'].size(0)

        # Periodic mark_step to avoid memory buildup on TPU
        if use_tpu and (batch_idx + 1) % 2 == 0:
            mark_step()

        if (batch_idx + 1) % max(1, num_batches // 4) == 0:
            print_once(f"  Processed {batch_idx + 1}/{num_batches} batches")

    # Final mark step for TPU
    if use_tpu:
        print_once("  Synchronizing TPU...")
        mark_step()

    avg_loss = total_loss / total_samples
    print_once(f"  Average causal LM loss: {avg_loss:.4f}")

    # Compute influence scores: score_i = sign(θ_i) × ∂L/∂θ_i
    print_once("\nComputing influence scores: score_i = sign(θ_i) × grad_i")

    influence_scores = ODict()
    total_params_scored = 0

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            # Compute sign-corrected influence
            score = torch.sign(param.data) * param.grad.data

            # Store on CPU to save device memory
            influence_scores[name] = score.cpu()
            total_params_scored += param.numel()

    print_once(f"\nInfluence scores computed:")
    print_once(f"  Parameters scored: {total_params_scored:,}")
    print_once(f"  Num parameter tensors: {len(influence_scores)}")

    # Restore original requires_grad states
    print_once("\nRestoring original parameter states...")
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
    Print statistics about computed influence scores.

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

    # Per-layer statistics (top layers by mean absolute score)
    print(f"\nTop {top_k} parameters by mean absolute score:")
    print(f"  {'Parameter Name':<60s} {'Mean':>12s} {'Std':>12s} {'Min':>12s} {'Max':>12s}")
    print("-" * 100)

    # Sort by mean absolute score
    sorted_params = sorted(
        influence_scores.items(),
        key=lambda x: x[1].abs().mean().item(),
        reverse=True
    )[:top_k]

    for name, scores in sorted_params:
        mean = scores.mean().item()
        std = scores.std().item()
        min_val = scores.min().item()
        max_val = scores.max().item()

        print(f"  {name:<60s} {mean:>12.6f} {std:>12.6f} {min_val:>12.6f} {max_val:>12.6f}")

    print("=" * 60)
