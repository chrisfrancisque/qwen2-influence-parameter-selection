"""
Parameter Selection for Masking

Selects parameters to mask based on influence scores:
1. Filter to INCLUDE only attention and MLP parameters
2. EXCLUDE embeddings, norms, classifier, rotary_emb
3. Sort by influence score (ascending)
4. Select bottom k% (most negative scores)
"""

import torch
from typing import Dict, OrderedDict, Set, List, Tuple
from collections import OrderedDict as ODict

from ..utils.tpu_utils import is_master_ordinal, print_once


# Default INCLUDE patterns (from experiment config)
DEFAULT_INCLUDE_PATTERNS = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj"
]

# Default EXCLUDE patterns (from experiment config)
DEFAULT_EXCLUDE_PATTERNS = [
    "embed_tokens",
    "norm",
    "score",
    "classifier",
    "rotary_emb"
]


def should_include_parameter(
    param_name: str,
    include_patterns: List[str] = None,
    exclude_patterns: List[str] = None
) -> bool:
    """
    Determine if a parameter should be included for masking

    Args:
        param_name: Full parameter name (e.g., "model.layers.0.self_attn.q_proj.weight")
        include_patterns: List of substrings that must be present
        exclude_patterns: List of substrings that exclude the parameter

    Returns:
        True if parameter should be included for masking
    """
    if include_patterns is None:
        include_patterns = DEFAULT_INCLUDE_PATTERNS

    if exclude_patterns is None:
        exclude_patterns = DEFAULT_EXCLUDE_PATTERNS

    # First check exclude patterns (higher priority)
    for pattern in exclude_patterns:
        if pattern in param_name:
            return False

    # Then check include patterns
    for pattern in include_patterns:
        if pattern in param_name:
            return True

    # If no include pattern matched, exclude by default
    return False


def select_parameters_to_mask(
    influence_scores: OrderedDict[str, torch.Tensor],
    mask_fraction: float = 0.05,
    include_patterns: List[str] = None,
    exclude_patterns: List[str] = None
) -> Dict[str, torch.Tensor]:
    """
    Select parameters to mask based on influence scores

    Strategy:
    1. Filter parameters using include/exclude patterns
    2. Flatten all eligible parameters into single array with tracking
    3. Sort by influence score (ascending)
    4. Select bottom k% (most negative scores)
    5. Return mask dict mapping param_name -> boolean mask tensor

    Args:
        influence_scores: OrderedDict of param_name -> influence score tensor
        mask_fraction: Fraction of parameters to mask (default 0.05 = 5%)
        include_patterns: Patterns for parameters to consider
        exclude_patterns: Patterns for parameters to exclude

    Returns:
        Dict mapping param_name -> boolean mask (True = mask this element)
    """
    print_once("\n" + "=" * 60)
    print_once("Selecting Parameters to Mask")
    print_once("=" * 60)

    if include_patterns is None:
        include_patterns = DEFAULT_INCLUDE_PATTERNS

    if exclude_patterns is None:
        exclude_patterns = DEFAULT_EXCLUDE_PATTERNS

    print_once(f"\nInclude patterns: {include_patterns}")
    print_once(f"Exclude patterns: {exclude_patterns}")
    print_once(f"Mask fraction: {mask_fraction:.2%}")

    # Step 1: Filter parameters
    print_once("\nFiltering parameters...")
    eligible_params = ODict()
    excluded_count = 0
    excluded_params_count = 0

    for name, scores in influence_scores.items():
        if should_include_parameter(name, include_patterns, exclude_patterns):
            eligible_params[name] = scores
        else:
            excluded_count += 1
            excluded_params_count += scores.numel()

    total_eligible = sum(s.numel() for s in eligible_params.values())
    total_all = sum(s.numel() for s in influence_scores.values())

    print_once(f"\nEligible parameters:")
    print_once(f"  Eligible tensors: {len(eligible_params)}")
    print_once(f"  Eligible parameters: {total_eligible:,}")
    print_once(f"  Excluded tensors: {excluded_count}")
    print_once(f"  Excluded parameters: {excluded_params_count:,}")
    print_once(f"  Total parameters: {total_all:,}")
    print_once(f"  Eligible fraction: {total_eligible / total_all:.2%}")

    if len(eligible_params) == 0:
        print_once("\nWARNING: No eligible parameters found!")
        return {}

    # Step 2: Flatten all eligible scores with tracking
    print_once("\nFlattening scores for ranking...")

    # Create list of (score_value, param_name, flat_index)
    score_list = []
    param_shapes = {}
    param_flat_ranges = {}

    current_offset = 0
    for name, scores in eligible_params.items():
        flat_scores = scores.flatten()
        param_shapes[name] = scores.shape

        # Track range of indices for this parameter
        param_flat_ranges[name] = (current_offset, current_offset + len(flat_scores))

        # Add to score list
        for i, score_val in enumerate(flat_scores):
            score_list.append((score_val.item(), name, current_offset + i))

        current_offset += len(flat_scores)

    print_once(f"  Total flattened scores: {len(score_list):,}")

    # Step 3: Sort by score (ascending = most negative first)
    print_once("\nSorting scores...")
    score_list.sort(key=lambda x: x[0])

    # Step 4: Select bottom k%
    num_to_mask = int(len(score_list) * mask_fraction)
    print_once(f"\nSelecting bottom {mask_fraction:.2%} = {num_to_mask:,} parameters to mask")

    # Get the threshold score
    if num_to_mask > 0:
        threshold_score = score_list[num_to_mask - 1][0]
        print_once(f"  Threshold score: {threshold_score:.6f}")
        print_once(f"  Min score: {score_list[0][0]:.6f}")
        print_once(f"  Max score: {score_list[-1][0]:.6f}")

    # Step 5: Create mask tensors
    print_once("\nCreating mask tensors...")

    # Initialize mask dict with False (don't mask) for all parameters
    masks = {}
    for name, scores in eligible_params.items():
        masks[name] = torch.zeros_like(scores, dtype=torch.bool)

    # Mark selected parameters as True (mask these)
    for i in range(num_to_mask):
        score_val, param_name, flat_idx = score_list[i]

        # Get the flat range for this parameter
        start_idx, end_idx = param_flat_ranges[param_name]
        local_idx = flat_idx - start_idx

        # Unflatten the index to get original coordinates
        shape = param_shapes[param_name]
        unflat_idx = torch.tensor(local_idx).unravel_index(shape)

        # Set mask
        masks[param_name][unflat_idx] = True

    # Verify mask counts
    total_masked = sum(mask.sum().item() for mask in masks.values())
    print_once(f"\nMask verification:")
    print_once(f"  Expected masked: {num_to_mask:,}")
    print_once(f"  Actually masked: {total_masked:,}")
    print_once(f"  Match: {total_masked == num_to_mask}")

    # Print per-parameter mask statistics
    print_once(f"\nPer-parameter mask statistics:")
    print_once(f"  {'Parameter Name':<60s} {'Total':>10s} {'Masked':>10s} {'%':>8s}")
    print_once("-" * 90)

    for name, mask in masks.items():
        total = mask.numel()
        masked = mask.sum().item()
        pct = (masked / total * 100) if total > 0 else 0

        print_once(f"  {name:<60s} {total:>10,} {masked:>10,} {pct:>7.2f}%")

    print_once("=" * 60)

    return masks


def apply_masks_to_model(
    model,
    masks: Dict[str, torch.Tensor],
    mask_value: float = 0.0
):
    """
    Apply masks to model parameters by setting masked values to mask_value

    Args:
        model: PyTorch model
        masks: Dict of param_name -> boolean mask tensor
        mask_value: Value to set masked parameters to (default 0.0)

    Returns:
        Number of parameters masked
    """
    print_once("\n" + "=" * 60)
    print_once("Applying Masks to Model")
    print_once("=" * 60)

    total_masked = 0

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in masks:
                mask = masks[name].to(param.device)

                # Apply mask
                param.data[mask] = mask_value

                num_masked = mask.sum().item()
                total_masked += num_masked

                print_once(f"  Masked {name}: {num_masked:,} / {param.numel():,} parameters")

    print_once(f"\nTotal parameters masked: {total_masked:,}")
    print_once("=" * 60)

    return total_masked
