"""
Parameter Selection for Masking (Targeting Specific Layers)

Selects parameters to mask based on influence scores:
1. Filter to INCLUDE only target layers (15-17 for Qwen2-0.5B coding)
2. Filter to attention and MLP parameters within those layers
3. EXCLUDE embeddings, norms, lm_head, rotary_emb
4. Sort by influence score and select top k% (most positive = most detrimental)
"""

import torch
from typing import Dict, OrderedDict, List
from collections import OrderedDict as ODict


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


def should_include_parameter(
    param_name: str,
    target_layers: List[int],
    include_patterns: List[str],
    exclude_patterns: List[str]
) -> bool:
    """
    Determine if a parameter should be included for masking.

    Args:
        param_name: Full parameter name (e.g., "model.layers.15.self_attn.q_proj.weight")
        target_layers: List of layer indices to target (e.g., [15, 16, 17])
        include_patterns: List of substrings that must be present
        exclude_patterns: List of substrings that exclude the parameter

    Returns:
        True if parameter should be included for masking
    """
    # First check exclude patterns (higher priority)
    for pattern in exclude_patterns:
        if pattern in param_name:
            return False

    # Check if parameter is in target layers
    # Format: "model.layers.{layer_num}.{component}.{param_type}"
    in_target_layer = False
    for layer_idx in target_layers:
        if f"layers.{layer_idx}." in param_name:
            in_target_layer = True
            break

    if not in_target_layer:
        return False

    # Then check include patterns
    for pattern in include_patterns:
        if pattern in param_name:
            return True

    # If no include pattern matched, exclude by default
    return False


def select_parameters_to_mask(
    influence_scores: OrderedDict[str, torch.Tensor],
    mask_fraction: float,
    target_layer_start: int,
    target_layer_end: int,
    include_patterns: List[str],
    exclude_patterns: List[str]
) -> Dict[str, torch.Tensor]:
    """
    Select parameters to mask based on influence scores within target layers.

    Strategy:
    1. Filter to target layers (start to end, inclusive)
    2. Filter parameters using include/exclude patterns
    3. Flatten all eligible parameters
    4. Find threshold using kthvalue for top k% (most positive = most detrimental)
    5. Return mask dict mapping param_name -> boolean mask tensor

    Args:
        influence_scores: OrderedDict of param_name -> influence score tensor
        mask_fraction: Fraction of parameters to mask (e.g., 0.05 = 5%)
        target_layer_start: Start layer index (inclusive)
        target_layer_end: End layer index (inclusive)
        include_patterns: Patterns for parameters to consider
        exclude_patterns: Patterns for parameters to exclude

    Returns:
        Dict mapping param_name -> boolean mask (True = mask this element)
    """
    print_once("\n" + "=" * 60)
    print_once("Selecting Parameters to Mask (Targeted Layers)")
    print_once("=" * 60)

    target_layers = list(range(target_layer_start, target_layer_end + 1))

    print_once(f"\nTarget layers: {target_layers}")
    print_once(f"Include patterns: {include_patterns}")
    print_once(f"Exclude patterns: {exclude_patterns}")
    print_once(f"Mask fraction: {mask_fraction:.2%}")

    # Step 1: Filter parameters
    print_once("\nFiltering parameters...")
    eligible_params = ODict()
    excluded_count = 0
    excluded_params_count = 0

    for name, scores in influence_scores.items():
        if should_include_parameter(name, target_layers, include_patterns, exclude_patterns):
            eligible_params[name] = scores
        else:
            excluded_count += 1
            excluded_params_count += scores.numel()

    total_eligible = sum(s.numel() for s in eligible_params.values())
    total_all = sum(s.numel() for s in influence_scores.values())

    print_once(f"\nEligible parameters (in target layers):")
    print_once(f"  Eligible tensors: {len(eligible_params)}")
    print_once(f"  Eligible parameters: {total_eligible:,}")
    print_once(f"  Excluded tensors: {excluded_count}")
    print_once(f"  Excluded parameters: {excluded_params_count:,}")
    print_once(f"  Total parameters: {total_all:,}")
    print_once(f"  Eligible fraction: {total_eligible / total_all:.2%}")

    if len(eligible_params) == 0:
        print_once("\nWARNING: No eligible parameters found!")
        return {}

    # Step 2: Concatenate all scores (vectorized)
    print_once("\nConcatenating scores from target layers...")

    flattened_scores = []
    for name, scores in eligible_params.items():
        flattened_scores.append(scores.flatten())

    # Concatenate into single tensor
    all_scores = torch.cat(flattened_scores)
    total_params = len(all_scores)

    print_once(f"  Total parameters in target layers: {total_params:,}")

    # Step 3: Find threshold using kthvalue
    num_to_mask = int(total_params * mask_fraction)
    print_once(f"\nSelecting top {mask_fraction:.2%} = {num_to_mask:,} parameters (most detrimental)")

    if num_to_mask == 0:
        print_once("  No parameters to mask!")
        return {}

    # Find the k-th largest value as threshold
    print_once("  Finding threshold...")
    k_largest_index = total_params - num_to_mask + 1
    threshold_score = torch.kthvalue(all_scores, k_largest_index).values.item()

    print_once(f"  Threshold score: {threshold_score:.6f}")
    print_once(f"  Min score: {all_scores.min().item():.6f}")
    print_once(f"  Max score: {all_scores.max().item():.6f}")
    print_once(f"  Median score: {all_scores.median().item():.6f}")

    # Step 4: Create masks by thresholding (vectorized)
    print_once("\nCreating mask tensors...")

    masks = {}
    for name, scores in eligible_params.items():
        # Vectorized comparison: True where score >= threshold (top scores = most detrimental)
        mask = scores >= threshold_score

        num_masked_here = mask.sum().item()
        if num_masked_here > 0:
            masks[name] = mask

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
    Apply masks to model parameters by setting masked values to mask_value.

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
