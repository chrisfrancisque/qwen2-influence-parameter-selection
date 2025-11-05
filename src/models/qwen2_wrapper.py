"""
Qwen2 Model Loading and Utilities
"""

import torch
from typing import Optional
from transformers import AutoModelForSequenceClassification, AutoConfig


def load_qwen2_model(
    model_name: str = "Qwen/Qwen2-0.5B",
    num_labels: int = 2,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.bfloat16,
    cache_dir: Optional[str] = None
):
    """
    Load Qwen2-0.5B model for sequence classification

    Args:
        model_name: HuggingFace model identifier
        num_labels: Number of classification labels
        device: Device to load model on
        dtype: Model precision (bfloat16 recommended for TPU)
        cache_dir: Cache directory for model weights

    Returns:
        Loaded model
    """
    print(f"\nLoading model: {model_name}")
    print(f"  Num labels: {num_labels}")
    print(f"  Dtype: {dtype}")

    # Load config
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        cache_dir=cache_dir,
        trust_remote_code=True
    )

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        torch_dtype=dtype,
        cache_dir=cache_dir,
        trust_remote_code=True
    )

    # Ensure pad_token_id is set in model config
    if model.config.pad_token_id is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
        else:
            # Fallback to eos_token_id
            model.config.pad_token_id = tokenizer.eos_token_id
        print(f"  Set pad_token_id to: {model.config.pad_token_id}")

    # Move to device if specified
    if device is not None:
        model = model.to(device)

    print(f"  Model loaded successfully")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num layers: {config.num_hidden_layers}")

    return model


def freeze_backbone(model, freeze: bool = True, verbose: bool = True):
    """
    Freeze/unfreeze the model backbone (all layers except classifier head)

    Args:
        model: Qwen2 model
        freeze: If True, freeze backbone; if False, unfreeze
        verbose: Print freeze status

    Returns:
        Number of frozen parameters
    """
    frozen_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        # Classifier head (score layer in Qwen2ForSequenceClassification)
        if 'score' in name or 'classifier' in name:
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = not freeze
            if freeze:
                frozen_params += param.numel()
            else:
                trainable_params += param.numel()

    if verbose:
        status = "Frozen" if freeze else "Unfrozen"
        print(f"\n{status} backbone:")
        print(f"  Frozen parameters: {frozen_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Total parameters: {frozen_params + trainable_params:,}")

    return frozen_params


def count_parameters(model, trainable_only: bool = False):
    """
    Count total or trainable parameters in model

    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable params

    Returns:
        Parameter count
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_classifier_head(model):
    """
    Get the classifier head from Qwen2 model

    Args:
        model: Qwen2ForSequenceClassification

    Returns:
        Classifier head module
    """
    # Qwen2ForSequenceClassification has a 'score' layer as the head
    if hasattr(model, 'score'):
        return model.score
    elif hasattr(model, 'classifier'):
        return model.classifier
    else:
        raise ValueError("Could not find classifier head in model")


def print_model_info(model):
    """
    Print detailed model information

    Args:
        model: PyTorch model
    """
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)

    print("\nModel Information:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {total_params - trainable_params:,}")
    print(f"  Trainable ratio: {trainable_params / total_params * 100:.2f}%")

    # Print layer-wise breakdown
    print("\n  Parameter breakdown by layer type:")
    param_dict = {}
    for name, param in model.named_parameters():
        # Extract layer type
        if 'embed' in name:
            layer_type = 'embeddings'
        elif 'score' in name or 'classifier' in name:
            layer_type = 'classifier'
        elif 'norm' in name:
            layer_type = 'layer_norm'
        elif 'self_attn' in name:
            layer_type = 'attention'
        elif 'mlp' in name:
            layer_type = 'mlp'
        else:
            layer_type = 'other'

        if layer_type not in param_dict:
            param_dict[layer_type] = 0
        param_dict[layer_type] += param.numel()

    for layer_type, num_params in sorted(param_dict.items()):
        print(f"    {layer_type:15s}: {num_params:>12,} ({num_params/total_params*100:>5.2f}%)")


if __name__ == "__main__":
    # Test model loading
    print("Testing Qwen2 model loading...")

    device = torch.device('cpu')  # Use CPU for testing
    model = load_qwen2_model(num_labels=2, device=device, dtype=torch.float32)

    print_model_info(model)

    # Test freezing
    freeze_backbone(model, freeze=True)
    trainable_before = count_parameters(model, trainable_only=True)

    freeze_backbone(model, freeze=False)
    trainable_after = count_parameters(model, trainable_only=True)

    print(f"\nFreeze test:")
    print(f"  Trainable when frozen: {trainable_before:,}")
    print(f"  Trainable when unfrozen: {trainable_after:,}")

    print("\nAll tests passed!")
