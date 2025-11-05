"""
Baseline Head Trainer

Trains classifier head for 1 epoch on head_init split (500 samples)
with frozen backbone to create baseline_start checkpoint.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from ..data.utils import create_dataloader
from ..utils.tpu_utils import (
    get_device, mark_step, optimizer_step,
    is_master_ordinal, print_once
)


def create_baseline_checkpoint(
    model,
    tokenized_dataset,
    output_dir: str,
    dataset_name: str,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    grad_clip: float = 1.0,
    device: Optional[torch.device] = None,
    use_tpu: bool = False
):
    """
    Train classifier head for 1 epoch on head_init split

    Args:
        model: Qwen2 model with frozen backbone
        tokenized_dataset: Tokenized head_init dataset (500 samples)
        output_dir: Base output directory
        dataset_name: Name of dataset (e.g., "sst2")
        batch_size: Effective batch size (128)
        learning_rate: Learning rate for head training
        weight_decay: Weight decay
        grad_clip: Gradient clipping value
        device: Device to train on
        use_tpu: Whether using TPU

    Returns:
        Trained model, save path
    """
    print_once("\n" + "=" * 60)
    print_once(f"Creating baseline_start checkpoint for {dataset_name}")
    print_once("=" * 60)

    # Get device
    if device is None:
        device = get_device(use_tpu=use_tpu)

    model = model.to(device)

    # Verify backbone is frozen
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print_once(f"\nModel parameter status:")
    print_once(f"  Total parameters: {total_params:,}")
    print_once(f"  Trainable parameters: {trainable_params:,}")
    print_once(f"  Frozen parameters: {total_params - trainable_params:,}")

    if trainable_params > 10_000_000:  # Sanity check: head should be <10M params
        print_once(f"  WARNING: Unexpectedly high trainable params ({trainable_params:,})")
        print_once(f"  Expected only classifier head to be trainable (~1-2M params)")

    # Create dataloader
    # 500 samples / 128 batch = 4 steps per epoch
    dataloader = create_dataloader(
        tokenized_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    steps_per_epoch = len(dataloader)
    print_once(f"\nTraining configuration:")
    print_once(f"  Dataset size: {len(tokenized_dataset)}")
    print_once(f"  Batch size: {batch_size}")
    print_once(f"  Steps per epoch: {steps_per_epoch}")
    print_once(f"  Learning rate: {learning_rate}")
    print_once(f"  Weight decay: {weight_decay}")
    print_once(f"  Gradient clip: {grad_clip}")

    # Optimizer (only for trainable parameters)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Training loop (1 epoch)
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    print_once(f"\nTraining for 1 epoch...")

    for step, batch in enumerate(dataloader):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        logits = outputs.logits

        # Compute loss
        loss = criterion(logits, batch['labels'])

        # Backward pass
        loss.backward()

        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # Optimizer step
        if use_tpu:
            optimizer_step(optimizer, barrier=True)
        else:
            optimizer.step()
            optimizer.zero_grad()

        # Mark step for TPU
        if use_tpu:
            mark_step()

        # Track metrics
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == batch['labels']).sum().item()
        total += batch['labels'].size(0)

        if is_master_ordinal() and (step + 1) % max(1, steps_per_epoch // 4) == 0:
            avg_loss = total_loss / (step + 1)
            accuracy = correct / total
            print(f"  Step {step + 1}/{steps_per_epoch}: "
                  f"Loss = {avg_loss:.4f}, Acc = {accuracy:.2%}")

    # Final metrics
    final_loss = total_loss / steps_per_epoch
    final_accuracy = correct / total

    print_once(f"\nTraining complete:")
    print_once(f"  Final loss: {final_loss:.4f}")
    print_once(f"  Final accuracy: {final_accuracy:.2%}")

    # Save checkpoint (only from master process)
    save_path = Path(output_dir) / dataset_name / "baseline_start"

    if is_master_ordinal():
        save_path.mkdir(parents=True, exist_ok=True)

        # Save model
        model.save_pretrained(str(save_path))

        # Save training info
        import json
        info = {
            'dataset': dataset_name,
            'training_samples': len(tokenized_dataset),
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'final_loss': final_loss,
            'final_accuracy': final_accuracy,
            'trainable_params': trainable_params,
            'total_params': total_params
        }

        with open(save_path / 'training_info.json', 'w') as f:
            json.dump(info, f, indent=2)

        print(f"\nCheckpoint saved to: {save_path}")

    # Synchronize all processes
    if use_tpu:
        from ..utils.tpu_utils import rendezvous
        rendezvous(f"baseline_save_{dataset_name}")

    return model, str(save_path)


if __name__ == "__main__":
    # Test head training
    print("Testing baseline head training...")

    from ..data.loader import load_and_split_dataset
    from ..data.utils import tokenize_dataset, get_tokenizer
    from .qwen2_wrapper import load_qwen2_model, freeze_backbone

    # Load data
    head_init, _, _, config = load_and_split_dataset(
        "sst2",
        config_dir="../../config/datasets",
        output_dir="../../outputs/splits"
    )

    # Take small subset for testing
    head_init_small = head_init.select(range(32))

    # Tokenize
    tokenizer = get_tokenizer()
    tokenized = tokenize_dataset(
        head_init_small,
        config.text_column,
        config.label_column,
        tokenizer
    )

    # Load model
    device = torch.device('cpu')
    model = load_qwen2_model(
        num_labels=config.num_labels,
        device=device,
        dtype=torch.float32
    )

    # Freeze backbone
    freeze_backbone(model, freeze=True)

    # Train
    model, save_path = create_baseline_checkpoint(
        model=model,
        tokenized_dataset=tokenized,
        output_dir="../../outputs/checkpoints",
        dataset_name="sst2_test",
        batch_size=16,
        learning_rate=1e-3,
        device=device,
        use_tpu=False
    )

    print(f"\nTest completed! Checkpoint saved to: {save_path}")
