"""
Arm D: Full Fine-tuning

1. Loads baseline_start checkpoint
2. Unfreezes entire model (backbone + head)
3. Trains on train split with early stopping
4. Evaluates on validation set
"""

import time
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional
from transformers import AutoModelForSequenceClassification

from ..data.utils import create_dataloader
from ..evaluation.evaluator import evaluate_model
from ..models.qwen2_wrapper import freeze_backbone
from ..utils.tpu_utils import (
    get_device, mark_step, optimizer_step,
    is_master_ordinal, print_once
)


def run_fullft_arm(
    checkpoint_path: str,
    train_dataset,
    val_dataset,
    num_labels: int,
    batch_size: int = 128,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.01,
    max_epochs: int = 10,
    early_stop_patience: int = 2,
    early_stop_min_delta: float = 0.001,
    device: Optional[torch.device] = None,
    use_tpu: bool = False,
    model_dtype: str = "bfloat16"
) -> Dict[str, float]:
    """
    Arm D: Full fine-tuning with early stopping

    Args:
        checkpoint_path: Path to baseline_start checkpoint
        train_dataset: Tokenized training dataset (1000 samples)
        val_dataset: Tokenized validation dataset
        num_labels: Number of classification labels
        batch_size: Training batch size
        learning_rate: Learning rate
        weight_decay: Weight decay
        max_epochs: Maximum epochs
        early_stop_patience: Early stopping patience
        early_stop_min_delta: Minimum improvement for early stopping
        device: Device to run on
        use_tpu: Whether using TPU
        model_dtype: Model precision

    Returns:
        Dictionary with metrics and metadata
    """
    print_once("\n" + "=" * 60)
    print_once("ARM D: FULL FINE-TUNING")
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

    # Unfreeze entire model
    print_once("\nUnfreezing entire model...")
    freeze_backbone(model, freeze=False, verbose=True)

    model = model.to(device)

    # Create dataloaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    steps_per_epoch = len(train_loader)

    print_once(f"\nTraining setup:")
    print_once(f"  Training samples: {len(train_dataset)}")
    print_once(f"  Validation samples: {len(val_dataset)}")
    print_once(f"  Batch size: {batch_size}")
    print_once(f"  Steps per epoch: {steps_per_epoch}")
    print_once(f"  Max epochs: {max_epochs}")
    print_once(f"  Learning rate: {learning_rate}")
    print_once(f"  Weight decay: {weight_decay}")
    print_once(f"  Early stop patience: {early_stop_patience}")
    print_once(f"  Early stop min delta: {early_stop_min_delta}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Training loop with early stopping
    best_val_accuracy = 0.0
    epochs_without_improvement = 0
    total_optimizer_steps = 0
    total_tokens = 0

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Optimizer step
            if use_tpu:
                optimizer_step(optimizer, barrier=True)
            else:
                optimizer.step()
                optimizer.zero_grad()

            if use_tpu:
                mark_step()

            # Track metrics
            train_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            train_correct += (preds == batch['labels']).sum().item()
            train_total += batch['labels'].size(0)

            total_optimizer_steps += 1
            total_tokens += batch['input_ids'].numel()

            if is_master_ordinal() and (step + 1) % max(1, steps_per_epoch // 4) == 0:
                avg_loss = train_loss / (step + 1)
                accuracy = train_correct / train_total
                print(f"  Step {step + 1}/{steps_per_epoch}: "
                      f"Loss = {avg_loss:.4f}, Acc = {accuracy:.2%}")

        # Validation
        print_once(f"\nValidating...")
        val_metrics = evaluate_model(
            model=model,
            dataloader=val_loader,
            device=device,
            num_labels=num_labels,
            use_tpu=use_tpu,
            desc=f"Epoch {epoch+1} Val"
        )

        val_accuracy = val_metrics['accuracy']
        val_loss = val_metrics['loss']

        print_once(f"\nEpoch {epoch + 1} results:")
        print_once(f"  Train Loss: {train_loss / steps_per_epoch:.4f}")
        print_once(f"  Train Acc:  {train_correct / train_total:.2%}")
        print_once(f"  Val Loss:   {val_loss:.4f}")
        print_once(f"  Val Acc:    {val_accuracy:.2%}")

        # Early stopping check
        if val_accuracy > best_val_accuracy + early_stop_min_delta:
            print_once(f"  Validation accuracy improved: {best_val_accuracy:.2%} -> {val_accuracy:.2%}")
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print_once(f"  No improvement ({epochs_without_improvement}/{early_stop_patience})")

            if epochs_without_improvement >= early_stop_patience:
                print_once(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

    # Final evaluation
    print_once(f"\nFinal evaluation on validation set...")
    final_metrics = evaluate_model(
        model=model,
        dataloader=val_loader,
        device=device,
        num_labels=num_labels,
        use_tpu=use_tpu,
        desc="Final Eval"
    )

    # Calculate runtime
    wall_time_minutes = (time.time() - start_time) / 60.0

    # Results
    results = {
        'arm': 'fullft',
        'val_accuracy': final_metrics['accuracy'],
        'val_macro_f1': final_metrics['macro_f1'],
        'val_loss': final_metrics['loss'],
        'epochs_run': epoch + 1,
        'optimizer_steps': total_optimizer_steps,
        'tokens_processed': total_tokens,
        'wall_time_minutes': wall_time_minutes,
        'notes': 'Full fine-tuning (entire model)'
    }

    # Print results
    print_once("\n" + "=" * 60)
    print_once("ARM D RESULTS")
    print_once("=" * 60)
    print_once(f"  Epochs run:          {results['epochs_run']}")
    print_once(f"  Optimizer steps:     {results['optimizer_steps']:,}")
    print_once(f"  Validation Accuracy: {results['val_accuracy']:.2%}")
    print_once(f"  Validation Macro F1: {results['val_macro_f1']:.4f}")
    print_once(f"  Validation Loss:     {results['val_loss']:.4f}")
    print_once(f"  Wall Time:           {results['wall_time_minutes']:.2f} minutes")
    print_once("=" * 60)

    return results
