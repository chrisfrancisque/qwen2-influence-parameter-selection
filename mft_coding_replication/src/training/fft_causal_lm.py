"""
Full Fine-Tuning for Causal Language Modeling

Trains Qwen2-0.5B on coding datasets using causal LM objective.
"""

import time
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional
from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup


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


def optimizer_step(optimizer):
    """Optimizer step with XLA support."""
    try:
        import torch_xla.core.xla_model as xm
        xm.optimizer_step(optimizer)
    except ImportError:
        optimizer.step()


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


def create_dataloader(dataset, batch_size, shuffle=True, drop_last=False):
    """Create DataLoader with proper collation for causal LM."""
    from torch.utils.data import DataLoader

    def collate_fn(examples):
        # Pad sequences to max length in batch
        input_ids = [ex['input_ids'] for ex in examples]
        attention_mask = [ex['attention_mask'] for ex in examples]

        max_len = max(len(ids) for ids in input_ids)

        # Pad with 0 (pad_token_id)
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


def train_causal_lm(
    model_name: str,
    train_dataset,
    config: Dict,
    save_path: str,
    device: Optional[torch.device] = None,
    use_tpu: bool = False
) -> Dict[str, float]:
    """
    Full fine-tuning on causal language modeling task.

    Args:
        model_name: HuggingFace model name (e.g., "Qwen/Qwen2-0.5B")
        train_dataset: Tokenized training dataset
        config: Training configuration dictionary
        save_path: Path to save final checkpoint
        device: Device to run on
        use_tpu: Whether using TPU

    Returns:
        Dictionary with training metrics
    """
    print_once("\n" + "=" * 60)
    print_once("FULL FINE-TUNING (Causal LM)")
    print_once("=" * 60)

    start_time = time.time()

    # Get device
    if device is None:
        device = get_device(use_tpu=use_tpu)

    # Load model
    print_once(f"\nLoading model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print_once(f"  Total parameters: {total_params:,}")
    print_once(f"  Trainable parameters: {trainable_params:,}")

    # Training config
    train_config = config['training']
    batch_size = train_config['batch_size']
    grad_accum_steps = train_config['gradient_accumulation_steps']
    num_epochs = train_config['num_epochs']
    learning_rate = train_config['learning_rate']
    weight_decay = train_config['weight_decay']
    warmup_steps = train_config['warmup_steps']
    max_grad_norm = train_config['max_grad_norm']

    # Create dataloader
    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * num_epochs // grad_accum_steps

    print_once(f"\nTraining setup:")
    print_once(f"  Training samples: {len(train_dataset)}")
    print_once(f"  Batch size: {batch_size}")
    print_once(f"  Gradient accumulation steps: {grad_accum_steps}")
    print_once(f"  Effective batch size: {batch_size * grad_accum_steps}")
    print_once(f"  Steps per epoch: {steps_per_epoch}")
    print_once(f"  Num epochs: {num_epochs}")
    print_once(f"  Total optimization steps: {total_steps}")
    print_once(f"  Learning rate: {learning_rate}")
    print_once(f"  Weight decay: {weight_decay}")
    print_once(f"  Warmup steps: {warmup_steps}")
    print_once(f"  Max grad norm: {max_grad_norm}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Training loop
    global_step = 0
    total_tokens_trained = 0

    for epoch in range(num_epochs):
        print_once(f"\n{'='*60}")
        print_once(f"Epoch {epoch + 1}/{num_epochs}")
        print_once(f"{'='*60}")

        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass (causal LM loss computed internally)
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['input_ids']  # Causal LM: predict next token
            )

            loss = outputs.loss / grad_accum_steps  # Normalize for gradient accumulation
            loss.backward()

            epoch_loss += loss.item() * grad_accum_steps
            total_tokens_trained += batch['input_ids'].numel()

            # Optimizer step every grad_accum_steps
            if (step + 1) % grad_accum_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                # Optimizer step
                optimizer_step(optimizer)
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                # Mark step for TPU
                if use_tpu:
                    mark_step()

                # Logging
                if global_step % train_config.get('logging_steps', 50) == 0:
                    avg_loss = epoch_loss / (step + 1)
                    current_lr = scheduler.get_last_lr()[0]
                    print_once(
                        f"  Step {global_step}/{total_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {current_lr:.2e} | "
                        f"Tokens: {total_tokens_trained:,}"
                    )

        avg_epoch_loss = epoch_loss / len(train_loader)
        print_once(f"\nEpoch {epoch + 1} completed | Avg Loss: {avg_epoch_loss:.4f}")

    training_time = time.time() - start_time

    # Save model
    print_once(f"\nSaving model to: {save_path}")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Move to CPU before saving (for TPU compatibility)
    if use_tpu:
        print_once("  Moving model to CPU for saving...")
        model_cpu = model.cpu()
        model_cpu.save_pretrained(save_path)
    else:
        model.save_pretrained(save_path)

    print_once(f"Model saved successfully!")

    # Return metrics
    metrics = {
        'final_loss': epoch_loss / len(train_loader),
        'training_time_seconds': training_time,
        'total_steps': global_step,
        'total_tokens': total_tokens_trained
    }

    print_once(f"\nTraining completed in {training_time / 60:.2f} minutes")
    print_once("=" * 60)

    return metrics
