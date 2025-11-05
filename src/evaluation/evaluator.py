"""
Model Evaluation Utilities
"""

import torch
import numpy as np
from typing import Dict, Optional
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from ..utils.tpu_utils import mark_step, is_master_ordinal


def compute_metrics(predictions: np.ndarray, labels: np.ndarray, num_labels: int) -> Dict[str, float]:
    """
    Compute evaluation metrics

    Args:
        predictions: Predicted labels
        labels: True labels
        num_labels: Number of classes

    Returns:
        Dictionary of metrics
    """
    accuracy = accuracy_score(labels, predictions)

    # Macro F1 (average across classes)
    if num_labels == 2:
        # Binary classification
        macro_f1 = f1_score(labels, predictions, average='binary')
    else:
        # Multi-class classification
        macro_f1 = f1_score(labels, predictions, average='macro')

    return {
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1)
    }


def evaluate_model(
    model,
    dataloader: DataLoader,
    device: torch.device,
    num_labels: int,
    use_tpu: bool = False,
    desc: str = "Evaluating"
) -> Dict[str, float]:
    """
    Evaluate model on a dataset

    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        num_labels: Number of classification labels
        use_tpu: Whether using TPU
        desc: Description for progress bar

    Returns:
        Dictionary with metrics: accuracy, macro_f1, loss
    """
    model.eval()

    all_predictions = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0

    # Use tqdm only from master process
    iterator = tqdm(dataloader, desc=desc, disable=not is_master_ordinal())

    with torch.no_grad():
        for batch in iterator:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )

            # Get loss and logits
            loss = outputs.loss
            logits = outputs.logits

            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1

            # Get predictions
            predictions = torch.argmax(logits, dim=-1)

            # Move to CPU for metric computation
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())

            # Mark step for TPU
            if use_tpu:
                mark_step()

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Compute metrics
    metrics = compute_metrics(all_predictions, all_labels, num_labels)

    # Add average loss
    metrics['loss'] = total_loss / num_batches

    return metrics


if __name__ == "__main__":
    # Test evaluation
    print("Testing evaluation utilities...")

    # Create fake predictions and labels
    predictions = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    labels = np.array([0, 1, 0, 0, 1, 0, 1, 1])

    metrics = compute_metrics(predictions, labels, num_labels=2)

    print(f"\nTest metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.2%}")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")

    # Expected: 6/8 = 0.75 accuracy
    assert abs(metrics['accuracy'] - 0.75) < 0.01, "Accuracy calculation error"

    print("\nAll tests passed!")
