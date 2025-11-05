"""
TPU-specific utilities for PyTorch XLA
"""

import os
from typing import Optional


def setup_tpu_environment():
    """
    Set up environment variables for TPU training
    """
    # Increase XLA compilation cache to avoid recompilation
    os.environ.setdefault('XLA_IR_SHAPE_CACHE_SIZE', '2048')
    os.environ.setdefault('XLA_COMPILATION_CACHE_SIZE', '256')

    # Enable async data loading
    os.environ.setdefault('XLA_TENSOR_ALLOCATOR_MAXSIZE', '1000000000')

    print("TPU environment variables set:")
    print(f"  XLA_IR_SHAPE_CACHE_SIZE: {os.environ['XLA_IR_SHAPE_CACHE_SIZE']}")
    print(f"  XLA_COMPILATION_CACHE_SIZE: {os.environ['XLA_COMPILATION_CACHE_SIZE']}")


def get_device(use_tpu: bool = True):
    """
    Get the appropriate device (TPU, CUDA, or CPU)

    Args:
        use_tpu: Whether to use TPU if available

    Returns:
        torch device
    """
    import torch

    if use_tpu:
        try:
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            print(f"Using TPU device: {device}")
            return device
        except ImportError:
            print("torch_xla not available, falling back to CUDA/CPU")

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {device}")
    else:
        device = torch.device('cpu')
        print(f"Using CPU device: {device}")

    return device


def is_master_ordinal():
    """
    Check if current process is master (ordinal 0)
    Used for logging and checkpointing on single core

    Returns:
        bool: True if master process
    """
    try:
        import torch_xla.core.xla_model as xm
        return xm.get_ordinal() == 0
    except:
        return True  # Single-process mode


def mark_step():
    """
    Mark XLA step boundary (TPU synchronization point)
    Safe wrapper that works even without TPU
    """
    try:
        import torch_xla.core.xla_model as xm
        xm.mark_step()
    except ImportError:
        pass  # No-op if not using TPU


def optimizer_step(optimizer, barrier: bool = True):
    """
    Perform optimizer step with TPU synchronization

    Args:
        optimizer: PyTorch optimizer
        barrier: Whether to add synchronization barrier
    """
    try:
        import torch_xla.core.xla_model as xm
        xm.optimizer_step(optimizer, barrier=barrier)
    except ImportError:
        optimizer.step()


def rendezvous(tag: str):
    """
    Synchronize all TPU cores at a rendezvous point

    Args:
        tag: Unique identifier for this rendezvous
    """
    try:
        import torch_xla.core.xla_model as xm
        xm.rendezvous(tag)
    except ImportError:
        pass


def mesh_reduce(tag: str, data, reduce_fn):
    """
    Reduce data across all TPU cores

    Args:
        tag: Unique identifier
        data: Tensor to reduce
        reduce_fn: Reduction function (e.g., lambda x: x.sum())

    Returns:
        Reduced tensor
    """
    try:
        import torch_xla.core.xla_model as xm
        return xm.mesh_reduce(tag, data, reduce_fn)
    except ImportError:
        return data  # No reduction in single-process mode


def save_checkpoint(model, path: str, optimizer=None, epoch: Optional[int] = None):
    """
    Save model checkpoint (only from master process)

    Args:
        model: PyTorch model
        path: Save path
        optimizer: Optional optimizer state
        epoch: Optional epoch number
    """
    if not is_master_ordinal():
        return

    import torch
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if epoch is not None:
        checkpoint['epoch'] = epoch

    torch.save(checkpoint, path)
    print(f"Checkpoint saved to: {path}")


def print_once(msg: str):
    """
    Print message only from master process

    Args:
        msg: Message to print
    """
    if is_master_ordinal():
        print(msg)
