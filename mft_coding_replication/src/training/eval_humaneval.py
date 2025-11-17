"""
HumanEval and HumanEval+ Evaluation

Evaluates code generation models on HumanEval benchmarks.
"""

import torch
from typing import Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def get_device(use_tpu: bool = False):
    """Get appropriate device."""
    if use_tpu:
        import torch_xla.core.xla_model as xm
        return xm.xla_device()
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


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


def evaluate_humaneval(
    model_path: str,
    eval_config: Dict,
    device=None,
    use_tpu: bool = False,
    eval_humaneval_plus: bool = False
) -> Dict[str, float]:
    """
    Evaluate model on HumanEval or HumanEval+.

    Args:
        model_path: Path to model checkpoint
        eval_config: Evaluation configuration
        device: Device to run on
        use_tpu: Whether using TPU
        eval_humaneval_plus: Whether to evaluate on HumanEval+ instead of HumanEval

    Returns:
        Dictionary with evaluation metrics
    """
    print_once("\n" + "=" * 60)
    dataset_name = "HumanEval+" if eval_humaneval_plus else "HumanEval"
    print_once(f"Evaluating on {dataset_name}")
    print_once("=" * 60)

    # Get device
    if device is None:
        device = get_device(use_tpu=use_tpu)

    # Load model and tokenizer
    print_once(f"\nLoading model from: {model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load HumanEval dataset
    print_once(f"\nLoading {dataset_name} dataset...")

    try:
        if eval_humaneval_plus:
            # HumanEval+ is in the evalplus package
            dataset = load_dataset("evalplus/humanevalplus", split="test")
        else:
            # Standard HumanEval
            dataset = load_dataset("openai_humaneval", split="test")
    except Exception as e:
        print_once(f"Warning: Could not load {dataset_name}: {e}")
        print_once("Returning dummy metrics")
        return {
            'pass@1': 0.0,
            'total_problems': 0,
            'note': f'Could not load {dataset_name} dataset'
        }

    print_once(f"  Total problems: {len(dataset)}")

    # Generation config
    temperature = eval_config.get('temperature', 0.2)
    top_p = eval_config.get('top_p', 0.95)
    max_new_tokens = eval_config.get('max_new_tokens', 512)

    print_once(f"\nGeneration config:")
    print_once(f"  Temperature: {temperature}")
    print_once(f"  Top-p: {top_p}")
    print_once(f"  Max new tokens: {max_new_tokens}")

    # Generate completions
    print_once(f"\nGenerating completions...")

    completions = []
    num_problems = len(dataset)

    with torch.no_grad():
        for idx, problem in enumerate(dataset):
            # Get prompt
            prompt = problem.get('prompt', '')

            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id
            )

            # Decode
            completion = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove the prompt from completion
            if completion.startswith(prompt):
                completion = completion[len(prompt):]

            completions.append(completion)

            if (idx + 1) % 10 == 0:
                print_once(f"  Generated {idx + 1}/{num_problems} completions")

    print_once(f"  Generated all {num_problems} completions")

    # Evaluate completions
    # NOTE: Actual pass@k evaluation requires running code execution
    # This is a placeholder - you would need to use the HumanEval evaluation harness
    print_once(f"\nEvaluation:")
    print_once(f"  Note: Full pass@k evaluation requires code execution harness")
    print_once(f"  Completions generated successfully")
    print_once(f"  Use the official HumanEval evaluation script for accurate metrics")

    # Placeholder metrics (would be computed by execution harness)
    metrics = {
        'pass@1': None,  # Requires code execution
        'total_problems': num_problems,
        'completions_generated': len(completions),
        'note': 'Use official HumanEval evaluation harness for pass@1 metric'
    }

    print_once("=" * 60)

    return metrics


def save_humaneval_predictions(
    completions: list,
    save_path: str
):
    """
    Save HumanEval predictions to file for later evaluation.

    Args:
        completions: List of generated code completions
        save_path: Path to save predictions
    """
    import json
    from pathlib import Path

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w') as f:
        json.dump(completions, f, indent=2)

    print_once(f"Predictions saved to: {save_path}")
