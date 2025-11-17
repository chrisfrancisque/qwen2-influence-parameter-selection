"""
Data loader for coding datasets.
Loads 3 datasets, applies chat template formatting, and splits into FFT and SCI sets.
"""

import random
from typing import Dict, List, Tuple
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import PreTrainedTokenizer


def convert_to_messages(example: Dict, dataset_name: str) -> Dict:
    """
    Convert dataset examples to messages format.

    Args:
        example: Dictionary with dataset fields
        dataset_name: Name of the dataset to determine conversion strategy

    Returns:
        Dictionary with 'messages' field added
    """
    if dataset_name == "evol_code_alpaca":
        # theblackcat102/evol-codealpaca-v1 has: instruction, input, output
        if example.get('input', '').strip():
            user_content = f"{example['instruction']}\n\nInput:\n{example['input']}"
        else:
            user_content = example['instruction']

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": example['output']}
        ]

    elif dataset_name == "code_alpaca":
        # sahil2801/CodeAlpaca-20k has: instruction, input, output
        if example.get('input', '').strip():
            user_content = f"{example['instruction']}\n\nInput:\n{example['input']}"
        else:
            user_content = example['instruction']

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": example['output']}
        ]

    elif dataset_name == "tulu3_persona_python":
        # allenai/tulu-3-sft-personas-math-grade
        # This dataset likely has 'messages' already, or we need to check format
        # For now, assuming it has instruction/response or similar
        if 'messages' in example:
            messages = example['messages']
        else:
            # Fallback: try to construct from available fields
            user_content = example.get('instruction', example.get('prompt', ''))
            assistant_content = example.get('response', example.get('completion', ''))
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    example['messages'] = messages
    return example


def apply_chat_template_to_example(
    example: Dict,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048
) -> Dict:
    """
    Apply chat template and tokenize.

    Args:
        example: Dictionary with 'messages' field
        tokenizer: HuggingFace tokenizer with chat template
        max_length: Maximum sequence length

    Returns:
        Dictionary with tokenized fields
    """
    # Apply chat template (returns string)
    formatted_text = tokenizer.apply_chat_template(
        example['messages'],
        tokenize=False,
        add_generation_prompt=False
    )

    # Tokenize
    tokenized = tokenizer(
        formatted_text,
        truncation=True,
        max_length=max_length,
        padding=False,  # We'll pad in the DataLoader
        return_tensors=None
    )

    return {
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask']
    }


def load_coding_datasets(
    tokenizer: PreTrainedTokenizer,
    config: Dict,
    seed: int = 42
) -> Tuple[Dataset, Dataset]:
    """
    Load and prepare coding datasets for FFT and SCI.

    Args:
        tokenizer: HuggingFace tokenizer
        config: Configuration dictionary with data settings
        seed: Random seed for reproducibility

    Returns:
        Tuple of (fft_dataset, sci_dataset)
    """
    random.seed(seed)

    data_config = config['data']
    fft_samples_per_dataset = data_config['fft_samples_per_dataset']
    sci_samples_per_dataset = data_config['sci_samples_per_dataset']
    max_seq_length = data_config['max_seq_length']

    all_fft_datasets = []
    all_sci_datasets = []

    # Load each dataset
    for dataset_name, dataset_info in data_config['datasets'].items():
        print(f"\nLoading {dataset_name}...")

        # Load from HuggingFace
        hf_name = dataset_info['hf_name']
        split = dataset_info.get('split', 'train')

        try:
            dataset = load_dataset(hf_name, split=split)
        except Exception as e:
            print(f"Warning: Could not load {hf_name}: {e}")
            print(f"Skipping {dataset_name}")
            continue

        # Convert to messages format
        print(f"Converting {dataset_name} to messages format...")
        dataset = dataset.map(
            lambda x: convert_to_messages(x, dataset_name),
            desc=f"Converting {dataset_name}"
        )

        # Sample FFT and SCI splits (non-overlapping)
        total_needed = fft_samples_per_dataset + sci_samples_per_dataset

        if len(dataset) < total_needed:
            print(f"Warning: {dataset_name} has only {len(dataset)} samples, needed {total_needed}")
            print(f"Using all available samples")
            # Use proportional split
            sci_count = int(len(dataset) * sci_samples_per_dataset / total_needed)
            fft_count = len(dataset) - sci_count
        else:
            sci_count = sci_samples_per_dataset
            fft_count = fft_samples_per_dataset

        # Shuffle and split
        dataset = dataset.shuffle(seed=seed)

        sci_dataset = dataset.select(range(sci_count))
        fft_dataset = dataset.select(range(sci_count, sci_count + fft_count))

        print(f"  FFT samples: {len(fft_dataset)}")
        print(f"  SCI samples: {len(sci_dataset)}")

        # Apply chat template and tokenize (batched to avoid XLA compilation issues)
        print(f"Tokenizing {dataset_name}...")
        fft_dataset = fft_dataset.map(
            lambda x: apply_chat_template_to_example(x, tokenizer, max_seq_length),
            remove_columns=fft_dataset.column_names,
            batched=False,  # Process one at a time to avoid XLA issues
            desc=f"Tokenizing FFT {dataset_name}"
        )

        sci_dataset = sci_dataset.map(
            lambda x: apply_chat_template_to_example(x, tokenizer, max_seq_length),
            remove_columns=sci_dataset.column_names,
            batched=False,  # Process one at a time to avoid XLA issues
            desc=f"Tokenizing SCI {dataset_name}"
        )

        all_fft_datasets.append(fft_dataset)
        all_sci_datasets.append(sci_dataset)

    # Concatenate all datasets
    print("\nConcatenating datasets...")
    fft_combined = concatenate_datasets(all_fft_datasets)
    sci_combined = concatenate_datasets(all_sci_datasets)

    # Shuffle combined datasets
    fft_combined = fft_combined.shuffle(seed=seed)
    sci_combined = sci_combined.shuffle(seed=seed)

    print(f"\nFinal dataset sizes:")
    print(f"  FFT: {len(fft_combined)} samples")
    print(f"  SCI: {len(sci_combined)} samples")

    return fft_combined, sci_combined
