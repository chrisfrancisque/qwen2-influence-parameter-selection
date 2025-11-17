"""
Main Orchestration Script for MFT Coding Replication

Pipeline:
1. Load and prepare 3 coding datasets (FFT: 30k, SCI: 1k)
2. Full fine-tune Qwen2-0.5B on 30k samples
3. Evaluate FFT model on HumanEval/HumanEval+
4. Compute SCI scores on 1k holdout samples
5. Mask top 5% parameters in layers 15-17
6. Evaluate SCI-masked model on HumanEval/HumanEval+
7. Compare results
"""

import sys
import yaml
import torch
from pathlib import Path
from transformers import AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.coding_loader import load_coding_datasets
from training.fft_causal_lm import train_causal_lm
from training.eval_humaneval import evaluate_humaneval
from influence.compute_scores import compute_influence_scores, print_influence_statistics
from influence.parameter_selector import select_parameters_to_mask, apply_masks_to_model


def main():
    print("\n" + "=" * 80)
    print("MFT CODING DOMAIN REPLICATION")
    print("Qwen2-0.5B with Sign-Corrected Influence Masking")
    print("=" * 80)

    # Load config
    config_path = Path(__file__).parent.parent / "config" / "experiment.yaml"
    print(f"\nLoading config from: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Get paths
    paths = config['paths']
    output_dir = Path(paths['output_dir'])
    fft_checkpoint = Path(paths['fft_checkpoint'])
    sci_masked_checkpoint = Path(paths['sci_masked_checkpoint'])
    sci_scores_path = Path(paths['sci_scores_path'])

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    fft_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    sci_masked_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    sci_scores_path.parent.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # STEP 1: Load and prepare datasets (BEFORE TPU initialization)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: Loading Coding Datasets (on CPU)")
    print("=" * 80)

    model_name = config['model']['name']
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    fft_dataset, sci_dataset = load_coding_datasets(
        tokenizer=tokenizer,
        config=config,
        seed=config['data']['seed']
    )

    print(f"\nDatasets loaded successfully:")
    print(f"  FFT dataset: {len(fft_dataset)} samples")
    print(f"  SCI dataset: {len(sci_dataset)} samples")

    # =========================================================================
    # Setup device - auto-detect TPU (AFTER data loading)
    # =========================================================================
    use_tpu = False
    device = None

    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        use_tpu = True
        print("\nUsing TPU for training")
    except ImportError:
        if torch.cuda.is_available():
            print("\nUsing CUDA for training")
            device = torch.device('cuda')
        else:
            print("\nUsing CPU for training")
            device = torch.device('cpu')

    # =========================================================================
    # STEP 2: Full Fine-Tuning
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: Full Fine-Tuning on Coding Datasets")
    print("=" * 80)

    fft_metrics = train_causal_lm(
        model_name=model_name,
        train_dataset=fft_dataset,
        config=config,
        save_path=str(fft_checkpoint),
        device=device,
        use_tpu=use_tpu
    )

    print(f"\nFFT Training Metrics:")
    for key, value in fft_metrics.items():
        print(f"  {key}: {value}")

    # =========================================================================
    # STEP 3: Evaluate FFT Model
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: Evaluating FFT Model")
    print("=" * 80)

    print("\n--- HumanEval ---")
    fft_humaneval_metrics = evaluate_humaneval(
        model_path=str(fft_checkpoint),
        eval_config=config['evaluation']['humaneval'],
        device=device,
        use_tpu=use_tpu,
        eval_humaneval_plus=False
    )

    print("\n--- HumanEval+ ---")
    fft_humaneval_plus_metrics = evaluate_humaneval(
        model_path=str(fft_checkpoint),
        eval_config=config['evaluation']['humaneval_plus'],
        device=device,
        use_tpu=use_tpu,
        eval_humaneval_plus=True
    )

    # =========================================================================
    # STEP 4: Compute Sign-Corrected Influence Scores
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: Computing Sign-Corrected Influence Scores")
    print("=" * 80)

    # Load FFT model
    from transformers import AutoModelForCausalLM

    print(f"\nLoading FFT model from: {fft_checkpoint}")
    model = AutoModelForCausalLM.from_pretrained(
        str(fft_checkpoint),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # Compute influence scores
    sci_config = config['sci']
    influence_scores = compute_influence_scores(
        model=model,
        sci_dataset=sci_dataset,
        batch_size=sci_config['gradient_batch_size'],
        device=device,
        use_tpu=use_tpu
    )

    # Print statistics
    print_influence_statistics(influence_scores, top_k=20)

    # Save influence scores
    print(f"\nSaving influence scores to: {sci_scores_path}")
    torch.save(influence_scores, sci_scores_path)

    # =========================================================================
    # STEP 5: Select and Apply Masks (Layers 15-17 only)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: Selecting and Applying Masks (Layers 15-17)")
    print("=" * 80)

    masks = select_parameters_to_mask(
        influence_scores=influence_scores,
        mask_fraction=sci_config['mask_fraction'],
        target_layer_start=sci_config['target_layers']['start'],
        target_layer_end=sci_config['target_layers']['end'],
        include_patterns=sci_config['include_patterns'],
        exclude_patterns=[]  # No global excludes since we filter by layer
    )

    # Apply masks to model
    num_masked = apply_masks_to_model(model, masks, mask_value=0.0)

    # Save masked model
    print(f"\nSaving SCI-masked model to: {sci_masked_checkpoint}")
    sci_masked_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    if use_tpu:
        print("  Moving model to CPU for saving...")
        model_cpu = model.cpu()
        model_cpu.save_pretrained(str(sci_masked_checkpoint))
        tokenizer.save_pretrained(str(sci_masked_checkpoint))
    else:
        model.save_pretrained(str(sci_masked_checkpoint))
        tokenizer.save_pretrained(str(sci_masked_checkpoint))

    # =========================================================================
    # STEP 6: Evaluate SCI-Masked Model
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: Evaluating SCI-Masked Model")
    print("=" * 80)

    print("\n--- HumanEval ---")
    sci_humaneval_metrics = evaluate_humaneval(
        model_path=str(sci_masked_checkpoint),
        eval_config=config['evaluation']['humaneval'],
        device=device,
        use_tpu=use_tpu,
        eval_humaneval_plus=False
    )

    print("\n--- HumanEval+ ---")
    sci_humaneval_plus_metrics = evaluate_humaneval(
        model_path=str(sci_masked_checkpoint),
        eval_config=config['evaluation']['humaneval_plus'],
        device=device,
        use_tpu=use_tpu,
        eval_humaneval_plus=True
    )

    # =========================================================================
    # STEP 7: Compare Results
    # =========================================================================
    print("\n" + "=" * 80)
    print("FINAL RESULTS COMPARISON")
    print("=" * 80)

    print("\n--- Training Metrics ---")
    print(f"FFT Training Time: {fft_metrics['training_time_seconds'] / 60:.2f} minutes")
    print(f"FFT Final Loss: {fft_metrics['final_loss']:.4f}")

    print("\n--- HumanEval Results ---")
    print(f"FFT Model:")
    print(f"  {fft_humaneval_metrics}")
    print(f"\nSCI-Masked Model:")
    print(f"  {sci_humaneval_metrics}")

    print("\n--- HumanEval+ Results ---")
    print(f"FFT Model:")
    print(f"  {fft_humaneval_plus_metrics}")
    print(f"\nSCI-Masked Model:")
    print(f"  {sci_humaneval_plus_metrics}")

    print("\n--- Masking Summary ---")
    print(f"Mask fraction: {sci_config['mask_fraction']:.2%}")
    print(f"Target layers: {sci_config['target_layers']['start']}-{sci_config['target_layers']['end']}")
    print(f"Total parameters masked: {num_masked:,}")

    # Save results summary
    results_file = output_dir / "results" / "summary.yaml"
    results_file.parent.mkdir(parents=True, exist_ok=True)

    results = {
        'fft_metrics': fft_metrics,
        'fft_humaneval': fft_humaneval_metrics,
        'fft_humaneval_plus': fft_humaneval_plus_metrics,
        'sci_humaneval': sci_humaneval_metrics,
        'sci_humaneval_plus': sci_humaneval_plus_metrics,
        'masking_summary': {
            'mask_fraction': sci_config['mask_fraction'],
            'target_layers': f"{sci_config['target_layers']['start']}-{sci_config['target_layers']['end']}",
            'total_masked': num_masked
        }
    }

    with open(results_file, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)

    print(f"\nResults saved to: {results_file}")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
