#!/usr/bin/env python3
"""
Script 2: Run All Datasets

Runs the full experiment (all 4 arms) on all 4 datasets.
Creates a combined results file for cross-dataset analysis.

Usage:
    python scripts/2_run_all_datasets.py
    python scripts/2_run_all_datasets.py --use_tpu
    python scripts/2_run_all_datasets.py --datasets sst2 agnews
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.tpu_utils import setup_tpu_environment

# Import from the 1_run_experiment module
import importlib.util
spec = importlib.util.spec_from_file_location("run_experiment", Path(__file__).parent / "1_run_experiment.py")
run_experiment_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_experiment_module)
run_experiment = run_experiment_module.run_experiment
load_config = run_experiment_module.load_config


def run_all_datasets(
    datasets: list = None,
    arms: list = None,
    config_dir: str = "config/datasets",
    splits_dir: str = "outputs/splits",
    checkpoints_dir: str = "outputs/checkpoints",
    results_dir: str = "outputs/results",
    use_tpu: bool = False
):
    """
    Run experiment on all datasets

    Args:
        datasets: List of datasets to run (default: all)
        arms: List of arms to run (default: all)
        config_dir: Directory with dataset configs
        splits_dir: Directory with dataset splits
        checkpoints_dir: Directory with baseline checkpoints
        results_dir: Directory to save results
        use_tpu: Whether using TPU
    """
    if datasets is None:
        datasets = ["sst2", "agnews", "dbpedia", "yelp"]

    if arms is None:
        arms = ['baseline', 'influence', 'lora', 'fullft']

    print("\n" + "=" * 80)
    print("RUNNING ALL DATASETS")
    print("=" * 80)
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Arms: {', '.join(arms)}")
    print(f"Use TPU: {use_tpu}")
    print("=" * 80)

    # Load experiment config
    experiment_config = load_config()

    # Run each dataset
    all_datasets_results = {}
    failed_datasets = []

    for dataset_name in datasets:
        print(f"\n\n{'='*80}")
        print(f"DATASET {datasets.index(dataset_name) + 1}/{len(datasets)}: {dataset_name.upper()}")
        print(f"{'='*80}\n")

        try:
            results = run_experiment(
                dataset_name=dataset_name,
                arms=arms,
                config_dir=config_dir,
                splits_dir=splits_dir,
                checkpoints_dir=checkpoints_dir,
                results_dir=results_dir,
                use_tpu=use_tpu,
                experiment_config=experiment_config
            )

            all_datasets_results[dataset_name] = results

        except Exception as e:
            print(f"\n{'='*80}")
            print(f"ERROR running {dataset_name}")
            print(f"{'='*80}")
            print(f"{str(e)}")
            import traceback
            traceback.print_exc()

            failed_datasets.append({
                'dataset': dataset_name,
                'error': str(e)
            })

    # Save combined results
    combined_results = {
        'timestamp': datetime.now().isoformat(),
        'datasets_run': datasets,
        'arms_run': arms,
        'use_tpu': use_tpu,
        'results': all_datasets_results,
        'failures': failed_datasets
    }

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_file = results_path / f"combined_results_{timestamp}.json"

    with open(combined_file, 'w') as f:
        json.dump(combined_results, f, indent=2)

    # Print final summary
    print("\n\n" + "=" * 80)
    print("ALL DATASETS COMPLETE")
    print("=" * 80)
    print(f"\nCombined results saved to: {combined_file}")

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    # Summary table
    print("\nValidation Accuracy by Dataset and Arm:")
    print(f"\n{'Dataset':<12s} {'Baseline':>10s} {'Influence':>10s} {'LoRA':>10s} {'FullFT':>10s}")
    print("-" * 55)

    for dataset_name in datasets:
        if dataset_name in all_datasets_results:
            results = all_datasets_results[dataset_name]['arms']
            baseline_acc = results.get('baseline', {}).get('val_accuracy', float('nan'))
            influence_acc = results.get('influence', {}).get('val_accuracy', float('nan'))
            lora_acc = results.get('lora', {}).get('val_accuracy', float('nan'))
            fullft_acc = results.get('fullft', {}).get('val_accuracy', float('nan'))

            print(f"{dataset_name:<12s} {baseline_acc:>9.2%} {influence_acc:>9.2%} "
                  f"{lora_acc:>9.2%} {fullft_acc:>9.2%}")
        else:
            print(f"{dataset_name:<12s} {'FAILED':>10s}")

    if failed_datasets:
        print(f"\n\nFailed datasets ({len(failed_datasets)}):")
        for failure in failed_datasets:
            print(f"  - {failure['dataset']}: {failure['error']}")

    print("\n" + "=" * 80)

    num_success = len(all_datasets_results)
    num_total = len(datasets)
    print(f"\nCompleted: {num_success}/{num_total} datasets")

    if num_success == num_total:
        print("\nAll datasets completed successfully!")
        return 0
    else:
        print(f"\nWARNING: {num_total - num_success} datasets failed")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Run influence experiment on all datasets"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        choices=["sst2", "agnews", "dbpedia", "yelp"],
        help="Datasets to run (default: all)"
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        default=None,
        choices=["baseline", "influence", "lora", "fullft"],
        help="Which arms to run (default: all)"
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        default="config/datasets",
        help="Directory containing dataset configs"
    )
    parser.add_argument(
        "--splits_dir",
        type=str,
        default="outputs/splits",
        help="Directory for dataset splits"
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default="outputs/checkpoints",
        help="Directory with baseline checkpoints"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="outputs/results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--use_tpu",
        action="store_true",
        help="Use TPU for training"
    )

    args = parser.parse_args()

    # Setup TPU environment if needed
    if args.use_tpu:
        setup_tpu_environment()

    # Run all datasets
    exit_code = run_all_datasets(
        datasets=args.datasets,
        arms=args.arms,
        config_dir=args.config_dir,
        splits_dir=args.splits_dir,
        checkpoints_dir=args.checkpoints_dir,
        results_dir=args.results_dir,
        use_tpu=args.use_tpu
    )

    return exit_code


if __name__ == "__main__":
    exit(main())
