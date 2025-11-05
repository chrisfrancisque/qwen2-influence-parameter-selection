#!/usr/bin/env python3
"""
Script 3: Analyze Results

Analyzes experiment results and creates comparison plots.

Usage:
    python scripts/3_analyze_results.py --results outputs/results/combined_results_*.json
    python scripts/3_analyze_results.py --results_dir outputs/results
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(results_file: str) -> dict:
    """Load results from JSON file"""
    with open(results_file) as f:
        return json.load(f)


def create_summary_dataframe(results: dict) -> pd.DataFrame:
    """
    Create summary DataFrame from results

    Args:
        results: Combined results dict

    Returns:
        DataFrame with columns: dataset, arm, val_accuracy, val_macro_f1, etc.
    """
    rows = []

    for dataset_name, dataset_results in results['results'].items():
        for arm_name, arm_results in dataset_results['arms'].items():
            row = {
                'dataset': dataset_name,
                'arm': arm_name,
                'val_accuracy': arm_results['val_accuracy'],
                'val_macro_f1': arm_results['val_macro_f1'],
                'val_loss': arm_results['val_loss'],
                'epochs_run': arm_results['epochs_run'],
                'optimizer_steps': arm_results['optimizer_steps'],
                'wall_time_minutes': arm_results['wall_time_minutes']
            }

            # Add arm-specific fields
            if 'num_masked_params' in arm_results:
                row['num_masked_params'] = arm_results['num_masked_params']
                row['mask_fraction'] = arm_results['mask_fraction']

            if 'lora_rank' in arm_results:
                row['lora_rank'] = arm_results['lora_rank']
                row['lora_alpha'] = arm_results['lora_alpha']

            rows.append(row)

    return pd.DataFrame(rows)


def print_summary_table(df: pd.DataFrame):
    """Print summary table of results"""
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    # Pivot table: datasets x arms
    pivot = df.pivot(index='dataset', columns='arm', values='val_accuracy')

    print("\nValidation Accuracy:")
    print(pivot.to_string(float_format=lambda x: f"{x:.2%}"))

    pivot_f1 = df.pivot(index='dataset', columns='arm', values='val_macro_f1')

    print("\n\nValidation Macro F1:")
    print(pivot_f1.to_string(float_format=lambda x: f"{x:.4f}"))

    # Statistics
    print("\n\nStatistics by Arm:")
    print("-" * 80)

    stats = df.groupby('arm')['val_accuracy'].agg(['mean', 'std', 'min', 'max'])
    print("\nAccuracy:")
    for arm in stats.index:
        mean = stats.loc[arm, 'mean']
        std = stats.loc[arm, 'std']
        min_val = stats.loc[arm, 'min']
        max_val = stats.loc[arm, 'max']
        print(f"  {arm:12s}: {mean:.2%} ± {std:.2%}  (range: {min_val:.2%} - {max_val:.2%})")

    # Relative to baseline
    print("\n\nRelative to Baseline:")
    print("-" * 80)

    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        baseline_acc = dataset_df[dataset_df['arm'] == 'baseline']['val_accuracy'].values[0]

        print(f"\n{dataset.upper()}:")
        for arm in ['influence', 'lora', 'fullft']:
            if arm in dataset_df['arm'].values:
                arm_acc = dataset_df[dataset_df['arm'] == arm]['val_accuracy'].values[0]
                diff = arm_acc - baseline_acc
                pct_change = (diff / baseline_acc) * 100

                sign = "+" if diff > 0 else ""
                print(f"  {arm:12s}: {sign}{diff:>7.2%} ({sign}{pct_change:>6.2f}% relative)")

    # Wall time
    print("\n\nWall Time (minutes):")
    print("-" * 80)
    time_stats = df.groupby('arm')['wall_time_minutes'].agg(['mean', 'std'])
    for arm in time_stats.index:
        mean = time_stats.loc[arm, 'mean']
        std = time_stats.loc[arm, 'std']
        print(f"  {arm:12s}: {mean:>7.2f} ± {std:>6.2f}")


def plot_results(df: pd.DataFrame, output_dir: str = "outputs/plots"):
    """
    Create visualization plots

    Args:
        df: Results DataFrame
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")
    sns.set_palette("husl")

    # Plot 1: Accuracy by dataset and arm
    plt.figure(figsize=(12, 6))
    pivot = df.pivot(index='dataset', columns='arm', values='val_accuracy')
    pivot.plot(kind='bar', figsize=(12, 6))
    plt.title('Validation Accuracy by Dataset and Arm', fontsize=14, fontweight='bold')
    plt.xlabel('Dataset', fontsize=12)
    plt.ylabel('Validation Accuracy', fontsize=12)
    plt.legend(title='Arm', fontsize=10)
    plt.xticks(rotation=0)
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(output_path / 'accuracy_by_dataset.png', dpi=300)
    print(f"\nSaved: {output_path / 'accuracy_by_dataset.png'}")

    # Plot 2: F1 by dataset and arm
    plt.figure(figsize=(12, 6))
    pivot_f1 = df.pivot(index='dataset', columns='arm', values='val_macro_f1')
    pivot_f1.plot(kind='bar', figsize=(12, 6))
    plt.title('Validation Macro F1 by Dataset and Arm', fontsize=14, fontweight='bold')
    plt.xlabel('Dataset', fontsize=12)
    plt.ylabel('Validation Macro F1', fontsize=12)
    plt.legend(title='Arm', fontsize=10)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path / 'f1_by_dataset.png', dpi=300)
    print(f"Saved: {output_path / 'f1_by_dataset.png'}")

    # Plot 3: Relative improvement over baseline
    plt.figure(figsize=(12, 6))
    rel_improvements = []

    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        baseline_acc = dataset_df[dataset_df['arm'] == 'baseline']['val_accuracy'].values[0]

        for arm in ['influence', 'lora', 'fullft']:
            if arm in dataset_df['arm'].values:
                arm_acc = dataset_df[dataset_df['arm'] == arm]['val_accuracy'].values[0]
                rel_improvement = ((arm_acc - baseline_acc) / baseline_acc) * 100
                rel_improvements.append({
                    'dataset': dataset,
                    'arm': arm,
                    'relative_improvement': rel_improvement
                })

    rel_df = pd.DataFrame(rel_improvements)
    pivot_rel = rel_df.pivot(index='dataset', columns='arm', values='relative_improvement')
    pivot_rel.plot(kind='bar', figsize=(12, 6))
    plt.title('Relative Improvement over Baseline (%)', fontsize=14, fontweight='bold')
    plt.xlabel('Dataset', fontsize=12)
    plt.ylabel('Relative Improvement (%)', fontsize=12)
    plt.legend(title='Arm', fontsize=10)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path / 'relative_improvement.png', dpi=300)
    print(f"Saved: {output_path / 'relative_improvement.png'}")

    # Plot 4: Wall time comparison
    plt.figure(figsize=(10, 6))
    df.groupby('arm')['wall_time_minutes'].mean().plot(kind='bar')
    plt.title('Average Wall Time by Arm', fontsize=14, fontweight='bold')
    plt.xlabel('Arm', fontsize=12)
    plt.ylabel('Wall Time (minutes)', fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path / 'wall_time.png', dpi=300)
    print(f"Saved: {output_path / 'wall_time.png'}")

    print(f"\nAll plots saved to: {output_path}")


def analyze_results(results_file: str, output_dir: str = "outputs/plots"):
    """
    Main analysis function

    Args:
        results_file: Path to combined results JSON
        output_dir: Directory to save plots
    """
    print("\n" + "=" * 80)
    print("ANALYZING RESULTS")
    print("=" * 80)
    print(f"\nResults file: {results_file}")

    # Load results
    results = load_results(results_file)

    print(f"\nDatasets: {', '.join(results['results'].keys())}")
    print(f"Arms: {results['arms_run']}")
    print(f"Timestamp: {results['timestamp']}")

    # Create DataFrame
    df = create_summary_dataframe(results)

    # Print summary
    print_summary_table(df)

    # Create plots
    print("\n" + "=" * 80)
    print("CREATING PLOTS")
    print("=" * 80)

    plot_results(df, output_dir)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze experiment results"
    )
    parser.add_argument(
        "--results",
        type=str,
        help="Path to combined results JSON file"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="outputs/results",
        help="Directory to search for latest results file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/plots",
        help="Directory to save plots"
    )

    args = parser.parse_args()

    # Find results file
    if args.results:
        results_file = args.results
    else:
        # Find most recent combined results file
        results_path = Path(args.results_dir)
        combined_files = sorted(results_path.glob("combined_results_*.json"))

        if not combined_files:
            print(f"ERROR: No combined results files found in {args.results_dir}")
            print("Please run scripts/2_run_all_datasets.py first")
            return 1

        results_file = combined_files[-1]
        print(f"Using most recent results file: {results_file}")

    if not Path(results_file).exists():
        print(f"ERROR: Results file not found: {results_file}")
        return 1

    # Analyze
    analyze_results(results_file, args.output_dir)

    return 0


if __name__ == "__main__":
    exit(main())
