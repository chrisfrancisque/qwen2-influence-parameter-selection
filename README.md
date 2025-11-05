# Qwen2-0.5B Influence-Based Parameter Selection Experiment

Adapting influence-based data selection (from ICLR 2024 paper "What Data Benefits My Classifier?") to parameter-level pruning.

## Overview

This experiment compares 4 approaches on Qwen2-0.5B across 4 text classification datasets:

- **Arm A (Baseline)**: Pretrained backbone + 1-epoch trained head (no adaptation)
- **Arm B (Influence)**: Static masking of 5% "detrimental" parameters using sign-corrected gradients
- **Arm C (LoRA)**: Low-rank adaptation (rank=8) with early stopping
- **Arm D (Full FT)**: Full fine-tuning with early stopping

## Datasets

1. **SST-2**: Binary sentiment (2 classes)
2. **AG News**: News topics (4 classes)
3. **DBPedia**: Wikipedia categories (14 classes)
4. **Yelp Review Full**: Star ratings (5 classes)

Each dataset split:
- **Head init**: 500 stratified samples (for baseline head training)
- **Train**: 1000 stratified samples (for methods)
- **Val**: Standard HuggingFace validation split

## Key Innovation

**Influence Computation**: `score_i = sign(θ_i) × ∂L/∂θ_i`

- Computed as **single gradient snapshot** on full 1000-sample batch
- No averaging across batches (unlike traditional influence functions)
- Selects bottom 5% of scores (most negative) for masking
- Only masks attention (Q/K/V/O) and MLP (gate/up/down) weights

## Project Structure

```
qwen2_influence_experiment/
├── config/                     # YAML configurations
│   ├── experiment.yaml         # Global experiment config
│   └── datasets/               # Per-dataset configs
├── src/
│   ├── data/                   # Dataset loading & splitting
│   ├── models/                 # Model wrappers
│   ├── training/               # Trainers for each arm (A/B/C/D)
│   ├── influence/              # Influence computation & parameter selection
│   ├── evaluation/             # Metrics (accuracy, F1)
│   └── utils/                  # TPU utilities
├── scripts/
│   ├── 0_create_baselines.py   # Create baseline checkpoints
│   ├── 1_run_experiment.py     # Run single dataset
│   ├── 2_run_all_datasets.py   # Run all datasets
│   ├── 3_analyze_results.py    # Generate plots & analysis
│   └── quick_test.py           # Quick integration test
├── tests/
│   └── test_data_splits.py     # Unit tests
└── outputs/
    ├── checkpoints/            # Baseline checkpoints
    ├── splits/                 # Cached dataset splits
    ├── results/                # JSON results files
    └── plots/                  # Analysis plots
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For TPU support
pip install torch-xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
```

## Usage

### Quick Start (Recommended)

Test the pipeline on a small dataset first:

```bash
# Step 0: Create baseline checkpoint for SST-2
python scripts/0_create_baselines.py --datasets sst2

# Step 1: Quick test (50 samples)
python scripts/quick_test.py --dataset sst2

# Step 2: Run full experiment on SST-2
python scripts/1_run_experiment.py --dataset sst2

# Step 3: Run all datasets
python scripts/2_run_all_datasets.py
```

### Detailed Workflow

#### 1. Create Baseline Checkpoints

```bash
# Create for all datasets
python scripts/0_create_baselines.py --datasets sst2 agnews dbpedia yelp

# Or create for single dataset
python scripts/0_create_baselines.py --datasets sst2

# With TPU
python scripts/0_create_baselines.py --datasets sst2 --use_tpu
```

This creates `baseline_start` checkpoints (frozen Qwen2-0.5B + 1-epoch trained head).

#### 2. Run Experiment (Single Dataset)

```bash
# Run all arms for SST-2
python scripts/1_run_experiment.py --dataset sst2

# Run specific arms only
python scripts/1_run_experiment.py --dataset sst2 --arms baseline influence

# With TPU
python scripts/1_run_experiment.py --dataset sst2 --use_tpu
```

This runs all 4 arms (Baseline, Influence, LoRA, Full FT) and saves results.

#### 3. Run All Datasets

```bash
# Run all datasets with all arms
python scripts/2_run_all_datasets.py

# Run subset of datasets
python scripts/2_run_all_datasets.py --datasets sst2 agnews

# With TPU
python scripts/2_run_all_datasets.py --use_tpu
```

Runs all 16 conditions (4 datasets × 4 arms) and creates combined results.

#### 4. Analyze Results

```bash
# Analyze most recent results
python scripts/3_analyze_results.py

# Analyze specific results file
python scripts/3_analyze_results.py --results outputs/results/combined_results_20240101_120000.json
```

Generates comparison plots and statistical analysis.

## Expected Runtime

- **Per-condition**: 5 min (Baseline) to 2.5 hours (Full FT)
- **Total (16 conditions)**: ~45 hours
- **With 4-core TPU parallelism**: ~12-15 hours wall-clock
- **Cost**: ~$140 (TPU v5p)

## Expected Results

| Dataset  | Baseline | Influence | LoRA  | Full FT |
|----------|----------|-----------|-------|---------|
| SST-2    | 72%      | 72±1%     | 84%   | 86.5%   |
| AG News  | 68%      | 68±1%     | 82%   | 84%     |
| DBPedia  | 64%      | 64±1%     | 79%   | 81.5%   |
| Yelp     | 70%      | 70±1%     | 83%   | 85%     |

**Hypothesis**: If Influence > Baseline, masking successfully removed detrimental parameters.

## Configuration

Edit `config/experiment.yaml` to adjust:
- Mask fraction (default: 5%)
- LoRA rank/alpha
- Learning rates
- Early stopping patience
- Include/exclude patterns for masking

## Testing

```bash
# Run unit tests
pytest tests/

# Quick integration test (50 samples per split)
python scripts/quick_test.py --dataset sst2

# Single dataset test (1000 train samples)
python scripts/1_run_experiment.py --dataset sst2 --arms baseline influence
```

## Citation

```bibtex
@inproceedings{chhabra2024what,
  title={What Data Benefits My Classifier? Enhancing Model Performance and Interpretability through Influence-Based Data Selection},
  author={Chhabra, Anshuman and Li, Peizhao and Mohapatra, Prasant and Liu, Hongfu},
  booktitle={ICLR},
  year={2024}
}
```

## License

MIT License
