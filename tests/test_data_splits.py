"""
Test data loading and splitting functionality
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import (
    load_and_split_dataset,
    stratified_split_indices,
    check_stratification,
    DatasetConfig
)
from src.data.utils import (
    tokenize_dataset,
    get_tokenizer,
    verify_dataset_integrity
)


class TestStratifiedSplit:
    """Test stratified splitting logic"""

    def test_stratified_split_sizes(self):
        """Verify split sizes are correct"""
        # Create fake labels
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1] * 200)  # 2000 balanced labels
        np.random.shuffle(labels)

        head_init_idx, train_idx, remaining_idx = stratified_split_indices(
            labels,
            sizes=(500, 1000),
            seed=42
        )

        assert len(head_init_idx) == 500
        assert len(train_idx) == 1000
        assert len(remaining_idx) == 500

    def test_stratified_split_no_overlap(self):
        """Verify no overlap between splits"""
        labels = np.array([0, 0, 0, 1, 1, 1] * 300)
        np.random.shuffle(labels)

        head_init_idx, train_idx, remaining_idx = stratified_split_indices(
            labels,
            sizes=(500, 1000),
            seed=42
        )

        # Check no overlap
        assert len(set(head_init_idx) & set(train_idx)) == 0
        assert len(set(head_init_idx) & set(remaining_idx)) == 0
        assert len(set(train_idx) & set(remaining_idx)) == 0

    def test_stratified_split_deterministic(self):
        """Verify same seed produces same splits"""
        labels = np.array([0, 1, 2, 3] * 500)
        np.random.shuffle(labels)

        h1, t1, r1 = stratified_split_indices(labels, sizes=(500, 1000), seed=42)
        h2, t2, r2 = stratified_split_indices(labels, sizes=(500, 1000), seed=42)

        assert np.array_equal(h1, h2)
        assert np.array_equal(t1, t2)
        assert np.array_equal(r1, r2)

    def test_stratified_split_preserves_distribution(self):
        """Verify class distribution is preserved"""
        # Imbalanced labels
        labels = np.array([0] * 800 + [1] * 1200)  # 40% class 0, 60% class 1
        np.random.shuffle(labels)

        head_init_idx, train_idx, _ = stratified_split_indices(
            labels,
            sizes=(500, 1000),
            seed=42
        )

        # Check head_init distribution
        head_labels = labels[head_init_idx]
        head_class0_ratio = (head_labels == 0).mean()
        assert abs(head_class0_ratio - 0.4) < 0.05  # Within 5%

        # Check train distribution
        train_labels = labels[train_idx]
        train_class0_ratio = (train_labels == 0).mean()
        assert abs(train_class0_ratio - 0.4) < 0.05


class TestDatasetLoading:
    """Test dataset loading from HuggingFace"""

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary directory for test outputs"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_load_sst2(self, temp_output_dir):
        """Test loading SST-2 dataset"""
        head_init, train, val, config = load_and_split_dataset(
            "sst2",
            config_dir="config/datasets",
            output_dir=temp_output_dir,
            seed=42
        )

        assert len(head_init) == 500
        assert len(train) == 1000
        assert len(val) > 0

        assert config.num_labels == 2
        assert config.text_column == "sentence"
        assert config.label_column == "label"

    def test_cached_splits_are_loaded(self, temp_output_dir):
        """Test that cached splits are reused"""
        # First load
        _, _, _, _ = load_and_split_dataset(
            "sst2",
            config_dir="config/datasets",
            output_dir=temp_output_dir,
            seed=42
        )

        # Check files exist
        split_dir = Path(temp_output_dir) / "sst2" / "seed42"
        assert (split_dir / "head500.txt").exists()
        assert (split_dir / "train1000.txt").exists()

        # Second load (should use cache)
        head_init2, train2, _, _ = load_and_split_dataset(
            "sst2",
            config_dir="config/datasets",
            output_dir=temp_output_dir,
            seed=42,
            force_resplit=False
        )

        # Should be same splits
        assert len(head_init2) == 500
        assert len(train2) == 1000

    def test_force_resplit(self, temp_output_dir):
        """Test force_resplit flag"""
        # First load
        head_init1, _, _, _ = load_and_split_dataset(
            "sst2",
            config_dir="config/datasets",
            output_dir=temp_output_dir,
            seed=42
        )

        # Force resplit
        head_init2, _, _, _ = load_and_split_dataset(
            "sst2",
            config_dir="config/datasets",
            output_dir=temp_output_dir,
            seed=42,
            force_resplit=True
        )

        # Should be same since seed is same
        assert len(head_init1) == len(head_init2)


class TestTokenization:
    """Test tokenization functionality"""

    @pytest.fixture
    def tokenizer(self):
        """Load Qwen2 tokenizer"""
        return get_tokenizer("Qwen/Qwen2-0.5B")

    def test_get_tokenizer(self, tokenizer):
        """Test tokenizer loading"""
        assert tokenizer is not None
        assert tokenizer.pad_token is not None
        assert tokenizer.pad_token_id is not None

    def test_tokenize_dataset(self, tokenizer, temp_output_dir):
        """Test dataset tokenization"""
        head_init, _, _, config = load_and_split_dataset(
            "sst2",
            config_dir="config/datasets",
            output_dir=temp_output_dir,
            seed=42
        )

        tokenized = tokenize_dataset(
            head_init,
            config.text_column,
            config.label_column,
            tokenizer,
            max_length=256
        )

        assert 'input_ids' in tokenized.column_names
        assert 'attention_mask' in tokenized.column_names
        assert 'labels' in tokenized.column_names

        # Check shapes
        assert tokenized[0]['input_ids'].shape == (256,)
        assert tokenized[0]['attention_mask'].shape == (256,)

    def test_verify_dataset_integrity(self, tokenizer, temp_output_dir):
        """Test dataset integrity verification"""
        head_init, train, val, config = load_and_split_dataset(
            "sst2",
            config_dir="config/datasets",
            output_dir=temp_output_dir,
            seed=42
        )

        # Tokenize
        head_init = tokenize_dataset(head_init, config.text_column, config.label_column, tokenizer)
        train = tokenize_dataset(train, config.text_column, config.label_column, tokenizer)
        val = tokenize_dataset(val, config.text_column, config.label_column, tokenizer)

        # Should not raise
        verify_dataset_integrity(head_init, train, val, config)


class TestDatasetConfig:
    """Test DatasetConfig class"""

    def test_load_config_from_yaml(self):
        """Test loading config from YAML"""
        config = DatasetConfig.from_yaml("config/datasets/sst2.yaml")

        assert config.dataset_name == "sst2"
        assert config.num_labels == 2
        assert config.text_column == "sentence"
        assert config.label_column == "label"

    def test_all_dataset_configs_valid(self):
        """Test all dataset configs can be loaded"""
        datasets = ["sst2", "agnews", "dbpedia", "yelp"]

        for dataset_name in datasets:
            config = DatasetConfig.from_yaml(f"config/datasets/{dataset_name}.yaml")
            assert config.dataset_name == dataset_name
            assert config.num_labels > 0
            assert config.text_column is not None
            assert config.label_column is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
