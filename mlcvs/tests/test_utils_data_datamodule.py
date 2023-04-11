#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Tests for the members of the mlcvs.data.datamodule module.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import pytest
import torch

from mlcvs.data.dataset import DictionaryDataset
from mlcvs.data.datamodule import DictionaryDataModule


# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.parametrize('lengths', [[40, 10], [0.8, 0.2], [0.7, 0.2, 0.1]])
@pytest.mark.parametrize('fields', [[], ['labels', 'weights']])
@pytest.mark.parametrize('random_split', [True, False])
def test_dictionary_data_module_split(lengths, fields, random_split):
    """The datamodule splits the dataset into train/val/test subsets.

    Tests that:
    * The sub-datasets have the correct lengths.
    * The dataloader returns all the fields in the DictionaryDataset.
    """
    # Create a dataset.
    n_samples = 50
    dataset = {'data': torch.randn((n_samples, 2))}
    dataset.update({k: torch.randn((n_samples, 1)) for k in fields})
    dataset = DictionaryDataset(dataset)

    # Splits the dataset.
    batch_size = 5
    datamodule = DictionaryDataModule(dataset, lengths=lengths, batch_size=batch_size, random_split=random_split)
    datamodule.setup('fit')

    # The length of the datasets is correct.
    # The test dataloader is skipped if len(lengths) < 3.
    for length, loader_fn in zip(
            lengths,
            [datamodule.train_dataloader, datamodule.val_dataloader, datamodule.test_dataloader]
    ):
        loader = loader_fn()

        # Check that the dataset has the correct number of samples.
        if isinstance(length, int):
            assert len(loader.dataset) == length
        else:  # Fraction.
            assert len(loader.dataset) == int(n_samples * length)

        # Check also the fields.
        for data in loader:
            # Data is always there.
            assert data['data'].shape == (batch_size, 2)
            for field in fields:
                assert data[field].shape == (batch_size, 1)

    # An error is raised if the length of the test set has not been specified.
    if len(lengths) < 3:
        with pytest.raises(ValueError, match='you need to pass three lengths'):
            datamodule.test_dataloader()


@pytest.mark.parametrize('random_split', [True, False])
def test_dictionary_data_module_multidataset(random_split):
    """The datamodule combines multiple datasets into one."""
    n_datasets = 3
    n_samples = 10
    batch_size = 2
    lengths = [.8, .2]

    # Create the datasets.
    datasets = []
    for dataset_idx in range(n_datasets):
        dataset = DictionaryDataset({
            f'data{dataset_idx}': torch.randn(n_samples, 2),
            f'labels{dataset_idx}': torch.randn(n_samples),
        })
        datasets.append(dataset)

    # Create the dataloader.
    datamodule = DictionaryDataModule(datasets, batch_size=batch_size, lengths=lengths, random_split=random_split)
    datamodule.setup()
    dataloader = datamodule.train_dataloader()
    assert dataloader.dataset_len == int(lengths[0]*n_samples)

    # Test that the batches are correct.
    for batch in dataloader:
        assert len(batch) == n_datasets
        for dataset_idx in range(n_datasets):
            batch_dataset = batch[f'dataset{dataset_idx}']
            assert batch_dataset[f'data{dataset_idx}'].shape == (batch_size, 2)
            assert batch_dataset[f'labels{dataset_idx}'].shape == (batch_size,)

    # If datasets are not of the same dimension, the dataloader should explode on init.
    datasets.append(DictionaryDataset({
        f'data': torch.randn(n_samples+3, 2),
        f'labels': torch.randn(n_samples+3),
    }))
    datamodule = DictionaryDataModule(datasets)
    datamodule.setup()
    with pytest.raises(ValueError, match='must have the same number of samples'):
        datamodule.train_dataloader()
