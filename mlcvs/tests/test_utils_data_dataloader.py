#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Tests for the members of the mlcvs.data.dataloader module.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import pytest
import torch
from torch.utils.data import Subset

from mlcvs.data.dataset import DictionaryDataset
from mlcvs.data.dataloader import FastDictionaryLoader


# =============================================================================
# TESTS
# =============================================================================

def test_fast_dictionary_loader_init():
    """FastDictionaryLoader can be initialized from dict, DictionaryDataset, and Subsets."""
    x = torch.arange(1, 11)
    y = x**2
    x = x.unsqueeze(1)

    batch_size = 2

    def _check_dataloader(dl, n_samples=10, n_batches=5, shuffled=False):
        assert dl.dataset_len == n_samples
        assert len(dl) == n_batches

        # The detaset is converted to a DictionaryDataset.
        assert isinstance(dl.dataset, DictionaryDataset)

        # Check the batch.
        batch = next(iter(dl))
        assert set(batch.keys()) == set(['data', 'labels'])
        if not shuffled:
            assert torch.all(batch['data'] == torch.tensor([[1.], [2.]]))
            assert torch.all(batch['labels'] == torch.tensor([1., 4.]))

    # Start from dictionary
    d = {'data': x, 'labels': y}
    dataloader = FastDictionaryLoader(d, batch_size=batch_size, shuffle=False)
    _check_dataloader(dataloader)

    # or from DictionaryDataset
    dict_dataset = DictionaryDataset(d)
    dataloader = FastDictionaryLoader(dict_dataset, batch_size=batch_size, shuffle=False)
    _check_dataloader(dataloader)

    # or from subset
    train, _ = torch.utils.data.random_split(dict_dataset, [0.5, 0.5])
    dataloader = FastDictionaryLoader(train, batch_size=batch_size, shuffle=False)
    _check_dataloader(dataloader, n_samples=5, n_batches=3, shuffled=True)

    # an error is raised if initialized from anything else.
    dataset = torch.utils.data.TensorDataset(x)
    with pytest.raises(ValueError, match='must be of type'):
        FastDictionaryLoader(dataset)


def test_fast_dictionary_loader_multidataset():
    """FastDictionaryLoader combines multiple datasets into one."""
    # Create datasets with different fields.
    n_samples = 10
    datasets = [
        Subset(DictionaryDataset({'data': torch.randn(n_samples+2, 2)}), indices=list(range(1, 11))),
        DictionaryDataset({'data': torch.randn(n_samples, 2), 'labels': torch.randn(n_samples)}),
        {'data': torch.randn(n_samples, 2), 'labels': torch.randn(n_samples), 'weights': torch.randn(n_samples)},
    ]

    # Create the dataloader.
    batch_size = 2
    dataloader = FastDictionaryLoader(datasets, batch_size=batch_size)

    # Check that dataset_len and number of batches are computed correctly.
    assert dataloader.dataset_len == n_samples
    assert len(dataloader) == 5

    # Test that the batches are correct.
    for batch in dataloader:
        assert len(batch) == len(datasets)
        for i in range(3):
            assert len(batch[f'dataset{i}']) == i+1
            assert batch[f'dataset{i}']['data'].shape == (batch_size, 2)
        for i in range(1, 3):
            assert batch[f'dataset{i}']['labels'].shape == (batch_size,)
        assert batch[f'dataset{i}']['weights'].shape == (batch_size,)

    # If datasets are not of the same dimension, the datamodule explodes.
    datasets.append(DictionaryDataset({
        'data': torch.randn(n_samples+1, 2),
        'labels': torch.randn(n_samples+1),
    }))
    with pytest.raises(ValueError, match='must have the same number of samples'):
        FastDictionaryLoader(datasets)
