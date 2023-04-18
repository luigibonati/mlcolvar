#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Tests for the members of the mlcolvar.data.dataloader module.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import pytest
import torch
from torch.utils.data import Subset

from mlcolvar.data.dataset import DictDataset
from mlcolvar.data.dataloader import FastDictionaryLoader


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def random_datasets():
    """A list of datasets with different keys."""
    n_samples = 10
    datasets = [
        Subset(DictDataset({'data': torch.randn(n_samples+2, 2)}), indices=list(range(1, 11))),
        DictDataset({'data': torch.randn(n_samples, 2), 'labels': torch.randn(n_samples)}),
        {'data': torch.randn(n_samples, 2), 'labels': torch.randn(n_samples), 'weights': torch.randn(n_samples)},
    ]
    return datasets


# =============================================================================
# TESTS
# =============================================================================

def test_fast_dictionary_loader_init():
    """FastDictionaryLoader can be initialized from dict, DictDataset, and Subsets."""
    x = torch.arange(1, 11)
    y = x**2
    x = x.unsqueeze(1)

    batch_size = 2

    def _check_dataloader(dl, n_samples=10, n_batches=5, shuffled=False):
        assert dl.dataset_len == n_samples
        assert len(dl) == n_batches

        # The detaset is converted to a DictDataset.
        assert isinstance(dl.dataset, DictDataset)

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

    # or from DictDataset
    dict_dataset = DictDataset(d)
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


def test_fast_dictionary_loader_multidataset_batch(random_datasets):
    """FastDictionaryLoader combines multiple datasets into a single batch."""
    n_samples = len(random_datasets[0])

    # Create the dataloader.
    batch_size = 2
    dataloader = FastDictionaryLoader(random_datasets, batch_size=batch_size)

    # Check that dataset_len and number of batches are computed correctly.
    assert dataloader.dataset_len == n_samples
    assert len(dataloader) == 5

    # Test that the batches are correct.
    for batch in dataloader:
        assert len(batch) == len(random_datasets)
        # Check shape 'data' (all datasets have it).
        for i in range(3):
            n_dataset_keys = len(batch[f'dataset{i}'])
            assert n_dataset_keys == i+1
            assert batch[f'dataset{i}']['data'].shape == (batch_size, 2)

        # Check shape 'labels' (only the last two datasets have it).
        for i in range(1, 3):
            assert batch[f'dataset{i}']['labels'].shape == (batch_size,)

        # Check shape 'weights' (only the last dataset has it).
        assert batch[f'dataset{i}']['weights'].shape == (batch_size,)


def test_fast_dictionary_loader_multidataset_different_lengths(random_datasets):
    """FastDictionaryLoader complains if multiple datasets of different dimensions are passed."""
    n_samples = len(random_datasets[0])
    random_datasets.append(DictDataset({
        'data': torch.randn(n_samples+1, 2),
        'labels': torch.randn(n_samples+1),
    }))
    with pytest.raises(ValueError, match='must have the same number of samples'):
        FastDictionaryLoader(random_datasets)


def test_fast_dictionary_loader_multidataset_get_stats(random_datasets):
    """FastDictionaryLoader compute stats for all or a subset of datasets."""
    dataloader = FastDictionaryLoader(random_datasets)

    # Stats for all datasets.
    stats_all = dataloader.get_stats()
    assert tuple(stats_all.keys()) == tuple(f'dataset{i}' for i in range(len(random_datasets)))

    # Check that statistics for all the datasets's key have been computed.
    for dataset_idx, dataset_keys in enumerate(dataloader.keys):
        dataset_name = 'dataset' + str(dataset_idx)
        assert tuple(stats_all[dataset_name].keys()) == dataset_keys

    # Stats for single datasets.
    stats_1 = dataloader.get_stats(dataset_idx=1)

    # Check that the statistics for dataset1 agree for both implementations.
    for dataset_key in dataloader.keys[1]:
        for stat_name, stat in stats_1[dataset_key].items():
            assert torch.allclose(stat, stats_all['dataset1'][dataset_key][stat_name])
