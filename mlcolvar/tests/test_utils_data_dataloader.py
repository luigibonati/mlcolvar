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

import math

import pytest
import torch
from torch.utils.data import Subset

from mlcolvar.data.dataset import DictDataset
from mlcolvar.data.dataloader import DictLoader


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def create_random_datasets(same_batch_size=True, create_ref_idx=False):
    """A list of datasets with different keys and different number of samples."""
    # One number of samples for each of the dataset.
    if same_batch_size:
        n_samples = [10, 10, 10]
    else:
        n_samples = [i * 10 for i in (1, 2, 3)]

    # Create datasets.
    datasets = [
        # Subset of a DictDataset.
        Subset(
            DictDataset({"data": torch.randn(n_samples[0] + 2, 2)}),
            indices=list(range(1, n_samples[0] + 1)),
        ),
        # DictDataset.
        DictDataset(
            {"data": torch.randn(n_samples[1], 2), "labels": torch.randn(n_samples[1])}
        ),
        # Standard dictionary.
        {
            "data": torch.randn(n_samples[2], 2),
            "labels": torch.randn(n_samples[2]),
            "weights": torch.randn(n_samples[2]),
        },
    ]
    return datasets


# =============================================================================
# TESTS
# =============================================================================


def test_fast_dictionary_loader_init():
    """DictLoader can be initialized from dict, DictDataset, and Subsets."""
    x = torch.arange(1, 11)
    y = x**2
    x = x.unsqueeze(1)

    batch_size = 2

    def _check_dataloader(dl, n_samples=10, n_batches=5, shuffled=False, ref_idx=False):
        assert dl.dataset_len == n_samples
        assert len(dl) == n_batches

        # The detaset is converted to a DictDataset.
        assert isinstance(dl.dataset, DictDataset)

        # Check the batch.
        batch = next(iter(dl))
        if not ref_idx:
            assert set(batch.keys()) == set(["data", "labels"])
        else:
            assert set(batch.keys()) == set(["data", "labels", "ref_idx"]) 
        if not shuffled:
            assert torch.all(batch["data"] == torch.tensor([[1.0], [2.0]]))
            assert torch.all(batch["labels"] == torch.tensor([1.0, 4.0]))
            if ref_idx:
                assert torch.all(batch["ref_idx"] == torch.tensor([0.0, 1.0]))

        # Count number of iterated batches.
        counter = 0
        for batch in dl:
            counter += 1
        assert counter == n_batches

    # Start from dictionary
    d = {"data": x, "labels": y}
    dataloader = DictLoader(d, batch_size=batch_size, shuffle=False)
    _check_dataloader(dataloader)

    # or from DictDataset
    dict_dataset = DictDataset(d)
    dataloader = DictLoader(dict_dataset, batch_size=batch_size, shuffle=False)
    _check_dataloader(dataloader)

    # check with ref indeces
    dict_dataset = DictDataset(d, create_ref_idx=True)
    dataloader = DictLoader(dict_dataset, batch_size=batch_size, shuffle=False)
    _check_dataloader(dataloader, ref_idx=True)

    # or from subset
    dict_dataset = DictDataset(d)
    train, _ = torch.utils.data.random_split(dict_dataset, [0.5, 0.5])
    dataloader = DictLoader(train, batch_size=batch_size, shuffle=False)
    _check_dataloader(dataloader, n_samples=5, n_batches=3, shuffled=True)

    # an error is raised if initialized from anything else.
    dataset = torch.utils.data.TensorDataset(x)
    with pytest.raises(ValueError, match="must be of type"):
        DictLoader(dataset)


@pytest.mark.parametrize("same_batch_size", [True, False])
@pytest.mark.parametrize("shuffle", [True, False])
def test_fast_dictionary_loader_multidataset_batch(same_batch_size, shuffle):
    """DictLoader combines multiple datasets into a single batch."""
    datasets = create_random_datasets(same_batch_size)
    n_datasets = len(datasets)

    # Create the dataloader.
    if same_batch_size:
        batch_size = 2
        batch_size_list = [batch_size for _ in range(n_datasets)]
    else:
        # The second/third datasets have twice/thrice many samples.
        batch_size = [2, 4, 6]
        batch_size_list = batch_size
    n_batches = math.ceil(len(datasets[0]) / batch_size_list[0])
    dataloader = DictLoader(datasets, batch_size=batch_size, shuffle=True)

    # Check that dataset_len, number of batches, and batch sizes are computed correctly.
    assert dataloader.dataset_len == [len(d) for d in dataloader.dataset]
    assert len(dataloader) == n_batches
    assert dataloader.batch_size == batch_size_list

    # Test that the batches are correct.
    batch_counter = 0
    for batch in dataloader:
        batch_counter += 1

        # batch has one key for each dataset.
        assert len(batch) == n_datasets

        # Check that the batches have the expected number of keys created in create_random_datasets().
        for i in range(n_datasets):
            n_dataset_keys = len(batch[f"dataset{i}"])
            assert n_dataset_keys == i + 1

        # Check shape 'data' (all datasets from create_random_datasets() have this key).
        for i in range(n_datasets):
            assert batch[f"dataset{i}"]["data"].shape == (batch_size_list[i], 2)

        # Check shape 'labels' (only the last two datasets have it).
        for i in range(1, n_datasets):
            assert batch[f"dataset{i}"]["labels"].shape == (batch_size_list[i],)

        # Check shape 'weights' (only the last dataset has it).
        assert batch[f"dataset{i}"]["weights"].shape == (batch_size_list[i],)

    # Iterator returns correct number of batches.
    assert batch_counter == n_batches


def test_fast_dictionary_loader_multidataset_different_n_batches():
    """DictLoader complains if the number of batches for different datasets is different."""
    # Two datasets with same or different number of samples.
    n_samples = 10
    datasets_same = [
        {"data": torch.randn(n_samples, 2)},
        {"data": torch.randn(n_samples, 2)},
    ]
    datasets_different = [
        {"data": torch.randn(n_samples, 2)},
        {"data": torch.randn(n_samples * 2, 2)},
    ]

    # Test error raising.
    err_msg = "must have the same number of batches"
    with pytest.raises(ValueError, match=err_msg):
        DictLoader(datasets_different, batch_size=1)
    with pytest.raises(ValueError, match=err_msg):
        DictLoader(datasets_different, batch_size=[2, 2])

    # Same number of samples but different batch sizes.
    with pytest.raises(ValueError, match=err_msg):
        DictLoader(datasets_same, batch_size=[1, 2])

    # Same error if we try to set separately incompatible datasets or batch_sizes.
    loader = DictLoader(datasets_same, batch_size=2)
    with pytest.raises(ValueError, match=err_msg):
        loader.dataset = datasets_different
    with pytest.raises(ValueError, match=err_msg):
        loader.batch_size = [2, 4]

    # While setting them together works.
    loader.set_dataset_and_batch_size(dataset=datasets_different, batch_size=[2, 4])
    assert loader.batch_size == [2, 4]


def test_fast_dictionary_loader_multidataset_get_stats():
    """DictLoader compute stats for all or a subset of datasets."""
    datasets = create_random_datasets()
    dataloader = DictLoader(datasets)

    # Stats for all datasets.
    stats_all = dataloader.get_stats()
    assert tuple(stats_all.keys()) == tuple(f"dataset{i}" for i in range(len(datasets)))

    # Check that statistics for all the datasets's key have been computed.
    for dataset_idx, dataset_keys in enumerate(dataloader.keys):
        dataset_name = "dataset" + str(dataset_idx)
        assert tuple(stats_all[dataset_name].keys()) == dataset_keys

    # Stats for single datasets.
    stats_1 = dataloader.get_stats(dataset_idx=1)

    # Check that the statistics for dataset1 agree for both implementations.
    for dataset_key in dataloader.keys[1]:
        for stat_name, stat in stats_1[dataset_key].items():
            assert torch.allclose(stat, stats_all["dataset1"][dataset_key][stat_name])
