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

@pytest.mark.parametrize('lengths', [[0.8, 0.2], [0.75, 0.2, 0.05]])
@pytest.mark.parametrize('fields', [[], ['labels', 'weights']])
def test_dictionary_data_module_split(lengths, fields):
    """The datamodule splits the dataset into train/val/test subsets.

    Tests that:
       1. The sub-datasets have the correct lengths.
       2. The dataloader returns all the fields in the DictionaryDataset.
    """
    # Create a dataset.
    n_samples = 100
    dataset = {'data': torch.randn((n_samples, 2))}
    dataset.update({k: torch.randn((n_samples, 1)) for k in fields})
    dataset = DictionaryDataset(dataset)

    # Splits the dataset.
    batch_size = 5
    datamodule = DictionaryDataModule(dataset, lengths=lengths, batch_size=batch_size)
    datamodule.setup('fit')

    # The length of the datasets is correct.
    # The test dataloader is skipped if len(lengths) < 3.
    for length, loader_fn in zip(
            lengths,
            [datamodule.train_dataloader, datamodule.val_dataloader, datamodule.test_dataloader]
    ):
        loader = loader_fn()
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
