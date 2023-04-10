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
        assert len(dl.dataset) == n_samples
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
    dataloader = FastDictionaryLoader(dict_dataset,batch_size=batch_size, shuffle=False)
    _check_dataloader(dataloader)

    # or from subset
    train, _ = torch.utils.data.random_split(dict_dataset, [0.5, 0.5])
    dataloader = FastDictionaryLoader(train, batch_size=batch_size, shuffle=False)
    _check_dataloader(dataloader, n_samples=5, n_batches=3, shuffled=True)
