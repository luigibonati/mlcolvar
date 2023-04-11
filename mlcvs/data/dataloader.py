#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
PyTorch Lightning DataModule object for DictionaryDatasets.
"""

__all__ = ["FastDictionaryLoader"]


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

from typing import Union
import torch
from torch.utils.data import Subset
from mlcvs.data import DictionaryDataset
from mlcvs.core.transform.utils import Statistics


# =============================================================================
# FAST DICTIONARY LOADER CLASS
# =============================================================================

class FastDictionaryLoader:
    """PyTorch DataLoader for :class:`~mlcvs.data.dataset.DictionaryDataset`s.
    
    It is much faster than ``TensorDataset`` + ``DataLoader`` because ``DataLoader``
    grabs individual indices of the dataset and calls cat (slow).

    The class can also merge multiple :class:`~mlcvs.data.dataset.DictionaryDataset`s
    that have different keys (see example below). The datasets must all have the
    same number of samples.

    Notes
    -----

    Adapted from https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6.

    Examples
    --------

    >>> x = torch.arange(1,11)

    A ``FastDictionaryLoader`` can be initialize from a ``dict``, a :class:`~mlcvs.data.dataset.DictionaryDataset`,
    or a ``Subset`` wrapping a :class:`~mlcvs.data.dataset.DictionaryDataset`.

    >>> # Initialize from a dictionary.
    >>> d = {'data': x.unsqueeze(1), 'labels': x**2}
    >>> dataloader = FastDictionaryLoader(d, batch_size=1, shuffle=False)
    >>> dataloader.dataset_len  # number of samples
    10
    >>> next(iter(dataloader))  # first batch
    {'data': tensor([[1]]), 'labels': tensor([1])}

    >>> # Initialize from a DictionaryDataset.
    >>> dict_dataset = DictionaryDataset(d)
    >>> dataloader = FastDictionaryLoader(dict_dataset, batch_size=2, shuffle=False)
    >>> len(dataloader)  # Number of batches
    5
    >>> batch = next(iter(dataloader))  # first batch
    >>> batch['data']
    tensor([[1],
            [2]])

    >>> # Initialize from a Subset.
    >>> train, _ = torch.utils.data.random_split(dict_dataset, [0.5, 0.5])
    >>> dataloader = FastDictionaryLoader(train, batch_size=1, shuffle=False)

    It is also possible to iterate over multiple dictionary datasets having
    different keys for multi-task learning

    >>> dataloader = FastDictionaryLoader(
    ...     dataset=[dict_dataset, {'some_unlabeled_data': torch.arange(11,21)}],
    ...     batch_size=1, shuffle=False,
    ... )
    >>> batch = next(iter(dataloader))  # first batch

    >>> from pprint import pprint
    >>> pprint(batch)
    {'dataset0': {'data': tensor([[1]]), 'labels': tensor([1])},
     'dataset1': {'some_unlabeled_data': tensor([11])}}

    """
    def __init__(
            self,
            dataset: Union[dict, DictionaryDataset, Subset],
            batch_size: int = 0,
            shuffle: bool = True,
    ):
        """Initialize a ``FastDictionaryLoader``.

        Parameters
        ----------
        dataset : dict or DictionaryDataset or Subset of DictionaryDataset or list-like.
            The dataset or a list of datasets. If a list, the datasets can have
            different keys but they must all have the same number of samples.
        batch_size : int, optional
            Batch size, by default 0 (==single batch).
        shuffle : bool, optional
            If ``True``, shuffle the data *in-place* whenever an
            iterator is created out of this object, by default ``True``.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    @property
    def dataset(self):
        """DictionaryDataset or list[DictionaryDataset]: The dictionary dataset(s)."""
        return self._dataset

    @dataset.setter
    def dataset(self, dataset):
        try:
            self._dataset = _to_dict_dataset(dataset)
        except ValueError:
            # This is a sequence of datasets.
            datasets = [_to_dict_dataset(d) for d in dataset]

            # Check that all datasets have the same number of samples.
            if len(set([len(d) for d in datasets])) != 1:
                raise ValueError('All the datasets must have the same number of samples.')

            self._dataset = datasets

    @property
    def dataset_len(self):
        """int: Number of samples in the dataset(s)."""
        if isinstance(self._dataset, DictionaryDataset):
            return len(self.dataset)
        # List of datasets.
        return len(self.dataset[0])

    @property
    def batch_size(self):
        """int: Batch size."""
        return self._batch_size if self._batch_size > 0 else self.dataset_len

    @batch_size.setter
    def batch_size(self, batch_size):
        self._batch_size = batch_size

    def __iter__(self):
        # Even with multiple datasets (of the same length), we generate a single
        # indices permutation since these datasets are normally uncorrelated.
        if self.shuffle:
            self.indices = torch.randperm(self.dataset_len)
        else:
            self.indices = None
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration

        if isinstance(self.dataset, DictionaryDataset):
            batch = self._get_batch(self.dataset)
        else:
            # List of dict datasets.
            batch = {}
            for dataset_idx, dataset in enumerate(self.dataset):
                batch[f'dataset{dataset_idx}'] = self._get_batch(dataset)

        self.i += self.batch_size
        return batch

    def __len__(self):
        # Number of batches.
        return (self.dataset_len + self.batch_size - 1) // self.batch_size

    @property
    def keys(self):
        return self.dataset.keys
    
    def __repr__(self) -> str:
        string = f'FastDictionaryLoader(length={self.dataset_len}, batch_size={self.batch_size}, shuffle={self.shuffle})'
        return string

    def get_stats(self):
        """Compute statistics ``('mean','std','min','max')`` of the dataloader.

        Returns
        -------
        stats 
            dictionary of dictionaries with statistics
        """
        stats = {}
        for batch in iter(self):
            for k in self.keys:
                #initialize
                if k not in stats:
                    stats[k] = Statistics(batch[k])
                # or accumulate
                else:
                    stats[k].update(batch[k])

        # convert to dictionaries
        for k in stats.keys():
            stats[k] = stats[k].to_dict()

        return stats

    def _get_batch(self, dataset):
        """Return the current batch from the dataset."""
        if self.indices is not None:
            indices = self.indices[self.i:self.i+self.batch_size]
            batch = dataset[indices]
        else:
            batch = dataset[self.i:self.i+self.batch_size]
        return batch


def _to_dict_dataset(d):
    """Convert Dict[Tensor] and Subset[DictionaryDataset] to DictionaryDataset.

    An error is raised if ``d`` cannot is of any other type.
    """
    # Convert to DictionaryDataset if a dict is given.
    if isinstance(d, dict):
        d = DictionaryDataset(d)
    elif isinstance(d, Subset) and isinstance(d.dataset, DictionaryDataset):
        # TODO: This might not not safe for classes that inherit from Subset or DictionaryDatset.
        # Retrieve selection if it a subset.
        d = d.dataset.__class__(d.dataset[d.indices])
    elif not isinstance(d, DictionaryDataset):
        raise ValueError('The data must be of type dict, DictionaryDataset or Subset[DictionaryDataset].')
    return d


if __name__ == '__main__':
    import doctest
    doctest.testmod()
