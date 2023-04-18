#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
PyTorch Lightning DataModule object for DictDatasets.
"""

__all__ = ["FastDictionaryLoader"]


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

from typing import Optional, Union, Sequence
import torch
from torch.utils.data import Subset
from mlcolvar.data import DictDataset
from mlcolvar.core.transform.utils import Statistics


# =============================================================================
# FAST DICTIONARY LOADER CLASS
# =============================================================================

class FastDictionaryLoader:
    """PyTorch DataLoader for :class:`~mlcolvar.data.dataset.DictDataset`s.
    
    It is much faster than ``TensorDataset`` + ``DataLoader`` because ``DataLoader``
    grabs individual indices of the dataset and calls cat (slow).

    The class can also merge multiple :class:`~mlcolvar.data.dataset.DictDataset`s
    that have different keys (see example below). The datasets must all have the
    same number of samples.

    Notes
    -----

    Adapted from https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6.

    Examples
    --------

    >>> x = torch.arange(1,11)

    A ``FastDictionaryLoader`` can be initialize from a ``dict``, a :class:`~mlcolvar.data.dataset.DictDataset`,
    or a ``Subset`` wrapping a :class:`~mlcolvar.data.dataset.DictDataset`.

    >>> # Initialize from a dictionary.
    >>> d = {'data': x.unsqueeze(1), 'labels': x**2}
    >>> dataloader = FastDictionaryLoader(d, batch_size=1, shuffle=False)
    >>> dataloader.dataset_len  # number of samples
    10
    >>> # Print first batch.
    >>> for batch in dataloader:
    ...     print(batch)
    ...     break
    {'data': tensor([[1]]), 'labels': tensor([1])}

    >>> # Initialize from a DictDataset.
    >>> dict_dataset = DictDataset(d)
    >>> dataloader = FastDictionaryLoader(dict_dataset, batch_size=2, shuffle=False)
    >>> len(dataloader)  # Number of batches
    5

    >>> # Initialize from a PyTorch Subset object.
    >>> train, _ = torch.utils.data.random_split(dict_dataset, [0.5, 0.5])
    >>> dataloader = FastDictionaryLoader(train, batch_size=1, shuffle=False)

    It is also possible to iterate over multiple dictionary datasets having
    different keys for multi-task learning

    >>> dataloader = FastDictionaryLoader(
    ...     dataset=[dict_dataset, {'some_unlabeled_data': torch.arange(10)+11}],
    ...     batch_size=1, shuffle=False,
    ... )
    >>> dataloader.dataset_len  # This is the number of samples in one dataset.
    10
    >>>  # Print first batch.
    >>> from pprint import pprint
    >>> for batch in dataloader:
    ...     pprint(batch)
    ...     break
    {'dataset0': {'data': tensor([[1]]), 'labels': tensor([1])},
     'dataset1': {'some_unlabeled_data': tensor([11])}}

    """
    def __init__(
            self,
            dataset: Union[dict, DictDataset, Subset, Sequence],
            batch_size: int = 0,
            shuffle: bool = True,
    ):
        """Initialize a ``FastDictionaryLoader``.

        Parameters
        ----------
        dataset : dict or DictDataset or Subset of DictDataset or list-like.
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
        """DictDataset or list[DictDataset]: The dictionary dataset(s)."""
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
        if self.has_multiple_datasets:
            # List of datasets.
            return len(self.dataset[0])
        return len(self.dataset)

    @property
    def batch_size(self):
        """int: Batch size."""
        return self._batch_size if self._batch_size > 0 else self.dataset_len

    @batch_size.setter
    def batch_size(self, batch_size):
        self._batch_size = batch_size

    @property
    def keys(self):
        """tuple[str] or tuple[tuple[str]]: The keys of all the datasets in this loader."""
        if self.has_multiple_datasets:
            return tuple(d.keys for d in self.dataset)
        return self.dataset.keys

    @property
    def has_multiple_datasets(self):
        return not isinstance(self.dataset, DictDataset)

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

        if self.has_multiple_datasets:
            batch = {}
            for dataset_idx, dataset in enumerate(self.dataset):
                batch[f'dataset{dataset_idx}'] = self._get_batch(dataset)
        else:
            batch = self._get_batch(self.dataset)

        self.i += self.batch_size
        return batch

    def __len__(self):
        # Number of batches.
        return (self.dataset_len + self.batch_size - 1) // self.batch_size
    
    def __repr__(self) -> str:
        string = f'FastDictionaryLoader(length={self.dataset_len}, batch_size={self.batch_size}, shuffle={self.shuffle})'
        return string

    def get_stats(self, dataset_idx: Optional[int] = None):
        """Compute statistics ``('mean','std','min','max')`` of the dataloader.

        Parameters
        ----------
        dataset_idx : int, optional
            If given and the loader has multiple datasets, only the statistics
            of the ``dataset_idx``-th dataset will be returned.

        Returns
        -------
        stats : Dict[Dict] or List[Dict[Dict]]
            A dictionary mapping the datasets' keys (e.g., ``'data'``, ``'weights'``)
            to their statistics. If the loader has multiple datasets, ``stats[i]``
            is the dictionary for the ``i``-th dataset.

        """
        # Check whether this loader has multiple datasets.
        if self.has_multiple_datasets:
            datasets = self.dataset
        else:
            datasets = [self.dataset]

        # Select requested dataset.
        is_selected_dataset = dataset_idx is not None
        if is_selected_dataset:
            datasets = [datasets[dataset_idx]]

        # Compute stats.
        stats = {f'dataset{i}': {} for i in range(len(datasets))}
        for dataset_idx, dataset in enumerate(datasets):
            for k in dataset.keys:
                stats[f'dataset{dataset_idx}'][k] = Statistics(dataset[k]).to_dict()

        # Return only a single dictionary if there are no multiple datasets.
        if is_selected_dataset or not self.has_multiple_datasets:
            return stats['dataset0']
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
    """Convert Dict[Tensor] and Subset[DictDataset] to DictDataset.

    An error is raised if ``d`` cannot is of any other type.
    """
    # Convert to DictDataset if a dict is given.
    if isinstance(d, dict):
        d = DictDataset(d)
    elif isinstance(d, Subset) and isinstance(d.dataset, DictDataset):
        # TODO: This might not not safe for classes that inherit from Subset or DictionaryDatset.
        # Retrieve selection if it a subset.
        d = d.dataset.__class__(d.dataset[d.indices])
    elif not isinstance(d, DictDataset):
        raise ValueError('The data must be of type dict, DictDataset or Subset[DictDataset].')
    return d


if __name__ == '__main__':
    import doctest
    doctest.testmod()
