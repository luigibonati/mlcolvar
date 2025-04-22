#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
PyTorch Lightning DataModule object for DictDatasets.
"""

__all__ = ["DictLoader"]


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import collections.abc
import math
from typing import Optional, Union, Sequence

import torch
from torch.utils.data import Subset

from mlcolvar.data import DictDataset
from mlcolvar.core.transform.utils import Statistics


# =============================================================================
# FAST DICTIONARY LOADER CLASS
# =============================================================================


class DictLoader:
    """PyTorch DataLoader for :class:`~mlcolvar.data.dataset.DictDataset` .

    It is much faster than ``TensorDataset`` + ``DataLoader`` because ``DataLoader``
    grabs individual indices of the dataset and calls cat (slow).

    The class can also merge multiple :class:`~mlcolvar.data.dataset.DictDataset`s
    that have different keys (see example below). Different datasets can have
    different number of samples. In this case, it is necessary to specify the
    batch sizes so that the number of batches per epoch is the same for all datasets.

    Notes
    -----

    Adapted from https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6.

    Examples
    --------

    >>> x = torch.arange(1,11)

    A ``DictLoader`` can be initialize from a ``dict``, a :class:`~mlcolvar.data.dataset.DictDataset`,
    or a ``Subset`` wrapping a :class:`~mlcolvar.data.dataset.DictDataset`.

    >>> # Initialize from a dictionary.
    >>> d = {'data': x.unsqueeze(1), 'labels': x**2}
    >>> dataloader = DictLoader(d, batch_size=1, shuffle=False)
    >>> dataloader.dataset_len  # number of samples
    10
    >>> # Print first batch.
    >>> for batch in dataloader:
    ...     print(batch)
    ...     break
    {'data': tensor([[1]]), 'labels': tensor([1])}

    >>> # Initialize from a DictDataset.
    >>> dict_dataset = DictDataset(d)
    >>> dataloader = DictLoader(dict_dataset, batch_size=2, shuffle=False)
    >>> len(dataloader)  # Number of batches
    5

    >>> # Initialize from a PyTorch Subset object.
    >>> train, _ = torch.utils.data.random_split(dict_dataset, [0.5, 0.5])
    >>> dataloader = DictLoader(train, batch_size=1, shuffle=False)

    It is also possible to iterate over multiple dictionary datasets having
    different keys for multi-task learning

    >>> dataloader = DictLoader(
    ...     dataset=[dict_dataset, {'some_unlabeled_data': torch.arange(20)+11}],
    ...     batch_size=[1, 2], shuffle=False,
    ... )
    >>> dataloader.dataset_len  # This is the number of samples in the datasets.
    [10, 20]
    >>>  # Print first batch.
    >>> from pprint import pprint
    >>> for batch in dataloader:
    ...     pprint(batch)
    ...     break
    {'dataset0': {'data': tensor([[1]]), 'labels': tensor([1])},
     'dataset1': {'some_unlabeled_data': tensor([11, 12])}}

    """

    def __init__(
        self,
        dataset: Union[dict, DictDataset, Subset, Sequence],
        batch_size: Union[int, Sequence[int]] = 0,
        shuffle: bool = True,
    ):
        """Initialize a ``DictLoader``.

        Parameters
        ----------
        dataset : dict or DictDataset or Subset of DictDataset or list-like.
            The dataset or a list of datasets. If a list, the datasets can have
            different keys but they must all have the same number of samples.
        batch_size : int or list-like of int, optional
            Batch size, by default 0 (==single batch). If multiple datasets are
            passed, this can be a list specifying the batch size for each dataset.
            Otherwise, if an ``int``, this uses the same batch size for al
            datasets. This must be set so that the total number of batches per
            epoch is the same for all datasets.
        shuffle : bool, optional
            If ``True``, shuffle the data *in-place* whenever an
            iterator is created out of this object, by default ``True``.
        """
        # This checks that dataset and batch_size are consistent.
        self._dataset = None
        self._batch_size = None
        self.set_dataset_and_batch_size(dataset=dataset, batch_size=batch_size)
        self.shuffle = shuffle

        # These are lazily initialized in __iter__().
        self.indices = None
        self.current_batch_idx = None

    @property
    def dataset(self):
        """DictDataset or list[DictDataset]: The dictionary dataset(s)."""
        return self._dataset

    @dataset.setter
    def dataset(self, dataset):
        self.set_dataset_and_batch_size(dataset=dataset, batch_size=None)

    @property
    def has_multiple_datasets(self):
        return not isinstance(self.dataset, DictDataset)

    @property
    def dataset_len(self):
        """int: Number of samples in the dataset(s)."""
        if self.has_multiple_datasets:
            return [len(d) for d in self.dataset]
        return len(self.dataset)

    @property
    def batch_size(self):
        """int or List[int]: Batch size or, in case of multiple datasets, a list of batch sizes."""
        if self.has_multiple_datasets:
            return [
                b if b > 0 else l for b, l in zip(self._batch_size, self.dataset_len)
            ]
        return self._batch_size if self._batch_size > 0 else self.dataset_len

    @batch_size.setter
    def batch_size(self, batch_size):
        self.set_dataset_and_batch_size(dataset=None, batch_size=batch_size)

    @property
    def keys(self):
        """tuple[str] or tuple[tuple[str]]: The keys of all the datasets in this loader."""
        if self.has_multiple_datasets:
            return tuple(d.keys for d in self.dataset)
        return self.dataset.keys

    def set_dataset_and_batch_size(
        self,
        dataset: Union[None, dict, DictDataset, Subset, Sequence],
        batch_size: Union[None, int, Sequence[int]],
    ):
        """Set a compatible pair of datasets and batch sizes.

        With multiple datasets, ``dataset`` and ``batch_size`` must be compatible
        so that each dataset has the same number of batches per epoch so it might
        not be possible to set the two attributes singularly without leaving the
        object in an inconsistent state. Instead, this setter can be used safely.

        Parameters
        ----------
        dataset: None or dict or DictDataset or Subset of DictDataset or list-like.
            The dataset or a list of datasets. If a list, the datasets can have
            different keys but they must all have the same number of samples.
            If ``None``, only ``batch_size`` is set.
        batch_size : None or int or list-like of int
            Batch size, by default 0 (==single batch). If multiple datasets are
            passed, this can be a list specifying the batch size for each dataset.
            Otherwise, if an ``int``, this uses the same batch size for al
            datasets. This must be set so that the total number of batches per
            epoch is the same for all datasets. If ``None``, only ``dataset``
            is set.
        """
        # Save the previous dataset and batch_size. We'll restore them if we find
        # an error to leave the object in a consistent state.
        old_dataset = self._dataset
        old_batch_size = self._batch_size

        if dataset is not None:
            # Convert dicts and Subsets to DictDatasets.
            try:
                dataset = _to_dict_dataset(dataset)
            except ValueError:  # Assume this is a sequence of datasets.
                dataset = [_to_dict_dataset(d) for d in dataset]
            self._dataset = dataset

        # Set batch size.
        if batch_size is not None:
            if self.has_multiple_datasets and not isinstance(
                batch_size, collections.abc.Sequence
            ):
                # If an integer is passed, we set the same batch size to all datasets.
                batch_size = [batch_size] * len(dataset)
            self._batch_size = batch_size

        # Now check for errors.
        if self.has_multiple_datasets:
            # batch_size must have the same length as dataset.
            if len(self._batch_size) != len(self._dataset):
                self._dataset = old_dataset
                self._batch_size = old_batch_size
                raise ValueError(
                    f"batch_size (length {len(self._batch_size)} must have length equal to the number of datasets (length {len(self.dataset)}."
                )

            # The number of batches per epoch must be the same for all datasets.
            n_batches = [
                math.ceil(dl / b) for dl, b in zip(self.dataset_len, self.batch_size)
            ]
            if len(set(n_batches)) > 1:
                self._dataset = old_dataset
                self._batch_size = old_batch_size
                raise ValueError(
                    "Multiple datasets must have the same number of batches per epoch. "
                    f"With batch_size {self._batch_size} the number of batches are {n_batches}."
                )

    def __iter__(self):
        # Since multiple datasets might have different length, we need to generate
        # separate shuffling indices for all of them.
        if not self.shuffle:
            self.indices = None
        elif self.has_multiple_datasets:
            self.indices = [torch.randperm(l) for l in self.dataset_len]
        else:
            self.indices = torch.randperm(self.dataset_len)

        # Rewind internal batch counter.
        self.current_batch_idx = 0
        return self

    def __next__(self):
        if self.current_batch_idx >= len(self):
            raise StopIteration

        if self.has_multiple_datasets:
            batch = {}
            for dataset_idx in range(len(self.dataset)):
                batch[f"dataset{dataset_idx}"] = self._get_batch(
                    dataset_idx=dataset_idx
                )
        else:
            batch = self._get_batch()

        self.current_batch_idx += 1
        return batch

    def __len__(self):
        """Return the number of batches in the loader."""
        if self.has_multiple_datasets:
            # All datasets have the same number of batches per epoch.
            dataset_len = self.dataset_len[0]
            batch_size = self.batch_size[0]
        else:
            dataset_len = self.dataset_len
            batch_size = self.batch_size
        return (dataset_len + batch_size - 1) // batch_size

    def __repr__(self) -> str:
        string = f"DictLoader(length={self.dataset_len}, batch_size={self.batch_size}, shuffle={self.shuffle})"
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
        # Make datasets always a list to simplify the code.
        if self.has_multiple_datasets:
            datasets = self.dataset
        else:
            datasets = [self.dataset]

        # Select requested dataset.
        is_selected_dataset = dataset_idx is not None
        if is_selected_dataset:
            datasets = [datasets[dataset_idx]]

        # Compute stats.
        stats = {f"dataset{i}": {} for i in range(len(datasets))}
        for dataset_idx, dataset in enumerate(datasets):
            for k in dataset.keys:
                stats[f"dataset{dataset_idx}"][k] = Statistics(dataset[k]).to_dict()

        # Return only a single dictionary if there are no multiple datasets.
        if is_selected_dataset or not self.has_multiple_datasets:
            return stats["dataset0"]
        return stats

    def _get_batch(self, dataset_idx=None):
        """Return the current batch from the dataset."""
        # Determine dataset and batch size.
        if dataset_idx is None:  # Only one dataset.
            dataset = self.dataset
            batch_size = self.batch_size
        else:
            dataset = self.dataset[dataset_idx]
            batch_size = self.batch_size[dataset_idx]

        # Determine start and end sample indices.
        start = self.current_batch_idx * batch_size
        end = start + batch_size

        # Handle shuffling.
        if self.indices is None:
            batch = dataset[start:end]
        else:
            if dataset_idx is None:
                indices = self.indices
            else:
                indices = self.indices[dataset_idx]
            batch = dataset[indices[start:end]]

        return batch


# =============================================================================
# PRIVATE UTILITY FUNCTIONS
# =============================================================================


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
        raise ValueError(
            "The data must be of type dict, DictDataset or Subset[DictDataset]."
        )
    return d


if __name__ == "__main__":
    import doctest

    doctest.testmod()
