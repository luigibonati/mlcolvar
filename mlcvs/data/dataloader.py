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
    
    It is much faster than TensorDataset + DataLoader because ``DataLoader``
    grabs individual indices of the dataset and calls cat (slow).

    Notes
    -----

    Adapted from https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6. 

    """
    def __init__(self, dataset: Union[dict, DictionaryDataset], batch_size: int = 0, shuffle: bool = True):
        """Initialize a ``FastDictionaryLoader``.

        Parameters
        ----------
        dataset : DictionaryDataset or dict
            The dataset.
        batch_size : int, optional
            Batch size, by default 0 (==single batch).
        shuffle : bool, optional
            If ``True``, shuffle the data *in-place* whenever an
            iterator is created out of this object, by default ``True``.
        """

        # Convert to DictionaryDataset if a dict is given
        if isinstance(dataset, dict):
            dataset = DictionaryDataset(dataset)
        
        # Retrieve selection if it a subset
        if isinstance(dataset, Subset):
            if isinstance(dataset.dataset, DictionaryDataset):
                dataset = DictionaryDataset(dataset.dataset[dataset.indices])

        # Save parameters
        self.dataset = dataset
        self._batch_size = batch_size
        self.shuffle = shuffle

    @property
    def batch_size(self):
        """Batch size."""
        return self._batch_size if self._batch_size > 0 else len(self.dataset)

    @batch_size.setter
    def batch_size(self, new_batch_size):
        self._batch_size = new_batch_size

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(len(self.dataset))
        else:
            self.indices = None
        self.i = 0
        return self

    def __next__(self):
        if self.i >= len(self.dataset):
            raise StopIteration
        
        if self.indices is not None:
            indices = self.indices[self.i:self.i+self.batch_size]
            batch = self.dataset[indices]
        else:
            batch = self.dataset[self.i:self.i+self.batch_size]

        self.i += self.batch_size
        return batch

    def __len__(self):
        # Number of batches
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    @property
    def keys(self):
        return self.dataset.keys
    
    def __repr__(self) -> str:
        string = f'FastDictionaryLoader(length={len(self.dataset)}, batch_size={self.batch_size}, shuffle={self.shuffle})'
        return string

    def get_stats(self):
        """Compute statistics ('mean','Std','Min','Max') of the dataloader. 

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


def test_FastDictionaryLoader(): 
    X = torch.arange(1,11).unsqueeze(1)
    y = X**2

    # Start from dictionary
    d = {'data': X, 'labels': y}
    dataloader = FastDictionaryLoader(d,batch_size=1,shuffle=False)
    print(len(dataloader))
    print(next(iter(dataloader)))

    # or from dict dataset
    dict_dataset = DictionaryDataset(d)
    dataloader = FastDictionaryLoader(dict_dataset,batch_size=1,shuffle=False)
    print(len(dataloader))
    print(next(iter(dataloader)))

    # or from subset
    train, _ = torch.utils.data.random_split(dict_dataset, [0.5,0.5])
    dataloader = FastDictionaryLoader(train,batch_size=1,shuffle=False)
    print(len(dataloader))
    print(next(iter(dataloader)))

if __name__ == "__main__":
    test_FastDictionaryLoader()