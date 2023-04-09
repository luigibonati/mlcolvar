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
        if isinstance(dataset ,Subset):
            if isinstance(dataset.dataset, DictionaryDataset):
                dataset = DictionaryDataset(dataset.dataset[dataset.indices])

        # Save parameters
        self.dataset = dataset
        self.dataset_len = len(self.dataset)
        self.batch_size = batch_size if batch_size > 0 else self.dataset_len
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(self.dataset_len)
        else:
            self.indices = None
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        
        if self.indices is not None:
            indices = self.indices[self.i:self.i+self.batch_size]
            batch = self.dataset[indices]
        else:
            batch = self.dataset[self.i:self.i+self.batch_size]

        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches

    @property
    def keys(self):
        return self.dataset.keys
    
    def __repr__(self) -> str:
        string = f'FastDictionaryLoader(length={self.dataset_len}, batch_size={self.batch_size}, shuffle={self.shuffle})'
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