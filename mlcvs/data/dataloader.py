"""Datasets."""

import torch 
from mlcvs.data import DictionaryDataset
from mlcvs.core.transform.utils import RunningStats

__all__ = ["FastDictionaryLoader"]

from torch.utils.data import Dataset,Subset

class FastDictionaryLoader:
    """
    A DataLoader-like object for a set of tensors.
    
    It is much faster than TensorDataset + DataLoader because dataloader grabs individual indices of the dataset and calls cat (slow).

    Adapted to work with dictionaries (incl. Dictionary Dataloader).

    Notes
    =====

    Adapted from https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6. 

    """
    def __init__(self, dataset : DictionaryDataset or dict, batch_size : int = 0, shuffle : bool = False):
        """Initialize a FastDictionaryLoader.

        Parameters
        ----------
        dataset : DictionaryDataset or dict
        batch_size : int, optional
            batch size, by default 0 (==single batch)
        shuffle : bool, optional
            if True, shuffle the data *in-place* whenever an
            iterator is created out of this object, by default False

        Returns
        -------
        FastDictionaryLoader
            dataloader-like object

        """

        # Convert to DictionaryDataset if a dict is given
        if isinstance(dataset,dict):
            dataset = DictionaryDataset(dataset)
        
        # Retrieve selection if it a subset
        if isinstance(dataset,Subset): 
            if isinstance(dataset.dataset,DictionaryDataset):
                dataset = DictionaryDataset(dataset.dataset[dataset.indices])

        # Save parameters
        self.dictionary = dataset
        self.dataset_len = len(self.dictionary)
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
            batch = self.dictionary[indices]
        else:
            batch = self.dictionary[self.i:self.i+self.batch_size]

        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches

    @property
    def keys(self):
        return self.dictionary.keys
    
    def get_stats(self):
        """Compute statistics ('Mean','Std','Min','Max') of the dataloader. 

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
                    stats[k] = RunningStats(batch[k])
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