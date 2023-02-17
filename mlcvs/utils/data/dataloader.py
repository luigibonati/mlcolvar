"""Datasets."""

import torch 
from torch.utils.data import TensorDataset
from .dataset import DictionaryDataset

__all__ = ["FastTensorDataLoader"]

from torch.utils.data import Dataset,Subset

class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors.
    
    It is much faster than TensorDataset + DataLoader because dataloader grabs individual indices of the dataset and calls cat (slow).

    Adapted to work also with dictionaries (incl. Dictionary Dataloader).

    Notes
    =====

    Adapted from https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6. 

    """
    def __init__(self, tensors, batch_size=0, shuffle=False):
        """Initialize a FastTensorDataLoader.

        Parameters
        ----------
        tensors : list of tensors or torch.Dataset or torch.Subset or list of torch.Subset or dict object containing a tensors object
            tensors to store. Must have the same length @ dim 0.
        batch_size : int, optional
            batch size, by default 0 (==single batch)
        shuffle : bool, optional
            if True, shuffle the data *in-place* whenever an
            iterator is created out of this object, by default False

        Returns
        -------
        FastTensorDataLoader
            dataloader-like object

        """
        # allocate 
        self.names = None

        # check input type
        if isinstance(tensors,Subset): 
            if isinstance(tensors.dataset,DictionaryDataset):
                data = tensors.dataset[tensors.indices]
                self.names = [ t for t in data.keys()]
                tensors = [ t for t in data.values() ]
            else:
                tensors = [ tensors.dataset.tensors[i][tensors.indices] for i in range(len(tensors.dataset.tensors)) ]
        elif isinstance(tensors,dict): # decouple it in names and tensors and recreate it in next
            self.names = [ t for t in tensors.keys()]
            tensors = [ t for t in tensors.values() ]
        elif isinstance(tensors,DictionaryDataset): # decouple it in names and tensors and recreate it in next
            self.names = [ t for t in tensors.dictionary.keys()]
            tensors = [ t for t in tensors.dictionary.values() ]
        elif isinstance(tensors,Dataset):
            tensors = [ tensors.tensors[i] for i in range(len(tensors.tensors)) ]
        # check for input type list of Subset, and create a list of tensors
        elif (isinstance(tensors,list) and isinstance(tensors[0],Subset) ):
            new_tensors = []
            tensor = torch.Tensor()
            for j in range( len( tensors[0].dataset.tensors ) ):
                for i in range( len( tensors ) ):
                    if i == 0:
                        tensor = tensors[i].dataset.tensors[j][tensors[i].indices]
                    else:    
                        tensor = torch.cat( (tensor, tensors[i].dataset.tensors[j][tensors[i].indices]), 0 )
                new_tensors.append(tensor)
            tensors = new_tensors

        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
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
            batch = tuple(torch.index_select(t, 0, indices) for t in self.tensors)
        else:
            batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        if self.names is not None: # then return a dictionary object
            batch = dict(zip(self.names, batch))

        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches

def test_FastTensorDataLoader(): 
    X = torch.arange(1,11).unsqueeze(1)
    y = X**2
    dataloader = FastTensorDataLoader([X,y],batch_size=2)
    print(next(iter(dataloader)))

    dict_dataset = {'data': X, 'labels': y}
    dataloader = FastTensorDataLoader(dict_dataset,batch_size=2)
    print(next(iter(dataloader)))

if __name__ == "__main__":
    test_FastTensorDataLoader()