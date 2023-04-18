import torch
import numpy as np
from mlcolvar.core.transform.utils import Statistics
from torch.utils.data import Dataset

__all__ = ["DictionaryDataset"]

class DictionaryDataset(Dataset):
    """Define a torch dataset from a dictionary of lists/array/tensors and names.
    E.g. { 'data' : torch.Tensor([1,2,3,4]), 
           'labels' : [0,0,1,1],
           'weights' : np.asarray([0.5,1.5,1.5,0.5]) }
    """
    def __init__(self, dictionary: dict = None, **kwargs):
        """Create a Dataset from a dictionary or from a list of kwargs.

        Parameters
        ----------
        dictionary : dict
            Dictionary with names and tensors

        """
        # assert type dict 
        if (dictionary is not None) and (not isinstance(dictionary,dict)):
            raise TypeError(f'DictionaryDataset requires a dictionary , not {type(dictionary)}.')

        # Add kwargs to dict
        if dictionary is None:
            dictionary = {}
        dictionary = {**dictionary, **kwargs}
        if len(dictionary) == 0:
            raise ValueError('Empty datasets are not supported')
        
        # convert to torch.Tensors
        for key,val in dictionary.items():
            if not isinstance(val,torch.Tensor):
                dictionary[key] = torch.Tensor(val)

        # save dictionary
        self._dictionary = dictionary
        
        # check that all elements of dict have same length
        it = iter(dictionary.values())
        self.length = len(next(it))
        if not all(len(l) == self.length for l in it):
            raise ValueError('not all arrays in dictionary have same length!')

    def __getitem__(self, index):
        if isinstance(index,str):
            #raise TypeError(f'Index ("{index}") should be a slice, and not a string. To access the stored dictionary use .dictionary["{index}"] instead.')
            return self._dictionary[index]
        else:
            slice_dict = {}
            for key,val in self._dictionary.items():
                slice_dict[key] = val[index]
            return slice_dict
    
    def __setitem__(self, index, value):
        if isinstance(index,str):
            # check lengths 
            if len(value) != len(self):
                raise ValueError(f'length of value ({len(value)}) != length of dataset ({len(self)}).')
            self._dictionary[index] = value
        else:
            raise NotImplementedError(f'Only string indexes can be set, {type(index)} is not supported.')
    
    def __len__(self):
        value = next(iter(self._dictionary.values()))
        return len(value)
    
    def get_stats(self):
        """Compute statistics ('mean','Std','Min','Max') of the dataset. 

        Returns
        -------
        stats 
            dictionary of dictionaries with statistics
        """
        stats = {}
        for k in self.keys:
            stats[k] = Statistics(self._dictionary[k]).to_dict()
        return stats
    
    def __repr__(self) -> str:
        string = 'DictionaryDataset('
        for key,val in self._dictionary.items():
            string += f' "{key}": {list(val.shape)},'
        string = string[:-1]+' )'
        return string

    @property
    def keys(self):
        return tuple(self._dictionary.keys())

def test_DictionaryDataset():
    # from list
    dataset_dict = { 'data' : torch.Tensor([[1.],[2.],[.3],[.4]]), 
                     'labels' : [0,0,1,1],
                     'weights' : np.asarray([0.5,1.5,1.5,0.5]) }

    dataset = DictionaryDataset(dataset_dict)
    print(len(dataset))
    print(dataset[0])
    print(dataset[0:2]['data'])
    print(dataset[0:2]['data'].dtype)

    # test with dataloader
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset,batch_size=1)
    batch=next(iter(loader))
    print(batch['data'])

    # test with fastdataloader
    from .dataloader import FastDictionaryLoader
    loader = FastDictionaryLoader(dataset,batch_size=1)
    batch=next(iter(loader))
    print(batch)

if __name__ == "__main__":
    test_DictionaryDataset()