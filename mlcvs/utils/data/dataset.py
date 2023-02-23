import torch
import numpy as np
from torch.utils.data import Dataset

__all__ = ["DictionaryDataset"]

class DictionaryDataset(Dataset):
    """Define a torch dataset from a dictionary of lists/array/tensors and names.
    E.g. { 'data' : torch.tensor([1,2,3,4]), 
           'labels' : [0,0,1,1],
           'weights' : np.asarray([0.5,1.5,1.5,0.5]) }
    """
    def __init__(self, dictionary: dict):
        """Create a Dataset from  a dictionary.

        Parameters
        ----------
        dictionary : dict
            Dictionary with names and tensors

        """
        # assert type dict 
        if not isinstance(dictionary,dict):
            raise TypeError(f'DictionaryDataset requires a dictionary , not {type(dictionary)}.')

        # convert to torch.tensors
        for key,val in dictionary.items():
            if not isinstance(val,torch.Tensor):
                dictionary[key] = torch.tensor(val)

        # save dictionary
        self.dictionary = dictionary
        
        # check that all elements of dict have same length
        it = iter(dictionary.values())
        self.length = len(next(it))
        if not all(len(l) == self.length for l in it):
            raise ValueError('not all arrays in dictionary have same length!')

    def __getitem__(self, index):
        slice_dict = {}
        for key,val in self.dictionary.items():
            slice_dict[key] = val[index]
        
        return slice_dict
    
    def __len__(self):
        return self.length

def test_DictionaryDataset():
    # from list
    dataset_dict = { 'data' : torch.tensor([[1.],[2.],[.3],[.4]]), 
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
    print(loader.names)
    batch=next(iter(loader))
    print(batch)

if __name__ == "__main__":
    test_DictionaryDataset()