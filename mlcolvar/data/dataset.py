import torch
import torch_geometric
import numpy as np
from mlcolvar.core.transform.utils import Statistics
from torch.utils.data import Dataset
from operator import itemgetter

__all__ = ["DictDataset"]


class DictDataset(Dataset):
    """Define a torch dataset from a dictionary of lists/array/tensors and names.

    E.g. { 'data' : torch.Tensor([1,2,3,4]),
           'labels' : [0,0,1,1],
           'weights' : np.asarray([0.5,1.5,1.5,0.5]) }
    """

    def __init__(self, 
                 dictionary: dict=None, 
                 feature_names = None, 
                 metadata: dict = None, 
                 data_type : str = 'descriptors', 
                 create_ref_idx : bool = False, 
                 **kwargs):
        """Create a Dataset from a dictionary or from a list of kwargs.

        Parameters
        ----------
        dictionary : dict
            Dictionary with names and tensors
        feature_names : array-like
            List or numpy array with feature names
        metadata : dict
            Dictionary with metadata quantities shared across the whole dataset.
        data_type : str
            Type of data stored in the dataset, either 'descriptors' or 'graphs', by default 'descriptors'.
            This will be stored in the dataset.metadata dictionary.


        """
        # assert type dict
        if (dictionary is not None) and (not isinstance(dictionary, dict)):
            raise TypeError(
                f"DictDataset requires a dictionary , not {type(dictionary)}."
            )
        
        if (metadata is not None) and (not isinstance(metadata, dict)):
            raise TypeError(
                f"DictDataset metadata requires a dictionary , not {type(metadata)}."
            )
        
        # assert data_type is 'descriptors' or 'graphs'
        if not data_type in ['descriptors', 'graphs']:
            raise TypeError(
                f"data_type expected to be either 'descriptors' or 'graph', found {data_type}"
            )
        
        # Add kwargs to dict
        if dictionary is None:
            dictionary = {}
        dictionary = {**dictionary, **kwargs}
        if len(dictionary) == 0:
            raise ValueError("Empty datasets are not supported")

        # initialize metadata as dict
        if metadata is None:
            metadata = {}
        
        if 'data_type' in metadata.keys():
            if not metadata['data_type'] == data_type:
                raise ValueError(f"Two different data_type specified. Found {metadata['data_type']} in metadata and {data_type} as keyword")
        else:
            metadata['data_type'] = data_type

        # convert to torch.Tensors
        for key, val in dictionary.items():
            if not isinstance(val, torch.Tensor):
                if key in ["data_list", "data_list_lag"]:
                    dictionary[key] = val
                else:
                    dictionary[key] = torch.Tensor(val)

        # save dictionary
        self._dictionary = dictionary

        # save feature names
        self.feature_names = feature_names

        # save metadata
        self.metadata = metadata

        # check that all elements of dict have same length
        it = iter(dictionary.values())
        self.length = len(next(it))
        if not all([len(l) == self.length for l in it]):
            raise ValueError("not all arrays in dictionary have same length!")
        
        # add indexing of entries for shuffling and slicing reference
        if create_ref_idx and "ref_idx" not in self._dictionary.keys():
            dictionary['ref_idx'] = torch.arange(len(self), dtype=torch.int)
        

    def __getitem__(self, index):
        if isinstance(index, str):
            return self._dictionary[index]
        else: 
            slice_dict = {}
            for key, val in self._dictionary.items():
                try:
                    slice_dict[key] = val[index]
                except Exception:
                    slice_dict[key] = list(itemgetter(*index)(val))
            return slice_dict

    def __setitem__(self, index, value):
        if isinstance(index, str):
            # check lengths
            if len(value) != len(self):
                raise ValueError(
                    f"length of value ({len(value)}) != length of dataset ({len(self)})."
                )
            self._dictionary[index] = value
        else:
            raise NotImplementedError(
                f"Only string indexes can be set, {type(index)} is not supported."
            )

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
        if self.metadata == 'graph':
            raise ValueError (
                "Method get_stats not supported for graph-based dataset!"
            )
        stats = {}
        for k in self.keys:
            print("KEY: ", k, end="\n\n\n")
            if k != "ref_idx":
                stats[k] = Statistics(self._dictionary[k]).to_dict()
        return stats

    def __repr__(self) -> str:
        parts = ["DictDataset("]
        for key, val in self._dictionary.items():
            if key in ["data_list", "data_list_lag"]:
                parts.append(f' "{key}": {len(val)},')
            else:
                parts.append(f' "{key}": {list(val.shape)},')
        if self.metadata:
            parts.append(" metadata={")
            for key, val in self.metadata.items():
                parts.append(f' "{key}": {val},')
            if parts[-1].endswith(","):
                parts[-1] = parts[-1][:-1]
            parts.append(" },")
        if parts[-1].endswith(","):
            parts[-1] = parts[-1][:-1]
        parts.append(" )")
        return "".join(parts)

    @property
    def keys(self):
        return tuple(self._dictionary.keys())

    @property
    def feature_names(self):
        """Feature names."""
        return self._feature_names

    @feature_names.setter
    def feature_names(self, value):
        self._feature_names = (
            np.asarray(value, dtype=str) if value is not None else value
        )

    def get_graph_inputs(self):
        """Generate and input suitable for graph models. Returns the whole dataset as a single batch not shuffled"""
        assert self.metadata['data_type'] == 'graphs', (
            'Graph inputs can only be generated for graph-based datasets'
        )
        loader = torch_geometric.loader.DataLoader(self, 
                                                   batch_size=len(self), 
                                                   shuffle=False )
        return next(iter(loader))['data_list']

def test_DictDataset():
    # descriptors based
    # from list
    data = torch.Tensor([[1.0], [2.0], [0.3], [0.4]])
    labels = [0, 0, 1, 1]
    weights = np.asarray([0.5, 1.5, 1.5, 0.5])
    dataset_dict = {
        "data": data,
        "labels": labels,
        "weights": weights,
    }
    
    dataset = DictDataset(dataset_dict)
    print(len(dataset))
    print(dataset[0])
    print(dataset[0:2]["data"])
    print(dataset[0:2]["data"].dtype)

    # test with dataloader
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=1)
    batch = next(iter(loader))
    print(batch["data"])

    # test with fastdataloader
    from mlcolvar.data import DictLoader
    loader = DictLoader(dataset, batch_size=1)
    batch = next(iter(loader))
    print(batch)

    from mlcolvar.data.graph.atomic import AtomicNumberTable, Configuration
    from mlcolvar.data.graph.utils import create_dataset_from_configurations
    # graphs based
    numbers = [8, 1, 1]
    positions = np.array(
        [[[0.0, 0.0, 0.0], [0.07, 0.07, 0.0], [0.07, -0.07, 0.0]],
        [[0.0, 0.0, 0.0], [0.07, 0.07, 0.0], [0.07, -0.07, 0.0]],
        [[0.0, 0.0, 0.0], [0.07, 0.07, 0.0], [0.07, -0.07, 0.0]]],
        dtype=float
    )
    cell = np.identity(3, dtype=float) * 0.2
    graph_labels = np.array([[1], [0], [1]])
    node_labels = np.array([[0], [1], [1]])
    z_table = AtomicNumberTable.from_zs(numbers)

    config = [Configuration(
        atomic_numbers=numbers,
        positions=positions[i] + 0.1*i,
        cell=cell,
        pbc=[True] * 3,
        node_labels=node_labels[i],
        graph_labels=graph_labels,
    ) for i in range(3)]
    graph_dataset = create_dataset_from_configurations(config, 
                                              z_table, 
                                              0.1, 
                                              show_progress=False
                                            )
    print(graph_dataset)
    assert(isinstance(graph_dataset, DictDataset))

    # check __getitem__
    # string
    out = dataset['data']
    assert( torch.allclose(out, data) ) 
    out = graph_dataset['data_list']
    assert( torch.allclose(out[1]['positions'], torch.Tensor(positions+0.1))) 
    
    # int
    out = dataset[1]
    assert( torch.allclose(out['data'], data[1]) ) 
    out = graph_dataset[1]
    assert( torch.allclose(out['data_list']['positions'], torch.Tensor(positions+0.1))) 


    # list
    out = dataset[[0,1,2]]
    assert( torch.allclose(out['data'], data[[0,1,2]]) ) 
    out = graph_dataset[[0,1,2]]
    for i in [0,1,2]: 
        assert( torch.allclose(out['data_list'][i]['positions'], torch.Tensor(positions+0.1*i))) 

    # slice
    out = dataset[0:2]
    assert( torch.allclose(out['data'], data[[0,1]]) ) 
    out = graph_dataset[0:2]
    for i in [0,1]: 
        assert( torch.allclose(out['data_list'][i]['positions'], torch.Tensor(positions+0.1*i))) 

    # range
    out = dataset[range(0,2)]
    assert( torch.allclose(out['data'], data[[0,1]]) ) 
    out = graph_dataset[range(0,2)]
    for i in [0,1]: 
        assert( torch.allclose(out['data_list'][i]['positions'], torch.Tensor(positions+0.1*i))) 

    # np.ndarray
    out = dataset[np.array(1)]
    assert( torch.allclose(out['data'], data[1]) ) 
    out = graph_dataset[np.array(1)]
    assert( torch.allclose(out['data_list']['positions'], torch.Tensor(positions+0.1))) 
    
    out = dataset[np.array([0,1,2])]
    assert( torch.allclose(out['data'], data[[0,1,2]]) ) 
    out = graph_dataset[np.array([0,1,2])]
    for i in [0,1,2]:
        assert( torch.allclose(out['data_list'][i]['positions'], torch.Tensor(positions+0.1*i))) 

    # torch.Tensor
    out = dataset[torch.tensor([1], dtype=torch.long)]
    assert( torch.allclose(out['data'], data[1]) ) 
    out = graph_dataset[torch.tensor([1], dtype=torch.long)]
    assert( torch.allclose(out['data_list']['positions'], torch.Tensor(positions+0.1))) 

    out = dataset[torch.tensor([0,1,2], dtype=torch.long)]
    assert( torch.allclose(out['data'], data[[0,1,2]]) ) 
    out = graph_dataset[torch.tensor([0,1,2], dtype=torch.long)]
    for i in [0,1,2]:
        assert( torch.allclose(out['data_list'][i]['positions'], torch.Tensor(positions+0.1*i))) 


if __name__ == "__main__":
    test_DictDataset()
