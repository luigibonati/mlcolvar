import torch
import torch_geometric
import numpy as np
from mlcolvar.core.transform.utils import Statistics
from torch.utils.data import Dataset

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
        """Return a field, one sample, or a sliced DictDataset."""

        # Access one complete field.
        if isinstance(index, str):
            return self._dictionary[index]

        # Scalar indexing returns one sample as a dictionary.
        is_scalar = (
            isinstance(index, (int, np.integer))
            or (
                isinstance(index, (torch.Tensor, np.ndarray))
                and index.ndim == 0
            )
        )

        if is_scalar:
            if isinstance(index, (torch.Tensor, np.ndarray)):
                index = index.item()

            return {
                key: value[index]
                for key, value in self._dictionary.items()
            }

        # Slicing can be applied directly to all supported containers.
        if isinstance(index, slice):
            sliced = {
                key: value[index]
                for key, value in self._dictionary.items()
            }

        else:
            # Convert advanced indexing to a Python list of integer indices.
            if isinstance(index, torch.Tensor):
                indices = (
                    torch.where(index)[0].tolist()
                    if index.dtype == torch.bool
                    else index.flatten().tolist()
                )

            elif isinstance(index, np.ndarray):
                indices = (
                    np.flatnonzero(index).tolist()
                    if index.dtype == bool
                    else index.flatten().tolist()
                )

            else:
                indices = list(index)

                # Python boolean mask.
                if indices and all(
                    isinstance(item, (bool, np.bool_))
                    for item in indices
                ):
                    indices = [
                        i
                        for i, selected in enumerate(indices)
                        if selected
                    ]

            sliced = {
                key: (
                    value[indices]
                    if isinstance(value, (torch.Tensor, np.ndarray))
                    else [value[i] for i in indices]
                )
                for key, value in self._dictionary.items()
            }

        return DictDataset(
            dictionary=sliced,
            feature_names=self.feature_names,
            metadata=self.metadata.copy(),
            data_type=self.metadata.get("data_type", "descriptors"),
        )

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
            parts.append("\n\t    metadata={")
            for key, val in self.metadata.items():
                parts.append(f'"{key}": {val},\n\t\t      ')
            if parts[-1].endswith(",\n\t\t      "):
                parts[-1] = parts[-1][:-10]
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