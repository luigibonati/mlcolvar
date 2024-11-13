import torch
import numpy as np
from mlcolvar.core.transform.utils import Statistics
from torch.utils.data import Dataset
import torch_geometric
from mlcolvar.data.graph.atomic import AtomicNumberTable

__all__ = ["DictDataset"]


class DictDataset(Dataset):
    """Define a torch dataset from a dictionary of lists/array/tensors and names.

    E.g. { 'data' : torch.Tensor([1,2,3,4]),
           'labels' : [0,0,1,1],
           'weights' : np.asarray([0.5,1.5,1.5,0.5]) }
    """

    def __init__(self, dictionary: dict = None, feature_names=None, **kwargs):
        """Create a Dataset from a dictionary or from a list of kwargs.

        Parameters
        ----------
        dictionary : dict
            Dictionary with names and tensors
        feature_names : array-like
            List or numpy array with feature names

        """
        # assert type dict
        if (dictionary is not None) and (not isinstance(dictionary, dict)):
            raise TypeError(
                f"DictDataset requires a dictionary , not {type(dictionary)}."
            )

        # Add kwargs to dict
        if dictionary is None:
            dictionary = {}
        dictionary = {**dictionary, **kwargs}
        if len(dictionary) == 0:
            raise ValueError("Empty datasets are not supported")

        # convert to torch.Tensors
        for key, val in dictionary.items():
            if not isinstance(val, torch.Tensor):
                if key in ["data_list", "z_table", "cutoff"]:
                    dictionary[key] = val
                else:
                    dictionary[key] = torch.Tensor(val)

        # save dictionary
        self._dictionary = dictionary

        # save feature names
        self.feature_names = feature_names

        # check that all elements of dict have same length
        it = iter(dictionary.items())
        self.length = len(next(it)[1])
        if not all([len(v)==self.length for l,v in it if l not in ["cutoff", "z_table"]]):
            raise ValueError("not all arrays in dictionary have same length!")

    def __getitem__(self, index):
        if isinstance(index, str):
            # raise TypeError(f'Index ("{index}") should be a slice, and not a string. To access the stored dictionary use .dictionary["{index}"] instead.')
            return self._dictionary[index]
        else:
            slice_dict = {}
            for key, val in self._dictionary.items():
                if key in ["z_table", "cutoff"]:
                    slice_dict[key] = val
                else:
                    slice_dict[key] = val[index]
            return slice_dict

    def __setitem__(self, index, value):
        if isinstance(index, str):
            # check lengths
            if len(value) != len(self):
                if index not in ["z_table", "cutoff"]:
                    raise ValueError(
                    f"length of value ({len(value)}) != length of dataset ({len(self)})."
                    )
                elif index=="z_table" and len(value) != len(self._dictionary[index]):
                    raise ValueError(
                    f"length of value ({len(value)}) != length of original dataset['{index}'] ({self._dictionary[index]})."
                    )
                elif index=="cutoff" and not isinstance(value, float):
                    raise ValueError(
                    f" 'cutoff' must be type float, found {type(value)} "
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
        stats = {}
        for k in self.keys:
            stats[k] = Statistics(self._dictionary[k]).to_dict()
        return stats

    def __repr__(self) -> str:
        string = "DictDataset("
        for key, val in self._dictionary.items():
            if key in ["data_list", "z_table"]:
                string += f' "{key}": {len(val)},'
            elif key in ["cutoff"]:
                string += f' "{key}": {val},'
            else:
                string += f' "{key}": {list(val.shape)},'
        string = string[:-1] + " )"
        return string

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


def test_DictDataset():
    # from list
    dataset_dict = {
        "data": torch.Tensor([[1.0], [2.0], [0.3], [0.4]]),
        "labels": [0, 0, 1, 1],
        "weights": np.asarray([0.5, 1.5, 1.5, 0.5]),
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
    from .dataloader import DictLoader

    loader = DictLoader(dataset, batch_size=1)
    batch = next(iter(loader))
    print(batch)


if __name__ == "__main__":
    test_DictDataset()
